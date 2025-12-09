# src/agents/routing_agent.py
"""
Routing Agent for calculating optimal routes using Google Maps Directions API.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from time import sleep
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import requests


class TrafficCondition(Enum):
    """Traffic condition severity levels."""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class RouteSummary:
    """Detailed route summary with metrics."""
    route_index: int
    summary: str
    distance_text: str
    distance_meters: int
    duration_text: str
    duration_seconds: int
    duration_in_traffic_text: Optional[str] = None
    duration_in_traffic_seconds: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    waypoint_order: List[int] = field(default_factory=list)
    steps_count: int = 0
    
    @property
    def traffic_delay_seconds(self) -> int:
        """Calculate delay due to traffic."""
        if self.duration_in_traffic_seconds:
            return max(0, self.duration_in_traffic_seconds - self.duration_seconds)
        return 0
    
    @property
    def traffic_delay_percentage(self) -> float:
        """Calculate traffic delay as percentage."""
        if self.duration_seconds > 0 and self.duration_in_traffic_seconds:
            return ((self.duration_in_traffic_seconds - self.duration_seconds) 
                    / self.duration_seconds * 100)
        return 0.0
    
    @property
    def traffic_condition(self) -> TrafficCondition:
        """Estimate traffic condition based on delay."""
        delay_pct = self.traffic_delay_percentage
        if delay_pct < 10:
            return TrafficCondition.LIGHT
        elif delay_pct < 25:
            return TrafficCondition.MODERATE
        elif delay_pct < 50:
            return TrafficCondition.HEAVY
        elif delay_pct >= 50:
            return TrafficCondition.SEVERE
        return TrafficCondition.UNKNOWN


@dataclass
class RouteComparison:
    """Comparison between multiple routes with recommendation."""
    routes: List[RouteSummary]
    recommended_index: int
    recommendation_reason: str
    comparison_time: datetime = field(default_factory=datetime.now)
    
    @property
    def recommended_route(self) -> Optional[RouteSummary]:
        if 0 <= self.recommended_index < len(self.routes):
            return self.routes[self.recommended_index]
        return None
    
    @property
    def fastest_route(self) -> Optional[RouteSummary]:
        if not self.routes:
            return None
        return min(
            self.routes, 
            key=lambda r: r.duration_in_traffic_seconds or r.duration_seconds
        )
    
    @property
    def shortest_route(self) -> Optional[RouteSummary]:
        if not self.routes:
            return None
        return min(self.routes, key=lambda r: r.distance_meters)


class RoutingAgentError(Exception):
    """Base exception for RoutingAgent errors."""
    pass


class ConfigurationError(RoutingAgentError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(RoutingAgentError):
    """Raised when input validation fails."""
    pass


class GoogleMapsError(RoutingAgentError):
    """Raised for Google Maps API errors."""
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(f"{status}: {message}")


class QuotaExceededError(GoogleMapsError):
    """Raised when API quota is exceeded."""
    pass


class RoutingAgent:
    """
    Agent responsible for calculating optimal routes and alternatives
    using Google Maps Directions API.
    
    Attributes:
        BASE_URL: Google Maps Directions API endpoint
        DEFAULT_TIMEOUT: Request timeout in seconds
        MAX_RETRIES: Maximum retry attempts for failed requests
        VALID_AVOID_OPTIONS: Valid values for avoid parameter
    
    Example:
        >>> agent = RoutingAgent()
        >>> route = agent.fetch_route("New York, NY", "Boston, MA")
        >>> summaries = agent.analyze_routes(route)
        >>> print(summaries[0].duration_text)
    """
    
    BASE_URL = "https://maps.googleapis.com/maps/api/directions/json"
    DEFAULT_TIMEOUT = 10
    MAX_RETRIES = 3
    VALID_AVOID_OPTIONS = frozenset({"tolls", "highways", "ferries", "indoor"})
    
    GOOGLE_API_ERRORS = {
        "ZERO_RESULTS": "No routes found between origin and destination",
        "NOT_FOUND": "Origin or destination could not be geocoded",
        "MAX_WAYPOINTS_EXCEEDED": "Too many waypoints in request",
        "MAX_ROUTE_LENGTH_EXCEEDED": "Route is too long to process",
        "INVALID_REQUEST": "Invalid request parameters",
        "OVER_DAILY_LIMIT": "API key quota exceeded or billing issue",
        "OVER_QUERY_LIMIT": "Too many requests, rate limited",
        "REQUEST_DENIED": "API key invalid or service not enabled",
        "UNKNOWN_ERROR": "Server error, try again"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RoutingAgent.
        
        Args:
            api_key: Google Maps API key. If not provided,
                     reads from GOOGLE_MAPS_API_KEY environment variable.
        
        Raises:
            ConfigurationError: If API key is not available.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not self.api_key:
            raise ConfigurationError(
                "Google Maps API key not provided. Set GOOGLE_MAPS_API_KEY "
                "environment variable or pass api_key parameter."
            )
        
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _validate_location(self, location: str, field_name: str) -> str:
        """
        Validate and sanitize location input.
        
        Args:
            location: Location string (address or coordinates)
            field_name: Name of field for error messages
            
        Returns:
            Sanitized location string
            
        Raises:
            ValidationError: If location is invalid
        """
        if not location or not isinstance(location, str):
            raise ValidationError(f"{field_name} must be a non-empty string")
        
        location = location.strip()
        
        if len(location) < 2:
            raise ValidationError(f"{field_name} must be at least 2 characters")
        
        if len(location) > 500:
            raise ValidationError(f"{field_name} exceeds maximum length of 500 characters")
        
        # Check for malicious characters
        if re.search(r'[<>{}|\[\]\\^`]', location):
            raise ValidationError(f"{field_name} contains invalid characters")
        
        return location
    
    def _validate_avoid(self, avoid: Optional[str]) -> Optional[str]:
        """
        Validate avoid parameter.
        
        Args:
            avoid: Comma or pipe-separated avoid options
            
        Returns:
            Validated avoid string or None
            
        Raises:
            ValidationError: If avoid options are invalid
        """
        if avoid is None:
            return None
        
        # Support both comma and pipe separators
        separator = "|" if "|" in avoid else ","
        avoid_options = [opt.strip().lower() for opt in avoid.split(separator)]
        
        invalid = set(avoid_options) - self.VALID_AVOID_OPTIONS
        if invalid:
            raise ValidationError(
                f"Invalid avoid options: {invalid}. "
                f"Valid options: {self.VALID_AVOID_OPTIONS}"
            )
        
        return "|".join(avoid_options)
    
    def _make_request(
        self, 
        params: Dict[str, str],
        max_retries: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request with retry logic and error handling.
        
        Args:
            params: Request parameters
            max_retries: Override default retry count
            
        Returns:
            API response data or None
        """
        params["key"] = self.api_key
        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        
        for attempt in range(retries):
            try:
                self.logger.debug(f"API request attempt {attempt + 1}/{retries}")
                
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                status = data.get("status", "UNKNOWN_ERROR")
                
                # Handle API-level errors
                if status != "OK":
                    self._handle_api_error(status, data)
                    return None
                
                return data
                
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1}/{retries})"
                )
                if attempt < 
