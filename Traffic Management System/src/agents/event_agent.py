# src/agents/event_agent.py
"""
Event Agent for fetching and analyzing public events for traffic impact.
"""

import os
import re
import logging
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from time import sleep
from typing import List, Optional, Dict, Any

import requests


class ImpactLevel(Enum):
    """Traffic impact severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TrafficImpact:
    """Container for traffic impact analysis results."""
    level: ImpactLevel
    event_count: int
    total_expected_attendance: int
    peak_times: List[str]
    message: str
    recommendations: List[str]


class EventAgentError(Exception):
    """Base exception for EventAgent errors."""
    pass


class ConfigurationError(EventAgentError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(EventAgentError):
    """Raised when input validation fails."""
    pass


class EventAgent:
    """
    Agent for fetching and analyzing public events for traffic impact.
    
    Attributes:
        BASE_URL: Eventbrite API base URL
        DEFAULT_TIMEOUT: Request timeout in seconds
        MAX_RETRIES: Maximum number of retry attempts
        CACHE_TTL: Cache time-to-live
    
    Example:
        >>> agent = EventAgent()
        >>> data = agent.fetch_event_data("New York")
        >>> impact = agent.analyze_event_impact(data)
        >>> print(impact.level)
    """
    
    BASE_URL = "https://www.eventbriteapi.com/v3/events/search/"
    DEFAULT_TIMEOUT = 10
    MAX_RETRIES = 3
    CACHE_TTL = timedelta(minutes=15)
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the EventAgent.
        
        Args:
            token: Optional Eventbrite API token. If not provided,
                   reads from EVENTBRITE_API_KEY environment variable.
        
        Raises:
            ConfigurationError: If API token is not available.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.token = token or os.getenv("EVENTBRITE_API_KEY")
        
        if not self.token:
            raise ConfigurationError(
                "EVENTBRITE_API_KEY not set. Provide token parameter or "
                "set EVENTBRITE_API_KEY environment variable."
            )
        
        self._cache: Dict[str, tuple] = {}
    
    def _validate_city(self, city: str) -> str:
        """
        Validate and sanitize city input.
        
        Args:
            city: City name to validate
            
        Returns:
            Sanitized city name
            
        Raises:
            ValidationError: If city is invalid
        """
        if not city or not isinstance(city, str):
            raise ValidationError("City name must be a non-empty string")
        
        city = city.strip()
        
        if len(city) < 2:
            raise ValidationError("City name must be at least 2 characters")
        
        if len(city) > 100:
            raise ValidationError("City name must not exceed 100 characters")
        
        # Allow letters, spaces, hyphens, periods, apostrophes
        if not re.match(r"^[a-zA-Z\s\-\.']+$", city):
            raise ValidationError(
                f"City name contains invalid characters: {city}"
            )
        
        return city
    
    def _get_cache_key(self, city: str) -> str:
        """Generate cache key for a city."""
        return hashlib.md5(city.lower().encode()).hexdigest()
    
    def _get_cached_data(self, city: str) -> Optional[dict]:
        """Retrieve data from cache if valid."""
        cache_key = self._get_cache_key(city)
        
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self.CACHE_TTL:
                self.logger.debug(f"Cache hit for city: {city}")
                return cached_data
            else:
                # Expired - remove from cache
                del self._cache[cache_key]
        
        return None
    
    def _update_cache(self, city: str, data: dict) -> None:
        """Update cache with new data."""
        cache_key = self._get_cache_key(city)
        self._cache[cache_key] = (data, datetime.now())
    
    def fetch_event_data(
        self, 
        city: str, 
        use_cache: bool = True,
        max_retries: Optional[int] = None
    ) -> Optional[dict]:
        """
        Fetch event data from Eventbrite API for a given city.
        
        Args:
            city: City name (e.g., "New York", "Los Angeles")
            use_cache: Whether to use cached data if available
            max_retries: Override default retry count
            
        Returns:
            Event data dictionary or None if request fails
            
        Raises:
            ValidationError: If city name is invalid
        """
        city = self._validate_city(city)
        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        
        # Check cache first
        if use_cache:
            cached = self._get_cached_data(city)
            if cached:
                return cached
        
        # Prepare request
        headers = {"Authorization": f"Bearer {self.token}"}
        params = {"location.address": city}
        
        # Retry loop
        for attempt in range(retries):
            try:
                self.logger.debug(
                    f"Fetching events for {city} (attempt {attempt + 1}/{retries})"
                )
                
                response = requests.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                self._update_cache(city, data)
                
                self.logger.info(
                    f"Successfully fetched {len(data.get('events', []))} events for {city}"
                )
                return data
                
            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Timeout fetching events for {city} "
                    f"(attempt {attempt + 1}/{retries})"
                )
                if attempt < retries - 1:
                    sleep(2 ** attempt)
                    
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                
                if status_code == 401:
                    self.logger.error("Invalid API token")
                    return None
                elif status_code == 429:
                    self.logger.warning("Rate limited, waiting 60 seconds")
                    sleep(60)
                elif status_code >= 500:
                    self.logger.warning(
                        f"Server error {status_code}, retrying..."
                    )
                    if attempt < retries - 1:
                        sleep(2 ** attempt)
                else:
                    self.logger.error(f"HTTP error {status_code}: {e}")
                    return None
                    
            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Connection error (attempt {attempt + 1}/{retries})"
                )
                if attempt < retries - 1:
                    sleep(2 ** attempt)
                    
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {e}", exc_info=True)
                return None
        
        self.logger.error(f"All {retries} attempts failed for city: {city}")
        return None
    
    def analyze_event_impact(self, event_data: Optional[dict]) -> TrafficImpact:
        """
        Analyze events for traffic impact.
        
        Args:
            event_data: Raw event data from Eventbrite API
            
        Returns:
            TrafficImpact object with detailed analysis
        """
        events = event_data.get("events", []) if event_data else []
        
        if not events:
            return TrafficImpact(
                level=ImpactLevel.LOW,
                event_count=0,
                total_expected_attendance=0,
                peak_times=[],
                message="No significant events in the area.",
                recommendations=[]
            )
        
        # Calculate metrics
        total_attendance = 0
        peak_times = []
        
        for event in events:
            capacity = event.get("capacity") or 0
            total_attendance += capacity
            
            start_time = event.get("start", {}).get("local", "")
            if start_time:
                peak_times.append(start_time)
        
        # Determine impact level
        level = self._calculate_impact_level(len(events), total_attendance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(level)
        
        return TrafficImpact(
            level=level,
            event_count=len(events),
            total_expected_attendance=total_attendance,
            peak_times=sorted(peak_times)[:5],
            message=self._format_impact_message(len(events), total_attendance),
            recommendations=recommendations
        )
    
    def _calculate_impact_level(
        self, 
        event_count: int, 
        total_attendance: int
    ) -> ImpactLevel:
        """Calculate traffic impact level based on metrics."""
        if event_count >= 10 or total_attendance >= 50000:
            return ImpactLevel.CRITICAL
        elif event_count >= 5 or total_attendance >= 10000:
            return ImpactLevel.HIGH
        elif event_count >= 2 or total_attendance >= 1000:
            return ImpactLevel.MEDIUM
        else:
            return ImpactLevel.LOW
    
    def _generate_recommendations(self, level: ImpactLevel) -> List[str]:
        """Generate traffic management recommendations."""
        recommendations = {
            ImpactLevel.CRITICAL: [
                "Activate emergency traffic management protocol."
                "Deploy additional traffic officers to all affected intersections."
                "Extend green light duration on main arterials by 50%",
                "Activate all dynamic message signs with alternate routes".
                "Coordinate with public transit for increased service",
                "Consider temporary road closures near event venues"
            ],
            ImpactLevel.HIGH: [
                "Extend green light duration on main arterials",
                "Deploy traffic officers to key intersections",
                "Activate dynamic message signs with alternate routes",
                "Alert emergency services of potential congestion"
            ],
            ImpactLevel.MEDIUM: [
                "Monitor traffic flow in real-time",
                "Prepare alternate signal timing plans",
                "Have traffic officers on standby."
            ],
            ImpactLevel.LOW: []
        }
        return recommendations.get(level, [])
    
    def _format_impact_message(
        self, 
        event_count: int, 
        total_attendance: int
    ) -> str:
        """Format human-readable impact message."""
        if event_count == 0:
            return "No significant events in the area."
        
        attendance_str = f"{total_attendance:,}" if total_attendance > 0 else "unknown"
        
        return (
            f"{event_count} event{'s' if event_count != 1 else ''} detected "
            f"with approximately {attendance_str} expected attendees."
        )
    
    def clear_cache(self) -> None:
        """Clear the event data cache."""
        self._cache.clear()
        self.logger.debug("Event cache cleared")
