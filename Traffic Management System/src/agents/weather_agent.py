# src/agents/weather_agent.py
"""
Weather Agent for fetching and analyzing weather data for traffic impact.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from hashlib import md5
from time import sleep
from typing import Dict, List, Optional, Any

import requests


class WeatherSeverity(Enum):
    """Weather severity levels for traffic impact."""
    CLEAR = "clear"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class WeatherAgentError(Exception):
    """Base exception for WeatherAgent errors."""
    pass


class ConfigurationError(WeatherAgentError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(WeatherAgentError):
    """Raised when input validation fails."""
    pass


@dataclass
class WeatherImpact:
    """Weather impact analysis result."""
    severity: WeatherSeverity
    condition: str
    condition_code: int
    description: str
    temperature: float
    feels_like: float
    humidity: int
    wind_speed: float
    wind_gust: Optional[float]
    visibility: int
    precipitation_1h: float
    snow_1h: float
    recommendations: List[str]
    traffic_speed_reduction_pct: int
    forecast_time: Optional[datetime] = None
    
    @property
    def has_precipitation(self) -> bool:
        """Check if there's any precipitation."""
        return self.precipitation_1h > 0 or self.snow_1h > 0
    
    @property
    def is_hazardous(self) -> bool:
        """Check if conditions are hazardous for driving."""
        return self.severity in [WeatherSeverity.SEVERE, WeatherSeverity.EXTREME]
    
    @property
    def visibility_category(self) -> str:
        """Categorize visibility conditions."""
        if self.visibility >= 10000:
            return "excellent"
        elif self.visibility >= 5000:
            return "good"
        elif self.visibility >= 1000:
            return "moderate"
        elif self.visibility >= 200:
            return "poor"
        else:
            return "very_poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "severity": self.severity.value,
            "condition": self.condition,
            "condition_code": self.condition_code,
            "description": self.description,
            "temperature": self.temperature,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "wind_speed": self.wind_speed,
            "wind_gust": self.wind_gust,
            "visibility": self.visibility,
            "visibility_category": self.visibility_category,
            "precipitation_1h": self.precipitation_1h,
            "snow_1h": self.snow_1h,
            "recommendations": self.recommendations,
            "traffic_speed_reduction_pct": self.traffic_speed_reduction_pct,
            "has_precipitation": self.has_precipitation,
            "is_hazardous": self.is_hazardous,
            "forecast_time": self.forecast_time.isoformat() if self.forecast_time else None
        }


class WeatherAgent:
    """
    Agent for fetching and analyzing weather data for traffic impact.
    
    Uses OpenWeatherMap API to get current conditions and forecasts,
    then analyzes their potential impact on traffic flow.
    
    Attributes:
        BASE_URL: OpenWeatherMap current weather endpoint
        FORECAST_URL: OpenWeatherMap forecast endpoint
        CACHE_TTL: Cache time-to-live for weather data
    
    Example:
        >>> agent = WeatherAgent()
        >>> impact = agent.get_traffic_impact("New York")
        >>> print(impact.severity, impact.recommendations)
    """
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    DEFAULT_TIMEOUT = 10
    MAX_RETRIES = 3
    CACHE_TTL = timedelta(minutes=10)
    
    # Weather condition to severity mapping
    WEATHER_SEVERITY = {
        # Thunderstorm group (2xx)
        "thunderstorm": WeatherSeverity.SEVERE,
        # Drizzle group (3xx)
        "drizzle": WeatherSeverity.MINOR,
        # Rain group (5xx)
        "rain": WeatherSeverity.MODERATE,
        # Snow group (6xx)
        "snow": WeatherSeverity.SEVERE,
        # Atmosphere group (7xx)
        "mist": WeatherSeverity.MINOR,
        "fog": WeatherSeverity.MODERATE,
        "haze": WeatherSeverity.MINOR,
        "dust": WeatherSeverity.MODERATE,
        "sand": WeatherSeverity.MODERATE,
        "ash": WeatherSeverity.SEVERE,
        "squall": WeatherSeverity.SEVERE,
        "tornado": WeatherSeverity.EXTREME,
        # Clear/Clouds (800, 80x)
        "clear": WeatherSeverity.CLEAR,
        "clouds": WeatherSeverity.CLEAR,
    }
    
    # Speed reduction percentages by severity
    SPEED_REDUCTION = {
        WeatherSeverity.CLEAR: 0,
        WeatherSeverity.MINOR: 10,
        WeatherSeverity.MODERATE: 25,
        WeatherSeverity.SEVERE: 40,
        WeatherSeverity.EXTREME: 60,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize WeatherAgent.
        
        Args:
            api_key: OpenWeatherMap API key. If not provided,
                     reads from OPENWEATHER_API_KEY environment variable.
        
        Raises:
            ConfigurationError: If API key is not available.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        
        if not self.api_key:
            raise ConfigurationError(
                "OpenWeatherMap API key not provided. Set OPENWEATHER_API_KEY "
                "environment variable or pass api_key parameter."
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
            raise ValidationError("City must be a non-empty string")
        
        city = city.strip()
        
        if len(city) < 2:
            raise ValidationError("City name must be at least 2 characters")
        
        if len(city) > 100:
            raise ValidationError("City name too long (max 100 characters)")
        
        # Allow letters, spaces, hyphens, commas, periods, apostrophes
        if not re.match(r"^[a-zA-Z\s\-,.']+$", city):
            raise ValidationError(f"City name contains invalid characters: {city}")
        
        return city
    
    def _validate_coordinates(self, lat: float, lon: float) -> tuple:
        """Validate geographic coordinates."""
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValidationError("Coordinates must be numeric")
        
        if not -90 <= lat <= 90:
            raise ValidationError(f"Invalid latitude: {lat} (must be -90 to 90)")
        
        if not -180 <= lon <= 180:
            raise ValidationError(f"Invalid longitude: {lon} (must be -180 to 180)")
        
        return float(lat), float(lon)
    
    def _get_cache_key(self, identifier: str) -> str:
        """Generate cache key from identifier."""
        return md5(identifier.lower().strip().encode()).hexdigest()
    
    def _make_request(
        self,
        url: str,
        params: Dict[str, Any],
        max_retries: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make API request with retry logic.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            max_retries: Override default retry count
            
        Returns:
            API response data or None
        """
        params["appid"] = self.api_key
        params["units"] = "metric"
        
        retries = max_retries if max_retries is not None else self.MAX_RETRIES
        
        for attempt in range(retries):
            try:
                self.logger.debug(f"API request attempt {attempt + 1}/{retries}")
                
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.DEFAULT_TIMEOUT
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    sleep(min(retry_after, 120))
                    continue
                
                response.raise_for_status()
                
                data = response.json()
                
                # Check API-level errors
                cod = data.get("cod")
                if cod and str(cod) != "200":
                    self._handle_api_error(data)
                    return None
                
                return data
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    sleep(2 ** attempt)
                    
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP error: {e}")
                if e.response.status_code >= 500 and attempt < retries - 1:
                    sleep(2 ** attempt)
                    continue
                return None
                
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {e}", exc_info=True)
                return None
        
        self.logger.error(f"All {retries} retry attempts failed")
        return None
    
    def _handle_api_error(self, data: Dict) -> None:
        """Handle OpenWeatherMap API error responses."""
        cod = data.get("cod")
        message = data.get("message", "Unknown error")
        self.logger.error(f"OpenWeatherMap API error {cod}: {message}")
    
    def fetch_weather_data(
        self,
        city: str,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch current weather data for a city.
        
        Args:
            city: City name (e.g., "London" or "London,UK")
            use_cache: Whether to use cached data if available
            
        Returns:
            Weather data dictionary or None
            
        Raises:
            ValidationError: If city name is invalid
        """
        city = self._validate_city(city)
        cache_key = self._get_cache_key(f"city:{city}")
        
        # Check cache
        if use_cache and cache_key in self._cache:
            data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self.CACHE_TTL:
                self.logger.debug(f"Cache hit for {city}")
                return data
        
        # Fetch fresh data
        data = self._make_request(self.BASE_URL, {"q": city})
        
        # Update cache
        if data:
            self._cache[cache_key] = (data, datetime.now())
            self.logger.info(f"Fetched weather for {city}: {data.get('weather', [{}])[0].get('main')}")
        
        return data
    
    def fetch_weather_by_coordinates(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch current weather data by coordinates.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            use_cache: Whether to use cached data
            
        Returns:
            Weather data dictionary or None
        """
        lat, lon = self._validate_coordinates(lat, lon)
        cache_key = self._get_cache_key(f"coords:{lat},{lon}")
        
        # Check cache
        if use_cache and cache_key in self._cache:
            data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self.CACHE_TTL:
                self.logger.debug(f"Cache hit for coordinates ({lat}, {lon})")
                return data
        
        # Fetch fresh data
        data = self._make_request(self.BASE_URL, {"lat": lat, "lon": lon})
        
        # Update cache
        if data:
            self._cache[cache_key] = (data, datetime.now())
        
        return data
    
    def fetch_weather(
        self,
        location: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data by city name OR coordinates.
        
        Args:
            location: City name (e.g., "London" or "London,UK")
            lat: Latitude (use with lon)
            lon: Longitude (use with lat)
            use_cache: Whether to use cached data
            
        Returns:
            Weather data dictionary or None
            
        Raises:
            ValueError: If neither location nor coordinates provided
        """
        if location:
            return self.fetch_weather_data(location, use_cache)
        elif lat is not None and lon is not None:
            return self.fetch_weather_by_coordinates(lat, lon, use_cache)
        else:
            raise ValueError("Must provide either location or lat/lon coordinates")
    
    def fetch_forecast(
        self,
        city: str,
        hours: int = 24
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch weather forecast for upcoming hours.
        
        Args:
            city: City name
            hours: Number of hours to forecast (max 120, 5-day limit)
            
        Returns:
            List of forecast data points (3-hour intervals)
        """
        city = self._validate_city(city)
        
        # OpenWeatherMap returns 3-hour intervals, max 40 data points (5 days)
        cnt = min(hours // 3 + 1, 40)
        
        data = self._make_request(self.FORECAST_URL, {"q": city, "cnt": cnt})
        
        if data:
            return data.get("list", [])
        return None
    
    def analyze_weather_impact(
        self,
        weather_data: Optional[Dict[str, Any]]
    ) -> WeatherImpact:
        """
        Analyze weather data for traffic impact.
        
        Args:
            weather_data: Raw weather data from API
            
        Returns:
            WeatherImpact with analysis and recommendations
            
        Raises:
            ValueError: If weather data is invalid
        """
        if not weather_data:
            raise ValueError("Weather data is required")
        
        # Extract weather info
        weather_list = weather_data.get("weather", [{}])
        weather_info = weather_list[0] if weather_list else {}
        main_data = weather_data.get("main", {})
        wind_data = weather_data.get("wind", {})
        
        condition = weather_info.get("main", "Unknown")
        condition_code = weather_info.get("id", 0)
        description = weather_info.get("description", "")
        
        temperature = main_data.get("temp", 20.0)
        feels_like = main_data.get("feels_like", temperature)
        humidity = main_data.get("humidity", 50)
        
        wind_speed = wind_data.get("speed", 0.0)
        wind_gust = wind_data.get("gust")
        
        visibility = weather_data.get("visibility", 10000)
        
        rain_1h = weather_data.get("rain", {}).get("1h", 0.0)
        snow_1h = weather_data.get("snow", {}).get("1h", 0.0)
        
        # Determine severity
        severity = self._calculate_severity(
            condition, temperature, visibility, rain_1h, snow_1h, wind_speed
        )
        
        # Calculate speed reduction
        speed_reduction = self.SPEED_REDUCTION.get(severity, 0)
        
        # Adjust for specific conditions
        if visibility < 1000:
            speed_reduction = max(speed_reduction, 30)
        if wind_speed > 20:
            speed_reduction = max(speed_reduction, 20)
        if temperature < -10:
            speed_reduction = max(speed_reduction, 25)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, condition, temperature, visibility, rain_1h, snow_1h, wind_speed
        )
        
        return WeatherImpact(
            severity=severity,
            condition=condition,
            condition_code=condition_code,
            description=description.capitalize(),
            temperature=temperature,
            feels_like=feels_like,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_gust=wind_gust,
            visibility=visibility,
            precipitation_1h=rain_1h,
            snow_1h=snow_1h,
            recommendations=recommendations,
            traffic_speed_reduction_pct=speed_reduction
        )
    
    def _calculate_severity(
        self,
        condition: str,
        temperature: float,
        visibility: int,
        rain_1h: float,
        snow_1h: float,
        wind_speed: float
    ) -> WeatherSeverity:
        """Calculate overall weather severity for traffic."""
        # Start with condition-based severity
        base_severity = self.WEATHER_SEVERITY.get(
            condition.lower(),
            WeatherSeverity.CLEAR
        )
        
        severity_score = list(WeatherSeverity).index(base_severity)
        
        # Adjust for extreme temperatures
        if temperature < -15 or temperature > 40:
            severity_score = max(severity_score, 3)  # At least SEVERE
        elif temperature < 0:
            severity_score = max(severity_score, 2)  # At least MODERATE
        
        # Adjust for visibility
        if visibility < 200:
            severity_score = max(severity_score, 3)  # SEVERE
        elif visibility < 1000:
            severity_score = max(severity_score, 2)  # MODERATE
        
        # Adjust for heavy precipitation
        if rain_1h > 10 or snow_1h > 5:
            severity_score = max(severity_score, 3)  # SEVERE
        elif rain_1h > 5 or snow_1h > 2:
            severity_score = max(severity_score, 2)  # MODERATE
        
        # Adjust for high winds
        if wind_speed > 25:
            severity_score = max(severity_score, 3)  # SEVERE
        elif wind_speed > 15:
            severity_score = max(severity_score, 2)  # MODERATE
        
        # Cap at EXTREME
        severity_score = min(severity_score, 4)
        
        return list(WeatherSeverity)[severity_score]
    
    def _generate_recommendations(
        self,
        severity: WeatherSeverity,
        condition: str,
        temperature: float,
        visibility: int,
        rain_1h: float,
        snow_1h: float,
        wind_speed: float
    ) -> List[str]:
        """Generate traffic management recommendations based on weather."""
        recommendations = []
        
        # Severity-based recommendations
        if severity == WeatherSeverity.EXTREME:
            recommendations.extend([
                "Consider road closures in affected areas",
                "Activate emergency traffic protocols",
                "Issue severe weather alerts to all drivers"
            ])
        elif severity == WeatherSeverity.SEVERE:
            recommendations.extend([
                "Reduce speed limits on highways",
                "Increase signal timing for slower traffic",
                "Deploy additional emergency responders"
            ])
        elif severity == WeatherSeverity.MODERATE:
            recommendations.extend([
                "Monitor traffic speeds closely",
                "Prepare for potential incidents",
                "Consider advisory speed limits"
            ])
        
        # Condition-specific recommendations
        if snow_1h > 0:
            recommendations.append("Deploy snow plows and salt trucks")
            recommendations.append("Monitor for ice formation on bridges")
        
        if rain_1h > 5:
            recommendations.append("Watch for flooding in low-lying areas")
            recommendations.append("Increase following distance signage")
        
        if visibility < 1000:
            recommendations.append("Activate fog warning signs")
            recommendations.append("Reduce highway speed limits")
        
        if temperature < 0:
            recommendations.append("Monitor for black ice conditions")
            recommendations.append("Pre-treat bridges and overpasses")
        
        if wind_speed > 15:
            recommendations.append("Issue high-wind warnings for high-profile vehicles")
        
        return recommendations
    
    def get_traffic_impact(
        self,
        location: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> WeatherImpact:
        """
        Get weather impact analysis for a location.
        
        Convenience method that fetches weather and analyzes impact.
        
        Args:
            location: City name
            lat: Latitude (use with lon)
            lon: Longitude (use with lat)
            
        Returns:
            WeatherImpact with analysis and recommendations
        """
        weather_data = self.fetch_weather(location=location, lat=lat, lon=lon)
        
        if not weather_data:
            raise WeatherAgentError(
                f"Could not fetch weather for location: {location or f'({lat}, {lon})'}"
            )
        
        return self.analyze_weather_impact(weather_data)
    
    def get_forecast_impacts(
        self,
        city: str,
        hours: int = 24
    ) -> List[WeatherImpact]:
        """
        Get weather impact analysis for upcoming hours.
        
        Args:
            city: City name
            hours: Number of hours to forecast
            
        Returns:
            List of WeatherImpact for each forecast period
        """
        forecast = self.fetch_forecast(city, hours)
        
        if not forecast:
            return []
        
        impacts = []
        for period in forecast:
            # Convert forecast format to current weather format
            weather_data = {
                "weather": period.get("weather", []),
                "main": period.get("main", {}),
                "wind": period.get("wind", {}),
                "visibility": period.get("visibility", 10000),
                "rain": period.get("rain", {}),
                "snow": period.get("snow", {})
            }
            
            try:
                impact = self.analyze_weather_impact(weather_data)
                impact.forecast_time = datetime.fromtimestamp(period.get("dt", 0))
                impacts.append(impact)
            except Exception as e:
                self.logger.warning(f"Error analyzing forecast period: {e}")
                continue
        
        return impacts
    
    def clear_cache(self) -> None:
        """Clear the weather data cache."""
        self._cache.clear()
        self.logger.debug("Weather cache cleared")
