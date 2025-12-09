# src/agents/traffic_agent.py
"""
Traffic Agent for congestion prediction and traffic management recommendations.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# Import sibling agents (assumes they're in same package)
try:
    from .routing_agent import RoutingAgent
    from .weather_agent import WeatherAgent
except ImportError:
    RoutingAgent = None
    WeatherAgent = None


class CongestionLevel(Enum):
    """Traffic congestion severity levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class TrafficAgentError(Exception):
    """Base exception for TrafficAgent errors."""
    pass


class ModelLoadError(TrafficAgentError):
    """Raised when model fails to load."""
    pass


class TrafficDataError(TrafficAgentError):
    """Raised when traffic data cannot be fetched or processed."""
    pass


class ConfigurationError(TrafficAgentError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class TrafficFeatures:
    """Features extracted for congestion prediction."""
    # Core traffic metrics
    duration_seconds: int
    duration_in_traffic_seconds: Optional[int]
    distance_meters: int
    
    # Derived features
    traffic_ratio: float
    speed_kmh: float
    
    # Weather features
    temperature: Optional[float] = None
    rain_1h: Optional[float] = None
    cloud_cover: Optional[int] = None
    
    # Time features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_rush_hour: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input."""
        return {
            "duration_seconds": self.duration_seconds,
            "duration_in_traffic_seconds": self.duration_in_traffic_seconds or self.duration_seconds,
            "distance_meters": self.distance_meters,
            "traffic_ratio": self.traffic_ratio,
            "speed_kmh": self.speed_kmh,
            "temperature": self.temperature if self.temperature is not None else 20.0,
            "rain_1h": self.rain_1h if self.rain_1h is not None else 0.0,
            "cloud_cover": self.cloud_cover if self.cloud_cover is not None else 50,
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": int(self.is_weekend),
            "is_rush_hour": int(self.is_rush_hour)
        }


@dataclass
class CongestionPrediction:
    """Congestion prediction result."""
    level: float
    confidence: Optional[float]
    threshold: float
    is_congested: bool
    
    @property
    def congestion_level(self) -> CongestionLevel:
        """Get categorical congestion level."""
        if self.level < 0.2:
            return CongestionLevel.VERY_LOW
        elif self.level < 0.4:
            return CongestionLevel.LOW
        elif self.level < 0.6:
            return CongestionLevel.MODERATE
        elif self.level < 0.8:
            return CongestionLevel.HIGH
        else:
            return CongestionLevel.SEVERE
    
    @property
    def confidence_description(self) -> str:
        """Human-readable confidence level."""
        if self.confidence is None:
            return "unknown"
        elif self.confidence >= 0.9:
            return "high"
        elif self.confidence >= 0.7:
            return "medium"
        else:
            return "low"


@dataclass
class TrafficRecommendation:
    """Complete traffic recommendation with actions."""
    origin: str
    destination: str
    prediction: CongestionPrediction
    features: TrafficFeatures
    recommended_action: str
    alternative_routes_suggested: bool
    signal_timing_adjustment: Optional[str]
    estimated_delay_minutes: int
    confidence_level: str
    factors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "origin": self.origin,
            "destination": self.destination,
            "congestion_level": self.prediction.congestion_level.value,
            "congestion_score": round(self.prediction.level, 3),
            "is_congested": self.prediction.is_congested,
            "recommended_action": self.recommended_action,
            "alternative_routes_suggested": self.alternative_routes_suggested,
            "signal_timing_adjustment": self.signal_timing_adjustment,
            "estimated_delay_minutes": self.estimated_delay_minutes,
            "confidence": self.confidence_level,
            "contributing_factors": self.factors,
            "timestamp": self.timestamp.isoformat()
        }


class TrafficAgent:
    """
    Agent for traffic congestion prediction and management recommendations.
    
    Uses machine learning model trained on historical traffic data to predict
    congestion and provide actionable recommendations.
    
    Attributes:
        RUSH_HOURS: Time ranges considered rush hour
        EXPECTED_FEATURES: Features expected by the ML model
    
    Example:
        >>> agent = TrafficAgent()
        >>> recommendation = agent.reason_and_act("New York, NY", "Boston, MA")
        >>> print(recommendation.recommended_action)
    """
    
    RUSH_HOURS = {
        "morning": (7, 9),
        "evening": (16, 19)
    }
    
    EXPECTED_FEATURES = [
        "duration_seconds",
        "duration_in_traffic_seconds",
        "distance_meters",
        "traffic_ratio",
        "speed_kmh",
        "temperature",
        "rain_1h",
        "cloud_cover",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_rush_hour"
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.7,
        routing_agent: Optional['RoutingAgent'] = None,
        weather_agent: Optional['WeatherAgent'] = None,
        log_predictions: Optional[bool] = None
    ):
        """
        Initialize TrafficAgent.
        
        Args:
            model_path: Path to trained model file
            threshold: Congestion threshold (0.0-1.0)
            routing_agent: Optional RoutingAgent instance for data fetching
            weather_agent: Optional WeatherAgent instance for weather data
            log_predictions: Whether to log predictions for monitoring
            
        Raises:
            ModelLoadError: If model cannot be loaded
            ValueError: If threshold is invalid
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold
        
        # Resolve model path
        if model_path is None:
            model_path = os.getenv(
                "TRAFFIC_MODEL_PATH",
                str(Path(__file__).parent.parent.parent / "models" / "traffic_model.pkl")
            )
        self.model_path = Path(model_path)
        
        # Load model
        self.model = self._load_model()
        self._validate_model()
        
        # Initialize dependent agents
        self.routing_agent = routing_agent
        self.weather_agent = weather_agent
        
        # Lazy initialization of routing agent if needed
        if self.routing_agent is None and RoutingAgent is not None:
            try:
                self.routing_agent = RoutingAgent()
            except Exception as e:
                self.logger.warning(f"Could not initialize RoutingAgent: {e}")
        
        # Prediction logging
        self._log_predictions = (
            log_predictions if log_predictions is not None
            else os.getenv("LOG_PREDICTIONS", "false").lower() == "true"
        )
        self._prediction_log: List[dict] = []
    
    def _load_model(self) -> Any:
        """Load the ML model from disk."""
        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model file not found: {self.model_path}. "
                "Run model training first: python notebooks/model-training.py"
            )
        
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            model = joblib.load(self.model_path)
            self.logger.info(f"Model loaded: {type(model).__name__}")
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}") from e
    
    def _validate_model(self) -> None:
        """Validate loaded model has required interface."""
        if not hasattr(self.model, 'predict'):
            raise ModelLoadError("Model must have 'predict' method")
        
        if hasattr(self.model, 'feature_names_in_'):
            model_features = list(self.model.feature_names_in_)
            self.logger.info(f"Model expects features: {model_features}")
            
            missing = set(self.EXPECTED_FEATURES) - set(model_features)
            extra = set(model_features) - set(self.EXPECTED_FEATURES)
            
            if missing or extra:
                self.logger.warning(
                    f"Feature mismatch - Missing: {missing}, Extra: {extra}"
                )
    
    def reload_model(self, model_path: Optional[str] = None) -> None:
        """
        Reload model from disk.
        
        Args:
            model_path: Optional new path, uses current path if not provided
        """
        if model_path:
            self.model_path = Path(model_path)
        
        self.logger.info(f"Reloading model from {self.model_path}")
        self.model = self._load_model()
        self._validate_model()
    
    def fetch_traffic_data(
        self, 
        origin: str, 
        destination: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch traffic data between origin and destination.
        
        Args:
            origin: Starting location
            destination: End location
            
        Returns:
            Traffic data from API or None
        """
        if self.routing_agent:
            return self.routing_agent.fetch_route(
                origin,
                destination,
                departure_time=int(datetime.now().timestamp())
            )
        
        # Fallback to direct API call
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ConfigurationError("GOOGLE_MAPS_API_KEY not set")
        
        import requests
        
        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params={
                    "origin": origin,
                    "destination": destination,
                    "departure_time": "now",
                    "key": api_key
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Error fetching traffic data: {e}")
            return None
    
    def fetch_weather_data(self, location: str) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data for a location.
        
        Args:
            location: Location to get weather for
            
        Returns:
            Weather data or None
        """
        if self.weather_agent:
            try:
                return self.weather_agent.fetch_weather(location)
            except Exception as e:
                self.logger.warning(f"Could not fetch weather: {e}")
        return None
    
    def extract_features(
        self,
        traffic_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None
    ) -> TrafficFeatures:
        """
        Extract features from traffic and weather data.
        
        Args:
            traffic_data: Response from Google Maps Directions API
            weather_data: Optional response from weather API
            
        Returns:
            TrafficFeatures object
            
        Raises:
            TrafficDataError: If traffic data is invalid
        """
        if not traffic_data:
            raise TrafficDataError("Traffic data is empty")
        
        if traffic_data.get("status") != "OK":
            raise TrafficDataError(
                f"Invalid traffic data status: {traffic_data.get('status')}"
            )
        
        routes = traffic_data.get("routes", [])
        if not routes:
            raise TrafficDataError("No routes in traffic data")
        
        leg = routes[0].get("legs", [{}])[0]
        
        # Extract core metrics
        duration = leg.get("duration", {}).get("value", 0)
        duration_traffic = leg.get("duration_in_traffic", {}).get("value")
        distance = leg.get("distance", {}).get("value", 0)
        
        # Calculate derived features
        traffic_ratio = (
            duration_traffic / duration 
            if duration and duration_traffic 
            else 1.0
        )
        speed_kmh = (
            (distance / 1000) / (duration / 3600) 
            if duration > 0 
            else 0.0
        )
        
        # Time features
        now = datetime.now()
        hour = now.hour
        day = now.weekday()
        is_weekend = day >= 5
        is_rush_hour = any(
            start <= hour <= end
            for start, end in self.RUSH_HOURS.values()
        )
        
        # Weather features
        temp = rain = clouds = None
        if weather_data:
            main = weather_data.get("main", {})
            temp = main.get("temp")
            clouds = weather_data.get("clouds", {}).get("all")
            rain = weather_data.get("rain", {}).get("1h", 0)
        
        return TrafficFeatures(
            duration_seconds=duration,
            duration_in_traffic_seconds=duration_traffic,
            distance_meters=distance,
            traffic_ratio=traffic_ratio,
            speed_kmh=round(speed_kmh, 2),
            temperature=temp,
            rain_1h=rain,
            cloud_cover=clouds,
            hour_of_day=hour,
            day_of_week=day,
            is_weekend=is_weekend,
            is_rush_hour=is_rush_hour
        )
    
    def predict_congestion(self, features: TrafficFeatures) -> CongestionPrediction:
        """
        Predict congestion level using the loaded model.
        
        Args:
            features: Extracted traffic features
            
        Returns:
            CongestionPrediction with level and confidence
        """
        feature_dict = features.to_dict()
        
        # Build input DataFrame with correct feature order
        try:
            input_data = pd.DataFrame([feature_dict])
            # Reorder to match expected features
            available_features = [f for f in self.EXPECTED_FEATURES if f in input_data.columns]
            input_data = input_data[available_features]
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            raise
        
        # Get prediction
        prediction = self.model.predict(input_data)
        pred_value = float(prediction[0]) if hasattr(prediction, "__getitem__") else float(prediction)
        
        # Get probability if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(input_data)[0]
                confidence = float(max(probabilities))
            except Exception:
                pass
        
        result = CongestionPrediction(
            level=pred_value,
            confidence=confidence,
            threshold=self.threshold,
            is_congested=pred_value > self.threshold
        )
        
        # Log prediction if enabled
        if self._log_predictions:
            self._log_prediction(features, result)
        
        return result
    
    def _identify_factors(
        self,
        features: TrafficFeatures,
        prediction: CongestionPrediction
    ) -> List[str]:
        """Identify factors contributing to congestion."""
        factors = []
        
        if features.is_rush_hour:
            factors.append("Rush hour traffic")
        
        if features.traffic_ratio > 1.5:
            pct = int((features.traffic_ratio - 1) * 100)
            factors.append(f"Traffic {pct}% slower than normal")
        
        if features.rain_1h and features.rain_1h > 0:
            factors.append(f"Rain affecting conditions ({features.rain_1h}mm/h)")
        
        if features.speed_kmh < 20:
            factors.append(f"Very slow speed: {features.speed_kmh:.1f} km/h")
        
        if not features.is_weekend and features.day_of_week in [0, 4]:
            factors.append("Monday/Friday typically busier")
        
        if features.cloud_cover and features.cloud_cover > 80:
            factors.append("Overcast conditions may affect visibility")
        
        return factors
    
    def _generate_action(
        self,
        prediction: CongestionPrediction,
        features: TrafficFeatures
    ) -> tuple:
        """Generate recommended action and signal adjustment."""
        level = prediction.congestion_level
        
        actions = {
            CongestionLevel.VERY_LOW: (
                "Traffic is clear. Proceed normally.",
                None
            ),
            CongestionLevel.LOW: (
                "Light traffic. Normal travel time expected.",
                None
            ),
            CongestionLevel.MODERATE: (
                "Moderate traffic. Consider leaving 10-15 minutes earlier.",
                "Extend green light by 10%"
            ),
            CongestionLevel.HIGH: (
                "Heavy traffic. Alternative routes recommended. Expect 20-30 min delays.",
                "Extend green light by 25%, activate adaptive timing"
            ),
            CongestionLevel.SEVERE: (
                "Severe congestion. Delay travel if possible or use alternatives. 45+ min delays.",
                "Maximum green time, coordinate adjacent intersections"
            )
        }
        
        return actions.get(level, ("Unable to determine action", None))
    
    def _estimate_delay(self, features: TrafficFeatures) -> int:
        """Estimate delay in minutes."""
        if features.duration_in_traffic_seconds and features.duration_seconds:
            delay = features.duration_in_traffic_seconds - features.duration_seconds
            return max(0, delay // 60)
        return 0
    
    def reason_and_act(
        self,
        origin: str,
        destination: str
    ) -> TrafficRecommendation:
        """
        Analyze traffic and provide comprehensive recommendations.
        
        Args:
            origin: Starting location
            destination: End location
            
        Returns:
            TrafficRecommendation with detailed analysis
            
        Raises:
            TrafficDataError: If traffic data cannot be fetched
        """
        # Fetch traffic data
        traffic_data = self.fetch_traffic_data(origin, destination)
        if not traffic_data:
            raise TrafficDataError(f"Unable to fetch traffic data for {origin} -> {destination}")
        
        # Fetch weather data (optional)
        weather_data = self.fetch_weather_data(origin)
        
        # Extract features
        features = self.extract_features(traffic_data, weather_data)
        
        # Predict congestion
        prediction = self.predict_congestion(features)
        
        # Generate insights
        factors = self._identify_factors(features, prediction)
        action, signal_adj = self._generate_action(prediction, features)
        delay = self._estimate_delay(features)
        
        return TrafficRecommendation(
            origin=origin,
            destination=destination,
            prediction=prediction,
            features=features,
            recommended_action=action,
            alternative_routes_suggested=prediction.is_congested,
            signal_timing_adjustment=signal_adj,
            estimated_delay_minutes=delay,
            confidence_level=prediction.confidence_description,
            factors=factors
        )
    
    def _log_prediction(
        self,
        features: TrafficFeatures,
        prediction: CongestionPrediction
    ) -> None:
        """Log prediction for monitoring."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "features": features.to_dict(),
            "prediction": prediction.level,
            "confidence": prediction.confidence,
            "is_congested": prediction.is_congested,
            "congestion_level": prediction.congestion_level.value
        }
        self._prediction_log.append(entry)
        
        if len(self._prediction_log) >= 100:
            self._flush_prediction_log()
    
    def _flush_prediction_log(self) -> None:
        """Flush prediction log to storage."""
        if not self._prediction_log:
            return
        
        log_dir = Path("logs/predictions")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        filename = log_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(filename, 'w') as f:
            for entry in self._prediction_log:
                f.write(json.dumps(entry) + "\n")
        
        self.logger.info(f"Flushed {len(self._prediction_log)} predictions to {filename}")
        self._prediction_log.clear()
    
    def predict_batch(
        self,
        features_list: List[TrafficFeatures]
    ) -> List[CongestionPrediction]:
        """
        Predict congestion for multiple feature sets efficiently.
        
        Args:
            features_list: List of TrafficFeatures objects
            
        Returns:
            List of CongestionPrediction results
        """
        if not features_list:
            return []
        
        input_data = pd.DataFrame([f.to_dict() for f in features_list])
        available_features = [f for f in self.EXPECTED_FEATURES if f in input_data.columns]
        input_data = input_data[available_features]
        
        predictions = self.model.predict(input_data)
        
        confidences = [None] * len(predictions)
        if hasattr(self.model, 'predict_proba'):
            try:
                probabilities = self.model.predict_proba(input_data)
                confidences = [float(max(p)) for p in probabilities]
            except Exception:
                pass
        
        return [
            CongestionPrediction(
                level=float(pred),
                confidence=conf,
                threshold=self.threshold,
                is_congested=float(pred) > self.threshold
            )
            for pred, conf in zip(predictions, confidences)
        ]
