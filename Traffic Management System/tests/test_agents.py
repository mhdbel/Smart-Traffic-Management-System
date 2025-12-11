# tests/test_agents.py
"""
Unit tests for the Smart Traffic Management System agents.

Tests cover:
- TrafficAgent: Traffic prediction and recommendations
- WeatherAgent: Weather data fetching and impact analysis
- EventAgent: Event data fetching and traffic impact
- RoutingAgent: Route calculation and optimization
- Orchestrator: Agent coordination

Uses mocking to avoid real API calls and ensure fast, reliable tests.
"""

import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import agents
from agents.traffic_agent import (
    TrafficAgent,
    TrafficFeatures,
    CongestionPrediction,
    TrafficRecommendation,
    CongestionLevel,
    ModelLoadError,
    TrafficDataError,
)
from agents.weather_agent import (
    WeatherAgent,
    WeatherImpact,
    WeatherSeverity,
    WeatherAgentError,
)
from agents.event_agent import (
    EventAgent,
    TrafficImpact as EventTrafficImpact,
    ImpactLevel,
)
from agents.routing_agent import (
    RoutingAgent,
    RouteSummary,
    RouteComparison,
    TrafficCondition,
)
from agents.orchestrator import (
    Orchestrator,
    OrchestrationResult,
    AgentResult,
    AgentStatus,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

class MockResponses:
    """Mock API responses for testing."""
    
    @staticmethod
    def google_maps_directions():
        """Mock Google Maps Directions API response."""
        return {
            "status": "OK",
            "routes": [
                {
                    "summary": "I-95 N",
                    "legs": [
                        {
                            "distance": {"text": "50 mi", "value": 80467},
                            "duration": {"text": "1 hour", "value": 3600},
                            "duration_in_traffic": {"text": "1 hour 15 min", "value": 4500},
                            "steps": [{"instruction": "Turn right"}]
                        }
                    ],
                    "warnings": [],
                    "waypoint_order": []
                },
                {
                    "summary": "US-1 N",
                    "legs": [
                        {
                            "distance": {"text": "55 mi", "value": 88514},
                            "duration": {"text": "1 hour 10 min", "value": 4200},
                            "duration_in_traffic": {"text": "1 hour 20 min", "value": 4800},
                            "steps": []
                        }
                    ],
                    "warnings": [],
                    "waypoint_order": []
                }
            ]
        }
    
    @staticmethod
    def openweather():
        """Mock OpenWeatherMap API response."""
        return {
            "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
            "main": {
                "temp": 25.5,
                "feels_like": 26.0,
                "humidity": 60
            },
            "wind": {"speed": 5.5, "gust": 8.0},
            "visibility": 10000,
            "rain": {},
            "snow": {},
            "cod": 200
        }
    
    @staticmethod
    def openweather_rain():
        """Mock OpenWeatherMap response with rain."""
        return {
            "weather": [{"id": 501, "main": "Rain", "description": "moderate rain"}],
            "main": {"temp": 18.0, "feels_like": 17.0, "humidity": 85},
            "wind": {"speed": 10.0},
            "visibility": 5000,
            "rain": {"1h": 5.5},
            "snow": {},
            "cod": 200
        }
    
    @staticmethod
    def eventbrite():
        """Mock Eventbrite API response."""
        return {
            "events": [
                {
                    "name": {"text": "Concert"},
                    "capacity": 5000,
                    "start": {"local": "2024-06-15T19:00:00"}
                },
                {
                    "name": {"text": "Sports Game"},
                    "capacity": 20000,
                    "start": {"local": "2024-06-15T14:00:00"}
                }
            ],
            "pagination": {"object_count": 2}
        }
    
    @staticmethod
    def eventbrite_empty():
        """Mock empty Eventbrite response."""
        return {"events": [], "pagination": {"object_count": 0}}


# =============================================================================
# TRAFFIC AGENT TESTS
# =============================================================================

class TestTrafficAgent(unittest.TestCase):
    """Tests for TrafficAgent."""
    
    @patch('agents.traffic_agent.joblib.load')
    @patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test_key'})
    def setUp(self, mock_load):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([0.65])
        self.mock_model.feature_names_in_ = [
            'duration_seconds', 'duration_in_traffic_seconds',
            'distance_meters', 'traffic_ratio', 'speed_kmh',
            'temperature', 'rain_1h', 'cloud_cover',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour'
        ]
        mock_load.return_value = self.mock_model
        
        # Mock routing agent
        self.mock_routing_agent = MagicMock()
        
        # Create agent
        self.agent = TrafficAgent(
            model_path=None,
            threshold=0.7,
            routing_agent=self.mock_routing_agent
        )
    
    def test_init_with_valid_model(self):
        """Test agent initialization with valid model."""
        self.assertIsNotNone(self.agent.model)
        self.assertEqual(self.agent.threshold, 0.7)
    
    @patch('agents.traffic_agent.joblib.load')
    def test_init_missing_model_raises_error(self, mock_load):
        """Test that missing model file raises ModelLoadError."""
        mock_load.side_effect = FileNotFoundError("Model not found")
        
        with self.assertRaises(ModelLoadError):
            TrafficAgent(model_path="/nonexistent/model.pkl")
    
    def test_invalid_threshold_raises_error(self):
        """Test that invalid threshold raises ValueError."""
        with self.assertRaises(ValueError):
            TrafficAgent(threshold=1.5)
        
        with self.assertRaises(ValueError):
            TrafficAgent(threshold=-0.1)
    
    def test_extract_features(self):
        """Test feature extraction from traffic data."""
        traffic_data = MockResponses.google_maps_directions()
        
        features = self.agent.extract_features(traffic_data)
        
        self.assertIsInstance(features, TrafficFeatures)
        self.assertEqual(features.duration_seconds, 3600)
        self.assertEqual(features.duration_in_traffic_seconds, 4500)
        self.assertEqual(features.distance_meters, 80467)
        self.assertAlmostEqual(features.traffic_ratio, 1.25, places=2)
    
    def test_extract_features_with_weather(self):
        """Test feature extraction with weather data."""
        traffic_data = MockResponses.google_maps_directions()
        weather_data = MockResponses.openweather()
        
        features = self.agent.extract_features(traffic_data, weather_data)
        
        self.assertEqual(features.temperature, 25.5)
        self.assertEqual(features.rain_1h, 0)
    
    def test_extract_features_invalid_data_raises_error(self):
        """Test that invalid traffic data raises error."""
        with self.assertRaises(TrafficDataError):
            self.agent.extract_features(None)
        
        with self.assertRaises(TrafficDataError):
            self.agent.extract_features({"status": "ZERO_RESULTS"})
    
    def test_predict_congestion(self):
        """Test congestion prediction."""
        features = TrafficFeatures(
            duration_seconds=3600,
            duration_in_traffic_seconds=4500,
            distance_meters=80000,
            traffic_ratio=1.25,
            speed_kmh=80.0,
            hour_of_day=17,
            day_of_week=2,
            is_weekend=False,
            is_rush_hour=True
        )
        
        prediction = self.agent.predict_congestion(features)
        
        self.assertIsInstance(prediction, CongestionPrediction)
        self.assertEqual(prediction.level, 0.65)
        self.assertFalse(prediction.is_congested)  # 0.65 < 0.7 threshold
    
    def test_predict_congestion_high_level(self):
        """Test high congestion prediction."""
        self.mock_model.predict.return_value = np.array([0.85])
        
        features = TrafficFeatures(
            duration_seconds=3600,
            duration_in_traffic_seconds=5400,
            distance_meters=80000,
            traffic_ratio=1.5,
            speed_kmh=60.0,
            hour_of_day=17,
            day_of_week=4,
            is_weekend=False,
            is_rush_hour=True
        )
        
        prediction = self.agent.predict_congestion(features)
        
        self.assertTrue(prediction.is_congested)
        self.assertEqual(prediction.congestion_level, CongestionLevel.SEVERE)
    
    @patch('agents.traffic_agent.requests.get')
    def test_reason_and_act(self, mock_get):
        """Test full reason_and_act flow."""
        # Mock routing agent response
        self.mock_routing_agent.fetch_route.return_value = MockResponses.google_maps_directions()
        
        result = self.agent.reason_and_act("New York", "Boston")
        
        self.assertIsInstance(result, TrafficRecommendation)
        self.assertIsNotNone(result.prediction)
        self.assertIsNotNone(result.recommended_action)
    
    def test_congestion_level_enum(self):
        """Test congestion level calculation."""
        # Test all levels
        test_cases = [
            (0.1, CongestionLevel.VERY_LOW),
            (0.3, CongestionLevel.LOW),
            (0.5, CongestionLevel.MODERATE),
            (0.7, CongestionLevel.HIGH),
            (0.9, CongestionLevel.SEVERE),
        ]
        
        for level, expected in test_cases:
            prediction = CongestionPrediction(
                level=level,
                confidence=None,
                threshold=0.7,
                is_congested=level > 0.7
            )
            self.assertEqual(prediction.congestion_level, expected)


# =============================================================================
# WEATHER AGENT TESTS
# =============================================================================

class TestWeatherAgent(unittest.TestCase):
    """Tests for WeatherAgent."""
    
    @patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'test_key'})
    def setUp(self):
        """Set up test fixtures."""
        self.agent = WeatherAgent()
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises ConfigurationError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key
            if 'OPENWEATHER_API_KEY' in os.environ:
                del os.environ['OPENWEATHER_API_KEY']
            
            from agents.weather_agent import ConfigurationError
            with self.assertRaises(ConfigurationError):
                WeatherAgent()
    
    def test_validate_city_valid(self):
        """Test city validation with valid input."""
        result = self.agent._validate_city("New York")
        self.assertEqual(result, "New York")
        
        result = self.agent._validate_city("  London  ")
        self.assertEqual(result, "London")
    
    def test_validate_city_invalid(self):
        """Test city validation with invalid input."""
        from agents.weather_agent import ValidationError
        
        with self.assertRaises(ValidationError):
            self.agent._validate_city("")
        
        with self.assertRaises(ValidationError):
            self.agent._validate_city("A")  # Too short
        
        with self.assertRaises(ValidationError):
            self.agent._validate_city("City<script>")  # Invalid chars
    
    @patch('agents.weather_agent.requests.get')
    def test_fetch_weather_data_success(self, mock_get):
        """Test successful weather data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.openweather()
        mock_get.return_value = mock_response
        
        result = self.agent.fetch_weather_data("London")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["main"]["temp"], 25.5)
    
    @patch('agents.weather_agent.requests.get')
    def test_fetch_weather_data_uses_cache(self, mock_get):
        """Test that cache is used for repeated requests."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.openweather()
        mock_get.return_value = mock_response
        
        # First call
        self.agent.fetch_weather_data("London")
        # Second call (should use cache)
        self.agent.fetch_weather_data("London")
        
        # Should only make one API call
        self.assertEqual(mock_get.call_count, 1)
    
    @patch('agents.weather_agent.requests.get')
    def test_fetch_weather_data_api_error(self, mock_get):
        """Test handling of API errors."""
        mock_get.side_effect = Exception("Network error")
        
        result = self.agent.fetch_weather_data("London")
        
        self.assertIsNone(result)
    
    def test_analyze_weather_impact_clear(self):
        """Test weather impact analysis for clear weather."""
        weather_data = MockResponses.openweather()
        
        impact = self.agent.analyze_weather_impact(weather_data)
        
        self.assertIsInstance(impact, WeatherImpact)
        self.assertEqual(impact.severity, WeatherSeverity.CLEAR)
        self.assertEqual(impact.condition, "Clear")
        self.assertEqual(impact.temperature, 25.5)
        self.assertFalse(impact.is_hazardous)
        self.assertEqual(impact.traffic_speed_reduction_pct, 0)
    
    def test_analyze_weather_impact_rain(self):
        """Test weather impact analysis for rainy weather."""
        weather_data = MockResponses.openweather_rain()
        
        impact = self.agent.analyze_weather_impact(weather_data)
        
        self.assertEqual(impact.severity, WeatherSeverity.MODERATE)
        self.assertEqual(impact.condition, "Rain")
        self.assertTrue(impact.has_precipitation)
        self.assertGreater(impact.traffic_speed_reduction_pct, 0)
        self.assertGreater(len(impact.recommendations), 0)
    
    def test_analyze_weather_impact_none_raises_error(self):
        """Test that None weather data raises error."""
        with self.assertRaises(ValueError):
            self.agent.analyze_weather_impact(None)
    
    @patch('agents.weather_agent.requests.get')
    def test_get_traffic_impact(self, mock_get):
        """Test convenience method get_traffic_impact."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.openweather()
        mock_get.return_value = mock_response
        
        impact = self.agent.get_traffic_impact(location="Paris")
        
        self.assertIsInstance(impact, WeatherImpact)


# =============================================================================
# EVENT AGENT TESTS
# =============================================================================

class TestEventAgent(unittest.TestCase):
    """Tests for EventAgent."""
    
    @patch.dict(os.environ, {'EVENTBRITE_API_KEY': 'test_key'})
    def setUp(self):
        """Set up test fixtures."""
        self.agent = EventAgent()
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            if 'EVENTBRITE_API_KEY' in os.environ:
                del os.environ['EVENTBRITE_API_KEY']
            
            with self.assertRaises(ValueError):
                EventAgent()
    
    @patch('agents.event_agent.requests.get')
    def test_fetch_event_data_success(self, mock_get):
        """Test successful event data fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.eventbrite()
        mock_get.return_value = mock_response
        
        result = self.agent.fetch_event_data("New York")
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["events"]), 2)
    
    @patch('agents.event_agent.requests.get')
    def test_fetch_event_data_with_cache(self, mock_get):
        """Test caching behavior."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.eventbrite()
        mock_get.return_value = mock_response
        
        # First call
        self.agent.fetch_event_data("Boston")
        # Second call
        self.agent.fetch_event_data("Boston")
        
        # Should only call API once due to caching
        self.assertEqual(mock_get.call_count, 1)
    
    def test_analyze_event_impact_with_events(self):
        """Test event impact analysis with events."""
        event_data = MockResponses.eventbrite()
        
        impact = self.agent.analyze_event_impact(event_data)
        
        self.assertIsInstance(impact, EventTrafficImpact)
        self.assertEqual(impact.event_count, 2)
        self.assertEqual(impact.total_expected_attendance, 25000)
        self.assertEqual(impact.level, ImpactLevel.HIGH)
        self.assertGreater(len(impact.recommendations), 0)
    
    def test_analyze_event_impact_no_events(self):
        """Test event impact analysis with no events."""
        event_data = MockResponses.eventbrite_empty()
        
        impact = self.agent.analyze_event_impact(event_data)
        
        self.assertEqual(impact.event_count, 0)
        self.assertEqual(impact.level, ImpactLevel.LOW)
    
    def test_analyze_event_impact_none(self):
        """Test event impact analysis with None data."""
        impact = self.agent.analyze_event_impact(None)
        
        self.assertEqual(impact.event_count, 0)
        self.assertEqual(impact.level, ImpactLevel.LOW)


# =============================================================================
# ROUTING AGENT TESTS
# =============================================================================

class TestRoutingAgent(unittest.TestCase):
    """Tests for RoutingAgent."""
    
    @patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test_key'})
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RoutingAgent()
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            if 'GOOGLE_MAPS_API_KEY' in os.environ:
                del os.environ['GOOGLE_MAPS_API_KEY']
            
            from agents.routing_agent import ConfigurationError
            with self.assertRaises(ConfigurationError):
                RoutingAgent()
    
    def test_validate_location_valid(self):
        """Test location validation with valid input."""
        result = self.agent._validate_location("New York, NY", "origin")
        self.assertEqual(result, "New York, NY")
    
    def test_validate_location_invalid(self):
        """Test location validation with invalid input."""
        from agents.routing_agent import ValidationError
        
        with self.assertRaises(ValidationError):
            self.agent._validate_location("", "origin")
        
        with self.assertRaises(ValidationError):
            self.agent._validate_location("A", "origin")
    
    def test_validate_avoid_valid(self):
        """Test avoid parameter validation."""
        result = self.agent._validate_avoid("tolls|highways")
        self.assertEqual(result, "tolls|highways")
        
        result = self.agent._validate_avoid("ferries")
        self.assertEqual(result, "ferries")
    
    def test_validate_avoid_invalid(self):
        """Test invalid avoid parameter."""
        from agents.routing_agent import ValidationError
        
        with self.assertRaises(ValidationError):
            self.agent._validate_avoid("invalid_option")
    
    @patch('agents.routing_agent.requests.get')
    def test_fetch_route_success(self, mock_get):
        """Test successful route fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.google_maps_directions()
        mock_get.return_value = mock_response
        
        result = self.agent.fetch_route("NYC", "Boston")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["routes"]), 2)
    
    @patch('agents.routing_agent.requests.get')
    def test_get_alternative_routes(self, mock_get):
        """Test fetching alternative routes."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.google_maps_directions()
        mock_get.return_value = mock_response
        
        result = self.agent.get_alternative_routes("NYC", "Boston")
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["routes"]), 2)
    
    def test_analyze_routes(self):
        """Test route analysis."""
        route_data = MockResponses.google_maps_directions()
        
        summaries = self.agent.analyze_routes(route_data)
        
        self.assertIsInstance(summaries, list)
        self.assertEqual(len(summaries), 2)
        
        # Check first route
        first_route = summaries[0]
        self.assertIsInstance(first_route, RouteSummary)
        self.assertEqual(first_route.summary, "I-95 N")
        self.assertEqual(first_route.distance_meters, 80467)
        self.assertEqual(first_route.duration_seconds, 3600)
    
    def test_analyze_routes_empty(self):
        """Test route analysis with empty data."""
        result = self.agent.analyze_routes(None)
        self.assertEqual(result, [])
        
        result = self.agent.analyze_routes({"status": "ZERO_RESULTS"})
        self.assertEqual(result, [])
    
    @patch('agents.routing_agent.requests.get')
    def test_compare_routes(self, mock_get):
        """Test route comparison."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = MockResponses.google_maps_directions()
        mock_get.return_value = mock_response
        
        comparison = self.agent.compare_routes("NYC", "Boston", preference="fastest")
        
        self.assertIsInstance(comparison, RouteComparison)
        self.assertEqual(len(comparison.routes), 2)
        self.assertIsNotNone(comparison.recommended_route)
    
    def test_route_summary_traffic_condition(self):
        """Test traffic condition calculation."""
        # Light traffic (< 10% delay)
        summary = RouteSummary(
            route_index=0,
            summary="Test",
            distance_text="10 mi",
            distance_meters=16000,
            duration_text="20 min",
            duration_seconds=1200,
            duration_in_traffic_seconds=1250,  # ~4% delay
            duration_in_traffic_text="21 min",
            warnings=[],
            traffic_condition="light"
        )
        self.assertEqual(summary.traffic_condition, "light")


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

class TestOrchestrator(unittest.TestCase):
    """Tests for Orchestrator."""
    
    def setUp(self):
        """Set up test fixtures with mocked agents."""
        self.mock_traffic = MagicMock()
        self.mock_weather = MagicMock()
        self.mock_event = MagicMock()
        self.mock_routing = MagicMock()
        
        self.orchestrator = Orchestrator(
            traffic_agent=self.mock_traffic,
            event_agent=self.mock_event,
            weather_agent=self.mock_weather,
            routing_agent=self.mock_routing
        )
    
    def test_init_with_none_agent_raises_error(self):
        """Test that None agents raise ValueError."""
        with self.assertRaises(ValueError):
            Orchestrator(
                traffic_agent=None,
                event_agent=self.mock_event,
                weather_agent=self.mock_weather,
                routing_agent=self.mock_routing
            )
    
    def test_validate_location(self):
        """Test location validation."""
        result = self.orchestrator._validate_location("New York", "origin")
        self.assertEqual(result, "New York")
        
        from agents.orchestrator import ValidationError
        with self.assertRaises(ValidationError):
            self.orchestrator._validate_location("", "origin")
    
    def test_run_success(self):
        """Test successful orchestration run."""
        # Setup mock responses
        self.mock_traffic.reason_and_act.return_value = MagicMock(
            to_dict=lambda: {"congestion_level": "low", "is_congested": False}
        )
        self.mock_weather.fetch_weather_data.return_value = MockResponses.openweather()
        self.mock_weather.analyze_weather_impact.return_value = MagicMock(
            to_dict=lambda: {"severity": "clear", "condition": "Clear"}
        )
        self.mock_event.fetch_event_data.return_value = MockResponses.eventbrite()
        self.mock_event.analyze_event_impact.return_value = MagicMock(
            to_dict=lambda: {"level": "medium", "event_count": 2}
        )
        self.mock_routing.get_alternative_routes.return_value = MockResponses.google_maps_directions()
        self.mock_routing.analyze_routes.return_value = [
            MagicMock(summary="I-95 N", duration_text="1 hour")
        ]
        
        result = self.orchestrator.run("NYC", "Boston", "NYC")
        
        self.assertIsInstance(result, OrchestrationResult)
        self.assertEqual(result.traffic.status, AgentStatus.SUCCESS)
        self.assertEqual(result.weather.status, AgentStatus.SUCCESS)
    
    def test_run_with_agent_failure(self):
        """Test orchestration continues when one agent fails."""
        # Traffic agent fails
        self.mock_traffic.reason_and_act.side_effect = Exception("API Error")
        
        # Other agents succeed
        self.mock_weather.fetch_weather_data.return_value = MockResponses.openweather()
        self.mock_weather.analyze_weather_impact.return_value = MagicMock(
            to_dict=lambda: {"severity": "clear"}
        )
        self.mock_event.fetch_event_data.return_value = None
        self.mock_event.analyze_event_impact.return_value = MagicMock(
            to_dict=lambda: {"level": "low"}
        )
        self.mock_routing.get_alternative_routes.return_value = None
        self.mock_routing.analyze_routes.return_value = []
        
        result = self.orchestrator.run("NYC", "Boston", "NYC")
        
        # Should still return result
        self.assertIsInstance(result, OrchestrationResult)
        self.assertEqual(result.traffic.status, AgentStatus.FAILED)
        self.assertIsNotNone(result.traffic.error)
    
    def test_run_quick_backward_compatible(self):
        """Test run_quick returns simple dict."""
        self.mock_traffic.reason_and_act.return_value = "Traffic is moderate"
        self.mock_weather.fetch_weather_data.return_value = None
        self.mock_weather.analyze_weather_impact.return_value = "No data"
        self.mock_event.fetch_event_data.return_value = None
        self.mock_event.analyze_event_impact.return_value = "No events"
        self.mock_routing.get_alternative_routes.return_value = None
        self.mock_routing.analyze_routes.return_value = []
        
        result = self.orchestrator.run_quick("NYC", "Boston", "NYC")
        
        self.assertIsInstance(result, dict)
        self.assertIn("traffic", result)
    
    def test_health_check(self):
        """Test health check returns agent status."""
        health = self.orchestrator.health_check()
        
        self.assertIn("traffic_agent", health)
        self.assertIn("event_agent", health)
        self.assertIn("weather_agent", health)
        self.assertIn("routing_agent", health)
        self.assertTrue(all(health.values()))
    
    def test_caching(self):
        """Test that results are cached."""
        self.mock_traffic.reason_and_act.return_value = MagicMock(
            to_dict=lambda: {"level": "low"}
        )
        self.mock_weather.fetch_weather_data.return_value = MockResponses.openweather()
        self.mock_weather.analyze_weather_impact.return_value = MagicMock(
            to_dict=lambda: {"severity": "clear"}
        )
        self.mock_event.fetch_event_data.return_value = None
        self.mock_event.analyze_event_impact.return_value = MagicMock(to_dict=lambda: {})
        self.mock_routing.get_alternative_routes.return_value = None
        self.mock_routing.analyze_routes.return_value = []
        
        # First call
        self.orchestrator.run("NYC", "Boston", "NYC", use_cache=True)
        # Second call (should use cache)
        self.orchestrator.run("NYC", "Boston", "NYC", use_cache=True)
        
        # Traffic agent should only be called once
        self.assertEqual(self.mock_traffic.reason_and_act.call_count, 1)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agent interactions."""
    
    def setUp(self):
        """Set up with all mocked agents."""
        self.mock_traffic = MagicMock()
        self.mock_weather = MagicMock()
        self.mock_event = MagicMock()
        self.mock_routing = MagicMock()
    
    def test_orchestrator_combines_all_agents(self):
        """Test that orchestrator properly combines all agent results."""
        # Setup comprehensive mock responses
        self.mock_traffic.reason_and_act.return_value = MagicMock(
            to_dict=lambda: {
                "congestion_level": "high",
                "is_congested": True,
                "recommended_action": "Use alternative route",
                "estimated_delay_minutes": 25
            }
        )
        
        self.mock_weather.fetch_weather_data.return_value = MockResponses.openweather_rain()
        self.mock_weather.analyze_weather_impact.return_value = MagicMock(
            to_dict=lambda: {
                "severity": "moderate",
                "is_hazardous": False,
                "recommendations": ["Drive carefully in rain"]
            }
        )
        
        self.mock_event.fetch_event_data.return_value = MockResponses.eventbrite()
        self.mock_event.analyze_event_impact.return_value = MagicMock(
            to_dict=lambda: {
                "level": "high",
                "event_count": 2,
                "recommendations": ["Expect delays near event venues"]
            }
        )
        
        self.mock_routing.get_alternative_routes.return_value = MockResponses.google_maps_directions()
        self.mock_routing.analyze_routes.return_value = [
            {"summary": "I-95 N", "duration": "1h 15m"},
            {"summary": "US-1 N", "duration": "1h 20m"}
        ]
        
        orchestrator = Orchestrator(
            traffic_agent=self.mock_traffic,
            event_agent=self.mock_event,
            weather_agent=self.mock_weather,
            routing_agent=self.mock_routing
        )
        
        result = orchestrator.run("NYC", "Boston", "NYC")
        
        # Verify all agents were called
        self.mock_traffic.reason_and_act.assert_called_once()
        self.mock_weather.fetch_weather_data.assert_called_once()
        self.mock_event.fetch_event_data.assert_called_once()
        self.mock_routing.get_alternative_routes.assert_called_once()
        
        # Verify result structure
        self.assertEqual(result.successful_agents, 4)
        self.assertEqual(result.overall_severity, "high")


# =============================================================================
# PYTEST FIXTURES (for pytest users)
# =============================================================================

@pytest.fixture
def mock_traffic_agent():
    """Pytest fixture for mocked traffic agent."""
    with patch('agents.traffic_agent.joblib.load') as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        mock_model.feature_names_in_ = ['hour', 'day_of_week']
        mock_load.return_value = mock_model
        
        with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test'}):
            yield TrafficAgent()


@pytest.fixture
def mock_weather_agent():
    """Pytest fixture for mocked weather agent."""
    with patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'test'}):
        yield WeatherAgent()


@pytest.fixture  
def mock_event_agent():
    """Pytest fixture for mocked event agent."""
    with patch.dict(os.environ, {'EVENTBRITE_API_KEY': 'test'}):
        yield EventAgent()


@pytest.fixture
def mock_routing_agent():
    """Pytest fixture for mocked routing agent."""
    with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test'}):
        yield RoutingAgent()


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    # Run with unittest
    unittest.main(verbosity=2)
