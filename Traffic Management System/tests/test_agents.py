import unittest
from src.agents.traffic_agent import TrafficAgent
from src.agents.weather_agent import WeatherAgent
from src.agents.event_agent import EventAgent
from src.agents.routing_agent import RoutingAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.traffic_agent = TrafficAgent()
        self.weather_agent = WeatherAgent()
        self.event_agent = EventAgent()
        self.routing_agent = RoutingAgent()

    def test_traffic_agent_prediction(self):
        result = self.traffic_agent.analyze_traffic("LocationA", "LocationB")
        self.assertIsInstance(result, dict)
        self.assertIn("congestion_level", result)

    def test_weather_agent_forecast(self):
        result = self.weather_agent.get_weather("TestCity")
        self.assertIsInstance(result, dict)
        self.assertIn("temperature", result)

    def test_event_agent_schedule(self):
        result = self.event_agent.get_events("TestCity")
        self.assertIsInstance(result, list)

    def test_routing_agent_optimization(self):
        result = self.routing_agent.optimize_route("LocationA", "LocationB")
        self.assertIsInstance(result, dict)
        self.assertIn("best_route", result)

if __name__ == '__main__':
    unittest.main()