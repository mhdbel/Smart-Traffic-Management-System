import unittest

class MockTrafficAgent:
    def reason_and_act(self, origin, destination):
        return "Traffic is smooth"

class MockEventAgent:
    def fetch_event_data(self, city):
        return {"events": [{"name": "Test Event"}]}
    def analyze_event_impact(self, event_data):
        return "Upcoming events detected. Expect increased traffic."

class MockWeatherAgent:
    def fetch_weather_data(self, city):
        return {"weather": [{"main": "Clear"}], "main": {"temp": 20}}
    def analyze_weather_impact(self, weather_data):
        return "Weather conditions normal. No significant impact on traffic expected."

class MockRoutingAgent:
    def get_alternative_routes(self, origin, destination):
        return {"routes": [{"summary": "Route 1", "legs": [{"distance": {"text": "5 km"}, "duration": {"text": "10 mins"}}]}]}
    def analyze_routes(self, route_data):
        return [
            {"summary": "Route 1", "distance": "5 km", "duration": "10 mins"}
        ]

from Traffic_Management_System.src.agents.orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator(
            MockTrafficAgent(),
            MockEventAgent(),
            MockWeatherAgent(),
            MockRoutingAgent()
        )

    def test_run(self):
        result = self.orchestrator.run("A", "B", "TestCity")
        self.assertIn("traffic", result)
        self.assertIn("event", result)
        self.assertIn("weather", result)
        self.assertIn("routes", result)
        self.assertEqual(result["traffic"], "Traffic is smooth")
        self.assertEqual(result["event"], "Upcoming events detected. Expect increased traffic.")
        self.assertEqual(result["weather"], "Weather conditions normal. No significant impact on traffic expected.")
        self.assertEqual(result["routes"], [
            {"summary": "Route 1", "distance": "5 km", "duration": "10 mins"}
        ])

if __name__ == "__main__":
    unittest.main()
