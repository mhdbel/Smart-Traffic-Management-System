import unittest
from src.api.app import app

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_full_flow(self):
        # Simulate a prediction request as a user would make
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("traffic", data)
        self.assertIn("weather", data)
        self.assertIn("event", data)
        self.assertIn("routes", data)

if __name__ == '__main__':
    unittest.main()