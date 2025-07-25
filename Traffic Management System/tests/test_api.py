import unittest
from flask import Flask
from flask.testing import FlaskClient

from Traffic_Management_System.src.api import create_app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()

    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Smart Traffic Management System", response.data)

    def test_home_post_not_allowed(self):
        response = self.client.post('/')
        self.assertEqual(response.status_code, 405)

    def test_predict_valid(self):
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("traffic", data)
        self.assertIn("event", data)
        self.assertIn("weather", data)
        self.assertIn("routes", data)

    def test_predict_missing_origin(self):
        payload = {"destination": "LocationB", "city": "TestCity"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_destination(self):
        payload = {"origin": "LocationA", "city": "TestCity"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_city(self):
        payload = {"origin": "LocationA", "destination": "LocationB"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_payload(self):
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, 400)

    def test_predict_malformed_json(self):
        response = self.client.post('/predict', data="not a json", content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_types(self):
        payload = {"origin": 123, "destination": ["A", "B"], "city": {"name": "TestCity"}}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_nonexistent_city(self):
        payload = {"origin": "LocationA", "destination": "LocationB", "city": "NoSuchCity"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)  # Should still return 200 but with fallback data
        data = response.get_json()
        self.assertIn("traffic", data)
        self.assertIn("event", data)
        self.assertIn("weather", data)
        self.assertIn("routes", data)

    def test_predict_get_not_allowed(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 405)

    def test_predict_wrong_content_type(self):
        payload = "origin=LocationA&destination=LocationB&city=TestCity"
        response = self.client.post('/predict', data=payload, content_type='text/plain')
        self.assertEqual(response.status_code, 400)

    # You can add more tests for rate limiting, authentication, or error propagation as needed.

if __name__ == '__main__':
    unittest.main()
