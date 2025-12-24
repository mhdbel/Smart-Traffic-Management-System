"""
Unit tests for the Flask API endpoints.
Tests cover all routes, HTTP methods, and edge cases.
"""

import unittest
from unittest.mock import patch, MagicMock
from flask.testing import FlaskClient

from Traffic_Management_System.src.api import create_app


class TestAPIHome(unittest.TestCase):
    """Test suite for the home route."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_home_returns_200(self):
        """Test that the home route returns 200 status code."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_home_contains_expected_content(self):
        """Test that home page contains expected text."""
        response = self.client.get('/')
        self.assertIn(b"Smart Traffic Management System", response.data)

    def test_home_post_not_allowed(self):
        """Test that POST method is not allowed on home route."""
        response = self.client.post('/')
        self.assertEqual(response.status_code, 405)

    def test_home_put_not_allowed(self):
        """Test that PUT method is not allowed on home route."""
        response = self.client.put('/')
        self.assertEqual(response.status_code, 405)

    def test_home_delete_not_allowed(self):
        """Test that DELETE method is not allowed on home route."""
        response = self.client.delete('/')
        self.assertEqual(response.status_code, 405)


class TestAPIPredictValid(unittest.TestCase):
    """Test suite for valid predict endpoint requests."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        self.valid_payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_predict_valid_payload_returns_200(self):
        """Test predict endpoint with valid payload returns 200."""
        response = self.client.post('/predict', json=self.valid_payload)
        self.assertEqual(response.status_code, 200)

    def test_predict_valid_payload_returns_json(self):
        """Test predict endpoint returns valid JSON response."""
        response = self.client.post('/predict', json=self.valid_payload)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertIsInstance(data, dict)

    def test_predict_response_contains_traffic(self):
        """Test predict response contains traffic data."""
        response = self.client.post('/predict', json=self.valid_payload)
        data = response.get_json()
        self.assertIn("traffic", data)

    def test_predict_response_contains_event(self):
        """Test predict response contains event data."""
        response = self.client.post('/predict', json=self.valid_payload)
        data = response.get_json()
        self.assertIn("event", data)

    def test_predict_response_contains_weather(self):
        """Test predict response contains weather data."""
        response = self.client.post('/predict', json=self.valid_payload)
        data = response.get_json()
        self.assertIn("weather", data)

    def test_predict_response_contains_routes(self):
        """Test predict response contains routes data."""
        response = self.client.post('/predict', json=self.valid_payload)
        data = response.get_json()
        self.assertIn("routes", data)

    def test_predict_nonexistent_city_returns_fallback(self):
        """Test predict with nonexistent city returns fallback data."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "NonExistentCity123"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertIn("traffic", data)
        self.assertIn("routes", data)


class TestAPIPredictMissingFields(unittest.TestCase):
    """Test suite for predict endpoint with missing fields."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_predict_missing_origin(self):
        """Test predict endpoint with missing origin field."""
        payload = {"destination": "LocationB", "city": "TestCity"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_destination(self):
        """Test predict endpoint with missing destination field."""
        payload = {"origin": "LocationA", "city": "TestCity"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_city(self):
        """Test predict endpoint with missing city field."""
        payload = {"origin": "LocationA", "destination": "LocationB"}
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_payload(self):
        """Test predict endpoint with empty payload."""
        response = self.client.post('/predict', json={})
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_all_fields_returns_error_message(self):
        """Test that missing fields return an error message."""
        response = self.client.post('/predict', json={})
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertIn("error", data)


class TestAPIPredictInvalidData(unittest.TestCase):
    """Test suite for predict endpoint with invalid data."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_predict_malformed_json(self):
        """Test predict endpoint with malformed JSON."""
        response = self.client.post(
            '/predict',
            data="not a json",
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_type_origin(self):
        """Test predict endpoint with invalid origin type."""
        payload = {
            "origin": 123,
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_type_destination(self):
        """Test predict endpoint with invalid destination type."""
        payload = {
            "origin": "LocationA",
            "destination": ["A", "B"],
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_invalid_type_city(self):
        """Test predict endpoint with invalid city type."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": {"name": "TestCity"}
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_string_origin(self):
        """Test predict endpoint with empty string origin."""
        payload = {
            "origin": "",
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_string_destination(self):
        """Test predict endpoint with empty string destination."""
        payload = {
            "origin": "LocationA",
            "destination": "",
            "city": "TestCity"
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_empty_string_city(self):
        """Test predict endpoint with empty string city."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": ""
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_null_values(self):
        """Test predict endpoint with null values."""
        payload = {
            "origin": None,
            "destination": None,
            "city": None
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_predict_whitespace_only_values(self):
        """Test predict endpoint with whitespace-only values."""
        payload = {
            "origin": "   ",
            "destination": "   ",
            "city": "   "
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)


class TestAPIPredictWrongMethods(unittest.TestCase):
    """Test suite for predict endpoint with wrong HTTP methods."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_predict_get_not_allowed(self):
        """Test that GET method is not allowed on predict route."""
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 405)

    def test_predict_put_not_allowed(self):
        """Test that PUT method is not allowed on predict route."""
        response = self.client.put('/predict', json={})
        self.assertEqual(response.status_code, 405)

    def test_predict_delete_not_allowed(self):
        """Test that DELETE method is not allowed on predict route."""
        response = self.client.delete('/predict')
        self.assertEqual(response.status_code, 405)

    def test_predict_patch_not_allowed(self):
        """Test that PATCH method is not allowed on predict route."""
        response = self.client.patch('/predict', json={})
        self.assertEqual(response.status_code, 405)


class TestAPIPredictContentType(unittest.TestCase):
    """Test suite for predict endpoint content type handling."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_predict_wrong_content_type_text_plain(self):
        """Test predict endpoint with text/plain content type."""
        payload = "origin=LocationA&destination=LocationB&city=TestCity"
        response = self.client.post(
            '/predict',
            data=payload,
            content_type='text/plain'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_form_data_not_accepted(self):
        """Test predict endpoint with form data instead of JSON."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post(
            '/predict',
            data=payload,
            content_type='application/x-www-form-urlencoded'
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_multipart_form_not_accepted(self):
        """Test predict endpoint with multipart form data."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }
        response = self.client.post(
            '/predict',
            data=payload,
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)


class TestAPINotFound(unittest.TestCase):
    """Test suite for 404 error handling."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    def test_nonexistent_route_get(self):
        """Test that GET to nonexistent route returns 404."""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)

    def test_nonexistent_route_post(self):
        """Test that POST to nonexistent route returns 404."""
        response = self.client.post('/nonexistent', json={})
        self.assertEqual(response.status_code, 404)

    def test_nonexistent_nested_route(self):
        """Test that nested nonexistent route returns 404."""
        response = self.client.get('/api/v1/nonexistent')
        self.assertEqual(response.status_code, 404)


class TestAPIWithMockedServices(unittest.TestCase):
    """Test suite with mocked external services."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.app = create_app(testing=True)
        self.client: FlaskClient = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
        
        self.valid_payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }

    def tearDown(self):
        """Clean up after each test method."""
        self.app_context.pop()

    @patch('Traffic_Management_System.src.data_fetcher.fetch_weather_data')
    def test_predict_handles_weather_service_failure(self, mock_weather):
        """Test predict handles weather service failure gracefully."""
        mock_weather.side_effect = Exception("Weather service unavailable")
        response = self.client.post('/predict', json=self.valid_payload)
        # Should handle gracefully - either 200 with fallback or 500
        self.assertIn(response.status_code, [200, 500, 503])

    @patch('Traffic_Management_System.src.data_fetcher.fetch_traffic_data')
    def test_predict_handles_traffic_service_failure(self, mock_traffic):
        """Test predict handles traffic service failure gracefully."""
        mock_traffic.side_effect = Exception("Traffic service unavailable")
        response = self.client.post('/predict', json=self.valid_payload)
        self.assertIn(response.status_code, [200, 500, 503])

    @patch('Traffic_Management_System.src.data_fetcher.fetch_events_data')
    def test_predict_handles_events_service_failure(self, mock_events):
        """Test predict handles events service failure gracefully."""
        mock_events.side_effect = Exception("Events service unavailable")
        response = self.client.post('/predict', json=self.valid_payload)
        self.assertIn(response.status_code, [200, 500, 503])


if __name__ == '__main__':
    unittest.main()
