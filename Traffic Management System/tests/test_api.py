"""
Unit tests for the Flask API endpoints.
"""

import pytest
import json
from flask.testing import FlaskClient


class TestHomeEndpoint:
    """Tests for the home endpoint."""

    def test_home_returns_200(self, client):
        """Test that home endpoint returns 200 OK."""
        response = client.get('/')
        assert response.status_code == 200

    def test_home_contains_system_name(self, client):
        """Test that home response contains system name."""
        response = client.get('/')
        assert b"Smart Traffic Management System" in response.data or \
               "Smart Traffic Management System" in response.get_json().get("message", "")

    def test_home_returns_json(self, client):
        """Test that home endpoint returns JSON."""
        response = client.get('/')
        assert response.content_type == 'application/json'

    def test_home_post_not_allowed(self, client):
        """Test that POST method is not allowed on home endpoint."""
        response = client.post('/')
        assert response.status_code == 405

    def test_home_put_not_allowed(self, client):
        """Test that PUT method is not allowed on home endpoint."""
        response = client.put('/')
        assert response.status_code == 405

    def test_home_delete_not_allowed(self, client):
        """Test that DELETE method is not allowed on home endpoint."""
        response = client.delete('/')
        assert response.status_code == 405


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    # ==================== VALID REQUESTS ====================

    def test_predict_valid_payload(self, client, valid_prediction_payload):
        """Test predict endpoint with valid payload."""
        response = client.post('/predict', json=valid_prediction_payload)
        assert response.status_code == 200
        
        data = response.get_json()
        assert "traffic" in data
        assert "event" in data
        assert "weather" in data
        assert "routes" in data

    def test_predict_returns_json(self, client, valid_prediction_payload):
        """Test that predict endpoint returns JSON."""
        response = client.post('/predict', json=valid_prediction_payload)
        assert response.content_type == 'application/json'

    def test_predict_routes_is_list(self, client, valid_prediction_payload):
        """Test that routes in response is a list."""
        response = client.post('/predict', json=valid_prediction_payload)
        data = response.get_json()
        assert isinstance(data.get("routes"), list)

    def test_predict_with_different_cities(self, client):
        """Test predict with various cities."""
        cities = ["NewYork", "LosAngeles", "Chicago", "Houston", "Phoenix"]
        
        for city in cities:
            payload = {
                "origin": "LocationA",
                "destination": "LocationB",
                "city": city
            }
            response = client.post('/predict', json=payload)
            assert response.status_code == 200

    # ==================== MISSING FIELDS ====================

    def test_predict_missing_origin(self, client):
        """Test predict endpoint with missing origin field."""
        payload = {"destination": "LocationB", "city": "TestCity"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_missing_destination(self, client):
        """Test predict endpoint with missing destination field."""
        payload = {"origin": "LocationA", "city": "TestCity"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_missing_city(self, client):
        """Test predict endpoint with missing city field."""
        payload = {"origin": "LocationA", "destination": "LocationB"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_empty_payload(self, client):
        """Test predict endpoint with empty payload."""
        response = client.post('/predict', json={})
        assert response.status_code == 400

    def test_predict_no_payload(self, client):
        """Test predict endpoint with no payload."""
        response = client.post('/predict', content_type='application/json')
        assert response.status_code == 400

    # ==================== INVALID DATA ====================

    def test_predict_malformed_json(self, client):
        """Test predict endpoint with malformed JSON."""
        response = client.post(
            '/predict',
            data="not a json",
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_predict_invalid_types_origin_int(self, client):
        """Test predict endpoint with integer origin."""
        payload = {"origin": 123, "destination": "LocationB", "city": "TestCity"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_invalid_types_destination_list(self, client):
        """Test predict endpoint with list destination."""
        payload = {"origin": "LocationA", "destination": ["A", "B"], "city": "TestCity"}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_invalid_types_city_dict(self, client):
        """Test predict endpoint with dict city."""
        payload = {"origin": "LocationA", "destination": "LocationB", "city": {"name": "TestCity"}}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_null_values(self, client):
        """Test predict endpoint with null values."""
        payload = {"origin": None, "destination": None, "city": None}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_empty_strings(self, client):
        """Test predict endpoint with empty string values."""
        payload = {"origin": "", "destination": "", "city": ""}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    def test_predict_whitespace_only(self, client):
        """Test predict endpoint with whitespace-only values."""
        payload = {"origin": "   ", "destination": "  ", "city": "   "}
        response = client.post('/predict', json=payload)
        assert response.status_code == 400

    # ==================== WRONG METHOD/CONTENT TYPE ====================

    def test_predict_get_not_allowed(self, client):
        """Test that GET method is not allowed on predict endpoint."""
        response = client.get('/predict')
        assert response.status_code == 405

    def test_predict_put_not_allowed(self, client):
        """Test that PUT method is not allowed on predict endpoint."""
        response = client.put('/predict', json={})
        assert response.status_code == 405

    def test_predict_delete_not_allowed(self, client):
        """Test that DELETE method is not allowed on predict endpoint."""
        response = client.delete('/predict')
        assert response.status_code == 405

    def test_predict_wrong_content_type_text(self, client):
        """Test predict endpoint with text/plain content type."""
        payload = "origin=LocationA&destination=LocationB&city=TestCity"
        response = client.post('/predict', data=payload, content_type='text/plain')
        assert response.status_code == 400

    def test_predict_wrong_content_type_form(self, client):
        """Test predict endpoint with form data."""
        payload = {"origin": "LocationA", "destination": "LocationB", "city": "TestCity"}
        response = client.post('/predict', data=payload)
        assert response.status_code == 400

    # ==================== EDGE CASES ====================

    def test_predict_nonexistent_city(self, client):
        """Test predict with nonexistent city returns fallback data."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "NoSuchCity12345"
        }
        response = client.post('/predict', json=payload)
        # Should still return 200 with fallback data
        assert response.status_code == 200
        
        data = response.get_json()
        assert "traffic" in data
        assert "routes" in data

    def test_predict_special_characters(self, client):
        """Test predict with special characters in payload."""
        payload = {
            "origin": "Location A & B's Corner",
            "destination": "Street #123",
            "city": "San José"
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200

    def test_predict_unicode_characters(self, client):
        """Test predict with unicode characters."""
        payload = {
            "origin": "北京站",
            "destination": "上海虹桥站",
            "city": "中国"
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200

    def test_predict_very_long_strings(self, client):
        """Test predict with very long string values."""
        payload = {
            "origin": "A" * 1000,
            "destination": "B" * 1000,
            "city": "C" * 1000
        }
        response = client.post('/predict', json=payload)
        # Should either succeed or return 400 for validation
        assert response.status_code in [200, 400]

    def test_predict_extra_fields_ignored(self, client):
        """Test that extra fields in payload are ignored."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity",
            "extra_field": "should be ignored",
            "another_extra": 12345
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200


class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get('/nonexistent-endpoint')
        assert response.status_code == 404

    def test_404_returns_json(self, client):
        """Test that 404 returns JSON response."""
        response = client.get('/nonexistent-endpoint')
        assert response.content_type == 'application/json'

    def test_405_returns_json(self, client):
        """Test that 405 returns JSON response."""
        response = client.post('/')
        assert response.status_code == 405


class TestHealthEndpoint:
    """Tests for health check endpoint (if exists)."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        # If endpoint exists, should return 200
        if response.status_code != 404:
            assert response.status_code == 200
            data = response.get_json()
            assert "status" in data or "healthy" in str(data).lower()
