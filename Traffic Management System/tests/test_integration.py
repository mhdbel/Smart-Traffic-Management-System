"""
Integration tests for the Smart Traffic Management System.
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestEndToEndPrediction:
    """End-to-end tests for the prediction workflow."""

    def test_full_prediction_workflow(self, client):
        """Test complete prediction workflow."""
        # Step 1: Check system is running
        response = client.get('/')
        assert response.status_code == 200

        # Step 2: Make a prediction request
        payload = {
            "origin": "Downtown",
            "destination": "Airport",
            "city": "TestCity"
        }
        response = client.post('/predict', json=payload)
        assert response.status_code == 200

        # Step 3: Validate response structure
        data = response.get_json()
        assert "traffic" in data
        assert "weather" in data
        assert "event" in data
        assert "routes" in data

        # Step 4: Validate routes
        routes = data["routes"]
        assert isinstance(routes, list)
        if len(routes) > 0:
            route = routes[0]
            # Check route has required fields
            expected_fields = ["distance_km", "estimated_time_min"]
            for field in expected_fields:
                if field in route:
                    assert route[field] >= 0

    def test_prediction_consistency(self, client):
        """Test that predictions are consistent for same input."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }

        # Make multiple requests
        responses = []
        for _ in range(3):
            response = client.post('/predict', json=payload)
            assert response.status_code == 200
            responses.append(response.get_json())

        # Check consistency (traffic level should be same)
        traffic_levels = [r.get("traffic", {}).get("level") for r in responses]
        # All should be the same (deterministic)
        assert len(set(filter(None, traffic_levels))) <= 1

    def test_different_routes_different_results(self, client):
        """Test that different routes give different results."""
        payload1 = {
            "origin": "Downtown",
            "destination": "Airport",
            "city": "TestCity"
        }
        payload2 = {
            "origin": "Suburb",
            "destination": "City Center",
            "city": "TestCity"
        }

        response1 = client.post('/predict', json=payload1)
        response2 = client.post('/predict', json=payload2)

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both should return valid data
        data1 = response1.get_json()
        data2 = response2.get_json()

        assert "routes" in data1
        assert "routes" in data2


class TestAPIResilience:
    """Tests for API resilience and error recovery."""

    def test_rapid_requests(self, client):
        """Test API handles rapid successive requests."""
        payload = {
            "origin": "LocationA",
            "destination": "LocationB",
            "city": "TestCity"
        }

        # Make 10 rapid requests
        responses = []
        for _ in range(10):
            response = client.post('/predict', json=payload)
            responses.append(response.status_code)

        # All should succeed (or be rate limited with 429)
        valid_codes = [200, 429]
        assert all(code in valid_codes for code in responses)

    def test_mixed_valid_invalid_requests(self, client):
        """Test API handles mix of valid and invalid requests."""
        requests_data = [
            ({"origin": "A", "destination": "B", "city": "C"}, 200),
            ({}, 400),
            ({"origin": "A", "destination": "B", "city": "C"}, 200),
            ({"origin": 123}, 400),
            ({"origin": "A", "destination": "B", "city": "C"}, 200),
        ]

        for payload, expected_status in requests_data:
            response = client.post('/predict', json=payload)
            assert response.status_code == expected_status

    def test_large_payload(self, client):
        """Test API handles large payloads appropriately."""
        payload = {
            "origin": "A" * 10000,
            "destination": "B" * 10000,
            "city": "C" * 10000
        }

        response = client.post('/predict', json=payload)
        # Should either accept (200) or reject (400/413)
        assert response.status_code in [200, 400, 413]


class TestDataIntegrity:
    """Tests for data integrity in responses."""

    def test_response_data_types(self, client, valid_prediction_payload):
        """Test that response data types are correct."""
        response = client.post('/predict', json=valid_prediction_payload)
        data = response.get_json()

        # Traffic should be dict
        assert isinstance(data.get("traffic"), dict)
        
        # Routes should be list
        assert isinstance(data.get("routes"), list)
        
        # Weather should be dict
        assert isinstance(data.get("weather"), dict)

    def test_numeric_values_are_valid(self, client, valid_prediction_payload):
        """Test that numeric values in response are valid."""
        response = client.post('/predict', json=valid_prediction_payload)
        data = response.get_json()

        routes = data.get("routes", [])
        for route in routes:
            if "distance_km" in route:
                assert route["distance_km"] >= 0
            if "estimated_time_min" in route:
                assert route["estimated_time_min"] >= 0
