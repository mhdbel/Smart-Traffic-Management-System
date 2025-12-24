"""
Unit tests for utility functions.
Tests cover helper functions, validators, and formatters.
"""

import unittest
from datetime import datetime, timedelta

from Traffic_Management_System.src.utils import (
    validate_input,
    format_response,
    calculate_eta,
    parse_coordinates,
    sanitize_string,
    get_current_timestamp,
    is_valid_city,
    is_valid_location
)


class TestValidateInput(unittest.TestCase):
    """Test suite for the validate_input function."""

    def test_validate_input_valid(self):
        """Test validate_input with valid input."""
        result = validate_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertTrue(result)

    def test_validate_input_empty_origin(self):
        """Test validate_input with empty origin."""
        result = validate_input(
            origin="",
            destination="LocationB",
            city="TestCity"
        )
        self.assertFalse(result)

    def test_validate_input_empty_destination(self):
        """Test validate_input with empty destination."""
        result = validate_input(
            origin="LocationA",
            destination="",
            city="TestCity"
        )
        self.assertFalse(result)

    def test_validate_input_empty_city(self):
        """Test validate_input with empty city."""
        result = validate_input(
            origin="LocationA",
            destination="LocationB",
            city=""
        )
        self.assertFalse(result)

    def test_validate_input_none_values(self):
        """Test validate_input with None values."""
        result = validate_input(
            origin=None,
            destination=None,
            city=None
        )
        self.assertFalse(result)

    def test_validate_input_whitespace_only(self):
        """Test validate_input with whitespace-only values."""
        result = validate_input(
            origin="   ",
            destination="   ",
            city="   "
        )
        self.assertFalse(result)

    def test_validate_input_invalid_types(self):
        """Test validate_input with invalid types."""
        result = validate_input(
            origin=123,
            destination=["A", "B"],
            city={"name": "City"}
        )
        self.assertFalse(result)

    def test_validate_input_special_characters(self):
        """Test validate_input with special characters."""
        result = validate_input(
            origin="Location-A",
            destination="Location_B",
            city="Test City"
        )
        self.assertTrue(result)


class TestFormatResponse(unittest.TestCase):
    """Test suite for the format_response function."""

    def test_format_response_returns_dict(self):
        """Test that format_response returns a dictionary."""
        data = {"key": "value"}
        result = format_response(data)
        self.assertIsInstance(result, dict)

    def test_format_response_contains_status(self):
        """Test that formatted response contains status."""
        data = {"key": "value"}
        result = format_response(data)
        self.assertIn("status", result)

    def test_format_response_contains_data(self):
        """Test that formatted response contains data."""
        data = {"key": "value"}
        result = format_response(data)
        self.assertIn("data", result)

    def test_format_response_contains_timestamp(self):
        """Test that formatted response contains timestamp."""
        data = {"key": "value"}
        result = format_response(data)
        self.assertIn("timestamp", result)

    def test_format_response_success_status(self):
        """Test format_response with success status."""
        data = {"key": "value"}
        result = format_response(data, success=True)
        self.assertEqual(result["status"], "success")

    def test_format_response_error_status(self):
        """Test format_response with error status."""
        data = {"error": "Something went wrong"}
        result = format_response(data, success=False)
        self.assertEqual(result["status"], "error")

    def test_format_response_empty_data(self):
        """Test format_response with empty data."""
        result = format_response({})
        self.assertIsInstance(result, dict)
        self.assertEqual(result["data"], {})

    def test_format_response_preserves_data(self):
        """Test that format_response preserves original data."""
        data = {"traffic": "heavy", "weather": "sunny"}
        result = format_response(data)
        self.assertEqual(result["data"]["traffic"], "heavy")
        self.assertEqual(result["data"]["weather"], "sunny")


class TestCalculateETA(unittest.TestCase):
    """Test suite for the calculate_eta function."""

    def test_calculate_eta_returns_datetime(self):
        """Test that calculate_eta returns datetime."""
        result = calculate_eta(distance=10, speed=60)
        self.assertIsInstance(result, datetime)

    def test_calculate_eta_future_time(self):
        """Test that ETA is in the future."""
        result = calculate_eta(distance=10, speed=60)
        self.assertGreater(result, datetime.now())

    def test_calculate_eta_zero_distance(self):
        """Test calculate_eta with zero distance."""
        result = calculate_eta(distance=0, speed=60)
        now = datetime.now()
        # Should be approximately now
        self.assertLess((result - now).total_seconds(), 1)

    def test_calculate_eta_negative_distance(self):
        """Test calculate_eta with negative distance."""
        with self.assertRaises(ValueError):
            calculate_eta(distance=-10, speed=60)

    def test_calculate_eta_zero_speed(self):
        """Test calculate_eta with zero speed."""
        with self.assertRaises((ValueError, ZeroDivisionError)):
            calculate_eta(distance=10, speed=0)

    def test_calculate_eta_negative_speed(self):
        """Test calculate_eta with negative speed."""
        with self.assertRaises(ValueError):
            calculate_eta(distance=10, speed=-60)

    def test_calculate_eta_correct_calculation(self):
        """Test calculate_eta calculates correctly."""
        # 60 km at 60 km/h = 1 hour
        now = datetime.now()
        result = calculate_eta(distance=60, speed=60)
        expected = now + timedelta(hours=1)
        
        # Allow 1 second tolerance
        diff = abs((result - expected).total_seconds())
        self.assertLess(diff, 1)


class TestParseCoordinates(unittest.TestCase):
    """Test suite for the parse_coordinates function."""

    def test_parse_coordinates_valid(self):
        """Test parse_coordinates with valid input."""
        result = parse_coordinates("40.7128,-74.0060")
        self.assertEqual(result, (40.7128, -74.0060))

    def test_parse_coordinates_returns_tuple(self):
        """Test that parse_coordinates returns tuple."""
        result = parse_coordinates("40.7128,-74.0060")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_parse_coordinates_with_spaces(self):
        """Test parse_coordinates with spaces."""
        result = parse_coordinates("40.7128, -74.0060")
        self.assertEqual(result, (40.7128, -74.0060))

    def test_parse_coordinates_invalid_format(self):
        """Test parse_coordinates with invalid format."""
        with self.assertRaises(ValueError):
            parse_coordinates("invalid")

    def test_parse_coordinates_missing_longitude(self):
        """Test parse_coordinates with missing longitude."""
        with self.assertRaises(ValueError):
            parse_coordinates("40.7128")

    def test_parse_coordinates_out_of_range_latitude(self):
        """Test parse_coordinates with out of range latitude."""
        with self.assertRaises(ValueError):
            parse_coordinates("91.0,-74.0060")

    def test_parse_coordinates_out_of_range_longitude(self):
        """Test parse_coordinates with out of range longitude."""
        with self.assertRaises(ValueError):
            parse_coordinates("40.7128,-181.0")

    def test_parse_coordinates_empty_string(self):
        """Test parse_coordinates with empty string."""
        with self.assertRaises(ValueError):
            parse_coordinates("")


class TestSanitizeString(unittest.TestCase):
    """Test suite for the sanitize_string function."""

    def test_sanitize_string_removes_leading_spaces(self):
        """Test that leading spaces are removed."""
        result = sanitize_string("   hello")
        self.assertEqual(result, "hello")

    def test_sanitize_string_removes_trailing_spaces(self):
        """Test that trailing spaces are removed."""
        result = sanitize_string("hello   ")
        self.assertEqual(result, "hello")

    def test_sanitize_string_removes_both_spaces(self):
        """Test that both leading and trailing spaces are removed."""
        result = sanitize_string("   hello   ")
        self.assertEqual(result, "hello")

    def test_sanitize_string_preserves_internal_spaces(self):
        """Test that internal spaces are preserved."""
        result = sanitize_string("hello world")
        self.assertEqual(result, "hello world")

    def test_sanitize_string_empty_string(self):
        """Test sanitize_string with empty string."""
        result = sanitize_string("")
        self.assertEqual(result, "")

    def test_sanitize_string_only_spaces(self):
        """Test sanitize_string with only spaces."""
        result = sanitize_string("     ")
        self.assertEqual(result, "")

    def test_sanitize_string_removes_special_chars(self):
        """Test that dangerous characters are removed."""
        result = sanitize_string("<script>alert('xss')</script>")
        self.assertNotIn("<script>", result)

    def test_sanitize_string_none_input(self):
        """Test sanitize_string with None input."""
        with self.assertRaises((ValueError, TypeError, AttributeError)):
            sanitize_string(None)


class TestGetCurrentTimestamp(unittest.TestCase):
    """Test suite for the get_current_timestamp function."""

    def test_get_current_timestamp_returns_string(self):
        """Test that get_current_timestamp returns string."""
        result = get_current_timestamp()
        self.assertIsInstance(result, str)

    def test_get_current_timestamp_iso_format(self):
        """Test that timestamp is in ISO format."""
        result = get_current_timestamp()
        # Should be parseable as datetime
        try:
            datetime.fromisoformat(result.replace('Z', '+00:00'))
        except ValueError:
            self.fail("Timestamp is not in valid ISO format")

    def test_get_current_timestamp_recent(self):
        """Test that timestamp is recent."""
        result = get_current_timestamp()
        timestamp = datetime.fromisoformat(result.replace('Z', '+00:00'))
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        
        # Should be within last minute
        diff = abs((now - timestamp).total_seconds())
        self.assertLess(diff, 60)


class TestIsValidCity(unittest.TestCase):
    """Test suite for the is_valid_city function."""

    def test_is_valid_city_valid(self):
        """Test is_valid_city with valid city name."""
        self.assertTrue(is_valid_city("New York"))

    def test_is_valid_city_empty(self):
        """Test is_valid_city with empty string."""
        self.assertFalse(is_valid_city(""))

    def test_is_valid_city_none(self):
        """Test is_valid_city with None."""
        self.assertFalse(is_valid_city(None))

    def test_is_valid_city_whitespace(self):
        """Test is_valid_city with whitespace only."""
        self.assertFalse(is_valid_city("   "))

    def test_is_valid_city_with_numbers(self):
        """Test is_valid_city with numbers."""
        # Some cities have numbers (e.g., "District 9")
        result = is_valid_city("District 9")
        self.assertIsInstance(result, bool)

    def test_is_valid_city_unicode(self):
        """Test is_valid_city with unicode characters."""
        self.assertTrue(is_valid_city("東京"))


class TestIsValidLocation(unittest.TestCase):
    """Test suite for the is_valid_location function."""

    def test_is_valid_location_valid(self):
        """Test is_valid_location with valid location."""
        self.assertTrue(is_valid_location("Times Square"))

    def test_is_valid_location_empty(self):
        """Test is_valid_location with empty string."""
        self.assertFalse(is_valid_location(""))

    def test_is_valid_location_none(self):
        """Test is_valid_location with None."""
        self.assertFalse(is_valid_location(None))

    def test_is_valid_location_coordinates(self):
        """Test is_valid_location with coordinates."""
        self.assertTrue(is_valid_location("40.7128,-74.0060"))

    def test_is_valid_location_address(self):
        """Test is_valid_location with address."""
        self.assertTrue(is_valid_location("123 Main St"))


if __name__ == '__main__':
    unittest.main()
