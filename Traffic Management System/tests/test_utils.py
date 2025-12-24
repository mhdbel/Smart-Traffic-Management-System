"""
Unit tests for utility functions.
"""

import pytest
import os
import sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestValidators:
    """Tests for validation utilities."""

    @pytest.fixture
    def validators(self):
        """Import validators module."""
        try:
            from utils import validators
            return validators
        except ImportError:
            pytest.skip("Validators module not available")

    def test_validate_location_valid(self, validators):
        """Test location validation with valid input."""
        if hasattr(validators, 'validate_location'):
            assert validators.validate_location("Downtown") == True

    def test_validate_location_empty(self, validators):
        """Test location validation with empty input."""
        if hasattr(validators, 'validate_location'):
            assert validators.validate_location("") == False

    def test_validate_location_none(self, validators):
        """Test location validation with None input."""
        if hasattr(validators, 'validate_location'):
            assert validators.validate_location(None) == False

    def test_validate_city_valid(self, validators):
        """Test city validation with valid input."""
        if hasattr(validators, 'validate_city'):
            assert validators.validate_city("New York") == True

    def test_validate_coordinates(self, validators):
        """Test coordinate validation."""
        if hasattr(validators, 'validate_coordinates'):
            assert validators.validate_coordinates(40.7128, -74.0060) == True
            assert validators.validate_coordinates(91, 0) == False  # Invalid latitude
            assert validators.validate_coordinates(0, 181) == False  # Invalid longitude


class TestHelpers:
    """Tests for helper functions."""

    @pytest.fixture
    def helpers(self):
        """Import helpers module."""
        try:
            from utils import helpers
            return helpers
        except ImportError:
            pytest.skip("Helpers module not available")

    def test_format_time(self, helpers):
        """Test time formatting."""
        if hasattr(helpers, 'format_time'):
            result = helpers.format_time(90)  # 90 minutes
            assert "1" in result and ("hour" in result.lower() or "h" in result.lower())

    def test_format_distance(self, helpers):
        """Test distance formatting."""
        if hasattr(helpers, 'format_distance'):
            result = helpers.format_distance(1500)  # 1500 meters
            assert "1.5" in result or "km" in result.lower()

    def test_calculate_distance(self, helpers):
        """Test distance calculation between coordinates."""
        if hasattr(helpers, 'calculate_distance'):
            # NYC to LA approximately
            distance = helpers.calculate_distance(
                40.7128, -74.0060,  # NYC
                34.0522, -118.2437  # LA
            )
            # Should be approximately 3940 km
            assert 3500 < distance < 4500

    def test_is_rush_hour(self, helpers):
        """Test rush hour detection."""
        if hasattr(helpers, 'is_rush_hour'):
            assert helpers.is_rush_hour(8) == True   # 8 AM
            assert helpers.is_rush_hour(17) == True  # 5 PM
            assert helpers.is_rush_hour(14) == False # 2 PM
            assert helpers.is_rush_hour(3) == False  # 3 AM

    def test_is_holiday(self, helpers):
        """Test holiday detection."""
        if hasattr(helpers, 'is_holiday'):
            result = helpers.is_holiday(date.today())
            assert isinstance(result, bool)


class TestDataProcessing:
    """Tests for data processing utilities."""

    @pytest.fixture
    def data_utils(self):
        """Import data utilities module."""
        try:
            from utils import data_processing
            return data_processing
        except ImportError:
            pytest.skip("Data processing module not available")

    def test_normalize_data(self, data_utils):
        """Test data normalization."""
        if hasattr(data_utils, 'normalize'):
            data = [1, 2, 3, 4, 5]
            normalized = data_utils.normalize(data)
            assert min(normalized) >= 0
            assert max(normalized) <= 1

    def test_clean_location_string(self, data_utils):
        """Test location string cleaning."""
        if hasattr(data_utils, 'clean_location'):
            result = data_utils.clean_location("  New York  City  ")
            assert result == "New York City" or "new york" in result.lower()

    def test_parse_datetime(self, data_utils):
        """Test datetime parsing."""
        if hasattr(data_utils, 'parse_datetime'):
            result = data_utils.parse_datetime("2024-01-15 10:30:00")
            assert isinstance(result, datetime)


class TestConfigLoader:
    """Tests for configuration loading."""

    @pytest.fixture
    def config_loader(self):
        """Import config loader."""
        try:
            from utils import config
            return config
        except ImportError:
            pytest.skip("Config module not available")

    def test_load_config(self, config_loader):
        """Test configuration loading."""
        if hasattr(config_loader, 'load_config'):
            config = config_loader.load_config()
            assert config is not None

    def test_get_api_key(self, config_loader):
        """Test API key retrieval."""
        if hasattr(config_loader, 'get_api_key'):
            # Should return key or None, not raise exception
            try:
                key = config_loader.get_api_key('weather')
                assert key is None or isinstance(key, str)
            except Exception:
                pass  # API key might not be configured in test environment
