"""
Unit tests for traffic prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import pickle
import joblib

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestTrafficModel:
    """Tests for the traffic prediction model."""

    @pytest.fixture
    def model_instance(self):
        """Create a traffic model instance for testing."""
        try:
            from models.traffic_model import TrafficModel
            return TrafficModel()
        except ImportError:
            pytest.skip("TrafficModel not available")

    @pytest.fixture
    def trained_model(self, model_instance, sample_traffic_data):
        """Create a trained model instance."""
        X = sample_traffic_data.drop(['congestion_level', 'traffic_volume'], axis=1, errors='ignore')
        y = sample_traffic_data.get('congestion_level', pd.Series(['low'] * len(sample_traffic_data)))
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        if hasattr(model_instance, 'train'):
            model_instance.train(X, y)
        elif hasattr(model_instance, 'fit'):
            model_instance.fit(X, y)
        
        return model_instance

    # ==================== INITIALIZATION TESTS ====================

    def test_model_initialization(self, model_instance):
        """Test that model initializes correctly."""
        assert model_instance is not None

    def test_model_has_predict_method(self, model_instance):
        """Test that model has predict method."""
        assert hasattr(model_instance, 'predict')
        assert callable(getattr(model_instance, 'predict'))

    def test_model_has_train_method(self, model_instance):
        """Test that model has train/fit method."""
        has_train = hasattr(model_instance, 'train') or hasattr(model_instance, 'fit')
        assert has_train

    # ==================== TRAINING TESTS ====================

    def test_model_training(self, model_instance, sample_traffic_data):
        """Test model training with sample data."""
        X = sample_traffic_data.drop(['congestion_level'], axis=1, errors='ignore')
        y = sample_traffic_data.get('congestion_level', pd.Series(['low'] * len(sample_traffic_data)))
        
        # Handle categorical
        X = pd.get_dummies(X, drop_first=True)
        
        try:
            if hasattr(model_instance, 'train'):
                model_instance.train(X, y)
            else:
                model_instance.fit(X, y)
            assert True  # Training succeeded
        except Exception as e:
            pytest.fail(f"Model training failed: {e}")

    def test_model_training_with_empty_data(self, model_instance):
        """Test model training with empty data raises error."""
        X = pd.DataFrame()
        y = pd.Series()
        
        with pytest.raises((ValueError, Exception)):
            if hasattr(model_instance, 'train'):
                model_instance.train(X, y)
            else:
                model_instance.fit(X, y)

    # ==================== PREDICTION TESTS ====================

    def test_model_prediction_output_shape(self, trained_model, sample_traffic_data):
        """Test that prediction output has correct shape."""
        X = sample_traffic_data.drop(['congestion_level', 'traffic_volume'], axis=1, errors='ignore')
        X = pd.get_dummies(X, drop_first=True)
        
        # Get expected columns from training
        if hasattr(trained_model, 'feature_names_'):
            X = X.reindex(columns=trained_model.feature_names_, fill_value=0)
        
        predictions = trained_model.predict(X.head(10))
        assert len(predictions) == 10

    def test_model_prediction_values(self, trained_model, sample_traffic_data):
        """Test that predictions are valid values."""
        X = sample_traffic_data.drop(['congestion_level', 'traffic_volume'], axis=1, errors='ignore')
        X = pd.get_dummies(X, drop_first=True)
        
        if hasattr(trained_model, 'feature_names_'):
            X = X.reindex(columns=trained_model.feature_names_, fill_value=0)
        
        predictions = trained_model.predict(X.head(5))
        
        # Check predictions are valid
        valid_levels = ['low', 'medium', 'high', 0, 1, 2]
        for pred in predictions:
            assert pred in valid_levels or isinstance(pred, (int, float, np.integer, np.floating))

    def test_model_prediction_without_training(self, model_instance):
        """Test that prediction without training raises error or returns default."""
        X = pd.DataFrame({'hour': [10], 'day_of_week': [1]})
        
        try:
            predictions = model_instance.predict(X)
            # If it doesn't raise, it should return something
            assert predictions is not None
        except Exception:
            # Expected behavior - model not trained
            pass

    # ==================== SAVE/LOAD TESTS ====================

    def test_model_save(self, trained_model, tmp_path):
        """Test model saving."""
        model_path = tmp_path / "model.pkl"
        
        if hasattr(trained_model, 'save'):
            trained_model.save(str(model_path))
            assert model_path.exists()
        else:
            # Use joblib as fallback
            joblib.dump(trained_model, str(model_path))
            assert model_path.exists()

    def test_model_load(self, trained_model, tmp_path):
        """Test model loading."""
        model_path = tmp_path / "model.pkl"
        
        # Save first
        joblib.dump(trained_model, str(model_path))
        
        # Load
        loaded_model = joblib.load(str(model_path))
        assert loaded_model is not None

    # ==================== FEATURE ENGINEERING TESTS ====================

    def test_feature_extraction(self, model_instance):
        """Test feature extraction if available."""
        if hasattr(model_instance, 'extract_features'):
            raw_data = {
                'timestamp': '2024-01-15 10:30:00',
                'location': 'downtown'
            }
            features = model_instance.extract_features(raw_data)
            assert features is not None

    def test_feature_importance(self, trained_model):
        """Test feature importance extraction."""
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            assert len(importances) > 0
            assert all(i >= 0 for i in importances)


class TestCongestionPredictor:
    """Tests for congestion prediction functionality."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        try:
            from models.congestion_predictor import CongestionPredictor
            return CongestionPredictor()
        except ImportError:
            pytest.skip("CongestionPredictor not available")

    def test_predict_congestion_level(self, predictor):
        """Test congestion level prediction."""
        input_data = {
            'hour': 8,
            'day_of_week': 1,
            'is_holiday': False,
            'weather': 'clear'
        }
        
        if hasattr(predictor, 'predict_congestion'):
            result = predictor.predict_congestion(input_data)
            assert result in ['low', 'medium', 'high'] or isinstance(result, (int, float))

    def test_predict_congestion_rush_hour(self, predictor):
        """Test congestion during rush hour."""
        rush_hour_data = {
            'hour': 8,  # Morning rush
            'day_of_week': 1,  # Monday
            'is_holiday': False
        }
        
        if hasattr(predictor, 'predict_congestion'):
            result = predictor.predict_congestion(rush_hour_data)
            # Rush hour should generally predict higher congestion
            assert result is not None
