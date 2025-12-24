"""
Unit tests for the Machine Learning model module.
Tests cover model predictions, input validation, and edge cases.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from Traffic_Management_System.src.ml_model import (
    predict_traffic,
    load_model,
    preprocess_input,
    TrafficPredictor
)


class TestPredictTraffic(unittest.TestCase):
    """Test suite for the predict_traffic function."""

    def test_predict_traffic_returns_dict(self):
        """Test that predict_traffic returns a dictionary."""
        result = predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIsInstance(result, dict)

    def test_predict_traffic_contains_prediction(self):
        """Test that result contains prediction key."""
        result = predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIn("prediction", result)

    def test_predict_traffic_contains_confidence(self):
        """Test that result contains confidence key."""
        result = predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIn("confidence", result)

    def test_predict_traffic_prediction_is_valid(self):
        """Test that prediction value is valid."""
        result = predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        # Assuming prediction is a traffic level or numeric value
        self.assertIsNotNone(result.get("prediction"))

    def test_predict_traffic_confidence_in_range(self):
        """Test that confidence is between 0 and 1."""
        result = predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        confidence = result.get("confidence", 0)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)

    def test_predict_traffic_with_special_characters(self):
        """Test predict_traffic with special characters in input."""
        result = predict_traffic(
            origin="Location-A (Test)",
            destination="Location_B #123",
            city="Test City"
        )
        self.assertIsInstance(result, dict)

    def test_predict_traffic_with_unicode(self):
        """Test predict_traffic with unicode characters."""
        result = predict_traffic(
            origin="東京",
            destination="大阪",
            city="日本"
        )
        self.assertIsInstance(result, dict)


class TestPredictTrafficInvalidInput(unittest.TestCase):
    """Test suite for predict_traffic with invalid inputs."""

    def test_predict_traffic_empty_origin(self):
        """Test predict_traffic with empty origin raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="",
                destination="LocationB",
                city="TestCity"
            )

    def test_predict_traffic_empty_destination(self):
        """Test predict_traffic with empty destination raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination="",
                city="TestCity"
            )

    def test_predict_traffic_empty_city(self):
        """Test predict_traffic with empty city raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination="LocationB",
                city=""
            )

    def test_predict_traffic_none_origin(self):
        """Test predict_traffic with None origin raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin=None,
                destination="LocationB",
                city="TestCity"
            )

    def test_predict_traffic_none_destination(self):
        """Test predict_traffic with None destination raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination=None,
                city="TestCity"
            )

    def test_predict_traffic_none_city(self):
        """Test predict_traffic with None city raises error."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination="LocationB",
                city=None
            )

    def test_predict_traffic_invalid_type_origin(self):
        """Test predict_traffic with invalid origin type."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin=123,
                destination="LocationB",
                city="TestCity"
            )

    def test_predict_traffic_invalid_type_destination(self):
        """Test predict_traffic with invalid destination type."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination=["A", "B"],
                city="TestCity"
            )

    def test_predict_traffic_invalid_type_city(self):
        """Test predict_traffic with invalid city type."""
        with self.assertRaises((ValueError, TypeError)):
            predict_traffic(
                origin="LocationA",
                destination="LocationB",
                city={"name": "TestCity"}
            )


class TestLoadModel(unittest.TestCase):
    """Test suite for the load_model function."""

    def test_load_model_returns_model(self):
        """Test that load_model returns a model object."""
        model = load_model()
        self.assertIsNotNone(model)

    def test_load_model_is_callable(self):
        """Test that loaded model has predict method."""
        model = load_model()
        self.assertTrue(hasattr(model, 'predict') or callable(model))

    @patch('Traffic_Management_System.src.ml_model.joblib.load')
    def test_load_model_file_not_found(self, mock_load):
        """Test load_model handles missing file gracefully."""
        mock_load.side_effect = FileNotFoundError("Model file not found")
        with self.assertRaises(FileNotFoundError):
            load_model(path="nonexistent_model.pkl")

    @patch('Traffic_Management_System.src.ml_model.joblib.load')
    def test_load_model_corrupted_file(self, mock_load):
        """Test load_model handles corrupted file."""
        mock_load.side_effect = Exception("Corrupted model file")
        with self.assertRaises(Exception):
            load_model(path="corrupted_model.pkl")


class TestPreprocessInput(unittest.TestCase):
    """Test suite for the preprocess_input function."""

    def test_preprocess_input_returns_array(self):
        """Test that preprocess_input returns numpy array."""
        result = preprocess_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIsInstance(result, np.ndarray)

    def test_preprocess_input_correct_shape(self):
        """Test that preprocessed input has correct shape."""
        result = preprocess_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        # Assuming 2D array with shape (1, n_features)
        self.assertEqual(len(result.shape), 2)
        self.assertEqual(result.shape[0], 1)

    def test_preprocess_input_consistent_output(self):
        """Test that same input produces same output."""
        result1 = preprocess_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        result2 = preprocess_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        np.testing.assert_array_equal(result1, result2)

    def test_preprocess_input_different_inputs(self):
        """Test that different inputs produce different outputs."""
        result1 = preprocess_input(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        result2 = preprocess_input(
            origin="LocationC",
            destination="LocationD",
            city="OtherCity"
        )
        self.assertFalse(np.array_equal(result1, result2))


class TestTrafficPredictor(unittest.TestCase):
    """Test suite for the TrafficPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = TrafficPredictor()

    def test_predictor_initialization(self):
        """Test TrafficPredictor initializes correctly."""
        self.assertIsNotNone(self.predictor)

    def test_predictor_has_predict_method(self):
        """Test TrafficPredictor has predict method."""
        self.assertTrue(hasattr(self.predictor, 'predict'))

    def test_predictor_predict_returns_dict(self):
        """Test predict method returns dictionary."""
        result = self.predictor.predict(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIsInstance(result, dict)

    def test_predictor_predict_valid_output(self):
        """Test predict method returns valid output."""
        result = self.predictor.predict(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        self.assertIn("prediction", result)

    def test_predictor_model_loaded(self):
        """Test that model is loaded in predictor."""
        self.assertIsNotNone(self.predictor.model)


class TestModelPerformance(unittest.TestCase):
    """Test suite for model performance characteristics."""

    def test_prediction_time_reasonable(self):
        """Test that prediction completes in reasonable time."""
        import time
        start = time.time()
        predict_traffic(
            origin="LocationA",
            destination="LocationB",
            city="TestCity"
        )
        elapsed = time.time() - start
        # Should complete within 5 seconds
        self.assertLess(elapsed, 5.0)

    def test_multiple_predictions_consistent(self):
        """Test that multiple predictions are consistent."""
        results = []
        for _ in range(5):
            result = predict_traffic(
                origin="LocationA",
                destination="LocationB",
                city="TestCity"
            )
            results.append(result.get("prediction"))
        
        # All predictions should be the same for same input
        self.assertTrue(all(r == results[0] for r in results))


if __name__ == '__main__':
    unittest.main()
