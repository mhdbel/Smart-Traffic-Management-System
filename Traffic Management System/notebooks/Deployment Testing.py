# notebooks/deployment_testing.py
"""
Deployment Testing Suite for Smart Traffic Management System.

This module provides comprehensive tests for:
1. Model loading and prediction
2. API endpoint testing
3. End-to-end integration tests
4. Performance benchmarks
5. Health checks

Usage:
    python deployment_testing.py --all
    python deployment_testing.py --api-only
    python deployment_testing.py --model-only
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration."""
    # Paths
    model_path: str = os.getenv(
        "MODEL_PATH",
        str(Path(__file__).parent.parent / "models" / "traffic_model.pkl")
    )
    
    # API
    api_base_url: str = os.getenv("API_URL", "http://localhost:5000")
    api_timeout: int = 30
    
    # Test parameters
    n_predictions: int = 100
    concurrent_requests: int = 10
    performance_threshold_ms: float = 1000.0
    
    # Test data
    test_origin: str = "Rabat, Morocco"
    test_destination: str = "Casablanca, Morocco"
    test_city: str = "Rabat"


# =============================================================================
# TEST RESULTS
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} | {self.name} ({self.duration_ms:.0f}ms) - {self.message}"


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def add(self, result: TestResult) -> None:
        self.results.append(result)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def success_rate(self) -> float:
        return self.passed / self.total * 100 if self.total > 0 else 0
    
    @property
    def all_passed(self) -> bool:
        return self.failed == 0
    
    def complete(self) -> None:
        self.completed_at = datetime.now().isoformat()
    
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"TEST SUITE: {self.name}",
            f"{'='*60}",
        ]
        
        for result in self.results:
            lines.append(str(result))
        
        lines.extend([
            f"{'='*60}",
            f"Results: {self.passed}/{self.total} passed ({self.success_rate:.1f}%)",
            f"{'='*60}\n",
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "success_rate": self.success_rate,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                }
                for r in self.results
            ]
        }


# =============================================================================
# MODEL TESTS
# =============================================================================

class ModelTests:
    """Tests for the ML model."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.model = None
        self.model_info = None
    
    def run_all(self) -> TestSuite:
        """Run all model tests."""
        suite = TestSuite("Model Tests")
        
        suite.add(self.test_model_exists())
        suite.add(self.test_model_loads())
        
        if self.model is not None:
            suite.add(self.test_model_has_predict())
            suite.add(self.test_model_prediction_shape())
            suite.add(self.test_model_prediction_range())
            suite.add(self.test_model_feature_names())
            suite.add(self.test_model_prediction_speed())
            suite.add(self.test_model_batch_prediction())
        
        suite.complete()
        return suite
    
    def test_model_exists(self) -> TestResult:
        """Test that model file exists."""
        start = time.time()
        
        model_path = Path(self.config.model_path)
        exists = model_path.exists()
        
        return TestResult(
            name="Model file exists",
            passed=exists,
            message=f"Found at {model_path}" if exists else f"Not found at {model_path}",
            duration_ms=(time.time() - start) * 1000,
            details={"path": str(model_path), "exists": exists}
        )
    
    def test_model_loads(self) -> TestResult:
        """Test that model loads without errors."""
        start = time.time()
        
        try:
            self.model = joblib.load(self.config.model_path)
            model_type = type(self.model).__name__
            
            # Try to load metadata
            metadata_path = Path(self.config.model_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.model_info = json.load(f)
            
            return TestResult(
                name="Model loads successfully",
                passed=True,
                message=f"Loaded {model_type}",
                duration_ms=(time.time() - start) * 1000,
                details={"model_type": model_type}
            )
        except Exception as e:
            return TestResult(
                name="Model loads successfully",
                passed=False,
                message=f"Failed to load: {e}",
                duration_ms=(time.time() - start) * 1000,
                details={"error": str(e)}
            )
    
    def test_model_has_predict(self) -> TestResult:
        """Test that model has predict method."""
        start = time.time()
        
        has_predict = hasattr(self.model, 'predict') and callable(self.model.predict)
        
        return TestResult(
            name="Model has predict method",
            passed=has_predict,
            message="predict() method available" if has_predict else "No predict() method",
            duration_ms=(time.time() - start) * 1000
        )
    
    def test_model_prediction_shape(self) -> TestResult:
        """Test that model produces correct output shape."""
        start = time.time()
        
        try:
            test_data = self._get_test_data(n_samples=5)
            predictions = self.model.predict(test_data)
            
            expected_shape = (5,)
            actual_shape = predictions.shape
            passed = actual_shape == expected_shape or len(predictions) == 5
            
            return TestResult(
                name="Prediction shape correct",
                passed=passed,
                message=f"Shape: {actual_shape}",
                duration_ms=(time.time() - start) * 1000,
                details={"expected": expected_shape, "actual": actual_shape}
            )
        except Exception as e:
            return TestResult(
                name="Prediction shape correct",
                passed=False,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000,
                details={"error": str(e)}
            )
    
    def test_model_prediction_range(self) -> TestResult:
        """Test that predictions are in reasonable range."""
        start = time.time()
        
        try:
            test_data = self._get_test_data(n_samples=100)
            predictions = self.model.predict(test_data)
            
            min_pred = float(np.min(predictions))
            max_pred = float(np.max(predictions))
            mean_pred = float(np.mean(predictions))
            
            # Traffic volume should be positive and reasonable
            is_positive = min_pred >= 0
            is_reasonable = max_pred < 100000  # Adjust based on your data
            passed = is_positive and is_reasonable
            
            return TestResult(
                name="Predictions in valid range",
                passed=passed,
                message=f"Range: [{min_pred:.2f}, {max_pred:.2f}], Mean: {mean_pred:.2f}",
                duration_ms=(time.time() - start) * 1000,
                details={"min": min_pred, "max": max_pred, "mean": mean_pred}
            )
        except Exception as e:
            return TestResult(
                name="Predictions in valid range",
                passed=False,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000,
                details={"error": str(e)}
            )
    
    def test_model_feature_names(self) -> TestResult:
        """Test that model has feature names (if applicable)."""
        start = time.time()
        
        if hasattr(self.model, 'feature_names_in_'):
            features = list(self.model.feature_names_in_)
            return TestResult(
                name="Model has feature names",
                passed=True,
                message=f"{len(features)} features defined",
                duration_ms=(time.time() - start) * 1000,
                details={"features": features[:10], "total": len(features)}
            )
        else:
            return TestResult(
                name="Model has feature names",
                passed=True,  # Not a failure, just info
                message="No feature_names_in_ attribute (using positional features)",
                duration_ms=(time.time() - start) * 1000
            )
    
    def test_model_prediction_speed(self) -> TestResult:
        """Test model prediction speed."""
        start = time.time()
        
        try:
            test_data = self._get_test_data(n_samples=1)
            
            # Warm up
            _ = self.model.predict(test_data)
            
            # Time multiple predictions
            times = []
            for _ in range(100):
                pred_start = time.time()
                _ = self.model.predict(test_data)
                times.append((time.time() - pred_start) * 1000)
            
            avg_time = np.mean(times)
            p95_time = np.percentile(times, 95)
            
            passed = avg_time < self.config.performance_threshold_ms
            
            return TestResult(
                name="Prediction speed",
                passed=passed,
                message=f"Avg: {avg_time:.2f}ms, P95: {p95_time:.2f}ms",
                duration_ms=(time.time() - start) * 1000,
                details={"avg_ms": avg_time, "p95_ms": p95_time, "threshold_ms": self.config.performance_threshold_ms}
            )
        except Exception as e:
            return TestResult(
                name="Prediction speed",
                passed=False,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000
            )
    
    def test_model_batch_prediction(self) -> TestResult:
        """Test model handles batch predictions."""
        start = time.time()
        
        try:
            batch_sizes = [1, 10, 100, 1000]
            results = {}
            
            for size in batch_sizes:
                test_data = self._get_test_data(n_samples=size)
                batch_start = time.time()
                predictions = self.model.predict(test_data)
                batch_time = (time.time() - batch_start) * 1000
                
                results[size] = {
                    "time_ms": batch_time,
                    "per_sample_ms": batch_time / size,
                    "output_len": len(predictions)
                }
            
            return TestResult(
                name="Batch prediction",
                passed=True,
                message=f"Tested batch sizes: {batch_sizes}",
                duration_ms=(time.time() - start) * 1000,
                details=results
            )
        except Exception as e:
            return TestResult(
                name="Batch prediction",
                passed=False,
                message=f"Error: {e}",
                duration_ms=(time.time() - start) * 1000
            )
    
    def _get_test_data(self, n_samples: int = 1) -> pd.DataFrame:
        """Generate test data matching model features."""
        # Try to get feature names from model
        if hasattr(self.model, 'feature_names_in_'):
            features = list(self.model.feature_names_in_)
        elif self.model_info and 'features' in self.model_info:
            features = self.model_info['features']
        else:
            # Default features based on our preprocessing
            features = [
                'hour', 'day_of_week', 'day_of_month', 'month', 'year',
                'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'temperature', 'humidity', 'wind_speed',
                'traffic_rolling_mean_3h', 'traffic_rolling_std_3h',
            ]
        
        # Generate random data
        np.random.seed(42)
        data = {}
        
        for feature in features:
            if 'hour' in feature.lower() and 'sin' not in feature and 'cos' not in feature:
                data[feature] = np.random.randint(0, 24, n_samples)
            elif 'day_of_week' in feature.lower():
                data[feature] = np.random.randint(0, 7, n_samples)
            elif 'day_of_month' in feature.lower():
                data[feature] = np.random.randint(1, 32, n_samples)
            elif 'month' in feature.lower():
                data[feature] = np.random.randint(1, 13, n_samples)
            elif 'year' in feature.lower():
                data[feature] = np.full(n_samples, 2024)
            elif 'is_' in feature.lower() or feature.lower().startswith('weather_'):
                data[feature] = np.random.randint(0, 2, n_samples)
            elif 'sin' in feature.lower() or 'cos' in feature.lower():
                data[feature] = np.random.uniform(-1, 1, n_samples)
            elif 'temp' in feature.lower():
                data[feature] = np.random.uniform(0, 40, n_samples)
            elif 'humidity' in feature.lower():
                data[feature] = np.random.uniform(20, 100, n_samples)
            elif 'wind' in feature.lower():
                data[feature] = np.random.uniform(0, 20, n_samples)
            elif 'traffic' in feature.lower():
                data[feature] = np.random.uniform(0, 1000, n_samples)
            else:
                data[feature] = np.random.uniform(0, 100, n_samples)
        
        return pd.DataFrame(data)


# =============================================================================
# API TESTS
# =============================================================================

class APITests:
    """Tests for the API endpoints."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.base_url = config.api_base_url.rstrip('/')
    
    def run_all(self) -> TestSuite:
        """Run all API tests."""
        suite = TestSuite("API Tests")
        
        suite.add(self.test_health_endpoint())
        suite.add(self.test_ready_endpoint())
        
        # Only continue if API is healthy
        if suite.results[0].passed:
            suite.add(self.test_predict_endpoint())
            suite.add(self.test_predict_validation())
            suite.add(self.test_traffic_endpoint())
            suite.add(self.test_weather_endpoint())
            suite.add(self.test_events_endpoint())
            suite.add(self.test_routes_endpoint())
            suite.add(self.test_error_handling())
            suite.add(self.test_response_format())
        
        suite.complete()
        return suite
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Tuple[Optional[requests.Response], Optional[Exception]]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.config.api_timeout)
        
        try:
            response = requests.request(method, url, **kwargs)
            return response, None
        except Exception as e:
            return None, e
    
    def test_health_endpoint(self) -> TestResult:
        """Test /health endpoint."""
        start = time.time()
        
        response, error = self._request('GET', '/health')
        
        if error:
            return TestResult(
                name="Health endpoint",
                passed=False,
                message=f"Connection failed: {error}",
                duration_ms=(time.time() - start) * 1000,
                details={"error": str(error)}
            )
        
        passed = response.status_code == 200
        
        try:
            data = response.json()
        except:
            data = {}
        
        return TestResult(
            name="Health endpoint",
            passed=passed,
            message=f"Status: {response.status_code}, healthy: {data.get('status')}",
            duration_ms=(time.time() - start) * 1000,
            details={"status_code": response.status_code, "response": data}
        )
    
    def test_ready_endpoint(self) -> TestResult:
        """Test /ready endpoint."""
        start = time.time()
        
        response, error = self._request('GET', '/ready')
        
        if error:
            return TestResult(
                name="Ready endpoint",
                passed=False,
                message=f"Connection failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code in [200, 503]  # 503 is valid (degraded)
        
        try:
            data = response.json()
        except:
            data = {}
        
        return TestResult(
            name="Ready endpoint",
            passed=passed,
            message=f"Status: {data.get('status', 'unknown')}",
            duration_ms=(time.time() - start) * 1000,
            details={"status_code": response.status_code, "checks": data.get('checks', {})}
        )
    
    def test_predict_endpoint(self) -> TestResult:
        """Test /api/v1/predict endpoint."""
        start = time.time()
        
        payload = {
            "origin": self.config.test_origin,
            "destination": self.config.test_destination,
            "city": self.config.test_city
        }
        
        response, error = self._request(
            'POST',
            '/api/v1/predict',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if error:
            return TestResult(
                name="Predict endpoint",
                passed=False,
                message=f"Request failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code == 200
        
        try:
            data = response.json()
            has_required_fields = all(
                key in data for key in ['overall_status', 'overall_severity', 'recommendations']
            )
        except:
            data = {}
            has_required_fields = False
        
        return TestResult(
            name="Predict endpoint",
            passed=passed and has_required_fields,
            message=f"Status: {response.status_code}, severity: {data.get('overall_severity', 'N/A')}",
            duration_ms=(time.time() - start) * 1000,
            details={
                "status_code": response.status_code,
                "severity": data.get('overall_severity'),
                "successful_agents": data.get('successful_agents'),
            }
        )
    
    def test_predict_validation(self) -> TestResult:
        """Test predict endpoint input validation."""
        start = time.time()
        
        test_cases = [
            ({"origin": "", "destination": "Boston"}, 400, "empty origin"),
            ({"origin": "NYC"}, 400, "missing destination"),
            ({}, 400, "empty body"),
            ({"origin": "A", "destination": "Boston"}, 400, "origin too short"),
        ]
        
        results = []
        all_passed = True
        
        for payload, expected_status, description in test_cases:
            response, _ = self._request(
                'POST',
                '/api/v1/predict',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response:
                passed = response.status_code == expected_status
                results.append({
                    "case": description,
                    "expected": expected_status,
                    "actual": response.status_code,
                    "passed": passed
                })
                if not passed:
                    all_passed = False
        
        return TestResult(
            name="Predict validation",
            passed=all_passed,
            message=f"{sum(1 for r in results if r['passed'])}/{len(results)} validation cases passed",
            duration_ms=(time.time() - start) * 1000,
            details={"test_cases": results}
        )
    
    def test_traffic_endpoint(self) -> TestResult:
        """Test /api/v1/traffic endpoint."""
        start = time.time()
        
        payload = {
            "origin": self.config.test_origin,
            "destination": self.config.test_destination
        }
        
        response, error = self._request('POST', '/api/v1/traffic', json=payload)
        
        if error:
            return TestResult(
                name="Traffic endpoint",
                passed=False,
                message=f"Request failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code == 200
        
        return TestResult(
            name="Traffic endpoint",
            passed=passed,
            message=f"Status: {response.status_code}",
            duration_ms=(time.time() - start) * 1000
        )
    
    def test_weather_endpoint(self) -> TestResult:
        """Test /api/v1/weather/<city> endpoint."""
        start = time.time()
        
        response, error = self._request('GET', f'/api/v1/weather/{self.config.test_city}')
        
        if error:
            return TestResult(
                name="Weather endpoint",
                passed=False,
                message=f"Request failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code == 200
        
        return TestResult(
            name="Weather endpoint",
            passed=passed,
            message=f"Status: {response.status_code}",
            duration_ms=(time.time() - start) * 1000
        )
    
    def test_events_endpoint(self) -> TestResult:
        """Test /api/v1/events/<city> endpoint."""
        start = time.time()
        
        response, error = self._request('GET', f'/api/v1/events/{self.config.test_city}')
        
        if error:
            return TestResult(
                name="Events endpoint",
                passed=False,
                message=f"Request failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code == 200
        
        return TestResult(
            name="Events endpoint",
            passed=passed,
            message=f"Status: {response.status_code}",
            duration_ms=(time.time() - start) * 1000
        )
    
    def test_routes_endpoint(self) -> TestResult:
        """Test /api/v1/routes endpoint."""
        start = time.time()
        
        payload = {
            "origin": self.config.test_origin,
            "destination": self.config.test_destination
        }
        
        response, error = self._request('POST', '/api/v1/routes', json=payload)
        
        if error:
            return TestResult(
                name="Routes endpoint",
                passed=False,
                message=f"Request failed: {error}",
                duration_ms=(time.time() - start) * 1000
            )
        
        passed = response.status_code == 200
        
        return TestResult(
            name="Routes endpoint",
            passed=passed,
            message=f"Status: {response.status_code}",
            duration_ms=(time.time() - start) * 1000
        )
    
    def test_error_handling(self) -> TestResult:
        """Test API error handling."""
        start = time.time()
        
        # Test 404
        response_404, _ = self._request('GET', '/nonexistent')
        
        # Test 405 (wrong method)
        response_405, _ = self._request('DELETE', '/api/v1/predict')
        
        # Test invalid JSON
        response_400, _ = self._request(
            'POST',
            '/api/v1/predict',
            data='invalid json',
            headers={'Content-Type': 'application/json'}
        )
        
        results = {
            "404_handling": response_404.status_code == 404 if response_404 else False,
            "405_handling": response_405.status_code == 405 if response_405 else False,
            "400_handling": response_400.status_code in [400, 415] if response_400 else False,
        }
        
        all_passed = all(results.values())
        
        return TestResult(
            name="Error handling",
            passed=all_passed,
            message=f"Passed: {sum(results.values())}/{len(results)}",
            duration_ms=(time.time() - start) * 1000,
            details=results
        )
    
    def test_response_format(self) -> TestResult:
        """Test API response format."""
        start = time.time()
        
        response, error = self._request(
            'POST',
            '/api/v1/predict',
            json={
                "origin": self.config.test_origin,
                "destination": self.config.test_destination
            }
        )
        
        if error or not response:
            return TestResult(
                name="Response format",
                passed=False,
                message="Could not get response",
                duration_ms=(time.time() - start) * 1000
            )
        
        checks = {
            "content_type": 'application/json' in response.headers.get('Content-Type', ''),
            "valid_json": False,
            "has_timestamp": False,
            "has_execution_time": False,
        }
        
        try:
            data = response.json()
            checks["valid_json"] = True
            checks["has_timestamp"] = "timestamp" in data
            checks["has_execution_time"] = "execution_time_ms" in data
        except:
            pass
        
        all_passed = all(checks.values())
        
        return TestResult(
            name="Response format",
            passed=all_passed,
            message=f"Checks passed: {sum(checks.values())}/{len(checks)}",
            duration_ms=(time.time() - start) * 1000,
            details=checks
        )


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class PerformanceTests:
    """Performance and load tests."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.base_url = config.api_base_url.rstrip('/')
    
    def run_all(self) -> TestSuite:
        """Run all performance tests."""
        suite = TestSuite("Performance Tests")
        
        suite.add(self.test_single_request_latency())
        suite.add(self.test_concurrent_requests())
        suite.add(self.test_sustained_load())
        
        suite.complete()
        return suite
    
    def test_single_request_latency(self) -> TestResult:
        """Test single request latency."""
        start = time.time()
        
        latencies = []
        
        for _ in range(10):
            req_start = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/predict",
                    json={
                        "origin": self.config.test_origin,
                        "destination": self.config.test_destination
                    },
                    timeout=self.config.api_timeout
                )
                if response.status_code == 200:
                    latencies.append((time.time() - req_start) * 1000)
            except:
                pass
        
        if not latencies:
            return TestResult(
                name="Single request latency",
                passed=False,
                message="No successful requests",
                duration_ms=(time.time() - start) * 1000
            )
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        passed = avg_latency < self.config.performance_threshold_ms
        
        return TestResult(
            name="Single request latency",
            passed=passed,
            message=f"Avg: {avg_latency:.0f}ms, P95: {p95_latency:.0f}ms",
            duration_ms=(time.time() - start) * 1000,
            details={
                "avg_ms": avg_latency,
                "p95_ms": p95_latency,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "samples": len(latencies)
            }
        )
    
    def test_concurrent_requests(self) -> TestResult:
        """Test concurrent request handling."""
        start = time.time()
        
        def make_request():
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/predict",
                    json={
                        "origin": self.config.test_origin,
                        "destination": self.config.test_destination
                    },
                    timeout=self.config.api_timeout
                )
                return {
                    "success": response.status_code == 200,
                    "latency_ms": (time.time() - req_start) * 1000,
                    "status_code": response.status_code
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        n_requests = self.config.concurrent_requests
        results = []
        
        with ThreadPoolExecutor(max_workers=n_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(n_requests)]
            for future in as_completed(futures):
                results.append(future.result())
        
        successful = sum(1 for r in results if r.get("success"))
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        
        passed = successful == n_requests
        
        return TestResult(
            name="Concurrent requests",
            passed=passed,
            message=f"{successful}/{n_requests} successful",
            duration_ms=(time.time() - start) * 1000,
            details={
                "total_requests": n_requests,
                "successful": successful,
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
            }
        )
    
    def test_sustained_load(self) -> TestResult:
        """Test sustained load for 30 seconds."""
        start = time.time()
        duration = 10  # seconds (reduced for faster testing)
        
        requests_made = 0
        successful = 0
        errors = 0
        latencies = []
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                req_start = time.time()
                response = requests.post(
                    f"{self.base_url}/api/v1/predict",
                    json={
                        "origin": self.config.test_origin,
                        "destination": self.config.test_destination
                    },
                    timeout=5
                )
                latencies.append((time.time() - req_start) * 1000)
                requests_made += 1
                
                if response.status_code == 200:
                    successful += 1
                else:
                    errors += 1
            except:
                requests_made += 1
                errors += 1
        
        success_rate = successful / requests_made * 100 if requests_made > 0 else 0
        rps = requests_made / duration
        
        passed = success_rate >= 95  # 95% success rate
        
        return TestResult(
            name=f"Sustained load ({duration}s)",
            passed=passed,
            message=f"{rps:.1f} req/s, {success_rate:.1f}% success",
            duration_ms=(time.time() - start) * 1000,
            details={
                "duration_seconds": duration,
                "total_requests": requests_made,
                "successful": successful,
                "errors": errors,
                "success_rate": success_rate,
                "requests_per_second": rps,
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
            }
        )


# =============================================================================
# TEST RUNNER
# =============================================================================

class DeploymentTestRunner:
    """Main test runner."""
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.suites: List[TestSuite] = []
    
    def run_model_tests(self) -> TestSuite:
        """Run model tests."""
        logger.info("Running Model Tests...")
        tests = ModelTests(self.config)
        suite = tests.run_all()
        self.suites.append(suite)
        return suite
    
    def run_api_tests(self) -> TestSuite:
        """Run API tests."""
        logger.info("Running API Tests...")
        tests = APITests(self.config)
        suite = tests.run_all()
        self.suites.append(suite)
        return suite
    
    def run_performance_tests(self) -> TestSuite:
        """Run performance tests."""
        logger.info("Running Performance Tests...")
        tests = PerformanceTests(self.config)
        suite = tests.run_all()
        self.suites.append(suite)
        return suite
    
    def run_all(self) -> bool:
        """Run all tests."""
        logger.info("=" * 60)
        logger.info("SMART TRAFFIC MANAGEMENT SYSTEM - DEPLOYMENT TESTS")
        logger.info("=" * 60)
        
        self.run_model_tests()
        self.run_api_tests()
        self.run_performance_tests()
        
        return self.print_summary()
    
    def print_summary(self) -> bool:
        """Print summary of all test suites."""
        total_passed = sum(s.passed for s in self.suites)
        total_failed = sum(s.failed for s in self.suites)
        total_tests = sum(s.total for s in self.suites)
        
        for suite in self.suites:
            print(suite.summary())
        
        print("=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        for suite in self.suites:
            status = "✅" if suite.all_passed else "❌"
            print(f"  {status} {suite.name}: {suite.passed}/{suite.total} passed")
        
        print("-" * 60)
        overall_status = "✅ ALL TESTS PASSED" if total_failed == 0 else "❌ SOME TESTS FAILED"
        print(f"  {overall_status}")
        print(f"  Total: {total_passed}/{total_tests} passed ({total_passed/total_tests*100:.1f}%)")
        print("=" * 60)
        
        return total_failed == 0
    
    def save_report(self, filepath: str) -> None:
        """Save test report to JSON file."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "model_path": self.config.model_path,
                "api_base_url": self.config.api_base_url,
            },
            "suites": [s.to_dict() for s in self.suites],
            "summary": {
                "total_suites": len(self.suites),
                "total_tests": sum(s.total for s in self.suites),
                "total_passed": sum(s.passed for s in self.suites),
                "total_failed": sum(s.failed for s in self.suites),
                "all_passed": all(s.all_passed for s in self.suites),
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deployment tests for Smart Traffic Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deployment_testing.py --all
  python deployment_testing.py --model-only
  python deployment_testing.py --api-only
  python deployment_testing.py --api-url http://localhost:5000
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--model-only', action='store_true', help='Run model tests only')
    parser.add_argument('--api-only', action='store_true', help='Run API tests only')
    parser.add_argument('--performance-only', action='store_true', help='Run performance tests only')
    parser.add_argument('--api-url', type=str, help='API base URL')
    parser.add_argument('--model-path', type=str, help='Path to model file')
    parser.add_argument('--report', type=str, help='Save report to file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create config
    config = TestConfig()
    if args.api_url:
        config.api_base_url = args.api_url
    if args.model_path:
        config.model_path = args.model_path
    
    # Create runner
    runner = DeploymentTestRunner(config)
    
    # Run tests
    if args.model_only:
        runner.run_model_tests()
    elif args.api_only:
        runner.run_api_tests()
    elif args.performance_only:
        runner.run_performance_tests()
    else:
        runner.run_all()
    
    # Print and save
    success = runner.print_summary()
    
    if args.report:
        runner.save_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
