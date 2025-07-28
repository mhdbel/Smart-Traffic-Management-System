import unittest
from src.mobile_app.main import fetch_city_data

class TestMobileApp(unittest.TestCase):
    def test_fetch_city_data(self):
        result = fetch_city_data("TestCity")
        self.assertIsInstance(result, dict)
        self.assertIn("traffic", result)
        self.assertIn("weather", result)

if __name__ == '__main__':
    unittest.main()