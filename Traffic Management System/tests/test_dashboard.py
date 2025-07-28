import unittest
from src.dashboard.app import app

class TestDashboard(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_dashboard_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Dashboard", response.data)

    def test_dashboard_city_stats(self):
        response = self.app.get('/city-stats?city=TestCity')
        self.assertEqual(response.status_code, 200)
        # Add more assertions for JSON or HTML content as needed

if __name__ == '__main__':
    unittest.main()