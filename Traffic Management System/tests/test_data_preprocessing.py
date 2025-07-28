import unittest
from src.utils.data_preprocessing import clean_data, feature_engineer

class TestDataPreprocessing(unittest.TestCase):
    def test_clean_data(self):
        raw_data = [{"city": "TestCity", "traffic": None}]
        cleaned = clean_data(raw_data)
        self.assertTrue(all("city" in item for item in cleaned))

    def test_feature_engineer(self):
        data = [{"city": "TestCity", "traffic": 3}]
        features = feature_engineer(data)
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)

if __name__ == '__main__':
    unittest.main()