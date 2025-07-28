import unittest
import pickle

class TestModels(unittest.TestCase):
    def test_model_load_and_predict(self):
        with open('models/traffic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample_input = [[1, 0, 3]]  # Example features
        prediction = model.predict(sample_input)
        self.assertIsInstance(prediction, (list, tuple, np.ndarray))

if __name__ == '__main__':
    unittest.main()