import unittest
from Traffic_Management_System.src.utils.helpers import safe_get

class TestUtils(unittest.TestCase):
    def test_safe_get_existing_key(self):
        data = {"a": 1, "b": 2}
        self.assertEqual(safe_get(data, "a"), 1)

    def test_safe_get_missing_key(self):
        data = {"a": 1, "b": 2}
        self.assertIsNone(safe_get(data, "c"))

    def test_safe_get_default_value(self):
        data = {"a": 1, "b": 2}
        self.assertEqual(safe_get(data, "c", default=100), 100)

    def test_safe_get_nested_dict(self):
        data = {"a": {"x": 5}, "b": 2}
        self.assertEqual(safe_get(data, "a"), {"x": 5})

    def test_safe_get_non_dict(self):
        self.assertIsNone(safe_get(None, "a"))
        self.assertIsNone(safe_get(123, "a"))

if __name__ == '__main__':
    unittest.main()
