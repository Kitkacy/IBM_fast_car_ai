import unittest
from torcs_scr.utils import clip, destringify

class TestScrUtils(unittest.TestCase):
    def test_clip(self):
        self.assertEqual(clip(5, 0, 10), 5)
        self.assertEqual(clip(-1, 0, 10), 0)
        self.assertEqual(clip(11, 0, 10), 10)
        self.assertEqual(clip(0, 0, 10), 0)
        self.assertEqual(clip(10, 0, 10), 10)

    def test_destringify_single_float(self):
        self.assertEqual(destringify("1.23"), 1.23)
        self.assertEqual(destringify(["4.56"]), 4.56)

    def test_destringify_list(self):
        self.assertEqual(destringify(["1.0", "2.0", "3.0"]), [1.0, 2.0, 3.0])

    def test_destringify_invalid(self):
        self.assertEqual(destringify("abc"), "abc")

    def test_destringify_empty(self):
        self.assertEqual(destringify([]), [])
        self.assertEqual(destringify(""), "")

if __name__ == "__main__":
    unittest.main()
