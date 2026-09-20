"""Tests for src.utils.seed."""

import unittest

from src.utils.seed import DEFAULT_SEED, set_seed


class TestSetSeed(unittest.TestCase):
    def test_returns_seed(self):
        self.assertEqual(set_seed(123), 123)

    def test_default_seed(self):
        self.assertEqual(set_seed(), DEFAULT_SEED)

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            set_seed(-1)

    def test_rejects_non_int(self):
        for bad in (3.5, "42", None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    set_seed(bad)

    def test_deterministic_numpy(self):
        import numpy as np

        set_seed(7)
        a = np.random.rand(5)
        set_seed(7)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
