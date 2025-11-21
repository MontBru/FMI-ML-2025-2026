import numpy as np
from ml_lib.linear_model import Ridge
import unittest


class TestFit(unittest.TestCase):
    def test_when_perfect_linear_data_then_fit_predicts_perfectly(self):
        # Arrange
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = 2*X.flatten() + 1
        expected = y
        # Act
        model = Ridge()
        model.fit(X, y)
        actual = model.predict(X)
        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_when_mismatched_shapes_then_raises(self):
        # Arrange
        X = np.random.randn(10, 3)
        y = np.random.randn(9)
        # Act & Assert
        model = Ridge()
        with self.assertRaises(Exception):
            model.fit(X, y)

class TestScore(unittest.TestCase):
    def test_when_perfect_fit_then_r2_is_one(self):
        # Arrange
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = 2*X.flatten() + 1
        expected = 1.0
        # Act
        model = Ridge()
        model.fit(X, y)
        actual = model.score(X, y)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_model_worse_than_baseline_then_r2_negative(self):
        # Arrange
        X = np.array([[0], [1], [2], [3]], dtype=float)
        y = np.array([0, 0, 0, 0], dtype=float)
        y_bad_pred = np.array([10, 10, 10, 10])
        expected = True  # r2 < 0
        # Act
        model = Ridge()
        model.fit(X, y_bad_pred)
        actual = model.score(X, y)
        # Assert
        self.assertTrue(actual < 0)

if __name__ == '__main__':
    unittest.main()