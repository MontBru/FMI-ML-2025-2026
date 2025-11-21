import unittest
import numpy as np
from ml_lib import stats

class TestSigmoid(unittest.TestCase):
    def test_when_matrix_given_then_elementwise_sigmoid(self):
        # Arrange
        x = np.array([[0, 1], [-1, 2]])
        expected = 1 / (1 + np.exp(-x))
        # Act
        actual = stats.sigmoid(x)
        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_zero_matrix_then_half_everywhere(self):
        # Arrange
        x = np.zeros((2, 2))
        expected = np.full((2, 2), 0.5)
        # Act
        actual = stats.sigmoid(x)
        # Assert
        np.testing.assert_allclose(actual, expected)

class TestSoftmax(unittest.TestCase):
    def test_when_matrix_given_then_rows_sum_to_one(self):
        # Arrange
        x = np.array([[1, 2, 3], [0, 0, 0]])
        expected = np.ones(x.shape[0])
        # Act
        actual = stats.softmax(x)
        row_sums = np.sum(actual, axis=1)
        # Assert
        np.testing.assert_allclose(row_sums, expected)

    def test_when_matrix_given_then_values_nonnegative(self):
        # Arrange
        x = np.random.randn(3, 4)
        expected = True
        # Act
        actual = stats.softmax(x)
        # Assert
        self.assertTrue(np.all(actual >= 0))

    def test_when_constant_shift_then_invariant(self):
        # Arrange
        x = np.array([[1, 2, 3]])
        shifted_x = x + 100
        # Act
        actual = stats.softmax(x)
        shifted_actual = stats.softmax(shifted_x)
        # Assert
        np.testing.assert_allclose(actual, shifted_actual)

class TestEntropy(unittest.TestCase):
    def test_when_nonuniform_distribution_then_entropy_computed(self):
        # Arrange
        x = np.array([.3, .7], dtype=float)
        expected = 0.8812908992306927
        # Act
        actual = stats.entropy(x)
        # Assert
        self.assertIsInstance(actual, float)
        self.assertTrue(np.isclose(actual, expected))

    def test_when_degenerate_distribution_then_entropy_zero(self):
        # Arrange
        x = np.array([1, 0, 0, 0], dtype=float)
        expected = 0.0
        # Act
        actual = stats.entropy(x)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

class TestGiniIndex(unittest.TestCase):
    def test_when_pure_distribution_then_gini_zero(self):
        # Arrange
        x = np.array([1, 0, 0], dtype=float)
        expected = 0.0
        # Act
        actual = stats.gini_index(x)
        # Assert
        self.assertIsInstance(actual, float)
        self.assertTrue(np.isclose(actual, expected))

    def test_when_two_classes_equal_then_gini_half(self):
        # Arrange
        x = np.array([.5, .5], dtype=float)
        expected = 0.5
        # Act
        actual = stats.gini_index(x)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

    def test_when_three_class_uniform_then_gini_two_thirds(self):
        # Arrange
        x = np.array([1/3, 1/3, 1/3], dtype=float)
        expected = 2/3
        # Act
        actual = stats.gini_index(x)
        # Assert
        self.assertTrue(np.isclose(actual, expected))

if __name__ == '__main__':
    unittest.main()
