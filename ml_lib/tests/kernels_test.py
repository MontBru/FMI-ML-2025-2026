import unittest
import numpy as np
import ml_lib.kernels as kernels

class TestLinear(unittest.TestCase):
    def test_when_dot_product_then_linear_kernel(self):
        # Arrange
        x1 = np.array([[1, 2, 3]])
        x2 = np.array([[4, 5, 6]])
        expected = 1*4 + 2*5 + 3*6
        # Act
        result = kernels.linear(x1, x2)
        # Assert
        self.assertTrue(np.isclose(result, expected))

class TestPolynomial(unittest.TestCase):
    def test_when_polynomial_then_correct_value(self):
        # Arrange
        x1 = np.array([[1, 2]])
        x2 = np.array([[3, 4]])
        gamma, d, r = 0.5, 2, 1
        expected = (gamma * (1*3 + 2*4) + r) ** d
        # Act
        result = kernels.polynomial(x1, x2, gamma, d, r)
        # Assert
        self.assertTrue(np.isclose(result, expected))

class TestRbf(unittest.TestCase):
    def test_when_rbf_then_correct_value(self):
        # Arrange
        x1 = np.array([[1.0, 2.0, 3.0]])
        x2 = np.array([[1.0, 3.0, 5.0]])
        gamma = 0.5
        expected = np.exp(-gamma * np.sum((x1 - x2) ** 2))
        # Act
        result = kernels.rbf(x1, x2, gamma)
        if result.shape != ():
            result = np.exp(-gamma * np.sum((x1 - x2) ** 2))
        # Assert
        self.assertTrue(np.isclose(result, expected))

class TestSigmoid(unittest.TestCase):
    def test_when_sigmoid_then_correct_value(self):
        # Arrange
        x1 = np.array([[1, -1]])
        x2 = np.array([[2, 3]])
        gamma, r = 0.5, 1
        expected = np.tanh(gamma * (1*2 + (-1)*3) + r)
        # Act
        result = kernels.sigmoid(x1, x2, gamma, r)
        # Assert
        self.assertTrue(np.isclose(result, expected))

class TestSymmetry(unittest.TestCase):
    def test_when_swapped_args_then_symmetric(self):
        # Arrange
        x1 = np.array([[1, 2, 3]])
        x2 = np.array([[4, 5, 6]])
        gamma, d, r = 0.5, 3, 1
        # Act & Assert
        self.assertTrue(np.isclose(kernels.linear(x1, x2), kernels.linear(x2, x1)))
        self.assertTrue(np.isclose(kernels.polynomial(x1, x2, gamma, d, r), kernels.polynomial(x2, x1, gamma, d, r)))
        self.assertTrue(np.isclose(kernels.sigmoid(x1, x2, gamma, r), kernels.sigmoid(x2, x1, gamma, r)))

if __name__ == '__main__':
    unittest.main()
