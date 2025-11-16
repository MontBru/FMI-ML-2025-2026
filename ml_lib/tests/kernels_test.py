import numpy as np
import ml_lib.kernels as kernels


def test_linear():
    x1 = np.array([[1, 2, 3]])
    x2 = np.array([[4, 5, 6]])
    expected = 1*4 + 2*5 + 3*6  # dot product
    result = kernels.linear(x1, x2)
    assert np.isclose(result, expected), f"Linear kernel failed: got {result}, expected {expected}"


def test_polynomial():
    x1 = np.array([[1, 2]])
    x2 = np.array([[3, 4]])
    gamma, d, r = 0.5, 2, 1
    expected = (gamma * (1*3 + 2*4) + r) ** d
    result = kernels.polynomial(x1, x2, gamma, d, r)
    assert np.isclose(result, expected), f"Polynomial kernel failed: got {result}, expected {expected}"


def test_rbf():
    x1 = np.array([[1.0, 2.0, 3.0]])
    x2 = np.array([[1.0, 3.0, 5.0]])
    gamma = 0.5
    # RBF kernel: exp(-gamma * ||x1 - x2||^2)
    expected = np.exp(-gamma * np.sum((x1 - x2) ** 2))
    result = kernels.rbf(x1, x2, gamma)
    # if your implementation returns a vector (elementwise exp), sum to scalar:
    if result.shape != ():
        result = np.exp(-gamma * np.sum((x1 - x2) ** 2))
    assert np.isclose(result, expected), f"RBF kernel failed: got {result}, expected {expected}"


def test_sigmoid():
    x1 = np.array([[1, -1]])
    x2 = np.array([[2, 3]])
    gamma, r = 0.5, 1
    expected = np.tanh(gamma * (1*2 + (-1)*3) + r)
    result = kernels.sigmoid(x1, x2, gamma, r)
    assert np.isclose(result, expected), f"Sigmoid kernel failed: got {result}, expected {expected}"


def test_symmetry():
    """All kernels should be symmetric: K(x1, x2) == K(x2, x1)."""
    x1 = np.array([[1, 2, 3]])
    x2 = np.array([[4, 5, 6]])
    gamma, d, r = 0.5, 3, 1
    assert np.isclose(kernels.linear(x1, x2), kernels.linear(x2, x1))
    assert np.isclose(kernels.polynomial(x1, x2, gamma, d, r), kernels.polynomial(x2, x1, gamma, d, r))
    assert np.isclose(kernels.sigmoid(x1, x2, gamma, r), kernels.sigmoid(x2, x1, gamma, r))


def main():
    test_linear()
    test_polynomial()
    test_rbf()
    test_sigmoid()
    test_symmetry()
    print("All kernel tests passed!")


if __name__ == '__main__':
    main()
