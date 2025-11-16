from ml_lib import stats
import numpy as np

def test_sigmoid_matrix():
    x = np.array([[0, 1], [-1, 2]])
    expected = 1 / (1 + np.exp(-x))
    result = stats.sigmoid(x)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_sigmoid_zero_matrix():
    x = np.zeros((2, 2))
    expected = np.full((2, 2), 0.5)
    result = stats.sigmoid(x)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_softmax_matrix_rows_sum_to_one():
    x = np.array([[1, 2, 3], [0, 0, 0]])
    result = stats.softmax(x)
    row_sums = np.sum(result, axis=1)
    expected = np.ones(x.shape[0])
    assert np.allclose(row_sums, expected), f"Each row should sum to 1, got {row_sums}"

def test_softmax_matrix_values_nonnegative():
    x = np.random.randn(3, 4)
    result = stats.softmax(x)
    assert np.all(result >= 0), f"Softmax should produce nonnegative values, got {result}"

def test_softmax_invariance_to_constant_shift():
    x = np.array([[1, 2, 3]])
    shifted_x = x + 100
    result = stats.softmax(x)
    shifted_result = stats.softmax(shifted_x)
    assert np.allclose(result, shifted_result), "Softmax should be invariant to constant shifts"

def test_softmax():
    test_softmax_invariance_to_constant_shift()
    test_softmax_matrix_rows_sum_to_one()
    test_softmax_matrix_values_nonnegative()
    print("All softmax tests passed successfully!")

def test_sigmoid():
    test_sigmoid_matrix()
    test_sigmoid_zero_matrix()
    print("All sigmoid tests passed successfully!")

def test_entropy():
    # Uniform distribution — entropy should be log(n)
    x = np.array([.3, .7], dtype=float)
    expected = 0.8812908992306927
    
    result = stats.entropy(x)
    assert isinstance(result, float), "entropy should return float"
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    # Degenerate distribution — entropy = 0
    x = np.array([1, 0, 0, 0], dtype=float)
    expected = 0.0
    result = stats.entropy(x)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    print("test_entropy passed!")


def test_gini_index():
    # Pure distribution (only one class) — gini = 0
    x = np.array([1, 0, 0], dtype=float)
    expected = 0.0
    result = stats.gini_index(x)
    assert isinstance(result, float), "gini_index should return float"
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    # Two equally likely classes — gini = 0.5
    x = np.array([.5, .5], dtype=float)
    expected = 0.5 
    result = stats.gini_index(x)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    # Three-class uniform distribution — gini = 1 - 3*(1/3)^2 = 2/3
    x = np.array([1/3, 1/3, 1/3], dtype=float)
    expected = 2/3
    result = stats.gini_index(x)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    print("test_gini_index passed!")


def main():
    test_sigmoid()
    test_softmax()
    test_entropy()
    test_gini_index()

if __name__ == '__main__':
    main()