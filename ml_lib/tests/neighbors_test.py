import unittest
import numpy as np
from ml_lib.neighbors import KNeighborsClassifier

class TestFit(unittest.TestCase):
    def test_when_fit_called_then_model_stores_training_data(self):
        # Arrange
        X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
        y_train = np.array([0, 0, 1, 1])
        expected_X = X_train
        expected_y = y_train
        # Act
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        # Assert
        np.testing.assert_array_equal(model.x, expected_X)
        np.testing.assert_array_equal(model.y, expected_y)

class TestPredict(unittest.TestCase):
    def test_when_k1_then_predicts_nearest_neighbor(self):
        # Arrange
        X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1.2, 1.9], [5.5, 8.5]])
        expected = np.array([0, 1])
        # Act
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        actual = model.predict(X_test)
        # Assert
        np.testing.assert_array_equal(actual, expected)

    def test_when_k3_then_predicts_majority_vote(self):
        # Arrange
        X_train = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1.2, 1.9], [5.5, 8.5]])
        expected = np.array([0, 1])
        # Act
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        actual = model.predict(X_test)
        # Assert
        np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
