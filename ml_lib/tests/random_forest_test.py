
import unittest
import numpy as np
from ml_lib.tree import RandomForestClassifier


class TestFitPredict(unittest.TestCase):
    def test_when_small_dataset_then_predict_shape_and_labels(self):
        # Arrange
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 0, 1, 1])
        expected_shape = y.shape
        # Act
        rf = RandomForestClassifier(n_estimators=3, max_features=2)
        rf.fit(X, y)
        actual = rf.predict(X)
        # Assert
        self.assertEqual(actual.shape, expected_shape)
        self.assertTrue(set(np.unique(actual)).issubset({0, 1}))

class TestBootstrapSampling(unittest.TestCase):
    def test_when_fit_called_then_estimators_created(self):
        # Arrange
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, size=20)
        expected = 5
        # Act
        rf = RandomForestClassifier(n_estimators=expected, max_features=2)
        rf.fit(X, y)
        # Assert
        self.assertTrue(hasattr(rf, "estimators"))
        self.assertEqual(len(rf.estimators), expected)

class TestMajorityVote(unittest.TestCase):
    def test_when_estimators_predict_then_majority_vote(self):
        # Arrange
        rf = RandomForestClassifier(n_estimators=3)
        rf.estimators = [None, None, None]
        rf.feature_indices_for_estimators = [None, None, None]
        class Est:
            def __init__(self, pred):
                self.pred = pred
            def predict(self, X):
                return np.array(self.pred)
        rf.estimators[0] = Est([0, 1, 1])
        rf.estimators[1] = Est([1, 1, 0])
        rf.estimators[2] = Est([1, 1, 1])
        X_dummy = np.zeros((3, 2))
        expected = np.array([1, 1, 1])
        # Act
        actual = rf.predict(X_dummy)
        # Assert
        np.testing.assert_array_equal(actual, expected)

class TestMaxFeatures(unittest.TestCase):
    def test_when_fit_then_each_estimator_uses_max_features(self):
        # Arrange
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        max_features = 3
        # Act
        rf = RandomForestClassifier(n_estimators=4, max_features=max_features)
        rf.fit(X, y)
        # Assert
        for feature_idx in rf.feature_indices_for_estimators:
            self.assertEqual(len(feature_idx), max_features)
            self.assertLess(max(feature_idx), X.shape[1])

if __name__ == "__main__":
    unittest.main()