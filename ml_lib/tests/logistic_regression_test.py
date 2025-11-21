import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from ml_lib import linear_model

class TestLogisticRegressionBinary(unittest.TestCase):
    def test_when_binary_data_then_high_accuracy(self):
        # Arrange
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            random_state=42
        )
        expected = True  # accuracy > 0.85
        # Act
        model = linear_model.LogisticRegression()
        model.fit(X, y, verbose=True)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        actual = acc > 0.85
        # Assert
        self.assertTrue(actual)

class TestLogisticRegressionMulticlass(unittest.TestCase):
    def test_when_multiclass_data_then_high_accuracy_and_all_classes(self):
        # Arrange
        X, y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=4,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=0
        )
        expected = True  # accuracy > 0.75
        # Act
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        actual = acc > 0.75 and set(np.unique(y_pred)) <= set(np.unique(y))
        # Assert
        self.assertTrue(actual)

if __name__ == '__main__':
    unittest.main()
