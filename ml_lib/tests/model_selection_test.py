import unittest
import numpy as np
import pandas as pd
from collections import Counter
from ml_lib import model_selection

class TestTrainTestSplit(unittest.TestCase):
    def test_when_basic_split_then_shapes_and_labels_correct(self):
        # Arrange
        X = pd.DataFrame(np.arange(20).reshape(10, 2))
        y = pd.Series(np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))
        # Act
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, stratify=None)
        # Assert
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))
        self.assertEqual(X_train.shape[1], X.shape[1])
        for index in range(X_train.shape[0]):
            row = X_train.iloc[index]
            mask = (X == row).all(axis=1)
            match_idx = X.index[mask]
            self.assertEqual(len(match_idx), 1)
            self.assertEqual(y.loc[match_idx[0]], y_train.iloc[index])
        for index in range(X_test.shape[0]):
            row = X_test.iloc[index]
            mask = (X == row).all(axis=1)
            match_idx = X.index[mask]
            self.assertEqual(len(match_idx), 1)
            self.assertEqual(y.loc[match_idx[0]], y_test.iloc[index])

    def test_when_stratify_then_class_distributions_preserved(self):
        # Arrange
        rng = np.random.default_rng(123)
        X = pd.DataFrame(np.arange(200).reshape(100, 2))
        y = pd.Series(np.array([rng.integers(0,3) for i in range(100)]))
        # Act
        X_train_s, X_test_s, y_train_s, y_test_s = model_selection.train_test_split(X, y, test_size=0.3, stratify=y)
        orig_dist = np.array(list(Counter(y).values()), dtype=float)
        train_dist = np.array(list(Counter(y_train_s).values()), dtype=float)
        test_dist = np.array(list(Counter(y_test_s).values()), dtype=float)
        orig_dist /= orig_dist.sum()
        train_dist /= train_dist.sum()
        test_dist /= test_dist.sum()
        # Assert
        self.assertTrue(np.allclose(train_dist, orig_dist, atol=0.1))
        self.assertTrue(np.allclose(test_dist, orig_dist, atol=0.1))

if __name__ == '__main__':
    unittest.main()