import unittest
import numpy as np
from scipy.sparse import csr_matrix

from ml_lib.naive_bayes import MultinomialNB


class TestMultinomialNB(unittest.TestCase):

    def setUp(self):
        # Simple toy dataset
        self.X_dense = np.array([
            [2, 1, 0],
            [1, 0, 1],
            [0, 2, 1],
            [0, 1, 2]
        ])

        self.y = np.array([0, 0, 1, 1])

        self.X_sparse = csr_matrix(self.X_dense)

    def test_fit_runs(self):
        model = MultinomialNB()
        model.fit(self.X_dense, self.y)

        self.assertTrue(hasattr(model, 'feature_log_prob_'))
        self.assertTrue(hasattr(model, 'class_log_prior_'))

    def test_predict_shape(self):
        model = MultinomialNB()
        model.fit(self.X_dense, self.y)

        preds = model.predict(self.X_dense)
        self.assertEqual(preds.shape, self.y.shape)

    def test_predict_valid_labels(self):
        model = MultinomialNB()
        model.fit(self.X_dense, self.y)

        preds = model.predict(self.X_dense)
        for p in preds:
            self.assertIn(p, model.classes_)

    def test_score_range(self):
        model = MultinomialNB()
        model.fit(self.X_dense, self.y)

        score = model.score(self.X_dense, self.y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_sparse_input(self):
        model = MultinomialNB()
        model.fit(self.X_sparse, self.y)

        preds = model.predict(self.X_sparse)
        self.assertEqual(len(preds), len(self.y))

    def test_perfect_separation(self):
        # Clearly separable dataset
        X = np.array([
            [5, 0],
            [4, 0],
            [0, 5],
            [0, 4]
        ])
        y = np.array([0, 0, 1, 1])

        model = MultinomialNB()
        model.fit(X, y)

        preds = model.predict(X)
        self.assertTrue(np.array_equal(preds, y))

    def test_laplace_smoothing_no_minus_inf(self):
        model = MultinomialNB(alpha=1.0)
        model.fit(self.X_dense, self.y)

        self.assertFalse(np.isneginf(model.feature_log_prob_).any())


if __name__ == "__main__":
    unittest.main()
