import unittest
import numpy as np
from ml_lib.cluster import KMeans


class TestKMeans(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_fit_sets_centroids_and_inertia(self):
        X = np.random.rand(20, 3)
        kmeans = KMeans(n_clusters=3)

        kmeans.fit(X)

        self.assertTrue(hasattr(kmeans, "centroids"))
        self.assertTrue(hasattr(kmeans, "inertia_"))

    def test_centroids_shape(self):
        X = np.random.rand(50, 4)
        kmeans = KMeans(n_clusters=5)

        kmeans.fit(X)

        self.assertEqual(kmeans.centroids.shape, (5, 4))

    def test_predict_shape(self):
        X = np.random.rand(30, 2)
        kmeans = KMeans(n_clusters=4)

        kmeans.fit(X)
        labels = kmeans.predict(X)

        self.assertEqual(labels.shape, (30,))

    def test_predict_cluster_range(self):
        X = np.random.rand(40, 2)
        kmeans = KMeans(n_clusters=3)

        kmeans.fit(X)
        labels = kmeans.predict(X)

        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < 3))

    def test_two_obvious_clusters(self):
        X = np.vstack([
            np.random.randn(50, 2) * 0.1 + np.array([0, 0]),
            np.random.randn(50, 2) * 0.1 + np.array([5, 5])
        ])

        kmeans = KMeans(n_clusters=2)
        labels = kmeans.fit_predict(X)

        self.assertEqual(len(np.unique(labels)), 2)

    def test_inertia_non_negative(self):
        X = np.random.rand(25, 3)
        kmeans = KMeans(n_clusters=3)

        kmeans.fit(X)

        self.assertGreaterEqual(kmeans.inertia_, 0)

    def test_fit_predict_returns_labels(self):
        X = np.random.rand(20, 2)
        kmeans = KMeans(n_clusters=2)

        labels = kmeans.fit_predict(X)

        self.assertIsNotNone(labels)
        self.assertEqual(labels.shape, (20,))


if __name__ == "__main__":
    unittest.main()
