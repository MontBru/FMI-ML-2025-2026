import unittest
import numpy as np
from ml_lib.hierarchy import linkage, AgglomerativeClustering


class TestLinkage(unittest.TestCase):

    def setUp(self):
        # Two simple point sets in 2D
        self.A = np.array([[0.0, 0.0],
                           [1.0, 0.0]])

        self.B = np.array([[0.0, 1.0],
                           [1.0, 1.0]])

        # Precomputed pairwise distances:
        # A[0]-B[0] = 1
        # A[0]-B[1] = sqrt(2)
        # A[1]-B[0] = sqrt(2)
        # A[1]-B[1] = 1
        self.distances = np.array([
            [1.0, np.sqrt(2)],
            [np.sqrt(2), 1.0]
        ])

    def test_single_linkage(self):
        result = linkage(self.A, self.B, method="single")
        expected = np.min(self.distances)
        self.assertAlmostEqual(result, expected)

    def test_complete_linkage(self):
        result = linkage(self.A, self.B, method="complete")
        expected = np.max(self.distances)
        self.assertAlmostEqual(result, expected)

    def test_average_linkage(self):
        result = linkage(self.A, self.B, method="average")
        expected = np.mean(self.distances)
        self.assertAlmostEqual(result, expected)

    def test_centroid_linkage(self):
        # Correct centroid distance
        centroid_A = np.mean(self.A, axis=0)
        centroid_B = np.mean(self.B, axis=0)
        expected = np.linalg.norm(centroid_A - centroid_B)

        result = linkage(self.A, self.B, method="centroid")
        self.assertAlmostEqual(result, expected)

    def test_invalid_method(self):
        result = linkage(self.A, self.B, method="unknown")
        self.assertIsNone(result)

    def test_single_point_clusters(self):
        A = np.array([[0.0, 0.0]])
        B = np.array([[3.0, 4.0]])

        result = linkage(A, B, method="single")
        self.assertAlmostEqual(result, 5.0)

class TestFit(unittest.TestCase):
    def test_single_point(self):
        X = np.array([[0.0, 0.0]])

        model = AgglomerativeClustering(distance_threshold=1.0)
        model.fit(X)

        self.assertEqual(len(model.cluster_history), 1)
        clusters, _ = model.cluster_history[-1]
        self.assertEqual(clusters, [[0]])

    def test_two_close_points_merge(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0]
        ])

        model = AgglomerativeClustering(linkage="single", distance_threshold=0.5)
        model.fit(X)

        final_clusters, _ = model.cluster_history[-1]
        self.assertEqual(len(final_clusters), 1)
        self.assertCountEqual(final_clusters[0], [0, 1])

    def test_two_far_points_no_merge(self):
        X = np.array([
            [0.0, 0.0],
            [10.0, 0.0]
        ])

        model = AgglomerativeClustering(linkage="single", distance_threshold=1.0)
        model.fit(X)

        final_clusters, _ = model.cluster_history[-1]
        self.assertEqual(len(final_clusters), 2)

    def test_three_points_transitive_merge(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0]
        ])

        model = AgglomerativeClustering(linkage="single", distance_threshold=0.15)
        model.fit(X)

        final_clusters, _ = model.cluster_history[-1]
        self.assertEqual(len(final_clusters), 1)
        self.assertCountEqual(final_clusters[0], [0, 1, 2])

    def test_cluster_history_monotonic(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 0.0]
        ])

        model = AgglomerativeClustering(linkage="single", distance_threshold=0.2)
        model.fit(X)

        sizes = [len(clusters) for clusters, _ in model.cluster_history]
        self.assertTrue(all(sizes[i] >= sizes[i+1] for i in range(len(sizes)-1)))

    def test_all_points_accounted_for(self):
        X = np.random.rand(5, 2)

        model = AgglomerativeClustering(distance_threshold=10.0)
        model.fit(X)

        final_clusters, _ = model.cluster_history[-1]
        merged = sorted(sum(final_clusters, []))
        self.assertEqual(merged, list(range(len(X))))


if __name__ == "__main__":
    unittest.main()
