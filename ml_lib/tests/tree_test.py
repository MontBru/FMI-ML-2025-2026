import unittest
import numpy as np
from ml_lib.tree import DecisionTreeClassifier, Node

class TestMakeSplit(unittest.TestCase):
    def test_when_simple_dataset_and_valid_params_then_split_successful(self):
        # Arrange
        X = np.array([
            [1.0, 0],
            [2.0, 1],
            [3.0, 1],
            [4.0, 0]
        ])
        y = np.array([0, 0, 1, 1])
        node = Node(X, y)
        expected = True

        # Act
        actual = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="gini_index")

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(node.feature, 0)
        self.assertIsNotNone(node.true_node)
        self.assertIsNotNone(node.false_node)

    def test_when_too_few_samples_then_split_fails(self):
        # Arrange
        X = np.array([[1.0]])
        y = np.array([0])
        node = Node(X, y)
        expected = False

        # Act
        actual = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="entropy")

        # Assert
        self.assertEqual(actual, expected)
        self.assertTrue(node.is_leaf())

    def test_when_split_then_children_have_correct_samples(self):
        # Arrange
        X = np.array([
            [1.0],
            [2.0],
            [10.0],
            [12.0],
        ])
        y = np.array([0, 0, 1, 1])
        node = Node(X, y)
        expected = len(X)

        # Act
        node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="gini_index")
        left = node.true_node
        right = node.false_node
        actual = len(left.data_entries) + len(right.data_entries)

        # Assert
        self.assertTrue(np.all(left.data_entries.flatten() <= node.split_point))
        self.assertTrue(np.all(right.data_entries.flatten() > node.split_point))
        self.assertEqual(actual, expected)

    def test_when_split_then_split_point_and_feature_are_set(self):
        # Arrange
        X = np.array([
            [0],
            [1],
            [5],
            [6],
        ])
        y = np.array([0, 0, 1, 1])
        node = Node(X, y)
        expected = True

        # Act
        actual = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="entropy")

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(node.feature, 0)
        self.assertIsNotNone(node.split_point)

class TestFit(unittest.TestCase):
    def test_when_linearly_separable_then_predicts_correctly(self):
        # Arrange
        X = np.array([[1], [2], [3], [4], [10], [11], [12]])
        y = np.array([0, 0, 0, 0, 1, 1, 1])
        clf = DecisionTreeClassifier(criterion="gini_index")
        expected = np.array([0, 0, 1, 1])

        # Act
        clf.fit(X, y)
        actual = clf.predict(np.array([[0], [4], [7], [20]]))

        # Assert
        np.testing.assert_array_equal(actual, expected)

    def test_when_min_samples_split_too_large_then_no_split_and_predicts_majority(self):
        # Arrange
        X = np.array([[1], [10]])
        y = np.array([0, 1])
        clf = DecisionTreeClassifier(min_samples_split=10)
        expected = np.full((2,), np.argmax(np.bincount(y)))

        # Act
        clf.fit(X, y)
        root = clf.root
        actual = clf.predict(np.array([[0], [50]]))

        # Assert
        self.assertTrue(root.is_leaf())
        np.testing.assert_array_equal(actual, expected)

    def test_when_split_possible_then_chooses_correct_feature(self):
        # Arrange
        X = np.array([
            [100, 1],
            [120, 2],
            [120, 9],
            [310, 10],
        ])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(criterion="entropy")
        expected = 1

        # Act
        clf.fit(X, y)
        actual = clf.root.feature

        # Assert
        self.assertEqual(actual, expected)

    def test_when_fit_on_multidim_then_predict_matches_labels(self):
        # Arrange
        X = np.array([
            [1, 5],
            [2, 4],
            [10, 1],
            [11, 2]
        ])
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier()
        expected = y

        # Act
        clf.fit(X, y)
        actual = clf.predict(X)

        # Assert
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
