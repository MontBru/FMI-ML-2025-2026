from ml_lib.tree import DecisionTreeClassifier
import numpy as np

def test_make_split_success():
    """
    A simple dataset where the best split is at feature 0 ≤ 2.5
    """
    X = np.array([
        [1.0, 0],
        [2.0, 1],
        [3.0, 1],
        [4.0, 0]
    ])
    y = np.array([0, 0, 1, 1])

    node = DecisionTreeClassifier.Node(X, y)
    result = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="gini_index")

    assert result is True
    assert node.feature == 0
    assert node.true_node is not None
    assert node.false_node is not None


def test_make_split_min_samples_split_fail():
    """
    Node should not split when too few samples.
    """
    X = np.array([[1.0]])
    y = np.array([0])

    node = DecisionTreeClassifier.Node(X, y)
    result = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="entropy")

    assert result is False
    assert node.is_leaf()


def test_children_have_correct_samples():
    """
    Check that the split correctly partitions the dataset.
    """
    X = np.array([
        [1.0],
        [2.0],
        [10.0],
        [12.0],
    ])
    y = np.array([0, 0, 1, 1])

    node = DecisionTreeClassifier.Node(X, y)
    node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="gini_index")

    left = node.true_node
    right = node.false_node

    # Most likely split: <= 6 on feature 0
    assert np.all(left.data_entries.flatten() <= node.split_point)
    assert np.all(right.data_entries.flatten() > node.split_point)

    assert len(left.data_entries) + len(right.data_entries) == len(X)


def test_split_point_and_feature_are_set():
    """
    Ensure the node stores split_point and feature after splitting.
    """
    X = np.array([
        [0],
        [1],
        [5],
        [6],
    ])
    y = np.array([0, 0, 1, 1])

    node = DecisionTreeClassifier.Node(X, y)
    success = node.make_split(min_samples_leaf=1, min_samples_split=2, max_depth=15, criterion="entropy")

    assert success
    assert node.feature == 0
    assert node.split_point is not None

def test_basic_fit_and_predict():
    """
    Simple linearly separable dataset:
        class 0: x < 5
        class 1: x >= 5
    """
    X = np.array([[1], [2], [3], [4], [10], [11], [12]])
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf = DecisionTreeClassifier(criterion="gini_index")
    clf.fit(X, y)

    preds = clf.predict(np.array([[0], [4], [7], [20]]))
    assert np.array_equal(preds, np.array([0, 0, 1, 1]))


def test_no_split_due_to_min_samples_split():
    """
    When min_samples_split > len(dataset), the tree should NOT split.
    Prediction must return majority class of root.
    """
    X = np.array([[1], [10]])
    y = np.array([0, 1])

    clf = DecisionTreeClassifier(min_samples_split=10)  # too large
    clf.fit(X, y)

    # majority class = 0 or 1 (tie — depends on implementation)
    # So check leaf status and predict from root only
    root = clf.root
    assert root.is_leaf()

    preds = clf.predict(np.array([[0], [50]]))

    # Both predictions must match the root's majority class
    root_majority = np.argmax(np.bincount(y))
    assert np.all(preds == root_majority)


def test_split_chooses_correct_feature():
    """
    Dataset designed so that only feature 1 gives a good split.
    """
    X = np.array([
        [100, 1],
        [120, 2],
        [120, 9],
        [310, 10],
    ])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)

    assert clf.root.feature == 1  # correct feature chosen


def test_predict_matches_labels_after_fit():
    """
    Train on multi-dimensional dataset and ensure predictions match truth.
    """
    X = np.array([
        [1, 5],
        [2, 4],
        [10, 1],
        [11, 2]
    ])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    preds = clf.predict(X)
    assert np.array_equal(preds, y)



def main():
    test_make_split_success()
    test_make_split_min_samples_split_fail()
    test_children_have_correct_samples()
    test_split_point_and_feature_are_set()

    print("All tests for node passed successfully!")

    test_basic_fit_and_predict()
    test_no_split_due_to_min_samples_split()
    test_split_chooses_correct_feature()
    test_predict_matches_labels_after_fit()

    print("All tests passed correctly for tree")

if __name__ == '__main__':
    main()