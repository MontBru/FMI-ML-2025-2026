from sklearn import base
import numpy as np
from ml_lib.stats import entropy, gini_index

class DecisionTreeClassifier(base.BaseEstimator, base.ClassifierMixin):

    class Node:
        def __init__(self, data_entries, data_entries_classes, depth = 1, feature = None, split_point = None, true_node=None, false_node=None):
            self.feature = feature
            self.split_point = split_point
            self.true_node = true_node
            self.false_node = false_node
            self.data_entries = data_entries
            self.data_entries_classes = data_entries_classes
            self.depth = depth

        def is_leaf(self):
            return self.true_node is None and self.false_node is None
        
        def get_function_for_criterion(self, criterion):
            if criterion == "entropy":
                return entropy
            if criterion == "gini_index":
                return gini_index
            return None
        
        def make_split(self, min_samples_leaf, min_samples_split, max_depth, criterion):
            if self.data_entries.shape[0] < min_samples_split:
                return False
            
            if self.depth >= max_depth:
                return False
            
            if np.unique(self.data_entries_classes).size == 1:
                return False
            
            criterion_func = self.get_function_for_criterion(criterion)

            class_counts = np.bincount(self.data_entries_classes)
            probabilities = class_counts / self.data_entries_classes.shape[0]
            impurity_parent = criterion_func(probabilities)

            best_information_gain = 0
            best_left_data_entries = None
            best_right_data_entries = None
            
            for temp_category in range(self.data_entries.shape[1]):
                min_element_for_category = np.min(self.data_entries[:, temp_category])
                max_element_for_category = np.max(self.data_entries[:, temp_category])
                step = (max_element_for_category- min_element_for_category)/1000
                for temp_split_point in np.arange(min_element_for_category, max_element_for_category, step):
                    left_entries = np.argwhere(self.data_entries[:, temp_category] <= temp_split_point).reshape(-1,)
                    right_entries = np.argwhere(self.data_entries[:, temp_category] > temp_split_point).reshape(-1,)

                    if left_entries.shape[0] < min_samples_leaf or right_entries.shape[0] < min_samples_leaf:
                        continue

                    left_class_counts = np.bincount(self.data_entries_classes[left_entries], minlength=class_counts.size)
                    probabilities_left = left_class_counts / left_entries.size
                    impurity_left = criterion_func(probabilities_left)

                    right_class_counts = np.bincount(self.data_entries_classes[right_entries], minlength=class_counts.size)
                    probabilities_right = right_class_counts / right_entries.size
                    impurity_right = criterion_func(probabilities_right)

                    information_gain = impurity_parent - (left_entries.shape[0]/self.data_entries_classes.shape[0]) * impurity_left - (right_entries.shape[0]/self.data_entries_classes.shape[0])*impurity_right

                    if information_gain > best_information_gain:
                        best_information_gain = information_gain
                        self.split_point = temp_split_point
                        self.feature = temp_category
                        best_left_data_entries = left_entries
                        best_right_data_entries = right_entries

            if best_left_data_entries is None:
                return False
            
            self.true_node = DecisionTreeClassifier.Node(self.data_entries[best_left_data_entries], self.data_entries_classes[best_left_data_entries], self.depth + 1)
            self.false_node = DecisionTreeClassifier.Node(self.data_entries[best_right_data_entries], self.data_entries_classes[best_right_data_entries], self.depth + 1)
            return True
        
        def fit(self, min_samples_leaf, min_samples_split, max_depth, criterion):
            split_success = self.make_split(min_samples_leaf, min_samples_split, max_depth, criterion)
            if split_success == False:
                return
            
            self.true_node.fit(min_samples_leaf, min_samples_split, max_depth, criterion)
            self.false_node.fit(min_samples_leaf, min_samples_split, max_depth, criterion)

        def predict(self, X_entry):
            #X_entry is 1D np array
            if self.is_leaf():
                class_counts = np.bincount(self.data_entries_classes)
                return np.argmax(class_counts)
            
            if X_entry[self.feature] <= self.split_point:
                return self.true_node.predict(X_entry)
        
            return self.false_node.predict(X_entry)

    def __init__(self, min_samples_leaf = 1, min_samples_split = 2, max_depth = 15, criterion='entropy'):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        y = np.array(y).reshape(-1,)
        self.root = DecisionTreeClassifier.Node(X, y)
        self.root.fit(self.min_samples_leaf, self.min_samples_split, self.max_depth, self.criterion)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(y.size):
            y[i] = self.root.predict(X[i])

        return y
