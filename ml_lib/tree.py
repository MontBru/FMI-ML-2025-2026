from sklearn import base
import numpy as np
from ml_lib.stats import entropy, gini_index

class Node:
    def __init__(self, data_entries, data_entries_classes, depth = 0, feature = None, split_point = None, true_node=None, false_node=None):
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

        if self.data_entries.ndim == 1:
            self.data_entries = self.data_entries.reshape(-1, 1)

        
        for temp_category in range(self.data_entries.shape[1]):
            min_element_for_category = np.min(self.data_entries[:, temp_category])
            max_element_for_category = np.max(self.data_entries[:, temp_category])
            step = (max_element_for_category- min_element_for_category)/1000

            try:
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
            except:
                print(min_element_for_category)
                print(max_element_for_category)
                print(step)
        if best_left_data_entries is None:
            return False
        
        self.true_node = Node(self.data_entries[best_left_data_entries], self.data_entries_classes[best_left_data_entries], self.depth + 1)
        self.false_node = Node(self.data_entries[best_right_data_entries], self.data_entries_classes[best_right_data_entries], self.depth + 1)
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


class DecisionTreeClassifier(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self, min_samples_leaf = 1, min_samples_split = 2, max_depth = 15, criterion='entropy'):
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        y = np.array(y).reshape(-1,)
        self.root = Node(X, y)
        self.root.fit(self.min_samples_leaf, self.min_samples_split, self.max_depth, self.criterion)

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in range(y.size):
            y[i] = self.root.predict(X[i])

        return y



class RandomForestClassifier(base.BaseEstimator):
    def __init__(self, n_estimators, criterion = 'gini_index', max_depth = 15, min_samples_split = 2, min_samples_leaf = 1, max_features = 1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.feature_indices_for_estimators = []
        self.estimators = []
        
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice( X.shape[0], X.shape[0], replace=True)
            bootstrap_sample = X[indices]

            feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
            train_set = bootstrap_sample[:, feature_indices]

            self.feature_indices_for_estimators.append(feature_indices)

            estimator = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split, max_depth=self.max_depth, criterion=self.criterion)
            estimator.fit(train_set, y[indices])

            self.estimators.append(estimator)
        

    def predict(self, X):
        pred_y = []
        for i in range(self.n_estimators):
            y_pred_i = self.estimators[i].predict(
                X[:, self.feature_indices_for_estimators[i]]
            )
            pred_y.append(y_pred_i.reshape(-1))

        pred_y = np.array(pred_y).astype(int)

        return np.array([np.bincount(col).argmax() for col in pred_y.T])

class AdaBoostClassifier(base.BaseEstimator):
    def __init__(self, n_estimators = 50, learning_rate = 1, criterion = 'gini_index'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.estimators = []
        self.amounts_of_say = []

    def fit(self, X, y):

        self.n_classes = len(np.unique(y))
        samples = X
        sample_categories = y
        weights = np.full(X.shape[0], 1/X.shape[0])

        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1, criterion=self.criterion)
            estimator.fit(samples, sample_categories)
            y_pred = estimator.predict(X)

            indices_of_wrong_pred = np.argwhere(y_pred != y).reshape(-1,)
            indices_of_correct_pred = np.argwhere(y_pred == y).reshape(-1,)
            
            total_estimator_error = np.sum(weights[indices_of_wrong_pred])
            estimator_amount_of_say = .5 * np.log((1 - total_estimator_error)/total_estimator_error) * self.learning_rate

            weights[indices_of_wrong_pred] *= np.exp(estimator_amount_of_say)
            weights[indices_of_correct_pred] *= np.exp(-estimator_amount_of_say)


            weights = weights/(np.sum(weights) + 1e-10)

            self.estimators.append(estimator)
            self.amounts_of_say.append(estimator_amount_of_say)

            if indices_of_wrong_pred.size == 0:
                self.n_estimators = _ + 1
                break

            sample_indices = np.random.choice(X.shape[0], X.shape[0], replace=True, p=weights)
            samples = X[sample_indices]
            sample_categories = y[sample_indices]

    def predict(self, X):
        weighted_votes = np.zeros((X.shape[0], self.n_classes))

        for estimator, alpha in zip(self.estimators, self.amounts_of_say):
            preds = estimator.predict(X).astype(int)
            for i in range(X.shape[0]):
                weighted_votes[i, preds[i]] += alpha

        print(weighted_votes)
        y_pred = np.argmax(weighted_votes, axis=1)
        print(y_pred)

        return y_pred
