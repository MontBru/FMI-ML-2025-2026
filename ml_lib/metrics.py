from fractions import Fraction
import numpy as np

def accuracy_score(y_true, y_pred, normalize = True):
    '''
    y_true: the ground truth labels;
    y_pred: the predicted labels;
    normalize: (optional) whether to return the fraction of correctly classified samples. If False, returns the number instead of the fraction. Default value: True
    '''

    if normalize == True:
        return Fraction((y_true == y_pred).sum(), y_pred.size)
    else:
        return (y_true == y_pred).sum()/y_pred.size
    
def euclidean_distance(p1, p2):
    coord_dists = p1 - p2
    coord_dists *= coord_dists
    return np.sqrt(np.sum(coord_dists))


def manhattan_distance(p1, p2):
    coord_dists = np.abs(p1 - p2)
    return np.sum(coord_dists)

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = y_true.mean()

    return 1 - np.sum((y_true-y_pred) ** 2) / (np.sum((y_true - y_true_mean) **2) + 1e-8)

def root_mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise Exception()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def recall_score(y_true, y_pred):
    #We assume that the negative is stored as 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    false_negatives = np.sum(np.logical_and(y_true != y_pred, y_true == 0))
    true_positives = np.sum(np.logical_and(y_pred == y_true, y_true == 1))
    return true_positives/(false_negatives + true_positives)

def precision_score(y_true, y_pred):
    #We assume that the negative is stored as 0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    false_positives = np.sum(np.logical_and(y_true != y_pred, y_true == 1))
    true_positives = np.sum(np.logical_and(y_true == y_pred, y_true == 1))
    return true_positives/(false_positives + true_positives)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return (2*precision*recall)/(precision + recall)

def log_loss(y_true, y_pred):
    #y_true is a vector containing all the true categories
    #y_pred is a matrix that in each row has the predicted probability for that category to be the true one
    
    y_true = np.array(y_true, int)
    y_pred = np.array(y_pred)
    if len(y_pred.shape) == 1:
        y_pred = np.vstack([1 - y_pred, y_pred])
        y_pred = y_pred.T

    y_pred_probability_for_true_categories = y_pred[np.arange(y_pred.shape[0]), y_true]
    return - np.mean(np.log(y_pred_probability_for_true_categories))