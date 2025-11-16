import math
import numpy as np

def sigmoid(x):
    return 1/(1 + math.e**(-x))

def softmax(x):
    e_pow_x = math.e ** x
    row_sum = np.sum(e_pow_x, axis=1)
    return (e_pow_x.T / row_sum).T

def gini_index(probabilities):
    return float(1 - np.sum(probabilities**2))


def entropy(probabilities):
    eps = 1e-10
    probabilities = np.where(probabilities == 0, eps, probabilities)
    return float(-np.sum( probabilities * np.log2(probabilities)))