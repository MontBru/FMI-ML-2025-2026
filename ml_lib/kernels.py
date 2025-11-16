import numpy as np

def linear(x1, x2, gamma=0):
    #gamma is never used but is included so that all kernels have it
    return x1 @ x2.T

def polynomial(x1, x2, gamma, d=1, r=0):
    return (gamma * linear(x1, x2) + r) ** d

def rbf(x1, x2, gamma):
    temp = np.abs(x1[:, None, :] - x2[None, :, :])
    return np.exp(-gamma * np.sum(temp * temp, axis = 2))

def sigmoid(x1, x2, gamma, r=0):
    return np.tanh(gamma * linear(x1, x2) + r)