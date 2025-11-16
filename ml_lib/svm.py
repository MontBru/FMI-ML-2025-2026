import quadprog as qp
import numpy as np
import ml_lib.kernels as kernels
from sklearn import base

class SVC(base.BaseEstimator):
    def __init__(self, c=1, kernel='rbf', gamma=1):
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.eps = 1e-10

    def get_kernel_func(self):
        kernel_func = None
        if self.kernel == "linear":
            kernel_func = kernels.linear
        elif self.kernel == "polynomial":
            kernel_func = kernels.polynomial
        elif self.kernel == "rbf":
            kernel_func = kernels.rbf
        elif self.kernel == "sigmoid":
            kernel_func = kernels.sigmoid
        else:
            raise Exception("Kernel string not supported!")
        return kernel_func

    def fit(self, X, y):
        #Saves support_vectors_ and intercept_
        y = np.where(y == 0, -1, y)
        kernel_func = self.get_kernel_func()
        # kernel_func = lambda x, y, z: np.sqrt(x.T @ y) 
        K = kernel_func(X, X, self.gamma)

        G = (y.reshape(-1, 1) @ y.reshape(1, -1)) * K
    
        a = np.ones(X.shape[0])
        C = np.vstack([y.reshape(1, -1), np.identity(X.shape[0]), -np.identity(X.shape[0])]).T
        b = np.hstack((np.zeros(y.size + 1), -self.c * np.ones_like(y)))
        
        #i do this to ensure that G is positive definite
        eps = self.eps
        G += np.eye(G.shape[0]) * eps
                
        meq = 1
        qp_result = qp.solve_qp(G, a, C, b, meq, False)
        self.alphas = qp_result[0]

        support_vector_indices = np.argwhere(np.logical_and(self.alphas > eps, self.alphas < self.c - eps)).reshape(-1,)
        
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_values = y[support_vector_indices]
        self.alphas = self.alphas[support_vector_indices]
        self.intercept_ = 0

        for i in support_vector_indices:
            self.intercept_ += y[i] - np.sum(a * y * G[i])

        self.intercept_ /= support_vector_indices.size

    def predict(self, X):
        kernel_func = self.get_kernel_func()
        return np.sum(self.alphas*self.support_vector_values*kernel_func(self.support_vectors_, X, self.gamma).T, axis=1) + self.intercept_ >= 0