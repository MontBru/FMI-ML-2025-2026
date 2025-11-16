import numpy as np
from ml_lib import metrics, stats
from sklearn import base

class LinearRegression(base.BaseEstimator):

    def __init__(self):
        self.betas = 0

    def fit(self, X, y):
        try:
            #X is shape (N, 1)
            #y is shape (N,)
            #X.T is shape (1, N)
            X = np.column_stack((np.ones(X.shape[0]), X))
            self.betas = np.linalg.inv(X.T @ X) @ X.T @ y
        except:
            raise ValueError("Matrix is not invertible. Please remove collinear features.")
        

    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return np.reshape(self.betas.T @ X.T, (-1,))
    
    def score(self, X, y):
        return metrics.r2_score(y, self.predict(X))
    


class LogisticRegression(base.BaseEstimator) :

    class BinaryClassification:
        def __init__(self, c, max_iter, lr,pd):
            self.betas = None
            self.pd = pd
            self.c = c
            self.lr = lr
            self.max_iter = max_iter

        def fit(self, X, y, random_state, verbose):
            #c is regularization strength
            #X is (N, D) matrix
            # where N - number of samples
            # D - dimension of sample (feature count)
            #y is (N,) vector
            feature_count = X.shape[1]

            rng = np.random.default_rng(random_state['seed'])
            self.betas = rng.normal(random_state['mean'], random_state['deviation'],(self.pd, feature_count))
            
            for _ in range(self.max_iter):
                y_pred = self.predict_proba(X)
                for deg in range(self.pd):
                    dL = (y_pred - y).T @ X**(deg+1) - self.c*2*self.betas[deg]
                    self.betas[deg] = self.betas[deg] - dL * self.lr
                reg =  self.c * np.sum(self.betas ** 2)
                loss = metrics.log_loss(y_pred=y_pred, y_true=y) + reg
                if verbose == True:
                    print(f'Iteration({_ + 1}/{self.max_iter})  Loss: {loss} Reg: {reg}')

        def predict_proba(self, X):
            if self.betas is None:
                raise Exception("Trying to predict without fitting!")
            X = np.array(X)
            h_beta = 0
            for deg in range(self.pd):
                h_beta += self.betas[deg].T @ X.T ** (deg + 1)
            return stats.sigmoid(h_beta)
        
        def predict(self, X):
            probabilities = self.predict_proba(X)
            return np.argmax(probabilities)
    
        def score(self, X, y):
            y_pred = self.predict(X)
            return metrics.accuracy_score(y, y_pred)

    def __init__(self, c = 1, max_iter = 50,lr = 1e-5, pd = 5 ):
        self.class_count = 0
        self.models = []
        self.c = c
        self.lr = lr
        self.max_iter = max_iter
        self.pd = pd
    
    def fit(self, X, y, random_state = {'seed': 123, 'mean':0, 'deviation':1e-3}, verbose = False):
        #c is regularization strength
        #X is (N, D) matrix
        # where N - number of samples
        # D - dimension of sample (feature count)
        #y is (N,) vector
        self.class_count = np.max(y) + 1
        for i in range(self.class_count):
            self.models.append(self.BinaryClassification(c=self.c, lr=self.lr, max_iter=self.max_iter, pd=self.pd))
            if verbose == True:
                print(f'Training model for class {i}')
            self.models[i].fit(X, y == i, random_state, verbose)


    def predict_proba(self, X):
        model_predictions = []
        for i in range(self.class_count):
            model_predictions.append(self.models[i].predict_proba(X))
        
        np_model_predictions = np.array(model_predictions)

        result = stats.softmax(np_model_predictions.T)
        return result


    def predict(self, X):
        probabilities = self.predict_proba(X)
        result = np.argmax(probabilities, axis = 1)

        return result
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.accuracy_score(y, y_pred)

class Ridge(base.BaseEstimator):
    def __init__(self, alpha = 1):
        self.coef = 0
        self.intercept = 0
        self.alpha = alpha

    def fit(self, X, y):
        try:
            #X is shape (N, 1)
            #y is shape (N,)
            #X.T is shape (1, N)

            D = np.identity(X.shape[1])
            D[0,0] = 0
            X = np.column_stack((np.ones(X.shape[0]), X))
            
            parameters = np.linalg.inv(X.T @ X + self.alpha * D) @ X.T @ y
        except:
            raise ValueError("Matrix is not invertible. Please remove collinear features.")
        
        self.intercept = parameters[0]
        self.coef = parameters[1]
        

    def predict(self, X):
        return np.reshape(self.coef * X + self.intercept, (-1,))
    
    def score(self, X, y):
        return metrics.r2_score(y, self.predict(X))
    
class Lasso(base.BaseEstimator):
    def __init__(self, alpha = 0):
        self.betas = 0
        self.alpha = alpha

    def fit(self, X, y, lr = 1e-1, max_iter = 50, random_state = {'seed': 123, 'mean':0, 'deviation':1e-3}, verbose = False):
        #X is shape (N, 1)
        #y is shape (N,)
        #X.T is shape (1, N)

        X = np.column_stack((np.ones(X.shape[0]), X))
        feature_count = X.shape[1]
        rng = np.random.default_rng(random_state['seed'])
        self.betas = rng.normal(random_state['mean'], random_state['deviation'],(feature_count))

        for _ in range(max_iter):
            y_pred = self.predict(X[:, 1:])
            loss = np.mean((y - y_pred) ** 2)
            if verbose == True:
                print(f'Iteration ({_ + 1}/{max_iter})    Loss: {loss}')

            for j in range(self.betas.shape[0]):
                if j == 0:
                    reg = 0
                else:
                    reg = self.alpha* (1 if self.betas[j] > 0 else -1)
                self.betas[j] -= lr * (np.mean((y_pred - y) * X[:, j]) + reg)

    def predict(self, X):
        X = np.column_stack((np.ones(X.shape[0]), X))
        return np.reshape(self.betas.T @ X.T, (-1,))
    
    def score(self, X, y):
        return metrics.r2_score(y, self.predict(X))
            