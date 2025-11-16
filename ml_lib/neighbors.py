from ml_lib import metrics
import numpy as np

class KNeighborsClassifier:

    def __init__(self, n_neighbors = 1, metric = 'euclidean'):
        if metric != 'euclidean' and metric != 'manhattan':
            raise Exception("Metric is not correct!")
        self.metric = metric    
        self.n_neighbors = n_neighbors    

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise Exception("Shapes aren't compatible between x and y")
        self.x = x
        self.y = y

    def calculate_distance(self, p1, p2):
        if self.metric == 'euclidean':
            return metrics.euclidean_distance(p1, p2)
        else:
            return metrics.manhattan_distance(p1, p2)


    #Normally it would be much faster to use the vectorized functions in numpy instead of doing it with loops
    #But I am doing it this way so that i can use the metrics module from the previous tasks
    def predict(self, x):
        result = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            distances_for_current = np.zeros(self.x.shape[0])
            for j in range(self.x.shape[0]):
                try:
                    distances_for_current[j] = self.calculate_distance(x[i], self.x[j])
                except:
                    print(i)
                    print(x.shape)
                    print(x.loc[0])
                    print(j)
                    print(self.x.shape)
                    print(type(self.x))
                    print(self.x.iloc[0])
                    exit()
            neighbours = np.argsort(distances_for_current)[:self.n_neighbors]
            #in the task description it wasn't specified how to resolve if two categories have the same amount
            #of cases so i am just selecting one of them
            neighbours_labels = self.y[neighbours]
            labels, label_counts = np.unique(neighbours_labels, return_counts=True)
            most_common_label = labels[np.argmax(label_counts)]
            result[i] = most_common_label
        return result
    
    def score(self, x, y):
        y_pred = self.predict(x)
        return metrics.accuracy_score(y, y_pred)

