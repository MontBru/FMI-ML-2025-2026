from sklearn import base
import numpy as np

class KMeans(base.BaseEstimator):
    def __init__(self, n_clusters=1, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y = None):
        # X is a (N,D) np matrix
        # centroids is a (n_clusters,) np vector
        centroids = X[np.random.random_integers(0,X.shape[0]-1, self.n_clusters)]
        
        
        for _ in range(self.max_iter):
            #distances is a (N, n_clusters) matrix which contains
            #the distance from each point to each centroid
            diff = X[:, None, :] - centroids[None, :, :]   # (N, K, D)
            distances = np.linalg.norm(diff, axis=2)       # (N, K)

            point_to_cluster = np.argmin(distances, axis = 1)

            inertia = 0

            new_centroids = np.zeros_like(centroids)
            for i in range(self.n_clusters):
                points_inside_cluster_i = np.where(point_to_cluster == i)[0]
                new_centroids[i] = np.mean(X[points_inside_cluster_i], axis=0)
                inertia += np.sum(distances[points_inside_cluster_i, i] ** 2)

            if np.allclose(new_centroids, centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        self.inertia_ = inertia

    def predict(self, X):
        diff = X[:, None, :] - self.centroids[None, :, :]   # (N, K, D)
        distances = np.linalg.norm(diff, axis=2)       # (N, K)

        point_to_cluster = np.argmin(distances, axis = 1)
        return point_to_cluster


    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.predict(X)