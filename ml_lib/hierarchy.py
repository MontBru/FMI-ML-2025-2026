import numpy as np
from sklearn import base

def linkage(A, B, method):
    #A, B are (N1, D), (N2, D) numpy matrixes
    #where N is the number of points in the set
    #D is the dimension of the points

    #distances is a (N1,N2) numpy matrix of the distances
    diff = A[:, None, :] - B[None, :, :]      # (N1, N2, D)
    distances = np.linalg.norm(diff, axis=2)  # (N1, N2)
    
    if method == 'single':
        return np.min(distances)
    elif method == 'complete':
        return np.max(distances)
    elif method == 'average':
        return np.mean(distances)
    elif method == 'centroid':
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        return np.linalg.norm(centroid_A - centroid_B)
    else:
        return
    

class AgglomerativeClustering(base.BaseEstimator):
    def __init__(self, linkage = 'complete', distance_threshold = 1):
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def merge_lists(self, C, I):
        parent = list(range(len(C)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Union step
        for i, j in I:
            union(i, j)

        # Group indices by root
        groups = {}
        for idx in range(len(C)):
            root = find(idx)
            groups.setdefault(root, []).extend(C[idx])

        return list(groups.values())

    def fit(self, X, y=None):
        X = np.array(X)
        #X is a (N, D) np matrix of N points with dimension D
        
        N = X.shape[0]

        #cluster_history is a list,
        #each element of this list is a pair (clusters, distance)
        cluster_history = []

        #clusters is a list, each element of this list
        #is a list that contains the indexes of the points inside a cluster
        #Initially we interpret every point as a cluster
        clusters = np.arange(N).reshape(N,1).tolist()
        clusters_num = len(clusters)

        time = 0

        while clusters_num > 1:
            cluster_history.append((clusters, time))

            #cluster_distances is a (clusters_num,clusters_num) np matrix of the distance between every 2 clusters
            cluster_distances = np.zeros((clusters_num,clusters_num))
            for i in range(clusters_num):
                for j in range(i+1, clusters_num):
                    try:
                        cluster_distances[i][j] = linkage(X[clusters[i]], X[clusters[j]], self.linkage)
                    except Exception as e:
                        print("Exception raised:")
                        print(f"{clusters[i]=}")
                        print(f"{X.shape=}")
                        raise e

            # applying mask because i don't want to include values for distance between a cluster and itself and don't 
            # want to take into account distance between A and B if I already took into 
            # account the distance between B and A
            mask = np.triu(np.ones_like(cluster_distances, dtype=bool), k=1)
            clusters_distance_less_than_threshold = np.argwhere(mask & (cluster_distances < self.distance_threshold))

            if len(clusters_distance_less_than_threshold) == 0:
                #No merge occured so I have to break
                #otherwise infinite loop
                break

            clusters = self.merge_lists(clusters, clusters_distance_less_than_threshold)
            clusters_num = len(clusters)
            time += 1
        
        cluster_history.append((clusters, time))
        self.cluster_history = cluster_history

    def fit_predict(self, X, y = None):
        self.fit(X)
        last_clusters = self.cluster_history[-1][0]
        result = np.zeros((X.shape[0],))

        i = 0
        for cluster in last_clusters:
            result[cluster] = i
            i += 1
        
        return result
        

