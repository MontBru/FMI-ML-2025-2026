from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

global base_model_silhouette

def base_model(X, ws, diagrams_dir = './diagrams'):
    global base_model_silhouette

    labels = np.zeros(X.shape[0])

    inertia = None  # only for KMeans / MiniBatchKMeans
    base_model_silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else -1


    ws.append([
            f"Base model",
            '',
            '',
            '',
            '',
            X.shape[1],
            '' ,
            inertia if inertia is not None else "",
            base_model_silhouette if base_model_silhouette is not None else "",
            0
        ])


    if not isinstance(X, csr_matrix):

        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = np.array(X)
        # ----------- SCATTERPLOT OF CLUSTERS -----------
        plt.figure(figsize=(7, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
        plt.title(f"base model — Cluster Scatterplot")

        plt.tight_layout()

        scatter_path = f"{diagrams_dir}/base_model_scatter.png"
        plt.savefig(scatter_path)
        plt.cla()

        img = openpyxl.drawing.image.Image(scatter_path)
        row = ws.max_row
        col = 'J'
        cell_ref = f"{col}{row}"

        img_width, img_height = img.width, img.height
        ws.column_dimensions[col].width = img_width / 7
        ws.row_dimensions[row].height = img_height

        img.anchor = cell_ref
        ws.add_image(img)

        col = chr(ord(col) + 1)


def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    n_clusters = len(set(labels))
    if n_clusters < 2 or n_clusters >= len(X):
        return -1
    return silhouette_score(X, labels)


def save_crosstab_image(df, path):
    plt.figure(figsize=(8, 4))
    plt.axis('off')

    # Create table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()


def clusterize(X,
               name,
               y=None,
               n_features=2,
               scaling=True,
               pca=True,
               tsne=True,
               truncatedsvd=False,
               args=None,
               export_to_file=True,
               ws=None,
               diagrams_dir='./diagrams',
               real_labels=None,
               n_clusters = None,
               clusterization_distance=1):
    global base_model_silhouette

    # print(args)
    param_grid = None
    reg = None

    if scaling == True:
        steps = [('scaler', StandardScaler())]
    else:
        steps = []

    if pca == True:
        steps.append(('pca', PCA(n_components=n_features)))

    # if tsne == True:
    #     steps.append(('tsne', TSNE(n_components=n_features)))

    if truncatedsvd == True:
        steps.append(('truncatedsvd', TruncatedSVD(n_components=n_features)))

    if 'hdbscan' in name:
        if args == None:
            param_grid = {
                # 'hdbscan__min_cluster_size': np.arange(1, 50, 1)
                # 'hdbscan__min_cluster_size': [2, 1000, 5000, 10000, 20000]
                'hdbscan__min_cluster_size': [5000]
            }
        else:
            param_grid = args['params']
        reg = hdbscan.HDBSCAN()
        steps.append(('hdbscan', reg))
    elif 'dbscan' in name:
        if args == None:
            param_grid = {
                'dbscan__eps':
                np.arange(.1, clusterization_distance,
                          .05 * clusterization_distance),
                # 'dbscan__min_samples': np.arange(1, 50, 1)
                'dbscan__min_samples': [10]
            }
        else:
            param_grid = args['params']
        reg = DBSCAN()
        steps.append(('dbscan', reg))
    elif 'agglomerative' in name:
        clusterization_distance *= 12000
        if args == None:
            param_grid = {
                'agglomerative__linkage': ['single', 'average', 'complete'],
                # 'agglomerative__distance_threshold':
                # np.arange(.1, clusterization_distance,
                #           .05 * clusterization_distance)
                # 'agglomerative__n_clusters': np.arange(7, 20)
            }
        else:
            param_grid = args['params']
        reg = AgglomerativeClustering(distance_threshold=None,
                                      n_clusters=n_clusters,
                                      compute_full_tree=True)

        steps.append(('agglomerative', reg))

    pipeline = Pipeline(steps)

    cv = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

    cv = RandomizedSearchCV(pipeline,
                            param_distributions=param_grid,
                            scoring=silhouette_scorer,
                            cv=cv,
                            n_iter=30)

    cv.fit(X=X)

    if export_to_file:
        pipeline = cv.best_estimator_

        clusterer = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
        labels = pipeline.fit_predict(X)

        # Calculate metrics
        inertia = getattr(clusterer, "inertia_",
                          None)  # only for KMeans / MiniBatchKMeans
        silhouette = silhouette_score(X, labels) if len(
            set(labels)) > 1 and len(set(labels)) < len(X) else -1

        scatter_path_test = None
        silhouette_test = None

        if y is not None:
            labels_y = pipeline.fit_predict(y)
            silhouette_test = silhouette_score(y, labels_y) if len(
                set(labels_y)) > 1 else None

            # ----------- SCATTERPLOT OF CLUSTERS -----------
            plt.figure(figsize=(7, 5))
            plt.scatter(y.iloc[:, 0], y.iloc[:, 1], c=labels_y)
            plt.title(f"{name} — Cluster Test Scatterplot")
            plt.xlabel(y.columns[0])
            plt.ylabel(y.columns[1])
            plt.tight_layout()

            scatter_path_test = f"{diagrams_dir}/{name}_scatter_test.png"
            plt.savefig(scatter_path_test)
            plt.cla()

        if not isinstance(X, csr_matrix):
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
            else:
                X_2d = np.array(X)
            # ----------- SCATTERPLOT OF CLUSTERS -----------
            plt.figure(figsize=(7, 5))
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
            plt.title(f"{name} — Cluster Scatterplot")

            plt.tight_layout()

            scatter_path = f"{diagrams_dir}/{name}_scatter.png"
            plt.savefig(scatter_path)
            plt.cla()

        # ----------- DENDROGRAM (ONLY FOR HIERARCHICAL) -----------
        dendro_path = None
        if clusterer.__class__.__name__.startswith("Agglomerative"):
            distance_threshold = clusterer.distance_threshold

            if distance_threshold == None:

                Z = linkage(X, method=clusterer.linkage)

                n_clusters = clusterer.n_clusters
                distances = Z[:, 2]
                distance_threshold = distances[-(n_clusters - 1)]

            plt.figure(figsize=(10, 5))
            Z = linkage(X, method=clusterer.linkage)
            dendrogram(Z, labels=X.index, color_threshold=distance_threshold)
            plt.title(f"{name} — Dendrogram ({clusterer.linkage} linkage)")
            plt.tight_layout()

            dendro_path = f"{diagrams_dir}/{name}_dendrogram.png"
            plt.savefig(dendro_path)
            plt.cla()

        pca_vairance_path = None
        if pca == True:
            pca = pipeline.named_steps["pca"]

            variances = pca.explained_variance_

            plt.figure(figsize=(8, 5))
            plt.bar(np.arange(1, len(variances) + 1), variances)
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance")
            plt.title("PCA: Explained Variance per Component")
            plt.tight_layout()

            pca_vairance_path = f"{diagrams_dir}/{name}_pca_variance.png"
            plt.savefig(pca_vairance_path)
            plt.cla()

        crosstab_path = None
        if real_labels is not None:
            crosstab_path = f"{diagrams_dir}/{name}_crosstab_output.png"
            cross_tab = pd.crosstab(real_labels,
                                    labels,
                                    rownames=["Actual"],
                                    colnames=["Predicted"])
            save_crosstab_image(cross_tab, crosstab_path)

        # ----------- EXPORT TO EXCEL -----------
        col = 'M'
        for img_path in [
                scatter_path, dendro_path, scatter_path_test,
                pca_vairance_path, crosstab_path
        ]:
            if not img_path:
                col = chr(ord(col) + 1)
                continue

            img = openpyxl.drawing.image.Image(img_path)
            row = ws.max_row + 1
            cell_ref = f"{col}{row}"

            img_width, img_height = img.width, img.height
            ws.column_dimensions[col].width = img_width / 7
            ws.row_dimensions[row].height = img_height

            img.anchor = cell_ref
            ws.add_image(img)

            col = chr(ord(col) + 1)

        # ----------- APPEND SUMMARY ROW -----------
        ws.append([
            f"{name} Clustering",
            str(scaling),
            str(pca),
            str(tsne),
            str(truncatedsvd),
            X.shape[1],
            str(cv.best_params_),
            inertia if inertia is not None else "",
            silhouette if silhouette is not None else "",
            silhouette / base_model_silhouette * 100 -
            100 if silhouette is not None else 0,
            silhouette_test if silhouette_test is not None else "",
            silhouette_test / base_model_silhouette * 100 -
            100 if silhouette_test is not None else 0,
        ])

        return pipeline, silhouette

    return pipeline, 0
