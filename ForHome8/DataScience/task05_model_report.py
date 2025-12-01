import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import Homework1.clusterization as clusterization
import openpyxl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from task03_data import samples, species

def main():
    X = pd.read_csv('eurovision_voting.csv', index_col=0)

    X.columns = X.columns.astype(str)

    filename = './model_report_task05.xlsx'
    diagrams_folder_path = './diagrams'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Scaling/PCA/t-SNE/TruncatedSVD', 'Number of variables','Hyperparams', 'Inertia', 'Silhouette', 'Silhouette Increase from base model %', 'Silhouette for test data', 'Silhouette for test data increase %','Scatter plot train', 'Dendrogram', 'Scatter plot test', 'PCA variance', 'Cross-Tabulation'])

    clusterization.base_model(X, ws)

    model_options = [
        'agglomerative',
        'dbscan',
        'hdbscan'
    ]

    data_preprocessing = [
        {'scaling':False,
         'pca':False,
         'tsne': False,
         'truncatedsvd':False},

        {'scaling':False,
         'pca':True,
         'tsne': False,
         'truncatedsvd':False},

        {'scaling':True,
         'pca':True,
         'tsne': False,
         'truncatedsvd':False},

        {'scaling':False,
         'pca':False,
         'tsne': False,
         'truncatedsvd':True},

        {'scaling':True,
         'pca':False,
         'tsne': False,
         'truncatedsvd':True},
    ]

    n_components = np.append(np.arange(1, min(X.columns.size, 10), 1), X.columns.size)
    print(f'{n_components=}')

    for name in model_options:
        for preprocessing in data_preprocessing:
            for n_component in n_components:
                scaling = preprocessing['scaling']
                pca = preprocessing['pca']
                tsne = preprocessing['tsne']
                truncatedsvd = preprocessing['truncatedsvd']
                
                clusterization.clusterize(X=X, name=name, n_features=n_component, scaling=scaling, pca=pca, tsne=tsne, truncatedsvd=truncatedsvd, ws=ws, clusterization_distance=12)

    wb.save(filename)

if __name__ == '__main__':
    main()