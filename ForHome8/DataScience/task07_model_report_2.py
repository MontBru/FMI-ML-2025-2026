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

from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

def main():
    X = pd.read_csv('wikipedia-vectors.csv', index_col=0).T
    X.columns = X.columns.astype(str)

    filename = './model_report_task07_2.xlsx'
    diagrams_folder_path = './diagrams'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    model = NMF(n_components=10)
    W = model.fit_transform(X)
    H = model.components_

    pos_anne_hathaway = X.index.get_loc("Anne Hathaway")
    pos_denzel_washington = X.index.get_loc("Denzel Washington")

    nmf_anne_hathaway = W[pos_anne_hathaway]
    nmf_denzel_washington = W[pos_denzel_washington]

    highest_component_anne_hathaway = nmf_anne_hathaway.argmax()
    highest_component_denzel_washington = nmf_denzel_washington.argmax()

    idx = np.argsort(H[highest_component_anne_hathaway])[-5:]
    top5 = H[highest_component_anne_hathaway][idx] 

    idx_d = np.argsort(H[highest_component_denzel_washington])[-5:]
    top5_d = H[highest_component_denzel_washington][idx]

    with open("wikipedia-vocabulary-utf8.txt", "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()

    for i in idx:
        ws.append([vocab[i]])

    wb.save(filename)

if __name__ == '__main__':
    main()