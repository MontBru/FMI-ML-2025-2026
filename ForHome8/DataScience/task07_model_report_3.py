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

    filename = './model_report_task07_3.xlsx'
    diagrams_folder_path = './diagrams'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    model = NMF(n_components=10)
    W = model.fit_transform(X)
    H = model.components_

    pos = X.index.get_loc("Cristiano Ronaldo")
    nmf = W[pos]

    #Now i have to find the closest vectors to nmf

    similarities = W.dot(nmf)

    idx = similarities.argsort()[-10:]

    ws.append(list(X.index[idx]))
    ws.append(list(similarities[idx]))

    wb.save(filename)

if __name__ == '__main__':
    main()