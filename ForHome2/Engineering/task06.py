import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from ml_lib import model_selection, neighbors
import os

def main():
    current_dir = os.path.dirname(__file__)
    df = pd.read_csv(f'{current_dir}/telecom_churn_clean.csv')

    X = df[['account_length', 'customer_service_calls']]
    y = df['churn']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

    n_neighbors_list = [i+1 for i in range(15)]
    metric_list = ['manhattan', 'euclidean']

    scores = {}

    for metric in metric_list:
        scores[metric] = []
        for n_neighbours in n_neighbors_list:
            model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbours, metric=metric)
            model.fit(np.array(X_train), np.array(y_train))
            score = model.score(np.array(X_test), np.array(y_test))
            scores[metric].append(score)

    filename  = f'{current_dir}/model_report_made_with_ml_lib.xlsx'
    pd.DataFrame(scores).to_excel(filename, sheet_name='Sheet1')
    
    plt.plot(scores['euclidean'])
    plt.title("Scores for k neighbors using euclidean metric")
    plt.xlabel("k-neighbors")
    plt.ylabel("score")
    plt.savefig(f'{current_dir}/diagrams/knn_euclidean_scores.png')
    plt.cla()

    plt.plot(scores['manhattan'])
    plt.title("Scores for k neighbors using manhattan metric")
    plt.xlabel("k-neighbors")
    plt.ylabel("score")
    plt.savefig(f'{current_dir}/diagrams/knn_manhattan_scores.png')
    plt.cla()
    
    

if __name__ == '__main__':
    main()