from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

def main():
    df = pd.read_csv('telecom_churn_clean.csv')

    X = df[['account_length', 'customer_service_calls']]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

    n_neighbors_list = [i+1 for i in range(15)]
    metric_list = ['manhattan', 'euclidean']

    scores = {}

    for metric in metric_list:
        scores[metric] = []
        for n_neighbours in n_neighbors_list:
            model = KNeighborsClassifier(n_neighbors=n_neighbours, metric=metric)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores[metric].append(score)

    filename  = 'model_report.xlsx'
    pd.DataFrame(scores).to_excel(filename, sheet_name='Sheet1')
    
    plt.plot(scores['euclidean'])
    plt.title("Scores for k neighbors using euclidean metric")
    plt.xlabel("k-neighbors")
    plt.ylabel("score")
    plt.savefig('./diagrams/knn_euclidean_scores.png')
    plt.cla()

    plt.plot(scores['manhattan'])
    plt.title("Scores for k neighbors using manhattan metric")
    plt.xlabel("k-neighbors")
    plt.ylabel("score")
    plt.savefig('./diagrams/knn_manhattan_scores.png')
    plt.cla()
    
    

if __name__ == '__main__':
    main()