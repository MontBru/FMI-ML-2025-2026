from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('telecom_churn_clean.csv')

    X = df[['account_length', 'customer_service_calls']]
    y = df['churn']
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(X, y)

    X_new = np.array([[30.0, 17.5],
                    [107.0, 24.1],
                    [213.0, 10.9]])

    print(model.predict(X_new))

if __name__ == '__main__':
    main()