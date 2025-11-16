from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import openpyxl
import numpy as np
from sklearn.metrics import r2_score


model_number = 1
base_model_score = None

def base_model( y, ws):
    global model_number
    global base_model_score

    y_train, y_test = train_test_split( y, test_size=0.3, random_state=21)
    pred = np.mean(y_train)

    y_pred = np.ones_like(y_test) * pred
    base_model_score = r2_score(y_test, y_pred)

    ws.append([f'Base model', '', '', base_model_score, 0])
    model_number += 1
    

def train_and_test_model(X, y, ws, name):

    global model_number
    global base_model_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)


    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    param_grid = None
    reg = None

    if name == "linear":
        param_grid = {}
        reg = LinearRegression()
    elif name == "ridge":
        param_grid = {
            'alpha': np.arange(.1, 1, .1),
            'solver': ['sag', 'lsqr']
        }
        reg = Ridge()
    elif name == "lasso":
        param_grid = {
            'alpha': np.arange(.1, 1, .1)
        }
        reg = Lasso()
    else:
        raise Exception("Invalid model name!")

    cv = GridSearchCV(reg, param_grid=param_grid,cv=kf, return_train_score=True, scoring='r2')
    cv.fit(X=X_train, y=y_train)

    results = pd.DataFrame(cv.cv_results_)

    results['params_label'] = results['params'].apply(
        lambda p: ', '.join(f'{k}={v}' for k, v in p.items())
    )

    results = results.sort_values(by='mean_test_score', ascending=True)

    y_labels = results['params_label']
    train_scores = results['mean_train_score']
    test_scores = results['mean_test_score']

    plt.figure(figsize=(10, 5))
    bar_width = 0.4
    y_positions = range(len(y_labels))

    plt.barh(
        [y - bar_width/2 for y in y_positions],
        train_scores,
        height=bar_width,
        label='Train score',
    )
    plt.barh(
        [y + bar_width/2 for y in y_positions],
        test_scores,
        height=bar_width,
        label='Test score',
    )

    plt.yticks(y_positions, y_labels)
    plt.xlabel("Score")
    plt.title("Train vs Test Score for Each Hyperparameter Combination")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./diagrams/{name}_regression_diagram_{X_test.columns[0]}.png')
    plt.cla()

    img = openpyxl.drawing.image.Image(f'diagrams/{name}_regression_diagram_{X_test.columns[0]}.png')
    

    cell_ref = f'F{1 + model_number}'

    img_width, img_height = img.width, img.height

    column_letter = 'F'
    col_width = img_width / 7
    row_height = img_height / 1.333


    ws.column_dimensions[column_letter].width = col_width
    ws.row_dimensions[1 + model_number].height = row_height

    img.anchor = cell_ref
    ws.add_image(img, cell_ref)

    ws.append([f'{name} Regression on {X.columns}', X.shape[1], str(cv.best_params_), cv.best_score_])
    model_number += 1


    return cv.best_score_, cv.best_params_


def main():
    df = pd.read_json('music_dirty.txt')
    df_dummies = pd.get_dummies(df['genre'], drop_first=True, dtype=int)
    filename = './model_report.xlsx'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Number of variables','Hyperparams', 'R2 score'])
    y = df['popularity']

    
    base_model(y, ws)


    X_options = [
        pd.concat([df_dummies,df.drop(columns=['popularity', 'genre'])], axis=1),
        df.drop(columns=['popularity', 'genre']),
        pd.concat([df_dummies,df[['loudness', 'energy', 'danceability']]], axis=1),
        df[['loudness', 'energy', 'danceability']]
    ]

    
    for X in X_options:
        train_and_test_model(X, y, wb['ModelReport'], "linear")
        train_and_test_model(X,y, wb['ModelReport'], "ridge")
        train_and_test_model(X,y, wb['ModelReport'], "lasso")



    wb.save(filename)
    

if __name__ == '__main__':
    main()