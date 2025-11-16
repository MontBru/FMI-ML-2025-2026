from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_curve, recall_score, precision_score
import openpyxl
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model_number = 1
base_model_precision = None
base_model_recall = None
base_model_f1 = None

def base_model(y, ws):
    global model_number
    global base_model_f1
    global base_model_recall
    global base_model_precision

    pred = 1

    y_pred = np.ones_like(y) * pred

    base_model_recall = recall_score(y, y_pred)
    base_model_precision = precision_score(y, y_pred)
    base_model_f1 = 2*base_model_precision*base_model_recall/(base_model_precision + base_model_recall)


    ws.append([f'Base model', '', '', base_model_precision, 0, base_model_recall, 0, base_model_f1, 0])
    model_number += 1


def train_and_test_model(X, y, ws, name):
    global model_number

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)


    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    steps = [('scaler', StandardScaler())]

    if name == 'logistic':
        param_grid = {
            'logistic__C': np.arange(.1,1, .1),
            'logistic__penalty': ['l2']
        }
        reg = LogisticRegression()
        steps.append(('logistic', reg))

    elif name == 'svm':
        param_grid = {
            'svm__kernel':['rbf', 'polynomial', 'linear'],
            'svm__C': np.arange(.1,1, .1),
            'svm__gamma': ['scale'],
        }
        reg =  svm.SVC()
        steps.append(('svm', reg))

    pipeline = Pipeline(steps)

    cv = GridSearchCV(pipeline, param_grid=param_grid,cv=kf,return_train_score=True, scoring='f1')
    cv.fit(X=X_train, y=y_train)

    y_pred = cv.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.savefig(f'./diagrams/{name}_confusion_matrix_{X_test.columns[0]}.png')
    plt.cla()

    img = openpyxl.drawing.image.Image(f'diagrams/{name}_confusion_matrix_{X_test.columns[0]}.png')
    cell_ref = f'J{1 + model_number}'

    img_width, img_height = img.width, img.height

    column_letter = 'J'
    col_width = img_width / 7
    row_height = img_height / 1.333

    ws.column_dimensions[column_letter].width = col_width
    ws.row_dimensions[1 + model_number].height = row_height

    img.anchor = cell_ref
    ws.add_image(img, cell_ref)

    scores = classification_report(y_test, y_pred, labels=[1], output_dict=True)

    ws.append([f'{name} Regression on {X.columns}', X.shape[1], str(cv.best_params_), scores['1']['precision'], scores['1']['precision']/base_model_precision * 100 - 100,scores['1']['recall'], scores['1']['recall']/base_model_recall * 100 - 100, scores['1']['f1-score'], scores['1']['f1-score']/base_model_f1 * 100 -100])
    model_number += 1

    return cv, [y_test, y_pred]


def main():
    df = pd.read_json('music_dirty.txt')
    filename = './model_report_task04.xlsx'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Number of variables','Hyperparams', "Precision", "Precision increase from base model (in %)", "Recall", "Recall increase from base model (in %)", "F1 Score", "F1 score increase from base model (in %)", "Confusion matrix", "Comments"])

    y = df['genre'] == 'Rock'

    base_model(y, ws)

    best_model = None
    best_model_accuracy = 0
    best_model_cache = []

    
    X_options = [
        df.drop(columns=['genre']),
        df[['popularity', 'energy', 'loudness']]
    ]

    model_options = [
        'logistic',
        'svm'
    ]

    for X in X_options:
        for name in model_options:
            curr_model, curr_model_cache = train_and_test_model(X, y, wb['ModelReport'], name)
            if curr_model.best_score_ > best_model_accuracy:
                best_model_accuracy = curr_model.best_score_
                best_model = curr_model
                best_model_cache = curr_model_cache

    RocCurveDisplay.from_predictions(best_model_cache[0], best_model_cache[1])
    plt.title('Logistic Regression ROC Curve')
    plt.savefig('./diagrams/roc_curve.png')
    plt.cla()

    img = openpyxl.drawing.image.Image(f'./diagrams/roc_curve.png')
    img.anchor = 'A7'
    ws.add_image(img)


    wb.save(filename)
    

if __name__ == '__main__':
    main()