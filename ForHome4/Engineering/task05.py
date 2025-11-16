from ml_lib import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_curve
import openpyxl
import numpy as np


model_number = 1
base_model_accuracy = 0.797109726548979
base_model_precision = 0.75
base_model_recall = 0.448275862068966
base_model_f1 = 0.561151079136691

dir = './ForHome4/Engineering/diagrams/'

def train_and_test_model(X, y, ws, name):

    global model_number
    global base_model_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)


    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    param_grid = {
        'c': np.arange(0,1, .0001),
        'lr':np.logspace(1e-8, 1e-3, 50),
        'max_iter': np.array([50, 100, 200, 300]),
        'pd': np.array([5, 7, 10, 12, 15])
    }
    reg = linear_model.LogisticRegression()

    cv = RandomizedSearchCV(reg, param_distributions=param_grid,cv=kf,return_train_score=True)
    cv.fit(X=X_train, y=y_train)


    y_pred = cv.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.tight_layout()
    plt.savefig(f'{dir}{name}_confusion_matrix_{X_test.columns[0]}.png')
    plt.cla()

    img = openpyxl.drawing.image.Image(f'{dir}{name}_confusion_matrix_{X_test.columns[0]}.png')
    cell_ref = f'L{1 + model_number}'

    img_width, img_height = img.width, img.height

    column_letter = 'L'
    col_width = img_width / 7
    row_height = img_height / 1.333

    ws.column_dimensions[column_letter].width = col_width
    ws.row_dimensions[1 + model_number].height = row_height

    img.anchor = cell_ref
    ws.add_image(img, cell_ref)

    scores = classification_report(y_test, y_pred, labels=[1], output_dict=True)

    ws.append([f'{name} Regression on {X.columns}', X.shape[1], str(cv.best_params_), cv.best_score_, cv.best_score_/base_model_accuracy * 100 - 100, scores['1']['precision'], scores['1']['precision']/base_model_precision * 100 - 100,scores['1']['recall'], scores['1']['recall']/base_model_recall * 100 - 100, scores['1']['f1-score'], scores['1']['f1-score']/base_model_f1 * 100 -100])
    model_number += 1

    return cv, [y_test, y_pred]


def main():
    df = pd.read_csv('./ForHome4/DataScience/diabetes_clean.csv')
    filename = './ForHome4/Engineering/model_report_task05_engineering.xlsx'
    

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Number of variables','Hyperparams', 'Accuracy', "Accuracy increase from base model (in %)", "Precision", "Precision increase from base model (in %)", "Recall", "Recall increase from base model (in %)", "F1 Score", "F1 score increase from base model (in %)", "Confusion matrix", "ROC", "Comments"])

    best_model = None
    best_model_accuracy = 0
    best_model_cache = []

    y = df['diabetes']
    X_options = [
        df[['pregnancies', 'glucose', 'diastolic','triceps','insulin','bmi','dpf','age']],
        df[['bmi', 'glucose', 'age', 'dpf']],
        df[['glucose', 'diastolic','triceps','insulin','bmi','dpf','age']],
    ]

    for X in X_options:
        curr_model, curr_model_cache = train_and_test_model(X, y, wb['ModelReport'], 'logistic')
        if curr_model.best_score_ > best_model_accuracy:
            best_model_accuracy = curr_model.best_score_
            best_model = curr_model
            best_model_cache = curr_model_cache

    RocCurveDisplay.from_predictions(best_model_cache[0], best_model_cache[1])
    plt.title('Logistic Regression ROC Curve')
    plt.savefig(f'{dir}roc_curve.png')
    plt.cla()

    img = openpyxl.drawing.image.Image(f'{dir}roc_curve.png')
    img.anchor = 'A7'
    ws.add_image(img)


    wb.save(filename)
    

if __name__ == '__main__':
    main()