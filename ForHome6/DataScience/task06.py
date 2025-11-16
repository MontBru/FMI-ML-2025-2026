from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report, roc_curve, recall_score, precision_score
import openpyxl
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer

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


def train_and_test_model(X, y, ws, name, args = None, export_to_file = True):
    global model_number

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    steps = [('imp', imp),('scaler', StandardScaler())]

    if 'logistic' in name:
        if args == None:
            param_grid = {
                'logistic__C': np.arange(.1,1, .1),
                'logistic__penalty': ['l2']
            }
        else:
            param_grid = args.params
        reg = LogisticRegression()
        steps.append(('logistic', reg))

    elif 'svm' in name:
        if args == None:
            param_grid = {
                'svm__kernel':['rbf', 'poly', 'linear'],
                'svm__C': np.arange(.1,1, .1),
                'svm__gamma': ['scale'],
            }
        else:
            param_grid = args.params
        reg =  svm.SVC()
        steps.append(('svm', reg))

    elif 'cart' in name:
        if args == None:
            param_grid = {
                'cart__criterion':['gini','entropy'],
                'cart__max_depth': np.arange(1, 100, 1),
                'cart__min_samples_split': np.arange(2, 50, 1)
            }
        else:
            param_grid = args.params
        reg = DecisionTreeClassifier()
        steps.append(('cart', reg))

    elif 'ensemble' in name:
        if args == None or (not ('models' in args)):
            models = ['svm']
        else:
            models = args['models']


        if args == None or (not ('params' in args)) :
            param_grid = {}
        else:
            param_grid = args['params']

        trained_models = []
        for model in models:
            cv, cache = train_and_test_model(X, y, ws, model, export_to_file=False)
            trained_models.append((model, cv))

        reg = VotingClassifier(trained_models)
        steps.append(('ensemble', reg))

    pipeline = Pipeline(steps)

    cv = RandomizedSearchCV(pipeline, param_distributions=param_grid,cv=kf,return_train_score=True, scoring='f1')
    cv.fit(X=X_train, y=y_train)

    y_pred = cv.predict(X_test)

    if export_to_file:
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

        ws.append([f'{name} model on {X.columns}', X.shape[1], str(cv.best_params_), scores['1']['precision'], scores['1']['precision']/base_model_precision * 100 - 100,scores['1']['recall'], scores['1']['recall']/base_model_recall * 100 - 100, scores['1']['f1-score'], scores['1']['f1-score']/base_model_f1 * 100 -100])
        model_number += 1

    return cv, [y_test, y_pred]


def main():
    df = pd.read_csv("indian_liver_patient_dataset.csv", names = ['age', 'gender', 'tb', 'db', 'alkphos', 'sgpt', 'sgot', 'tp', 'alb', 'ag ratio', 'has_liver_disease'])
    df_dummies = pd.get_dummies(df['gender'], drop_first=True, dtype=int)
    df = pd.concat([df_dummies,df.drop(columns=['gender'])], axis=1)

    df['has_liver_disease'] = 2 - df['has_liver_disease']

    y = df['has_liver_disease']
    filename = './model_report_task06.xlsx'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Number of variables','Hyperparams', "Precision", "Precision increase from base model (in %)", "Recall", "Recall increase from base model (in %)", "F1 Score", "F1 score increase from base model (in %)", "Confusion matrix", "Comments"])

    base_model(y, ws)

    best_model = None
    best_model_accuracy = 0
    best_model_cache = []

    
    X_options = [
        df, #training model with all data
        df.drop(columns=['tb', 'sgpt', 'tp']), 
        df.drop(columns=['tb', 'sgpt', 'tp', 'ag ratio']), 
    ]

    model_options = [
        'logistic',
        'svm',
        'cart',
        'ensemble'
    ]

    ensemble_models = [
        ["logistic", "svm", 'cart'],
        ["svm", 'cart']
    ]

    for X in X_options:
        for name in model_options:
            if name == 'ensemble':
                for i in range(len(ensemble_models)):
                    args = {'models': ensemble_models[i]}
                    curr_model, curr_model_cache = train_and_test_model(X, y, wb['ModelReport'], name, args=args)

                    if curr_model.best_score_ > best_model_accuracy:
                        best_model_accuracy = curr_model.best_score_
                        best_model = curr_model
                        best_model_cache = curr_model_cache

            else:
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