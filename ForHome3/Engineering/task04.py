from ml_lib.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import openpyxl


def train_and_test_model(X, y, ws):
    #I'm purposefully not doing stratify because here it's not a classification model and the y can take too many values.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)

    if X_test.shape[1] == 1:
        plt.scatter(X_test, y_test, s=.5) # plot the true target values
        plt.plot(X_test, predictions, 'red') # plot the predicted target values
        plt.ylabel('Money earned in sales')
        plt.xlabel(f'Money spent on {X_test.columns[0]}')
        plt.savefig(f'./ForHome3//diagrams/regression_diagram_{X_test.columns[0]}.png')
        plt.cla()

        img = openpyxl.drawing.image.Image(f'./ForHome3//diagrams/regression_diagram_{X_test.columns[0]}.png')
        img.anchor = 'E5'
        ws.add_image(img)

    ws.append([f'Regression on {X_test.columns}', X_test.shape[1], reg.score(X_test, y_test)])
    



def main():
    df = pd.read_csv('./ForHome3/advertising_and_sales_clean.csv')
    filename = './ForHome3/Engineering/model_report.xlsx'

    wb = openpyxl.Workbook()
    wb.create_sheet('ModelReport')
    ws = wb['ModelReport']

    ws.append(['Model', 'Number of variables','Accuracy'])

    X,y = df[['tv']], df['sales']
    train_and_test_model(X, y, wb['ModelReport'])


    X,y = df[['social_media']], df['sales']
    train_and_test_model(X, y, wb['ModelReport'])

    X,y = df[['radio']], df['sales']
    train_and_test_model(X, y, wb['ModelReport'])

    wb.save(filename)
    

if __name__ == '__main__':
    main()