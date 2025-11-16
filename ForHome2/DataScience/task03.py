from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('telecom_churn_clean.csv')
    filename = './data_audit.xlsx'
    
    excel_writer = pd.ExcelWriter(filename)

    bar_chart_cols = ['area_code', 'international_plan', 'voice_mail_plan', 'churn']

    df.describe().T.to_excel(excel_writer=excel_writer, sheet_name='data_audit')
    for col in df.columns:
        if col == 'Unnamed: 0':
            continue

        value_counts = df[col].value_counts()
        value_counts.T.to_excel(excel_writer=excel_writer, sheet_name=col)
        if bar_chart_cols.count(col) == 1:
            value_counts.plot(kind='bar')
            plt.title(f'{col} bar chart')
        else:
            plt.hist(df[col])
            plt.title(f'{col} histogram')
            plt.ylabel('count')

        plt.xlabel(col)
        plt.savefig(f'./diagrams/{col}.png')
        plt.cla()

    excel_writer.close()

    wb = openpyxl.load_workbook(filename)
    for col in df.columns:
        if col == 'Unnamed: 0':
            continue
        ws = wb[col]
        img = openpyxl.drawing.image.Image(f'./diagrams/{col}.png')
        img.anchor = 'D1'
        ws.add_image(img)

    wb.save(filename)
        

if __name__ == '__main__':
    main()