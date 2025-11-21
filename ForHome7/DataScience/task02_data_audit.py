import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import numpy as np

def create_correlation_heatmap(df, diagrams_folder_path):
    corr = df.select_dtypes('number').corr()
    sns.heatmap(data=corr)
    plt.savefig(f'{diagrams_folder_path}/heatmap.png', bbox_inches='tight')
    plt.clf()
    
    print("Created correlation heatmap")

def create_pairplot(df, diagrams_folder_path):
    # sns.pairplot(df.select_dtypes('number'), hue='has_liver_disease')
    sns.pairplot(df.select_dtypes('number'))
    plt.savefig(f'{diagrams_folder_path}/pairplot.png', bbox_inches='tight')
    plt.clf()
    plt.figure(figsize=(8, 6))

def save_correlation_heatmap_to_excel(wb, diagram_folder_path):
    ws = wb['data_audit']
    img = openpyxl.drawing.image.Image(f'{diagram_folder_path}/heatmap.png')
    img.anchor = 'A1'
    ws.add_image(img)
    print("Saved correlation heatmap in Excel")

def save_pairplot_to_excel(wb, diagram_folder_path):
    ws = wb['data_audit']
    img = openpyxl.drawing.image.Image(f'{diagram_folder_path}/pairplot.png')
    img.anchor = 'A1'
    ws.add_image(img)

    print("Saved pairplot in Excel")

def create_bar_or_histogram_chart_for_column_and_save_data_to_excel(df, col, bar_chart_cols, excel_writer, diagram_folder_path):
    value_counts = df[col].value_counts()
    value_counts = value_counts.sort_index()
    value_counts.T.to_excel(excel_writer=excel_writer, sheet_name=col)
    if col in bar_chart_cols:
        sns.barplot(value_counts)
        plt.title(f'{col} bar chart')
    else:
        sns.histplot(df[col])
        plt.title(f'{col} histogram')
        plt.ylabel('count')

    plt.xlabel(col)
    plt.savefig(f'{diagram_folder_path}/{col}.png')
    plt.cla()

    print("Saved column data to Excel")
    print(f"Created histograms and bar charts for column {col}")

def save_column_chart_to_excel(wb, col, diagram_folder_path):
    ws = wb[col]
    img = openpyxl.drawing.image.Image(f'{diagram_folder_path}/{col}.png')
    img.anchor = 'D1'
    ws.add_image(img)

def describe_df(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0']).describe().T
    desc = df.describe().T
    desc['num_unique']        = df.nunique()
    desc['pct_unique']        = df.nunique() / len(df) * 100
    desc['num_missing']       = df.isna().sum()
    desc['pct_missing']       = df.isna().sum() / len(df) * 100

    desc = desc.round(2)
    return desc

def main():
    df = pd.read_csv("bike_sharing.csv")
    # df_dummies = pd.get_dummies(df['origin'], drop_first=True, dtype=int)
    #df_dummies = pd.get_dummies(df['gender'], drop_first=False, dtype=int)

    # df = pd.concat([df_dummies,df.drop(columns=['origin'])], axis=1)
    #df = pd.concat([df_dummies,df], axis=1)

    df["datetime"] = pd.to_datetime(df["datetime"])   # ensure dtype

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second

    df = df.drop(columns=['datetime'])


    filename = './data_audit_task02.xlsx'
    diagrams_folder_path = './diagrams'
    excel_writer = pd.ExcelWriter(filename)

    describe_df(df).to_excel(excel_writer=excel_writer, sheet_name='data_audit')

    bar_chart_cols = []

    create_correlation_heatmap(df, diagrams_folder_path)
    create_pairplot(df, diagrams_folder_path)

    for col in df.columns:
        if col == 'Unnamed: 0':
            continue

        create_bar_or_histogram_chart_for_column_and_save_data_to_excel(df, col, bar_chart_cols, excel_writer, diagrams_folder_path)

    excel_writer.close()

    wb = openpyxl.load_workbook(filename)
    save_correlation_heatmap_to_excel(wb, diagrams_folder_path)
    save_pairplot_to_excel(wb, diagrams_folder_path)

    for col in df.columns:
        if col == 'Unnamed: 0':
            continue
        save_column_chart_to_excel(wb, col, diagrams_folder_path)

    print("Saved column charts in Excel")
    wb.save(filename)

if __name__ == '__main__':
    main()