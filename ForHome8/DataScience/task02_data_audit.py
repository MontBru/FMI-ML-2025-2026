import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import numpy as np
from sklearn.decomposition import PCA

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Homework1.data_audit import save_pairplot_to_excel, save_column_chart_to_excel, save_correlation_heatmap_to_excel,create_pairplot, create_correlation_heatmap, create_bar_or_histogram_chart_for_column_and_save_data_to_excel, describe_df


def create_scatter_plot(X, name, labels, diagrams_dir):
    # ----------- SCATTERPLOT OF CLUSTERS -----------
    plt.figure(figsize=(7, 5))
    if labels is not None:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
    else:
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1])
    plt.title(f"{name} — Cluster Scatterplot")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.tight_layout()

    scatter_path = f"{diagrams_dir}/{name}_scatter.png"
    plt.savefig(scatter_path)
    plt.cla()

    return scatter_path

def export_to_excel(img_path, ws):
    img = openpyxl.drawing.image.Image(img_path)
    row = ws.max_row + 1
    col = 'A'
    cell_ref = f"{col}{row}"

    img_width, img_height = img.width, img.height
    ws.column_dimensions[col].width = img_width / 7
    ws.row_dimensions[row].height = img_height

    img.anchor = cell_ref
    ws.add_image(img)

    col = chr(ord(col) + 1)

def main():
    df = pd.read_csv("seeds_dataset.txt", 
                 sep=r"\s+",        # split on whitespace / tabs
                 header=None,       # no header in your snippet
                 engine="python",   # allows regex separators
                 names = ['area', 'perimeter', 'compactness', 'kernel_length', 'kernel_width', 'assymetry_coef', 'kernel_groove_length']
                 )  

    filename = './data_audit_task02.xlsx'
    diagrams_folder_path = './diagrams'
    excel_writer = pd.ExcelWriter(filename)

    describe_df(df).to_excel(excel_writer=excel_writer, sheet_name='data_audit')

    bar_chart_cols = []

    create_correlation_heatmap(df, diagrams_folder_path, filename='heatmap1.png')
    create_pairplot(df, diagrams_folder_path)

    for col in df.columns:
        if col == 'Unnamed: 0':
            continue

        create_bar_or_histogram_chart_for_column_and_save_data_to_excel(df, col, bar_chart_cols, excel_writer, diagrams_folder_path)

    excel_writer.close()

    wb = openpyxl.load_workbook(filename)
    save_correlation_heatmap_to_excel(wb, diagrams_folder_path, filename='heatmap1.png')
    save_pairplot_to_excel(wb, diagrams_folder_path)

    for col in df.columns:
        if col == 'Unnamed: 0':
            continue
        save_column_chart_to_excel(wb, col, diagrams_folder_path)

    print("Saved column charts in Excel")

    #these two columns are correlated

    pca = PCA()
    correlated_columns = df[['area', 'compactness']]
    transformed_correlated_columns = pd.DataFrame(pca.fit_transform(correlated_columns))

    name = ['before', 'after']
    i = 0
    for corr_col in [correlated_columns, transformed_correlated_columns]:
        create_correlation_heatmap(corr_col, diagrams_folder_path, title=f'correlation between area and compactness {name[i]} PCA', filename=f'heatmap{i+2}.png')
        save_correlation_heatmap_to_excel(wb, diagrams_folder_path, filename=f'heatmap{i+2}.png')

        scatter_plot_path = create_scatter_plot(X=corr_col, name=f'area compactness {name[i]} PCA', labels=None, diagrams_dir=diagrams_folder_path)
        export_to_excel(scatter_plot_path, wb['data_audit'])
        i += 1


    wb.save(filename)



if __name__ == '__main__':
    main()