from task01_data import points
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from Homework1.data_audit import save_pairplot_to_excel, save_column_chart_to_excel, save_correlation_heatmap_to_excel,create_pairplot, create_correlation_heatmap, create_bar_or_histogram_chart_for_column_and_save_data_to_excel, describe_df


def main():
    df = pd.DataFrame(points, columns=['X', 'Y'])
    filename = './data_audit_task01.xlsx'
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