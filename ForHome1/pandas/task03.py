import pandas as pd


def main():
    dataframe = pd.read_csv('cars.csv')
    print(dataframe)

    print("\nAfter setting first column as index:")

    dataframe = dataframe.rename(columns={dataframe.columns[0]: ""})
    dataframe = dataframe.set_index(dataframe.columns[0])

    print(dataframe)

if __name__ == "__main__":
    main()
