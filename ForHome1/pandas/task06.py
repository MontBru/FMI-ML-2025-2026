import pandas as pd


def main():
    df = pd.read_csv('cars_advanced.csv', index_col=0)
    for label, row in df.iterrows():
        print(f'Label is "{label}"')
        print("Row contents:")
        print(row)
        print("\n")

if __name__ == '__main__':
    main()
