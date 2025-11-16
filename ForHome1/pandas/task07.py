import pandas as pd


def main():
    df = pd.read_csv("cars_advanced.csv", index_col=0)
    for label, row in df.iterrows():
        print(f"{label}: {row['cars_per_cap']}")

if __name__ == "__main__":
    main()
