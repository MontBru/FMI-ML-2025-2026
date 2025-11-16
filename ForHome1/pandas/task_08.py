import pandas as pd


def main():
    df = pd.read_csv("cars_advanced.csv", index_col=0)
    print("Before:")
    print(df)

    countries = df['country']
    countries = pd.DataFrame(countries).applymap(
        lambda x: str(x).upper()).rename(columns={'country': 'COUNTRY'})

    print(countries)
    df = df.join(countries)
    print("\nAfter:")
    print(df)

if __name__ == "__main__":
    main()
