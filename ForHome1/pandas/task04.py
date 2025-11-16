import pandas as pd


def main():
    df = pd.read_csv("cars.csv", index_col=0)
    country_series = df['country']
    print(f"{country_series}\n")
    country_df = pd.DataFrame(country_series)
    print(f"{country_df}\n")
    country_dr_df = pd.DataFrame([df['country'], df['drives_right']]).T
    print(f"{country_dr_df}\n")
    first_three_entries = df.iloc[:3, :]
    print(f"{first_three_entries}\n")
    fourth_to_sixth_entries = df.iloc[3:6, :]
    print(f"{fourth_to_sixth_entries}\n")

    df_advanced = pd.read_csv("cars_advanced.csv", index_col=0)
    print(f"{df_advanced.loc['JPN']}\n")
    print(f"{pd.DataFrame([df_advanced.loc["JPN"],df_advanced.loc["EG"]]).T}\n")
    print(f"{df_advanced.at['MOR', 'drives_right']}\n")
    print(f"{df_advanced[["country", "drives_right"]].loc[["RU", "MOR"]]}\n")

if __name__ == '__main__':
    main()
