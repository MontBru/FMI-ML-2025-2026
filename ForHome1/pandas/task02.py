import pandas as pd
import datasets


def main():
    dataframe = pd.DataFrame({
        "country": datasets.names,
        "cars_per_cap": datasets.cpc,
        "drives_right": datasets.dr
    }, index=datasets.labels)
    print(dataframe)

if __name__ == "__main__":
    main()
