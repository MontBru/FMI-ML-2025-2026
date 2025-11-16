import pandas as pd


def main():
    df = pd.read_csv("cars_advanced.csv", index_col=0)
    print(f"{df[df['drives_right']]}\n")

    print(f"{df[df['cars_per_cap']>500]['country']}\n")

    print(
        f"{df[(df['cars_per_cap']>=10) & (df['cars_per_cap'] <= 80)]['country']}\n"
    )

    print("Alternative:")
    print(f"{df[df['cars_per_cap'].between(10, 80, 'both')]['country']}\n")


if __name__ == '__main__':
    main()
