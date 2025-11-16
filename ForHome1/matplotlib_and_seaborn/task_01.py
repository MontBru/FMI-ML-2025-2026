import datasets
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print(f"Last year with estimation: {datasets.year[-1]}")
    print(f"Last estimated population: {datasets.pop[-1]}")

    sns.set_theme(style='darkgrid')

    plt.plot(datasets.year, datasets.pop)
    plt.show()

    sns.lineplot(x=datasets.year, y=datasets.pop)
    plt.show()

    #In approximately 2070 there will be more than 10 billion humans on the planet

    plt.scatter(datasets.gdp_cap, datasets.life_exp)
    plt.show()
    sns.scatterplot(x=datasets.gdp_cap, y=datasets.life_exp)
    plt.show()

    #the GDP per capita has a high influence on the age for the first 10'000 dollars but after that the age is mostly between 70 to 80 years with a small correlation with the GDP per capita

if __name__ == '__main__':
    main()
