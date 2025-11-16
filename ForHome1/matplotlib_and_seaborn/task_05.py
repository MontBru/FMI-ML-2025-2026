import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import numpy as np


def main():
    gdp_cap = np.array(datasets.gdp_cap) * 2
    life_exp = np.array(datasets.life_exp)

    plt.scatter(gdp_cap, life_exp, s=datasets.pop2, color=datasets.colors)
    plt.title("World Development in 2007")
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectandy [in years]")
    plt.xscale('log')

    plt.show()

    sns.scatterplot(x=gdp_cap, y=life_exp, s=datasets.pop2, color = datasets.colors)
    plt.title("World Development in 2007")
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectandy [in years]")
    plt.xscale('log')

    plt.show()


if __name__ == '__main__':
    main()
