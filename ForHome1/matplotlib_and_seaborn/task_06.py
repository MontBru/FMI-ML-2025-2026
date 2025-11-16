import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import numpy as np


def main():
    gdp_cap = np.array(datasets.gdp_cap) * 2
    life_exp = np.array(datasets.life_exp)
    pop = datasets.pop2
    pop_asc_indices = np.argsort(pop)
    china_index = pop_asc_indices[-1]
    india_index = pop_asc_indices[-2]

    plt.scatter(gdp_cap, life_exp, s=datasets.pop2, color=datasets.colors)
    plt.title("World Development in 2007")
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectandy [in years]")
    plt.xscale('log')
    plt.grid(visible=True)
    plt.text(gdp_cap[china_index], life_exp[china_index], "China")
    plt.text(gdp_cap[india_index], life_exp[india_index], "India")

    plt.show()

    sns.scatterplot(x=gdp_cap,
                    y=life_exp,
                    s=datasets.pop2,
                    color=datasets.colors)
    plt.title("World Development in 2007")
    plt.xlabel("GDP per Capita [in USD]")
    plt.ylabel("Life Expectandy [in years]")
    plt.xscale('log')
    plt.grid(visible=True)
    plt.text(gdp_cap[china_index], life_exp[china_index], "China")
    plt.text(gdp_cap[india_index], life_exp[india_index], "India")

    plt.show()

#Answer to question: A


if __name__ == '__main__':
    main()
