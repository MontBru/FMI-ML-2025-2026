import matplotlib.pyplot as plt
import seaborn as sns
import datasets


def main():
    plt.scatter(datasets.pop2, datasets.life_exp)
    plt.xlabel("Population [in millions of people]")
    plt.ylabel("Life Expextancy [in years]")
    plt.title("World Development in 2007")

    plt.show()

    plt.scatter(datasets.pop2, datasets.life_exp)
    plt.xlabel("Population [in millions of people]")
    plt.ylabel("Life Expextancy [in years]")
    plt.title("World Development in 2007")
    plt.xscale('log')
    plt.show()

    sns.scatterplot(x=datasets.pop2, y=datasets.life_exp)
    plt.xlabel("Population [in millions of people]")
    plt.ylabel("Life Expextancy [in years]")
    plt.title("World Development in 2007")

    plt.show()

    sns.scatterplot(x=datasets.pop2, y=datasets.life_exp)
    plt.xlabel("Population [in millions of people]")
    plt.ylabel("Life Expextancy [in years]")
    plt.title("World Development in 2007")
    plt.xscale('log')

    plt.show()


    #From the chart we can see that there is no clear correlation between population size and life expectancy.

if __name__ == '__main__':
    main()
