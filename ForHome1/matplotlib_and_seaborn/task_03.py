import matplotlib.pyplot as plt
import seaborn as sns
import datasets


def main():
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.hist(datasets.life_exp)
    f.suptitle("Life Expectancy distribution in 2007 vs 1950")
    f.supxlabel("Life Expectancy [in years]")
    f.supylabel("Number of countries with this life expectancy")

    ax2.hist(datasets.life_exp1950)

    #from the charts we can see that today most of the countries have a life expectancy between 70 and 80 years old, but during the 50s a big part of the countries had a life expectancy below 50 years old.

    #We can see a big difference between the two time periods since the life expectancy of 70 to 80 has become common while in the past a life expectancy wasn't even recorded.
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.suptitle("Life Expectancy distribution in 2007 vs 1950")
    f.supxlabel("Life Expectancy [in years]")
    f.supylabel("Number of countries with this life expectancy")

    sns.histplot(datasets.life_exp, ax=ax1)
    sns.histplot(datasets.life_exp1950, ax=ax2)
    sns.set_theme(style='darkgrid')

    plt.show()


if __name__ == '__main__':
    main()
