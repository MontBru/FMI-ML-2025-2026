import numpy as np
import ForHome1.numpy.datasets as datasets

def get_statistics(arr):
    print(f"Number of rows and columns: {arr.shape}")
    means = arr.mean(axis=0)
    medians = np.median(arr, axis=0)
    maxes = arr.max(axis=0)
    mins = arr.min(axis=0)

    print("Summary statistics for height:")
    print(f"Mean height: {means[0]}")
    print(f"Median height: {medians[0]}")
    print(f"Max height: {maxes[0]}")
    print(f"Min height: {mins[0]}")

    print("Summary statistics for weight:")
    print(f"Mean weight: {means[1]}")
    print(f"Median weight: {medians[1]}")
    print(f"Max weight: {maxes[1]}")
    print(f"Min weight: {mins[1]}")

    print("Summary statistics for age:")
    print(f"Mean age: {means[2]}")
    print(f"Median age: {medians[2]}")
    print(f"Max age: {maxes[2]}")
    print(f"Min age: {mins[2]}\n")


def main():

    np_baseball = np.array(datasets.baseball_dataset)
    get_statistics(np_baseball)

    #The heights are abnormal, because we find that there are values that look like this 76000 which is impossible

    np_baseball = np.array(datasets.baseball_dataset_corrected)
    get_statistics(np_baseball)

    print("Additional statistics:")
    print("Correlation matrix:")
    print(np.corrcoef(np_baseball.T))

    #From the correlation matrix we find that there is a .53 correlation between height and weight which means that they are correlated




if __name__ == '__main__':
    main()
