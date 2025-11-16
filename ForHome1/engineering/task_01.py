import numpy as np


def main():
    rng = np.random.default_rng(123)
    randnum = rng.random()
    ints = rng.integers(1, 7, 2)
    step = 50
    dice = rng.integers(1, 7)

    print(f"Random float: {randnum}")
    print(f"Random integer 1: {ints[0]}")
    print(f"Random integer 2: {ints[1]}")
    print(f"Before throw step = {step}")
    print(f"After throw dice = {dice}")

    go_up = 0
    if dice <= 2:
        go_up = -1
    elif dice <= 5:
        go_up = 1
    else:
        go_up = rng.integers(1, 7)

    step += go_up
    print(f"After throw step = {step}")


if __name__ == '__main__':
    main()
