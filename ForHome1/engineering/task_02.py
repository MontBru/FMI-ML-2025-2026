import numpy as np


def main():
    rng = np.random.default_rng(123)
    step = 0
    step_history = [0]

    for i in range(100):
        dice = rng.integers(1, 7)
        go_up = 0
        if dice <= 2:
            go_up = -1
        elif dice <= 5:
            go_up = 1
        else:
            go_up = rng.integers(1, 7)

        step += go_up
        step_history.append(step)

    print(step_history)

#It was expected but there is a time where we are on step -1 which shouldn't be possible



if __name__ == '__main__':
    main()
