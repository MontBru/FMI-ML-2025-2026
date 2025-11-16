import numpy as np
import matplotlib.pyplot as plt


def main():
    rng = np.random.default_rng(123)

    all_walks = []

    for j in range(500):
        step = 0
        step_history = [0]
        for i in range(100):
            clumsiness = rng.random()

            if (clumsiness <= .005):
                step = 0
            else:
                dice = rng.integers(1, 7)
                go_up = 0
                if dice <= 2:
                    go_up = -1
                elif dice <= 5:
                    go_up = 1
                else:
                    go_up = rng.integers(1, 7)

                step += go_up
                step = max(step, 0)

            step_history.append(step)
        all_walks.append(step_history[-1])

    np_all_walks = np.array(all_walks)
    np_all_walks_over_60_steps = np_all_walks[np_all_walks >= 60]
    print(f"Probability of reaching 60 steps: {np_all_walks_over_60_steps.size/np_all_walks.size}")
    #The probability of reaching 60 steps is .536

    plt.hist(all_walks)
    plt.title("Random walks")
    plt.xlabel("End step")

    plt.show()


if __name__ == '__main__':
    main()
