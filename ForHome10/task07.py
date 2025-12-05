from task05 import sigmoid
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Generate x-values from -10 to 10
    x = np.linspace(-10, 10, 400)

    # Compute f(x)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Function Plot from -10 to 10")
    plt.grid(True)
    plt.show()