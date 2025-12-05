from task01 import create_dataset, initialize_weights
import numpy as np

def calculate_loss(model, data):
    loss = 0
    for entry in data:
        loss += np.pow((model*entry[0])  - entry[1], 2)

    loss /= len(data)
    return loss

if __name__ == '__main__':
    data = create_dataset(6)
    w = initialize_weights(0, 10, 42)

    print(calculate_loss(w, data))
    print(calculate_loss(w + 0.001 * 2, data))
    print(calculate_loss(w + 0.001, data))
    print(calculate_loss(w - 0.001, data))
    print(calculate_loss(w - 0.002, data))

    #when w increases the loss increses