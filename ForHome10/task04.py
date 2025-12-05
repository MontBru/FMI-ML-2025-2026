from task02 import calculate_loss
from task01 import create_dataset, initialize_weights
from task03 import gradient_descent_step


if __name__ == '__main__':
    
    print("Seed is 42:")
    data = create_dataset(6)
    w = initialize_weights(0, 10, 42)

    print(calculate_loss(w, data))

    epoch_count = 500
    for i in range(epoch_count):
        w = gradient_descent_step(w, data)
        if (i+1)%100 == 0:
            print(f"Epoch({i+1}/{epoch_count}): Loss: {calculate_loss(w, data)}, W: {w} ")

    print("Seed is random:")

    data = create_dataset(6)
    w = initialize_weights(0, 10)

    print(calculate_loss(w, data))

    epoch_count = 500
    for i in range(epoch_count):
        w = gradient_descent_step(w, data)
        if (i+1)%100 == 0:
            print(f"Epoch({i+1}/{epoch_count}): Loss: {calculate_loss(w, data)}, W: {w} ")
