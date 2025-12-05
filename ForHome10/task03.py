from task02 import calculate_loss
from task01 import create_dataset, initialize_weights

def gradient_descent_step(model, data,eps=1e-8,learning_rate=0.001):
    grad = (calculate_loss(model+eps, data) - calculate_loss(model, data))/eps
    return model - learning_rate*grad
    

if __name__ == '__main__':
    data = create_dataset(6)
    w = initialize_weights(0, 10, 42)

    print(calculate_loss(w, data))

    learning_rates = [.001, .01, .1, 1, 10]

    for lr in learning_rates:
        new_w = gradient_descent_step(w, data,learning_rate=lr)
        print(calculate_loss(new_w, data))

    epoch_count = 10
    for i in range(epoch_count):
        w = gradient_descent_step(w, data)
        print(f"Epoch({i+1}/{epoch_count}): Loss: {calculate_loss(w, data)}, W: {w} ")