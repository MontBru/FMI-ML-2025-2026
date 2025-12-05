from task05 import predict, calculate_loss, X, gradient_descent_step, initialize_model, train_model
import numpy as np

nand_y = np.array([1, 1, 1, 0])
data_nand = (X, nand_y)

if __name__ == '__main__':
    nand_model = initialize_model(0, 1, W_shape=(2,), b_shape=(1,))

    print("Training NAND model:")
    nand_model = train_model(nand_model, data_nand, epochs=100000, lr=1e-1)

    print("NAND predictions after training:")
    print(predict(nand_model, data_nand[0]))

    #The model is really good and confident in its' predictions
    #by just using the same structure like AND and OR but with
    #different data

    