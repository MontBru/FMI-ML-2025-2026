from task05 import X, predict
import numpy as np

xor_y = np.array([0, 1, 1, 0])
and_y = np.array([0,0,0,1])
or_y = np.array([0,1,1,1])

def calculate_loss(model, data):
    #Model will be a 2d tuple (w, b)
        #where w is a (D,) np array that has weights
        #b is a (D,) np array that has biases
        #where D is dimension of the network in this case it's 2
    #data will be a 2d tuple (X, y)
        #where X is a (N, D) np array
        #y is a (N,) np array of the results

    X = data[0]
    y = data[1]

    if isinstance(model, tuple):
        y_pred = predict(model, X)
    elif isinstance(model, list):
        y_pred = X
        for layer in model:
            y_pred = predict(layer, y_pred)
    else:
        y_pred = model.forward(X)

    y_pred = y_pred.reshape(y.shape)

    # print(f'{y=}')
    # print(f'{y_pred=}')
    # print(f"{y=}")
    # print(f"{y_pred=}")
    loss = np.mean((y - y_pred)**2)

    return loss

def gradient_descent_step(model, data,eps=1e-8,learning_rate=0.001):
    #Model will be a 2d tuple (w, b)
        #where w is a (D,) np array that has weights
        #b is a (D,) np array that has biases
        #where D is dimension of the network in this case it's 2
    #data will be a 2d tuple (X, y)
        #where X is a (N, D) np array
        #y is a (N,) np array of the results

    result = []

    for layer_num,layer  in enumerate(model):

        temp_model = model.copy()

        W = layer[0]
        b = layer[1]

        #I have to find gradient of W and gradient of b

        grad_W = np.zeros(W.size)
        grad_b = np.zeros(b.size)

        #moving W by epsilon in every dimension and seeing how the loss changes
        #(so this is dL/dW)
        for dim in range(W.size):
            eps_vector = np.zeros(W.size)
            eps_vector[dim] = eps
            eps_vector = eps_vector.reshape(W.shape)
            temp_model[layer_num] = (W+eps_vector, b)
            # print(f'{W=}')
            
            grad_W[dim] = (calculate_loss(temp_model, data) - calculate_loss(model, data))/eps
            # print(f"{grad_W[dim]=}")
            # print(f"{np.array(temp_model[layer_num][0]) - np.array(model[layer_num][0])=}")
            # print(f"{model=}")
            # print(f"{temp_model=}")
            
        #moving b by epsilon in every dimension and seeing how the loss changes
        #(so this is dL/db)
        for dim in range(b.size):
            eps_vector = np.zeros(b.size)
            eps_vector[dim] = eps
            eps_vector = eps_vector.reshape(b.shape)
            temp_model[layer_num] = (W, b+eps_vector)

            grad_b[dim] = (calculate_loss(temp_model, data) - calculate_loss(model, data))/eps

        grad_W = grad_W.reshape(W.shape)
        grad_b = grad_b.reshape(b.shape)

        # print(f"{grad_W=}")
        # print(f"{grad_b=}")

        new_layer = (W - grad_W*learning_rate, b - grad_b*learning_rate)
        result.append(new_layer)

    return result
    
def initialize_model(x, y, rng, W_shape = (1,), b_shape=(1,)):
    
    W = rng.uniform(x, y, W_shape)
    b = rng.uniform(x,y, b_shape)

    return (W, b)

class Xor:

    #The architecture of Xor should be 1 hidden layer with 2 neurons so:
    #it is (2, 2, 1) two inputs, two hidden layer outputs, one final output

    def __init__(self, seed = 0):
        self.layers = []
        rng = np.random.default_rng(seed)
        self.layers.append(initialize_model(-1, 1, rng=rng, W_shape=(2,2), b_shape=2))
        self.layers.append(initialize_model(-1, 1, rng=rng, W_shape=(2,1), b_shape=(1,)))
        print(self.layers)


    def forward(self, X):
        out = X
        for layer in self.layers:
            new_out = predict(layer, out)
            out = new_out

        return out
    
    def backward(self, X, y, eps=1e-8,lr=1):
        return gradient_descent_step(self.layers, (X,y), eps, lr)

    def train(self, X, y, eps=1e-8, lr=1e-3, epochs=1000, verbose=True):
        for i in range(epochs):
            # print(f"Model before: {self.layers}")
            self.layers = self.backward(X,y, lr=lr, eps=eps)
            # print(f"Model after: {self.layers}")
            if (i+1)%1000 == 0 and verbose==True:
                print(f"Epoch({i+1}/{epochs}): Loss: {calculate_loss(self.layers, (X,y))}")
                
                # print(self.layers)
                # print(self.forward(X))


if __name__ == '__main__':

    y = xor_y

    model = Xor()
    y_pred = model.forward(X)
    print(y_pred)
    print(f'Loss before training: {calculate_loss(model, (X, y))}')

    model.train(X, y,eps=0.00001, lr=10, epochs=100000)
    print(f'Loss after training: {calculate_loss(model, (X, y))}')

    y_pred = model.forward(X)
    print(y_pred)
