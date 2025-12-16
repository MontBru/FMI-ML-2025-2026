import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

X = np.array([
    0,0,
    0,1,
    1,0,
    1,1
]).reshape(4,2)

y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

data_and = (X, y_and)

data_or = (X, y_or)


#the model will look like this sigmoid(w1*x1 + w2*x2 + b1) = y


#w1*x1 and w2*x2 looks like dot product so i will implement 
#them using np arrays

def predict(model, X):
    #Model will be a 2d tuple (w, b)
        #where w is a (D,) np array that has weights
        #b is a (D,) np array that has biases
        #where D is dimension of the network in this case it's 2

    #X is a (N, D) np array
    W = model[0]
    b = model[1]

    X = np.array(X)

    X_W = X @ W
    # print(f"{X=}")
    # print(f"{W=}")
    # print(f"{X_W=}")
    if b.size > 0:
        X_W = X_W + b

    #apply sigmoid and after that sum row-wise

    # print(f"{X_W=}")
    return sigmoid(X_W)
    # return X_W

def calculate_loss(model, data):
    #Model will be a 2d tuple (w, b)
        #where w is a (D,) np array that has weights
        #b is a (D,) np array that has biases
        #where D is dimension of the network in this case it's 2
    #data will be a 2d tuple (X, y)
        #where X is a (N, D) np array
        #y is a (N,) np array of the results

    X = np.array(data[0])
    y = np.array(data[1])

    y_pred = predict(model, X).reshape(y.shape)
    # print(f'{y=}')
    # print(f'{y_pred=}')
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
    
    W = np.array(model[0])
    b = np.array(model[1])

    #I have to find gradient of W and gradient of b

    grad_W = np.zeros(W.size)
    grad_b = np.zeros(b.size)

    #moving W by epsilon in every dimension and seeing how the loss changes
    #(so this is dL/dW)
    for dim in range(W.size):
        eps_vector = np.zeros(W.size)
        eps_vector[dim] = eps
        eps_vector = eps_vector.reshape(W.shape)

        eps_changed_loss = calculate_loss((W+eps_vector, b), data)
        model_loss = calculate_loss(model, data)

        grad_W[dim] = (eps_changed_loss - model_loss)/eps
        
    
    
    #moving b by epsilon in every dimension and seeing how the loss changes
    #(so this is dL/db)
    for dim in range(b.size):
        eps_vector = np.zeros(b.size)
        eps_vector[dim] = eps
        eps_vector = eps_vector.reshape(b.shape)

        grad_b[dim] = (calculate_loss((W, b+eps_vector), data) - calculate_loss(model, data))/eps
        

    
    
    grad_W = np.array(grad_W)
    grad_b = np.array(grad_b)

    grad_W = grad_W.reshape(W.shape)
    grad_b = grad_b.reshape(b.shape)

    new_W = W - grad_W*learning_rate
    new_b = b - grad_b*learning_rate

    return (new_W, new_b)
    
def initialize_model(x, y, seed=0, W_shape = (1,), b_shape=(1,)):
    rng = np.random.default_rng(seed)
    W = rng.uniform(x, y, W_shape)
    b = rng.uniform(x,y, b_shape)

    return (W, b)

def train_model(model, data, epochs=10, verbose=True, lr=1e-3):
    for i in range(epochs):
        model = gradient_descent_step(model, data, learning_rate=lr)
        if (i+1)%10000 == 0 and verbose==True:
            W = model[0]
            b = model[1]
            print(f"Epoch({i+1}/{epochs}): Loss: {calculate_loss(model, data)}, W: {W}, b: {b}")
    return model


if __name__ == '__main__':
    and_model = initialize_model(0, 1, W_shape=(2,), b_shape=(0,))
    or_model = initialize_model(0, 1, W_shape=(2,), b_shape=(0,))

    print("Training AND model:")
    and_model = train_model(and_model, data_and, epochs=100000, lr=1e-1)

    print("Training OR model:")
    or_model = train_model(or_model, data_or, epochs=100000, lr=1e-1)

    print("AND predictions after training:")
    print(predict(and_model, data_and[0]))
    print("OR predictions after training:")
    print(predict(or_model, data_or[0]))

    #The model isn't confident in its' decisions because it 
    #outputs values near .5