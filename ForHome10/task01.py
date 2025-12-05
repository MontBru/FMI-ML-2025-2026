import numpy as np

def create_dataset(n):
    return [(i, 2*i) for i in range(n)]

#The model looks like wx + b in its' general form

def initialize_weights(x, y, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(x, y)

if __name__ == '__main__':
    print(create_dataset(4))
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))