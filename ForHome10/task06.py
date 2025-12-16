from task05 import initialize_model, train_model, data_and, data_or, predict


if __name__ == '__main__':
    and_model = initialize_model(0, 1, W_shape=(2,1), b_shape=(1,))
    or_model = initialize_model(0, 1, W_shape=(2,1), b_shape=(1,))

    print(and_model[0])
    print(and_model[1])

    print("Training AND model:")
    and_model = train_model(and_model, data_and, epochs=100000, lr=1e-1)

    print("Training OR model:")
    or_model = train_model(or_model, data_or, epochs=100000, lr=1e-1)

    print("AND predictions after training:")
    print(predict(and_model, data_and[0]))
    print("OR predictions after training:")
    print(predict(or_model, data_or[0]))

    #The model is now much more confident in its' decisions
    #because it outputs values like 0.02 and 0.98