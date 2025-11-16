import pandas as pd
import numpy as np
import datetime

def shuffle_sets(x, y, random_state):
    indices = np.arange(y.shape[0])

    rng = np.random.default_rng(seed = random_state)
    rng.shuffle(indices)

    x = x.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)
    return x,y

def train_test_split(x, y, test_size = .25, train_size = -1, shuffle = True, random_state=None, stratify=None):
    '''
    X: a pandas dataframe with features;
    y: a pandas series with labels;
    test_size: (optional) the proportion of the dataset to include in the test split. Default value: 0.25;
    train_size: (optional) the proportion of the dataset to include in the train split. Default value: the complement of the test_size;
    shuffle: (optional) whether to shuffle the data. Default value: True;
    random_state: (optional) a seed for the shuffling. Default value: None, meaning that results will not be reproducible;
    stratify: (optional) a pandas series with labels. Default value: None, meaning that the split will not be stratified.
    '''


    if random_state == None and shuffle == True:
        random_state = int(datetime.datetime.now().timestamp())

    if (test_size > 1 and train_size == -1) or (train_size + test_size > 1 and train_size != -1):
        return -1
    
    x_train = pd.DataFrame([])
    y_train = pd.Series([])
    x_test = pd.DataFrame([])
    y_test = pd.Series([])
    


    if stratify is None:
        test_size_num = int(test_size * y.size)
        train_size_num = int(train_size * y.size)

        if shuffle == True:
            x,y = shuffle_sets(x,y,random_state)

        x_test = x[:test_size_num]
        y_test = y[:test_size_num]
        if train_size == -1:
            x_train = x[test_size_num:]
            y_train = y[test_size_num:]
        else:
            x_train = x[test_size_num:test_size_num+train_size_num]
            y_train = y[test_size_num:test_size_num+train_size_num]
    else:
        labels = stratify.unique()

        for label in labels:
            stratified_x = x[y == label]
            stratified_y = y[y == label]

            test_size_num = int(test_size * stratified_y.size)
            train_size_num = int(train_size * stratified_y.size)

            if shuffle == True:
                shuffle_sets(stratified_x, stratified_y, random_state)

            x_test = pd.concat([x_test,stratified_x[:test_size_num]])
            y_test = pd.concat([y_test, stratified_y[:test_size_num]])
            if train_size == -1:
                x_train = pd.concat([x_train, stratified_x[test_size_num:]])
                y_train = pd.concat([y_train, stratified_y[test_size_num:]])
            else:
                x_train = pd.concat([x_train, stratified_x[test_size_num:test_size_num+train_size_num]])
                y_train = pd.concat([y_train, stratified_y[test_size_num:test_size_num+train_size_num]])
            
    return x_train, x_test, y_train, y_test
