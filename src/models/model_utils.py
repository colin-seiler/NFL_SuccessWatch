import pandas as pd

def down_split(train, test):
    train = train.copy()
    test = test.copy()

    train_1 = train[train['down'] == 1]
    train_2 = train[train['down'] == 2]
    train_34 = train[(train['down'] == 3) | (train['down'] == 4)]

    test_1 = test[test['down'] == 1]
    test_2 = test[test['down'] == 2]
    test_34 = test[(test['down'] == 3) | (test['down'] == 4)]

    return train_1, train_2, train_34, test_1, test_2, test_34