import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import sys
import random

from model_utils import down_split

def model_train(train, test, file_number, ax=None):
    train = train.copy()
    test = test.copy()

    down = train['down'].iloc[0]

    train_X = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    train_y = train['success']
    test_X = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    test_y = test['success']

    model = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C = .1,
        max_iter=5000,
        class_weight='balanced'
    )

    model.fit(train_X, train_y)

    y_preds = model.predict(test_X)
    y_proba = model.predict_proba(test_X)[:,1]
    auc_score = roc_auc_score(test_y, y_proba)

    down_str = 'DOWN ' + str(down)
    print(f"{down_str:-^15}")
    print(f"{'':-^15}")
    print(classification_report(test_y, y_preds))
    print("AUC:", auc_score)
    print(f"{'':-^15}")
    print(confusion_matrix(test_y, y_preds))
    print(f"{'':-^15}")

    test['pred'] = y_preds
    test['prob'] = y_proba

    test.to_csv(f'data/evals/log_{down}_preds.csv')
    X, Y, _ = roc_curve(test_y, y_proba)

    if not ax:
        fig, ax = plt.figure(figsize=(8,6))
    plt.plot(X, Y, label=f"DOWN {down} AUC = {auc_score:.3f}", linewidth=2)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

if __name__ == '__main__':
    try:
        print(f"📂 Loading: log_train.csv")
        train = pd.read_csv('data/processed/log_train.csv')
        print("✅ Training DataFrame loaded!")
        print(f"📂 Loading: log_test.csv")
        test = pd.read_csv('data/processed/log_test.csv')
        print("✅ Testing DataFrame loaded!")
    except:
        print('❌ Unable to load file, please try again')
        sys.exit()

    file_number = random.randint(1, 9999)
    train1, train2, train34, test1, test2, test34 = down_split(train, test)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    model_train(train1, test1, file_number, ax)
    model_train(train2, test2, file_number, ax)
    model_train(train34, test34, file_number, ax)

    plt.savefig(f'plots/roc_log_{file_number}.png')