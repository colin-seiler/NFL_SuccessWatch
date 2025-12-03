import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import sys
import random
import yaml

def model_train(train, test, file_number):
    train = train.copy()
    test = test.copy()

    train_X = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    train_y = train['success']
    test_X = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    test_y = test['success']

    model = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C = .1,
        max_iter=5000,
        class_weight='balanced',
        random_state=42
    )

    model.fit(train_X, train_y)

    y_preds = model.predict(test_X)
    y_proba = model.predict_proba(test_X)[:,1]
    auc_score = roc_auc_score(test_y, y_proba)

    print(f"{'LOG':-^15}")
    print(f"{'':-^15}")
    print(classification_report(test_y, y_preds))
    print("AUC:", auc_score)
    print(f"{'':-^15}")
    print(confusion_matrix(test_y, y_preds))
    print(f"{'':-^15}")

    test['pred'] = y_preds
    test['prob'] = y_proba

    test.to_csv(f'data/evals/log_preds_{file_number}.csv')
    X, Y, _ = roc_curve(test_y, y_proba)

    plt.figure(figsize=(8,6))
    plt.plot(X, Y, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'plots/roc_log_34_{file_number}.png')
    plt.close()

    return {
        'model':model,
        'y_preds':y_preds,
        'y_proba':y_proba,
        'auc':auc_score,
        'fpr':X,
        'tpr':Y
    }

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

    try:
        with open("cfg/down.yml") as f:
            cfg = yaml.safe_load(f)
    except:
        print('❌ Unable to load cfg, using all features instead of best')

    file_number = random.randint(1, 9999)
    train = train[(train['down'] == 3) | (train['down'] == 4)]
    test = test[(test['down'] == 3) | (test['down'] == 4)]

    model_train(train, test, file_number)