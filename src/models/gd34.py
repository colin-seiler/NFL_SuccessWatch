import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import sys
import random
import shap
import yaml

def model_train(train, test, file_number):
    train = train.copy()
    test = test.copy()

    train_X = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    train_y = train['success']
    test_X = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    test_y = test['success']

    best_param_grid = {
        'subsample': .95, 
        'n_estimators': 300, 
        'min_child_weight': 9, 
        'max_depth': 5, 
        'learning_rate': .1, 
        'gamma': 3, 
    }
    pos = test_y.mean()
    weight = (1-pos)/pos

    model = XGBClassifier(
        **best_param_grid,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        scale_pos_weight=weight,
        random_state=42
    )

    model.fit(train_X, train_y)

    y_preds = model.predict(test_X)
    y_proba = model.predict_proba(test_X)[:,1]
    auc_score = roc_auc_score(test_y, y_proba)

    print(f"{'GD':-^15}")
    print(f"{'':-^15}")
    print(classification_report(test_y, y_preds))
    print("AUC:", auc_score)
    print(f"{'':-^15}")
    print(confusion_matrix(test_y, y_preds))
    print(f"{'':-^15}")

    test['pred'] = y_preds
    test['prob'] = y_proba

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_X)
    shap.summary_plot(shap_values, train_X)
    plt.savefig(f"plots/shap_gd_34_{file_number}.png", dpi=300, bbox_inches='tight')
    plt.close()

    test.to_csv(f'data/evals/gd_preds_{file_number}.csv', index=False)
    X, Y, _ = roc_curve(test_y, y_proba)

    plt.figure(figsize=(8,6))
    plt.plot(X, Y, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'plots/roc_gd_34_{file_number}.png')
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
        print(f"📂 Loading: gd_train.csv")
        train = pd.read_csv('data/processed/gd_train.csv')
        print("✅ Training DataFrame loaded!")
        print(f"📂 Loading: gd_test.csv")
        test = pd.read_csv('data/processed/gd_test.csv')
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