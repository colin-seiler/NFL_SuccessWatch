import numpy as np
import pandas as pd
import sys
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import shap
import matplotlib.pyplot as plt
import random

def model(train, test):
    X_train = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    X_test = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    y_train = train['success']
    y_test = test['success']

    pos = y_train.mean()
    ratio = (1-pos)/(pos)

    best_param_grid = {
        'subsample': .95, 
        'n_estimators': 300, 
        'min_child_weight': 9, 
        'max_depth': 5, 
        'learning_rate': .1, 
        'gamma': 3, 
    }

    xgb = XGBClassifier(
        **best_param_grid,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        scale_pos_weight=ratio
    )

    xgb.fit(X_train, y_train)

    y_preds = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_preds))
    print("AUC:", roc_auc_score(y_test, y_proba))

    test['predict'] = y_preds
    test['predict_prob'] = y_proba

    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nCONFUSION MATRIX")
    print("----------------")
    print(f"          Pred 0    Pred 1")
    print(f"Actual 0   {tn:6}    {fp:6}")
    print(f"Actual 1   {fn:6}    {tp:6}")
    print()

    importance = xgb.feature_importances_
    idx = np.argsort(importance)[::-1]   # sort descending

    print("\nXGBoost Feature Importances")
    print("---------------------------")
    for i in idx:
        print(f"{X_train.columns[i]:30} {importance[i]:.5f}")
    print()

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    file_number = random.randint(1, 9999)
    plt.savefig(f'plots/roc_{file_number}.png')
    plt.close()

    print(f'Saved plot to {file_number}.png')

    try:
        test.to_csv(f'data/processed/{file_number}_preds.csv')
        print(f'Saved PREDs to {file_number}')
    except:
        print(f'Unable to Save PREDs')

    result = permutation_importance(xgb, X_train, y_train, scoring="roc_auc")
    pi = pd.Series(result.importances_mean, index=X_train.columns).sort_values(ascending=False)
    print(pi)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
    plt.savefig(f"plots/shap_summary_{file_number}.png", dpi=300, bbox_inches='tight')
    plt.close()


        

if __name__ == "__main__":
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

    model(train, test)

    