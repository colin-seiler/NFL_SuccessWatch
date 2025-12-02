import numpy as np
import pandas as pd
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix



def model(train, test):
    X_train = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    X_test = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    y_train = train['success']
    y_test = test['success']

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist'
    )

    param_grid = {
        'n_estimators':np.arange(100, 1000, 100),
        'max_depth':np.arange(2,20),
        'learning_rate':np.linspace(.005, .3, 25),
        'gamma': np.linspace(0, 5, 40),
        'subsample': np.linspace(0.8, 1.0, 8),
        'colsample_bytree': np.linspace(0.5, 1.0, 10),
        'min_child_weight': np.arange(1, 10)
    }

    best_param_grid = {
        'subsample': np.float64(0.8), 
        'n_estimators': np.int64(300), 
        'min_child_weight': np.int64(9), 
        'max_depth': np.int64(3), 
        'learning_rate': np.float64(0.041874999999999996), 
        'gamma': np.float64(3.0769230769230766), 
        'colsample_bytree': np.float64(0.9444444444444444)
    }

    rand = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=100,      # number of trials
        scoring='roc_auc',
        cv=5,
        verbose=2,
        random_state=42
    )

    rand.fit(X_train, y_train)

    print("BEST PARAMS:", rand.best_params_)
    print("BEST SCORE:", rand.best_score_)

    best_model = rand.best_estimator_

    y_preds = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_preds))
    print("AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nCONFUSION MATRIX")
    print("----------------")
    print(f"          Pred 0    Pred 1")
    print(f"Actual 0   {tn:6}    {fp:6}")
    print(f"Actual 1   {fn:6}    {tp:6}")
    print()

    importance = best_model.feature_importances_
    idx = np.argsort(importance)[::-1]   # sort descending

    print("\nXGBoost Feature Importances")
    print("---------------------------")
    for i in idx:
        print(f"{X_train.columns[i]:30} {importance[i]:.5f}")
    print()

    cv_results = pd.DataFrame(rand.cv_results_)
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)

    print(cv_results.head())


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

    