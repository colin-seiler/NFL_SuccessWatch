import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix



def model(train, test):
    X_train = train.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    X_test = test.drop(columns = ['nflverse_game_id', 'play_id', 'success'])
    y_train = train['success']
    y_test = test['success']

    pos = y_train.mean()
    ratio = (1-pos)/pos
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight={0:1,1:ratio})

    rf.fit(X_train, y_train)

    y_preds = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
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

    y_proba_preds = (y_proba > .5).astype(int)
    cm = confusion_matrix(y_test, y_proba_preds)
    tn, fp, fn, tp = cm.ravel()
    print("\nCONFUSION MATRIX")
    print("----------------")
    print(f"          Pred 0    Pred 1")
    print(f"Actual 0   {tn:6}    {fp:6}")
    print(f"Actual 1   {fn:6}    {tp:6}")
    print()
    print(classification_report(y_test, y_proba_preds))

    print(y_proba.mean())


if __name__ == "__main__":
    try:
        print(f"📂 Loading: rdf_train.csv")
        train = pd.read_csv('data/processed/rdf_train.csv')
        print("✅ Training DataFrame loaded!")
        print(f"📂 Loading: rdf_test.csv")
        test = pd.read_csv('data/processed/rdf_test.csv')
        print("✅ Testing DataFrame loaded!")
    except:
        print('❌ Unable to load file, please try again')
        sys.exit()

    model(train, test)

    