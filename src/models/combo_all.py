import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

from gd34 import model_train as gd_model_train
from log34 import model_train as log_model_train
from rf34 import model_train as rdf_model_train

def load_model_csv(model_name):
    try:
        print(f"📂 Loading: rdf_train.csv")
        train = pd.read_csv(f'data/processed/{model_name}_train.csv')
        print("✅ Training DataFrame loaded!")
        print(f"📂 Loading: rdf_test.csv")
        test = pd.read_csv(f'data/processed/{model_name}_test.csv')
        print("✅ Testing DataFrame loaded!")
    except:
        print(f'❌ Unable to load {model_name} file, please try again')
        sys.exit()
    return train, test

def ensemble_test(p_log, p_gd, p_rdf, test_log):
    y_test = test_log['success']
    weights = np.linspace(0, 1, 21)

    best_auc = max(out_log['auc'], out_gd['auc'], out_rdf['auc'])
    best_w_auc = None

    best_acc = 0
    best_w_acc = None

    for w1 in tqdm(weights, desc='Running through all weight combinations!'):
        for w2 in weights:
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            combined = w1*p_gd+w2*p_rdf+w3*p_log
            auc = roc_auc_score(y_test, combined)
            acc = accuracy_score(y_test, (combined > .5).astype(int))

            if auc > best_auc:
                best_auc = auc
                best_w_auc = (w1, w2, w3)
            if acc > best_acc:
                best_acc = acc
                best_w_acc = (w1, w2, w3)

    print(best_auc, best_w_auc)
    print(best_acc, best_w_acc)

if __name__ == '__main__':
    test = True

    file_number = random.randint(1, 9999)

    train_log, test_log = load_model_csv('log')
    train_gd, test_gd = load_model_csv('gd')
    train_rdf, test_rdf = load_model_csv('rdf')

    out_log = log_model_train(train_log, test_log, file_number)
    out_gd = gd_model_train(train_gd, test_gd, file_number)
    out_rdf = rdf_model_train(train_rdf, test_rdf, file_number)

    p_log = out_log['y_proba']
    p_gd = out_gd['y_proba']
    p_rdf = out_rdf['y_proba']

    if test:
        ensemble_test(p_log, p_gd, p_rdf, test_log)

    print(f'Doc Save Key is: {file_number}')

    