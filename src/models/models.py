import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def build_rf(config_path="cfg/random_forest.yml"):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return RandomForestClassifier(**params)

def build_xgb(config_path="cfg/xgboost.yml"):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return XGBClassifier(**params)

def build_log(config_path="cfg/logreg.yml"):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    return LogisticRegression(**params)

def build_svm(config_path="cfg/svm.yml"):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return SVC(**params)

MODEL_REGISTRY = {
    'random_forest':build_rf,
    'xgboost':build_xgb,
    'logistic':build_log,
    'svm':build_svm
}