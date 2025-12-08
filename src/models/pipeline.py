import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import FunctionTransformer

from src.models.models import MODEL_REGISTRY

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

with open('cfg/pipeline.yml') as f:
    config = yaml.safe_load(f)

COLS = config['features']['keep']
TARGET = config['features']['target']
DROP_COLS = config['features']['drop']
CAT_COLS = config['features']['categorical']
NUM_COLS = config['features']['normalize']
LOG = config['log']

def add_log_features(df):
    df = df.copy()
    df["log_ydstogo"] = np.log1p(df["ydstogo"])
    df["log_yardline_100"] = np.log1p(df["yardline_100"])
    return df

def preproc_rf():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ], remainder="passthrough")

def preproc_xgb():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ], remainder="passthrough")

def preproc_log():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ])

def preproc_svm():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ])

def build_ensemble(config_path="cfg/ensemble.yml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    estimators = [
        ("logreg", build_pipeline("logistic")),
        ("rf", build_pipeline("random_forest")),
        ("xgb", build_pipeline("xgboost"))
    ]

    final_model = LogisticRegression(
        C=cfg.get("C", 1.0),
        penalty=cfg.get("penalty", "l2"),
        solver="lbfgs",
        max_iter=2000
    )

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_model,
        stack_method="predict_proba",
        passthrough=cfg.get("passthrough", False),
        n_jobs=cfg.get("n_jobs", -1)
    )

    return Pipeline([
        ("preproc", "passthrough"),   # no preprocessing at ensemble level
        ("model", model)
    ])

    
PREPROC_REGISTRY = {
    'random_forest':preproc_rf,
    'xgboost':preproc_xgb,
    'logistic':preproc_log,
    'svm':preproc_svm
}

def build_pipeline(model_name):
    if model_name == 'ensemble':
        return build_ensemble()
    elif model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    if LOG == True:
        return Pipeline([
            ("add_logs", FunctionTransformer(add_log_features, validate=False)),
            ('preproc', PREPROC_REGISTRY[model_name]()),
            ('model', MODEL_REGISTRY[model_name]())
        ])
    else:
        return Pipeline([
            ('preproc', PREPROC_REGISTRY[model_name]()),
            ('model', MODEL_REGISTRY[model_name]())
        ])