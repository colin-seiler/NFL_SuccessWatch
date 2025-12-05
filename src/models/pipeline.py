import yaml
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier

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

def build_ensemble(config_path="configs/ensemble.yaml"):
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    
    estimators = [
        ("logreg", build_pipeline('logistic')),
        ("rf", build_pipeline('random_forest')),
        ("xgb", build_pipeline('xgboost'))
    ]

    ensemble = VotingClassifier(
        estimators=estimators,
        voting=params.get("voting", "soft"),
        weights=params.get("weights"),
        n_jobs=params.get("n_jobs", -1)
    )

    return ensemble

    
PREPROC_REGISTRY = {
    'random_forest':preproc_rf,
    'xgboost':preproc_xgb,
    'logistic':preproc_log,
}

def build_pipeline(model_name):
    if model_name == 'ensemble':
        return build_ensemble()
    elif model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    return Pipeline([
        ('preproc', PREPROC_REGISTRY[model_name]()),
        ('model', MODEL_REGISTRY[model_name]())
    ])