import argparse
import joblib
import numpy as np
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score

from src.data.load_data import load_data
from src.models.pipeline import build_pipeline
from src.models.models import MODEL_REGISTRY
from src.utils import update_yaml_config

import warnings
warnings.filterwarnings("ignore")

def build_model_with_params(model_name, params):

    builder = MODEL_REGISTRY[model_name]
    return builder(config_path=None, **params)

def suggest_params(model_name, trial):
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None])
        }

    elif model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0)
        }

    elif model_name == "logistic":
        solver = trial.suggest_categorical(
            "solver", ["lbfgs", "liblinear", "saga"]
        )
        if solver == "liblinear":
            penalty = trial.suggest_categorical("penalty_liblinear", ["l1", "l2"])
        elif solver == "lbfgs":
            penalty = trial.suggest_categorical("penalty_lbfgs", ["l2", None])
        elif solver == "saga":
            penalty = trial.suggest_categorical("penalty_saga", ["l1", "l2", "elasticnet", None])

        if solver.startswith("lib") and penalty in ["l1", "l2"]:
            chosen_penalty = penalty
        else:
            chosen_penalty = penalty

        l1_ratio = None
        if chosen_penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "solver": solver,
            "penalty": penalty,
            "l1_ratio": l1_ratio,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "max_iter": 2000
        }

    else:
        raise ValueError(f"No tuning space defined for model {model_name}")

def objective(trial, model_name, X, y):
    params = suggest_params(model_name, trial)
    pipeline = build_pipeline(model_name)
    pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)

        f_scores.append(fbeta_score(y_valid, preds, beta=.5))

    return np.mean(f_scores)


def tune(model_name, years, data_path, n_trials=30, save_dir="models/"):
    X, y = load_data(years, data_path)

    study = optuna.create_study(direction="maximize")

    pbar = tqdm(total=n_trials, desc=f"Tuning {model_name}", colour="cyan")
    def tqdm_callback(study, trial):
        pbar.update(1)

        best_val = study.best_value
        last_val = trial.value

        pbar.set_postfix({
            "best_f1": f"{best_val:.4f}" if best_val is not None else None,
            "last_f1": f"{last_val:.4f}" if last_val is not None else None,
        })

    study.optimize(
        lambda trial: objective(trial, model_name, X, y),
        n_trials=n_trials,
        callbacks=[tqdm_callback]
    )
    pbar.close()

    print("\n🏆 Best Trial:")
    print(study.best_trial)

    best_params = suggest_params(model_name, study.best_trial)
    update_yaml_config(model_name, best_params)
    
    final_pipeline = build_pipeline(model_name)
    final_pipeline.set_params(**{f"model__{k}": v for k, v in best_params.items()})

    final_pipeline.fit(X, y)

    best_path = f"{save_dir}{model_name}_optuna.joblib"
    joblib.dump(final_pipeline, best_path)

    print(f"\n💾 Saved best tuned model to:\n{best_path}")

    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning for ML Models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    tune(args.model, args.years, args.data, args.trials)