import argparse
import joblib
import numpy as np
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score

from src.data.load_data import load_data
from src.models.pipeline import build_pipeline
from src.models.models import MODEL_REGISTRY
from src.utils import update_yaml_config

import warnings
warnings.filterwarnings("ignore")

def suggest_params(model_name, trial):
    if model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
        if params["bootstrap"]:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
        else:
            params["max_samples"] = None

        return params

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
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])

        pen_lbfgs     = trial.suggest_categorical("pen_lbfgs",     ["l2", None])
        pen_liblinear = trial.suggest_categorical("pen_liblinear", ["l1", "l2"])
        pen_saga      = trial.suggest_categorical("pen_saga",      ["l1", "l2", "elasticnet", None])

        if solver == "lbfgs":
            penalty = pen_lbfgs
        elif solver == "liblinear":
            penalty = pen_liblinear
        else:
            penalty = pen_saga

        if solver == "saga" and penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:
            l1_ratio = None

        return {
            "C": trial.suggest_float("C", 0.001, 10.0, log=True),
            "solver": solver,
            "penalty": penalty,
            "l1_ratio": l1_ratio,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            "max_iter": 2000
        }
    
    elif model_name == 'svm':
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
        if kernel in ["rbf", "poly", "sigmoid"]:
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        else:
            gamma = "scale"
        if kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
        else:
            degree = 3 
        if kernel in ["poly", "sigmoid"]:
            coef0 = trial.suggest_float("coef0", -1.0, 1.0)
        else:
            coef0 = 0.0

        return {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": gamma,
            "degree": degree,
            "coef0": coef0,
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "probability": True,
            "max_iter": -1
        }
    
    elif model_name == "ensemble":
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "n_jobs": -1
        }

    else:
        raise ValueError(f"No tuning space defined for model {model_name}")

def objective(trial, model_name, X, y):
    params = suggest_params(model_name, trial)
    pipeline = build_pipeline(model_name)
    if model_name == "ensemble":
        pipeline.set_params(
            model__final_estimator__C=params["C"]
        )
    else:
        pipeline.set_params(**{f"model__{k}": v for k, v in params.items()})

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_valid)
        proba = pipeline.predict_proba(X_valid)[:, 1]


        eval_scores.append(roc_auc_score(y_valid, proba))

    return np.mean(eval_scores)


def tune(model_name, years, data_path, n_trials=30, save_dir="models/"):
    X, y = load_data(years, data_path)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

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

    print("\n🏆 Best Trial Parameters Found!")

    best_params = study.best_trial.params
    allowed_keys = {"C", "solver", "penalty", "l1_ratio", "class_weight", "max_iter"}
    filtered_params = {k: v for k, v in best_params.items() if k in allowed_keys}
    update_yaml_config(model_name, filtered_params)
    
    final_pipeline = build_pipeline(model_name)
    if model_name == "ensemble":
        final_pipeline.set_params(
            model__final_estimator__C=best_params["C"]
        )
    else:
        final_pipeline.set_params(**{f"model__{k}": v for k, v in filtered_params.items()})

    final_pipeline.fit(X, y)

    best_path = f"{save_dir}{model_name}_optuna.joblib"
    joblib.dump(final_pipeline, best_path)

    print(f"\n💾 Saved best tuned model to:\n{best_path}")

    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning for ML Models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--data", type=str, default='data/processed/processed_all.csv', help="CSV file containing features for prediction, defaults to data/processed/processed_all.csv")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    tune(args.model, args.years, args.data, args.trials)