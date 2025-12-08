import argparse
import joblib
import pandas as pd
from src.data.load_data import load_data

def load_pipeline(model_path):
    return joblib.load(model_path)

def predict(model_path, years, data_path, save=False):
    pipeline = load_pipeline(model_path)

    X, y = load_data(years, data_path)
    orig_idx = X.index

    preds = pipeline.predict(X)

    probas = (
        pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    if save:
        df = pd.DataFrame({
            'orig_index': orig_idx,
            'y_true': y, 
            'y_pred': preds, 
            'y_prob': probas
            })
        df = df.sort_values("orig_index").reset_index(drop=True)
        save_path = model_path.split('.')[0].split('/')[-1]
        df.to_csv(f'outputs/predictions/{save_path}_vals.csv')

    return preds, probas, X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained model pipeline.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved pipeline (e.g., models/random_forest_pipeline.joblib)"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="List of years for use in test set"
    )

    parser.add_argument(
        "--data",
        type=str,
        default='data/processed/processed_all.csv',
        help="CSV file containing features for prediction, defaults to data/processed/processed_all.csv"
    )

    parser.add_argument(
        "--save",
        action='store_true',
        help='Save predictions and ground truth to files', 
        required=False
    )

    args = parser.parse_args()

    preds, probas, X, y = predict(args.model_path, args.years, args.data, args.save)