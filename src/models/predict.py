import argparse
import joblib
from src.data.load_data import load_data

def load_pipeline(model_path):
    return joblib.load(model_path)

def predict(model_path, years, data_path):
    pipeline = load_pipeline(model_path)

    X, y = load_data(years, data_path)
    preds = pipeline.predict(X)

    probas = (
        pipeline.predict_proba(X)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

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
        default='data/clean',
        help="CSV file containing features for prediction, defaults to data/clean"
    )

    args = parser.parse_args()

    preds, probas, X, y = predict(args.model_path, args.years, args.data)

    print("\n🔮 Predictions (first 20):")
    print(preds[:20])

    if probas is not None:
        print("\n📈 Probabilities (first 20):")
        print(probas[:20])
