import argparse
import joblib
import os

from src.models.pipeline import build_pipeline
from src.data.load_data import load_data

def train(model_name, years, save_dir = 'models/'):
    print(f"{f'👨‍💻 Training {model_name} on years {years}':=^100}")

    X_train, y_train = load_data(years)
    print(f"📂 Loaded {len(X_train)} training rows")

    print("👷‍♂️ Building pipeline...")
    pipeline = build_pipeline(model_name)

    print("🎶 Fitting model...")
    pipeline.fit(X_train, y_train)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_pipeline.joblib")
    joblib.dump(pipeline, save_path)

    print(f"\n💽 Saved trained pipeline to:\n  {save_path}")
    print(f"{'':=^100}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an NFL ML model.")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (random_forest, xgboost, logistic, ensemble)"
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="List of seasons to train on (e.g. --years 2016 2017 2018 2019)"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Where to save trained pipeline"
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        years=args.years,
        save_dir=args.save_dir
    )