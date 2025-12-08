import argparse
import joblib

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from src.models.predict import predict


def load_pipeline(model_path):
    return joblib.load(model_path)

def evaluate(model_path, years, data_path):
    preds, probas, X, y = predict(model_path, years, data_path)

    results = {
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "report": classification_report(y, preds),
        "confusion_matrix": confusion_matrix(y, preds),
        "roc_auc": None
    }

    if probas is not None:
        try:
            results["roc_auc"] = roc_auc_score(y, probas)
        except ValueError:
            results["roc_auc"] = None

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model pipeline.")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--data", type=str, default='data/processed/processed_all.csv', help="CSV file containing features for prediction, defaults to data/processed/processed_all.csv")

    args = parser.parse_args()

    metrics = evaluate(args.model_path, args.years, args.data)

    print("\n📊 Evaluation Metrics")
    print("---------------------")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    print("\nClassification Report:\n", metrics["report"])
    print("\nConfusion Matrix:\n", metrics["confusion_matrix"])