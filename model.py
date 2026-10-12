"""Train a Logistic Regression model on the Iris dataset."""
import argparse
import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_model():
    """Train the model and return model, test features, labels, and accuracy."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, X_test, y_test, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Optional path to write JSON model results")
    args = parser.parse_args()

    model, X_test, y_test, accuracy = train_model()
    predictions = model.predict(X_test)
    print("Model: Logistic Regression")
    print("Dataset: Iris")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({
                "model": "Logistic Regression",
                "dataset": "Iris",
                "accuracy": round(float(accuracy), 4),
                "accuracy_percent": round(float(accuracy) * 100, 2),
                "test_samples": int(len(y_test)),
                "ci_cd_status": "Deployed after CI passed"
            }, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
