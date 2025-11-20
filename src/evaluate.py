# src/evaluate.py
import argparse
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow

def read_params(params_path: str):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def main(params_path: str):
    params = read_params(params_path)
    train_params = params["train"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    model_path = "models/model.pkl"
    model = joblib.load(model_path)

    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Evaluate -> accuracy: {acc:.4f}, f1_macro: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_param("stage", "evaluate")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        report_text = classification_report(y_test, y_pred)
        with open("evaluation_report.txt", "w") as f:
            f.write(report_text)
        mlflow.log_artifact("evaluation_report.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)