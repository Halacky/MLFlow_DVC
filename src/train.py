import os
import argparse
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def read_params(params_path: str):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def main(params_path: str):
    params = read_params(params_path)
    train_params = params["train"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    model_type = train_params["model_type"]
    random_state = train_params["random_state"]

    if model_type == "logreg":
        C = train_params["logreg"]["C"]
        max_iter = train_params["logreg"]["max_iter"]
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "rf":
        n_estimators = train_params["rf"]["n_estimators"]
        max_depth = train_params["rf"]["max_depth"]
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)

        if model_type == "logreg":
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
        else:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)