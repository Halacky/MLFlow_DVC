import os
import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def read_params(params_path: str):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def main(params_path: str):
    params = read_params(params_path)
    prepare_params = params["prepare"]

    input_path = "data/raw/iris.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=prepare_params["test_size"],
        random_state=prepare_params["random_state"],
        stratify=y,
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df = X_train.copy()
    train_df["label"] = y_train
    test_df = X_test.copy()
    test_df["label"] = y_test

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train saved to {train_path}, test saved to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.params)