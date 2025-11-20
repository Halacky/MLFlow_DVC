import os
import pandas as pd
from sklearn.datasets import load_iris

def main():
    os.makedirs("data/raw", exist_ok=True)

    iris = load_iris(as_frame=True)
    df = iris.frame
    
    df.rename(columns={"target": "label"}, inplace=True)

    output_path = "data/raw/iris.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved iris dataset to {output_path}")

if __name__ == "__main__":
    main()