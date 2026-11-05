import os
import sys

import pandas as pd

RAW_PATH = "data/raw/dataset.csv"


def main():
    if not os.path.exists(RAW_PATH):
        print(f"Dataset not found at {RAW_PATH}")
        print("Download it from Kaggle (blastchar/telco-customer-churn)")
        print("and save it as data/raw/dataset.csv")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    print("Data collected successfully")
    print("Rows, columns:", df.shape)
    print("Churn counts:")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    main()