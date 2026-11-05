import os

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/dataset.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def clean_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 1. remove the ID column
    df = df.drop(columns=["customerID"], errors="ignore")

    # 2. remove extra spaces in text columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # 3. TotalCharges is text with blanks, so convert it to a number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    estimate = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges"] = df["TotalCharges"].fillna(estimate)

    # 4. "No internet service" and "No phone service" mean "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # 5. Churn: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    # 6. remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main():
    df = pd.read_csv(RAW_PATH)
    print("Raw data:", df.shape)

    df = clean_data(df)
    print("Clean data:", df.shape)

    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Churn"]
    )

    os.makedirs("data/processed", exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print("Saved train:", train.shape, "and test:", test.shape)


if __name__ == "__main__":
    main()