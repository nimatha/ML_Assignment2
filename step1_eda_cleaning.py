"""Step 1: EDA and cleaning for the credit default dataset."""

from pathlib import Path

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/350/data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_credit_data.csv"
TARGET_COLUMN = "default payment next month"


def load_dataset():
    """Load the raw UCI dataset and rename columns to the credit-default schema."""
    df = pd.read_csv(DATA_URL)

    rename_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": TARGET_COLUMN,
    }
    df = df.rename(columns=rename_map)

    # The assignment specifically says EDUCATION values 0, 5, and 6 are not valid,
    # and MARRIAGE value 0 is invalid. We correct these to the nearest valid label so
    # the dataset matches the original UCI specification without dropping rows.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # The ID column is a row identifier and not a predictive feature, so it should not be
    # included in model training and evaluation.
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    return df


def print_eda_summary(df):
    """Display key exploratory checks for review before modeling."""
    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Class Balance ===")
    print(df[TARGET_COLUMN].value_counts().to_string())
    print("\nNormalized class proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).to_string())

    print("\n=== Null Values ===")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

    print("\n=== Duplicate Rows ===")
    print(f"Duplicate count: {df.duplicated().sum()}")

    print("\n=== EDUCATION values after repair ===")
    print(df["EDUCATION"].value_counts().sort_index().to_string())

    print("\n=== MARRIAGE values after repair ===")
    print(df["MARRIAGE"].value_counts().sort_index().to_string())


def main():
    df = load_dataset()
    print_eda_summary(df)

    # Null and duplicate checks are essential because missing values or repeated records can
    # distort model performance and training stability.
    null_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    if null_count > 0:
        print(f"\nFound {null_count} null values and will handle them during preprocessing.")
    else:
        print("\nNo null values found.")

    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows; removing them to avoid leakage.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicate rows found.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
