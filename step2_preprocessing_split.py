"""Step 2: preprocess, split, and save the raw test set."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "cleaned_credit_data.csv"
TEST_OUTPUT_PATH = Path(__file__).resolve().parent / "test_data.csv"
TARGET_COLUMN = "default payment next month"


def main():
    # Use the cleaned dataset from Step 1 so the split is based on cleaned, validated data.
    df = pd.read_csv(DATA_PATH)

    # Keep the target separate from the feature set. This is the standard setup for
    # supervised classification, and it makes the 80/20 split interpretable.
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Categorical variables are encoded in the model pipeline later, but this split is done
    # before fitting so that the test set remains untouched and truly represents unseen data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Save the raw test split in a format that can be uploaded into the Streamlit app.
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Target distribution in training set:")
    print(y_train.value_counts(normalize=True).to_string())
    print(f"Saved raw test split to: {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
