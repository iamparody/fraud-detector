# src/features/build_features.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_processed_data(processed_dir: str):
    """Load train and test data from processed directory."""
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train, test


def build_features(train: pd.DataFrame, test: pd.DataFrame, target_col: str = "Class"):
    """Scale numeric features and split target."""
    # Split X and y
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


def save_features(X_train, X_test, y_train, y_test, output_dir: str):
    """Save scaled data and labels to features directory."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"✅ Features saved to: {output_dir}")


def main():
    processed_dir = "data/processed"
    output_dir = "data/features"

    train, test = load_processed_data(processed_dir)
    X_train, X_test, y_train, y_test = build_features(train, test)
    save_features(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
