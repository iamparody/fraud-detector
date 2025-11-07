# monitoring/generate_extreme_drift.py
import pandas as pd
import numpy as np
from pathlib import Path

def generate_extreme_drift():
    """Generate extremely drifted data that WILL be detected"""
    print("=== GENERATING EXTREME DRIFT ===")
    
    # Load original test data
    X_test = pd.read_csv("data/features/X_test.csv")
    y_test = pd.read_csv("data/features/y_test.csv")
    
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]
    
    # Create extreme drift
    X_drifted = X_test.copy()
    
    # Apply massive shifts to multiple features
    np.random.seed(42)
    
    # Select 10 random features and shift them dramatically
    features_to_shift = np.random.choice(X_test.columns, 10, replace=False)
    print(f"Applying extreme drift to features: {list(features_to_shift)}")
    
    for feature in features_to_shift:
        # Massive mean shift + high noise
        shift = np.random.choice([-8.0, 8.0])  # Large shift in either direction
        X_drifted[feature] = X_drifted[feature] + shift + np.random.normal(0, 3.0, len(X_drifted))
    
    # Also create label shift
    y_drifted = y_test.copy()
    # Increase fraud rate significantly
    fraud_indices = y_drifted[y_drifted == 1].index
    additional_frauds = int(0.04 * len(y_drifted))  # 4% more frauds
    new_fraud_indices = np.random.choice(y_drifted[y_drifted == 0].index, additional_frauds, replace=False)
    y_drifted.loc[new_fraud_indices] = 1
    
    # Save files
    X_drifted.to_csv("data/features/X_test_drifted.csv", index=False)
    y_drifted.to_csv("data/features/y_test_drifted.csv", index=False)
    
    # Create combined file
    full_drifted = X_drifted.copy()
    full_drifted['target'] = y_drifted
    full_drifted.to_csv("data/processed/drifted_test.csv", index=False)
    
    print("✅ Extreme drift generated!")
    print(f"Original fraud rate: {(y_test == 1).mean():.4f}")
    print(f"Drifted fraud rate: {(y_drifted == 1).mean():.4f}")
    
    # Show some feature differences
    print("\nFeature differences (first 5 features):")
    for feature in features_to_shift[:5]:
        orig_mean = X_test[feature].mean()
        drifted_mean = X_drifted[feature].mean()
        diff = abs(orig_mean - drifted_mean)
        print(f"  {feature}: {orig_mean:.2f} -> {drifted_mean:.2f} (diff: {diff:.2f})")

if __name__ == "__main__":
    generate_extreme_drift()