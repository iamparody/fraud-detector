# monitoring/debug_drift.py
import pandas as pd
import numpy as np

def debug_drift_data():
    print("=== DRIFT DATA DEBUG ===")
    
    # Load all datasets
    X_train = pd.read_csv("data/features/X_train.csv")
    X_test = pd.read_csv("data/features/X_test.csv") 
    X_drifted = pd.read_csv("data/features/X_test_drifted.csv")
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_drifted shape: {X_drifted.shape}")
    
    # Check if drifted data is actually different
    print("\n=== DATA COMPARISON ===")
    
    # Compare basic statistics
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()[:5]  # First 5 numeric cols
    
    for col in numeric_cols:
        if col in X_train.columns and col in X_drifted.columns:
            train_mean = X_train[col].mean()
            drifted_mean = X_drifted[col].mean()
            diff = abs(train_mean - drifted_mean)
            
            print(f"{col}:")
            print(f"  Train mean: {train_mean:.4f}")
            print(f"  Drifted mean: {drifted_mean:.4f}")
            print(f"  Difference: {diff:.4f}")
            print(f"  Significant diff: {'YES' if diff > 0.1 else 'NO'}")
            print()

def check_drift_generation():
    """Verify the drift generation worked"""
    print("=== CHECKING DRIFT GENERATION ===")
    
    # Check if drifted files exist and have content
    try:
        X_drifted = pd.read_csv("data/features/X_test_drifted.csv")
        y_drifted = pd.read_csv("data/features/y_test_drifted.csv")
        
        print(f"Drifted files exist:")
        print(f"  X_test_drifted.csv: {X_drifted.shape}")
        print(f"  y_test_drifted.csv: {y_drifted.shape}")
        
        # Check if drifted data is different from original test
        X_test = pd.read_csv("data/features/X_test.csv")
        are_different = not X_drifted.equals(X_test)
        print(f"Drifted data is different from original: {are_different}")
        
        if not are_different:
            print("❌ PROBLEM: Drifted data is identical to original test data!")
            print("The drift generation might have failed.")
            
    except FileNotFoundError as e:
        print(f"❌ Drifted files not found: {e}")

if __name__ == "__main__":
    debug_drift_data()
    check_drift_generation()