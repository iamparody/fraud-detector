# monitoring/analyze_drift_results.py
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import json

def analyze_drift_in_detail():
    print("=== DETAILED DRIFT ANALYSIS ===")
    
    # Load data
    X_ref = pd.read_csv("data/features/X_train.csv")
    y_ref = pd.read_csv("data/features/y_train.csv")
    X_cur = pd.read_csv("data/features/X_test_drifted.csv")
    y_cur = pd.read_csv("data/features/y_test_drifted.csv")
    
    if y_ref.shape[1] == 1:
        y_ref = y_ref.iloc[:, 0]
    if y_cur.shape[1] == 1:
        y_cur = y_cur.iloc[:, 0]
    
    reference = X_ref.assign(target=y_ref)
    current = X_cur.assign(target=y_cur)
    
    # Run drift analysis
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()
    
    drift_info = result['metrics'][0]['result']
    
    print(f"Dataset drift: {drift_info['dataset_drift']}")
    print(f"Drifted columns: {drift_info['number_of_drifted_columns']}/{drift_info['number_of_columns']}")
    print(f"Drift share: {drift_info.get('share_of_drifted_columns', 0):.3f}")
    print(f"Drift threshold: {drift_info.get('drift_share_threshold', 0.5)}")
    
    # Analyze individual column drift
    print("\n=== COLUMN-LEVEL DRIFT ANALYSIS ===")
    drift_by_columns = drift_info.get('drift_by_columns', {})
    
    drifted_cols = []
    non_drifted_cols = []
    
    for col_name, col_info in drift_by_columns.items():
        drift_detected = col_info.get('drift_detected', False)
        p_value = col_info.get('p_value', 'N/A')
        stat_test = col_info.get('stat_test', 'unknown')
        
        if drift_detected:
            drifted_cols.append((col_name, p_value, stat_test))
        else:
            non_drifted_cols.append((col_name, p_value, stat_test))
    
    print(f"\nDRIFTED COLUMNS ({len(drifted_cols)}):")
    for col, p_val, test in drifted_cols[:10]:  # Show first 10
        print(f"  {col}: p={p_val:.6f} ({test})")
    
    print(f"\nNON-DRIFTED COLUMNS ({len(non_drifted_cols)}):")
    for col, p_val, test in non_drifted_cols[:10]:  # Show first 10
        print(f"  {col}: p={p_val:.6f} ({test})")
    
    # Manual statistical analysis
    print("\n=== MANUAL STATISTICAL COMPARISON ===")
    numeric_cols = reference.select_dtypes(include=[np.number]).columns
    
    print("Top 10 most different features (by Z-score):")
    differences = []
    
    for col in numeric_cols:
        if col in reference.columns and col in current.columns:
            ref_mean = reference[col].mean()
            cur_mean = current[col].mean()
            ref_std = reference[col].std()
            
            if ref_std > 0:
                z_score = abs(ref_mean - cur_mean) / ref_std
                differences.append((col, z_score, ref_mean, cur_mean))
    
    # Sort by largest differences
    differences.sort(key=lambda x: x[1], reverse=True)
    
    for col, z_score, ref_mean, cur_mean in differences[:10]:
        print(f"  {col}: Z={z_score:.2f}, Ref={ref_mean:.3f} -> Cur={cur_mean:.3f}")
    
    # Check what threshold would trigger dataset drift
    total_cols = len(numeric_cols)
    drift_threshold = drift_info.get('drift_share_threshold', 0.5)
    required_drifted = int(total_cols * drift_threshold)
    
    print(f"\n=== DRIFT THRESHOLD ANALYSIS ===")
    print(f"Total columns: {total_cols}")
    print(f"Drift threshold: {drift_threshold} ({required_drifted} columns need to drift)")
    print(f"Currently drifted: {len(drifted_cols)} columns")
    print(f"Need {required_drifted - len(drifted_cols)} more columns to drift for dataset-level detection")

def check_evidently_settings():
    """Check Evidently's default drift settings"""
    print("\n=== EVIDENTLY DEFAULT SETTINGS ===")
    print("By default, Evidently uses:")
    print("  - Dataset drift threshold: 0.5 (50% of columns must drift)")
    print("  - Individual test threshold: p < 0.05")
    print("  - Uses KS test for numerical features")
    print("  - Uses chi-square test for categorical features")
    
    print("\nTo change sensitivity, you can:")
    print("  1. Lower the dataset drift threshold")
    print("  2. Use more sensitive statistical tests")
    print("  3. Increase the p-value threshold for individual tests")

if __name__ == "__main__":
    analyze_drift_in_detail()
    check_evidently_settings()