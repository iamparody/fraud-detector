# monitoring/advanced_drift_pipeline.py
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from monitoring.configs.db_config import get_engine
from monitoring.postgres.init_tables import DataDrift, TargetDrift
import json

def run_advanced_drift_detection():
    print("=== ADVANCED DRIFT DETECTION ===")
    
    # Load data
    X_ref = pd.read_csv("data/features/X_train.csv")
    y_ref = pd.read_csv("data/features/y_train.csv")
    X_cur = pd.read_csv("data/features/X_test_drifted.csv")
    y_cur = pd.read_csv("data/features/y_test_drifted.csv")
    
    if y_ref.shape[1] == 1:
        y_ref = y_ref.iloc[:, 0]
    if y_cur.shape[1] == 1:
        y_cur = y_cur.iloc[:, 0]
    
    # Use smaller, more comparable samples for better sensitivity
    sample_size = min(10000, len(X_ref), len(X_cur))
    
    reference_sample = X_ref.sample(sample_size, random_state=42).assign(
        target=y_ref.sample(sample_size, random_state=42)
    )
    current_sample = X_cur.sample(sample_size, random_state=42).assign(
        target=y_cur.sample(sample_size, random_state=42)
    )
    
    print(f"Using sample size: {sample_size}")
    print(f"Reference sample: {reference_sample.shape}")
    print(f"Current sample: {current_sample.shape}")
    
    # Run drift detection on samples
    drift_metric = DatasetDriftMetric()
    report = Report(metrics=[drift_metric])
    report.run(reference_data=reference_sample, current_data=current_sample)
    result = report.as_dict()
    
    drift_info = result['metrics'][0]['result']
    dataset_drift = drift_info['dataset_drift']
    drifted_cols = drift_info['number_of_drifted_columns']
    total_cols = drift_info['number_of_columns']
    drift_share = drifted_cols / total_cols if total_cols > 0 else 0
    
    print(f"Dataset drift: {dataset_drift}")
    print(f"Drifted columns: {drifted_cols}/{total_cols}")
    print(f"Drift share: {drift_share:.3f}")
    
    # Apply multiple thresholds for sensitivity analysis
    thresholds = [0.5, 0.4, 0.3, 0.2]
    drift_at_thresholds = {}
    
    for threshold in thresholds:
        drift_at_thresholds[threshold] = drift_share >= threshold
        print(f"Threshold {threshold:.0%}: {'DRIFT' if drift_share >= threshold else 'no drift'}")
    
    # Log detailed results
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        drift_by_columns = drift_info.get('drift_by_columns', {})
        
        # Log column-level drift
        for col_name, col_info in drift_by_columns.items():
            drift_detected = col_info.get('drift_detected', False)
            p_value = col_info.get('p_value')
            stat_test = col_info.get('stat_test', 'unknown')
            
            drift_row = DataDrift(
                timestamp=datetime.utcnow(),
                feature_name=col_name,
                drift_score=1.0 if drift_detected else 0.0,
                p_value=p_value,
                details=json.dumps({
                    "stat_test": stat_test,
                    "drift_detected": drift_detected,
                    "sample_based": True,
                    "sample_size": sample_size
                })
            )
            session.add(drift_row)
        
        # Log comprehensive summary
        summary_row = TargetDrift(
            timestamp=datetime.utcnow(),
            drift_score=drift_share,
            p_value=None,
            details=json.dumps({
                "dataset_drift_default": dataset_drift,
                "drifted_columns": drifted_cols,
                "total_columns": total_cols,
                "drift_share": drift_share,
                "sample_size": sample_size,
                "threshold_analysis": drift_at_thresholds,
                "advanced_detection": True
            })
        )
        session.add(summary_row)
        
        session.commit()
        print("SUCCESS: Advanced drift results stored.")
        
        # Alert based on most sensitive threshold
        if drift_share >= 0.3:
            print("🚨 ALERT: Significant drift detected (30% threshold)")
        elif drift_share >= 0.2:
            print("⚠️  WARNING: Moderate drift detected (20% threshold)")
        else:
            print("✅ Minimal or no drift detected")
            
    except Exception as e:
        print(f"ERROR: {e}")
        session.rollback()
    finally:
        session.close()

def manual_drift_analysis():
    """Additional manual analysis for verification"""
    print("\n=== MANUAL DRIFT VERIFICATION ===")
    
    X_ref = pd.read_csv("data/features/X_train.csv")
    X_cur = pd.read_csv("data/features/X_test_drifted.csv")
    
    # Calculate mean differences
    numeric_cols = X_ref.select_dtypes(include=[np.number]).columns
    
    significant_drifts = 0
    print("Top 10 features with largest differences:")
    
    differences = []
    for col in numeric_cols:
        ref_mean = X_ref[col].mean()
        cur_mean = X_cur[col].mean()
        ref_std = X_ref[col].std()
        
        if ref_std > 0:
            z_score = abs(ref_mean - cur_mean) / ref_std
            differences.append((col, z_score, ref_mean, cur_mean))
    
    # Sort by largest differences
    differences.sort(key=lambda x: x[1], reverse=True)
    
    for col, z_score, ref_mean, cur_mean in differences[:10]:
        status = "SIGNIFICANT" if z_score > 1.0 else "moderate" if z_score > 0.5 else "minor"
        print(f"  {col}: Z={z_score:.2f} ({status})")
        if z_score > 1.0:
            significant_drifts += 1
    
    print(f"\nSignificant drifts (Z > 1.0): {significant_drifts}/{len(numeric_cols)}")

if __name__ == "__main__":
    run_advanced_drift_detection()
    manual_drift_analysis()