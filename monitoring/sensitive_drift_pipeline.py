# monitoring/sensitive_drift_pipeline.py
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from monitoring.configs.db_config import get_engine
from monitoring.postgres.init_tables import DataDrift, TargetDrift
import json

def run_sensitive_drift_detection():
    print("=== SENSITIVE DRIFT DETECTION ===")
    
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
    
    # Create drift metric with custom threshold in the options
    drift_metric = DatasetDriftMetric()
    
    report = Report(metrics=[drift_metric])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()
    
    drift_info = result['metrics'][0]['result']
    dataset_drift = drift_info['dataset_drift']
    drifted_cols = drift_info['number_of_drifted_columns']
    total_cols = drift_info['number_of_columns']
    drift_share = drifted_cols / total_cols if total_cols > 0 else 0
    
    print(f"Dataset drift: {dataset_drift}")
    print(f"Drifted columns: {drifted_cols}/{total_cols}")
    print(f"Drift share: {drift_share:.3f}")
    
    # Apply custom threshold logic in code (since we can't set it in the metric)
    CUSTOM_THRESHOLD = 0.3  # 30% instead of default 50%
    custom_dataset_drift = drift_share >= CUSTOM_THRESHOLD
    
    print(f"Custom threshold ({CUSTOM_THRESHOLD:.0%}): {custom_dataset_drift}")
    
    # Log to database with both original and custom detection
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        drift_by_columns = drift_info.get('drift_by_columns', {})
        
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
                    "sensitive_analysis": True,
                    "custom_threshold_applied": CUSTOM_THRESHOLD
                })
            )
            session.add(drift_row)
        
        # Log both original and custom drift detection
        summary_row = TargetDrift(
            timestamp=datetime.utcnow(),
            drift_score=drift_share,
            p_value=None,
            details=json.dumps({
                "dataset_drift_original": dataset_drift,
                "dataset_drift_custom": custom_dataset_drift,
                "drifted_columns": drifted_cols,
                "total_columns": total_cols,
                "drift_share": drift_share,
                "default_threshold": 0.5,
                "custom_threshold": CUSTOM_THRESHOLD,
                "sensitive_detection": True
            })
        )
        session.add(summary_row)
        
        session.commit()
        print("SUCCESS: Sensitive drift results stored.")
        
        if custom_dataset_drift:
            print("🚨 ALERT: Dataset drift detected with custom threshold!")
        else:
            print("✅ No dataset drift with custom threshold.")
            
    except Exception as e:
        print(f"ERROR: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    run_sensitive_drift_detection()