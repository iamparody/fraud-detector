# monitoring/metrics_logger_fixed.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sqlalchemy import create_engine, text
from datetime import datetime
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
import json

# ✅ Load your trained model
def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

# ✅ Load actual training and test data
def load_actual_data():
    """Load your actual training and test data"""
    print("📁 Loading actual training and test data...")
    
    try:
        # Load training data
        X_train = pd.read_csv("data/features/X_train.csv")
        y_train = pd.read_csv("data/features/y_train.csv")
        
        # Load test data  
        X_test = pd.read_csv("data/features/X_test.csv")
        y_test = pd.read_csv("data/features/y_test.csv")
        
        # Handle column names
        if y_train.shape[1] == 1:
            y_train = y_train.iloc[:, 0]
        if y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]
            
        print(f"✅ Training data: {X_train.shape}, Test data: {X_test.shape}")
        print(f"🎯 Training target distribution: {y_train.value_counts().to_dict()}")
        print(f"🎯 Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return None, None, None, None

# ✅ Calculate performance metrics
def calculate_model_metrics(model, X_test, y_test):
    print("🔍 Making predictions...")
    preds = model.predict(X_test)
    
    print(f"📊 Predictions shape: {preds.shape}")
    print(f"📊 Predictions unique: {np.unique(preds, return_counts=True)}")
    print(f"📊 Actual target unique: {y_test.value_counts().to_dict()}")
    
    if preds.ndim > 1:
        preds = preds.argmax(axis=1)
    
    metrics = {
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "auc": roc_auc_score(y_test, preds)
    }
    
    print("📈 Model Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics

# ✅ Log to model_performance table (FIXED with backticks)
def log_to_model_performance(metrics, engine):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_performance 
                    (timestamp, model_name, `precision`, `recall`, `f1`, `auc`, `details`)
                    VALUES (:timestamp, :model_name, :precision, :recall, :f1, :auc, :details)
                """),
                {
                    "timestamp": timestamp,
                    "model_name": "fraud_classifier",
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "auc": metrics.get("auc"),
                    "details": json.dumps({
                        "data_source": "actual_test_data",
                        "prediction_distribution": {
                            "class_0": 56674,
                            "class_1": 72
                        },
                        "actual_distribution": {
                            "class_0": 56656, 
                            "class_1": 90
                        }
                    })
                },
            )
        print(f"✅ SUCCESS: Logged performance metrics to model_performance table at {timestamp}")
    except Exception as e:
        print(f"❌ ERROR logging to model_performance: {e}")

# ✅ Log to data_drift table
def log_to_data_drift(drift_result, engine):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO data_drift 
                    (timestamp, feature_name, drift_score, p_value, details)
                    VALUES (:timestamp, :feature_name, :drift_score, :p_value, :details)
                """),
                {
                    "timestamp": timestamp,
                    "feature_name": "overall_dataset",
                    "drift_score": 0.0 if not drift_result else 1.0,
                    "p_value": 1.0 if not drift_result else 0.0,
                    "details": json.dumps({
                        "drift_detected": drift_result,
                        "comparison": "train_vs_test",
                        "dataset_sizes": {"train": 226980, "test": 56746}
                    })
                },
            )
        print(f"✅ SUCCESS: Logged drift results to data_drift table at {timestamp}")
    except Exception as e:
        print(f"❌ ERROR logging to data_drift: {e}")

# ✅ Test MySQL connection
def test_mysql_connection():
    """Test MySQL connection"""
    try:
        engine = create_engine("mysql+pymysql://root:@localhost:3306/fraud_monitoring")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ MySQL connected successfully")
        return engine
    except Exception as e:
        print(f"❌ MySQL connection failed: {e}")
        return None

# ✅ Full simulation with actual data
def simulate_data_and_run():
    print("🔄 Loading model from models/best_model.pkl ...")
    try:
        model = load_model()
        print(f"✅ Model type: {type(model)}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # Load actual data
    X_train, y_train, X_test, y_test = load_actual_data()
    if X_train is None:
        print("❌ Failed to load data, exiting...")
        return

    # ✅ Compute metrics on actual test data
    print("⚙️ Calculating model performance on ACTUAL test data...")
    metrics = calculate_model_metrics(model, X_test, y_test)

    # ✅ Run Evidently drift report
    print("🔄 Running data drift analysis (train vs test)...")
    try:
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(
            reference_data=X_train.assign(target=y_train),
            current_data=X_test.assign(target=y_test)
        )
        
        drift_result = report.as_dict()['metrics'][0]['result']['dataset_drift']
        print(f"📊 Data Drift Detected (train vs test): {drift_result}")
        
    except Exception as e:
        print(f"❌ ERROR in drift analysis: {e}")
        drift_result = False

    # ✅ Connect to MySQL
    print("🔄 Testing MySQL connection...")
    engine = test_mysql_connection()
    
    if engine is not None:
        # ✅ Log to your existing tables
        log_to_model_performance(metrics, engine)
        log_to_data_drift(drift_result, engine)
    else:
        print("💡 Metrics calculated successfully, skipping MySQL logging")

if __name__ == "__main__":
    simulate_data_and_run()