# orchestration/unified_pipeline_clean.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier, log_evaluation
from mlflow.models import infer_signature
import platform
import getpass
import subprocess
import json
from datetime import datetime
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric
from sqlalchemy import create_engine, text
import time
import logging
from typing import Dict, Any, Tuple
import os

# ========================
# CONFIGURATION
# ========================
class Config:
    DATA_PATH = Path("data/features")
    MODEL_PATH = Path("models/best_model.pkl")
    ARTIFACTS_DIR = Path("artifacts")
    MLRUNS_DIR = Path("mlruns")
    MONITORING_DIR = Path("monitoring")
    DB_ENGINE = "mysql+pymysql://root:@localhost:3306/fraud_monitoring"
    
    # Create directories
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    
    # MLflow setup
    TRACKING_URI = f"file:///{Path.cwd()/'mlruns'}".replace("\\", "/")

# Setup logging with ASCII-only characters for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestration/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load preprocessed training and test data"""
    logger.info(f"Loading data from: {Config.DATA_PATH.resolve()}")
    
    try:
        X_train = pd.read_csv(Config.DATA_PATH / "X_train.csv")
        y_train = pd.read_csv(Config.DATA_PATH / "y_train.csv").values.ravel()
        X_test = pd.read_csv(Config.DATA_PATH / "X_test.csv")
        y_test = pd.read_csv(Config.DATA_PATH / "y_test.csv")
        
        # Handle column names for y_test
        if y_test.shape[1] == 1:
            y_test = y_test.iloc[:, 0]
        
        logger.info(f"SUCCESS - Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"SUCCESS - Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"ERROR loading data: {e}")
        raise

def train_model_cv(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[LGBMClassifier, float, str]:
    """Train LightGBM with 5-fold cross-validation"""
    
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced",
        "verbosity": -1,
        "force_col_wise": True,
    }
    
    logger.info("Starting 5-fold Cross-Validation...")
    
    # Setup MLflow
    mlflow.set_tracking_uri(Config.TRACKING_URI)
    mlflow.set_experiment("Fraud_Detection_Experiment")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    fold_scores = []
    
    # Suppress MLflow environment variable warnings
    os.environ['MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING'] = 'false'
    
    with mlflow.start_run(run_name="LightGBM_CV_Training") as run:
        # System tags
        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("os", platform.platform())
        mlflow.set_tag("python_version", platform.python_version())
        mlflow.set_tag("framework", "LightGBM")
        
        try:
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            mlflow.set_tag("git_commit", commit_hash)
        except Exception:
            mlflow.set_tag("git_commit", "N/A")
        
        logger.info(f"Run ID: {run.info.run_id}")
        mlflow.log_params(params)
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
            logger.info(f"Processing Fold {fold}/5")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[log_evaluation(period=100)],
            )
            
            val_preds = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_preds)
            oof_preds[val_idx] = val_preds
            fold_scores.append(fold_auc)
            
            logger.info(f"SUCCESS - Fold {fold} AUC: {fold_auc:.5f}")
            mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)
        
        # Overall metrics
        overall_auc = roc_auc_score(y_train, oof_preds)
        logger.info(f"SUCCESS - Overall CV AUC: {overall_auc:.5f}")
        mlflow.log_metric("overall_cv_auc", overall_auc)
        
        # Feature importance plot
        fig, ax = plt.subplots(figsize=(10, 8))
        import lightgbm as lgb
        lgb.plot_importance(
            model,
            max_num_features=20,
            importance_type="gain",
            ax=ax,
            title="Top 20 Features (Gain)"
        )
        plt.tight_layout()
        imp_path = Config.ARTIFACTS_DIR / "feature_importance.png"
        plt.savefig(imp_path)
        plt.close()
        mlflow.log_artifact(str(imp_path), artifact_path="plots")
        
        # Log model with signature (using updated MLflow API)
        input_example = X_train.head(5)
        pred_example = model.predict_proba(input_example)
        signature = infer_signature(input_example, pred_example)
        
        # Use updated MLflow logging without deprecated parameters
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="fraud_model",
            signature=signature,
            input_example=input_example
        )
        
        # Save locally
        joblib.dump(model, Config.MODEL_PATH)
        mlflow.log_artifact(str(Config.MODEL_PATH), artifact_path="local_model")
        
        # Register model
        model_uri = f"runs:/{run.info.run_id}/fraud_model"
        try:
            registered = mlflow.register_model(model_uri, "FraudDetector")
            logger.info(f"SUCCESS - Registered: FraudDetector v{registered.version}")
        except Exception as e:
            logger.warning(f"Model registration warning: {e}")
        
        return model, overall_auc, run.info.run_id

def evaluate_model(model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model on test data and calculate metrics"""
    logger.info("Evaluating model on test data...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"SUCCESS - Test ROC-AUC: {auc:.5f}")
    logger.info(f"SUCCESS - Precision: {precision:.5f}")
    logger.info(f"SUCCESS - Recall: {recall:.5f}")
    logger.info(f"SUCCESS - F1-Score: {f1:.5f}")
    
    metrics = {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    return metrics

def run_drift_analysis(X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> bool:
    """Run data drift analysis between train and test data"""
    logger.info("Running Data Drift Analysis...")
    
    try:
        # Use smaller sample for faster drift analysis on large datasets
        sample_size = min(5000, len(X_train), len(X_test))
        
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        y_train_sample = y_train[X_train_sample.index]
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test[X_test_sample.index]
        
        logger.info(f"Using sample size {sample_size} for drift analysis")
        
        report = Report(metrics=[DatasetDriftMetric()])
        report.run(
            reference_data=X_train_sample.assign(target=y_train_sample),
            current_data=X_test_sample.assign(target=y_test_sample)
        )
        
        drift_result = report.as_dict()['metrics'][0]['result']['dataset_drift']
        logger.info(f"Data Drift Detected (train vs test): {drift_result}")
        
        return drift_result
        
    except Exception as e:
        logger.error(f"ERROR in drift analysis: {e}")
        return False

def log_to_monitoring_db(metrics: Dict[str, float], drift_result: bool, model_run_id: str) -> bool:
    """Log metrics and drift results to MySQL database"""
    logger.info("Logging to Monitoring Database...")
    
    try:
        engine = create_engine(Config.DB_ENGINE)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("SUCCESS - Database connected")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to model_performance table
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_performance 
                    (timestamp, model_name, `precision`, recall, f1, auc, details)
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
                        "mlflow_run_id": model_run_id,
                        "data_source": "test_data",
                        "pipeline_version": "cleaned_v1"
                    })
                },
            )
        logger.info("SUCCESS - Logged to model_performance table")
        
        # Log to data_drift table
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
                        "mlflow_run_id": model_run_id,
                        "sample_size_used": min(5000, len(X_train), len(X_test)) if 'X_train' in locals() else "unknown"
                    })
                },
            )
        logger.info("SUCCESS - Logged to data_drift table")
        
        return True
        
    except Exception as e:
        logger.error(f"ERROR logging to database: {e}")
        return False

def generate_monitoring_report(metrics: Dict[str, float], drift_result: bool, 
                             model_run_id: str, execution_time: float) -> Path:
    """Generate a comprehensive monitoring report"""
    logger.info("Generating Monitoring Report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_run_id": model_run_id,
        "performance_metrics": metrics,
        "data_drift": {
            "detected": drift_result,
            "analysis_type": "train_vs_test"
        },
        "execution_time_seconds": execution_time,
        "pipeline_status": "completed",
        "next_steps": [
            "Check MLflow UI for detailed experiment tracking",
            "Monitor database tables for ongoing metrics",
            "Set up Grafana dashboard for visualization",
            "Schedule regular monitoring runs"
        ]
    }
    
    # Save report to file
    report_path = Config.MONITORING_DIR / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"SUCCESS - Monitoring report saved: {report_path}")
    return report_path

def unified_fraud_detection_pipeline() -> Dict[str, Any]:
    """
    End-to-end unified fraud detection pipeline combining:
    - MLflow for experiment tracking
    - Evidently for data drift detection  
    - MySQL for monitoring storage
    """
    
    logger.info("=" * 70)
    logger.info("UNIFIED FRAUD DETECTION PIPELINE")
    logger.info("MLflow + Evidently + MySQL + Grafana")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        X_train, y_train, X_test, y_test = load_data()
        
        # Step 2: Train with CV
        model, cv_auc, run_id = train_model_cv(X_train, y_train)
        
        # Step 3: Evaluate on test data
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 4: Run drift analysis
        drift_result = run_drift_analysis(X_train, y_train, X_test, y_test)
        
        # Step 5: Log to monitoring database
        db_success = log_to_monitoring_db(metrics, drift_result, run_id)
        
        execution_time = time.time() - start_time
        
        # Step 6: Generate monitoring report
        report_path = generate_monitoring_report(metrics, drift_result, run_id, execution_time)
        
        logger.info("=" * 70)
        logger.info("SUCCESS - UNIFIED PIPELINE COMPLETED!")
        logger.info("=" * 70)
        logger.info("MODEL PERFORMANCE:")
        logger.info(f"  - CV AUC: {cv_auc:.5f}")
        logger.info(f"  - Test AUC: {metrics['auc']:.5f}")
        logger.info(f"  - Precision: {metrics['precision']:.5f}")
        logger.info(f"  - Recall: {metrics['recall']:.5f}")
        logger.info(f"  - F1-Score: {metrics['f1']:.5f}")
        logger.info(f"DATA DRIFT: {'Detected' if drift_result else 'Not Detected'}")
        logger.info(f"DATABASE LOGGING: {'Success' if db_success else 'Failed'}")
        logger.info(f"MLFLOW RUN ID: {run_id}")
        logger.info(f"MONITORING REPORT: {report_path}")
        logger.info(f"EXECUTION TIME: {execution_time:.2f} seconds")
        logger.info(f"MLFLOW UI: mlflow ui --port 5000")
        logger.info("=" * 70)
        
        return {
            "success": True,
            "model_path": str(Config.MODEL_PATH),
            "cv_auc": cv_auc,
            "test_metrics": metrics,
            "drift_detected": drift_result,
            "mlflow_run_id": run_id,
            "db_logging_success": db_success,
            "execution_time": execution_time,
            "report_path": str(report_path)
        }
        
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def monitoring_only_pipeline() -> Dict[str, Any]:
    """
    Run only the monitoring part (useful for scheduled monitoring)
    """
    logger.info("Running Monitoring-Only Pipeline...")
    start_time = time.time()
    
    try:
        # Load data
        X_train, y_train, X_test, y_test = load_data()
        
        # Load latest model
        model = joblib.load(Config.MODEL_PATH)
        
        # Run monitoring tasks
        metrics = evaluate_model(model, X_test, y_test)
        drift_result = run_drift_analysis(X_train, y_train, X_test, y_test)
        db_success = log_to_monitoring_db(metrics, drift_result, "monitoring_only")
        
        execution_time = time.time() - start_time
        report_path = generate_monitoring_report(metrics, drift_result, "monitoring_only", execution_time)
        
        logger.info("SUCCESS - MONITORING PIPELINE COMPLETED!")
        
        return {
            "success": True,
            "metrics": metrics,
            "drift_detected": drift_result,
            "db_success": db_success,
            "execution_time": execution_time,
            "report_path": str(report_path)
        }
        
    except Exception as e:
        logger.error(f"MONITORING PIPELINE FAILED: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Create orchestration directory
    Path("orchestration").mkdir(exist_ok=True)
    
    # Run the full unified pipeline
    result = unified_fraud_detection_pipeline()
    
    # For monitoring only (uncomment to use):
    # result = monitoring_only_pipeline()
    
    # Exit with appropriate code
    exit(0 if result.get("success", False) else 1)