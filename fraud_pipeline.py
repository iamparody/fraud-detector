from prefect import flow, task
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier, log_evaluation
from mlflow.models import infer_signature
import platform
import getpass
import subprocess

# ========================
# PATHS & CONFIG
# ========================
DATA_PATH = Path("data/features")
MODEL_PATH = Path("models/best_model.pkl")
ARTIFACTS_DIR = Path("artifacts")
MLRUNS_DIR = Path("mlruns")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow setup
TRACKING_URI = f"file:///{Path.cwd()/'mlruns'}".replace("\\", "/")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Fraud_Detection_Experiment")

@task(name="Load Training Data")
def load_data():
    """Load preprocessed training data"""
    print(f"📂 Loading from: {DATA_PATH.resolve()}")
    
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv").values.ravel()
    
    print(f"✅ Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    return X_train, y_train

@task(name="Train LightGBM with CV")
def train_model_cv(X_train, y_train):
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
    
    print("\n🔄 Starting 5-fold CV...\n")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    fold_scores = []
    
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
        
        print(f"🔖 Run ID: {run.info.run_id}")
        mlflow.log_params(params)
        
        # Cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
            print(f"\n📊 Fold {fold}")
            
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
            
            print(f"✅ Fold {fold} AUC: {fold_auc:.5f}")
            mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)
        
        # Overall metrics
        overall_auc = roc_auc_score(y_train, oof_preds)
        print(f"\n🎯 Overall CV AUC: {overall_auc:.5f}")
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
        imp_path = ARTIFACTS_DIR / "feature_importance.png"
        plt.savefig(imp_path)
        plt.close()
        mlflow.log_artifact(str(imp_path), artifact_path="plots")
        
        # Log model with signature
        input_example = X_train.head(5)
        pred_example = model.predict_proba(input_example)
        signature = infer_signature(input_example, pred_example)
        
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        
        # Save locally
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="local_model")
        
        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            registered = mlflow.register_model(model_uri, "FraudDetector")
            print(f"✅ Registered: FraudDetector v{registered.version}")
        except Exception as e:
            print(f"⚠️  Model registration: {e}")
        
        return model, overall_auc, run.info.run_id

@task(name="Evaluate Model")
def evaluate_model(model, X_train, y_train):
    """Evaluate final model"""
    print("\n📊 Final Evaluation...")
    
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    
    auc = roc_auc_score(y_train, y_proba)
    print(f"✅ Training ROC-AUC: {auc:.5f}")
    print("\n" + classification_report(y_train, y_pred))
    
    return {"auc": auc}

@flow(
    name="Fraud Detection Pipeline",
    log_prints=True,
    flow_run_name="fraud-detection-{date}",
)
def fraud_detection_pipeline():
    """End-to-end fraud detection pipeline with LightGBM + MLflow"""
    
    print("="*60)
    print("🚀 FRAUD DETECTION PIPELINE - LightGBM + MLflow")
    print("="*60)
    
    # Step 1: Load data
    X_train, y_train = load_data()
    
    # Step 2: Train with CV (depends on load_data)
    model, cv_auc, run_id = train_model_cv(X_train, y_train)
    
    # Step 3: Evaluate (depends on train_model_cv)
    metrics = evaluate_model(model, X_train, y_train)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED!")
    print(f"📊 CV AUC: {cv_auc:.5f}")
    print(f"💾 Model saved: {MODEL_PATH}")
    print(f"🔖 MLflow Run ID: {run_id}")
    print(f"🌐 MLflow UI: mlflow ui --port 5000")
    print("="*60)
    
    return {"model_path": str(MODEL_PATH), "cv_auc": cv_auc, "run_id": run_id}

if __name__ == "__main__":
    fraud_detection_pipeline()