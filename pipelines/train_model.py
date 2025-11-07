from prefect import task
import numpy as np
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import platform, getpass, subprocess, os
from mlflow.models import infer_signature

# === CONFIG ===
ARTIFACTS_DIR = Path("artifacts")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.pkl"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# === HELPER ===
def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "N/A"

# === MAIN TASK ===
@task(name="Train LightGBM with CV", log_prints=True)
def train_model_cv(X_train, y_train):
    # Ensure MLflow tracking is set up correctly
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Fraud_Detection_Experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

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

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    fold_scores = []

    with mlflow.start_run(run_name="LightGBM_CV_Training") as run:
        # === Tags for Traceability ===
        mlflow.set_tag("user", getpass.getuser())
        mlflow.set_tag("os", platform.platform())
        mlflow.set_tag("git_commit", _get_git_commit())
        mlflow.set_tag("training_framework", "Prefect + Airflow Hybrid")

        # === Log Params ===
        mlflow.log_params(params)

        # === CV Training ===
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
            print(f"📊 Training Fold {fold}/5")
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

            mlflow.log_metric(f"fold_{fold}_auc", fold_auc, step=fold)

        # === Overall Metrics ===
        overall_auc = roc_auc_score(y_train, oof_preds)
        mlflow.log_metric("overall_cv_auc", overall_auc)
        print(f"✅ Overall CV AUC: {overall_auc:.4f}")

        # === Save + Log Model ===
        joblib.dump(model, MODEL_PATH)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        # === Register Model ===
        model_uri = f"runs:/{run.info.run_id}/model"
        try:
            mlflow.register_model(model_uri, "FraudDetector")
            print("📦 Model registered successfully.")
        except Exception as e:
            print(f"⚠️ Registration skipped or failed: {e}")

    return model, overall_auc
