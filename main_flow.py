from prefect import flow
from pipelines.load_data import load_data
from pipelines.train_model import train_model_cv
from pipelines.evaluate_model import evaluate_model

@flow(name="Fraud Detection Pipeline", log_prints=True)
def fraud_detection_pipeline():
    X_train, y_train = load_data()
    model, cv_auc = train_model_cv(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train)
    print("✅ Pipeline finished!")
    return {"cv_auc": cv_auc, "train_auc": metrics["auc"]}

if __name__ == "__main__":
    fraud_detection_pipeline()
