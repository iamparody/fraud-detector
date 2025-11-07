from prefect import task
from sklearn.metrics import roc_auc_score, classification_report

@task(name="Evaluate Model", log_prints=True)
def evaluate_model(model, X_train, y_train):
    print("📊 Evaluating Model...")
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_proba)
    print(f"✅ Training ROC-AUC: {auc:.5f}")
    print(classification_report(y_train, y_pred))
    return {"auc": auc}
