from prefect import task
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/features")

@task(name="Load Training Data", log_prints=True)
def load_data():
    print(f"📂 Loading from: {DATA_PATH.resolve()}")
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    y_train = pd.read_csv(DATA_PATH / "y_train.csv").values.ravel()
    print(f"✅ Loaded X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train
