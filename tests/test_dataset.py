# tests/test_make_dataset.py

import os
import pandas as pd
from src.data.make_dataset import load_raw_data, preprocess_data

def test_load_raw_data():
    df = load_raw_data("data/raw/creditcard.csv")
    assert not df.empty, "Data failed to load"

def test_preprocess_data():
    df = pd.DataFrame({"A": [1, 1, None]})
    processed = preprocess_data(df)
    assert processed.isnull().sum().sum() == 0, "Missing values not handled"
