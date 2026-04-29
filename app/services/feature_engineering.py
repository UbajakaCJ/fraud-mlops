"""
Feature Engineering Service
----------------------------
Must be kept in sync with the training pipeline in the Airflow DAG.
Any change here requires a corresponding change to fraud_train_dag.py.
"""
import numpy as np


def engineer_features(raw: dict) -> np.ndarray:
    """
    Transform raw transaction dict into the feature vector expected by the model.

    Transformations applied (must match training):
    - log(Amount + 1)
    - Cyclical time encoding: sin/cos of hour-of-day
    - Drop original Amount and Time columns
    """
    v_features = [raw[f"V{i}"] for i in range(1, 29)]

    amount = raw["Amount"]
    time   = raw["Time"]

    log_amount  = np.log1p(amount)
    time_hour   = (time % 86400) / 3600
    time_sin    = np.sin(2 * np.pi * time_hour / 24)
    time_cos    = np.cos(2 * np.pi * time_hour / 24)

    features = v_features + [log_amount, time_hour, time_sin, time_cos]
    return np.array(features, dtype=np.float32).reshape(1, -1)
