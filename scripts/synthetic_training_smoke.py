"""Exercise the eight-feature training path on generated, non-empirical data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
FEATURES = (
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balanceDiffOrig",
    "balanceDiffDest",
)
CATEGORICAL = ["type"]
NUMERICAL = [column for column in FEATURES if column not in CATEGORICAL]


def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = 240
    amount = rng.uniform(5, 5000, rows)
    old_origin = rng.uniform(100, 12000, rows)
    old_destination = rng.uniform(0, 20000, rows)
    fraud = np.array([0, 1] * (rows // 2))
    rng.shuffle(fraud)
    origin_drop = np.where(fraud == 1, amount, np.minimum(amount, old_origin))
    destination_add = np.where(fraud == 1, 0.0, amount)
    frame = pd.DataFrame({
        "type": np.where(fraud == 1, "TRANSFER", rng.choice(["PAYMENT", "CASH_OUT"], rows)),
        "amount": amount,
        "oldbalanceOrg": old_origin,
        "newbalanceOrig": old_origin - origin_drop,
        "oldbalanceDest": old_destination,
        "newbalanceDest": old_destination + destination_add,
    })
    frame["balanceDiffOrig"] = frame["oldbalanceOrg"] - frame["newbalanceOrig"]
    frame["balanceDiffDest"] = frame["newbalanceDest"] - frame["oldbalanceDest"]
    return frame.loc[:, FEATURES], pd.Series(fraud, name="isFraud")


def run_smoke() -> dict[str, object]:
    features, target = synthetic_data()
    if tuple(features.columns) != FEATURES:
        raise AssertionError("Smoke fixture feature schema mismatch")
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=RANDOM_STATE
    )
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAL),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    result = {
        "data": "synthetic smoke fixture; not empirical dataset evaluation",
        "feature_count": len(FEATURES),
        "test_rows": len(y_test),
        "fraud_class_precision": report["1"]["precision"],
        "fraud_class_recall": report["1"]["recall"],
        "fraud_class_f1": report["1"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    assert set(model.predict(x_test)).issubset({0, 1})
    return result


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
