from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "models" / "fraud_detection_model.pkl"
INPUT_COLUMNS = (
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)
FEATURE_COLUMNS = INPUT_COLUMNS + ("balanceDiffOrig", "balanceDiffDest")
TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


@st.cache_resource
def load_model():
    """Load and validate the committed scikit-learn pipeline."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The saved model does not provide a predict() method")

    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is not None and tuple(model_columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Saved model feature schema does not match the application inputs: "
            f"expected {FEATURE_COLUMNS}, received {tuple(model_columns)}"
        )
    return model


def build_transaction_features(transaction):
    """Build the exact eight-column feature row used to train the saved model."""
    if set(transaction) != set(INPUT_COLUMNS):
        missing = sorted(set(INPUT_COLUMNS) - set(transaction))
        extra = sorted(set(transaction) - set(INPUT_COLUMNS))
        raise ValueError(f"Invalid transaction fields; missing={missing}, extra={extra}")

    features = dict(transaction)
    # These equations match the feature engineering in the training notebook.
    features["balanceDiffOrig"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )
    features["balanceDiffDest"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )
    return pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])


def predict_transaction(model, transaction):
    """Return the saved classifier's binary prediction for one transaction."""
    prediction = model.predict(build_transaction_features(transaction))[0]
    return int(prediction)


def main():
    st.title("Fraud Detection Prediction App")
    st.markdown(
        "Enter transaction details to view the saved model's classification."
    )
    st.caption(
        "A research demonstration; predictions are not a substitute for a reviewed risk decision."
    )
    st.divider()

    transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    old_balance = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
    new_balance = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
    old_balance_dest = st.number_input(
        "Old Balance (Receiver)", min_value=0.0, value=0.0
    )
    new_balance_dest = st.number_input(
        "New Balance (Receiver)", min_value=0.0, value=0.0
    )

    if st.button("Predict"):
        transaction = {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": new_balance,
            "oldbalanceDest": old_balance_dest,
            "newbalanceDest": new_balance_dest,
        }
        prediction = predict_transaction(load_model(), transaction)
        if prediction == 1:
            st.subheader("Prediction: Fraud")
            st.error("The model classifies this transaction as potentially fraudulent.")
        else:
            st.subheader("Prediction: Not flagged as fraud")
            st.success("The model does not flag this transaction as fraudulent.")


if __name__ == "__main__":
    main()
