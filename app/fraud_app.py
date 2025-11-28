import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Model path relative to this file 
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # fraud-detection-streamlit-ml/
MODEL_PATH = BASE_DIR / 'models' / 'fraud_detection_model.pkl'

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------
st.title('Fraud Detection Prediction App')
st.markdown('Please enter the transaction details and press **Predict** to evaluate the risk.')

st.divider()

# ---------------------------------------------------------
# Input fields for the transaction details
# These correspond exactly to the model feature schema
# ---------------------------------------------------------

transaction_type = st.selectbox(
    'Transaction Type', 
    ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT']
)

amount = st.number_input(
    'Amount', 
    min_value=0.0, 
    value=1000.0
)

oldbalance = st.number_input(
    'Old Balance (Sender)', 
    min_value=0.0, 
    value=10000.0
)

newbalance = st.number_input(
    'New Balance (Sender)', 
    min_value=0.0, 
    value=9000.0
)

oldbalanceDest = st.number_input(
    'Old Balance (Receiver)', 
    min_value=0.0, 
    value=0.0
)

newbalanceDest = st.number_input(
    'New Balance (Receiver)', 
    min_value=0.0, 
    value=0.0
)

# ---------------------------------------------------------
# Prediction block
# Build a single-row DataFrame and pass it to the model
# ---------------------------------------------------------
if st.button('Predict'):

    # Prepare input in the exact format used during training
    input_data = pd.DataFrame([{
        'type': transaction_type,
        'amount': amount,
        'oldbalanceOrg': oldbalance,
        'newbalanceOrig': newbalance,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Model prediction (0 = legitimate, 1 = fraud)
    prediction = model.predict(input_data)[0]

    st.subheader(f'Prediction: {int(prediction)}')

    # ---------------------------------------------------------
    # Display result with appropriate formatting
    # ---------------------------------------------------------
    if prediction == 1:
        st.error('This transaction is likely to be **fraudulent**.')
    else:
        st.success('This transaction is likely to be **legitimate**.')

from pathlib import Path
MODEL_PATH = Path("models") / "fraud_detection_model.pkl"
print(MODEL_PATH.resolve())
print(MODEL_PATH.exists())
