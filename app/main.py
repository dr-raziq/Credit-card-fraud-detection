import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("results", "models", "xgboost_model.pkl")
DATA_PATH = os.path.join("data", "creditcard_clean.csv")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("Credit Card Fraud Detection")
st.write("Enter a transaction number to check whether it is fraudulent.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Cleaned dataset not found. Run make_dataset.py first.")
        st.stop()
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

max_txn = int(df['TransactionNumber'].max())

txn_no = st.number_input(
    "Transaction Number",
    min_value=1,
    max_value=max_txn,
    value=1,
    step=1
)

if st.button("Predict Fraud Probability", type="primary"):
    row = df[df['TransactionNumber'] == txn_no]

    if row.empty:
        st.error("Transaction number not found in dataset.")
    else:
        X = row.drop(columns=['TransactionNumber', 'Class'])

        feature_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        X = X[feature_order]

        proba = model.predict_proba(X)[0][1]
        actual = int(row['Class'].values[0])

        st.write(f"### Fraud Probability: **{proba:.4f}**")

        if proba > 0.5:
            st.error("This transaction is likely fraudulent!")
        else:
            st.success("This transaction seems legitimate.")

        st.caption(f"Actual label in dataset: {'Fraud' if actual == 1 else 'Legitimate'}")