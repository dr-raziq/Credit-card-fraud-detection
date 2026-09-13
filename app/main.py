import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("results", "models", "xgboost_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("Credit Card Fraud Detection")
st.write("Enter the transaction features to get a fraud prediction.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please train the model first.")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Transaction Details")
col1, col2, col3 = st.columns(3)

with col1:
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)
with col2:
    amount = st.number_input("Amount", min_value=0.0, value=0.0)
with col3:
    st.write("")  

st.write("Principal Components (V1-V28)")
v_cols = st.columns(4)
v_values = {}
for i in range(1, 29):
    col_idx = (i-1) % 4
    with v_cols[col_idx]:
        v_values[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f")

if st.button("Predict Fraud Probability", type="primary"):
    input_dict = {'Time': time, 'Amount': amount}
    for i in range(1, 29):
        input_dict[f'V{i}'] = v_values[f'V{i}']
    input_df = pd.DataFrame([input_dict])

    feature_order = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
    input_df = input_df[feature_order]

    proba = model.predict_proba(input_df)[0][1]
    st.write(f"### Fraud Probability: **{proba:.4f}**")
    if proba > 0.5:
        st.error("This transaction is likely fraudulent!")
    else:
        st.success("This transaction seems legitimate.")