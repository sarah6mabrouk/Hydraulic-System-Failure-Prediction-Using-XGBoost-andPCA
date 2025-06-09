import streamlit as st
import pandas as pd
import joblib as jbl
import numpy as np

# Load trained artifacts
model = jbl.load("xgb_internal_pump_leakage.pkl")
scaler = jbl.load("scaler.pkl")
pca = jbl.load("pca.pkl")

# Optional: Load original feature names (only if saved previously)
try:
    original_columns = jbl.load("original_columns.pkl")  # list of 202 column names
except:
    original_columns = [f"feature_{i+1}" for i in range(202)]  # fallback

# Optional: Load default row to simplify inputs
try:
    default_input = jbl.load("default_input.pkl")  # a Series or DataFrame row with 202 values
except:
    default_input = [0.0] * 202  # fallback

st.title("🛠️ Hydraulic System Failure Prediction")
st.write("Predict **Internal Pump Leakage** based on input sensor readings.")

# Input form
with st.form("prediction_form"):
    inputs = []
    st.markdown("### Input Sensor Readings")
    for i in range(202):
        val = st.number_input(
            f"{original_columns[i]}", 
            value=float(default_input[i]) if default_input else 0.0
        )
        inputs.append(val)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Convert to DataFrame
    X = pd.DataFrame([inputs], columns=original_columns)

    # Scale and apply PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Predict
    prediction = model.predict(X_pca)[0]
    if prediction == 0:
        message = "No leakage"
    elif prediction == 1:
        message = "weak leakage"
    else:
        message = "severe leakage"
    st.success(f"✅ Predicted Class: **{prediction}** : **{message}**")
