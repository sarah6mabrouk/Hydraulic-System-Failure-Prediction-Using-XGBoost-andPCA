import streamlit as st
import pandas as pd
import joblib as jbl
import numpy as np

# Load the saved model, scaler, and feature names
model = jbl.load("xgb_means_model.pkl")
scaler = jbl.load("means_scaler.pkl")
mean_columns = jbl.load("mean_columns.pkl")
default_input = jbl.load("default_input_mean.pkl")  # Optional default values

# App title
st.title("Hydraulic System Failure Prediction")
st.subheader("Predicting Internal Pump Leakage")
st.write("This model uses **mean sensor values** to predict the internal condition of the pump.")

# Input form
with st.form("input_form"):
    inputs = []
    st.markdown("### Input Mean Sensor Values")

    for i, col_name in enumerate(mean_columns):
        default_val = default_input[i] if default_input else 0.0
        val = st.number_input(f"{col_name}", value=float(default_val))
        inputs.append(val)

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    X_input = pd.DataFrame([inputs], columns=mean_columns)
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]

    if prediction == 0:
        message = "No leakage"
    elif prediction == 1:
        message = "weak leakage"
    else:
        message = "severe leakage"

    st.success(f"Predicted internal pump leakage class: **{prediction}** : **{message}**")
