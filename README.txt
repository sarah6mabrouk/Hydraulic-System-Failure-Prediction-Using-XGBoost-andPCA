# Hydraulic System Failure Prediction

This project predicts internal pump leakage in a hydraulic system using time-series data from 17 industrial sensors. 

## Highlights
- Feature extraction from raw sensor data (mean, median, skew, etc.)
- PCA for dimensionality reduction
- XGBoost model with >90% accuracy
- Model deployed using Streamlit

## Files
- `notebook.ipynb` – main ML pipeline
- `streamlit_app.py` – interactive prediction app
- `model.pkl`, `scaler.pkl`, `pca.pkl` – serialized components

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run streamlit_app.py`


NB: i created 2 streamlit deployments (.py)
the first one with 202 columns, but bec ause it's not practical for use nor for testing i created another one with only the means of each feature : PS1_mean, PS2_mean , etc...