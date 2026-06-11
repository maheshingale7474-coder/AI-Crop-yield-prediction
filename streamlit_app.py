
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('crop_yield_model.pkl')
        xgb_model = joblib.load('crop_yield_xgboost_model.pkl')
        return rf_model, xgb_model
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'crop_yield_model.pkl' and 'crop_yield_xgboost_model.pkl' are in the same directory as this app.")
        st.stop()

rf_model, xgb_model = load_models()

# --- App Title and Description ---
st.title("🌾 Crop Yield Prediction App")
st.markdown("Enter the environmental parameters to predict crop yield using Random Forest and XGBoost models.")

# --- Crop Selection (for display purposes, models are not crop-specific yet) ---
crops = ['Maize', 'Wheat', 'Rice', 'Barley', 'Soybean', 'Potato', 'Tomato', 'Cotton']
selected_crop = st.selectbox("Select a Crop", crops)

st.write(f"You selected: **{selected_crop}**")

# --- Input Features ---
st.header("Input Crop Parameters")

# Using columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=29.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=68.0, step=0.1)
    nitrogen = st.number_input("Nitrogen (N) (kg/ha)", min_value=0.0, max_value=200.0, value=82.0, step=0.1)

with col2:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=140.0, step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=42.0, step=0.1)
    phosphorus = st.number_input("Phosphorus (P) (kg/ha)", min_value=0.0, max_value=200.0, value=42.0, step=0.1)

with col3:
    potassium = st.number_input("Potassium (K) (kg/ha)", min_value=0.0, max_value=200.0, value=52.0, step=0.1)

# --- Prediction Button ---
if st.button("Predict Yield"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Rainfall': [rainfall],
        'Humidity': [humidity],
        'Soil_Moisture': [soil_moisture],
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium]
    })

    # Ensure the order of columns matches the training data
    expected_features = ['Temperature', 'Rainfall', 'Humidity', 'Soil_Moisture', 'Nitrogen', 'Phosphorus', 'Potassium']
    input_data = input_data[expected_features]
    input_data = input_data.astype(float) # Ensure all input features are floats

    # Make predictions
    rf_prediction = rf_model.predict(input_data)[0]
    xgb_prediction = xgb_model.predict(input_data)[0]

    st.subheader("Prediction Results")
    st.success(f"Predicted Yield (Random Forest): **{rf_prediction:.2f}** ton per hectare")
    st.info(f"Predicted Yield (XGBoost): **{xgb_prediction:.2f}** ton per hectare")
    st.markdown("--- Request data ---")
    st.json(input_data.to_dict(orient='records')[0])

