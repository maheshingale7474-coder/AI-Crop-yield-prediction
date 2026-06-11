
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai

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

# --- Configure Gemini API ---
# Ensure GOOGLE_API_KEY is set in your Streamlit Cloud secrets or environment variables
if "GOOGLE_API_KEY" not in os.environ:
    st.warning("GOOGLE_API_KEY not found in environment variables. Chatbot functionality may not work.")
    # Optionally, you can add a placeholder or disable the chatbot if no key is found
    gemini_model = None
else:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- App Title and Description ---
st.title("🌾 Crop Yield Prediction App")
st.markdown("Enter the environmental parameters to predict crop yield using Random Forest and XGBoost models.")

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
    st.success(f"Predicted Yield (Random Forest): **{rf_prediction:.2f}** units")
    st.info(f"Predicted Yield (XGBoost): **{xgb_prediction:.2f}** units")
    st.markdown("--- Request data ---")
    st.json(input_data.to_dict(orient='records')[0])

# --- Chatbot for Suggestions ---
st.header("💡 Crop Optimization Suggestions")
if gemini_model:
    user_query = st.text_area("Ask for crop optimization suggestions:", "What are some general tips for optimizing potato yield in moderate rainfall?")
    if st.button("Get Suggestion"):
        if user_query:
            with st.spinner("Generating suggestions..."):
                try:
                    response = gemini_model.generate_content(user_query)
                    st.markdown("**Gemini's Suggestion:**")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating suggestion: {e}")
        else:
            st.warning("Please enter a query for suggestions.")
elif st.button("Get Suggestion"):
    st.error("Chatbot not available. Please configure your GOOGLE_API_KEY in Streamlit secrets.")

