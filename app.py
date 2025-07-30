import streamlit as st
import joblib
import numpy as np
import requests

# Page setup
st.set_page_config(page_title="Crop Recommender", page_icon="🌾")
st.title("🌾 Crop Recommendation System")

# Weather API (optional)
api_key = "d56fb2ef217db80dee4a005b2c8e25e4"

# Get weather data
def get_weather(lat, lon):
    res = requests.get(
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    ).json()
    if "main" not in res:
        raise ValueError("Weather data not available for this location.")
    return (
        round(res['main']['temp'], 2),
        round(res['main']['humidity'], 2),
        round(res.get('rain', {}).get('1h', 0.0), 2)
    )

# Session defaults
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {
        "temp": 25.0,
        "humidity": 80.0,
        "rainfall": 200.0
    }

# Input section
st.subheader("🧪 Enter Soil and Weather Data")

N = st.number_input("Nitrogen", min_value=0)
P = st.number_input("Phosphorus", min_value=0)
K = st.number_input("Potassium", min_value=0)

temperature = st.number_input("Temperature (°C)", value=st.session_state.weather_data["temp"])
humidity = st.number_input("Humidity (%)", value=st.session_state.weather_data["humidity"])
ph = st.number_input("pH", min_value=5.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", value=st.session_state.weather_data["rainfall"])

# Prediction
if st.button("Predict Crop"):
    try:
        # Load model and scaler
        model = joblib.load("crop_recommendation_model.pkl")
        scaler = joblib.load("scaler.pkl")  # Make sure you saved it!

        # Prepare and scale inputs
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)

        # Predict
        pred = model.predict(scaled_input)[0]
        st.success(f"🌱 Recommended Crop: *{pred.upper()}*")

    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")
