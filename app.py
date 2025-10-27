import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained regression model
model = joblib.load("delivery_time_model.pkl")

st.title("🚚 Delivery Time Prediction System")
st.write("This prototype predicts delivery time using regression models to help prevent food spoilage.")

# --- Input fields ---
st.subheader("Input Delivery Details")

age = st.number_input("Delivery Person Age", min_value=18, max_value=70, value=30)
rating = st.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, value=4.5)
restaurant_lat = st.number_input("Restaurant Latitude", value=14.5995)
restaurant_long = st.number_input("Restaurant Longitude", value=120.9842)
delivery_lat = st.number_input("Delivery Location Latitude", value=14.6760)
delivery_long = st.number_input("Delivery Location Longitude", value=121.0437)
traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Fog", "Stormy", "Windy"])
vehicle = st.selectbox("Vehicle Type", ["Bike", "Motorcycle", "Scooter", "Car"])

# Encode categorical inputs manually (depends on your model)
traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
weather_map = {"Sunny": 0, "Cloudy": 1, "Fog": 2, "Stormy": 3, "Windy": 4}
vehicle_map = {"Bike": 0, "Motorcycle": 1, "Scooter": 2, "Car": 3}

# Convert inputs to numeric array
features = np.array([[
    age, rating, restaurant_lat, restaurant_long,
    delivery_lat, delivery_long,
    traffic_map[traffic],
    weather_map[weather],
    vehicle_map[vehicle]
]])

# --- Predict button ---
if st.button("Predict Delivery Time"):
    prediction = model.predict(features)
    st.success(f"🕒 Estimated Delivery Time: {round(prediction[0], 2)} minutes")

