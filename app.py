import streamlit as st
import pandas as pd
import joblib
import math
from datetime import datetime

# Load model
model = joblib.load("delivery_time_model.pkl")

st.title("🚚 Improved Delivery Time Prediction System")
st.write("Predict estimated delivery time using regression model.")

# Sidebar inputs
st.sidebar.header("📦 Delivery Details")

age = st.sidebar.number_input("Delivery Person Age", min_value=18, max_value=65, value=30)
rating = st.sidebar.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
restaurant_lat = st.sidebar.number_input("Restaurant Latitude", value=16.40)
restaurant_lon = st.sidebar.number_input("Restaurant Longitude", value=120.59)
delivery_lat = st.sidebar.number_input("Delivery Latitude", value=13.00)
delivery_lon = st.sidebar.number_input("Delivery Longitude", value=77.00)

weather = st.sidebar.selectbox("Weather", ["Sunny", "Stormy", "Rainy", "Cloudy"])
traffic = st.sidebar.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
order_type = st.sidebar.selectbox("Type of Order", ["Meat", "Vegetables", "Fruits and Vegetables"])
vehicle = st.sidebar.selectbox("Vehicle Type", ["motorcycle", "truck"])
multiple = st.sidebar.selectbox("Multiple Deliveries", [0, 1])
festival = st.sidebar.selectbox("Festival", ["Yes", "No"])

order_time = st.sidebar.time_input("Order Time")
pickup_time = st.sidebar.time_input("Pickup Time")
order_date = st.sidebar.date_input("Order Date")

# --- Helper Functions ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

distance = calculate_distance(restaurant_lat, restaurant_lon, delivery_lat, delivery_lon)

# Compute derived time features
try:
    order_dt = datetime.combine(order_date, order_time)
    pickup_dt = datetime.combine(order_date, pickup_time)
    time_diff = (pickup_dt - order_dt).total_seconds() / 60
    order_hour = order_time.hour
    pickup_hour = pickup_time.hour
    day_of_week = order_date.weekday()
except Exception:
    time_diff = 0
    order_hour = 0
    pickup_hour = 0
    day_of_week = 0

# Create dataframe
input_data = pd.DataFrame({
    'Delivery_person_Age': [age],
    'Delivery_person_Ratings': [rating],
    'Restaurant_latitude': [restaurant_lat],
    'Restaurant_longitude': [restaurant_lon],
    'Delivery_location_latitude': [delivery_lat],
    'Delivery_location_longitude': [delivery_lon],
    'Weatherconditions': [weather],
    'Road_traffic_density': [traffic],
    'Type_of_order': [order_type],
    'Type_of_vehicle': [vehicle],
    'multiple_deliveries': [multiple],
    'Festival': [festival],
    'distance': [distance],
    'time_diff': [time_diff],
    'order_hour': [order_hour],
    'pickup_hour': [pickup_hour],
    'day_of_week': [day_of_week]
})

st.write("### Input Summary")
st.write(input_data)

# Encode categorical columns
def encode_inputs(df):
    weather_map = {'Sunny': 0, 'Stormy': 1, 'Rainy': 2, 'Cloudy': 3}
    traffic_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
    order_map = {'Meat': 0, 'Vegetables': 1, 'Fruits and Vegetables': 2}
    vehicle_map = {'motorcycle': 0, 'truck': 1}
    festival_map = {'No': 0, 'Yes': 1}

    df['Weatherconditions'] = df['Weatherconditions'].map(weather_map)
    df['Road_traffic_density'] = df['Road_traffic_density'].map(traffic_map)
    df['Type_of_order'] = df['Type_of_order'].map(order_map)
    df['Type_of_vehicle'] = df['Type_of_vehicle'].map(vehicle_map)
    df['Festival'] = df['Festival'].map(festival_map)
    return df

encoded_data = encode_inputs(input_data)

# --- Prediction ---
if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(encoded_data)
        st.success(f"⏱️ Estimated Delivery Time: {round(prediction[0], 2)} minutes")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
