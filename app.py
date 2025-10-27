import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("⏱️ Delivery Time Prediction App")
st.write("Predict estimated delivery time based on order details.")

# Sidebar input form
st.sidebar.header("📦 Input Details")

delivery_person_age = st.sidebar.number_input("Delivery Person Age", min_value=18, max_value=65, value=30)
delivery_person_rating = st.sidebar.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
restaurant_lat = st.sidebar.number_input("Restaurant Latitude", value=16.40)
restaurant_lon = st.sidebar.number_input("Restaurant Longitude", value=120.59)
delivery_lat = st.sidebar.number_input("Delivery Location Latitude", value=13.00)
delivery_lon = st.sidebar.number_input("Delivery Location Longitude", value=77.00)
weather = st.sidebar.selectbox("Weather Conditions", ["Sunny", "Stormy", "Rainy", "Cloudy"])
traffic = st.sidebar.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
order_type = st.sidebar.selectbox("Type of Order", ["Meat", "Vegetables", "Fruits and Vegetables"])
vehicle = st.sidebar.selectbox("Type of Vehicle", ["motorcycle", "truck"])
multiple_deliveries = st.sidebar.selectbox("Multiple Deliveries", [0, 1])
festival = st.sidebar.selectbox("Festival", ["Yes", "No"])

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Delivery_person_Age': [delivery_person_age],
    'Delivery_person_Ratings': [delivery_person_rating],
    'Restaurant_latitude': [restaurant_lat],
    'Restaurant_longitude': [restaurant_lon],
    'Delivery_location_latitude': [delivery_lat],
    'Delivery_location_longitude': [delivery_lon],
    'Weatherconditions': [weather],
    'Road_traffic_density': [traffic],
    'Type_of_order': [order_type],
    'Type_of_vehicle': [vehicle],
    'multiple_deliveries': [multiple_deliveries],
    'Festival': [festival]
})

# Predict
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_data)
    st.success(f"⏰ Estimated Delivery Time: {round(prediction[0], 2)} minutes")
