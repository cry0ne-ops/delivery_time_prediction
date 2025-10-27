import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("delivery_time_model.pkl")

st.title("🚚 Delivery Time Prediction System")
st.write("Enter the delivery details below to predict the estimated delivery time.")

# Sidebar for user input
st.sidebar.header("📦 Input Delivery Details")

# Collect user inputs
age = st.sidebar.number_input("Delivery Person Age", min_value=18, max_value=65, value=30)
rating = st.sidebar.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
restaurant_lat = st.sidebar.number_input("Restaurant Latitude", value=16.40)
restaurant_lon = st.sidebar.number_input("Restaurant Longitude", value=120.59)
delivery_lat = st.sidebar.number_input("Delivery Location Latitude", value=13.00)
delivery_lon = st.sidebar.number_input("Delivery Location Longitude", value=77.00)
weather = st.sidebar.selectbox("Weather Conditions", ["Sunny", "Stormy", "Rainy", "Cloudy"])
traffic = st.sidebar.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
order_type = st.sidebar.selectbox("Type of Order", ["Meat", "Vegetables", "Fruits and Vegetables"])
vehicle = st.sidebar.selectbox("Type of Vehicle", ["motorcycle", "truck"])
multiple = st.sidebar.selectbox("Multiple Deliveries", [0, 1])
festival = st.sidebar.selectbox("Festival", ["Yes", "No"])

# Create input DataFrame
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
    'Festival': [festival]
})

st.write("### Input Summary:")
st.write(input_data)

# Predict button
if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(input_data)
        st.success(f"⏱️ Estimated Delivery Time: {round(prediction[0], 2)} minutes")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
