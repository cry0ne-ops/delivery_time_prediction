import streamlit as st
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("delivery_time_model.pkl")

st.title("🚚 Delivery Time Prediction System")
st.write("Enter delivery details to predict estimated delivery time.")

# Sidebar inputs
st.sidebar.header("📦 Input Delivery Details")

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

# Create DataFrame
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

# 🔧 Encode categorical columns exactly like training
def encode_inputs(df):
    # Create mappings (you can adjust these based on your training data)
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

# Encode before prediction
encoded_data = encode_inputs(input_data)

if st.button("Predict Delivery Time"):
    try:
        prediction = model.predict(encoded_data)
        st.success(f"⏱️ Estimated Delivery Time: {round(prediction[0], 2)} minutes")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
