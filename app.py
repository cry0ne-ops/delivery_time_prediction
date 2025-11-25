import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import requests
import random
from datetime import datetime, timedelta, time

# =============================================
# API KEY for OpenRouteService
# =============================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="

# =============================================
# Streamlit Page Setup
# =============================================
st.set_page_config(page_title="Delivery Time Prediction", layout="wide")
st.title("🚚 Improved Delivery Time Prediction System")
st.caption("Prototype for Thesis: Prevent Food Spoilage through Regression Models")

# =============================================
# Random Data Generator
# =============================================
def generate_random_delivery_details():
    st.session_state['age'] = random.randint(18, 60)
    st.session_state['rating'] = round(random.uniform(1.0, 5.0), 1)
    st.session_state['multiple_deliveries'] = random.randint(0, 10)

    # Random order and pick times
    random_date = datetime.today() - timedelta(days=random.randint(0, 30))
    st.session_state['order_date'] = random_date.date()
    
    random_hour_ordered = random.randint(8, 20)
    random_min_ordered = random.randint(0, 59)
    st.session_state['time_ordered'] = time(random_hour_ordered, random_min_ordered)

    delta_minutes = random.randint(5, 60)
    picked_datetime = datetime.combine(random_date.date(), st.session_state['time_ordered']) + timedelta(minutes=delta_minutes)
    st.session_state['time_picked'] = picked_datetime.time()

    # Random selections
    st.session_state['weather'] = random.choice(["Sunny", "Cloudy", "Rainy", "Stormy"])
    st.session_state['traffic'] = random.choice(["Low", "Medium", "High", "Jam"])
    st.session_state['festival'] = random.choice(["No", "Yes"])
    st.session_state['order_type'] = random.choice(["Meat", "Fruits", "Fruits and Vegetables"])
    st.session_state['vehicle'] = random.choice(["motorcycle", "scooter", "truck"])

# =============================================
# Delivery Person & Order Details
# =============================================
st.header("🧠 Delivery & Order Details")

# Button to generate random details
if st.button("🎲 Generate Random Delivery & Order Details"):
    generate_random_delivery_details()

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Delivery Person Age", 18, 60, 30, key="age")
    rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5, 0.1, key="rating")
    multiple_deliveries = st.number_input("Multiple Deliveries", 0, 10, 0, key="multiple_deliveries")
with col2:
    order_date = st.date_input("Order Date", key="order_date")
    time_ordered = st.time_input("Time Ordered", key="time_ordered")
    time_picked = st.time_input("Time Picked", key="time_picked")

col3, col4, col5 = st.columns(3)
with col3:
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"], key="weather")
with col4:
    traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"], key="traffic")
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"], key="festival")

order_type = st.selectbox("Order Type", ["Meat", "Fruits", "Fruits and Vegetables"], key="order_type")
vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "truck"], key="vehicle")

# =============================================
# Model Selection (unchanged)
# =============================================
st.header("🧩 Select Regression Model")

model_choice = st.selectbox(
    "Choose a regression model:",
    [
        "Random Forest (delivery_time_model.pkl)",
        "Linear Regression (linear_regression_model.pkl)",
        "Decision Tree (decision_tree_model.pkl)"
    ]
)

model_file_map = {
    "Random Forest (delivery_time_model.pkl)": "delivery_time_model.pkl",
    "Linear Regression (linear_regression_model.pkl)": "linear_regression_model.pkl",
    "Decision Tree (decision_tree_model.pkl)": "decision_tree_model.pkl"
}

selected_model_file = model_file_map[model_choice]

@st.cache_resource
def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model {model_filename}: {e}")
        st.stop()

model = load_model(selected_model_file)
