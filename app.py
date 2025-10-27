import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# =========================================================
# 🔹 Load trained model safely
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("delivery_time_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

st.set_page_config(page_title="Delivery Time Prediction", layout="wide")
st.title("🚚 Improved Delivery Time Prediction System")
st.caption("Prototype for Thesis: Prevent Food Spoilage through Regression Models")

# =========================================================
# 📍 Helper for geolocation
# =========================================================
def get_lat_lon(address):
    geolocator = Nominatim(user_agent="delivery_time_app")
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception:
        return None, None

# =========================================================
# 🧠 Delivery & Order Details
# =========================================================
st.header("🧠 Delivery & Order Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Delivery Person Age", 18, 60, 30)
    rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = st.number_input("Number of Multiple Deliveries", 0, 10, 0)
with col2:
    order_date = st.date_input("Order Date")
    time_ordered = st.time_input("Time Ordered")
    time_picked = st.time_input("Time Picked")

# =========================================================
# 🌦️ Environmental & Order Info
# =========================================================
st.header("🌦️ Environment & Traffic")

col3, col4, col5 = st.columns(3)
with col3:
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Stormy"])
with col4:
    traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"])

# ✅ Updated order options
order_type = st.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "truck"])

# =========================================================
# 📍 Address Input — Auto Detection
# =========================================================
st.header("📍 Real Location Detection")

restaurant_address = st.text_input("Enter Restaurant Address", "SM City Baguio, Philippines")
delivery_address = st.text_input("Enter Delivery Address", "Burnham Park, Baguio City, Philippines")

if restaurant_address and delivery_address:
    rest_lat, rest_lon = get_lat_lon(restaurant_address)
    del_lat, del_lon = get_lat_lon(delivery_address)

    if rest_lat and del_lat:
        distance = geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km
        st.success("✅ Coordinates fetched successfully!")
        st.write(f"**Restaurant:** {restaurant_address} → ({rest_lat:.5f}, {rest_lon:.5f})")
        st.write(f"**Delivery:** {delivery_address} → ({del_lat:.5f}, {del_lon:.5f})")
        st.write(f"📏 **Distance:** {distance:.2f} km")

        # --- Map visualization
        m = folium.Map(location=[(rest_lat + del_lat) / 2, (rest_lon + del_lon) / 2], zoom_start=13)
        folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color='green')).add_to(m)
        st_folium(m, width=700, height=400)

        # =========================================================
        # 🔢 Encoding (must match training setup)
        # =========================================================
        weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
        traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
        order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
        vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
        festival_map = {"No": 0, "Yes": 1}

        # Compute time difference
        time_diff = abs(
            (pd.to_datetime(str(time_picked)) - pd.to_datetime(str(time_ordered))).total_seconds()
        ) / 60

        # =========================================================
        # 🧩 Construct 17-feature input (must match model)
        # =========================================================
        input_data = pd.DataFrame([{
            "ID": 1,
            "Delivery_person_ID": 1001,
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Restaurant_latitude": rest_lat,
            "Restaurant_longitude": rest_lon,
            "Delivery_location_latitude": del_lat,
            "Delivery_location_longitude": del_lon,
            "Order_Date": int(order_date.strftime("%Y%m%d")),
            "Time_Orderd": int(time_ordered.strftime("%H%M")),
            "Time_Order_picked": int(time_picked.strftime("%H%M")),
            "Weatherconditions": weather_map[weather],
            "Road_traffic_density": traffic_map[traffic],
            "Type_of_order": order_map[order_type],
            "Type_of_vehicle": vehicle_map[vehicle],
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival_map[festival]
        }])

        # =========================================================
        # 🧮 Predict
        # =========================================================
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"⏱️ **Predicted Delivery Time:** {prediction:.2f} minutes")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Could not locate one or both addresses. Please check spelling or try nearby landmarks.")
else:
    st.info("ℹ️ Please enter both restaurant and delivery addresses to continue.")
