import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
from streamlit_geocoder import st_geocoder
import folium
from streamlit_folium import st_folium
import requests

# =========================================================
# 🔹 API Key for OpenRouteService
# =========================================================
ORS_API_KEY = "YOUR_API_KEY_HERE"  # Paste your real key here safely

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
# 🌦️ Environment & Order Info
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

st.header("🌦️ Environment & Traffic")
col3, col4, col5 = st.columns(3)
with col3:
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Stormy"])
with col4:
    traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"])

order_type = st.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "truck"])

# =========================================================
# 📍 Address Inputs (with Autocomplete)
# =========================================================
st.header("📍 Select Locations")

colA, colB = st.columns(2)
with colA:
    restaurant_location = st_geocoder("Select Restaurant Location", key="rest")
with colB:
    delivery_location = st_geocoder("Select Delivery Location", key="del")

if restaurant_location and delivery_location:
    rest_lat, rest_lon = restaurant_location["latitude"], restaurant_location["longitude"]
    del_lat, del_lon = delivery_location["latitude"], delivery_location["longitude"]

    st.success(f"✅ Restaurant: {restaurant_location['address']}")
    st.success(f"✅ Delivery: {delivery_location['address']}")

    # =========================================================
    # 🗺️ Fetch Route Using OpenRouteService
    # =========================================================
    def get_route(lat1, lon1, lat2, lon2):
        try:
            url = "https://api.openrouteservice.org/v2/directions/driving-car"
            headers = {"Authorization": ORS_API_KEY}
            params = {"start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"}
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                route_coords = data["features"][0]["geometry"]["coordinates"]
                route_points = [(coord[1], coord[0]) for coord in route_coords]
                return route_points
            else:
                st.error(f"Routing error: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Routing exception: {e}")
            return None

    route = get_route(rest_lat, rest_lon, del_lat, del_lon)
    distance = geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km
    st.info(f"📏 Estimated Distance: **{distance:.2f} km**")

    if route:
        m = folium.Map(location=[(rest_lat + del_lat) / 2, (rest_lon + del_lon) / 2], zoom_start=13)
        folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color='blue')).add_to(m)
        folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color='green')).add_to(m)
        folium.PolyLine(route, color="purple", weight=5).add_to(m)
        st_folium(m, width=800, height=500)

        # =========================================================
        # 🧩 Encode Data & Predict
        # =========================================================
        weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
        traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
        order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
        vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
        festival_map = {"No": 0, "Yes": 1}

        time_diff = abs(
            (pd.to_datetime(str(time_picked)) - pd.to_datetime(str(time_ordered))).total_seconds()
        ) / 60

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
        # 🧮 Prediction
        # =========================================================
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"⏱️ **Predicted Delivery Time:** {prediction:.2f} minutes")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.warning("⚠️ Could not fetch route from OpenRouteService.")
else:
    st.info("ℹ️ Please select both restaurant and delivery locations to continue.")
