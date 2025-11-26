import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import requests
from datetime import datetime

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
# Delivery Person & Order Details
# =============================================
st.header("🧠 Delivery & Order Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Delivery Person Age", 18, 60, 30)
    rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = st.number_input("Multiple Deliveries", 0, 10, 0)
with col2:
    order_date = st.date_input("Order Date")
    time_ordered = st.time_input("Time Ordered")
    time_picked = st.time_input("Time Picked")

col3, col4, col5 = st.columns(3)
with col3:
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
with col4:
    traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"])

order_type = st.selectbox("Order Type", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "truck"])

# =============================================
# Model Selection
# =============================================
st.header("🧩 Select Regression Model")

model_choice = st.selectbox(
    "Choose a regression model:",
    [
        "Random Forest (random_forest_model.pkl)",
        "Linear Regression (linear_regression_model.pkl)",
        "Decision Tree (decision_tree_model.pkl)"
    ]
)

model_file_map = {
    "Random Forest (random_forest_model.pkl)": "random_forest_model.pkl",
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

# =============================================
# Geocoding Helper (OpenRouteService)
# =============================================
def geocode_address(address):
    if not address:
        return None
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": ORS_API_KEY, "text": address, "size": 1}
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code == 200 and res.json().get("features"):
            coords = res.json()["features"][0]["geometry"]["coordinates"]
            return {"lat": coords[1], "lon": coords[0]}
    except Exception as e:
        st.warning(f"⚠️ Geocoding failed for '{address}': {e}")
    return None

# =============================================
# Address Input
# =============================================
st.header("📍 Address Selection")

colA, colB = st.columns(2)
with colA:
    restaurant_address = st.text_input("🏪 Restaurant Address", placeholder="Enter restaurant location")
    restaurant_data = geocode_address(restaurant_address) if restaurant_address else None
with colB:
    delivery_address = st.text_input("🏠 Delivery Address", placeholder="Enter delivery location")
    delivery_data = geocode_address(delivery_address) if delivery_address else None

# =============================================
# Proceed if Both Locations Selected
# =============================================
if restaurant_data and delivery_data:
    rest_lat, rest_lon = restaurant_data["lat"], restaurant_data["lon"]
    del_lat, del_lon = delivery_data["lat"], delivery_data["lon"]

    st.success(f"🏪 Restaurant: ({rest_lat:.5f}, {rest_lon:.5f})")
    st.success(f"🏠 Delivery: ({del_lat:.5f}, {del_lon:.5f})")

    # =============================================
    # Get Road Route and Details via OpenRouteService
    # =============================================
    def get_route_and_details(lat1, lon1, lat2, lon2):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY}
        params = {"start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"}
        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()["features"][0]
                coords = data["geometry"]["coordinates"]
                route = [(c[1], c[0]) for c in coords]
                distance_km = data["properties"]["segments"][0]["distance"] / 1000  # Meters to km
                duration_min = data["properties"]["segments"][0]["duration"] / 60  # Seconds to minutes
                return route, distance_km, duration_min
        except Exception as e:
            st.warning(f"⚠️ Route data fetch failed: {e}. Using fallback.")
        # Fallback: Geodesic distance, no duration
        return None, geodesic((lat1, lon1), (lat2, lon2)).km, None

    route, actual_distance_km, estimated_duration_min = get_route_and_details(rest_lat, rest_lon, del_lat, del_lon)
    st.info(f"📏 Actual Driving Distance: {actual_distance_km:.2f} km")
    if estimated_duration_min:
        st.info(f"⏱️ Estimated Travel Time (from API): {estimated_duration_min:.2f} minutes")

    # =============================================
    # Map Display
    # =============================================
    m = folium.Map(location=[(rest_lat + del_lat) / 2, (rest_lon + del_lon) / 2], zoom_start=13)
    folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color="green")).add_to(m)

    if route:
        folium.PolyLine(route, color="purple", weight=5, opacity=0.8).add_to(m)
    else:
        folium.PolyLine([(rest_lat, rest_lon), (del_lat, del_lon)], color="gray", dash_array="5").add_to(m)

    st_folium(m, width=900, height=500)

    # =============================================
    # Enhanced Feature Engineering
    # =============================================
    weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
    vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
    festival_map = {"No": 0, "Yes": 1}

    # Derived time features
    order_datetime = pd.to_datetime(f"{order_date} {time_ordered}")
    picked_datetime = pd.to_datetime(f"{order_date} {time_picked}")
    time_diff_min = (picked_datetime - order_datetime).total_seconds() / 60
    hour_of_day = time_ordered.hour
    day_of_week = order_date.weekday()

    # =============================================
    # Create Input DataFrame (Temporarily 17 Features - Uncomment New Ones After Retraining)
    # =============================================
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
        # Uncomment after retraining models on 22 features:
        # "Actual_Distance_km": actual_distance_km,
        # "Estimated_Travel_Time_min": estimated_duration_min if estimated_duration_min else 0,
        # "Time_Diff_Order_to_Pickup_min": time_diff_min,
        # "Hour_of_Day": hour_of_day,
        # "Day_of_Week": day_of_week
    }])

    # Debug: Display feature count
    st.write(f"🔍 Debug: Input DataFrame has {input_data.shape[1]} features. Model expects {model.n_features_in_}.")

    # =============================================
    # Prediction
    # =============================================
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🧮 Model Used: **{model_choice}**")
        st.success(f"⏱️ Predicted Delivery Time: **{prediction:.2f} minutes**")
        if estimated_duration_min:
            st.info(f"🔍 API Estimated Travel Time: {estimated_duration_min:.2f} min | Model Prediction: {prediction:.2f} min")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}. Retrain models for 22 features.")

else:
    st.info("ℹ️ Please enter both Restaurant and Delivery addresses.")
