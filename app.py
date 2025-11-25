import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import requests

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
        "Random Forest (delivery_time_model.pkl)",
        "Linear Regression (linear_regression_model.pkl)",
        "Decision Tree (decision_tree_model.pkl)"
    ]
)

# Mapping models to their files
model_file_map = {
    "Random Forest (delivery_time_model.pkl)": "delivery_time_model.pkl",
    "Linear Regression (linear_regression_model.pkl)": "linear_regression_model.pkl",
    "Decision Tree (decision_tree_model.pkl)": "decision_tree_model.pkl"
}

scaler_file_map = {
    "Random Forest (delivery_time_model.pkl)": "random_forest_scaler.pkl",
    "Linear Regression (linear_regression_model.pkl)": "linear_scaler.pkl",
    "Decision Tree (decision_tree_model.pkl)": "decision_tree_scaler.pkl"
}

selected_model_file = model_file_map[model_choice]
selected_scaler_file = scaler_file_map[model_choice]

# =============================================
# Load model and scaler
# =============================================
@st.cache_resource
def load_model(model_filename):
    try:
        model = joblib.load(model_filename)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model {model_filename}: {e}")
        st.stop()

@st.cache_resource
def load_scaler(scaler_filename):
    try:
        scaler = joblib.load(scaler_filename)
        return scaler
    except Exception as e:
        st.error(f"❌ Could not load scaler {scaler_filename}: {e}")
        st.stop()

model = load_model(selected_model_file)
scaler = load_scaler(selected_scaler_file)

# =============================================
# Geocoding helper
# =============================================
def geocode_address(address):
    if not address:
        return None
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": ORS_API_KEY, "text": address, "size": 1}
    res = requests.get(url, params=params)
    if res.status_code == 200 and res.json().get("features"):
        coords = res.json()["features"][0]["geometry"]["coordinates"]
        return {"lat": coords[1], "lon": coords[0]}
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
# Proceed if both locations selected
# =============================================
if restaurant_data and delivery_data:
    rest_lat, rest_lon = restaurant_data["lat"], restaurant_data["lon"]
    del_lat, del_lon = delivery_data["lat"], delivery_data["lon"]

    st.success(f"🏪 Restaurant: ({rest_lat:.5f}, {rest_lon:.5f})")
    st.success(f"🏠 Delivery: ({del_lat:.5f}, {del_lon:.5f})")

    # =============================================
    # Get Route
    # =============================================
    def get_route(lat1, lon1, lat2, lon2):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY}
        params = {"start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"}
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            coords = res.json()["features"][0]["geometry"]["coordinates"]
            return [(c[1], c[0]) for c in coords]
        else:
            st.warning("⚠️ Route data could not be fetched. Showing straight line instead.")
            return None

    route = get_route(rest_lat, rest_lon, del_lat, del_lon)
    distance_km = geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km
    st.info(f"📏 Distance: {distance_km:.2f} km")

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
    # Encode categorical data
    # =============================================
    weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
    vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
    festival_map = {"No": 0, "Yes": 1}

    # =============================================
    # Create input DataFrame
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
    }])

    # =============================================
    # Numeric features for scaling
    # =============================================
    numeric_features = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Order_Date",
        "Time_Orderd",
        "Time_Order_picked",
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "multiple_deliveries",
        "Festival"
    ]

    # =============================================
    # Scale and predict
    # =============================================
    try:
        input_scaled = input_data.copy()
        input_scaled[numeric_features] = scaler.transform(input_data[numeric_features])

        prediction = model.predict(input_scaled)[0]

        st.success(f"🧮 Model Used: **{model_choice}**")
        st.success(f"⏱️ Predicted Delivery Time: **{prediction:.2f} minutes**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

else:
    st.info("ℹ️ Please enter both Restaurant and Delivery addresses.")
