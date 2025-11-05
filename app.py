import streamlit as st
import pandas as pd
import joblib
from streamlit_geolocation import streamlit_geolocation
import folium
from streamlit_folium import st_folium
import requests
from geopy.distance import geodesic

# =============================================
# API KEY for OpenRouteService
# =============================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="

# =============================================
# Load Trained Model
# =============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("delivery_time_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

model = load_model()

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
# Location Picker
# =============================================
st.header("📍 Select Real Locations on Map")

colA, colB = st.columns(2)
with colA:
    st.subheader("🏪 Restaurant Location")
    restaurant = streamlit_geolocation(key="restaurant_location")
with colB:
    st.subheader("🏠 Delivery Location")
    delivery = streamlit_geolocation(key="delivery_location")

# =============================================
# Process locations if available
# =============================================
if restaurant and delivery:
    rest_lat, rest_lon = restaurant["latitude"], restaurant["longitude"]
    del_lat, del_lon = delivery["latitude"], delivery["longitude"]

    st.success(f"**Restaurant:** ({rest_lat:.5f}, {rest_lon:.5f})")
    st.success(f"**Delivery:** ({del_lat:.5f}, {del_lon:.5f})")

    # =============================================
    # Get Road Route via OpenRouteService
    # =============================================
    def get_route(lat1, lon1, lat2, lon2):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[lon1, lat1], [lon2, lat2]]}

        try:
            res = requests.post(url, json=body, headers=headers)
            res.raise_for_status()
            data = res.json()

            route_coords = [(coord[1], coord[0]) for coord in data["features"][0]["geometry"]["coordinates"]]
            distance_km = data["features"][0]["properties"]["segments"][0]["distance"] / 1000
            duration_min = data["features"][0]["properties"]["segments"][0]["duration"] / 60
            return route_coords, distance_km, duration_min
        except Exception as e:
            st.warning(f"⚠️ Could not fetch route from ORS: {e}")
            return None, geodesic((lat1, lon1), (lat2, lon2)).km, None

    route, distance_km, duration_min = get_route(rest_lat, rest_lon, del_lat, del_lon)

    # =============================================
    # Map Visualization
    # =============================================
    m = folium.Map(location=[(rest_lat + del_lat) / 2, (rest_lon + del_lon) / 2], zoom_start=13)
    folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color="green")).add_to(m)

    if route:
        folium.PolyLine(route, color="purple", weight=5, opacity=0.8).add_to(m)
    else:
        folium.PolyLine([(rest_lat, rest_lon), (del_lat, del_lon)], color="gray", dash_array="5").add_to(m)

    st_folium(m, width=900, height=500)

    # =============================================
    # Display Route Info
    # =============================================
    st.subheader("🛣️ Route Information")
    st.metric("📏 Road Distance (km)", f"{distance_km:.2f}")
    if duration_min:
        st.metric("⏱️ Estimated Driving Time (min)", f"{duration_min:.1f}")

    # =============================================
    # Feature Encoding for Model
    # =============================================
    weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
    vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
    festival_map = {"No": 0, "Yes": 1}

    time_diff = abs(
        (pd.to_datetime(str(time_picked)) - pd.to_datetime(str(time_ordered))).total_seconds()
    ) / 60

    # =============================================
    # Prepare Input Data for Prediction
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
    # Predict Delivery Time
    # =============================================
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"⏱️ **Predicted Delivery Time:** {prediction:.2f} minutes")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

else:
    st.info("ℹ️ Please select both Restaurant and Delivery locations to continue.")
