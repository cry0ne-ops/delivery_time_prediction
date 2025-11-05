import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
from streamlit_geocoder import st_geocoder
import folium
from streamlit_folium import st_folium
import requests

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="  
@st.cache_resource
def load_model():
    try:
        return joblib.load("delivery_time_model.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

st.set_page_config(page_title="Delivery Time Prediction", layout="wide")
st.title("🚚 Delivery Time Prediction System")

# Delivery person & order info
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
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"])

order_type = st.selectbox("Order Type", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "truck"])

st.header("📍 Select Locations")
colA, colB = st.columns(2)
with colA:
    rest = st_geocoder("Restaurant Location", key="rest")
with colB:
    dest = st_geocoder("Delivery Location", key="del")

if rest and dest:
    r_lat, r_lon = rest["latitude"], rest["longitude"]
    d_lat, d_lon = dest["latitude"], dest["longitude"]
    st.success(f"Restaurant: {rest['address']}")
    st.success(f"Delivery: {dest['address']}")

    def get_route(lat1, lon1, lat2, lon2):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY}
        params = {"start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"}
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            coords = res.json()["features"][0]["geometry"]["coordinates"]
            return [(c[1], c[0]) for c in coords]
        return None

    route = get_route(r_lat, r_lon, d_lat, d_lon)
    dist = geodesic((r_lat, r_lon), (d_lat, d_lon)).km
    st.info(f"Distance: {dist:.2f} km")

    if route:
        m = folium.Map(location=[(r_lat + d_lat) / 2, (r_lon + d_lon) / 2], zoom_start=13)
        folium.Marker([r_lat, r_lon], tooltip="Restaurant", icon=folium.Icon(color="blue")).add_to(m)
        folium.Marker([d_lat, d_lon], tooltip="Delivery", icon=folium.Icon(color="green")).add_to(m)
        folium.PolyLine(route, color="purple", weight=5).add_to(m)
        st_folium(m, width=800, height=500)

        # encode and predict once
        weather_map = {"Sunny":1, "Cloudy":2, "Rainy":3, "Stormy":4}
        traffic_map = {"Low":1, "Medium":2, "High":3, "Jam":4}
        order_map = {"Meat":1, "Fruits":2, "Fruits and Vegetables":3}
        vehicle_map = {"motorcycle":1, "scooter":2, "truck":3}
        festival_map = {"No":0, "Yes":1}

        time_diff = abs((pd.to_datetime(str(time_picked)) - pd.to_datetime(str(time_ordered))).total_seconds()) / 60
        x = pd.DataFrame([{
            "ID":1,
            "Delivery_person_ID":1001,
            "Delivery_person_Age":age,
            "Delivery_person_Ratings":rating,
            "Restaurant_latitude":r_lat,
            "Restaurant_longitude":r_lon,
            "Delivery_location_latitude":d_lat,
            "Delivery_location_longitude":d_lon,
            "Order_Date":int(order_date.strftime("%Y%m%d")),
            "Time_Orderd":int(time_ordered.strftime("%H%M")),
            "Time_Order_picked":int(time_picked.strftime("%H%M")),
            "Weatherconditions":weather_map[weather],
            "Road_traffic_density":traffic_map[traffic],
            "Type_of_order":order_map[order_type],
            "Type_of_vehicle":vehicle_map[vehicle],
            "multiple_deliveries":multiple_deliveries,
            "Festival":festival_map[festival]
        }])

        try:
            y_pred = model.predict(x)[0]
            st.success(f"Predicted Delivery Time: {y_pred:.2f} minutes")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Could not get route data.")
else:
    st.info("Please select both locations to continue.")
