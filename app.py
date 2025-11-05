import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from datetime import datetime

# =========================================================
# 🔹 App Configuration
# =========================================================
st.set_page_config(
    page_title="🚚 Real-Time Delivery Time Prediction Dashboard",
    layout="wide",
    page_icon="🚀"
)

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

# =========================================================
# 🎨 Header Section
# =========================================================
st.markdown(
    """
    <style>
        .main-title { 
            font-size: 2.2rem; 
            color: #0C4B33; 
            font-weight: 700;
        }
        .subheader {
            color: #666;
            font-size: 1.1rem;
        }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="main-title">🚚 Improved Real-Time Delivery Time Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Prototype for Thesis: Prevent Food Spoilage through Regression Models</p>', unsafe_allow_html=True)
st.markdown("---")

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
    order_date = st.date_input("Order Date", datetime.now().date())
    time_ordered = st.time_input("Time Ordered", datetime.now().time())
    time_picked = st.time_input("Time Picked", datetime.now().time())

st.markdown("---")

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

order_type = st.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "truck"])

st.markdown("---")

# =========================================================
# 🗺️ Interactive Location Selection
# =========================================================
st.header("📍 Select Restaurant & Delivery Locations on Map")

default_center = [16.4119, 120.5990]
m = folium.Map(location=default_center, zoom_start=13)
m.add_child(folium.LatLngPopup())

st.markdown("👉 Click once for **Restaurant**, click again for **Delivery** location.")
map_data = st_folium(m, width=700, height=450)

if map_data and map_data["last_clicked"]:
    if "clicks" not in st.session_state:
        st.session_state.clicks = []
    if len(st.session_state.clicks) < 2:
        st.session_state.clicks.append(map_data["last_clicked"])
    if st.button("🔄 Reset Map Points"):
        st.session_state.clicks = []

# =========================================================
# 📍 Once 2 Points Selected
# =========================================================
if "clicks" in st.session_state and len(st.session_state.clicks) == 2:
    rest_lat = st.session_state.clicks[0]["lat"]
    rest_lon = st.session_state.clicks[0]["lng"]
    del_lat = st.session_state.clicks[1]["lat"]
    del_lon = st.session_state.clicks[1]["lng"]

    distance = geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km

    st.success("✅ Coordinates selected successfully!")
    st.write(f"**Restaurant:** ({rest_lat:.5f}, {rest_lon:.5f})")
    st.write(f"**Delivery:** ({del_lat:.5f}, {del_lon:.5f})")
    st.write(f"📏 **Distance:** {distance:.2f} km")

    # Display map with markers
    m = folium.Map(location=[(rest_lat + del_lat) / 2, (rest_lon + del_lon) / 2], zoom_start=13)
    folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color='green')).add_to(m)
    folium.PolyLine([(rest_lat, rest_lon), (del_lat, del_lon)], color="purple", weight=3).add_to(m)
    st_folium(m, width=700, height=400)

    # =========================================================
    # 🌐 Encode and Predict (Auto-refresh every few seconds)
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
    # 🕒 Live Prediction Loop
    # =========================================================
    placeholder = st.empty()
    st.info("🔄 Live prediction updating every 5 seconds...")

    for i in range(100):  # 100 iterations = about 8 minutes of updates
        try:
            prediction = model.predict(input_data)[0]
            with placeholder.container():
                st.metric(
                    label="⏱️ Predicted Delivery Time (minutes)",
                    value=f"{prediction:.2f}",
                    delta=f"Update #{i+1}"
                )
                progress = min(100, int((i+1) / 100 * 100))
                st.progress(progress)
            time.sleep(5)
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
            break

elif "clicks" in st.session_state and len(st.session_state.clicks) == 1:
    st.info("🟦 Restaurant location selected. Now click for the **Delivery** point.")
else:
    st.info("🗺️ Click once for Restaurant, again for Delivery.")
