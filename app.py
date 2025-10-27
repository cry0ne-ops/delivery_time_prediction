import streamlit as st
import pandas as pd
import pickle
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium

# ====================================================
# 🔍 LOAD TRAINED MODEL SAFELY
# ====================================================
@st.cache_resource
def load_model():
    try:
        with open("delivery_time_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

st.title("🚚 Improved Delivery Time Prediction System")
st.caption("Prototype for Thesis: Prevent Food Spoilage through Regression Models")

# ====================================================
# 📍 LOCATION INPUT SECTION
# ====================================================
geolocator = Nominatim(user_agent="delivery_app")

st.subheader("📍 Input Delivery and Restaurant Locations")

mode = st.radio(
    "Choose location input method:",
    ["Manual Coordinates", "Address Search", "Interactive Map"]
)

restaurant_lat, restaurant_lon, delivery_lat, delivery_lon = None, None, None, None

# --- Option 1: Manual Coordinates ---
if mode == "Manual Coordinates":
    restaurant_lat = st.number_input("Restaurant Latitude", value=16.408176, format="%.6f")
    restaurant_lon = st.number_input("Restaurant Longitude", value=120.594594, format="%.6f")
    delivery_lat = st.number_input("Delivery Latitude", value=16.420000, format="%.6f")
    delivery_lon = st.number_input("Delivery Longitude", value=120.600000, format="%.6f")

# --- Option 2: Address Search ---
elif mode == "Address Search":
    restaurant_address = st.text_input("Restaurant Address", "SM City Baguio, Philippines")
    delivery_address = st.text_input("Delivery Address", "Burnham Park, Baguio City, Philippines")

    def get_coordinates(address):
        try:
            location = geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            else:
                st.warning(f"⚠️ Could not find coordinates for: {address}")
                return None, None
        except Exception:
            st.error("❌ Error connecting to geolocation service.")
            return None, None

    if st.button("🔍 Get Coordinates"):
        restaurant_lat, restaurant_lon = get_coordinates(restaurant_address)
        delivery_lat, delivery_lon = get_coordinates(delivery_address)
        if restaurant_lat and delivery_lat:
            st.success("✅ Coordinates fetched successfully!")

# --- Option 3: Interactive Map ---
elif mode == "Interactive Map":
    st.markdown("### 🗺️ Select Restaurant and Delivery Locations on the Map")
    st.info("Left-click to add restaurant, right-click to add delivery location.")

    # Initialize default coordinates
    start_coords = [16.412, 120.595]
    m = folium.Map(location=start_coords, zoom_start=13)

    # Add instructions
    folium.Marker(
        location=start_coords,
        popup="Center (Default View)",
        icon=folium.Icon(color="gray")
    ).add_to(m)

    # Display map
    map_data = st_folium(m, width=700, height=500)

    if map_data and "last_object_clicked" in map_data and map_data["last_object_clicked"]:
        clicked = map_data["last_object_clicked"]
        lat, lon = clicked["lat"], clicked["lng"]
        st.write(f"📍 You clicked at: {lat:.6f}, {lon:.6f}")

        # User chooses whether click is restaurant or delivery
        point_type = st.radio("Assign clicked point as:", ["Restaurant", "Delivery"])
        if point_type == "Restaurant":
            restaurant_lat, restaurant_lon = lat, lon
            st.success(f"🏪 Restaurant location set: {lat:.6f}, {lon:.6f}")
        elif point_type == "Delivery":
            delivery_lat, delivery_lon = lat, lon
            st.success(f"📦 Delivery location set: {lat:.6f}, {lon:.6f}")

# ====================================================
# 🧾 OTHER INPUT FEATURES
# ====================================================
st.subheader("🧠 Delivery Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Delivery Person Age", min_value=18, max_value=60, value=30)
    rating = st.number_input("Delivery Person Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
    weather = st.selectbox("Weather Conditions", ["Sunny", "Cloudy", "Rainy", "Stormy"])
    vehicle = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "truck"])
with col2:
    order_type = st.selectbox("Type of Order", ["Meal", "Meat", "Vegetables", "Fruits and Vegetables"])
    multiple_deliveries = st.number_input("Multiple Deliveries", min_value=0, max_value=3, value=0)
    festival = st.selectbox("Festival", ["No", "Yes"])
    order_hour = st.number_input("Order Time (24h format)", min_value=0, max_value=23, value=12)

# ====================================================
# 📏 COMPUTE DISTANCE
# ====================================================
if restaurant_lat and delivery_lat:
    distance = geodesic((restaurant_lat, restaurant_lon), (delivery_lat, delivery_lon)).km
    st.write(f"📏 **Calculated Distance:** {distance:.2f} km")
else:
    distance = 0.0

# ====================================================
# 🧮 PREDICTION
# ====================================================
if st.button("🚀 Predict Delivery Time"):
    try:
        # 🔢 Encode categorical features (manual encoding)
        weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 3}
        traffic_map = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}
        vehicle_map = {"motorcycle": 0, "scooter": 1, "truck": 2}
        order_map = {
            "Meal": 0,
            "Meat": 1,
            "Vegetables": 2,
            "Fruits and Vegetables": 3
        }
        festival_map = {"No": 0, "Yes": 1}

        # 🧾 Prepare encoded input
        input_data = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Restaurant_latitude": restaurant_lat,
            "Restaurant_longitude": restaurant_lon,
            "Delivery_location_latitude": delivery_lat,
            "Delivery_location_longitude": delivery_lon,
            "Weatherconditions": weather_map[weather],
            "Road_traffic_density": traffic_map[traffic],
            "Type_of_order": order_map[order_type],
            "Type_of_vehicle": vehicle_map[vehicle],
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival_map[festival],
            "distance": distance,
            "Order_hour": order_hour
        }])

        # ✅ Predict using the model
        prediction = model.predict(input_data)[0]

        st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
