import streamlit as st
import pandas as pd
import pickle
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium

# ====================================================
# 🔍 LOAD TRAINED MODEL SAFELY (.pkl format)
# ====================================================
@st.cache_resource
def load_model():
    try:
        with open("delivery_time_model.pkl", "rb") as f:
            model = pickle.load(f)
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
    st.markdown("### Select Restaurant & Delivery Locations on Map")

    start_coords = [16.412, 120.595]
    m = folium.Map(location=start_coords, zoom_start=13)

    folium.Marker(location=start_coords, popup="Restaurant", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker(location=[16.422, 120.600], popup="Delivery", icon=folium.Icon(color="green")).add_to(m)

    map_data = st_folium(m, width=700, height=500)

    if map_data and "last_object_clicked" in map_data:
        clicked = map_data["last_object_clicked"]
        delivery_lat, delivery_lon = clicked["lat"], clicked["lng"]
        st.info(f"📍 Delivery location: {delivery_lat}, {delivery_lon}")

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
        # Build input DataFrame (⚠️ Must match training features)
        input_data = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Restaurant_latitude": restaurant_lat,
            "Restaurant_longitude": restaurant_lon,
            "Delivery_location_latitude": delivery_lat,
            "Delivery_location_longitude": delivery_lon,
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Type_of_order": order_type,
            "Type_of_vehicle": vehicle,
            "multiple_deliveries": multiple_deliveries,
            "Festival": festival,
            "distance": distance,
            "Order_hour": order_hour
        }])

        # Encode categorical values if model requires numeric input
        input_data_encoded = pd.get_dummies(input_data)

        # Align columns with model training
        missing_cols = set(model.feature_names_in_) - set(input_data_encoded.columns)
        for c in missing_cols:
            input_data_encoded[c] = 0  # fill missing columns

        input_data_encoded = input_data_encoded[model.feature_names_in_]

        # Predict
        prediction = model.predict(input_data_encoded)[0]
        st.success(f"⏱️ Estimated Delivery Time: **{prediction:.2f} minutes**")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
