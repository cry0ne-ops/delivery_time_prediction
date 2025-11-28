import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import random
import folium
from streamlit_folium import st_folium
from openrouteservice import Client
import math

# ============================================
# ORS API Key
# ============================================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="

# ============================================
# Load Dataset
# ============================================
df = pd.read_csv("update dataset (1).csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")

# ============================================
# Feature Engineering
# ============================================
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"].astype(str), format="%d/%m/%Y", errors="coerce")
    df["order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["order_month"] = df["Order_Date"].dt.month

def clean_time_to_hhmm_int(time_str):
    time_str = str(time_str).strip()
    if ':' in time_str:
        try:
            dt_obj = pd.to_datetime(time_str, format='%H:%M:%S').time()
            return dt_obj.hour*100 + dt_obj.minute
        except:
            return np.nan
    else:
        try:
            return int(time_str.zfill(4))
        except:
            return np.nan

for col in ["Time_Orderd", "Time_Order_picked"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_time_to_hhmm_int)

if "Time_Orderd" in df.columns and "Time_Order_picked" in df.columns:
    df.dropna(subset=["Time_Orderd", "Time_Order_picked"], inplace=True)
    df["order_hour"] = df["Time_Orderd"] // 100
    df["pickup_hour"] = df["Time_Order_picked"] // 100
    df["pickup_delay_min"] = ((df["pickup_hour"] - df["order_hour"])*60).clip(lower=0)

if "Time_taken(min)" in df.columns:
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.replace('(min) ', '', regex=False).astype(float)

TARGET = "Time_taken(min)"
FEATURES = [
    "Delivery_person_Age","Delivery_person_Ratings",
    "Restaurant_latitude","Restaurant_longitude",
    "Delivery_location_latitude","Delivery_location_longitude",
    "multiple_deliveries","order_day_of_week","order_month",
    "order_hour","pickup_hour","pickup_delay_min",
    "Weatherconditions","Road_traffic_density",
    "Type_of_order","Type_of_vehicle","Festival"
]
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# Load Models
# ============================================
preprocessor = joblib.load("preprocessing_pipeline.pkl")
lr_model = joblib.load("linear_regression_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# ============================================
# Session State Defaults
# ============================================
default_values = {
    "Delivery_person_Age": 25,
    "Delivery_person_Ratings": 4.0,
    "pickup_delay_min": 5,
    "Type_of_order": "Meat",
    "Type_of_vehicle": "Bike",
    "Festival": "No",
    "Restaurant_latitude": 12.9716,
    "Restaurant_longitude": 77.5946,
    "Delivery_location_latitude": 12.9352,
    "Delivery_location_longitude": 77.6245,
    "multiple_deliveries": 1,
    "order_day_of_week": 0,
    "order_month": 1,
    "order_hour": 12,
    "pickup_hour": 12,
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Low"
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# Random Data Generator
# ============================================
def generate_random_delivery_data():
    return {
        "Delivery_person_Age": random.randint(18, 60),
        "Delivery_person_Ratings": round(random.uniform(2.5, 5.0), 1),
        "Restaurant_latitude": round(random.uniform(12.90, 13.00), 6),
        "Restaurant_longitude": round(random.uniform(77.55, 77.65), 6),
        "Delivery_location_latitude": round(random.uniform(12.90, 13.00), 6),
        "Delivery_location_longitude": round(random.uniform(77.55, 77.65), 6),
        "multiple_deliveries": random.randint(1, 5),
        "order_day_of_week": random.randint(0, 6),
        "order_month": random.randint(1, 12),
        "order_hour": random.randint(8, 22),
        "pickup_hour": random.randint(8, 23),
        "pickup_delay_min": random.randint(0, 30),
        "Weatherconditions": random.choice(["Sunny","Cloudy","Rainy","Stormy","Fog"]),
        "Road_traffic_density": random.choice(["Low","Medium","High","Jam"]),
        "Type_of_order": random.choice(["Meat","Vegetables","Meat or Vegetables"]),
        "Type_of_vehicle": random.choice(["Bike","Car","Scooter"]),
        "Festival": random.choice(["Yes","No"])
    }

@st.cache_data(ttl=600)
def get_ors_distance(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    """
    Calculate driving distance (km) using ORS without affecting prediction.
    """
    client = Client(key=ORS_API_KEY)
    coords = [[restaurant_long, restaurant_lat], [delivery_long, delivery_lat]]
    try:
        route = client.directions(coords, profile='driving-car', format='geojson')
        # Distance is in meters, convert to km
        distance_km = route['features'][0]['properties']['segments'][0]['distance'] / 1000
        duration_min = route['features'][0]['properties']['segments'][0]['duration'] / 60
        return distance_km, duration_min
    except:
        return None, None


# ============================================
# Prediction Function
# ============================================
def predict_delivery_time(input_data):
    df_input = pd.DataFrame([input_data])
    numeric_features = [
        "Delivery_person_Age","Delivery_person_Ratings",
        "Restaurant_latitude","Restaurant_longitude",
        "Delivery_location_latitude","Delivery_location_longitude",
        "multiple_deliveries","order_day_of_week","order_month",
        "order_hour","pickup_hour","pickup_delay_min"
    ]
    df_input[numeric_features] = df_input[numeric_features].astype(float)
    return {
        "Linear Regression": round(lr_model.predict(df_input)[0],2),
        "Decision Tree": round(dt_model.predict(df_input)[0],2),
        "Random Forest": round(rf_model.predict(df_input)[0],2)
    }

# ============================================
# ORS Route
# ============================================
@st.cache_data(ttl=600)
def get_ors_route(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    client = Client(key=ORS_API_KEY)
    coords = [[restaurant_long, restaurant_lat], [delivery_long, delivery_lat]]
    try:
        return client.directions(coords, profile='driving-car', format='geojson')
    except:
        return None

def visualize_route_simple(restaurant_lat, restaurant_long, delivery_lat, delivery_long):
    map_center = [(restaurant_lat + delivery_lat)/2, (restaurant_long + delivery_long)/2]
    m = folium.Map(location=map_center, zoom_start=13)
    folium.Marker([restaurant_lat, restaurant_long], tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([delivery_lat, delivery_long], tooltip="Delivery Location", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([(restaurant_lat, restaurant_long), (delivery_lat, delivery_long)], color="blue", weight=3, opacity=0.8).add_to(m)
    return m

# ============================================
# Streamlit Dashboard Layout
# ============================================
st.set_page_config(page_title="Delivery Time Dashboard 🚀", layout="wide")
st.title("🛵 Delivery Time Prediction Dashboard")

# Use 3 columns for full-width layout
col_input, col_pred, col_map = st.columns([1,1,1.5])

# --- Column 1: Compact Delivery Details Card ---
with col_input:
    st.subheader("🔧 Delivery Details")

    # Random data button
    if st.button("🎲 Generate Random Delivery Details"):
        random_data = generate_random_delivery_data()
        for k, v in random_data.items():
            st.session_state[k] = v
        st.success("✅ Random delivery details generated!")

    # ----------------------
    # Single Compact Card
    # ----------------------
    with st.container():
        st.markdown("### 📋 Delivery Details Card")

        # --- Delivery Person + Pickup Delay in 1 row ---
        dp_col1, dp_col2, dp_col3 = st.columns([1,1,1])
        with dp_col1:
            st.slider("Age", 18, 60, st.session_state["Delivery_person_Age"], key="Delivery_person_Age")
        with dp_col2:
            st.slider("Rating", 0.0, 5.0, st.session_state["Delivery_person_Ratings"], step=0.1, key="Delivery_person_Ratings")
        with dp_col3:
            st.number_input("Pickup Delay (min)", 0, 120, st.session_state["pickup_delay_min"], key="pickup_delay_min")

        # --- Order Info + Type + Festival + Vehicle in 2 rows ---
        order_col1, order_col2 = st.columns([1,1])
        with order_col1:
            st.selectbox("Order Type", ["Meat","Vegetables","Meat or Vegetables"], key="Type_of_order")
            st.selectbox("Vehicle", ["Bike","Car","Scooter"], key="Type_of_vehicle")
        with order_col2:
            st.selectbox("Festival?", ["Yes","No"], key="Festival")
            st.selectbox("Weather", ["Sunny","Cloudy","Rainy","Stormy","Fog"], key="Weatherconditions")
            st.selectbox("Traffic", ["Low","Medium","High","Jam"], key="Road_traffic_density")

        # --- Location Info compact in 2 columns ---
        loc_col1, loc_col2 = st.columns([1,1])
        with loc_col1:
            st.number_input("Supplier Latitude", 12.90, 13.00, st.session_state["Restaurant_latitude"], format="%.6f", key="Restaurant_latitude")
            st.number_input("Supplier Longitude", 77.55, 77.65, st.session_state["Restaurant_longitude"], format="%.6f", key="Restaurant_longitude")
        with loc_col2:
            st.number_input("Restaurant Latitude", 12.90, 13.00, st.session_state["Delivery_location_latitude"], format="%.6f", key="Delivery_location_latitude")
            st.number_input("Restaurant Longitude", 77.55, 77.65, st.session_state["Delivery_location_longitude"], format="%.6f", key="Delivery_location_longitude")

        # --- Predict Button ---
        st.markdown("")
        if st.button("Predict Delivery Time"):
            input_data = {k: st.session_state[k] for k in default_values.keys()}
            st.session_state["predictions"] = predict_delivery_time(input_data)

            # Compute model metrics
            models = {"Linear Regression": lr_model, "Decision Tree": dt_model, "Random Forest": rf_model}
            metrics = []
            for name, model in models.items():
                y_pred = model.predict(X_test)
                metrics.append({
                    "Model": name,
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R²": r2_score(y_test, y_pred)
                })
            st.session_state["metrics_df"] = pd.DataFrame(metrics).set_index("Model")



# --- Column 2: Predictions & Metrics ---
with col_pred:
    st.subheader("📊 Predictions & Metrics")
    if "predictions" in st.session_state:
        preds = st.session_state["predictions"]
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        pred_col1.metric("Linear Regression", f"{preds['Linear Regression']} min")
        pred_col2.metric("Decision Tree", f"{preds['Decision Tree']} min")
        pred_col3.metric("Random Forest", f"{preds['Random Forest']} min")

    if "metrics_df" in st.session_state:
        st.subheader("📈 Model Accuracy on Test Set")
        st.dataframe(st.session_state["metrics_df"].style.format("{:.2f}"))
        best_model = st.session_state["metrics_df"]["RMSE"].idxmin()

# --- Column 3: Map ---
with col_map:
    st.subheader("🗺️ Delivery Route")
    route = get_ors_route(
        st.session_state["Restaurant_latitude"], st.session_state["Restaurant_longitude"],
        st.session_state["Delivery_location_latitude"], st.session_state["Delivery_location_longitude"]
    )
    if route:
        map_center = [
            (st.session_state["Restaurant_latitude"] + st.session_state["Delivery_location_latitude"]) / 2,
            (st.session_state["Restaurant_longitude"] + st.session_state["Delivery_location_longitude"]) / 2
        ]
        m = folium.Map(location=map_center, zoom_start=13)
        folium.GeoJson(route, name="Route").add_to(m)
        folium.Marker([st.session_state["Restaurant_latitude"], st.session_state["Restaurant_longitude"]],
                      tooltip="Restaurant", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([st.session_state["Delivery_location_latitude"], st.session_state["Delivery_location_longitude"]],
                      tooltip="Delivery", icon=folium.Icon(color='red')).add_to(m)
    else:
        m = visualize_route_simple(
            st.session_state["Restaurant_latitude"], st.session_state["Restaurant_longitude"],
            st.session_state["Delivery_location_latitude"], st.session_state["Delivery_location_longitude"]
        )
    st_folium(m, width=700, height=500)

    # --- ORS Driving Distance Display ---
    distance_km, duration_min = get_ors_distance(
        st.session_state["Restaurant_latitude"], st.session_state["Restaurant_longitude"],
        st.session_state["Delivery_location_latitude"], st.session_state["Delivery_location_longitude"]
    )
    if distance_km is not None:
        st.markdown(f"**🛣️ Driving Distance:** {distance_km:.2f} km")
        st.markdown(f"**⏱️ Estimated Driving Duration:** {duration_min:.1f} min")
    else:
        st.markdown("**⚠️ Could not calculate driving distance.**")
