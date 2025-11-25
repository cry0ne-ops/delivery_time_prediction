# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ===========================
# CONFIG
# ===========================
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6Ijc2Y2I5NmExMzM4MTRlNjhiOTY5OTIwMjk3MWRhMWExIiwiaCI6Im11cm11cjY0In0="
st.set_page_config(page_title="Delivery Time Prediction — Multi-Model", layout="wide")

# ===========================
# MODEL LOADING UTILITIES
# ===========================
@st.cache_resource
def load_model_cached(path):
    if not os.path.isfile(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Failed to load model at {path}: {e}")
        return None

# load three models (they may be missing — we'll handle that)
main_model = load_model_cached("delivery_time_model.pkl")
tree_model = load_model_cached("decision_tree_model.pkl")
linear_model = load_model_cached("linear_regression_model.pkl")

# ===========================
# PAGE HEADER
# ===========================
st.title("🚚 Improved Delivery Time Prediction System")
st.caption("Prototype for Thesis — compare multiple regression models and view accuracy.")

# ===========================
# DELIVERY / ORDER INPUTS
# ===========================
st.header("🧠 Delivery & Order Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Delivery Person Age", 18, 60, 30)
    rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5, 0.1)
    multiple_deliveries = st.number_input("Number of Multiple Deliveries", 0, 10, 0, step=1)
with col2:
    order_date = st.date_input("Order Date")
    time_ordered = st.time_input("Time Ordered")
    time_picked = st.time_input("Time Picked")

col3, col4, col5 = st.columns(3)
with col3:
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy", "Stormy"])
with col4:
    traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
with col5:
    festival = st.selectbox("Festival Day?", ["No", "Yes"])

order_type = st.selectbox("Type of Order", ["Meat", "Fruits", "Fruits and Vegetables"])
vehicle = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "truck"])

st.markdown("---")

# ===========================
# GEOCODING HELPER (ORS)
# ===========================
def geocode_address(address):
    if not address:
        return None
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"api_key": ORS_API_KEY, "text": address, "size": 1}
    try:
        res = requests.get(url, params=params, timeout=8)
        if res.status_code == 200 and res.json().get("features"):
            coords = res.json()["features"][0]["geometry"]["coordinates"]
            return {"lat": coords[1], "lon": coords[0]}
    except Exception as e:
        st.warning(f"Geocoding failed: {e}")
    return None

# ===========================
# ADDRESS INPUTS WITH GEOCODING
# ===========================
st.header("📍 Address Selection (type addresses, then press Enter)")

colA, colB = st.columns(2)
with colA:
    restaurant_address = st.text_input("🏪 Restaurant Address", placeholder="e.g. SM City Baguio, Philippines")
    restaurant_data = geocode_address(restaurant_address) if restaurant_address else None
with colB:
    delivery_address = st.text_input("🏠 Delivery Address", placeholder="e.g. Burnham Park, Baguio City, Philippines")
    delivery_data = geocode_address(delivery_address) if delivery_address else None

# ===========================
# PROCEED IF BOTH LOCATIONS ARE GEOCODED
# ===========================
if restaurant_data and delivery_data:
    rest_lat, rest_lon = restaurant_data["lat"], restaurant_data["lon"]
    del_lat, del_lon = delivery_data["lat"], delivery_data["lon"]

    st.success(f"🏪 Restaurant: ({rest_lat:.5f}, {rest_lon:.5f})")
    st.success(f"🏠 Delivery: ({del_lat:.5f}, {del_lon:.5f})")

    # ROUTE (ORS directions) - use POST for robustness
    def get_route_and_stats(lat1, lon1, lat2, lon2):
        url = "https://api.openrouteservice.org/v2/directions/driving-car"
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        body = {"coordinates": [[lon1, lat1], [lon2, lat2]]}
        try:
            res = requests.post(url, json=body, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            coords = [(c[1], c[0]) for c in data["features"][0]["geometry"]["coordinates"]]
            distance_m = data["features"][0]["properties"]["segments"][0]["distance"]
            duration_s = data["features"][0]["properties"]["segments"][0]["duration"]
            return coords, distance_m/1000.0, duration_s/60.0
        except Exception:
            # fallback: no route from ORS — return None but still compute straight-line distance
            return None, geodesic((lat1, lon1), (lat2, lon2)).km, None

    route_coords, road_distance_km, road_duration_min = get_route_and_stats(rest_lat, rest_lon, del_lat, del_lon)
    st.info(f"📏 Straight-line distance: {geodesic((rest_lat, rest_lon), (del_lat, del_lon)).km:.2f} km")
    if road_distance_km:
        st.success(f"🛣️ Road distance (ORS): {road_distance_km:.2f} km")
    if road_duration_min:
        st.success(f"⏱️ ORS estimated travel time: {road_duration_min:.1f} min")

    # MAP
    m = folium.Map(location=[(rest_lat + del_lat)/2, (rest_lon + del_lon)/2], zoom_start=13)
    folium.Marker([rest_lat, rest_lon], tooltip="Restaurant", icon=folium.Icon(color="blue")).add_to(m)
    folium.Marker([del_lat, del_lon], tooltip="Delivery", icon=folium.Icon(color="green")).add_to(m)
    if route_coords:
        folium.PolyLine(route_coords, color="purple", weight=5, opacity=0.8).add_to(m)
    else:
        folium.PolyLine([(rest_lat, rest_lon), (del_lat, del_lon)], color="gray", dash_array="5").add_to(m)
    st_folium(m, width=900, height=500)

    # ===========================
    # Prepare input_data for models
    # Note: align column names with what your models expect.
    # If your models expect extra engineered features, compute them here.
    # ===========================
    weather_map = {"Sunny": 1, "Cloudy": 2, "Rainy": 3, "Stormy": 4}
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    order_map = {"Meat": 1, "Fruits": 2, "Fruits and Vegetables": 3}
    vehicle_map = {"motorcycle": 1, "scooter": 2, "truck": 3}
    festival_map = {"No": 0, "Yes": 1}

    time_diff = abs((pd.to_datetime(str(time_picked)) - pd.to_datetime(str(time_ordered))).total_seconds())/60.0

    input_dict = {
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
        "Festival": festival_map[festival],
        # optional: include road_distance_km and road_duration_min if your model used them
        "Road_distance_km": road_distance_km if road_distance_km is not None else np.nan,
        "Road_duration_min": road_duration_min if road_duration_min is not None else np.nan,
        "Time_diff_minutes": time_diff
    }

    # Build DataFrame (1-row) and attempt to align to model.feature_names_in_ when possible
    input_df = pd.DataFrame([input_dict])

    def align_input_for_model(df, model):
        """
        Align input dataframe columns to the features expected by the model, if possible.
        If the model exposes `feature_names_in_` we will reorder/select columns accordingly.
        Otherwise return df as-is and hope the model can handle it.
        """
        if model is None:
            return None
        try:
            expected = getattr(model, "feature_names_in_", None)
            if expected is not None:
                expected = list(expected)
                # Add missing expected cols with NaN so shape matches
                for c in expected:
                    if c not in df.columns:
                        df[c] = np.nan
                return df[expected]
        except Exception:
            pass
        return df

    # ===========================
    # PREDICTIONS: compute for each model
    # ===========================
    st.header("📊 Model Predictions Comparison")
    col1, col2, col3 = st.columns(3)

    preds = {}
    models = {
        "Main Model": main_model,
        "Decision Tree": tree_model,
        "Linear Regression": linear_model
    }

    for (name, model), col in zip(models.items(), [col1, col2, col3]):
        with col:
            if model is None:
                st.warning(f"{name} not found.")
                preds[name] = None
            else:
                X_for_model = align_input_for_model(input_df.copy(), model)
                try:
                    pred = model.predict(X_for_model)[0]
                    preds[name] = float(pred)
                    # style the output a bit
                    if name == "Main Model":
                        st.success(f"🔮 {name}: {pred:.2f} min")
                    elif name == "Decision Tree":
                        st.info(f"🌳 {name}: {pred:.2f} min")
                    else:
                        st.warning(f"📐 {name}: {pred:.2f} min")
                except Exception as e:
                    st.error(f"{name} prediction failed: {e}")
                    preds[name] = None

    # ===========================
    # ACCURACY COMPARISON (if validation file exists)
    # ===========================
    st.markdown("---")
    st.header("📈 Accuracy Comparison (validation dataset)")

    # Try a few common filenames for validation set
    validation_paths = ["validation_data.csv", "validation.csv", "val.csv"]
    val_path = next((p for p in validation_paths if os.path.exists(p)), None)

    if val_path is None:
        st.info(
            "No validation file found. To show accuracy metrics place a CSV named "
            "'validation_data.csv' (or validation.csv / val.csv) in the app folder. "
            "The file must contain the same features used to train the models and a target column named 'Time_taken(min)'."
        )
    else:
        try:
            val_df = pd.read_csv(val_path)
            if "Time_taken(min)" not in val_df.columns:
                st.error("Validation CSV found but missing 'Time_taken(min)' target column.")
            else:
                y_val = val_df["Time_taken(min)"]
                X_val = val_df.drop(columns=["Time_taken(min)"])
                accuracy_rows = []
                for name, model in models.items():
                    if model is None:
                        accuracy_rows.append([name, None, None, None])
                        continue
                    try:
                        # Align X_val to model expected features if possible
                        X_val_aligned = X_val.copy()
                        expected = getattr(model, "feature_names_in_", None)
                        if expected is not None:
                            for c in expected:
                                if c not in X_val_aligned.columns:
                                    X_val_aligned[c] = np.nan
                            X_val_aligned = X_val_aligned[expected]
                        preds_val = model.predict(X_val_aligned)
                        mae = mean_absolute_error(y_val, preds_val)
                        rmse = mean_squared_error(y_val, preds_val, squared=False)
                        r2 = r2_score(y_val, preds_val)
                        accuracy_rows.append([name, mae, rmse, r2])
                    except Exception as e:
                        accuracy_rows.append([name, None, None, None])
                        st.warning(f"Could not compute metrics for {name}: {e}")

                acc_df = pd.DataFrame(accuracy_rows, columns=["Model", "MAE", "RMSE", "R2"])
                st.dataframe(acc_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}))

                # Simple bar charts (MAE & RMSE)
                plot_df = acc_df.set_index("Model")[["MAE", "RMSE"]].dropna()
                if not plot_df.empty:
                    st.subheader("MAE / RMSE Comparison")
                    st.bar_chart(plot_df)

        except Exception as e:
            st.error(f"Failed to load/compute validation metrics: {e}")

else:
    st.info("ℹ️ Please enter both Restaurant and Delivery addresses (and press Enter after typing).")
