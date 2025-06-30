import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="🚖 NYC Taxi Fare Predictor", layout="centered")

st.title("🚖 NYC Taxi Fare Prediction App")
st.markdown("Enter trip details to estimate the total fare.")

model_filename = 'best_random_forest_model.pkl'

# Check if model exists
if not os.path.exists(model_filename):
    st.error(f"❌ Model file `{model_filename}` not found.\n\nPlease ensure the model is trained and saved in this directory.")
    st.stop()

# Load trained model
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Input form
with st.form("fare_form"):
    pickup_long = st.number_input("Pickup Longitude", value=-73.985)
    pickup_lat = st.number_input("Pickup Latitude", value=40.758)
    dropoff_long = st.number_input("Dropoff Longitude", value=-73.985)
    dropoff_lat = st.number_input("Dropoff Latitude", value=40.761)
    hour = st.slider("Hour of Pickup (0–23)", 0, 23, 14)
    month = st.slider("Month", 1, 12, 6)
    passenger_count = st.selectbox("Passenger Count", [1, 2, 3, 4, 5, 6])
    is_weekend = st.radio("Is it a Weekend?", ["Yes", "No"]) == "Yes"
    store_flag = st.radio("Store and Forward Flag", ["Yes", "No"]) == "Yes"
    payment_type = st.selectbox("Payment Type", ["1", "2", "3", "4", "5", "6"])  # Payment encoding assumed as 1–6
    submitted = st.form_submit_button("Predict Fare")

# Haversine function
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

if submitted:
    try:
        trip_distance = haversine(pickup_long, pickup_lat, dropoff_long, dropoff_lat)
        pickup_day_encoded = int(is_weekend)
        store_encoded = int(store_flag)

        # One-hot encoding for payment type (1–6)
        pay_encoded = [0] * 6
        pay_encoded[int(payment_type) - 1] = 1

        # Final input features
        features = [trip_distance, hour, month, pickup_day_encoded, store_encoded] + pay_encoded

        prediction = model.predict([features])[0]
        st.success(f"💰 Estimated Total Fare: **${prediction:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
