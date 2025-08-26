import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load('MODEL.pkl')
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
# owner_encoder = joblib.load("owner_encoder.pkl")
# fuel_encoder = joblib.load("fuel_encoder.pkl")
# brand_encoder = joblib.load("brand_encoder.pkl")
# model_encoder = joblib.load("brand_encoder.pkl")
# transmission_encoder = joblib.load("transmission_encoder.pkl")
# seller_type_encoder = joblib.load("seller_type_encoder.pkl")


# Page config
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction App")

# Sidebar inputs
st.sidebar.header("Enter Car Details")

year = st.sidebar.slider("Year", 1990, 2025, 2015)
km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0)
seats = st.sidebar.selectbox("Seats", [2, 4, 5, 6, 7, 8])
converted_mileage = st.sidebar.number_input("Mileage (kmpl or km/kg)", min_value=0.0)
engine_size = st.sidebar.number_input("Engine Size (cc)", min_value=500.0)
max_power_num = st.sidebar.number_input("Max Power (bhp)", min_value=20.0)
torque_Nm = st.sidebar.number_input("Torque (Nm)", min_value=50.0)
mean_rpm = st.sidebar.number_input("Mean RPM", min_value=500)

owner = st.sidebar.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
brand = st.sidebar.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Audi'])  # Add more as needed
model_name = st.sidebar.text_input("Model Name", "City")
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic'])
fuel = st.sidebar.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.sidebar.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])

# Prepare input
categorical_inputs = [['transmission', 'owner', 'brand', 'model', 'seller_type', 'fuel']]
encoded_values = ordinal_encoder.transform(categorical_inputs)[0]
transmission_encoded, owner_encoded, brand_encoded, model_encoded, seller_type_encoded, fuel_encoded = encoded_values


input_data = pd.DataFrame({
    'owner': [owner_encoded], 'brand': [brand_encoded],
    'model': [model_encoded], 'transmission': [transmission_encoded],
    'fuel': [fuel_encoded], 'year': [year], 'seller_type': [seller_type_encoded],
    'km_driven': [km_driven], 'seats': [seats],
    'converted_mileage': [converted_mileage],
    'engine_size': [engine_size],
    'max_power_num': [max_power_num],
    'torque_Nm': [torque_Nm],
    'mean_rpm': [mean_rpm]
})


# Predict
if st.button("Predict Selling Price"):
    prediction = model.predict(input_data)[0]
    predicted_price = np.exp(prediction)  # or 10 ** if log10 was used

    st.success(f"Estimated Selling Price: ₹ {int(predicted_price):,}")
