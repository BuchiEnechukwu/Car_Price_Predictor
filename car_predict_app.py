# # import streamlit as st
# # import pandas as pd
# # import joblib

# # # Load dataset
# # model = joblib.load("MODEL.pkl")
# # encoder = joblib.load("LabelEncoder.pkl")
# # #owner_encoder= joblib.load("ENCODER.pkl")
# # #fuel_encoder =joblib.load("ENCODER.pkl")
# # # brand_encoder= joblib.load("ENCODER.pkl")
# # # model_encoder= joblib.load("ENCODER.pkl")
# # # transmission_encoder= joblib.load("ENCODER.pkl")
# # # seller_type_encoder= joblib.load("ENCODER.pkl")

# # # Setting the title of the app
# # st.title("Car Price Prediction")
# # st.set_page_config(page_title="Car Price Prediction")
# # st.write("Fill in the car details below")

# # # Input fields
# # owner_raw = st.selectbox("Owner", encoder.classes_)
# # brand_raw = st.selectbox("Car Brand", encoder.classes_)
# # model_raw = st.selectbox("Model", encoder.classes_)
# # transmission_raw = st.selectbox("Transmission Type", encoder.classes_)
# # fuel_raw = st.selectbox("Fuel Type", encoder.classes_)
# # seller_type_raw = st.selectbox("Seller Type", encoder.classes_)


# # year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
# # km_driven = st.number_input("Kilometers Driven", min_value=0)
# # mileage = st.text_input("Mileage (e.g. 20 kmpl)")
# # engine = st.text_input("Engine Capacity (e.g. 1248 CC)")
# # max_power = st.text_input("Max Power (e.g. 74 bhp)")
# # torque = st.text_input("Torque (e.g. 190Nm@ 2000rpm)")
# # seats = st.number_input("Number of Seats", min_value=2, max_value=10)
# # mean_rpm = st.number_input("RPM")

# # # --- Predict button ---
# # if st.button("Estimate Price"):
# #     try:
# #         # Encode categorical variables
# #         owner_encoded = encoder.transform([owner_raw])[0]
# #         brand_encoded = encoder.transform([brand_raw])[0]
# #         model_encoded = encoder.transform([model_raw])[0]
# #         transmission_encoded = encoder.transform([transmission_raw])[0]
# #         fuel_encoded = encoder.transform([fuel_raw])[0]
# #         seller_type_encoded = encoder.transform([seller_type_raw])[0]


# #         # Extract numeric values from text fields
# #         def extract_number(text):
# #             import re
# #             match = re.search(r'\d+\.?\d*', text)
# #             return float(match.group()) if match else 0.0

# #         mileage_val = extract_number(mileage)
# #         engine_val = extract_number(engine)
# #         max_power_val = extract_number(max_power)



# #         # Prepare input data
# #         input_data = pd.DataFrame([{'owner': owner_encoded, 'brand': brand_encoded, 
# #                                     'model': model_encoded, 
# #                                     'transmission': transmission_encoded,
# #                                     'fuel': fuel_encoded, 'year': year,
# #                                     'seller_type': seller_type_encoded,
# #                                     'km_driven': km_driven, 'seats': seats,
# #                                     'converted_mileage': mileage,
# #                                     'engine_size': engine,
# #                                     'max_power_num': max_power,
# #                                     'torque_Nm': torque,
# #                                     'mean_rpm': mean_rpm
# #         }])

     
# #         # Making the prediction
        
# #         prediction = model.predict(input_data)[0]
# #         st.success(f"Estimated Selling Price: ₹{round(prediction, 2)}")

# #     except Exception as e:
# #         st.error(f"Error making prediction: {e}")






# import streamlit as st
# import pandas as pd
# import joblib
# import re

# # Load the model and a single dictionary of all the encoders
# model = joblib.load("MODEL.pkl")
# all_encoders = joblib.load("ENCODER.pkl")

# # Use the loaded encoders by name from the dictionary
# owner_encoder = all_encoders['owner_encoder']
# fuel_encoder = all_encoders['fuel_encoder']
# brand_encoder = all_encoders['brand_encoder']
# model_encoder = all_encoders['model_encoder']
# transmission_encoder = all_encoders['transmission_encoder']
# seller_type_encoder = all_encoders['seller_type_encoder']

# st.title("Car Price Prediction")
# st.set_page_config(page_title="Car Price Prediction")
# st.write("Fill in the car details below")

# # Input fields
# owner_raw = st.selectbox("Owner", owner_encoder.classes_)
# brand_raw = st.selectbox("Car Brand", brand_encoder.classes_)
# model_raw = st.selectbox("Model", model_encoder.classes_)
# transmission_raw = st.selectbox("Transmission Type", transmission_encoder.classes_)
# fuel_raw = st.selectbox("Fuel Type", fuel_encoder.classes_)
# seller_type_raw = st.selectbox("Seller Type", seller_type_encoder.classes_)

# year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
# km_driven = st.number_input("Kilometers Driven", min_value=0)
# mileage = st.text_input("Mileage (e.g. 20 kmpl)")
# engine = st.text_input("Engine Capacity (e.g. 1248 CC)")
# max_power = st.text_input("Max Power (e.g. 74 bhp)")
# torque = st.text_input("Torque (e.g. 190Nm@ 2000rpm)")
# seats = st.number_input("Number of Seats", min_value=2, max_value=10)
# mean_rpm = st.number_input("RPM")


# # --- Predict button ---
# if st.button("Estimate Price"):
#     try:
#         # Encode categorical variables
#         owner_encoded = owner_encoder.transform([owner_raw])[0]
#         brand_encoded = brand_encoder.transform([brand_raw])[0]
#         model_encoded = model_encoder.transform([model_raw])[0]
#         transmission_encoded = transmission_encoder.transform([transmission_raw])[0]
#         fuel_encoded = fuel_encoder.transform([fuel_raw])[0]
#         seller_type_encoded = seller_type_encoder.transform([seller_type_raw])[0]

#         # Extract numeric values from text fields
#         def extract_number(text):
#             match = re.search(r'\d+\.?\d*', text)
#             return float(match.group()) if match else 0.0

#         mileage_val = extract_number(mileage)
#         engine_val = extract_number(engine)
#         max_power_val = extract_number(max_power)
#         torque_val = extract_number(torque) # Extract torque value here

#         # Prepare input data (ensure column order matches training data)
#         input_data = pd.DataFrame([{'owner': owner_encoded, 'brand': brand_encoded, 
#                                     'model': model_encoded, 
#                                     'transmission': transmission_encoded,
#                                     'fuel': fuel_encoded, 'year': year,
#                                     'seller_type': seller_type_encoded,
#                                     'km_driven': km_driven, 'seats': seats,
#                                     'mileage': mileage_val, # Use the extracted numeric value
#                                     'engine_size': engine_val,
#                                     'max_power_num': max_power_val,
#                                     'torque_Nm': torque_val, # Use the extracted numeric value
#                                     'mean_rpm': mean_rpm
#         }])

#         # Making the prediction
#         prediction = model.predict(input_data)[0]
#         st.success(f"Estimated Selling Price: ₹{round(prediction, 2)}")

#     except Exception as e:
#         st.error(f"Error making prediction: {e}")

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

    st.success(f"Estimated Selling Price: ${int(predicted_price):,}")