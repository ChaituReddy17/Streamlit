import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Title
st.title("✈️ Flight Passenger Satisfaction Predictor")

st.markdown("Provide passenger details to predict satisfaction.")

# Collect input features
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.slider("Age", 7, 85, 30)
travel_type = st.selectbox("Type of Travel", ["Business Travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_distance = st.number_input("Flight Distance (miles)", 30, 5000, 500)

wifi = st.slider("Inflight Wifi Service (0-5)", 0, 5, 3)
online_booking = st.slider("Ease of Online Booking (0-5)", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort (0-5)", 0, 5, 3)
inflight_entertainment = st.slider("Inflight Entertainment (0-5)", 0, 5, 3)
onboard_service = st.slider("On-board Service (0-5)", 0, 5, 3)
cleanliness = st.slider("Cleanliness (0-5)", 0, 5, 3)
departure_delay = st.slider("Departure Delay (in minutes)", 0, 1000, 0)
arrival_delay = st.slider("Arrival Delay (in minutes)", 0, 1000, 0)

# Create a DataFrame for prediction (must match your training features!)
input_data = pd.DataFrame({
    'Age': [age],
    'Flight Distance': [flight_distance],
    'Inflight wifi service': [wifi],
    'Ease of Online booking': [online_booking],
    'Seat comfort': [seat_comfort],
    'Inflight entertainment': [inflight_entertainment],
    'On-board service': [onboard_service],
    'Cleanliness': [cleanliness],
    'Departure Delay in Minutes': [departure_delay],
    'Arrival Delay in Minutes': [arrival_delay],

    # One-hot encoded features
    'Gender_male': [1 if gender == 'Male' else 0],
    'Customer Type_loyal customer': [1 if customer_type.lower() == 'loyal customer' else 0],
    'Type of Travel_Personal Travel': [1 if travel_type.lower() == 'personal travel' else 0],
    'Class_Eco': [1 if travel_class == 'Eco' else 0],
    'Class_Eco Plus': [1 if travel_class == 'Eco Plus' else 0]
})

# Prediction
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Passenger is **Satisfied**.")
    else:
        st.error("❌ Passenger is **Not Satisfied**.")
