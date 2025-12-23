import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Predict house price using a trained Machine Learning model.")

st.subheader("Enter House Details")
square_feet = st.number_input("Square Feet", min_value=100, step=10)
num_rooms = st.number_input("Number of Rooms", min_value=1, step=1)
age = st.number_input("House Age (years)", min_value=0, step=1)
distance = st.number_input("Distance to City (km)", min_value=0.0, step=0.5)
neighborhood = st.text_input("Neighborhood")
age_group = st.selectbox("Age Group", options=['New', 'Mid', 'Old'])
distance_group = st.selectbox("Distance Group", options=['Near', 'Medium', 'Far'])

if st.button("Predict Price 💰"):
    input_data = pd.DataFrame({
        "square_feet": [square_feet],
        "num_rooms": [num_rooms],
        "age": [age],
        "distance_to_city(km)": [distance],
        "neighborhood": [neighborhood] ,
        "age_group" : [age_group] ,
        "distance_group": [distance_group]

    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")