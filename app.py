# ================================
# 🔹 IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================
# 🔹 LOAD MODEL & ENCODERS
# ================================
model = joblib.load("aqi_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# ================================
# 🔹 LOAD DATASET
# ================================
df = pd.read_csv("city_day.csv")

# ================================
# 🔹 TITLE
# ================================
st.title("🌆 Smart City Digital Twin - AQI System")

# ================================
# 🔹 CITY SELECTION
# ================================
city = st.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == city].copy()
city_df['Date'] = pd.to_datetime(city_df['Date'])

# ================================
# 🔹 DATE FILTER
# ================================
start_date = st.date_input("Start Date", city_df['Date'].min())
end_date = st.date_input("End Date", city_df['Date'].max())

filtered = city_df[
    (city_df['Date'] >= pd.to_datetime(start_date)) &
    (city_df['Date'] <= pd.to_datetime(end_date))
]

# ================================
# 🔹 AQI TREND
# ================================
st.subheader("📈 AQI Trend")

if not filtered.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(filtered['Date'], filtered['AQI'])
    ax.set_title(f"AQI Trend - {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("No data available")

# ================================
# 🔹 ML PREDICTION SECTION
# ================================
st.subheader("🏙️ AQI Prediction (ML Model)")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
    no2 = st.number_input("NO2", 0.0, 200.0, 40.0)
    so2 = st.number_input("SO2", 0.0, 200.0, 10.0)

with col2:
    pm10 = st.number_input("PM10", 0.0, 500.0, 80.0)
    co = st.number_input("CO", 0.0, 10.0, 1.0)
    o3 = st.number_input("O3", 0.0, 200.0, 30.0)

with col3:
    no = st.number_input("NO", 0.0, 200.0, 20.0)
    nox = st.number_input("NOx", 0.0, 300.0, 30.0)
    nh3 = st.number_input("NH3", 0.0, 200.0, 15.0)

if st.button("🔮 Predict AQI Category"):

    input_data = pd.DataFrame([[
        pm25, pm10, no2, co, so2, o3, no, nox, nh3
    ]], columns=['PM2.5','PM10','NO2','CO','SO2','O3','NO','NOx','NH3'])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    result = le.inverse_transform(prediction.reshape(-1))

    st.success(f"Predicted AQI Category: {result[0]}")

# ================================
# 🔹 OPENWEATHER API CONFIG
# ================================
API_KEY = "601305bad7bcfc09d34f378bcf3c96f2"

def get_live_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        return None, None

    aqi = data['list'][0]['main']['aqi']
    components = data['list'][0]['components']

    return aqi, components

# ================================
# 🔹 REVERSE GEOCODING (CITY NAME)
# ================================
def get_city_name(lat, lon):
    url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    # 🔥 DEBUG (optional but useful)
    st.write("DEBUG CITY API:", data)

    # ✅ check if response is valid list
    if isinstance(data, list) and len(data) > 0:
        return data[0].get("name", "Unknown City")

    return "Unknown City"
# ================================
# 🔹 LIVE AQI SECTION
# ================================
st.subheader("🌍 Live Air Quality (Real-Time)")

lat = st.number_input("Latitude", value=23.2599)
lon = st.number_input("Longitude", value=77.4126)

if st.button("📡 Get Live AQI"):

    city_name = get_city_name(lat, lon)
    aqi, comp = get_live_aqi(lat, lon)

    st.write(f"📍 City: {city_name}")

    if aqi is None:
        st.error("API Error: Unable to fetch data")
    else:
        st.success(f"Live AQI Level: {aqi}")
        st.json(comp)
