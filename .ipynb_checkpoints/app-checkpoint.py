import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ================================
# 🔹 LOAD MODEL & FILES
# ================================
model = joblib.load("aqi_model.pkl")   # saved with joblib
scaler = joblib.load("scaler.pkl")
le = pickle.load(open("label_encoder.pkl", "rb"))

# ================================
# 🔹 LOAD DATASET
# ================================
df = pd.read_csv("city_day.csv")

# ================================
# 🔹 TITLE
# ================================
st.title("🌆 Smart City Digital Twin - AQI Prediction")

# ================================
# 🔹 1. CITY SELECTION
# ================================
city = st.selectbox("Select City", df['City'].unique())

city_df = df[df['City'] == city].copy()
city_df['Date'] = pd.to_datetime(city_df['Date'])

# ================================
# 🔹 2. DATE FILTER
# ================================
start_date = st.date_input("Start Date", city_df['Date'].min())
end_date = st.date_input("End Date", city_df['Date'].max())

filtered = city_df[
    (city_df['Date'] >= pd.to_datetime(start_date)) &
    (city_df['Date'] <= pd.to_datetime(end_date))
]

# ================================
# 🔹 3. GRAPH (DASHBOARD)
# ================================
st.subheader("📈 AQI Trend")

if filtered.empty:
    st.warning("No data available for selected date range")
else:
    fig, ax = plt.subplots(figsize=(18,10))

    ax.plot(filtered['Date'], filtered['AQI'])

    # X-axis formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel("AQI", fontsize=16)
    ax.set_title(f"AQI Trend for {city}", fontsize=18)

    plt.xticks(rotation=45)

    st.pyplot(fig)

# ================================
# 🔹 4. USER INPUT (9 FEATURES)
# ================================
st.subheader("🔮 Predict AQI Category")

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

# ================================
# 🔹 5. PREDICTION
# ================================
if st.button("Predict AQI Category"):

    input_data = pd.DataFrame(
        [[pm25, pm10, no2, co, so2, o3, no, nox, nh3]],
        columns=['PM2.5','PM10','NO2','CO','SO2','O3','NO','NOx','NH3']
    )

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted AQI Category: {result[0]}")

# ================================
# 🔹 6. DIGITAL TWIN EXPLANATION
# ================================
st.subheader("🧠 Digital Twin Concept")

st.write("""
This system acts as a Digital Twin of urban air quality.

- Real-world data (pollution levels)
- Virtual model (machine learning)
- Prediction (future AQI category)

It helps simulate and analyze environmental conditions in smart cities.
""")