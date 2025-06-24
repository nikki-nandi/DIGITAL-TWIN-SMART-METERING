import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import pydeck as pdk
from tensorflow.keras.models import load_model
import joblib
import os

# Suppress oneDNN log clutter
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# App Config
st.set_page_config(page_title="Smart Meter Monitoring", layout="wide")
st.title("🔌 Real-Time Smart Meter Monitoring Dashboard")

# Sidebar Controls
refresh_rate = st.sidebar.slider("🔁 Refresh interval (sec)", 1, 10, 5)
data_choice = st.sidebar.radio("🧪 Data Source", ["Real", "Simulated"])

# Load Monitoring Data
DATA_PATH = "data/preprocessed_data.csv" if data_choice == "Real" else "data/simulated_data.csv"
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["RealtimeClockDateandTime"],
    dayfirst=True  # Set this to True if your CSV uses DD-MM-YYYY format
)

# Simulated Real-Time Loop
placeholder = st.empty()
i = 0

while True:
    with placeholder.container():
        data_batch = df.iloc[i:i+1]
        if data_batch.empty:
            st.success("✔️ Stream complete.")
            break

        row = data_batch.iloc[0]
        st.subheader(f"Meter ID: `{row['METERSNO']}` | Time: {row['RealtimeClockDateandTime']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔋 Voltage (V)", f"{row['Voltage']:.2f}")
        col2.metric("⚡ Phase Current (A)", f"{row['NormalPhaseCurrent']:.2f}")
        col3.metric("📊 Power Factor", f"{row['SystemPowerFactor']:.2f}")
        col4.metric("🌐 Frequency (Hz)", f"{row['Frequency']:.2f}")

        st.markdown("### 📈 Energy Usage")
        st.line_chart(df.iloc[:i+1][['BlockEnergykWh', 'CumulativeEnergykWh']])

        st.markdown("### 🗺️ Location")
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=row['Latitude'],
                longitude=row['Longitude'],
                zoom=12,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data_batch,
                    get_position='[Longitude, Latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=500,
                ),
            ],
        ))

    i += 1
    time.sleep(refresh_rate)

# ------------------------
# 🔮 Energy Forecast Section
# ------------------------

st.markdown("---")
st.title("📊 7-Day Smart Meter Energy Forecast")

try:
    # Load Model and Scalers
    model = load_model("models/final_energy_forecast_model.h5")
    feature_scaler = joblib.load("models/final_feature_scaler.pkl")
    target_scaler = joblib.load("models/final_target_scaler.pkl")

    # File Uploader
    uploaded_file = st.file_uploader("14_day_inp", type="csv")

    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        if new_data.shape[0] != 14:
            st.error("❌ Upload must contain exactly 14 rows (days) of input features.")
        else:
            # Predict
            input_scaled = feature_scaler.transform(new_data)
            input_seq = np.expand_dims(input_scaled, axis=0)

            pred = model.predict(input_seq)
            pred = pred.reshape(7, 2)
            pred_inv = target_scaler.inverse_transform(pred)

            # Display Prediction
            st.subheader("✅ Predicted Energy Usage (Next 7 Days)")
            forecast_df = pd.DataFrame(pred_inv, columns=["Predicted_kWh", "Predicted_kVAh"])
            forecast_df.index = [f"Day {i+1}" for i in range(7)]
            st.dataframe(forecast_df)

            st.line_chart(forecast_df)

except Exception as e:
    st.error(f"⚠️ Forecast model loading error: {e}")
