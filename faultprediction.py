# anomaly_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("models/anomaly_xgboost_model.pkl")
scaler = joblib.load("models/feature_scaler.pkl")

st.set_page_config(page_title="🔍 Anomaly Detector", layout="wide")
st.title("⚡ Smart Meter Anomaly Detection App")

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'])

    features = ['Voltage', 'SystemPowerFactor', 'ActivePower_kW', 'Frequency', 'BlockEnergykWh']

    if all(col in df.columns for col in features):
        # Preprocess input
        X = df[features]
        X_scaled = scaler.transform(X)

        # Predict anomalies
        df['Predicted'] = model.predict(X_scaled)
        df['Anomaly_Label'] = df['Predicted'].apply(lambda x: "🔴 Anomaly" if x == 1 else "🟢 Normal")

        # Show summary
        st.success(f"✅ Data Processed: {len(df)} records")
        st.write("### 📊 Sample Output", df.head())

        # Plot results
        st.write("### 🔍 Anomaly Timeline (Voltage)")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['RealtimeClockDateandTime'], df['Voltage'], label='Voltage')
        ax.scatter(df[df['Predicted'] == 1]['RealtimeClockDateandTime'],
                   df[df['Predicted'] == 1]['Voltage'],
                   color='red', label='Anomalies', s=10)
        ax.set_title("Voltage with Predicted Anomalies")
        ax.set_xlabel("Time")
        ax.set_ylabel("Voltage")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Download results
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", csv_download, "anomaly_predictions.csv", "text/csv")
    else:
        st.error("❌ Required columns missing: Ensure your CSV includes Voltage, SystemPowerFactor, ActivePower_kW, Frequency, BlockEnergykWh")
else:
    st.info("📁 Upload a CSV file with smart meter readings to begin.")
