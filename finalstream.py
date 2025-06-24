# import streamlit as st
# import pandas as pd
# import numpy as np
# import pydeck as pdk
# import time
# import joblib
# import os
# import smtplib
# from datetime import datetime
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.message import EmailMessage
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

# # ========== CONFIG ==========
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# st.set_page_config(page_title="💡 Digital Twin - Smart Metering", layout="wide")
# st.title("💡 DIGITAL TWIN DASHBOARD FOR SMART METERING")

# # ========== SIDEBAR NAVIGATION ==========
# page = st.sidebar.selectbox("📂 Choose Module", [
#     "📡 Real-Time Monitoring",
#     "🔮 Energy Prediction",
#     "💰 Billing Estimation",
#     "🚨 Fault Detection",
#     "🔍 Anomaly Classification"
# ])

# # ========== MONITORING PAGE ==========
# if page == "📡 Real-Time Monitoring":
#     st.header("📡 Real-Time Smart Meter Monitoring")
#     refresh_rate = st.sidebar.slider("🔁 Refresh Interval (sec)", 1, 10, 5)
#     data_choice = st.sidebar.radio("🧪 Data Source", ["Real", "Simulated"])
#     path = "data/preprocessed_data.csv" if data_choice == "Real" else "data/simulated_data.csv"
#     df = pd.read_csv(path, parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
#     placeholder = st.empty()
#     i = 0

#     while True:
#         with placeholder.container():
#             data_batch = df.iloc[i:i+1]
#             if data_batch.empty:
#                 st.success("✔️ Stream complete.")
#                 break
#             row = data_batch.iloc[0]
#             st.subheader(f"Meter ID: `{row['METERSNO']}` | Time: {row['RealtimeClockDateandTime']}")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("🔋 Voltage (V)", f"{row['Voltage']:.2f}")
#             col2.metric("⚡ Current (A)", f"{row['NormalPhaseCurrent']:.2f}")
#             col3.metric("📊 Power Factor", f"{row['SystemPowerFactor']:.2f}")
#             col4.metric("🌐 Frequency (Hz)", f"{row['Frequency']:.2f}")
#             st.markdown("### 📈 Energy Usage")
#             st.line_chart(df.iloc[:i+1][['BlockEnergykWh', 'CumulativeEnergykWh']])
#             st.markdown("### 🗺️ Location")
#             st.pydeck_chart(pdk.Deck(
#                 map_style='mapbox://styles/mapbox/light-v9',
#                 initial_view_state=pdk.ViewState(
#                     latitude=row['Latitude'],
#                     longitude=row['Longitude'],
#                     zoom=12,
#                     pitch=50,
#                 ),
#                 layers=[pdk.Layer(
#                     'ScatterplotLayer',
#                     data=data_batch,
#                     get_position='[Longitude, Latitude]',
#                     get_color='[200, 30, 0, 160]',
#                     get_radius=500,
#                 )],
#             ))
#         i += 1
#         time.sleep(refresh_rate)

# # ========== ENERGY PREDICTION ==========
# elif page == "🔮 Energy Prediction":
#     st.header("🔮 Predict Energy Usage")
#     prediction_type = st.radio("📊 Select Prediction Type:", ["Next Day", "Next Month"])
#     expected_days = 7 if prediction_type == "Next Day" else 30
#     uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (KWHhh)", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         if 'KWHhh' in df.columns and len(df) >= expected_days:
#             input_values = df['KWHhh'].values[-expected_days:]
#             st.success(f"✅ Uploaded {len(input_values)} records. Ready to predict.")
#             if st.button("🔮 Predict"):
#                 day_model = load_model("models/energy_lstm_model.h5", compile=False)
#                 month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
#                 day_scaler = joblib.load("models/energy_scaler.pkl")
#                 month_scaler = joblib.load("models/energy_scaler_month.pkl")

#                 input_array = np.array(input_values).reshape(-1, 1)
#                 if prediction_type == "Next Day":
#                     input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
#                     prediction = day_model.predict(input_scaled)
#                     pred_value = day_scaler.inverse_transform(prediction)[0][0]
#                     st.metric("📈 Predicted Energy (Tomorrow)", f"{pred_value:.2f} kWh")
#                 else:
#                     input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
#                     pred = month_model.predict(input_scaled).reshape(30, 1)
#                     prediction = month_scaler.inverse_transform(pred).flatten()
#                     st.line_chart(prediction)
#         else:
#             st.error("❌ Invalid CSV format.")

# # ========== BILLING ESTIMATION ==========
# elif page == "💰 Billing Estimation":
#     st.header("💰 Bill Prediction & Alerts")
#     uploaded_file = st.file_uploader("📁 Upload Billing Input CSV", type="csv")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         df.rename(columns={"Amount_Billed": "Previous_Bill"}, inplace=True)
#         features = ['Energy_Consumption_KWh', 'Units_Consumed_KWh', 'Tariff_Per_KWh', 'Average_Daily_Consumption_KWh']
#         model = joblib.load("models/bill_predictor_model.pkl")
#         df['Predicted_Bill'] = model.predict(df[features])
#         df['Anomaly_Flag'] = df['Predicted_Bill'] > df['Previous_Bill'] * 1.2
#         st.dataframe(df[['Name', 'Email', 'Previous_Bill', 'Predicted_Bill', 'Anomaly_Flag']])

# # ========== FAULT DETECTION ==========
# elif page == "🚨 Fault Detection":
#     st.header("🚨 Fault & Anomaly Alerts")
#     uploaded_file = st.file_uploader("📂 Upload Smart Meter Data", type="csv")
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file, parse_dates=["RealtimeClockDateandTime"])
#         def detect_anomalies(row):
#             issues = []
#             if row['Voltage'] < 180 or row['Voltage'] > 250:
#                 issues.append("Voltage issue")
#             if row['SystemPowerFactor'] < 0.5:
#                 issues.append("Power factor low")
#             if row['ActivePower_kW'] < 0:
#                 issues.append("Negative power")
#             if row['Frequency'] < 48.5 or row['Frequency'] > 51.5:
#                 issues.append("Frequency deviation")
#             return issues

#         log = []
#         for i, row in df.iterrows():
#             anomalies = detect_anomalies(row)
#             if anomalies:
#                 log.append(f"[{row['RealtimeClockDateandTime']}] {'; '.join(anomalies)}")
#         st.code("\n".join(log) if log else "✅ No anomalies detected")

# # ========== ANOMALY CLASSIFIER ==========
# elif page == "🔍 Anomaly Classification":
#     st.header("🔍 ML-Based Anomaly Classification")
#     uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'])
#         features = ['Voltage', 'SystemPowerFactor', 'ActivePower_kW', 'Frequency', 'BlockEnergykWh']

#         if all(f in df.columns for f in features):
#             model = joblib.load("models/anomaly_xgboost_model.pkl")
#             scaler = joblib.load("models/feature_scaler.pkl")
#             X_scaled = scaler.transform(df[features])
#             df['Predicted'] = model.predict(X_scaled)
#             df['Label'] = df['Predicted'].map({0: '🟢 Normal', 1: '🔴 Anomaly'})

#             st.dataframe(df[['RealtimeClockDateandTime', 'Voltage', 'Label']])
#             fig, ax = plt.subplots(figsize=(12, 5))
#             ax.plot(df['RealtimeClockDateandTime'], df['Voltage'], label='Voltage')
#             ax.scatter(df[df['Predicted'] == 1]['RealtimeClockDateandTime'],
#                        df[df['Predicted'] == 1]['Voltage'], color='red', label='Anomalies', s=10)
#             ax.legend()
#             st.pyplot(fig)
#         else:
#             st.error("Missing columns in uploaded data.")
# ###############################################################################################################################################






# import streamlit as st
# import pandas as pd
# import numpy as np
# import pydeck as pdk
# import time
# import joblib
# import os
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model

# # ========== CONFIG ==========
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# st.set_page_config(page_title="💡 Digital Twin - Smart Metering", layout="wide")
# st.title("💡 DIGITAL TWIN DASHBOARD FOR SMART METERING")

# # ========== SIDEBAR NAVIGATION ==========
# page = st.sidebar.selectbox("📂 Choose Module", [
#     "🛁 Real-Time Monitoring",
#     "🔮 Energy Prediction",
#     "💰 Billing Estimation",
#     "🚨 Fault Detection",
#     "🔍 Anomaly Classification"
# ])

# # ========== MONITORING PAGE ==========
# if page == "🛁 Real-Time Monitoring":
#     st.header("🛁 Real-Time Smart Meter Monitoring")
#     refresh_rate = st.sidebar.slider("🔁 Refresh Interval (sec)", 1, 10, 5)
#     data_choice = st.sidebar.radio("🧪 Data Source", ["Real", "Simulated"])
#     path = "data/preprocessed_data.csv" if data_choice == "Real" else "data/simulated_data.csv"
#     df = pd.read_csv(path, parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
#     placeholder = st.empty()
#     i = 0

#     while True:
#         with placeholder.container():
#             data_batch = df.iloc[i:i+1]
#             if data_batch.empty:
#                 st.success("✔️ Stream complete.")
#                 break
#             row = data_batch.iloc[0]
#             st.subheader(f"Meter ID: `{row['METERSNO']}` | Time: {row['RealtimeClockDateandTime']}")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("🔋 Voltage (V)", f"{row['Voltage']:.2f}")
#             col2.metric("⚡ Current (A)", f"{row['NormalPhaseCurrent']:.2f}")
#             col3.metric("📊 Power Factor", f"{row['SystemPowerFactor']:.2f}")
#             col4.metric("🌐 Frequency (Hz)", f"{row['Frequency']:.2f}")
#             st.markdown("### 📈 Energy Usage")
#             st.line_chart(df.iloc[:i+1][['BlockEnergykWh', 'CumulativeEnergykWh']])
#             st.markdown("### 🗜️ Location")
#             st.pydeck_chart(pdk.Deck(
#                 map_style='mapbox://styles/mapbox/light-v9',
#                 initial_view_state=pdk.ViewState(
#                     latitude=row['Latitude'],
#                     longitude=row['Longitude'],
#                     zoom=12,
#                     pitch=50,
#                 ),
#                 layers=[pdk.Layer(
#                     'ScatterplotLayer',
#                     data=data_batch,
#                     get_position='[Longitude, Latitude]',
#                     get_color='[200, 30, 0, 160]',
#                     get_radius=500,
#                 )],
#             ))
#         i += 1
#         time.sleep(refresh_rate)

# # ========== ENERGY PREDICTION ==========
# elif page == "🔮 Energy Prediction":
#     st.header("🔮 Predict Energy Usage")

#     @st.cache_resource
#     def load_all_models():
#         day_model = load_model("models/energy_lstm_model.h5", compile=False)
#         month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
#         day_scaler = joblib.load("models/energy_scaler.pkl")
#         month_scaler = joblib.load("models/energy_scaler_month.pkl")
#         return day_model, month_model, day_scaler, month_scaler

#     day_model, month_model, day_scaler, month_scaler = load_all_models()

#     prediction_type = st.radio("📈 Select Prediction Type:", ["Next Day", "Next Month"])
#     expected_days = 7 if prediction_type == "Next Day" else 30
#     uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (column: `KWHhh`)", type=["csv"])

#     def generate_sample_csv(days=30):
#         sample_dates = [(datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
#         sample_values = np.random.uniform(5, 50, size=days)
#         sample_df = pd.DataFrame({
#             "Date": sample_dates,
#             "KWHhh": np.round(sample_values, 2)
#         })
#         return sample_df.to_csv(index=False)

#     if st.button("📅 Download Sample CSV"):
#         sample_csv = generate_sample_csv(expected_days)
#         st.download_button("⬇️ Click to Download Sample CSV", data=sample_csv, file_name=f"sample_{expected_days}_days.csv", mime="text/csv")

#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if 'KWHhh' not in df.columns:
#                 st.error("❌ CSV must contain a `KWHhh` column.")
#             elif len(df) < expected_days:
#                 st.error(f"❌ Please provide at least {expected_days} rows of data.")
#             else:
#                 input_values = df['KWHhh'].values[-expected_days:]
#                 st.success(f"✅ Uploaded {len(input_values)} records. Ready to predict.")

#                 if st.button("🔮 Predict"):
#                     input_array = np.array(input_values).reshape(-1, 1)

#                     if prediction_type == "Next Day":
#                         input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
#                         prediction_scaled = day_model.predict(input_scaled)
#                         prediction = day_scaler.inverse_transform(prediction_scaled)[0][0]
#                         st.metric(label="📈 Predicted Energy (Tomorrow)", value=f"{prediction:.2f} kWh")

#                         csv = pd.DataFrame({"Predicted kWh": [prediction]}).to_csv(index=False)
#                         st.download_button("📅 Download Result CSV", data=csv, file_name="next_day_prediction.csv", mime="text/csv")

#                     else:
#                         input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
#                         prediction_scaled = month_model.predict(input_scaled).reshape(30, 1)
#                         prediction = month_scaler.inverse_transform(prediction_scaled).flatten()

#                         st.success("📆 Predicted Energy for Next 30 Days")
#                         st.line_chart(prediction)

#                         previous_energy = df['KWHhh'].values[-30:]
#                         result_df = pd.DataFrame({
#                             "Day": [f"Day {i+1}" for i in range(30)],
#                             "Previous Energy (kWh)": previous_energy,
#                             "Predicted Energy (kWh)": prediction
#                         })

#                         csv = result_df.to_csv(index=False)
#                         st.download_button("📅 Download 30-Day Forecast (With Previous)", data=csv, file_name="next_month_prediction.csv", mime="text/csv")
#         except Exception as e:
#             st.error(f"❌ Error reading file: {e}")

# # ========== BILLING ESTIMATION ==========
# elif page == "💰 Billing Estimation":
#     st.header("💰 Bill Prediction & Alerts")
#     uploaded_file = st.file_uploader("Upload Billing Input CSV", type="csv")
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         df.rename(columns={"Amount_Billed": "Previous_Bill"}, inplace=True)
#         features = ['Energy_Consumption_KWh', 'Units_Consumed_KWh', 'Tariff_Per_KWh', 'Average_Daily_Consumption_KWh']
#         model = joblib.load("models/bill_predictor_model.pkl")
#         df['Predicted_Bill'] = model.predict(df[features])
#         df['Anomaly_Flag'] = df['Predicted_Bill'] > df['Previous_Bill'] * 1.2
#         st.dataframe(df[['Name', 'Email', 'Previous_Bill', 'Predicted_Bill', 'Anomaly_Flag']])

# # ========== FAULT DETECTION ==========
# elif page == "🚨 Fault Detection":
#     st.header("🚨 Fault & Anomaly Alerts")
#     uploaded_file = st.file_uploader("Upload Smart Meter Data", type="csv")
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file, parse_dates=["RealtimeClockDateandTime"])
#         def detect_anomalies(row):
#             issues = []
#             if row['Voltage'] < 180 or row['Voltage'] > 250:
#                 issues.append("Voltage issue")
#             if row['SystemPowerFactor'] < 0.5:
#                 issues.append("Power factor low")
#             if row['ActivePower_kW'] < 0:
#                 issues.append("Negative power")
#             if row['Frequency'] < 48.5 or row['Frequency'] > 51.5:
#                 issues.append("Frequency deviation")
#             return issues

#         log = []
#         for _, row in df.iterrows():
#             anomalies = detect_anomalies(row)
#             if anomalies:
#                 log.append(f"[{row['RealtimeClockDateandTime']}] {'; '.join(anomalies)}")
#         st.code("\n".join(log) if log else "✅ No anomalies detected")

# # ========== ANOMALY CLASSIFICATION ==========
# elif page == "🔍 Anomaly Classification":
#     st.header("🔍 ML-Based Anomaly Classification")
#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'])
#         features = ['Voltage', 'SystemPowerFactor', 'ActivePower_kW', 'Frequency', 'BlockEnergykWh']

#         if all(f in df.columns for f in features):
#             model = joblib.load("models/anomaly_xgboost_model.pkl")
#             scaler = joblib.load("models/feature_scaler.pkl")
#             X_scaled = scaler.transform(df[features])
#             df['Predicted'] = model.predict(X_scaled)
#             df['Label'] = df['Predicted'].map({0: '🟢 Normal', 1: '🔴 Anomaly'})

#             st.dataframe(df[['RealtimeClockDateandTime', 'Voltage', 'Label']])
#             fig, ax = plt.subplots(figsize=(12, 5))
#             ax.plot(df['RealtimeClockDateandTime'], df['Voltage'], label='Voltage')
#             ax.scatter(df[df['Predicted'] == 1]['RealtimeClockDateandTime'],
#                        df[df['Predicted'] == 1]['Voltage'], color='red', label='Anomalies', s=10)
#             ax.legend()
#             st.pyplot(fig)
#         else:
#             st.error("Missing columns in uploaded data.")


################################################################################################################################
#################################################main code######################################################################

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ========== CONFIG ==========
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
st.set_page_config(page_title="💡 Digital Twin - Smart Metering", layout="wide")
st.title("💡 DIGITAL TWIN DASHBOARD FOR SMART METERING")

# ========== SIDEBAR NAVIGATION ==========
page = st.sidebar.selectbox("📂 Choose Module", [
    "🛁 Real-Time Monitoring",
    "🔮 Energy Prediction",
    "💰 Billing Estimation",
    "🚨 Fault Detection",
    "🔍 Anomaly Classification"
])

# ========== MONITORING PAGE ==========
if page == "🛁 Real-Time Monitoring":
    st.header("🛁 Real-Time Smart Meter Monitoring")
    refresh_rate = st.sidebar.slider("🔁 Refresh Interval (sec)", 1, 10, 5)
    data_choice = st.sidebar.radio("🧪 Data Source", ["Real", "Simulated"])
    path = "data/preprocessed_data.csv" if data_choice == "Real" else "data/simulated_data.csv"
    df = pd.read_csv(path, parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
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
            col2.metric("⚡ Current (A)", f"{row['NormalPhaseCurrent']:.2f}")
            col3.metric("📊 Power Factor", f"{row['SystemPowerFactor']:.2f}")
            col4.metric("🌐 Frequency (Hz)", f"{row['Frequency']:.2f}")
            st.markdown("### 📈 Energy Usage")
            st.line_chart(df.iloc[:i+1][['BlockEnergykWh', 'CumulativeEnergykWh']])
            st.markdown("### 🗜️ Location")
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=row['Latitude'],
                    longitude=row['Longitude'],
                    zoom=12,
                    pitch=50,
                ),
                layers=[pdk.Layer(
                    'ScatterplotLayer',
                    data=data_batch,
                    get_position='[Longitude, Latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius=500,
                )],
            ))
        i += 1
        time.sleep(refresh_rate)

# ========== ENERGY PREDICTION ==========
elif page == "🔮 Energy Prediction":
    st.header("🔮 Predict Energy Usage")

    @st.cache_resource
    def load_all_models():
        day_model = load_model("models/energy_lstm_model.h5", compile=False)
        month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
        day_scaler = joblib.load("models/energy_scaler.pkl")
        month_scaler = joblib.load("models/energy_scaler_month.pkl")
        return day_model, month_model, day_scaler, month_scaler

    day_model, month_model, day_scaler, month_scaler = load_all_models()

    prediction_type = st.radio("📈 Select Prediction Type:", ["Next Day", "Next Month"])
    expected_days = 7 if prediction_type == "Next Day" else 30
    uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (column: `KWHhh`)", type=["csv"])

    def generate_sample_csv(days=30):
        sample_dates = [(datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
        sample_values = np.random.uniform(5, 50, size=days)
        sample_df = pd.DataFrame({
            "Date": sample_dates,
            "KWHhh": np.round(sample_values, 2)
        })
        return sample_df.to_csv(index=False)

    if st.button("📅 Download Sample CSV"):
        sample_csv = generate_sample_csv(expected_days)
        st.download_button("⬇️ Click to Download Sample CSV", data=sample_csv, file_name=f"sample_{expected_days}_days.csv", mime="text/csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'KWHhh' not in df.columns:
                st.error("❌ CSV must contain a `KWHhh` column.")
            elif len(df) < expected_days:
                st.error(f"❌ Please provide at least {expected_days} rows of data.")
            else:
                input_values = df['KWHhh'].values[-expected_days:]
                st.success(f"✅ Uploaded {len(input_values)} records. Ready to predict.")

                if st.button("🔮 Predict"):
                    input_array = np.array(input_values).reshape(-1, 1)

                    if prediction_type == "Next Day":
                        input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
                        prediction_scaled = day_model.predict(input_scaled)
                        prediction = day_scaler.inverse_transform(prediction_scaled)[0][0]
                        st.metric(label="📈 Predicted Energy (Tomorrow)", value=f"{prediction:.2f} kWh")

                        csv = pd.DataFrame({"Predicted kWh": [prediction]}).to_csv(index=False)
                        st.download_button("📅 Download Result CSV", data=csv, file_name="next_day_prediction.csv", mime="text/csv")

                    else:
                        input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
                        prediction_scaled = month_model.predict(input_scaled).reshape(30, 1)
                        prediction = month_scaler.inverse_transform(prediction_scaled).flatten()

                        st.success("📆 Predicted Energy for Next 30 Days")
                        st.line_chart(prediction)

                        previous_energy = df['KWHhh'].values[-30:]
                        result_df = pd.DataFrame({
                            "Day": [f"Day {i+1}" for i in range(30)],
                            "Previous Energy (kWh)": previous_energy,
                            "Predicted Energy (kWh)": prediction
                        })

                        csv = result_df.to_csv(index=False)
                        st.download_button("📅 Download 30-Day Forecast (With Previous)", data=csv, file_name="next_month_prediction.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ========== BILLING ESTIMATION ==========
elif page == "💰 Billing Estimation":
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    st.header("💰 Electricity Bill Prediction & Alert System")

    @st.cache_resource
    def load_model():
        return joblib.load("models/bill_predictor_model.pkl")

    model = load_model()

    def send_email(to_email, name, predicted, previous):
        sender_email = "nikithnandi08@gmail.com"
        sender_password = "sshz jpyi pibg jxev"
        subject = "⚠️ High Electricity Usage Alert"

        body = f"""
        Dear {name},

        Our system predicts that your next electricity bill will be ₹{predicted:.2f}, 
        which is higher than your previous bill of ₹{previous:.2f}.

        Please consider reducing your usage to avoid higher charges.

        Regards,
        Energy Monitoring Team
        """

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            st.success(f"✅ Email sent to {to_email}")
        except Exception as e:
            st.error(f"❌ Failed to send email to {to_email}: {e}")

    uploaded_file = st.file_uploader("📁 Upload your Training_Data.csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.rename(columns={"Amount_Billed": "Previous_Bill"}, inplace=True)

        features = [
            'Energy_Consumption_KWh',
            'Units_Consumed_KWh',
            'Tariff_Per_KWh',
            'Average_Daily_Consumption_KWh'
        ]

        df['Predicted_Bill'] = model.predict(df[features])
        df['Anomaly_Flag'] = df['Predicted_Bill'] > df['Previous_Bill'] * 1.20

        st.subheader("📊 Predicted Results")
        st.dataframe(df[['Name', 'Email', 'Previous_Bill', 'Predicted_Bill', 'Anomaly_Flag']])

        anomalies = df[df['Anomaly_Flag'] == True]
        st.subheader("🚨 Anomalies Detected (High Usage)")
        st.dataframe(anomalies[['Name', 'Email', 'Previous_Bill', 'Predicted_Bill']])

        if st.button("📧 Send Alerts to All Anomalous Customers"):
            for _, row in anomalies.iterrows():
                if pd.notna(row['Email']) and pd.notna(row['Name']):
                    send_email(
                        to_email=row['Email'],
                        name=row['Name'],
                        predicted=row['Predicted_Bill'],
                        previous=row['Previous_Bill']
                    )

# ========== FAULT DETECTION ==========
# ========== FAULT DETECTION ==========
elif page == "🚨 Fault Detection":
    import time
    import os
    import smtplib
    from datetime import datetime
    from email.message import EmailMessage

    st.header("📡 Smart Meter Monitoring & Anomaly Detection")

    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    EMAIL_ALERTS = True
    ALERT_EMAIL = "nikithnandi2004@gmail.com"   # ✅ Receiver
    EMAIL_SENDER = "nikithnandi08@gmail.com"    # ✅ Sender Gmail
    EMAIL_PASSWORD = "sshz jpyi pibg jxev"       # ✅ Gmail App Password

    st.sidebar.header("🚨 Live Alerts")
    alert_box = st.sidebar.empty()

    uploaded_file = st.file_uploader("📂 Upload Smart Meter CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["RealtimeClockDateandTime"])
    else:
        DATA_PATH = "data/preprocessed_data.csv"
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH, parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
        else:
            st.warning("⚠️ Upload a file or add 'data/preprocessed_data.csv'")
            st.stop()

    # Function to detect anomalies
    def detect_row_anomalies(row):
        issues = []
        if row['Voltage'] < 180 or row['Voltage'] > 250:
            issues.append("Voltage anomaly")
        if row['SystemPowerFactor'] < 0.5:
            issues.append("Low power factor")
        if row['ActivePower_kW'] < 0:
            issues.append("Negative active power")
        if row['Frequency'] < 48.5 or row['Frequency'] > 51.5:
            issues.append("Frequency anomaly")
        if row['BlockEnergykWh'] == 0:
            issues.append("Zero energy consumption")
        return issues

    # Function to send email
    def send_email_alert(anomalies):
        if not anomalies or not EMAIL_ALERTS:
            return
        msg = EmailMessage()
        msg["Subject"] = "⚠️ Smart Meter Anomaly Alert"
        msg["From"] = EMAIL_SENDER
        msg["To"] = ALERT_EMAIL
        msg.set_content("Anomalies Detected:\n\n" + "\n".join(anomalies))
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            return True
        except Exception as e:
            st.error(f"❌ Email sending failed: {e}")
            return False

    log_file = os.path.join(LOG_DIR, f"{datetime.today().strftime('%Y-%m-%d')}_realtime_log.txt")
    anomaly_log = []

    placeholder = st.empty()
    for i in range(len(df)):
        with placeholder.container():
            row = df.iloc[i]
            st.subheader(f"⏱️ Time: {row['RealtimeClockDateandTime']}")
            st.metric("🔋 Voltage", f"{row['Voltage']:.2f} V")
            st.metric("⚡ Power (kW)", f"{row['ActivePower_kW']:.2f}")
            st.metric("🔌 Power Factor", f"{row['SystemPowerFactor']:.2f}")
            st.metric("🌐 Frequency", f"{row['Frequency']:.2f} Hz")

            anomalies = detect_row_anomalies(row)
            alert_lines = [f"[{row['RealtimeClockDateandTime']}] - {reason}" for reason in anomalies]

            if anomalies:
                alert_box.error("\n".join(alert_lines))
                anomaly_log.extend(alert_lines)
                with open(log_file, "a") as f:
                    f.write("\n".join(alert_lines) + "\n")
                send_email_alert(alert_lines)
            else:
                alert_box.success("✅ No anomalies detected")

            time.sleep(0.7)

    st.success("✅ Monitoring Completed")

    if anomaly_log:
        st.subheader("📋 Final Anomaly Summary")
        st.code("\n".join(anomaly_log))

    st.markdown("---")
    st.subheader("📂 View Past Anomaly Logs")
    if os.path.exists(LOG_DIR):
        log_files = sorted(os.listdir(LOG_DIR), reverse=True)
        selected_log = st.selectbox("📄 Select log file", log_files)
        if selected_log:
            with open(os.path.join(LOG_DIR, selected_log)) as f:
                st.code(f.read(), language="text")
    else:
        st.info("No logs found.")

# ========== ANOMALY CLASSIFICATION ==========
# ========== ANOMALY CLASSIFICATION ==========
elif page == "🔍 Anomaly Classification":
    st.header("🔍 ML-Based Anomaly Classification Using XGBoost")

    # Upload CSV
    uploaded_file = st.file_uploader("📂 Upload CSV with Smart Meter Data", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'])

        # Required feature columns
        features = ['Voltage', 'SystemPowerFactor', 'ActivePower_kW', 'Frequency', 'BlockEnergykWh']

        if all(col in df.columns for col in features):
            # Load model and scaler
            model = joblib.load("models/anomaly_xgboost_model.pkl")
            scaler = joblib.load("models/feature_scaler.pkl")

            # Scale features
            X_scaled = scaler.transform(df[features])

            # Predict anomalies
            df['Predicted'] = model.predict(X_scaled)
            df['Anomaly_Label'] = df['Predicted'].apply(lambda x: "🔴 Anomaly" if x == 1 else "🟢 Normal")

            # Output Preview
            st.success(f"✅ Classification Completed: {len(df)} records")
            st.dataframe(df[['RealtimeClockDateandTime', 'Voltage', 'SystemPowerFactor', 'Anomaly_Label']].head(10))

            # Plotting
            st.write("### 📈 Voltage Over Time with Anomalies")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df['RealtimeClockDateandTime'], df['Voltage'], label='Voltage', linewidth=1)
            ax.scatter(df[df['Predicted'] == 1]['RealtimeClockDateandTime'],
                       df[df['Predicted'] == 1]['Voltage'],
                       color='red', label='Anomaly', s=20)
            ax.set_title("Smart Meter Voltage with Anomaly Detection")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Voltage (V)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # CSV download
            result_csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Anomaly Classification CSV",
                               data=result_csv,
                               file_name="anomaly_classification_results.csv",
                               mime="text/csv")
        else: 
            st.error("❌ Required columns missing. Your CSV must include:")
            st.code(", ".join(features))
    else:
        st.info("📁 Please upload a CSV file to begin classification.")
