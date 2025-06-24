# import streamlit as st
# st.set_page_config(page_title="Energy Predictor", layout="centered")

# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model
# from io import BytesIO

# # === Load Models and Scalers ===
# @st.cache_resource
# def load_all_models():
#     day_model = load_model("models/energy_lstm_model.h5", compile=False)
#     month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
#     day_scaler = joblib.load("models/energy_scaler.pkl")
#     month_scaler = joblib.load("models/energy_scaler_month.pkl")
#     return day_model, month_model, day_scaler, month_scaler

# day_model, month_model, day_scaler, month_scaler = load_all_models()

# # === Styling ===
# st.markdown("""
#     <style>
#     body {
#         color: #ffffff;
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         background-color: #3a3f52;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5em 1em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # === App Header ===
# st.title("🔋 Energy Usage Predictor")
# st.write("Upload past energy usage data to forecast the next day or next 30 days.")

# # === Prediction Type ===
# prediction_type = st.radio("📊 Select Prediction Type:", ["Next Day", "Next Month"])

# # === File Upload ===
# expected_days = 7 if prediction_type == "Next Day" else 30
# uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (column: `KWHhh`)", type=["csv"])

# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         if 'KWHhh' not in df.columns:
#             st.error("❌ CSV must contain a `KWHhh` column.")
#         elif len(df) < expected_days:
#             st.error(f"❌ Please provide at least {expected_days} rows of data.")
#         else:
#             input_values = df['KWHhh'].values[-expected_days:]
#             st.success(f"✅ Uploaded {len(input_values)} records. Ready to predict.")

#             if st.button("🔮 Predict"):
#                 try:
#                     input_array = np.array(input_values).reshape(-1, 1)

#                     if prediction_type == "Next Day":
#                         input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
#                         prediction_scaled = day_model.predict(input_scaled)
#                         prediction = day_scaler.inverse_transform(prediction_scaled)[0][0]
#                         st.metric(label="📈 Predicted Energy (Tomorrow)", value=f"{prediction:.2f} kWh")

#                         # Download
#                         csv = pd.DataFrame({"Predicted kWh": [prediction]}).to_csv(index=False)
#                         st.download_button("📥 Download Result CSV", data=csv, file_name="next_day_prediction.csv", mime="text/csv")

#                     else:  # Next Month
#                         input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
#                         prediction_scaled = month_model.predict(input_scaled).reshape(30, 1)
#                         prediction = month_scaler.inverse_transform(prediction_scaled).flatten()

#                         st.success("📆 Predicted Energy for Next 30 Days")
#                         st.line_chart(prediction)

#                         # Download
#                         result_df = pd.DataFrame({
#                             "Day": [f"Day {i+1}" for i in range(30)],
#                             "Predicted kWh": prediction
#                         })
#                         csv = result_df.to_csv(index=False)
#                         st.download_button("📥 Download 30-Day Forecast", data=csv, file_name="next_month_prediction.csv", mime="text/csv")

#                 except Exception as e:
#                     st.error(f"❌ Prediction failed: {e}")
#     except Exception as e:
#         st.error(f"❌ Error reading file: {e}")

##################################################code 2###########################################################################
# import streamlit as st
# st.set_page_config(page_title="Energy Predictor", layout="centered")

# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model
# from io import BytesIO
# from datetime import datetime, timedelta

# # === Load Models and Scalers ===
# @st.cache_resource
# def load_all_models():
#     day_model = load_model("models/energy_lstm_model.h5", compile=False)
#     month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
#     day_scaler = joblib.load("models/energy_scaler.pkl")
#     month_scaler = joblib.load("models/energy_scaler_month.pkl")
#     return day_model, month_model, day_scaler, month_scaler

# day_model, month_model, day_scaler, month_scaler = load_all_models()

# # === Styling ===
# st.markdown("""
#     <style>
#     body {
#         color: #ffffff;
#         background-color: #0e1117;
#     }
#     .stButton>button {
#         background-color: #3a3f52;
#         color: white;
#         border-radius: 8px;
#         padding: 0.5em 1em;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # === App Header ===
# st.title("🔋 Energy Usage Predictor")
# st.write("Upload past energy usage data to forecast the next day or next 30 days.")

# # === Prediction Type ===
# prediction_type = st.radio("📊 Select Prediction Type:", ["Next Day", "Next Month"])

# # === File Upload ===
# expected_days = 7 if prediction_type == "Next Day" else 30
# uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (column: `KWHhh`)", type=["csv"])

# # === Sample CSV Download Option ===
# def generate_sample_csv(days=30):
#     sample_dates = [(datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
#     sample_values = np.random.uniform(5, 50, size=days)
#     sample_df = pd.DataFrame({
#         "Date": sample_dates,
#         "KWHhh": np.round(sample_values, 2)
#     })
#     return sample_df.to_csv(index=False)

# if st.button("📥 Download Sample CSV"):
#     sample_csv = generate_sample_csv(expected_days)
#     st.download_button("⬇️ Click to Download Sample CSV", data=sample_csv, file_name=f"sample_{expected_days}_days.csv", mime="text/csv")

# # === Prediction Logic ===
# if uploaded_file is not None:
#     try:
#         df = pd.read_csv(uploaded_file)
#         if 'KWHhh' not in df.columns:
#             st.error("❌ CSV must contain a `KWHhh` column.")
#         elif len(df) < expected_days:
#             st.error(f"❌ Please provide at least {expected_days} rows of data.")
#         else:
#             input_values = df['KWHhh'].values[-expected_days:]
#             st.success(f"✅ Uploaded {len(input_values)} records. Ready to predict.")

#             if st.button("🔮 Predict"):
#                 try:
#                     input_array = np.array(input_values).reshape(-1, 1)

#                     if prediction_type == "Next Day":
#                         input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
#                         prediction_scaled = day_model.predict(input_scaled)
#                         prediction = day_scaler.inverse_transform(prediction_scaled)[0][0]
#                         st.metric(label="📈 Predicted Energy (Tomorrow)", value=f"{prediction:.2f} kWh")

#                         # Download
#                         csv = pd.DataFrame({"Predicted kWh": [prediction]}).to_csv(index=False)
#                         st.download_button("📥 Download Result CSV", data=csv, file_name="next_day_prediction.csv", mime="text/csv")

#                     else:  # Next Month
#                         input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
#                         prediction_scaled = month_model.predict(input_scaled).reshape(30, 1)
#                         prediction = month_scaler.inverse_transform(prediction_scaled).flatten()

#                         st.success("📆 Predicted Energy for Next 30 Days")
#                         st.line_chart(prediction)

#                         # Download
#                         result_df = pd.DataFrame({
#                             "Day": [f"Day {i+1}" for i in range(30)],
#                             "Predicted kWh": prediction
#                         })
#                         csv = result_df.to_csv(index=False)
#                         st.download_button("📥 Download 30-Day Forecast", data=csv, file_name="next_month_prediction.csv", mime="text/csv")

#                 except Exception as e:
#                     st.error(f"❌ Prediction failed: {e}")
#     except Exception as e:
#         st.error(f"❌ Error reading file: {e}")
#################################code 3##########################################################################################
import streamlit as st
st.set_page_config(page_title="Energy Predictor", layout="centered")

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# === Load Models and Scalers ===
@st.cache_resource
def load_all_models():
    day_model = load_model("models/energy_lstm_model.h5", compile=False)
    month_model = load_model("models/energy_lstm_month_model.h5", compile=False)
    day_scaler = joblib.load("models/energy_scaler.pkl")
    month_scaler = joblib.load("models/energy_scaler_month.pkl")
    return day_model, month_model, day_scaler, month_scaler

day_model, month_model, day_scaler, month_scaler = load_all_models()

# === Styling ===
st.markdown("""
    <style>
    body {
        color: #ffffff;
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #3a3f52;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)

# === App Header ===
st.title("🔋 Energy Usage Predictor")
st.write("Upload past energy usage data to forecast the next day or next 30 days.")

# === Prediction Type ===
prediction_type = st.radio("📊 Select Prediction Type:", ["Next Day", "Next Month"])

# === File Upload ===
expected_days = 7 if prediction_type == "Next Day" else 30
uploaded_file = st.file_uploader(f"📄 Upload CSV with last {expected_days} daily energy values (column: `KWHhh`)", type=["csv"])

# === Sample CSV Download Option ===
def generate_sample_csv(days=30):
    sample_dates = [(datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
    sample_values = np.random.uniform(5, 50, size=days)
    sample_df = pd.DataFrame({
        "Date": sample_dates,
        "KWHhh": np.round(sample_values, 2)
    })
    return sample_df.to_csv(index=False)

if st.button("📥 Download Sample CSV"):
    sample_csv = generate_sample_csv(expected_days)
    st.download_button("⬇️ Click to Download Sample CSV", data=sample_csv, file_name=f"sample_{expected_days}_days.csv", mime="text/csv")

# === Prediction Logic ===
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
                try:
                    input_array = np.array(input_values).reshape(-1, 1)

                    if prediction_type == "Next Day":
                        input_scaled = day_scaler.transform(input_array).reshape(1, 7, 1)
                        prediction_scaled = day_model.predict(input_scaled)
                        prediction = day_scaler.inverse_transform(prediction_scaled)[0][0]
                        st.metric(label="📈 Predicted Energy (Tomorrow)", value=f"{prediction:.2f} kWh")

                        # Download
                        csv = pd.DataFrame({"Predicted kWh": [prediction]}).to_csv(index=False)
                        st.download_button("📥 Download Result CSV", data=csv, file_name="next_day_prediction.csv", mime="text/csv")

                    else:  # Next Month
                        input_scaled = month_scaler.transform(input_array).reshape(1, 30, 1)
                        prediction_scaled = month_model.predict(input_scaled).reshape(30, 1)
                        prediction = month_scaler.inverse_transform(prediction_scaled).flatten()

                        st.success("📆 Predicted Energy for Next 30 Days")
                        st.line_chart(prediction)

                        # Combine Previous + Predicted
                        previous_energy = df['KWHhh'].values[-30:]
                        result_df = pd.DataFrame({
                            "Day": [f"Day {i+1}" for i in range(30)],
                            "Previous Energy (kWh)": previous_energy,
                            "Predicted Energy (kWh)": prediction
                        })

                        csv = result_df.to_csv(index=False)
                        st.download_button("📥 Download 30-Day Forecast (With Previous)", data=csv, file_name="next_month_prediction.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
