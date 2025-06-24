import streamlit as st
import pandas as pd
import time
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

# === CONFIG ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
EMAIL_ALERTS = True
ALERT_EMAIL = "nikithnandi2004@gmail.com"   # ✅ Receiver
EMAIL_SENDER = "nikithnandi08@gmail.com"    # ✅ Sender Gmail
EMAIL_PASSWORD = "sshz jpyi pibg jxev"       # ✅ Gmail app password

# Streamlit Page Setup
st.set_page_config(page_title="⚠️ Anomaly Monitoring", layout="wide")
st.title("📡 Smart Meter Monitoring & Anomaly Detection")

# Sidebar Alerts Box
st.sidebar.header("🚨 Live Alerts")
alert_box = st.sidebar.empty()

# File Upload or default data
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

# Anomaly Detection Function
def detect_row_anomalies(row):
    reasons = []
    if row['Voltage'] < 180 or row['Voltage'] > 250:
        reasons.append("Voltage anomaly")
    if row['SystemPowerFactor'] < 0.5:
        reasons.append("Low power factor")
    if row['ActivePower_kW'] < 0:
        reasons.append("Negative active power")
    if row['Frequency'] < 48.5 or row['Frequency'] > 51.5:
        reasons.append("Frequency anomaly")
    if row['BlockEnergykWh'] == 0:
        reasons.append("Zero energy consumption")
    return reasons

# Email Alert Function
def send_email_alert(anomalies):
    if not anomalies or not EMAIL_ALERTS:
        return
    msg = EmailMessage()
    msg["Subject"] = "⚠️ Smart Meter Anomaly Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = ALERT_EMAIL
    body = "\n".join(anomalies)
    msg.set_content(f"Anomalies Detected:\n\n{body}")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"❌ Email sending failed: {e}")
        return False

# Create daily log file
log_file = os.path.join(LOG_DIR, f"{datetime.today().strftime('%Y-%m-%d')}_realtime_log.txt")
anomaly_log = []

# Real-Time Monitoring Loop
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
        timestamp = row['RealtimeClockDateandTime']
        alert_lines = [f"[{timestamp}] - {reason}" for reason in anomalies]

        if anomalies:
            alert_box.error("\n".join(alert_lines))
            anomaly_log.extend(alert_lines)
            with open(log_file, "a") as log:
                for line in alert_lines:
                    log.write(line + "\n")
            send_email_alert(alert_lines)  # ✅ Send email immediately
        else:
            alert_box.success("✅ No anomalies detected")

        time.sleep(0.7)

# End of Monitoring
st.success("✅ Monitoring Completed")

if anomaly_log:
    st.subheader("📋 Final Anomaly Summary")
    st.code("\n".join(anomaly_log))

# Log Viewer Section
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

















################################################code 1#############################################################
# import streamlit as st
# import pandas as pd
# import time
# import os
# import smtplib
# from datetime import datetime
# from email.message import EmailMessage

# # === CONFIG ===
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)
# EMAIL_ALERTS = True
# ALERT_EMAIL = "nikithnandi2004@gmail.com"  # ✅ Replace with your recipient email
# EMAIL_SENDER = "nikithnandi08@gmail.com"  # ✅ Your Gmail
# EMAIL_PASSWORD = "sshz jpyi pibg jxev"     # ✅ App password only

# # Streamlit setup
# st.set_page_config(page_title="🔍 Real-Time Anomaly Monitoring", layout="wide")
# st.title("📡 Smart Meter Monitoring & Anomaly Detection")

# # Sidebar for real-time alerts
# st.sidebar.header("🚨 Live Alerts")
# alert_box = st.sidebar.empty()

# # File Upload or Default File
# uploaded_file = st.file_uploader("📂 Upload Smart Meter CSV", type="csv")
# if uploaded_file:
#     df = pd.read_csv(uploaded_file, parse_dates=["RealtimeClockDateandTime"])
# else:
#     DATA_PATH = "data/preprocessed_data.csv"
#     if os.path.exists(DATA_PATH):
#         df = pd.read_csv(DATA_PATH, parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
#     else:
#         st.warning("⚠️ Upload a file or add 'data/preprocessed_data.csv'")
#         st.stop()

# # --- Anomaly Detection Function ---
# def detect_row_anomalies(row):
#     reasons = []
#     if row['Voltage'] < 180 or row['Voltage'] > 250:
#         reasons.append("Voltage anomaly")
#     if row['SystemPowerFactor'] < 0.5:
#         reasons.append("Low power factor")
#     if row['ActivePower_kW'] < 0:
#         reasons.append("Negative active power")
#     if row['Frequency'] < 48.5 or row['Frequency'] > 51.5:
#         reasons.append("Frequency anomaly")
#     if row['BlockEnergykWh'] == 0:
#         reasons.append("Zero energy consumption")
#     return reasons

# # --- Email Sender ---
# def send_email_alert(anomalies):
#     if not anomalies or not EMAIL_ALERTS:
#         return
#     msg = EmailMessage()
#     msg["Subject"] = "⚠️ Smart Meter Anomaly Alert"
#     msg["From"] = EMAIL_SENDER
#     msg["To"] = ALERT_EMAIL
#     body = "\n".join(anomalies)
#     msg.set_content(f"Anomalies Detected:\n\n{body}")
#     try:
#         with smtplib.SMTP("smtp.gmail.com", 587) as server:
#             server.starttls()
#             server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#             server.send_message(msg)
#         return True
#     except Exception as e:
#         st.error(f"❌ Email sending failed: {e}")
#         return False

# # Log file path
# log_file = os.path.join(LOG_DIR, f"{datetime.today().strftime('%Y-%m-%d')}_realtime_log.txt")

# # --- Real-Time Monitoring Loop ---
# placeholder = st.empty()
# anomaly_log = []

# for i in range(len(df)):
#     with placeholder.container():
#         row = df.iloc[i]
#         st.subheader(f"⏱️ Time: {row['RealtimeClockDateandTime']}")
#         st.metric("🔋 Voltage", f"{row['Voltage']:.2f} V")
#         st.metric("⚡ Power (kW)", f"{row['ActivePower_kW']:.2f}")
#         st.metric("🔌 Power Factor", f"{row['SystemPowerFactor']:.2f}")
#         st.metric("🌐 Frequency", f"{row['Frequency']:.2f} Hz")

#         # Detect anomalies
#         anomalies = detect_row_anomalies(row)
#         ts = row['RealtimeClockDateandTime']
#         alert_lines = [f"[{ts}] - {reason}" for reason in anomalies]

#         if anomalies:
#             alert_box.error("\n".join(alert_lines))
#             anomaly_log.extend(alert_lines)
#             with open(log_file, "a") as log:
#                 for line in alert_lines:
#                     log.write(line + "\n")
#         else:
#             alert_box.success("✅ No anomalies detected")

#         time.sleep(0.7)

# # Final Message
# st.success("✅ Monitoring Completed")

# # Show all collected anomalies
# if anomaly_log:
#     st.subheader("📋 Final Anomaly Summary")
#     st.code("\n".join(anomaly_log))

#     if st.button("📧 Send Email Alert"):
#         sent = send_email_alert(anomaly_log)
#         if sent:
#             st.success("📨 Email sent successfully!")

# st.markdown("---")
# st.subheader("📂 View Past Anomaly Logs")
# if os.path.exists(LOG_DIR):
#     log_files = sorted(os.listdir(LOG_DIR), reverse=True)
#     selected_log = st.selectbox("Select log file", log_files)
#     if selected_log:
#         with open(os.path.join(LOG_DIR, selected_log)) as f:
#             st.code(f.read(), language="text")
# else:
#     st.info("No logs yet.")
