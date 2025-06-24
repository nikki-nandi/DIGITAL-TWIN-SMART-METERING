import streamlit as st
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/bill_predictor_model.pkl")

model = load_model()

# Send email function
def send_email(to_email, name, predicted, previous):

    sender_email = "nikithnandi08@gmail.com"
    sender_password = "sshz jpyi pibg jxev"  # Use App Password
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

# Streamlit UI
st.title("🔌 Electricity Bill Prediction & Alert System")

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

    # Predict future bills
    df['Predicted_Bill'] = model.predict(df[features])

    # Flag anomalies
    df['Anomaly_Flag'] = df['Predicted_Bill'] > df['Previous_Bill'] * 1.20

    st.subheader("📊 Predicted Results")
    st.dataframe(df[['Name', 'Email', 'Previous_Bill', 'Predicted_Bill', 'Anomaly_Flag']])

    # Only show anomalous rows
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
