import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("data/preprocessed_data.csv", parse_dates=["RealtimeClockDateandTime"], dayfirst=True)

# Drop columns that won't help in time-series forecasting
drop_cols = ["METERSNO", "MDkWTS", "MDkVATS", "TimeOfDay", "Latitude", "Longitude"]
df.drop(columns=drop_cols, inplace=True)

# Fill missing values (interpolate for time series)
df.interpolate(method='linear', inplace=True)

# Encode categorical time features
df["Weekday"] = df["Weekday"].astype(int)
df["Month"] = df["Month"].astype(int)

# Sort by time
df = df.sort_values("RealtimeClockDateandTime")

# Create lag features
for lag in range(1, 8):
    df[f"BlockEnergykWh_lag{lag}"] = df["BlockEnergykWh"].shift(lag)
    df[f"BlockEnergykVAh_lag{lag}"] = df["BlockEnergykVAh"].shift(lag)

# Drop rows with lag-induced NaNs
df.dropna(inplace=True)

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
scaled_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop("BlockEnergykWh")
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Save preprocessed data
df.to_csv("data/final_processed_data.csv", index=False)

# Quick check
print("✅ Preprocessing Done | Shape:", df.shape)
print(df[["BlockEnergykWh", "BlockEnergykWh_lag1", "BlockEnergykWh_lag7"]].head())




# import pandas as pd
# import os
# import numpy as np

# def preprocess_smart_meter_data(input_csv, output_csv):
#     # Ensure the input file exists
#     if not os.path.exists(input_csv):
#         raise FileNotFoundError(f"Input file not found: {input_csv}")

#     df = pd.read_csv(input_csv)

#     # Convert datetime with error handling
#     df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'], errors='coerce')
#     df.dropna(subset=['RealtimeClockDateandTime'], inplace=True)

#     df = df.sort_values('RealtimeClockDateandTime')
#     df.set_index('RealtimeClockDateandTime', inplace=True)

#     # Drop duplicates
#     df = df.drop_duplicates()

#     # Handle missing values
#     df = df.fillna(method='ffill').fillna(method='bfill')

#     # Convert METERSNO to categorical
#     if 'METERSNO' in df.columns:
#         df['METERSNO'] = df['METERSNO'].astype(str)

#     # Outlier removal
#     df = df[(df['Voltage'] > 100) & (df['Voltage'] < 300)]
#     df = df[(df['Frequency'] > 45) & (df['Frequency'] < 65)]

#     # Feature engineering
#     df['Hour'] = df.index.hour
#     df['Day'] = df.index.day
#     df['Weekday'] = df.index.weekday
#     df['Month'] = df.index.month

#     # Active power calculation
#     if all(col in df.columns for col in ['AvgPhaseCurrent', 'Voltage', 'Signedpowerfactor']):
#         df['ActivePower_kW'] = (df['AvgPhaseCurrent'] * df['Voltage'] * df['Signedpowerfactor']) / 1000

#     # Optional features
#     df['ReactivePower_kVAR'] = (df['AvgPhaseCurrent'] * df['Voltage'] * np.sqrt(1 - df['Signedpowerfactor']**2)) / 1000
#     df['ApparentPower_kVA'] = (df['AvgPhaseCurrent'] * df['Voltage']) / 1000

#     # Time of day classification
#     df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)

#     # Save processed file
#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df.to_csv(output_csv)
#     print(f"[INFO] Preprocessed data saved to {output_csv}")

# if __name__ == "__main__":
#     # Use relative path adjustment
#     input_csv = "../data/final_digital_twin.csv"
#     output_csv = "../data/preprocessed_data.csv"

#     # Debug: Print working directory
#     print("Current working directory:", os.getcwd())

#     preprocess_smart_meter_data(input_csv, output_csv)