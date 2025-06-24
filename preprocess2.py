# preprocess2.py

# ---------------------------------------------------------------------------
# STEP 0 — Import libraries
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# STEP 1 — Loading the data
# ---------------------------------------------------------------------------
# File path to your CSV
file_path = r"C:\Users\gani_\OneDrive\Desktop\final_project\data\cleaned.csv"

# Loading CSV with parsed dates
df = pd.read_csv(file_path, parse_dates=['DateTime'])

# ---------------------------------------------------------------------------
# STEP 2 — Handle missing values
# ---------------------------------------------------------------------------
# Fill missing values forward
df = df.sort_values(by='DateTime')
df = df.ffill().bfill()

# ---------------------------------------------------------------------------
# STEP 3 — Filter out non-numeric columns
# ---------------------------------------------------------------------------
# Keep only numerical columns alongside DateTime
numerical_df = df.copy()
numerical_df = numerical_df.set_index('DateTime')
numerical_df = numerical_df.select_dtypes(include='number')

# ---------------------------------------------------------------------------
# STEP 4 — Resampling to Daily Average
# ---------------------------------------------------------------------------
resampled = numerical_df.resample('D').mean().reset_index()

# ---------------------------------------------------------------------------
# STEP 5 — Time-Based Features
# ---------------------------------------------------------------------------
resampled = resampled.copy()
resampled['hour'] = resampled['DateTime'].dt.hour
resampled['day'] = resampled['DateTime'].dt.day
resampled['weekday'] = resampled['DateTime'].dt.weekday
resampled['month'] = resampled['DateTime'].dt.month

# ---------------------------------------------------------------------------
# STEP 6 — Lag Features
# ---------------------------------------------------------------------------
for lag in [1, 7, 30]:
    for col in numerical_df.columns:
        resampled[f'{col}_lag{lag}'] = resampled[col].shift(lag)

# ---------------------------------------------------------------------------
# STEP 7 — Window (Moving Average) Features
# ---------------------------------------------------------------------------
for col in numerical_df.columns:
    resampled[f'{col}_moving_average_7d'] = resampled[col].rolling(window=7).mean()

# ---------------------------------------------------------------------------
# STEP 8 — Handle missing after transformation
# ---------------------------------------------------------------------------
resampled = resampled.ffill().bfill()

# ---------------------------------------------------------------------------
# STEP 9 — Normalize or Scale (Optionally)
# ---------------------------------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(resampled.select_dtypes(include='number'))
resampled_scaled = resampled.copy()
resampled_scaled[resampled.select_dtypes(include='number').columns] = scaler.transform(
    resampled.select_dtypes(include='number')
)

# ---------------------------------------------------------------------------
# STEP 10 — Save Preprocessed Data
# ---------------------------------------------------------------------------
resampled_scaled.to_csv(r"C:\Users\gani_\OneDrive\Desktop\final_project\data\cleaned_preprocessed.csv", index=False)

print("Preprocessing completed successfully.")
print(r"C:\Users\gani_\OneDrive\Desktop\final_project\data\cleaned_preprocessed.csv")