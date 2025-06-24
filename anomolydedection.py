import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# STEP 1: Load the Data
df = pd.read_csv("generated_anomaly_training_data.csv")
df['RealtimeClockDateandTime'] = pd.to_datetime(df['RealtimeClockDateandTime'])

# STEP 2: Feature Selection
features = ['Voltage', 'SystemPowerFactor', 'ActivePower_kW', 'Frequency', 'BlockEnergykWh']
X = df[features]
y = df['Anomaly']

# STEP 3: Normalize the Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# STEP 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 5: Train XGBoost Classifier
model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# STEP 6: Predictions & Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_prob))

print(f"\nAccuracy: {accuracy:.4f}")
print(f"R² Score (based on probabilities): {r2:.4f}")
print(f"RMSE (based on probabilities): {rmse:.4f}")

# STEP 7: Predict All
df['predicted'] = model.predict(X_scaled)

# STEP 8: Voltage Anomaly Plot
plt.figure(figsize=(14, 6))
plt.plot(df['RealtimeClockDateandTime'], df['Voltage'], label='Voltage')
plt.scatter(df[df['predicted'] == 1]['RealtimeClockDateandTime'],
            df[df['predicted'] == 1]['Voltage'],
            color='red', label='Predicted Anomaly', s=10)
plt.title("Voltage with Predicted Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Voltage")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# STEP 9: Actual vs Predicted Anomalies
plt.figure(figsize=(14, 6))
plt.plot(df['RealtimeClockDateandTime'], df['Anomaly'], label='Actual Anomaly', linestyle='--')
plt.plot(df['RealtimeClockDateandTime'], df['predicted'], label='Predicted Anomaly')
plt.title("Actual vs Predicted Anomalies")
plt.xlabel("Timestamp")
plt.ylabel("Anomaly")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# STEP 10: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# STEP 11: Save the Model
joblib.dump(model, 'anomaly_xgboost_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
