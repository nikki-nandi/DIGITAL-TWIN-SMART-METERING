import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load dataset
df = pd.read_csv("Training_Data.csv")

# Optional: Rename for clarity
df.rename(columns={"Amount_Billed": "Previous_Bill"}, inplace=True)

# Step 2: Select features and target
features = [
    'Energy_Consumption_KWh',
    'Units_Consumed_KWh',
    'Tariff_Per_KWh',
    'Average_Daily_Consumption_KWh'
]
target = 'Projected_Bill'

X = df[features]
y = df[target]

# Step 3: Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 RMSE: {rmse:.4f}")
print(f"📈 R² Score: {r2:.5f}")

# Step 7: Save the model for reuse
joblib.dump(model, "bill_predictor_model.pkl")
print("💾 Model saved as 'bill_predictor_model.pkl'")
import matplotlib.pyplot as plt

# Step 6.1: Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='dodgerblue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel("Actual Projected Bill")
plt.ylabel("Predicted Projected Bill")
plt.title("Actual vs Predicted Electricity Bill")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
