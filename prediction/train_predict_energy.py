# !pip install keras-tuner --quiet
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import keras_tuner as kt
# import joblib
# import random
# import os
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from keras.models import Model
# from keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional,
#                           MultiHeadAttention, LayerNormalization,
#                           Add, Reshape, TimeDistributed)
# from keras.callbacks import EarlyStopping

# # 1. Set seed
# def set_seed(seed=42):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     random.seed(seed)
# set_seed()

# # 2. Load Data
# df = pd.read_csv("final_processed_data.csv", parse_dates=["RealtimeClockDateandTime"], dayfirst=True)
# df['Date'] = df['RealtimeClockDateandTime'].dt.date
# daily = df.groupby('Date')[['BlockEnergykWh', 'BlockEnergykVAh']].sum().reset_index()
# daily.rename(columns={'BlockEnergykWh': 'Daily_kWh', 'BlockEnergykVAh': 'Daily_kVAh'}, inplace=True)

# # 3. Feature Engineering (Added More!)
# daily['Lag_kWh'] = daily['Daily_kWh'].shift(1)
# daily['Lag_kVAh'] = daily['Daily_kVAh'].shift(1)
# daily['Rolling_kWh_3'] = daily['Daily_kWh'].rolling(3).mean()
# daily['Rolling_kVAh_3'] = daily['Daily_kVAh'].rolling(3).mean()
# daily['Rolling_kWh_std'] = daily['Daily_kWh'].rolling(3).std()
# daily['Energy_Ratio'] = daily['Daily_kWh'] / (daily['Daily_kVAh'] + 1e-5)
# daily['Weekday'] = pd.to_datetime(daily['Date']).dt.weekday
# daily['kWh_diff'] = daily['Daily_kWh'].diff()
# daily['7D_avg_kWh'] = daily['Daily_kWh'].rolling(window=7).mean()
# daily['3D_avg_diff'] = daily['kWh_diff'].rolling(3).mean()
# daily['IsWeekend'] = daily['Weekday'].isin([5,6]).astype(int)
# daily['kVAh_diff'] = daily['Daily_kVAh'].diff()
# daily['7D_std_kWh'] = daily['Daily_kWh'].rolling(window=7).std()

# daily.dropna(inplace=True)

# # 4. Scaling
# FEATURE_COLS = ['Daily_kWh', 'Daily_kVAh', 'Lag_kWh', 'Lag_kVAh', 'Rolling_kWh_3',
#                 'Rolling_kVAh_3', 'Rolling_kWh_std', 'Energy_Ratio', 'Weekday',
#                 'kWh_diff', '7D_avg_kWh', '3D_avg_diff', 'IsWeekend', 'kVAh_diff', '7D_std_kWh']

# scaler = MinMaxScaler()
# scaled_features = pd.DataFrame(scaler.fit_transform(daily[FEATURE_COLS]), columns=[f'Scaled_{col}' for col in FEATURE_COLS])
# joblib.dump(scaler, "final_feature_scaler.pkl")

# target_scaler = MinMaxScaler()
# target_scaled = target_scaler.fit_transform(daily[['Daily_kWh', 'Daily_kVAh']])
# scaled_features['Scaled_Daily_kWh'] = target_scaled[:, 0]
# scaled_features['Scaled_Daily_kVAh'] = target_scaled[:, 1]
# joblib.dump(target_scaler, "final_target_scaler.pkl")

# # 5. Create Sequences
# def create_sequences(data, input_len=14, output_len=7):
#     X, y = [], []
#     for i in range(len(data) - input_len - output_len):
#         X.append(data.iloc[i:i+input_len].drop(columns=['Scaled_Daily_kWh', 'Scaled_Daily_kVAh']).values)
#         y.append(data[['Scaled_Daily_kWh', 'Scaled_Daily_kVAh']].iloc[i+input_len:i+input_len+output_len].values)
#     return np.array(X), np.array(y)

# SEQ_LEN, OUT_LEN = 14, 7
# X, y = create_sequences(scaled_features, SEQ_LEN, OUT_LEN)
# split = int(0.8 * len(X))
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # 6. Build Model
# def build_model(hp):
#     inputs = Input(shape=(SEQ_LEN, X.shape[2]))
#     x = Bidirectional(LSTM(hp.Int('lstm1', 64, 128, step=32), return_sequences=True))(inputs)
#     x = Dropout(hp.Float('dropout1', 0.2, 0.5))(x)

#     attn = MultiHeadAttention(num_heads=hp.Int('heads', 2, 8), key_dim=hp.Choice('key_dim', [8, 16]))(x, x)
#     attn = LayerNormalization()(attn)
#     x = Add()([x, attn])

#     x = LSTM(hp.Int('lstm2', 32, 64, step=16), return_sequences=True)(x)
#     x = Dropout(hp.Float('dropout2', 0.2, 0.4))(x)
#     x = TimeDistributed(Dense(2))(x)
#     x = x[:, -OUT_LEN:, :]
#     out = Reshape((OUT_LEN * 2,))(x)

#     model = Model(inputs, out)
#     model.compile(optimizer='adam', loss='huber')
#     return model

# # 7. Tune Hyperparameters
# tuner = kt.Hyperband(build_model,
#                      objective='val_loss',
#                      max_epochs=50,
#                      factor=3,
#                      directory='tune_dir',
#                      project_name='final_energy_forecast')

# early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# tuner.search(X_train, y_train.reshape(-1, OUT_LEN * 2),
#              validation_data=(X_test, y_test.reshape(-1, OUT_LEN * 2)),
#              epochs=50,
#              callbacks=[early])

# # 8. Train Final Model
# best_model = tuner.get_best_models(1)[0]
# history = best_model.fit(
#     X_train, y_train.reshape(-1, OUT_LEN * 2),
#     validation_data=(X_test, y_test.reshape(-1, OUT_LEN * 2)),
#     epochs=100,
#     batch_size=16,
#     callbacks=[early]
# )

# # 9. Evaluate
# y_pred = best_model.predict(X_test).reshape(-1, OUT_LEN, 2)
# mse = mean_squared_error(y_test.reshape(-1, 2), y_pred.reshape(-1, 2))
# r2 = r2_score(y_test.reshape(-1, 2), y_pred.reshape(-1, 2))
# print(f"✅ MSE: {mse:.6f}")
# print(f"✅ R² Score: {r2:.6f}")

# # 10. Plot
# plt.figure(figsize=(14, 6))
# plt.plot(y_test[:, :, 0].flatten(), label='Actual kWh')
# plt.plot(y_pred[:, :, 0].flatten(), label='Predicted kWh')
# plt.plot(y_test[:, :, 1].flatten(), label='Actual kVAh', linestyle='--')
# plt.plot(y_pred[:, :, 1].flatten(), label='Predicted kVAh', linestyle='--')
# plt.title("🔮 Forecast: Actual vs Predicted (7 Days)")
# plt.xlabel("Days")
# plt.ylabel("Scaled Energy Values")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # 11. Save
# best_model.save("final_energy_forecast_model.h5")
# print("📦 Saved model to models/final_energy_forecast_model.h5")








#---------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import joblib

# === Step 1: Load and preprocess data ===
df = pd.read_csv('cleaned.csv', parse_dates=['DateTime'], index_col='DateTime')
df = df[['KWHhh']]
df = df.resample('D').sum().dropna()  # Daily energy consumption

# === Step 2: Scale ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === Step 3: Create sliding window sequences ===
def create_multi_step_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps:i + input_steps + output_steps])
    return np.array(X), np.array(y)

lookback = 30  # 30 days input
forecast_horizon = 30  # predict next 30 days
X, y = create_multi_step_sequences(scaled_data, lookback, forecast_horizon)

# === Step 4: Train-Test Split ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === Step 5: Reshape for LSTM ===
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Step 6: Define model ===
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(lookback, 1)))
model.add(RepeatVector(forecast_horizon))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# === Step 7: Train model ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stop])

# === Step 8: Predict and inverse transform ===
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# === Step 9: Evaluation ===
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())

print("\n✅ Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# === Step 10: Visualization ===
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv[0], label='Actual')
plt.plot(y_pred_inv[0], label='Predicted')
plt.title('Energy Usage Forecast (Next 30 Days)')
plt.xlabel('Day')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 11: Save model and scaler ===
model.save('energy_lstm_month_model.h5')
joblib.dump(scaler, 'energy_scaler_month.pkl')
print("✅ Model and scaler saved to disk.")
#############################################################################
##month prediction 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import joblib
 
# === Step 1: Load and preprocess data ===
df = pd.read_csv('/mnt/data/cleaned.csv', parse_dates=['DateTime'], index_col='DateTime')
df = df[['KWHhh']]
df = df.resample('D').sum().dropna()  # Daily energy consumption

# === Step 2: Scale ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === Step 3: Create sliding window sequences ===
def create_multi_step_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i + input_steps])
        y.append(data[i + input_steps:i + input_steps + output_steps])
    return np.array(X), np.array(y)

lookback = 30  # 30 days input
forecast_horizon = 30  # predict next 30 days
X, y = create_multi_step_sequences(scaled_data, lookback, forecast_horizon)

# === Step 4: Train-Test Split ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === Step 5: Reshape for LSTM ===
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# === Step 6: Define model ===
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(lookback, 1)))
model.add(RepeatVector(forecast_horizon))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

# === Step 7: Train model ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stop])

# === Step 8: Predict and inverse transform ===
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# === Step 9: Evaluation ===
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), y_pred_inv.flatten()))
mae = mean_absolute_error(y_test_inv.flatten(), y_pred_inv.flatten())

print("\n✅ Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# === Step 10: Visualization ===
plt.figure(figsize=(14, 6))
plt.plot(y_test_inv[0], label='Actual')
plt.plot(y_pred_inv[0], label='Predicted')
plt.title('Energy Usage Forecast (Next 30 Days)')
plt.xlabel('Day')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 11: Save model and scaler ===
model.save('energy_lstm_month_model.h5')
joblib.dump(scaler, 'energy_scaler_month.pkl')
print("✅ Model and scaler saved to disk.")
########################################################################
# train_model_fixed.py
####day prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib

# === Step 1: Load and Format Data ===
df = pd.read_csv('cleaned.csv', parse_dates=['DateTime'])
df.set_index('DateTime', inplace=True)

# === Step 2: Select Energy Column and Resample ===
df = df[['KWHhh']].resample('D').sum().dropna()  # Daily energy usage

# === Step 3: Scale Data ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# === Step 4: Create Sequences ===
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

lookback = 7
X, y = create_sequences(scaled_data, lookback)

# === Step 5: Train-Test Split ===
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === Step 6: Build LSTM Model ===
model = Sequential()
model.add(LSTM(64, input_shape=(lookback, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# === Step 7: Train Model ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stop])

# === Step 8: Predict and Inverse Scale ===
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# === Step 9: Evaluate ===
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"\n✅ Evaluation Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# === Step 10: Plot Results ===
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title("Energy Usage Prediction (Next Day)")
plt.xlabel("Time Step")
plt.ylabel("Energy (kWh)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 11: Save Model and Scaler ===
model.save("energy_lstm_model.h5")
joblib.dump(scaler, "energy_scaler.pkl")
print("✅ Model and scaler saved to disk.")