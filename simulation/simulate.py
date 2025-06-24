# simulation/simulate.py
import pandas as pd
import numpy as np
from datetime import timedelta
import random

def simulate_variation(df):
    df_sim = df.copy()

    # Add random fluctuations to key features
    df_sim['NormalPhaseCurrent'] *= np.random.normal(1.0, 0.05, size=len(df_sim))
    df_sim['Voltage'] += np.random.normal(0, 1.5, size=len(df_sim))
    df_sim['Frequency'] += np.random.normal(0, 0.01, size=len(df_sim))
    df_sim['CumulativeEnergykWh'] += np.random.normal(0.01, 0.05, size=len(df_sim))
    
    return df_sim

def simulate_theft(df, theft_meter_ids):
    df_theft = df.copy()
    for meter in theft_meter_ids:
        mask = df_theft['METERSNO'] == meter
        df_theft.loc[mask, 'CumulativeEnergykWh'] *= 0.5  # Under-reporting
        df_theft.loc[mask, 'SystemPowerFactor'] = df_theft.loc[mask, 'SystemPowerFactor'] * 0.6
    return df_theft

def inject_spikes(df, spike_indices):
    df_spike = df.copy()
    for idx in spike_indices:
        df_spike.at[idx, 'NormalPhaseCurrent'] *= 2.5
        df_spike.at[idx, 'Voltage'] += 20
    return df_spike

# Usage Example
if __name__ == "__main__":
    df = pd.read_csv("data/preprocessed_data.csv")

    # Simulate
    df_varied = simulate_variation(df)
    df_theft = simulate_theft(df_varied, theft_meter_ids=["A00002", "A00029"])
    df_spiked = inject_spikes(df_theft, spike_indices=[5, 15, 25])

    df_spiked.to_csv("data/simulated_data.csv", index=False)
    print("✅ Simulation complete: saved as simulated_data.csv")
