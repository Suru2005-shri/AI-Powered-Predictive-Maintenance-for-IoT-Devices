"""
generate_dataset.py
Generates a realistic synthetic turbofan engine sensor dataset
that mirrors the NASA CMAPSS structure exactly.

Each engine unit degrades over time until failure.
Sensors show increasing anomalies as the engine approaches breakdown.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_engine_data(unit_id, max_cycles, n_sensors=21, n_ops=3):
    cycles = np.arange(1, max_cycles + 1)
    n = len(cycles)

    # Degradation factor: 0 (new) → 1 (failed)
    degradation = (cycles / max_cycles) ** 1.5

    rows = []
    for i, cycle in enumerate(cycles):
        deg = degradation[i]

        # Operational settings (random but consistent per engine)
        op1 = np.random.choice([-0.0087, 0.0000, 0.0014, 0.0024], p=[0.25,0.25,0.25,0.25])
        op2 = np.random.choice([0.0004, 0.0003, 0.0001, 0.0], p=[0.25,0.25,0.25,0.25])
        op3 = np.random.choice([100.0, 60.0, 80.0], p=[0.5,0.25,0.25])

        # Sensor readings — 14 variable + 7 near-constant
        # Sensors that INCREASE with degradation (e.g. temperature, pressure)
        s2  = 641.82  + deg * 15.0  + np.random.normal(0, 0.5)
        s3  = 1590.09 + deg * 50.0  + np.random.normal(0, 1.0)
        s4  = 1408.94 + deg * 30.0  + np.random.normal(0, 1.5)
        s7  = 554.36  + deg * 10.0  + np.random.normal(0, 0.3)
        s8  = 2388.06 + deg * 20.0  + np.random.normal(0, 2.0)
        s9  = 9065.80 + deg * 150.0 + np.random.normal(0, 5.0)
        s11 = 47.47   + deg * 2.0   + np.random.normal(0, 0.2)
        s12 = 522.28  + deg * 8.0   + np.random.normal(0, 0.5)
        s13 = 2388.09 + deg * 25.0  + np.random.normal(0, 2.0)
        s14 = 8138.62 + deg * 100.0 + np.random.normal(0, 5.0)

        # Sensors that DECREASE with degradation (e.g. efficiency)
        s17 = 392.0   - deg * 10.0  + np.random.normal(0, 0.5)
        s20 = 39.06   - deg * 1.5   + np.random.normal(0, 0.2)
        s21 = 23.4190 - deg * 0.5   + np.random.normal(0, 0.05)

        # Vibration sensor (increases sharply near failure)
        vibration_spike = 0.0
        if deg > 0.75:
            vibration_spike = (deg - 0.75) * 40.0 * abs(np.random.normal(1, 0.3))
        s15 = 38.86 + vibration_spike + np.random.normal(0, 0.3)

        # Near-constant sensors (low information, will be dropped in preprocessing)
        s1  = 518.67 + np.random.normal(0, 0.001)
        s5  = 14.62  + np.random.normal(0, 0.001)
        s6  = 21.61  + np.random.normal(0, 0.001)
        s10 = 1.3    + np.random.normal(0, 0.0001)
        s16 = 0.03   + np.random.normal(0, 0.0001)
        s18 = 2388.0 + np.random.normal(0, 0.001)
        s19 = 100.0  + np.random.normal(0, 0.001)

        row = [unit_id, cycle, op1, op2, op3,
               s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
               s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21]
        rows.append(row)

    return rows


def generate_full_dataset(n_units=100, seed=42):
    np.random.seed(seed)
    all_rows = []

    for uid in range(1, n_units + 1):
        # Each engine has a random lifespan between 120 and 350 cycles
        max_cycles = np.random.randint(120, 350)
        rows = generate_engine_data(uid, max_cycles)
        all_rows.extend(rows)

    columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + \
              [f's{i}' for i in range(1, 22)]

    df = pd.DataFrame(all_rows, columns=columns)
    return df


if __name__ == "__main__":
    print("Generating synthetic NASA CMAPSS-style dataset...")
    df = generate_full_dataset(n_units=100)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/train_FD001.csv", index=False, sep=' ')
    print(f"Dataset saved: data/raw/train_FD001.csv")
    print(f"Shape: {df.shape}")
    print(f"Units: {df['unit'].nunique()}")
    print(f"Total cycles: {len(df)}")
    print(df.head(3))
