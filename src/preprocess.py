"""
src/preprocess.py
─────────────────────────────────────────────────────
Loads and cleans the turbofan engine sensor dataset.
Steps:
  1. Load raw CSV data
  2. Compute Remaining Useful Life (RUL) per engine
  3. Create binary failure label (1 = failure within 30 cycles)
  4. Drop near-constant sensors that carry no signal
  5. Normalize sensor readings to [0, 1]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset. Supports space-separated or CSV format.
    Columns: unit, cycle, 3 op settings, 21 sensor readings.
    """
    try:
        df = pd.read_csv(filepath, sep=r'\s+', header=0, index_col=False)
    except Exception:
        df = pd.read_csv(filepath, header=0, index_col=False)

    # Drop any fully-NaN trailing columns
    df.dropna(axis=1, how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[preprocess] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def compute_RUL(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remaining Useful Life (RUL) = cycles left before failure.
    For each engine unit:
        RUL at cycle t  =  max_cycle_for_unit  -  t
    At the last recorded cycle, RUL = 0 (engine has failed).
    """
    max_cycles = df.groupby('unit')['cycle'].max().rename('max_cycle')
    df = df.join(max_cycles, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    print(f"[preprocess] RUL computed. Range: {df['RUL'].min()}–{df['RUL'].max()} cycles")
    return df


def create_failure_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Binary classification target:
        1 → failure will occur within 'threshold' cycles  (DANGER)
        0 → engine is healthy                             (NORMAL)

    threshold=30 means we alert maintenance 30 cycles before breakdown.
    """
    df['failure_label'] = (df['RUL'] <= threshold).astype(int)
    counts = df['failure_label'].value_counts()
    pct = counts[1] / len(df) * 100
    print(f"[preprocess] Labels created (threshold={threshold} cycles)")
    print(f"             Healthy: {counts[0]:,}  |  Failure: {counts[1]:,}  ({pct:.1f}% failure rate)")
    return df


def drop_constant_sensors(df: pd.DataFrame, sensor_cols: list) -> tuple:
    """
    Sensors with near-zero standard deviation provide no information.
    Drop them to reduce noise and speed up training.
    Returns (cleaned_df, list_of_useful_sensor_names)
    """
    stds = df[sensor_cols].std()
    useful = stds[stds > 0.01].index.tolist()
    dropped = [s for s in sensor_cols if s not in useful]
    print(f"[preprocess] Dropped {len(dropped)} constant sensors: {dropped}")
    print(f"[preprocess] Keeping {len(useful)} informative sensors")
    keep_cols = ['unit', 'cycle', 'RUL', 'failure_label'] + useful
    return df[keep_cols].copy(), useful


def normalize_sensors(df: pd.DataFrame, sensor_cols: list) -> tuple:
    """
    Scale all sensor values to [0, 1] using MinMaxScaler.
    This ensures no single sensor dominates due to scale differences.
    Returns (normalized_df, fitted_scaler)
    """
    scaler = MinMaxScaler()
    df = df.copy()
    df[sensor_cols] = scaler.fit_transform(df[sensor_cols])
    print(f"[preprocess] Sensors normalized to [0, 1]")
    return df, scaler


def run_preprocessing(input_path: str, output_path: str, threshold: int = 30):
    """Full preprocessing pipeline."""
    df = load_data(input_path)
    df = compute_RUL(df)
    df = create_failure_label(df, threshold)

    all_sensor_cols = [f's{i}' for i in range(1, 22)]
    # Only keep sensor cols that actually exist in df
    sensor_cols = [c for c in all_sensor_cols if c in df.columns]

    df, useful_sensors = drop_constant_sensors(df, sensor_cols)
    df, scaler = normalize_sensors(df, useful_sensors)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Saved cleaned data → {output_path}\n")
    return df, scaler, useful_sensors


if __name__ == "__main__":
    run_preprocessing(
        input_path="data/raw/train_FD001.csv",
        output_path="data/processed/cleaned_data.csv",
        threshold=30
    )
