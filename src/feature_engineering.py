"""
src/feature_engineering.py
─────────────────────────────────────────────────────
Creates time-series statistical features from raw sensor readings.

Why? A single sensor reading at one moment is weak evidence.
But the TREND (rolling mean) and VOLATILITY (rolling std) over
the last 5 cycles are far more predictive of failure.

Features created per sensor:
  - Rolling mean  (window=5): captures trend direction
  - Rolling std   (window=5): captures instability / noise
  - Lag-1 value             : captures rate of change
"""

import pandas as pd
import numpy as np
import os


def add_rolling_features(df: pd.DataFrame, sensor_cols: list, window: int = 5) -> pd.DataFrame:
    """
    Rolling mean and standard deviation over the last `window` cycles.
    Grouped by unit so we don't mix data between different engines.

    Rising rolling mean of temperature → engine getting hotter → danger
    Rising rolling std of vibration   → erratic shaking → danger
    """
    df = df.copy()
    for col in sensor_cols:
        grp = df.groupby('unit')[col]
        df[f'{col}_roll_mean'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'{col}_roll_std'] = grp.transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )
    print(f"[features] Added rolling mean+std (window={window}) for {len(sensor_cols)} sensors")
    return df


def add_lag_features(df: pd.DataFrame, sensor_cols: list, lag: int = 1) -> pd.DataFrame:
    """
    Previous cycle's reading (lag-1).
    Difference between current and previous = rate of change.
    Sudden jumps often signal developing faults.
    """
    df = df.copy()
    for col in sensor_cols:
        df[f'{col}_lag{lag}'] = df.groupby('unit')[col].shift(lag).fillna(0)
    print(f"[features] Added lag-{lag} features for {len(sensor_cols)} sensors")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns all ML feature columns (excludes metadata and target)."""
    exclude = {'unit', 'cycle', 'RUL', 'failure_label'}
    return [col for col in df.columns if col not in exclude]


def run_feature_engineering(input_path: str, output_path: str):
    """Full feature engineering pipeline."""
    df = pd.read_csv(input_path)

    # Identify base sensor columns (before rolling features are added)
    exclude = {'unit', 'cycle', 'RUL', 'failure_label'}
    sensor_cols = [c for c in df.columns if c not in exclude]

    df = add_rolling_features(df, sensor_cols, window=5)
    df = add_lag_features(df, sensor_cols, lag=1)

    # Drop rows with NaN (from rolling at the start of each engine)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"[features] Dropped {before - after} NaN rows after rolling ops")

    feature_cols = get_feature_columns(df)
    print(f"[features] Total feature columns: {len(feature_cols)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[features] Saved feature dataset → {output_path}\n")
    return df, feature_cols


if __name__ == "__main__":
    run_feature_engineering(
        input_path="data/processed/cleaned_data.csv",
        output_path="data/processed/features_df.csv"
    )
