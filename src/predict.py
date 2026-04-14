"""
src/predict.py
─────────────────────────────────────────────────────
Simulates a real-time IoT failure prediction system.

In a real deployment:
  - Sensors send data every second/minute via MQTT or REST API
  - This module loads the saved model and scores each reading
  - An alert fires when failure probability crosses a threshold

Here we simulate this with sample rows from the dataset.
"""

import pandas as pd
import numpy as np
import joblib
import os
import time


def load_model_and_features(model_dir: str = "models"):
    """Load the persisted Random Forest model and feature column names."""
    model = joblib.load(f"{model_dir}/random_forest_model.pkl")
    feature_cols = joblib.load(f"{model_dir}/feature_columns.pkl")
    return model, feature_cols


def predict_failure_probability(model, feature_cols: list, sensor_reading: dict) -> float:
    """
    Score a single IoT sensor snapshot.

    sensor_reading: dict mapping feature_name → sensor_value
    Returns: float between 0.0 (healthy) and 1.0 (certain failure)
    """
    row = pd.DataFrame([sensor_reading])[feature_cols]
    prob = model.predict_proba(row)[0][1]
    return float(prob)


def classify_alert(prob: float) -> tuple:
    """
    Convert probability to human-readable status.
    Returns (status_string, color_code)
    """
    if prob < 0.30:
        return "NORMAL   ✓", "green"
    elif prob < 0.60:
        return "WARNING  ⚠", "yellow"
    else:
        return "CRITICAL ✗", "red"


def print_alert(unit_id, prob: float, cycle: int = None):
    """Print a formatted alert line to terminal."""
    status, _ = classify_alert(prob)
    cycle_str = f" | Cycle {cycle:4d}" if cycle else ""
    bar_filled = int(prob * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  Engine {unit_id:>8s}{cycle_str} | [{bar}] {prob:.1%} | {status}")


def run_live_simulation(df: pd.DataFrame, model, feature_cols: list,
                        n_samples: int = 15, delay: float = 0.0):
    """
    Simulate live IoT monitoring by scoring random sensor snapshots.

    In production this would be replaced by a real sensor stream.
    """
    print("\n" + "═"*75)
    print("   LIVE IoT MONITORING SIMULATION — AI Predictive Maintenance")
    print("═"*75)
    print(f"  {'Engine':>10} | {'Cycle':>6} | {'Probability Bar':20} {'Prob':>6} | Status")
    print("─"*75)

    exclude = {'unit', 'cycle', 'RUL', 'failure_label'}
    X = df[[c for c in df.columns if c not in exclude]]

    # Sample across different engines for variety
    samples = df.sample(n=n_samples, random_state=99)

    critical_count = 0
    warning_count  = 0
    normal_count   = 0

    for _, row in samples.iterrows():
        sensor_vals = row[feature_cols].to_dict()
        prob = predict_failure_probability(model, feature_cols, sensor_vals)
        unit_label = f"Unit-{int(row['unit']):03d}"
        cycle = int(row['cycle'])
        print_alert(unit_label, prob, cycle)

        status, _ = classify_alert(prob)
        if "CRITICAL" in status: critical_count += 1
        elif "WARNING" in status: warning_count += 1
        else: normal_count += 1

        if delay > 0:
            time.sleep(delay)

    print("─"*75)
    print(f"\n  Summary: {normal_count} Normal | {warning_count} Warning | {critical_count} Critical")
    print("═"*75 + "\n")


def run_batch_prediction(df: pd.DataFrame, model, feature_cols: list,
                         output_path: str = "outputs/predictions_output.csv"):
    """
    Score entire dataset and save results with predictions.
    Useful for batch monitoring reports.
    """
    exclude = {'unit', 'cycle', 'RUL', 'failure_label'}
    X = df[[c for c in df.columns if c not in exclude]]

    df = df.copy()
    df['predicted_prob'] = model.predict_proba(X)[:, 1]
    df['predicted_label'] = model.predict(X)
    df['alert_status'] = df['predicted_prob'].apply(
        lambda p: classify_alert(p)[0].strip()
    )

    alerts = df[df['predicted_label'] == 1]
    print(f"[predict] Total readings  : {len(df):,}")
    print(f"[predict] Failure alerts  : {len(alerts):,}")
    print(f"[predict] Alert rate      : {len(alerts)/len(df):.2%}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[['unit', 'cycle', 'RUL', 'failure_label',
        'predicted_prob', 'predicted_label', 'alert_status']].to_csv(
        output_path, index=False
    )
    print(f"[predict] Results saved → {output_path}\n")
    return df


if __name__ == "__main__":
    model, feature_cols = load_model_and_features()
    df = pd.read_csv("data/processed/features_df.csv")

    # Live simulation
    run_live_simulation(df, model, feature_cols, n_samples=20)

    # Batch prediction
    run_batch_prediction(df, model, feature_cols)
