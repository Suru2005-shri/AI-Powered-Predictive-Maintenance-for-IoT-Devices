"""
main.py
═══════════════════════════════════════════════════════════════════
AI-Powered Predictive Maintenance System for IoT Devices
Dataset : Synthetic NASA CMAPSS-style turbofan engine sensor data
Model   : Random Forest Classifier (scikit-learn)
Author  : Your Name

HOW TO RUN:
    python main.py

WHAT IT DOES:
    Step 1 — Generate/load sensor dataset
    Step 2 — Preprocess (RUL, labels, normalize)
    Step 3 — Feature engineering (rolling stats, lag)
    Step 4 — Train Random Forest model
    Step 5 — Evaluate (accuracy, F1, ROC-AUC)
    Step 6 — Generate all visualizations
    Step 7 — Run live failure prediction simulation
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def print_header():
    print("\n" + "═"*65)
    print("  AI-POWERED PREDICTIVE MAINTENANCE FOR IoT DEVICES")
    print("  Turbofan Engine Failure Prediction System")
    print("  Dataset: NASA CMAPSS-style Synthetic Sensor Data")
    print("═"*65 + "\n")


def step(n, total, desc):
    print(f"\n{'─'*65}")
    print(f"  STEP {n}/{total}: {desc}")
    print(f"{'─'*65}")


def main():
    print_header()
    t_start = time.time()

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — Generate Dataset
    # ─────────────────────────────────────────────────────────────
    step(1, 7, "Generating synthetic sensor dataset")
    raw_path = "data/raw/train_FD001.csv"
    if not os.path.exists(raw_path):
        print("  Dataset not found. Generating now...")
        from generate_dataset import generate_full_dataset
        import pandas as pd, os as _os
        _os.makedirs("data/raw", exist_ok=True)
        df_raw = generate_full_dataset(n_units=100)
        df_raw.to_csv(raw_path, index=False, sep=' ')
        print(f"  ✓ Generated {len(df_raw):,} sensor readings from 100 engines")
    else:
        import pandas as pd
        df_raw = pd.read_csv(raw_path, sep=r'\s+')
        print(f"  ✓ Dataset found: {len(df_raw):,} rows | {df_raw['unit'].nunique()} engines")

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — Preprocessing
    # ─────────────────────────────────────────────────────────────
    step(2, 7, "Preprocessing — RUL computation + labels + normalization")
    from src.preprocess import run_preprocessing
    df_clean, scaler, useful_sensors = run_preprocessing(
        input_path=raw_path,
        output_path="data/processed/cleaned_data.csv",
        threshold=30
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — Feature Engineering
    # ─────────────────────────────────────────────────────────────
    step(3, 7, "Feature Engineering — rolling stats + lag features")
    from src.feature_engineering import run_feature_engineering
    df_features, feature_cols = run_feature_engineering(
        input_path="data/processed/cleaned_data.csv",
        output_path="data/processed/features_df.csv"
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — Model Training
    # ─────────────────────────────────────────────────────────────
    step(4, 7, "Model Training — Random Forest Classifier")
    from src.train_model import run_training
    model, X_test, y_test, y_pred, y_prob, feature_cols, metrics = run_training(
        features_path="data/processed/features_df.csv"
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 5 — Print metrics summary
    # ─────────────────────────────────────────────────────────────
    step(5, 7, "Evaluation Summary")
    print(f"\n  ┌───────────────────────────────────┐")
    print(f"  │  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)     │")
    print(f"  │  F1 Score   : {metrics['f1']:.4f}                    │")
    print(f"  │  ROC-AUC    : {metrics['roc_auc']:.4f}                    │")
    print(f"  │  CV ROC-AUC : {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}         │")
    print(f"  └───────────────────────────────────┘")

    # ─────────────────────────────────────────────────────────────
    # STEP 6 — Visualizations
    # ─────────────────────────────────────────────────────────────
    step(6, 7, "Generating all visualizations")
    import pandas as pd
    df_full = pd.read_csv("data/processed/features_df.csv")

    from src.visualize import run_all_visualizations
    run_all_visualizations(df_full, model, feature_cols,
                           y_test, y_pred, y_prob, metrics)

    print("\n  Charts saved in outputs/:")
    charts = [
        "sensor_degradation_unit1.png",
        "confusion_matrix.png",
        "feature_importance.png",
        "roc_curve.png",
        "failure_prediction_timeline.png",
        "class_distribution.png",
        "sensor_heatmap.png",
        "performance_summary.png",
    ]
    for c in charts:
        exists = "✓" if os.path.exists(f"outputs/{c}") else "✗"
        print(f"    {exists}  {c}")

    # ─────────────────────────────────────────────────────────────
    # STEP 7 — Live simulation
    # ─────────────────────────────────────────────────────────────
    step(7, 7, "Live IoT Failure Prediction Simulation")
    from src.predict import load_model_and_features, run_live_simulation, run_batch_prediction
    model_loaded, feat_cols = load_model_and_features()
    run_live_simulation(df_full, model_loaded, feat_cols, n_samples=20)
    run_batch_prediction(df_full, model_loaded, feat_cols)

    # ─────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("═"*65)
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  ✓ Model  → models/random_forest_model.pkl")
    print(f"  ✓ Charts → outputs/*.png")
    print(f"  ✓ Report → outputs/classification_report.txt")
    print(f"  ✓ Preds  → outputs/predictions_output.csv")
    print("═"*65 + "\n")


if __name__ == "__main__":
    main()
