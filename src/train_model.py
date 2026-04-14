"""
src/train_model.py
─────────────────────────────────────────────────────
Trains a Random Forest classifier to predict machine failure.

Why Random Forest?
  - Handles high-dimensional sensor data well
  - Works without feature scaling (we scale anyway for consistency)
  - Provides feature importance out of the box
  - Robust to noisy sensor readings
  - No GPU required — runs on any laptop
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)


def load_features(filepath: str):
    """Load feature-engineered dataset and split into X, y."""
    df = pd.read_csv(filepath)
    exclude = {'unit', 'cycle', 'RUL', 'failure_label'}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df['failure_label']
    print(f"[train] Dataset loaded: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"[train] Failure class rate: {y.mean():.2%}")
    return X, y, feature_cols


def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified 80/20 split preserving class distribution."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[train] Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest with 150 trees.

    Key parameters:
      n_estimators=150     → 150 decision trees (more = more stable)
      max_depth=12         → prevent overfitting
      min_samples_leaf=4   → each leaf needs 4+ samples (smoother predictions)
      class_weight=balanced → compensates for fewer failure samples
      n_jobs=-1            → use all CPU cores for speed
    """
    print("[train] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[train] Training complete.")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Compute and print all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    roc  = roc_auc_score(y_test, y_prob)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='roc_auc', n_jobs=-1)

    print("\n" + "="*50)
    print("  MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  ROC-AUC          : {roc:.4f}")
    print(f"  CV ROC-AUC       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Healthy (0)', 'Failure (1)']))

    # Save report
    os.makedirs("outputs", exist_ok=True)
    report_text = (
        f"AI Predictive Maintenance — Model Results\n"
        f"{'='*45}\n"
        f"Accuracy  : {acc:.4f}\n"
        f"F1 Score  : {f1:.4f}\n"
        f"ROC-AUC   : {roc:.4f}\n"
        f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
        f"Classification Report:\n"
        f"{classification_report(y_test, y_pred, target_names=['Healthy','Failure'])}"
    )
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report_text)
    print("[train] Report saved → outputs/classification_report.txt")

    return y_pred, y_prob, {
        'accuracy': acc, 'f1': f1, 'roc_auc': roc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
    }


def save_model(model, feature_cols, model_dir="models"):
    """Persist model and feature column list for later inference."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/random_forest_model.pkl")
    joblib.dump(feature_cols, f"{model_dir}/feature_columns.pkl")
    print(f"[train] Model saved → {model_dir}/random_forest_model.pkl")
    print(f"[train] Features saved → {model_dir}/feature_columns.pkl\n")


def run_training(features_path="data/processed/features_df.csv"):
    """Full training pipeline."""
    X, y, feature_cols = load_features(features_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred, y_prob, metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    save_model(model, feature_cols)
    return model, X_test, y_test, y_pred, y_prob, feature_cols, metrics


if __name__ == "__main__":
    run_training()
