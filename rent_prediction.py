"""Train the rent-prediction model.

Builds a sklearn Pipeline (preprocessing + estimator), evaluates with
cross-validation and a held-out test set, then persists the fitted pipeline,
JSON metrics, and permutation-importance CSV next to this script.

Usage:
    python rent_prediction.py            # train only if no model exists
    python rent_prediction.py --retrain  # force re-train and overwrite outputs
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor  # alternative estimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "House_Rent_Dataset.csv")
MODEL_PATH = os.path.join(HERE, "rent_prediction_model.pkl")
METRICS_PATH = os.path.join(HERE, "model_metrics.json")
IMPORTANCES_PATH = os.path.join(HERE, "feature_importances.csv")

NUMERIC_COLS = ["BHK", "Size", "Bathroom"]
CATEGORICAL_COLS = [
    "City",
    "Area Type",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact",
]
DROP_COLS = ["Posted On", "Area Locality", "Floor"]
TARGET = "Rent"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    cap = df[TARGET].quantile(0.95)
    df = df[df[TARGET] < cap].reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    estimator = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        random_state=RANDOM_STATE,
    )
    # Alternative:
    # estimator = RandomForestRegressor(
    #     n_estimators=300, max_depth=18, min_samples_split=5,
    #     random_state=RANDOM_STATE, n_jobs=-1,
    # )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)


def cross_validate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    neg_mae = cross_val_score(pipeline, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    neg_mse = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
    r2 = cross_val_score(pipeline, X, y, scoring="r2", cv=cv, n_jobs=-1)
    return {
        "cv_mae_mean": float(-neg_mae.mean()),
        "cv_mae_std": float(neg_mae.std()),
        "cv_rmse_mean": float(np.sqrt(-neg_mse).mean()),
        "cv_rmse_std": float(np.sqrt(-neg_mse).std()),
        "cv_r2_mean": float(r2.mean()),
        "cv_r2_std": float(r2.std()),
    }


def evaluate_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": rmse,
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mape": mean_absolute_percentage_error(y_test.values, y_pred),
    }


def compute_permutation_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def print_summary(metrics: dict[str, Any]) -> None:
    rows = [
        ("CV MAE",  f"{metrics['cv_mae_mean']:>12,.2f} ± {metrics['cv_mae_std']:,.2f}"),
        ("CV RMSE", f"{metrics['cv_rmse_mean']:>12,.2f} ± {metrics['cv_rmse_std']:,.2f}"),
        ("CV R²",   f"{metrics['cv_r2_mean']:>12.4f} ± {metrics['cv_r2_std']:.4f}"),
        ("Test MAE",  f"{metrics['test_mae']:>12,.2f}"),
        ("Test RMSE", f"{metrics['test_rmse']:>12,.2f}"),
        ("Test R²",   f"{metrics['test_r2']:>12.4f}"),
        ("Test MAPE", f"{metrics['test_mape']:>12.2f}%"),
    ]
    width = max(len(r[0]) for r in rows)
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    for label, value in rows:
        print(f"{label:<{width}}  {value}")
    print("=" * 50)
    print(f"Trained at:   {metrics['trained_at']}")
    print(f"Samples:      {metrics['n_samples']:,}")
    print(f"Model file:   {MODEL_PATH}")
    print(f"Metrics file: {METRICS_PATH}")
    print(f"Importances:  {IMPORTANCES_PATH}")
    print("=" * 50)


def train(force: bool = False) -> dict[str, Any]:
    if not force and os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        print(f"Model already exists at {MODEL_PATH}. Use --retrain to overwrite.")
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Loading dataset from {DATASET_PATH}")
    df = load_dataset(DATASET_PATH)
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    print("Running 5-fold cross-validation...")
    cv_metrics = cross_validate(pipeline, X, y)

    print("Fitting on training set...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    test_metrics = evaluate_test(pipeline, X_test, y_test)

    print("Computing permutation importances...")
    importances = compute_permutation_importance(pipeline, X_test, y_test)
    importances.to_csv(IMPORTANCES_PATH, index=False)

    metrics = {
        **cv_metrics,
        **test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "estimator": type(pipeline.named_steps["model"]).__name__,
    }

    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print_summary(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rent-prediction model.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if a model already exists.")
    args = parser.parse_args()
    train(force=args.retrain)


if __name__ == "__main__":
    main()
