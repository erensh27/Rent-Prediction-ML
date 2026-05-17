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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor  # alternative estimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

def parse_floor(floor_str: Any) -> tuple[int, int]:
    try:
        if not isinstance(floor_str, str):
            return 0, 1
        parts = floor_str.split(" out of ")
        level_str = parts[0].strip().lower()
        if level_str == "ground":
            level = 0
        elif level_str == "lower basement":
            level = -1
        elif level_str == "upper basement":
            level = -2
        else:
            level = int(level_str)

        total = int(parts[1].strip()) if len(parts) > 1 else level + 1
        return level, total
    except (ValueError, IndexError):
        return 0, 1

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Feature engineering for Floor
    if "Floor" in df.columns:
        floor_data = df["Floor"].apply(parse_floor)
        df["floor_level"] = floor_data.apply(lambda x: x[0])
        df["total_floors"] = floor_data.apply(lambda x: x[1])
        df = df.drop(columns=["Floor"])

    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    cap = df[config.TARGET].quantile(0.95)
    df = df[df[config.TARGET] < cap].reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLS),
        ]
    )
    estimator = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        random_state=config.RANDOM_STATE,
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
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
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
        n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def generate_trend_graphs(df_raw: pd.DataFrame) -> None:
    os.makedirs(config.STATIC_DIR, exist_ok=True)

    plt.figure()
    plt.scatter(df_raw["Size"], df_raw["Rent"], alpha=0.4)
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Rent")
    plt.title("Rent vs Size")
    plt.tight_layout()
    plt.savefig(os.path.join(config.STATIC_DIR, "rent_vs_size.png"))
    plt.close()

    plt.figure()
    df_raw.groupby("City")["Rent"].mean().sort_values().plot(kind="bar")
    plt.ylabel("Average Rent")
    plt.title("Average Rent by City")
    plt.tight_layout()
    plt.savefig(os.path.join(config.STATIC_DIR, "rent_by_city.png"))
    plt.close()

    # Calculate correlation on numeric columns only
    plt.figure()
    numeric_df = df_raw.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(config.STATIC_DIR, "correlation.png"))
    plt.close()


def generate_feature_importance_chart() -> None:
    if not os.path.exists(config.IMPORTANCES_PATH):
        return
    try:
        imp = pd.read_csv(config.IMPORTANCES_PATH).head(15).iloc[::-1]
    except Exception as exc:
        print(f"WARNING: could not read importances: {exc}")
        return
    os.makedirs(config.STATIC_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"])
    plt.xlabel("Permutation Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(config.STATIC_DIR, "feature_importances.png"))
    plt.close()


def save_form_options(df: pd.DataFrame) -> None:
    options = {
        "cities": sorted(df["City"].dropna().unique().tolist()),
        "area_types": sorted(df["Area Type"].dropna().unique().tolist()),
        "furnishings": sorted(df["Furnishing Status"].dropna().unique().tolist()),
        "tenants": sorted(df["Tenant Preferred"].dropna().unique().tolist()),
        "contacts": sorted(df["Point of Contact"].dropna().unique().tolist()),
        "bhk_min": int(df["BHK"].min()),
        "bhk_max": int(df["BHK"].max()),
        "size_min": int(df["Size"].min()),
        "size_max": int(df["Size"].max()),
        "bath_min": int(df["Bathroom"].min()),
        "bath_max": int(df["Bathroom"].max()),
    }
    with open(config.FORM_OPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(options, f, indent=2)


def save_dataset_stats(df: pd.DataFrame) -> None:
    stats = {
        "rows": int(len(df)),
        "cities": int(df["City"].nunique()),
        "avg_rent": int(df["Rent"].mean()),
        "median_rent": int(df["Rent"].median()),
        "min_rent": int(df["Rent"].min()),
        "max_rent": int(df["Rent"].max()),
        "city_avg": {
            city: int(rent)
            for city, rent in df.groupby("City")["Rent"].mean().sort_values().items()
        },
        "prediction_cap": int(df["Rent"].quantile(0.99))
    }
    with open(config.STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def save_recommendations(df: pd.DataFrame) -> None:
    cols = ["Rent", "BHK", "Size", "City", "Furnishing Status", "Tenant Preferred"]
    sorted_df = df[cols].sort_values("Rent")

    def _listing_dict(row):
        return {
            "rent": int(row["Rent"]),
            "bhk": int(row["BHK"]),
            "size": int(row["Size"]),
            "city": row["City"],
            "furnishing": row["Furnishing Status"],
            "tenant": row["Tenant Preferred"],
        }

    cheapest = [_listing_dict(row) for _, row in sorted_df.head(5).iterrows()]
    premium = [_listing_dict(row) for _, row in sorted_df.tail(5).iloc[::-1].iterrows()]

    with open(config.RECS_PATH, "w", encoding="utf-8") as f:
        json.dump({"cheapest": cheapest, "premium": premium}, f, indent=2)


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
    print(f"Model file:   {config.MODEL_PATH}")
    print(f"Metrics file: {config.METRICS_PATH}")
    print(f"Importances:  {config.IMPORTANCES_PATH}")
    print("=" * 50)


def train(force: bool = False) -> dict[str, Any]:
    if not force and os.path.exists(config.MODEL_PATH) and os.path.exists(config.METRICS_PATH):
        print(f"Model already exists at {config.MODEL_PATH}. Use --retrain to overwrite.")
        with open(config.METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Loading dataset from {config.DATASET_PATH}")
    df = load_dataset(config.DATASET_PATH)
    feature_cols = config.NUMERIC_COLS + config.CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    pipeline = build_pipeline()

    print("Running hyperparameter tuning (RandomizedSearchCV)...")
    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")

    print("Running 5-fold cross-validation with best model...")
    cv_metrics = cross_validate(pipeline, X, y)

    print("Evaluating on held-out test set...")
    test_metrics = evaluate_test(pipeline, X_test, y_test)

    print("Computing permutation importances...")
    importances = compute_permutation_importance(pipeline, X_test, y_test)
    importances.to_csv(config.IMPORTANCES_PATH, index=False)

    print("Generating charts...")
    generate_trend_graphs(df)
    generate_feature_importance_chart()

    print("Saving form options, stats and recommendations...")
    save_form_options(df)
    save_dataset_stats(df)
    save_recommendations(df)

    metrics = {
        **cv_metrics,
        **test_metrics,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "estimator": type(pipeline.named_steps["model"]).__name__,
    }

    joblib.dump(pipeline, config.MODEL_PATH)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
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
