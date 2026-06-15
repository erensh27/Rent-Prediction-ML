"""Train, evaluate, and benchmark rent-prediction models.

Usage:
    python rent_prediction.py                # train only if no model exists
    python rent_prediction.py --retrain      # force re-train and overwrite
    python rent_prediction.py --compare      # benchmark GBR vs CatBoost
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
OUTLIER_QUANTILE = 0.95

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "House_Rent_Dataset.csv")
MODEL_PATH = os.path.join(HERE, "rent_prediction_model.pkl")
MODEL_B_PATH = os.path.join(HERE, "rent_prediction_model_catboost.pkl")
METRICS_PATH = os.path.join(HERE, "model_metrics.json")
IMPORTANCES_PATH = os.path.join(HERE, "feature_importances.csv")
BENCHMARK_PATH = os.path.join(HERE, "benchmark_report.json")

CATEGORICAL_COLS = [
    "City",
    "Area Type",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact",
]
TARGET = "Rent"

_GROUND_MAP = {"Ground": 0, "Lower Basement": -1, "Upper Basement": -2}
_FLOOR_RE = re.compile(r"^(?:Ground|Lower Basement|Upper Basement|(\d+)) out of (\d+)$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rent_prediction")

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------


def parse_floor(floor_series: pd.Series) -> pd.DataFrame:
    matches = floor_series.str.extract(_FLOOR_RE, expand=False)
    # matches[0] is NaN when "Ground"/"Lower Basement"/"Upper Basement" matched
    # (the \d+ group did not participate). Use integer sentinel, then patch.
    floor_num = matches[0].fillna(-999).astype(int).copy()
    ground_mask = matches[0].isna()
    if ground_mask.any():
        ground_vals = (
            floor_series[ground_mask]
            .str.extract(r"^(Ground|Lower Basement|Upper Basement)", expand=False)
            .map(_GROUND_MAP)
            .fillna(0)
            .astype(int)
        )
        floor_num[ground_mask] = ground_vals.values
    total = matches[1].fillna(1).astype(int)
    return pd.DataFrame(
        {"floor_number": floor_num, "total_floors": total},
        index=floor_series.index,
    )


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bhk_x_size"] = df["BHK"] * df["Size"]
    df["bhk_x_bath"] = df["BHK"] * df["Bathroom"]
    return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency-encode a high-cardinality categorical column, grouping
    rare levels (frequency < *min_count*) into an "other" bucket."""

    def __init__(self, min_count: int = 5):
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None) -> FrequencyEncoder:
        col = X.columns[0]
        freq = X[col].value_counts(dropna=False)
        self.mapping_ = freq[freq >= self.min_count].to_dict()
        rare_sum = freq[freq < self.min_count].sum()
        self.other_value_ = float(rare_sum) / float(len(X)) if len(X) else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        col = X.columns[0]
        result = X.copy()
        result[col + "_freq"] = result[col].map(self.mapping_).fillna(self.other_value_)
        result = result.drop(columns=[col])
        return result.astype({col + "_freq": float})


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the GBR pipeline: numeric passthrough + cat OHE."""
    numeric_cols = [
        "BHK", "Size", "Bathroom",
        "floor_number", "total_floors", "locality_freq",
        "bhk_x_size", "bhk_x_bath",
    ]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )


def get_feature_cols() -> list[str]:
    return [
        "BHK", "Size", "Bathroom",
        "floor_number", "total_floors", "locality_freq",
        "bhk_x_size", "bhk_x_bath",
        *CATEGORICAL_COLS,
    ]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def build_gbr_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])


def build_catboost_pipeline() -> Pipeline:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=50,
    )
    return Pipeline(steps=[
        ("passthrough", FunctionTransformer()),
        ("model", model),
    ])


def _catboost_cat_indices(X: pd.DataFrame) -> list[int]:
    """Return column indices of categorical features for CatBoost."""
    return [i for i, c in enumerate(X.columns) if c in CATEGORICAL_COLS]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cap = df[TARGET].quantile(OUTLIER_QUANTILE)
    before = len(df)
    df = df[df[TARGET] <= cap].reset_index(drop=True)
    log.info("Loaded %d rows; removed %d rows above %s quantile (₹%d)",
             len(df), before - len(df), OUTLIER_QUANTILE, int(cap))
    return df


def prepare_features(df: pd.DataFrame, locality_encoder: FrequencyEncoder | None,
                     fit_encoder: bool = False) -> tuple[pd.DataFrame, FrequencyEncoder | None]:
    df = df.copy()

    # 1. Parse Floor
    floor_df = parse_floor(df["Floor"])
    df = pd.concat([df, floor_df], axis=1)

    # 2. Frequency-encode Area Locality
    if fit_encoder:
        locality_encoder = FrequencyEncoder(min_count=5)
        locality_encoder.fit(df[["Area Locality"]])
    if locality_encoder is not None:
        encoded = locality_encoder.transform(df[["Area Locality"]])
        df["locality_freq"] = encoded["Area Locality_freq"]

    # 3. Interaction features
    df = add_interactions(df)

    # 4. Drop unused columns
    df = df.drop(columns=[c for c in ["Posted On", "Area Locality", "Floor"]
                          if c in df.columns])

    return df, locality_encoder


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def tune_gbr(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    log.info("Tuning GradientBoostingRegressor with RandomizedSearchCV …")
    pipeline = build_gbr_pipeline()
    param_dist = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__min_samples_leaf": [1, 3, 5, 10],
    }
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    log.info("Best GBR params: %s   (CV MAE: ₹%.0f)",
             search.best_params_, -search.best_score_)
    return search.best_estimator_


def tune_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Tune CatBoostRegressor using CatBoost's own randomized_search (avoids
    sklearn clone() incompatibility with ``cat_features``)."""
    from catboost import CatBoostRegressor

    cat_indices = _catboost_cat_indices(X_train)

    param_dist = {
        "iterations": [500, 1000, 1500, 2000],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
        "depth": [4, 5, 6, 7, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
    best_model = CatBoostRegressor(
        cat_features=cat_indices,
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=50,
    )
    log.info("Tuning CatBoostRegressor with randomized_search \u2026")
    result = best_model.randomized_search(
        param_dist,
        X=X_train,
        y=y_train,
        cv=N_SPLITS,
        n_iter=20,
        partition_random_seed=RANDOM_STATE,
        verbose=False,
        plot=False,
        search_by_train_test_split=False,
    )
    log.info("Best CatBoost params: %s", result["params"])
    # best_model is already fitted on full training data with best params
    return Pipeline(steps=[
        ("passthrough", FunctionTransformer()),
        ("model", best_model),
    ])


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def compute_permutation_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def per_segment_metrics(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute test metrics broken down by City and Furnishing Status."""
    segments: dict[str, dict[str, float]] = {}

    for col in ["City", "Furnishing Status"]:
        for val in X_test[col].unique():
            mask = X_test[col] == val
            if mask.sum() < 5:
                continue
            y_true = y_test[mask]
            y_pred = pipeline.predict(X_test[mask])
            segments[f"{col}_{val}"] = evaluate_regression(y_true.values, y_pred)

    return segments


def baseline_metrics(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    return evaluate_regression(y_test, y_pred)


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


def compute_shap(pipeline: Pipeline, X_test: pd.DataFrame) -> dict[str, Any]:
    """Compute SHAP values for the trained pipeline."""
    import shap

    try:
        # Unwrap the model from Pipeline
        model_step = pipeline.named_steps["model"]

        if hasattr(model_step, "feature_names_in_"):
            feature_names = list(model_step.feature_names_in_)
        else:
            feature_names = list(X_test.columns)

        # For sklearn GBR with OHE, we need to use the preprocessed features
        # Use Pipeline's transform to get the feature matrix
        X_test_transformed = pipeline[:-1].transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        # Use a subsample for SHAP (speed)
        sample_size = min(200, len(X_test))
        X_sample = X_test_transformed[:sample_size]
        X_raw_sample = X_test.iloc[:sample_size]

        if isinstance(model_step, GradientBoostingRegressor):
            explainer = shap.Explainer(model_step, X_sample)
            shap_values = explainer(X_sample)

            return {
                "feature_names": feature_names,
                "shap_values": shap_values.values.tolist(),
                "base_value": float(shap_values.base_values[0]),
                "mean_abs_shap": [
                    float(np.abs(shap_values.values).mean(axis=0).tolist()[i])
                    for i in range(len(feature_names))
                ],
            }
        elif "CatBoostRegressor" in type(model_step).__name__:
            explainer = shap.Explainer(model_step, X_raw_sample)
            shap_values = explainer(X_raw_sample)
            raw_feature_names = list(X_raw_sample.columns)
            return {
                "feature_names": raw_feature_names,
                "shap_values": shap_values.values.tolist(),
                "base_value": float(shap_values.base_values[0]),
                "mean_abs_shap": [
                    float(np.abs(shap_values.values).mean(axis=0).tolist()[i])
                    for i in range(len(raw_feature_names))
                ],
            }
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def print_summary(metrics: dict[str, Any]) -> None:
    rows = [
        ("CV MAE",   f"₹{metrics['cv_mae_mean']:>10,.0f} ± ₹{metrics['cv_mae_std']:,.0f}"),
        ("CV RMSE",  f"₹{metrics['cv_rmse_mean']:>10,.0f} ± ₹{metrics['cv_rmse_std']:,.0f}"),
        ("CV R²",    f"{metrics['cv_r2_mean']:>12.4f} ± {metrics['cv_r2_std']:.4f}"),
        ("Test MAE", f"₹{metrics['test_mae']:>10,.0f}"),
        ("Test RMSE",f"₹{metrics['test_rmse']:>10,.0f}"),
        ("Test R²",  f"{metrics['test_r2']:>12.4f}"),
        ("Test MAPE",f"{metrics['test_mape']:>11.2f}%"),
    ]
    w = max(len(r[0]) for r in rows)
    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    for label, val in rows:
        print(f"  {label:<{w}}  {val}")
    print("=" * 55)
    print(f"  Trained at:   {metrics['trained_at']}")
    print(f"  Estimator:    {metrics['estimator']}")
    print(f"  Samples:      {metrics['n_samples']:,}  "
          f"(train {metrics['n_train']:,},  test {metrics['n_test']:,})")
    print(f"  Features:     {len(metrics.get('features', []))}")

    if "baseline_vs_model" in metrics:
        b = metrics["baseline_vs_model"]
        print(f"\n  --- vs DummyRegressor (median) ---")
        print(f"  Δ MAE:  ₹{b['mae_improvement']:+,.0f}  "
              f"({b['mae_improvement_pct']:+.1f}%)")
        print(f"  Δ RMSE: ₹{b['rmse_improvement']:+,.0f}  "
              f"({b['rmse_improvement_pct']:+.1f}%)")
        print(f"  Δ R²:   {b['r2_improvement']:+.4f}")

    print("=" * 55)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def _create_benchmark_report(
    pipeline: Pipeline,
    X_test: pd.DataFrame, y_test: pd.Series,
    baseline: dict[str, float],
    test_metrics: dict[str, float],
    name: str,
) -> dict[str, Any]:
    segments = per_segment_metrics(pipeline, X_test, y_test)
    importances_df = compute_permutation_importance(pipeline, X_test, y_test)
    shap_data = compute_shap(pipeline, X_test)

    return {
        "model_name": name,
        "test_metrics": test_metrics,
        "baseline_median": baseline,
        "segments": segments,
        "permutation_importance_top10": importances_df.head(10).to_dict(orient="records"),
        "shap": shap_data,
    }


def train(force: bool = False, compare: bool = False) -> dict[str, Any]:
    # -----------------------------------------------------------------------
    # 1. Load & prepare data
    # -----------------------------------------------------------------------
    if not force and os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        log.info("Model already exists at %s. Use --retrain to overwrite.", MODEL_PATH)
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    df = load_dataset(DATASET_PATH)
    locality_encoder: FrequencyEncoder | None = None
    df, locality_encoder = prepare_features(df, locality_encoder, fit_encoder=True)

    # Log-transform target
    y = np.log1p(df[TARGET])
    feature_cols = get_feature_cols()
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # 2. Cross-validate baseline
    # -----------------------------------------------------------------------
    log.info("Fitting baseline (DummyRegressor median) …")
    baseline = baseline_metrics(X_train, y_train, X_test, y_test)

    # -----------------------------------------------------------------------
    # 3. Train & evaluate selected pipeline
    # -----------------------------------------------------------------------
    use_catboost = compare
    if compare:
        log.info("=" * 55)
        log.info("  BENCHMARK MODE: Training GBR + CatBoost side-by-side")
        log.info("=" * 55)

        pipe_gbr = tune_gbr(X_train, np.expm1(y_train))
        pipe_cb = tune_catboost(X_train, y_train)

        # Evaluate both on test set (log-scale for GBR was trained on expm1)
        # GBR was trained on expm1(y_train) — so it predicts rent directly
        # CatBoost was trained on log1p(y_train) — so it predicts log(rent)
        for pipe, name, y_tr, trained_on_log in [
            (pipe_gbr, "GBR", np.expm1(y_train), False),
            (pipe_cb, "CatBoost", y_train, True),
        ]:
            if trained_on_log:
                y_pred_log = pipe.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                y_t = np.expm1(y_test)
            else:
                y_pred = pipe.predict(X_test)
                y_t = np.expm1(y_test)
            m = evaluate_regression(y_t, y_pred)
            log.info("%s test metrics: MAE=₹%.0f  RMSE=₹%.0f  R²=%.4f  MAPE=%.1f%%",
                     name, m["mae"], m["rmse"], m["r2"], m["mape"])

            benchmark = _create_benchmark_report(
                pipe, X_test, y_test, baseline, m, name,
            )
            bp = BENCHMARK_PATH.replace(".json", f"_{name.lower()}.json")
            with open(bp, "w", encoding="utf-8") as f:
                json.dump(benchmark, f, indent=2)
            log.info("Benchmark saved → %s", bp)

        # Pick the better model (lower MAE)
        # Re-evaluate both properly
        y_pred_gbr = pipe_gbr.predict(X_test)
        m_gbr = evaluate_regression(np.expm1(y_test), y_pred_gbr)

        y_pred_cb_log = pipe_cb.predict(X_test)
        y_pred_cb = np.expm1(y_pred_cb_log)
        m_cb = evaluate_regression(np.expm1(y_test), y_pred_cb)

        if m_gbr["mae"] <= m_cb["mae"]:
            pipeline = pipe_gbr
            log.info("Picking GBR (MAE ₹%.0f vs ₹%.0f)", m_gbr["mae"], m_cb["mae"])
            train_target_was_logged = False
        else:
            pipeline = pipe_cb
            log.info("Picking CatBoost (MAE ₹%.0f vs ₹%.0f)", m_cb["mae"], m_gbr["mae"])
            train_target_was_logged = True

        # Save the runner-up too
        joblib.dump(pipe_cb if m_gbr["mae"] <= m_cb["mae"] else pipe_gbr, MODEL_B_PATH)
        log.info("Runner-up saved → %s", MODEL_B_PATH)

    else:
        pipeline = tune_gbr(X_train, np.expm1(y_train))
        train_target_was_logged = False

    # -----------------------------------------------------------------------
    # 4. Evaluate on test set
    # -----------------------------------------------------------------------
    if train_target_was_logged:
        y_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_orig = np.expm1(y_test)
    else:
        y_pred = pipeline.predict(X_test)
        y_test_orig = np.expm1(y_test)

    test_metrics = evaluate_regression(y_test_orig, y_pred)

    # -----------------------------------------------------------------------
    # 5. Cross-validation on original scale
    # -----------------------------------------------------------------------
    log.info("Running %d-fold cross-validation …", N_SPLITS)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    neg_mae = cross_val_score(pipeline, X, y, scoring="neg_mean_absolute_error",
                              cv=cv, n_jobs=-1)
    neg_mse = cross_val_score(pipeline, X, y, scoring="neg_mean_squared_error",
                              cv=cv, n_jobs=-1)
    r2_cv = cross_val_score(pipeline, X, y, scoring="r2", cv=cv, n_jobs=-1)

    cv_metrics = {
        "cv_mae_mean": float(-neg_mae.mean()),
        "cv_mae_std": float(neg_mae.std()),
        "cv_rmse_mean": float(np.sqrt(-neg_mse).mean()),
        "cv_rmse_std": float(np.sqrt(-neg_mse).std()),
        "cv_r2_mean": float(r2_cv.mean()),
        "cv_r2_std": float(r2_cv.std()),
    }

    # -----------------------------------------------------------------------
    # 6. Permutation importance & segment breakdown
    # -----------------------------------------------------------------------
    log.info("Computing permutation importance …")
    importances_df = compute_permutation_importance(pipeline, X_test, y_test)
    importances_df.to_csv(IMPORTANCES_PATH, index=False)

    log.info("Breaking down test metrics by City & Furnishing Status …")
    segments = per_segment_metrics(pipeline, X_test, y_test)

    # -----------------------------------------------------------------------
    # 7. SHAP explainability
    # -----------------------------------------------------------------------
    log.info("Computing SHAP values …")
    shap_data = compute_shap(pipeline, X_test)

    # -----------------------------------------------------------------------
    # 8. Assemble metrics
    # -----------------------------------------------------------------------
    estimator_name = type(pipeline.named_steps["model"]).__name__
    n_samples = len(df)

    # Improvement vs baseline
    baseline_vs_model = {
        "baseline_mae": baseline["mae"],
        "baseline_rmse": baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "mae_improvement": baseline["mae"] - test_metrics["mae"],
        "mae_improvement_pct": (baseline["mae"] - test_metrics["mae"]) / baseline["mae"] * 100,
        "rmse_improvement": baseline["rmse"] - test_metrics["rmse"],
        "rmse_improvement_pct": (baseline["rmse"] - test_metrics["rmse"]) / baseline["rmse"] * 100,
        "r2_improvement": test_metrics["r2"] - baseline["r2"],
    }

    metrics: dict[str, Any] = {
        **cv_metrics,
        **test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "categorical_features": CATEGORICAL_COLS,
        "estimator": estimator_name,
        "target_transform": "log1p" if train_target_was_logged else "none",
        "baseline_vs_model": baseline_vs_model,
        "segments": segments,
        "permutation_importance_top10": importances_df.head(10).to_dict(orient="records"),
        "shap": shap_data,
        "datasets": {
            "n_rows": n_samples,
            "n_outliers_removed": int(pd.read_csv(DATASET_PATH).shape[0]) - n_samples,
        },
    }

    # -----------------------------------------------------------------------
    # 9. Persist model & metrics
    # -----------------------------------------------------------------------
    joblib.dump((pipeline, locality_encoder), MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print_summary(metrics)
    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark rent-prediction models.")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retraining even if a model already exists.")
    parser.add_argument("--compare", action="store_true",
                        help="Benchmark GradientBoosting vs CatBoost and pick the best.")
    args = parser.parse_args()
    train(force=args.retrain, compare=args.compare)


if __name__ == "__main__":
    main()
