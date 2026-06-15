"""Flask server for the Rent Prediction app."""
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request

from rent_prediction import (
    FrequencyEncoder,
    add_interactions,
    get_feature_cols,
    parse_floor,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(HERE, "House_Rent_Dataset.csv")
MODEL_PATH = os.path.join(HERE, "rent_prediction_model.pkl")
METRICS_PATH = os.path.join(HERE, "model_metrics.json")
IMPORTANCES_PATH = os.path.join(HERE, "feature_importances.csv")
STATIC_DIR = os.path.join(HERE, "static")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("DEBUG", "False").lower() in {"1", "true", "yes"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("app")


def _load_artifacts() -> tuple[Any, Any, Any]:
    """Load model, locality_encoder, and metrics. Exits on failure."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file not found at {MODEL_PATH}\n"
              f"Run: python rent_prediction.py --retrain", file=sys.stderr)
        sys.exit(1)
    try:
        data = joblib.load(MODEL_PATH)
        if isinstance(data, tuple):
            pipeline, locality_encoder = data
        else:
            pipeline, locality_encoder = data, None
    except Exception as exc:
        print(f"ERROR: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as exc:
            print(f"WARNING: failed to read metrics: {exc}", file=sys.stderr)

    return pipeline, locality_encoder, metrics


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        return pd.read_csv(DATASET_PATH)
    except Exception as exc:
        print(f"ERROR: failed to read dataset: {exc}", file=sys.stderr)
        sys.exit(1)


model, locality_encoder, METRICS = _load_artifacts()
df_raw = _load_dataset()
MODEL_VERSION = METRICS.get("trained_at", "unknown")
TARGET_TRANSFORM = METRICS.get("target_transform", "none")
IS_GBR = "GradientBoosting" in METRICS.get("estimator", "")

app = Flask(__name__)


@lru_cache(maxsize=1)
def get_form_options() -> dict[str, Any]:
    return {
        "cities": sorted(df_raw["City"].dropna().unique().tolist()),
        "area_types": sorted(df_raw["Area Type"].dropna().unique().tolist()),
        "furnishings": sorted(df_raw["Furnishing Status"].dropna().unique().tolist()),
        "tenants": sorted(df_raw["Tenant Preferred"].dropna().unique().tolist()),
        "contacts": sorted(df_raw["Point of Contact"].dropna().unique().tolist()),
        "bhk_min": int(df_raw["BHK"].min()),
        "bhk_max": int(df_raw["BHK"].max()),
        "size_min": int(df_raw["Size"].min()),
        "size_max": int(df_raw["Size"].max()),
        "bath_min": int(df_raw["Bathroom"].min()),
        "bath_max": int(df_raw["Bathroom"].max()),
    }


@lru_cache(maxsize=1)
def get_dataset_stats() -> dict[str, Any]:
    return {
        "rows": int(len(df_raw)),
        "cities": int(df_raw["City"].nunique()),
        "avg_rent": int(df_raw["Rent"].mean()),
        "median_rent": int(df_raw["Rent"].median()),
        "min_rent": int(df_raw["Rent"].min()),
        "max_rent": int(df_raw["Rent"].max()),
        "city_avg": {
            city: int(rent)
            for city, rent in df_raw.groupby("City")["Rent"].mean().sort_values().items()
        },
    }


@lru_cache(maxsize=1)
def get_prediction_cap() -> int:
    return int(df_raw["Rent"].quantile(0.99))


@lru_cache(maxsize=1)
def heap_rent_ranking() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cols = ["Rent", "BHK", "Size", "City", "Furnishing Status", "Tenant Preferred"]
    sorted_df = df_raw[cols].sort_values("Rent")
    cheapest = [_listing_dict(row) for _, row in sorted_df.head(5).iterrows()]
    premium = [_listing_dict(row) for _, row in sorted_df.tail(5).iloc[::-1].iterrows()]
    return cheapest, premium


def _listing_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "rent": int(row["Rent"]),
        "bhk": int(row["BHK"]),
        "size": int(row["Size"]),
        "city": row["City"],
        "furnishing": row["Furnishing Status"],
        "tenant": row["Tenant Preferred"],
    }


def _format_prediction(value: float) -> str:
    cap = get_prediction_cap()
    if value > cap:
        return f"Above ₹{cap:,}"
    return f"₹ {int(value):,}"


# Feature columns used by the model (excluding categoricals that are stored
# directly in the form data).
_NUMERIC_FEATURE_KEYS = ["BHK", "Size", "Bathroom"]
_CATEGORICAL_FEATURE_KEYS = [
    "City", "Area Type", "Furnishing Status", "Tenant Preferred", "Point of Contact",
]


def _build_input_df(form_data: dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame with all engineered features expected by the model."""
    row = {k: form_data.get(k) for k in _NUMERIC_FEATURE_KEYS + _CATEGORICAL_FEATURE_KEYS}
    row["Floor"] = f"{form_data.get('floor_number', 'Ground')} out of {form_data.get('total_floors', 1)}"
    df = pd.DataFrame([row])

    # 1. Parse floor
    floor_df = parse_floor(df["Floor"])
    df = pd.concat([df, floor_df], axis=1)

    # 2. Frequency-encode locality (if encoder available)
    if locality_encoder is not None:
        if "Area Locality" not in df.columns:
            df["Area Locality"] = "Unknown"
        encoded = locality_encoder.transform(df[["Area Locality"]])
        df["locality_freq"] = encoded["Area Locality_freq"]

    # 3. Interaction features
    df = add_interactions(df)

    # 4. Select columns in correct order
    return df[get_feature_cols()]


def _model_info() -> dict[str, Any]:
    if not METRICS:
        return {}
    return {
        "test_r2": METRICS.get("test_r2"),
        "test_mae": METRICS.get("test_mae"),
        "test_rmse": METRICS.get("test_rmse"),
        "n_samples": METRICS.get("n_samples"),
        "trained_at": METRICS.get("trained_at"),
        "estimator": METRICS.get("estimator"),
        "features": METRICS.get("features"),
    }


@app.route("/", methods=["GET", "POST"])
def home() -> Any:
    prediction = None
    submitted: dict[str, Any] = {}
    error: str | None = None

    if request.method == "POST":
        try:
            form_data = {
                "BHK": int(request.form["bhk"]),
                "Size": int(request.form["size"]),
                "Bathroom": int(request.form["bathroom"]),
                "City": request.form["city"],
                "Area Type": request.form["area_type"],
                "Furnishing Status": request.form["furnishing"],
                "Tenant Preferred": request.form["tenant"],
                "Point of Contact": request.form["contact"],
                "floor_number": request.form.get("floor_number", "Ground"),
                "total_floors": int(request.form.get("total_floors", 1)),
            }
            input_df = _build_input_df(form_data)
            raw_prediction = float(model.predict(input_df)[0])

            # Reverse log-transform if needed
            if TARGET_TRANSFORM == "log1p":
                raw_prediction = float(np.expm1(raw_prediction))

            prediction = _format_prediction(raw_prediction)
            submitted = {
                "bhk": int(form_data["BHK"]),
                "size": int(form_data["Size"]),
                "bathroom": int(form_data["Bathroom"]),
                "city": form_data["City"],
                "area_type": form_data["Area Type"],
                "furnishing": form_data["Furnishing Status"],
                "tenant": form_data["Tenant Preferred"],
                "contact": form_data["Point of Contact"],
            }
        except (KeyError, ValueError) as exc:
            error = f"Invalid form input: {exc}"

    server_data = {
        "options": get_form_options(),
        "stats": get_dataset_stats(),
        "submitted": submitted,
        "prediction": prediction,
        "error": error,
        "model_info": _model_info(),
    }
    return render_template(
        "index.html",
        server_data_json=json.dumps(server_data),
        model_info=_model_info(),
    )


@app.route("/graphs")
def graphs() -> Any:
    return render_template("graphs.html")


@app.route("/recommendations")
def recommendations() -> Any:
    cheapest, premium = heap_rent_ranking()
    return render_template(
        "recommendations.html",
        cheapest=cheapest,
        premium=premium,
    )


@app.route("/metrics")
def metrics_page() -> Any:
    return render_template("metrics.html", metrics=METRICS or {})


@app.route("/health")
def health() -> Any:
    return jsonify({
        "status": "ok",
        "model": "loaded" if model is not None else "missing",
        "dataset_rows": int(len(df_raw)),
        "model_version": MODEL_VERSION,
        "estimator": METRICS.get("estimator", "unknown"),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict() -> Any:
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400
    required = _NUMERIC_FEATURE_KEYS + _CATEGORICAL_FEATURE_KEYS
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        payload.setdefault("floor_number", "Ground")
        payload.setdefault("total_floors", 1)
        input_df = _build_input_df(payload)
        raw_prediction = float(model.predict(input_df)[0])
        if TARGET_TRANSFORM == "log1p":
            raw_prediction = float(np.expm1(raw_prediction))
    except ValueError as exc:
        return jsonify({"error": f"invalid input: {exc}"}), 400
    cap = get_prediction_cap()
    return jsonify({
        "predicted_rent": int(raw_prediction),
        "currency": "INR",
        "capped": raw_prediction > cap,
        "cap_threshold": cap,
    })


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def generate_trend_graphs() -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)

    plt.figure()
    plt.scatter(df_raw["Size"], df_raw["Rent"], alpha=0.4, s=15)
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Rent (₹)")
    plt.title("Rent vs Size")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "rent_vs_size.png"), dpi=120)
    plt.close()

    plt.figure()
    df_raw.groupby("City")["Rent"].mean().sort_values().plot(kind="bar", color="steelblue")
    plt.ylabel("Average Rent (₹)")
    plt.title("Average Rent by City")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "rent_by_city.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.heatmap(df_raw[["BHK", "Size", "Bathroom", "Rent"]].corr(),
                annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "correlation.png"), dpi=120)
    plt.close()


def generate_feature_importance_chart() -> None:
    if not os.path.exists(IMPORTANCES_PATH):
        return
    try:
        imp = pd.read_csv(IMPORTANCES_PATH).head(15).iloc[::-1]
    except Exception:
        return
    os.makedirs(STATIC_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh(imp["feature"], imp["importance_mean"],
             xerr=imp["importance_std"], color="teal", alpha=0.85)
    plt.xlabel("Permutation Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "feature_importances.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_trend_graphs()
    generate_feature_importance_chart()
    log.info("Starting server on %s:%s (DEBUG=%s)", HOST, PORT, DEBUG)
    app.run(host=HOST, port=PORT, debug=DEBUG)
