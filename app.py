"""Flask server for the Rent Prediction app."""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request

import config

def _load_model() -> Any:
    if not os.path.exists(config.MODEL_PATH):
        print(
            f"ERROR: model file not found at {config.MODEL_PATH}\n"
            f"Run: python rent_prediction.py --retrain",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        return joblib.load(config.MODEL_PATH)
    except Exception as exc:
        print(f"ERROR: failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(config.DATASET_PATH):
        print(f"ERROR: dataset not found at {config.DATASET_PATH}", file=sys.stderr)
        sys.exit(1)
    try:
        return pd.read_csv(config.DATASET_PATH)
    except Exception as exc:
        print(f"ERROR: failed to read dataset: {exc}", file=sys.stderr)
        sys.exit(1)


def _load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default or {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"WARNING: failed to read JSON file {path}: {exc}", file=sys.stderr)
        return default or {}


model = _load_model()
# df_raw no longer needed as all stats/recs are pre-calculated
# df_raw = _load_dataset()
METRICS = _load_json(config.METRICS_PATH)
OPTIONS = _load_json(config.FORM_OPTIONS_PATH)
STATS = _load_json(config.STATS_PATH)
RECS = _load_json(config.RECS_PATH)
MODEL_VERSION = METRICS.get("trained_at", "unknown")

app = Flask(__name__)


def get_form_options() -> dict[str, Any]:
    return OPTIONS


def get_dataset_stats() -> dict[str, Any]:
    return STATS


def get_prediction_cap() -> int:
    return STATS.get("prediction_cap", 200000)


def heap_rent_ranking() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return RECS.get("cheapest", []), RECS.get("premium", [])


def _format_prediction(value: float) -> str:
    cap = get_prediction_cap()
    if value > cap:
        return f"Above ₹{cap:,}"
    return f"₹ {int(value):,}"


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

def _build_input_df(form_data: dict[str, Any]) -> pd.DataFrame:
    floor_level, total_floors = parse_floor(form_data.get("Floor", "Ground out of 1"))
    return pd.DataFrame([{
        "BHK": int(form_data["BHK"]),
        "Size": int(form_data["Size"]),
        "Bathroom": int(form_data["Bathroom"]),
        "City": form_data["City"],
        "Area Type": form_data["Area Type"],
        "Furnishing Status": form_data["Furnishing Status"],
        "Tenant Preferred": form_data["Tenant Preferred"],
        "Point of Contact": form_data["Point of Contact"],
        "floor_level": floor_level,
        "total_floors": total_floors,
    }])


def _model_info() -> dict[str, Any]:
    if not METRICS:
        return {}
    return {
        "test_r2": METRICS.get("test_r2"),
        "test_mae": METRICS.get("test_mae"),
        "test_rmse": METRICS.get("test_rmse"),
        "n_samples": METRICS.get("n_samples"),
        "trained_at": METRICS.get("trained_at"),
    }


@app.route("/", methods=["GET", "POST"])
def home() -> Any:
    prediction = None
    submitted: dict[str, Any] = {}
    error: str | None = None

    if request.method == "POST":
        try:
            form_data = {
                "BHK": request.form["bhk"],
                "Size": request.form["size"],
                "Bathroom": request.form["bathroom"],
                "City": request.form["city"],
                "Area Type": request.form["area_type"],
                "Furnishing Status": request.form["furnishing"],
                "Tenant Preferred": request.form["tenant"],
                "Point of Contact": request.form["contact"],
            }
            input_df = _build_input_df(form_data)
            raw_prediction = float(model.predict(input_df)[0])
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
        "dataset_rows": STATS.get("rows", 0),
        "model_version": MODEL_VERSION,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict() -> Any:
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400
    required = ["BHK", "Size", "Bathroom", "City", "Area Type",
                "Furnishing Status", "Tenant Preferred", "Point of Contact"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        input_df = _build_input_df(payload)
        raw_prediction = float(model.predict(input_df)[0])
    except ValueError as exc:
        return jsonify({"error": f"invalid input: {exc}"}), 400
    cap = get_prediction_cap()
    return jsonify({
        "predicted_rent": int(raw_prediction),
        "currency": "INR",
        "capped": raw_prediction > cap,
        "cap_threshold": cap,
    })


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
