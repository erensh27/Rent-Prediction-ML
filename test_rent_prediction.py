"""Tests for rent_prediction.py and app.py helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rent_prediction import (
    CATEGORICAL_COLS,
    FrequencyEncoder,
    add_interactions,
    build_gbr_pipeline,
    build_catboost_pipeline,
    evaluate_regression,
    get_feature_cols,
    load_dataset,
    mape,
    parse_floor,
)


# ---------------------------------------------------------------------------
# parse_floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, exp_floor, exp_total",
    [
        ("Ground out of 2", 0, 2),
        ("1 out of 3", 1, 3),
        ("2 out of 5", 2, 5),
        ("Lower Basement out of 2", -1, 2),
        ("Upper Basement out of 10", -2, 10),
        ("5 out of 7", 5, 7),
    ],
)
def test_parse_floor_standard(raw: str, exp_floor: int, exp_total: int) -> None:
    result = parse_floor(pd.Series([raw]))
    assert result["floor_number"].iloc[0] == exp_floor
    assert result["total_floors"].iloc[0] == exp_total


def test_parse_floor_edge_cases() -> None:
    result = parse_floor(pd.Series(["Ground out of 1"]))
    assert result["floor_number"].iloc[0] == 0
    assert result["total_floors"].iloc[0] == 1


# ---------------------------------------------------------------------------
# FrequencyEncoder
# ---------------------------------------------------------------------------

def test_frequency_encoder_basic() -> None:
    data = pd.DataFrame({"loc": [f"area_{i % 3}" for i in range(100)]})
    encoder = FrequencyEncoder(min_count=5)
    encoder.fit(data)
    transformed = encoder.transform(data)
    assert "loc_freq" in transformed.columns
    assert "loc" not in transformed.columns
    assert transformed["loc_freq"].dtype.kind == "f"


def test_frequency_encoder_unknown() -> None:
    train = pd.DataFrame({"loc": ["a", "a", "b"]})
    encoder = FrequencyEncoder(min_count=2)
    encoder.fit(train)
    test = pd.DataFrame({"loc": ["a", "c"]})
    transformed = encoder.transform(test)
    # "a" maps to raw count 2 (min_count=2, so it's kept in mapping)
    assert transformed["loc_freq"].iloc[0] == 2.0
    # unknown "c" gets other_value = rare_sum / len(train) = 1/3
    assert transformed["loc_freq"].iloc[1] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# add_interactions
# ---------------------------------------------------------------------------

def test_add_interactions() -> None:
    df = pd.DataFrame({"BHK": [2, 3], "Size": [1000, 1500], "Bathroom": [1, 2]})
    result = add_interactions(df)
    assert "bhk_x_size" in result.columns
    assert "bhk_x_bath" in result.columns
    assert result["bhk_x_size"].iloc[0] == 2000
    assert result["bhk_x_bath"].iloc[0] == 2


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------

def test_get_feature_cols() -> None:
    cols = get_feature_cols()
    assert "BHK" in cols
    assert "Size" in cols
    assert "floor_number" in cols
    assert "total_floors" in cols
    assert "locality_freq" in cols
    assert "bhk_x_size" in cols
    assert "bhk_x_bath" in cols
    for c in CATEGORICAL_COLS:
        assert c in cols


# ---------------------------------------------------------------------------
# evaluate_regression
# ---------------------------------------------------------------------------

def test_evaluate_regression_perfect() -> None:
    y = np.array([100, 200, 300])
    result = evaluate_regression(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0
    assert result["mape"] == 0.0


def test_evaluate_regression_known() -> None:
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = evaluate_regression(y_true, y_pred)
    assert result["mae"] == pytest.approx(10.0)
    assert result["rmse"] == pytest.approx(np.sqrt((100 + 100 + 100) / 3))
    assert result["mape"] == pytest.approx(np.mean([10 / 100, 10 / 200, 10 / 300]) * 100)


# ---------------------------------------------------------------------------
# mape edge cases
# ---------------------------------------------------------------------------

def test_mape_zero_division() -> None:
    assert mape(np.array([0, 100]), np.array([10, 90])) > 0
    assert mape(np.array([0, 0]), np.array([1, 1])) == 100.0


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset(tmp_path: pytest.TempPathFactory) -> None:
    csv_path = tmp_path / "test.csv"
    csv_content = """Posted On,BHK,Rent,Size,Floor,Area Type,Area Locality,City,Furnishing Status,Tenant Preferred,Bathroom,Point of Contact
2022-01-01,2,10000,1000,Ground out of 2,Super Area,TestLoc,Kolkata,Unfurnished,Bachelors/Family,1,Contact Owner
2022-01-02,3,500000,2000,1 out of 3,Carpet Area,TestLoc2,Mumbai,Furnished,Bachelors,2,Contact Agent
2022-01-03,1,5000,500,Ground out of 1,Super Area,TestLoc3,Delhi,Semi-Furnished,Family,1,Contact Owner
"""
    csv_path.write_text(csv_content)
    df = load_dataset(str(csv_path))
    assert len(df) == 2  # row 2 (500000) should be removed as outlier (>95th quantile)
    assert "Rent" in df.columns


# ---------------------------------------------------------------------------
# Pipeline builds
# ---------------------------------------------------------------------------

def test_build_gbr_pipeline() -> None:
    pipe = build_gbr_pipeline()
    assert pipe is not None
    assert hasattr(pipe, "fit")


def test_build_catboost_pipeline() -> None:
    pipe = build_catboost_pipeline()
    assert pipe is not None
    assert hasattr(pipe, "fit")
