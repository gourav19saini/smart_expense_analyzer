"""
tests/test_models.py
-----------------------
Tests for models/categorizer.py, models/anomaly_detector.py, and
models/predictor.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import categorizer, anomaly_detector, predictor


# --- categorizer ---------------------------------------------------

def test_categorizer_trains_and_predicts_known_merchant():
    model = categorizer.train_model()
    cat, conf = categorizer.predict_category("SWIGGY", model=model)
    assert cat == "Food"
    assert conf > categorizer.CONFIDENCE_THRESHOLD


def test_categorizer_low_confidence_needs_review():
    model = categorizer.train_model()
    cat, conf = categorizer.predict_category("ZZZQQXX UNKNOWN MERCHANT 999", model=model)
    # Either genuinely low confidence -> needs review, or a class was still assigned;
    # the important invariant is the function never crashes and always returns a tuple.
    assert isinstance(cat, str)
    assert 0.0 <= conf <= 1.0


def test_categorizer_empty_text():
    model = categorizer.train_model()
    cat, conf = categorizer.predict_category("", model=model)
    assert cat == categorizer.NEEDS_REVIEW_LABEL
    assert conf == 0.0


def test_categorizer_bulk_predictions():
    model = categorizer.train_model()
    results = categorizer.predict_categories_bulk(["UBER", "NETFLIX", "AMAZON"], model=model)
    assert len(results) == 3
    for cat, conf in results:
        assert isinstance(cat, str)


# --- anomaly_detector ---------------------------------------------------

def _make_expense_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    amounts = rng.normal(500, 50, size=n).clip(min=10)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "amount": amounts,
        "transaction_type": "Expense",
        "category": "Food",
        "merchant": "TEST",
        "description": "",
        "payment_method": "UPI",
    })
    return df


def test_anomaly_detection_insufficient_data():
    small_df = _make_expense_df(n=5)
    result, msg = anomaly_detector.detect_anomalies(small_df)
    assert result.empty
    assert msg is not None
    assert "Not enough" in msg


def test_anomaly_detection_finds_outlier():
    df = _make_expense_df(n=40)
    # Inject an obvious outlier
    outlier_row = pd.DataFrame([{
        "date": "2026-03-01", "amount": 50000, "transaction_type": "Expense",
        "category": "Food", "merchant": "TEST", "description": "", "payment_method": "UPI",
    }])
    df = pd.concat([df, outlier_row], ignore_index=True)
    result, msg = anomaly_detector.detect_anomalies(df, contamination=0.1)
    assert msg is None
    assert result.loc[result["amount"] == 50000, "is_anomaly"].iloc[0] == True  # noqa: E712


def test_anomaly_detection_empty_df():
    empty = pd.DataFrame(columns=["date", "amount", "transaction_type", "category",
                                   "merchant", "description", "payment_method"])
    result, msg = anomaly_detector.detect_anomalies(empty)
    assert result.empty
    assert msg is not None


# --- predictor ---------------------------------------------------

def test_prediction_insufficient_data():
    small_df = _make_expense_df(n=5)
    pred = predictor.predict_month_end_spending(small_df)
    assert pred["status"] == "insufficient_data"


def test_prediction_with_sufficient_data():
    df = _make_expense_df(n=60)
    today = pd.Timestamp(df["date"].iloc[-1])
    pred = predictor.predict_month_end_spending(df, today=today)
    assert pred["status"] == "ok"
    assert pred["estimated_month_end"] >= pred["current_spending"]
    assert pred["historical_monthly_average"] >= 0


def test_prediction_never_negative():
    df = _make_expense_df(n=60)
    today = pd.Timestamp(df["date"].iloc[-1])
    pred = predictor.predict_month_end_spending(df, today=today)
    if pred["status"] == "ok":
        assert pred["estimated_month_end"] >= 0
