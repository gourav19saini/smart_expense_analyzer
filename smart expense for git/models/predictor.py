"""
models/predictor.py
---------------------
Predicts expected spending for the remainder of the current month
(and gives a simple next-month estimate) using historical daily
expense totals and a Linear Regression model.

Features per day:
    day_of_month, day_of_week, rolling_avg_7d, cumulative_month_spend

Only produces a prediction when there is enough historical data;
otherwise it clearly says so rather than fabricating a number.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import calendar

MIN_DAYS_REQUIRED = 14  # need at least ~2 weeks of history for a trend


def _daily_expense_series(df: pd.DataFrame) -> pd.Series:
    expenses = df[df["transaction_type"] == "Expense"].copy()
    if expenses.empty:
        return pd.Series(dtype=float)
    expenses["date"] = pd.to_datetime(expenses["date"]).dt.normalize()
    daily = expenses.groupby("date")["amount"].sum()
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    return daily


def _build_feature_frame(daily: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame({"date": daily.index, "amount": daily.values})
    feat["day_of_month"] = feat["date"].dt.day
    feat["day_of_week"] = feat["date"].dt.dayofweek
    feat["rolling_avg_7d"] = feat["amount"].rolling(7, min_periods=1).mean().shift(1).fillna(0)
    feat["month"] = feat["date"].dt.to_period("M")
    feat["cumulative_month_spend"] = feat.groupby("month")["amount"].cumsum() - feat["amount"]
    return feat


def predict_month_end_spending(df: pd.DataFrame, today: pd.Timestamp = None):
    """
    Predict total spending for the remainder of the current month.

    Returns a dict with keys:
        status: 'ok' | 'insufficient_data'
        message: str (only when insufficient_data)
        current_spending, estimated_month_end, historical_monthly_average
    """
    if today is None:
        today = pd.Timestamp(datetime.now().date())

    daily = _daily_expense_series(df)
    if daily.empty or len(daily) < MIN_DAYS_REQUIRED:
        return {
            "status": "insufficient_data",
            "message": "Not enough historical data for a reliable prediction. "
                       "Add more transactions to enable prediction.",
        }

    feat = _build_feature_frame(daily)
    feature_cols = ["day_of_month", "day_of_week", "rolling_avg_7d", "cumulative_month_spend"]

    model = LinearRegression()
    model.fit(feat[feature_cols], feat["amount"])

    # Current month spend so far
    current_month = today.to_period("M")
    month_mask = feat["month"] == current_month
    current_spending = feat.loc[month_mask, "amount"].sum()

    # Predict remaining days of the current month day-by-day
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    last_known_rolling = feat["rolling_avg_7d"].iloc[-1] if not feat.empty else 0.0
    cumulative = current_spending
    remaining_pred_total = 0.0

    recent_history = list(feat["amount"].tail(7).values) if len(feat) >= 1 else []

    for day in range(today.day + 1, days_in_month + 1):
        future_date = pd.Timestamp(year=today.year, month=today.month, day=day)
        rolling_avg = np.mean(recent_history[-7:]) if recent_history else 0.0
        row = pd.DataFrame([{
            "day_of_month": day,
            "day_of_week": future_date.dayofweek,
            "rolling_avg_7d": rolling_avg,
            "cumulative_month_spend": cumulative,
        }])
        pred = max(0.0, float(model.predict(row[feature_cols])[0]))
        remaining_pred_total += pred
        cumulative += pred
        recent_history.append(pred)

    estimated_month_end = current_spending + remaining_pred_total

    # Historical monthly average (excluding current, possibly-partial month)
    monthly_totals = feat[feat["month"] != current_month].groupby("month")["amount"].sum()
    historical_average = float(monthly_totals.mean()) if not monthly_totals.empty else current_spending

    return {
        "status": "ok",
        "current_spending": round(float(current_spending), 2),
        "estimated_month_end": round(float(estimated_month_end), 2),
        "historical_monthly_average": round(historical_average, 2),
        "days_of_history": len(daily),
    }
