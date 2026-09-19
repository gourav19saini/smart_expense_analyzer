"""
models/anomaly_detector.py
----------------------------
Unusual-transaction detection using scikit-learn's IsolationForest.

IMPORTANT: This is purely a statistical outlier detector based on the
amount, category, weekday and recency of a transaction relative to the
user's own history. It is NOT fraud detection - a flagged transaction
is simply unusual, not necessarily wrong, illegal, or fraudulent.

For the first version we prioritize amount-based detection because it
is both easier to explain in a viva and more reliable with the small
datasets typical of a personal-finance demo.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MIN_TRANSACTIONS_REQUIRED = 15


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    expenses = df[df["transaction_type"] == "Expense"].copy()
    expenses["date"] = pd.to_datetime(expenses["date"])
    expenses["day_of_week"] = expenses["date"].dt.dayofweek

    # Category encoded as its historical mean amount (simple, explainable
    # numeric encoding instead of one-hot, which keeps the feature set small).
    category_avg = expenses.groupby("category")["amount"].transform("mean")
    expenses["category_avg_amount"] = category_avg

    features = expenses[["amount", "day_of_week", "category_avg_amount"]].copy()
    return expenses, features


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05):
    """
    Run IsolationForest over expense transactions.

    Returns
    -------
    result_df : DataFrame of expense transactions with an added
                'is_anomaly' (bool) and 'anomaly_score' (float, lower = more unusual)
                column. Empty DataFrame with a 'message' if there isn't
                enough data yet.
    message : str or None - explanation when detection could not run.
    """
    if df.empty:
        return pd.DataFrame(), "No transactions available yet."

    expenses, features = _build_features(df)

    if len(expenses) < MIN_TRANSACTIONS_REQUIRED:
        return pd.DataFrame(), (
            f"Not enough expense data for reliable anomaly detection "
            f"(need at least {MIN_TRANSACTIONS_REQUIRED}, have {len(expenses)}). "
            f"Add more transactions."
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    preds = model.fit_predict(features)          # -1 = anomaly, 1 = normal
    scores = model.decision_function(features)    # lower = more anomalous

    expenses = expenses.copy()
    expenses["is_anomaly"] = preds == -1
    expenses["anomaly_score"] = scores

    return expenses.sort_values("anomaly_score"), None


def get_flagged_transactions(df: pd.DataFrame, contamination: float = 0.05, top_n: int = 10):
    """Convenience wrapper: returns just the flagged anomalies, most unusual first."""
    result, message = detect_anomalies(df, contamination=contamination)
    if message:
        return pd.DataFrame(), message
    flagged = result[result["is_anomaly"]].head(top_n)
    return flagged, None
