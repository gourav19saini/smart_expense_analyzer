"""
analytics/expense_analysis.py
-------------------------------
Pandas / NumPy based analytics over the transactions DataFrame.
Every function here works directly off real data pulled from the
database - nothing is hard-coded.
"""

import numpy as np
import pandas as pd


def _expenses(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_type"] == "Expense"].copy()


def _income(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["transaction_type"] == "Income"].copy()


def overall_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_income": 0.0, "total_expenses": 0.0, "balance": 0.0,
            "avg_expense": 0.0, "median_expense": 0.0, "max_expense": 0.0,
            "min_expense": 0.0, "num_transactions": 0,
        }
    exp = _expenses(df)
    inc = _income(df)
    total_income = float(inc["amount"].sum())
    total_expenses = float(exp["amount"].sum())

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": total_income - total_expenses,
        "avg_expense": float(exp["amount"].mean()) if not exp.empty else 0.0,
        "median_expense": float(exp["amount"].median()) if not exp.empty else 0.0,
        "max_expense": float(exp["amount"].max()) if not exp.empty else 0.0,
        "min_expense": float(exp["amount"].min()) if not exp.empty else 0.0,
        "num_transactions": int(len(df)),
    }


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    exp = _expenses(df)
    if exp.empty:
        return pd.DataFrame(columns=["category", "total", "percentage", "avg_transaction", "count"])

    grouped = exp.groupby("category")["amount"].agg(total="sum", avg_transaction="mean", count="count")
    grouped["percentage"] = (grouped["total"] / grouped["total"].sum() * 100).round(2)
    grouped = grouped.reset_index().sort_values("total", ascending=False)
    return grouped[["category", "total", "percentage", "avg_transaction", "count"]]


def daily_spending(df: pd.DataFrame) -> pd.Series:
    exp = _expenses(df)
    if exp.empty:
        return pd.Series(dtype=float)
    exp["date"] = pd.to_datetime(exp["date"])
    return exp.groupby(exp["date"].dt.date)["amount"].sum()


def weekly_spending(df: pd.DataFrame) -> pd.Series:
    exp = _expenses(df)
    if exp.empty:
        return pd.Series(dtype=float)
    exp["date"] = pd.to_datetime(exp["date"])
    return exp.groupby(pd.Grouper(key="date", freq="W"))["amount"].sum()


def monthly_spending(df: pd.DataFrame) -> pd.Series:
    exp = _expenses(df)
    if exp.empty:
        return pd.Series(dtype=float)
    exp["date"] = pd.to_datetime(exp["date"])
    return exp.groupby(exp["date"].dt.to_period("M"))["amount"].sum()


def monthly_income_vs_expense(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "Income", "Expense"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M")
    pivot = d.pivot_table(index="month", columns="transaction_type", values="amount", aggfunc="sum", fill_value=0)
    for col in ("Income", "Expense"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot.reset_index()
    pivot["month"] = pivot["month"].astype(str)
    return pivot[["month", "Income", "Expense"]]


def weekday_vs_weekend(df: pd.DataFrame) -> dict:
    exp = _expenses(df)
    if exp.empty:
        return {"weekday_avg": 0.0, "weekend_avg": 0.0}
    exp["date"] = pd.to_datetime(exp["date"])
    exp["is_weekend"] = exp["date"].dt.dayofweek >= 5

    weekday_daily = exp[~exp["is_weekend"]].groupby(exp["date"].dt.date)["amount"].sum()
    weekend_daily = exp[exp["is_weekend"]].groupby(exp["date"].dt.date)["amount"].sum()

    return {
        "weekday_avg": float(weekday_daily.mean()) if not weekday_daily.empty else 0.0,
        "weekend_avg": float(weekend_daily.mean()) if not weekend_daily.empty else 0.0,
    }


def payment_method_breakdown(df: pd.DataFrame) -> pd.Series:
    exp = _expenses(df)
    if exp.empty:
        return pd.Series(dtype=float)
    return exp.groupby("payment_method")["amount"].sum().sort_values(ascending=False)


def month_over_month_change(df: pd.DataFrame):
    """Returns (pct_change, current_month_total, previous_month_total) for total expenses."""
    monthly = monthly_spending(df)
    if len(monthly) < 2:
        return None, None, None
    current = float(monthly.iloc[-1])
    previous = float(monthly.iloc[-2])
    if previous == 0:
        return None, current, previous
    pct_change = ((current - previous) / previous) * 100
    return round(pct_change, 1), current, previous


def category_month_over_month(df: pd.DataFrame) -> pd.DataFrame:
    """Per-category comparison of the two most recent months present in the data."""
    exp = _expenses(df)
    if exp.empty:
        return pd.DataFrame(columns=["category", "previous_month", "current_month", "pct_change"])
    exp["date"] = pd.to_datetime(exp["date"])
    exp["month"] = exp["date"].dt.to_period("M")
    months = sorted(exp["month"].unique())
    if len(months) < 2:
        return pd.DataFrame(columns=["category", "previous_month", "current_month", "pct_change"])

    prev_m, cur_m = months[-2], months[-1]
    pivot = exp.groupby(["category", "month"])["amount"].sum().unstack(fill_value=0)
    result = pd.DataFrame({
        "category": pivot.index,
        "previous_month": pivot[prev_m] if prev_m in pivot.columns else 0,
        "current_month": pivot[cur_m] if cur_m in pivot.columns else 0,
    }).reset_index(drop=True)
    result["pct_change"] = np.where(
        result["previous_month"] > 0,
        ((result["current_month"] - result["previous_month"]) / result["previous_month"] * 100).round(1),
        np.nan,
    )
    return result.sort_values("current_month", ascending=False)


def frequent_small_transactions(df: pd.DataFrame, threshold: float = 200.0) -> pd.DataFrame:
    """Categories with many transactions below `threshold`, for the current month."""
    exp = _expenses(df)
    if exp.empty:
        return pd.DataFrame(columns=["category", "count_below_threshold"])
    exp["date"] = pd.to_datetime(exp["date"])
    current_month = pd.Timestamp.now().to_period("M")
    this_month = exp[exp["date"].dt.to_period("M") == current_month]
    small = this_month[this_month["amount"] < threshold]
    if small.empty:
        return pd.DataFrame(columns=["category", "count_below_threshold"])
    counts = small.groupby("category").size().reset_index(name="count_below_threshold")
    return counts.sort_values("count_below_threshold", ascending=False)


def budget_progress(df: pd.DataFrame, budgets_df: pd.DataFrame, month: str) -> pd.DataFrame:
    """
    Compare actual spend per category (for `month`, format 'YYYY-MM') against
    the budgets set for that month.
    """
    exp = _expenses(df)
    if exp.empty or budgets_df.empty:
        cols = ["category", "budget_amount", "spent", "percentage_used", "remaining", "exceeded"]
        base = budgets_df.copy() if not budgets_df.empty else pd.DataFrame(columns=["category", "budget_amount"])
        if base.empty:
            return pd.DataFrame(columns=cols)
        base["spent"] = 0.0
        base["percentage_used"] = 0.0
        base["remaining"] = base["budget_amount"]
        base["exceeded"] = False
        return base[cols]

    exp["date"] = pd.to_datetime(exp["date"])
    month_exp = exp[exp["date"].dt.strftime("%Y-%m") == month]
    spent_by_cat = month_exp.groupby("category")["amount"].sum()

    rows = []
    for _, b in budgets_df[budgets_df["month"] == month].iterrows():
        spent = float(spent_by_cat.get(b["category"], 0.0))
        pct = (spent / b["budget_amount"] * 100) if b["budget_amount"] else 0.0
        rows.append({
            "category": b["category"],
            "budget_amount": b["budget_amount"],
            "spent": spent,
            "percentage_used": round(pct, 1),
            "remaining": round(b["budget_amount"] - spent, 2),
            "exceeded": spent > b["budget_amount"],
        })
    return pd.DataFrame(rows)
