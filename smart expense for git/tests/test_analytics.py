"""
tests/test_analytics.py
--------------------------
Tests for analytics/expense_analysis.py and utils/data_cleaning.py.
"""

import os
import sys
import io
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import expense_analysis as ea
from utils import data_cleaning


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"id": 1, "date": "2026-01-05", "amount": 500, "transaction_type": "Expense",
         "category": "Food", "merchant": "SWIGGY", "description": "", "payment_method": "UPI"},
        {"id": 2, "date": "2026-01-06", "amount": 300, "transaction_type": "Expense",
         "category": "Transport", "merchant": "UBER", "description": "", "payment_method": "Cash"},
        {"id": 3, "date": "2026-01-07", "amount": 50000, "transaction_type": "Income",
         "category": "Salary", "merchant": "PAYROLL", "description": "", "payment_method": "Bank Transfer"},
        {"id": 4, "date": "2026-02-05", "amount": 700, "transaction_type": "Expense",
         "category": "Food", "merchant": "ZOMATO", "description": "", "payment_method": "UPI"},
    ])


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["id", "date", "amount", "transaction_type",
                                  "category", "merchant", "description", "payment_method"])


# --- overall_summary ---------------------------------------------------

def test_overall_summary_basic(sample_df):
    s = ea.overall_summary(sample_df)
    assert s["total_income"] == 50000
    assert s["total_expenses"] == 1500
    assert s["balance"] == 48500
    assert s["num_transactions"] == 4


def test_overall_summary_empty(empty_df):
    s = ea.overall_summary(empty_df)
    assert s["total_income"] == 0.0
    assert s["total_expenses"] == 0.0
    assert s["num_transactions"] == 0


# --- category_breakdown ---------------------------------------------------

def test_category_breakdown(sample_df):
    cat = ea.category_breakdown(sample_df)
    food_row = cat[cat["category"] == "Food"].iloc[0]
    assert food_row["total"] == 1200
    assert food_row["count"] == 2


def test_category_breakdown_empty(empty_df):
    cat = ea.category_breakdown(empty_df)
    assert cat.empty


# --- monthly / weekly / daily spending ---------------------------------------------------

def test_monthly_spending(sample_df):
    monthly = ea.monthly_spending(sample_df)
    assert len(monthly) == 2  # Jan and Feb


def test_month_over_month_change(sample_df):
    pct, current, previous = ea.month_over_month_change(sample_df)
    assert previous == 800  # Jan expenses
    assert current == 700   # Feb expenses


def test_month_over_month_insufficient_data(empty_df):
    pct, current, previous = ea.month_over_month_change(empty_df)
    assert pct is None


# --- budget_progress ---------------------------------------------------

def test_budget_progress(sample_df):
    budgets = pd.DataFrame([{"month": "2026-01", "category": "Food", "budget_amount": 400}])
    progress = ea.budget_progress(sample_df, budgets, "2026-01")
    row = progress.iloc[0]
    assert row["spent"] == 500
    assert row["exceeded"] is True or bool(row["exceeded"]) is True


def test_budget_progress_no_budgets(sample_df):
    empty_budgets = pd.DataFrame(columns=["month", "category", "budget_amount"])
    progress = ea.budget_progress(sample_df, empty_budgets, "2026-01")
    assert progress.empty


# --- data_cleaning ---------------------------------------------------

def test_clean_csv_valid_rows():
    raw = pd.DataFrame([
        {"date": "2026-01-01", "amount": 500, "transaction_type": "Expense",
         "category": "Food", "merchant": "SWIGGY", "description": "", "payment_method": "UPI"},
    ])
    clean, report = data_cleaning.clean_transactions_csv(raw)
    assert report["imported"] == 1
    assert report["skipped"] == 0


def test_clean_csv_empty_file():
    clean, report = data_cleaning.clean_transactions_csv(pd.DataFrame())
    assert report["imported"] == 0
    assert "no rows" in report["issues"][0]


def test_clean_csv_missing_required_columns():
    raw = pd.DataFrame({"foo": [1], "bar": [2]})
    clean, report = data_cleaning.clean_transactions_csv(raw)
    assert report["imported"] == 0
    assert any("not found" in issue for issue in report["issues"])


def test_clean_csv_invalid_rows_skipped():
    raw = pd.DataFrame([
        {"date": "bad-date", "amount": 100, "transaction_type": "Expense", "category": "Food"},
        {"date": "2026-01-01", "amount": "not-a-number", "transaction_type": "Expense", "category": "Food"},
        {"date": "2026-01-01", "amount": 500, "transaction_type": "Expense", "category": "Food"},
    ])
    clean, report = data_cleaning.clean_transactions_csv(raw)
    assert report["imported"] == 1
    assert report["skipped"] == 2


def test_clean_csv_duplicate_detection():
    raw = pd.DataFrame([
        {"date": "2026-01-01", "amount": 500, "transaction_type": "Expense",
         "category": "Food", "merchant": "SWIGGY", "description": "lunch"},
        {"date": "2026-01-01", "amount": 500, "transaction_type": "Expense",
         "category": "Food", "merchant": "SWIGGY", "description": "lunch"},
    ])
    clean, report = data_cleaning.clean_transactions_csv(raw)
    assert report["imported"] == 1
    assert report["duplicates"] == 1


def test_clean_csv_alternate_headers():
    raw = pd.DataFrame([
        {"Date": "2026-01-01", "Amount": "500", "Type": "Debit",
         "Merchant": "UBER", "Desc": "cab ride"},
    ])
    clean, report = data_cleaning.clean_transactions_csv(raw)
    assert report["imported"] == 1
    assert clean.iloc[0]["transaction_type"] == "Expense"
