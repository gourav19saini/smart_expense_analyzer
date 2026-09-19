"""
tests/test_database.py
------------------------
Tests for database/db.py: schema creation, transaction CRUD, budget CRUD,
and input validation. Uses a temporary on-disk SQLite file so it never
touches the real data/expenses.db used by the running app.
"""

import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db


@pytest.fixture
def temp_db(monkeypatch):
    """Point db.DB_PATH at a fresh temp file for the duration of one test."""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "test_expenses.db")
    monkeypatch.setattr(db, "DB_PATH", tmp_path)
    db.init_db()
    yield tmp_path


def test_init_db_creates_tables(temp_db):
    with db.get_connection() as conn:
        tables = {row["name"] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "transactions" in tables
    assert "budgets" in tables


def test_add_and_get_transaction(temp_db):
    db.add_transaction("2026-01-15", 500, "Expense", "Food", "SWIGGY", "lunch", "UPI")
    df = db.get_all_transactions()
    assert len(df) == 1
    assert df.iloc[0]["amount"] == 500
    assert df.iloc[0]["category"] == "Food"


def test_add_transaction_rejects_negative_amount(temp_db):
    with pytest.raises(ValueError):
        db.add_transaction("2026-01-15", -50, "Expense", "Food")


def test_add_transaction_rejects_zero_amount(temp_db):
    with pytest.raises(ValueError):
        db.add_transaction("2026-01-15", 0, "Expense", "Food")


def test_add_transaction_rejects_invalid_type(temp_db):
    with pytest.raises(ValueError):
        db.add_transaction("2026-01-15", 100, "Transfer", "Food")


def test_add_transaction_rejects_invalid_date(temp_db):
    with pytest.raises(ValueError):
        db.add_transaction("not-a-date", 100, "Expense", "Food")


def test_add_transactions_bulk(temp_db):
    frame = pd.DataFrame([
        {"date": "2026-01-01", "amount": 100, "transaction_type": "Expense",
         "category": "Food", "merchant": "A", "description": "", "payment_method": "Cash"},
        {"date": "2026-01-02", "amount": 200, "transaction_type": "Income",
         "category": "Salary", "merchant": "B", "description": "", "payment_method": "Bank Transfer"},
    ])
    n = db.add_transactions_bulk(frame)
    assert n == 2
    assert db.transaction_count() == 2


def test_update_transaction(temp_db):
    db.add_transaction("2026-01-15", 500, "Expense", "Food")
    df = db.get_all_transactions()
    tx_id = int(df.iloc[0]["id"])
    db.update_transaction(tx_id, amount=750, category="Shopping")
    df2 = db.get_all_transactions()
    row = df2[df2["id"] == tx_id].iloc[0]
    assert row["amount"] == 750
    assert row["category"] == "Shopping"


def test_delete_transaction(temp_db):
    db.add_transaction("2026-01-15", 500, "Expense", "Food")
    df = db.get_all_transactions()
    tx_id = int(df.iloc[0]["id"])
    db.delete_transaction(tx_id)
    assert db.transaction_count() == 0


def test_delete_all_transactions(temp_db):
    db.add_transaction("2026-01-15", 500, "Expense", "Food")
    db.add_transaction("2026-01-16", 300, "Expense", "Transport")
    db.delete_all_transactions()
    assert db.transaction_count() == 0


def test_set_and_get_budget(temp_db):
    db.set_budget("2026-01", "Food", 5000)
    budgets = db.get_budgets(month="2026-01")
    assert len(budgets) == 1
    assert budgets.iloc[0]["budget_amount"] == 5000


def test_set_budget_upsert(temp_db):
    db.set_budget("2026-01", "Food", 5000)
    db.set_budget("2026-01", "Food", 6000)  # should update, not duplicate
    budgets = db.get_budgets(month="2026-01")
    assert len(budgets) == 1
    assert budgets.iloc[0]["budget_amount"] == 6000


def test_set_budget_rejects_non_positive(temp_db):
    with pytest.raises(ValueError):
        db.set_budget("2026-01", "Food", 0)
    with pytest.raises(ValueError):
        db.set_budget("2026-01", "Food", -100)


def test_empty_database_returns_empty_frame(temp_db):
    df = db.get_all_transactions()
    assert df.empty
