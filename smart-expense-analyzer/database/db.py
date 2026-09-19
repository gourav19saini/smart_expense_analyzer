"""
database/db.py
---------------
SQLite database layer for the Smart Expense Pattern Analyzer.

Handles:
    - Database + table creation (auto-runs on first import)
    - Transaction CRUD (create, read, update, delete)
    - Budget CRUD
    - All queries use parameterized SQL to avoid injection issues.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager
import pandas as pd

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "expenses.db")

VALID_TYPES = ("Income", "Expense")
VALID_CATEGORIES = (
    "Food", "Transport", "Shopping", "Bills", "Entertainment",
    "Education", "Healthcare", "Salary", "Other",
)
VALID_PAYMENT_METHODS = ("UPI", "Cash", "Card", "Bank Transfer", "Other")


@contextmanager
def get_connection():
    """Context-managed SQLite connection with foreign keys / row factory set."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they do not already exist. Safe to call many times."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL CHECK (amount > 0),
                transaction_type TEXT NOT NULL CHECK (transaction_type IN ('Income','Expense')),
                category TEXT NOT NULL,
                merchant TEXT,
                description TEXT,
                payment_method TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                month TEXT NOT NULL,
                category TEXT NOT NULL,
                budget_amount REAL NOT NULL CHECK (budget_amount > 0),
                created_at TEXT NOT NULL,
                UNIQUE(month, category)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category)")


# ---------------------------------------------------------------------------
# Transaction operations
# ---------------------------------------------------------------------------

def add_transaction(date, amount, transaction_type, category, merchant="",
                     description="", payment_method="Other"):
    """Insert a single transaction. Raises ValueError on invalid input."""
    if amount is None or float(amount) <= 0:
        raise ValueError("Amount must be a positive number.")
    if transaction_type not in VALID_TYPES:
        raise ValueError(f"transaction_type must be one of {VALID_TYPES}")
    try:
        pd.to_datetime(date)
    except Exception:
        raise ValueError("Invalid date.")

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO transactions
               (date, amount, transaction_type, category, merchant, description,
                payment_method, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (str(pd.to_datetime(date).date()), float(amount), transaction_type,
             category or "Other", merchant, description, payment_method,
             datetime.now().isoformat(timespec="seconds")),
        )


def add_transactions_bulk(df: pd.DataFrame):
    """Insert many transactions at once from a cleaned DataFrame."""
    required = ["date", "amount", "transaction_type", "category",
                "merchant", "description", "payment_method"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    rows = []
    now = datetime.now().isoformat(timespec="seconds")
    for _, r in df.iterrows():
        rows.append((
            str(pd.to_datetime(r["date"]).date()),
            float(r["amount"]),
            r["transaction_type"],
            r["category"] if r["category"] else "Other",
            r.get("merchant", ""),
            r.get("description", ""),
            r.get("payment_method", "Other") or "Other",
            now,
        ))

    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO transactions
               (date, amount, transaction_type, category, merchant, description,
                payment_method, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
    return len(rows)


def get_all_transactions() -> pd.DataFrame:
    """Return every transaction as a DataFrame, sorted by date descending."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY date DESC, id DESC", conn
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def update_transaction(tx_id, **fields):
    """Update arbitrary allowed fields of a transaction by id."""
    allowed = {"date", "amount", "transaction_type", "category",
               "merchant", "description", "payment_method"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [tx_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE transactions SET {set_clause} WHERE id = ?", values)


def delete_transaction(tx_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))


def delete_all_transactions():
    with get_connection() as conn:
        conn.execute("DELETE FROM transactions")


def transaction_count() -> int:
    with get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) AS c FROM transactions")
        return cur.fetchone()["c"]


# ---------------------------------------------------------------------------
# Budget operations
# ---------------------------------------------------------------------------

def set_budget(month, category, budget_amount):
    if budget_amount is None or float(budget_amount) <= 0:
        raise ValueError("Budget amount must be positive.")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO budgets (month, category, budget_amount, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(month, category)
               DO UPDATE SET budget_amount = excluded.budget_amount""",
            (month, category, float(budget_amount),
             datetime.now().isoformat(timespec="seconds")),
        )


def get_budgets(month=None) -> pd.DataFrame:
    with get_connection() as conn:
        if month:
            df = pd.read_sql_query(
                "SELECT * FROM budgets WHERE month = ?", conn, params=(month,)
            )
        else:
            df = pd.read_sql_query("SELECT * FROM budgets", conn)
    return df


def delete_budget(budget_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))


# Initialize automatically on import so every entry point (app or tests)
# is guaranteed to have the schema in place.
init_db()
