"""
utils/data_cleaning.py
-----------------------
Robust preprocessing for user-uploaded CSV transaction files.

Pipeline:
    raw CSV -> column normalization -> type coercion -> validation
    -> duplicate detection -> report

Returns a clean DataFrame plus a report dict, never raises on bad
data - bad rows are dropped and counted instead.
"""

import pandas as pd
import numpy as np
from database.db import VALID_TYPES, VALID_PAYMENT_METHODS

# Map many possible incoming header spellings to our canonical schema.
COLUMN_ALIASES = {
    "date": "date", "transaction date": "date", "txn date": "date",
    "amount": "amount", "amt": "amount", "value": "amount",
    "transaction_type": "transaction_type", "type": "transaction_type",
    "txn type": "transaction_type", "transaction type": "transaction_type",
    "category": "category", "cat": "category",
    "merchant": "merchant", "payee": "merchant", "source": "merchant",
    "vendor": "merchant",
    "description": "description", "desc": "description", "narration": "description",
    "remarks": "description", "particulars": "description",
    "payment_method": "payment_method", "payment method": "payment_method",
    "mode": "payment_method", "payment mode": "payment_method",
}

CANONICAL_COLUMNS = [
    "date", "amount", "transaction_type", "category",
    "merchant", "description", "payment_method",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns using COLUMN_ALIASES (case/space-insensitive)."""
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    df = df.rename(columns=rename_map)
    return df


def normalize_transaction_type(value) -> str:
    if pd.isna(value):
        return ""
    v = str(value).strip().lower()
    if v in ("income", "credit", "cr", "in", "deposit"):
        return "Income"
    if v in ("expense", "debit", "dr", "out", "withdrawal", "spend"):
        return "Expense"
    return ""


def normalize_payment_method(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "Other"
    v = str(value).strip().lower()
    mapping = {
        "upi": "UPI", "cash": "Cash", "card": "Card",
        "credit card": "Card", "debit card": "Card",
        "bank transfer": "Bank Transfer", "neft": "Bank Transfer",
        "imps": "Bank Transfer", "rtgs": "Bank Transfer",
    }
    return mapping.get(v, "Other")


def normalize_category(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    return str(value).strip().title()


def clean_transactions_csv(raw_df: pd.DataFrame):
    """
    Clean an uploaded transactions DataFrame.

    Returns
    -------
    clean_df : pd.DataFrame  (canonical columns, ready for DB insert;
                               'category' may be '' meaning "needs auto-categorization")
    report   : dict with counts of imported / skipped / duplicate rows
               and a list of human-readable reasons for skipped rows.
    """
    report = {"total_rows": len(raw_df), "imported": 0, "skipped": 0,
              "duplicates": 0, "issues": []}

    if raw_df.empty:
        report["issues"].append("The uploaded file has no rows.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS), report

    df = normalize_columns(raw_df.copy())

    for required in ("date", "amount"):
        if required not in df.columns:
            report["issues"].append(
                f"Required column '{required}' not found in file. "
                f"Found columns: {list(raw_df.columns)}"
            )
            return pd.DataFrame(columns=CANONICAL_COLUMNS), report

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # --- type coercion -----------------------------------------------
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=False)
    # try dayfirst too for rows that failed, common in Indian bank exports
    still_na = df["date_parsed"].isna()
    if still_na.any():
        df.loc[still_na, "date_parsed"] = pd.to_datetime(
            df.loc[still_na, "date"], errors="coerce", dayfirst=True
        )

    df["amount_parsed"] = (
        df["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    )
    df["amount_parsed"] = pd.to_numeric(df["amount_parsed"], errors="coerce")
    df["amount_parsed"] = df["amount_parsed"].abs()  # store as positive; type is separate

    df["transaction_type"] = df["transaction_type"].apply(normalize_transaction_type)
    # If type missing but original amount was negative, infer Expense; positive -> Income
    raw_amount_numeric = pd.to_numeric(
        df["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
    )
    inferred_expense = raw_amount_numeric < 0
    missing_type = df["transaction_type"] == ""
    df.loc[missing_type & inferred_expense, "transaction_type"] = "Expense"
    df.loc[missing_type & ~inferred_expense & raw_amount_numeric.notna(), "transaction_type"] = "Expense"

    df["category"] = df["category"].apply(normalize_category)
    df["payment_method"] = df["payment_method"].apply(normalize_payment_method)
    df["merchant"] = df["merchant"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # --- validation -----------------------------------------------------
    valid_mask = (
        df["date_parsed"].notna()
        & df["amount_parsed"].notna()
        & (df["amount_parsed"] > 0)
        & df["transaction_type"].isin(VALID_TYPES)
    )

    n_invalid = (~valid_mask).sum()
    if n_invalid:
        report["issues"].append(
            f"{n_invalid} row(s) skipped due to missing/invalid date, amount, or type."
        )

    clean = df.loc[valid_mask].copy()
    clean["date"] = clean["date_parsed"].dt.strftime("%Y-%m-%d")
    clean["amount"] = clean["amount_parsed"]
    clean = clean[CANONICAL_COLUMNS]

    # --- duplicate detection ------------------------------------------
    before = len(clean)
    clean = clean.drop_duplicates(
        subset=["date", "amount", "transaction_type", "merchant", "description"]
    )
    n_dupes = before - len(clean)

    report["imported"] = len(clean)
    report["skipped"] = int(n_invalid)
    report["duplicates"] = int(n_dupes)

    return clean.reset_index(drop=True), report
