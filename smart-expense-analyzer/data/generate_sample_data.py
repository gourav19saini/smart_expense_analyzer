"""
One-off script used to generate data/sample_transactions.csv.
This is SYNTHETIC DEMO DATA ONLY - it does not represent any real
person's finances or any real bank/UPI transactions.

Run: python data/generate_sample_data.py
"""

import numpy as np
import pandas as pd
import random
from datetime import timedelta

random.seed(42)
np.random.seed(42)

START = pd.Timestamp("2025-10-01")
END = pd.Timestamp("2026-09-16")  # ~11.5 months of history

MERCHANTS = {
    "Food": ["SWIGGY", "ZOMATO", "DOMINOS PIZZA", "CAFE COFFEE DAY", "MCDONALDS",
              "BIGBASKET GROCERY", "LOCAL GROCERY STORE", "STARBUCKS", "BLINKIT"],
    "Transport": ["UBER", "OLA CABS", "PETROL PUMP", "METRO CARD RECHARGE",
                  "IRCTC TRAIN TICKET", "PARKING FEE", "RAPIDO"],
    "Shopping": ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "DECATHLON", "RELIANCE TRENDS"],
    "Bills": ["ELECTRICITY BILL", "MOBILE RECHARGE", "BROADBAND BILL", "WATER BILL",
              "GAS CYLINDER", "RENT PAYMENT"],
    "Entertainment": ["NETFLIX", "SPOTIFY", "BOOKMYSHOW", "PVR CINEMAS", "AMAZON PRIME"],
    "Education": ["UDEMY COURSE", "COLLEGE FEE", "BOOK STORE", "COURSERA SUBSCRIPTION"],
    "Healthcare": ["APOLLO PHARMACY", "DOCTOR CONSULTATION", "MEDICAL STORE", "LAB TEST"],
    "Other": ["ATM WITHDRAWAL", "MISCELLANEOUS", "GIFT", "DONATION"],
}

CATEGORY_AMOUNT_RANGE = {
    "Food": (80, 900),
    "Transport": (40, 600),
    "Shopping": (300, 4500),
    "Bills": (250, 3500),
    "Entertainment": (150, 800),
    "Education": (200, 6000),
    "Healthcare": (150, 2500),
    "Other": (100, 2000),
}

PAYMENT_METHODS = ["UPI", "Cash", "Card", "Bank Transfer"]

rows = []

# --- Monthly salary income --------------------------------------------------
month_starts = pd.date_range(START, END, freq="MS")
for m in month_starts:
    salary_date = m + timedelta(days=random.choice([0, 1, 2]))
    rows.append({
        "date": salary_date.strftime("%Y-%m-%d"),
        "amount": round(np.random.normal(65000, 2500), 2),
        "transaction_type": "Income",
        "category": "Salary",
        "merchant": "COMPANY PAYROLL",
        "description": "Monthly salary credit",
        "payment_method": "Bank Transfer",
    })
    # occasional freelance / other income
    if random.random() < 0.3:
        freelance_date = m + timedelta(days=random.randint(5, 25))
        rows.append({
            "date": freelance_date.strftime("%Y-%m-%d"),
            "amount": round(np.random.uniform(2000, 8000), 2),
            "transaction_type": "Income",
            "category": "Other",
            "merchant": "FREELANCE PAYMENT",
            "description": "Freelance project payment",
            "payment_method": "Bank Transfer",
        })

# --- Regular expenses across the whole period --------------------------------------------------
current = START
while current <= END:
    # Number of transactions per day varies; weekends slightly higher for food/shopping
    is_weekend = current.dayofweek >= 5
    n_tx = np.random.poisson(1.2 if is_weekend else 0.85)

    for _ in range(n_tx):
        category = random.choices(
            population=list(CATEGORY_AMOUNT_RANGE.keys()),
            weights=[30, 18, 15, 12, 10, 5, 7, 3],
            k=1,
        )[0]
        low, high = CATEGORY_AMOUNT_RANGE[category]
        amount = round(np.random.uniform(low, high), 2)
        merchant = random.choice(MERCHANTS[category])
        rows.append({
            "date": current.strftime("%Y-%m-%d"),
            "amount": amount,
            "transaction_type": "Expense",
            "category": category,
            "merchant": merchant,
            "description": f"{merchant.title()} purchase",
            "payment_method": random.choice(PAYMENT_METHODS),
        })
    current += timedelta(days=1)

# --- Recurring monthly bills (more realistic than pure randomness) --------------------------------------------------
for m in month_starts:
    for merchant, cat, amt_range in [
        ("ELECTRICITY BILL", "Bills", (900, 2200)),
        ("BROADBAND BILL", "Bills", (699, 999)),
        ("RENT PAYMENT", "Bills", (12000, 12000)),
        ("MOBILE RECHARGE", "Bills", (299, 599)),
    ]:
        bill_date = m + timedelta(days=random.randint(1, 8))
        amt = round(np.random.uniform(*amt_range), 2) if amt_range[0] != amt_range[1] else amt_range[0]
        rows.append({
            "date": bill_date.strftime("%Y-%m-%d"),
            "amount": amt,
            "transaction_type": "Expense",
            "category": "Bills",
            "merchant": merchant,
            "description": f"{merchant.title()}",
            "payment_method": random.choice(["UPI", "Bank Transfer"]),
        })

# --- Intentional anomalies for demonstrating anomaly detection (clearly synthetic) --------------------------------------------------
anomaly_dates = pd.date_range(START + timedelta(days=40), END - timedelta(days=10), periods=6)
anomaly_specs = [
    ("Shopping", 18500, "AMAZON", "Large one-off electronics purchase (synthetic demo anomaly)"),
    ("Healthcare", 22000, "HOSPITAL BILL", "Unplanned hospital visit (synthetic demo anomaly)"),
    ("Shopping", 15200, "FLIPKART", "Festival sale big-ticket purchase (synthetic demo anomaly)"),
    ("Entertainment", 9800, "BOOKMYSHOW", "Concert tickets for a group (synthetic demo anomaly)"),
    ("Transport", 7600, "IRCTC TRAIN TICKET", "Multiple long-distance tickets booked together (synthetic demo anomaly)"),
    ("Other", 25000, "MISCELLANEOUS", "Large unexpected miscellaneous expense (synthetic demo anomaly)"),
]
for dt, (cat, amt, merchant, desc) in zip(anomaly_dates, anomaly_specs):
    rows.append({
        "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
        "amount": amt,
        "transaction_type": "Expense",
        "category": cat,
        "merchant": merchant,
        "description": desc,
        "payment_method": "Card",
    })

df = pd.DataFrame(rows)
df = df.sort_values("date").reset_index(drop=True)

print(f"Generated {len(df)} transactions from {df['date'].min()} to {df['date'].max()}")
df.to_csv("data/sample_transactions.csv", index=False)
print("Saved to data/sample_transactions.csv")
