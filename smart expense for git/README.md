# Smart Expense Pattern Analyzer

A personal expense analytics application built with Streamlit, Pandas, NumPy,
Matplotlib, Scikit-learn, and SQLite. Built as an industrial-training / Data
Science project to demonstrate a complete, working data pipeline: data entry
→ cleaning → storage → analysis → machine learning → visualization.

> **Not a real banking app.** This project does not connect to any bank,
> UPI provider, or payment gateway. All data comes from what you type in
> manually or import from a CSV file you provide.

---

## 1. Project Description

Smart Expense Pattern Analyzer lets a user track income and expenses, import
historical bank/UPI statements, automatically categorize transactions using
a small trained ML model, detect unusual transactions, predict end-of-month
spending, track category budgets, and see everything on a single dashboard.

## 2. Features

- Manual income/expense entry with full validation
- CSV import with robust cleaning (handles messy headers, bad dates,
  invalid amounts, duplicates) and a clear import report
- Automatic transaction categorization (TF-IDF + Logistic Regression)
  with a "needs review" fallback when confidence is low
- Expense analytics: totals, averages, category breakdowns, daily / weekly
  / monthly trends, weekday vs weekend comparison, payment-method split
- Anomaly detection (IsolationForest) for unusually large/unusual transactions
- Expense prediction (Linear Regression) for estimated month-end spending
- Category-wise monthly budgets with progress bars and over-budget alerts
- Dynamically generated "Smart Insights" - every message is computed from
  real data, never a canned string
- Searchable/filterable expense history with inline edit, delete, and
  CSV export
- Graceful handling of an empty database and of too-little-data-for-ML cases
- Full pytest test suite covering the database, analytics, and ML modules

## 3. Technology Stack

| Layer            | Technology                     |
|-------------------|--------------------------------|
| UI                | Streamlit                      |
| Data processing   | Pandas, NumPy                  |
| Visualization     | Matplotlib                     |
| Machine Learning  | Scikit-learn (TF-IDF + Logistic Regression, IsolationForest, Linear Regression) |
| Database          | SQLite (via Python's built-in `sqlite3`) |
| Testing           | Pytest                         |

## 4. Architecture

```
User
  |
Streamlit UI  (app.py)
  |
Application Logic (page handlers in app.py)
  |
Data Processing Layer (utils/data_cleaning.py)
  |
SQLite Database (database/db.py)
  |
Analytics / ML Layer (analytics/expense_analysis.py, models/*.py)
  |
Dashboard (rendered back in app.py via Matplotlib + Streamlit widgets)
```

Every page reads fresh data from SQLite on each interaction, so adding,
editing, or deleting a transaction is immediately reflected everywhere -
dashboard KPIs, charts, insights, and budget progress all recompute live
from the database.

## 5. Folder Structure

```
smart-expense-analyzer/
│
├── app.py                       # Streamlit application (all pages)
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md      # Viva-oriented documentation
├── .gitignore
│
├── database/
│   └── db.py                    # Schema, CRUD, validation
│
├── models/
│   ├── categorizer.py           # TF-IDF + Logistic Regression
│   ├── predictor.py             # Linear Regression month-end prediction
│   └── anomaly_detector.py      # IsolationForest outlier detection
│
├── analytics/
│   └── expense_analysis.py      # Pandas/NumPy calculations
│
├── utils/
│   ├── data_cleaning.py         # CSV import cleaning pipeline
│   └── insights.py              # Dynamic insight-string generation
│
├── data/
│   ├── sample_transactions.csv  # ~375 synthetic demo transactions
│   ├── generate_sample_data.py  # Script that generated the CSV above
│   └── expenses.db              # Created automatically on first run
│
└── tests/
    ├── test_database.py
    ├── test_analytics.py
    └── test_models.py
```

## 6. Installation

Requires Python 3.11+ (tested on 3.12).

```bash
git clone <this-repo-or-copy-the-folder>
cd smart-expense-analyzer
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Running the Application

```bash
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). The
SQLite database (`data/expenses.db`) and any trained ML model file
(`data/categorizer_model.joblib`) are created automatically on first run -
there is no manual setup step.

## 8. Using the Sample Dataset

Two ways to load the bundled demo data:

1. In the app, go to **📥 Import CSV** → click **"📂 Load sample dataset"**.
2. Or upload `data/sample_transactions.csv` yourself through the same page.

The sample file has ~375 transactions spanning about a year, including
several intentionally unusual transactions (clearly documented as synthetic
in `generate_sample_data.py`) so anomaly detection and prediction have
something meaningful to show immediately. To regenerate it with different
random data:

```bash
python data/generate_sample_data.py
```

## 9. Running Tests

```bash
pytest tests/ -v
```

Tests cover: database CRUD and validation, CSV cleaning edge cases (empty
file, missing columns, invalid rows, duplicates, alternate headers),
category/monthly/budget analytics, the categorizer, anomaly detector, and
predictor (including their "not enough data" fallbacks).

## 10. ML Methodology (Plain-Language Summary)

- **Auto-categorization**: Turns a merchant/description string into
  categorized "words" using **TF-IDF** (a way of scoring which words matter
  most in short text), then a **Logistic Regression** classifier maps those
  scores to a category. It's trained on ~90 example merchant names covering
  every category, plus any transactions the user has manually categorized
  themselves - so it slowly personalizes. If the model isn't confident, it
  says "Other / Needs Review" instead of guessing.
- **Anomaly detection**: Uses **IsolationForest**, an algorithm that isolates
  data points by repeatedly splitting on random feature values. Transactions
  that are "easy to isolate" (unusual amount, unusual weekday, far from the
  category's typical amount) get flagged. This needs at least 15 expense
  transactions to run reliably, and is explicitly labeled as a statistical
  outlier detector, not fraud detection.
- **Expense prediction**: A **Linear Regression** model is trained on the
  user's own daily spending history (day of month, day of week, 7-day
  rolling average, cumulative month spend so far) and used to project
  spending for each remaining day of the current month. Needs at least 14
  days of history; otherwise the app says so rather than fabricating a
  number.

## 11. Database Schema

**transactions**

| column            | type | notes                                  |
|-------------------|------|-----------------------------------------|
| id                | INTEGER PK AUTOINCREMENT | |
| date              | TEXT | `YYYY-MM-DD` |
| amount            | REAL | must be > 0 |
| transaction_type  | TEXT | `Income` or `Expense` |
| category          | TEXT | Food, Transport, Shopping, Bills, Entertainment, Education, Healthcare, Salary, Other |
| merchant          | TEXT | |
| description       | TEXT | |
| payment_method    | TEXT | UPI, Cash, Card, Bank Transfer, Other |
| created_at        | TEXT | ISO timestamp, set automatically |

**budgets**

| column         | type | notes                          |
|----------------|------|----------------------------------|
| id             | INTEGER PK AUTOINCREMENT | |
| month          | TEXT | `YYYY-MM` |
| category       | TEXT | |
| budget_amount  | REAL | must be > 0 |
| created_at     | TEXT | ISO timestamp |

`(month, category)` is unique - setting a budget for a category you've
already budgeted that month updates it instead of creating a duplicate.

## 12. Screenshots

_Add screenshots here after running the app locally, e.g.:_

- `screenshots/dashboard.png`
- `screenshots/add_transaction.png`
- `screenshots/ml_insights.png`
- `screenshots/budget.png`

## 13. Future Enhancements

- Real bank/UPI API integration (would require authorized, user-consented
  API access - intentionally out of scope for this student project)
- Multi-user accounts with authentication
- Recurring-transaction detection and automatic entry
- More advanced anomaly detection combining amount, frequency, and merchant
  novelty together
- Export to PDF monthly statements
- Mobile-friendly layout

## 14. Limitations

- Single-user, local-only application (no authentication)
- The categorizer's accuracy depends on the size and variety of its
  training examples; unfamiliar merchants may need manual correction
- Anomaly detection needs a minimum amount of history (15+ expense
  transactions) to run at all, and by design will also occasionally flag
  legitimate but infrequent large transactions (e.g. rent) - this is a
  known and expected characteristic of unsupervised outlier detection, not
  a bug
- Prediction is a simple linear trend model, not a sophisticated
  time-series forecaster; it is meant to be explainable, not maximally
  accurate
- CSV date parsing can be ambiguous for slash-separated dates without a
  clear day/month order (e.g. `01/09/2026`) - ISO format (`YYYY-MM-DD`) is
  always parsed unambiguously
