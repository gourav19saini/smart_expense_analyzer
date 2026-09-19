# PROJECT DOCUMENTATION
## Smart Expense Pattern Analyzer

---

## 1. Introduction

Personal financial management is a challenge for most individuals because
spending data is scattered across cash, cards, UPI apps, and bank
statements, with no single place to see patterns. Smart Expense Pattern
Analyzer is a self-contained desktop-style web application (built with
Streamlit) that consolidates manually entered and CSV-imported transactions
into one SQLite database, then applies data analysis and machine learning
to help the user understand and predict their own spending behavior.

## 2. Problem Statement

Individuals who want to understand their spending habits typically have to:
manually reconcile multiple payment sources, manually categorize every
transaction, and manually notice patterns or unusual spending - a slow,
error-prone, and easily abandoned process. There is a need for a single
tool that ingests transaction data (manual or CSV), automatically organizes
and categorizes it, and surfaces analysis and predictions without requiring
the user to write a single spreadsheet formula.

## 3. Objectives

1. Provide a simple interface for manual transaction entry and CSV import.
2. Store all transactions reliably and queryably in a relational database.
3. Automatically categorize transactions using a trainable ML model.
4. Compute descriptive statistics and visualizations over income/expenses.
5. Detect transactions that deviate from the user's normal spending pattern.
6. Predict likely spending for the remainder of the current month.
7. Let the user set and track category-level monthly budgets.
8. Present all of the above through a single, coherent dashboard.

## 4. Existing System

Most people currently rely on: (a) manual spreadsheets, which require
ongoing manual data entry and formula maintenance; (b) bank/UPI app
statements, which show a list of transactions but rarely provide
cross-source analysis, categorization, or prediction; or (c) generic
budgeting apps, which are often subscription-based, require linking a real
bank account, and provide little transparency into how their "insights"
are computed.

## 5. Proposed System

Smart Expense Pattern Analyzer is a local, transparent alternative: every
number shown on the dashboard is computed live, in view-able Python code,
from data the user explicitly provided. There is no black-box bank
integration and no hidden fee. The system combines:

- A validated data-entry and CSV-import pipeline
- A relational (SQLite) store as the single source of truth
- Three explainable ML components (categorization, anomaly detection,
  prediction) chosen specifically because their logic can be explained in
  a few sentences
- A Streamlit dashboard that recomputes everything from the database on
  every interaction, so there is never stale or hard-coded data

## 6. System Architecture

```
User
  |
Streamlit UI  (app.py - 8 pages via sidebar navigation)
  |
Application Logic  (per-page handlers: validate input, call lower layers)
  |
Data Processing Layer  (utils/data_cleaning.py - CSV normalization/validation)
  |
SQLite Database  (database/db.py - transactions & budgets tables)
  |
Analytics / ML Layer  (analytics/expense_analysis.py, models/categorizer.py,
                        models/anomaly_detector.py, models/predictor.py)
  |
Dashboard  (Matplotlib charts + Streamlit metrics/tables rendered in app.py)
```

## 7. Functional Requirements

- FR1: The system shall allow adding a transaction with type, date, amount,
  category, merchant, description, and payment method, with validation.
- FR2: The system shall allow importing a CSV of transactions, reporting
  imported/skipped/duplicate counts.
- FR3: The system shall auto-suggest a category for transactions missing one.
- FR4: The system shall compute total income, total expenses, balance, and
  descriptive statistics (mean, median, min, max) over expenses.
- FR5: The system shall break down spending by category, by day, by week,
  by month, and by weekday vs weekend.
- FR6: The system shall flag statistically unusual transactions.
- FR7: The system shall predict estimated month-end spending when enough
  history exists.
- FR8: The system shall allow setting and tracking category budgets per month.
- FR9: The system shall allow searching, filtering, editing, deleting, and
  exporting transaction history.
- FR10: The system shall function correctly with zero transactions.

## 8. Non-Functional Requirements

- **Usability**: Navigable via a persistent sidebar; every page shows
  clear success/warning/error messages, never a raw Python traceback.
- **Reliability**: All SQL uses parameterized queries; all numeric inputs
  are validated before reaching the database.
- **Performance**: Designed for a single user's personal transaction volume
  (hundreds to low thousands of rows) - Pandas operations comfortably
  handle this in memory.
- **Portability**: Pure Python + SQLite; runs identically on Windows,
  macOS, and Linux with no external services.
- **Maintainability**: Each concern (database, cleaning, analytics, each ML
  model) lives in its own module with a docstring explaining its role.

## 9. Database Design

See `README.md` section 11 for the full schema. In short: a `transactions`
table (one row per income/expense entry) and a `budgets` table (one row per
month+category budget, unique on that pair so re-setting a budget updates
it in place).

## 10. Data Flow

**Manual entry**: Streamlit form → `db.validate_transaction()` →
`db.add_transaction()` → SQLite → next `db.get_all_transactions()` call
picks it up automatically.

**CSV import**:
```
CSV file
  -> pandas.read_csv
  -> utils.data_cleaning.normalize_columns (header aliasing)
  -> type coercion (dates, amounts) + normalization (type, category, payment method)
  -> validity mask (drops rows with missing/invalid date, amount, or type)
  -> duplicate drop (same date+amount+type+merchant+description)
  -> rows missing a category run through models.categorizer
  -> db.add_transactions_bulk
  -> dashboard reflects new data on next read
```

## 11. Algorithms

### 11.1 Pandas Preprocessing
Column renaming via a lookup dictionary (handles header variants like
`Txn Date`, `Desc`, `Type`); `pd.to_datetime(..., errors="coerce")` turns
unparseable dates into `NaT` so they can be filtered out instead of
crashing the import; `pd.to_numeric(..., errors="coerce")` does the same
for amounts.

### 11.2 NumPy Calculations
Used inside anomaly detection features (day-of-week extraction, category
average encoding) and inside the predictor's rolling-average and cumulative
sum calculations, as well as `np.where` for vectorized month-over-month
percentage-change calculations that avoid division by zero.

### 11.3 TF-IDF (Term Frequency - Inverse Document Frequency)
Converts each merchant/description string into a vector of numbers, where
words that are distinctive to a particular category (e.g. "SWIGGY",
"ELECTRICITY") get a high score, and common/uninformative words get a low
score. This turns text into something a classifier can work with.

### 11.4 Logistic Regression
A classifier that, given the TF-IDF vector, estimates the probability of
each possible category and picks the highest one. Chosen for
categorization because it is fast to train on small datasets, and its
predicted probabilities give a natural "confidence" score used to decide
when to say "Other / Needs Review" instead of guessing.

### 11.5 Linear Regression
Fits a straight-line-style relationship between a day's features
(day of month, day of week, 7-day rolling average, cumulative spend so far
that month) and that day's actual spending. Used to project spending for
each remaining day of the month, summed into a month-end estimate. Chosen
for its transparency - the influence of each feature can be inspected
directly via its coefficient - which matters for a project meant to be
explained in a viva.

### 11.6 Isolation Forest
An unsupervised algorithm that builds random decision trees which split
data on random feature values. Points that get isolated into their own
leaf node in very few splits are considered anomalies, because they differ
sharply from the bulk of the data on at least one feature. Applied here to
amount, day-of-week, and category-average-amount for each expense. Chosen
over simpler z-score methods because it naturally handles multiple features
at once without assuming a normal distribution.

### 11.7 K-Means
Not used in the current version. (Left here because the project brief asks
it to be documented if implemented - a natural extension would be
clustering transactions by amount+category+frequency to discover spending
"personas", but this was intentionally left out of v1 to keep the ML
surface area small, explainable, and reliable on small personal datasets.)

## 12. Results

On the bundled ~375-transaction synthetic demo dataset (spanning
2025-10-01 to the day the data was last generated):

- The categorizer correctly classifies well-known merchants (SWIGGY →
  Food, UBER → Transport, NETFLIX → Entertainment, AMAZON → Shopping,
  ELECTRICITY BILL → Bills) with confidence typically above 80%, and
  correctly falls back to "Other / Needs Review" for unrecognized text.
- The anomaly detector correctly flags all of the intentionally-injected
  large synthetic transactions (e.g. a ₹25,000 miscellaneous expense, a
  ₹22,000 hospital bill, an ₹18,500 electronics purchase) as unusual. It
  also occasionally flags recurring but large fixed costs (like a monthly
  ₹12,000 rent payment) - a known, expected limitation of unsupervised
  outlier detection discussed in the Limitations section.
- The predictor produces a month-end estimate that tracks the historical
  monthly average within a reasonable margin once at least two weeks of
  data exist, and correctly reports "not enough data" before that point.

## 13. Limitations

See `README.md` section 14 for the full list (single-user/local only,
categorizer depends on training data size and variety, anomaly detector
will sometimes flag legitimate recurring large costs, prediction is a
simple linear model, and CSV date parsing can be ambiguous for
non-ISO-format slash dates).

## 14. Future Scope

See `README.md` section 13 (real bank/UPI integration with proper
authorization, multi-user accounts, recurring-transaction detection, richer
multi-feature anomaly detection, PDF export, mobile layout).

## 15. Conclusion

Smart Expense Pattern Analyzer demonstrates a complete, working, end-to-end
data application: validated data entry and import, a normalized relational
store, real descriptive analytics, and three distinct, explainable machine
learning techniques applied to a genuinely useful personal-finance problem
- all while being transparent about what each component can and cannot do,
and handling missing or insufficient data gracefully rather than
fabricating results.

---

## 16. Twenty Likely Viva Questions & Answers

**Q1. What problem does this project solve?**
It consolidates manually entered and CSV-imported personal transactions
into one place and applies analytics and ML to reveal spending patterns,
flag unusual transactions, and predict future spending - without needing
any real bank integration.

**Q2. Why did you choose SQLite instead of MySQL/PostgreSQL?**
SQLite requires no separate server process, ships with Python's standard
library, and is perfectly suited to a single-user local application of
this size. It still supports full SQL, transactions, and constraints.

**Q3. How do you prevent SQL injection?**
Every query uses parameterized placeholders (`?`) via `sqlite3`'s
parameter-binding, never Python string formatting/concatenation into SQL.

**Q4. How does automatic categorization work?**
A merchant/description string is vectorized with TF-IDF (scoring which
words are distinctive) and classified with Logistic Regression, trained on
~90 seed examples plus the user's own manually-categorized history. If the
model's confidence is below a threshold (35%), it returns "Other / Needs
Review" instead of a low-confidence guess.

**Q5. What is TF-IDF and why use it here?**
TF-IDF (Term Frequency - Inverse Document Frequency) turns text into
numeric vectors where words common across all categories are down-weighted
and words distinctive to a category are up-weighted. It's simple, fast,
and works well on short merchant strings, unlike deep NLP models that need
much more training data.

**Q6. Why Logistic Regression for categorization rather than a neural network?**
The training set is small (dozens to low hundreds of examples). Logistic
Regression trains in milliseconds, doesn't overfit on small data the way a
neural network could, and its probability outputs give a natural, easily
explained confidence score.

**Q7. How does anomaly detection work, and what algorithm powers it?**
IsolationForest, an unsupervised scikit-learn algorithm, builds random
trees that split on feature values; points isolated in very few splits are
flagged as anomalies. Features used are amount, day-of-week, and the
transaction's category's average amount.

**Q8. Is the anomaly detector a fraud detector?**
No. It only identifies statistical outliers relative to the user's own
history. It cannot know whether a transaction was authorized, legitimate,
or fraudulent - it is explicitly labeled as such in the UI.

**Q9. What happens if there isn't enough data for anomaly detection or prediction?**
Both modules check a minimum data threshold (15 expense transactions for
anomaly detection, 14 days of history for prediction) and return a
clear "not enough data" message instead of producing an unreliable or
fabricated result.

**Q10. How does the expense prediction model work?**
A Linear Regression model is trained on the user's daily expense history
using features like day of month, day of week, a 7-day rolling average,
and cumulative spend so far that month. It then predicts each remaining
day of the current month one at a time, feeding each prediction back in as
history for the next day, and sums them into a month-end estimate.

**Q11. Why Linear Regression instead of a more advanced time-series model
(e.g. ARIMA, LSTM)?**
Personal expense data is noisy and limited in volume (a year of data is
only ~365 points). Linear Regression is transparent, trains instantly, and
its coefficients can be directly explained, which matters both for user
trust and for a student viva. More advanced models would need far more
data to outperform it meaningfully here.

**Q12. How is the CSV import made robust to messy real-world data?**
A column-alias dictionary maps many header spellings to canonical names;
dates and amounts are parsed with `errors="coerce"` so bad values become
missing rather than crashing; a validity mask drops rows missing a
required field; duplicates are dropped by matching on
date+amount+type+merchant+description; and a report of
imported/skipped/duplicate counts, plus reasons, is shown to the user.

**Q13. How do you decide whether a CSV row is Income or Expense if the
column is missing or unclear?**
`normalize_transaction_type()` matches common synonyms (credit/debit,
cr/dr, deposit/withdrawal). If still missing, the sign of the raw amount
column is used as a fallback signal before defaulting to Expense.

**Q14. What happens when the database is empty?**
Every analytics function checks for an empty DataFrame first and returns
a zeroed/empty result instead of raising an exception; the UI shows a
friendly "No transactions available" message rather than a blank or
broken page.

**Q15. How are budgets tracked?**
A `budgets` table stores a `(month, category)` unique pair with a monthly
amount. `analytics.budget_progress()` sums the month's actual expenses per
category from the `transactions` table and compares them against the
budgeted amount to compute percentage used and whether it's exceeded.

**Q16. How does editing or deleting a transaction affect the dashboard?**
The UI never caches transaction data across the session - every page call
re-reads `db.get_all_transactions()` fresh, so any edit or delete is
immediately reflected in every KPI, chart, and insight the next time a
page renders.

**Q17. What testing was done, and how?**
A pytest suite (`tests/`) covers: database schema creation, transaction
and budget CRUD plus their validation rules, CSV cleaning edge cases
(empty file, missing columns, invalid rows, duplicates, alternate
headers), category/monthly/budget analytics calculations, and all three ML
modules including their "insufficient data" fallback paths.

**Q18. How would this scale to multiple users?**
The current schema has no `user_id` column and no authentication layer.
Adding multi-user support would mean adding a `users` table, a `user_id`
foreign key on `transactions` and `budgets`, and an authentication/session
layer in Streamlit (e.g. `streamlit-authenticator`) - a natural next step
noted in Future Scope.

**Q19. Why does the anomaly detector sometimes flag a legitimate recurring
transaction like rent?**
IsolationForest looks at amount relative to the whole distribution of
expenses, not at whether a transaction repeats. A consistently large but
recurring cost (e.g. monthly rent) can still be far from the bulk of
smaller day-to-day expenses and get flagged. This is disclosed as a known
limitation rather than hidden.

**Q20. What would you improve if you had more time?**
Recurring-transaction awareness so the anomaly detector doesn't flag known
regular bills; a richer feature set for anomaly detection (recency,
transaction frequency;) proper user authentication for multi-user use; and
optionally a more sophisticated time-series forecasting model once more
historical data is available.
