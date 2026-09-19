"""
app.py
-------
Smart Expense Pattern Analyzer - main Streamlit application.

Run with:
    streamlit run app.py

Pages (sidebar navigation):
    Dashboard, Add Transaction, Import CSV, Expense History,
    Analytics, ML Insights, Budget, Settings
"""

import io
import sys
import os
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import db
from utils import data_cleaning, insights as insights_mod
from analytics import expense_analysis as ea
from models import categorizer, anomaly_detector, predictor

# ---------------------------------------------------------------------------
# Page config & one-time setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Expense Pattern Analyzer",
    page_icon="💰",
    layout="wide",
)

db.init_db()

CATEGORIES = list(db.VALID_CATEGORIES)
PAYMENT_METHODS = list(db.VALID_PAYMENT_METHODS)
CURRENCY = "\u20b9"  # ₹


def load_data() -> pd.DataFrame:
    """Always read fresh from the database so the UI reflects the latest state."""
    return db.get_all_transactions()


def fmt_money(x) -> str:
    try:
        return f"{CURRENCY}{x:,.0f}"
    except Exception:
        return f"{CURRENCY}0"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("💰 Expense Analyzer")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "➕ Add Transaction",
        "📥 Import CSV",
        "📜 Expense History",
        "📊 Analytics",
        "🤖 ML Insights",
        "💰 Budget",
        "⚙️ Settings",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Personal-finance demo project. This app does not connect to any real "
    "bank, UPI provider, or payment system. All data is entered manually "
    "or imported by you from a CSV file."
)

df = load_data()

# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------

if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")

    if df.empty:
        st.info(
            "No transactions available.\n\n"
            "Add a transaction or import a CSV file to start analyzing your expenses."
        )
    else:
        summary = ea.overall_summary(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Income", fmt_money(summary["total_income"]))
        c2.metric("Total Expenses", fmt_money(summary["total_expenses"]))
        c3.metric("Balance", fmt_money(summary["balance"]))
        daily = ea.daily_spending(df)
        avg_daily = float(daily.mean()) if not daily.empty else 0.0
        c4.metric("Avg Daily Expense", fmt_money(avg_daily))

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Expense Trend (Monthly)")
            monthly = ea.monthly_spending(df)
            if not monthly.empty:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(monthly.index.astype(str), monthly.values, marker="o", color="#d62728")
                ax.set_ylabel(f"Amount ({CURRENCY})")
                ax.set_xlabel("Month")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("Not enough data yet.")

        with col2:
            st.subheader("Category Distribution")
            cat = ea.category_breakdown(df)
            if not cat.empty:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(cat["category"], cat["total"], color="#1f77b4")
                ax.set_ylabel(f"Total ({CURRENCY})")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("Not enough data yet.")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Income vs Expense")
            iv = ea.monthly_income_vs_expense(df)
            if not iv.empty:
                fig, ax = plt.subplots(figsize=(6, 3.5))
                x = np.arange(len(iv))
                width = 0.35
                ax.bar(x - width / 2, iv["Income"], width, label="Income", color="#2ca02c")
                ax.bar(x + width / 2, iv["Expense"], width, label="Expense", color="#d62728")
                ax.set_xticks(x)
                ax.set_xticklabels(iv["month"], rotation=45, ha="right")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.caption("Not enough data yet.")

        with col4:
            st.subheader("Top Categories")
            cat = ea.category_breakdown(df)
            if not cat.empty:
                st.dataframe(
                    cat.head(5).rename(columns={
                        "category": "Category", "total": "Total",
                        "percentage": "% of total", "avg_transaction": "Avg / txn",
                        "count": "Count",
                    }).style.format({"Total": "{:.0f}", "% of total": "{:.1f}", "Avg / txn": "{:.0f}"}),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.caption("Not enough data yet.")

        st.markdown("---")
        st.subheader("Recent Transactions")
        recent = df.head(8)[["date", "transaction_type", "category", "merchant", "amount", "payment_method"]]
        st.dataframe(recent, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("💡 Smart Insights")
        budgets_df = db.get_budgets()
        all_insights = insights_mod.generate_insights(df, budgets_df)
        if not all_insights:
            st.caption("Add more transactions to unlock insights.")
        else:
            icon_map = {"insight": "💡", "alert": "⚠️", "prediction": "📊"}
            for item in all_insights:
                st.write(f"{icon_map.get(item['type'], '•')} {item['text']}")

# ---------------------------------------------------------------------------
# ADD TRANSACTION
# ---------------------------------------------------------------------------

elif page == "➕ Add Transaction":
    st.title("➕ Add Transaction")

    with st.form("add_transaction_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            t_type = st.selectbox("Transaction Type", list(db.VALID_TYPES))
            t_date = st.date_input("Date", value=date.today())
            amount = st.number_input("Amount", min_value=0.0, step=10.0, format="%.2f")
            category = st.selectbox("Category", CATEGORIES)
        with col2:
            merchant = st.text_input("Merchant")
            description = st.text_input("Description")
            payment_method = st.selectbox("Payment Method", PAYMENT_METHODS)

        submitted = st.form_submit_button("💾 Save Transaction")

        if submitted:
            try:
                db.add_transaction(
                    date=t_date.strftime("%Y-%m-%d"),
                    amount=amount,
                    transaction_type=t_type,
                    category=category,
                    merchant=merchant,
                    description=description,
                    payment_method=payment_method,
                )
                st.success(
                    f"Saved: {t_type} of {fmt_money(amount)} in {category} on {t_date.strftime('%d-%m-%Y')}."
                )
                st.rerun()
            except ValueError as e:
                st.error(f"Could not save transaction: {e}")
            except Exception:
                st.error("Something went wrong while saving this transaction. Please check your inputs.")

    st.markdown("---")
    st.caption(
        "Tip: leave Category as 'Other' if unsure - the ML Insights page can "
        "suggest a category automatically based on merchant/description."
    )

# ---------------------------------------------------------------------------
# IMPORT CSV
# ---------------------------------------------------------------------------

elif page == "📥 Import CSV":
    st.title("📥 Import CSV")
    st.write(
        "Upload a transaction file with columns similar to: "
        "`date, amount, transaction_type, category, merchant, description, payment_method`. "
        "Common header variations (e.g. `Type`, `Desc`, `Txn Date`) are handled automatically."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read this file. Please make sure it is a valid CSV.")
            raw_df = None

        if raw_df is not None:
            clean_df, report = data_cleaning.clean_transactions_csv(raw_df)

            st.subheader("Import Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Imported", report["imported"])
            c2.metric("Skipped", report["skipped"])
            c3.metric("Duplicates removed", report["duplicates"])

            for issue in report["issues"]:
                st.warning(issue)

            if not clean_df.empty:
                needs_category = clean_df["category"] == ""
                if needs_category.any():
                    st.info(
                        f"{int(needs_category.sum())} row(s) have no category and will be "
                        f"auto-categorized using the ML categorizer."
                    )
                    history = df if not df.empty else None
                    model = categorizer.load_or_train_model(history_df=history)
                    texts = (
                        clean_df.loc[needs_category, "merchant"].fillna("") + " " +
                        clean_df.loc[needs_category, "description"].fillna("")
                    )
                    preds = categorizer.predict_categories_bulk(texts, model=model)
                    clean_df.loc[needs_category, "category"] = [p[0] for p in preds]

                st.subheader("Preview (first 20 rows)")
                st.dataframe(clean_df.head(20), use_container_width=True, hide_index=True)

                if st.button("✅ Confirm Import"):
                    n = db.add_transactions_bulk(clean_df)
                    st.success(f"Imported {n} transactions successfully.")
                    st.rerun()
            else:
                st.error("No valid rows could be imported from this file.")

    st.markdown("---")
    st.subheader("Or try the bundled sample dataset")
    st.caption(
        "`data/sample_transactions.csv` contains ~375 synthetic demo transactions "
        "spanning about a year, useful for exploring every feature immediately."
    )
    if st.button("📂 Load sample dataset"):
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_transactions.csv")
        if os.path.exists(sample_path):
            raw_df = pd.read_csv(sample_path)
            clean_df, report = data_cleaning.clean_transactions_csv(raw_df)
            n = db.add_transactions_bulk(clean_df)
            st.success(f"Loaded {n} sample transactions. Go to the Dashboard to explore.")
            st.rerun()
        else:
            st.error("Sample dataset file not found.")

# ---------------------------------------------------------------------------
# EXPENSE HISTORY
# ---------------------------------------------------------------------------

elif page == "📜 Expense History":
    st.title("📜 Expense History")

    if df.empty:
        st.info("No transactions available. Add a transaction or import a CSV file to get started.")
    else:
        with st.expander("🔍 Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                min_date = pd.to_datetime(df["date"]).min().date()
                max_date = pd.to_datetime(df["date"]).max().date()
                date_range = st.date_input("Date range", value=(min_date, max_date))
            with fc2:
                type_filter = st.multiselect("Transaction Type", list(db.VALID_TYPES), default=list(db.VALID_TYPES))
                category_filter = st.multiselect("Category", sorted(df["category"].unique()), default=list(sorted(df["category"].unique())))
            with fc3:
                payment_filter = st.multiselect("Payment Method", sorted(df["payment_method"].dropna().unique()), default=list(sorted(df["payment_method"].dropna().unique())))
                min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
                amt_range = st.slider("Amount range", min_value=0.0, max_value=max(max_amt, 1.0), value=(0.0, max(max_amt, 1.0)))

        filtered = df.copy()
        filtered["date"] = pd.to_datetime(filtered["date"])
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered = filtered[
                (filtered["date"].dt.date >= date_range[0]) & (filtered["date"].dt.date <= date_range[1])
            ]
        if type_filter:
            filtered = filtered[filtered["transaction_type"].isin(type_filter)]
        if category_filter:
            filtered = filtered[filtered["category"].isin(category_filter)]
        if payment_filter:
            filtered = filtered[filtered["payment_method"].isin(payment_filter)]
        filtered = filtered[(filtered["amount"] >= amt_range[0]) & (filtered["amount"] <= amt_range[1])]

        st.caption(f"Showing {len(filtered)} of {len(df)} transactions.")
        st.dataframe(
            filtered[["id", "date", "transaction_type", "category", "merchant", "description", "amount", "payment_method"]],
            use_container_width=True, hide_index=True,
        )

        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export filtered transactions to CSV", data=csv_bytes,
                            file_name="filtered_transactions.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Edit or Delete a Transaction")
        tx_id = st.number_input("Transaction ID", min_value=0, step=1)
        if tx_id and tx_id in df["id"].values:
            row = df[df["id"] == tx_id].iloc[0]
            ec1, ec2 = st.columns(2)
            with ec1:
                new_amount = st.number_input("New amount", value=float(row["amount"]), min_value=0.0, step=10.0)
                new_category = st.selectbox("New category", CATEGORIES, index=CATEGORIES.index(row["category"]) if row["category"] in CATEGORIES else 0)
            with ec2:
                new_merchant = st.text_input("New merchant", value=row["merchant"] or "")
                new_description = st.text_input("New description", value=row["description"] or "")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Update Transaction"):
                    db.update_transaction(int(tx_id), amount=new_amount, category=new_category,
                                           merchant=new_merchant, description=new_description)
                    st.success("Transaction updated.")
                    st.rerun()
            with b2:
                if st.button("🗑️ Delete Transaction"):
                    db.delete_transaction(int(tx_id))
                    st.success("Transaction deleted.")
                    st.rerun()
        elif tx_id:
            st.warning("No transaction found with that ID in the current data.")

# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------

elif page == "📊 Analytics":
    st.title("📊 Analytics")

    if df.empty:
        st.info("No transactions available. Add a transaction or import a CSV file to start analyzing your expenses.")
    else:
        summary = ea.overall_summary(df)
        st.subheader("Overall Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Expense", fmt_money(summary["avg_expense"]))
        s2.metric("Median Expense", fmt_money(summary["median_expense"]))
        s3.metric("Max Expense", fmt_money(summary["max_expense"]))
        s4.metric("Min Expense", fmt_money(summary["min_expense"]))

        st.markdown("---")
        st.subheader("Category-wise Spending")
        cat = ea.category_breakdown(df)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(cat["category"], cat["total"], color="#ff7f0e")
            ax.set_ylabel(f"Total ({CURRENCY})")
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.dataframe(cat, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Daily Spending Trend")
        daily = ea.daily_spending(df)
        if not daily.empty:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(list(daily.index), daily.values, color="#9467bd", linewidth=1)
            ax.set_ylabel(f"Amount ({CURRENCY})")
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Weekday vs Weekend")
            wk = ea.weekday_vs_weekend(df)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(["Weekday avg", "Weekend avg"], [wk["weekday_avg"], wk["weekend_avg"]],
                   color=["#1f77b4", "#d62728"])
            ax.set_ylabel(f"Avg daily spend ({CURRENCY})")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col4:
            st.subheader("Payment Method Breakdown")
            pay = ea.payment_method_breakdown(df)
            if not pay.empty:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.pie(pay.values, labels=pay.index, autopct="%1.0f%%")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        st.subheader("Month-over-Month Category Change")
        cat_mom = ea.category_month_over_month(df)
        if not cat_mom.empty:
            st.dataframe(cat_mom, use_container_width=True, hide_index=True)
        else:
            st.caption("Need at least two months of data to compare.")

# ---------------------------------------------------------------------------
# ML INSIGHTS
# ---------------------------------------------------------------------------

elif page == "🤖 ML Insights":
    st.title("🤖 ML Insights")

    if df.empty:
        st.info("No transactions available. Add a transaction or import a CSV file first.")
    else:
        tab1, tab2, tab3 = st.tabs(["🏷️ Auto-Categorization", "⚠️ Anomaly Detection", "📈 Spending Prediction"])

        with tab1:
            st.subheader("Try the Categorizer")
            st.caption(
                "This uses a TF-IDF + Logistic Regression model trained on common "
                "merchant/description examples plus your own categorized history. "
                "Low-confidence predictions are labeled 'Other / Needs Review' instead of guessed."
            )
            sample_text = st.text_input("Enter a merchant or description", "SWIGGY ORDER")
            if st.button("Predict Category"):
                model = categorizer.load_or_train_model(history_df=df)
                cat_pred, conf = categorizer.predict_category(sample_text, model=model)
                st.write(f"**Predicted category:** {cat_pred}  (confidence: {conf:.0%})")

        with tab2:
            st.subheader("Unusual Transactions")
            st.caption(
                "Detected using IsolationForest on amount, weekday, and category patterns. "
                "This is an algorithmic outlier detector, NOT a fraud detection system - "
                "a flagged transaction is simply statistically unusual."
            )
            flagged, msg = anomaly_detector.get_flagged_transactions(df, top_n=10)
            if msg:
                st.warning(msg)
            elif flagged.empty:
                st.success("No unusual transactions detected in your current data.")
            else:
                for _, row in flagged.iterrows():
                    st.warning(
                        f"⚠️ Unusual Transaction\n\n"
                        f"Amount: {fmt_money(row['amount'])}  |  Category: {row['category']}  |  "
                        f"Date: {pd.to_datetime(row['date']).strftime('%d-%m-%Y')}\n\n"
                        f"This transaction is significantly different from your historical spending pattern."
                    )

        with tab3:
            st.subheader("Expense Prediction")
            st.caption("Uses a Linear Regression model over your daily spending history.")
            pred = predictor.predict_month_end_spending(df)
            if pred["status"] == "insufficient_data":
                st.warning(pred["message"])
            else:
                p1, p2, p3 = st.columns(3)
                p1.metric("Current Spending", fmt_money(pred["current_spending"]))
                p2.metric("Estimated Month-End", fmt_money(pred["estimated_month_end"]))
                p3.metric("Historical Average", fmt_money(pred["historical_monthly_average"]))
                st.caption(f"Prediction is based on {pred['days_of_history']} days of expense history.")

# ---------------------------------------------------------------------------
# BUDGET
# ---------------------------------------------------------------------------

elif page == "💰 Budget":
    st.title("💰 Budget")

    current_month = datetime.now().strftime("%Y-%m")
    st.subheader(f"Set Budget for {current_month}")

    with st.form("budget_form", clear_on_submit=True):
        b1, b2 = st.columns(2)
        with b1:
            budget_category = st.selectbox("Category", CATEGORIES)
        with b2:
            budget_amount = st.number_input("Monthly budget amount", min_value=0.0, step=100.0)
        month_input = st.text_input("Month (YYYY-MM)", value=current_month)
        set_budget_btn = st.form_submit_button("💾 Save Budget")

        if set_budget_btn:
            try:
                db.set_budget(month_input, budget_category, budget_amount)
                st.success(f"Budget for {budget_category} in {month_input} set to {fmt_money(budget_amount)}.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Budget Status")
    budgets_df = db.get_budgets(month=current_month)
    if budgets_df.empty:
        st.info("No budgets set for this month yet. Use the form above to set one.")
    else:
        progress = ea.budget_progress(df, budgets_df, current_month)
        for _, row in progress.iterrows():
            st.write(f"**{row['category']}**")
            pct = min(row["percentage_used"] / 100, 1.0)
            st.progress(pct)
            status_line = f"{fmt_money(row['spent'])} / {fmt_money(row['budget_amount'])} ({row['percentage_used']:.0f}% used)"
            if row["exceeded"]:
                st.error(f"⚠️ {row['category']} budget exceeded. {status_line}")
            else:
                st.caption(status_line)

    st.markdown("---")
    st.subheader("All Budgets")
    all_budgets = db.get_budgets()
    if not all_budgets.empty:
        st.dataframe(all_budgets[["id", "month", "category", "budget_amount"]], use_container_width=True, hide_index=True)
        del_id = st.number_input("Budget ID to delete", min_value=0, step=1)
        if st.button("🗑️ Delete Budget") and del_id:
            db.delete_budget(int(del_id))
            st.success("Budget deleted.")
            st.rerun()

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.subheader("Export Data")
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export all transactions to CSV", data=csv_bytes,
                            file_name="all_transactions.csv", mime="text/csv")
    else:
        st.caption("No data to export yet.")

    st.markdown("---")
    st.subheader("⚠️ Danger Zone")
    st.caption("This permanently deletes all transactions from the database. This cannot be undone.")
    confirm = st.checkbox("I understand this will permanently delete all my transaction data.")
    if st.button("🗑️ Reset / Delete All Transactions", disabled=not confirm):
        db.delete_all_transactions()
        st.success("All transactions deleted.")
        st.rerun()

    st.markdown("---")
    st.subheader("About")
    st.write(
        "**Smart Expense Pattern Analyzer** is a personal-finance analytics demo built with "
        "Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn, and SQLite.\n\n"
        "This application does not connect to any real bank account, UPI provider, or "
        "payment gateway. It works only with data you enter manually or import via CSV. "
        "Anomaly detection here is a statistical outlier detector, not a fraud-detection system."
    )
