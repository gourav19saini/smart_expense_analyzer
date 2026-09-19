"""
utils/insights.py
-------------------
Generates human-readable insight strings purely from computed
analytics. Every message here is backed by a real calculation -
nothing is a canned/static claim.
"""

import pandas as pd
from analytics import expense_analysis as ea
from models import anomaly_detector, predictor


def generate_insights(df: pd.DataFrame, budgets_df: pd.DataFrame = None) -> list:
    """Returns a list of dicts: {'type': 'insight'|'alert'|'prediction', 'text': str}"""
    insights = []
    if df.empty:
        return insights

    # --- category share -------------------------------------------------
    cat = ea.category_breakdown(df)
    if not cat.empty:
        top = cat.iloc[0]
        insights.append({
            "type": "insight",
            "text": f"{top['category']} represents {top['percentage']:.0f}% of your total expenses "
                    f"and is currently your largest expense category."
        })

    # --- month over month -------------------------------------------------
    pct_change, current, previous = ea.month_over_month_change(df)
    if pct_change is not None:
        direction = "increased" if pct_change > 0 else "decreased"
        insights.append({
            "type": "insight",
            "text": f"Your spending {direction} by {abs(pct_change):.0f}% compared with last month."
        })

    # --- category-level increases -------------------------------------------------
    cat_mom = ea.category_month_over_month(df)
    if not cat_mom.empty:
        risers = cat_mom[cat_mom["pct_change"] > 20].sort_values("pct_change", ascending=False)
        for _, r in risers.head(2).iterrows():
            insights.append({
                "type": "insight",
                "text": f"Your {r['category']} spending increased by {r['pct_change']:.0f}% "
                        f"compared with the previous month."
            })

    # --- weekend vs weekday -------------------------------------------------
    wk = ea.weekday_vs_weekend(df)
    if wk["weekday_avg"] > 0 or wk["weekend_avg"] > 0:
        if wk["weekend_avg"] > wk["weekday_avg"]:
            insights.append({
                "type": "insight",
                "text": "Your average weekend spending is higher than your weekday spending."
            })
        elif wk["weekday_avg"] > wk["weekend_avg"]:
            insights.append({
                "type": "insight",
                "text": "Your average weekday spending is higher than your weekend spending."
            })

    # --- frequent small transactions -------------------------------------------------
    small = ea.frequent_small_transactions(df)
    if not small.empty:
        total_small = int(small["count_below_threshold"].sum())
        insights.append({
            "type": "insight",
            "text": f"You made {total_small} small transactions below \u20b9200 this month."
        })

    # --- anomalies -------------------------------------------------
    flagged, msg = anomaly_detector.get_flagged_transactions(df, top_n=3)
    if msg is None and not flagged.empty:
        top_anomaly = flagged.iloc[0]
        insights.append({
            "type": "alert",
            "text": f"A transaction of \u20b9{top_anomaly['amount']:,.0f} in {top_anomaly['category']} "
                    f"on {pd.to_datetime(top_anomaly['date']).strftime('%d-%m-%Y')} is unusually high "
                    f"compared with your historical spending pattern."
        })

    # --- prediction -------------------------------------------------
    pred = predictor.predict_month_end_spending(df)
    if pred["status"] == "ok":
        insights.append({
            "type": "prediction",
            "text": f"Based on your historical data, estimated month-end expenses are "
                    f"\u20b9{pred['estimated_month_end']:,.0f}."
        })

    # --- budget alerts -------------------------------------------------
    if budgets_df is not None and not budgets_df.empty:
        current_month = pd.Timestamp.now().strftime("%Y-%m")
        progress = ea.budget_progress(df, budgets_df, current_month)
        for _, row in progress.iterrows():
            if row.get("exceeded"):
                insights.append({
                    "type": "alert",
                    "text": f"{row['category']} budget exceeded: \u20b9{row['spent']:,.0f} spent "
                            f"against a \u20b9{row['budget_amount']:,.0f} budget."
                })

    return insights
