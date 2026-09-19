"""
models/categorizer.py
-----------------------
Automatic expense categorization using a simple, explainable NLP pipeline:

    TF-IDF Vectorizer  ->  Logistic Regression

Trained on a small seed dataset of merchant/description -> category
examples plus (when available) the user's own manually-categorized
transaction history, so it improves the more the user uses the app.

If the model's confidence for a prediction is below a threshold, the
transaction is labeled "Other / Needs Review" instead of a guess.
"""

import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_PATH = os.path.join(MODEL_DIR, "categorizer_model.joblib")

CONFIDENCE_THRESHOLD = 0.35
NEEDS_REVIEW_LABEL = "Other / Needs Review"

# Seed training examples: (merchant/description text, category)
SEED_TRAINING_DATA = [
    ("SWIGGY", "Food"), ("ZOMATO", "Food"), ("SWIGGY INSTAMART", "Food"),
    ("DOMINOS PIZZA", "Food"), ("MCDONALDS", "Food"), ("STARBUCKS", "Food"),
    ("CAFE COFFEE DAY", "Food"), ("RESTAURANT BILL", "Food"), ("PIZZA HUT", "Food"),
    ("KFC", "Food"), ("BIGBASKET GROCERY", "Food"), ("GROCERY STORE", "Food"),
    ("BLINKIT", "Food"), ("ZEPTO", "Food"),

    ("UBER", "Transport"), ("OLA CABS", "Transport"), ("RAPIDO", "Transport"),
    ("PETROL PUMP", "Transport"), ("FUEL STATION", "Transport"), ("INDIAN OIL", "Transport"),
    ("METRO CARD RECHARGE", "Transport"), ("IRCTC TRAIN TICKET", "Transport"),
    ("PARKING FEE", "Transport"), ("BUS TICKET", "Transport"), ("OLA AUTO", "Transport"),

    ("AMAZON", "Shopping"), ("FLIPKART", "Shopping"), ("MYNTRA", "Shopping"),
    ("AJIO", "Shopping"), ("SHOPPING MALL", "Shopping"), ("RELIANCE TRENDS", "Shopping"),
    ("DECATHLON", "Shopping"), ("NYKAA", "Shopping"), ("H AND M CLOTHING", "Shopping"),

    ("ELECTRICITY BILL", "Bills"), ("WATER BILL", "Bills"), ("MOBILE RECHARGE", "Bills"),
    ("BROADBAND BILL", "Bills"), ("GAS CYLINDER", "Bills"), ("WIFI BILL", "Bills"),
    ("DTH RECHARGE", "Bills"), ("RENT PAYMENT", "Bills"), ("MAINTENANCE FEE", "Bills"),

    ("NETFLIX", "Entertainment"), ("SPOTIFY", "Entertainment"), ("AMAZON PRIME", "Entertainment"),
    ("HOTSTAR", "Entertainment"), ("MOVIE TICKET", "Entertainment"), ("PVR CINEMAS", "Entertainment"),
    ("BOOKMYSHOW", "Entertainment"), ("GAMING SUBSCRIPTION", "Entertainment"),

    ("COLLEGE FEE", "Education"), ("SCHOOL FEE", "Education"), ("TUITION FEE", "Education"),
    ("UDEMY COURSE", "Education"), ("COURSERA SUBSCRIPTION", "Education"),
    ("BOOK STORE", "Education"), ("EXAM FEE", "Education"),

    ("PHARMACY", "Healthcare"), ("HOSPITAL BILL", "Healthcare"), ("DOCTOR CONSULTATION", "Healthcare"),
    ("MEDICAL STORE", "Healthcare"), ("APOLLO PHARMACY", "Healthcare"), ("HEALTH INSURANCE", "Healthcare"),
    ("DENTAL CLINIC", "Healthcare"), ("LAB TEST", "Healthcare"),

    ("SALARY CREDIT", "Salary"), ("MONTHLY SALARY", "Salary"), ("COMPANY PAYROLL", "Salary"),

    ("ATM WITHDRAWAL", "Other"), ("BANK CHARGES", "Other"), ("MISCELLANEOUS", "Other"),
    ("GIFT", "Other"), ("DONATION", "Other"),
]


def _clean_text(text: str) -> str:
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_training_frame(history_df: pd.DataFrame = None) -> pd.DataFrame:
    """Combine seed data with any manually-categorized user history."""
    seed = pd.DataFrame(SEED_TRAINING_DATA, columns=["text", "category"])

    if history_df is not None and not history_df.empty:
        hist = history_df.copy()
        hist["text"] = (hist.get("merchant", "").fillna("") + " " +
                         hist.get("description", "").fillna(""))
        hist = hist[hist["category"].notna() & (hist["category"] != "") &
                    (hist["category"] != NEEDS_REVIEW_LABEL)]
        hist = hist[hist["text"].str.strip() != ""]
        hist = hist[["text", "category"]]
        combined = pd.concat([seed, hist], ignore_index=True)
    else:
        combined = seed

    combined["text"] = combined["text"].apply(_clean_text)
    combined = combined[combined["text"] != ""]
    return combined


def train_model(history_df: pd.DataFrame = None) -> Pipeline:
    """Train (or retrain) the TF-IDF + Logistic Regression pipeline."""
    training = build_training_frame(history_df)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, C=10.0)),
    ])
    pipeline.fit(training["text"], training["category"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_or_train_model(history_df: pd.DataFrame = None, force_retrain: bool = False) -> Pipeline:
    """Load a cached model from disk, or train a new one if missing/forced."""
    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_model(history_df)


def predict_category(text: str, model: Pipeline = None, history_df: pd.DataFrame = None):
    """
    Predict a category for a single merchant/description string.

    Returns (category, confidence). If confidence is below the
    threshold, category is NEEDS_REVIEW_LABEL.
    """
    if model is None:
        model = load_or_train_model(history_df)

    cleaned = _clean_text(text)
    if not cleaned:
        return NEEDS_REVIEW_LABEL, 0.0

    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    confidence = float(proba[best_idx])
    category = classes[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return NEEDS_REVIEW_LABEL, confidence
    return category, confidence


def predict_categories_bulk(texts, model: Pipeline = None, history_df: pd.DataFrame = None):
    """Predict categories for a list/Series of merchant+description strings."""
    if model is None:
        model = load_or_train_model(history_df)
    return [predict_category(t, model=model) for t in texts]
