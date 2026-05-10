"""
Steps 6–8: Data cleaning and NLP feature extraction.
Extracted from Capstone_MIS_790.ipynb — Steps 6–8.
"""

import re
import numpy as np
import pandas as pd
from textblob import TextBlob
import textstat


def clean_data(df):
    """Steps 6–7: Standardize timestamps, enforce rating range, remove duplicates."""
    print("Cleaning data...")
    df = df.copy()

    # Standardize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["parent_asin", "asin", "user_id", "timestamp", "rating"])

    # Remove duplicate review events
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "asin", "timestamp"], keep="first")
    print(f"  Removed duplicates: {before - len(df)}")

    # Fill optional metadata
    df["product_title"] = df["product_title"].fillna("Unknown")
    df["description"]   = df["description"].fillna("")
    df["main_category"] = df["main_category"].fillna("Unknown")
    df["categories"]    = df["categories"].fillna("Unknown")
    df["store"]         = df["store"].fillna("Unknown")
    df["price"]         = pd.to_numeric(
        df["price"].astype(str).str.replace("—", "").str.strip(), errors="coerce"
    )

    # Enforce rating range 1–5
    before = len(df)
    df = df[df["rating"].between(1, 5)]
    print(f"  Removed invalid ratings: {before - len(df)}")

    # Helpful vote cleanup
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df = df[df["helpful_vote"] >= 0]

    # Verified purchase as boolean
    if df["verified_purchase"].dtype != bool:
        df["verified_purchase"] = (
            df["verified_purchase"].astype(str).str.lower().isin(["true", "1", "yes"])
        )

    # Trim string columns
    for col in ["parent_asin", "asin", "user_id"]:
        df[col] = df[col].astype(str).str.strip()
    df["review_title"] = df["review_title"].fillna("").astype(str).str.strip()
    df["review_text"]  = df["review_text"].fillna("").astype(str).str.strip()

    # Drop empty reviews
    before = len(df)
    df = df[df["review_text"].str.len() > 0]
    print(f"  Removed empty review_text: {before - len(df)}")

    # Re-check duplicates after normalization
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "asin", "timestamp"], keep="first")
    print(f"  Removed duplicates after normalization: {before - len(df)}")

    # Month period and helpfulness weight
    df["period"]    = df["timestamp"].dt.to_period("M").dt.start_time
    df["help_w"]    = df["helpful_vote"].astype(float) + 1.0
    df["review_text_l"] = df["review_text"].str.lower()

    print(f"Final cleaned shape: {df.shape}")
    return df


# ── NLP feature helpers ───────────────────────────────────────────────────────

def _sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

def _subjectivity(text):
    return TextBlob(str(text)).sentiment.subjectivity

def _readability(text):
    try:
        return textstat.flesch_reading_ease(str(text))
    except Exception:
        return 0.0

def _diversity(text):
    words = str(text).lower().split()
    return len(set(words)) / len(words) if words else 0.0


def extract_nlp_features(df):
    """Step 7: Compute sentiment, subjectivity, readability, length, diversity."""
    print("Computing NLP features (this takes several minutes on 300K rows)...")
    df = df.copy()
    texts = df["review_text"].fillna("")

    df["sentiment"]     = texts.apply(_sentiment)
    df["subjectivity"]  = texts.apply(_subjectivity)
    df["readability"]   = texts.apply(_readability)
    df["review_length"] = texts.str.len().apply(np.log1p)
    df["diversity"]     = texts.apply(_diversity)

    print("NLP features computed:")
    print(df[["sentiment", "subjectivity", "readability", "review_length", "diversity"]].describe())
    return df
