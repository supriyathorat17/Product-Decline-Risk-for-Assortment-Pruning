"""
Generates synthetic sample data matching the real capstone CSV structure.
Run this to test the dashboard before downloading your actual Google Drive files.

    python generate_sample_data.py
"""

import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

# ── Config ────────────────────────────────────────────────────────────────────
N_PRODUCTS = 200
N_MONTHS   = 24   # 2 years of monthly history per product
OUT_DIR    = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = ["Cables & Adapters", "Cameras", "Laptops", "Phone Cases",
              "Headphones", "Chargers", "Tablets", "Smart Home", "Gaming"]
STORES     = ["Anker", "Belkin", "Samsung", "Apple", "Logitech",
              "Sony", "JBL", "Corsair", "TP-Link", "Unknown Brand"]

# ── Product-level metadata ─────────────────────────────────────────────────
asins  = [f"B{rng.integers(10**8, 10**9):09d}" for _ in range(N_PRODUCTS)]
titles = [
    f"{rng.choice(STORES)} {rng.choice(['USB-C Hub', 'Charging Cable', 'Wireless Headphones', 'Phone Case', 'Laptop Stand', 'Webcam', 'Smart Plug', 'HDMI Cable', 'Keyboard', 'Mouse'])} {rng.choice(['Pro', 'Plus', 'Lite', 'Gen 2', 'Max', ''])}"
    for _ in range(N_PRODUCTS)
]

products = pd.DataFrame({
    "parent_asin":  asins,
    "product_title": titles,
    "main_category": rng.choice(CATEGORIES, N_PRODUCTS),
    "store":         rng.choice(STORES,     N_PRODUCTS),
    "price":         rng.uniform(9.99, 199.99, N_PRODUCTS).round(2),
})

# ── Time-series vitality scores (product × month) ─────────────────────────
periods = pd.date_range("2022-01-01", periods=N_MONTHS, freq="MS")
rows = []

for i, asin in enumerate(asins):
    # Each product has a base health & popularity, with some drift
    base_health = rng.uniform(0.2, 0.9)
    base_pop    = rng.uniform(0.05, 0.7)
    drift       = rng.uniform(-0.015, 0.010)

    for j, period in enumerate(periods):
        noise_h = rng.normal(0, 0.03)
        noise_p = rng.normal(0, 0.02)

        health     = float(np.clip(base_health + drift * j + noise_h, 0.05, 0.99))
        popularity = float(np.clip(base_pop    + drift * j * 0.5 + noise_p, 0.01, 0.95))
        vitality   = float(np.clip(0.5 * health + 0.5 * popularity, 0.05, 0.99))

        rows.append({
            "parent_asin":      asin,
            "period":           period,
            "health_score":     round(health, 4),
            "popularity_score": round(popularity, 4),
            "vitality_score":   round(vitality, 4),
            "avg_rating":       round(np.clip(rng.normal(3.8 + health, 0.4), 1, 5), 2),
            "sentiment_mean":   round(np.clip(rng.normal(health - 0.1, 0.15), -1, 1), 4),
            "review_count":     max(0, int(rng.poisson(popularity * 30 + 5))),
        })

ts_df = pd.DataFrame(rows)

# Compute 6-month trend slope per product-month
def rolling_slope(series):
    out = [0.0] * len(series)
    for k in range(3, len(series)):
        window = series[max(0, k-6):k+1]
        if len(window) >= 3:
            x = np.arange(len(window), dtype=float)
            out[k] = float(np.polyfit(x, window, 1)[0])
    return out

ts_df = ts_df.sort_values(["parent_asin", "period"]).reset_index(drop=True)
slopes = []
for _, g in ts_df.groupby("parent_asin"):
    slopes.extend(rolling_slope(g["vitality_score"].tolist()))
ts_df["vitality_trend_6m"] = slopes

ts_path = os.path.join(OUT_DIR, "product_vitality_scores.csv")
ts_df.to_csv(ts_path, index=False)
print(f"Saved time-series:  {ts_path}  ({len(ts_df):,} rows)")

# ── Product-level summary (last 6 months) ─────────────────────────────────
last_6 = periods[-6:]
recent = ts_df[ts_df["period"].isin(last_6)]

summary = (
    recent.groupby("parent_asin")
    .agg(
        avg_health_score     = ("health_score",     "mean"),
        avg_popularity_score = ("popularity_score",  "mean"),
        avg_vitality_score   = ("vitality_score",   "mean"),
        avg_vitality_trend   = ("vitality_trend_6m","mean"),
        avg_rating           = ("avg_rating",        "mean"),
        avg_sentiment        = ("sentiment_mean",    "mean"),
        avg_review_count     = ("review_count",      "mean"),
        months_in_window     = ("vitality_score",    "count"),
    )
    .reset_index()
)

summary["at_risk"] = (
    (summary["avg_vitality_score"] < 0.45) &
    (summary["avg_vitality_trend"] < -0.001)
).astype(int)

summary = summary.merge(products, on="parent_asin", how="left")

# Round for readability
for col in ["avg_health_score","avg_popularity_score","avg_vitality_score",
            "avg_vitality_trend","avg_rating","avg_sentiment"]:
    summary[col] = summary[col].round(4)
summary["avg_review_count"] = summary["avg_review_count"].round(1)

all_path = os.path.join(OUT_DIR, "product_summary.csv")
summary.to_csv(all_path, index=False)
print(f"Saved product summary: {all_path}  ({len(summary):,} rows)")

# ── Top-50 pruning candidates ─────────────────────────────────────────────
at_risk = summary[summary["at_risk"] == 1].sort_values("avg_vitality_score")
pruning = at_risk.head(50).copy()
pruning.insert(0, "rank", range(1, len(pruning) + 1))

pr_path = os.path.join(OUT_DIR, "pruning_candidates_vitality_top50.csv")
pruning.to_csv(pr_path, index=False)
print(f"Saved pruning candidates: {pr_path}  ({len(pruning)} rows)")
print(f"\nAt-risk products: {summary['at_risk'].sum()} / {len(summary)}")
print("Done. Open the dashboard: streamlit run app.py")
