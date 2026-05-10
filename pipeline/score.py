"""
Steps 14–18: Health, Popularity, Vitality scoring and at-risk flagging.
Extracted from Capstone_MIS_790.ipynb — Steps 14–18.

Scoring design:
  Health    = 0.70 × (avg_rating, helpfulness_rating, verified_rate, sentiment,
                       diversity, review_length, readability)
            − 0.30 × (subjectivity, lda_topic_3)   ← Topic 3 = reliability complaints
  Popularity = 0.40×log(total_reviews) + 0.30×log(avg_monthly) + 0.20×active_months + 0.10×verified_rate
  Vitality   = 0.50 × Health + 0.50 × Popularity
  At-Risk    = Vitality < 0.45  AND  6m trend < −0.001
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

VITALITY_THRESHOLD = 0.45
TREND_THRESHOLD    = -0.001
LAST_K_MONTHS      = 6

HEALTH_POS = [
    "avg_rating_ffill",
    "helpfulness_weighted_rating_ffill",
    "verified_rate",
    "sentiment_mean",
    "diversity_mean",
    "review_length_mean",
    "readability_mean",
]
HEALTH_NEG = [
    "subjectivity_mean",
    "lda_topic_3_mean",
]


def _slope_series(arr):
    arr = arr.dropna()
    if len(arr) < 3:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr.values, 1)[0])


def compute_health(sku_month_dense):
    """Step 14: Composite Health Score (0–1) from positive/negative signals."""
    print("Computing Health Score...")
    df     = sku_month_dense.copy()
    scaler = MinMaxScaler()

    for col in HEALTH_POS + HEALTH_NEG:
        if col in df.columns:
            vals        = df[col].fillna(0).values.reshape(-1, 1)
            df[f"{col}_norm"] = scaler.fit_transform(vals)

    pos_cols = [f"{c}_norm" for c in HEALTH_POS if f"{c}_norm" in df.columns]
    neg_cols = [f"{c}_norm" for c in HEALTH_NEG if f"{c}_norm" in df.columns]

    df["health_score"] = df[pos_cols].mean(axis=1) * 0.7 - df[neg_cols].mean(axis=1) * 0.3
    hs                  = df["health_score"].values.reshape(-1, 1)
    df["health_score"]  = scaler.fit_transform(hs)

    print(df["health_score"].describe())
    return df


def compute_popularity(sku_month_dense_scored):
    """Step 15: Popularity Score (0–1) from log-transformed review volume signals."""
    print("Computing Popularity Score...")
    df      = sku_month_dense_scored.copy()
    scaler  = MinMaxScaler()

    product_pop = (
        df.groupby("parent_asin")
        .agg(
            total_reviews       = ("review_count", "sum"),
            avg_monthly_reviews = ("review_count", "mean"),
            avg_verified_rate   = ("verified_rate", "mean"),
            active_months       = ("review_count", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    product_pop["total_reviews_raw"]  = product_pop["total_reviews"].copy()
    product_pop["total_reviews"]      = np.log1p(product_pop["total_reviews"])
    product_pop["avg_monthly_reviews"]= np.log1p(product_pop["avg_monthly_reviews"])

    for col in ["total_reviews", "avg_monthly_reviews", "avg_verified_rate", "active_months"]:
        product_pop[f"{col}_norm"] = scaler.fit_transform(product_pop[[col]])

    product_pop["popularity_score"] = (
        product_pop["total_reviews_norm"]        * 0.40
        + product_pop["avg_monthly_reviews_norm"] * 0.30
        + product_pop["active_months_norm"]        * 0.20
        + product_pop["avg_verified_rate_norm"]    * 0.10
    )
    product_pop["popularity_score"] = scaler.fit_transform(product_pop[["popularity_score"]])

    pop_map                        = product_pop.set_index("parent_asin")["popularity_score"].to_dict()
    df["popularity_score"]         = df["parent_asin"].map(pop_map)

    print(df["popularity_score"].describe())
    return df


def compute_vitality(sku_month_dense_scored):
    """Step 16: Vitality Score = 0.5 × Health + 0.5 × Popularity, plus 6m trend slope."""
    print("Computing Vitality Score and 6-month trend...")
    df     = sku_month_dense_scored.copy()
    scaler = MinMaxScaler()

    df["vitality_score"] = 0.5 * df["health_score"] + 0.5 * df["popularity_score"]
    df["vitality_score"] = scaler.fit_transform(df[["vitality_score"]])

    df = df.sort_values(["parent_asin", "period"]).reset_index(drop=True)
    g  = df.groupby("parent_asin", group_keys=False)

    df["vitality_trend_6m"] = (
        g["vitality_score"]
        .rolling(6, min_periods=3)
        .apply(_slope_series, raw=False)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    print(df[["health_score", "popularity_score", "vitality_score", "vitality_trend_6m"]].describe())
    return df


def flag_at_risk(sku_month_dense_scored, df_reviews,
                 last_k=LAST_K_MONTHS,
                 vitality_thresh=VITALITY_THRESHOLD,
                 trend_thresh=TREND_THRESHOLD):
    """Step 17: Flag products where Vitality < threshold AND trend < threshold."""
    print(f"Flagging at-risk products (last {last_k} months)...")
    scored     = sku_month_dense_scored.copy()
    max_period = scored["period"].max()
    recent_p   = pd.date_range(end=max_period, periods=last_k, freq="MS")
    recent     = scored[scored["period"].isin(recent_p)].copy()

    product_health = (
        recent.groupby("parent_asin")
        .agg(
            avg_health_score     = ("health_score",     "mean"),
            avg_popularity_score = ("popularity_score",  "mean"),
            avg_vitality_score   = ("vitality_score",   "mean"),
            avg_vitality_trend   = ("vitality_trend_6m","mean"),
            months_in_window     = ("vitality_score",    "count"),
            avg_rating           = ("avg_rating_ffill",  "mean"),
            avg_sentiment        = ("sentiment_mean",    "mean"),
            avg_review_count     = ("review_count",      "mean"),
        )
        .reset_index()
    )

    product_health["at_risk"] = (
        (product_health["avg_vitality_score"] < vitality_thresh) &
        (product_health["avg_vitality_trend"] < trend_thresh)
    ).astype(int)

    product_health = product_health.sort_values(
        ["avg_vitality_trend", "avg_vitality_score"], ascending=[True, True]
    )

    # Attach product metadata
    context        = df_reviews.groupby("parent_asin")[
        ["product_title", "main_category", "store", "price"]
    ].first().reset_index()
    product_health = product_health.merge(context, on="parent_asin", how="left")

    n_risk = product_health["at_risk"].sum()
    total  = len(product_health)
    print(f"Total products: {total}  |  At-risk: {n_risk} ({n_risk/total*100:.1f}%)")
    return product_health


def run_analysis(product_health):
    """Steps 19–22: Benchmark comparison, sensitivity analysis, business impact, error analysis."""

    # ── Step 19: Benchmark ───────────────────────────────────────────────────
    RATING_THRESHOLD = 3.5
    baseline         = product_health.copy()
    baseline["baseline_at_risk"] = (baseline["avg_rating"] < RATING_THRESHOLD).astype(int)
    total             = len(baseline)
    sys_flagged       = int(baseline["at_risk"].sum())
    base_flagged      = int(baseline["baseline_at_risk"].sum())
    both              = int((baseline["at_risk"] & baseline["baseline_at_risk"]).sum())
    sys_only          = int((baseline["at_risk"] & ~baseline["baseline_at_risk"]).sum())

    benchmark = {
        "total_products":         total,
        "vitality_flagged":       sys_flagged,
        "rating_only_flagged":    base_flagged,
        "both_agree":             both,
        "vitality_only":          sys_only,
        "high_rated_at_risk":     int(
            baseline[(baseline["at_risk"] == 1) & (baseline["avg_rating"] >= 4.0)].shape[0]
        ),
    }

    # ── Step 20: Sensitivity ─────────────────────────────────────────────────
    sensitivity = []
    for vt in [0.40, 0.45, 0.50]:
        for tt in [0.000, -0.001, -0.002]:
            count = int(
                ((product_health["avg_vitality_score"] < vt) &
                 (product_health["avg_vitality_trend"] < tt)).sum()
            )
            sensitivity.append({
                "vitality_threshold": vt,
                "trend_threshold":    tt,
                "flagged":            count,
                "pct":                round(count / total * 100, 1),
                "chosen":             (vt == 0.45 and tt == -0.001),
            })

    # ── Step 21: Business Impact ─────────────────────────────────────────────
    inv_per_sku   = 2500
    holding_rate  = 0.25
    overstock_rate= 0.15
    cycle_months  = 6

    def _impact(n):
        inv      = n * inv_per_sku
        holding  = inv * holding_rate * (cycle_months / 12)
        overstock= inv * overstock_rate
        return {"inventory": inv, "holding": holding, "overstock": overstock,
                "total": holding + overstock}

    business_impact = {
        "top_50":   _impact(50),
        "all_risk": _impact(sys_flagged),
    }

    # ── Step 22: Error Analysis ───────────────────────────────────────────────
    failure_modes = [
        {"risk": "Medium", "mode": "Seasonal products misread as declining",
         "why":  "Off-season volume drop → false negative trend.",
         "fix":  "Add seasonal decomposition in a future version."},
        {"risk": "Low",    "mode": "Young products at 6-7 month boundary",
         "why":  "Limited data → slope may overstate decline.",
         "fix":  "Quality filter removes most; monitor boundary separately."},
        {"risk": "Medium", "mode": "Temporary disruption (supply chain, recall)",
         "why":  "Short-term drop may trigger false flag.",
         "fix":  "Cross-check flagged products against news or seller notes."},
        {"risk": "Low",    "mode": "Category-specific review velocity norms",
         "why":  "Gaming vs cables have different natural review rates.",
         "fix":  "Category-specific thresholds in a future version."},
    ]

    return {
        "benchmark":       benchmark,
        "sensitivity":     sensitivity,
        "business_impact": business_impact,
        "failure_modes":   failure_modes,
    }
