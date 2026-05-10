"""
Steps 10–13: Product-by-month aggregation, quality filtering, trend features, densification.
Extracted from Capstone_MIS_790.ipynb — Steps 10–13.
"""

import numpy as np
import pandas as pd

NUM_TOPICS = 5


def _wmean(values, weights):
    values  = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        return np.nan
    return float(np.sum(values * weights) / np.sum(weights))


def _slope(arr):
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.asarray(arr, dtype=float)
    if np.all(np.isnan(y)):
        return 0.0
    if np.isnan(y).any():
        y = np.where(np.isnan(y), np.nanmean(y), y)
    return float(np.polyfit(x, y, 1)[0])


def aggregate_by_month(df):
    """Step 10: Aggregate reviews to product × month (SKU-month table)."""
    print("Aggregating reviews by product × month...")
    lda_cols = [f"lda_topic_{i}" for i in range(NUM_TOPICS)]

    sku_month = (
        df
        .groupby(["parent_asin", "period"])
        .apply(lambda g: pd.Series({
            "review_count":               len(g),
            "avg_rating":                 g["rating"].mean(),
            "rating_std":                 g["rating"].std(ddof=0),
            "pct_1_star":                 (g["rating"] == 1).mean(),
            "pct_5_star":                 (g["rating"] == 5).mean(),
            "polarization":               (g["rating"] == 1).mean() + (g["rating"] == 5).mean(),
            "helpful_votes_sum":          g["helpful_vote"].sum(),
            "helpful_votes_avg":          g["helpful_vote"].mean(),
            "helpfulness_weighted_rating":_wmean(g["rating"], g["help_w"]),
            "verified_rate":              g["verified_purchase"].mean(),
            "sentiment_mean":             g["sentiment"].mean(),
            "subjectivity_mean":          g["subjectivity"].mean(),
            "readability_mean":           g["readability"].mean(),
            "review_length_mean":         g["review_length"].mean(),
            "diversity_mean":             g["diversity"].mean(),
            **{f"{c}_mean": g[c].mean() for c in lda_cols if c in g.columns},
        }))
        .reset_index()
    )

    sku_month["rating_std"] = sku_month["rating_std"].fillna(0.0)
    print(f"Product-month table: {sku_month.shape}")
    return sku_month


def filter_quality(sku_month, df, min_reviews=20, min_months=6):
    """Steps 11–12: Remove products with too few reviews or too little history."""
    print(f"Filtering: min_reviews={min_reviews}, min_months={min_months}")

    valid_skus = df.groupby("parent_asin").size()
    valid_skus = valid_skus[valid_skus >= min_reviews].index
    sku_month_f = sku_month[sku_month["parent_asin"].isin(valid_skus)].copy()
    print(f"  After review filter: {sku_month_f['parent_asin'].nunique()} products")

    months_per = sku_month_f.groupby("parent_asin")["period"].nunique()
    valid2      = months_per[months_per >= min_months].index
    sku_month_f = sku_month_f[sku_month_f["parent_asin"].isin(valid2)].copy()
    print(f"  After month filter:  {sku_month_f['parent_asin'].nunique()} products")

    return sku_month_f


def densify(sku_month_f):
    """Step 13: Insert zero-review rows so each product has a continuous monthly series."""
    print("Densifying timeline (filling month gaps)...")
    sku_month_f = sku_month_f.sort_values(["parent_asin", "period"]).reset_index(drop=True)
    all_products = []

    for pid, g in sku_month_f.groupby("parent_asin"):
        g        = g.sort_values("period").set_index("period")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="MS")
        g        = g.reindex(full_idx)
        g["parent_asin"]   = pid
        g.index.name       = "period"
        g["review_count"]  = g["review_count"].fillna(0)

        fill_zero = (
            [c for c in g.columns if c.endswith("_rate") or c.endswith("_w_rate")]
            + ["helpful_votes_sum", "helpful_votes_avg", "verified_rate",
               "polarization", "rating_std"]
        )
        for c in fill_zero:
            if c in g.columns:
                g[c] = g[c].fillna(0)

        all_products.append(g.reset_index())

    sku_month_dense = pd.concat(all_products, ignore_index=True)
    print(f"Dense rows: {len(sku_month_dense):,}")
    return sku_month_dense


def fill_gaps(sku_month_dense):
    """Step 13 continued: Forward-fill ratings and zero-fill linguistic/LDA columns."""
    sku_month_dense = sku_month_dense.sort_values(
        ["parent_asin", "period"]
    ).reset_index(drop=True)
    g = sku_month_dense.groupby("parent_asin", group_keys=False)

    # Rating: forward-fill (carry last known rating forward into silent months)
    sku_month_dense["avg_rating_ffill"] = g["avg_rating"].ffill()
    sku_month_dense["avg_rating_ffill"] = g["avg_rating_ffill"].bfill()

    sku_month_dense["helpfulness_weighted_rating_ffill"] = g["helpfulness_weighted_rating"].ffill()
    sku_month_dense["helpfulness_weighted_rating_ffill"] = (
        g["helpfulness_weighted_rating_ffill"].bfill()
    )

    # Linguistic features and LDA topics: fill 0 for inserted months
    for col in ["sentiment_mean", "subjectivity_mean", "readability_mean",
                "review_length_mean", "diversity_mean"]:
        if col in sku_month_dense.columns:
            sku_month_dense[col] = sku_month_dense[col].fillna(0)

    lda_cols = [c for c in sku_month_dense.columns if c.startswith("lda_topic")]
    for c in lda_cols:
        sku_month_dense[c] = sku_month_dense[c].fillna(0)

    return sku_month_dense


def compute_trends(sku_month_dense):
    """Step 13 continued: Compute velocity, EWMA, and 6-month slope on dense timeline."""
    print("Computing trend features on dense timeline...")
    sku_month_dense = sku_month_dense.sort_values(
        ["parent_asin", "period"]
    ).reset_index(drop=True)
    g = sku_month_dense.groupby("parent_asin", group_keys=False)

    sku_month_dense["review_velocity"]   = g["review_count"].diff().fillna(0)
    prev = g["review_count"].shift(1)
    sku_month_dense["review_growth_rate"] = (
        (sku_month_dense["review_count"] - prev) / prev.replace(0, np.nan)
    ).fillna(0)
    sku_month_dense["rating_change"]      = g["avg_rating"].diff().fillna(0)
    sku_month_dense["avg_rating_ffill"]   = g["avg_rating"].ffill()
    sku_month_dense["avg_rating_ewm3"]    = g["avg_rating_ffill"].apply(
        lambda s: s.ewm(span=3, adjust=False).mean()
    )
    sku_month_dense["rating_slope_6m"] = (
        g["avg_rating_ffill"]
        .rolling(6, min_periods=3)
        .apply(_slope, raw=True)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    return sku_month_dense
