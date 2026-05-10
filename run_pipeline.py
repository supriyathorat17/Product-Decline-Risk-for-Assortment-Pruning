"""
End-to-end pipeline runner for the Product Vitality Scoring System.
Runs Steps 1–18 from Capstone_MIS_790.ipynb as a single script.

Usage:
    python run_pipeline.py                    # full pipeline, saves to ./data/
    python run_pipeline.py --out /path/to/dir # save outputs elsewhere
    python run_pipeline.py --skip-download    # use cached HuggingFace files

Outputs written to ./data/:
    cleaned_electronics_reviews.csv
    product_month_aggregated.csv
    product_month_densified.csv
    product_vitality_scores.csv
    product_summary.csv
    pruning_candidates_vitality_top50.csv
"""

import argparse
import os
import time

import pandas as pd

from pipeline.ingest    import download_data, load_reviews, load_metadata, merge_data
from pipeline.clean     import clean_data, extract_nlp_features
from pipeline.model     import run_lda
from pipeline.aggregate import aggregate_by_month, filter_quality, densify, fill_gaps, compute_trends
from pipeline.score     import compute_health, compute_popularity, compute_vitality, flag_at_risk


def banner(step, title):
    print(f"\n{'='*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'='*60}")


def save(df, path, label):
    df.to_csv(path, index=False)
    print(f"Saved {label}: {path}  ({len(df):,} rows)")


def run(out_dir, skip_download=False):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── Steps 1–2: Download ───────────────────────────────────────────────────
    banner("1–2", "Environment Setup & Data Download")
    reviews_path, meta_path = download_data()

    # ── Step 3: Load reviews ──────────────────────────────────────────────────
    banner(3, "Load Customer Reviews (300,000 sample)")
    reviews_df = load_reviews(reviews_path)

    # ── Step 4: Load metadata ─────────────────────────────────────────────────
    banner(4, "Load Product Metadata")
    meta_df = load_metadata(meta_path, reviews_df["parent_asin"].unique())

    # ── Step 5: Merge ─────────────────────────────────────────────────────────
    banner(5, "Merge Reviews with Metadata")
    merged = merge_data(reviews_df, meta_df)

    # ── Steps 6–7: Clean + NLP ────────────────────────────────────────────────
    banner("6–7", "Data Cleaning & NLP Feature Extraction")
    df = clean_data(merged)
    df = extract_nlp_features(df)
    save(df, os.path.join(out_dir, "cleaned_electronics_reviews.csv"), "cleaned reviews")

    # ── Step 9: LDA Topic Modeling ────────────────────────────────────────────
    banner(9, "LDA Topic Modeling (5 topics)")
    df, lda_model, dictionary = run_lda(df)

    # ── Step 10: Product-Month Aggregation ────────────────────────────────────
    banner(10, "Product × Month Aggregation")
    sku_month = aggregate_by_month(df)
    save(sku_month, os.path.join(out_dir, "product_month_aggregated.csv"), "product-month table")

    # ── Steps 11–12: Quality Filtering ───────────────────────────────────────
    banner("11–12", "Quality Filtering (min 20 reviews, 6 months)")
    sku_month_f = filter_quality(sku_month, df)

    # ── Step 13: Densification + Trend Features ───────────────────────────────
    banner(13, "Timeline Densification & Trend Features")
    sku_month_dense = densify(sku_month_f)
    sku_month_dense = fill_gaps(sku_month_dense)
    sku_month_dense = compute_trends(sku_month_dense)
    save(sku_month_dense, os.path.join(out_dir, "product_month_densified.csv"), "dense product-month")

    # ── Step 14: Health Score ─────────────────────────────────────────────────
    banner(14, "Product Health Score")
    scored = compute_health(sku_month_dense)

    # ── Step 15: Popularity Score ─────────────────────────────────────────────
    banner(15, "Product Popularity Score")
    scored = compute_popularity(scored)

    # ── Step 16: Vitality Score ───────────────────────────────────────────────
    banner(16, "Product Vitality Score & 6-Month Trend")
    scored = compute_vitality(scored)
    save(scored, os.path.join(out_dir, "product_vitality_scores.csv"), "vitality scores (time-series)")

    # ── Step 17: At-Risk Flagging ─────────────────────────────────────────────
    banner(17, "At-Risk Flagging")
    product_summary = flag_at_risk(scored, df)
    save(product_summary, os.path.join(out_dir, "product_summary.csv"), "product summary (all)")

    # ── Step 18: Top-50 Pruning Candidates ───────────────────────────────────
    banner(18, "Export Top-50 Pruning Candidates")
    pruning = (
        product_summary[product_summary["at_risk"] == 1]
        .sort_values("avg_vitality_score")
        .head(50)
        .copy()
    )
    pruning = pruning[pruning["product_title"] != "Unknown"]
    pruning.insert(0, "rank", range(1, len(pruning) + 1))
    save(pruning, os.path.join(out_dir, "pruning_candidates_vitality_top50.csv"), "pruning candidates")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Outputs: {out_dir}")
    print(f"  Launch dashboard: streamlit run app.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Product Vitality Scoring Pipeline")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Output directory for CSV files")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HuggingFace download (use cached files)")
    args = parser.parse_args()
    run(args.out)
