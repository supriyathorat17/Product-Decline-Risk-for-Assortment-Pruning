"""
Steps 1–5: Data download, loading, and merge.
Extracted from Capstone_MIS_790.ipynb — Steps 2–5.
"""

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "McAuley-Lab/Amazon-Reviews-2023"

REVIEW_KEEP_COLS = [
    "parent_asin", "asin", "user_id",
    "rating", "title", "text",
    "timestamp", "helpful_vote", "verified_purchase",
]

META_KEEP_COLS = [
    "parent_asin", "title", "description", "main_category",
    "categories", "average_rating", "rating_number", "store", "price",
]


def download_data():
    """Download Electronics reviews and metadata from HuggingFace (cached after first run)."""
    print("Downloading reviews from HuggingFace (cached after first download)...")
    reviews_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename="raw/review_categories/Electronics.jsonl",
    )
    print("Downloading product metadata...")
    meta_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename="raw/meta_categories/meta_Electronics.jsonl",
    )
    print(f"Reviews:  {reviews_path}")
    print(f"Metadata: {meta_path}")
    return reviews_path, meta_path


def load_reviews(reviews_path, target=300_000, chunk_size=100_000):
    """Step 3: Load up to `target` reviews from the JSONL file in chunks."""
    print(f"Loading up to {target:,} reviews in chunks of {chunk_size:,}...")
    chunks = []
    count  = 0
    for chunk in pd.read_json(reviews_path, lines=True, chunksize=chunk_size):
        cols  = [c for c in REVIEW_KEEP_COLS if c in chunk.columns]
        chunk = chunk[cols].dropna(
            subset=["parent_asin", "asin", "user_id", "timestamp", "rating"]
        )
        chunks.append(chunk)
        count += len(chunk)
        print(f"  Loaded: {count:,}")
        if count >= target:
            break

    reviews_df = pd.concat(chunks, ignore_index=True).head(target)
    reviews_df = reviews_df.rename(columns={"title": "review_title", "text": "review_text"})
    print(f"Reviews shape: {reviews_df.shape}")
    return reviews_df


def load_metadata(meta_path, product_ids, target_products=50_000):
    """Step 4: Load metadata for products in `product_ids`."""
    print("Loading product metadata...")
    product_ids = set(list(product_ids)[:target_products])
    hits        = []
    seen        = 0

    for chunk in pd.read_json(meta_path, lines=True, chunksize=50_000):
        seen += len(chunk)
        chunk = chunk[chunk["parent_asin"].isin(product_ids)]
        cols  = [c for c in META_KEEP_COLS if c in chunk.columns]
        chunk = chunk[cols]
        if len(chunk) > 0:
            hits.append(chunk)

        matched = sum(len(x) for x in hits)
        print(f"  Scanned: {seen:,}  Matched: {matched:,}")

        if hits:
            matched_asins = pd.concat(hits)["parent_asin"].nunique()
            if matched_asins >= int(0.9 * len(product_ids)):
                break

    meta_df = pd.concat(hits, ignore_index=True).drop_duplicates(subset=["parent_asin"])
    meta_df = meta_df.rename(columns={"title": "product_title"})
    print(f"Metadata shape: {meta_df.shape}")
    return meta_df


def merge_data(reviews_df, meta_df):
    """Step 5: Left-join reviews onto product metadata."""
    merged = reviews_df.merge(meta_df, on="parent_asin", how="left")
    print(f"Merged shape: {merged.shape}")
    print(f"Missing product_title (%): {merged['product_title'].isna().mean() * 100:.2f}")
    return merged
