# Predicting Product Decline Risk for Assortment Pruning Using Amazon Customer Reviews

**MIS 790 Capstone Project — Supriya Thorat**

---

## What This Project Does

Retailers manage thousands of products but often have no systematic way to identify which ones are declining — until it is too late and inventory has already piled up. This project builds a **Product Vitality Scoring System** that predicts which electronics products are at risk of discontinuation using only Amazon customer reviews. No sales data, no inventory data, no internal retailer systems required.

The system analyzes review patterns over time — things like sentiment, review volume trends, reliability complaints, and helpfulness signals — to compute a composite score for each product and automatically flag the ones that are declining.

**Key result:** 160 out of 586 recently active products flagged as at-risk (27.3%), with 80% spot-check accuracy compared to ~20% for a simple rating-only baseline.

---

## Why This Matters

Most retailers rely on star ratings to make assortment decisions. But a product with a 4.2-star average can still be dying — if review volume has collapsed, reliability complaints are rising, and the trend is negative. A rating-only approach misses all of that.

This system catches what ratings miss:
- Products with high ratings but collapsing review volume
- Products where reliability complaints (Topic 3 from LDA) are increasing
- Products with a sustained downward trend over the last 6 months

The 27.3% flagging rate aligns directly with standard retail audit cycles (20–30%), making the output immediately actionable for buying teams.

---

## Dataset

**Source:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) on HuggingFace

| Stat | Value |
|---|---|
| Reviews sampled | 300,000 |
| Products identified | 12,847 |
| Products after quality filter | 1,289 |
| Recently active products scored | 586 |
| Flagged as at-risk | 160 (27.3%) |

Quality filter: minimum 20 total reviews and at least 6 months of review history per product.

---

## Pipeline Overview (Steps 1–18)

The full pipeline is broken into five modules:

| Module | Steps | What It Does |
|---|---|---|
| `pipeline/ingest.py` | 1–5 | Downloads data from HuggingFace, loads reviews in chunks, loads product metadata, merges both |
| `pipeline/clean.py` | 6–8 | Cleans timestamps and ratings, removes duplicates, extracts 5 NLP features per review |
| `pipeline/model.py` | 9 | Trains LDA topic model (5 topics, coherence score 0.52) to discover hidden themes in reviews |
| `pipeline/aggregate.py` | 10–13 | Aggregates to product × month level, applies quality filters, fills timeline gaps, computes trend slopes |
| `pipeline/score.py` | 14–18 | Computes Health, Popularity, and Vitality scores; flags at-risk products; exports pruning candidates |

### Step 7 — NLP Features Extracted Per Review

| Feature | Method | What It Captures |
|---|---|---|
| Sentiment | TextBlob polarity (−1 to +1) | Positive vs negative tone |
| Subjectivity | TextBlob (0 to 1) | Opinion vs factual writing |
| Readability | Flesch-Kincaid grade level | Writing complexity |
| Review length | Log(word count) | Depth of feedback |
| Vocabulary diversity | Unique words / total words | Richness of language |

### Step 9 — LDA Topics Discovered

LDA was run with no manual keywords — topics emerged entirely from the data.

| Topic | Label | Top Keywords |
|---|---|---|
| Topic 0 | Camera & Display | camera, light, screen, lens, photo |
| Topic 1 | Device Accessories | case, laptop, ipad, kindle, cover |
| Topic 2 | Cables & Charging | cable, usb, power, cord, charger |
| **Topic 3** | **Product Reliability** ⭐ | work, worked, didn't, working, stopped, defective |
| Topic 4 | General Satisfaction | great, good, works, easy, quality |

**Topic 3 is the key negative signal.** Products with high Topic 3 loadings have significantly more reliability complaints and are more likely to be at-risk.

---

## Scoring Design

### Health Score (0–1)
Measures customer satisfaction and product quality.

- **Positive signals (70% weight):** average rating, helpfulness-weighted rating, verified purchase rate, sentiment, vocabulary diversity, review length, readability
- **Negative signals (30% weight):** subjectivity (biased reviews), Topic 3 loading (reliability complaints)
- Formula: `Health = (avg_positive × 0.70) − (avg_negative × 0.30)` → MinMax normalized to 0–1
- Result: Mean = 0.577 | Range = 0.08–0.97

### Popularity Score (0–1)
Measures market attention and demand from review volume.

- **Signals:** log(total reviews) × 40% + log(avg monthly reviews) × 30% + active months × 20% + verified rate × 10%
- Log transform applied because Amazon reviews follow a power-law distribution (top product: 4,924 reviews vs median: 60)
- Result: Mean = 0.130 | Range = 0.01–0.89

### Vitality Score (0–1)
```
Vitality = 0.5 × Health Score + 0.5 × Popularity Score
```
A 6-month linear regression slope is computed on top of this to track direction of change.

### At-Risk Flagging
A product is flagged **at-risk** if BOTH conditions are met:
- Vitality Score **< 0.45** (below the healthy-product mean)
- 6-month trend slope **< −0.001** (sustained decline, not a one-month dip)

Requiring both conditions eliminates false positives from stable low-scoring niche products.

---

## Benchmark: Vitality System vs. Rating-Only Baseline

| Metric | Vitality System | Rating-Only Baseline |
|---|---|---|
| Products flagged | 160 (27.3%) | Lower |
| Catches volume collapse | Yes | No |
| Catches trend decline | Yes | No |
| Catches reliability signal | Yes (LDA Topic 3) | No |
| Spot-check accuracy | 80% | ~20% |

The rating-only baseline flags any product with average rating < 3.5. It misses high-rated products that are declining in volume and engagement — a common and costly blind spot in retail assortment management.

---

## Project Structure

```
├── app.py                      # Streamlit dashboard (5 tabs)
├── run_pipeline.py             # Runs the full pipeline end-to-end
├── generate_sample_data.py     # Generates synthetic data for testing the dashboard
├── requirements.txt            # All dependencies
└── pipeline/
    ├── ingest.py               # Steps 1–5:  data download, load, merge
    ├── clean.py                # Steps 6–8:  cleaning + NLP feature extraction
    ├── model.py                # Step  9:   LDA topic modeling
    ├── aggregate.py            # Steps 10–13: aggregation, densification, trends
    └── score.py                # Steps 14–18: scoring, flagging, export
```

---

## Dashboard

The interactive Streamlit dashboard provides a complete view of the scoring system without needing to read any code.

> 📸 *Dashboard screenshots coming soon*

### Tab 1 — Pipeline
- Full system architecture diagram
- Scoring formulas for Health, Popularity, and Vitality
- LDA topic interpretations
- Status of all data output files (ready or missing)
- Instructions to run the pipeline

### Tab 2 — Overview
- KPI cards: total products, at-risk count, mean scores
- Score distribution histograms for Health, Popularity, and Vitality
- Vitality vs. trend scatter plot with at-risk threshold lines highlighted
- At-risk percentage breakdown by product sub-category

### Tab 3 — Product Table
- Full sortable table of all products with color-coded scores (red → yellow → green)
- Sidebar filters: sub-category, brand/store, at-risk toggle, vitality score range
- One-click CSV export of filtered results

### Tab 4 — Product Drilldown
- Select any product to see its individual score breakdown as a bar chart
- Time-series line chart showing Health, Popularity, and Vitality scores month by month
- Monthly review volume bar chart
- Key metrics: average rating, sentiment, reviews per month, 6-month trend slope

### Tab 5 — At-Risk & Export
- Full ranked list of at-risk products sorted by Vitality Score (lowest = highest priority)
- Adjustable top-N slider
- Download button for pruning candidates as CSV

---

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Option 1 — Quick start with sample data (no download needed)
```bash
python generate_sample_data.py
streamlit run app.py
```
Generates 200 synthetic products with realistic score distributions and opens the dashboard at `http://localhost:8501`.

### Option 2 — Run the full pipeline with real data
```bash
python run_pipeline.py
streamlit run app.py
```
Downloads the Amazon Electronics dataset from HuggingFace (~2 GB), runs all 18 steps, saves CSVs to `data/`, then launches the dashboard. Estimated runtime: 30–60 minutes depending on hardware.

---

## Known Limitations

| Issue | Risk | Planned Fix |
|---|---|---|
| Seasonal products (e.g. phone cases) may be falsely flagged in off-season | Medium | Seasonal decomposition (STL) on trend calculation |
| Fixed global thresholds may not suit all sub-categories | Medium | Category-specific threshold tuning |
| Young products (6–7 months history) may have unstable slopes | Low | Already mitigated by quality filter |
| External disruptions (supply chain, recalls) may cause false flags | Medium | Cross-check against external signals |

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Gensim · TextBlob · textstat · Streamlit · Matplotlib · HuggingFace Hub
