# Predicting Product Decline Risk for Assortment Pruning Using Amazon Customer Reviews

**MIS 790 Capstone Project — Supriya Thorat**

An end-to-end **Product Vitality Scoring System** that identifies electronics products at risk of discontinuation using only Amazon customer reviews -  no sales or inventory data required.

---

## Overview

Retailers carry thousands of SKUs but lack systematic tools to identify which products are declining before it is too late. This system builds a composite **Vitality Score** for each product by combining customer review signals - sentiment, volume trends, reliability complaints, and helpfulness into an actionable at-risk flag that aligns with standard retail audit cycles (20–30% flagging rate).

**Key result:** 160 of 586 recently active products flagged as at-risk (27.3%), with an 80% spot-check accuracy vs. ~20% for a rating-only baseline.

---

## Pipeline (Steps 1–18)

| Module | Steps | Description |
|---|---|---|
| `pipeline/ingest.py` | 1–5 | Download from HuggingFace → load reviews → load metadata → merge |
| `pipeline/clean.py` | 6–8 | Clean timestamps/ratings → NLP features (TextBlob, textstat) |
| `pipeline/model.py` | 9 | LDA topic modeling — 5 topics, coherence score 0.52 |
| `pipeline/aggregate.py` | 10–13 | Product × month aggregation → quality filter → densify → trend features |
| `pipeline/score.py` | 14–18 | Health score → Popularity score → Vitality score → at-risk flagging |

### LDA Topics Discovered (Electronics Dataset)

| Topic | Label | Key Words |
|---|---|---|
| Topic 0 | Camera & Display | camera, light, screen, lens, photo |
| Topic 1 | Device Accessories | case, laptop, ipad, kindle, cover |
| Topic 2 | Cables & Charging | cable, usb, power, cord, charger |
| **Topic 3** | **Product Reliability** ⭐ | work, worked, didn't, working, stopped, defective |
| Topic 4 | General Satisfaction | great, good, works, easy, quality |

Topic 3 is the key negative signal weighted in the Health Score.

### Scoring Design

**Health Score (0–1)**
- +70%: avg rating, helpfulness-weighted rating, verified purchase rate, sentiment, vocabulary diversity, review length, readability
- −30%: subjectivity, Topic 3 (reliability complaints)

**Popularity Score (0–1)**
- 40% log(total reviews) + 30% log(avg monthly reviews) + 20% active months + 10% verified rate
- Log transform handles the power-law distribution of review counts

**Vitality Score = 0.5 × Health + 0.5 × Popularity**

**At-Risk flag:** Vitality < 0.45 AND 6-month trend slope < −0.001

---

## Project Structure
