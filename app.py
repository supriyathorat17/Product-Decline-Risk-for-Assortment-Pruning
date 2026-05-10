"""
Product Vitality Scoring System — Streamlit Dashboard
MIS 790 Capstone — Supriya Thorat

The full pipeline lives in pipeline/:
  ingest.py     Steps 1–5   Data download, load, merge
  clean.py      Steps 6–8   Cleaning, NLP features (TextBlob, textstat)
  model.py      Step  9     LDA topic modeling (5 topics, coherence 0.52)
  aggregate.py  Steps 10–13 Product-month aggregation, densification, trends
  score.py      Steps 14–18 Health, Popularity, Vitality scores, at-risk flagging

Run pipeline first:  python run_pipeline.py
Then launch this UI: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(
    page_title="Product Vitality Scoring System",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading product summary...")
def load_summary():
    for fname in ["product_summary.csv", "pruning_candidates_vitality_top50.csv"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "at_risk" not in df.columns:
                df["at_risk"] = 0
            return df
    return None


@st.cache_data(show_spinner="Loading time-series...")
def load_timeseries():
    path = os.path.join(DATA_DIR, "product_vitality_scores.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["period"])


def data_file_status():
    files = {
        "cleaned_electronics_reviews.csv":    "Step 6–8  Cleaned reviews (NLP features)",
        "product_month_aggregated.csv":        "Step 10   Product × month aggregation",
        "product_month_densified.csv":         "Step 13   Dense timeline + trend features",
        "product_vitality_scores.csv":         "Step 16   Vitality scores (time-series)",
        "product_summary.csv":                 "Step 17   Product summary (all products)",
        "pruning_candidates_vitality_top50.csv":"Step 18  Top-50 pruning candidates",
    }
    rows = []
    for fname, desc in files.items():
        path   = os.path.join(DATA_DIR, fname)
        exists = os.path.exists(path)
        size   = f"{os.path.getsize(path)/1e6:.1f} MB" if exists else "–"
        rows.append({"File": fname, "Description": desc,
                     "Status": "✅ Ready" if exists else "❌ Missing", "Size": size})
    return pd.DataFrame(rows)


# ── Colour helpers ────────────────────────────────────────────────────────────
def score_color(val):
    rgba = plt.cm.RdYlGn(float(np.clip(val, 0, 1)))
    return mcolors.to_hex(rgba)

def color_score_col(series):
    return [f"background-color: {score_color(v)}; color: #111" for v in series]


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar_filters(df):
    st.sidebar.header("Filters")
    cats   = ["All"] + sorted(df["main_category"].dropna().unique().tolist())
    cat    = st.sidebar.selectbox("Category", cats)
    stores = ["All"] + sorted(df["store"].dropna().unique().tolist())
    store  = st.sidebar.selectbox("Store / Brand", stores)
    at_risk_only   = st.sidebar.checkbox("At-risk products only", False)
    vitality_range = st.sidebar.slider("Vitality Score range", 0.0, 1.0, (0.0, 1.0), 0.01)
    return cat, store, at_risk_only, vitality_range


def apply_filters(df, cat, store, at_risk_only, vitality_range):
    mask = pd.Series([True] * len(df), index=df.index)
    if cat   != "All": mask &= df["main_category"] == cat
    if store != "All": mask &= df["store"] == store
    if at_risk_only:   mask &= df["at_risk"] == 1
    mask &= (df["avg_vitality_score"] >= vitality_range[0]) & \
            (df["avg_vitality_score"] <= vitality_range[1])
    return df[mask]


# ── Tab: Pipeline ─────────────────────────────────────────────────────────────
def tab_pipeline():
    st.subheader("System Architecture")
    st.markdown("""
The pipeline is split into five modules in `pipeline/`. Run `python run_pipeline.py` once to
generate all data files, then the dashboard loads them automatically.

```
App/
├── run_pipeline.py          ← Run this to execute the full pipeline
├── app.py                   ← This dashboard
└── pipeline/
    ├── ingest.py            Steps 1–5   Download → load reviews → load metadata → merge
    ├── clean.py             Steps 6–8   Clean timestamps/ratings → NLP features (TextBlob, textstat)
    ├── model.py             Step  9     LDA topic modeling (5 topics, coherence 0.52)
    ├── aggregate.py         Steps 10–13 Product-month aggregation → quality filter → densify → trends
    └── score.py             Steps 14–18 Health → Popularity → Vitality → at-risk flagging
```
""")

    st.subheader("LDA Topics Discovered (Electronics Dataset)")
    topics = pd.DataFrame({
        "Topic": ["Topic 0", "Topic 1", "Topic 2", "Topic 3 ⭐", "Topic 4"],
        "Label": ["Camera & Display", "Device Accessories", "Cables & Charging",
                  "Product Reliability (KEY NEGATIVE SIGNAL)", "General Satisfaction"],
        "Top Keywords": [
            "camera, light, screen, lens, photo",
            "case, laptop, ipad, kindle, cover",
            "cable, usb, power, cord, charger",
            "work, worked, didn't, working, stopped, defective",
            "great, good, works, easy, quality",
        ],
        "Role in Scoring": ["Neutral", "Neutral", "Neutral",
                            "Negative signal (30% weight in Health Score)", "Neutral"],
    })
    st.dataframe(topics, use_container_width=True, hide_index=True)

    st.subheader("Scoring Formulas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Health Score (0–1)**")
        st.markdown("""
- +70%: avg rating, helpfulness-weighted rating, verified rate, sentiment, diversity, length, readability
- −30%: subjectivity, Topic 3 (reliability)
- MinMax normalized → 0–1
- Range: 0.08 – 0.97 | Mean: 0.577
""")
    with col2:
        st.markdown("**Popularity Score (0–1)**")
        st.markdown("""
- 40%: log(total reviews)
- 30%: log(avg monthly reviews)
- 20%: active months
- 10%: verified rate
- Log transform handles power-law distribution
- Range: 0.01 – 0.89 | Mean: 0.130
""")
    with col3:
        st.markdown("**Vitality Score + At-Risk**")
        st.markdown("""
- Vitality = 0.5 × Health + 0.5 × Popularity
- 6-month trend slope (linear regression)
- **At-Risk** if BOTH:
  - Vitality < **0.45**
  - 6m trend < **−0.001**
- Result: 160 of 586 products flagged (27.3%)
""")

    st.subheader("Output Files")
    st.dataframe(data_file_status(), use_container_width=True, hide_index=True)

    st.subheader("How to Run the Full Pipeline")
    st.code("""# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads ~2GB from HuggingFace, takes ~30–60 min)
python run_pipeline.py

# Or use your pre-computed Google Drive CSVs:
# Download from Drive → place in App/data/ → run dashboard
streamlit run app.py""", language="bash")


# ── Tab: Overview ─────────────────────────────────────────────────────────────
def tab_overview(df):
    total    = len(df)
    at_risk  = int(df["at_risk"].sum())
    mean_vit = df["avg_vitality_score"].mean()
    mean_h   = df["avg_health_score"].mean()
    mean_p   = df["avg_popularity_score"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Products",   f"{total:,}")
    c2.metric("At-Risk",          f"{at_risk}",
              delta=f"{at_risk/total*100:.1f}% of total", delta_color="inverse")
    c3.metric("Mean Vitality",    f"{mean_vit:.3f}")
    c4.metric("Mean Health",      f"{mean_h:.3f}")
    c5.metric("Mean Popularity",  f"{mean_p:.3f}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Score Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, col, color, label in [
            (axes[0], "avg_health_score",     "#2ecc71", "Health Score"),
            (axes[1], "avg_popularity_score", "#3498db", "Popularity Score"),
            (axes[2], "avg_vitality_score",   "#9b59b6", "Vitality Score"),
        ]:
            ax.hist(df[col].dropna(), bins=25, color=color, edgecolor="white", alpha=0.85)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Score (0–1)", fontsize=8)
            ax.set_ylabel("# Products", fontsize=8)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_r:
        st.subheader("Vitality Score vs. 6-Month Trend")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors  = ["#e74c3c" if r == 1 else "#3498db" for r in df["at_risk"]]
        ax.scatter(df["avg_vitality_trend"], df["avg_vitality_score"],
                   c=colors, alpha=0.55, s=18, linewidths=0)
        ax.axhline(0.45,   color="#e74c3c", linestyle="--", linewidth=0.9)
        ax.axvline(-0.001, color="#e67e22", linestyle="--", linewidth=0.9)
        ax.set_xlabel("6-Month Trend Slope", fontsize=9)
        ax.set_ylabel("Avg Vitality Score",  fontsize=9)
        red  = plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="#e74c3c",ms=7,label="At-Risk")
        blue = plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="#3498db",ms=7,label="Healthy")
        ax.legend(handles=[red, blue], fontsize=7)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    if "main_category" in df.columns:
        st.subheader("At-Risk by Category")
        cat_s = (
            df.groupby("main_category")
            .agg(total=("at_risk","count"), at_risk=("at_risk","sum"))
            .assign(pct=lambda x: (x["at_risk"]/x["total"]*100).round(1))
            .sort_values("pct", ascending=False).reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.barh(cat_s["main_category"], cat_s["pct"], color="#e74c3c", alpha=0.7)
        ax.set_xlabel("% At-Risk", fontsize=9); ax.tick_params(labelsize=8)
        for bar, val in zip(bars, cat_s["pct"]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ── Tab: Product Table ────────────────────────────────────────────────────────
def tab_product_table(df):
    st.subheader(f"Products ({len(df):,} shown)")
    display_cols = [c for c in [
        "parent_asin","product_title","main_category","store","price",
        "avg_vitality_score","avg_health_score","avg_popularity_score",
        "avg_vitality_trend","avg_rating","avg_sentiment","avg_review_count","at_risk",
    ] if c in df.columns]

    rename = {
        "parent_asin":"ASIN","product_title":"Product","main_category":"Category",
        "store":"Store","price":"Price ($)","avg_vitality_score":"Vitality",
        "avg_health_score":"Health","avg_popularity_score":"Popularity",
        "avg_vitality_trend":"Trend (6m)","avg_rating":"Avg Rating",
        "avg_sentiment":"Sentiment","avg_review_count":"Reviews/mo","at_risk":"At-Risk",
    }
    disp = df[display_cols].rename(columns=rename).reset_index(drop=True)

    sort_by = st.selectbox("Sort by", ["Vitality","Health","Popularity","Avg Rating","At-Risk"])
    disp    = disp.sort_values(sort_by, ascending=(sort_by == "Vitality"))

    score_cols = [c for c in ["Vitality","Health","Popularity"] if c in disp.columns]
    styled = (
        disp.style
        .apply(color_score_col, subset=score_cols)
        .format({"Price ($)":"${:.2f}","Vitality":"{:.3f}","Health":"{:.3f}",
                 "Popularity":"{:.3f}","Trend (6m)":"{:.5f}","Avg Rating":"{:.2f}",
                 "Sentiment":"{:.3f}","Reviews/mo":"{:.1f}"}, na_rep="–")
    )
    st.dataframe(styled, use_container_width=True, height=520)
    csv = df[display_cols].to_csv(index=False).encode()
    st.download_button("Download filtered results as CSV", csv,
                       "filtered_products.csv", "text/csv")


# ── Tab: Drilldown ────────────────────────────────────────────────────────────
def tab_drilldown(df, ts):
    st.subheader("Product Drilldown")
    choices    = df[["parent_asin","product_title"]].drop_duplicates()
    asin_label = {r["parent_asin"]: f"{r['product_title']}  ({r['parent_asin']})"
                  for _, r in choices.iterrows()}
    label_asin = {v: k for k, v in asin_label.items()}

    default = (df[df["at_risk"]==1]["parent_asin"].iloc[0]
               if df["at_risk"].sum() > 0 else df["parent_asin"].iloc[0])
    labels  = list(asin_label.values())
    sel     = st.selectbox("Select a product", labels,
                            index=labels.index(asin_label[default]))
    asin    = label_asin[sel]
    row     = df[df["parent_asin"] == asin].iloc[0]

    st.markdown(f"### {row.get('product_title', asin)}")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**ASIN:** `{asin}`")
    c2.markdown(f"**Category:** {row.get('main_category','–')}")
    c3.markdown(f"**Store:** {row.get('store','–')}  &nbsp; **Price:** ${row.get('price',0):.2f}")
    flag = ":red[AT-RISK]" if row.get("at_risk",0)==1 else ":green[HEALTHY]"
    st.markdown(f"**Status:** {flag}")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Score Breakdown**")
        scores = {"Health":row.get("avg_health_score",0),
                  "Popularity":row.get("avg_popularity_score",0),
                  "Vitality":row.get("avg_vitality_score",0)}
        fig, ax = plt.subplots(figsize=(5, 2.5))
        colors  = [score_color(v) for v in scores.values()]
        bars    = ax.barh(list(scores.keys()), list(scores.values()), color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.45, color="#e74c3c", linestyle="--", linewidth=0.8, label="At-risk threshold")
        for bar, val in zip(bars, scores.values()):
            ax.text(val+0.01, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown("**Key Metrics**")
        for k, v in {
            "Avg Rating":      f"{row.get('avg_rating',0):.2f} / 5.0",
            "Avg Sentiment":   f"{row.get('avg_sentiment',0):.3f}",
            "Reviews / month": f"{row.get('avg_review_count',0):.1f}",
            "Months in window":f"{int(row.get('months_in_window',0))}",
            "6-Month Trend":   f"{row.get('avg_vitality_trend',0):.5f}",
        }.items():
            st.markdown(f"- **{k}:** {v}")

    if ts is not None:
        prod_ts = ts[ts["parent_asin"]==asin].sort_values("period")
        if len(prod_ts) > 0:
            st.markdown("---")
            st.markdown("**Vitality, Health & Popularity Over Time**")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(prod_ts["period"], prod_ts["vitality_score"],   label="Vitality",
                    color="#9b59b6", linewidth=2)
            ax.plot(prod_ts["period"], prod_ts["health_score"],     label="Health",
                    color="#2ecc71", linewidth=1.5, linestyle="--")
            ax.plot(prod_ts["period"], prod_ts["popularity_score"], label="Popularity",
                    color="#3498db", linewidth=1.5, linestyle="--")
            ax.axhline(0.45, color="#e74c3c", linestyle=":", linewidth=0.9, label="At-risk threshold")
            ax.set_ylim(0,1); ax.set_xlabel("Month",fontsize=9); ax.set_ylabel("Score",fontsize=9)
            ax.legend(fontsize=8); ax.tick_params(labelsize=8)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            if "review_count" in prod_ts.columns:
                st.markdown("**Monthly Review Volume**")
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                ax2.bar(prod_ts["period"], prod_ts["review_count"],
                        color="#3498db", alpha=0.7, width=20)
                ax2.set_xlabel("Month",fontsize=9); ax2.set_ylabel("Reviews",fontsize=9)
                ax2.tick_params(labelsize=8)
                fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    else:
        st.info("Time-series file not found. Run `python run_pipeline.py` or place "
                "`product_vitality_scores.csv` in `data/` for trend charts.")


# ── Tab: At-Risk & Export ─────────────────────────────────────────────────────
def tab_export(df):
    st.subheader("Pruning Candidates")
    at_risk = (
        df[df["at_risk"]==1]
        .sort_values("avg_vitality_score")
        .reset_index(drop=True)
    )
    at_risk.insert(0, "rank", range(1, len(at_risk)+1))

    st.markdown(f"**{len(at_risk)} at-risk products** — sorted by Vitality Score "
                f"(lowest = highest priority for pruning).")

    top_n = st.slider("Show top N", 10, min(200, max(10, len(at_risk))),
                       min(50, len(at_risk)), 5)
    st.dataframe(at_risk.head(top_n), use_container_width=True, height=400)

    csv = at_risk.head(top_n).to_csv(index=False).encode()
    st.download_button(f"Download top-{top_n} pruning candidates",
                       csv, f"pruning_candidates_top{top_n}.csv", "text/csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("Product Vitality Scoring System")
    st.caption("Amazon Electronics · Predicting Product Decline Risk · MIS 790 Capstone · Supriya Thorat")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Pipeline", "Overview", "Product Table", "Drilldown", "At-Risk & Export"]
    )

    with tab1:
        tab_pipeline()

    df = load_summary()
    if df is None:
        for t in [tab2, tab3, tab4, tab5]:
            with t:
                st.warning("No data yet. Run `python run_pipeline.py` or place your "
                           "Google Drive CSVs in `App/data/`, then refresh.")
        return

    ts = load_timeseries()
    cat, store, at_risk_only, vrange = sidebar_filters(df)
    filtered = apply_filters(df, cat, store, at_risk_only, vrange)

    if len(filtered) == 0:
        for t in [tab2, tab3, tab4, tab5]:
            with t:
                st.warning("No products match the current filters.")
        return

    with tab2: tab_overview(filtered)
    with tab3: tab_product_table(filtered)
    with tab4: tab_drilldown(filtered, ts)
    with tab5: tab_export(filtered)


if __name__ == "__main__":
    main()
