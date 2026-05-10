"""
Step 9: LDA topic modeling.
Extracted from Capstone_MIS_790.ipynb — Step 9.

Discovered topics (Electronics dataset, 5 topics):
  Topic 0 — Camera & Display:      camera, light, screen, lens, photo
  Topic 1 — Device Accessories:    case, laptop, ipad, kindle, cover
  Topic 2 — Cables & Charging:     cable, usb, power, cord, charger
  Topic 3 — Product Reliability:   work, worked, didn't, working, stopped  ← KEY NEGATIVE SIGNAL
  Topic 4 — General Satisfaction:  great, good, works, easy, quality
"""

import re
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import warnings
warnings.filterwarnings("ignore")

NUM_TOPICS = 5


def _preprocess(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def run_lda(df, num_topics=NUM_TOPICS):
    """Step 9: Train LDA on review text, add topic columns to df."""
    print(f"Preprocessing text for LDA ({len(df):,} reviews)...")
    df = df.copy()
    df["lda_tokens"] = df["review_text"].fillna("").apply(_preprocess)

    print("Building LDA dictionary and corpus...")
    dictionary = corpora.Dictionary(df["lda_tokens"])
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(tokens) for tokens in df["lda_tokens"]]

    print(f"Training LDA model ({num_topics} topics)...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=5,
        alpha="auto",
    )

    print("\nDiscovered Topics:")
    for idx, topic in lda_model.print_topics(num_words=8):
        print(f"  Topic {idx}: {topic}")

    print("\nExtracting per-review topic distributions...")
    def topic_vector(bow):
        topics = dict(lda_model.get_document_topics(bow, minimum_probability=0))
        return [topics.get(i, 0.0) for i in range(num_topics)]

    vectors  = [topic_vector(bow) for bow in corpus]
    topic_df = pd.DataFrame(vectors, columns=[f"lda_topic_{i}" for i in range(num_topics)])
    df       = pd.concat([df.reset_index(drop=True), topic_df], axis=1)

    print("LDA complete. Topic columns added:", list(topic_df.columns))
    return df, lda_model, dictionary
