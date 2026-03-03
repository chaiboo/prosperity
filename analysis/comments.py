"""
Comment analysis: sentiment (VADER), topic modeling (LDA), and key terms.
"""

import json, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

COMMENTS_DIR = Path("data/comments")

_vader = SentimentIntensityAnalyzer()


def load_comments(video_id: str) -> pd.DataFrame:
    path = COMMENTS_DIR / f"{video_id}.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(path)]
    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    scores = df["text"].apply(lambda t: _vader.polarity_scores(t))
    df["sentiment_compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment_pos"] = scores.apply(lambda s: s["pos"])
    df["sentiment_neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment_neu"] = scores.apply(lambda s: s["neu"])
    df["sentiment_label"] = df["sentiment_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    return df


def get_topic_model(df: pd.DataFrame, n_topics: int = 6, n_top_words: int = 8):
    texts = df["text"].tolist()
    vec = CountVectorizer(
        max_df=0.85, min_df=3, stop_words="english",
        max_features=2000, token_pattern=r"[a-z]{3,}",
    )
    dtm = vec.fit_transform([t.lower() for t in texts])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    lda.fit(dtm)

    feature_names = vec.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        topics.append({"topic": i, "words": top_words, "label": ", ".join(top_words[:4])})

    doc_topics = lda.transform(dtm)
    df["dominant_topic"] = doc_topics.argmax(axis=1)

    return topics, df


def get_top_terms(df: pd.DataFrame, top_n: int = 30) -> list[tuple[str, float]]:
    vec = TfidfVectorizer(
        max_df=0.8, min_df=3, stop_words="english",
        max_features=3000, token_pattern=r"[a-z]{3,}",
    )
    tfidf = vec.fit_transform([t.lower() for t in df["text"]])
    mean_scores = tfidf.mean(axis=0).A1
    feature_names = vec.get_feature_names_out()
    top_idx = mean_scores.argsort()[-top_n:][::-1]
    return [(feature_names[i], mean_scores[i]) for i in top_idx]


def get_bigrams(df: pd.DataFrame, top_n: int = 20) -> list[tuple[str, int]]:
    vec = CountVectorizer(
        ngram_range=(2, 2), stop_words="english",
        max_features=5000, min_df=3, token_pattern=r"[a-z]{3,}",
    )
    X = vec.fit_transform([t.lower() for t in df["text"]])
    counts = X.sum(axis=0).A1
    names = vec.get_feature_names_out()
    top_idx = counts.argsort()[-top_n:][::-1]
    return [(names[i], int(counts[i])) for i in top_idx]
