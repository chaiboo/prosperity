"""
Comment analysis: sentiment (VADER), topic modeling (LDA), lexicon echo,
engagement analysis, and LLM-based classification.
"""

import json, os, re, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from analysis.lexicon import (
    MARKET_TERMS, MARKET_PATTERNS, MARKET_PATTERN_NAMES,
    INDIVIDUAL_TERMS, CAUSALITY_TECHNIQUE, CAUSALITY_OUTCOME,
)

COMMENTS_DIR = Path("data/comments")
COMMENT_CLASSIFICATIONS_DIR = Path("data/comments/classified")

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


# ---------------------------------------------------------------------------
# Market lexicon echo — do commenters mirror prosperity language?
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\b[a-z0-9]+(?:'?[a-z]+)?\b")

def _count_matches(text: str, terms: list[str], patterns: list[re.Pattern] = []) -> int:
    tokens = set(_TOKEN_RE.findall(text.lower()))
    count = sum(1 for t in terms if t in tokens)
    for pat in patterns:
        count += len(pat.findall(text))
    return count


def add_lexicon_scores(df: pd.DataFrame) -> pd.DataFrame:
    df["market_hits"] = df["text"].apply(
        lambda t: _count_matches(t, MARKET_TERMS, MARKET_PATTERNS)
    )
    df["individual_hits"] = df["text"].apply(
        lambda t: _count_matches(t, INDIVIDUAL_TERMS)
    )
    df["technique_hits"] = df["text"].apply(
        lambda t: _count_matches(t, CAUSALITY_TECHNIQUE)
    )
    df["outcome_hits"] = df["text"].apply(
        lambda t: _count_matches(t, CAUSALITY_OUTCOME)
    )
    df["has_prosperity_language"] = (df["market_hits"] + df["technique_hits"] + df["outcome_hits"]) > 0
    return df


def get_prosperity_echo_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    with_lang = df["has_prosperity_language"].sum()
    return {
        "total": total,
        "with_prosperity_language": int(with_lang),
        "pct": 100 * with_lang / total if total > 0 else 0,
        "top_market_terms": (
            df[df["market_hits"] > 0]["text"]
            .apply(lambda t: [w for w in _TOKEN_RE.findall(t.lower()) if w in MARKET_TERMS])
            .explode().value_counts().head(10).to_dict()
        ),
        "top_technique_terms": (
            df[df["technique_hits"] > 0]["text"]
            .apply(lambda t: [w for w in _TOKEN_RE.findall(t.lower()) if w in CAUSALITY_TECHNIQUE])
            .explode().value_counts().head(10).to_dict()
        ),
    }


# ---------------------------------------------------------------------------
# Engagement x sentiment
# ---------------------------------------------------------------------------

def get_engagement_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("sentiment_label").agg(
        count=("likes", "size"),
        avg_likes=("likes", "mean"),
        median_likes=("likes", "median"),
        total_likes=("likes", "sum"),
    ).reset_index()
    return grouped


def get_engagement_by_prosperity(df: pd.DataFrame) -> pd.DataFrame:
    df["comment_type"] = df["has_prosperity_language"].map(
        {True: "Uses prosperity language", False: "No prosperity language"}
    )
    grouped = df.groupby("comment_type").agg(
        count=("likes", "size"),
        avg_likes=("likes", "mean"),
        median_likes=("likes", "median"),
    ).reset_index()
    return grouped


# ---------------------------------------------------------------------------
# LLM-based comment classification
# ---------------------------------------------------------------------------

COMMENT_CLASSIFY_SYSTEM = """You are analyzing YouTube comments on a prosperity theology video.
Classify each comment into ONE primary category:

- testimony: Personal story of blessing, healing, or miracle
- prayer_request: Asking for prayer or divine help
- affirmation: Agreement, "amen", praising the message
- gratitude: Thanking God, the preacher, or expressing thankfulness
- prosperity_echo: Echoing prosperity theology language (seed, harvest, breakthrough, claim it)
- scripture: Quoting or referencing Bible verses
- personal_struggle: Sharing hardship, difficulty, suffering
- criticism: Disagreeing with or questioning the message
- other: Doesn't fit above categories

For each comment, return: {"id": <number>, "category": "<category>", "confidence": <0.0-1.0>}
Return a JSON array of objects."""


def classify_comments_llm(
    video_id: str,
    df: pd.DataFrame,
    batch_size: int = 25,
    model: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    COMMENT_CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = COMMENT_CLASSIFICATIONS_DIR / f"{video_id}.jsonl"

    if cache_path.exists():
        cached = [json.loads(l) for l in open(cache_path)]
        cat_map = {r["id"]: r["category"] for r in cached}
        conf_map = {r["id"]: r.get("confidence", 1.0) for r in cached}
        df["llm_category"] = df.index.map(lambda i: cat_map.get(i, "other"))
        df["llm_confidence"] = df.index.map(lambda i: conf_map.get(i, 0.0))
        return df

    try:
        from google import genai
    except ImportError:
        df["llm_category"] = "unavailable"
        df["llm_confidence"] = 0.0
        return df

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        df["llm_category"] = "unavailable"
        df["llm_confidence"] = 0.0
        return df

    client = genai.Client(api_key=key)
    all_results = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        user_text = "\n".join(
            f"[{idx}] {row['text'][:200]}" for idx, row in batch.iterrows()
        )
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_text,
                config={
                    "system_instruction": COMMENT_CLASSIFY_SYSTEM,
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            parsed = json.loads(response.text.strip())
            if isinstance(parsed, list):
                all_results.extend(parsed)
        except Exception as e:
            print(f"  LLM classify batch error: {e}")

    with open(cache_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    cat_map = {r["id"]: r["category"] for r in all_results}
    conf_map = {r["id"]: r.get("confidence", 1.0) for r in all_results}
    df["llm_category"] = df.index.map(lambda i: cat_map.get(i, "other"))
    df["llm_confidence"] = df.index.map(lambda i: conf_map.get(i, 0.0))
    return df


def load_classifications(video_id: str) -> pd.DataFrame:
    cache_path = COMMENT_CLASSIFICATIONS_DIR / f"{video_id}.jsonl"
    if not cache_path.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(cache_path)]
    return pd.DataFrame(rows)
