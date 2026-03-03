"""Compute derived NLP features from corpus."""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .lexicon import (
    MARKET_TERMS,
    MARKET_PATTERNS,
    MARKET_PATTERN_NAMES,
    INDIVIDUAL_TERMS,
    STRUCTURAL_TERMS,
    STRUCTURAL_PATTERNS,
    SCRIPTURE_PATTERN,
    EMOTION_POSITIVE,
    EMOTION_NEGATIVE,
    TEMPORAL_FUTURE,
    TEMPORAL_PRESENT,
    TEMPORAL_PAST,
    AUTHORITY_TERMS,
    HEDGING_TERMS,
    CTA_TERMS,
    count_terms,
    count_per_term,
)

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "corpus.csv"


def compute_market_density(text: str, word_count: int) -> float:
    """Market lexicon density per 1,000 words."""
    if not text or word_count <= 0:
        return 0.0
    count = count_terms(text, MARKET_TERMS, MARKET_PATTERNS)
    return 1000 * count / word_count


def compute_market_per_term(text: str, word_count: int) -> Dict[str, float]:
    """Per-term market density per 1,000 words."""
    if not text or word_count <= 0:
        return {}
    counts = count_per_term(text, MARKET_TERMS, MARKET_PATTERN_NAMES, MARKET_PATTERNS)
    return {k: 1000 * v / word_count for k, v in counts.items()}


def compute_individualization_score(text: str, word_count: int) -> float:
    if not text or word_count <= 0:
        return 0.0
    count = count_terms(text, INDIVIDUAL_TERMS, None)
    return 1000 * count / word_count


def compute_structural_score(text: str, word_count: int) -> float:
    if not text or word_count <= 0:
        return 0.0
    count = count_terms(text, STRUCTURAL_TERMS, STRUCTURAL_PATTERNS)
    return 1000 * count / word_count


def compute_individualization_index(text: str, word_count: int) -> float:
    ind = compute_individualization_score(text, word_count)
    struct = compute_structural_score(text, word_count)
    return ind - struct


def compute_scripture_density(text: str, word_count: int) -> float:
    """Scripture citation count per 1,000 words."""
    if not text or word_count <= 0:
        return 0.0
    count = len(SCRIPTURE_PATTERN.findall(text.lower()))
    return 1000 * count / word_count


def compute_emotion_scores(text: str, word_count: int) -> tuple[float, float]:
    """(positive per 1k, negative per 1k)."""
    if not text or word_count <= 0:
        return 0.0, 0.0
    pos = count_terms(text, EMOTION_POSITIVE, None)
    neg = count_terms(text, EMOTION_NEGATIVE, None)
    return 1000 * pos / word_count, 1000 * neg / word_count


def compute_temporal_scores(text: str, word_count: int) -> tuple[float, float, float]:
    """(future per 1k, present per 1k, past per 1k)."""
    if not text or word_count <= 0:
        return 0.0, 0.0, 0.0
    # "was"/"were" are common; use word boundaries
    future = count_terms(text, TEMPORAL_FUTURE, None)
    present = count_terms(text, TEMPORAL_PRESENT, None)
    past = count_terms(text, TEMPORAL_PAST, None)
    return 1000 * future / word_count, 1000 * present / word_count, 1000 * past / word_count


def compute_authority_score(text: str, word_count: int) -> float:
    """Authority phrase occurrences per 1,000 words (counts each occurrence, not presence/absence)."""
    if not text or word_count <= 0:
        return 0.0
    text_lower = text.lower()
    count = 0
    for phrase in AUTHORITY_TERMS:
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b", re.I)
        count += len(pat.findall(text_lower))
    return 1000 * count / word_count


def compute_hedging_score(text: str, word_count: int) -> float:
    if not text or word_count <= 0:
        return 0.0
    count = count_terms(text, HEDGING_TERMS, None)
    return 1000 * count / word_count


def compute_cta_density(text: str, word_count: int) -> float:
    """Call-to-action terms per 1,000 words."""
    if not text or word_count <= 0:
        return 0.0
    count = count_terms(text, CTA_TERMS, None)
    return 1000 * count / word_count


def compute_sentiment_simple(text: str) -> float:
    """Lexicon-based sentiment: (pos - neg) / total words. Returns -1 to 1. Not per-1k."""
    if not text:
        return 0.0
    pos = count_terms(text, EMOTION_POSITIVE, None)
    neg = count_terms(text, EMOTION_NEGATIVE, None)
    words = text.lower().split()
    if not words:
        return 0.0
    return (pos - neg) / len(words)


def load_corpus(path: Optional[Path] = None) -> pd.DataFrame:
    """Load corpus and filter to rows with valid transcripts."""
    p = path or CORPUS_PATH
    if p.exists():
        df = pd.read_csv(p)
    else:
        # Fallback: load from raw jsonl files (data/raw/)
        raw_dir = p.parent.parent / "raw"  # data/processed -> data, then data/raw
        if not raw_dir.exists():
            raise FileNotFoundError(f"Corpus not found: {p}. Run build_corpus.py first.")
        import json
        rows = []
        for f in raw_dir.glob("transcripts_*.jsonl"):
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    if "transcript" in df.columns and "transcript_status" in df.columns:
        df = df[df["transcript_status"] == "ok"].copy()
        df = df[df["transcript"].notna() & (df["transcript"].astype(str).str.len() > 0)].copy()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived NLP features to corpus DataFrame."""
    out = df.copy()
    texts = out["transcript"].fillna("").astype(str)
    word_counts = out["word_count"].fillna(0).astype(int)
    word_counts = word_counts.replace(0, 1)

    out["market_density"] = [compute_market_density(t, w) for t, w in zip(texts, word_counts)]
    out["individualization_score"] = [compute_individualization_score(t, w) for t, w in zip(texts, word_counts)]
    out["structural_score"] = [compute_structural_score(t, w) for t, w in zip(texts, word_counts)]
    out["individualization_index"] = [compute_individualization_index(t, w) for t, w in zip(texts, word_counts)]
    out["scripture_density"] = [compute_scripture_density(t, w) for t, w in zip(texts, word_counts)]

    emotion = [compute_emotion_scores(t, w) for t, w in zip(texts, word_counts)]
    out["emotion_positive"] = [e[0] for e in emotion]
    out["emotion_negative"] = [e[1] for e in emotion]
    out["emotion_ratio"] = [
        e[0] / e[1] if e[1] > 0 else (e[0] if e[0] > 0 else 0) for e in emotion
    ]

    temporal = [compute_temporal_scores(t, w) for t, w in zip(texts, word_counts)]
    out["temporal_future"] = [t[0] for t in temporal]
    out["temporal_present"] = [t[1] for t in temporal]
    out["temporal_past"] = [t[2] for t in temporal]

    out["authority_score"] = [compute_authority_score(t, w) for t, w in zip(texts, word_counts)]
    out["hedging_score"] = [compute_hedging_score(t, w) for t, w in zip(texts, word_counts)]
    out["cta_density"] = [compute_cta_density(t, w) for t, w in zip(texts, word_counts)]
    out["sentiment"] = [compute_sentiment_simple(t) for t in texts]

    return out
