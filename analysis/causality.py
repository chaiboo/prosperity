"""Blessing linkage extraction: technique→outcome co-occurrence + template matching."""

import re
from typing import List, Tuple

from .lexicon import (
    CAUSALITY_TECHNIQUE,
    CAUSALITY_OUTCOME,
    CAUSALITY_PATTERNS,
)


def extract_causality_sentences(text: str) -> List[str]:
    """Extract sentences matching 'if you... then God will...' style patterns."""
    if not text:
        return []
    # Split into sentences (rough: . ! ? or newlines)
    sentences = re.split(r"[.!?]\s+|\n+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    found = []
    for s in sentences:
        for p in CAUSALITY_PATTERNS:
            if p.search(s):
                found.append(s[:200] + ("..." if len(s) > 200 else ""))
                break
    return found


def extract_technique_outcome_pairs(text: str, window: int = 50) -> List[Tuple[str, str]]:
    """Find co-occurrence of technique → outcome within a word window."""
    if not text:
        return []
    words = text.lower().split()
    pairs = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^\w]", "", w)
        if clean in CAUSALITY_TECHNIQUE:
            for j in range(i + 1, min(i + window + 1, len(words))):
                w2 = re.sub(r"[^\w]", "", words[j])
                if w2 in CAUSALITY_OUTCOME:
                    pairs.append((clean, w2))
    return pairs


def count_technique_outcome_edges(df) -> List[Tuple[str, str, int]]:
    """Aggregate (technique, outcome, count) across corpus."""
    from collections import Counter
    all_pairs = []
    for t in df["transcript"].fillna(""):
        all_pairs.extend(extract_technique_outcome_pairs(str(t)))
    c = Counter(all_pairs)
    return [(a, b, n) for (a, b), n in c.most_common(30)]


def get_top_causality_templates(df, top_n: int = 20) -> List[Tuple[str, str, str]]:
    """Return (channel, title, excerpt) for top causality sentences."""
    rows = []
    for _, r in df.iterrows():
        sents = extract_causality_sentences(str(r.get("transcript", "")))
        for s in sents[:3]:  # max 3 per video
            rows.append((r.get("slug", ""), r.get("title", ""), s))
    # Sort by length (prefer longer, more meaningful)
    rows.sort(key=lambda x: -len(x[2]))
    return rows[:top_n]
