"""Lexicon definitions for comparative analysis."""

import re
from typing import Dict, List, Optional

# Market / wealth positivity (count per 1k words) — also for per-term breakdown
MARKET_TERMS: List[str] = [
    "money", "debt", "wealth", "rich", "poor", "abundance", "seed", "harvest",
    "return", "business", "entrepreneur",
]
MARKET_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bprosper\w*", re.I),
    re.compile(r"\binvest\w*", re.I),
]
MARKET_PATTERN_NAMES: List[str] = ["prosper*", "invest*"]

# Individual / technique vocabulary
INDIVIDUAL_TERMS: List[str] = [
    "mindset", "believe", "declare", "vision", "favor", "breakthrough",
    "confidence", "discipline", "habit", "speak", "words", "purpose",
]

# Structural / political economy vocabulary
STRUCTURAL_TERMS: List[str] = [
    "wages", "labor", "inequality", "class", "unions", "regulation",
    "policy", "housing", "healthcare", "system", "exploitation",
]
STRUCTURAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcorporation\w*", re.I),
]

# Blessing causality: technique → outcome
CAUSALITY_TECHNIQUE: List[str] = ["faith", "words", "seed", "declare", "believe", "sow", "give"]
CAUSALITY_OUTCOME: List[str] = ["money", "blessing", "abundance", "harvest", "return", "prosper"]

# Scripture citation — book names + chapter:verse patterns
SCRIPTURE_BOOKS: List[str] = [
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges",
    "ruth", "samuel", "kings", "chronicles", "ezra", "nehemiah", "esther", "job",
    "psalm", "psalms", "proverbs", "ecclesiastes", "song", "isaiah", "jeremiah",
    "lamentations", "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah",
    "jonah", "micah", "nahum", "habakkuk", "zephaniah", "haggai", "zechariah",
    "malachi", "matthew", "mark", "luke", "john", "acts", "romans", "corinthians",
    "galatians", "ephesians", "philippians", "colossians", "thessalonians",
    "timothy", "titus", "philemon", "hebrews", "james", "peter", "jude", "revelation",
]
SCRIPTURE_PATTERN = re.compile(
    r"\b(" + "|".join(SCRIPTURE_BOOKS) + r")\s+\d+(?:\s*:\s*\d+)?(?:\s*[-–]\s*\d+)?",
    re.I
)

# Emotional / affective
EMOTION_POSITIVE: List[str] = ["joy", "hope", "blessed", "grateful", "peace", "love", "happy", "praise"]
EMOTION_NEGATIVE: List[str] = ["struggle", "battle", "enemy", "attack", "fear", "worry", "anxious", "suffer"]

# Temporal framing
TEMPORAL_FUTURE: List[str] = ["will", "going to", "one day", "destiny", "breakthrough", "future", "shall"]
TEMPORAL_PRESENT: List[str] = ["now", "today", "right now", "this moment", "currently"]
TEMPORAL_PAST: List[str] = ["was", "were", "testimony", "story", "testified", "remember"]

# Authority / certainty
AUTHORITY_TERMS: List[str] = ["god said", "bible says", "scripture", "thus saith", "the lord said", "word says"]
HEDGING_TERMS: List[str] = ["maybe", "perhaps", "might", "could", "possibly", "sometimes"]

# Call-to-action
CTA_TERMS: List[str] = ["give", "sow", "seed", "offer", "tithe", "donate", "declare", "speak", "believe", "receive"]

# Causality sentence patterns
CAUSALITY_PATTERNS: List[re.Pattern] = [
    re.compile(r"if you\s+\w+(?:\s+\w+){0,8}\s+(?:then\s+)?(?:god|he|the lord)\s+will", re.I),
    re.compile(r"when you\s+\w+(?:\s+\w+){0,8}\s+(?:god|he|the lord)\s+will", re.I),
    re.compile(r"as you\s+\w+(?:\s+\w+){0,8}\s+(?:god|he|the lord)\s+will", re.I),
    re.compile(r"god will\s+\w+(?:\s+\w+){0,6}\s+when you\s+\w+", re.I),
]


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens (alphanumeric sequences)."""
    if not text:
        return []
    return re.findall(r"\b[a-z0-9]+(?:'[a-z]+)?\b", text.lower())


def _count_token_matches(tokens: List[str], term: str, text_lower: str) -> int:
    """Count exact token matches. Single words: token match. Multi-word: regex on text."""
    if " " in term:
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.I)
        return len(pat.findall(text_lower))
    return sum(1 for t in tokens if t == term.lower())


def count_terms(text: str, terms: List[str], patterns: Optional[List[re.Pattern]] = None) -> int:
    """Count term occurrences using word-boundary token matching + regex for stems."""
    if not text:
        return 0
    text_lower = text.lower()
    tokens = _tokenize(text)
    count = sum(_count_token_matches(tokens, t, text_lower) for t in terms)
    if patterns:
        count += sum(len(p.findall(text_lower)) for p in patterns)
    return count


def count_per_term(
    text: str,
    terms: List[str],
    pattern_names: Optional[List[str]] = None,
    pattern_regexes: Optional[List[re.Pattern]] = None,
) -> Dict[str, int]:
    """Count each term separately. Uses word-boundary matching for terms; regex for stems."""
    if not text:
        return {}
    text_lower = text.lower()
    tokens = _tokenize(text)
    out: Dict[str, int] = {t: _count_token_matches(tokens, t, text_lower) for t in terms}
    if pattern_names and pattern_regexes:
        for name, p in zip(pattern_names, pattern_regexes):
            out[name] = len(p.findall(text_lower))
    return out
