"""LLM-based structured extraction for theological discourse analysis.

Three extraction schemas:
1. Blessing Causality — conditional claims linking human action to divine response
2. Agency Attribution — who causes blessings/provision (God, Human, Enemy, System)
3. Suffering Framing — how suffering is explained (test, attack, lack of faith, etc.)

Runs offline via `run_extraction.py`. Results saved to data/processed/extractions.jsonl.
Requires GEMINI_API_KEY environment variable (free tier available).
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

EXTRACTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "extractions.jsonl"

# ---------------------------------------------------------------------------
# Pre-filter keywords (cheap sentence selection before sending to LLM)
# ---------------------------------------------------------------------------

CAUSALITY_KEYWORDS = {
    "if you", "when you", "as you", "god will", "the lord will",
    "he will", "blessing", "prosper", "harvest", "return", "reward",
    "sow", "seed", "give", "tithe", "faith", "believe", "declare",
    "receive", "breakthrough", "favor", "abundance",
}

AGENCY_KEYWORDS = {
    "god", "lord", "he", "bless", "gave", "gives", "provide",
    "heal", "restore", "withhold", "activate", "release",
    "enemy", "devil", "attack", "steal", "system", "government",
}

SUFFERING_KEYWORDS = {
    "suffer", "pain", "struggle", "trial", "test", "attack",
    "enemy", "devil", "lack", "poverty", "sick", "disease",
    "hardship", "difficulty", "trouble", "storm", "valley",
    "persecution", "oppression", "injustice", "inequality",
}


def _split_sentences(text: str) -> List[str]:
    """Rough sentence splitter for transcripts."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if len(s.strip()) > 20]


def _keyword_filter(sentences: List[str], keywords: set, min_matches: int = 1) -> List[str]:
    """Keep sentences containing at least min_matches keywords."""
    out = []
    for s in sentences:
        lower = s.lower()
        hits = sum(1 for k in keywords if k in lower)
        if hits >= min_matches:
            out.append(s)
    return out


def prefilter_sentences(text: str, schema: str) -> List[str]:
    """Select candidate sentences for a given schema."""
    sentences = _split_sentences(text)
    if schema == "causality":
        return _keyword_filter(sentences, CAUSALITY_KEYWORDS, min_matches=2)
    elif schema == "agency":
        return _keyword_filter(sentences, AGENCY_KEYWORDS, min_matches=2)
    elif schema == "suffering":
        return _keyword_filter(sentences, SUFFERING_KEYWORDS, min_matches=1)
    return sentences


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CAUSALITY_SYSTEM = """You are a theological discourse analyst. Extract structured causal claims from transcript text.

For each conditional theological claim where a human action is linked to a divine response, extract:
- condition: The situation or prerequisite (e.g., "if you are faithful")
- action_required: What the human must do (e.g., "sow a seed", "tithe", "believe")
- divine_response: What God/the Lord will do in response (e.g., "bless you abundantly")
- material_outcome: true if the outcome involves money, wealth, health, or material provision
- spiritual_outcome: true if the outcome involves peace, salvation, spiritual growth, or relationship with God

Only extract claims that actually link human action to divine response. Skip generic statements.
Return a JSON object with key "items" containing an array. If no claims found, return {"items": []}.
"""

AGENCY_SYSTEM = """You are a theological discourse analyst. Extract agency attribution from transcript text about blessings and provision.

For each statement about blessing, healing, or provision, extract:
- event: Brief description of what happens (e.g., "financial breakthrough")
- agent: Who causes it — one of: "God", "Human", "Enemy", "System", "Unspecified"
  - "God" = God acts of his own will
  - "Human" = human action triggers or activates the outcome
  - "Enemy" = devil/enemy is blocking it
  - "System" = structural/systemic cause
  - "Unspecified" = agent unclear
- verb_type: The type of action — one of: "bless", "withhold", "activate", "give", "heal", "restore", "block", "other"

Return a JSON object with key "items" containing an array. If no relevant statements, return {"items": []}.
"""

SUFFERING_SYSTEM = """You are a theological discourse analyst. Extract suffering framing from transcript text.

For each reference to suffering, hardship, or difficulty, extract:
- suffering_reference: Brief description (e.g., "financial hardship", "illness")
- framing: How the suffering is explained — one of:
  - "test" = God is testing faith
  - "attack" = spiritual warfare / enemy attack
  - "lack_of_faith" = suffering caused by insufficient faith or disobedience
  - "inevitable" = suffering is part of life / following Christ
  - "systemic" = caused by social/economic systems
  - "discipline" = God is correcting/teaching
  - "unspecified" = framing unclear

Return a JSON object with key "items" containing an array. If no suffering references, return {"items": []}.
"""

SCHEMA_PROMPTS = {
    "causality": CAUSALITY_SYSTEM,
    "agency": AGENCY_SYSTEM,
    "suffering": SUFFERING_SYSTEM,
}

# Gemini free tier limits vary by project; we retry on RESOURCE_EXHAUSTED
DEFAULT_MODEL = "gemini-2.5-flash"
RATE_LIMIT_SLEEP = 0
MAX_RETRIES = 5

COMBINED_SYSTEM = """You are a theological discourse analyst. Analyze the following transcript excerpts and extract three types of structured data.

## 1. Causal Claims
For each conditional theological claim where a human action is linked to a divine response, extract:
- condition: The situation or prerequisite (e.g., "if you are faithful")
- action_required: What the human must do (e.g., "sow a seed", "tithe", "believe")
- divine_response: What God/the Lord will do in response (e.g., "bless you abundantly")
- material_outcome: true if the outcome involves money, wealth, health, or material provision
- spiritual_outcome: true if the outcome involves peace, salvation, spiritual growth, or relationship with God
Only extract claims that actually link human action to divine response. Skip generic statements.

## 2. Agency Attribution
For each statement about blessing, healing, or provision, extract:
- event: Brief description of what happens (e.g., "financial breakthrough")
- agent: Who causes it — one of: "God", "Human", "Enemy", "System", "Unspecified"
- verb_type: The type of action — one of: "bless", "withhold", "activate", "give", "heal", "restore", "block", "other"

## 3. Suffering Framing
For each reference to suffering, hardship, or difficulty, extract:
- suffering_reference: Brief description (e.g., "financial hardship", "illness")
- framing: How the suffering is explained — one of: "test", "attack", "lack_of_faith", "inevitable", "systemic", "discipline", "unspecified"

Return a JSON object with three keys: "causality", "agency", "suffering", each containing an array. Use empty arrays if nothing found for a category.
"""


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _get_client():
    """Create Gemini client. Raises if no API key."""
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "Missing google-genai. Install with: pip install google-genai"
        )
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY environment variable.\n"
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=key)


def _call_with_retry(client, model: str, contents: str, config: dict) -> str:
    """Call Gemini with automatic retry on rate limit (RESOURCE_EXHAUSTED)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                import re as _re
                match = _re.search(r"retry in ([\d.]+)s", err_str, _re.I)
                wait = float(match.group(1)) + 5 if match else 60
                print(f"\n  Rate limited. Waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Still rate-limited after {MAX_RETRIES} retries")


def _parse_response(content: str) -> List[dict]:
    """Parse LLM JSON response, handling both list and dict formats."""
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict):
        items = parsed.get("items", [])
        if isinstance(items, list):
            return items
        for v in parsed.values():
            if isinstance(v, list):
                return v
    return []


def extract_schema(
    client,
    sentences: List[str],
    schema: str,
    model: str = DEFAULT_MODEL,
    max_sentences_per_call: int = 30,
) -> List[dict]:
    """Send pre-filtered sentences to Gemini, return structured extractions."""
    if not sentences:
        return []

    system_prompt = SCHEMA_PROMPTS[schema]
    results = []

    for i in range(0, len(sentences), max_sentences_per_call):
        batch = sentences[i : i + max_sentences_per_call]
        user_text = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(batch))

        content = _call_with_retry(client, model, user_text, {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "temperature": 0.1,
        })

        try:
            results.extend(_parse_response(content))
        except json.JSONDecodeError:
            pass

        time.sleep(RATE_LIMIT_SLEEP)

    return results


def extract_all_schemas(
    client,
    text: str,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Run all three schemas in a single API call per batch of sentences."""
    all_keywords = CAUSALITY_KEYWORDS | AGENCY_KEYWORDS | SUFFERING_KEYWORDS
    sentences = _split_sentences(text)
    candidates = _keyword_filter(sentences, all_keywords, min_matches=1)

    if not candidates:
        return {"causality": [], "agency": [], "suffering": []}

    results = {"causality": [], "agency": [], "suffering": []}

    for i in range(0, len(candidates), 80):
        batch = candidates[i : i + 80]
        user_text = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(batch))

        content = _call_with_retry(client, model, user_text, {
            "system_instruction": COMBINED_SYSTEM,
            "response_mime_type": "application/json",
            "temperature": 0.1,
        })

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for key in ("causality", "agency", "suffering"):
                    items = parsed.get(key, [])
                    if isinstance(items, list):
                        results[key].extend(items)
        except json.JSONDecodeError:
            pass

        time.sleep(RATE_LIMIT_SLEEP)

    return results


# ---------------------------------------------------------------------------
# Loading saved extractions
# ---------------------------------------------------------------------------

def load_extractions(path: Optional[Path] = None) -> List[dict]:
    """Load extractions from JSONL file."""
    p = path or EXTRACTIONS_PATH
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
