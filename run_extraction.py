"""Run LLM-based structured extraction on the YouTube corpus.

Usage:
    export GEMINI_API_KEY='your-key'
    python run_extraction.py                   # All videos
    python run_extraction.py -c joel_osteen    # One channel
    python run_extraction.py --resume          # Skip already-extracted videos
    python run_extraction.py --limit 5         # Test with 5 videos

Results are saved incrementally to data/processed/extractions.jsonl.
Each line: {video_id, slug, title, causality: [...], agency: [...], suffering: [...]}

Free tier limits (gemini-2.5-flash-lite): 15 RPM, 1000 RPD.
At ~3 calls per video + rate limiting, expect ~2 videos/min → ~2.5 hours for 300 videos.
Use --resume to continue after interruption.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from analysis.features import load_corpus
from analysis.extract import (
    _get_client,
    extract_all_schemas,
    EXTRACTIONS_PATH,
    DEFAULT_MODEL,
)


def load_existing_ids(path: Path) -> set:
    """Load video_ids that have already been extracted."""
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="Run LLM extraction on YouTube corpus")
    parser.add_argument(
        "-c", "--channel", action="append", dest="channels", metavar="SLUG",
        help="Channel slug to extract (repeatable). Default: all.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip videos already in extractions.jsonl.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max videos to process (for testing).",
    )
    args = parser.parse_args()

    df = load_corpus()
    if args.channels:
        df = df[df["slug"].isin(args.channels)]

    if df.empty:
        print("No videos to process.")
        return

    done_ids = load_existing_ids(EXTRACTIONS_PATH) if args.resume else set()
    todo = df[~df["video_id"].isin(done_ids)]

    if args.limit:
        todo = todo.head(args.limit)

    if todo.empty:
        print("All videos already extracted. Use without --resume to re-extract.")
        return

    print(f"Extracting {len(todo)} videos with model={args.model}")
    print(f"Output: {EXTRACTIONS_PATH}")
    print(f"Rate limit sleep: built into extract module (~7s between API calls)")

    client = _get_client()

    EXTRACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXTRACTIONS_PATH.open("a", encoding="utf-8") as f:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="extracting"):
            text = str(row.get("transcript", ""))
            if not text or len(text) < 50:
                continue

            try:
                results = extract_all_schemas(client, text, model=args.model)
            except Exception as e:
                print(f"\nError on {row['video_id']}: {e}")
                continue

            record = {
                "video_id": row["video_id"],
                "slug": row["slug"],
                "title": row.get("title", ""),
                "causality": results.get("causality", []),
                "agency": results.get("agency", []),
                "suffering": results.get("suffering", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nDone. Results in {EXTRACTIONS_PATH}")

    from analysis.extract import load_extractions
    all_ext = load_extractions()
    total_causality = sum(len(r.get("causality", [])) for r in all_ext)
    total_agency = sum(len(r.get("agency", [])) for r in all_ext)
    total_suffering = sum(len(r.get("suffering", [])) for r in all_ext)
    print(f"Total extractions: {len(all_ext)} videos")
    print(f"  Causality claims: {total_causality}")
    print(f"  Agency attributions: {total_agency}")
    print(f"  Suffering framings: {total_suffering}")


if __name__ == "__main__":
    main()
