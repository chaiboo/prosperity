"""Diagnostic script for transcript fetch failures.

Run this before re-running the full corpus builder to understand
exactly why transcripts are returning 'error'.

Usage:
    python diagnose_transcripts.py

It will test a handful of video IDs from your corpus and print the
full exception type + message for each failure so you know what to fix.
"""

import traceback
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

try:
    from youtube_transcript_api._errors import TooManyRequests
except ImportError:
    class TooManyRequests(Exception):
        pass

CORPUS_PATH = "data/processed/corpus.csv"  # adjust if needed

# A few known-public videos as sanity checks independent of your corpus
SANITY_CHECK_IDS = [
    ("TED Talk (public)", "UF8uR6Z6KLc"),
    ("YouTube Help (public)", "tntdHmBh24k"),
]


def probe(video_id: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  https://www.youtube.com/watch?v={video_id}")
    print(f"{'='*60}")
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        print("  list_transcripts() succeeded.")
        for t in transcript_list:
            print(f"    lang={t.language_code}  generated={t.is_generated}  translatable={t.is_translatable}")
        # Try fetching the first available
        try:
            first = next(iter(YouTubeTranscriptApi.list_transcripts(video_id)))
            chunks = first.fetch()
            # Handle both dict-style and object-style chunks
            texts = []
            for c in chunks[:3]:  # just first 3 chunks
                if isinstance(c, dict):
                    texts.append(c.get("text", ""))
                else:
                    texts.append(getattr(c, "text", "") or "")
            print(f"  fetch() succeeded. First 3 chunks: {texts}")
        except Exception as e:
            print(f"  fetch() FAILED: {type(e).__name__}: {e}")
    except TranscriptsDisabled:
        print("  RESULT: TranscriptsDisabled — owner has turned off captions.")
    except NoTranscriptFound:
        print("  RESULT: NoTranscriptFound — no captions in any language.")
    except VideoUnavailable:
        print("  RESULT: VideoUnavailable — video is private or deleted.")
    except TooManyRequests:
        print("  RESULT: TooManyRequests — YouTube is rate-limiting this IP.")
    except Exception as e:
        print(f"  RESULT: Unexpected exception!")
        print(f"  Type   : {type(e).__name__}")
        print(f"  Message: {e}")
        print("  Traceback:")
        traceback.print_exc()


def main():
    print("\n=== youtube-transcript-api version ===")
    try:
        import importlib.metadata
        print(" ", importlib.metadata.version("youtube-transcript-api"))
    except Exception:
        print("  (could not determine version)")

    print("\n=== Sanity checks (known-public videos) ===")
    for label, vid in SANITY_CHECK_IDS:
        probe(vid, label)

    print("\n=== Corpus sample (first 5 error rows) ===")
    try:
        df = pd.read_csv(CORPUS_PATH)
        error_rows = df[df["transcript_status"] == "error"].head(5)
        if error_rows.empty:
            print("  No error rows found in corpus — nothing to diagnose here.")
        else:
            for _, row in error_rows.iterrows():
                probe(row["video_id"], row["title"])
    except FileNotFoundError:
        print(f"  Corpus not found at {CORPUS_PATH}. Skipping corpus sample.")

    print("\n=== Diagnosis summary ===")
    print("""
  Common causes and fixes:
  ---------------------------------------------------------------
  TranscriptsDisabled / NoTranscriptFound
      → Those specific videos have no captions. Expected for some.
        If ALL videos show this, the channels may have disabled captions.

  TooManyRequests / HTTP 429 / RequestBlocked
      → YouTube is blocking transcript requests from your IP.
        This is very common on cloud/server IPs (AWS, GCP, etc.).
        Fix: run the script from a residential IP or use a proxy.

  Unexpected exception with 'NoneType' / attribute errors
      → API version mismatch. Run:
            pip install --upgrade youtube-transcript-api
        The fetch() return type changed in v0.6 (dicts → objects).

  Connection errors / timeouts
      → Network issue or YouTube blocking. Check internet access.
  ---------------------------------------------------------------
""")


if __name__ == "__main__":
    main()