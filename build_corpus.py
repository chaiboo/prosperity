"""YouTube Prosperity Sermon Corpus Builder

Goal
----
Build a reproducible corpus of transcripts + metadata for:
- 2–3 prosperity / prosperity-adjacent YouTube channels
- 1 high-reach non-prosperity evangelical control channel

Sampling rule (per channel)
--------------------------
- Take top N most-viewed long-form videos
- Exclude Shorts/clips by requiring duration >= MIN_DURATION_SECONDS (default 8 minutes)
- Pull transcript text if available (manual or auto captions)

Outputs
-------
- data/raw/videos_<channel>.csv          (metadata)
- data/raw/transcripts_<channel>.jsonl   (one JSON per video: id, title, transcript, etc.)
- data/processed/corpus.csv              (flat table: one row per video)

Prereqs
-------
1) Create a YouTube Data API v3 key and set env var YT_API_KEY
   - macOS/Linux:
       export YT_API_KEY='your-api-key-here'
   - Windows (PowerShell):
       setx YT_API_KEY "your-api-key-here"

2) Create and activate a virtual environment (recommended):
   python3 -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows

3) Install required dependencies:
   pip install --upgrade pip
   pip install google-api-python-client "youtube-transcript-api>=1.0" pandas python-dateutil tqdm yt-dlp

4) (Recommended) Use a residential proxy to avoid YouTube blocking transcript requests.
   YouTube actively blocks transcript scraping from non-browser/datacenter IPs.
   The most reliable fix is routing requests through a residential proxy.

   NOTE: Cookie-based auth (cookies.txt) is currently broken in youtube-transcript-api
   v1.x due to recent YouTube API changes — do not rely on it.

   Option A — Webshare residential proxy (recommended, ~$5/mo for light use):
     a) Sign up at https://www.webshare.io and buy a "Residential" proxy package.
        (Do NOT buy "Proxy Server" or "Static Residential" — those won't work.)
     b) Copy your proxy username and password from Webshare Proxy Settings.
     c) Set env vars:
          export YT_PROXY_USER=your_username
          export YT_PROXY_PASS=your_password

   Option B — Generic HTTP proxy (if you already have one):
     a) Set env var:
          export YT_PROXY_URL=http://user:pass@host:port

   If neither is set, the script will still run but blocked videos will be skipped.

Usage
-----
  python build_corpus.py              # Build all channels
  python build_corpus.py --fresh      # Clear all data and rebuild from scratch
  python build_corpus.py -c elevation # Build only Elevation Church
  python build_corpus.py -c elevation -c joel_osteen  # Build multiple

Notes
-----
- Compatible with youtube-transcript-api v1.x (API was redesigned in v1.0).
- Some channels disable transcripts; those videos will be recorded with transcript_status.
- transcript_status values: ok, disabled, not_found, unavailable, blocked, rate_limited, error:<type>
- This script is designed for transparency + reproducibility.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import json
import time
import shutil
import pathlib
import tempfile
import subprocess
from typing import Dict, List, Optional, Iterable, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError as e:
    raise ImportError(
        "Missing required packages. Activate the virtualenv and install dependencies:\n"
        "  source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows\n"
        "  pip install -r requirements.txt\n"
        f"Original error: {e}"
    ) from e

# youtube-transcript-api v1.x API
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
)

try:
    from youtube_transcript_api._errors import CouldNotRetrieveTranscript
except ImportError:
    class CouldNotRetrieveTranscript(Exception):
        pass

try:
    from youtube_transcript_api._errors import TooManyRequests
except ImportError:
    class TooManyRequests(Exception):
        pass

try:
    from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
except ImportError:
    WebshareProxyConfig = None
    GenericProxyConfig = None


# ----------------------------
# Configuration
# ----------------------------

DATA_DIR = pathlib.Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YT_API_KEY = os.environ.get("YT_API_KEY", "").strip()

# Proxy config for bypassing YouTube transcript blocking (see header for setup instructions).
# Webshare residential proxy takes priority; falls back to generic proxy URL; falls back to none.
YT_PROXY_USER = os.environ.get("YT_PROXY_USER", "").strip() or None
YT_PROXY_PASS = os.environ.get("YT_PROXY_PASS", "").strip() or None
YT_PROXY_URL  = os.environ.get("YT_PROXY_URL",  "").strip() or None

TOP_N_PER_CHANNEL = 50
MIN_DURATION_SECONDS = 8 * 60
PREFERRED_LANGS = ["en", "en-US", "en-GB", "en-nl"]  # en-nl = auto-translated English

SLEEP_BETWEEN_API_CALLS = 0.15
SLEEP_BETWEEN_TRANSCRIPTS = 1.5   # pause between each transcript fetch to avoid blocking
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.5


# ----------------------------
# Targets
# ----------------------------

CHANNELS: Dict[str, Dict[str, str]] = {
    "joel_osteen": {
        "label": "Joel Osteen",
        "channel_id": "UCvxWyn4rfcI2H9APhfUIB1Q",
        "handle": "@joelosteen",
    },
    "elevation": {
        "label": "Elevation Church",
        "channel_id": "",
        "handle": "@elevationchurch",
    },
    "creflo": {
        "label": "Creflo Dollar",
        "channel_id": "",
        "handle": "@CrefloDollar1",
    },
    "desiring_god": {
        "label": "Desiring God (control)",
        "channel_id": "",
        "handle": "@desiringGod",
    },
    "mclean_bible": {
        "label": "McLean Bible Church / David Platt (control)",
        "channel_id": "",
        "handle": "@mcleanbiblechurch5357",
    },
    "village_church": {
        "label": "The Village Church / Matt Chandler (control)",
        "channel_id": "",
        "handle": "@thevillagechurch",
    },
}


# ----------------------------
# Helpers
# ----------------------------

ISO8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)


def parse_iso8601_duration_to_seconds(duration: str) -> Optional[int]:
    if not duration:
        return None
    m = ISO8601_DURATION_RE.match(duration)
    if not m:
        return None
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def backoff_sleep(attempt: int) -> None:
    base = BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1))
    jitter = 0.2 * base
    time.sleep(base + (jitter * (0.5 - (time.time() % 1))))


def require_api_key() -> None:
    if not YT_API_KEY:
        raise RuntimeError(
            "Missing YT_API_KEY. Set environment variable YT_API_KEY before running."
        )


def youtube_client():
    require_api_key()
    return build("youtube", "v3", developerKey=YT_API_KEY)


# ----------------------------
# Channel resolution
# ----------------------------


def resolve_channel_id(youtube, handle: str) -> str:
    handle = handle.strip()
    if not handle:
        raise ValueError("handle is empty")

    req = youtube.channels().list(part="id", forHandle=handle)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = req.execute()
            time.sleep(SLEEP_BETWEEN_API_CALLS)
            items = resp.get("items", [])
            if items:
                return items[0]["id"]
            break
        except HttpError:
            if attempt >= MAX_RETRIES:
                raise
            backoff_sleep(attempt)

    req2 = youtube.search().list(part="snippet", q=handle, type="channel", maxResults=5)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp2 = req2.execute()
            time.sleep(SLEEP_BETWEEN_API_CALLS)
            items2 = resp2.get("items", [])
            if not items2:
                raise RuntimeError(f"No channel found for handle query: {handle}")
            return items2[0]["snippet"]["channelId"]
        except HttpError:
            if attempt >= MAX_RETRIES:
                raise
            backoff_sleep(attempt)

    raise RuntimeError(f"Unable to resolve handle to channelId: {handle}")


# ----------------------------
# Video discovery
# ----------------------------


def fetch_uploads_playlist_id(youtube, channel_id: str) -> str:
    req = youtube.channels().list(part="contentDetails", id=channel_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = req.execute()
            time.sleep(SLEEP_BETWEEN_API_CALLS)
            items = resp.get("items", [])
            if not items:
                raise RuntimeError(f"Channel not found: {channel_id}")
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except HttpError:
            if attempt >= MAX_RETRIES:
                raise
            backoff_sleep(attempt)


def iter_playlist_video_ids(youtube, playlist_id: str) -> Iterable[str]:
    page_token = None
    while True:
        req = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page_token,
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = req.execute()
                time.sleep(SLEEP_BETWEEN_API_CALLS)
                break
            except HttpError:
                if attempt >= MAX_RETRIES:
                    raise
                backoff_sleep(attempt)

        for it in resp.get("items", []):
            yield it["contentDetails"]["videoId"]

        page_token = resp.get("nextPageToken")
        if not page_token:
            return


def chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_videos_details(youtube, video_ids: List[str]) -> pd.DataFrame:
    rows: List[dict] = []

    for batch in tqdm(list(chunked(video_ids, 50)), desc="videos.details"):
        req = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch),
            maxResults=50,
        )
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = req.execute()
                time.sleep(SLEEP_BETWEEN_API_CALLS)
                break
            except HttpError:
                if attempt >= MAX_RETRIES:
                    raise
                backoff_sleep(attempt)

        for it in resp.get("items", []):
            snippet = it.get("snippet", {})
            stats = it.get("statistics", {})
            cd = it.get("contentDetails", {})
            duration_iso = cd.get("duration")
            duration_s = parse_iso8601_duration_to_seconds(duration_iso) if duration_iso else None

            rows.append({
                "video_id": it.get("id"),
                "title": snippet.get("title"),
                "published_at": snippet.get("publishedAt"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "duration_iso": duration_iso,
                "duration_seconds": duration_s,
                "view_count": int(stats["viewCount"]) if stats.get("viewCount") else None,
                "like_count": int(stats["likeCount"]) if stats.get("likeCount") else None,
                "comment_count": int(stats["commentCount"]) if stats.get("commentCount") else None,
                "url": f"https://www.youtube.com/watch?v={it.get('id')}",
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df


def select_top_viewed_longform(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df2 = df[df["duration_seconds"].fillna(0) >= MIN_DURATION_SECONDS].copy()
    df2 = df2[df2["view_count"].notna()]
    df2 = df2.sort_values("view_count", ascending=False)
    return df2.head(top_n).reset_index(drop=True)


# ----------------------------
# Transcript fetching  (v1.x API)
# ----------------------------


def _make_api() -> YouTubeTranscriptApi:
    """Instantiate YouTubeTranscriptApi with the best available proxy config."""
    if YT_PROXY_USER and YT_PROXY_PASS and WebshareProxyConfig is not None:
        return YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=YT_PROXY_USER,
                proxy_password=YT_PROXY_PASS,
            )
        )
    if YT_PROXY_URL and GenericProxyConfig is not None:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=YT_PROXY_URL,
                https_url=YT_PROXY_URL,
            )
        )
    return YouTubeTranscriptApi()


def _fetch_transcript_ytdlp(video_id: str, preferred_langs: List[str] = PREFERRED_LANGS) -> Tuple[str, Optional[str]]:
    """Fallback transcript fetcher using yt-dlp.

    yt-dlp mimics a real browser far more convincingly than youtube-transcript-api
    and bypasses most IP-based blocks without needing a proxy.

    Requires yt-dlp to be installed:
        pip install yt-dlp

    Downloads only the subtitle file (no video), parses it, and cleans up.
    """
    # Prefer yt-dlp command; fall back to python -m yt_dlp (works when venv has it)
    ytdlp_cmd = shutil.which("yt-dlp") or (sys.executable, "-m", "yt_dlp")
    if isinstance(ytdlp_cmd, str):
        ytdlp_cmd = [ytdlp_cmd]
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("\n  [yt-dlp missing] Run: pip install yt-dlp")
        return "error:yt-dlp-not-installed", None

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, "%(id)s.%(ext)s")
        path_env = os.environ.get("PATH", "") + ":/opt/homebrew/bin:/usr/local/bin"

        def run_ytdlp(write_auto: bool, sub_lang: Optional[str]) -> Optional[str]:
            """Run yt-dlp and return transcript text if found."""
            base = list(ytdlp_cmd) + [
                "--skip-download",
                "--write-auto-sub" if write_auto else "--write-sub",
                "--sub-format", "vtt",
                "--output", out_template,
                "--quiet",
                "--no-warnings",
                url,
            ]
            if sub_lang:
                cmd = base + ["--sub-lang", sub_lang]
            else:
                cmd = base + ["--all-subs"]
            for use_cookies in (True, False):
                full_cmd = cmd + (["--cookies-from-browser", "chrome"] if use_cookies else [])
                try:
                    subprocess.run(
                        full_cmd, check=True, timeout=60,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        env={**os.environ, "PATH": path_env},
                    )
                    vtt_files = list(pathlib.Path(tmpdir).glob("*.vtt"))
                    if vtt_files:
                        # Prefer English when multiple languages (e.g. id.en.vtt, id.en-nl.vtt)
                        en_pref = [f for f in vtt_files if any(
                            f.stem.endswith(f".{x}") for x in ("en", "en-US", "en-GB", "en-nl")
                        )]
                        chosen = (en_pref or vtt_files)[0]
                        raw = chosen.read_text(encoding="utf-8", errors="replace")
                        text = _parse_vtt(raw)
                        if text:
                            return text
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass
            return None

        langs = ",".join(preferred_langs)
        # 1) Manual captions, preferred langs  2) Auto-generated, preferred langs
        # 3) Any manual captions  4) Any auto-generated
        for write_auto, sub_lang in [(False, langs), (True, langs), (False, None), (True, None)]:
            text = run_ytdlp(write_auto, sub_lang)
            if text:
                return "ok", text

    return "not_found", None


def _parse_vtt(vtt: str) -> Optional[str]:
    """Extract plain text from a WebVTT subtitle file, deduplicating overlap lines."""
    # Strip header and cue metadata; keep only text lines
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        # Skip header, timestamps, cue IDs, and empty lines
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}", line) or "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        # Strip VTT inline tags like <00:00:00.000> and <c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            lines.append(line)

    # Deduplicate consecutive identical or contained lines (VTT often repeats partials)
    deduped = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    text = " ".join(deduped)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def fetch_transcript(video_id: str, preferred_langs: List[str] = PREFERRED_LANGS) -> Tuple[str, Optional[str]]:
    """Return (status, transcript_text).

    Strategy:
    1. Try youtube-transcript-api (fast, no subprocess).
    2. If blocked, fall back to yt-dlp which mimics a real browser
       and bypasses most IP blocks without needing a proxy.
    """
    api = _make_api()
    time.sleep(SLEEP_BETWEEN_TRANSCRIPTS)

    def _to_text(fetched) -> str:
        parts = [getattr(s, "text", "") or "" for s in fetched]
        return re.sub(r"\s+", " ", " ".join(parts).strip())

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            fetched = api.fetch(video_id, languages=preferred_langs)
            text = _to_text(fetched)
            return "ok", text or None

        except TranscriptsDisabled:
            return "disabled", None

        except NoTranscriptFound:
            try:
                fetched = api.fetch(video_id)
                text = _to_text(fetched)
                return "ok", text or None
            except Exception:
                # API couldn't find transcript — try yt-dlp (often succeeds for
                # channels where the API fails, e.g. Elevation Church)
                return _fetch_transcript_ytdlp(video_id, preferred_langs)

        except VideoUnavailable:
            return "unavailable", None

        except RequestBlocked:
            # Primary method blocked — fall back to yt-dlp
            return _fetch_transcript_ytdlp(video_id, preferred_langs)

        except TooManyRequests:
            if attempt >= MAX_RETRIES:
                return "rate_limited", None
            backoff_sleep(attempt)

        except CouldNotRetrieveTranscript as e:
            if attempt >= MAX_RETRIES:
                return f"error:{type(e).__name__}", None
            backoff_sleep(attempt)

        except Exception as e:
            if attempt >= MAX_RETRIES:
                return f"error:{type(e).__name__}", None
            backoff_sleep(attempt)

    return "error", None


# ----------------------------
# Pipeline
# ----------------------------


def build_channel_corpus(youtube, slug: str, cfg: Dict[str, str], top_n: int = TOP_N_PER_CHANNEL) -> pd.DataFrame:
    label = cfg.get("label", slug)
    channel_id = (cfg.get("channel_id") or "").strip()
    handle = (cfg.get("handle") or "").strip()

    if not channel_id:
        if not handle:
            raise ValueError(f"Channel {slug} missing channel_id and handle")
        channel_id = resolve_channel_id(youtube, handle)

    uploads_playlist = fetch_uploads_playlist_id(youtube, channel_id)
    all_video_ids = list(iter_playlist_video_ids(youtube, uploads_playlist))

    details_df = fetch_videos_details(youtube, all_video_ids)
    top_df = select_top_viewed_longform(details_df, top_n)

    meta_path = RAW_DIR / f"videos_{slug}.csv"
    top_df.to_csv(meta_path, index=False)

    jsonl_path = RAW_DIR / f"transcripts_{slug}.jsonl"
    rows = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, r in tqdm(top_df.iterrows(), total=len(top_df), desc=f"transcripts:{slug}"):
            vid = r["video_id"]
            status, text = fetch_transcript(vid)
            rec = {
                "slug": slug,
                "label": label,
                "video_id": vid,
                "title": r["title"],
                "published_at": r["published_at"].isoformat() if pd.notna(r["published_at"]) else None,
                "view_count": r["view_count"],
                "like_count": r["like_count"],
                "comment_count": r["comment_count"],
                "duration_seconds": r["duration_seconds"],
                "url": r["url"],
                "transcript_status": status,
                "transcript": text,
                "word_count": len(text.split()) if text else 0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    return pd.DataFrame(rows)


def build_full_corpus(channels: Optional[List[str]] = None) -> pd.DataFrame:
    """Build corpus for all or specified channels.

    Args:
        channels: Channel slugs to build (e.g. ["elevation"]). If None, builds all.
    """
    youtube = youtube_client()

    to_build = {k: v for k, v in CHANNELS.items() if channels is None or k in channels}
    if not to_build:
        valid = ", ".join(CHANNELS.keys())
        raise SystemExit(f"No matching channels. Valid: {valid}")

    all_dfs = []
    for slug, cfg in to_build.items():
        print(f"\n=== Building channel: {slug} ===")
        try:
            df = build_channel_corpus(youtube, slug, cfg, top_n=TOP_N_PER_CHANNEL)
            all_dfs.append(df)
        except HttpError as e:
            if e.resp.status == 404:
                print(f"Warning: Channel {slug} — playlist/channel not found (404). Skipping.")
            else:
                raise

    out_path = PROCESSED_DIR / "corpus.csv"

    # Merge with existing corpus when rebuilding a subset
    if channels and out_path.exists():
        existing = pd.read_csv(out_path)
        if "slug" in existing.columns:
            existing = existing[~existing["slug"].isin(channels)]
            corpus = pd.concat([existing, *all_dfs], ignore_index=True)
        else:
            corpus = pd.concat(all_dfs, ignore_index=True)
    else:
        corpus = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    corpus.to_csv(out_path, index=False)
    print(f"\nWrote corpus to: {out_path}")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build YouTube prosperity sermon corpus")
    parser.add_argument(
        "-c", "--channel",
        action="append",
        dest="channels",
        metavar="SLUG",
        help="Channel slug to build (e.g. elevation). Can repeat. Default: all channels.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete all raw/processed data and rebuild from scratch.",
    )
    args = parser.parse_args()

    if args.fresh:
        for d in (RAW_DIR, PROCESSED_DIR):
            for f in d.iterdir():
                f.unlink()
        print("Cleared data/raw and data/processed")

    build_full_corpus(channels=args.channels)