"""
Fetch top-level YouTube comments for a given video ID.
Saves to data/comments/<video_id>.jsonl — one JSON object per comment.

Usage:
    export YT_API_KEY='...'
    python fetch_comments.py GA6uE2CPo1I
"""

import json, os, sys, time
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

COMMENTS_DIR = Path("data/comments")
COMMENTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_COMMENTS = 500


def fetch_comments(video_id: str, max_comments: int = MAX_COMMENTS) -> list[dict]:
    api_key = os.environ.get("YT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set YT_API_KEY environment variable.")

    youtube = build("youtube", "v3", developerKey=api_key)

    comments = []
    next_page = None

    while len(comments) < max_comments:
        resp = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page,
            textFormat="plainText",
            order="relevance",
        ).execute()

        for item in resp.get("items", []):
            snip = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "author": snip.get("authorDisplayName", ""),
                "text": snip.get("textDisplay", ""),
                "likes": snip.get("likeCount", 0),
                "published": snip.get("publishedAt", ""),
                "updated": snip.get("updatedAt", ""),
            })

        next_page = resp.get("nextPageToken")
        if not next_page:
            break
        time.sleep(0.2)

    return comments


def save_comments(video_id: str, comments: list[dict]):
    out = COMMENTS_DIR / f"{video_id}.jsonl"
    with open(out, "w") as f:
        for c in comments:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(comments)} comments to {out}")


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "GA6uE2CPo1I"
    print(f"Fetching comments for {vid}...")
    comments = fetch_comments(vid)
    save_comments(vid, comments)
