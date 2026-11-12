"""Evidence images on the Cloud (P1, 10/11): event snapshots and plate crops as JPEG files, paths kept in the database.

Plain files under V4_MEDIA_DIR (default ~/v4_media)/<kind>/; no imports beyond the standard library, so both the results
writer (qnn env) and the plate OCR worker (EasyOCR env) can use it.
prune(): files older than V4_MEDIA_KEEP_DAYS (30, like the plates / crowd rows) are deleted (the results writer runs it
hourly, 12/11).
"""
from __future__ import annotations

import base64
import os
import re
import time

MEDIA_DIR = os.path.expanduser(os.getenv("V4_MEDIA_DIR", "~/v4_media"))
MEDIA_KEEP_DAYS = float(os.getenv("V4_MEDIA_KEEP_DAYS", "30"))


def prune(media_dir: str = MEDIA_DIR, keep_days: float = MEDIA_KEEP_DAYS, now: float | None = None) -> int:
    """Deletes the JPEGs under media_dir older than keep_days -> how many."""
    cutoff, removed = (time.time() if now is None else now) - keep_days * 86400, 0
    for folder, _, files in os.walk(media_dir):
        for name in files:
            path = os.path.join(folder, name)
            try:
                if name.endswith(".jpg") and os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                pass   # gone meanwhile
    return removed


def save_media(kind: str, name: str, jpeg_b64: str | None, media_dir: str = MEDIA_DIR) -> str | None:
    """Writes one JPEG under media_dir/kind/; -> its path, or None. name: any id, made safe for a file name."""
    if not jpeg_b64:
        return None
    folder = os.path.join(media_dir, kind)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:200] + ".jpg")
    with open(path, "wb") as f:
        f.write(base64.b64decode(jpeg_b64))
    return path
