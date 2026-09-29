from __future__ import annotations
import json
import subprocess
from pathlib import Path
from music_transcribe.schema import Tags

_KEYS = {"title": ["TITLE", "title"], "artist": ["ARTIST", "artist", "album_artist"], "album": ["ALBUM", "album"],
         "embedded_lyrics": ["LYRICS", "lyrics", "UNSYNCEDLYRICS", "unsyncedlyrics"]}


def read_tags(path: Path, runner=subprocess.run) -> Tags:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    r = runner(cmd, check=False, capture_output=True, text=True)
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags", {}) or {}
    except (json.JSONDecodeError, AttributeError):
        tags = {}
    out = {}
    for field, keys in _KEYS.items():
        out[field] = next((str(tags[k]) for k in keys if k in tags), "")
    return Tags(**out)
