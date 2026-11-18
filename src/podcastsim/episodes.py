"""Episode discovery and the on-disk index.

Plan S14.13: "Drive is slow for many small files; ... avoid `os.listdir` on huge
Drive folders (maintain an index file)."

So we scan `data/raw` once, write `data/work/_index.json`, and every later run
reads the index instead of stat-ing thousands of Drive entries. `refresh()` is
incremental: files already in the index keep their id without being re-hashed,
which matters because hashing 200 h of MP3 off Drive is minutes of pure I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .ids import digest_of, episode_id

INDEX_NAME = "_index.json"
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".flac"}
DEFAULT_EXCLUDE = "configs/exclude.yaml"


@dataclass(frozen=True)
class Episode:
    episode_id: str
    source: str      # path relative to the raw root
    size: int
    mtime_ns: int

    @property
    def digest(self) -> str:
        return digest_of(self.episode_id)


def index_path(work_root: str | Path) -> Path:
    return Path(work_root) / INDEX_NAME


def load_index(work_root: str | Path) -> dict[str, Episode]:
    """Return {source -> Episode}; empty if no index yet."""
    try:
        with open(index_path(work_root)) as fh:
            raw = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: Episode(**v) for k, v in raw.get("episodes", {}).items()}


def save_index(work_root: str | Path, episodes: dict[str, Episode]) -> Path:
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    path = index_path(root)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(
            {"episodes": {k: asdict(v) for k, v in sorted(episodes.items())}},
            fh,
            indent=2,
        )
    os.replace(tmp, path)
    return path


def refresh(raw_root: str | Path, work_root: str | Path) -> dict[str, Episode]:
    """Scan `raw_root` and update the index incrementally.

    A file is re-hashed only when its size or mtime changed, so the common case
    (nothing new uploaded) costs one directory walk and no reads.
    """
    raw_root = Path(raw_root)
    known = load_index(work_root)
    found: dict[str, Episode] = {}

    for path in sorted(raw_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rel = str(path.relative_to(raw_root))
        st = path.stat()
        prev = known.get(rel)
        if prev and prev.size == st.st_size and prev.mtime_ns == st.st_mtime_ns:
            found[rel] = prev            # unchanged: trust the cached id
        else:
            found[rel] = Episode(
                episode_id=episode_id(path),
                source=rel,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )
    save_index(work_root, found)
    return found


def duplicates(episodes: dict[str, Episode]) -> dict[str, list[str]]:
    """{digest -> [source, ...]} for digests appearing more than once.

    Plan S5.2 dedupes on decoded PCM; this catches byte-identical re-uploads
    for free, before anything is decoded.
    """
    by_digest: dict[str, list[str]] = {}
    for ep in episodes.values():
        by_digest.setdefault(ep.digest, []).append(ep.source)
    return {d: sorted(s) for d, s in by_digest.items() if len(s) > 1}


def load_exclusions(path: str | Path = DEFAULT_EXCLUDE) -> dict[str, str]:
    """Return {match_string: reason} from configs/exclude.yaml.

    `match_string` is a source filename or an episode-id digest prefix, exactly
    as written in the file. Missing file means nothing is excluded.
    """
    import yaml

    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    out: dict[str, str] = {}
    for entry in data.get("exclude") or []:
        if isinstance(entry, dict) and entry.get("match"):
            out[str(entry["match"])] = str(entry.get("reason", "no reason given"))
    return out


def is_excluded(ep: Episode, exclusions: dict[str, str]) -> str | None:
    """Return the exclusion reason for an episode, or None if it is kept."""
    for match, reason in exclusions.items():
        if ep.source == match or ep.episode_id == match or ep.digest.startswith(match):
            return reason
    return None


def resolve(episodes: dict[str, Episode], selector: str) -> list[Episode]:
    """Resolve a selector to episodes: 'all', a source path, an id, or a digest prefix."""
    if selector == "all":
        return sorted(episodes.values(), key=lambda e: e.source)
    if selector in episodes:
        return [episodes[selector]]
    hits = [
        e
        for e in episodes.values()
        if e.episode_id == selector
        or e.digest.startswith(selector)
        or Path(e.source).stem == selector          # "22" matches "22.mp3"/"22.m4a"
    ]
    return sorted(hits, key=lambda e: e.source)
