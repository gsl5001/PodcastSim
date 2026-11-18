"""Episode identity.

Plan S3: "Episode ID: sha1(file bytes)[:12] plus a human slug from the filename.
Never key on filenames alone (duplicates and re-uploads are common)."

The hash is the identity; the slug is decoration so humans can read a directory
listing. Two uploads of the same bytes under different filenames collapse to the
same hash but keep whichever slug was seen first -- callers dedupe on `digest`,
never on `slug` or the full id.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Read in 1 MiB blocks: episodes are ~64 MB and we refuse to hold one in RAM
# just to hash it.
_CHUNK = 1 << 20

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(name: str, max_len: int = 40) -> str:
    """Filename -> lowercase hyphenated slug, safe as a path segment."""
    stem = Path(name).stem.lower()
    slug = _SLUG_STRIP.sub("-", stem).strip("-")
    return slug[:max_len].rstrip("-") or "untitled"


def _file_hash(path: str | Path, hasher) -> str:
    """Streamed hex digest of a file's bytes using the given hashlib object."""
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_sha1(path: str | Path) -> str:
    """Full hex sha1 of a file's bytes, streamed."""
    return _file_hash(path, hashlib.sha1())


def file_sha256(path: str | Path) -> str:
    """Full hex sha256 of a file's bytes, streamed.

    S0 dedupes on sha256 of *decoded PCM* (plan S5.2); this is the raw-file
    equivalent used for manifest input hashing.
    """
    return _file_hash(path, hashlib.sha256())


def episode_id(path: str | Path) -> str:
    """Stable episode id: `<sha1[:12]>__<slug>`.

    Hash first so the id sorts by identity and the prefix alone is enough to
    disambiguate; slug second so `ls data/work/` is readable.
    """
    path = Path(path)
    return f"{file_sha1(path)[:12]}__{slugify(path.name)}"


def digest_of(episode_id_str: str) -> str:
    """Extract the 12-char digest from an episode id."""
    return episode_id_str.split("__", 1)[0]
