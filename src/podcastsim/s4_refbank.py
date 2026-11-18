"""Reference-bank selection for zero-shot cloning (plan S9.2 step 1).

Zero-shot TTS copies timbre, pace and room tone from a short reference clip, so
the reference must be clean and representative. S9.2: prefer clips 6-15 s, high
quality, no boilerplate. We do not yet have DNSMOS/UTMOS (S3 v1), so we rank on
what we have: Whisper confidence (avg_logprob), a sane duration, and text that
looks like real speech (not one-word or repetitive).
"""

from __future__ import annotations

import json
from pathlib import Path


def score_clip(row: dict) -> float:
    """Higher is a better reference. Combines confidence + duration fit."""
    dur = row.get("duration", 0)
    # S9.2 wants 6-15 s; give a smooth preference peaking in that band.
    if dur < 3 or dur > 20:
        return -1e9
    dur_fit = 1.0 - abs(dur - 10.0) / 10.0        # 1.0 at 10 s, lower at edges
    logprob = row.get("avg_logprob", -5.0)         # ~ -0.1 (great) .. -1.0 (poor)
    nwords = len(row.get("text", "").split())
    if nwords < 4:
        return -1e9                                # too little content
    return logprob + 0.3 * dur_fit


def build_reference_bank(host_dir: str | Path, top_k: int = 12) -> list[dict]:
    """Pick the top_k reference clips for one host, best first.

    Reads ``<host_dir>/metadata.jsonl``, which is produced by
    ``s3_clips.aggregate_host_metadata`` (NOT written directly by the S3 stage,
    which emits per-episode ``clips.jsonl``). Run that aggregation first.
    """
    meta = Path(host_dir) / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} not found -- run s3_clips.aggregate_host_metadata(work, hosts) "
            "to build per-host metadata from the per-episode clips.jsonl files.")
    rows = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    scored = sorted(((score_clip(r), r) for r in rows), key=lambda x: x[0], reverse=True)
    return [r for s, r in scored if s > -1e8][:top_k]
