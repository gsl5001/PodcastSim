"""Split a long episode for diarization, then stitch the pieces back together.

Plan S6.1: "if a 3 h file OOMs or takes > 20 min, split at silences >= 2 s into
<= 60 min chunks and re-link chunk speakers by embedding cosine (> 0.7 -> same
local speaker)."

Here the motive is different but the mechanism is the same: free-tier Colab
reclaims the VM within ~15-20 min (S2.1), and a 78-minute episode's diarization
(~10 min) keeps losing that race. Splitting into ~20-minute pieces makes each
chunk diarize in ~2-3 min, comfortably inside the window.

Two things must be right:
  1. Split only at silence, so a speaker turn is never cut mid-word.
  2. Speaker labels are per-chunk (SPEAKER_00 in chunk 1 is unrelated to
     SPEAKER_00 in chunk 2), so they must be re-linked by voice embedding before
     the turns are merged.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import numpy as np

from . import audio as au


@dataclass
class Chunk:
    index: int
    start_s: float      # offset of this chunk within the full episode
    end_s: float


def find_silences(path: str, min_silence_s: float = 2.0,
                  noise_db: float = -30.0) -> list[tuple[float, float]]:
    """Silence intervals via ffmpeg's silencedetect filter."""
    cmd = [au.FFMPEG, "-v", "info", "-i", str(path), "-af",
           f"silencedetect=noise={noise_db}dB:d={min_silence_s}", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def plan_chunks(duration_s: float, silences: list[tuple[float, float]],
                target_s: float = 1200.0, max_s: float = 1500.0) -> list[Chunk]:
    """Choose cut points near each target boundary, landing in a silence.

    Cuts at the midpoint of whichever silence is closest to the target boundary
    and still leaves the chunk under `max_s`. If no silence is near (a
    relentless episode with no gaps), it cuts at the hard `target_s` -- worse,
    but bounded.
    """
    if duration_s <= max_s:
        return [Chunk(0, 0.0, duration_s)]

    mids = [( (a + b) / 2.0) for a, b in silences]
    chunks: list[Chunk] = []
    start = 0.0
    idx = 0
    while duration_s - start > max_s:
        target = start + target_s
        candidates = [m for m in mids if start + 60 < m < start + max_s]
        cut = min(candidates, key=lambda m: abs(m - target)) if candidates else start + target_s
        chunks.append(Chunk(idx, start, cut))
        start = cut
        idx += 1
    chunks.append(Chunk(idx, start, duration_s))
    return chunks


def cut_chunk(src_audio: str, chunk: Chunk, dst: str) -> str:
    """Extract one chunk to its own analysis file (same 16 kHz mono WAV)."""
    au._run([
        au.FFMPEG, "-v", "error", "-y", "-ss", str(chunk.start_s),
        "-t", str(chunk.end_s - chunk.start_s), "-i", str(src_audio),
        "-ac", "1", "-ar", str(au.ANALYSIS_SR), "-c:a", "pcm_s16le", str(dst),
    ])
    return dst


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def relink_speakers(chunk_results: list[dict], link_threshold: float = 0.7
                    ) -> tuple[list[tuple[float, float, str]], dict]:
    """Merge per-chunk diarization into one timeline with global speaker labels.

    Each element of `chunk_results` is:
        {"offset": float,
         "turns": [(start, end, local_label), ...],   # chunk-local times/labels
         "embeddings": {local_label: vector}}

    Greedy linking: a chunk speaker joins the existing global speaker whose
    centroid is closest above `link_threshold`, else starts a new one. Good
    enough for three recurring hosts; S2 re-checks identities globally anyway.
    """
    global_centroids: dict[str, np.ndarray] = {}
    global_counts: dict[str, int] = {}
    merged: list[tuple[float, float, str]] = []
    next_id = 0
    links: list[dict] = []

    for res in chunk_results:
        offset = res["offset"]
        emb = res.get("embeddings", {})
        local_to_global: dict[str, str] = {}

        for local, vec in emb.items():
            vec = np.asarray(vec, dtype=np.float32)
            best, best_sim = None, link_threshold
            for gid, centroid in global_centroids.items():
                sim = _cosine(vec, centroid)
                if sim >= best_sim:
                    best, best_sim = gid, sim
            if best is None:
                best = f"SPEAKER_{next_id:02d}"
                next_id += 1
                global_centroids[best] = vec.copy()
                global_counts[best] = 1
            else:
                n = global_counts[best]
                global_centroids[best] = (global_centroids[best] * n + vec) / (n + 1)
                global_counts[best] = n + 1
            local_to_global[local] = best
            links.append({"chunk": res.get("index"), "local": local,
                          "global": best, "sim": round(best_sim, 3)})

        for start, end, local in res["turns"]:
            gid = local_to_global.get(local, local)
            merged.append((start + offset, end + offset, gid))

    merged.sort()
    # Surface the global centroids so the caller can persist embeddings.npz for
    # long (chunked) episodes too -- S2 identity assignment depends on them.
    return merged, {"n_global_speakers": len(global_centroids), "links": links,
                    "global_centroids": global_centroids}
