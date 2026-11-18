"""Stage 1 - Speaker diarization (plan S6).

Goal: "per episode, who speaks when, including overlapped regions, with a
per-local-speaker embedding."

Outputs under `data/work/{ep}/s1_diarize/`:
    diarization.rttm   overlap-aware
    exclusive.rttm     one speaker at a time, for ASR alignment
    overlap.json       overlap regions, padded per S6.1
    embeddings.npz     per-local-speaker centroids + per-turn embeddings
    _DONE.json

Plan A is pyannote.audio 4 + `speaker-diarization-community-1`. GPU work, so
this runs on Colab (S2.2); the model is HF-gated, so it is snapshotted once and
loaded by path thereafter (S14.12).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import audio as au
from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest, StageGuard

STAGE = "s1_diarize"
PLAN_SECTION = "S6"

# Bump when output would change. History:
#   1  initial: pyannote community-1, music-masked, overlap padded
#   2  overlap_ratio reports TRUE overlap (padding is exclusion-only);
#      speech_ratio uses the union of turns, not the sum (it exceeded 1.0)
#   3  long episodes are chunked at silence and speaker-relinked (S6.1) so they
#      finish inside the free-tier reclamation window
CODE_VERSION = 3

DIAR_RTTM = "diarization.rttm"
EXCL_RTTM = "exclusive.rttm"
OVERLAP_JSON = "overlap.json"
EMBED_NPZ = "embeddings.npz"


def load_music_spans(work_root: Path, episode_id: str) -> list[dict]:
    """S0's spans, used to mask S1 output (plan S5.3).

    "diarization still runs on the whole file (it needs context), but its output
    is masked" -- so we diarize everything and drop turns inside music.
    """
    p = Path(work_root) / episode_id / "s0_ingest" / "segments_music.json"
    try:
        return json.load(open(p)).get("segments", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def music_intervals(spans: list[dict]) -> list[tuple[float, float]]:
    """Intervals that are music or noise -- i.e. not usable speech."""
    return [(s["start"], s["end"]) for s in spans
            if s.get("label") in ("music", "noise", "silence")]


def overlaps_any(start: float, end: float,
                 intervals: list[tuple[float, float]], min_frac: float = 0.5) -> bool:
    """True when >= min_frac of [start,end) falls inside any interval."""
    dur = max(end - start, 1e-9)
    covered = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
    return covered / dur >= min_frac


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total time covered by the union of intervals (overlaps counted once)."""
    if not intervals:
        return 0.0
    total = 0.0
    cur_a, cur_b = None, None
    for a, b in sorted(intervals):
        if cur_b is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    return total + (cur_b - cur_a if cur_b is not None else 0.0)


def write_rttm(path: Path, episode_id: str, turns: list[tuple[float, float, str]]) -> None:
    """NIST RTTM, the format pyannote.metrics scores (S12.1)."""
    with open(path, "w") as fh:
        for start, end, spk in sorted(turns):
            fh.write(
                f"SPEAKER {episode_id} 1 {start:.3f} {end - start:.3f} "
                f"<NA> <NA> {spk} <NA> <NA>\n"
            )


def compute_overlap(turns: list[tuple[float, float, str]],
                    pad_s: float) -> tuple[list[dict], float, float]:
    """Regions where >= 2 speakers are active.

    Returns (padded_regions, padded_seconds, true_seconds). S6.1 pads by 0.25 s
    each side so the *exclusion* zone is generous, but the reported
    `overlap_ratio` must use the UNPADDED total: gate S1-A's 2-10% band is about
    how much genuine crosstalk there is, and padding every region by 0.5 s
    inflates that badly when turns are short (this episode has 1004 turns in
    3040 s, i.e. one every ~3 s).
    """
    events: list[tuple[float, int]] = []
    for start, end, _ in turns:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    raw: list[tuple[float, float]] = []
    active = 0
    region_start: float | None = None
    for t, delta in events:
        was = active
        active += delta
        if was < 2 <= active:
            region_start = t
        elif was >= 2 > active and region_start is not None:
            raw.append((region_start, t))
            region_start = None

    true_total = sum(b - a for a, b in raw)

    merged: list[dict] = []
    for a, b in raw:
        r = {"start": max(0.0, a - pad_s), "end": b + pad_s}
        if merged and r["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append(r)
    padded_total = sum(r["end"] - r["start"] for r in merged)
    return merged, padded_total, true_total


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    work = Path(paths["work"])
    s0_manifest = work / episode.episode_id / "s0_ingest" / "_DONE.json"
    if not s0_manifest.exists():
        raise RuntimeError(f"{STAGE} requires s0_ingest first for {episode.source}")
    s0_hash = json.load(open(s0_manifest))["params_hash"]

    guard = StageGuard(
        work, episode.episode_id, STAGE,
        # Cite S0's params_hash: if S0 is redone, S1 must be too.
        inputs=[f"s0:{episode.episode_id}:{s0_hash}"],
        params=cfg.stage_params(STAGE),
        versions=_versions(),
        code_version=CODE_VERSION,
    )
    if not force and guard.should_skip():
        return None

    with guard:
        # S0 keeps the analysis copy in scratch; regenerate if it was dropped.
        audio = au.ensure_analysis_copy(
            Path(paths["raw"]) / episode.source,
            Path(paths.get("cache", "data/cache")) / episode.episode_id,
            codec=str(cfg.get("s0_ingest.analysis_codec", "wav")),
        )
        guard.metrics["analysis_format"] = audio.suffix.lstrip(".")

        turns, embeddings, backend_stats = _diarize(audio, cfg, paths)

        music = music_intervals(load_music_spans(work, episode.episode_id))
        kept = [t for t in turns if not overlaps_any(t[0], t[1], music)]
        guard.metrics["turns_dropped_to_music"] = len(turns) - len(kept)

        out = guard.dir
        write_rttm(out / DIAR_RTTM, episode.episode_id, kept)

        pad = float(cfg.get("s1_diarize.overlap_pad_s", 0.25))
        regions, padded_overlap_s, true_overlap_s = compute_overlap(kept, pad)
        (out / OVERLAP_JSON).write_text(json.dumps({"pad_s": pad, "regions": regions}, indent=2))

        exclusive = [t for t in kept
                     if not overlaps_any(t[0], t[1],
                                         [(r["start"], r["end"]) for r in regions], 0.01)]
        write_rttm(out / EXCL_RTTM, episode.episode_id, exclusive)

        if embeddings:
            np.savez_compressed(out / EMBED_NPZ, **embeddings)

        duration = json.load(open(work / episode.episode_id / "s0_ingest" / "meta.json"))["duration_s"]
        speakers = sorted({t[2] for t in kept})
        speaking = {s: sum(e - b for b, e, k in kept if k == s) for s in speakers}
        dominant = [s for s, t in speaking.items()
                    if t / duration >= float(cfg.get("s1_diarize.gate.dominant_speaker_min_share", 0.05))]

        # Union of speaking time, NOT the sum: two people talking at once is
        # one second of speech, not two. Summing per-speaker time gave a
        # "speech_ratio" of 1.0429, which is impossible.
        covered = _union_seconds([(b, e) for b, e, _ in kept])

        guard.metrics.update({
            "n_speakers": len(speakers),
            "n_dominant_speakers": len(dominant),
            "n_turns": len(kept),
            "n_exclusive_turns": len(exclusive),
            # Gate S1-A scores TRUE overlap; padded is the exclusion zone.
            "overlap_ratio": round(true_overlap_s / duration, 4) if duration else 0.0,
            "overlap_ratio_padded": round(padded_overlap_s / duration, 4) if duration else 0.0,
            "chunked": backend_stats.get("chunked", False),
            "n_chunks": backend_stats.get("n_chunks", 1),
            "speech_ratio": round(covered / duration, 4) if duration else 0.0,
            "speaker_time_ratio": round(sum(speaking.values()) / duration, 4) if duration else 0.0,
            **backend_stats,
        })

    return guard.existing


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import torch

        out["torch"] = torch.__version__
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        out["gpu"] = "cpu"
    try:
        import pyannote.audio

        out["pyannote.audio"] = pyannote.audio.__version__
    except ImportError:
        pass
    return out


def _diarize(audio16k: Path, cfg: Config, paths: dict):
    """Dispatch to the configured backend. Returns (turns, embeddings, stats)."""
    backend = cfg.get("s1_diarize.backend", "pyannote")
    if backend == "pyannote":
        from ..diarize_pyannote import diarize_pyannote

        return diarize_pyannote(audio16k, cfg, paths)
    raise NotImplementedError(
        f"diarization backend {backend!r} not implemented (plan {PLAN_SECTION})"
    )
