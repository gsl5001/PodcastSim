"""Stage 3 - Per-host training datasets (plan S8).

v1 scope: select clean host turns, cut at native rate -> 24 kHz, re-segment with
Silero VAD into 3-20 s clips, transcribe (faster-whisper or CrisperWhisper),
apply the cheap sanity filters (S8.6: avg_logprob, no_speech_prob,
compression_ratio, language, duration), loudness-normalise, write the clips plus
one `clips.jsonl` PER EPISODE into this stage's work dir (per-episode so a
resumed run never corrupts a shared file). The per-host `metadata.jsonl` that S4
consumes is assembled afterwards from all episodes' clips.jsonl by
`s3_clips.aggregate_host_metadata` (run it once S3 has finished all episodes).

Deferred to v2 (need more model installs): DNSMOS/UTMOS/WADA-SNR, AST tagging,
Qwen3 forced alignment, purity recompute, MinHash dedup. Structured so they slot
into the per-clip metadata later.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest, StageGuard

STAGE = "s3_dataset"
PLAN_SECTION = "S8"

#   1  v1: select + cut 24k + Silero VAD + faster-whisper + sanity filters
#   2  clipping check after peak-normalise (MP3 overshoot != clipping)
#   3  CrisperWhisper verbatim transcripts; per-episode metadata (resumable)
CODE_VERSION = 3


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    work = Path(paths["work"])
    for need in ("s1_diarize", "s2_identify"):
        if not (work / episode.episode_id / need / "_DONE.json").exists():
            raise RuntimeError(f"{STAGE} requires {need} first for {episode.source}")

    host_map_path = work / episode.episode_id / "s2_identify" / "host_map.json"
    if not host_map_path.exists():
        raise RuntimeError(f"{STAGE} needs host_map.json -- run `podcastsim identify` "
                           f"after s2 for {episode.source}")

    s2_hash = json.load(open(work / episode.episode_id / "s2_identify" / "_DONE.json"))["params_hash"]
    guard = StageGuard(
        work, episode.episode_id, STAGE,
        inputs=[f"s2:{episode.episode_id}:{s2_hash}",
                f"hostmap:{host_map_path.read_text()}"],
        params=cfg.stage_params(STAGE),
        versions=_versions(),
        code_version=CODE_VERSION,
    )
    if not force and guard.should_skip():
        return None

    with guard:
        from .. import audio as au
        from ..rttm import read_rttm
        from ..s3_select import host_hours, select_host_turns
        from ..s3_clips import cut_and_transcribe

        host_map = json.loads(host_map_path.read_text())
        s1 = work / episode.episode_id / "s1_diarize"
        turns = read_rttm(s1 / "exclusive.rttm")

        overlap = json.load(open(s1 / "overlap.json")).get("regions", [])
        overlap_regions = [(r["start"], r["end"]) for r in overlap]

        seg = json.load(open(work / episode.episode_id / "s0_ingest" / "segments_music.json"))
        exclude = [(s["start"], s["end"]) for s in seg.get("segments", [])
                   if s.get("label") in ("music", "speech_over_music", "noise")]

        candidates = select_host_turns(
            turns, host_map, overlap_regions, exclude,
            min_duration_s=float(cfg.get("s3_dataset.clip_duration_s", [1.0, 20.0])[0]),
            overlap_pad_s=float(cfg.get("s1_diarize.overlap_pad_s", 0.25)),
        )

        src_mp3 = Path(paths["raw"]) / episode.source
        hosts_root = Path(paths["hosts"])
        stats = cut_and_transcribe(src_mp3, candidates, hosts_root, episode.episode_id, cfg)
        # per-episode metadata (resumable; a shared append corrupts on resume)
        (guard.dir / "clips.jsonl").write_text(
            "\n".join(json.dumps(r) for r in stats.pop("rows", [])))

        guard.metrics.update({
            "n_candidate_turns": len(candidates),
            "candidate_hours": {k: round(v, 3) for k, v in host_hours(candidates).items()},
            "clips_written": stats["clips_written"],
            "clips_rejected": stats["clips_rejected"],
            "retained_seconds": {k: round(v, 1) for k, v in stats["retained_seconds"].items()},
            "reject_reasons": stats["reject_reasons"],
        })

    return guard.existing


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        out["gpu"] = "cpu"
    for pkg in ("faster_whisper",):
        try:
            import importlib.metadata as md
            out[pkg] = md.version(pkg)
        except Exception:
            pass
    return out
