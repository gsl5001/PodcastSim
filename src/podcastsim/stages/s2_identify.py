"""Stage 2 - Host identification across episodes (plan S7).

This stage has two parts:

  1. PER EPISODE (here): embed each local speaker from their clean turns and
     save a voice fingerprint. Runs on Colab, pushes to Drive.

  2. CROSS EPISODE (podcastsim.identify, run once locally): pool all fingerprints
     and cluster them to find the recurring hosts.

Plan A (S7.1) assigns local speakers to named hosts using the owner's enrollment
clips. Those do not exist yet, so we do the per-episode embedding now and reach
the naming via Plan B (S7.2): cluster all local speakers, and the largest
recurring clusters are the hosts. That also turns the owner's blocking task from
"pick enrollment clips blind" into "confirm these three voices".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest, StageGuard
from ..rttm import read_rttm, turns_by_speaker

STAGE = "s2_identify"
PLAN_SECTION = "S7"

#   1  per-episode local-speaker embedding via wespeaker-resnet34-LM
CODE_VERSION = 1

EMBED_NPZ = "local_embeddings.npz"
LOCAL_JSON = "local_speakers.json"


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    work = Path(paths["work"])
    s1_manifest = work / episode.episode_id / "s1_diarize" / "_DONE.json"
    if not s1_manifest.exists():
        raise RuntimeError(f"{STAGE} requires s1_diarize first for {episode.source}")
    s1_hash = json.load(open(s1_manifest))["params_hash"]

    guard = StageGuard(
        work, episode.episode_id, STAGE,
        inputs=[f"s1:{episode.episode_id}:{s1_hash}"],
        params=cfg.stage_params(STAGE),
        versions=_versions(),
        code_version=CODE_VERSION,
    )
    if not force and guard.should_skip():
        return None

    with guard:
        from .. import audio as au
        from ..embed_speaker import load_embedder, speaker_centroid

        # Embed from the exclusive (single-speaker) turns, not the overlap-aware
        # ones -- S7.1 wants clean, non-overlapped speech per speaker.
        excl = work / episode.episode_id / "s1_diarize" / "exclusive.rttm"
        turns = read_rttm(excl)
        by_spk = turns_by_speaker(turns)

        audio = au.ensure_analysis_copy(
            Path(paths["raw"]) / episode.source,
            Path(paths.get("cache", "data/cache")) / episode.episode_id,
            codec=str(cfg.get("s0_ingest.analysis_codec", "wav")),
        )

        from ..diarize_pyannote import read_token
        inf = load_embedder(read_token(),
                            model_id=cfg.get("s2_identify.embedding_model_repo",
                                             "pyannote/wespeaker-voxceleb-resnet34-LM"))

        min_turn = float(cfg.get("s2_identify.assignment.min_turn_s", 3.0))
        max_total = float(cfg.get("s2_identify.assignment.max_turn_seconds_total", 60.0))

        centroids: dict[str, np.ndarray] = {}
        records = []
        for spk, spk_turns in by_spk.items():
            total_s = sum(t.duration for t in spk_turns)
            cen, used_s, n_used = speaker_centroid(
                inf, audio, spk_turns, min_turn_s=min_turn, max_total_s=max_total)
            records.append({
                "local_speaker": spk,
                "total_exclusive_s": round(total_s, 1),
                "embedded_s": round(used_s, 1),
                "turns_used": n_used,
                "has_embedding": cen is not None,
            })
            if cen is not None:
                centroids[spk] = cen

        out = guard.dir
        if centroids:
            np.savez_compressed(
                out / EMBED_NPZ,
                labels=np.array(list(centroids.keys()), dtype="U64"),
                centroids=np.stack(list(centroids.values())),
            )
        (out / LOCAL_JSON).write_text(json.dumps(
            {"episode_id": episode.episode_id,
             "publish_date": _episode_date(work, episode.episode_id),
             "speakers": records}, indent=2))

        embeddable = [r for r in records if r["has_embedding"]]
        guard.metrics.update({
            "n_local_speakers": len(records),
            "n_embedded": len(embeddable),
            "n_too_short": len(records) - len(embeddable),
            "min_embedded_s": round(min((r["embedded_s"] for r in embeddable), default=0.0), 1),
        })
        if len(embeddable) < len(records):
            guard.warn(f"{len(records) - len(embeddable)} local speaker(s) had "
                       f"no turn >= {min_turn}s and could not be embedded")

    return guard.existing


def _episode_date(work: Path, episode_id: str) -> str | None:
    try:
        return json.load(open(work / episode_id / "s0_ingest" / "meta.json")).get("publish_date")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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
