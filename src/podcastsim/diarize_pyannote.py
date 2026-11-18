"""pyannote.audio 4 backend for S1 (plan S6.1, Plan A).

The pipeline is HF-gated. Plan S14.12: "HF-gated != online-only. Snapshot gated
models (pyannote, CrisperWhisper) to Drive once; load by path; keep tokens out
of the pipeline and out of git." So: snapshot to `models/cache/<repo>` on first
use with a token read from disk, then load by path forever after.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

REPO = "pyannote/speaker-diarization-community-1"
TOKEN_PATHS = (
    Path.home() / ".config" / "podcastsim" / "hf_token",
    Path("/content/hf_token"),
)


def read_token() -> str | None:
    """Token from the environment or a 0600 file outside the repo."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if v := os.environ.get(var):
            return v.strip()
    for p in TOKEN_PATHS:
        try:
            if v := p.read_text().strip():
                return v
        except OSError:
            continue
    return None


def ensure_snapshot(models_root: Path, repo: str = REPO) -> Path:
    """Return a local path to the pipeline, downloading once if needed."""
    dest = Path(models_root) / "cache" / repo.replace("/", "__")
    if (dest / "config.yaml").exists():
        return dest

    token = read_token()
    if token is None:
        raise RuntimeError(
            f"{repo} is HF-gated and no token was found. Accept the conditions at "
            f"https://huggingface.co/{repo} then write a read token to "
            f"~/.config/podcastsim/hf_token (or set HF_TOKEN). Plan S14.12."
        )
    from huggingface_hub import snapshot_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dest), token=token)
    return dest


def _run_pipeline(pipe, path, cfg):
    """Diarize one file; return (turns, {label: embedding vector})."""
    out = pipe(
        str(path),
        min_speakers=int(cfg.get("s1_diarize.min_speakers", 2)),
        max_speakers=int(cfg.get("s1_diarize.max_speakers", 6)),
    )
    diar = out.speaker_diarization
    turns = [(float(seg.start), float(seg.end), str(label))
             for seg, _, label in diar.itertracks(yield_label=True)]
    labels = list(diar.labels())
    emb = getattr(out, "speaker_embeddings", None)
    vecs: dict[str, np.ndarray] = {}
    if emb is not None:
        arr = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)
        for i, lbl in enumerate(labels):
            if i < len(arr):
                vecs[str(lbl)] = np.asarray(arr[i], dtype=np.float32)
    return turns, vecs


def diarize_pyannote(audio16k: Path, cfg, paths: dict
                     ) -> tuple[list[tuple[float, float, str]], dict[str, np.ndarray], dict[str, Any]]:
    """Diarize one analysis-WAV with pyannote (the S1 backend).

    Returns ``(turns, embeddings, stats)`` where turns is
    ``[(start_s, end_s, speaker_label), ...]``, embeddings is a dict ready for
    ``np.savez`` (keys ``labels`` + ``centroids``, or empty if unavailable), and
    stats carries device/backend/chunking info. Episodes longer than
    ``s1_diarize.chunk.over_duration_s`` are split at silence, diarized per
    chunk, and relinked by embedding similarity.
    """
    import torch
    from pyannote.audio import Pipeline

    from . import audio as au
    from .chunking import cut_chunk, find_silences, plan_chunks, relink_speakers

    models_root = Path(paths.get("models", "models"))
    local = ensure_snapshot(models_root, cfg.get("s1_diarize.model_repo", REPO))
    pipe = Pipeline.from_pretrained(str(local))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(torch.device(device))

    duration = au.probe(audio16k).duration_s
    chunk_over = float(cfg.get("s1_diarize.chunk.over_duration_s", 3600))

    if duration <= chunk_over:
        turns, vecs = _run_pipeline(pipe, audio16k, cfg)
        embeddings: dict[str, np.ndarray] = {}
        if vecs:
            embeddings["labels"] = np.array(list(vecs.keys()), dtype="U64")
            embeddings["centroids"] = np.stack(list(vecs.values()))
        return turns, embeddings, {"device": device, "backend": "pyannote", "chunked": False}

    # Long episode: split at silence, diarize each piece, relink by embedding.
    target = float(cfg.get("s1_diarize.chunk.target_s", 1200))
    hard = float(cfg.get("s1_diarize.chunk.max_s", 1500))
    link = float(cfg.get("s1_diarize.chunk.relink_cosine", 0.7))
    silences = find_silences(str(audio16k),
                             float(cfg.get("s1_diarize.long_episode.min_split_silence_s", 2.0)))
    chunks = plan_chunks(duration, silences, target, hard)

    chunk_results = []
    for ch in chunks:
        piece = Path(audio16k).with_name(f"_chunk{ch.index}.wav")
        cut_chunk(str(audio16k), ch, str(piece))
        try:
            t, v = _run_pipeline(pipe, piece, cfg)
        finally:
            piece.unlink(missing_ok=True)
        chunk_results.append({"index": ch.index, "offset": ch.start_s,
                              "turns": t, "embeddings": v})

    turns, link_stats = relink_speakers(chunk_results, link)
    # Persist the relinked global centroids (not {}), so long episodes write
    # embeddings.npz like short ones -- S2 identity assignment depends on it.
    embeddings = {}
    centroids = link_stats.get("global_centroids", {})
    if centroids:
        embeddings["labels"] = np.array(list(centroids.keys()), dtype="U64")
        embeddings["centroids"] = np.stack(list(centroids.values()))
    return turns, embeddings, {"device": device, "backend": "pyannote",
                               "chunked": True, "n_chunks": len(chunks),
                               "n_speakers_relinked": link_stats["n_global_speakers"]}
