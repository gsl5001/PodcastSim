"""Speaker embeddings for S2 (plan S7.1).

The plan's first choice is WeSpeaker ResNet293-LM. Its `wespeaker` package does
not pip-install on the Colab image (S2.6: do not fight the audio ML ecosystem),
so we use pyannote's own embedding model instead: `wespeaker-voxceleb-resnet34-LM`
-- same WeSpeaker family, already available because pyannote is installed, and
more than adequate for three distinct recurring hosts (S7.1: "Either is more
than adequate for three known voices"). It is 256-dim.

S14.5: cosine thresholds are model-specific -- the resnet34-LM thresholds are
NOT the resnet293 numbers in the plan and must be calibrated on our own data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def load_embedder(hf_token: str | None, device: str | None = None,
                  model_id: str = DEFAULT_MODEL):
    import torch
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(model_id, use_auth_token=hf_token)
    inf = Inference(model, window="whole")
    inf.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    return inf


def embed_segment(inf, wav_path: str | Path, start: float, end: float) -> np.ndarray:
    """L2-normalised embedding of one time span."""
    from pyannote.core import Segment

    vec = np.asarray(inf.crop(str(wav_path), Segment(start, end)), dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n else vec


def speaker_centroid(inf, wav_path: str | Path, turns, *,
                     min_turn_s: float = 3.0, max_total_s: float = 60.0
                     ) -> tuple[np.ndarray | None, float, int]:
    """Average embedding of a speaker's longest clean turns (plan S7.4).

    "embed its longest clean turns (>= 3 s, no overlap, up to 60 s total) and
    average". Returns (centroid or None, seconds embedded, turns used).
    """
    chosen = sorted((t for t in turns if t.duration >= min_turn_s),
                    key=lambda t: t.duration, reverse=True)
    vecs, used_s = [], 0.0
    for t in chosen:
        if used_s >= max_total_s:
            break
        vecs.append(embed_segment(inf, wav_path, t.start, t.end))
        used_s += t.duration
    if not vecs:
        return None, 0.0, 0
    centroid = np.mean(vecs, axis=0)
    n = np.linalg.norm(centroid)
    return (centroid / n if n else centroid), used_s, len(vecs)
