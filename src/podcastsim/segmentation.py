"""Speech / music / noise segmentation (plan S5.3).

The plan's Plan A is inaSpeechSegmenter, but it still requires
`tensorflow[and-cuda]` *and* `onnxruntime-gpu` (0.8.0, Sept 2025) -- roughly 3 GB
of CUDA extras and a dedicated venv (S2.6) for one coarse labelling pass.

We use **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`) instead, which S8.6
already requires for per-clip contamination filtering. Consequences:

- no new dependency, and one fewer venv;
- the same model and thresholds label spans in S0 and clips in S3, so there is
  no cross-model threshold mismatch (S14.5 warns about exactly this);
- AST is multi-label, so `speech_over_music` is simply a window scoring high on
  both Speech and Music -- no separate heuristic.

Backends are pluggable via `s0_ingest.segmentation.backend`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet indices, verified against the model's own config.json id2label.
CLS_SPEECH = 0
CLS_CONVERSATION = 4
CLS_LAUGHTER = 16
CLS_SINGING = 27
CLS_MUSIC = 137
CLS_INSTRUMENT = 138
CLS_SILENCE = 500
CLS_NOISE = 513

# Played media. Comedy podcasts play YouTube clips, movie audio and sound
# effects, which carry voices that are neither host nor guest. Diarization will
# happily cluster them as speakers, and a short clip can be absorbed into a
# host's cluster outright. These classes are already in the same AST forward
# pass, so flagging them costs nothing.
CLS_TELEVISION = 524
CLS_RADIO = 525
CLS_SOUND_EFFECT = 504
CLS_FIELD_RECORDING = 526
CLS_REVERBERATION = 511   # audio played through speakers into a mic
CLS_ECHO = 512
CLS_SPEECH_SYNTH = 7
CLS_NARRATION = 5
# The hosts are men, so these are strong "not a host" evidence.
CLS_FEMALE_SPEECH = 2
CLS_CHILD_SPEECH = 3

# A class needs real probability mass before it counts as evidence. AST is
# multi-label sigmoid and podcast speech sits around 0.36 on Sound effect, so
# anything below this is baseline, not a clip.
PLAYBACK_FLOOR = 0.50
# How far above the episode's own baseline a window must sit. Podcasts idle at
# different levels on the SFX classes (0.03 on one fixture, 0.36 on another).
PLAYBACK_MARGIN = 0.15

PLAYBACK_CLASSES = (CLS_TELEVISION, CLS_RADIO, CLS_SOUND_EFFECT,
                    CLS_FIELD_RECORDING, CLS_SPEECH_SYNTH)
NONHOST_VOICE_CLASSES = (CLS_FEMALE_SPEECH, CLS_CHILD_SPEECH)

# AST consumes 1024 frames of 128-bin log-mel at 16 kHz = 10.24 s.
AST_WINDOW_S = 10.24


@dataclass
class Span:
    start: float
    end: float
    label: str          # speech | music | speech_over_music | noise | silence
    p_speech: float
    p_music: float
    # Evidence that this span is played media rather than someone in the room.
    # Advisory: S2 still decides host vs guest from embeddings (S7.4/S7.6).
    p_playback: float = 0.0
    p_nonhost_voice: float = 0.0
    p_reverb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("start", "end", "p_speech", "p_music",
                  "p_playback", "p_nonhost_voice", "p_reverb"):
            d[k] = round(d[k], 3)
        return d


def label_window(p: np.ndarray, *, speech_th: float, music_th: float) -> str:
    """Map one window's AudioSet probabilities to a coarse label."""
    p_speech = float(max(p[CLS_SPEECH], p[CLS_CONVERSATION]))
    p_music = float(max(p[CLS_MUSIC], p[CLS_INSTRUMENT], p[CLS_SINGING]))
    speech, music = p_speech >= speech_th, p_music >= music_th
    if speech and music:
        return "speech_over_music"   # ad beds, intros (S5.3)
    if speech:
        return "speech"
    if music:
        return "music"
    if float(p[CLS_SILENCE]) >= max(speech_th, music_th):
        return "silence"
    return "noise"


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _playback_stats(spans: list["Span"], probs: np.ndarray | None = None,
                    hop_s: float = 5.0) -> dict[str, Any]:
    """Distribution of playback evidence.

    Statistics come from the per-WINDOW probabilities, not per-span. Spans
    aggregate p_playback by max (so a 5 s clip inside a longer span is not
    averaged away), but that makes span-level duration accounting nonsense: on
    45.mp3 the episode collapsed into two ~30-minute speech spans, one high
    window pushed each span's max over the floor, and 3780 s of 3804 got flagged
    while the median window sat at 0.03.
    """
    if not spans:
        return {}
    if probs is not None and len(probs):
        pb = [float(max(p[c] for c in PLAYBACK_CLASSES)) for p in probs]
        nh = [float(max(p[c] for c in NONHOST_VOICE_CLASSES)) for p in probs]
        rv = [float(max(p[CLS_REVERBERATION], p[CLS_ECHO])) for p in probs]
        unit = hop_s
    else:
        pb = [s.p_playback for s in spans]
        nh = [s.p_nonhost_voice for s in spans]
        rv = [s.p_reverb for s in spans]
        unit = None
    out = {
        "playback_p50": round(_pct(pb, 50), 3),
        "playback_p95": round(_pct(pb, 95), 3),
        "playback_max": round(max(pb), 3),
        "nonhost_voice_p95": round(_pct(nh, 95), 3),
        "nonhost_voice_max": round(max(nh), 3),
        "reverb_p95": round(_pct(rv, 95), 3),
    }
    # Outliers must be BOTH statistically unusual for this episode AND above an
    # absolute floor. MAD alone is not enough: on an episode with no clips the
    # distribution is nearly constant (p95 0.365, max 0.374), MAD collapses
    # toward zero, and every span becomes a "5-sigma outlier". The floor is what
    # encodes "a class needs real probability mass before we believe it".
    # A window counts as played media when it is BOTH above an absolute floor
    # (a class needs real probability mass) AND clearly above this episode's own
    # baseline (podcasts differ in how much the SFX classes idle at).
    #
    # A z-score was tried and abandoned: MAD is exactly 0 whenever most windows
    # share a baseline, which is the normal case -- 717 of 720 windows at 0.03
    # gave MAD 0.0 and suppressed a genuine clip entirely. Floor + margin has no
    # degenerate case.
    med = float(np.median(pb))
    out["playback_baseline"] = round(med, 3)
    threshold = max(PLAYBACK_FLOOR, med + PLAYBACK_MARGIN)
    out["playback_threshold"] = round(threshold, 3)
    if unit is not None:
        out["playback_outlier_seconds"] = round(
            unit * sum(1 for x in pb if x >= threshold), 1)
    else:
        out["playback_outlier_seconds"] = round(sum(
            s.end - s.start for s in spans if s.p_playback >= threshold), 1)
    return out


def playback_scores(p: np.ndarray) -> tuple[float, float, float]:
    """(p_playback, p_nonhost_voice, p_reverb) for one window."""
    return (
        float(max(p[c] for c in PLAYBACK_CLASSES)),
        float(max(p[c] for c in NONHOST_VOICE_CLASSES)),
        float(max(p[CLS_REVERBERATION], p[CLS_ECHO])),
    )


def merge_spans(labels: list[str], probs: np.ndarray, hop_s: float,
                min_span_s: float = 1.0) -> list[Span]:
    """Collapse consecutive equal labels into spans, dropping slivers."""
    if not labels:
        return []
    spans: list[Span] = []
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            sl = probs[start_i:i]
            # Playback evidence uses the max over the span, not the mean: a
            # 5 s clip inside a 60 s span must not be averaged away.
            pb = np.max(sl[:, list(PLAYBACK_CLASSES)]) if len(sl) else 0.0
            nh = np.max(sl[:, list(NONHOST_VOICE_CLASSES)]) if len(sl) else 0.0
            rv = np.max(sl[:, [CLS_REVERBERATION, CLS_ECHO]]) if len(sl) else 0.0
            spans.append(Span(
                start=start_i * hop_s,
                end=i * hop_s,
                label=labels[start_i],
                p_speech=float(np.mean(np.maximum(sl[:, CLS_SPEECH], sl[:, CLS_CONVERSATION]))),
                p_music=float(np.mean(np.maximum.reduce(
                    [sl[:, CLS_MUSIC], sl[:, CLS_INSTRUMENT], sl[:, CLS_SINGING]]))),
                p_playback=float(pb), p_nonhost_voice=float(nh), p_reverb=float(rv),
            ))
            start_i = i
    # Absorb sub-minimum spans into the preceding one.
    out: list[Span] = []
    for s in spans:
        if out and (s.end - s.start) < min_span_s:
            out[-1].end = s.end
            # Absorbing a sliver must not discard its playback evidence.
            out[-1].p_playback = max(out[-1].p_playback, s.p_playback)
            out[-1].p_nonhost_voice = max(out[-1].p_nonhost_voice, s.p_nonhost_voice)
            out[-1].p_reverb = max(out[-1].p_reverb, s.p_reverb)
        else:
            out.append(s)
    return out


def speech_ratio(spans: list[Span], total_s: float) -> float:
    """Fraction of the episode that is speech (clean or over music).

    Gate S0 expects 0.60-0.98. `speech_over_music` counts as speech here because
    it *is* someone talking; S3 excludes it from training clips separately.
    """
    if total_s <= 0:
        return 0.0
    spoken = sum(min(s.end, total_s) - s.start
                 for s in spans if s.label in ("speech", "speech_over_music"))
    return float(np.clip(spoken / total_s, 0.0, 1.0))


def segment_ast(audio16k: str | Path, *, speech_th: float = 0.35, music_th: float = 0.30,
                hop_s: float = 5.0, batch: int = 16, device: str | None = None,
                cache_dir: str | None = None) -> tuple[list[Span], dict[str, Any]]:
    """Label an episode with AST. Requires torch + transformers (runs on GPU, not the local dev box).

    Playback/non-host-voice detection is NOT parameterised here: it is driven by
    the module constants PLAYBACK_FLOOR / PLAYBACK_MARGIN via _playback_stats
    (an absolute floor plus a MAD-based baseline). Tune those to recalibrate.
    """
    import torch  # lazy: the local dev box has an old torch, so importing at
    from transformers import ASTForAudioClassification, ASTFeatureExtractor
    from . import audio as au

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL, cache_dir=cache_dir)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL, cache_dir=cache_dir)
    model.to(device).eval()

    x = au.read_pcm(audio16k, sr=16000)
    win = int(AST_WINDOW_S * 16000)
    hop = int(hop_s * 16000)
    starts = list(range(0, max(1, len(x) - win + 1), hop)) or [0]

    probs = np.zeros((len(starts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            waves = []
            for s in chunk:
                w = x[s:s + win]
                if len(w) < win:
                    w = np.pad(w, (0, win - len(w)))
                waves.append(w)
            inputs = fe(waves, sampling_rate=16000, return_tensors="pt")
            logits = model(inputs.input_values.to(device)).logits
            probs[b0:b0 + len(chunk)] = torch.sigmoid(logits).cpu().numpy()

    labels = [label_window(p, speech_th=speech_th, music_th=music_th) for p in probs]
    spans = merge_spans(labels, probs, hop_s)
    total_s = len(x) / 16000
    stats = {
        "backend": "ast",
        "model": AST_MODEL,
        "device": device,
        "windows": len(starts),
        "hop_s": hop_s,
        "speech_th": speech_th,
        "music_th": music_th,
        "speech_ratio": round(speech_ratio(spans, total_s), 4),
        "music_span_count": sum(1 for s in spans if s.label == "music"),
        "music_seconds": round(sum(s.end - s.start for s in spans if s.label == "music"), 1),
        "speech_over_music_seconds": round(
            sum(s.end - s.start for s in spans if s.label == "speech_over_music"), 1),
        # Advisory playback evidence (clips, sound effects, TV/radio audio).
        #
        # Reported as a DISTRIBUTION, not a thresholded count. AST is
        # multi-label sigmoid and ordinary podcast speech carries a non-trivial
        # baseline on classes like Sound effect, so an absolute cutoff is
        # meaningless: 0.20 flagged 2875 s of a 3040 s episode. What identifies
        # a clip is standing out against *this episode's own* baseline, so we
        # ship the percentiles and let S12.3 calibrate on episodes the owner
        # confirms contain clips.
        **_playback_stats(spans, probs, hop_s),
    }
    return spans, stats
