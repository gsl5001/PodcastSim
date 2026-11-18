"""Cut, VAD-segment, transcribe and filter training clips (plan S8.2-8.7).

GPU/model-heavy, so this lives apart from the pure selection logic and imports
its dependencies lazily (it only runs on Colab).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import audio as au


def _load_silero():
    """Silero VAD v6 via torch.hub (MIT, ~2 MB). Returns (get_ts fn, model)."""
    import torch

    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  trust_repo=True, onnx=False)
    get_speech_timestamps = utils[0]
    return get_speech_timestamps, model


def _load_whisper(cfg):
    from faster_whisper import WhisperModel

    size = cfg.get("s3_dataset.asr.model", "large-v3")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(size, device=device, compute_type=compute)


def _load_crisperwhisper(cfg):
    """CrisperWhisper via transformers -- verbatim transcripts that KEEP the
    disfluencies ("um", false starts) plain Whisper strips (plan S8.4)."""
    import os
    import torch
    from transformers import pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline(
        "automatic-speech-recognition",
        model=cfg.get("s3_dataset.asr.crisper_repo", "nyrahealth/CrisperWhisper"),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,
        token=os.environ.get("HF_TOKEN"),
    )


def _resegment(pcm24: np.ndarray, sr: int, get_ts, model, cfg) -> list[tuple[int, int]]:
    """Silero VAD -> speech (start, end) sample indices, split to 3-20 s (S8.3)."""
    import torch

    lo, hi = cfg.get("s3_dataset.vad.split_range_s", [3.0, 20.0])
    pad_ms = float(cfg.get("s3_dataset.vad.edge_padding_ms", 150))
    # Silero wants 16 kHz; resample a copy for detection, index back to 24 kHz.
    ratio = 16000 / sr
    x16 = np.interp(np.arange(0, len(pcm24) * ratio) / ratio,
                    np.arange(len(pcm24)), pcm24).astype(np.float32)
    ts = get_ts(torch.from_numpy(x16), model, sampling_rate=16000,
                min_speech_duration_ms=int(lo * 1000),
                max_speech_duration_s=hi, speech_pad_ms=int(pad_ms))
    out = []
    for seg in ts:
        a = int(seg["start"] / 16000 * sr)
        b = int(seg["end"] / 16000 * sr)
        if (b - a) / sr >= lo:
            out.append((a, b))
    return out


def cut_and_transcribe(src_mp3: Path, candidates, hosts_root: Path,
                       episode_id: str, cfg) -> dict:
    """Cut candidate turns to 24 kHz clips, VAD-split, transcribe, filter."""
    import soundfile as sf

    target_sr = int(cfg.get("s3_dataset.target_sample_rate", 24000))
    f = cfg.stage_params("s3_dataset").get("filters", {})
    lo_dur, hi_dur = cfg.get("s3_dataset.clip_duration_s", [1.0, 20.0])
    target_lufs = float(cfg.get("s3_dataset.loudness.target_lufs", -23.0))

    # Decode the whole MP3 once at native rate (S14.3), slice by sample index.
    native = au.read_pcm(src_mp3, sr=target_sr)   # resampled once to 24k
    get_ts, vad = _load_silero()
    asr_backend = cfg.get("s3_dataset.asr.primary", "faster_whisper_large_v3")
    use_crisper = "crisper" in asr_backend.lower()
    if use_crisper:
        crisper = _load_crisperwhisper(cfg)
    else:
        whisper = _load_whisper(cfg)
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(target_sr)
    except ImportError:
        meter = None

    stats = {"clips_written": 0, "clips_rejected": 0,
             "retained_seconds": {}, "reject_reasons": {}}

    def reject(reason):
        stats["clips_rejected"] += 1
        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

    for ci, cand in enumerate(candidates):
        a = int(cand.start * target_sr)
        b = int(cand.end * target_sr)
        turn_pcm = native[a:b]
        if len(turn_pcm) < lo_dur * target_sr:
            reject("too_short_turn"); continue

        for si, (s, e) in enumerate(_resegment(turn_pcm, target_sr, get_ts, vad, cfg)):
            clip = turn_pcm[s:e]
            dur = len(clip) / target_sr
            if not (lo_dur <= dur <= hi_dur):
                reject("duration"); continue
            # MP3 decoding can overshoot beyond +/-1.0 (45.mp3 peaks at 1.36),
            # and this podcast is mastered hot. That is not digital clipping.
            # Normalise to the clip's own peak first, THEN test for genuine
            # sustained clipping (a long run pinned at the very top).
            peak = float(np.max(np.abs(clip))) or 1.0
            clip = clip / peak
            if np.mean(np.abs(clip) >= 0.999) > f.get("max_clipped_sample_fraction", 0.02):
                reject("clipping"); continue

            if use_crisper:
                # CrisperWhisper: verbatim text (keeps disfluencies). No
                # per-segment logprob/no_speech metrics, so gate on audio
                # quality (SNR/clipping) + text sanity instead.
                r = crisper({"raw": clip.astype("float32"), "sampling_rate": target_sr}, return_timestamps=False)
                text = (r.get("text") or "").strip()
                # Thresholds come from config like the Whisper path's do (S3
                # config contract), so both ASR backends gate consistently.
                if len(text.split()) < int(f.get("crisper_min_words", 2)):
                    reject("empty_text"); continue
                from .clip_quality import clip_metrics
                m = clip_metrics(clip, target_sr)
                if m["clip_frac"] > f.get("max_clipped_sample_fraction", 0.02) or \
                   m["snr_db"] < float(f.get("crisper_min_snr_db", 12)):
                    reject("low_quality"); continue
                alp = nsp = crr = None
                lang = "en"
            else:
                segs, info = whisper.transcribe(
                    clip, beam_size=int(cfg.get("s3_dataset.asr.beam_size", 5)),
                    condition_on_previous_text=False, vad_filter=True)
                segs = list(segs)
                text = " ".join(s.text.strip() for s in segs).strip()
                if not text:
                    reject("empty_text"); continue
                if info.language != f.get("language", "en") or \
                   info.language_probability < f.get("language_prob_min", 0.8):
                    reject("language"); continue
                alp = float(np.mean([s.avg_logprob for s in segs]))
                nsp = float(np.mean([s.no_speech_prob for s in segs]))
                crr = float(np.mean([s.compression_ratio for s in segs]))
                lang = info.language
                if alp < f.get("whisper_avg_logprob_min", -0.7):
                    reject("low_logprob"); continue
                if nsp > f.get("whisper_no_speech_prob_max", 0.5):
                    reject("no_speech"); continue
                if crr > f.get("whisper_compression_ratio_max", 2.4):
                    reject("repetition"); continue

            if meter is not None:
                loud = meter.integrated_loudness(clip)
                if np.isfinite(loud):
                    clip = pyln.normalize.loudness(clip, loud, target_lufs)
            safe_peak = float(np.max(np.abs(clip))) or 1.0
            if safe_peak > 0.999:
                clip = clip * (0.999 / safe_peak)

            host_dir = hosts_root / cand.host_id / "clips"
            host_dir.mkdir(parents=True, exist_ok=True)
            name = f"{episode_id}_{ci:04d}_{si:02d}.flac"
            sf.write(host_dir / name, clip, target_sr)
            meta = {"clip": name, "host": cand.host_id, "episode_id": episode_id,
                    "start": round(cand.start + s / target_sr, 3),
                    "duration": round(dur, 3), "text": text,
                    "asr": asr_backend, "language": lang,
                    "avg_logprob": round(alp, 3) if alp is not None else None}
            stats.setdefault("rows", []).append(meta)
            stats["clips_written"] += 1
            stats["retained_seconds"][cand.host_id] = \
                stats["retained_seconds"].get(cand.host_id, 0.0) + dur

    return stats


def aggregate_host_metadata(work_root: str | Path, hosts_root: str | Path
                            ) -> dict[str, int]:
    """Assemble per-host ``metadata.jsonl`` from the per-episode ``clips.jsonl``.

    S3 writes one ``work/{episode}/s3_dataset/clips.jsonl`` per episode (so a
    resumed run never corrupts a shared file). S4 (``s4_refbank`` and the
    fine-tuning kernels) consumes one ``hosts/{host}/metadata.jsonl`` per host.
    This is the bridge between the two: it reads every episode's clip rows and
    writes them, grouped by host, into the per-host files.

    Returns ``{host: n_clips_written}``. Safe to re-run -- each host file is
    rewritten from scratch, so it always reflects the current clips.jsonl set.
    """
    work_root, hosts_root = Path(work_root), Path(hosts_root)
    by_host: dict[str, list[dict]] = {}
    for cj in sorted(work_root.glob("*/s3_dataset/clips.jsonl")):
        for line in cj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            host = row.get("host")
            if host:
                by_host.setdefault(host, []).append(row)

    counts: dict[str, int] = {}
    for host, rows in by_host.items():
        host_dir = hosts_root / host
        host_dir.mkdir(parents=True, exist_ok=True)
        (host_dir / "metadata.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows))
        counts[host] = len(rows)
    return counts
