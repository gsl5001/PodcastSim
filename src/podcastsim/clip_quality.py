"""Cheap per-clip audio-quality scoring for reference selection (no GPU).

S3 v1 deferred DNSMOS/UTMOS, and picking references by Whisper confidence (word
clarity) does not measure whether a clip *sounds* clean. These numpy/ffmpeg
proxies do: an SNR-like dynamic-range estimate, spectral flatness (noise/music
vs tonal speech), clipping, and level.
"""
from __future__ import annotations
import numpy as np
from . import audio as au


def clip_metrics(source: str | np.ndarray, sr: int = 24000) -> dict:
    """Audio-quality proxies for one clip.

    `source` is either a file path (decoded at `sr`) or an already-decoded mono
    float array at `sr`. Returns snr_db / flatness / clip_frac / rms.
    """
    x = source if isinstance(source, np.ndarray) else au.read_pcm(source, sr=sr)
    if len(x) < sr:
        return {"snr_db": -99.0, "flatness": 1.0, "clip_frac": 1.0, "rms": 0.0}
    win = 1024
    n = len(x) // win * win
    frames = x[:n].reshape(-1, win)
    energy = np.log(np.mean(frames**2, axis=1) + 1e-10)
    hi, lo = np.percentile(energy, 85), np.percentile(energy, 15)
    snr_db = float((hi - lo) * 10 / np.log(10))
    mag = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-10
    flat = float(np.mean(np.exp(np.mean(np.log(mag), axis=1)) / np.mean(mag, axis=1)))
    clip_frac = float(np.mean(np.abs(x) >= 0.98))
    rms = float(np.sqrt(np.mean(x**2)))
    return {"snr_db": round(snr_db, 1), "flatness": round(flat, 4),
            "clip_frac": round(clip_frac, 4), "rms": round(rms, 4)}


def quality_score(m: dict) -> float:
    """Higher = cleaner reference. Rewards dynamic range, penalises noise/clipping."""
    if m["clip_frac"] > 0.01:
        return -1e9
    return m["snr_db"] - 30 * m["flatness"] - 100 * m["clip_frac"]
