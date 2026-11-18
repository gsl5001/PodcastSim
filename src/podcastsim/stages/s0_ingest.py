"""Stage 0 - Ingest and normalise (plan S5).

Goal: "turn arbitrary MP3s into consistent analysis audio plus episode metadata,
and remove non-speech (intro music, ads with music beds) before diarization sees
it."

Outputs per episode, under `data/work/{ep}/s0/`:
    audio16k.wav        16 kHz mono analysis copy (S14.2)
    meta.json            duration, rates, bandwidth, tags, publish date
    segments_music.json  speech/music/noise spans
    _DONE.json

S0 is CPU-only, so per S2.4 (infra Plan C) it runs wherever Claude Code lives
rather than spending GPU session time.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import audio as au
from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest, StageGuard

STAGE = "s0_ingest"
PLAN_SECTION = "S5"

# Bump when this stage's output would change. History:
#   1  initial
#   2  -vn/-map 0:a:0 (cover art was being decoded instead of audio) and
#      -sample_fmt s16 (ffmpeg defaulted to 24-bit, ~1.5x larger)
#   3  bandwidth measured on the native-rate source, not the 16 kHz copy
#   4  AST segmentation backend replaces inaSpeechSegmenter (S5.3 deviation)
#   5  played-media flags (clips/SFX/TV/non-host voices) added to spans
#   6  playback reported as a distribution + MAD outliers, not an absolute
#      threshold (0.20 flagged 95% of an episode)
#   7  analysis copy is WAV, not FLAC (no per-read decode cost)
#   8  playback outliers need an absolute floor as well as MAD (a flat
#      distribution made every span a "5-sigma outlier")
#   9  playback stats computed per WINDOW, not per span (one high window in a
#      30-minute span was flagging the whole span)
CODE_VERSION = 9

ANALYSIS_STEM = "audio16k"
META_NAME = "meta.json"
SEGMENTS_NAME = "segments_music.json"


def _versions() -> dict[str, str]:
    import numpy

    out: dict[str, str] = {"numpy": numpy.__version__, "gpu": "cpu"}
    try:
        banner = au._run([au.FFMPEG, "-version"]).decode("utf-8", "replace")
        out["ffmpeg"] = banner.split("\n", 1)[0].split(" ")[2]
    except Exception:  # pragma: no cover - ffmpeg absent is caught earlier
        out["ffmpeg"] = "unknown"
    return out


def segment_music(audio16k: Path, cfg: Config, paths_models: str | Path | None = None
                  ) -> tuple[list[dict], dict, list[str]]:
    """Speech/music/noise spans (plan S5.3). Returns (spans, stats, warnings).

    See configs/pipeline.yaml for why this uses AST rather than the plan's
    inaSpeechSegmenter.
    """
    backend = cfg.get("s0_ingest.segmentation.backend", "none")
    if backend in (None, "none"):
        return [], {"backend": "none"}, [
            "segmentation backend 'none': segments_music.json is empty and "
            "speech_ratio is unmeasured; gate S0 cannot be evaluated yet"
        ]
    if backend == "ast":
        from ..segmentation import segment_ast

        spans, stats = segment_ast(
            audio16k,
            speech_th=float(cfg.get("s0_ingest.segmentation.speech_threshold", 0.35)),
            music_th=float(cfg.get("s0_ingest.segmentation.music_threshold", 0.30)),
            hop_s=float(cfg.get("s0_ingest.segmentation.hop_s", 5.0)),
            batch=int(cfg.get("s0_ingest.segmentation.batch", 16)),
            cache_dir=str(Path(paths_models) / "cache") if paths_models else None,
        )
        return [s.to_dict() for s in spans], stats, []
    raise NotImplementedError(
        f"segmentation backend {backend!r} not implemented (plan {PLAN_SECTION}.3)"
    )


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    src = Path(paths["raw"]) / episode.source
    guard = StageGuard(
        paths["work"],
        episode.episode_id,
        STAGE,
        # The episode id already pins the source bytes, so cite it as a token
        # rather than re-hashing a 60 MB file off Drive every run (S14.13).
        inputs=[f"src:{episode.episode_id}"],
        params=cfg.stage_params(STAGE),
        versions=_versions(),
        code_version=CODE_VERSION,
    )
    if not force and guard.should_skip():
        return None

    with guard:
        p = au.probe(src)
        out_dir = guard.dir

        # 1. One decode to the 16 kHz analysis copy (S5.1). It goes to the
        # scratch cache, not to Drive-backed work/: it is regenerable, and 71
        # of them would be ~7.8 GB.
        cache_dir = Path(paths.get("cache", "data/cache")) / episode.episode_id
        codec = str(cfg.get("s0_ingest.analysis_codec", "wav"))
        audio16k = au.decode_analysis_copy(
            src, cache_dir / f"{ANALYSIS_STEM}.{codec}",
            expect_duration_s=p.duration_s, codec=codec,
        )

        # 2. Real bandwidth (S5.1). Measured on the SOURCE at its native rate:
        # the 16 kHz analysis copy is band-limited to 8 kHz by construction, so
        # measuring it would only ever rediscover our own resampler.
        probe_s = float(cfg.get("s0_ingest.spectral_rolloff_probe_s", 60))
        start = max(0.0, min(600.0, p.duration_s / 2 - probe_s / 2))
        native_pcm = au.read_pcm(src, sr=p.sample_rate, start=start, duration=probe_s)
        r85, r99, cutoff = au.spectral_cutoff(native_pcm, p.sample_rate)
        honest_sr = au.honest_sample_rate(cutoff)

        # 3. Dedupe key over decoded PCM (S5.2).
        pcm_hash = au.pcm_sha256(audio16k)

        # 4. Publish date (S5.4).
        date, date_source = au.extract_publish_date(p.tags, episode.source)
        if date is None:
            guard.warn("no publish date derivable; S2 era assignment will fall back to order")

        # 5. Music/speech segmentation (S5.3).
        segments, seg_stats, seg_warnings = segment_music(
            audio16k, cfg, paths_models=paths.get("models")
        )
        for w in seg_warnings:
            guard.warn(w)

        meta = {
            "episode_id": episode.episode_id,
            "source": episode.source,
            "duration_s": round(p.duration_s, 3),
            "native_sample_rate": p.sample_rate,
            "native_channels": p.channels,
            "bit_rate": p.bit_rate,
            "codec": p.codec,
            "size": p.size,
            "pcm_sha256": pcm_hash,
            "bandwidth": {
                "rolloff85_hz": round(r85, 1),
                "rolloff99_hz": round(r99, 1),
                "cutoff_hz": round(cutoff, 1),
                "honest_sample_rate": honest_sr,
                "probe_start_s": round(start, 1),
                "probe_duration_s": probe_s,
            },
            "publish_date": date,
            "publish_date_source": date_source,
            "analysis_copy": str(audio16k),
            "tags": p.tags,
        }
        (out_dir / META_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        (out_dir / SEGMENTS_NAME).write_text(
            json.dumps({**seg_stats, "segments": segments}, indent=2)
        )

        target_sr = int(cfg.get("s3_dataset.target_sample_rate", 24000))
        if honest_sr < target_sr:
            guard.warn(
                f"source is lowpassed at {cutoff:.0f} Hz -> honest max {honest_sr} Hz, "
                f"below s3_dataset.target_sample_rate={target_sr}"
            )

        if not cfg.get("s0_ingest.keep_analysis_copy", True):
            audio16k.unlink(missing_ok=True)

        guard.metrics.update({
            "duration_s": round(p.duration_s, 1),
            "native_sample_rate": p.sample_rate,
            "native_channels": p.channels,
            "bit_rate_kbps": p.bit_rate // 1000,
            "cutoff_hz": round(cutoff, 1),
            "honest_sample_rate": honest_sr,
            "publish_date": date,
            "speech_ratio": seg_stats.get("speech_ratio"),
            "music_seconds": seg_stats.get("music_seconds"),
            "speech_over_music_seconds": seg_stats.get("speech_over_music_seconds"),
            "playback_p95": seg_stats.get("playback_p95"),
            "playback_max": seg_stats.get("playback_max"),
            "playback_outlier_seconds": seg_stats.get("playback_outlier_seconds"),
            "nonhost_voice_max": seg_stats.get("nonhost_voice_max"),
        })

    return guard.existing
