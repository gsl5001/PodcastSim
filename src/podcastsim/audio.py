"""ffmpeg/ffprobe helpers for Stage 0 (plan S5).

Kept dependency-light on purpose: S0 is CPU-only, so per plan S2.4 (infra Plan C)
it runs wherever Claude Code lives rather than burning GPU session time.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Plan S14.2: ALL analysis models are 16 kHz.
ANALYSIS_SR = 16000

_DATE_IN_TEXT = re.compile(
    r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b"  # "May 11, 2016"
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR_IN_NAME = re.compile(r"\b(19|20)\d{2}\b")

_MONTHS = {
    m: i
    for i, m in enumerate(
        "January February March April May June July August September "
        "October November December".split(),
        start=1,
    )
}


class AudioError(RuntimeError):
    """ffmpeg/ffprobe failed."""


@dataclass
class Probe:
    """What ffprobe knows about a source file."""

    duration_s: float
    bit_rate: int
    sample_rate: int
    channels: int
    codec: str
    size: int
    tags: dict[str, str] = field(default_factory=dict)


def _run(cmd: list[str]) -> bytes:
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise AudioError(
            f"{cmd[0]} failed ({p.returncode}): {p.stderr.decode('utf-8', 'replace')[:400]}"
        )
    return p.stdout


def probe(path: str | Path) -> Probe:
    """ffprobe one audio file into a Probe (duration, sample rate, channels,
    bit rate, codec, size, and ID3/format tags). Does not decode the audio."""
    out = _run([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration,bit_rate,size:format_tags:stream=sample_rate,channels,codec_name",
        "-select_streams", "a:0", "-of", "json", str(path),
    ])
    d = json.loads(out)
    fmt = d.get("format", {})
    streams = d.get("streams") or [{}]
    st = streams[0]
    if not fmt.get("duration"):
        raise AudioError(f"no decodable audio stream in {path}")
    return Probe(
        duration_s=float(fmt["duration"]),
        bit_rate=int(fmt.get("bit_rate", 0)),
        sample_rate=int(st.get("sample_rate", 0)),
        channels=int(st.get("channels", 0)),
        codec=str(st.get("codec_name", "")),
        size=int(fmt.get("size", 0)),
        # Drop XMP blobs: they are kilobytes of Adobe XML we never read.
        tags={k: v for k, v in (fmt.get("tags") or {}).items()
              if not k.lower().startswith("id3v2_priv") and len(str(v)) < 2000},
    )


def analysis_codec_args(codec: str) -> list[str]:
    """ffmpeg args for the analysis copy's container/codec.

    WAV is the default: it is uncompressed, so nothing has to be decoded at read
    time. FLAC is bit-identical (verified by PCM md5) and half the size, but any
    per-read decode cost is paid by every stage that touches the file, which on
    a 95-minute episode is not free. VM scratch is 65 GiB, so size is cheap and
    decode time is not.
    """
    if codec == "wav":
        return ["-c:a", "pcm_s16le"]
    if codec == "flac":
        return ["-sample_fmt", "s16", "-c:a", "flac"]
    raise AudioError(f"unsupported analysis codec {codec!r}; expected wav or flac")


def decode_analysis_copy(
    src: str | Path, dst: str | Path, *, expect_duration_s: float | None = None,
    tolerance_s: float = 1.0, codec: str = "wav",
) -> Path:
    """Decode to 16 kHz mono FLAC (plan S5.1).

    One decode, then everything downstream slices by sample index -- S14.3:
    "MP3 encoder delay (~1105 samples at 44.1 kHz) and seek inaccuracy will
    otherwise shift clip boundaries."

    ffmpeg can exit 0 having written a truncated file when the disk fills, so
    the result is verified against the source duration before it is published
    under `dst`. Observed for real: a full disk produced a 609 KB "FLAC"
    reporting 0.18 s from a 63-minute source, with returncode 0.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".tmp.{codec}")
    try:
        _run([
            FFMPEG, "-v", "error", "-y", "-i", str(src),
            # -vn + explicit audio map: many podcast MP3s carry cover art as a
            # video stream, and ffmpeg's default selection happily writes the
            # IMAGE into the output -- yielding a 0.13 s file from a 63 min
            # source, with returncode 0.
            "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", str(ANALYSIS_SR),
            *analysis_codec_args(codec),
            str(tmp),
        ])
        if expect_duration_s is not None:
            got = probe(tmp).duration_s
            if abs(got - expect_duration_s) > tolerance_s:
                raise AudioError(
                    f"decode truncated: {tmp.name} is {got:.2f}s but source is "
                    f"{expect_duration_s:.2f}s (disk full?)"
                )
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


def ensure_analysis_copy(src: str | Path, cache_dir: str | Path, *,
                         codec: str = "wav", stem: str = "audio16k") -> Path:
    """Return the analysis copy, decoding it if the scratch cache lost it.

    The copy is a cache, not an artifact, so any stage
    that needs it must be able to recreate it -- Colab runtimes are ephemeral
    and `/content` does not survive a session.
    """
    cache_dir = Path(cache_dir)
    wanted = cache_dir / f"{stem}.{codec}"
    if wanted.exists():
        return wanted
    # A copy in the other lossless format is equally valid; prefer not to redo
    # the decode just because the configured codec changed.
    for alt in ("wav", "flac"):
        p = cache_dir / f"{stem}.{alt}"
        if p.exists():
            return p
    return decode_analysis_copy(
        src, wanted, expect_duration_s=probe(src).duration_s, codec=codec
    )


def read_pcm(path: str | Path, *, sr: int = ANALYSIS_SR, start: float | None = None,
             duration: float | None = None, channels: int = 1) -> np.ndarray:
    """Decode (a slice of) a file to float32 PCM."""
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-vn", "-map", "0:a:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(channels), "-ar", str(sr), "-"]
    a = np.frombuffer(_run(cmd), dtype=np.float32)
    return a.reshape(-1, channels) if channels > 1 else a


def pcm_sha256(path: str | Path, *, sr: int = ANALYSIS_SR) -> str:
    """Hash of decoded PCM (plan S5.2 dedupe key).

    Content-based, so two encodes of the same audio at different bitrates still
    collide -- which raw-file hashing would miss.
    """
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0",
           "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        h = hashlib.sha256()
        assert p.stdout is not None
        while chunk := p.stdout.read(1 << 20):
            h.update(chunk)
        p.wait()
        if p.returncode != 0:
            raise AudioError(f"ffmpeg failed hashing {path}")
    return h.hexdigest()


def spectral_cutoff(x: np.ndarray, sr: int = ANALYSIS_SR, *, win: int = 8192,
                    floor_above_hz: float | None = None,
                    db_over_floor: float = 40.0) -> tuple[float, float, float]:
    """(rolloff85, rolloff99, cutoff) in Hz.

    Plan S5.1: "estimate real bandwidth ... this decides whether '24 kHz' TTS
    training clips are honest". `cutoff` is the highest frequency still well
    above the top-end noise floor -- for a 128 kbps MP3 that is the encoder's
    lowpass, which caps any honest training sample rate at 2x cutoff.
    """
    if len(x) < win:
        raise AudioError(f"need >= {win} samples for a spectral estimate, got {len(x)}")
    n = len(x) // win * win
    frames = x[:n].reshape(-1, win) * np.hanning(win)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.arange(len(power)) * sr / win

    c = np.cumsum(power)
    r85 = float(np.searchsorted(c, c[-1] * 0.85) * sr / win)
    r99 = float(np.searchsorted(c, c[-1] * 0.99) * sr / win)

    # The noise floor must be estimated in a band the source cannot contain
    # signal in. Measuring a 16 kHz analysis copy is meaningless: it is
    # band-limited to 8 kHz by construction, so the "floor" band is full of
    # real speech and the estimate collapses. Always pass native-rate audio.
    ref = floor_above_hz if floor_above_hz is not None else sr * 0.475
    tail = power[freqs > ref]
    db = 10 * np.log10(power + 1e-20)
    floor = float(np.median(10 * np.log10(tail + 1e-20))) if len(tail) else -200.0
    above = freqs[db > floor + db_over_floor]
    cutoff = float(above.max()) if len(above) else 0.0
    return r85, r99, cutoff


def honest_sample_rate(cutoff_hz: float) -> int:
    """Highest standard TTS rate this material honestly supports (2 x cutoff)."""
    for rate in (48000, 44100, 24000, 22050, 16000):
        if cutoff_hz >= rate / 2:
            return rate
    return 8000


def extract_publish_date(tags: dict[str, str], filename: str) -> tuple[str | None, str]:
    """(ISO date or None, which source it came from) -- plan S5.4.

    NOTE: the plan's order is TDRC/TYER -> filename -> mtime, but this archive
    carries the date inside the TDES description text ("May 11, 2016"), so that
    is checked too.
    """
    for key in ("TDRC", "TYER", "date", "creation_time"):
        v = tags.get(key) or tags.get(key.lower()) or tags.get(key.upper())
        if v:
            if m := _ISO_DATE.search(str(v)):
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", f"id3:{key}"
            if re.fullmatch(r"\d{4}", str(v).strip()):
                return f"{str(v).strip()}-01-01", f"id3:{key}"

    text_keys = [k for k in ("TDES", "description", "comment", "ID3v1 Comment")
                 if k in tags] + sorted(set(tags) - {"TDES", "description",
                                                     "comment", "ID3v1 Comment"})
    for key in text_keys:
        val = tags[key]
        if m := _DATE_IN_TEXT.search(str(val)):
            # Re-parse the matched span into parts; guard against the two patterns
            # drifting out of sync (a _DATE_IN_TEXT hit that this stricter regex
            # misses would otherwise raise AttributeError on .groups()).
            parts = re.match(r"([A-Z][a-z]+) (\d{1,2}), (\d{4})", m.group(1))
            if parts:
                month, day, year = parts.groups()
                if month in _MONTHS:
                    return f"{year}-{_MONTHS[month]:02d}-{int(day):02d}", f"id3_text:{key}"

    if m := _YEAR_IN_NAME.search(filename):
        return f"{m.group(0)}-01-01", "filename"
    return None, "none"
