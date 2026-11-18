"""Audio helpers (plan S5). Pure-numpy tests; no ffmpeg required."""

import numpy as np
import pytest

from podcastsim.audio import (
    AudioError,
    extract_publish_date,
    honest_sample_rate,
    spectral_cutoff,
)

pytestmark = pytest.mark.unit


def band_limited_noise(sr: int, cutoff: float, seconds: float = 4.0) -> np.ndarray:
    """White noise brick-walled at `cutoff`, like an MP3 encoder's lowpass."""
    rng = np.random.default_rng(0)
    n = int(sr * seconds)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec[freqs > cutoff] = 0
    return np.fft.irfft(spec, n).astype(np.float32)


@pytest.mark.parametrize("cutoff", [8000, 12000, 16000])
def test_spectral_cutoff_recovers_known_lowpass(cutoff):
    sr = 44100
    x = band_limited_noise(sr, cutoff)
    _, _, got = spectral_cutoff(x, sr)
    assert abs(got - cutoff) < 600, f"expected ~{cutoff}, got {got}"


def test_spectral_cutoff_rejects_short_input():
    with pytest.raises(AudioError, match="8192 samples"):
        spectral_cutoff(np.zeros(100, dtype=np.float32), 16000)


def test_rolloffs_are_ordered():
    x = band_limited_noise(44100, 12000)
    r85, r99, cutoff = spectral_cutoff(x, 44100)
    assert r85 <= r99 <= cutoff


@pytest.mark.parametrize(
    "cutoff,expected",
    [
        (16554, 24000),   # 0.mp3: stereo-era encode
        (12441, 24000),   # 22.mp3: mono-era encode, just clears 24k
        (11999, 22050),
        (7000, 8000),
        (0, 8000),
    ],
)
def test_honest_sample_rate(cutoff, expected):
    """S5.1: the measured lowpass caps any honest training rate at 2x cutoff."""
    assert honest_sample_rate(cutoff) == expected


def test_real_archive_supports_24k():
    """Every fixture measured >= 12 kHz cutoff, so 24 kHz training is honest."""
    for cutoff in (16554, 12441, 16597, 12484, 12559):
        assert honest_sample_rate(cutoff) >= 24000


# --- publish dates (plan S5.4) ----------------------------------------------

def test_date_from_id3_tdrc():
    d, src = extract_publish_date({"TDRC": "2016-05-11"}, "0.mp3")
    assert (d, src) == ("2016-05-11", "id3:TDRC")


def test_date_from_bare_year():
    d, src = extract_publish_date({"TYER": "2017"}, "x.mp3")
    assert (d, src) == ("2017-01-01", "id3:TYER")


def test_date_from_description_text():
    """This archive puts the date in TDES prose, not TDRC"""
    tags = {"TDES": "May 11, 2016 Cum Boys NYC Originals Nick and Stav sit down"}
    d, src = extract_publish_date(tags, "0.mp3")
    assert d == "2016-05-11"
    assert src == "id3_text:TDES"


def test_description_preferred_over_other_text_fields():
    """Deterministic source ordering, not dict iteration order."""
    tags = {"comment": "January 1, 2020 reissue", "TDES": "May 11, 2016 original"}
    d, src = extract_publish_date(tags, "0.mp3")
    assert (d, src) == ("2016-05-11", "id3_text:TDES")


def test_date_from_filename_fallback():
    d, src = extract_publish_date({}, "2018-03-04 episode.mp3")
    assert d == "2018-01-01" and src == "filename"


def test_no_date_available():
    assert extract_publish_date({}, "12.mp3") == (None, "none")


def test_single_digit_day_is_padded():
    tags = {"TDES": "March 4, 2019 something"}
    assert extract_publish_date(tags, "x.mp3")[0] == "2019-03-04"


# --- analysis codec (WAV over FLAC for read cost) ----------

from podcastsim.audio import analysis_codec_args  # noqa: E402


def test_wav_args_are_uncompressed_pcm():
    assert analysis_codec_args("wav") == ["-c:a", "pcm_s16le"]


def test_flac_args_pin_16_bit():
    """ffmpeg otherwise picks s32/24-bit and inflates every file ~1.5x."""
    args = analysis_codec_args("flac")
    assert "-sample_fmt" in args and args[args.index("-sample_fmt") + 1] == "s16"


def test_unknown_codec_rejected():
    with pytest.raises(AudioError, match="unsupported analysis codec"):
        analysis_codec_args("mp3")
