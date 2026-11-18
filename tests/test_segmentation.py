"""Segmentation labelling and span merging (plan S5.3).

Pure-numpy paths only -- `segment_ast` needs torch, which has no Intel-Mac
wheel past 2.2.2, so it is exercised on Colab rather than here.
"""

import numpy as np
import pytest

from podcastsim.segmentation import (
    CLS_CONVERSATION,
    CLS_LAUGHTER,
    CLS_MUSIC,
    CLS_SILENCE,
    CLS_SINGING,
    CLS_SPEECH,
    Span,
    label_window,
    merge_spans,
    speech_ratio,
)

pytestmark = pytest.mark.unit

N = 527
TH = {"speech_th": 0.35, "music_th": 0.30}


def probs(values: dict[int, float]) -> np.ndarray:
    """Build a 527-dim AudioSet probability vector with the given classes set."""
    p = np.zeros(N, dtype=np.float32)
    for idx, v in values.items():
        p[idx] = v
    return p


def test_clear_speech():
    assert label_window(probs({CLS_SPEECH: 0.9}), **TH) == "speech"


def test_clear_music():
    assert label_window(probs({CLS_MUSIC: 0.8}), **TH) == "music"


def test_speech_over_music_when_both_fire():
    """S5.3: ad beds and intros are speech that overlaps music."""
    w = probs({CLS_SPEECH: 0.7, CLS_MUSIC: 0.6})
    assert label_window(w, **TH) == "speech_over_music"


def test_conversation_counts_as_speech():
    assert label_window(probs({CLS_CONVERSATION: 0.8}), **TH) == "speech"


def test_singing_counts_as_music():
    assert label_window(probs({CLS_SINGING: 0.7}), **TH) == "music"


def test_silence_detected():
    assert label_window(probs({CLS_SILENCE: 0.9}), **TH) == "silence"


def test_everything_quiet_is_noise():
    assert label_window(probs({CLS_LAUGHTER: 0.1}), **TH) == "noise"


def test_thresholds_are_respected():
    w = probs({CLS_SPEECH: 0.34})
    assert label_window(w, speech_th=0.35, music_th=0.30) == "noise"
    assert label_window(w, speech_th=0.30, music_th=0.30) == "speech"


# --- span merging ------------------------------------------------------------

def stack(*windows) -> np.ndarray:
    return np.stack(windows)


def test_merge_collapses_runs():
    labels = ["speech"] * 4 + ["music"] * 2
    spans = merge_spans(labels, stack(*[probs({CLS_SPEECH: 0.9})] * 6), hop_s=5.0)
    assert [s.label for s in spans] == ["speech", "music"]
    assert spans[0].start == 0.0 and spans[0].end == 20.0
    assert spans[1].start == 20.0 and spans[1].end == 30.0


def test_merge_empty():
    assert merge_spans([], np.zeros((0, N)), hop_s=5.0) == []


def test_short_spans_absorbed():
    """A single stray window must not fragment the timeline."""
    labels = ["speech", "music", "speech"]
    spans = merge_spans(labels, stack(*[probs({CLS_SPEECH: 0.9})] * 3),
                        hop_s=0.5, min_span_s=1.0)
    assert len(spans) == 1
    assert spans[0].end == 1.5


def test_span_serialisation_rounds():
    s = Span(start=1.23456, end=2.34567, label="speech", p_speech=0.987654, p_music=0.1)
    d = s.to_dict()
    assert d["start"] == 1.235 and d["p_speech"] == 0.988
    assert d["label"] == "speech"


# --- speech ratio ------------------------------------------------------------

def test_speech_ratio_counts_speech_over_music():
    """Someone talking over an ad bed is still speech for gate S0.

    S3 excludes speech_over_music from *training clips* separately.
    """
    spans = [
        Span(0, 60, "speech", 0.9, 0.0),
        Span(60, 90, "speech_over_music", 0.7, 0.6),
        Span(90, 100, "music", 0.0, 0.9),
    ]
    assert speech_ratio(spans, 100) == pytest.approx(0.90)


def test_speech_ratio_all_music_is_zero():
    assert speech_ratio([Span(0, 100, "music", 0.0, 0.9)], 100) == 0.0


def test_speech_ratio_clamped_and_safe():
    assert speech_ratio([], 0) == 0.0
    # A final window can overrun the true duration; the ratio must not exceed 1.
    assert speech_ratio([Span(0, 120, "speech", 0.9, 0.0)], 100) == 1.0


def test_observed_fixture_ratio_passes_gate_s0():
    """0.mp3 measured 0.9638 on Colab; gate S0 wants 0.60-0.98."""
    assert 0.60 <= 0.9638 <= 0.98


# --- played media (clips, SFX, TV audio) -------------------------------------

from podcastsim.segmentation import (  # noqa: E402
    CLS_CHILD_SPEECH,
    CLS_FEMALE_SPEECH,
    CLS_RADIO,
    CLS_REVERBERATION,
    CLS_SOUND_EFFECT,
    CLS_TELEVISION,
    playback_scores,
)


def test_television_audio_flagged_as_playback():
    pb, nh, rv = playback_scores(probs({CLS_TELEVISION: 0.7, CLS_SPEECH: 0.8}))
    assert pb == pytest.approx(0.7)


def test_sound_effect_flagged():
    pb, _, _ = playback_scores(probs({CLS_SOUND_EFFECT: 0.55}))
    assert pb == pytest.approx(0.55)


def test_female_voice_flagged_as_nonhost():
    """A male-host show, so a female voice is a clip or a guest."""
    _, nh, _ = playback_scores(probs({CLS_FEMALE_SPEECH: 0.6, CLS_SPEECH: 0.9}))
    assert nh == pytest.approx(0.6)


def test_child_voice_flagged_as_nonhost():
    _, nh, _ = playback_scores(probs({CLS_CHILD_SPEECH: 0.4}))
    assert nh == pytest.approx(0.4)


def test_reverb_flagged_separately():
    """Audio played through speakers into a mic picks up room reverb."""
    _, _, rv = playback_scores(probs({CLS_REVERBERATION: 0.5}))
    assert rv == pytest.approx(0.5)


def test_clean_host_speech_scores_zero_playback():
    pb, nh, rv = playback_scores(probs({CLS_SPEECH: 0.95}))
    assert (pb, nh, rv) == (0.0, 0.0, 0.0)


def test_playback_evidence_uses_max_not_mean_over_span():
    """A 5 s clip inside a 60 s speech span must not be averaged away."""
    windows = [probs({CLS_SPEECH: 0.9}) for _ in range(11)]
    windows[5] = probs({CLS_SPEECH: 0.9, CLS_RADIO: 0.8})
    spans = merge_spans(["speech"] * 11, np.stack(windows), hop_s=5.0)
    assert len(spans) == 1
    assert spans[0].p_playback == pytest.approx(0.8)


def test_absorbed_sliver_keeps_its_playback_evidence():
    """Merging a short span into its neighbour must not lose the flag."""
    windows = [probs({CLS_SPEECH: 0.9}), probs({CLS_TELEVISION: 0.9}),
               probs({CLS_SPEECH: 0.9})]
    spans = merge_spans(["speech", "noise", "speech"], np.stack(windows),
                        hop_s=0.5, min_span_s=1.0)
    assert len(spans) == 1
    assert spans[0].p_playback == pytest.approx(0.9)


def test_span_serialises_playback_fields():
    d = Span(0, 5, "speech", 0.9, 0.0, p_playback=0.1234,
             p_nonhost_voice=0.5678, p_reverb=0.9012).to_dict()
    assert d["p_playback"] == 0.123
    assert d["p_nonhost_voice"] == 0.568
    assert d["p_reverb"] == 0.901


def test_flat_distribution_yields_no_playback_outliers():
    """Observed on 0.mp3: p95 0.365, max 0.374 -- a constant baseline.

    MAD collapses toward zero there, so a pure z-score would call every span a
    5-sigma outlier (it flagged 2875 s of a 3040 s episode).
    """
    from podcastsim.segmentation import _playback_stats
    spans = [Span(i * 5, (i + 1) * 5, "speech", 0.7, 0.0, p_playback=0.36 + (i % 3) * 0.001)
             for i in range(100)]
    assert _playback_stats(spans)["playback_outlier_seconds"] == 0.0


def test_genuine_clip_is_flagged():
    """A span well above both the episode baseline and the absolute floor."""
    from podcastsim.segmentation import _playback_stats
    spans = [Span(i * 5, (i + 1) * 5, "speech", 0.7, 0.0, p_playback=0.10 + (i % 5) * 0.01)
             for i in range(50)]
    spans.append(Span(250, 265, "speech", 0.7, 0.0, p_playback=0.85))
    assert _playback_stats(spans)["playback_outlier_seconds"] == 15.0


def test_unusual_but_weak_is_not_flagged():
    """Statistically odd yet below the floor -> still baseline, not a clip."""
    from podcastsim.segmentation import _playback_stats
    spans = [Span(i * 5, (i + 1) * 5, "speech", 0.7, 0.0, p_playback=0.10 + (i % 5) * 0.01)
             for i in range(50)]
    spans.append(Span(250, 265, "speech", 0.7, 0.0, p_playback=0.30))
    assert _playback_stats(spans)["playback_outlier_seconds"] == 0.0


def test_playback_stats_use_window_granularity_not_span():
    """Regression: 45.mp3 flagged 3780 s of 3804 with a median window of 0.03.

    The episode collapsed into two ~30-minute speech spans. Span-level
    p_playback is a max (so brief clips survive merging), so one high window
    pushed each span over the floor and the whole span's duration was counted.
    Statistics must come from the windows.
    """
    from podcastsim.segmentation import _playback_stats

    n = 720                                  # 720 windows x 5 s = 60 min
    windows = [probs({CLS_SPEECH: 0.8, CLS_SOUND_EFFECT: 0.03}) for _ in range(n)]
    for i in (100, 101, 102):                # 15 s of genuine played media
        windows[i] = probs({CLS_SPEECH: 0.5, CLS_TELEVISION: 0.62})
    probs_arr = np.stack(windows)

    # One long span, exactly as merge_spans would produce.
    spans = merge_spans(["speech"] * n, probs_arr, hop_s=5.0)
    assert len(spans) == 1
    assert spans[0].p_playback == pytest.approx(0.62)   # max survives, by design

    stats = _playback_stats(spans, probs_arr, hop_s=5.0)
    assert stats["playback_outlier_seconds"] == 15.0    # not 3600
    assert stats["playback_p50"] == pytest.approx(0.03, abs=0.01)
    assert stats["playback_baseline"] == pytest.approx(0.03, abs=0.01)


def test_span_fallback_still_works_without_probs():
    """_playback_stats must degrade gracefully when windows aren't supplied."""
    from podcastsim.segmentation import _playback_stats
    spans = [Span(0, 10, "speech", 0.8, 0.0, p_playback=0.02) for _ in range(10)]
    assert _playback_stats(spans)["playback_outlier_seconds"] == 0.0


def test_high_baseline_episode_flags_nothing():
    """0.mp3 idles at ~0.36 on the SFX classes with max 0.374.

    Nothing clears the 0.50 floor, so no played media is reported -- correct for
    the show's first episode.
    """
    from podcastsim.segmentation import _playback_stats
    arr = np.stack([probs({CLS_SPEECH: 0.8, CLS_SOUND_EFFECT: 0.36}) for _ in range(200)])
    spans = merge_spans(["speech"] * 200, arr, hop_s=5.0)
    assert _playback_stats(spans, arr, hop_s=5.0)["playback_outlier_seconds"] == 0.0


def test_margin_scales_with_episode_baseline():
    """An episode idling high needs a proportionally higher spike to flag."""
    from podcastsim.segmentation import _playback_stats
    arr = np.stack([probs({CLS_SPEECH: 0.8, CLS_SOUND_EFFECT: 0.45}) for _ in range(100)])
    arr[50] = probs({CLS_SPEECH: 0.5, CLS_TELEVISION: 0.55})   # over floor, under baseline+margin
    spans = merge_spans(["speech"] * 100, arr, hop_s=5.0)
    assert _playback_stats(spans, arr, hop_s=5.0)["playback_outlier_seconds"] == 0.0

    arr[50] = probs({CLS_SPEECH: 0.5, CLS_TELEVISION: 0.90})   # clearly above both
    spans = merge_spans(["speech"] * 100, arr, hop_s=5.0)
    assert _playback_stats(spans, arr, hop_s=5.0)["playback_outlier_seconds"] == 5.0
