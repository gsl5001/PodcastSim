"""S1 pure logic: overlap detection, music masking, RTTM (plan S6)."""

import json

import pytest

from podcastsim.stages.s1_diarize import (
    compute_overlap,
    load_music_spans,
    music_intervals,
    overlaps_any,
    write_rttm,
)

pytestmark = pytest.mark.unit


# --- overlap regions (S6.1) --------------------------------------------------

def test_no_overlap_between_sequential_turns():
    turns = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]
    regions, padded, true = compute_overlap(turns, pad_s=0.0)
    assert regions == [] and padded == 0.0 and true == 0.0


def test_overlap_detected_and_padded():
    """S6.1: frames where >= 2 speakers are active, padded 0.25 s each side."""
    turns = [(0.0, 6.0, "A"), (5.0, 10.0, "B")]
    regions, padded, true = compute_overlap(turns, pad_s=0.25)
    assert len(regions) == 1
    assert regions[0]["start"] == pytest.approx(4.75)
    assert regions[0]["end"] == pytest.approx(6.25)
    assert padded == pytest.approx(1.5)
    # Gate S1-A scores the genuine crosstalk, not the exclusion zone.
    assert true == pytest.approx(1.0)


def test_padding_never_goes_negative():
    turns = [(0.0, 2.0, "A"), (0.1, 3.0, "B")]
    regions, _, _ = compute_overlap(turns, pad_s=1.0)
    assert regions[0]["start"] == 0.0


def test_adjacent_overlap_regions_merge():
    turns = [(0.0, 3.0, "A"), (2.0, 5.0, "B"), (4.0, 7.0, "C")]
    regions, _, _ = compute_overlap(turns, pad_s=0.5)
    assert len(regions) == 1


def test_three_way_overlap_is_one_region():
    turns = [(0.0, 9.0, "A"), (1.0, 8.0, "B"), (2.0, 7.0, "C")]
    regions, padded, true = compute_overlap(turns, pad_s=0.0)
    assert len(regions) == 1
    assert padded == pytest.approx(7.0) and true == pytest.approx(7.0)


# --- music masking (S5.3) ----------------------------------------------------

def test_music_intervals_selects_non_speech():
    spans = [
        {"start": 0, "end": 15, "label": "music"},
        {"start": 15, "end": 35, "label": "speech_over_music"},
        {"start": 35, "end": 300, "label": "speech"},
        {"start": 300, "end": 305, "label": "noise"},
    ]
    assert music_intervals(spans) == [(0, 15), (300, 305)]


def test_speech_over_music_is_not_masked():
    """It is someone talking; S3 excludes it from training clips separately."""
    spans = [{"start": 0, "end": 60, "label": "speech_over_music"}]
    assert music_intervals(spans) == []


def test_turn_inside_music_is_dropped():
    assert overlaps_any(2.0, 8.0, [(0.0, 15.0)]) is True


def test_turn_outside_music_is_kept():
    assert overlaps_any(40.0, 50.0, [(0.0, 15.0)]) is False


def test_partial_overlap_below_threshold_is_kept():
    """A turn only 25% inside music survives at the default 0.5 threshold."""
    assert overlaps_any(10.0, 20.0, [(0.0, 12.5)], min_frac=0.5) is False


def test_partial_overlap_above_threshold_is_dropped():
    assert overlaps_any(10.0, 20.0, [(0.0, 16.0)], min_frac=0.5) is True


def test_zero_length_turn_does_not_divide_by_zero():
    """Degenerate turns must not raise; zero covered time means "not inside"."""
    assert overlaps_any(5.0, 5.0, [(0.0, 10.0)]) is False


def test_load_music_spans_missing_file(tmp_path):
    assert load_music_spans(tmp_path, "nope__x") == []


def test_load_music_spans_reads_s0_output(tmp_path):
    d = tmp_path / "ep1" / "s0_ingest"
    d.mkdir(parents=True)
    (d / "segments_music.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 5, "label": "music"}]})
    )
    assert load_music_spans(tmp_path, "ep1")[0]["label"] == "music"


# --- RTTM (S12.1 scores these with pyannote.metrics) -------------------------

def test_rttm_format(tmp_path):
    p = tmp_path / "d.rttm"
    write_rttm(p, "ep1", [(1.5, 4.25, "SPEAKER_01")])
    line = p.read_text().strip()
    fields = line.split()
    assert fields[0] == "SPEAKER" and fields[1] == "ep1"
    assert fields[3] == "1.500" and fields[4] == "2.750"   # start, DURATION
    assert fields[7] == "SPEAKER_01"


def test_rttm_is_sorted_by_start(tmp_path):
    p = tmp_path / "d.rttm"
    write_rttm(p, "ep1", [(9.0, 10.0, "B"), (1.0, 2.0, "A")])
    starts = [float(l.split()[3]) for l in p.read_text().splitlines()]
    assert starts == sorted(starts)


def test_padding_inflates_overlap_when_turns_are_short():
    """Why gate S1-A must score the unpadded total.

    0.mp3 has 1004 turns in 3040 s -- a turn every ~3 s. Padding every overlap
    region by 0.5 s reported 18.1% overlap against a 2-10% gate band.
    """
    turns = []
    for i in range(20):                       # 20 brief crosstalk moments
        turns.append((i * 10.0, i * 10.0 + 2.0, "A"))
        turns.append((i * 10.0 + 1.9, i * 10.0 + 4.0, "B"))
    _, padded, true = compute_overlap(turns, pad_s=0.25)
    assert true == pytest.approx(20 * 0.1, abs=0.01)   # 0.1 s each
    assert padded == pytest.approx(20 * 0.6, abs=0.01)  # 6x larger
    assert padded > true * 5


# --- speech ratio must be a union, not a sum ---------------------------------

from podcastsim.stages.s1_diarize import _union_seconds  # noqa: E402


def test_union_counts_simultaneous_speech_once():
    """Two people talking at once is one second of speech, not two.

    Summing per-speaker time gave speech_ratio = 1.0429, which is impossible.
    """
    assert _union_seconds([(0.0, 10.0), (5.0, 15.0)]) == pytest.approx(15.0)


def test_union_of_disjoint_intervals():
    assert _union_seconds([(0.0, 5.0), (10.0, 15.0)]) == pytest.approx(10.0)


def test_union_of_nested_intervals():
    assert _union_seconds([(0.0, 20.0), (5.0, 10.0)]) == pytest.approx(20.0)


def test_union_empty():
    assert _union_seconds([]) == 0.0


def test_union_is_order_independent():
    a = _union_seconds([(9.0, 12.0), (0.0, 10.0), (5.0, 6.0)])
    b = _union_seconds([(0.0, 10.0), (5.0, 6.0), (9.0, 12.0)])
    assert a == b == pytest.approx(12.0)


def test_union_never_exceeds_episode_duration():
    """The invariant that speech_ratio > 1.0 violated."""
    turns = [(0.0, 100.0), (10.0, 90.0), (50.0, 100.0)]
    assert _union_seconds(turns) <= 100.0
