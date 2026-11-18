"""Long-episode chunking and speaker re-linking (plan S6.1)."""

import numpy as np
import pytest

from podcastsim.chunking import Chunk, plan_chunks, relink_speakers

pytestmark = pytest.mark.unit


# --- planning cut points -----------------------------------------------------

def test_short_episode_is_one_chunk():
    chunks = plan_chunks(1000.0, [], target_s=1200, max_s=1500)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0 and chunks[0].end_s == 1000.0


def test_long_episode_splits_at_silence():
    # 40-min episode, a clean 3 s silence near the 20-min mark.
    silences = [(1198.0, 1201.0)]
    chunks = plan_chunks(2400.0, silences, target_s=1200, max_s=1500)
    assert len(chunks) == 2
    # cut at the silence midpoint, 1199.5
    assert chunks[0].end_s == pytest.approx(1199.5)
    assert chunks[1].start_s == pytest.approx(1199.5)


def test_chunks_are_contiguous_and_cover_the_episode():
    silences = [(1190.0, 1195.0), (2400.0, 2405.0)]
    chunks = plan_chunks(3600.0, silences, target_s=1200, max_s=1500)
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == 3600.0
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_s == b.start_s


def test_falls_back_to_hard_cut_without_silence():
    """A relentless episode with no gaps still gets bounded chunks."""
    chunks = plan_chunks(3000.0, [], target_s=1200, max_s=1500)
    assert all(c.end_s - c.start_s <= 1500 + 1e-6 for c in chunks)
    assert len(chunks) >= 2


def test_no_chunk_exceeds_max():
    silences = [(t, t + 3) for t in range(500, 3600, 400)]
    chunks = plan_chunks(3600.0, silences, target_s=1200, max_s=1500)
    assert all(c.end_s - c.start_s <= 1500 + 1e-6 for c in chunks)


# --- speaker re-linking across chunks ----------------------------------------

def vec(*xs):
    return np.array(xs, dtype=np.float32)


def test_same_voice_relinks_across_chunks():
    """A host in both chunks must get ONE global label, not two."""
    a = vec(1, 0, 0)
    results = [
        {"index": 0, "offset": 0.0, "turns": [(0, 10, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": a}},
        {"index": 1, "offset": 1200.0, "turns": [(0, 10, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": a * 0.95}},   # same voice, slight drift
    ]
    turns, stats = relink_speakers(results, link_threshold=0.7)
    assert stats["n_global_speakers"] == 1
    assert {t[2] for t in turns} == {"SPEAKER_00"}


def test_different_voices_stay_separate():
    results = [
        {"index": 0, "offset": 0.0, "turns": [(0, 10, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(1, 0, 0)}},
        {"index": 1, "offset": 1200.0, "turns": [(0, 10, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(0, 1, 0)}},
    ]
    _, stats = relink_speakers(results, link_threshold=0.7)
    assert stats["n_global_speakers"] == 2


def test_turn_times_get_chunk_offset():
    """A turn at 5 s in chunk 2 (offset 1200) must land at 1205 s globally."""
    results = [
        {"index": 0, "offset": 0.0, "turns": [(0, 10, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(1, 0, 0)}},
        {"index": 1, "offset": 1200.0, "turns": [(5, 15, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(1, 0, 0)}},
    ]
    turns, _ = relink_speakers(results, link_threshold=0.7)
    assert (1205.0, 1215.0, "SPEAKER_00") in turns


def test_two_hosts_across_three_chunks():
    """Two recurring voices, appearing in different orders per chunk."""
    h1, h2 = vec(1, 0, 0), vec(0, 1, 0)
    results = [
        {"index": 0, "offset": 0, "turns": [(0, 5, "SPEAKER_00"), (5, 10, "SPEAKER_01")],
         "embeddings": {"SPEAKER_00": h1, "SPEAKER_01": h2}},
        {"index": 1, "offset": 1200, "turns": [(0, 5, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": h2 * 0.98}},   # host 2 labelled 00 here
        {"index": 2, "offset": 2400, "turns": [(0, 5, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": h1 * 0.97}},
    ]
    _, stats = relink_speakers(results, link_threshold=0.7)
    assert stats["n_global_speakers"] == 2


def test_empty_chunks():
    turns, stats = relink_speakers([], link_threshold=0.7)
    assert turns == [] and stats["n_global_speakers"] == 0


def test_merged_turns_are_sorted():
    results = [
        {"index": 0, "offset": 0, "turns": [(0, 5, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(1, 0, 0)}},
        {"index": 1, "offset": 1200, "turns": [(0, 5, "SPEAKER_00")],
         "embeddings": {"SPEAKER_00": vec(1, 0, 0)}},
    ]
    turns, _ = relink_speakers(results, link_threshold=0.7)
    assert turns == sorted(turns)
