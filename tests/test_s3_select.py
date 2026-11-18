"""S3 turn selection (plan S8.1)."""

import pytest

from podcastsim.rttm import Turn
from podcastsim.s3_select import Candidate, host_hours, select_host_turns

pytestmark = pytest.mark.unit

HOSTS = {"SPEAKER_00": "host_0", "SPEAKER_01": "host_1", "SPEAKER_02": "guest"}


def test_only_host_turns_kept():
    turns = [Turn(0, 5, "SPEAKER_00"), Turn(5, 10, "SPEAKER_02")]  # host + guest
    got = select_host_turns(turns, HOSTS, [], [])
    assert len(got) == 1 and got[0].host_id == "host_0"


def test_short_turns_dropped():
    turns = [Turn(0, 0.5, "SPEAKER_00"), Turn(1, 5, "SPEAKER_00")]
    got = select_host_turns(turns, HOSTS, [], [], min_duration_s=1.0)
    assert len(got) == 1 and got[0].start == 1


def test_turn_in_overlap_region_dropped():
    """Overlap contaminates timbre -- discard, don't separate (S8.1)."""
    turns = [Turn(10, 20, "SPEAKER_00")]
    got = select_host_turns(turns, HOSTS, [(15, 18)], [])
    assert got == []


def test_padding_widens_overlap_exclusion():
    """A turn ending just before an overlap region is still dropped (0.25 pad)."""
    turns = [Turn(10, 15, "SPEAKER_00")]
    got = select_host_turns(turns, HOSTS, [(15.1, 18)], [], overlap_pad_s=0.25)
    assert got == []


def test_turn_in_music_dropped():
    turns = [Turn(0, 10, "SPEAKER_00")]
    assert select_host_turns(turns, HOSTS, [], [(2, 8)]) == []


def test_clean_host_turn_kept():
    turns = [Turn(30, 40, "SPEAKER_00")]
    got = select_host_turns(turns, HOSTS, [(100, 110)], [(200, 210)])
    assert len(got) == 1


def test_host_hours_sums_per_host():
    cands = [Candidate(0, 3600, "host_0"), Candidate(0, 1800, "host_1"),
             Candidate(0, 1800, "host_0")]
    hours = host_hours(cands)
    assert hours["host_0"] == pytest.approx(1.5)
    assert hours["host_1"] == pytest.approx(0.5)


def test_guest_only_episode_yields_nothing():
    turns = [Turn(0, 10, "SPEAKER_02")]
    assert select_host_turns(turns, HOSTS, [], []) == []
