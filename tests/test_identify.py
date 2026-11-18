"""Cross-episode host clustering (plan S7.2)."""

import numpy as np
import pytest

from podcastsim.identify import (
    Cluster,
    LocalSpeaker,
    cluster_speakers,
    summarise_clusters,
)

pytestmark = pytest.mark.unit


def spk(ep, label, vec, speech_s=120.0, date=None):
    return LocalSpeaker(ep, label, np.array(vec, dtype=np.float32), speech_s, date)


def test_three_hosts_across_episodes_form_three_clusters():
    """Three distinct voices, each appearing in several episodes."""
    h = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    speakers = []
    for ep in ("ep1", "ep2", "ep3"):
        for i, base in enumerate(h):
            jitter = base + np.random.default_rng(i).normal(0, 0.02, 3)
            speakers.append(spk(ep, f"SPEAKER_0{i}", jitter))
    clusters = cluster_speakers(speakers, sim_threshold=0.8)
    assert len([c for c in clusters if c.n_episodes >= 2]) == 3


def test_recurring_hosts_rank_above_one_off_guest():
    h1 = np.array([1, 0, 0]); h2 = np.array([0, 1, 0])
    speakers = [
        spk("ep1", "S0", h1, 600), spk("ep2", "S0", h1 * 0.99, 600),
        spk("ep1", "S1", h2, 500), spk("ep2", "S1", h2 * 0.98, 500),
        spk("ep1", "S2", np.array([0, 0, 1]), 90),   # a one-off guest
    ]
    clusters = cluster_speakers(speakers, sim_threshold=0.8)
    # top two clusters (by speech time) are the recurring hosts
    assert clusters[0].n_episodes == 2
    assert clusters[1].n_episodes == 2
    assert clusters[-1].total_speech_s == 90    # guest is smallest


def test_host_split_into_two_local_clusters_merges():
    """A host diarized as two local speakers in one episode should merge if
    their voices match -- the common S7.4 case."""
    h = np.array([1, 0, 0])
    speakers = [
        spk("ep1", "S0", h, 300),
        spk("ep1", "S1", h * 0.99, 200),   # same voice, split by diarizer
    ]
    clusters = cluster_speakers(speakers, sim_threshold=0.8)
    assert len(clusters) == 1
    assert len(clusters[0].members) == 2


def test_empty_input():
    assert cluster_speakers([], 0.8) == []


def test_single_speaker():
    clusters = cluster_speakers([spk("ep1", "S0", [1, 0, 0])], 0.8)
    assert len(clusters) == 1


def test_clusters_ranked_by_speech_time():
    speakers = [
        spk("ep1", "S0", [1, 0, 0], 100),
        spk("ep1", "S1", [0, 1, 0], 500),
        spk("ep1", "S2", [0, 0, 1], 300),
    ]
    clusters = cluster_speakers(speakers, sim_threshold=0.9)
    times = [c.total_speech_s for c in clusters]
    assert times == sorted(times, reverse=True)


def test_summarise_labels_top_n_as_hosts():
    speakers = [spk("ep1", f"S{i}", np.eye(4)[i], 500 - i * 100) for i in range(4)]
    clusters = cluster_speakers(speakers, sim_threshold=0.9)
    report = summarise_clusters(clusters, n_hosts=3)
    hosts = [r for r in report["clusters"] if r["candidate"] == "host"]
    guests = [r for r in report["clusters"] if r["candidate"] == "guest/other"]
    assert len(hosts) == 3 and len(guests) == 1


# --- host assignment (recurrence + speech share) -----------------------------

from podcastsim.identify import assign_hosts  # noqa: E402


def test_recurring_guest_is_not_promoted_to_host():
    """The real fixture bug: a 1-min voice recurring across 2 episodes, present
    in the duo-era episode, was being labelled a host."""
    speakers = [
        spk("ep1", "S0", [1, 0, 0], 880), spk("ep2", "S0", [1, 0, 0], 880),
        spk("ep1", "S1", [0, 1, 0], 350), spk("ep2", "S1", [0, 1, 0], 350),
        spk("ep1", "S2", [0, 0, 1], 30), spk("ep2", "S2", [0, 0, 1], 30),  # recurring guest
    ]
    clusters = cluster_speakers(speakers, sim_threshold=0.9)
    assignment = assign_hosts(clusters, min_episodes=2, n_hosts=3, min_speech_share=0.20)
    hosts = {h for ep in assignment.values() for h in ep.values() if h != "guest"}
    assert hosts == {"host_0", "host_1"}   # not 3
    assert assignment["ep1"]["S2"] == "guest"


def test_single_episode_voice_is_guest():
    speakers = [
        spk("ep1", "S0", [1, 0, 0], 600), spk("ep2", "S0", [1, 0, 0], 600),
        spk("ep1", "S1", [0, 1, 0], 90),   # one-off
    ]
    clusters = cluster_speakers(speakers, sim_threshold=0.9)
    assignment = assign_hosts(clusters, min_episodes=2)
    assert assignment["ep1"]["S1"] == "guest"


def test_assign_hosts_respects_n_hosts():
    """The project assumes TWO hosts (the default), so even three substantial
    recurring speakers yield only the top two by speech; the third is 'guest'.
    Passing n_hosts=3 explicitly still picks up all three."""
    speakers = []
    for ep in ("ep1", "ep2", "ep3"):
        speakers += [spk(ep, "S0", [1, 0, 0], 500), spk(ep, "S1", [0, 1, 0], 400),
                     spk(ep, "S2", [0, 0, 1], 300)]
    clusters = cluster_speakers(speakers, sim_threshold=0.9)

    default_hosts = {h for ep in assign_hosts(clusters).values()
                     for h in ep.values() if h != "guest"}
    assert default_hosts == {"host_0", "host_1"}          # 2 by default

    three = {h for ep in assign_hosts(clusters, n_hosts=3).values()
             for h in ep.values() if h != "guest"}
    assert three == {"host_0", "host_1", "host_2"}        # param still works


def test_empty_clusters_assign_nothing():
    assert assign_hosts([]) == {}
