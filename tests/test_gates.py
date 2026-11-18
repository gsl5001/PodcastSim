"""Automated gate evaluation (plan S12)."""

import json

import pytest

from podcastsim.config import Config
from podcastsim.gates import (
    ESCALATE,
    NA,
    PASS,
    WARN,
    gate_s0,
    gate_s1,
    load_manifests,
    summarise,
)

pytestmark = pytest.mark.unit

CFG = Config.load("configs/pipeline.yaml")


def man(stage: str, metrics: dict, wall: float = 60.0) -> dict:
    return {"stage": stage, "episode_id": "x", "input_hash": "i",
            "params_hash": "p", "wall_s": wall, "metrics": metrics,
            "versions": {}, "warnings": []}


def status(checks, name):
    return next(c.status for c in checks if c.name == name)


# --- S0 ----------------------------------------------------------------------

def test_s0_all_in_band_passes():
    ms = [man("s0_ingest", {"speech_ratio": 0.8, "music_seconds": 30, "duration_s": 3600})
          for _ in range(5)]
    checks = gate_s0(ms, CFG)
    assert status(checks, "speech_ratio in band") == PASS
    assert status(checks, "music detected") == PASS


def test_s0_saturated_speech_ratio_warns():
    """The real fixtures: 0.964-0.999, above the 0.98 ceiling."""
    ms = [man("s0_ingest", {"speech_ratio": r, "music_seconds": 0, "duration_s": 3600})
          for r in (0.964, 0.980, 0.997, 0.999)]
    assert status(gate_s0(ms, CFG), "speech_ratio in band") == WARN


def test_s0_unmeasured_segmentation_is_na_not_pass():
    """An unrun backend must never look like a pass."""
    ms = [man("s0_ingest", {"speech_ratio": None, "music_seconds": None, "duration_s": 3600})]
    checks = gate_s0(ms, CFG)
    assert status(checks, "speech_ratio in band") == NA
    assert status(checks, "music detected") == NA


def test_s0_no_manifests_is_na():
    assert gate_s0([], CFG)[0].status == NA


def test_s0_wall_rate_uses_audio_hours():
    ms = [man("s0_ingest", {"speech_ratio": 0.8, "music_seconds": 1, "duration_s": 3600}, wall=20)]
    assert status(gate_s0(ms, CFG), "wall per audio-hour") == PASS
    ms = [man("s0_ingest", {"speech_ratio": 0.8, "music_seconds": 1, "duration_s": 3600}, wall=200)]
    assert status(gate_s0(ms, CFG), "wall per audio-hour") == WARN


# --- S1 ----------------------------------------------------------------------

def test_s1_gold_checks_are_na_until_gold_exists():
    checks = gate_s1([man("s1_diarize", {"overlap_ratio": 0.05, "duration_s": 3600})], CFG)
    assert status(checks, "DER vs gold set") == NA
    assert status(checks, "JER vs gold set") == NA


def test_s1_overlap_in_band_passes():
    ms = [man("s1_diarize", {"overlap_ratio": 0.05, "n_dominant_speakers": 3, "duration_s": 3600})]
    assert status(gate_s1(ms, CFG), "overlap ratio") == PASS


def test_s1_overlap_in_show_range_passes():
    """This show runs 0.12-0.30 overlap (heavy but real crosstalk, per the owner),
    now inside the recalibrated 0.05-0.35 pass band."""
    ms = [man("s1_diarize", {"overlap_ratio": 0.30, "n_dominant_speakers": 3, "duration_s": 3600})]
    assert status(gate_s1(ms, CFG), "overlap ratio") == PASS


def test_s1_overlap_escalates_only_on_gross_failure():
    """Above 0.50 suggests music/noise leaking as a speaker, not crosstalk."""
    ms = [man("s1_diarize", {"overlap_ratio": 0.55, "n_dominant_speakers": 3, "duration_s": 3600})]
    assert status(gate_s1(ms, CFG), "overlap ratio") == ESCALATE


def test_s1_worst_episode_drives_the_gate():
    ms = [man("s1_diarize", {"overlap_ratio": r, "n_dominant_speakers": 3, "duration_s": 3600})
          for r in (0.10, 0.15, 0.60)]
    assert status(gate_s1(ms, CFG), "overlap ratio") == ESCALATE


def test_s1_duo_episodes_warn_not_escalate():
    """The show began as a duo, so 2 dominant speakers is legitimate."""
    ms = [man("s1_diarize", {"overlap_ratio": 0.05, "n_dominant_speakers": 2, "duration_s": 3600})]
    assert status(gate_s1(ms, CFG), "dominant speakers in {3,4}") == WARN


# --- summary + loading -------------------------------------------------------

def test_summarise_reports_worst_status():
    from podcastsim.gates import Check
    assert summarise([Check("a", PASS, "")]) == PASS
    assert summarise([Check("a", PASS, ""), Check("b", WARN, "")]) == WARN
    assert summarise([Check("a", WARN, ""), Check("b", ESCALATE, "")]) == ESCALATE


def test_summarise_counts_unevaluable():
    from podcastsim.gates import Check
    assert "1 not evaluable" in summarise([Check("a", PASS, ""), Check("b", NA, "")])


def test_load_manifests_skips_corrupt(tmp_path):
    good = tmp_path / "ep1" / "s0_ingest"
    good.mkdir(parents=True)
    (good / "_DONE.json").write_text(json.dumps(man("s0_ingest", {"speech_ratio": 0.8})))
    bad = tmp_path / "ep2" / "s0_ingest"
    bad.mkdir(parents=True)
    (bad / "_DONE.json").write_text("{truncated")

    assert len(load_manifests(tmp_path, "s0_ingest")) == 1


def test_load_manifests_empty_dir(tmp_path):
    assert load_manifests(tmp_path, "s0_ingest") == []
