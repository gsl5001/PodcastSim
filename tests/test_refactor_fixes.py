"""Tests for the 2026-11 refactor: locks the bug fixes and the new seams.

Covers the S3->S4 aggregation bridge, clip_quality accepting an in-memory array,
relink_speakers surfacing global centroids (so long episodes keep embeddings),
the stage dependency guards, the stages.load error, the gate registry, RTTM
parsing, and s4_refbank reference selection.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# --- S3 -> S4 aggregation bridge -------------------------------------------

def test_aggregate_host_metadata_groups_by_host(tmp_path):
    from podcastsim.s3_clips import aggregate_host_metadata

    work = tmp_path / "work"
    for ep, rows in {
        "ep1": [{"clip": "a.flac", "host": "host_0", "text": "hi there friend"},
                {"clip": "b.flac", "host": "host_1", "text": "yeah for sure"}],
        "ep2": [{"clip": "c.flac", "host": "host_0", "text": "another one here"}],
    }.items():
        d = work / ep / "s3_dataset"
        d.mkdir(parents=True)
        (d / "clips.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    hosts = tmp_path / "hosts"
    counts = aggregate_host_metadata(work, hosts)

    assert counts == {"host_0": 2, "host_1": 1}
    h0 = [json.loads(l) for l in (hosts / "host_0" / "metadata.jsonl").read_text().splitlines()]
    assert {r["clip"] for r in h0} == {"a.flac", "c.flac"}


def test_aggregate_is_idempotent_and_skips_blank_lines(tmp_path):
    from podcastsim.s3_clips import aggregate_host_metadata

    work = tmp_path / "work"
    d = work / "ep1" / "s3_dataset"
    d.mkdir(parents=True)
    (d / "clips.jsonl").write_text(
        json.dumps({"clip": "a.flac", "host": "host_0", "text": "x y"}) + "\n\n")
    hosts = tmp_path / "hosts"

    assert aggregate_host_metadata(work, hosts) == {"host_0": 1}
    # re-running rewrites from scratch, not append
    assert aggregate_host_metadata(work, hosts) == {"host_0": 1}
    lines = (hosts / "host_0" / "metadata.jsonl").read_text().splitlines()
    assert len([l for l in lines if l.strip()]) == 1


# --- clip_quality accepts an in-memory array (dedup of _cq_from_array) ------

def test_clip_metrics_accepts_array_and_path(tmp_path):
    from podcastsim.clip_quality import clip_metrics

    rng = np.random.default_rng(0)
    x = rng.standard_normal(24000).astype(np.float32) * 0.1
    m = clip_metrics(x, sr=24000)
    assert set(m) == {"snr_db", "flatness", "clip_frac", "rms"}
    assert 0.0 <= m["clip_frac"] <= 1.0


def test_clip_metrics_flags_clipping():
    from podcastsim.clip_quality import clip_metrics, quality_score

    x = np.ones(24000, dtype=np.float32)        # fully pinned -> clipped
    m = clip_metrics(x, sr=24000)
    assert m["clip_frac"] > 0.01
    assert quality_score(m) == -1e9             # hard veto on clipping


# --- relink_speakers surfaces centroids (fixes long-episode empty embeddings)

def test_relink_returns_global_centroids():
    from podcastsim.chunking import relink_speakers

    v = np.array([1.0, 0.0], dtype=np.float32)
    results = [
        {"index": 0, "offset": 0.0, "turns": [(0.0, 1.0, "S0")], "embeddings": {"S0": v}},
        {"index": 1, "offset": 10.0, "turns": [(0.0, 1.0, "S0")], "embeddings": {"S0": v}},
    ]
    _, stats = relink_speakers(results, link_threshold=0.7)
    assert stats["n_global_speakers"] == 1
    assert "global_centroids" in stats and len(stats["global_centroids"]) == 1


# --- stage dependency guards (resumability contract) ------------------------

def _ep():
    from podcastsim.episodes import Episode
    return Episode(episode_id="abc123__x", source="x.mp3", size=0, mtime_ns=0)


def test_s1_requires_s0(tmp_path):
    from podcastsim.config import Config
    from podcastsim.stages import s1_diarize

    paths = {"work": tmp_path / "work", "raw": tmp_path / "raw"}
    with pytest.raises(RuntimeError, match="requires s0_ingest"):
        s1_diarize.run_episode(_ep(), Config.load("configs/pipeline.yaml"), paths)


def test_s3_requires_upstream(tmp_path):
    from podcastsim.config import Config
    from podcastsim.stages import s3_dataset

    paths = {"work": tmp_path / "work", "raw": tmp_path / "raw",
             "hosts": tmp_path / "hosts"}
    with pytest.raises(RuntimeError, match="requires s1_diarize"):
        s3_dataset.run_episode(_ep(), Config.load("configs/pipeline.yaml"), paths)


def test_stages_load_unknown_raises():
    from podcastsim.stages import load

    with pytest.raises(ValueError, match="unknown stage"):
        load("s9_nope")


def test_stage_modules_match_config_stages():
    from podcastsim.config import STAGES
    from podcastsim.stages import STAGE_MODULES

    assert tuple(STAGE_MODULES) == tuple(STAGES)


# --- gate registry + CLI validation ----------------------------------------

def test_gates_registry_has_known_stages():
    from podcastsim.gates import GATES

    assert set(GATES) == {"s0_ingest", "s1_diarize"}


def test_gates_cli_rejects_stage_without_gate(tmp_path):
    from typer.testing import CliRunner
    from podcastsim.cli import app

    res = CliRunner().invoke(app, ["gates", "--stage", "s2_identify"])
    assert res.exit_code == 2
    assert "no gate defined" in res.stdout


# --- RTTM parsing (read side of a format whose write side was tested) -------

def test_rttm_round_trip(tmp_path):
    from podcastsim.stages.s1_diarize import write_rttm
    from podcastsim.rttm import read_rttm, turns_by_speaker

    p = tmp_path / "x.rttm"
    write_rttm(p, "ep1", [(1.5, 4.0, "A"), (5.0, 6.0, "B"), (7.0, 8.0, "A")])
    turns = read_rttm(p)
    assert [t.speaker for t in turns] == ["A", "B", "A"]
    assert turns[0].start == 1.5 and abs(turns[0].duration - 2.5) < 1e-9
    assert set(turns_by_speaker(turns)) == {"A", "B"}


def test_rttm_skips_malformed_lines(tmp_path):
    from podcastsim.rttm import read_rttm

    p = tmp_path / "x.rttm"
    p.write_text("not a speaker line\nSPEAKER ep1 1 2.0 3.0 <NA> <NA> A <NA>\nSHORT ep1\n")
    turns = read_rttm(p)
    assert len(turns) == 1 and turns[0].speaker == "A"


# --- s4_refbank reference selection ----------------------------------------

def test_score_clip_prefers_duration_band_and_rejects_short_text():
    from podcastsim.s4_refbank import score_clip

    good = {"duration": 10.0, "avg_logprob": -0.2, "text": "this is a clean sentence"}
    edge = {"duration": 3.5, "avg_logprob": -0.2, "text": "this is a clean sentence"}
    assert score_clip(good) > score_clip(edge)
    assert score_clip({"duration": 10.0, "avg_logprob": -0.2, "text": "too short"}) == -1e9
    assert score_clip({"duration": 25.0, "avg_logprob": -0.2, "text": "a b c d e"}) == -1e9


def test_build_reference_bank_missing_metadata_raises(tmp_path):
    from podcastsim.s4_refbank import build_reference_bank

    with pytest.raises(FileNotFoundError, match="aggregate_host_metadata"):
        build_reference_bank(tmp_path / "host_0")


def test_build_reference_bank_ranks_best_first(tmp_path):
    from podcastsim.s4_refbank import build_reference_bank

    hd = tmp_path / "host_0"
    hd.mkdir()
    rows = [
        {"clip": "best.flac", "duration": 10.0, "avg_logprob": -0.1, "text": "a b c d e f"},
        {"clip": "ok.flac", "duration": 7.0, "avg_logprob": -0.6, "text": "a b c d e"},
        {"clip": "rej.flac", "duration": 1.0, "avg_logprob": -0.1, "text": "a b c d e"},
    ]
    (hd / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    bank = build_reference_bank(hd, top_k=5)
    assert [r["clip"] for r in bank] == ["best.flac", "ok.flac"]   # rej excluded
