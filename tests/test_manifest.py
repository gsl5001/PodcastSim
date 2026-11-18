"""Manifest and skip logic (plan S0 rule 3, S3 manifest contract).

These tests encode the resumability property the S2.2 infra gate checks:
"a deliberately killed session followed by re-run must resume with zero
re-processing of completed episodes".
"""

import json

import pytest

from podcastsim.manifest import (
    MANIFEST_NAME,
    Manifest,
    StageGuard,
    canonical_hash,
    input_hash,
    read_manifest,
    write_manifest,
)

pytestmark = pytest.mark.unit

EP = "abc123def456__episode-7"


def guard(tmp_path, *, params=None, inputs=(), stage="s1_diarize"):
    return StageGuard(tmp_path / "work", EP, stage, inputs=inputs, params=params)


def test_canonical_hash_is_key_order_independent():
    """Reordering pipeline.yaml keys must not invalidate finished work."""
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_canonical_hash_detects_value_change():
    assert canonical_hash({"der_pass": 0.10}) != canonical_hash({"der_pass": 0.11})


def test_canonical_hash_distinguishes_types():
    assert canonical_hash({"n": 1}) != canonical_hash({"n": "1"})


def test_input_hash_uses_content_for_files(tmp_path):
    p = tmp_path / "a.flac"
    p.write_bytes(b"one")
    first = input_hash([p])
    p.write_bytes(b"two")
    assert input_hash([p]) != first


def test_input_hash_accepts_opaque_tokens():
    """Downstream stages cite the upstream hash instead of re-reading Drive."""
    assert input_hash(["upstream:deadbeef"]) == input_hash(["upstream:deadbeef"])
    assert input_hash(["upstream:deadbeef"]) != input_hash(["upstream:cafe"])


def test_input_hash_tolerates_very_long_token():
    """A token longer than PATH_MAX must not raise ENAMETOOLONG."""
    assert input_hash(["x" * 10_000])


def test_fresh_stage_is_not_skipped(tmp_path):
    assert guard(tmp_path).should_skip() is False
    assert guard(tmp_path).skip_reason() == "no manifest"


def test_completed_stage_is_skipped(tmp_path):
    with guard(tmp_path, params={"threshold": 0.6}) as g:
        g.metrics["n_speakers"] = 4

    again = guard(tmp_path, params={"threshold": 0.6})
    assert again.should_skip() is True
    assert again.skip_reason() is None
    assert again.existing.metrics["n_speakers"] == 4


def test_param_change_forces_rerun(tmp_path):
    """Plan S0.3: 'Re-running skips completed work unless the parameter hash changed.'"""
    with guard(tmp_path, params={"threshold": 0.6}):
        pass
    changed = guard(tmp_path, params={"threshold": 0.7})
    assert changed.should_skip() is False
    assert changed.skip_reason() == "params or code changed"


def test_input_change_forces_rerun(tmp_path):
    src = tmp_path / "audio16k.flac"
    src.write_bytes(b"v1")
    with guard(tmp_path, inputs=[src], params={"t": 1}):
        pass
    src.write_bytes(b"v2")
    g = guard(tmp_path, inputs=[src], params={"t": 1})
    assert g.should_skip() is False
    assert g.skip_reason() == "inputs changed"


def test_failed_stage_writes_no_manifest(tmp_path):
    """The kill-and-resume property: a crash must leave nothing behind, so the
    episode reruns and no *other* episode is redone."""
    g = guard(tmp_path, params={"t": 1})
    with pytest.raises(RuntimeError):
        with g:
            raise RuntimeError("CUDA OOM")

    assert not (g.dir / MANIFEST_NAME).exists()
    assert guard(tmp_path, params={"t": 1}).should_skip() is False


def test_manifest_records_contract_fields(tmp_path):
    with guard(tmp_path, params={"t": 1}) as g:
        g.versions.update({"pyannote.audio": "4.0.7", "gpu": "T4"})
        g.metrics.update({"overlap_ratio": 0.041})
        g.warn("speech_ratio near lower bound")

    m = read_manifest(tmp_path / "work", EP, "s1_diarize")
    assert m.stage == "s1_diarize"
    assert m.episode_id == EP
    assert m.versions["gpu"] == "T4"          # S14.14: record the GPU
    assert m.metrics["overlap_ratio"] == 0.041
    assert m.warnings == ["speech_ratio near lower bound"]
    assert m.started and m.finished
    assert m.wall_s >= 0.0


def test_corrupt_manifest_is_treated_as_not_done(tmp_path):
    """A runtime killed mid-write must not wedge the pipeline."""
    with guard(tmp_path, params={"t": 1}):
        pass
    path = (tmp_path / "work") / EP / "s1_diarize" / MANIFEST_NAME
    path.write_text('{"stage": "s1_diar')  # truncated JSON

    assert read_manifest(tmp_path / "work", EP, "s1_diarize") is None
    assert guard(tmp_path, params={"t": 1}).should_skip() is False


def test_write_leaves_no_temp_files(tmp_path):
    with guard(tmp_path, params={"t": 1}) as g:
        pass
    leftovers = [p.name for p in g.dir.iterdir() if ".tmp" in p.name]
    assert leftovers == []


def test_manifest_json_is_readable_and_small(tmp_path):
    """S14.13: keep _DONE.json small -- Drive is slow for many small files."""
    with guard(tmp_path, params={"t": 1}) as g:
        g.metrics["speech_ratio"] = 0.87
    path = (tmp_path / "work") / EP / "s1_diarize" / MANIFEST_NAME
    payload = json.loads(path.read_text())
    assert payload["metrics"]["speech_ratio"] == 0.87
    assert path.stat().st_size < 4096


def test_stages_are_independent(tmp_path):
    """Completing s0 must not mark s1 done for the same episode."""
    with guard(tmp_path, stage="s0_ingest", params={"t": 1}):
        pass
    assert guard(tmp_path, stage="s0_ingest", params={"t": 1}).should_skip() is True
    assert guard(tmp_path, stage="s1_diarize", params={"t": 1}).should_skip() is False


def test_roundtrip_ignores_unknown_fields(tmp_path):
    """Forward compatibility: a manifest from a newer version must still load."""
    m = Manifest(stage="s0_ingest", episode_id=EP, input_hash="i", params_hash="p")
    write_manifest(tmp_path / "work", m)
    path = (tmp_path / "work") / EP / "s0_ingest" / MANIFEST_NAME
    d = json.loads(path.read_text())
    d["future_field"] = {"unexpected": True}
    path.write_text(json.dumps(d))

    loaded = read_manifest(tmp_path / "work", EP, "s0_ingest")
    assert loaded is not None and loaded.episode_id == EP


# --- code_version ------------------------------------------------------------

def test_code_version_change_forces_rerun(tmp_path):
    """Fixing a bug in a stage's code must invalidate its manifests.

    Without this, an episode "done" by a broken decoder stays done forever --
    which is exactly what happened when the cover-art bug was fixed.
    """
    work = tmp_path / "work"
    StageGuard(work, EP, "s0_ingest", params={"t": 1}, code_version=1).__enter__().__exit__(None, None, None)

    same = StageGuard(work, EP, "s0_ingest", params={"t": 1}, code_version=1)
    assert same.should_skip() is True

    bumped = StageGuard(work, EP, "s0_ingest", params={"t": 1}, code_version=2)
    assert bumped.should_skip() is False
    assert bumped.skip_reason() == "params or code changed"


def test_code_version_recorded_in_manifest(tmp_path):
    with StageGuard(tmp_path / "work", EP, "s0_ingest", params={}, code_version=3):
        pass
    m = read_manifest(tmp_path / "work", EP, "s0_ingest")
    assert m.versions["code_version"] == 3
