"""CLI surface (plan S3: `podcastsim <stage> --episodes ...`)."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from podcastsim.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    """A throwaway project dir with the real configs and two fake episodes."""
    (tmp_path / "configs").mkdir()
    for name in ("pipeline.yaml", "hosts.yaml"):
        shutil.copy(REPO / "configs" / name, tmp_path / "configs" / name)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "0.mp3").write_bytes(b"episode zero")
    (raw / "1.mp3").write_bytes(b"episode one")
    return tmp_path


def run(project, *args):
    """Invoke the CLI with `project` as cwd."""
    import os

    prev = os.getcwd()
    os.chdir(project)
    try:
        return runner.invoke(app, list(args))
    finally:
        os.chdir(prev)


def test_check_validates_real_config(project):
    r = run(project, "check")
    assert r.exit_code == 0
    assert "config OK" in r.stdout
    assert "hosts OK: 2 defined" in r.stdout


def test_check_reports_hosts(project):
    """Hosts come from clustering (S2), so `check` just confirms the 2 are defined;
    it no longer nags about enrollment clips (that approach was abandoned)."""
    r = run(project, "check")
    assert r.exit_code == 0
    assert "hosts OK: 2 defined" in r.stdout
    assert "enrollment clips not yet picked" not in r.stdout


def test_check_rejects_broken_config(project):
    (project / "configs" / "pipeline.yaml").write_text("s0_ingest:\n  a: 1\n")
    r = run(project, "check")
    assert r.exit_code == 2
    assert "config error" in r.stderr


def test_index_counts_episodes(project):
    r = run(project, "index")
    assert r.exit_code == 0
    assert "indexed 2 episodes" in r.stdout


def test_index_reports_duplicates(project):
    (project / "data" / "raw" / "dupe.mp3").write_bytes(b"episode zero")
    r = run(project, "index")
    assert r.exit_code == 0
    assert "duplicate digest" in r.stdout


def test_index_fails_without_raw_dir(project):
    shutil.rmtree(project / "data" / "raw")
    r = run(project, "index")
    assert r.exit_code == 2
    assert "raw directory not found" in r.stderr


def test_status_before_any_work(project):
    run(project, "index")
    r = run(project, "status")
    assert r.exit_code == 0
    assert "2 episodes indexed" in r.stdout
    # every stage shows 0 of 2 complete
    assert r.stdout.count("0/2") == 6   # 6 stages after dropping s6_qa


def test_status_without_index_is_graceful(project):
    r = run(project, "status")
    assert r.exit_code == 0
    assert "no episode index yet" in r.stdout


def test_run_requires_index(project):
    r = run(project, "run", "s0_ingest")
    assert r.exit_code == 2
    assert "empty episode index" in r.stderr


def test_run_rejects_unknown_stage(project):
    r = run(project, "run", "s9_nope")
    assert r.exit_code == 2
    assert "unknown stage" in r.stderr


def test_run_rejects_unmatched_selector(project):
    run(project, "index")
    r = run(project, "run", "s0_ingest", "--episodes", "zzz")
    assert r.exit_code == 2
    assert "no episodes matched" in r.stderr


def test_dry_run_lists_without_running(project):
    run(project, "index")
    r = run(project, "run", "s4_voices", "--dry-run")
    assert r.exit_code == 0
    assert "[todo] 0.mp3" in r.stdout
    assert "[todo] 1.mp3" in r.stdout


def test_run_reports_unimplemented_stage_clearly(project):
    """Stages still stubbed must say which plan section covers them.

    s0-s3 are implemented, so this checks s4_voices.
    """
    run(project, "index")
    r = run(project, "run", "s4_voices")
    assert r.exit_code == 3
    assert "not implemented yet" in r.stdout
    assert "plan S9" in r.stdout
