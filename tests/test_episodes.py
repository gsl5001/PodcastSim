"""Episode discovery and index (plan S14.13)."""

import json

import pytest

from podcastsim import episodes as ep_mod

pytestmark = pytest.mark.unit


@pytest.fixture
def archive(tmp_path):
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    (raw / "0.mp3").write_bytes(b"episode zero")
    (raw / "1.mp3").write_bytes(b"episode one")
    (raw / "notes.txt").write_bytes(b"not audio")
    return raw, work


def test_refresh_indexes_audio_only(archive):
    raw, work = archive
    found = ep_mod.refresh(raw, work)
    assert sorted(found) == ["0.mp3", "1.mp3"]


def test_refresh_writes_index_file(archive):
    raw, work = archive
    ep_mod.refresh(raw, work)
    payload = json.loads(ep_mod.index_path(work).read_text())
    assert sorted(payload["episodes"]) == ["0.mp3", "1.mp3"]


def test_index_roundtrips(archive):
    raw, work = archive
    written = ep_mod.refresh(raw, work)
    assert ep_mod.load_index(work) == written


def test_missing_index_is_empty(tmp_path):
    assert ep_mod.load_index(tmp_path / "nothing") == {}


def test_corrupt_index_is_empty(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    ep_mod.index_path(work).write_text("{not json")
    assert ep_mod.load_index(work) == {}


def test_unchanged_files_are_not_rehashed(archive, monkeypatch):
    """S14.13: re-scanning 200 h off Drive must not re-read every byte."""
    raw, work = archive
    ep_mod.refresh(raw, work)

    def boom(_):
        raise AssertionError("re-hashed an unchanged file")

    monkeypatch.setattr(ep_mod, "episode_id", boom)
    again = ep_mod.refresh(raw, work)
    assert sorted(again) == ["0.mp3", "1.mp3"]


def test_changed_file_is_rehashed(archive):
    raw, work = archive
    before = ep_mod.refresh(raw, work)["0.mp3"].episode_id
    (raw / "0.mp3").write_bytes(b"different content entirely")
    after = ep_mod.refresh(raw, work)["0.mp3"].episode_id
    assert after != before


def test_new_file_is_picked_up(archive):
    raw, work = archive
    ep_mod.refresh(raw, work)
    (raw / "2.mp3").write_bytes(b"episode two")
    assert "2.mp3" in ep_mod.refresh(raw, work)


def test_removed_file_leaves_index(archive):
    raw, work = archive
    ep_mod.refresh(raw, work)
    (raw / "1.mp3").unlink()
    assert sorted(ep_mod.refresh(raw, work)) == ["0.mp3"]


def test_duplicates_detected_by_content(tmp_path):
    raw, work = tmp_path / "raw", tmp_path / "work"
    raw.mkdir()
    (raw / "12.mp3").write_bytes(b"same bytes")
    (raw / "12 (copy).mp3").write_bytes(b"same bytes")
    (raw / "13.mp3").write_bytes(b"other")

    dups = ep_mod.duplicates(ep_mod.refresh(raw, work))
    assert len(dups) == 1
    # `source` is the raw relative path, not the slug.
    assert sorted(next(iter(dups.values()))) == ["12 (copy).mp3", "12.mp3"]


def test_no_false_duplicates(archive):
    raw, work = archive
    assert ep_mod.duplicates(ep_mod.refresh(raw, work)) == {}


def test_resolve_selectors(archive):
    raw, work = archive
    found = ep_mod.refresh(raw, work)
    one = found["0.mp3"]

    assert len(ep_mod.resolve(found, "all")) == 2
    assert ep_mod.resolve(found, "0.mp3") == [one]
    assert ep_mod.resolve(found, one.episode_id) == [one]
    assert ep_mod.resolve(found, one.digest[:6]) == [one]
    # stem match: "0" resolves "0.mp3" -- the bug that made the resilient
    # per-episode runner silently match nothing.
    assert ep_mod.resolve(found, "0") == [found["0.mp3"]]
    assert ep_mod.resolve(found, "nope") == []


def test_nested_directories_are_scanned(tmp_path):
    raw, work = tmp_path / "raw", tmp_path / "work"
    (raw / "2019").mkdir(parents=True)
    (raw / "2019" / "a.mp3").write_bytes(b"nested")
    found = ep_mod.refresh(raw, work)
    assert "2019/a.mp3" in found


# --- exclusions (the owner: drop troublesome episodes) --------------------------

from podcastsim.episodes import Episode, is_excluded, load_exclusions  # noqa: E402


def test_load_real_exclusions():
    ex = load_exclusions()
    assert "trailer.mp3" in ex
    assert ex["trailer.mp3"]  # has a reason


def test_load_exclusions_missing_file(tmp_path):
    assert load_exclusions(tmp_path / "none.yaml") == {}


def test_is_excluded_by_source():
    ep = Episode("abc123__68", "68.mp3", 1, 1)
    assert is_excluded(ep, {"68.mp3": "live"}) == "live"


def test_is_excluded_by_digest_prefix():
    ep = Episode("abc123def__68", "68.mp3", 1, 1)
    assert is_excluded(ep, {"abc123": "by digest"}) == "by digest"


def test_kept_episode_returns_none():
    ep = Episode("xyz__22", "22.mp3", 1, 1)
    assert is_excluded(ep, {"68.mp3": "live"}) is None


def test_empty_exclusions_keeps_everything():
    ep = Episode("xyz__22", "22.mp3", 1, 1)
    assert is_excluded(ep, {}) is None
