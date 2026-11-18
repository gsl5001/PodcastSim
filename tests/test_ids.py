"""Episode identity (plan S3)."""

import pytest

from podcastsim.ids import digest_of, episode_id, file_sha1, slugify

pytestmark = pytest.mark.unit


def test_slugify_basic():
    assert slugify("Episode 412 - Lighthouses.mp3") == "episode-412-lighthouses"
    assert slugify("12.mp3") == "12"


def test_slugify_degenerate_names():
    assert slugify("___.mp3") == "untitled"
    assert slugify("") == "untitled"


def test_slugify_truncates_without_trailing_hyphen():
    s = slugify("a-" * 100, max_len=10)
    assert len(s) <= 10
    assert not s.endswith("-")


def test_episode_id_keys_on_bytes_not_filename(tmp_path):
    """Plan S3: 'Never key on filenames alone (duplicates and re-uploads
    are common).' Same bytes under two names must share a digest."""
    a = tmp_path / "12.mp3"
    b = tmp_path / "re-upload of 12.mp3"
    a.write_bytes(b"identical audio bytes")
    b.write_bytes(b"identical audio bytes")

    assert digest_of(episode_id(a)) == digest_of(episode_id(b))
    # ...but the slug still distinguishes them for a human reading `ls`.
    assert episode_id(a) != episode_id(b)


def test_episode_id_differs_on_content(tmp_path):
    a = tmp_path / "x.mp3"
    b = tmp_path / "y.mp3"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert digest_of(episode_id(a)) != digest_of(episode_id(b))


def test_episode_id_shape(tmp_path):
    p = tmp_path / "Episode 7.mp3"
    p.write_bytes(b"abc")
    eid = episode_id(p)
    digest, _, slug = eid.partition("__")
    assert len(digest) == 12
    assert slug == "episode-7"


def test_file_sha1_matches_hashlib_over_chunks(tmp_path):
    """Streaming in 1 MiB blocks must equal a one-shot hash."""
    import hashlib

    p = tmp_path / "big.bin"
    payload = b"\xa5" * (3 * (1 << 20) + 17)  # spans several chunks, unaligned
    p.write_bytes(payload)
    assert file_sha1(p) == hashlib.sha1(payload).hexdigest()
