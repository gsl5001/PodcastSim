"""Per-episode, per-stage manifests and skip logic.

Plan S0 rule 3: "Idempotent, resumable stages. Each stage writes a per-episode
manifest (`_DONE.json`) with an input hash, parameter hash, package versions and
metrics. Re-running skips completed work unless the parameter hash changed. This
is non-negotiable because the compute environment (Colab) is ephemeral."

Contract (plan S3):

    data/work/{episode_id}/{stage}/_DONE.json

A `_DONE.json` exists **only** for work that finished successfully. A crashed or
killed stage leaves no manifest, so re-running redoes exactly that episode and
nothing else -- which is the kill-and-resume property the S2.2 infra gate tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from .ids import file_sha256

MANIFEST_NAME = "_DONE.json"


def canonical_hash(obj: Any) -> str:
    """Stable sha256 over arbitrary JSON-able data.

    `sort_keys` makes dict ordering irrelevant, so reordering keys in
    pipeline.yaml does not invalidate completed work. `default=str` keeps Paths
    and similar from exploding.
    """
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def input_hash(inputs: Iterable[str | Path]) -> str:
    """Combined hash of a stage's inputs.

    Each entry is either an existing file (hashed by content) or an opaque token
    (hashed as text). Tokens let a downstream stage cite the upstream manifest's
    hash instead of re-reading a multi-hundred-MB FLAC off Drive every run --
    plan S14.13 warns that Drive is slow for exactly this kind of access.
    """
    parts: list[str] = []
    for item in inputs:
        p = Path(item)
        # Guard against absurdly long "paths" -- a token can be any string, and
        # os.path.exists() raises ENAMETOOLONG rather than returning False.
        is_file = len(str(item)) < 4096 and p.is_file()
        parts.append(f"file:{file_sha256(p)}" if is_file else f"tok:{item}")
    return canonical_hash(parts)


@dataclass
class Manifest:
    """The contents of a `_DONE.json`."""

    stage: str
    episode_id: str
    input_hash: str
    params_hash: str
    versions: dict[str, Any] = field(default_factory=dict)
    started: str = ""
    finished: str = ""
    wall_s: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def stage_dir(work_root: str | Path, episode_id: str, stage: str) -> Path:
    return Path(work_root) / episode_id / stage


def manifest_path(work_root: str | Path, episode_id: str, stage: str) -> Path:
    return stage_dir(work_root, episode_id, stage) / MANIFEST_NAME


def read_manifest(work_root: str | Path, episode_id: str, stage: str) -> Manifest | None:
    """Return the manifest, or None if absent/unreadable.

    A truncated `_DONE.json` (killed mid-write on a dying runtime, despite the
    atomic write below) is treated as "not done" rather than raising -- the
    stage simply reruns.
    """
    path = manifest_path(work_root, episode_id, stage)
    try:
        with open(path) as fh:
            return Manifest.from_dict(json.load(fh))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, AttributeError, ValueError):
        # A JSON file that parses but is not a dict (list/string) raises
        # AttributeError in from_dict; treat any malformed manifest as "not done".
        return None


def write_manifest(work_root: str | Path, m: Manifest) -> Path:
    """Atomically write `_DONE.json` (tmp file + rename)."""
    d = stage_dir(work_root, m.episode_id, m.stage)
    d.mkdir(parents=True, exist_ok=True)
    final = d / MANIFEST_NAME
    tmp = d / f"{MANIFEST_NAME}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(m.to_dict(), fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)  # atomic within a filesystem
    return final


class StageGuard:
    """Skip-or-run wrapper around one (episode, stage) unit of work.

    Usage:

        guard = StageGuard(work_root, ep_id, "s0_ingest", inputs=[mp3], params=cfg)
        if guard.should_skip():
            return guard.existing
        with guard:
            ...do the work...
            guard.metrics["speech_ratio"] = 0.87

    On a clean exit the manifest is written; on an exception nothing is written
    and the exception propagates, so the episode is retried next run.
    """

    def __init__(
        self,
        work_root: str | Path,
        episode_id: str,
        stage: str,
        inputs: Iterable[str | Path] = (),
        params: Any = None,
        versions: dict[str, Any] | None = None,
        code_version: str | int = 0,
    ) -> None:
        self.work_root = Path(work_root)
        self.episode_id = episode_id
        self.stage = stage
        self.input_hash = input_hash(inputs)
        # `code_version` folds into params_hash so that fixing a bug in a
        # stage's *code* invalidates its manifests, not just editing its config.
        # Without this, a decode fix leaves already-"done" episodes carrying
        # data produced by the broken version. Bump the stage's CODE_VERSION
        # whenever its output would change.
        self.code_version = code_version
        self.params_hash = canonical_hash(
            {"params": params, "code_version": code_version}
        )
        self.versions = versions or {}
        self.metrics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self._t0: float | None = None
        self._started_iso: str = ""

    @property
    def dir(self) -> Path:
        return stage_dir(self.work_root, self.episode_id, self.stage)

    @property
    def existing(self) -> Manifest | None:
        return read_manifest(self.work_root, self.episode_id, self.stage)

    def should_skip(self) -> bool:
        """True when a prior run covered these exact inputs and params."""
        m = self.existing
        return (
            m is not None
            and m.input_hash == self.input_hash
            and m.params_hash == self.params_hash
        )

    def skip_reason(self) -> str | None:
        """Why this will rerun -- for logs and the runbook."""
        m = self.existing
        if m is None:
            return "no manifest"
        if m.input_hash != self.input_hash:
            return "inputs changed"
        if m.params_hash != self.params_hash:
            return "params or code changed"
        return None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __enter__(self) -> "StageGuard":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.monotonic()
        self._started_iso = _now_iso()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            return False  # propagate; deliberately leave no _DONE.json
        write_manifest(
            self.work_root,
            Manifest(
                stage=self.stage,
                episode_id=self.episode_id,
                input_hash=self.input_hash,
                params_hash=self.params_hash,
                versions={**self.versions, "code_version": self.code_version},
                started=self._started_iso,
                finished=_now_iso(),
                wall_s=round(time.monotonic() - (self._t0 or 0.0), 3),
                metrics=self.metrics,
                warnings=self.warnings,
            ),
        )
        return False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
