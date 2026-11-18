"""Config loading.

Plan S3 config contract: "every threshold in this document must be a key in
`configs/pipeline.yaml` with the default given here; gates read from config so
calibration (S12.3) is a config change, not a code change."

The important structural property is that `stage_params()` returns *only* that
stage's subtree. Stage params feed the manifest's `params_hash`, so recalibrating
S4's gates must not invalidate 200 hours of completed S1 diarization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/pipeline.yaml")
DEFAULT_HOSTS = Path("configs/hosts.yaml")

STAGES = (
    "s0_ingest",
    "s1_diarize",
    "s2_identify",
    "s3_dataset",
    "s4_voices",
    "s5_render",
)


class ConfigError(ValueError):
    """Raised for a malformed or incomplete config."""


class _Missing:
    """Sentinel: distinguishes "no default given" from a default of None."""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Config":
        path = Path(path)
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError as e:
            raise ConfigError(f"config not found: {path}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        cfg = cls(data=data, path=path)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Every pipeline stage must have a (possibly empty) section."""
        missing = [s for s in STAGES if s not in self.data]
        if missing:
            raise ConfigError(
                f"{self.path}: missing stage sections: {', '.join(missing)}"
            )

    def stage_params(self, stage: str) -> dict[str, Any]:
        """The subtree hashed into a stage's `params_hash`.

        Deliberately excludes every other stage's section.
        """
        if stage not in STAGES:
            raise ConfigError(f"unknown stage {stage!r}; expected one of {STAGES}")
        section = self.data.get(stage) or {}
        if not isinstance(section, dict):
            raise ConfigError(f"config section {stage!r} must be a mapping")
        return section

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Fetch `a.b.c`. Raises ConfigError if absent and no default given."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if isinstance(default, _Missing):
                    raise ConfigError(f"missing config key: {dotted}")
                return default
            node = node[part]
        return node


@dataclass(frozen=True)
class Host:
    host_id: str
    display_name: str
    enrollment_clips: list[str]


def load_hosts(path: str | Path = DEFAULT_HOSTS) -> list[Host]:
    """Load `configs/hosts.yaml` (host_id, display name, enrollment clip list)."""
    path = Path(path)
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError as e:
        raise ConfigError(f"hosts config not found: {path}") from e

    entries = data.get("hosts")
    if not isinstance(entries, list):
        raise ConfigError(f"{path}: expected a top-level 'hosts:' list")

    hosts: list[Host] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or "host_id" not in e:
            raise ConfigError(f"{path}: hosts[{i}] needs a 'host_id'")
        hid = str(e["host_id"])
        if hid in seen:
            raise ConfigError(f"{path}: duplicate host_id {hid!r}")
        seen.add(hid)
        hosts.append(
            Host(
                host_id=hid,
                display_name=str(e.get("display_name", hid)),
                enrollment_clips=list(e.get("enrollment_clips") or []),
            )
        )
    return hosts
