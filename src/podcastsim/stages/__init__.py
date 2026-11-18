"""Pipeline stages (plan S5-S11).

Each stage module exposes:

    run_episode(episode, cfg, paths, *, force=False) -> Manifest | None

returning None when the episode was skipped because a valid `_DONE.json`
already covered these inputs and params.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from ..config import STAGES

# Derived from the single source of truth in config.STAGES, so adding or
# renaming a stage is a one-line change there (the module path is conventional:
# podcastsim.stages.<stage>).
STAGE_MODULES = {stage: f"podcastsim.stages.{stage}" for stage in STAGES}


def load(stage: str) -> ModuleType:
    """Import and return a stage module by name, e.g. ``load("s1_diarize")``.

    Raises ValueError for an unknown stage (one not in config.STAGES).
    """
    try:
        return import_module(STAGE_MODULES[stage])
    except KeyError:
        raise ValueError(
            f"unknown stage {stage!r}; expected one of {sorted(STAGE_MODULES)}"
        ) from None
