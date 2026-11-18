"""Stage 4 - Voice models per host.

Plan S9. Not yet implemented -- see the README for status.
"""

from __future__ import annotations

from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest

STAGE = "s4_voices"
PLAN_SECTION = "S9"


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    raise NotImplementedError(
        f"{STAGE} is not implemented yet (plan {PLAN_SECTION})"
    )
