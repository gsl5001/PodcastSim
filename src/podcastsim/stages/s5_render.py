"""Stage 5 - Script format and episode rendering.

Plan S10. Not yet implemented -- see the README for status.
"""

from __future__ import annotations

from ..config import Config
from ..episodes import Episode
from ..manifest import Manifest

STAGE = "s5_render"
PLAN_SECTION = "S10"


def run_episode(
    episode: Episode, cfg: Config, paths: dict, *, force: bool = False
) -> Manifest | None:
    raise NotImplementedError(
        f"{STAGE} is not implemented yet (plan {PLAN_SECTION})"
    )
