"""Run a pipeline stage on a Colab VM (plan S2.2 / S3 `scripts/run_stage.py`).

Reads its parameters from /content/run.json so it can be launched with a bare
`colab exec -f`, which takes no script arguments.
"""

import json
import os
import sys
import time

ROOT = "/content/podcastsim"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

# Kernel state persists across `colab exec`, so a freshly synced source file is
# IGNORED if the module was already imported -- sys.modules wins. That silently
# ran stage code from a previous sync: a CODE_VERSION bump had no effect and the
# episode was skipped as already-done. Purge the package before importing.
import importlib  # noqa: E402

for _name in [m for m in sys.modules if m == "podcastsim" or m.startswith("podcastsim.")]:
    del sys.modules[_name]
importlib.invalidate_caches()

cfgfile = "/content/run.json"
opts = json.load(open(cfgfile)) if os.path.exists(cfgfile) else {}
stage = opts.get("stage", "s0_ingest")
selector = opts.get("episodes", "all")

import shutil                                     # noqa: E402
import subprocess                                  # noqa: E402

from podcastsim import episodes as ep_mod          # noqa: E402
from podcastsim.config import Config               # noqa: E402
from podcastsim.stages import load as load_stage   # noqa: E402

cfg = Config.load("configs/pipeline.yaml")
paths = {k: v for k, v in cfg.get("paths").items()}
# VM-local layout: everything under /content, which has ~65 GiB.
paths.update({
    "raw": "/content/raw",
    "work": "/content/work",
    "cache": "/content/cache",
    "models": "/content/models",
    "out": "/content/out",
})
from pathlib import Path                            # noqa: E402
paths = {k: Path(v) for k, v in paths.items()}

# Where per-episode stage state is persisted. Overridable via run.json so a new
# dataset (e.g. a different podcast) keeps its state separate from any other.
DRIVE_STATE = opts.get("drive_state", "gdrive_rw:podcastsim_state/work")


def drive_push(episode_id: str) -> str:
    """Persist one episode's stage state to Drive immediately.

    Plan S2.2 puts state on Drive precisely because runtimes are ephemeral;
    pushing per episode (rather than per batch) means a reclaimed VM loses at
    most the episode in flight. Manifests are KB, so this is nearly free.
    """
    if not shutil.which("rclone"):
        return "rclone absent"
    src = paths["work"] / episode_id
    if not src.is_dir():
        return "nothing to push"
    r = subprocess.run(
        ["rclone", "copy", str(src), f"{DRIVE_STATE}/{episode_id}", "-q"],
        capture_output=True, text=True,
    )
    return "pushed" if r.returncode == 0 else f"push failed: {r.stderr.strip()[:120]}"


known = ep_mod.refresh(paths["raw"], paths["work"])
print(f"[run] indexed {len(known)} episode(s) in {paths['raw']}")

selected = ep_mod.resolve(known, selector)

exclusions = ep_mod.load_exclusions()
_kept = []
for _ep in selected:
    _reason = ep_mod.is_excluded(_ep, exclusions)
    if _reason:
        print(f"[run] excluded {_ep.source}: {_reason}", flush=True)
    else:
        _kept.append(_ep)
selected = _kept
module = load_stage(stage)
ran = skipped = failed = 0
t0 = time.time()

for ep in selected:
    tick = time.time()
    try:
        result = module.run_episode(ep, cfg, paths, force=opts.get("force", False))
    except Exception as e:
        failed += 1
        print(f"[run] FAIL {ep.source}: {type(e).__name__}: {e}", flush=True)
        continue
    if result is None:
        skipped += 1
        print(f"[run] skip {ep.source}", flush=True)
    else:
        ran += 1
        pushed = drive_push(ep.episode_id)
        print(f"[run] done {ep.source} in {time.time()-tick:.1f}s [{pushed}] "
              f"metrics={json.dumps(result.metrics)}", flush=True)

print(f"[run] {stage}: {ran} ran, {skipped} skipped, {failed} failed "
      f"in {time.time()-t0:.1f}s")
if failed:
    sys.exit(1)
