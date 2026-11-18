#!/usr/bin/env bash
# Pull /content/work off a Colab session into the local data/work tree.
#
# Plan S2.2 wants stage state on Drive, never on the runtime disk, because
# free-tier runtimes are reclaimed without warning. `colab drivemount` is not
# agent-runnable, so the laptop is the durable store instead: the VM computes,
# we pull results after every batch, and colab_bootstrap.sh pushes them back.
#
# Manifests are small (S0: ~4 KB for 5 episodes) so this is cheap to do often.
# Usage: scripts/colab_pull_work.sh [session] [dest]
set -euo pipefail
SESSION="${1:-ps}"; DEST="${2:-data/work}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
mkdir -p "$DEST"

cat > /tmp/_pack_work.py <<'PY'
import os, tarfile
out = "/content/work_state.tgz"
if not os.path.isdir("/content/work"):
    print("NO_WORK"); raise SystemExit(0)
with tarfile.open(out, "w:gz") as t:
    for root, _, files in os.walk("/content/work"):
        for f in files:
            p = os.path.join(root, f)
            t.add(p, arcname=os.path.relpath(p, "/content/work"))
print("PACKED", os.path.getsize(out))
PY
./scripts/colab_exec.sh "$SESSION" /tmp/_pack_work.py 600 | grep -E "PACKED|NO_WORK" || true
colab --auth=adc download -s "$SESSION" /content/work_state.tgz /tmp/work_state.tgz >/dev/null
tar xzf /tmp/work_state.tgz -C "$DEST"
echo "[pull] $(find "$DEST" -name _DONE.json | wc -l | tr -d ' ') manifests now in $DEST"
