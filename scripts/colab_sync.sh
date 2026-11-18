#!/usr/bin/env bash
# Push the orchestrator package + configs to a Colab session.
# One tarball rather than N uploads: the source tree is small but has ~15 files,
# and each `colab upload` is a separate round trip.
set -euo pipefail
SESSION="${1:-ps}"
REMOTE_ROOT="${2:-/content/podcastsim}"
COLAB=(colab --auth=adc)

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar czf "$TMP/pkg.tgz" src configs scripts pyproject.toml
echo "[sync] $(wc -c < "$TMP/pkg.tgz" | tr -d ' ') bytes"
"${COLAB[@]}" upload -s "$SESSION" "$TMP/pkg.tgz" /content/pkg.tgz >/dev/null

cat > "$TMP/unpack.py" <<PY
import subprocess, sys, os, tarfile, shutil
root = "$REMOTE_ROOT"
shutil.rmtree(os.path.join(root, "src"), ignore_errors=True)
os.makedirs(root, exist_ok=True)
with tarfile.open("/content/pkg.tgz") as t:
    t.extractall(root, filter="data")  # block path-traversal/zip-slip
os.remove("/content/pkg.tgz")
# The orchestrator's own deps; everything ML is already on the image.
missing = []
for mod, pkg in (("yaml", "pyyaml"), ("typer", "typer")):
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    print("installed:", missing)
src = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)
import podcastsim
print("podcastsim", podcastsim.__version__, "at", root)
PY
"${COLAB[@]}" exec -s "$SESSION" --timeout 600 -f "$TMP/unpack.py"
