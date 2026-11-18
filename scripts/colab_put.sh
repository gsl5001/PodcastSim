#!/usr/bin/env bash
# Upload a large file to a Colab session, in chunks, verified by sha256.
#
# `colab upload` goes through the Jupyter contents API, which base64-encodes the
# payload and rejects large bodies: 32 MB succeeds, 46 MB returns 500. This
# splits, uploads the parts, reassembles on the VM, and checks the hash before
# removing the parts.
#
# Usage: scripts/colab_put.sh <session> <local-file> <remote-path> [chunk_mb]
set -euo pipefail

SESSION="$1"; LOCAL="$2"; REMOTE="$3"; CHUNK_MB="${4:-24}"
# `colab exec` gives up on the output stream after ~30 s while the kernel keeps
# running, so joining a large file looks like a hang. Always pass --timeout.
COLAB=(colab --auth=adc)
EXEC_TIMEOUT="${COLAB_EXEC_TIMEOUT:-600}"

[ -f "$LOCAL" ] || { echo "no such file: $LOCAL" >&2; exit 1; }

SUM=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
SIZE=$(wc -c < "$LOCAL" | tr -d ' ')
BASE=$(basename "$REMOTE")
STAGE="/content/.upload/${BASE}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
split -b "${CHUNK_MB}m" "$LOCAL" "$TMP/part."
N=$(ls "$TMP" | wc -l | tr -d ' ')
echo "[put] $LOCAL -> $REMOTE  (${SIZE} bytes, $N chunks of ${CHUNK_MB}MB)"

python3 - "$STAGE" <<'PY' > "$TMP/mk.py"
import sys
print(f'import os; os.makedirs("{sys.argv[1]}", exist_ok=True); print("staged")')
PY
"${COLAB[@]}" exec -s "$SESSION" --timeout "$EXEC_TIMEOUT" -f "$TMP/mk.py" >/dev/null

i=0
for p in "$TMP"/part.*; do
  i=$((i+1))
  printf "\r[put] chunk %d/%d" "$i" "$N"
  "${COLAB[@]}" upload -s "$SESSION" "$p" "$STAGE/$(basename "$p")" >/dev/null
done
echo

cat > "$TMP/join.py" <<PY
import hashlib, os, glob, shutil
stage, remote, want, size = "$STAGE", "$REMOTE", "$SUM", $SIZE
os.makedirs(os.path.dirname(remote) or "/content", exist_ok=True)
parts = sorted(glob.glob(os.path.join(stage, "part.*")))
h = hashlib.sha256()
with open(remote, "wb") as out:
    for p in parts:
        with open(p, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk); out.write(chunk)
got, n = h.hexdigest(), os.path.getsize(remote)
if got != want or n != size:
    raise SystemExit(f"MISMATCH: {n}b {got[:12]} != {size}b {want[:12]}")
shutil.rmtree(stage, ignore_errors=True)
print(f"OK {remote} {n} bytes sha256={got[:12]}")
PY
"${COLAB[@]}" exec -s "$SESSION" --timeout "$EXEC_TIMEOUT" -f "$TMP/join.py"
