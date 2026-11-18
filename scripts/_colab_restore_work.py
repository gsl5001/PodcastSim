"""Unpack previously-pulled stage state onto a fresh Colab VM.

Plan S0 rule 3 makes stages resumable via `_DONE.json`, but that only helps if
the manifests outlive the runtime. Free-tier VMs are reclaimed without warning,
so the laptop is the durable store and this restores it on bootstrap.
"""

import os
import tarfile

ARCHIVE = "/content/work_restore.tgz"
DEST = "/content/work"

os.makedirs(DEST, exist_ok=True)
if os.path.exists(ARCHIVE):
    with tarfile.open(ARCHIVE) as t:
        t.extractall(DEST, filter="data")
    os.remove(ARCHIVE)

n = sum(1 for _r, _d, fs in os.walk(DEST) for f in fs if f == "_DONE.json")
print(f"restored {n} manifests")
