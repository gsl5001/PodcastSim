"""Pull source episodes from Drive onto the VM.

Reads the wanted filenames from /content/fetch.json. Uses the READ-ONLY remote
(plan rule 2: sources must not be reachable by anything that can write).
"""

import json
import os
import subprocess
import time

_spec = json.load(open("/content/fetch.json"))
WANT = _spec["episodes"]
# Source remote:path. Default is the original read-only sources; overridable so a
# new dataset (e.g. a different podcast) can be fetched from its own location.
REMOTE = _spec.get("remote", "gdrive:PodcastSim/audio")
os.makedirs("/content/raw", exist_ok=True)

for name in WANT:
    dest = f"/content/raw/{name}"
    if os.path.exists(dest):
        print(f"  have {name} ({os.path.getsize(dest)/2**20:.1f} MiB)")
        continue
    t0 = time.time()
    r = subprocess.run(
        ["rclone", "copy", f"{REMOTE}/{name}", "/content/raw/",
         "--transfers", "4", "-q"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FAIL {name}: {r.stderr.strip()[:160]}")
        continue
    print(f"  got  {name} ({os.path.getsize(dest)/2**20:.1f} MiB) in {time.time()-t0:.1f}s")

total = sum(os.path.getsize(f"/content/raw/{f}") for f in os.listdir("/content/raw")
            if f.endswith(".mp3"))
print(f"raw total: {total/2**30:.2f} GiB")
