"""Install rclone on the Colab VM so it can reach Drive directly.

With this in place the VM pulls source audio from Drive (instead of chunked
uploads through the Jupyter contents API, which caps around 32 MB) and writes
stage state straight back — plan S2.2's "state lives on Google Drive".

The config carries OAuth tokens, so it is written 0600 and never committed.
Two remotes, deliberately asymmetric:
    gdrive     drive.readonly  -- sources; cannot be modified by any bug here
    gdrive_rw  drive.file      -- state; cannot see anything it did not create
"""

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

URL = "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
DEST = "/usr/local/bin/rclone"

if shutil.which("rclone"):
    print("rclone already present:", shutil.which("rclone"))
else:
    zpath = "/tmp/rclone.zip"
    # downloads.rclone.org 403s the default Python-urllib UA -> send a browser UA.
    _req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(_req, timeout=120) as _r, open(zpath, "wb") as _f:
        shutil.copyfileobj(_r, _f)
    with zipfile.ZipFile(zpath) as z:
        binary = next(n for n in z.namelist() if n.endswith("/rclone"))
        with z.open(binary) as src, open(DEST, "wb") as out:
            shutil.copyfileobj(src, out)
    os.chmod(DEST, 0o755)
    print("installed rclone ->", DEST)

conf_dir = os.path.expanduser("~/.config/rclone")
os.makedirs(conf_dir, exist_ok=True)
staged = "/content/rclone.conf"
if os.path.exists(staged):
    shutil.move(staged, os.path.join(conf_dir, "rclone.conf"))
    os.chmod(os.path.join(conf_dir, "rclone.conf"), 0o600)

out = subprocess.run([DEST if os.path.exists(DEST) else "rclone", "listremotes"],
                     capture_output=True, text=True)
print("remotes:", out.stdout.strip().replace("\n", " ") or "NONE")
if "gdrive_rw:" not in out.stdout:
    sys.exit("rclone config missing gdrive_rw — was rclone.conf staged?")
