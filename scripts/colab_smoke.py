"""Minimal Colab smoke test: what hardware did we actually get, and what's on it?"""
import subprocess, sys, shutil, os

print(f"python      : {sys.version.split()[0]}")
print(f"cwd         : {os.getcwd()}")

total, used, free = shutil.disk_usage("/content")
print(f"/content    : {free/2**30:.1f} GiB free of {total/2**30:.1f} GiB")

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30)
    print(f"gpu         : {out.stdout.strip() or 'NONE (rc=%d)' % out.returncode}")
except FileNotFoundError:
    print("gpu         : NONE (no nvidia-smi -> CPU-only runtime)")

try:
    import torch
    print(f"torch       : {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
except ImportError:
    print("torch       : not preinstalled")

print(f"ffmpeg      : {shutil.which('ffmpeg') or 'absent'}")
print(f"drive mount : {'PRESENT' if os.path.isdir('/content/drive') else 'absent (needs interactive drivemount)'}")
print("SMOKE_OK")
