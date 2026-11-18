"""Spike (overnight): full-episode speaker-labelled transcripts.

For each episode: transcribe the WHOLE audio with faster-whisper (not just the
S3-filtered clips), then label each segment host_0/host_1 by max time-overlap
with the S1 diarization turns (exclusive.rttm) mapped through host_map.json.
This recovers the continuous conversation — including the short turns and rapid
back-and-forth that S3 dropped — for the personality spike.

Output: trih_state/transcripts/{episode_id}.txt  (lines "host_x: text", ordered).
Resumable (skips episodes already transcribed). Runs on Kaggle GPU (P100 ok).
"""
import json, os, re, shutil, subprocess, sys, time
import numpy as np
from pathlib import Path

WORK = "/kaggle/working"; RAW = f"{WORK}/raw"
DRIVE = "gdrive_rw:trih_state"; SRC = "gdrive_rw:trih_mp3s"


def sh(c, check=True):
    print(f"$ {c}", flush=True); return subprocess.run(c, shell=True, check=check)


def find_secrets():
    for r, _d, fs in os.walk("/kaggle/input"):
        if "rclone.conf" in fs:
            return r
    sys.exit("secrets not attached")


def read_rttm(path):
    """[(start, end, speaker)] from an RTTM file."""
    turns = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            s, dur = float(f[3]), float(f[4])
            turns.append((s, s + dur, f[7]))
    return turns


def decode_16k(path):
    """Decode to 16 kHz mono float32 via ffmpeg (avoids faster-whisper's PyAV path,
    which breaks on Kaggle's av version)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", "16000", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def speaker_at(turns, s, e):
    """RTTM speaker label with the most overlap of [s,e], or None."""
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = max(0.0, min(e, te) - max(s, ts))
        if ov > best_ov:
            best, best_ov = spk, ov
    return best


def main():
    sec = find_secrets()
    if not shutil.which("rclone"):
        sh("curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip "
           "&& cd /tmp && unzip -q r.zip && cp rclone-*/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone")
    os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
    shutil.copy(f"{sec}/rclone.conf", os.path.expanduser("~/.config/rclone/rclone.conf"))
    sh(f"{sys.executable} -m pip install -q -U transformers accelerate", check=False)
    if os.path.exists(f"{sec}/hf_token"):
        os.environ["HF_TOKEN"] = open(f"{sec}/hf_token").read().strip()

    # pull the small state we need: exclusive.rttm + host_map.json per episode
    os.makedirs(f"{WORK}/work", exist_ok=True)
    sh(f"rclone copy {DRIVE}/work {WORK}/work --include '*/s1_diarize/exclusive.rttm' "
       f"--include '*/s2_identify/host_map.json' -q --transfers 16", check=False)
    done = set(subprocess.run(f"rclone lsf {DRIVE}/transcripts", shell=True,
                              capture_output=True, text=True).stdout.split())

    work = Path(f"{WORK}/work")
    eps = []
    for d in sorted(work.iterdir()):
        rttm = d / "s1_diarize" / "exclusive.rttm"
        hm = d / "s2_identify" / "host_map.json"
        if rttm.exists() and hm.exists() and f"{d.name}.txt" not in done:
            eps.append(d)
    print(f"[full] {len(eps)} episodes to transcribe", flush=True)

    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="nyrahealth/CrisperWhisper",
                   torch_dtype=torch.float16, device=0,
                   token=os.environ.get("HF_TOKEN"))

    os.makedirs(RAW, exist_ok=True)
    for i, d in enumerate(eps, 1):
        # episode_id is "<hash>__<slug>"; source mp3 name is the slug + .mp3
        slug = d.name.split("__", 1)[1] if "__" in d.name else d.name
        name = f"{slug}.mp3"
        t0 = time.time()
        r = subprocess.run(f"rclone copy '{SRC}/{name}' {RAW}/ -q", shell=True)
        src = Path(RAW) / name
        if not src.exists():
            print(f"[full] {i}/{len(eps)} MISSING {name}", flush=True); continue
        turns = read_rttm(d / "s1_diarize" / "exclusive.rttm")
        host_map = json.loads((d / "s2_identify" / "host_map.json").read_text())
        audio = decode_16k(src)
        res = asr({"raw": audio, "sampling_rate": 16000},
                  chunk_length_s=30, stride_length_s=5, return_timestamps=True)
        lines = []
        for ch in res.get("chunks", []):
            ts = ch.get("timestamp") or (None, None)
            s0, e0 = ts[0], ts[1]
            if s0 is None:
                continue
            spk = speaker_at(turns, s0, e0 if e0 is not None else s0 + 1)
            host = host_map.get(spk, "other") if spk else "other"
            txt = (ch.get("text") or "").strip()
            if txt:
                lines.append(f"{host}: {txt}")
        out = Path(WORK) / f"{d.name}.txt"
        out.write_text("\n".join(lines))
        sh(f"rclone copy {out} {DRIVE}/transcripts/ -q", check=False)
        src.unlink(missing_ok=True)
        print(f"[full] {i}/{len(eps)} {d.name} {time.time()-t0:.0f}s, {len(lines)} segs", flush=True)
    print("[full] done", flush=True)


if __name__ == "__main__":
    main()
