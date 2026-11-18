"""S3 (dataset build) with CrisperWhisper verbatim transcripts, across every
episode that already has a host map (S2 done).

Verbatim = keeps the "um"/"uh"/false-starts plain Whisper strips, so the clones
can learn the hosts' real disfluent delivery (user's expressiveness request).

Per-episode + resumable: for each episode we fetch its one source MP3, cut +
transcribe, push work/{ep}/s3_dataset and the new clips to Drive, then delete
the local MP3. A session that dies is continued by re-running -- episodes whose
s3_dataset/_DONE.json already matches CODE_VERSION 3 skip. Runs on GPU.

"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

WORK = "/kaggle/working"; RAW = f"{WORK}/raw"; ROOT = f"{WORK}/podcastsim"
HOSTS = f"{WORK}/work_hosts"
DRIVE = "gdrive_rw:trih_state"
SRC_ORIG = "gdrive_rw:trih_mp3s"          # original 75 (read-only remote)
SRC_NEW = "gdrive_rw:trih_mp3s"              # (same; single source)


def sh(c, check=True):
    print(f"$ {c}", flush=True); return subprocess.run(c, shell=True, check=check)


def find_secrets():
    for r, _d, fs in os.walk("/kaggle/input"):
        if "rclone.conf" in fs:
            return r
    sys.exit("secrets not attached")


def fetch_source(name):
    """Pull one MP3 into RAW, trying the original remote then new_audio."""
    for remote in (SRC_ORIG, SRC_NEW):
        r = subprocess.run(f"rclone copy '{remote}/{name}' {RAW}/ -q",
                           shell=True)
        if r.returncode == 0 and os.path.exists(f"{RAW}/{name}"):
            return True
    return False


def main():
    sec = find_secrets()
    if not shutil.which("rclone"):
        sh("curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip "
           "&& cd /tmp && unzip -q r.zip && cp rclone-*/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone")
    os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
    shutil.copy(f"{sec}/rclone.conf", os.path.expanduser("~/.config/rclone/rclone.conf"))
    if os.path.exists(f"{sec}/hf_token"):
        os.environ["HF_TOKEN"] = open(f"{sec}/hf_token").read().strip()

    # write the real GPU name to Drive so we can confirm T4 vs P100 mid-run
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception as e:
        gpu = f"unknown ({type(e).__name__})"
    Path(f"{WORK}/s3_gpu.txt").write_text(gpu + "\n")
    sh(f"rclone copy {WORK}/s3_gpu.txt {DRIVE}/ -q", check=False)
    print(f"[s3] GPU = {gpu}", flush=True)

    # deps: CrisperWhisper (transformers), Silero VAD (torch.hub), loudness, io.
    sh(f"{sys.executable} -m pip install -q -U transformers accelerate", check=False)
    sh(f"{sys.executable} -m pip install -q pyloudnorm soundfile scikit-learn pyyaml typer", check=False)
    sh(f"{sys.executable} -m pip install -q 'numpy<2.1'", check=False)

    # package + all prior work state (small JSON/npz; needed for s1/s2 inputs)
    sh(f"rclone copy {DRIVE}/pkg.tgz {WORK} -q")
    os.makedirs(ROOT, exist_ok=True); sh(f"tar xzf {WORK}/pkg.tgz -C {ROOT}")
    sys.path.insert(0, f"{ROOT}/src")
    os.makedirs(f"{WORK}/work", exist_ok=True)
    print("[s3] pulling work state...", flush=True)
    sh(f"rclone copy {DRIVE}/work {WORK}/work -q", check=False)

    os.chdir(ROOT)
    from podcastsim.config import Config
    from podcastsim.episodes import Episode, load_exclusions, is_excluded
    from podcastsim.stages import s3_dataset as s3
    cfg = Config.load("configs/pipeline.yaml")
    paths = {k: Path(v) for k, v in {
        "raw": RAW, "work": f"{WORK}/work", "cache": f"{WORK}/cache",
        "models": f"{WORK}/models", "out": f"{WORK}/out", "hosts": HOSTS}.items()}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    excl = load_exclusions()

    work = Path(f"{WORK}/work")
    targets = []
    for d in sorted(work.iterdir()):
        if not (d / "s2_identify" / "host_map.json").exists():
            continue
        s3done = d / "s3_dataset" / "_DONE.json"
        if s3done.exists():
            try:
                if json.load(open(s3done)).get("versions", {}).get("code_version") == s3.CODE_VERSION:
                    continue                     # already verbatim -- resume skip
            except Exception:
                pass
        meta = json.load(open(d / "s0_ingest" / "meta.json"))
        ep = Episode(episode_id=d.name, source=meta["source"], size=0, mtime_ns=0)
        if is_excluded(ep, excl):
            continue
        targets.append(ep)

    print(f"[s3] {len(targets)} episode(s) to transcribe verbatim", flush=True)
    ok = fail = 0
    for i, ep in enumerate(targets, 1):
        t0 = time.time()
        if not fetch_source(ep.source):
            print(f"[s3] {i}/{len(targets)} MISSING SOURCE {ep.source}", flush=True)
            fail += 1; continue
        try:
            s3.run_episode(ep, cfg, paths, force=False)
        except Exception as e:
            print(f"[s3] {i}/{len(targets)} FAIL {ep.source}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            fail += 1
        else:
            # push manifest + this episode's new clips (incremental)
            sd = work / ep.episode_id / "s3_dataset"
            if sd.is_dir():
                sh(f"rclone copy '{sd}' {DRIVE}/work/{ep.episode_id}/s3_dataset -q", check=False)
            sh(f"rclone copy {HOSTS} {DRIVE}/hosts -q", check=False)
            n = 0
            try:
                n = len((sd / "clips.jsonl").read_text().split("\n")) if (sd / "clips.jsonl").exists() else 0
            except Exception:
                pass
            print(f"[s3] {i}/{len(targets)} {ep.source} {time.time()-t0:.0f}s "
                  f"clips={n}", flush=True)
            ok += 1
        # free disk: drop the source; clips already pushed
        try:
            os.remove(f"{RAW}/{ep.source}")
        except OSError:
            pass

    print(f"\n[s3] complete: {ok} ok, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
