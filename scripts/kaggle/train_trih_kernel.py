"""Fine-tune Chatterbox T3 for both hosts in one Kaggle GPU session
(plan S9.3, full T3 fine-tune), on the CrisperWhisper verbatim dataset.

Training text comes from the per-episode work/{ep}/s3_dataset/clips.jsonl
(authoritative verbatim labels), NOT the retired shared hosts/*/metadata.jsonl.

One host per iteration: build the trainer metadata, run
stlohrey/chatterbox-finetuning's finetune_t3.py, push that host's
t3_cfg.safetensors to Drive, move on. Resumable: a host whose checkpoint already
exists in Drive is skipped, so a session that dies mid-way is continued by
re-running. Exaggeration is a synth-time knob and is NOT set here.

Checkpoints go to s4/finetune_verbatim/ (old faster-whisper ones in
s4/finetune/ are kept for A/B).
"""
import json, os, subprocess, sys, glob, time

CONFIG = {"hosts": ["host_0", "host_1"],
          "epochs": 2, "batch": 2, "grad_accum": 4, "max_clips": 0}

WORK = "/kaggle/working"
DRIVE = "gdrive_rw:trih_state"
REPO = "/tmp/chatterbox-finetuning"
DATA = "/tmp/ftdata"          # keep clips/ckpts out of /kaggle/working


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def find_secrets():
    for root, _d, files in os.walk("/kaggle/input"):
        if "rclone.conf" in files:
            return root
    sys.exit("secrets not attached")


def already_done(host):
    r = subprocess.run(
        f"rclone lsf {DRIVE}/s4/finetune_verbatim/{host}/t3_cfg.safetensors",
        shell=True, capture_output=True, text=True)
    return "t3_cfg.safetensors" in r.stdout


def build_meta(host, hostdir):
    """train_meta.txt: <abs_audio_path>\\t<verbatim_text>, from clips.jsonl."""
    rows = []
    for f in glob.glob(f"{WORK}/work/*/s3_dataset/clips.jsonl"):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("host") == host:
                rows.append(r)
    if CONFIG.get("max_clips"):
        rows = rows[:CONFIG["max_clips"]]
    meta = f"{hostdir}/train_meta.txt"
    n = 0
    with open(meta, "w") as fh:
        for r in rows:
            ap = f"{hostdir}/clips/{r['clip']}"
            text = " ".join((r.get("text") or "").replace("\t", " ").split())
            if text and os.path.exists(ap):
                fh.write(f"{ap}\t{text}\n"); n += 1
    return meta, n, len(rows)


def main():
    import shutil
    secrets = find_secrets()
    if not shutil.which("rclone"):
        sh("curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip "
           "&& cd /tmp && unzip -q r.zip && cp rclone-*/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone")
    os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
    shutil.copy(f"{secrets}/rclone.conf", os.path.expanduser("~/.config/rclone/rclone.conf"))
    if os.path.exists(f"{secrets}/hf_token"):
        os.environ["HF_TOKEN"] = open(f"{secrets}/hf_token").read().strip()

    # trainer + Chatterbox's exact pins (transformers 5.2.0 / torch 2.6.0 cu124)
    sh(f"git clone -q https://github.com/stlohrey/chatterbox-finetuning {REPO}", check=False)
    sh(f"{sys.executable} -m pip install -q chatterbox-tts peft datasets", check=False)
    sh(f"{sys.executable} -m pip install -q --force-reinstall transformers==5.2.0", check=False)
    sh(f"{sys.executable} -m pip install -q torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 "
       f"--index-url https://download.pytorch.org/whl/cu124", check=False)
    sh(f"{sys.executable} -m pip install -q 'numpy<2.1'", check=False)

    # verbatim metadata source (small): all per-episode clips.jsonl
    os.makedirs(f"{WORK}/work", exist_ok=True)
    sh(f"rclone copy {DRIVE}/work {WORK}/work --include '*/s3_dataset/clips.jsonl' "
       f"-q --transfers 16", check=False)

    for host in CONFIG["hosts"]:
        if already_done(host):
            print(f"[train] {host}: checkpoint already in Drive -- skip", flush=True)
            continue
        t0 = time.time()
        hostdir = f"{DATA}/{host}"
        os.makedirs(f"{hostdir}/clips", exist_ok=True)
        print(f"[train] {host}: pulling clips...", flush=True)
        sh(f"rclone copy {DRIVE}/hosts/{host}/clips {hostdir}/clips -q --transfers 16")
        meta, n, total = build_meta(host, hostdir)
        print(f"[train] {host}: {n}/{total} verbatim clips in metadata", flush=True)
        if n < 20:
            print(f"[train] {host}: too few clips ({n}) -- skip", flush=True)
            continue

        ckpt = f"{DATA}/ckpt/{host}"
        os.makedirs(ckpt, exist_ok=True)
        cmd = (f"cd {REPO} && {sys.executable} src/finetune_t3.py "
               f"--output_dir {ckpt} --model_name_or_path ResembleAI/chatterbox "
               f"--metadata_file {meta} --eval_split_size 0.02 "
               f"--num_train_epochs {CONFIG['epochs']} "
               f"--per_device_train_batch_size {CONFIG['batch']} "
               f"--gradient_accumulation_steps {CONFIG['grad_accum']} "
               f"--learning_rate 5e-5 --warmup_steps 50 --fp16 "
               f"--do_train --do_eval --dataloader_pin_memory False "
               f"--eval_on_start False --label_names labels_speech "
               f"--save_strategy no --logging_steps 10")
        sh(cmd)

        # push ONLY the fine-tuned T3 weights (~1-2 GB, not the whole ~14 GB tree)
        sh(f"rclone copy {ckpt}/t3_cfg.safetensors {DRIVE}/s4/finetune_verbatim/{host}/ -q", check=False)
        print(f"[train] {host}: done in {(time.time()-t0)/60:.1f} min", flush=True)
        # free disk before the next host
        sh(f"rm -rf {hostdir} {ckpt}", check=False)

    print("[train] all hosts complete", flush=True)


if __name__ == "__main__":
    main()
