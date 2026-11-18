"""Synthesize a full parody episode from episode_script.txt using the fine-tuned
TRIH voices. Groups lines by host (generate() patches T3, so load each host's
model once), splits long lines into sentences for TTS stability, concatenates.
Per-line WAVs -> trih_state/s4/episode/line_NN.wav (stitched locally after)."""
import os, re, subprocess, sys, shutil
from collections import OrderedDict

WORK = "/kaggle/working"; DRIVE = "gdrive_rw:trih_state"
GEN = {"cfg_weight": 0.5, "exaggeration": 0.5}


def sh(c, check=True):
    print(f"$ {c}", flush=True); return subprocess.run(c, shell=True, check=check)


def find_secrets():
    for r, _d, fs in os.walk("/kaggle/input"):
        if "rclone.conf" in fs:
            return r
    sys.exit("secrets not attached")


def norm(t):
    return (t.replace("’", "'").replace("‘", "'").replace("“", '"')
             .replace("”", '"').replace("—", " - ").replace("–", "-")
             .replace("…", "...").strip())


def sentences(t):
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def main():
    sec = find_secrets()
    if not shutil.which("rclone"):
        sh("curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip "
           "&& cd /tmp && unzip -q r.zip && cp rclone-*/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone")
    os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
    shutil.copy(f"{sec}/rclone.conf", os.path.expanduser("~/.config/rclone/rclone.conf"))
    sh(f"{sys.executable} -m pip install -q chatterbox-tts", check=False)
    sh(f"{sys.executable} -m pip install -q --force-reinstall transformers==5.2.0", check=False)
    sh(f"{sys.executable} -m pip install -q torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 "
       f"--index-url https://download.pytorch.org/whl/cu124", check=False)
    sh(f"{sys.executable} -m pip install -q 'numpy<2.1'", check=False)

    sh(f"rclone copy {DRIVE}/episode_script.txt {WORK}/ -q")
    lines = []  # (global_idx, host, text)
    for ln in open(f"{WORK}/episode_script.txt"):
        m = re.match(r"\s*(host_[01])\s*:\s*(.+)", ln.strip())
        if m:
            lines.append([len(lines), m.group(1), norm(m.group(2))])
    print(f"[ep] {len(lines)} lines", flush=True)

    hosts = ["host_0", "host_1"]
    import torch, torchaudio
    from chatterbox.tts import ChatterboxTTS
    from safetensors.torch import load_file
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(f"{WORK}/ep", exist_ok=True)
    gap = None

    for h in hosts:
        sh(f"rclone copy {DRIVE}/s4/finetune_verbatim/{h}/t3_cfg.safetensors /tmp/{h}/ -q", check=False)
        ck = f"/tmp/{h}/t3_cfg.safetensors"
        if not os.path.exists(ck):
            print(f"[ep] no checkpoint {h}", flush=True); continue
        first = subprocess.run(f"rclone lsf {DRIVE}/hosts/{h}/clips | grep .flac | sort | head -1",
                               shell=True, capture_output=True, text=True).stdout.strip()
        sh(f"rclone copy '{DRIVE}/hosts/{h}/clips/{first}' /tmp/{h}ref/ -q", check=False)
        ref = f"/tmp/{h}ref/{first}"
        model = ChatterboxTTS.from_pretrained(device=dev)
        model.t3.load_state_dict(load_file(ck))
        sr = model.sr
        if gap is None:
            gap = torch.zeros(1, int(0.25 * sr))
        print(f"[ep] {h}: loaded", flush=True)
        for idx, host, text in lines:
            if host != h:
                continue
            chunks = []
            for s in sentences(text):
                wav = model.generate(s, audio_prompt_path=ref,
                                     cfg_weight=GEN["cfg_weight"], exaggeration=GEN["exaggeration"])
                w = wav.cpu() if hasattr(wav, "cpu") else wav
                if w.dim() == 1:
                    w = w.unsqueeze(0)
                chunks.append(w)
                chunks.append(torch.zeros(1, int(0.15 * sr)))
            line_wav = torch.cat(chunks, dim=1)
            out = f"{WORK}/ep/line_{idx:02d}.wav"
            torchaudio.save(out, line_wav, sr)
            sh(f"rclone copy {out} {DRIVE}/s4/episode/ -q", check=False)
            print(f"[ep] line {idx:02d} ({h}) {line_wav.shape[1]/sr:.1f}s", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("[ep] done", flush=True)


if __name__ == "__main__":
    main()
