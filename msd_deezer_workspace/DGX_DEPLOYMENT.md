# DGX Deployment — What to install after `git pull`

You already have SSH/Bitvise set up and the repo cloned on the DGX. This doc covers only the delta: what the new pretrained-embedding code needs that git alone won't give you.

---

## TL;DR

After `git pull`, you need to do two things that git can't do for you:

1. `pip install -r requirements-pretrained.txt` — the new Python deps
2. `python msd_deezer_workspace/install_encodecmae.py` — the one-command installer for EnCodecMAE (works around an upstream packaging bug)

That's it. HuggingFace model weights (MusicNN, MERT, EnCodecMAE — ~1.5 GB total) download automatically on the first extraction run and get cached under `~/.cache/huggingface/`.

Full procedure below.

---

## 1. Pull the new code

```bash
cd /storage/data4/up1072603/ML-song-recommendation-system
git pull origin main
```

Nothing of significant size is committed — the whole pretrained-models package is a few KB of Python. No big files are gitignored that you'd need to fetch separately *from this repo*. The only sizeable things the pipeline needs are model weights, and those come from HuggingFace at runtime.

---

## 2. Install the new Python dependencies

Activate your env (create one first if you haven't):

```bash
source /opt/anaconda3/bin/activate
conda activate msd-pretrained   # or create: conda create -n msd-pretrained python=3.11 -y

pip install -r requirements-pretrained.txt
```

This installs: `torch`, `transformers`, `tensorflow`, `tf-keras`, `musicnn`.

**Note:** `torch` from PyPI already includes CUDA support on the DGX. If `nvidia-smi` reports CUDA 12.x and you want to be explicit, use `pip install torch --index-url https://download.pytorch.org/whl/cu121` instead.

---

## 3. Install EnCodecMAE (one command via a helper script)

EnCodecMAE is **not** in `requirements-pretrained.txt` because its upstream pip package is broken — it omits five subpackages (`models/`, `configs/`, `tasks/`, `heareval_model/`, `scripts/`), so `from encodecmae.models import EncodecMAE` fails at runtime. Upstream has not fixed this.

The repo ships [install_encodecmae.py](install_encodecmae.py), which does the pip install, clones the source repo to a temp dir, copies the missing subpackages into the pip install location, and verifies the import — all in one command:

```bash
cd /storage/data4/up1072603/ML-song-recommendation-system
python msd_deezer_workspace/install_encodecmae.py
```

Idempotent — safe to re-run if you're unsure whether it's already done. Final line should read `OK -- import ok`. If you see that, EnCodecMAE is ready.

---

## 4. Verify with the built-in pre-flight check

```bash
cd /storage/data4/up1072603/ML-song-recommendation-system/msd_deezer_workspace
python extract_pretrained_embeddings.py --check-models --device cuda
```

Expected:

```
[musicnn]    OK -- musicnn (200-dim @ 16000 Hz)
[mert]       OK -- mert (768-dim @ 24000 Hz)
[encodecmae] OK -- encodecmae (1024-dim @ 24000 Hz)
All models loaded successfully.
```

**First run only:** the models will download from HuggingFace (~380 MB MERT + ~1 GB EnCodecMAE + ~60 MB MusicNN). That's a one-time cost — cached under `~/.cache/huggingface/` afterwards.

> **If you're tight on home-dir quota (50 GB limit),** move the HF cache to `/storage/data4`:
> ```bash
> export HF_HOME=/storage/data4/up1072603/hf_cache
> ```
> Put this in your `~/.bashrc` so it persists.

---

## 5. Troubleshooting

| Failure | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'encodecmae.models'` | Re-run `python msd_deezer_workspace/install_encodecmae.py` |
| `RuntimeError: batch_normalization is not available with Keras 3` | `tf-keras` not installed → `pip install tf-keras` |
| `Exception: Available models are: [...]` from encodecmae | MODEL_VARIANT string mismatch — should be `mel256-ec-large_st`; pull latest `main` |
| MERT download is slow / rate-limited | `export HF_HUB_ENABLE_HF_TRANSFER=1` or set `HF_TOKEN` |
| `torch.cuda.is_available()` returns False | CUDA/torch version mismatch — reinstall torch with the correct `--index-url` |

---

## 6. What's gitignored that you might care about

Nothing that the DGX needs you to install manually. For reference:

- `audio/` — preprocessed WAVs, produced on the DGX itself by the Deezer download pipeline. Not synced via git.
- `pretrained_embeddings/` — the output of the extraction run. Created on the DGX. Copy back to the laptop with `scp` when done.
- `sample_runs/`, `cache/`, `features/` — local test artefacts. Ignore.
- HuggingFace weights (`~/.cache/huggingface/`) — downloaded automatically on first run.

Nothing else. No third-party data drops, no vendored binaries, no Git LFS.

---

## 7. Run the extraction

Once §4 passes:

```bash
screen -S extract

# Pick a free GPU first — see "How to pick a GPU" below. Replace 0 with the chosen index.
export CUDA_VISIBLE_DEVICES=0

# Required by CEID admin: tell TF/XLA where the DGX's CUDA toolkit lives.
# MusicNN runs on TensorFlow and needs this to compile GPU kernels.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda

python extract_pretrained_embeddings.py --device cuda 2>&1 \
    | tee pretrained_embeddings/extraction_$(date +%Y%m%d_%H%M).log
```

### How to pick a GPU

Before the export, check what's free:

```bash
nvidia-smi                    # one-shot snapshot
# or
nvtop                         # live view, press q to quit
```

In `nvidia-smi` look at two columns per GPU:
- **GPU-Util %** — 0 % means idle, anything sustained means someone is computing.
- **Memory-Usage** — even 1–2 GB in use usually means another user has a model loaded; don't share.

Pick the lowest-indexed GPU where both utilization is ~0 % **and** memory is ~0 MiB **and** the `Processes` list at the bottom has no entry for that GPU. That index is what you put in `CUDA_VISIBLE_DEVICES`.

Rule of thumb from the admin message: use **at most one GPU** at a time.

Detach with `Ctrl-A d`, reattach with `screen -r extract`. Resume is ON by default — if the run crashes, rerun the same command and it skips songs already processed.
