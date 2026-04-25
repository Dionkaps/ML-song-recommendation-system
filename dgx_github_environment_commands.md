# DGX commands for environment setup and re-cloning the project

## 1) Activate Anaconda, activate the project environment, and move into the repo

```bash
source /opt/anaconda3/bin/activate
conda activate /storage/data4/up1072603/conda_envs/msdrec
cd /storage/data4/up1072603/projects/ML-song-recommendation-system
```

### When to use this block

Use this block every time you log into the DGX and want to work on the project with the correct Python environment.

---

## 2) Delete the current local repo folder and clone it again from GitHub

```bash
rm -rf /storage/data4/up1072603/projects/ML-song-recommendation-system
cd /storage/data4/up1072603/projects
git clone git@github.com:Dionkaps/ML-song-recommendation-system.git
```

### When to use this block

Use this block only when you want a clean fresh copy of the repo, for example:

- the local folder is broken or messy
- you want to discard the local copy completely
- you accidentally changed or deleted many files locally


## 3) Run the Deezer download pipeline

Move into the workspace folder that contains the download runner, then start the resilient download job:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python run_resilient_msd_deezer.py --workers 3 --max-songs-per-sec 3.0 --save-every 25
```

### When to use this block

Use this block when you want to download Deezer preview audio for the MSD/Deezer workspace.

---

## 4) Run the resilient Deezer download inside a named tmux session

If you want the long-running Deezer download job to keep running after you disconnect from the DGX, start it inside `tmux`:

```bash
# Start a named session
tmux new -s msd

# Inside tmux, run your stuff normally:
source /opt/anaconda3/bin/activate
conda activate /storage/data4/up1072603/conda_envs/msdrec
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python run_resilient_msd_deezer.py --workers 6 --max-songs-per-sec 4.0 \
  --save-every 100 --chunk-size 5000 --idle-timeout-sec 900

# Detach and leave it running:
# Ctrl-b then d
```

To manage the session later:

```bash
tmux ls
tmux attach -t msd
tmux kill-session -t msd
```

### When to use this block

Use this block when you want the resilient Deezer download to continue running safely in the background even if your SSH session disconnects.

### Important

Reattach with `tmux attach -t msd` whenever you want to inspect progress, and use `tmux kill-session -t msd` only when you want to stop and remove that named session.

---

## 5) Build the metadata CSV only

If you want to run only the extraction stage and stop before any Deezer matching or downloads, use:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python msd_deezer_pipeline.py --extract-only
```

### When to use this block

Use this block when you want to build or refresh the metadata CSV only, without starting the Deezer matching/download step.

---

## 6) Reset the MSD/Deezer workspace outputs

If you want to delete generated workspace artifacts and start fresh, use:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python reset_msd_deezer_workspace.py --all
```

### When to use this block

Use this block when you want to clear generated audio, features, cluster results, logs, caches, and similar workspace outputs before rerunning the pipeline.

### Important

This is a destructive cleanup command for generated workspace outputs, and `--all` deletes all discovered cleanup groups without prompting.

---

## 7) Preprocess the downloaded audio (dual-branch)

Preprocessing now produces two preprocessed copies of each downloaded mp3
instead of overwriting the source:

* `audio_handcrafted/` — 22050 Hz mono WAV, center-cropped to 29 s,
  normalized to -23 LUFS (EBU R128), sample-peak ceiling -1 dBFS. Feeds
  the MFCC / chroma / spectral / Tzanetakis feature extractor.
* `audio_pretrained/` — 24000 Hz mono WAV, center-cropped to 29 s,
  peak-capped only (no LUFS, so the SSL pretraining distribution is
  preserved). Feeds MERT, EnCodecMAE, and MusiCNN.

The original mp3 files in `audio/` are kept untouched. Rejections
(too short, silent, decode error) are synchronised between the two
output directories: if a song is dropped for one branch it is dropped
for the other.

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python preprocess_downloaded_audio.py
```

### When to use this block

Use this block after the download step, before either feature-extraction
stage. Resume is built in — re-running skips pairs that are already
present in both output directories.

---

## 8) Extract audio features (handcrafted branch)

Handcrafted feature extraction reads from `audio_handcrafted/` by default:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python extract_audio_features.py \
    --input-dir audio_handcrafted \
    --output-dir features \
    --sample-rate 22050 \
    --n-mfcc 20
```

### When to use this block

Use this block after preprocessing when you want to generate feature
outputs in `features/` from the 22 kHz LUFS-normalised copies in
`audio_handcrafted/`.

---

## 9) Stop running pretrained embedding extraction jobs

If you need to stop any running `extract_pretrained_embeddings.py` processes for your user, use:

```bash
pkill -u $USER -f extract_pretrained_embeddings.py
```

### When to use this block

Use this block when a pretrained embedding extraction run is stuck, needs to be restarted, or you want to stop all currently running extraction jobs owned by your user account.

### Important

This command stops matching `extract_pretrained_embeddings.py` processes for your current user.

---

## 10) Run the pretrained embedding models

Pretrained extraction reads from `audio_pretrained/` by default (the
24 kHz non-LUFS-normalised branch of the dual preprocessor). Verify the
models first, then launch the parallel extraction job:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda

python extract_pretrained_embeddings.py --check-models --device cuda
bash run_parallel_extraction.sh 16
```

### When to use this block

Use this block after `preprocess_downloaded_audio.py` has produced
`audio_pretrained/`, when you want to generate pretrained embeddings
with MusicNN, MERT, and EnCodecMAE on the DGX.

### Important

Replace `0` in `CUDA_VISIBLE_DEVICES=0` with a free GPU index on the DGX
before starting the run. Override the default input directory (e.g. for
a probe subset) by exporting `PRETRAINED_AUDIO_DIR` before the `bash`
invocation.

---

## 11) Run clustering on the extracted audio features

If you want to cluster the contents of `features/` and then inspect the result in the explorer, use:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python run_kmeans_clustering.py --features-path features/
python run_gmm_clustering.py --features-path features/
python run_hdbscan_clustering.py --features-path features/
python launch_cluster_explorer.py --features-path features/ --no-deezer-previews
```

### When to use this block

Use this block when you want to cluster the handcrafted audio features stored in `features/` and browse the results locally in the cluster explorer.

---

## 12) Run clustering on the pretrained embeddings

If you want to cluster the contents of `pretrained_embeddings/` and then inspect the result in the explorer, use:

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system/msd_deezer_workspace
python run_kmeans_clustering.py --features-path pretrained_embeddings/
python run_gmm_clustering.py --features-path pretrained_embeddings/
python run_hdbscan_clustering.py --features-path pretrained_embeddings/
python launch_cluster_explorer.py --features-path pretrained_embeddings/ --no-deezer-previews
```

### When to use this block

Use this block when you want to cluster the pretrained embedding outputs and inspect those clusters without loading Deezer preview audio in the explorer.

---
