# Full-Dataset K-Method Benchmark

This benchmark decides how to choose the number of KMeans clusters for the
full catalogue without touching the production audio or production outputs.

It builds on the first 10k K-selection test, which found that the strongest
representation was:

```text
pretrained_embeddings + PCA -> UMAP
```

So this benchmark focuses on the method question:

```text
Which K-selection rule gives a defensible full-catalogue K range?
```

## Isolation Guarantees

- Production audio is read-only.
- `select_repeated_audio_samples.py` copies sampled audio into
  `full_k_method_benchmark/samples/<sample_id>/audio/`.
- Each sample workspace receives its own copied pipeline scripts and packages.
- Preprocessing, pretrained extraction, K-selection, candidate validation,
  logs, and reports are written only under this benchmark folder.
- `safety.py` and the copied `clustering/shared.py` refuse write paths outside
  their current benchmark/sample root.

## What It Runs

Default repeated samples:

```text
5 samples x 10,000 songs
seeds: 101, 202, 303, 404, 505
```

Per sample:

1. Copy 10k random audio files into the sample workspace.
2. Preprocess copied audio.
3. Extract pretrained embeddings using 16 MusicNN CPU shards plus one MERT GPU
   process and one EnCodecMAE GPU process.
4. Run `run_kmeans_gap_stability.py` on `pretrained_embeddings` with
   `pca_then_umap`.
5. Run `run_candidate_k_validation.py` for:

```text
K = 5, 8, 10, 12, 15, 20, 30, 40, 50, 60, 80, 100
```

The final aggregation compares:

- gap-statistic selected K
- prediction-strength selected K
- candidate-K bootstrap ARI
- silhouette
- Davies-Bouldin
- cluster-size safety

## Run On The DGX

```bash
cd /storage/data4/up1072603/projects/ML-song-recommendation-system
git pull origin main

cd msd_deezer_workspace/full_k_method_benchmark

tmux new -s full-k-method
```

Inside tmux:

```bash
source /opt/anaconda3/bin/activate
conda activate /storage/data4/up1072603/conda_envs/msdrec

nvidia-smi
export CUDA_VISIBLE_DEVICES=<free_gpu>
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda

bash run_full_k_method_benchmark.sh 2>&1 | tee logs/full_k_method_$(date +%Y%m%d_%H%M%S).log
```

Detach:

```text
Ctrl-b then d
```

Reattach:

```bash
tmux attach -t full-k-method
```

## Outputs

Main results:

- `method_selection_repeated_samples.csv`
- `candidate_k_validation_all_samples.csv`
- `candidate_k_validation_summary.csv`
- `full_dataset_k_method_recommendation.md`

Per-sample outputs live under:

```text
samples/<sample_id>/cluster_results/pretrained_embeddings/pca_then_umap/
```

Important per-sample folders:

- `kmeans_gap_stability/`
- `candidate_k_validation/`

## Reruns

If the copied audio already exists and extraction finished, rerun only
selection + validation + aggregation:

```bash
bash run_full_k_method_benchmark.sh --selection-only
```

If the copied audio exists but you changed scripts and want to refresh the
per-sample script copies:

```bash
bash run_full_k_method_benchmark.sh --refresh-scripts --skip-extract
```

## Interpretation

The expected final conclusion is not a single universal "true K".

- Prediction strength gives the broad stable K range.
- Gap statistic gives a fine-grained K range.
- Candidate validation chooses the largest practical K that remains stable and
  avoids tiny clusters.

That final K is the one to test on the full catalogue.
