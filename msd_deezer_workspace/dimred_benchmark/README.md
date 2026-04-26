# Dim-Reduction Benchmark (10k subset)

Self-contained, isolated benchmark for comparing three dimensionality-reduction
recipes — **PCA only**, **UMAP only**, and **PCA → UMAP** — on a 10 000-song
random sample of the production `audio/` catalogue. Runs the full pipeline
(preprocess → handcrafted features → pretrained embeddings → clustering) for
all three reductions × three clustering algorithms (KMeans, BGMM, HDBSCAN)
and produces a side-by-side comparison report.

Full design rationale, parameter justifications, and literature citations
live in [calendar/2026-04-25.md](../../calendar/2026-04-25.md).

## Isolation guarantees

Nothing in this benchmark writes outside `dimred_benchmark/`.

- The packages `audio_preprocessing/`, `pretrained_models/`, `clustering/`
  are **copied** here, not imported from the production tree, so every
  `WORKSPACE_DIR = Path(__file__).resolve().parents[1]` lookup resolves
  inside this directory.
- `clustering/shared.py::ensure_output_dir`, `write_assignments`, and
  `write_json` all run `assert_inside_test_root(path)` before any I/O —
  if a path ever escapes, the script aborts.
- The 10 000 sample songs are **hard-copied** into `dimred_benchmark/audio/`
  by default (`select_random_sample.py`, ~30 GB), so the production
  `../audio/` directory is read once and never opened again. `--mode symlink`
  is available if disk is tight.

## How to run on the DGX

```bash
# 1. From a machine that already has the repo, push the dimred_benchmark/
#    directory:
git pull   # on the DGX

cd msd_deezer_workspace/dimred_benchmark

# 2. Pick a free GPU and export it (required before pretrained extraction).
nvidia-smi            # find a GPU with 0% util and ~0 MiB used
export CUDA_VISIBLE_DEVICES=<index>

# 3. Sample 10 000 songs into ./audio/ (hard copies, ~30 GB).
python select_random_sample.py
# Optional: --n 1000 for a quick smoke test, --mode symlink to save disk.

# 4. Run the full pipeline inside a screen session (~5-7 h on one A100).
screen -S dimred_bench
bash run_full_benchmark.sh
# Detach: Ctrl-A d   |   reattach: screen -r dimred_bench

# 5. After it finishes, the comparison report is here:
#    comparison_report.csv   (full table, sorted by overall_rank)
#    comparison_report.md    (human-readable, with per-algorithm winners)
```

If preprocessing / extraction has already run on a previous attempt and you
only want to re-run clustering and aggregation:

```bash
bash run_full_benchmark.sh --clustering-only
```

## What gets compared

For each of:

- **2 feature sources**: `features/` (handcrafted, ~234-D) and
  `pretrained_embeddings/` (MusicNN+MERT+EnCodecMAE fused, 2248-D)
- **3 reductions**: `pca_only`, `umap_only`, `pca_then_umap`
- **3 algorithms**: `kmeans`, `gmm`, `hdbscan`

we run the per-algorithm grid search exactly as production does, then compute
the additional benchmark metrics defined in `clustering/benchmark_metrics.py`:

| Metric | What it catches | Source |
|---|---|---|
| Silhouette / Calinski-Harabasz / Davies-Bouldin | Internal cluster validity (already in production) | Rousseeuw 1987 / 1974 / 1979 |
| Dunn index | Cleaner separation than silhouette captures, less sensitive to size imbalance | Dunn 1974 |
| Trustworthiness | How faithfully k-NN neighbourhoods survived the reduction (independent of the clustering algorithm) | Venna & Kaski 2001 |
| Bootstrap stability ARI | "Are the clusters a property of the data or of UMAP's optimizer?" — 5 bootstraps × 80% subsample, mean pairwise ARI | Lange 2004 / Ben-Hur 2002 |

`aggregate_results.py` ranks the three reductions per metric within each
(feature source, algorithm) group and reports an averaged `overall_rank`
column. Lower is better.

## UMAP parameters

Aligned to the official "for clustering" recommendations from the umap-learn
documentation (McInnes 2018):

- `n_neighbors=30` (the "doubling the default 15" recommendation)
- `min_dist=0.0` ("set min_dist to a very low value")
- `n_components=15` (inside the 10-20 clustering range)
- `metric=euclidean`, `random_state=42`

These same UMAP params are used in both `umap_only` and `pca_then_umap` so
the only thing differing across modes is whether PCA runs first.

## Output layout

```
dimred_benchmark/
├── audio/                                         # 10k mp3 copies (gitignored)
├── audio_handcrafted/, audio_pretrained/          # preprocessed (gitignored)
├── features/                                      # handcrafted features (gitignored)
├── pretrained_embeddings/                         # pretrained embeddings (gitignored)
├── cluster_results/
│   ├── features/{pca_only,umap_only,pca_then_umap}/{kmeans,gmm,hdbscan}/
│   │       run_metadata.json + cluster_assignments.csv + selection_metrics.csv + ...
│   └── pretrained_embeddings/  (same layout)
├── logs/                                          # per-stage stdout (gitignored)
├── comparison_report.csv                          # the deliverable
└── comparison_report.md                           # the deliverable
```
