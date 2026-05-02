# KMeans K-Selection Benchmark (10k subset)

Self-contained benchmark for testing the thesis workflow from the shared chat:

1. Use the gap statistic as the primary estimate of K.
2. Select the smallest K satisfying the one-standard-error rule.
3. Confirm reproducibility with prediction strength and bootstrap stability.
4. Report silhouette, Calinski-Harabasz, and Davies-Bouldin as supporting
   internal validity evidence.
5. If the indices disagree strongly, mark the run as needing interpretation
   review instead of pretending there is one objective answer.

This directory is intentionally isolated from the production workspace. Scripts
read the production `../audio/` directory only when sampling; every generated
file is written under this folder.

## Isolation Guarantees

- `select_random_sample.py` hard-copies the sampled songs into `./audio/`.
- `safety.py` refuses write paths outside this benchmark root.
- Preprocessing, handcrafted features, pretrained embeddings, clustering
  outputs, logs, manifests, and reports are all created inside this folder.
- The copied `clustering/shared.py` still guards clustering output paths.

## Run On The DGX

```bash
cd msd_deezer_workspace/kmeans_k_selection_benchmark

# 1. Copy 10,000 random downloaded songs into ./audio/.
python select_random_sample.py

# 2. Pick a free GPU for pretrained embedding extraction.
nvidia-smi
export CUDA_VISIBLE_DEVICES=<index>

# 3. Run the full K-selection benchmark.
screen -S kselect_bench
bash run_full_k_selection_benchmark.sh
```

Outputs:

```text
kmeans_k_selection_benchmark/
  audio/                         # 10k copied songs
  audio_handcrafted/
  audio_pretrained/
  features/
  pretrained_embeddings/
  cluster_results/
    features/{pca_only,umap_only,pca_then_umap}/kmeans_gap_stability/
    pretrained_embeddings/{pca_only,umap_only,pca_then_umap}/kmeans_gap_stability/
  k_selection_summary.csv
  k_selection_summary.md
```

Each run also writes:

- `selection_metrics.csv`: gap, gap standard error, prediction strength,
  silhouette, CH, DB, and inertia for each candidate K.
- `k_selection_report.md`: compact per-run decision report.
- `cluster_assignments.csv`: final KMeans labels using the selected K.
- `run_metadata.json`: machine-readable method, parameters, and metrics.

## Faster Resume Modes

```bash
# Only rerun K-selection and aggregation when features already exist.
bash run_full_k_selection_benchmark.sh --selection-only

# Skip only preprocessing; useful if audio_handcrafted/ and audio_pretrained/
# are already complete but feature extraction needs to resume.
bash run_full_k_selection_benchmark.sh --skip-preprocess
```

## Practical Notes

The gap statistic is expensive because it fits KMeans on multiple uniform
reference datasets for every candidate K. Defaults are thesis-oriented rather
than tiny-smoke-test oriented: `K=2..60`, 10 gap reference samples, 5 prediction
strength repeats, and 5 bootstrap stability runs.

For a quick check:

```bash
python run_kmeans_gap_stability.py --features-path <feature_dir> \
  --reduction-mode pca_only --max-clusters 8 \
  --gap-reference-samples 2 --prediction-strength-repeats 2 \
  --stability-bootstraps 2 --workers 1
```

## Method References

- Gap statistic: Tibshirani, Walther, and Hastie (2001),
  "Estimating the number of clusters in a data set via the gap statistic."
- Prediction strength: Tibshirani and Walther (2005),
  "Cluster Validation by Prediction Strength."
- Silhouette: Rousseeuw (1987), "Silhouettes: a graphical aid to the
  interpretation and validation of cluster analysis."
- Calinski-Harabasz: Calinski and Harabasz (1974), "A dendrite method for
  cluster analysis."
- Davies-Bouldin: Davies and Bouldin (1979), "A cluster separation measure."
- Stability validation: Ben-Hur, Elisseeff, and Guyon (2002),
  "A stability based method for discovering structure in clustered data."
