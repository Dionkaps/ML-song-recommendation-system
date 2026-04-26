#!/usr/bin/env bash
# Orchestrator for the dim-reduction benchmark on a 10k subset.
#
# Pipeline (all paths relative to this script's directory):
#   1. preprocess_downloaded_audio.py   audio/             -> audio_handcrafted/, audio_pretrained/
#   2. extract_audio_features.py        audio_handcrafted/ -> features/
#   3. extract_pretrained_embeddings.py audio_pretrained/  -> pretrained_embeddings/
#   4. For (handcrafted, pretrained) x (pca_only, umap_only, pca_then_umap) x (kmeans, gmm, hdbscan):
#        run the clustering script with the right --features-path and --reduction-mode.
#      That's 2 x 3 x 3 = 18 clustering runs in total.
#   5. aggregate_results.py             cluster_results/   -> comparison_report.{csv,md}
#
# Resumable: each stage skips work that already exists. preprocessing skips
# pairs whose outputs are already present; extraction scripts honour their
# own resume semantics; clustering runs are idempotent at the output dir
# level (they overwrite).
#
# Run inside `screen -S dimred_bench` so detaching doesn't kill it.
#
# Usage:
#   bash run_full_benchmark.sh                    # full pipeline
#   bash run_full_benchmark.sh --skip-preprocess  # skip step 1
#   bash run_full_benchmark.sh --skip-extract     # skip steps 1-3
#   bash run_full_benchmark.sh --clustering-only  # alias of --skip-extract

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ---- Flags -----------------------------------------------------------------
SKIP_PREPROCESS=0
SKIP_EXTRACT=0
for arg in "$@"; do
    case "$arg" in
        --skip-preprocess) SKIP_PREPROCESS=1 ;;
        --skip-extract)    SKIP_PREPROCESS=1; SKIP_EXTRACT=1 ;;
        --clustering-only) SKIP_PREPROCESS=1; SKIP_EXTRACT=1 ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *)
            echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

# ---- Sanity checks ---------------------------------------------------------
if [ ! -d "audio" ] || [ -z "$(ls -A audio 2>/dev/null)" ]; then
    echo "ERROR: ./audio/ is empty or missing. Run select_random_sample.py first." >&2
    exit 1
fi

# DGX requirement (per CLAUDE memory + DGX_DEPLOYMENT.md): pretrained extraction
# needs CUDA_VISIBLE_DEVICES set explicitly so we don't accidentally hop GPUs.
# Fail fast rather than silently consume a busy GPU.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$SKIP_EXTRACT" -eq 0 ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set. Pick a free GPU first:" >&2
    echo "         nvidia-smi   # find a GPU with 0% util and ~0 MiB used" >&2
    echo "         export CUDA_VISIBLE_DEVICES=<index>" >&2
    exit 1
fi

# Required by CEID admin: tell TF/XLA where the DGX's CUDA toolkit lives.
# MusicNN runs on TensorFlow and needs this. Set only if the toolkit path
# exists -- on non-DGX machines we don't want to pin a non-existent path.
if [ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda"
fi

mkdir -p logs

ts() { date +'%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    local logfile="logs/$(date +%Y%m%d_%H%M%S)_${label}.log"
    echo "[$(ts)] >>> $label"
    echo "[$(ts)] >>> log: $logfile"
    if "$@" 2>&1 | tee "$logfile"; then
        echo "[$(ts)] <<< $label DONE"
    else
        echo "[$(ts)] <<< $label FAILED" >&2
        exit 1
    fi
}

# Same as run_step but doesn't abort the whole script on failure -- used for
# clustering runs so a single broken (feature, reduction, algorithm) cell
# doesn't waste the 4-5 hours of extraction work that came before it.
# Tracks failures so we can report them and exit non-zero at the end.
CLUSTER_FAILURES=()

run_cluster_step() {
    local label="$1"; shift
    local logfile="logs/$(date +%Y%m%d_%H%M%S)_${label}.log"
    echo "[$(ts)] >>> $label"
    echo "[$(ts)] >>> log: $logfile"
    # `set +e` locally so a non-zero exit from the pipeline doesn't abort.
    set +e
    "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[$(ts)] <<< $label DONE"
    else
        echo "[$(ts)] <<< $label FAILED (continuing — see $logfile)" >&2
        CLUSTER_FAILURES+=("$label")
    fi
}

# ---- Stage 1: dual preprocessing ------------------------------------------
if [ "$SKIP_PREPROCESS" -eq 0 ]; then
    run_step preprocess python preprocess_downloaded_audio.py
else
    echo "[$(ts)] Skipping preprocessing (flag)."
fi

# ---- Stage 2: handcrafted feature extraction -------------------------------
if [ "$SKIP_EXTRACT" -eq 0 ]; then
    run_step extract_handcrafted python extract_audio_features.py
fi

# ---- Stage 3: pretrained embedding extraction ------------------------------
# Reuses the existing parallel launcher which co-runs MERT and EnCodecMAE
# on the assigned GPU and keeps MusicNN on a separate worker. The launcher
# already respects PRETRAINED_AUDIO_DIR / output paths from defaults, which
# resolve to dimred_benchmark/* because the package is copied here.
if [ "$SKIP_EXTRACT" -eq 0 ]; then
    run_step extract_pretrained bash run_parallel_extraction.sh 16
fi

# ---- Stage 4: clustering matrix --------------------------------------------
# 2 feature sources x 3 reductions x 3 algorithms = 18 runs.
# Each run writes into cluster_results/<feature_source>/<reduction>/<algorithm>/.
FEATURE_SOURCES=("features" "pretrained_embeddings")
REDUCTIONS=("pca_only" "umap_only" "pca_then_umap")
ALGOS=("kmeans" "gmm" "hdbscan")

for src in "${FEATURE_SOURCES[@]}"; do
    if [ ! -d "$src" ]; then
        echo "[$(ts)] WARNING: $src/ does not exist (extraction failed?). Skipping." >&2
        continue
    fi
    for red in "${REDUCTIONS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            label="cluster_${src}_${red}_${algo}"
            run_cluster_step "$label" python "run_${algo}_clustering.py" \
                --features-path "$src" \
                --reduction-mode "$red"
        done
    done
done

# ---- Stage 5: aggregation --------------------------------------------------
# Aggregate even if some clustering runs failed -- the report still has value
# for the runs that succeeded, and the diagnostics make the failures obvious.
run_cluster_step aggregate python aggregate_results.py

echo "[$(ts)] === Benchmark complete ==="
echo "Reports:"
echo "  $(pwd)/comparison_report.csv"
echo "  $(pwd)/comparison_report.md"

if [ ${#CLUSTER_FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "[$(ts)] WARNING: ${#CLUSTER_FAILURES[@]} step(s) failed:" >&2
    for f in "${CLUSTER_FAILURES[@]}"; do
        echo "  - $f" >&2
    done
    echo "  Inspect their .log files under $(pwd)/logs/ for details." >&2
    exit 3
fi
