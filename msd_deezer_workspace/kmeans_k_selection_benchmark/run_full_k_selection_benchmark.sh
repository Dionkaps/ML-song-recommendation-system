#!/usr/bin/env bash
# Orchestrator for the KMeans K-selection benchmark on a 10k copied subset.
#
# Pipeline, all paths relative to this script's directory:
#   1. preprocess_downloaded_audio.py
#   2. extract_audio_features.py
#   3. extract_pretrained_embeddings.py via run_parallel_extraction.sh
#   4. run_kmeans_gap_stability.py for:
#        features, pretrained_embeddings
#        x pca_only, umap_only, pca_then_umap
#   5. aggregate_k_selection_results.py

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

SKIP_PREPROCESS=0
SKIP_EXTRACT=0
for arg in "$@"; do
    case "$arg" in
        --skip-preprocess) SKIP_PREPROCESS=1 ;;
        --skip-extract)    SKIP_PREPROCESS=1; SKIP_EXTRACT=1 ;;
        --selection-only)  SKIP_PREPROCESS=1; SKIP_EXTRACT=1 ;;
        -h|--help)
            sed -n '2,24p' "$0"; exit 0 ;;
        *)
            echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -d "audio" ] || [ -z "$(ls -A audio 2>/dev/null)" ]; then
    echo "ERROR: ./audio/ is empty or missing. Run: python select_random_sample.py" >&2
    exit 1
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$SKIP_EXTRACT" -eq 0 ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set. Pick a free GPU first:" >&2
    echo "         nvidia-smi" >&2
    echo "         export CUDA_VISIBLE_DEVICES=<index>" >&2
    exit 1
fi

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

FAILURES=()

run_selection_step() {
    local label="$1"; shift
    local logfile="logs/$(date +%Y%m%d_%H%M%S)_${label}.log"
    echo "[$(ts)] >>> $label"
    echo "[$(ts)] >>> log: $logfile"
    set +e
    "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[$(ts)] <<< $label DONE"
    else
        echo "[$(ts)] <<< $label FAILED (continuing; see $logfile)" >&2
        FAILURES+=("$label")
    fi
}

if [ "$SKIP_PREPROCESS" -eq 0 ]; then
    run_step preprocess python preprocess_downloaded_audio.py
else
    echo "[$(ts)] Skipping preprocessing."
fi

if [ "$SKIP_EXTRACT" -eq 0 ]; then
    run_step extract_handcrafted python extract_audio_features.py --workers 16
    run_step extract_pretrained bash run_parallel_extraction.sh 16
else
    echo "[$(ts)] Skipping extraction."
fi

FEATURE_SOURCES=("features" "pretrained_embeddings")
REDUCTIONS=("pca_only" "umap_only" "pca_then_umap")

for src in "${FEATURE_SOURCES[@]}"; do
    if [ ! -d "$src" ]; then
        echo "[$(ts)] WARNING: $src/ does not exist. Skipping." >&2
        continue
    fi
    for red in "${REDUCTIONS[@]}"; do
        run_selection_step "kselect_${src}_${red}" \
            python run_kmeans_gap_stability.py \
                --features-path "$src" \
                --reduction-mode "$red" \
                --workers 16
    done
done

run_selection_step aggregate python aggregate_k_selection_results.py

echo "[$(ts)] === K-selection benchmark complete ==="
echo "Reports:"
echo "  $(pwd)/k_selection_summary.csv"
echo "  $(pwd)/k_selection_summary.md"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "" >&2
    echo "[$(ts)] WARNING: ${#FAILURES[@]} step(s) failed:" >&2
    for f in "${FAILURES[@]}"; do
        echo "  - $f" >&2
    done
    echo "Inspect logs/ for details." >&2
    exit 3
fi
