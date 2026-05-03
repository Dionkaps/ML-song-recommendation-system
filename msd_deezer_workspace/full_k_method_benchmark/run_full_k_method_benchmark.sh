#!/usr/bin/env bash
# Repeated-sample benchmark for deciding how to choose K on the full catalogue.
#
# This intentionally benchmarks the representation that won the first 10k test:
# pretrained_embeddings + PCA -> UMAP. It creates copied-audio sample
# workspaces under ./samples/ and runs all generated outputs inside those
# sample folders only.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

SAMPLE_SIZE=10000
SEEDS=(101 202 303 404 505)
WORKERS=16
MUSICNN_WORKERS=16
FEATURE_SOURCE="pretrained_embeddings"
REDUCTION_MODE="pca_then_umap"
MAX_CLUSTERS=100
CANDIDATE_KS="5,8,10,12,15,20,30,40,50,60,80,100"

SKIP_SAMPLING=0
SKIP_EXTRACT=0
SELECTION_ONLY=0
REFRESH_SCRIPTS=0

for arg in "$@"; do
    case "$arg" in
        --skip-sampling) SKIP_SAMPLING=1 ;;
        --skip-extract) SKIP_EXTRACT=1 ;;
        --selection-only) SKIP_SAMPLING=1; SKIP_EXTRACT=1; SELECTION_ONLY=1 ;;
        --refresh-scripts) REFRESH_SCRIPTS=1 ;;
        -h|--help)
            sed -n '2,28p' "$0"; exit 0 ;;
        *)
            echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

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
    local logfile="$SCRIPT_DIR/logs/$(date +%Y%m%d_%H%M%S)_${label}.log"
    echo "[$(ts)] >>> $label"
    echo "[$(ts)] >>> log: $logfile"
    if "$@" 2>&1 | tee "$logfile"; then
        echo "[$(ts)] <<< $label DONE"
    else
        echo "[$(ts)] <<< $label FAILED" >&2
        exit 1
    fi
}

if [ "$SKIP_SAMPLING" -eq 0 ]; then
    if [ ! -d "samples" ] || [ -z "$(find samples -mindepth 1 -maxdepth 1 -type d 2>/dev/null)" ]; then
        sample_args=(--sample-size "$SAMPLE_SIZE" --workers "$WORKERS" --seeds "${SEEDS[@]}")
        if [ "$REFRESH_SCRIPTS" -eq 1 ]; then
            sample_args+=(--refresh-scripts)
        fi
        run_step select_samples python select_repeated_audio_samples.py "${sample_args[@]}"
    else
        echo "[$(ts)] samples/ already exists; skipping sampling. Use --skip-sampling explicitly on reruns."
        if [ "$REFRESH_SCRIPTS" -eq 1 ]; then
            run_step refresh_sample_scripts python select_repeated_audio_samples.py \
                --sample-size "$SAMPLE_SIZE" --workers "$WORKERS" --seeds "${SEEDS[@]}" --refresh-scripts
        fi
    fi
fi

if [ ! -d "samples" ] || [ -z "$(find samples -mindepth 1 -maxdepth 1 -type d 2>/dev/null)" ]; then
    echo "ERROR: no sample workspaces found under ./samples/." >&2
    echo "Run python select_repeated_audio_samples.py first, or omit --skip-sampling." >&2
    exit 1
fi

FAILURES=()

run_sample_step() {
    local sample_id="$1"; shift
    local label="$1"; shift
    local logfile="$SCRIPT_DIR/logs/$(date +%Y%m%d_%H%M%S)_${sample_id}_${label}.log"
    echo "[$(ts)] >>> $sample_id / $label"
    echo "[$(ts)] >>> log: $logfile"
    set +e
    "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "[$(ts)] <<< $sample_id / $label DONE"
    else
        echo "[$(ts)] <<< $sample_id / $label FAILED (continuing)" >&2
        FAILURES+=("$sample_id/$label")
    fi
}

for sample_dir in samples/sample_*; do
    [ -d "$sample_dir" ] || continue
    sample_id="$(basename "$sample_dir")"
    echo ""
    echo "[$(ts)] === Sample: $sample_id ==="
    sample_failures_before=${#FAILURES[@]}

    pushd "$sample_dir" >/dev/null
    if [ "$SKIP_EXTRACT" -eq 0 ]; then
        run_sample_step "$sample_id" preprocess python preprocess_downloaded_audio.py
        run_sample_step "$sample_id" extract_pretrained bash run_parallel_extraction.sh "$MUSICNN_WORKERS"
    else
        echo "[$(ts)] $sample_id: skipping extraction."
    fi

    if [ ${#FAILURES[@]} -eq "$sample_failures_before" ]; then
        run_sample_step "$sample_id" k_selection \
            python run_kmeans_gap_stability.py \
                --features-path "$FEATURE_SOURCE" \
                --reduction-mode "$REDUCTION_MODE" \
                --max-clusters "$MAX_CLUSTERS" \
                --workers "$WORKERS"

        run_sample_step "$sample_id" candidate_validation \
            python run_candidate_k_validation.py \
                --features-path "$FEATURE_SOURCE" \
                --reduction-mode "$REDUCTION_MODE" \
                --candidate-ks "$CANDIDATE_KS" \
                --workers "$WORKERS"
    else
        echo "[$(ts)] $sample_id: skipping selection because an earlier sample step failed."
    fi
    popd >/dev/null
done

run_step aggregate python aggregate_full_k_method_benchmark.py \
    --feature-source "$FEATURE_SOURCE" \
    --reduction-mode "$REDUCTION_MODE"

echo "[$(ts)] === Full K-method benchmark complete ==="
echo "Reports:"
echo "  $SCRIPT_DIR/method_selection_repeated_samples.csv"
echo "  $SCRIPT_DIR/candidate_k_validation_summary.csv"
echo "  $SCRIPT_DIR/full_dataset_k_method_recommendation.md"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "" >&2
    echo "[$(ts)] WARNING: ${#FAILURES[@]} sample step(s) failed:" >&2
    for failure in "${FAILURES[@]}"; do
        echo "  - $failure" >&2
    done
    echo "Inspect logs/ for details." >&2
    exit 3
fi

if [ "$SELECTION_ONLY" -eq 1 ]; then
    echo "[$(ts)] Selection-only run finished."
fi
