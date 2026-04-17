#!/usr/bin/env bash
# Parallel pretrained-embedding extraction.
#
#   MusicNN runs as N CPU workers (sharded over the song list).
#   MERT       runs as 1 GPU process.
#   EnCodecMAE runs as 1 GPU process.
#
# When every background worker finishes, the script calls
# `merge_sharded_embeddings.py` to combine the three per-model output
# directories into one unified directory with the fused CSV.
#
# Usage:
#   ./run_parallel_extraction.sh [musicnn_workers]
#
# Default musicnn_workers = 16. Override by passing the count as the first
# argument, e.g.  ./run_parallel_extraction.sh 8
#
# Prerequisites (must be active before invoking this script):
#   * The msd-pretrained conda env (or equivalent) is activated.
#   * CUDA_VISIBLE_DEVICES is set to a free GPU index (for MERT + EnCodecMAE).
#   * You are cd'd into msd_deezer_workspace/.

set -euo pipefail

MUSICNN_WORKERS=${1:-16}

if ! [[ "$MUSICNN_WORKERS" =~ ^[0-9]+$ ]] || [[ "$MUSICNN_WORKERS" -lt 1 ]]; then
    echo "musicnn_workers must be a positive integer, got: $MUSICNN_WORKERS" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Preflight --------------------------------------------------------------

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "WARNING: no conda env active. Activate with:" >&2
    echo "  source /opt/anaconda3/bin/activate && conda activate msd-pretrained" >&2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set." >&2
    echo "Pick a free GPU index first, e.g.:  export CUDA_VISIBLE_DEVICES=3" >&2
    exit 1
fi

# Required by the CEID admin for TF/XLA to find the DGX's CUDA toolkit.
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda}

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="pretrained_embeddings_logs/${TS}"
mkdir -p "$LOG_DIR"

MUSICNN_OUT="pretrained_embeddings_musicnn"
MERT_OUT="pretrained_embeddings_mert"
ENCODEC_OUT="pretrained_embeddings_encodecmae"
MERGED_OUT="pretrained_embeddings"

mkdir -p "$MUSICNN_OUT" "$MERT_OUT" "$ENCODEC_OUT"

echo "=========================================================="
echo "Parallel pretrained-embedding extraction"
echo "=========================================================="
echo "  MusicNN workers  : $MUSICNN_WORKERS  (CPU, no GPU)"
echo "  MERT             : 1 process  (GPU $CUDA_VISIBLE_DEVICES)"
echo "  EnCodecMAE       : 1 process  (GPU $CUDA_VISIBLE_DEVICES)"
echo "  Logs             : $LOG_DIR/"
echo "  Merged output    : $MERGED_OUT/"
echo "=========================================================="

# --- Launch workers ---------------------------------------------------------

declare -a PIDS=()
declare -a LABELS=()

start_worker() {
    local label="$1"; shift
    local log="$1"; shift
    echo "  launching $label ..."
    "$@" > "$log" 2>&1 &
    PIDS+=($!)
    LABELS+=("$label")
}

# MusicNN workers -- force CPU-only (CUDA_VISIBLE_DEVICES="" prevents
# TF from allocating GPU memory even though --device cpu is set).
for ((i=0; i<MUSICNN_WORKERS; i++)); do
    start_worker \
        "musicnn[$i/$MUSICNN_WORKERS]" \
        "$LOG_DIR/musicnn_shard${i}of${MUSICNN_WORKERS}.log" \
        env CUDA_VISIBLE_DEVICES="" \
        python extract_pretrained_embeddings.py \
            --models musicnn \
            --output-dir "$MUSICNN_OUT" \
            --device cpu \
            --shard-index "$i" --num-shards "$MUSICNN_WORKERS"
done

# MERT (GPU) -- no sharding for GPU models (VRAM cost of duplicating
# the model outweighs the speedup).
start_worker \
    "mert" \
    "$LOG_DIR/mert.log" \
    python extract_pretrained_embeddings.py \
        --models mert \
        --output-dir "$MERT_OUT" \
        --device cuda

# EnCodecMAE (GPU)
start_worker \
    "encodecmae" \
    "$LOG_DIR/encodecmae.log" \
    python extract_pretrained_embeddings.py \
        --models encodecmae \
        --output-dir "$ENCODEC_OUT" \
        --device cuda

echo ""
echo "Launched ${#PIDS[@]} workers. Waiting for all to finish..."
echo "Tail a log in another shell, e.g.:  tail -f $LOG_DIR/mert.log"
echo ""

# --- Wait for every worker --------------------------------------------------

FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    label="${LABELS[$idx]}"
    if wait "$pid"; then
        echo "  [ok]   $label (pid $pid)"
    else
        status=$?
        echo "  [FAIL] $label (pid $pid, exit $status)" >&2
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "" >&2
    echo "$FAILED worker(s) failed. Inspect $LOG_DIR/ and rerun this script --" >&2
    echo "completed songs are skipped on resume, so lost work is bounded." >&2
    exit 2
fi

# --- Merge + fused CSV ------------------------------------------------------

echo ""
echo "All workers finished. Merging per-model outputs..."
python merge_sharded_embeddings.py \
    --sources "$MUSICNN_OUT" "$MERT_OUT" "$ENCODEC_OUT" \
    --output-dir "$MERGED_OUT" \
    --audio-dir audio \
    2>&1 | tee "$LOG_DIR/merge.log"

echo ""
echo "Done. Fused CSV: $MERGED_OUT/feature_vectors.csv"
