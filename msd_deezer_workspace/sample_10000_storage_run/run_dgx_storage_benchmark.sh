#!/usr/bin/env bash
# Run the storage-estimation pipeline with all outputs scoped to this folder.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$BENCH_DIR/.." && pwd)"
cd "$WORKSPACE_DIR"

BENCH_DIR="$(realpath -m "$BENCH_DIR")"
WORKSPACE_DIR="$(realpath -m "$WORKSPACE_DIR")"

if [[ "$(basename "$BENCH_DIR")" != "sample_10000_storage_run" ]]; then
    echo "ERROR: unexpected benchmark directory: $BENCH_DIR" >&2
    exit 1
fi

if [[ "$(realpath -m "$(dirname "$BENCH_DIR")")" != "$WORKSPACE_DIR" ]]; then
    echo "ERROR: benchmark directory is not directly under workspace: $BENCH_DIR" >&2
    exit 1
fi

case "$BENCH_DIR" in
    "$WORKSPACE_DIR"/sample_10000_storage_run) ;;
    *)
        echo "ERROR: refusing to run outside sample_10000_storage_run: $BENCH_DIR" >&2
        exit 1
        ;;
esac

DEVICE="${DEVICE:-cuda}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
FEATURE_WORKERS="${FEATURE_WORKERS:-32}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-16}"
GPU_MAX_BATCH_SIZE="${GPU_MAX_BATCH_SIZE:-128}"

AUDIO_DIR="$BENCH_DIR/audio_10000"
HANDCRAFTED_DIR="$BENCH_DIR/audio_handcrafted"
PRETRAINED_DIR="$BENCH_DIR/audio_pretrained"
FEATURES_DIR="$BENCH_DIR/features"
EMBEDDINGS_DIR="$BENCH_DIR/pretrained_embeddings"
LOG_DIR="$BENCH_DIR/logs"
CACHE_DIR="$BENCH_DIR/cache"
DATA_DIR="$BENCH_DIR/data"
PREPROCESS_SUMMARY="$DATA_DIR/preprocess_summary.json"
REPORT_PATH="$BENCH_DIR/data/storage_report.json"

assert_under_bench() {
    local label="$1"
    local path
    path="$(realpath -m "$2")"
    case "$path" in
        "$BENCH_DIR"/*) ;;
        *)
            echo "ERROR: $label escapes benchmark root: $path" >&2
            exit 1
            ;;
    esac
}

assert_output_dir() {
    local label="$1"
    local path
    path="$(realpath -m "$2")"
    assert_under_bench "$label" "$path"
    case "$path" in
        "$AUDIO_DIR"|"$AUDIO_DIR"/*)
            echo "ERROR: $label must not point at or inside source audio: $path" >&2
            exit 1
            ;;
    esac
}

if [[ "$(realpath -m "$AUDIO_DIR")" != "$BENCH_DIR/audio_10000" ]]; then
    echo "ERROR: source audio must be exactly $BENCH_DIR/audio_10000" >&2
    exit 1
fi

assert_under_bench "source audio" "$AUDIO_DIR"
assert_output_dir "handcrafted output" "$HANDCRAFTED_DIR"
assert_output_dir "pretrained output" "$PRETRAINED_DIR"
assert_output_dir "features output" "$FEATURES_DIR"
assert_output_dir "embeddings output" "$EMBEDDINGS_DIR"
assert_output_dir "log output" "$LOG_DIR"
assert_output_dir "cache output" "$CACHE_DIR"
assert_output_dir "data output" "$DATA_DIR"
assert_output_dir "preprocess summary output" "$PREPROCESS_SUMMARY"
assert_output_dir "report output" "$REPORT_PATH"

if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "ERROR: missing $AUDIO_DIR. Run copy_10000_from_downloaded.sh first." >&2
    exit 1
fi

audio_count="$(find "$AUDIO_DIR" -maxdepth 1 -type f \( -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.flac' -o -iname '*.wav' \) | wc -l)"
if [[ "$audio_count" -ne 10000 ]]; then
    echo "ERROR: expected 10000 files in $AUDIO_DIR, found $audio_count." >&2
    exit 1
fi

if [[ "$DEVICE" == "cuda" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set. Pick a free GPU first." >&2
    exit 1
fi

if [[ -d "/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda}"
fi

mkdir -p \
    "$LOG_DIR" \
    "$DATA_DIR" \
    "$CACHE_DIR/huggingface/hub" \
    "$CACHE_DIR/huggingface/transformers" \
    "$CACHE_DIR/torch" \
    "$CACHE_DIR/xdg" \
    "$CACHE_DIR/keras" \
    "$CACHE_DIR/tfhub" \
    "$CACHE_DIR/cuda" \
    "$CACHE_DIR/matplotlib" \
    "$CACHE_DIR/numba"

# Keep model/framework caches inside this benchmark run. This avoids hidden
# writes to $HOME/.cache or other shared DGX locations during first model load.
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TRANSFORMERS_CACHE="$CACHE_DIR/huggingface/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export KERAS_HOME="$CACHE_DIR/keras"
export TFHUB_CACHE_DIR="$CACHE_DIR/tfhub"
export CUDA_CACHE_PATH="$CACHE_DIR/cuda"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export NUMBA_CACHE_DIR="$CACHE_DIR/numba"

ts() { date +'%Y-%m-%d %H:%M:%S'; }

run_step() {
    local label="$1"; shift
    local logfile="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${label}.log"
    echo "[$(ts)] >>> $label"
    echo "[$(ts)] >>> log: $logfile"
    if "$@" 2>&1 | tee "$logfile"; then
        echo "[$(ts)] <<< $label DONE"
    else
        echo "[$(ts)] <<< $label FAILED" >&2
        exit 1
    fi
}

run_step check_pretrained_models \
    python extract_pretrained_embeddings.py \
        --models musicnn,mert,encodecmae \
        --check-models \
        --device "$DEVICE"

run_step preprocess \
    python audio_preprocessing/processor.py \
        --source-dir "$AUDIO_DIR" \
        --handcrafted-dir "$HANDCRAFTED_DIR" \
        --pretrained-dir "$PRETRAINED_DIR" \
        --summary-path "$PREPROCESS_SUMMARY" \
        --workers "$PREPROCESS_WORKERS"

run_step extract_handcrafted \
    python extract_audio_features.py \
        --input-dir "$HANDCRAFTED_DIR" \
        --output-dir "$FEATURES_DIR" \
        --workers "$FEATURE_WORKERS"

run_step extract_pretrained \
    python extract_pretrained_embeddings.py \
        --input-dir "$PRETRAINED_DIR" \
        --output-dir "$EMBEDDINGS_DIR" \
        --device "$DEVICE" \
        --batch-size "$GPU_BATCH_SIZE" \
        --max-batch-size "$GPU_MAX_BATCH_SIZE"

run_step verify_pretrained \
    python - "$EMBEDDINGS_DIR/extraction_summary.json" "$audio_count" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
expected = int(sys.argv[2])
required = ("musicnn", "mert", "encodecmae")

if not summary_path.is_file():
    raise SystemExit(f"Missing pretrained extraction summary: {summary_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
processed = int(summary.get("processed", -1))
if processed != expected:
    raise SystemExit(f"Expected {expected} processed songs, got {processed}")

success = summary.get("per_model_success", {})
errors = summary.get("per_model_errors", {})
bad = []
for model in required:
    if int(success.get(model, -1)) != expected:
        bad.append(f"{model} success={success.get(model)} expected={expected}")
    if int(errors.get(model, 0)) != 0:
        bad.append(f"{model} errors={errors.get(model)} expected=0")

if bad:
    raise SystemExit("Invalid pretrained extraction: " + "; ".join(bad))

print("pretrained extraction verified for all required models")
PY

run_step storage_report \
    python - "$BENCH_DIR" "$REPORT_PATH" "$audio_count" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
sample_count = int(sys.argv[3])
full_count = 812_353

items = {
    "audio_10000": "audio_10000",
    "audio_handcrafted": "audio_handcrafted",
    "audio_pretrained": "audio_pretrained",
    "features_total": "features",
    "features_raw": "features/raw",
    "features_feature_vectors_csv": "features/feature_vectors.csv",
    "pretrained_embeddings_total": "pretrained_embeddings",
    "pretrained_embeddings_raw": "pretrained_embeddings/raw",
    "pretrained_musicnn_vectors_csv": "pretrained_embeddings/musicnn_vectors.csv",
    "pretrained_mert_vectors_csv": "pretrained_embeddings/mert_vectors.csv",
    "pretrained_encodecmae_vectors_csv": "pretrained_embeddings/encodecmae_vectors.csv",
    "pretrained_fused_feature_vectors_csv": "pretrained_embeddings/feature_vectors.csv",
}

def files_under(path: Path):
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.is_file()]

def load_json(relative: str):
    path = root / relative
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def summarize(path: Path):
    files = files_under(path)
    total = sum(p.stat().st_size for p in files)
    return {
        "path": str(path),
        "exists": path.exists(),
        "files": len(files),
        "bytes": total,
        "gib": round(total / (1024 ** 3), 6),
        "extrapolated_812353_gib": round((total / sample_count * full_count) / (1024 ** 3), 3),
    }

copy_metadata = load_json("data/copy_metadata.json")
preprocess = load_json("data/preprocess_summary.json")
pretrained = load_json("pretrained_embeddings/extraction_summary.json")

payload = {
    "root": str(root),
    "sample_count": sample_count,
    "full_count": full_count,
    "copy_metadata": copy_metadata,
    "manifest": str(root / "data/copied_audio_manifest.csv"),
    "preprocess_summary": {
        key: preprocess.get(key)
        for key in (
            "total",
            "processed",
            "skipped_existing",
            "removed_too_short",
            "removed_silent",
            "errors",
            "peak_limited_handcrafted",
            "peak_limited_pretrained",
        )
    } if preprocess else None,
    "pretrained_summary": {
        "processed": pretrained.get("processed"),
        "errors": pretrained.get("errors"),
        "per_model_success": pretrained.get("per_model_success"),
        "per_model_errors": pretrained.get("per_model_errors"),
        "device_request": pretrained.get("device_request"),
        "batch_size": pretrained.get("batch_size"),
        "max_batch_size": pretrained.get("max_batch_size"),
        "elapsed_sec": pretrained.get("elapsed_sec"),
    } if pretrained else None,
    "items": {name: summarize(root / rel) for name, rel in items.items()},
}

payload["totals"] = {
    "audio_and_outputs_bytes": sum(
        payload["items"][name]["bytes"]
        for name in (
            "audio_10000",
            "audio_handcrafted",
            "audio_pretrained",
            "features_total",
            "pretrained_embeddings_total",
        )
    ),
}
payload["totals"]["audio_and_outputs_gib"] = round(
    payload["totals"]["audio_and_outputs_bytes"] / (1024 ** 3), 6
)
payload["totals"]["audio_and_outputs_extrapolated_812353_gib"] = round(
    (payload["totals"]["audio_and_outputs_bytes"] / sample_count * full_count) / (1024 ** 3),
    3,
)

# Cache is useful for DGX operational storage but should not be scaled as a
# per-song artifact in the storage request.
cache_summary = summarize(root / "cache")
payload["cache"] = {
    **cache_summary,
    "extrapolated_812353_gib": None,
    "note": "Model/framework cache scoped to this benchmark; not a per-song output.",
    }

report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "[$(ts)] Report: $REPORT_PATH"
