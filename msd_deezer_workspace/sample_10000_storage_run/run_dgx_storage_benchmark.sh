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
MUSICNN_WORKERS="${MUSICNN_WORKERS:-16}"
GPU_PREFETCH="${GPU_PREFETCH:-32}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-8}"
GPU_MAX_BATCH_SIZE="${GPU_MAX_BATCH_SIZE:-64}"
MERGE_WORKERS="${MERGE_WORKERS:-16}"

AUDIO_DIR="$BENCH_DIR/audio_10000"
HANDCRAFTED_DIR="$BENCH_DIR/audio_handcrafted"
PRETRAINED_DIR="$BENCH_DIR/audio_pretrained"
FEATURES_DIR="$BENCH_DIR/features"
EMBEDDINGS_DIR="$BENCH_DIR/pretrained_embeddings"
EMBEDDINGS_MUSICNN_DIR="$BENCH_DIR/pretrained_embeddings_musicnn"
EMBEDDINGS_MERT_DIR="$BENCH_DIR/pretrained_embeddings_mert"
EMBEDDINGS_ENCODEC_DIR="$BENCH_DIR/pretrained_embeddings_encodecmae"
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

require_positive_int() {
    local label="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
        echo "ERROR: $label must be a positive integer, got: $value" >&2
        exit 1
    fi
}

require_nonnegative_int() {
    local label="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $label must be a non-negative integer, got: $value" >&2
        exit 1
    fi
}

for setting in \
    PREPROCESS_WORKERS \
    FEATURE_WORKERS \
    MUSICNN_WORKERS \
    GPU_BATCH_SIZE \
    GPU_MAX_BATCH_SIZE \
    MERGE_WORKERS
do
    require_positive_int "$setting" "${!setting}"
done
require_nonnegative_int "GPU_PREFETCH" "$GPU_PREFETCH"

assert_under_bench "source audio" "$AUDIO_DIR"
assert_output_dir "handcrafted output" "$HANDCRAFTED_DIR"
assert_output_dir "pretrained output" "$PRETRAINED_DIR"
assert_output_dir "features output" "$FEATURES_DIR"
assert_output_dir "embeddings output" "$EMBEDDINGS_DIR"
assert_output_dir "MusicNN shard output" "$EMBEDDINGS_MUSICNN_DIR"
assert_output_dir "MERT shard output" "$EMBEDDINGS_MERT_DIR"
assert_output_dir "EnCodecMAE shard output" "$EMBEDDINGS_ENCODEC_DIR"
assert_output_dir "log output" "$LOG_DIR"
assert_output_dir "cache output" "$CACHE_DIR"
assert_output_dir "data output" "$DATA_DIR"
assert_output_dir "preprocess summary output" "$PREPROCESS_SUMMARY"
assert_output_dir "report output" "$REPORT_PATH"

if [[ ! -d "$AUDIO_DIR" ]]; then
    echo "ERROR: missing $AUDIO_DIR. Run copy_10000_from_downloaded.sh first." >&2
    exit 1
fi

count_audio_files() {
    find "$1" -maxdepth 1 -type f \( \
        -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.flac' -o -iname '*.wav' \
    \) 2>/dev/null | wc -l
}

audio_count="$(count_audio_files "$AUDIO_DIR")"
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

run_parallel_pretrained() {
    local expected_count="$1"
    local parallel_log_dir="$LOG_DIR/pretrained_parallel_$(date +%Y%m%d_%H%M%S)"
    local merged_resolved

    merged_resolved="$(realpath -m "$EMBEDDINGS_DIR")"
    if [[ "$merged_resolved" != "$BENCH_DIR/pretrained_embeddings" ]]; then
        echo "ERROR: refusing to clean unexpected merged output: $merged_resolved" >&2
        return 2
    fi
    rm -rf --one-file-system -- "$merged_resolved"

    mkdir -p \
        "$parallel_log_dir" \
        "$EMBEDDINGS_MUSICNN_DIR" \
        "$EMBEDDINGS_MERT_DIR" \
        "$EMBEDDINGS_ENCODEC_DIR" \
        "$EMBEDDINGS_DIR"

    echo "=========================================================="
    echo "Parallel pretrained-embedding extraction"
    echo "=========================================================="
    echo "  Pretrained audio : $PRETRAINED_DIR"
    echo "  Expected files   : $expected_count"
    echo "  MusicNN workers  : $MUSICNN_WORKERS  (CPU shards)"
    echo "  MERT             : 1 process  ($DEVICE)"
    echo "  EnCodecMAE       : 1 process  ($DEVICE)"
    echo "  GPU batch        : start=$GPU_BATCH_SIZE, cap=$GPU_MAX_BATCH_SIZE"
    echo "  GPU prefetch     : $GPU_PREFETCH"
    echo "  Logs             : $parallel_log_dir/"
    echo "  Merged output    : $EMBEDDINGS_DIR/"
    echo "=========================================================="

    declare -a PIDS=()
    declare -a LABELS=()

    cleanup_parallel() {
        local sig="${1:-INT}"
        echo "" >&2
        echo "Caught SIG${sig} -- terminating ${#PIDS[@]} pretrained worker(s)..." >&2
        for pid in "${PIDS[@]:-}"; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        for _ in 1 2 3 4 5; do
            local alive=0
            for pid in "${PIDS[@]:-}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    alive=1
                    break
                fi
            done
            [[ "$alive" -eq 0 ]] && break
            sleep 1
        done
        for pid in "${PIDS[@]:-}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        wait 2>/dev/null || true
        exit 130
    }
    trap 'cleanup_parallel INT' INT
    trap 'cleanup_parallel TERM' TERM

    start_worker() {
        local label="$1"; shift
        local log="$1"; shift
        echo "  launching $label ..."
        "$@" > "$log" 2>&1 &
        PIDS+=($!)
        LABELS+=("$label")
    }

    for ((i=0; i<MUSICNN_WORKERS; i++)); do
        start_worker \
            "musicnn[$i/$MUSICNN_WORKERS]" \
            "$parallel_log_dir/musicnn_shard${i}of${MUSICNN_WORKERS}.log" \
            env CUDA_VISIBLE_DEVICES="" \
            python extract_pretrained_embeddings.py \
                --models musicnn \
                --input-dir "$PRETRAINED_DIR" \
                --output-dir "$EMBEDDINGS_MUSICNN_DIR" \
                --device cpu \
                --prefetch 0 \
                --shard-index "$i" \
                --num-shards "$MUSICNN_WORKERS"
    done

    start_worker \
        "mert" \
        "$parallel_log_dir/mert.log" \
        python extract_pretrained_embeddings.py \
            --models mert \
            --input-dir "$PRETRAINED_DIR" \
            --output-dir "$EMBEDDINGS_MERT_DIR" \
            --device "$DEVICE" \
            --prefetch "$GPU_PREFETCH" \
            --batch-size "$GPU_BATCH_SIZE" \
            --max-batch-size "$GPU_MAX_BATCH_SIZE"

    start_worker \
        "encodecmae" \
        "$parallel_log_dir/encodecmae.log" \
        python extract_pretrained_embeddings.py \
            --models encodecmae \
            --input-dir "$PRETRAINED_DIR" \
            --output-dir "$EMBEDDINGS_ENCODEC_DIR" \
            --device "$DEVICE" \
            --prefetch "$GPU_PREFETCH" \
            --batch-size "$GPU_BATCH_SIZE" \
            --max-batch-size "$GPU_MAX_BATCH_SIZE"

    count_npz() {
        local raw="$1/raw"
        if [[ ! -d "$raw" ]]; then
            echo 0
            return
        fi
        find "$raw" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l
    }

    draw_bar() {
        local done_count=$1 total_count=$2 width=${3:-40}
        [[ "$total_count" -le 0 ]] && total_count=1
        local filled=$(( done_count * width / total_count ))
        [[ "$filled" -gt "$width" ]] && filled="$width"
        local empty=$(( width - filled ))
        local bar="" i
        for ((i=0; i<filled; i++)); do bar+="#"; done
        for ((i=0; i<empty; i++)); do bar+="."; done
        local pct=$(( done_count * 100 / total_count ))
        printf "[%s] %3d%%  %d/%d" "$bar" "$pct" "$done_count" "$total_count"
    }

    workers_alive() {
        local pid
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                return 0
            fi
        done
        return 1
    }

    draw_progress() {
        local musicnn_done mert_done encodec_done
        musicnn_done="$(count_npz "$EMBEDDINGS_MUSICNN_DIR")"
        mert_done="$(count_npz "$EMBEDDINGS_MERT_DIR")"
        encodec_done="$(count_npz "$EMBEDDINGS_ENCODEC_DIR")"
        printf "  musicnn     %s\n" "$(draw_bar "$musicnn_done" "$expected_count")"
        printf "  mert        %s\n" "$(draw_bar "$mert_done" "$expected_count")"
        printf "  encodecmae  %s\n" "$(draw_bar "$encodec_done" "$expected_count")"
    }

    echo ""
    echo "Launched ${#PIDS[@]} workers. Detailed logs:"
    echo "  tail -f $parallel_log_dir/mert.log"
    echo "  tail -f $parallel_log_dir/encodecmae.log"
    echo "  tail -f $parallel_log_dir/musicnn_shard0of${MUSICNN_WORKERS}.log"
    echo ""

    while workers_alive; do
        draw_progress
        sleep 60
    done
    draw_progress
    echo ""

    local failed=0
    for idx in "${!PIDS[@]}"; do
        local pid="${PIDS[$idx]}"
        local label="${LABELS[$idx]}"
        if wait "$pid"; then
            echo "  [ok]   $label (pid $pid)"
        else
            local status=$?
            echo "  [FAIL] $label (pid $pid, exit $status)" >&2
            failed=$((failed + 1))
        fi
    done

    if [[ "$failed" -gt 0 ]]; then
        echo "$failed pretrained worker(s) failed. Inspect $parallel_log_dir/ and rerun." >&2
        return 2
    fi

    echo "All pretrained workers finished. Merging per-model outputs..."
    python merge_sharded_embeddings.py \
        --sources "$EMBEDDINGS_MUSICNN_DIR" "$EMBEDDINGS_MERT_DIR" "$EMBEDDINGS_ENCODEC_DIR" \
        --output-dir "$EMBEDDINGS_DIR" \
        --audio-dir "$PRETRAINED_DIR" \
        --workers "$MERGE_WORKERS" \
        2>&1 | tee "$parallel_log_dir/merge.log"

    python - \
        "$EMBEDDINGS_DIR" \
        "$EMBEDDINGS_MUSICNN_DIR" \
        "$EMBEDDINGS_MERT_DIR" \
        "$EMBEDDINGS_ENCODEC_DIR" \
        "$expected_count" \
        "$DEVICE" \
        "$GPU_BATCH_SIZE" \
        "$GPU_MAX_BATCH_SIZE" \
        "$GPU_PREFETCH" \
        "$MUSICNN_WORKERS" \
        "$MERGE_WORKERS" \
        "$parallel_log_dir" <<'PY'
import json
import sys
from pathlib import Path

merged_dir = Path(sys.argv[1])
source_dirs = {
    "musicnn": Path(sys.argv[2]),
    "mert": Path(sys.argv[3]),
    "encodecmae": Path(sys.argv[4]),
}
expected = int(sys.argv[5])
device = sys.argv[6]
batch_size = int(sys.argv[7])
max_batch_size = int(sys.argv[8])
prefetch = int(sys.argv[9])
musicnn_workers = int(sys.argv[10])
merge_workers = int(sys.argv[11])
log_dir = Path(sys.argv[12])

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def raw_count(path: Path) -> int:
    raw = path / "raw"
    if not raw.is_dir():
        return 0
    return sum(1 for _ in raw.glob("*.npz"))

def csv_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)

per_model_success = {}
per_model_errors = {}
per_model_elapsed = {}
bad = []

musicnn_summaries = sorted(
    source_dirs["musicnn"].glob("extraction_summary_shard*of*.json")
)
if not musicnn_summaries:
    bad.append("musicnn has no shard extraction summaries")
else:
    success = 0
    errors = 0
    elapsed = 0.0
    for summary_path in musicnn_summaries:
        summary = load_json(summary_path)
        success += int(summary.get("per_model_success", {}).get("musicnn", 0))
        errors += int(summary.get("per_model_errors", {}).get("musicnn", 0))
        elapsed = max(elapsed, float(summary.get("elapsed_sec", 0) or 0))
    per_model_success["musicnn"] = success
    per_model_errors["musicnn"] = errors
    per_model_elapsed["musicnn_wall_sec"] = elapsed

for model in ("mert", "encodecmae"):
    summary_path = source_dirs[model] / "extraction_summary.json"
    if not summary_path.is_file():
        bad.append(f"{model} missing extraction_summary.json")
        continue
    summary = load_json(summary_path)
    per_model_success[model] = int(summary.get("per_model_success", {}).get(model, 0))
    per_model_errors[model] = int(summary.get("per_model_errors", {}).get(model, 0))
    per_model_elapsed[f"{model}_wall_sec"] = float(summary.get("elapsed_sec", 0) or 0)

for model, source in source_dirs.items():
    count = raw_count(source)
    if count != expected:
        bad.append(f"{model} raw npz count={count}, expected={expected}")
    if per_model_success.get(model) != expected:
        bad.append(f"{model} success={per_model_success.get(model)}, expected={expected}")
    if per_model_errors.get(model, 0) != 0:
        bad.append(f"{model} errors={per_model_errors.get(model)}, expected=0")

merged_count = raw_count(merged_dir)
if merged_count != expected:
    bad.append(f"merged raw npz count={merged_count}, expected={expected}")

csv_counts = {}
for name in (
    "musicnn_vectors.csv",
    "mert_vectors.csv",
    "encodecmae_vectors.csv",
    "feature_vectors.csv",
):
    rows = csv_rows(merged_dir / name)
    csv_counts[name] = rows
    if rows != expected:
        bad.append(f"{name} rows={rows}, expected={expected}")

if bad:
    raise SystemExit("Invalid parallel pretrained extraction: " + "; ".join(bad))

summary = {
    "mode": "parallel_per_model",
    "total": expected,
    "processed": expected,
    "errors": sum(per_model_errors.values()),
    "per_model_success": per_model_success,
    "per_model_errors": per_model_errors,
    "device_request": device,
    "batch_size": batch_size,
    "max_batch_size": max_batch_size,
    "prefetch": prefetch,
    "musicnn_workers": musicnn_workers,
    "merge_workers": merge_workers,
    "parallel_log_dir": str(log_dir),
    "per_model_elapsed": per_model_elapsed,
    "csv_rows": csv_counts,
}
(merged_dir / "extraction_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("parallel pretrained extraction verified and summarized")
PY

    trap - INT TERM
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

pretrained_audio_count="$(count_audio_files "$PRETRAINED_DIR")"
if [[ "$pretrained_audio_count" -le 0 ]]; then
    echo "ERROR: no preprocessed audio files found in $PRETRAINED_DIR." >&2
    exit 1
fi
echo "[$(ts)] Pretrained extraction input count: $pretrained_audio_count"

run_step extract_pretrained run_parallel_pretrained "$pretrained_audio_count"

run_step verify_pretrained \
    python - "$EMBEDDINGS_DIR/extraction_summary.json" "$pretrained_audio_count" <<'PY'
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
    python - "$BENCH_DIR" "$REPORT_PATH" "$audio_count" "$pretrained_audio_count" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2]).resolve()
sample_count = int(sys.argv[3])
pretrained_audio_count = int(sys.argv[4])
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
    "pretrained_intermediate_musicnn": "pretrained_embeddings_musicnn",
    "pretrained_intermediate_mert": "pretrained_embeddings_mert",
    "pretrained_intermediate_encodecmae": "pretrained_embeddings_encodecmae",
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
    "pretrained_audio_count": pretrained_audio_count,
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
        "mode": pretrained.get("mode"),
        "processed": pretrained.get("processed"),
        "errors": pretrained.get("errors"),
        "per_model_success": pretrained.get("per_model_success"),
        "per_model_errors": pretrained.get("per_model_errors"),
        "device_request": pretrained.get("device_request"),
        "batch_size": pretrained.get("batch_size"),
        "max_batch_size": pretrained.get("max_batch_size"),
        "prefetch": pretrained.get("prefetch"),
        "musicnn_workers": pretrained.get("musicnn_workers"),
        "per_model_elapsed": pretrained.get("per_model_elapsed"),
    } if pretrained else None,
    "items": {name: summarize(root / rel) for name, rel in items.items()},
}

payload["items"]["pretrained_intermediates_total"] = {
    "path": "pretrained_embeddings_musicnn + pretrained_embeddings_mert + pretrained_embeddings_encodecmae",
    "exists": all(
        payload["items"][name]["exists"]
        for name in (
            "pretrained_intermediate_musicnn",
            "pretrained_intermediate_mert",
            "pretrained_intermediate_encodecmae",
        )
    ),
    "files": sum(
        payload["items"][name]["files"]
        for name in (
            "pretrained_intermediate_musicnn",
            "pretrained_intermediate_mert",
            "pretrained_intermediate_encodecmae",
        )
    ),
    "bytes": sum(
        payload["items"][name]["bytes"]
        for name in (
            "pretrained_intermediate_musicnn",
            "pretrained_intermediate_mert",
            "pretrained_intermediate_encodecmae",
        )
    ),
}
payload["items"]["pretrained_intermediates_total"]["gib"] = round(
    payload["items"]["pretrained_intermediates_total"]["bytes"] / (1024 ** 3), 6
)
payload["items"]["pretrained_intermediates_total"]["extrapolated_812353_gib"] = round(
    (
        payload["items"]["pretrained_intermediates_total"]["bytes"]
        / sample_count
        * full_count
    )
    / (1024 ** 3),
    3,
)

payload["totals"] = {
    "audio_and_final_outputs_bytes": sum(
        payload["items"][name]["bytes"]
        for name in (
            "audio_10000",
            "audio_handcrafted",
            "audio_pretrained",
            "features_total",
            "pretrained_embeddings_total",
        )
    ),
    "parallel_pretrained_intermediates_bytes": payload["items"]["pretrained_intermediates_total"]["bytes"],
}
payload["totals"]["audio_and_final_outputs_gib"] = round(
    payload["totals"]["audio_and_final_outputs_bytes"] / (1024 ** 3), 6
)
payload["totals"]["audio_and_final_outputs_extrapolated_812353_gib"] = round(
    (
        payload["totals"]["audio_and_final_outputs_bytes"]
        / sample_count
        * full_count
    )
    / (1024 ** 3),
    3,
)
payload["totals"]["peak_with_parallel_intermediates_bytes"] = (
    payload["totals"]["audio_and_final_outputs_bytes"]
    + payload["totals"]["parallel_pretrained_intermediates_bytes"]
)
payload["totals"]["peak_with_parallel_intermediates_gib"] = round(
    payload["totals"]["peak_with_parallel_intermediates_bytes"] / (1024 ** 3), 6
)
payload["totals"]["peak_with_parallel_intermediates_extrapolated_812353_gib"] = round(
    (
        payload["totals"]["peak_with_parallel_intermediates_bytes"]
        / sample_count
        * full_count
    )
    / (1024 ** 3),
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
