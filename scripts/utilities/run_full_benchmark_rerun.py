from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean generated artifacts, verify the download state, rerun the remaining "
            "pipeline stages, and execute the benchmark suites with per-step logs."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the pipeline steps.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional suffix for the run-log directory name.",
    )
    parser.add_argument(
        "--preserve-download-state",
        action="store_true",
        help=(
            "Preserve the completed download inputs (audio_files/, data/songs.csv, "
            "checkpoint, and download cache) and rerun only the post-download stages."
        ),
    )
    parser.add_argument(
        "--include-plot",
        action="store_true",
        help="Include the bulk plotting stage in the rerun.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def remove_path(path: Path) -> Dict[str, object]:
    existed = path.exists()
    item_type = "missing"
    if existed:
        item_type = "directory" if path.is_dir() else "file"
        attempts = 5
        for attempt in range(1, attempts + 1):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt >= attempts:
                    raise
                sleep(1.0 * attempt)
    return {
        "path": str(path),
        "existed": bool(existed),
        "type": item_type,
    }


def run_logged_step(
    name: str,
    command: List[str],
    log_dir: Path,
) -> Dict[str, object]:
    log_path = log_dir / f"{name}.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    started_perf = perf_counter()

    print(f"\n[{name}] Starting")
    print("$", " ".join(command))

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    duration_sec = round(perf_counter() - started_perf, 2)
    print(f"[{name}] Exit code: {result.returncode} | duration: {duration_sec:.2f}s")
    print(f"[{name}] Log: {log_path}")

    if result.returncode != 0:
        tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        if tail_lines:
            print(f"[{name}] Last log lines:")
            for line in tail_lines:
                print(line)
        raise subprocess.CalledProcessError(result.returncode, command)

    return {
        "name": name,
        "command": command,
        "log_path": str(log_path),
        "started_at": started_at,
        "duration_sec": duration_sec,
        "returncode": int(result.returncode),
    }


def iter_audio_files(audio_dir: Path) -> List[Path]:
    if not audio_dir.exists():
        return []
    return sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def csv_row_has_audio(row: Dict[str, str]) -> bool:
    has_audio_raw = str(row.get("has_audio", "")).strip()
    if has_audio_raw:
        return has_audio_raw.lower() in {"true", "1", "yes"}
    return bool(str(row.get("filename", "")).strip())


def csv_row_audio_basename(row: Dict[str, str]) -> str:
    explicit = str(row.get("audio_basename", "")).strip()
    if explicit:
        return explicit
    filename = str(row.get("filename", "")).strip()
    return Path(filename).stem if filename else ""


def summarize_extension_counts(paths: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in paths:
        key = path.suffix.lower()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def load_checkpoint_counts(checkpoint_path: Path) -> Dict[str, int]:
    payload = read_json(checkpoint_path)
    if not payload:
        return {"processed": 0, "downloaded": 0, "failed": 0}

    processed_indices = set(payload.get("processed_indices", []))
    failed_indices = set(payload.get("failed_indices", []))
    downloaded_keys = {
        str(
            item.get("msd_track_id")
            or item.get("deezer_track_id")
            or item.get("filename")
            or ""
        ).strip()
        for item in payload.get("downloaded_songs", [])
        if isinstance(item, dict)
    }
    downloaded_keys = {key for key in downloaded_keys if key}

    return {
        "processed": int(len(processed_indices | failed_indices)),
        "downloaded": int(len(downloaded_keys)),
        "failed": int(len(failed_indices)),
    }


def read_log_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False


def verify_download_state() -> Dict[str, object]:
    audio_dir = PROJECT_ROOT / "audio_files"
    songs_csv = PROJECT_ROOT / "data" / "songs.csv"
    checkpoint_path = PROJECT_ROOT / "download_checkpoint_with_genre.json"
    latest_download_summary_path = (
        PROJECT_ROOT
        / "docs"
        / "reports"
        / "run_logs"
        / "latest_resilient_download"
        / "run_summary.json"
    )

    audio_files = iter_audio_files(audio_dir)
    audio_basenames = {path.stem for path in audio_files}
    zero_byte_files = [path.name for path in audio_files if path.stat().st_size <= 0]

    songs_rows = 0
    songs_audio_rows = 0
    songs_audio_basenames = set()
    if songs_csv.exists():
        with songs_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                songs_rows += 1
                if csv_row_has_audio(row):
                    songs_audio_rows += 1
                    basename = csv_row_audio_basename(row)
                    if basename:
                        songs_audio_basenames.add(basename)

    orphaned_audio_basenames = sorted(audio_basenames - songs_audio_basenames)
    missing_audio_basenames = sorted(songs_audio_basenames - audio_basenames)

    latest_summary = read_json(latest_download_summary_path)
    sessions = latest_summary.get("sessions", []) if latest_summary else []
    no_progress_sessions = [
        session
        for session in sessions
        if int(session.get("processed_gain", 0)) <= 0
        and int(session.get("downloaded_gain", 0)) <= 0
        and int(session.get("audio_gain", 0)) <= 0
    ]
    misleading_success_logs = [
        str(session.get("log_path", ""))
        for session in no_progress_sessions
        if str(session.get("log_path", "")).strip()
        and read_log_contains(
            Path(str(session.get("log_path", ""))),
            "All songs downloaded successfully!",
        )
    ]

    checkpoint_counts = load_checkpoint_counts(checkpoint_path)

    notes: List[str] = []
    unique_audio_file_count = len(audio_basenames)
    if (
        songs_audio_rows == unique_audio_file_count
        and not orphaned_audio_basenames
        and not missing_audio_basenames
    ):
        notes.append(
            "songs.csv audio-backed rows and on-disk audio basenames are aligned."
        )
    else:
        notes.append(
            "Metadata/audio alignment is imperfect and should be treated as a download-state issue."
        )
    if len(audio_files) != unique_audio_file_count:
        notes.append(
            "audio_files/ currently contains multiple formats for some songs "
            "(for example both .mp3 and normalized .wav), so raw file counts are "
            "larger than unique audio basenames."
        )
    if zero_byte_files:
        notes.append(
            f"Detected {len(zero_byte_files)} zero-byte audio file(s)."
        )
    else:
        notes.append("No zero-byte audio files were detected in audio_files/.")
    if int(latest_summary.get("remaining_rows", 0) or 0) > 0:
        notes.append(
            "The resilient downloader still reports remaining catalog rows, so the run is "
            "consistent but not a 100% complete catalog acquisition."
        )
    if misleading_success_logs:
        notes.append(
            "One or more no-progress downloader sessions still printed a success message in "
            "their terminal log, so the final downloader messaging is inconsistent with the "
            "checkpoint summary."
        )

    return {
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "songs_csv_path": str(songs_csv),
        "audio_dir": str(audio_dir),
        "checkpoint_path": str(checkpoint_path),
        "latest_download_summary_path": str(latest_download_summary_path),
        "songs_rows": int(songs_rows),
        "songs_audio_rows": int(songs_audio_rows),
        "audio_files_on_disk": int(len(audio_files)),
        "audio_basenames_on_disk": int(unique_audio_file_count),
        "audio_extension_counts": summarize_extension_counts(audio_files),
        "zero_byte_audio_files": int(len(zero_byte_files)),
        "zero_byte_audio_file_examples": zero_byte_files[:10],
        "orphaned_audio_basenames": int(len(orphaned_audio_basenames)),
        "orphaned_audio_examples": orphaned_audio_basenames[:10],
        "missing_audio_basenames": int(len(missing_audio_basenames)),
        "missing_audio_examples": missing_audio_basenames[:10],
        "checkpoint_processed_rows": int(checkpoint_counts["processed"]),
        "checkpoint_downloaded_rows": int(checkpoint_counts["downloaded"]),
        "checkpoint_failed_rows": int(checkpoint_counts["failed"]),
        "latest_download_catalog_rows": latest_summary.get("catalog_rows"),
        "latest_download_processed_rows": latest_summary.get("processed_rows"),
        "latest_download_downloaded_rows": latest_summary.get("downloaded_rows"),
        "latest_download_audio_files": latest_summary.get("audio_files"),
        "latest_download_remaining_rows": latest_summary.get("remaining_rows"),
        "latest_download_forced_restart_count": latest_summary.get("forced_restart_count"),
        "latest_download_session_count": int(len(sessions)),
        "latest_download_no_progress_sessions": int(len(no_progress_sessions)),
        "no_progress_sessions_with_success_message": int(len(misleading_success_logs)),
        "no_progress_success_log_examples": misleading_success_logs[:5],
        "notes": notes,
    }


def build_cleanup_targets(preserve_download_state: bool) -> List[Path]:
    targets = [
        PROJECT_ROOT / "output",
        PROJECT_ROOT / "data" / "songs_with_merged_genres.csv",
        PROJECT_ROOT / "data" / "songs_genre_list.csv",
        PROJECT_ROOT / "data" / "unique_genres.csv",
    ]
    if preserve_download_state:
        return targets

    return targets + [
        PROJECT_ROOT / "audio_files",
        PROJECT_ROOT / "src" / "data_collection" / "download_stats",
        PROJECT_ROOT / "docs" / "reports" / "run_logs" / "latest_resilient_download",
        PROJECT_ROOT / "data" / "songs.csv",
        PROJECT_ROOT / "data" / "songs_schema_summary.json",
        PROJECT_ROOT / "download_checkpoint_with_genre.json",
        PROJECT_ROOT / "deezer_search_cache.json",
    ]


def build_steps(
    args: argparse.Namespace,
    benchmark_output_dir: Path,
    experiment_root: Path,
) -> List[Dict[str, object]]:
    steps: List[Dict[str, object]] = []
    step_index = 1

    def add_step(label: str, command: List[str]) -> None:
        nonlocal step_index
        steps.append(
            {
                "name": f"{step_index:02d}_{label}",
                "command": [str(part) for part in command],
            }
        )
        step_index += 1

    if args.preserve_download_state:
        add_step(
            "run_audio_preprocessing",
            [
                args.python,
                "scripts/run_audio_preprocessing.py",
                "--audio-dir",
                "audio_files",
                "--target-duration",
                str(fv.baseline_target_duration_seconds),
                "--target-lufs",
                str(fv.baseline_target_lufs),
                "--max-peak-db",
                str(fv.baseline_max_true_peak_dbtp),
            ],
        )
        add_step(
            "rebuild_unified_metadata",
            [args.python, "scripts/utilities/migrate_to_unified_csv.py"],
        )
        add_step(
            "run_feature_extraction",
            [args.python, "src/features/extract_features.py"],
        )
        if args.include_plot:
            add_step(
                "run_feature_plots",
                [
                    args.python,
                    "scripts/visualization/ploting.py",
                    "--features_dir",
                    "output/features",
                    "--plots_dir",
                    "output/plots",
                ],
            )
        add_step(
            "run_feature_qc",
            [args.python, "scripts/analysis/run_feature_qc.py"],
        )
        add_step(
            "run_baseline_gmm",
            [args.python, "src/clustering/gmm.py", "--no-ui"],
        )
        add_step(
            "run_metadata_audit",
            [
                args.python,
                "scripts/analysis/audit_metadata_schema.py",
                "--output-dir",
                "output/metrics",
            ],
        )
    else:
        add_step(
            "run_pipeline",
            [
                args.python,
                "run_pipeline.py",
                "--skip",
                *(["plot"] if not args.include_plot else []),
                "--clustering-method",
                "gmm",
            ],
        )
        add_step(
            "rebuild_unified_metadata",
            [args.python, "scripts/utilities/migrate_to_unified_csv.py"],
        )
        add_step(
            "run_feature_qc",
            [args.python, "scripts/analysis/run_feature_qc.py"],
        )
        add_step(
            "run_metadata_audit",
            [
                args.python,
                "scripts/analysis/audit_metadata_schema.py",
                "--output-dir",
                "output/metrics",
            ],
        )

    add_step(
        "build_merged_genre_dataset",
        [args.python, "scripts/utilities/build_merged_genre_dataset.py"],
    )
    add_step(
        "run_all_clustering_profiles",
        [
            args.python,
            "scripts/run_all_clustering.py",
            "--profiles",
            "recommended_production",
            "all_audio_pca_comparison",
            "all_audio_zscore_comparison",
            "--publish-ui-snapshot",
            "--run-root",
            str(experiment_root),
        ],
    )
    add_step(
        "run_thesis_benchmark",
        [
            args.python,
            "scripts/analysis/thesis_clustering_benchmark.py",
            "--output-dir",
            str(benchmark_output_dir),
        ],
    )
    return steps


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_suffix = f"_{args.run_label.strip()}" if args.run_label.strip() else ""
    run_root = ensure_dir(
        PROJECT_ROOT / "docs" / "reports" / "run_logs" / f"full_pipeline_rerun_{stamp}{label_suffix}"
    )

    download_verification = verify_download_state()
    download_verification_path = run_root / "download_verification.json"
    download_verification_path.write_text(
        json.dumps(download_verification, indent=2),
        encoding="utf-8",
    )
    print(f"[verify] Download verification written to: {download_verification_path}")

    cleanup_targets = build_cleanup_targets(args.preserve_download_state)
    print("\n[cleanup] Removing generated artifacts...")
    cleanup_results = [remove_path(path) for path in cleanup_targets]
    for item in cleanup_results:
        status = "removed" if item["existed"] else "missing"
        print(f"[cleanup] {status}: {item['path']}")

    output_root = ensure_dir(PROJECT_ROOT / "output")
    metrics_root = ensure_dir(output_root / "metrics")
    experiment_root = ensure_dir(output_root / "experiment_runs_taxonomy")
    benchmark_output_dir = metrics_root / f"thesis_benchmark_full_rerun_{stamp}"
    steps = build_steps(
        args=args,
        benchmark_output_dir=benchmark_output_dir,
        experiment_root=experiment_root,
    )

    step_results = []
    for step in steps:
        step_results.append(
            run_logged_step(
                name=str(step["name"]),
                command=[str(part) for part in step["command"]],
                log_dir=run_root,
            )
        )

    benchmark_notes = []
    if args.preserve_download_state:
        benchmark_notes.append(
            "This rerun preserved the completed download state and restarted from the "
            "post-download stages only."
        )
    else:
        benchmark_notes.append(
            "This rerun cleaned the download stage as well and restarted from the full pipeline."
        )
    if not args.include_plot:
        benchmark_notes.append(
            "The bulk plotting stage was skipped because it is non-essential for the "
            "benchmark path and would create a very large PNG artifact set."
        )
    else:
        benchmark_notes.append(
            "The feature plotting stage was included in this rerun."
        )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "python": str(args.python),
        "run_root": str(run_root),
        "rerun_mode": (
            "post_download_fresh_start"
            if args.preserve_download_state
            else "full_fresh_start"
        ),
        "scope_summary": (
            "This report summarizes a post-download fresh rerun that verified the "
            "existing download outputs, preserved the completed download inputs, "
            "cleaned only derived artifacts, reran preprocessing and handcrafted "
            "feature extraction, rebuilt baseline and benchmark clustering outputs, "
            "and generated fresh benchmark artifacts."
            if args.preserve_download_state
            else
            "This report summarizes a full fresh rerun that removed generated "
            "artifacts including download-stage outputs, reran the pipeline from "
            "download onward, rebuilt the taxonomy-aware merged dataset, executed "
            "the clustering profile suite, and ran the thesis clustering benchmark."
        ),
        "download_inputs_preserved": bool(args.preserve_download_state),
        "included_plot_stage": bool(args.include_plot),
        "download_verification_path": str(download_verification_path),
        "download_verification": download_verification,
        "benchmark_rerun_notes": benchmark_notes,
        "cleanup_targets": cleanup_results,
        "steps": step_results,
        "benchmark_output_dir": str(benchmark_output_dir),
        "experiment_run_root": str(experiment_root),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = run_root / "rerun_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFull rerun complete.")
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "summary_path": str(summary_path),
                "download_verification_path": str(download_verification_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
