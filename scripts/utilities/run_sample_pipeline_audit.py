from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}_sample_audit"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
COPY_IGNORE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "audio_files",
    "output",
    "node_modules",
}
SNAPSHOT_IGNORE_NAMES = {
    ".git",
    ".venv",
    ".venv-encodecmae",
    ".venv-musicnn",
    "__pycache__",
    "node_modules",
}


@dataclass
class StepSpec:
    name: str
    command: List[str]
    required_csvs: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a clean two-pass sample pipeline audit from download through "
            "benchmark in an isolated workspace, with CSV inventories and validation."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--keep-workspace", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def normalize_rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def should_ignore_directory(relative_dir: str, names: Sequence[str]) -> List[str]:
    ignored = [name for name in names if name in COPY_IGNORE_NAMES]
    if relative_dir.startswith("docs/reports"):
        if "run_logs" in names:
            ignored.append("run_logs")
    if relative_dir == "data":
        for name in (
            "songs.csv",
            "songs_with_merged_genres.csv",
            "songs_schema_summary.json",
            "_tmp_rebuilt_songs.csv",
            "songs_genre_list.csv",
            "unique_genres.csv",
        ):
            if name in names:
                ignored.append(name)
    return sorted(set(ignored))


def copy_project_template(source_root: Path, workspace_root: Path) -> None:
    if workspace_root.exists():
        shutil.rmtree(workspace_root)

    def ignore(dir_path: str, names: Sequence[str]) -> List[str]:
        relative = Path(dir_path).resolve().relative_to(source_root.resolve())
        relative_text = str(relative).replace("\\", "/")
        if relative_text == ".":
            relative_text = ""
        return should_ignore_directory(relative_text, names)

    shutil.copytree(source_root, workspace_root, ignore=ignore)


def write_sample_catalog(source_catalog: Path, target_catalog: Path, sample_size: int) -> Dict[str, int]:
    target_catalog.parent.mkdir(parents=True, exist_ok=True)
    with source_catalog.open("r", encoding="utf-8", newline="") as src_handle:
        reader = csv.DictReader(src_handle)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= sample_size:
                break

    with target_catalog.open("w", encoding="utf-8", newline="") as dst_handle:
        writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {"requested_rows": int(sample_size), "written_rows": int(len(rows))}


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prepare_workspace_data(workspace_root: Path, sample_size: int) -> Dict[str, object]:
    source_catalog = PROJECT_ROOT / "data" / "millionsong_dataset.csv"
    workspace_catalog = workspace_root / "data" / "millionsong_dataset.csv"
    sample_summary = write_sample_catalog(source_catalog, workspace_catalog, sample_size)

    for relative in (
        Path("audio_files"),
        Path("output"),
        Path("docs") / "reports" / "run_logs",
        Path("src") / "data_collection" / "download_stats",
        Path("data") / "songs.csv",
        Path("data") / "songs_with_merged_genres.csv",
        Path("data") / "songs_schema_summary.json",
        Path("data") / "_tmp_rebuilt_songs.csv",
        Path("download_checkpoint_with_genre.json"),
        Path("deezer_search_cache.json"),
        Path("data") / "songs_data_with_genre.csv",
        Path("data") / "msd_matches.csv",
        Path("data") / "audio_msd_mapping.csv",
    ):
        remove_path(workspace_root / relative)

    # Keep only the MSD numeric metadata in backup_old_csvs so the sample run
    # does not inherit historical download/match CSVs from the main workspace.
    for relative in (
        Path("data") / "backup_old_csvs" / "songs_data_with_genre.csv",
        Path("data") / "backup_old_csvs" / "audio_msd_mapping.csv",
        Path("data") / "backup_old_csvs" / "msd_matches.csv",
        Path("data") / "backup_old_csvs" / "msd_unmatched.csv",
    ):
        remove_path(workspace_root / relative)

    (workspace_root / "audio_files").mkdir(parents=True, exist_ok=True)
    (workspace_root / "output").mkdir(parents=True, exist_ok=True)
    (workspace_root / "docs" / "reports" / "run_logs").mkdir(parents=True, exist_ok=True)
    return sample_summary


def snapshot_csvs(root: Path) -> Dict[str, Dict[str, int]]:
    snapshot: Dict[str, Dict[str, int]] = {}
    for path in root.rglob("*.csv"):
        if any(part in SNAPSHOT_IGNORE_NAMES for part in path.parts):
            continue
        stat = path.stat()
        snapshot[normalize_rel(path, root)] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return snapshot


def summarize_csv(path: Path) -> Dict[str, object]:
    summary = {
        "path": str(path),
        "exists": bool(path.exists()),
        "size": int(path.stat().st_size) if path.exists() else 0,
        "rows": 0,
        "columns": 0,
        "header": [],
        "error": "",
    }
    if not path.exists():
        return summary

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            row_count = sum(1 for _ in reader)
        summary["rows"] = int(row_count)
        summary["columns"] = int(len(header))
        summary["header"] = header[:30]
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def resolve_patterns(root: Path, patterns: Sequence[str]) -> Dict[str, List[str]]:
    resolved: Dict[str, List[str]] = {}
    for pattern in patterns:
        matches = sorted(
            normalize_rel(path, root)
            for path in root.glob(pattern)
            if path.is_file()
        )
        resolved[pattern] = matches
    return resolved


def run_command(command: Sequence[str], cwd: Path, log_path: Path) -> Dict[str, object]:
    started = datetime.now().isoformat(timespec="seconds")
    started_perf = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + subprocess.list2cmdline([str(part) for part in command]) + "\n\n")
        handle.flush()
        result = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    return {
        "started_at": started,
        "duration_sec": round(time.perf_counter() - started_perf, 2),
        "returncode": int(result.returncode),
        "log_path": str(log_path),
    }


def build_step_specs(python_executable: str, pass_label: str) -> List[StepSpec]:
    benchmark_output_dir = Path("output") / "metrics" / f"thesis_benchmark_sample_{pass_label}"
    return [
        StepSpec(
            name="01_rebuild_unified_pre_download",
            command=[python_executable, "scripts/utilities/migrate_to_unified_csv.py"],
            required_csvs=[
                "data/millionsong_dataset.csv",
                "data/backup_old_csvs/msd_features.csv",
            ],
        ),
        StepSpec(
            name="02_resilient_download",
            command=[
                python_executable,
                "scripts/utilities/run_resilient_download.py",
                "--chunk-size",
                "500",
                "--idle-timeout-sec",
                "240",
                "--max-stall-restarts",
                "20",
                "--max-no-progress-sessions",
                "2",
                "--log-dir",
                "docs/reports/run_logs/latest_resilient_download",
            ],
            required_csvs=[
                "data/millionsong_dataset.csv",
                "data/songs.csv",
            ],
        ),
        StepSpec(
            name="03_audio_preprocessing",
            command=[
                python_executable,
                "scripts/run_audio_preprocessing.py",
                "--audio-dir",
                "audio_files",
            ],
            required_csvs=[
                "data/songs.csv",
                "data/songs_data_with_genre.csv",
            ],
        ),
        StepSpec(
            name="04_rebuild_unified_post_preprocess",
            command=[python_executable, "scripts/utilities/migrate_to_unified_csv.py"],
            required_csvs=[
                "data/millionsong_dataset.csv",
                "data/songs_data_with_genre.csv",
                "data/backup_old_csvs/msd_features.csv",
            ],
        ),
        StepSpec(
            name="05_feature_extraction",
            command=[python_executable, "src/features/extract_features.py"],
            required_csvs=["data/songs.csv"],
        ),
        StepSpec(
            name="06_feature_qc",
            command=[python_executable, "scripts/analysis/run_feature_qc.py"],
            required_csvs=["data/songs.csv"],
        ),
        StepSpec(
            name="07_metadata_audit",
            command=[
                python_executable,
                "scripts/analysis/audit_metadata_schema.py",
                "--output-dir",
                "output/metrics",
            ],
            required_csvs=["data/songs.csv"],
        ),
        StepSpec(
            name="08_build_merged_genre_dataset",
            command=[python_executable, "scripts/utilities/build_merged_genre_dataset.py"],
            required_csvs=[
                "data/songs.csv",
                "data/acoustically_coherent_merged_genres_corrected.csv",
            ],
        ),
        StepSpec(
            name="09_baseline_gmm",
            command=[python_executable, "src/clustering/gmm.py", "--no-ui"],
            required_csvs=["data/songs_with_merged_genres.csv"],
        ),
        StepSpec(
            name="10_run_all_clustering_profiles",
            command=[
                python_executable,
                "scripts/run_all_clustering.py",
                "--profiles",
                "recommended_production",
                "all_audio_pca_comparison",
                "all_audio_zscore_comparison",
                "--publish-ui-snapshot",
                "--run-root",
                "output/experiment_runs_taxonomy",
            ],
            required_csvs=["data/songs_with_merged_genres.csv"],
        ),
        StepSpec(
            name="11_run_thesis_benchmark",
            command=[
                python_executable,
                "scripts/analysis/thesis_clustering_benchmark.py",
                "--output-dir",
                str(benchmark_output_dir),
            ],
            required_csvs=["data/songs_with_merged_genres.csv"],
        ),
    ]


def run_step_with_csv_audit(
    workspace_root: Path,
    reports_root: Path,
    step: StepSpec,
) -> Dict[str, object]:
    before_snapshot = snapshot_csvs(workspace_root)
    required_before = resolve_patterns(workspace_root, step.required_csvs)
    missing_before = {
        pattern: matches
        for pattern, matches in required_before.items()
        if not matches
    }

    log_path = reports_root / "logs" / f"{step.name}.log"
    command_result = run_command(step.command, workspace_root, log_path)
    if int(command_result["returncode"]) != 0:
        raise RuntimeError(
            f"Step {step.name} failed with exit code {command_result['returncode']}. "
            f"See {log_path}"
        )

    after_snapshot = snapshot_csvs(workspace_root)
    new_csvs = sorted(set(after_snapshot) - set(before_snapshot))
    modified_csvs = sorted(
        path
        for path in set(after_snapshot).intersection(before_snapshot)
        if after_snapshot[path] != before_snapshot[path]
    )
    deleted_csvs = sorted(set(before_snapshot) - set(after_snapshot))

    summarized_paths = sorted(set(new_csvs + modified_csvs))
    csv_summaries = {
        rel_path: summarize_csv(workspace_root / rel_path)
        for rel_path in summarized_paths
    }

    return {
        "name": step.name,
        "command": list(step.command),
        "required_csvs": required_before,
        "missing_required_csvs": missing_before,
        "run": command_result,
        "new_csvs": new_csvs,
        "modified_csvs": modified_csvs,
        "deleted_csvs": deleted_csvs,
        "csv_summaries": csv_summaries,
    }


def _bool_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def validate_pass(workspace_root: Path, sample_size: int, pass_label: str) -> Dict[str, object]:
    data_dir = workspace_root / "data"
    output_dir = workspace_root / "output"
    benchmark_dir = output_dir / "metrics" / f"thesis_benchmark_sample_{pass_label}"
    sample_catalog = pd.read_csv(data_dir / "millionsong_dataset.csv")
    songs = pd.read_csv(data_dir / "songs.csv")
    merged = pd.read_csv(data_dir / "songs_with_merged_genres.csv")
    sample_track_ids = set(sample_catalog["track_id"].fillna("").astype(str).str.strip()) - {""}
    songs_track_ids = set(songs["msd_track_id"].fillna("").astype(str).str.strip()) - {""}
    songs_data = (
        pd.read_csv(data_dir / "songs_data_with_genre.csv")
        if (data_dir / "songs_data_with_genre.csv").exists()
        else pd.DataFrame(columns=["filename", "genre"])
    )

    if not songs_data.empty:
        songs_data["audio_basename"] = songs_data["filename"].fillna("").astype(str).map(
            lambda value: Path(value).stem
        )
        songs_data["genre"] = songs_data["genre"].fillna("").astype(str).str.strip()
        songs_data_map = (
            songs_data.groupby("audio_basename")["genre"]
            .apply(lambda series: next((value for value in series if value), ""))
            .to_dict()
        )
    else:
        songs_data_map = {}

    audio_bases = {
        path.stem
        for path in (workspace_root / "audio_files").iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    }
    songs_audio_mask = _bool_mask(songs["has_audio"]) if "has_audio" in songs.columns else songs["filename"].astype(str).str.strip().ne("")
    songs_audio_bases = set(
        songs.loc[songs_audio_mask, "audio_basename"].fillna("").astype(str).str.strip()
    )

    merged_audio_mask = _bool_mask(merged["has_audio"])
    merged_include_mask = _bool_mask(merged["include_in_mrs"])
    merged_audio = merged[merged_audio_mask].copy()
    merged_excluded = merged_audio[~merged_include_mask.loc[merged_audio.index]].copy()
    leaked_legacy_mask = merged_excluded["original_genre"].fillna("").astype(str).str.strip().eq("")
    leaked_legacy_mask = leaked_legacy_mask & merged_excluded["audio_basename"].astype(str).map(
        lambda value: bool(str(songs_data_map.get(value, "")).strip())
    ).astype(bool)
    leaked_legacy_genres = merged_excluded[leaked_legacy_mask].copy()
    extra_song_rows = max(int(len(songs) - len(sample_catalog)), 0)
    missing_sample_track_ids = sorted(sample_track_ids - songs_track_ids)
    benchmark_expected = [
        "aligned_metadata.csv",
        "feature_group_variance_summary.csv",
        "feature_group_correlation_summary.csv",
        "representation_catalog.csv",
        "full_grid_results.csv",
        "native_best_results.csv",
        "matched_granularity_results.csv",
        "global_native_leaders.csv",
        "global_matched_leaders.csv",
    ]
    benchmark_files = {
        name: summarize_csv(benchmark_dir / name)
        for name in benchmark_expected
    }

    checks = [
        {
            "name": "sample_catalog_size",
            "passed": int(len(sample_catalog)) == int(sample_size),
            "details": {"expected": int(sample_size), "actual": int(len(sample_catalog))},
        },
        {
            "name": "sample_catalog_track_ids_present_in_songs",
            "passed": not missing_sample_track_ids,
            "details": {
                "sample_track_ids": int(len(sample_track_ids)),
                "songs_track_ids": int(len(songs_track_ids)),
                "missing_track_ids": missing_sample_track_ids[:10],
            },
        },
        {
            "name": "songs_audio_basenames_align_with_audio_dir",
            "passed": songs_audio_bases == audio_bases,
            "details": {
                "songs_audio_basenames": int(len(songs_audio_bases)),
                "audio_dir_basenames": int(len(audio_bases)),
                "orphaned_audio_examples": sorted(audio_bases - songs_audio_bases)[:10],
                "missing_audio_examples": sorted(songs_audio_bases - audio_bases)[:10],
            },
        },
        {
            "name": "merged_rows_match_songs_rows",
            "passed": int(len(merged)) == int(len(songs)),
            "details": {
                "merged_rows": int(len(merged)),
                "songs_rows": int(len(songs)),
                "extra_song_rows_vs_catalog": extra_song_rows,
                "rows_without_msd_track_id": int(
                    songs["msd_track_id"].fillna("").astype(str).str.strip().eq("").sum()
                ),
            },
        },
        {
            "name": "taxonomy_legacy_genre_loss_fixed",
            "passed": leaked_legacy_genres.empty,
            "details": {
                "leaked_rows": int(len(leaked_legacy_genres)),
                "examples": leaked_legacy_genres["audio_basename"].astype(str).head(10).tolist(),
            },
        },
        {
            "name": "benchmark_csvs_exist_and_are_readable",
            "passed": all(summary["exists"] and not summary["error"] for summary in benchmark_files.values()),
            "details": benchmark_files,
        },
    ]

    taxonomy_summary = {
        "audio_rows": int(len(merged_audio)),
        "audio_included_in_mrs": int((merged_audio_mask & merged_include_mask).sum()),
        "audio_excluded_from_mrs": int(len(merged_excluded)),
        "excluded_blank_original_genre": int(
            merged_excluded["original_genre"].fillna("").astype(str).str.strip().eq("").sum()
        ),
        "excluded_nonblank_original_genre": int(
            merged_excluded["original_genre"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        "excluded_by_metadata_origin": {
            str(key): int(value)
            for key, value in merged_excluded["metadata_origin"].fillna("").value_counts().items()
        },
    }

    return {
        "sample_size": int(sample_size),
        "checks": checks,
        "taxonomy_summary": taxonomy_summary,
    }


def cleanup_junk_csvs(project_root: Path, reports_root: Path, label: str) -> Dict[str, object]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "utilities" / "cleanup_project_junk.py"),
        "--project-root",
        str(project_root),
        "--apply",
        "--output-json",
        str(reports_root / f"{label}_junk_cleanup.json"),
    ]
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "junk cleanup failed")
    return json.loads(result.stdout)


def write_markdown_report(
    path: Path,
    *,
    workspace_root: Path,
    sample_summary: Dict[str, int],
    pass_reports: List[Dict[str, object]],
    source_cleanup: Dict[str, object],
) -> None:
    lines: List[str] = [
        "# Sample Pipeline Audit",
        "",
        f"- Workspace: `{workspace_root}`",
        f"- Sample rows requested: `{sample_summary['requested_rows']}`",
        f"- Sample rows written: `{sample_summary['written_rows']}`",
        "",
    ]

    for report in pass_reports:
        lines.extend(
            [
                f"## {report['pass_label']}",
                "",
                f"- Reports directory: `{report['reports_root']}`",
                f"- Junk CSVs removed in workspace after pass: `{report['workspace_cleanup']['removed_count']}`",
                "",
                "### Validation",
                "",
            ]
        )
        for check in report["validation"]["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {status}: `{check['name']}`")
        lines.extend(
            [
                "",
                "### Taxonomy",
                "",
            ]
        )
        for key, value in report["validation"]["taxonomy_summary"].items():
            lines.append(f"- {key}: `{value}`")
        lines.extend(["", "### Steps", ""])
        for step in report["steps"]:
            lines.append(
                f"- `{step['name']}`: new CSVs `{len(step['new_csvs'])}`, modified CSVs `{len(step['modified_csvs'])}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Source Cleanup",
            "",
            f"- Junk CSV candidates removed in source project: `{source_cleanup['removed_count']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_parent = resolve_path(args.workspace_root)
    run_root = workspace_parent / f"sample_pipeline_audit_{run_stamp}"
    workspace_root = run_root / "workspace"
    reports_root = run_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    copy_project_template(PROJECT_ROOT, workspace_root)
    sample_summary = prepare_workspace_data(workspace_root, args.sample_size)

    pass_reports: List[Dict[str, object]] = []
    for pass_index in range(1, args.passes + 1):
        pass_label = f"pass_{pass_index}"
        pass_reports_root = reports_root / pass_label
        pass_reports_root.mkdir(parents=True, exist_ok=True)

        if pass_index > 1:
            prepare_workspace_data(workspace_root, args.sample_size)

        steps = build_step_specs(args.python, pass_label)
        step_reports = [
            run_step_with_csv_audit(workspace_root, pass_reports_root, step)
            for step in steps
        ]
        validation = validate_pass(workspace_root, args.sample_size, pass_label)
        workspace_cleanup = cleanup_junk_csvs(workspace_root, pass_reports_root, pass_label)

        pass_payload = {
            "pass_label": pass_label,
            "workspace_root": str(workspace_root),
            "reports_root": str(pass_reports_root),
            "steps": step_reports,
            "validation": validation,
            "workspace_cleanup": workspace_cleanup,
        }
        (pass_reports_root / "pass_summary.json").write_text(
            json.dumps(pass_payload, indent=2),
            encoding="utf-8",
        )
        pass_reports.append(pass_payload)

    source_cleanup = cleanup_junk_csvs(PROJECT_ROOT, reports_root, "source_project")
    summary_payload = {
        "project_root": str(PROJECT_ROOT),
        "workspace_root": str(workspace_root),
        "sample_summary": sample_summary,
        "passes": pass_reports,
        "source_cleanup": source_cleanup,
    }
    (reports_root / "final_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        reports_root / "final_summary.md",
        workspace_root=workspace_root,
        sample_summary=sample_summary,
        pass_reports=pass_reports,
        source_cleanup=source_cleanup,
    )

    print(json.dumps(summary_payload, indent=2))

    print(f"Workspace available at: {workspace_root}")


if __name__ == "__main__":
    main()
