from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_qc import FEATURE_KEYS, collect_feature_bundle_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a detailed Markdown report for a full pipeline rerun by "
            "reading the produced logs, manifests, and benchmark artifacts."
        )
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="",
        help=(
            "Optional rerun log directory. Defaults to the latest "
            "full_pipeline_rerun_* or post_download_pipeline_* directory."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional Markdown output path. Defaults under docs/reports/ with the rerun slug.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final output summary as JSON instead of a human-readable message.",
    )
    return parser.parse_args()


def resolve_run_root(explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    run_logs_root = PROJECT_ROOT / "docs" / "reports" / "run_logs"
    candidates = sorted(
        [
            *run_logs_root.glob("full_pipeline_rerun_*"),
            *run_logs_root.glob("post_download_pipeline_*"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No full_pipeline_rerun_* or post_download_pipeline_* directory was found "
            "under docs/reports/run_logs."
        )
    return candidates[0]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def coerce_path(path_text: Any) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def rel(path: Path | str | None) -> str:
    if not path:
        return ""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(resolved).replace("\\", "/")


def escape_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", "<br>")
    return text.replace("|", "\\|")


def fmt_float(value: Any, digits: int = 4, default: str = "n/a") -> str:
    try:
        number = float(value)
    except Exception:
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return f"{number:.{digits}f}"


def fmt_int(value: Any, default: str = "n/a") -> str:
    try:
        return f"{int(value)}"
    except Exception:
        return default


def fmt_pct(value: Any, digits: int = 2, default: str = "n/a") -> str:
    try:
        number = float(value)
    except Exception:
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return f"{number * 100:.{digits}f}%"


def markdown_table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return ["_No rows available._"]
    header = "| " + " | ".join(escape_cell(col) for col in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(escape_cell(cell) for cell in row) + " |"
        for row in rows
    ]
    return [header, divider, *body]


def step_rows_from_summary(summary: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for item in summary.get("steps", []):
        command = " ".join(str(part) for part in item.get("command", []))
        rows.append(
            [
                str(item.get("name", "")),
                fmt_float(item.get("duration_sec"), digits=2),
                fmt_int(item.get("returncode")),
                command,
                rel(item.get("log_path")),
            ]
        )
    return rows


def fallback_step_rows(run_root: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    log_paths = sorted(run_root.glob("*.log")) + sorted(run_root.glob("*.err.log"))
    for log_path in log_paths:
        rows.append([log_path.stem, "n/a", "n/a", "", rel(log_path)])
    return rows


def cleanup_rows(summary: Optional[Dict[str, Any]]) -> List[List[str]]:
    if not summary:
        return []
    rows: List[List[str]] = []
    for item in summary.get("cleanup_targets", []):
        rows.append(
            [
                rel(item.get("path")),
                str(item.get("type", "")),
                "yes" if bool(item.get("existed")) else "no",
            ]
        )
    return rows


def resolve_download_summary_path(rerun_summary: Optional[Dict[str, Any]]) -> Path:
    verification = (rerun_summary or {}).get("download_verification") or {}
    candidates = [
        coerce_path(verification.get("latest_download_summary_path")),
        PROJECT_ROOT
        / "docs"
        / "reports"
        / "run_logs"
        / "latest_resilient_download"
        / "run_summary.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return candidates[-1]


def collect_download_summary(rerun_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    run_summary_path = resolve_download_summary_path(rerun_summary)
    payload = read_json(run_summary_path) or {}
    payload["path"] = run_summary_path
    payload["reference_only"] = bool((rerun_summary or {}).get("download_inputs_preserved"))
    sessions = payload.get("sessions", [])
    if sessions:
        last_session = sessions[-1]
        payload.setdefault("processed_rows", last_session.get("processed_after"))
        payload.setdefault("downloaded_rows", last_session.get("downloaded_after"))
        payload.setdefault("audio_files", last_session.get("audio_after"))
        if payload.get("catalog_rows") is not None and payload.get("processed_rows") is not None:
            payload.setdefault(
                "remaining_rows",
                max(0, int(payload["catalog_rows"]) - int(payload["processed_rows"])),
            )
    return payload


def download_session_rows(payload: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for session in payload.get("sessions", []):
        rows.append(
            [
                fmt_int(session.get("session_index")),
                fmt_float(session.get("duration_sec"), digits=2),
                fmt_int(session.get("limit")),
                fmt_int(session.get("processed_gain")),
                fmt_int(session.get("downloaded_gain")),
                fmt_int(session.get("audio_gain")),
                "yes" if bool(session.get("stalled")) else "no",
                fmt_int(session.get("exit_code")),
                rel(session.get("log_path")),
            ]
        )
    return rows


def collect_metadata_summary() -> Dict[str, Any]:
    songs_csv = PROJECT_ROOT / "data" / "songs.csv"
    merged_csv = PROJECT_ROOT / "data" / "songs_with_merged_genres.csv"
    schema_json = PROJECT_ROOT / "data" / "songs_schema_summary.json"
    audio_dir = PROJECT_ROOT / "audio_files"

    payload: Dict[str, Any] = {
        "songs_csv": songs_csv,
        "merged_csv": merged_csv,
        "schema_json": schema_json,
        "schema_summary": read_json(schema_json) or {},
        "audio_dir": audio_dir,
        "audio_files_on_disk": int(
            sum(
                1
                for path in audio_dir.glob("*")
                if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
            )
        )
        if audio_dir.exists()
        else 0,
    }

    songs_df = read_csv(songs_csv)
    if songs_df is not None:
        payload["songs_rows"] = int(len(songs_df))
        payload["songs_audio_rows"] = (
            int(bool_series(songs_df["has_audio"]).sum())
            if "has_audio" in songs_df.columns
            else 0
        )
        payload["songs_unique_primary_genres"] = (
            int(
                songs_df["genre"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )
            if "genre" in songs_df.columns
            else 0
        )

    merged_df = read_csv(merged_csv)
    if merged_df is not None:
        payload["merged_rows"] = int(len(merged_df))
        payload["merged_audio_rows"] = (
            int(bool_series(merged_df["has_audio"]).sum())
            if "has_audio" in merged_df.columns
            else 0
        )
        payload["merged_included_rows"] = (
            int(bool_series(merged_df["include_in_mrs"]).sum())
            if "include_in_mrs" in merged_df.columns
            else 0
        )
        payload["merged_unique_primary_genres"] = (
            int(
                merged_df["primary_genre"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )
            if "primary_genre" in merged_df.columns
            else 0
        )
        payload["merged_unique_primary_tag_sets"] = (
            int(
                merged_df["primary_tags"]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )
            if "primary_tags" in merged_df.columns
            else 0
        )
        if "secondary_tags" in merged_df.columns:
            secondary_tags = {
                tag.strip()
                for value in merged_df["secondary_tags"].fillna("").astype(str)
                for tag in value.split(",")
                if tag.strip()
            }
            payload["merged_unique_secondary_tags"] = int(len(secondary_tags))
    return payload


def collect_feature_summary() -> Dict[str, Any]:
    results_dir = PROJECT_ROOT / "output" / "features"
    manifest_path = results_dir / "feature_extraction_manifest.json"
    manifest = read_json(manifest_path) or {}
    payload: Dict[str, Any] = {
        "results_dir": results_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }

    if results_dir.exists():
        inventory = collect_feature_bundle_inventory(str(results_dir))
        complete_bundles = sum(
            1 for keys in inventory.values() if len(keys) == len(FEATURE_KEYS)
        )
        incomplete_bundles = sum(
            1 for keys in inventory.values() if len(keys) != len(FEATURE_KEYS)
        )
        npy_files = sum(1 for _ in results_dir.glob("*.npy"))
        payload.update(
            {
                "bundle_count": int(len(inventory)),
                "complete_bundles": int(complete_bundles),
                "incomplete_bundles": int(incomplete_bundles),
                "npy_files": int(npy_files),
            }
        )

    failure_log = results_dir / "feature_extraction_failures.txt"
    if failure_log.exists():
        try:
            payload["failure_count"] = int(
                sum(
                    1
                    for line in failure_log.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    if line.strip()
                )
            )
        except Exception:
            payload["failure_count"] = "n/a"
        payload["failure_log"] = failure_log

    return payload


def find_latest_metrics_file(pattern: str) -> Optional[Path]:
    metrics_root = PROJECT_ROOT / "output" / "metrics"
    candidates = sorted(
        metrics_root.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def collect_metadata_audit_summary() -> Dict[str, Any]:
    summary_path = find_latest_metrics_file("metadata_schema_audit_*_summary.json")
    payload: Dict[str, Any] = {"summary_path": summary_path}
    if summary_path is None:
        return payload

    summary = read_json(summary_path) or {}
    stamp = summary_path.stem.replace("metadata_schema_audit_", "").replace("_summary", "")
    payload["summary"] = summary
    payload["markdown_path"] = summary_path.parent / f"metadata_schema_audit_{stamp}.md"
    payload["primary_distribution_path"] = (
        summary_path.parent / f"metadata_primary_genre_distribution_{stamp}.csv"
    )
    payload["label_distribution_path"] = (
        summary_path.parent / f"metadata_label_genre_distribution_{stamp}.csv"
    )
    return payload


def collect_feature_qc_summary() -> Dict[str, Any]:
    summary_path = find_latest_metrics_file("feature_qc_*_summary.json")
    payload: Dict[str, Any] = {"summary_path": summary_path}
    if summary_path is None:
        return payload

    summary = read_json(summary_path) or {}
    stamp = summary_path.stem.replace("feature_qc_", "").replace("_summary", "")
    payload["summary"] = summary
    payload["markdown_path"] = summary_path.parent / f"feature_qc_{stamp}_summary.md"
    payload["track_status_path"] = summary_path.parent / f"feature_qc_{stamp}_track_status.csv"
    payload["variance_path"] = summary_path.parent / f"feature_qc_{stamp}_group_variance.csv"
    return payload


def find_single_experiment_manifest() -> Optional[Path]:
    experiment_root = PROJECT_ROOT / "output" / "experiment_runs_taxonomy"
    manifests = sorted(
        experiment_root.glob("run_*/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return manifests[0] if manifests else None


def collect_experiment_summary() -> Dict[str, Any]:
    manifest_path = find_single_experiment_manifest()
    payload: Dict[str, Any] = {"run_manifest_path": manifest_path}
    if manifest_path is None:
        return payload

    manifest = read_json(manifest_path) or {}
    payload["run_manifest"] = manifest
    profile_rows: List[Dict[str, Any]] = []

    for profile_id, profile_manifest in manifest.get("profiles", {}).items():
        evaluation = profile_manifest.get("evaluation") or {}
        summary_path = (
            Path(evaluation["summary_path"]) if evaluation.get("summary_path") else None
        )
        comparison_path = (
            Path(evaluation["comparison_csv"]) if evaluation.get("comparison_csv") else None
        )
        summary_payload = read_json(summary_path) if summary_path else None
        comparison_df = read_csv(comparison_path) if comparison_path else None
        top_rows = []
        if comparison_df is not None and not comparison_df.empty:
            top_df = comparison_df.sort_values("OverallRank", kind="stable").head(3)
            for row in top_df.to_dict(orient="records"):
                top_rows.append(
                    {
                        "rank": fmt_int(row.get("OverallRank")),
                        "method": row.get("Method", row.get("MethodId", "")),
                        "primary_tag_jaccard": fmt_float(row.get("PrimaryTagJaccard@K")),
                        "all_tag_jaccard": fmt_float(row.get("AllTagJaccard@K")),
                        "catalog_coverage": fmt_float(row.get("CatalogCoverage")),
                        "subsample_median_ari": fmt_float(row.get("SubsampleMedianARI")),
                        "cluster_count": fmt_int(row.get("ClusterCount")),
                    }
                )

        profile_rows.append(
            {
                "profile_id": profile_id,
                "title": (profile_manifest.get("profile") or {}).get("title", profile_id),
                "artifact_dir": profile_manifest.get("artifact_dir"),
                "metrics_dir": profile_manifest.get("metrics_dir"),
                "methods": list((profile_manifest.get("methods") or {}).keys()),
                "representation_contract": (
                    (summary_payload or {}).get("representation_contract", {})
                ),
                "decision_assessment": (
                    (summary_payload or {}).get("decision_assessment", {})
                ),
                "top_rows": top_rows,
                "comparison_csv": comparison_path,
                "comparison_report": (
                    Path(evaluation["comparison_report"])
                    if evaluation.get("comparison_report")
                    else None
                ),
                "summary_path": summary_path,
            }
        )

    payload["profiles"] = profile_rows
    return payload


def find_single_benchmark_dir() -> Optional[Path]:
    metrics_root = PROJECT_ROOT / "output" / "metrics"
    candidates = sorted(
        metrics_root.glob("thesis_benchmark_full_rerun_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_benchmark_dir(rerun_summary: Optional[Dict[str, Any]]) -> Optional[Path]:
    explicit_dir = coerce_path((rerun_summary or {}).get("benchmark_output_dir"))
    if explicit_dir is not None and explicit_dir.exists():
        return explicit_dir
    return find_single_benchmark_dir()


def collect_benchmark_summary(rerun_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    benchmark_dir = resolve_benchmark_dir(rerun_summary)
    payload: Dict[str, Any] = {"benchmark_dir": benchmark_dir}
    if benchmark_dir is None:
        return payload

    payload["dataset_summary"] = read_json(benchmark_dir / "dataset_summary.json") or {}
    payload["representation_catalog"] = read_csv(
        benchmark_dir / "representation_catalog.csv"
    )
    payload["global_native"] = read_csv(benchmark_dir / "global_native_leaders.csv")
    payload["global_matched"] = read_csv(benchmark_dir / "global_matched_leaders.csv")
    payload["report_path"] = benchmark_dir / "benchmark_report.md"

    native_best = read_csv(benchmark_dir / "native_best_results.csv")
    if native_best is not None and not native_best.empty:
        payload["native_best_top"] = native_best.sort_values(
            ["nmi", "silhouette", "stability_ari"],
            ascending=[False, False, False],
            kind="stable",
        ).head(10)

    matched_best = read_csv(benchmark_dir / "matched_granularity_results.csv")
    if matched_best is not None and not matched_best.empty:
        payload["matched_best_top"] = matched_best.sort_values(
            ["matched_target_clusters", "cluster_gap", "nmi", "stability_ari"],
            ascending=[True, True, False, False],
            kind="stable",
        ).head(12)
    return payload


def build_report(
    run_root: Path,
    rerun_summary: Optional[Dict[str, Any]],
    download_summary: Dict[str, Any],
    metadata_summary: Dict[str, Any],
    feature_summary: Dict[str, Any],
    metadata_audit_summary: Dict[str, Any],
    feature_qc_summary: Dict[str, Any],
    experiment_summary: Dict[str, Any],
    benchmark_summary: Dict[str, Any],
) -> str:
    generated_at = datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []

    rerun_mode = str((rerun_summary or {}).get("rerun_mode", "")).strip().lower()
    download_inputs_preserved = bool((rerun_summary or {}).get("download_inputs_preserved"))
    if rerun_mode in {"post_download_fresh_start", "post_download_pipeline"} or run_root.name.startswith(
        "post_download_pipeline_"
    ):
        title = "# Post-Download Pipeline Rerun Report"
    else:
        title = "# Full Pipeline Fresh Rerun Report"

    lines.append(title)
    lines.append("")
    lines.append(f"- Generated at: `{generated_at}`")
    lines.append(f"- Project root: `{rel(PROJECT_ROOT)}`")
    lines.append(f"- Rerun log root: `{rel(run_root)}`")
    lines.append(f"- Rerun summary present: `{'yes' if rerun_summary else 'no'}`")
    lines.append("")

    lines.append("## Scope")
    lines.append("")
    scope_summary = (rerun_summary or {}).get("scope_summary")
    if scope_summary:
        lines.append(str(scope_summary))
    else:
        lines.append(
            "This report summarizes a fresh end-to-end execution that removes generated artifacts, "
            "rebuilds metadata, reruns `run_pipeline.py` from the download stage, rebuilds the "
            "taxonomy-aware merged-genre dataset, executes the clustering profile suite, and then "
            "runs the thesis clustering benchmark."
        )
    if rerun_summary and rerun_summary.get("rerun_mode"):
        lines.append("")
        lines.append(f"- Rerun mode: `{rerun_summary.get('rerun_mode')}`")
        lines.append(
            f"- Download inputs preserved: `{'yes' if rerun_summary.get('download_inputs_preserved') else 'no'}`"
        )
    lines.append("")

    lines.append("## Step Execution")
    lines.append("")
    rerun_notes = (rerun_summary or {}).get("benchmark_rerun_notes", [])
    if rerun_notes:
        for note in rerun_notes:
            lines.append(f"- Note: `{note}`")
        lines.append("")
    step_rows = step_rows_from_summary(rerun_summary) if rerun_summary else fallback_step_rows(run_root)
    lines.extend(
        markdown_table(
            ["Step", "DurationSec", "ReturnCode", "Command", "Log"],
            step_rows,
        )
    )
    lines.append("")

    cleanup = cleanup_rows(rerun_summary)
    if cleanup:
        lines.append("## Cleanup Targets")
        lines.append("")
        lines.extend(
            markdown_table(
                ["Path", "Type", "ExistedBeforeCleanup"],
                cleanup,
            )
        )
        lines.append("")

    lines.append(
        "## Preserved Download Snapshot"
        if download_inputs_preserved
        else "## Download Stage"
    )
    lines.append("")
    if download_summary.get("catalog_rows") is not None:
        if download_inputs_preserved:
            lines.append(
                "This rerun did not execute a downloader step. The values below are a "
                "reference snapshot from the resilient-download summary recorded when "
                "the preserved inputs were verified."
            )
            lines.append("")
        catalog_rows = download_summary.get("catalog_rows")
        processed_rows = download_summary.get("processed_rows")
        downloaded_rows = download_summary.get("downloaded_rows")
        audio_files = download_summary.get("audio_files")
        remaining_rows = download_summary.get("remaining_rows")
        lines.append(f"- Catalog rows: `{fmt_int(catalog_rows)}`")
        lines.append(f"- Processed rows: `{fmt_int(processed_rows)}`")
        lines.append(f"- Downloaded rows recorded in checkpoint: `{fmt_int(downloaded_rows)}`")
        lines.append(f"- Audio files on disk after download stage: `{fmt_int(audio_files)}`")
        lines.append(f"- Remaining rows after the resilient download loop: `{fmt_int(remaining_rows)}`")
        lines.append(f"- Forced stall restarts: `{fmt_int(download_summary.get('forced_restart_count'))}`")
        if catalog_rows is not None and audio_files is not None:
            lines.append(
                f"- Catalog-to-audio yield: `{fmt_pct((float(audio_files) / float(catalog_rows)) if float(catalog_rows) else 0.0)}`"
            )
        if processed_rows is not None and audio_files is not None:
            lines.append(
                f"- Processed-to-audio yield: `{fmt_pct((float(audio_files) / float(processed_rows)) if float(processed_rows) else 0.0)}`"
            )
        summary_label = (
            "Referenced download summary artifact"
            if download_inputs_preserved
            else "Download summary artifact"
        )
        lines.append(f"- {summary_label}: `{rel(download_summary.get('path'))}`")
        lines.append("")

        verification = (rerun_summary or {}).get("download_verification") or {}
        if verification:
            lines.append("### Download Verification Snapshot")
            lines.append("")
            lines.append(f"- Verification artifact: `{rel((rerun_summary or {}).get('download_verification_path'))}`")
            lines.append(f"- `songs.csv` rows: `{fmt_int(verification.get('songs_rows'))}`")
            lines.append(f"- `songs.csv` audio-backed rows: `{fmt_int(verification.get('songs_audio_rows'))}`")
            lines.append(f"- Audio files on disk at verification time: `{fmt_int(verification.get('audio_files_on_disk'))}`")
            lines.append(f"- Zero-byte audio files: `{fmt_int(verification.get('zero_byte_audio_files'))}`")
            lines.append(f"- Orphaned audio basenames vs metadata: `{fmt_int(verification.get('orphaned_audio_basenames'))}`")
            lines.append(f"- Metadata basenames missing on disk: `{fmt_int(verification.get('missing_audio_basenames'))}`")
            lines.append(f"- Checkpoint processed rows: `{fmt_int(verification.get('checkpoint_processed_rows'))}`")
            lines.append(f"- Checkpoint downloaded rows: `{fmt_int(verification.get('checkpoint_downloaded_rows'))}`")
            lines.append(f"- Checkpoint failed rows: `{fmt_int(verification.get('checkpoint_failed_rows'))}`")
            lines.append(f"- No-progress downloader sessions: `{fmt_int(verification.get('latest_download_no_progress_sessions'))}`")
            lines.append(
                f"- No-progress sessions whose logs still printed success: `{fmt_int(verification.get('no_progress_sessions_with_success_message'))}`"
            )
            extension_counts = verification.get("audio_extension_counts") or {}
            if extension_counts:
                lines.append(f"- Audio extension counts: `{extension_counts}`")
            verification_notes = verification.get("notes") or []
            if verification_notes:
                lines.append("")
                for note in verification_notes:
                    lines.append(f"- {note}")
            lines.append("")

        session_rows = download_session_rows(download_summary)
        if session_rows:
            lines.append("### Download Sessions")
            lines.append("")
            lines.extend(
                markdown_table(
                    [
                        "Session",
                        "DurationSec",
                        "Limit",
                        "ProcessedGain",
                        "DownloadedGain",
                        "AudioGain",
                        "Stalled",
                        "ExitCode",
                        "Log",
                    ],
                    session_rows,
                )
            )
            lines.append("")
    else:
        lines.append("_Download summary JSON is not available yet._")
        lines.append("")

    lines.append("## Metadata Outputs")
    lines.append("")
    lines.append(f"- Unified songs CSV: `{rel(metadata_summary.get('songs_csv'))}`")
    lines.append(f"- Schema summary JSON: `{rel(metadata_summary.get('schema_json'))}`")
    lines.append(f"- Taxonomy-aware merged CSV: `{rel(metadata_summary.get('merged_csv'))}`")
    lines.append(f"- Audio files currently on disk: `{fmt_int(metadata_summary.get('audio_files_on_disk'))}`")
    lines.append(f"- Unified rows: `{fmt_int(metadata_summary.get('songs_rows'))}`")
    lines.append(f"- Unified audio-backed rows: `{fmt_int(metadata_summary.get('songs_audio_rows'))}`")
    lines.append(f"- Unified unique raw genre values: `{fmt_int(metadata_summary.get('songs_unique_primary_genres'))}`")
    lines.append(f"- Merged rows: `{fmt_int(metadata_summary.get('merged_rows'))}`")
    lines.append(f"- Merged audio-backed rows: `{fmt_int(metadata_summary.get('merged_audio_rows'))}`")
    lines.append(f"- Merged rows included in MRS: `{fmt_int(metadata_summary.get('merged_included_rows'))}`")
    lines.append(f"- Merged unique primary genres: `{fmt_int(metadata_summary.get('merged_unique_primary_genres'))}`")
    lines.append(f"- Merged unique primary tag sets: `{fmt_int(metadata_summary.get('merged_unique_primary_tag_sets'))}`")
    lines.append(f"- Merged unique secondary tags: `{fmt_int(metadata_summary.get('merged_unique_secondary_tags'))}`")
    lines.append("")

    schema_summary = metadata_summary.get("schema_summary") or {}
    if schema_summary:
        lines.append("### Schema Summary Snapshot")
        lines.append("")
        schema_rows = [[key, value] for key, value in sorted(schema_summary.items())]
        lines.extend(markdown_table(["Metric", "Value"], schema_rows))
        lines.append("")

    metadata_audit = metadata_audit_summary.get("summary") or {}
    if metadata_audit:
        lines.append("## Metadata Audit")
        lines.append("")
        lines.append(f"- Summary JSON: `{rel(metadata_audit_summary.get('summary_path'))}`")
        lines.append(f"- Markdown audit: `{rel(metadata_audit_summary.get('markdown_path'))}`")
        lines.append(
            f"- Primary genre distribution CSV: `{rel(metadata_audit_summary.get('primary_distribution_path'))}`"
        )
        lines.append(
            f"- Exploded label distribution CSV: `{rel(metadata_audit_summary.get('label_distribution_path'))}`"
        )
        lines.append(f"- Unified rows audited: `{fmt_int(metadata_audit.get('unified_rows'))}`")
        lines.append(f"- Audio-backed rows audited: `{fmt_int(metadata_audit.get('audio_backed_rows'))}`")
        lines.append(
            f"- Audio rows with MSD track id: `{fmt_int(metadata_audit.get('audio_rows_with_msd_track_id'))}`"
        )
        lines.append(
            f"- Audio rows with numeric MSD features: `{fmt_int(metadata_audit.get('audio_rows_with_numeric_msd_features'))}`"
        )
        lines.append(
            f"- Current audio basenames: `{fmt_int(metadata_audit.get('current_audio_basenames'))}`"
        )
        lines.append(
            f"- Raw audio files on disk: `{fmt_int(metadata_audit.get('current_audio_files'))}`"
        )
        lines.append(
            f"- Audio coverage gap: `{fmt_int(metadata_audit.get('audio_basename_gap'))}`"
        )
        lines.append(
            f"- Unique primary genres in audio subset: `{fmt_int(metadata_audit.get('audio_primary_genre_unique_count'))}`"
        )
        lines.append(f"- Genre Gini: `{fmt_float(metadata_audit.get('audio_primary_genre_gini'), digits=6)}`")
        lines.append(f"- Genre HHI: `{fmt_float(metadata_audit.get('audio_primary_genre_hhi'), digits=6)}`")
        lines.append(
            f"- Effective genre count: `{fmt_float(metadata_audit.get('audio_primary_genre_effective_count'), digits=2)}`"
        )
        lines.append(
            f"- Top-10 primary-genre share: `{fmt_float(metadata_audit.get('audio_primary_genre_top_10_share'), digits=6)}`"
        )
        lines.append(
            f"- Singleton primary-genre fraction: `{fmt_float(metadata_audit.get('audio_primary_genre_singleton_fraction'), digits=6)}`"
        )
        lines.append(
            f"- <=5-track primary-genre fraction: `{fmt_float(metadata_audit.get('audio_primary_genre_tail_fraction_le_5'), digits=6)}`"
        )
        lines.append(f"- Audit runtime seconds: `{fmt_float(metadata_audit.get('runtime_seconds'), digits=2)}`")
        lines.append("")

    lines.append("## Feature Extraction")
    lines.append("")
    manifest = feature_summary.get("manifest") or {}
    extraction_run_summary = manifest.get("run_summary") or {}
    audio_contract = manifest.get("audio_contract") or {}
    feature_settings = manifest.get("feature_settings") or {}
    library_versions = manifest.get("library_versions") or {}
    lines.append(f"- Feature output directory: `{rel(feature_summary.get('results_dir'))}`")
    lines.append(f"- Feature manifest: `{rel(feature_summary.get('manifest_path'))}`")
    lines.append(f"- Audio files seen by extraction: `{fmt_int(extraction_run_summary.get('total_audio_files'))}`")
    lines.append(f"- Newly processed files: `{fmt_int(extraction_run_summary.get('processed'))}`")
    lines.append(f"- Skipped existing bundles: `{fmt_int(extraction_run_summary.get('skipped'))}`")
    lines.append(f"- Extraction failures reported by manifest: `{fmt_int(extraction_run_summary.get('failed'))}`")
    lines.append(f"- Detected feature bundles in output folder: `{fmt_int(feature_summary.get('bundle_count'))}`")
    lines.append(f"- Complete feature bundles: `{fmt_int(feature_summary.get('complete_bundles'))}`")
    lines.append(f"- Incomplete feature bundles: `{fmt_int(feature_summary.get('incomplete_bundles'))}`")
    lines.append(f"- `.npy` files present: `{fmt_int(feature_summary.get('npy_files'))}`")
    if feature_summary.get("failure_log"):
        lines.append(f"- Failure log: `{rel(feature_summary.get('failure_log'))}`")
        lines.append(f"- Failure log lines: `{fmt_int(feature_summary.get('failure_count'))}`")
    lines.append("")

    if audio_contract or feature_settings or library_versions:
        lines.append("### Extraction Contract")
        lines.append("")
        contract_rows: List[List[str]] = []
        for key, value in sorted(audio_contract.items()):
            contract_rows.append([f"audio_contract.{key}", value])
        for key, value in sorted(feature_settings.items()):
            contract_rows.append([f"feature_settings.{key}", value])
        for key, value in sorted(library_versions.items()):
            contract_rows.append([f"library_versions.{key}", value])
        lines.extend(markdown_table(["Field", "Value"], contract_rows))
        lines.append("")

    feature_qc = feature_qc_summary.get("summary") or {}
    if feature_qc:
        lines.append("## Feature QC")
        lines.append("")
        lines.append(f"- QC summary JSON: `{rel(feature_qc_summary.get('summary_path'))}`")
        lines.append(f"- QC markdown: `{rel(feature_qc_summary.get('markdown_path'))}`")
        lines.append(f"- Track-status CSV: `{rel(feature_qc_summary.get('track_status_path'))}`")
        lines.append(f"- Group-variance CSV: `{rel(feature_qc_summary.get('variance_path'))}`")
        lines.append(f"- Audio tracks scanned: `{fmt_int(feature_qc.get('audio_tracks_scanned'))}`")
        lines.append(f"- Complete bundles before repair: `{fmt_int(feature_qc.get('complete_before_repair'))}`")
        lines.append(f"- Incomplete bundles before repair: `{fmt_int(feature_qc.get('incomplete_before_repair'))}`")
        lines.append(f"- Invalid bundles before repair: `{fmt_int(feature_qc.get('invalid_before_repair'))}`")
        lines.append(f"- Re-extraction attempted: `{fmt_int(feature_qc.get('reextract_attempted'))}`")
        lines.append(f"- Re-extraction succeeded: `{fmt_int(feature_qc.get('reextract_succeeded'))}`")
        lines.append(f"- Re-extraction failed: `{fmt_int(feature_qc.get('reextract_failed'))}`")
        lines.append(f"- Stale feature bundles: `{fmt_int(feature_qc.get('stale_feature_bundles'))}`")
        lines.append(
            f"- Stale genre-cache entries removed: `{fmt_int(feature_qc.get('stale_genre_cache_entries_removed'))}`"
        )
        lines.append(f"- Tracks loaded into clustering dataset: `{fmt_int(feature_qc.get('clustering_loaded_tracks'))}`")
        lines.append(f"- Clustering QC CSV: `{rel(feature_qc.get('clustering_qc_csv_path'))}`")
        lines.append(f"- Clustering QC JSON: `{rel(feature_qc.get('clustering_qc_json_path'))}`")
        lines.append(f"- QC runtime seconds: `{fmt_float(feature_qc.get('runtime_seconds'), digits=2)}`")
        lines.append("")

    lines.append("## Clustering Profile Suite")
    lines.append("")
    run_manifest = experiment_summary.get("run_manifest") or {}
    if run_manifest:
        lines.append(f"- Experiment run manifest: `{rel(experiment_summary.get('run_manifest_path'))}`")
        lines.append(f"- Requested profiles: `{run_manifest.get('requested_profiles')}`")
        lines.append(f"- Methods override: `{run_manifest.get('methods_override')}`")
        lines.append(f"- Include VaDE: `{run_manifest.get('include_vade')}`")
        lines.append(f"- Skip evaluation: `{run_manifest.get('skip_evaluation')}`")
        lines.append("")

        for profile in experiment_summary.get("profiles", []):
            lines.append(f"### Profile `{profile['profile_id']}`")
            lines.append("")
            lines.append(f"- Title: `{profile['title']}`")
            lines.append(f"- Artifact dir: `{rel(profile.get('artifact_dir'))}`")
            lines.append(f"- Metrics dir: `{rel(profile.get('metrics_dir'))}`")
            lines.append(f"- Methods run: `{profile.get('methods')}`")
            lines.append(f"- Evaluation summary JSON: `{rel(profile.get('summary_path'))}`")
            lines.append(f"- Comparison CSV: `{rel(profile.get('comparison_csv'))}`")
            lines.append(f"- Comparison report: `{rel(profile.get('comparison_report'))}`")
            lines.append("")

            representation_contract = profile.get("representation_contract") or {}
            if representation_contract:
                lines.append("Representation contract:")
                lines.append("")
                rep_rows = [[key, value] for key, value in sorted(representation_contract.items())]
                lines.extend(markdown_table(["Field", "Value"], rep_rows))
                lines.append("")

            top_rows = profile.get("top_rows") or []
            if top_rows:
                lines.append("Top-ranked methods:")
                lines.append("")
                lines.extend(
                    markdown_table(
                        [
                            "Rank",
                            "Method",
                            "PrimaryTagJaccard@10",
                            "AllTagJaccard@10",
                            "CatalogCoverage",
                            "SubsampleMedianARI",
                            "ClusterCount",
                        ],
                        [
                            [
                                row["rank"],
                                row["method"],
                                row["primary_tag_jaccard"],
                                row["all_tag_jaccard"],
                                row["catalog_coverage"],
                                row["subsample_median_ari"],
                                row["cluster_count"],
                            ]
                            for row in top_rows
                        ],
                    )
                )
                lines.append("")

            assessment = profile.get("decision_assessment") or {}
            if assessment:
                lines.append("Decision-policy assessment:")
                lines.append("")
                for key, value in assessment.items():
                    lines.append(f"- `{key}`: `{value}`")
                lines.append("")
    else:
        lines.append("_Experiment suite manifest is not available yet._")
        lines.append("")

    lines.append("## Thesis Benchmark")
    lines.append("")
    dataset_summary = benchmark_summary.get("dataset_summary") or {}
    if dataset_summary:
        lines.append(f"- Benchmark directory: `{rel(benchmark_summary.get('benchmark_dir'))}`")
        lines.append(f"- Benchmark report: `{rel(benchmark_summary.get('report_path'))}`")
        lines.append(f"- Songs evaluated: `{fmt_int(dataset_summary.get('n_songs'))}`")
        lines.append(f"- Unique genres: `{fmt_int(dataset_summary.get('n_genres'))}`")
        lines.append(f"- Unique artists: `{fmt_int(dataset_summary.get('n_artists'))}`")
        lines.append(f"- Raw feature dimensions: `{fmt_int(dataset_summary.get('n_raw_dims'))}`")
        lines.append(f"- Representation families: `{fmt_int(dataset_summary.get('n_representations'))}`")
        lines.append(f"- Preprocess modes: `{dataset_summary.get('preprocess_modes')}`")
        lines.append(f"- Matched cluster targets: `{dataset_summary.get('matched_targets')}`")
        lines.append(f"- Raw cache path: `{rel(dataset_summary.get('raw_cache'))}`")
        lines.append("")

        representation_catalog = benchmark_summary.get("representation_catalog")
        if isinstance(representation_catalog, pd.DataFrame) and not representation_catalog.empty:
            lines.append("### Representation Catalog")
            lines.append("")
            catalog_df = representation_catalog[["combo", "groups", "n_groups", "n_dims_raw", "rationale"]]
            lines.extend(
                markdown_table(
                    ["Combo", "Groups", "NGroups", "RawDims", "Rationale"],
                    catalog_df.astype(object).values.tolist(),
                )
            )
            lines.append("")

        global_native = benchmark_summary.get("global_native")
        if isinstance(global_native, pd.DataFrame) and not global_native.empty:
            lines.append("### Global Native Leaders")
            lines.append("")
            selected = [
                "method",
                "combo",
                "preprocess_mode",
                "n_clusters",
                "silhouette",
                "nmi",
                "stability_ari",
                "fit_time_sec",
            ]
            available = [column for column in selected if column in global_native.columns]
            lines.extend(
                markdown_table(
                    available,
                    global_native[available].astype(object).values.tolist(),
                )
            )
            lines.append("")

        global_matched = benchmark_summary.get("global_matched")
        if isinstance(global_matched, pd.DataFrame) and not global_matched.empty:
            lines.append("### Global Matched-Granularity Leaders")
            lines.append("")
            selected = [
                "matched_target_clusters",
                "method",
                "combo",
                "preprocess_mode",
                "n_clusters",
                "cluster_gap",
                "silhouette",
                "nmi",
                "stability_ari",
            ]
            available = [column for column in selected if column in global_matched.columns]
            lines.extend(
                markdown_table(
                    available,
                    global_matched[available].astype(object).values.tolist(),
                )
            )
            lines.append("")

        native_best_top = benchmark_summary.get("native_best_top")
        if isinstance(native_best_top, pd.DataFrame) and not native_best_top.empty:
            lines.append("### Top Native Operating Points Across All Representations")
            lines.append("")
            selected = [
                "method",
                "combo",
                "preprocess_mode",
                "n_clusters",
                "silhouette",
                "nmi",
                "stability_ari",
                "internal_selection_score",
            ]
            available = [column for column in selected if column in native_best_top.columns]
            lines.extend(
                markdown_table(
                    available,
                    native_best_top[available].astype(object).values.tolist(),
                )
            )
            lines.append("")

        matched_best_top = benchmark_summary.get("matched_best_top")
        if isinstance(matched_best_top, pd.DataFrame) and not matched_best_top.empty:
            lines.append("### Top Matched-Granularity Operating Points")
            lines.append("")
            selected = [
                "matched_target_clusters",
                "method",
                "combo",
                "preprocess_mode",
                "n_clusters",
                "cluster_gap",
                "nmi",
                "stability_ari",
            ]
            available = [column for column in selected if column in matched_best_top.columns]
            lines.extend(
                markdown_table(
                    available,
                    matched_best_top[available].astype(object).values.tolist(),
                )
            )
            lines.append("")
    else:
        lines.append("_Benchmark dataset summary is not available yet._")
        lines.append("")

    lines.append("## Key Artifact Paths")
    lines.append("")
    artifact_rows = [
        ["Rerun log root", rel(run_root)],
        ["Rerun summary JSON", rel(run_root / "rerun_summary.json")],
        ["Preprocessing details JSON", rel((rerun_summary or {}).get("preprocessing_details_path"))],
        [
            "Referenced resilient download summary"
            if download_inputs_preserved
            else "Latest resilient download summary",
            rel(download_summary.get("path")),
        ],
        ["Unified songs CSV", rel(metadata_summary.get("songs_csv"))],
        ["Schema summary JSON", rel(metadata_summary.get("schema_json"))],
        ["Metadata audit summary JSON", rel(metadata_audit_summary.get("summary_path"))],
        ["Metadata audit markdown", rel(metadata_audit_summary.get("markdown_path"))],
        ["Merged taxonomy CSV", rel(metadata_summary.get("merged_csv"))],
        ["Feature manifest JSON", rel(feature_summary.get("manifest_path"))],
        ["Feature QC summary JSON", rel(feature_qc_summary.get("summary_path"))],
        ["Feature QC markdown", rel(feature_qc_summary.get("markdown_path"))],
        ["Experiment run manifest", rel(experiment_summary.get("run_manifest_path"))],
        ["Benchmark directory", rel(benchmark_summary.get("benchmark_dir"))],
        ["Benchmark report", rel(benchmark_summary.get("report_path"))],
    ]
    lines.extend(markdown_table(["Artifact", "Path"], artifact_rows))
    lines.append("")

    lines.append("## Residual Notes")
    lines.append("")
    if download_summary.get("remaining_rows") is not None:
        remaining_rows = int(download_summary.get("remaining_rows", 0))
        if remaining_rows > 0:
            if download_inputs_preserved:
                lines.append(
                    f"- The referenced resilient download summary still reports `{remaining_rows}` remaining catalog rows. "
                    "This post-download rerun preserved those inputs rather than rerunning acquisition."
                )
            else:
                lines.append(
                    f"- The resilient downloader still reports `{remaining_rows}` remaining catalog rows after its stop condition. "
                    "That means the library is a fresh rerun, but not a 100% successful audio acquisition across the full catalog."
                )
        else:
            lines.append("- The resilient downloader reports no remaining catalog rows.")
    audio_files_on_disk = metadata_summary.get("audio_files_on_disk")
    songs_audio_rows = metadata_summary.get("songs_audio_rows")
    if audio_files_on_disk is not None and songs_audio_rows is not None:
        try:
            audio_gap = int(songs_audio_rows) - int(audio_files_on_disk)
        except Exception:
            audio_gap = 0
        if audio_gap > 0:
            lines.append(
                f"- Metadata/audio mismatch after the rerun: songs.csv marks `{fmt_int(songs_audio_rows)}` rows as audio-backed, "
                f"but only `{fmt_int(audio_files_on_disk)}` audio files are currently on disk (gap `{audio_gap}`)."
            )
        elif audio_gap < 0:
            lines.append(
                f"- Metadata/audio mismatch after the rerun: `{fmt_int(audio_files_on_disk)}` audio files are on disk, "
                f"which exceeds the `{fmt_int(songs_audio_rows)}` audio-backed rows in songs.csv by `{fmt_int(abs(audio_gap))}`. "
                "This points to orphaned or otherwise untracked audio files relative to the current metadata."
            )
    bundle_count = feature_summary.get("bundle_count")
    if audio_files_on_disk is not None and bundle_count is not None:
        try:
            eligibility_gap = int(audio_files_on_disk) - int(bundle_count)
        except Exception:
            eligibility_gap = 0
        if eligibility_gap > 0:
            lines.append(
                f"- The benchmarkable library is smaller than the downloaded library: `{fmt_int(audio_files_on_disk)}` audio files on disk "
                f"versus `{fmt_int(bundle_count)}` complete feature bundles (gap `{eligibility_gap}`)."
            )
        elif eligibility_gap < 0:
            lines.append(
                f"- Feature bundle inventory exceeds the current audio-file count: `{fmt_int(bundle_count)}` bundles versus "
                f"`{fmt_int(audio_files_on_disk)}` audio files (gap `{fmt_int(abs(eligibility_gap))}`)."
            )
    schema_summary = metadata_summary.get("schema_summary") or {}
    if schema_summary and songs_audio_rows not in (None, "n/a"):
        try:
            schema_audio_rows = int(schema_summary.get("current_audio_rows", 0))
            songs_audio_rows_int = int(songs_audio_rows)
        except Exception:
            schema_audio_rows = 0
            songs_audio_rows_int = 0
        if songs_audio_rows_int > 0 and schema_audio_rows == 0:
            lines.append(
                "- `songs_schema_summary.json` still reports zero current audio rows, so its audio counters appear stale relative to the rebuilt `songs.csv`."
            )
    if feature_summary.get("incomplete_bundles") not in (None, 0):
        lines.append(
            f"- Incomplete feature bundles detected: `{fmt_int(feature_summary.get('incomplete_bundles'))}`."
        )
    if feature_summary.get("failure_count") not in (None, 0, "0", "n/a"):
        lines.append(
            f"- Feature extraction failure log contains `{fmt_int(feature_summary.get('failure_count'))}` lines."
        )
    if not rerun_summary:
        lines.append(
            "- `rerun_summary.json` is missing, which typically means the full rerun has not finished yet or stopped before the wrapper wrote its final summary."
        )
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    run_root = resolve_run_root(args.run_root)
    rerun_summary = read_json(run_root / "rerun_summary.json")
    download_summary = collect_download_summary(rerun_summary)
    metadata_summary = collect_metadata_summary()
    feature_summary = collect_feature_summary()
    metadata_audit_summary = collect_metadata_audit_summary()
    feature_qc_summary = collect_feature_qc_summary()
    experiment_summary = collect_experiment_summary()
    benchmark_summary = collect_benchmark_summary(rerun_summary)

    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_name = f"{run_root.name}_detailed_report.md"
        output_path = PROJECT_ROOT / "docs" / "reports" / output_name
    ensure_parent(output_path)

    report = build_report(
        run_root=run_root,
        rerun_summary=rerun_summary,
        download_summary=download_summary,
        metadata_summary=metadata_summary,
        feature_summary=feature_summary,
        metadata_audit_summary=metadata_audit_summary,
        feature_qc_summary=feature_qc_summary,
        experiment_summary=experiment_summary,
        benchmark_summary=benchmark_summary,
    )
    output_path.write_text(report, encoding="utf-8")

    payload = {"run_root": str(run_root), "report_path": str(output_path)}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Detailed rerun report written to {output_path}")


if __name__ == "__main__":
    main()
