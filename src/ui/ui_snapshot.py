import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_UI_SNAPSHOT_ROOT = PROJECT_ROOT / "output" / "ui" / "latest_benchmark_snapshot"

METHOD_DISPLAY_NAMES = {
    "kmeans": "KMeans",
    "gmm": "GMM",
    "hdbscan": "HDBSCAN",
    "vade": "VaDE",
}

BENCHMARK_DASHBOARD_FILENAMES = (
    "dataset_summary.json",
    "representation_catalog.csv",
    "full_grid_results.csv",
    "native_best_results.csv",
    "matched_granularity_results.csv",
    "global_native_leaders.csv",
    "global_matched_leaders.csv",
    "benchmark_report.md",
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _snapshot_manifest_path(snapshot_root: Path) -> Path:
    return snapshot_root / "ui_bundle_manifest.json"


def _load_snapshot_manifest(snapshot_root: Path) -> Dict[str, Any]:
    manifest_path = _snapshot_manifest_path(snapshot_root)
    if manifest_path.exists():
        return _read_json(manifest_path)
    return {
        "snapshot_root": str(snapshot_root.resolve()),
        "updated_at_utc": None,
        "profile_snapshot": {},
        "benchmark_snapshot": {},
    }


def _copy_if_exists(source: Optional[Path], destination: Path) -> Optional[Path]:
    if source is None or not source.exists() or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.resolve()


def _coerce_existing_file(value: object) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    if path.exists() and path.is_file():
        return path
    return None


def publish_profile_ui_snapshot(
    profile_root: Path,
    profile_manifest: Dict[str, Any],
    snapshot_root: Path = DEFAULT_UI_SNAPSHOT_ROOT,
) -> Dict[str, Any]:
    snapshot_root = snapshot_root.resolve()
    manifest = _load_snapshot_manifest(snapshot_root)
    profile_payload = profile_manifest.get("profile") or {}
    profile_id = str(profile_payload.get("profile_id") or "unknown")

    metrics_dir = snapshot_root / "metrics"
    methods_dir = snapshot_root / "methods"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    methods_dir.mkdir(parents=True, exist_ok=True)

    evaluation_payload = profile_manifest.get("evaluation") or {}
    summary_src = _coerce_existing_file(evaluation_payload.get("summary_path"))
    comparison_csv_src = _coerce_existing_file(evaluation_payload.get("comparison_csv"))
    comparison_report_src = _coerce_existing_file(
        evaluation_payload.get("comparison_report")
    )

    summary_dst = _copy_if_exists(summary_src, metrics_dir / (summary_src.name if summary_src else ""))
    comparison_csv_dst = _copy_if_exists(
        comparison_csv_src,
        metrics_dir / (comparison_csv_src.name if comparison_csv_src else ""),
    )
    comparison_report_dst = _copy_if_exists(
        comparison_report_src,
        metrics_dir / (comparison_report_src.name if comparison_report_src else ""),
    )

    method_entries: Dict[str, Any] = {}
    for method_id, method_row in (profile_manifest.get("methods") or {}).items():
        method_id = str(method_id).strip().lower()
        method_dir = methods_dir / method_id
        method_dir.mkdir(parents=True, exist_ok=True)

        method_summary_src = _coerce_existing_file(method_row.get("method_summary_path"))
        method_summary_payload = method_row.get("method_summary") or (
            _read_json(method_summary_src) if method_summary_src is not None else {}
        )
        outputs = method_summary_payload.get("outputs") or {}

        results_csv_src = _coerce_existing_file(outputs.get("results_csv"))
        artifact_src = _coerce_existing_file(outputs.get("retrieval_artifact"))
        summary_copy = _copy_if_exists(
            method_summary_src,
            method_dir / (method_summary_src.name if method_summary_src else ""),
        )
        results_copy = _copy_if_exists(
            results_csv_src,
            method_dir / (results_csv_src.name if results_csv_src else ""),
        )
        artifact_copy = _copy_if_exists(
            artifact_src,
            method_dir / (artifact_src.name if artifact_src else ""),
        )

        method_entries[method_id] = {
            "display_name": METHOD_DISPLAY_NAMES.get(method_id, method_id.upper()),
            "results_csv": str(results_copy) if results_copy else None,
            "artifact_path": str(artifact_copy) if artifact_copy else None,
            "summary_path": str(summary_copy) if summary_copy else None,
            "source_results_csv": str(results_csv_src) if results_csv_src else None,
            "source_artifact_path": str(artifact_src) if artifact_src else None,
            "source_summary_path": (
                str(method_summary_src) if method_summary_src else None
            ),
        }

    manifest["profile_snapshot"] = {
        "profile_id": profile_id,
        "profile_root": str(profile_root.resolve()),
        "metrics_dir": str(metrics_dir.resolve()),
        "methods_dir": str(methods_dir.resolve()),
        "summary_path": str(summary_dst) if summary_dst else None,
        "comparison_csv": str(comparison_csv_dst) if comparison_csv_dst else None,
        "comparison_report": (
            str(comparison_report_dst) if comparison_report_dst else None
        ),
        "available_methods": sorted(method_entries.keys()),
        "methods": method_entries,
    }
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(_snapshot_manifest_path(snapshot_root), manifest)
    return manifest


def publish_benchmark_dashboard_snapshot(
    benchmark_dir: Path,
    snapshot_root: Path = DEFAULT_UI_SNAPSHOT_ROOT,
) -> Dict[str, Any]:
    snapshot_root = snapshot_root.resolve()
    manifest = _load_snapshot_manifest(snapshot_root)
    benchmark_dir = benchmark_dir.resolve()
    local_benchmark_dir = snapshot_root / "benchmark"
    local_benchmark_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: Dict[str, Optional[str]] = {}
    for filename in BENCHMARK_DASHBOARD_FILENAMES:
        source = benchmark_dir / filename
        copied = _copy_if_exists(source, local_benchmark_dir / filename)
        copied_paths[filename] = str(copied) if copied else None

    manifest["benchmark_snapshot"] = {
        "benchmark_name": benchmark_dir.name,
        "source_benchmark_dir": str(benchmark_dir),
        "benchmark_dir": str(local_benchmark_dir.resolve()),
        **copied_paths,
    }
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(_snapshot_manifest_path(snapshot_root), manifest)
    return manifest


def load_ui_snapshot(
    snapshot_root: Path = DEFAULT_UI_SNAPSHOT_ROOT,
) -> Dict[str, Any]:
    snapshot_root = snapshot_root.resolve()
    manifest_path = _snapshot_manifest_path(snapshot_root)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"UI snapshot manifest not found: {manifest_path}. "
            "Run the benchmark pipeline first."
        )
    return _read_json(manifest_path)
