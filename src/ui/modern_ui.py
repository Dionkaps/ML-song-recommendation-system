import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from config import feature_vars as fv
from src.ui.ui_snapshot import DEFAULT_UI_SNAPSHOT_ROOT, load_ui_snapshot

tk = None
tkfont = None
messagebox = None
ttk = None
pygame = None


BENCHMARK_DIR_PATTERNS = (
    "thesis_benchmark_full_rerun_*",
    "thesis_clustering_benchmark_*",
)

DEFAULT_BENCHMARK_PROFILE_ID = "recommended_production"
UI_METHOD_CHOICES = ("auto", "kmeans", "gmm", "hdbscan", "vade")

METHOD_DISPLAY_NAMES = {
    "kmeans": "KMeans",
    "gmm": "GMM",
    "hdbscan": "HDBSCAN",
    "vade": "VaDE",
}

METHOD_COLORS = {
    "kmeans": "#0f766e",
    "gmm": "#ea580c",
    "hdbscan": "#be123c",
    "vade": "#7c3aed",
}

PREPROCESS_DISPLAY_NAMES = {
    "raw_zscore": "Raw + z-score",
    "pca_per_group_2": "PCA / group = 2",
    "pca_per_group_5": "PCA / group = 5",
}

METRIC_DISPLAY_NAMES = {
    "nmi": "NMI",
    "silhouette": "Silhouette",
    "stability_ari": "Stability ARI",
    "coverage": "Coverage",
    "noise_fraction": "Noise fraction",
    "fit_time_sec": "Fit time",
    "internal_selection_score": "Internal score",
    "n_clusters": "Cluster count",
    "cluster_gap": "Target gap",
    "cluster_balance": "Cluster balance",
    "avg_confidence": "Avg confidence",
}

LOWER_IS_BETTER_METRICS = {
    "fit_time_sec",
    "noise_fraction",
    "davies_bouldin",
    "bic",
    "aic",
    "cluster_gap",
}


def _ensure_ui_dependencies() -> None:
    """Import Tkinter and pygame only when the interactive UI is requested."""

    global tk, tkfont, messagebox, ttk, pygame

    if tk is None or tkfont is None or messagebox is None or ttk is None:
        try:
            import tkinter as _tk
            from tkinter import font as _tkfont
            from tkinter import messagebox as _messagebox, ttk as _ttk
        except ImportError as exc:
            raise RuntimeError(
                "Tkinter isn't available on your Python installation."
            ) from exc

        tk = _tk
        tkfont = _tkfont
        messagebox = _messagebox
        ttk = _ttk

    if pygame is None:
        try:
            import pygame as _pygame
        except ImportError as exc:
            raise RuntimeError(
                "Pygame isn't available. Install with 'pip install pygame'"
            ) from exc

        pygame = _pygame


def _ensure_pygame_mixer() -> None:
    """Initialize pygame audio only when the interactive UI is actually launched."""

    _ensure_ui_dependencies()

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize pygame mixer: {exc}") from exc


def _parse_bounded_int(value: str, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer control value and clamp it to a safe range."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_bounded_float(
    value: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Parse a float control value and clamp it to a safe range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _align_artifact_rows(
    df: pd.DataFrame,
    labels: np.ndarray,
    payload: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Align a saved retrieval artifact to the current DataFrame row order."""

    artifact_songs = [str(song) for song in payload["songs"].tolist()]
    song_to_idx = {song: idx for idx, song in enumerate(artifact_songs)}
    missing = [song for song in df["Song"] if song not in song_to_idx]
    if missing:
        raise ValueError(
            "Retrieval artifact is missing songs required by the UI. "
            f"First missing song: {missing[0]}"
        )

    row_positions = np.array([song_to_idx[song] for song in df["Song"]], dtype=np.int32)
    aligned: Dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if values.ndim >= 1 and values.shape[0] == len(artifact_songs):
            aligned[key] = values[row_positions]
        else:
            aligned[key] = values

    artifact_labels = aligned.get("labels")
    if artifact_labels is not None and not np.array_equal(artifact_labels, labels):
        raise ValueError(
            "Saved retrieval artifact labels do not match the currently loaded "
            "clustering results. Re-run the clustering method to refresh artifacts."
        )

    return aligned


def _resolve_retrieval_payload(
    df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    retrieval_features: Optional[np.ndarray],
    retrieval_method_id: Optional[str],
    assignment_confidence: Optional[np.ndarray],
    posterior_probabilities: Optional[np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:
    """Resolve the prepared-space retrieval payload used by the UI."""

    if retrieval_features is None:
        if retrieval_method_id is None:
            raise ValueError(
                "launch_ui requires retrieval_features or retrieval_method_id so "
                "recommendations can run in the prepared feature space."
            )
        from src.clustering.kmeans import load_retrieval_artifact

        artifact = load_retrieval_artifact(retrieval_method_id)
        aligned = _align_artifact_rows(df, labels, artifact)
        retrieval_features = np.asarray(aligned["prepared_features"], dtype=np.float32)
        if assignment_confidence is None and "assignment_confidence" in aligned:
            assignment_confidence = np.asarray(
                aligned["assignment_confidence"], dtype=np.float32
            )
        if posterior_probabilities is None and "posterior_probabilities" in aligned:
            posterior_probabilities = np.asarray(
                aligned["posterior_probabilities"], dtype=np.float32
            )

    retrieval_features = np.asarray(retrieval_features, dtype=np.float32)
    if retrieval_features.ndim != 2 or retrieval_features.shape[0] != len(df):
        raise ValueError(
            "retrieval_features must have one row per song. "
            f"Got shape {retrieval_features.shape} for {len(df)} songs."
        )

    if assignment_confidence is not None:
        assignment_confidence = np.asarray(assignment_confidence, dtype=np.float32)
        if assignment_confidence.shape[0] != len(df):
            raise ValueError(
                "assignment_confidence must have one value per song. "
                f"Got shape {assignment_confidence.shape} for {len(df)} songs."
            )

    if posterior_probabilities is not None:
        posterior_probabilities = np.asarray(posterior_probabilities, dtype=np.float32)
        if (
            posterior_probabilities.ndim != 2
            or posterior_probabilities.shape[0] != len(df)
        ):
            raise ValueError(
                "posterior_probabilities must have one row per song. "
                f"Got shape {posterior_probabilities.shape} for {len(df)} songs."
            )

    return {
        "retrieval_features": retrieval_features,
        "assignment_confidence": assignment_confidence,
        "posterior_probabilities": posterior_probabilities,
        "coords": coords,
        "labels": labels,
    }


def _normalize_method_id(value: object) -> str:
    text = str(value or "").strip().lower()
    compact = text.replace("-", "").replace("_", "").replace(" ", "")
    if "kmeans" in compact:
        return "kmeans"
    if "hdbscan" in compact:
        return "hdbscan"
    if "gmm" in compact or "gaussianmixture" in compact:
        return "gmm"
    if "vade" in compact:
        return "vade"
    return text


def _display_method_name(value: object) -> str:
    method_id = _normalize_method_id(value)
    return METHOD_DISPLAY_NAMES.get(method_id, str(value or "Unknown"))


def _display_preprocess_mode(value: object) -> str:
    text = str(value or "").strip()
    return PREPROCESS_DISPLAY_NAMES.get(text, text or "Unknown")


def _metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def _metric_sort_ascending(metric: str) -> bool:
    return metric in LOWER_IS_BETTER_METRICS


def _format_metric_value(metric: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(numeric):
        return "--"
    if metric in {"coverage", "noise_fraction", "cluster_balance", "avg_confidence", "stability_ari"}:
        return f"{numeric:.1%}"
    if metric in {"fit_time_sec", "stability_time_sec"}:
        return f"{numeric:.3f}s"
    if metric in {"n_clusters", "matched_target_clusters", "cluster_gap", "matched_tolerance"}:
        return str(int(round(numeric)))
    return f"{numeric:.3f}"


def _format_param_summary(row: Dict[str, object]) -> str:
    method_id = _normalize_method_id(row.get("method"))
    param_1_name = str(row.get("param_1_name") or "").strip()
    param_2_name = str(row.get("param_2_name") or "").strip()
    param_1_value = row.get("param_1_value")
    param_2_value = row.get("param_2_value")

    if method_id == "kmeans" and pd.notna(param_1_value):
        return f"k={int(float(param_1_value))}"
    if method_id == "gmm" and pd.notna(param_1_value):
        covariance = str(param_2_value) if pd.notna(param_2_value) else "full"
        return f"components={int(float(param_1_value))}, cov={covariance}"
    if method_id == "hdbscan" and pd.notna(param_1_value):
        if pd.notna(param_2_value):
            return (
                f"min_cluster_size={int(float(param_1_value))}, "
                f"min_samples={int(float(param_2_value))}"
            )
        return f"min_cluster_size={int(float(param_1_value))}"

    parts: List[str] = []
    if param_1_name and pd.notna(param_1_value):
        parts.append(f"{param_1_name}={param_1_value}")
    if param_2_name and pd.notna(param_2_value):
        parts.append(f"{param_2_name}={param_2_value}")
    return ", ".join(parts) if parts else "method-selected operating point"


def _list_benchmark_directories(metrics_root: Path) -> List[Path]:
    discovered: Dict[str, Path] = {}
    for pattern in BENCHMARK_DIR_PATTERNS:
        for candidate in metrics_root.glob(pattern):
            if candidate.is_dir():
                discovered[str(candidate.resolve())] = candidate.resolve()
    return sorted(
        discovered.values(),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )


def _load_benchmark_bundle(benchmark_dir: Path) -> Dict[str, object]:
    def read_csv(name: str) -> pd.DataFrame:
        path = benchmark_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def read_json(name: str) -> Dict[str, object]:
        path = benchmark_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "dir": benchmark_dir,
        "dataset_summary": read_json("dataset_summary.json"),
        "representation_catalog": read_csv("representation_catalog.csv"),
        "full_grid_results": read_csv("full_grid_results.csv"),
        "native_best_results": read_csv("native_best_results.csv"),
        "matched_granularity_results": read_csv("matched_granularity_results.csv"),
        "global_native_leaders": read_csv("global_native_leaders.csv"),
        "global_matched_leaders": read_csv("global_matched_leaders.csv"),
    }


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_existing_path(value: object) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve()
    return path if path.exists() else None


def _is_within_path(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _list_experiment_run_manifests(experiment_root: Path) -> List[Path]:
    if not experiment_root.exists():
        return []

    manifests = [
        path.resolve()
        for path in experiment_root.glob("run_*/run_manifest.json")
        if path.is_file()
    ]
    return sorted(
        manifests,
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )


def _resolve_latest_benchmark_linked_manifest(
    benchmark_dir: Path,
    experiment_root: Path,
    benchmark_profile_id: str,
) -> Tuple[Path, Dict[str, Any]]:
    manifests = _list_experiment_run_manifests(experiment_root)
    if not manifests:
        raise FileNotFoundError(
            "No experiment run manifests were found under "
            f"{experiment_root}. Run the benchmark pipeline first."
        )

    benchmark_cutoff = benchmark_dir.stat().st_mtime + 5.0
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    for manifest_path in manifests:
        try:
            manifest_payload = _read_json_file(manifest_path)
        except Exception:
            continue

        profiles = manifest_payload.get("profiles") or {}
        if benchmark_profile_id not in profiles:
            continue

        manifest_mtime = manifest_path.stat().st_mtime
        if manifest_mtime <= benchmark_cutoff:
            candidates.append((manifest_mtime, manifest_path, manifest_payload))

    if not candidates:
        raise FileNotFoundError(
            "No benchmark-linked experiment snapshot was found for "
            f"{benchmark_dir.name}. Expected a completed run manifest under "
            f"{experiment_root} with the '{benchmark_profile_id}' profile."
        )

    _, manifest_path, manifest_payload = max(candidates, key=lambda item: item[0])
    return manifest_path, manifest_payload


def _normalize_method_lookup(
    methods: Dict[str, Any],
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    lookup: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for raw_key, payload in methods.items():
        method_id = _normalize_method_id(raw_key)
        if not method_id:
            continue
        lookup[method_id] = (str(raw_key), payload if isinstance(payload, dict) else {})
    return lookup


def _select_benchmark_ui_method(
    requested_method_id: str,
    summary_payload: Dict[str, Any],
    available_method_ids: List[str],
) -> Tuple[str, str]:
    if requested_method_id != "auto":
        if requested_method_id not in available_method_ids:
            raise ValueError(
                f"Method '{requested_method_id}' is not available in the latest "
                f"benchmark-linked snapshot. Available methods: {available_method_ids}"
            )
        return requested_method_id, "requested"

    for row in summary_payload.get("ranked_methods", []):
        method_id = _normalize_method_id(row.get("MethodId"))
        if method_id in available_method_ids:
            return method_id, "ranked_methods"

    if "gmm" in available_method_ids:
        return "gmm", "default_gmm"

    return available_method_ids[0], "first_available"


def _read_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _load_ui_bundle_from_files(
    *,
    selected_method_id: str,
    selected_via: str,
    available_method_ids: List[str],
    results_csv_path: Path,
    artifact_path: Path,
    method_display_name: str,
    run_manifest_path: Path,
    benchmark_dir_path: Optional[Path],
    profile_id: str,
    summary_path: Optional[Path],
) -> Dict[str, Any]:
    df = pd.read_csv(results_csv_path)
    artifact_payload = _read_npz_payload(artifact_path)

    if "prepared_features" not in artifact_payload:
        raise KeyError(f"{artifact_path} is missing 'prepared_features'")
    if "labels" not in artifact_payload:
        raise KeyError(f"{artifact_path} is missing 'labels'")

    retrieval_features = np.asarray(
        artifact_payload["prepared_features"],
        dtype=np.float32,
    )
    labels = np.asarray(artifact_payload["labels"])
    coords = (
        np.asarray(artifact_payload["coords"], dtype=np.float32)
        if "coords" in artifact_payload
        else df[["PCA1", "PCA2"]].to_numpy(dtype=np.float32)
    )
    if len(df) != retrieval_features.shape[0]:
        raise ValueError(
            "Benchmark UI snapshot is inconsistent: "
            f"CSV rows={len(df)} but artifact rows={retrieval_features.shape[0]}"
        )

    if "Cluster" in df.columns:
        df["Cluster"] = labels

    return {
        "benchmark_dir": benchmark_dir_path,
        "run_manifest_path": run_manifest_path,
        "profile_id": profile_id,
        "method_id": selected_method_id,
        "method_display_name": method_display_name,
        "selected_via": selected_via,
        "available_methods": available_method_ids,
        "results_csv_path": results_csv_path,
        "artifact_path": artifact_path,
        "summary_path": summary_path,
        "df": df,
        "coords": coords,
        "labels": labels,
        "retrieval_features": retrieval_features,
        "assignment_confidence": (
            np.asarray(artifact_payload["assignment_confidence"], dtype=np.float32)
            if "assignment_confidence" in artifact_payload
            else None
        ),
        "posterior_probabilities": (
            np.asarray(artifact_payload["posterior_probabilities"], dtype=np.float32)
            if "posterior_probabilities" in artifact_payload
            else None
        ),
    }


def _resolve_ui_snapshot_bundle(
    method_id: str,
    snapshot_root: Path = DEFAULT_UI_SNAPSHOT_ROOT,
) -> Dict[str, Any]:
    snapshot_payload = load_ui_snapshot(snapshot_root)
    profile_snapshot = snapshot_payload.get("profile_snapshot") or {}
    method_entries = profile_snapshot.get("methods") or {}
    if not method_entries:
        raise FileNotFoundError(
            "The UI snapshot does not contain any method outputs yet."
        )

    available_method_ids = sorted(
        _normalize_method_id(key) for key in method_entries.keys() if str(key).strip()
    )
    summary_path = _coerce_existing_path(profile_snapshot.get("summary_path"))
    summary_payload = _read_json_file(summary_path) if summary_path else {}
    selected_method_id, selected_via = _select_benchmark_ui_method(
        requested_method_id=_normalize_method_id(method_id),
        summary_payload=summary_payload,
        available_method_ids=available_method_ids,
    )
    method_payload = method_entries.get(selected_method_id) or {}
    results_csv_path = _coerce_existing_path(method_payload.get("results_csv"))
    artifact_path = _coerce_existing_path(method_payload.get("artifact_path"))
    if results_csv_path is None or artifact_path is None:
        raise FileNotFoundError(
            f"UI snapshot is missing files for method '{selected_method_id}'."
        )

    benchmark_snapshot = snapshot_payload.get("benchmark_snapshot") or {}
    benchmark_dir_path = (
        _coerce_existing_path(benchmark_snapshot.get("benchmark_dir"))
        or _coerce_existing_path(benchmark_snapshot.get("source_benchmark_dir"))
    )
    method_display_name = str(
        method_payload.get("display_name")
        or METHOD_DISPLAY_NAMES.get(selected_method_id, selected_method_id.upper())
    )
    return _load_ui_bundle_from_files(
        selected_method_id=selected_method_id,
        selected_via=selected_via,
        available_method_ids=available_method_ids,
        results_csv_path=results_csv_path,
        artifact_path=artifact_path,
        method_display_name=method_display_name,
        run_manifest_path=(snapshot_root.resolve() / "ui_bundle_manifest.json"),
        benchmark_dir_path=benchmark_dir_path,
        profile_id=str(
            profile_snapshot.get("profile_id") or DEFAULT_BENCHMARK_PROFILE_ID
        ),
        summary_path=summary_path,
    )


def resolve_latest_benchmark_ui_bundle(
    method_id: str = "auto",
    benchmark_dir: Optional[str] = None,
    benchmark_profile_id: str = DEFAULT_BENCHMARK_PROFILE_ID,
    benchmark_metrics_root: Optional[str] = None,
    experiment_run_root: Optional[str] = None,
) -> Dict[str, Any]:
    requested_method_id = _normalize_method_id(method_id)
    if requested_method_id not in UI_METHOD_CHOICES:
        raise ValueError(
            f"method_id must be one of {UI_METHOD_CHOICES}, got '{method_id}'"
        )

    if benchmark_dir is None:
        try:
            return _resolve_ui_snapshot_bundle(method_id=requested_method_id)
        except FileNotFoundError:
            pass

    if benchmark_dir:
        benchmark_dir_path = Path(benchmark_dir).expanduser().resolve()
        if not benchmark_dir_path.exists() or not benchmark_dir_path.is_dir():
            raise FileNotFoundError(
                f"Benchmark directory does not exist or is not a directory: {benchmark_dir_path}"
            )
        if _is_within_path(DEFAULT_UI_SNAPSHOT_ROOT, benchmark_dir_path):
            try:
                return _resolve_ui_snapshot_bundle(method_id=requested_method_id)
            except FileNotFoundError:
                pass
    else:
        metrics_root = Path(
            benchmark_metrics_root or (project_root / "output" / "metrics")
        ).resolve()
        benchmark_runs = _list_benchmark_directories(metrics_root)
        if not benchmark_runs:
            raise FileNotFoundError(
                "No thesis benchmark directory was found under output/metrics. "
                "Run the thesis benchmark pipeline first."
            )
        benchmark_dir_path = benchmark_runs[0]

    experiment_root = Path(
        experiment_run_root or (project_root / "output" / "experiment_runs_taxonomy")
    ).resolve()
    run_manifest_path, run_manifest = _resolve_latest_benchmark_linked_manifest(
        benchmark_dir=benchmark_dir_path,
        experiment_root=experiment_root,
        benchmark_profile_id=benchmark_profile_id,
    )

    profiles = run_manifest.get("profiles") or {}
    profile_payload = profiles.get(benchmark_profile_id)
    if not isinstance(profile_payload, dict):
        raise KeyError(
            f"Profile '{benchmark_profile_id}' was not found in {run_manifest_path}"
        )

    method_lookup = _normalize_method_lookup(profile_payload.get("methods") or {})
    if not method_lookup:
        raise FileNotFoundError(
            f"No method outputs were recorded for profile '{benchmark_profile_id}' "
            f"in {run_manifest_path}"
        )

    evaluation_payload = profile_payload.get("evaluation") or {}
    summary_path = _coerce_existing_path(evaluation_payload.get("summary_path"))
    summary_payload = _read_json_file(summary_path) if summary_path else {}

    available_method_ids = sorted(method_lookup.keys())
    selected_method_id, selected_via = _select_benchmark_ui_method(
        requested_method_id=requested_method_id,
        summary_payload=summary_payload,
        available_method_ids=available_method_ids,
    )

    _, method_record = method_lookup[selected_method_id]
    method_summary = method_record.get("method_summary") or {}
    method_outputs = method_summary.get("outputs") or {}
    summary_methods = summary_payload.get("methods") or {}
    summary_method_payload = summary_methods.get(selected_method_id) or {}

    results_csv_path = (
        _coerce_existing_path(summary_method_payload.get("results_csv"))
        or _coerce_existing_path(method_outputs.get("results_csv"))
    )
    artifact_path = (
        _coerce_existing_path(summary_method_payload.get("artifact_path"))
        or _coerce_existing_path(method_outputs.get("retrieval_artifact"))
    )

    if results_csv_path is None:
        raise FileNotFoundError(
            f"Could not resolve a results CSV for method '{selected_method_id}' "
            f"from {run_manifest_path}"
        )
    if artifact_path is None:
        raise FileNotFoundError(
            f"Could not resolve a retrieval artifact for method '{selected_method_id}' "
            f"from {run_manifest_path}"
        )

    method_display_name = str(
        summary_method_payload.get("display_name")
        or METHOD_DISPLAY_NAMES.get(selected_method_id, selected_method_id.upper())
    )

    return _load_ui_bundle_from_files(
        selected_method_id=selected_method_id,
        selected_via=selected_via,
        available_method_ids=available_method_ids,
        results_csv_path=results_csv_path,
        artifact_path=artifact_path,
        method_display_name=method_display_name,
        run_manifest_path=run_manifest_path,
        benchmark_dir_path=benchmark_dir_path,
        profile_id=benchmark_profile_id,
        summary_path=summary_path,
    )


def _draw_empty_axis(ax, title: str, message: str) -> None:
    ax.clear()
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def launch_ui(
    df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    top_n: int = 5,
    audio_dir: str = "genres_original",
    clustering_method: str = "K-means",
    retrieval_features: Optional[np.ndarray] = None,
    retrieval_method_id: Optional[str] = None,
    assignment_confidence: Optional[np.ndarray] = None,
    posterior_probabilities: Optional[np.ndarray] = None,
    benchmark_dir: Optional[str] = None,
    available_method_ids: Optional[List[str]] = None,
):
    _ensure_pygame_mixer()
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    df = df.reset_index(drop=True).copy()
    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels)

    if coords.ndim != 2 or coords.shape[0] != len(df) or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape ({len(df)}, 2), got {coords.shape}"
        )
    if labels.shape[0] != len(df):
        raise ValueError(f"labels length must match df length: {labels.shape[0]} != {len(df)}")

    payload = _resolve_retrieval_payload(
        df=df,
        coords=coords,
        labels=labels,
        retrieval_features=retrieval_features,
        retrieval_method_id=retrieval_method_id,
        assignment_confidence=assignment_confidence,
        posterior_probabilities=posterior_probabilities,
    )
    retrieval_features = payload["retrieval_features"]
    assignment_confidence = payload["assignment_confidence"]
    posterior_probabilities = payload["posterior_probabilities"]

    supports_confidence = assignment_confidence is not None
    supports_posteriors = (
        posterior_probabilities is not None and posterior_probabilities.shape[1] > 1
    )
    method_id = _normalize_method_id(clustering_method)
    normalized_available_method_ids: List[str] = []
    seen_method_ids = set()
    for candidate in list(available_method_ids or []):
        normalized = _normalize_method_id(candidate)
        if normalized and normalized not in seen_method_ids:
            normalized_available_method_ids.append(normalized)
            seen_method_ids.add(normalized)
    if method_id and method_id not in seen_method_ids:
        normalized_available_method_ids.insert(0, method_id)
        seen_method_ids.add(method_id)
    occupied_cluster_ids = sorted(int(value) for value in np.unique(labels) if int(value) != -1)
    noise_count = int(np.sum(labels == -1))
    cluster_counts_series = pd.Series(labels, dtype="int64").value_counts().sort_values(
        ascending=False
    )
    cluster_color_map: Dict[int, object] = {}
    palette = plt.get_cmap("tab20")
    for palette_idx, cluster_id in enumerate(occupied_cluster_ids):
        cluster_color_map[cluster_id] = palette(palette_idx % palette.N)
    if noise_count:
        cluster_color_map[-1] = "#94a3b8"
    point_colors = [cluster_color_map.get(int(label), "#64748b") for label in labels]

    root = tk.Tk()
    root.title(f"Clustering Explorer - {_display_method_name(method_id)}")
    root.geometry("1260x820")
    root.minsize(980, 640)

    audio_folder = audio_dir
    current_song = {"name": None, "playing": False}
    recommendation_rows: List[Dict[str, object]] = []
    selected_song_idx = {"value": None}
    song_to_index = {str(name): idx for idx, name in enumerate(df["Song"])}

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    header_font = tkfont.Font(family="Segoe UI", size=15, weight="bold")
    normal_font = tkfont.Font(size=11)
    small_font = tkfont.Font(size=10)

    header = ttk.Frame(root, padding=(10, 6))
    header.pack(fill="x")
    ttk.Label(header, text="Clustering Explorer", font=header_font).pack(side="left")
    header_summary_var = tk.StringVar(
        value=(
            f"{_display_method_name(method_id)} | {len(df)} songs | "
            f"{len(occupied_cluster_ids)} occupied clusters"
            + (f" | {noise_count} noise points" if noise_count else "")
        )
    )
    ttk.Label(header, textvariable=header_summary_var, font=small_font).pack(
        side="left",
        padx=(12, 0),
    )
    method_selector_var = tk.StringVar(value=_display_method_name(method_id))
    method_label_to_id = {
        _display_method_name(value): value for value in normalized_available_method_ids
    }
    if len(method_label_to_id) > 1:
        ttk.Label(header, text="Algorithm", font=small_font).pack(
            side="left",
            padx=(16, 6),
        )
        method_selector = ttk.Combobox(
            header,
            textvariable=method_selector_var,
            values=list(method_label_to_id.keys()),
            state="readonly",
            width=12,
        )
        method_selector.pack(side="left")
    else:
        method_selector = None
    benchmark_status_var = tk.StringVar(value="Benchmark dashboard: ready to load latest thesis run")
    ttk.Label(header, textvariable=benchmark_status_var, font=small_font).pack(
        side="right",
        padx=(0, 10),
    )
    benchmark_button = ttk.Button(header, text="Open benchmark dashboard")
    benchmark_button.pack(side="right")

    paned = ttk.Panedwindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=10, pady=6)

    left = ttk.Frame(paned, width=280)
    paned.add(left, weight=1)

    ttk.Label(left, text="Search songs", font=normal_font).pack(anchor="w", pady=(4, 0))
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.pack(fill="x", pady=5)

    ttk.Label(left, text="All songs", font=normal_font).pack(anchor="w")
    song_list_frame = ttk.Frame(left)
    song_list_frame.pack(fill="both", expand=True)
    song_list = tk.Listbox(song_list_frame, font=normal_font, activestyle="none")
    scroll_songs = ttk.Scrollbar(song_list_frame, orient="vertical", command=song_list.yview)
    song_list.config(yscrollcommand=scroll_songs.set)
    song_list.pack(side="left", fill="both", expand=True)
    scroll_songs.pack(side="right", fill="y")

    play_button = ttk.Button(left, text="Play selected")
    play_button.pack(fill="x", pady=6)

    for name in df["Song"]:
        song_list.insert("end", name)

    right = ttk.Frame(paned)
    paned.add(right, weight=3)

    plot_summary_var = tk.StringVar(
        value=(
            "PCA is used only for visualization. The right-side bar chart shows cluster sizes "
            "so you can see whether recommendations come from a broad or tight region."
        )
    )
    ttk.Label(
        right,
        textvariable=plot_summary_var,
        font=small_font,
        wraplength=820,
        justify="left",
    ).pack(fill="x", padx=5, pady=(0, 6))

    plot_frame = ttk.LabelFrame(right, text="Cluster map and size profile")
    plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 6))

    fig = plt.Figure(figsize=(7.8, 4.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.3], wspace=0.28)
    scatter_ax = fig.add_subplot(grid[0, 0])
    size_ax = fig.add_subplot(grid[0, 1])

    def full_scatter(alpha: float = 0.85):
        return scatter_ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=point_colors,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.25,
            s=44,
        )

    base_scatter = full_scatter()
    scatter_ax.set_xlabel("PCA-1")
    scatter_ax.set_ylabel("PCA-2")
    scatter_ax.set_title(f"{_display_method_name(method_id)} cluster map (PCA)")
    scatter_ax.grid(True, linestyle=":", linewidth=0.4)

    for cluster_id in occupied_cluster_ids:
        mask = labels == cluster_id
        if not mask.any():
            continue
        centroid = coords[mask].mean(axis=0)
        scatter_ax.scatter(
            centroid[0],
            centroid[1],
            marker="X",
            s=100,
            c=[cluster_color_map[cluster_id]],
            edgecolors="#0f172a",
            linewidths=0.8,
            zorder=4,
        )
        scatter_ax.text(
            centroid[0],
            centroid[1],
            f"C{cluster_id}",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            color="#0f172a",
            zorder=5,
        )

    ordered_counts = cluster_counts_series.sort_values(ascending=True)
    size_bars = size_ax.barh(
        range(len(ordered_counts)),
        ordered_counts.values,
        color=[cluster_color_map.get(int(cluster_id), "#64748b") for cluster_id in ordered_counts.index],
        alpha=0.92,
    )
    size_ax.set_yticks(range(len(ordered_counts)))
    size_ax.set_yticklabels(
        [
            "Noise" if int(cluster_id) == -1 else f"C{int(cluster_id)}"
            for cluster_id in ordered_counts.index
        ],
        fontsize=8,
    )
    size_ax.set_title("Cluster sizes")
    size_ax.set_xlabel("Songs")
    size_ax.grid(True, axis="x", linestyle=":", linewidth=0.4)
    max_count = float(max(ordered_counts.max(), 1.0))
    for bar, count in zip(size_bars, ordered_counts.values):
        size_ax.text(
            float(count) + (0.02 * max_count),
            bar.get_y() + (bar.get_height() / 2),
            str(int(count)),
            va="center",
            fontsize=8,
            color="#0f172a",
        )

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    recommendation_header = ttk.Frame(right)
    recommendation_header.pack(fill="x", pady=(4, 0))
    ttk.Label(recommendation_header, text="Recommendations", font=header_font).pack(
        side="left"
    )

    controls_frame = ttk.Frame(right)
    controls_frame.pack(fill="x", pady=(4, 6))

    default_ranking_mode = "Distance"
    if (
        supports_posteriors
        and str(getattr(fv, "default_recommendation_ranking_method", "distance")).lower()
        == "posterior_weighted"
    ):
        default_ranking_mode = "Posterior-weighted"

    top_n_var = tk.StringVar(value=str(top_n))
    ranking_mode_var = tk.StringVar(value=default_ranking_mode)
    min_confidence_var = tk.StringVar(
        value=f"{float(getattr(fv, 'default_min_assignment_confidence', 0.0)):.2f}"
    )
    min_posterior_var = tk.StringVar(
        value=f"{float(getattr(fv, 'default_min_selected_cluster_posterior', 0.0)):.2f}"
    )

    ttk.Label(controls_frame, text="Top N").pack(side="left")
    top_n_spinbox = ttk.Spinbox(
        controls_frame,
        from_=1,
        to=20,
        increment=1,
        textvariable=top_n_var,
        width=4,
    )
    top_n_spinbox.pack(side="left", padx=(6, 12))

    ttk.Label(controls_frame, text="Ranking").pack(side="left")
    ranking_values = ["Distance"]
    if supports_posteriors:
        ranking_values.append("Posterior-weighted")
    ranking_combo = ttk.Combobox(
        controls_frame,
        textvariable=ranking_mode_var,
        values=ranking_values,
        width=18,
        state="readonly",
    )
    ranking_combo.pack(side="left", padx=(6, 12))
    if not supports_posteriors:
        ranking_combo.state(["disabled"])

    ttk.Label(controls_frame, text="Min confidence").pack(side="left")
    min_confidence_box = ttk.Spinbox(
        controls_frame,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=min_confidence_var,
        width=6,
    )
    min_confidence_box.pack(side="left", padx=(6, 12))
    if not supports_confidence:
        min_confidence_box.state(["disabled"])

    ttk.Label(controls_frame, text="Min selected-cluster posterior").pack(side="left")
    min_posterior_box = ttk.Spinbox(
        controls_frame,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=min_posterior_var,
        width=6,
    )
    min_posterior_box.pack(side="left", padx=(6, 0))
    if not supports_posteriors:
        min_posterior_box.state(["disabled"])

    recommendation_summary_var = tk.StringVar(
        value=(
            "Retrieval uses the full prepared feature space inside the selected "
            "cluster. PCA-2 is kept for visualization only. Probabilistic methods "
            "default to posterior-weighted ranking unless you override it."
        )
    )
    recommendation_summary = ttk.Label(
        right,
        textvariable=recommendation_summary_var,
        font=small_font,
        wraplength=760,
        justify="left",
    )
    recommendation_summary.pack(fill="x", pady=(0, 6))

    recommendation_list_frame = ttk.Frame(right)
    recommendation_list_frame.pack(fill="both", expand=True, pady=(0, 6))
    rec_list = tk.Listbox(recommendation_list_frame, font=small_font, activestyle="none")
    scroll_rec = ttk.Scrollbar(
        recommendation_list_frame,
        orient="vertical",
        command=rec_list.yview,
    )
    rec_list.config(yscrollcommand=scroll_rec.set)
    rec_list.pack(side="left", fill="both", expand=True)
    scroll_rec.pack(side="right", fill="y")

    rec_play_button = ttk.Button(right, text="Play recommendation")
    rec_play_button.pack(fill="x", pady=(0, 6))

    now_playing_var = tk.StringVar(value="Now playing: None")
    now_playing_label = ttk.Label(root, textvariable=now_playing_var, font=normal_font)
    now_playing_label.pack(side="bottom", fill="x", padx=10, pady=6)

    def filter_songs(*_):
        query = search_var.get().lower()
        song_list.delete(0, "end")
        for name in df["Song"]:
            if query in name.lower():
                song_list.insert("end", name)

    search_var.trace_add("write", filter_songs)

    def redraw_plot(sel_idx: int, neighbour_indices: np.ndarray):
        scatter_ax.clear()
        size_ax.clear()

        full_scatter(alpha=0.22)
        selected_cluster = int(labels[sel_idx])
        if selected_cluster != -1:
            same_cluster = np.where(labels == selected_cluster)[0]
            scatter_ax.scatter(
                coords[same_cluster, 0],
                coords[same_cluster, 1],
                c=[cluster_color_map.get(selected_cluster, "#64748b")],
                alpha=0.52,
                edgecolors="white",
                linewidths=0.30,
                s=56,
                zorder=2,
            )

        scatter_ax.scatter(
            coords[sel_idx, 0],
            coords[sel_idx, 1],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=3,
        )
        for i in neighbour_indices:
            scatter_ax.scatter(
                coords[i, 0],
                coords[i, 1],
                marker="D",
                s=90,
                facecolors="#fde68a",
                edgecolors="#0f172a",
                linewidths=0.8,
                zorder=3,
            )
            scatter_ax.plot(
                [coords[sel_idx, 0], coords[i, 0]],
                [coords[sel_idx, 1], coords[i, 1]],
                linestyle="--",
                linewidth=1.0,
                color="#94a3b8",
                zorder=2,
            )

        pts = (
            np.vstack([coords[sel_idx], coords[neighbour_indices]])
            if len(neighbour_indices)
            else coords[[sel_idx]]
        )
        margin = 0.5
        xmin, ymin = pts.min(axis=0) - margin
        xmax, ymax = pts.max(axis=0) + margin
        scatter_ax.set_xlim(xmin, xmax)
        scatter_ax.set_ylim(ymin, ymax)
        scatter_ax.set_xlabel("PCA-1")
        scatter_ax.set_ylabel("PCA-2")
        scatter_ax.set_title(
            (
                f"{_display_method_name(method_id)} local view | noise track"
                if selected_cluster == -1
                else f"{_display_method_name(method_id)} local view | cluster C{selected_cluster}"
            )
        )
        scatter_ax.grid(True, linestyle=":", linewidth=0.4)

        ordered_counts = cluster_counts_series.sort_values(ascending=True)
        display_cluster_ids = [int(cluster_id) for cluster_id in ordered_counts.index]
        bars = size_ax.barh(
            range(len(ordered_counts)),
            ordered_counts.values,
            color=[cluster_color_map.get(cluster_id, "#64748b") for cluster_id in display_cluster_ids],
            alpha=0.92,
        )
        size_ax.set_yticks(range(len(ordered_counts)))
        size_ax.set_yticklabels(
            [
                "Noise" if cluster_id == -1 else f"C{cluster_id}"
                for cluster_id in display_cluster_ids
            ],
            fontsize=8,
        )
        size_ax.set_title("Cluster sizes")
        size_ax.set_xlabel("Songs")
        size_ax.grid(True, axis="x", linestyle=":", linewidth=0.4)
        if selected_cluster in display_cluster_ids:
            selected_bar_idx = display_cluster_ids.index(selected_cluster)
            bars[selected_bar_idx].set_edgecolor("#0f172a")
            bars[selected_bar_idx].set_linewidth(1.8)
        max_count = float(max(ordered_counts.max(), 1.0))
        for bar, count in zip(bars, ordered_counts.values):
            size_ax.text(
                float(count) + (0.02 * max_count),
                bar.get_y() + (bar.get_height() / 2),
                str(int(count)),
                va="center",
                fontsize=8,
                color="#0f172a",
            )
        canvas.draw_idle()

    def _format_recommendation_row(row: Dict[str, object]) -> str:
        parts = [
            f"{row['song']}",
            f"dist={float(row['distance']):.4f}",
        ]
        if row.get("confidence") is not None:
            parts.append(f"conf={float(row['confidence']):.3f}")
        if row.get("cluster_posterior") is not None:
            parts.append(f"p(cluster)={float(row['cluster_posterior']):.3f}")
        if row.get("ranking_score") is not None and row.get("ranking_mode") == "Posterior-weighted":
            parts.append(f"weighted={float(row['ranking_score']):.4f}")
        return " | ".join(parts)

    def refresh_recommendations(*_):
        if selected_song_idx["value"] is None:
            return

        sel_idx = int(selected_song_idx["value"])
        selected_song = str(df.at[sel_idx, "Song"])
        selected_cluster = int(labels[sel_idx])

        rec_list.delete(0, "end")
        recommendation_rows.clear()

        if selected_cluster == -1:
            plot_summary_var.set(
                f"{selected_song} is labeled as noise, so the explorer keeps the map local but disables within-cluster recommendations."
            )
            recommendation_summary_var.set(
                "The selected song is labeled as HDBSCAN noise, so within-cluster "
                "recommendations are intentionally disabled."
            )
            redraw_plot(sel_idx, np.array([], dtype=np.int32))
            return

        top_n_value = _parse_bounded_int(top_n_var.get(), top_n, 1, 20)
        ranking_mode = ranking_mode_var.get()
        min_confidence = _parse_bounded_float(min_confidence_var.get(), 0.0, 0.0, 1.0)
        min_posterior = _parse_bounded_float(min_posterior_var.get(), 0.0, 0.0, 1.0)

        candidate_indices = np.where(labels == selected_cluster)[0]
        candidate_indices = candidate_indices[candidate_indices != sel_idx]

        candidate_confidence = None
        if assignment_confidence is not None:
            candidate_confidence = assignment_confidence[candidate_indices]
            keep = candidate_confidence >= min_confidence
            candidate_indices = candidate_indices[keep]
            candidate_confidence = candidate_confidence[keep]

        candidate_cluster_posterior = None
        if (
            posterior_probabilities is not None
            and 0 <= selected_cluster < posterior_probabilities.shape[1]
        ):
            candidate_cluster_posterior = posterior_probabilities[
                candidate_indices, selected_cluster
            ]
            keep = candidate_cluster_posterior >= min_posterior
            candidate_indices = candidate_indices[keep]
            if candidate_confidence is not None:
                candidate_confidence = candidate_confidence[keep]
            candidate_cluster_posterior = candidate_cluster_posterior[keep]

        if len(candidate_indices) == 0:
            plot_summary_var.set(
                f"Cluster C{selected_cluster} is visible on the map, but the active recommendation filters removed every candidate."
            )
            recommendation_summary_var.set(
                f"No candidates remain for '{selected_song}' after the active "
                "prepared-space filters."
            )
            redraw_plot(sel_idx, np.array([], dtype=np.int32))
            return

        selected_vector = retrieval_features[sel_idx]
        candidate_vectors = retrieval_features[candidate_indices]
        distances = np.linalg.norm(candidate_vectors - selected_vector, axis=1)

        ranking_scores = distances.copy()
        if ranking_mode == "Posterior-weighted" and candidate_cluster_posterior is not None:
            ranking_scores = distances / np.clip(candidate_cluster_posterior, 1e-3, 1.0)
        else:
            ranking_mode = "Distance"

        order = np.lexsort((distances, ranking_scores))[:top_n_value]
        chosen_indices = candidate_indices[order]

        for rank_position, order_idx in enumerate(order, start=1):
            candidate_idx = int(candidate_indices[order_idx])
            row = {
                "rank": rank_position,
                "song": str(df.at[candidate_idx, "Song"]),
                "song_index": candidate_idx,
                "distance": float(distances[order_idx]),
                "confidence": (
                    None
                    if candidate_confidence is None
                    else float(candidate_confidence[order_idx])
                ),
                "cluster_posterior": (
                    None
                    if candidate_cluster_posterior is None
                    else float(candidate_cluster_posterior[order_idx])
                ),
                "ranking_score": float(ranking_scores[order_idx]),
                "ranking_mode": ranking_mode,
            }
            recommendation_rows.append(row)
            rec_list.insert("end", _format_recommendation_row(row))

        summary_parts = [
            f"Selected: {selected_song}",
            f"cluster={selected_cluster}",
            f"retrieval=prepared-space {ranking_mode.lower()}",
            f"candidates={len(candidate_indices)}",
        ]
        if assignment_confidence is not None:
            summary_parts.append(f"min_conf={min_confidence:.2f}")
        if candidate_cluster_posterior is not None:
            summary_parts.append(f"min_p(cluster)={min_posterior:.2f}")
        recommendation_summary_var.set(" | ".join(summary_parts))
        plot_summary_var.set(
            f"{selected_song} sits in cluster C{selected_cluster}. The highlighted bar shows cluster size, and the diamonds mark the {len(chosen_indices)} visible recommendations."
        )

        redraw_plot(sel_idx, chosen_indices)

    def on_song_select(_event):
        selection = song_list.curselection()
        if not selection:
            return
        song_name = song_list.get(selection[0])
        sel_idx = song_to_index[song_name]
        selected_song_idx["value"] = sel_idx
        refresh_recommendations()

    def play_audio(song_name: str):
        pygame.mixer.music.stop()

        song_row = df[df["Song"] == song_name]
        audio_file = None

        candidates = [song_name]
        if not song_name.lower().endswith((".wav", ".mp3")):
            candidates.append(f"{song_name}.wav")
            candidates.append(f"{song_name}.mp3")

        for candidate in candidates:
            potential_path = os.path.join(audio_folder, candidate)
            if os.path.exists(potential_path):
                audio_file = potential_path
                break

        if audio_file is None and not song_row.empty and "Genre" in df.columns:
            genre = song_row["Genre"].values[0]
            for candidate in candidates:
                potential_path = os.path.join(audio_folder, genre, candidate)
                if os.path.exists(potential_path):
                    audio_file = potential_path
                    break

        if audio_file is None:
            parts = song_name.split(".")
            if parts:
                genre = parts[0]
                for candidate in candidates:
                    potential_path = os.path.join(audio_folder, genre, candidate)
                    if os.path.exists(potential_path):
                        audio_file = potential_path
                        break

        if audio_file is None:
            for root_dir, _, files in os.walk(audio_folder):
                for candidate in candidates:
                    if candidate in files:
                        audio_file = os.path.join(root_dir, candidate)
                        break
                if audio_file:
                    break

        if audio_file is None:
            messagebox.showerror(
                "Error",
                f"Could not find audio file for {song_name}\nSearched in {audio_folder}",
            )
            return

        if current_song["name"] == song_name and current_song["playing"]:
            current_song["playing"] = False
            now_playing_var.set("Now playing: None")
            return

        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            current_song["name"] = song_name
            current_song["playing"] = True
            now_playing_var.set(f"Now playing: {song_name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to play audio: {exc}")

    def play_selected_song():
        selection = song_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a song first")
            return
        play_audio(song_list.get(selection[0]))

    def play_recommended_song():
        selection = rec_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recommendation first")
            return
        row_idx = int(selection[0])
        if not (0 <= row_idx < len(recommendation_rows)):
            messagebox.showinfo("Info", "That recommendation entry is not playable")
            return
        play_audio(str(recommendation_rows[row_idx]["song"]))

    song_list.bind("<Double-1>", lambda _event: play_selected_song())
    rec_list.bind("<Double-1>", lambda _event: play_recommended_song())
    play_button.config(command=play_selected_song)
    rec_play_button.config(command=play_recommended_song)
    song_list.bind("<<ListboxSelect>>", on_song_select)

    for control_var in (
        top_n_var,
        ranking_mode_var,
        min_confidence_var,
        min_posterior_var,
    ):
        control_var.trace_add("write", refresh_recommendations)

    benchmark_window_state: Dict[str, object] = {"window": None}

    def open_benchmark_dashboard() -> None:
        existing = benchmark_window_state.get("window")
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return

        metrics_root = project_root / "output" / "metrics"
        benchmark_runs = _list_benchmark_directories(metrics_root)
        if benchmark_dir:
            explicit_path = Path(benchmark_dir).expanduser().resolve()
            if explicit_path.exists() and explicit_path.is_dir():
                benchmark_runs = [explicit_path] + [
                    path for path in benchmark_runs if path != explicit_path
                ]

        dashboard = tk.Toplevel(root)
        benchmark_window_state["window"] = dashboard
        dashboard.title("Thesis Benchmark Dashboard")
        dashboard.geometry("1240x840")
        dashboard.minsize(980, 680)

        summary_var = tk.StringVar(
            value="Loading thesis benchmark artifacts from output/metrics..."
        )
        method_note_var = tk.StringVar(
            value="This dashboard compares the current clustering method against the thesis benchmark runs."
        )
        detail_var = tk.StringVar(
            value="Select a leaderboard row to see its representation rationale and operating point."
        )

        header_frame = ttk.Frame(dashboard, padding=(10, 8))
        header_frame.pack(fill="x")
        ttk.Label(header_frame, text="Benchmark Dashboard", font=header_font).pack(anchor="w")
        ttk.Label(
            header_frame,
            textvariable=summary_var,
            font=small_font,
            wraplength=1180,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(
            header_frame,
            textvariable=method_note_var,
            font=small_font,
            wraplength=1180,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        controls = ttk.LabelFrame(dashboard, text="Filters")
        controls.pack(fill="x", padx=10, pady=(0, 8))
        row1 = ttk.Frame(controls)
        row1.pack(fill="x", padx=10, pady=(8, 4))
        row2 = ttk.Frame(controls)
        row2.pack(fill="x", padx=10, pady=(0, 8))

        view_key_by_label = {
            "Matched leaders by target": "global_matched_leaders",
            "Global native leaders": "global_native_leaders",
            "Native best by representation": "native_best_results",
            "Matched candidates": "matched_granularity_results",
            "Full grid search": "full_grid_results",
        }
        metric_key_by_label = {
            "NMI": "nmi",
            "Silhouette": "silhouette",
            "Stability ARI": "stability_ari",
            "Coverage": "coverage",
            "Noise fraction": "noise_fraction",
            "Fit time": "fit_time_sec",
            "Internal score": "internal_selection_score",
            "Cluster count": "n_clusters",
        }

        run_var = tk.StringVar(value=benchmark_runs[0].name if benchmark_runs else "")
        view_var = tk.StringVar(value="Matched leaders by target")
        metric_var = tk.StringVar(value="NMI")
        method_var = tk.StringVar(value="All methods")
        preprocess_var = tk.StringVar(value="All preprocessing")
        target_var = tk.StringVar(value="All targets")
        top_rows_var = tk.StringVar(value="10")

        ttk.Label(row1, text="Run").pack(side="left")
        run_combo = ttk.Combobox(row1, textvariable=run_var, state="readonly", width=40)
        run_combo["values"] = [path.name for path in benchmark_runs]
        run_combo.pack(side="left", padx=(6, 12))
        ttk.Label(row1, text="View").pack(side="left")
        view_combo = ttk.Combobox(
            row1,
            textvariable=view_var,
            values=list(view_key_by_label.keys()),
            state="readonly",
            width=28,
        )
        view_combo.pack(side="left", padx=(6, 12))
        ttk.Label(row1, text="Metric").pack(side="left")
        metric_combo = ttk.Combobox(
            row1,
            textvariable=metric_var,
            values=list(metric_key_by_label.keys()),
            state="readonly",
            width=18,
        )
        metric_combo.pack(side="left", padx=(6, 0))

        ttk.Label(row2, text="Method").pack(side="left")
        method_combo = ttk.Combobox(
            row2,
            textvariable=method_var,
            values=["All methods"],
            state="readonly",
            width=16,
        )
        method_combo.pack(side="left", padx=(6, 12))
        ttk.Label(row2, text="Preprocess").pack(side="left")
        preprocess_combo = ttk.Combobox(
            row2,
            textvariable=preprocess_var,
            values=["All preprocessing"],
            state="readonly",
            width=18,
        )
        preprocess_combo.pack(side="left", padx=(6, 12))
        ttk.Label(row2, text="Target").pack(side="left")
        target_combo = ttk.Combobox(
            row2,
            textvariable=target_var,
            values=["All targets"],
            state="readonly",
            width=12,
        )
        target_combo.pack(side="left", padx=(6, 12))
        ttk.Label(row2, text="Top rows").pack(side="left")
        ttk.Spinbox(
            row2,
            from_=3,
            to=25,
            increment=1,
            textvariable=top_rows_var,
            width=4,
        ).pack(side="left", padx=(6, 0))

        content = ttk.Panedwindow(dashboard, orient="horizontal")
        content.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        charts_panel = ttk.Frame(content)
        table_panel = ttk.Frame(content, width=430)
        content.add(charts_panel, weight=3)
        content.add(table_panel, weight=2)

        lead_chart_frame = ttk.LabelFrame(charts_panel, text="Leaderboard")
        lead_chart_frame.pack(fill="both", expand=True, pady=(0, 8))
        tradeoff_frame = ttk.LabelFrame(charts_panel, text="Tradeoff / target trend")
        tradeoff_frame.pack(fill="both", expand=True)

        lead_fig = plt.Figure(figsize=(7.2, 4.2))
        lead_ax = lead_fig.add_subplot(111)
        lead_canvas = FigureCanvasTkAgg(lead_fig, master=lead_chart_frame)
        lead_canvas.draw()
        lead_canvas.get_tk_widget().pack(fill="both", expand=True)

        trade_fig = plt.Figure(figsize=(7.2, 3.8))
        trade_ax = trade_fig.add_subplot(111)
        trade_canvas = FigureCanvasTkAgg(trade_fig, master=tradeoff_frame)
        trade_canvas.draw()
        trade_canvas.get_tk_widget().pack(fill="both", expand=True)

        table_frame = ttk.LabelFrame(table_panel, text="Visible leaderboard")
        table_frame.pack(fill="both", expand=True)
        columns = (
            "rank",
            "method",
            "combo",
            "preprocess",
            "clusters",
            "metric",
            "silhouette",
            "nmi",
            "stability",
            "coverage",
        )
        table = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        for column, title, width in (
            ("rank", "#", 36),
            ("method", "Method", 82),
            ("combo", "Representation", 138),
            ("preprocess", "Preprocess", 110),
            ("clusters", "Clusters", 64),
            ("metric", "Metric", 86),
            ("silhouette", "Silhouette", 78),
            ("nmi", "NMI", 64),
            ("stability", "Stability", 80),
            ("coverage", "Coverage", 80),
        ):
            table.heading(column, text=title)
            table.column(column, width=width, anchor="center")
        table.column("combo", anchor="w")
        table.column("preprocess", anchor="w")
        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=scroll_y.set)
        table.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")

        ttk.Label(
            table_panel,
            textvariable=detail_var,
            font=small_font,
            wraplength=420,
            justify="left",
        ).pack(fill="x", pady=(8, 0))

        bundle_cache: Dict[str, Dict[str, object]] = {}
        dashboard_state: Dict[str, object] = {"bundle": None, "rows": {}}

        def update_detail(*_args) -> None:
            selection = table.selection()
            row_lookup = dashboard_state.get("rows", {})
            if not selection or selection[0] not in row_lookup:
                return
            row = row_lookup[selection[0]]
            detail_var.set(
                f"{_display_method_name(row.get('method'))} | {row.get('combo')} / "
                f"{_display_preprocess_mode(row.get('preprocess_mode'))}\n"
                f"{_format_param_summary(row)} | clusters={_format_metric_value('n_clusters', row.get('n_clusters'))} | "
                f"silhouette={_format_metric_value('silhouette', row.get('silhouette'))} | "
                f"NMI={_format_metric_value('nmi', row.get('nmi'))} | "
                f"coverage={_format_metric_value('coverage', row.get('coverage'))}\n"
                f"{row.get('representation_rationale', 'No representation rationale stored for this row.')}"
            )

        def refresh_dashboard(*_args) -> None:
            bundle = dashboard_state.get("bundle")
            if bundle is None:
                _draw_empty_axis(lead_ax, "Leaderboard", "No thesis benchmark run is available.")
                _draw_empty_axis(trade_ax, "Tradeoff / target trend", "Run the thesis benchmark or choose a different benchmark directory.")
                lead_canvas.draw_idle()
                trade_canvas.draw_idle()
                return

            full_grid = bundle.get("full_grid_results", pd.DataFrame())
            methods_available = sorted(
                {
                    _display_method_name(value)
                    for value in full_grid.get("method", [])
                    if str(value).strip()
                }
            )
            method_values = ["All methods"] + methods_available
            method_combo["values"] = method_values
            if method_var.get() not in method_values:
                method_var.set("All methods")

            preprocess_available = sorted(
                {
                    _display_preprocess_mode(value)
                    for value in full_grid.get("preprocess_mode", [])
                    if str(value).strip()
                }
            )
            preprocess_values = ["All preprocessing"] + preprocess_available
            preprocess_combo["values"] = preprocess_values
            if preprocess_var.get() not in preprocess_values:
                preprocess_var.set("All preprocessing")

            matched_source = bundle.get("matched_granularity_results", pd.DataFrame())
            target_values = ["All targets"]
            if "matched_target_clusters" in matched_source.columns:
                target_values.extend(
                    [
                        str(int(value))
                        for value in sorted(
                            pd.to_numeric(
                                matched_source["matched_target_clusters"], errors="coerce"
                            ).dropna().unique()
                        )
                    ]
                )
            target_combo["values"] = target_values
            if target_var.get() not in target_values:
                target_var.set("All targets")

            dataset_key = view_key_by_label[view_var.get()]
            metric_key = metric_key_by_label[metric_var.get()]
            view_df = bundle.get(dataset_key, pd.DataFrame()).copy()
            if method_var.get() != "All methods" and "method" in view_df.columns:
                view_df = view_df[
                    view_df["method"].map(_display_method_name) == method_var.get()
                ]
            if preprocess_var.get() != "All preprocessing" and "preprocess_mode" in view_df.columns:
                view_df = view_df[
                    view_df["preprocess_mode"].map(_display_preprocess_mode)
                    == preprocess_var.get()
                ]
            if target_var.get() != "All targets" and "matched_target_clusters" in view_df.columns:
                view_df = view_df[
                    pd.to_numeric(view_df["matched_target_clusters"], errors="coerce")
                    == int(target_var.get())
                ]

            ranked = view_df.copy()
            if metric_key not in ranked.columns:
                ranked = pd.DataFrame()
            else:
                ranked["_metric_value"] = pd.to_numeric(ranked[metric_key], errors="coerce")
                ranked = ranked.dropna(subset=["_metric_value"]).copy()
                ascending = _metric_sort_ascending(metric_key)
                ranked = ranked.sort_values("_metric_value", ascending=ascending).reset_index(drop=True)

            if ranked.empty:
                _draw_empty_axis(lead_ax, "Leaderboard", "No rows survived the active benchmark filters.")
                _draw_empty_axis(trade_ax, "Tradeoff / target trend", "Try another view, metric, or filter combination.")
                lead_canvas.draw_idle()
                trade_canvas.draw_idle()
                for item in table.get_children():
                    table.delete(item)
                dashboard_state["rows"] = {}
                summary_var.set("No rows survived the current benchmark filter combination.")
                method_note_var.set("Relax the filters or switch benchmark views to compare methods.")
                return

            top_rows = _parse_bounded_int(top_rows_var.get(), 10, 3, 25)
            top_df = ranked.head(top_rows).copy()
            plot_df = top_df.iloc[::-1].copy()

            lead_ax.clear()
            lead_ax.barh(
                range(len(plot_df)),
                plot_df["_metric_value"],
                color=[
                    METHOD_COLORS.get(_normalize_method_id(row_method), "#64748b")
                    for row_method in plot_df["method"]
                ],
                alpha=0.92,
            )
            lead_ax.set_yticks(range(len(plot_df)))
            lead_ax.set_yticklabels(
                [
                    shorten(
                        f"{row.combo} | {_display_preprocess_mode(row.preprocess_mode)} | {_display_method_name(row.method)}",
                        width=56,
                        placeholder="...",
                    )
                    for row in plot_df.itertuples(index=False)
                ],
                fontsize=8,
            )
            lead_ax.set_xlabel(_metric_display_name(metric_key))
            lead_ax.set_title(
                f"{'Lowest' if _metric_sort_ascending(metric_key) else 'Highest'} "
                f"{_metric_display_name(metric_key).lower()} rows"
            )
            lead_ax.grid(True, axis="x", linestyle=":", linewidth=0.4)
            max_metric = float(max(plot_df["_metric_value"].max(), 1.0))
            for row_position, (_, row) in enumerate(plot_df.iterrows()):
                lead_ax.text(
                    float(row["_metric_value"]) + (0.015 * max_metric),
                    row_position,
                    _format_metric_value(metric_key, row["_metric_value"]),
                    va="center",
                    fontsize=8,
                )
            lead_canvas.draw_idle()

            trade_ax.clear()
            if "matched_target_clusters" in ranked.columns and "matched" in dataset_key:
                trend_source = ranked.copy()
                for method_name, group in trend_source.groupby("method", sort=False):
                    group = group.sort_values("matched_target_clusters")
                    trade_ax.plot(
                        group["matched_target_clusters"],
                        group["_metric_value"],
                        marker="o",
                        linewidth=2.0,
                        color=METHOD_COLORS.get(_normalize_method_id(method_name), "#64748b"),
                        label=_display_method_name(method_name),
                    )
                trade_ax.set_xlabel("Matched target clusters")
                trade_ax.set_ylabel(_metric_display_name(metric_key))
                trade_ax.set_title(
                    f"Matched-target trend for {_metric_display_name(metric_key).lower()}"
                )
                trade_ax.legend(frameon=False, loc="best")
            else:
                scatter_source = ranked.copy()
                scatter_source["silhouette"] = pd.to_numeric(
                    scatter_source.get("silhouette"), errors="coerce"
                )
                scatter_source["nmi"] = pd.to_numeric(scatter_source.get("nmi"), errors="coerce")
                scatter_source = scatter_source.dropna(subset=["silhouette", "nmi"]).copy()
                if scatter_source.empty:
                    _draw_empty_axis(trade_ax, "Tradeoff / target trend", "Silhouette and NMI are not both available for the current selection.")
                else:
                    for method_name, group in scatter_source.groupby("method", sort=False):
                        trade_ax.scatter(
                            group["silhouette"],
                            group["nmi"],
                            s=42 + (pd.to_numeric(group.get("n_clusters"), errors="coerce").fillna(2.0) * 2.0),
                            alpha=0.60 if len(group) > 25 else 0.80,
                            c=METHOD_COLORS.get(_normalize_method_id(method_name), "#64748b"),
                            edgecolors="white",
                            linewidths=0.35,
                            label=_display_method_name(method_name),
                        )
                    for _, row in ranked.head(min(5, len(ranked))).iterrows():
                        if pd.notna(row.get("silhouette")) and pd.notna(row.get("nmi")):
                            trade_ax.annotate(
                                shorten(str(row.get("combo", "")), width=18, placeholder="..."),
                                (float(row["silhouette"]), float(row["nmi"])),
                                textcoords="offset points",
                                xytext=(5, 5),
                                fontsize=8,
                            )
                    trade_ax.set_xlabel("Silhouette")
                    trade_ax.set_ylabel("NMI")
                    trade_ax.set_title("Tradeoff map: semantic alignment vs cluster separation")
                    trade_ax.legend(frameon=False, loc="best")
            trade_ax.grid(True, linestyle=":", linewidth=0.4)
            trade_canvas.draw_idle()

            table.heading("metric", text=_metric_display_name(metric_key))
            for item in table.get_children():
                table.delete(item)
            row_lookup: Dict[str, Dict[str, object]] = {}
            for rank_idx, (_, row) in enumerate(ranked.head(max(top_rows, 16)).iterrows(), start=1):
                item_id = f"row_{rank_idx}"
                row_lookup[item_id] = row.to_dict()
                table.insert(
                    "",
                    "end",
                    iid=item_id,
                    values=(
                        rank_idx,
                        _display_method_name(row.get("method")),
                        row.get("combo", ""),
                        _display_preprocess_mode(row.get("preprocess_mode")),
                        _format_metric_value("n_clusters", row.get("n_clusters")),
                        _format_metric_value(metric_key, row.get("_metric_value")),
                        _format_metric_value("silhouette", row.get("silhouette")),
                        _format_metric_value("nmi", row.get("nmi")),
                        _format_metric_value("stability_ari", row.get("stability_ari")),
                        _format_metric_value("coverage", row.get("coverage")),
                    ),
                )
            dashboard_state["rows"] = row_lookup
            if row_lookup:
                first_row = next(iter(row_lookup))
                table.selection_set(first_row)
                table.focus(first_row)
                update_detail()

            summary = bundle.get("dataset_summary", {})
            top_row = ranked.iloc[0].to_dict()
            summary_var.set(
                f"{Path(bundle['dir']).name} | {summary.get('n_songs', '--')} songs | "
                f"{summary.get('n_representations', '--')} representations | "
                f"{len(ranked)} visible rows | leader: {_display_method_name(top_row.get('method'))} "
                f"{top_row.get('combo')} ({_format_metric_value(metric_key, top_row.get('_metric_value'))})"
            )
            if method_id == "vade":
                method_note_var.set(
                    "VaDE is not part of the current thesis benchmark artifacts, so the dashboard compares KMeans, GMM, and HDBSCAN only."
                )
            else:
                current_method_rows = ranked[
                    ranked["method"].map(_normalize_method_id) == method_id
                ]
                if current_method_rows.empty:
                    method_note_var.set(
                        f"{_display_method_name(method_id)} has no visible rows under the current benchmark filters."
                    )
                else:
                    best_method_row = current_method_rows.iloc[0].to_dict()
                    method_note_var.set(
                        f"Current method callout: {_display_method_name(method_id)} is best here with "
                        f"{best_method_row.get('combo')} / {_display_preprocess_mode(best_method_row.get('preprocess_mode'))} "
                        f"({_format_param_summary(best_method_row)}), "
                        f"{_metric_display_name(metric_key)} "
                        f"{_format_metric_value(metric_key, best_method_row.get('_metric_value'))}."
                    )

        def load_run(*_args) -> None:
            selected_path = next((path for path in benchmark_runs if path.name == run_var.get()), None)
            if selected_path is None:
                dashboard_state["bundle"] = None
                benchmark_status_var.set("Benchmark dashboard: no thesis benchmark run selected")
                refresh_dashboard()
                return
            cache_key = str(selected_path)
            if cache_key not in bundle_cache:
                bundle_cache[cache_key] = _load_benchmark_bundle(selected_path)
            dashboard_state["bundle"] = bundle_cache[cache_key]
            benchmark_status_var.set(f"Benchmark dashboard: {selected_path.name}")
            refresh_dashboard()

        for control_var in (view_var, metric_var, method_var, preprocess_var, target_var, top_rows_var):
            control_var.trace_add("write", refresh_dashboard)

        table.bind("<<TreeviewSelect>>", update_detail)
        run_combo.bind("<<ComboboxSelected>>", load_run)

        def close_dashboard() -> None:
            benchmark_window_state["window"] = None
            dashboard.destroy()

        dashboard.protocol("WM_DELETE_WINDOW", close_dashboard)
        if benchmark_runs:
            load_run()
        else:
            summary_var.set("No thesis benchmark directory was found under output/metrics.")
            method_note_var.set("Run the thesis benchmark first, or pass a benchmark directory explicitly.")
            _draw_empty_axis(lead_ax, "Leaderboard", "No thesis benchmark directory found.")
            _draw_empty_axis(trade_ax, "Tradeoff / target trend", "No thesis benchmark directory found.")
            lead_canvas.draw_idle()
            trade_canvas.draw_idle()

    benchmark_button.config(command=open_benchmark_dashboard)

    def switch_algorithm(*_args) -> None:
        if method_selector is None:
            return
        selected_method_id = method_label_to_id.get(method_selector_var.get())
        if not selected_method_id or selected_method_id == method_id:
            return

        command = [
            sys.executable,
            str(project_root / "scripts" / "run_ui.py"),
            "--method",
            selected_method_id,
            "--audio-dir",
            audio_folder,
        ]
        if benchmark_dir:
            command.extend(["--benchmark-dir", str(benchmark_dir)])

        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            os.execv(sys.executable, command)
        except Exception as exc:
            messagebox.showerror(
                "Algorithm Switch Failed",
                f"Could not reopen the UI for {selected_method_id}: {exc}",
            )

    if method_selector is not None:
        method_selector.bind("<<ComboboxSelected>>", switch_algorithm)

    def on_closing():
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the interactive recommendation UI using the latest "
            "benchmark-linked recommended-production snapshot."
        )
    )
    parser.add_argument(
        "--method",
        choices=UI_METHOD_CHOICES,
        default="auto",
        help=(
            "Which clustering method snapshot to open. 'auto' uses the "
            "top-ranked method from the latest recommended-production summary."
        ),
    )
    parser.add_argument(
        "--audio-dir",
        default="audio_files",
        help="Directory containing the playable audio files shown by the UI.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=None,
        help=(
            "Optional explicit benchmark directory. If omitted, the newest "
            "thesis benchmark folder under output/metrics is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    bundle = resolve_latest_benchmark_ui_bundle(
        method_id=args.method,
        benchmark_dir=args.benchmark_dir,
    )
    print(
        "Launching UI from benchmark snapshot -> "
        f"benchmark={(bundle['benchmark_dir'].name if bundle['benchmark_dir'] is not None else 'not_available')}, "
        f"method={bundle['method_id']}, "
        f"source={bundle['run_manifest_path']}"
    )
    launch_ui(
        bundle["df"],
        bundle["coords"],
        bundle["labels"],
        audio_dir=args.audio_dir,
        clustering_method=bundle["method_display_name"],
        retrieval_features=bundle["retrieval_features"],
        assignment_confidence=bundle["assignment_confidence"],
        posterior_probabilities=bundle["posterior_probabilities"],
        benchmark_dir=(
            str(bundle["benchmark_dir"]) if bundle["benchmark_dir"] is not None else None
        ),
        available_method_ids=bundle["available_methods"],
    )


if __name__ == "__main__":
    main()
