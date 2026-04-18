from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import unicodedata
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from clustering.shared import (
    DEFAULT_FEATURES_DIR,
    default_cluster_results_dir,
    feature_source_key,
    prepare_dataset,
)


DEEZER_TRACK_API = "https://api.deezer.com/track/{track_id}"
DEEZER_API_TIMEOUT_SEC = 5.0


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA_CSV = WORKSPACE_DIR / "data" / "msd_deezer_matches.csv"
DEFAULT_HTML_PATH = WORKSPACE_DIR / "interface" / "cluster_explorer.html"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RECOMMENDATIONS = 5
DISPLAY_ALGORITHMS = ("kmeans", "gmm", "hdbscan")

# Built-in feature sources the dashboard auto-discovers when no --feature-source
# flags are passed. Order here controls the default-source preference order.
AUTO_DISCOVERY_SOURCES: tuple[tuple[str, str, Path], ...] = (
    ("features", "Audio Features", WORKSPACE_DIR / "features"),
    ("pretrained_embeddings", "Pretrained Embeddings", WORKSPACE_DIR / "pretrained_embeddings"),
)
KNOWN_SOURCE_LABELS: dict[str, str] = {
    "features": "Audio Features",
    "pretrained_embeddings": "Pretrained Embeddings",
}


@dataclass
class ExplorerSource:
    source_id: str
    label: str
    state_bytes: bytes
    songs_by_id: dict[str, dict[str, Any]]
    song_ids: list[str]
    song_index_by_id: dict[str, int]
    list_order_song_ids: list[str]
    scaled_matrix: np.ndarray
    neighbors_model: NearestNeighbors
    cluster_labels: dict[str, np.ndarray]
    available_algorithms: list[str]
    default_algorithm: str


@dataclass
class ExplorerState:
    html_bytes: bytes
    sources: dict[str, ExplorerSource]
    default_source_id: str
    sources_manifest: list[dict[str, str]]
    audio_paths_by_id: dict[str, Path]
    recommendation_count: int
    deezer_track_ids_by_id: dict[str, str] = field(default_factory=dict)
    deezer_preview_cache: dict[str, str | None] = field(default_factory=dict)
    preview_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    prefer_deezer_previews: bool = True


def _clean_deezer_track_id(value: Any) -> str:
    """Normalize a Deezer track ID to a bare digit string ("" if invalid)."""
    text = _clean_text(value)
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return ""


def _resolve_deezer_preview(track_id: str) -> str | None:
    """Look up a track's preview URL via the public Deezer API.

    Returns None on any failure (network error, non-200, missing field).
    The result is meant to be cached by the caller.
    """
    if not track_id:
        return None
    url = DEEZER_TRACK_API.format(track_id=track_id)
    request = urllib.request.Request(
        url, headers={"User-Agent": "cluster-explorer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=DEEZER_API_TIMEOUT_SEC) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    preview = payload.get("preview")
    if isinstance(preview, str) and preview.startswith("http"):
        return preview
    return None


def _safe_int(value: Any) -> int | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if np.isfinite(numeric) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the clustering explorer interface.")
    parser.add_argument(
        "--feature-source",
        action="append",
        default=None,
        dest="feature_sources",
        metavar="ID=PATH",
        help=(
            "Register a feature source for the dashboard, repeatable. "
            "Format: 'id=path' (e.g. 'features=./features', "
            "'pretrained_embeddings=./pretrained_embeddings'). "
            "A bare path is accepted and its id is derived from the directory name. "
            "When omitted, both built-in sources are auto-discovered."
        ),
    )
    parser.add_argument(
        "--features-path",
        default=None,
        help=(
            "Legacy single-source shortcut: path to a features directory or a "
            "feature summary CSV. Ignored when --feature-source is given."
        ),
    )
    parser.add_argument("--cluster-results-dir", default=None, help="Override the clustering outputs directory for the single-source --features-path mode. Ignored when --feature-source is given.")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV), help="Optional metadata CSV generated by the Deezer pipeline.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port for the local server.")
    parser.add_argument("--limit", type=int, help="Optional limit for faster smoke tests.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--recommendations", type=int, default=DEFAULT_RECOMMENDATIONS, help="How many recommendations to return for the selected song.")
    parser.add_argument(
        "--no-deezer-previews", dest="prefer_deezer_previews",
        action="store_false", default=True,
        help=(
            "Always serve local audio files instead of streaming previews "
            "from the Deezer API. Default: prefer Deezer and fall back to "
            "the local file only if the API call fails."
        ),
    )
    return parser.parse_args()


@dataclass
class FeatureSourceSpec:
    source_id: str
    label: str
    features_path: Path
    cluster_results_dir: Path


def _label_for_source_id(source_id: str) -> str:
    if source_id in KNOWN_SOURCE_LABELS:
        return KNOWN_SOURCE_LABELS[source_id]
    return source_id.replace("_", " ").replace("-", " ").title()


def _spec_for_path(source_id: str | None, path: Path, label: str | None = None) -> FeatureSourceSpec:
    resolved_path = path.resolve()
    derived_id = source_id or feature_source_key(resolved_path)
    return FeatureSourceSpec(
        source_id=derived_id,
        label=label or _label_for_source_id(derived_id),
        features_path=resolved_path,
        cluster_results_dir=Path(default_cluster_results_dir(resolved_path)).resolve(),
    )


def _parse_feature_source_flag(raw: str) -> FeatureSourceSpec:
    text = raw.strip()
    if not text:
        raise ValueError("Empty --feature-source value.")
    if "=" in text:
        source_id, path_text = text.split("=", 1)
        source_id = source_id.strip()
        path_text = path_text.strip()
        if not source_id or not path_text:
            raise ValueError(f"Invalid --feature-source value: {raw!r}")
        return _spec_for_path(source_id, Path(path_text))
    return _spec_for_path(None, Path(text))


def _resolve_feature_source_specs(args: argparse.Namespace) -> list[FeatureSourceSpec]:
    if args.feature_sources:
        specs = [_parse_feature_source_flag(value) for value in args.feature_sources]
        # De-duplicate by id while preserving user-provided order.
        seen: set[str] = set()
        unique: list[FeatureSourceSpec] = []
        for spec in specs:
            if spec.source_id in seen:
                continue
            seen.add(spec.source_id)
            unique.append(spec)
        return unique

    if args.features_path:
        spec = _spec_for_path(None, Path(args.features_path))
        if args.cluster_results_dir:
            spec.cluster_results_dir = Path(args.cluster_results_dir).resolve()
        return [spec]

    discovered: list[FeatureSourceSpec] = []
    for source_id, label, path in AUTO_DISCOVERY_SOURCES:
        if not path.exists():
            continue
        spec = _spec_for_path(source_id, path, label=label)
        if not spec.cluster_results_dir.exists():
            continue
        discovered.append(spec)

    if discovered:
        return discovered

    # Fall back to the legacy default so helpful errors surface downstream.
    return [_spec_for_path(None, Path(DEFAULT_FEATURES_DIR))]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _clean_year(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        year = int(float(text))
    except ValueError:
        return text
    return "" if year <= 0 else str(year)


def _ascii_header_filename(filename: str) -> str:
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe = ascii_only.replace("\\", "_").replace('"', "_").strip()
    return safe or "audio"


def _content_disposition_value(filename: str) -> str:
    fallback = _ascii_header_filename(filename)
    encoded = quote(filename, safe="")
    return f"inline; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def _song_id_from_row(row: pd.Series, fallback_index: int | None = None) -> str:
    msd_track_id = _clean_text(row.get("msd_track_id"))
    if msd_track_id:
        return msd_track_id

    deezer_track_id = _clean_text(row.get("deezer_track_id"))
    if deezer_track_id:
        return f"deezer_{deezer_track_id}"

    raw_feature_path = _clean_text(row.get("raw_feature_path"))
    if raw_feature_path:
        return Path(raw_feature_path).stem

    audio_path = _clean_text(row.get("audio_path"))
    if audio_path:
        return Path(audio_path).stem

    if fallback_index is None:
        raise ValueError("Could not derive a stable song id for a row.")
    return f"song_{fallback_index:05d}"


def _first_non_blank(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _parse_artist_title_from_filename(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    if "[" in stem:
        stem = stem.split("[", 1)[0].strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", stem.strip()


def _normalize_cluster_label(value: Any) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _normalize_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _load_metadata_frame(metadata_csv: Path) -> pd.DataFrame | None:
    if not metadata_csv.exists():
        return None

    metadata = pd.read_csv(metadata_csv, low_memory=False)
    metadata["song_id"] = [
        _song_id_from_row(row, index)
        for index, (_, row) in enumerate(metadata.iterrows())
    ]
    keep_columns = [
        "song_id",
        "msd_title",
        "msd_artist_name",
        "msd_release",
        "msd_year",
        "msd_duration",
        "deezer_title",
        "deezer_artist",
        "deezer_album",
        "deezer_duration",
        "deezer_link",
        "deezer_match_score",
        "deezer_match_status",
        "deezer_download_status",
    ]
    keep_columns = [column for column in keep_columns if column in metadata.columns]
    return metadata[keep_columns].drop_duplicates(subset=["song_id"])


def _load_algorithm_assignments(
    algorithm: str,
    cluster_results_dir: Path,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    algorithm_dir = cluster_results_dir / algorithm
    assignments_path = algorithm_dir / "cluster_assignments.csv"
    metadata_path = algorithm_dir / "run_metadata.json"

    if not assignments_path.exists():
        return None, {}

    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments["song_id"] = [
        _song_id_from_row(row, index)
        for index, (_, row) in enumerate(assignments.iterrows())
    ]
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return assignments.drop_duplicates(subset=["song_id"]), metadata


def _build_algorithm_detail(
    algorithm: str,
    cluster_results_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    algorithm_dir = cluster_results_dir / algorithm
    cluster_summary_path = algorithm_dir / "cluster_summary.csv"
    selection_metrics_path = algorithm_dir / "selection_metrics.csv"

    cluster_distribution: list[dict[str, Any]] = []
    if cluster_summary_path.exists():
        cluster_summary = pd.read_csv(cluster_summary_path, low_memory=False)
        for _, row in cluster_summary.iterrows():
            label = _safe_int(row.get("cluster_label"))
            size = _safe_int(row.get("size")) or 0
            is_noise = algorithm == "hdbscan" and label == -1
            cluster_distribution.append(
                {
                    "label": label,
                    "size": size,
                    "isNoise": is_noise,
                }
            )

    active_clusters = [item for item in cluster_distribution if not item["isNoise"]]
    active_clusters.sort(key=lambda item: (-item["size"], item["label"] if item["label"] is not None else 0))
    cluster_distribution.sort(
        key=lambda item: (
            item["isNoise"],
            -item["size"],
            item["label"] if item["label"] is not None else 0,
        )
    )

    samples = _safe_int(metadata.get("samples")) or sum(item["size"] for item in cluster_distribution)
    noise_points = _safe_int(metadata.get("noise_points"))
    if noise_points is None:
        noise_points = next((item["size"] for item in cluster_distribution if item["isNoise"]), 0)

    clustered_songs = sum(item["size"] for item in active_clusters)
    cluster_count = _safe_int(metadata.get("cluster_count")) or len(active_clusters)
    cluster_sizes = [item["size"] for item in active_clusters]
    average_cluster_size = float(np.mean(cluster_sizes)) if cluster_sizes else None
    median_cluster_size = float(np.median(cluster_sizes)) if cluster_sizes else None
    largest_cluster = active_clusters[0] if active_clusters else None
    smallest_cluster = min(active_clusters, key=lambda item: (item["size"], item["label"])) if active_clusters else None

    for item in cluster_distribution:
        item["ratio"] = (item["size"] / samples) if samples else 0.0

    top_metrics: dict[str, Any] = {}
    if selection_metrics_path.exists():
        selection_metrics = pd.read_csv(selection_metrics_path, low_memory=False)
        if not selection_metrics.empty:
            top_row = selection_metrics.iloc[0].to_dict()
            normalized_top_row: dict[str, Any] = {}
            for key, value in top_row.items():
                numeric = _safe_float(value)
                normalized_top_row[key] = numeric if numeric is not None else _clean_text(value)
            top_metrics = normalized_top_row

    metric_cards: list[dict[str, str]] = []
    if algorithm == "kmeans":
        metric_cards = [
            {"label": "Best silhouette", "value": f"{(_safe_float(metadata.get('best_silhouette_score')) or 0.0):.4f}"},
            {"label": "Inertia", "value": f"{(_safe_float(metadata.get('inertia')) or 0.0):,.0f}"},
            {"label": "PCA variance", "value": f"{((_safe_float(metadata.get('pca_explained_variance_ratio')) or 0.0) * 100):.1f}%"},
        ]
    elif algorithm == "gmm":
        metric_cards = [
            {"label": "Covariance", "value": _clean_text(metadata.get("covariance_type")).upper() or "n/a"},
            {"label": "Best BIC", "value": f"{(_safe_float(metadata.get('best_bic')) or 0.0):,.0f}"},
            {"label": "Mean confidence", "value": f"{(_safe_float(top_metrics.get('mean_membership_confidence')) or 0.0):.3f}"},
        ]
    elif algorithm == "hdbscan":
        min_cluster_size = _safe_int(metadata.get("min_cluster_size"))
        min_samples = metadata.get("min_samples")
        metric_cards = [
            {"label": "Min cluster size", "value": str(min_cluster_size) if min_cluster_size is not None else "n/a"},
            {"label": "Min samples", "value": str(min_samples) if min_samples is not None else "None"},
            {"label": "Validity", "value": f"{(_safe_float(metadata.get('validity_index')) or 0.0):.4f}"},
        ]

    return {
        "clusterCount": cluster_count,
        "samples": samples,
        "clusteredSongs": clustered_songs,
        "noisePoints": noise_points,
        "clusteredRatio": (clustered_songs / samples) if samples else 0.0,
        "averageClusterSize": average_cluster_size,
        "medianClusterSize": median_cluster_size,
        "largestCluster": largest_cluster,
        "smallestCluster": smallest_cluster,
        "metricCards": metric_cards,
        "topMetrics": top_metrics,
        "clusterDistribution": cluster_distribution,
    }


def _build_song_table(
    features_path: Path,
    metadata_csv: Path,
    cluster_results_dir: Path,
    limit: int | None,
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset = prepare_dataset(features_path=features_path, limit=limit)
    base_frame = dataset.metadata.copy().reset_index(drop=True)
    base_frame["song_id"] = [
        _song_id_from_row(row, index)
        for index, (_, row) in enumerate(base_frame.iterrows())
    ]

    metadata_frame = _load_metadata_frame(metadata_csv)
    if metadata_frame is not None:
        base_frame = base_frame.merge(metadata_frame, on="song_id", how="left")

    parsed_pairs = base_frame["file"].fillna("").map(_parse_artist_title_from_filename)
    base_frame["parsed_artist"] = parsed_pairs.map(lambda value: value[0])
    base_frame["parsed_title"] = parsed_pairs.map(lambda value: value[1])

    base_frame["display_title"] = [
        _first_non_blank(deezer_title, msd_title, parsed_title, file_name)
        for deezer_title, msd_title, parsed_title, file_name in zip(
            base_frame.get("deezer_title", pd.Series(index=base_frame.index, dtype="object")),
            base_frame.get("msd_title", pd.Series(index=base_frame.index, dtype="object")),
            base_frame["parsed_title"],
            base_frame["file"],
        )
    ]
    base_frame["display_artist"] = [
        _first_non_blank(deezer_artist, msd_artist_name, parsed_artist, "Unknown Artist")
        for deezer_artist, msd_artist_name, parsed_artist in zip(
            base_frame.get("deezer_artist", pd.Series(index=base_frame.index, dtype="object")),
            base_frame.get("msd_artist_name", pd.Series(index=base_frame.index, dtype="object")),
            base_frame["parsed_artist"],
        )
    ]
    base_frame["display_album"] = [
        _first_non_blank(deezer_album, msd_release)
        for deezer_album, msd_release in zip(
            base_frame.get("deezer_album", pd.Series(index=base_frame.index, dtype="object")),
            base_frame.get("msd_release", pd.Series(index=base_frame.index, dtype="object")),
        )
    ]
    base_frame["display_year"] = [
        _clean_year(value)
        for value in base_frame.get("msd_year", pd.Series(index=base_frame.index, dtype="object"))
    ]

    projection = PCA(n_components=2, svd_solver="full").fit_transform(dataset.scaled_matrix)
    base_frame["projection_x"] = projection[:, 0]
    base_frame["projection_y"] = projection[:, 1]

    algorithm_metadata: dict[str, dict[str, Any]] = {}
    algorithm_details: dict[str, dict[str, Any]] = {}
    available_algorithms: list[str] = []

    for algorithm in DISPLAY_ALGORITHMS:
        assignments, metadata = _load_algorithm_assignments(algorithm, cluster_results_dir)
        if assignments is None:
            continue

        available_algorithms.append(algorithm)
        algorithm_metadata[algorithm] = metadata
        algorithm_details[algorithm] = _build_algorithm_detail(algorithm, cluster_results_dir, metadata)

        base_frame = base_frame.merge(
            assignments[["song_id", "cluster_label", "cluster_size"]].rename(
                columns={
                    "cluster_label": f"{algorithm}_cluster_label",
                    "cluster_size": f"{algorithm}_cluster_size",
                }
            ),
            on="song_id",
            how="left",
        )

        if algorithm == "kmeans":
            if "distance_to_centroid" in assignments.columns:
                base_frame = base_frame.merge(
                    assignments[["song_id", "distance_to_centroid"]].rename(
                        columns={"distance_to_centroid": "kmeans_distance_to_centroid"}
                    ),
                    on="song_id",
                    how="left",
                )
        elif algorithm == "gmm":
            metric_columns = [column for column in ("membership_confidence", "membership_entropy") if column in assignments.columns]
            if metric_columns:
                renamed = {column: f"gmm_{column}" for column in metric_columns}
                base_frame = base_frame.merge(
                    assignments[["song_id", *metric_columns]].rename(columns=renamed),
                    on="song_id",
                    how="left",
                )
        elif algorithm == "hdbscan":
            metric_columns = [column for column in ("membership_probability", "outlier_score", "is_noise") if column in assignments.columns]
            if metric_columns:
                renamed = {column: f"hdbscan_{column}" for column in metric_columns}
                base_frame = base_frame.merge(
                    assignments[["song_id", *metric_columns]].rename(columns=renamed),
                    on="song_id",
                    how="left",
                )

    return base_frame, dataset.scaled_matrix, available_algorithms, algorithm_metadata, algorithm_details


def _cluster_info_for_song(row: pd.Series, available_algorithms: list[str]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for algorithm in available_algorithms:
        cluster_label = _normalize_cluster_label(row.get(f"{algorithm}_cluster_label"))
        cluster_size = _normalize_cluster_label(row.get(f"{algorithm}_cluster_size"))
        algorithm_payload: dict[str, Any] = {
            "label": cluster_label,
            "clusterSize": cluster_size,
        }

        if algorithm == "kmeans":
            algorithm_payload["distanceToCentroid"] = _normalize_float(row.get("kmeans_distance_to_centroid"))
        elif algorithm == "gmm":
            algorithm_payload["membershipConfidence"] = _normalize_float(row.get("gmm_membership_confidence"))
            algorithm_payload["membershipEntropy"] = _normalize_float(row.get("gmm_membership_entropy"))
        elif algorithm == "hdbscan":
            algorithm_payload["membershipProbability"] = _normalize_float(row.get("hdbscan_membership_probability"))
            algorithm_payload["outlierScore"] = _normalize_float(row.get("hdbscan_outlier_score"))
            algorithm_payload["isNoise"] = _normalize_bool(row.get("hdbscan_is_noise"))

        payload[algorithm] = algorithm_payload
    return payload


def _build_source(
    spec: FeatureSourceSpec,
    metadata_csv: Path,
    args: argparse.Namespace,
    sources_manifest: list[dict[str, str]],
    default_source_id: str,
) -> tuple[ExplorerSource, dict[str, Path], dict[str, str]]:
    song_table, scaled_matrix, available_algorithms, algorithm_metadata, algorithm_details = _build_song_table(
        features_path=spec.features_path,
        metadata_csv=metadata_csv,
        cluster_results_dir=spec.cluster_results_dir,
        limit=args.limit,
    )
    if song_table.empty:
        raise ValueError(f"No songs are available for feature source {spec.source_id!r} at {spec.features_path}.")
    if not available_algorithms:
        raise FileNotFoundError(
            f"No clustering outputs were found for source {spec.source_id!r} under {spec.cluster_results_dir}"
        )

    song_ids = song_table["song_id"].tolist()
    song_index_by_id = {song_id: index for index, song_id in enumerate(song_ids)}
    list_order_song_ids = (
        song_table.assign(
            _sort_artist=song_table["display_artist"].str.casefold(),
            _sort_title=song_table["display_title"].str.casefold(),
        )
        .sort_values(by=["_sort_artist", "_sort_title", "song_id"], kind="stable")["song_id"]
        .tolist()
    )

    recommendation_neighbors = min(max(int(args.recommendations) * 20, 60), len(song_table))
    neighbors_model = NearestNeighbors(n_neighbors=recommendation_neighbors, metric="euclidean")
    neighbors_model.fit(scaled_matrix)

    audio_paths_by_id: dict[str, Path] = {}
    deezer_track_ids_by_id: dict[str, str] = {}
    songs_payload: list[dict[str, Any]] = []
    songs_by_id: dict[str, dict[str, Any]] = {}
    cluster_labels: dict[str, np.ndarray] = {}

    for algorithm in available_algorithms:
        labels = song_table[f"{algorithm}_cluster_label"].apply(_normalize_cluster_label).tolist()
        cluster_labels[algorithm] = np.array(
            [np.nan if value is None else float(value) for value in labels], dtype=np.float64
        )

    for _, row in song_table.iterrows():
        song_id = row["song_id"]
        audio_path_text = _clean_text(row.get("audio_path"))
        audio_path = Path(audio_path_text).resolve() if audio_path_text else None
        if audio_path is not None and audio_path.exists():
            audio_paths_by_id[song_id] = audio_path

        deezer_track_id = _clean_deezer_track_id(row.get("deezer_track_id"))
        if deezer_track_id:
            deezer_track_ids_by_id[song_id] = deezer_track_id

        has_local_audio = bool(audio_path is not None and audio_path.exists())
        has_audio = has_local_audio or bool(deezer_track_id)

        song_payload = {
            "id": song_id,
            "fileName": _clean_text(row.get("file")),
            "title": _clean_text(row.get("display_title")),
            "artist": _clean_text(row.get("display_artist")),
            "album": _clean_text(row.get("display_album")),
            "year": _clean_text(row.get("display_year")),
            "msdTitle": _clean_text(row.get("msd_title")),
            "msdArtist": _clean_text(row.get("msd_artist_name")),
            "deezerTitle": _clean_text(row.get("deezer_title")),
            "deezerArtist": _clean_text(row.get("deezer_artist")),
            "deezerLink": _clean_text(row.get("deezer_link")),
            "deezerTrackId": deezer_track_id,
            "audioUrl": f"/audio/{song_id}",
            "hasAudio": has_audio,
            "coords": {
                "x": float(row["projection_x"]),
                "y": float(row["projection_y"]),
            },
            "clusters": _cluster_info_for_song(row, available_algorithms),
        }
        songs_payload.append(song_payload)
        songs_by_id[song_id] = song_payload

    projection_x = song_table["projection_x"].to_numpy(dtype=np.float64)
    projection_y = song_table["projection_y"].to_numpy(dtype=np.float64)
    ui_payload = {
        "summary": {
            "songs": int(len(song_table)),
            "availableAlgorithms": available_algorithms,
            "defaultAlgorithm": available_algorithms[0],
            "activeSource": spec.source_id,
            "availableSources": sources_manifest,
            "defaultSource": default_source_id,
        },
        "projection": {
            "xMin": float(projection_x.min()),
            "xMax": float(projection_x.max()),
            "yMin": float(projection_y.min()),
            "yMax": float(projection_y.max()),
        },
        "algorithms": {
            algorithm: {
                "label": algorithm.upper() if algorithm != "hdbscan" else "HDBSCAN",
                "metadata": algorithm_metadata.get(algorithm, {}),
                "details": algorithm_details.get(algorithm, {}),
            }
            for algorithm in available_algorithms
        },
        "listOrderSongIds": list_order_song_ids,
        "songs": songs_payload,
    }

    source = ExplorerSource(
        source_id=spec.source_id,
        label=spec.label,
        state_bytes=json.dumps(ui_payload, ensure_ascii=False).encode("utf-8"),
        songs_by_id=songs_by_id,
        song_ids=song_ids,
        song_index_by_id=song_index_by_id,
        list_order_song_ids=list_order_song_ids,
        scaled_matrix=scaled_matrix,
        neighbors_model=neighbors_model,
        cluster_labels=cluster_labels,
        available_algorithms=available_algorithms,
        default_algorithm=available_algorithms[0],
    )
    return source, audio_paths_by_id, deezer_track_ids_by_id


def _build_state(args: argparse.Namespace) -> ExplorerState:
    specs = _resolve_feature_source_specs(args)
    metadata_csv = Path(args.metadata_csv).resolve()
    html_path = DEFAULT_HTML_PATH.resolve()

    sources_manifest = [
        {"id": spec.source_id, "label": spec.label}
        for spec in specs
    ]
    default_source_id = specs[0].source_id

    sources: dict[str, ExplorerSource] = {}
    combined_audio_paths: dict[str, Path] = {}
    combined_deezer_ids: dict[str, str] = {}
    build_errors: list[str] = []

    for spec in specs:
        try:
            source, audio_paths, deezer_ids = _build_source(
                spec,
                metadata_csv=metadata_csv,
                args=args,
                sources_manifest=sources_manifest,
                default_source_id=default_source_id,
            )
        except (FileNotFoundError, ValueError) as error:
            build_errors.append(f"[{spec.source_id}] {error}")
            continue

        sources[spec.source_id] = source
        combined_audio_paths.update(audio_paths)
        combined_deezer_ids.update(deezer_ids)

    if not sources:
        details = "\n  - ".join(build_errors) if build_errors else "No feature sources were resolved."
        raise FileNotFoundError(f"Could not initialise any feature source for the dashboard:\n  - {details}")

    # Trim the manifest to sources that actually loaded, while preserving order.
    sources_manifest = [entry for entry in sources_manifest if entry["id"] in sources]
    if default_source_id not in sources:
        default_source_id = sources_manifest[0]["id"]

    # Rebuild every source's cached state_bytes so the manifest reflects only
    # the sources that actually loaded. Cheap: we just re-serialise JSON.
    for source in sources.values():
        payload = json.loads(source.state_bytes.decode("utf-8"))
        payload["summary"]["availableSources"] = sources_manifest
        payload["summary"]["defaultSource"] = default_source_id
        source.state_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    return ExplorerState(
        html_bytes=html_path.read_bytes(),
        sources=sources,
        default_source_id=default_source_id,
        sources_manifest=sources_manifest,
        audio_paths_by_id=combined_audio_paths,
        recommendation_count=max(1, int(args.recommendations)),
        deezer_track_ids_by_id=combined_deezer_ids,
        prefer_deezer_previews=bool(getattr(args, "prefer_deezer_previews", True)),
    )


def _agreement_count(source: ExplorerSource, anchor_index: int, candidate_index: int) -> int:
    agreement_count = 0
    for algorithm in source.available_algorithms:
        anchor_label = source.cluster_labels[algorithm][anchor_index]
        candidate_label = source.cluster_labels[algorithm][candidate_index]
        if not np.isfinite(anchor_label) or not np.isfinite(candidate_label):
            continue
        if int(anchor_label) != int(candidate_label):
            continue
        if algorithm == "hdbscan" and int(anchor_label) == -1:
            continue
        agreement_count += 1
    return agreement_count


def _shares_active_cluster(
    source: ExplorerSource, algorithm: str, anchor_index: int, candidate_index: int
) -> bool:
    if algorithm not in source.cluster_labels:
        return False

    anchor_label = source.cluster_labels[algorithm][anchor_index]
    candidate_label = source.cluster_labels[algorithm][candidate_index]
    if not np.isfinite(anchor_label) or not np.isfinite(candidate_label):
        return False
    if int(anchor_label) != int(candidate_label):
        return False
    if algorithm == "hdbscan" and int(anchor_label) == -1:
        return False
    return True


def _build_reason_text(
    active_algorithm: str,
    same_active_cluster: bool,
    agreement_count: int,
) -> str:
    parts: list[str] = []
    if same_active_cluster:
        label = active_algorithm.upper() if active_algorithm != "hdbscan" else "HDBSCAN"
        parts.append(f"same {label} cluster")
    if agreement_count > 0:
        noun = "algorithm" if agreement_count == 1 else "algorithms"
        parts.append(f"{agreement_count} clustering {noun} agree")
    parts.append("close in feature space")
    return ", ".join(parts)


def _recommendations_response(
    state: ExplorerState, source: ExplorerSource, song_id: str, algorithm: str
) -> dict[str, Any]:
    if song_id not in source.song_index_by_id:
        raise KeyError(song_id)

    if algorithm not in source.available_algorithms:
        algorithm = source.default_algorithm

    anchor_index = source.song_index_by_id[song_id]
    distances, indices = source.neighbors_model.kneighbors(
        source.scaled_matrix[anchor_index : anchor_index + 1]
    )

    same_cluster_matches: list[dict[str, Any]] = []
    nearest_matches: list[dict[str, Any]] = []

    for distance, candidate_index in zip(distances[0], indices[0]):
        if candidate_index == anchor_index:
            continue

        candidate_id = source.song_ids[int(candidate_index)]
        candidate_song = source.songs_by_id[candidate_id]
        same_active_cluster = _shares_active_cluster(source, algorithm, anchor_index, int(candidate_index))
        agreement_count = _agreement_count(source, anchor_index, int(candidate_index))
        candidate_payload = {
            **candidate_song,
            "distance": round(float(distance), 6),
            "sameActiveCluster": same_active_cluster,
            "agreementCount": agreement_count,
            "reason": _build_reason_text(algorithm, same_active_cluster, agreement_count),
        }
        if same_active_cluster:
            same_cluster_matches.append(candidate_payload)
        else:
            nearest_matches.append(candidate_payload)

    same_cluster_matches.sort(key=lambda item: (item["distance"], -item["agreementCount"], item["title"]))
    nearest_matches.sort(key=lambda item: (-item["agreementCount"], item["distance"], item["title"]))
    recommendations = same_cluster_matches[: state.recommendation_count]

    if len(recommendations) < state.recommendation_count:
        remaining = state.recommendation_count - len(recommendations)
        recommendations.extend(nearest_matches[:remaining])

    return {
        "sourceId": source.source_id,
        "songId": song_id,
        "algorithm": algorithm,
        "selectedSong": source.songs_by_id[song_id],
        "recommendations": recommendations,
    }


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int] | None:
    if not range_header:
        return None
    if not range_header.startswith("bytes="):
        raise ValueError("Unsupported range unit.")

    requested_range = range_header.split("=", 1)[1].split(",", 1)[0].strip()
    if "-" not in requested_range:
        raise ValueError("Malformed range header.")

    start_text, end_text = requested_range.split("-", 1)
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix byte range.")
        if suffix_length >= file_size:
            return 0, file_size - 1
        return file_size - suffix_length, file_size - 1

    start = int(start_text)
    end = file_size - 1 if not end_text else int(end_text)
    if start < 0 or start >= file_size:
        raise ValueError("Range start is outside the file.")
    if end < start:
        raise ValueError("Range end is before the start.")
    return start, min(end, file_size - 1)


def _choose_server(host: str, preferred_port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    for offset in range(0, 20):
        try:
            return ThreadingHTTPServer((host, preferred_port + offset), handler)
        except OSError:
            continue
    raise OSError(f"Could not bind the cluster explorer on {host}:{preferred_port}-{preferred_port + 19}")


def _build_handler(state: ExplorerState) -> type[BaseHTTPRequestHandler]:
    class ClusterExplorerHandler(BaseHTTPRequestHandler):
        explorer_state = state
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_headers(
            self,
            *,
            content_type: str,
            content_length: int,
            status: HTTPStatus = HTTPStatus.OK,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()

        def _write_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            extra_headers: dict[str, str] | None = None,
            head_only: bool = False,
        ) -> None:
            self._send_headers(
                content_type=content_type,
                content_length=len(payload),
                status=status,
                extra_headers=extra_headers,
            )
            if head_only:
                return
            self.wfile.write(payload)

        def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            self._write_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _get_cached_deezer_preview(self, song_id: str) -> str | None:
            """Return a cached preview URL, resolving it via the Deezer API if needed."""
            state = self.explorer_state
            with state.preview_cache_lock:
                if song_id in state.deezer_preview_cache:
                    return state.deezer_preview_cache[song_id]

            track_id = state.deezer_track_ids_by_id.get(song_id, "")
            preview_url = _resolve_deezer_preview(track_id) if track_id else None

            with state.preview_cache_lock:
                # Cache both hits and misses; misses expire when the server restarts.
                state.deezer_preview_cache[song_id] = preview_url
            return preview_url

        def _redirect_to(self, url: str) -> None:
            self.send_response(HTTPStatus.FOUND.value)
            self.send_header("Location", url)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _serve_audio(self, song_id: str, head_only: bool = False) -> None:
            state = self.explorer_state

            # Prefer streaming from the Deezer CDN so the dashboard works on
            # any machine, not just the one that did the downloads.
            if state.prefer_deezer_previews and song_id in state.deezer_track_ids_by_id:
                preview_url = self._get_cached_deezer_preview(song_id)
                if preview_url:
                    self._redirect_to(preview_url)
                    return

            audio_path = self.explorer_state.audio_paths_by_id.get(song_id)
            if audio_path is None or not audio_path.exists():
                self._write_json({"error": f"No audio found for {song_id}"}, status=HTTPStatus.NOT_FOUND)
                return

            file_size = audio_path.stat().st_size
            content_type = _guess_mime_type(audio_path)
            try:
                byte_range = _parse_range_header(self.headers.get("Range"), file_size)
            except ValueError:
                self._send_headers(
                    content_type=content_type,
                    content_length=0,
                    status=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    extra_headers={
                        "Accept-Ranges": "bytes",
                        "Content-Range": f"bytes */{file_size}",
                    },
                )
                return

            start = 0
            end = file_size - 1
            status = HTTPStatus.OK
            extra_headers = {
                "Accept-Ranges": "bytes",
                "Content-Disposition": _content_disposition_value(audio_path.name),
            }
            if byte_range is not None:
                start, end = byte_range
                status = HTTPStatus.PARTIAL_CONTENT
                extra_headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

            content_length = end - start + 1
            self._send_headers(
                content_type=content_type,
                content_length=content_length,
                status=status,
                extra_headers=extra_headers,
            )
            if head_only:
                return

            remaining = content_length
            with audio_path.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def _resolve_source(self, query: dict[str, list[str]]) -> ExplorerSource:
            requested = (query.get("source", [""])[0] or "").strip()
            state = self.explorer_state
            if requested and requested in state.sources:
                return state.sources[requested]
            return state.sources[state.default_source_id]

        def _handle_request(self, head_only: bool = False) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._write_bytes(self.explorer_state.html_bytes, "text/html; charset=utf-8", head_only=head_only)
                return

            if parsed.path == "/api/sources":
                payload = {
                    "sources": self.explorer_state.sources_manifest,
                    "defaultSource": self.explorer_state.default_source_id,
                }
                self._write_bytes(
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                    head_only=head_only,
                )
                return

            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                source = self._resolve_source(query)
                self._write_bytes(source.state_bytes, "application/json; charset=utf-8", head_only=head_only)
                return

            if parsed.path == "/api/recommendations":
                if head_only:
                    self._write_bytes(b"", "application/json; charset=utf-8", head_only=True)
                    return
                query = parse_qs(parsed.query)
                source = self._resolve_source(query)
                song_id = query.get("song_id", [""])[0]
                algorithm = query.get("algorithm", [source.default_algorithm])[0]
                if not song_id:
                    self._write_json({"error": "song_id is required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                try:
                    payload = _recommendations_response(self.explorer_state, source, song_id, algorithm)
                except KeyError:
                    self._write_json({"error": f"Unknown song_id: {song_id}"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._write_json(payload)
                return

            if parsed.path.startswith("/audio/"):
                song_id = parsed.path.split("/audio/", 1)[1]
                self._serve_audio(song_id, head_only=head_only)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT.value)
                self.end_headers()
                return

            self._write_json({"error": f"Unknown route: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            self._handle_request(head_only=False)

        def do_HEAD(self) -> None:
            self._handle_request(head_only=True)

    return ClusterExplorerHandler


def main() -> None:
    args = parse_args()
    state = _build_state(args)
    handler = _build_handler(state)
    server = _choose_server(args.host, args.port, handler)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}"

    print(f"Cluster explorer ready at {url}")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
