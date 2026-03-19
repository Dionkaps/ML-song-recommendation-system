import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hdbscan  # type: ignore  # noqa: E402

from config.experiment_profiles import (  # noqa: E402
    FUTURE_VALIDATION_BACKLOG,
    PROXY_EVALUATION_LIMITATION,
    build_decision_policy_contract,
)
from config import feature_vars as fv  # noqa: E402
from src.clustering.kmeans import load_retrieval_artifact  # noqa: E402


DEFAULT_METHODS: Tuple[str, ...] = ("kmeans", "gmm", "hdbscan")
DEFAULT_KS: Tuple[int, ...] = (5, 10, 20)
DEFAULT_METHOD_LABELS: Dict[str, str] = {
    "kmeans": "K-Means",
    "gmm": "GMM",
    "hdbscan": "HDBSCAN",
}
DEFAULT_RANKING_METHOD = str(
    getattr(fv, "default_recommendation_ranking_method", "distance")
)
DEFAULT_MIN_CONFIDENCE = float(
    getattr(fv, "default_min_assignment_confidence", 0.0)
)
DEFAULT_MIN_POSTERIOR = float(
    getattr(fv, "default_min_selected_cluster_posterior", 0.0)
)
DEFAULT_SUBSAMPLE_RUNS = 50
DEFAULT_SUBSAMPLE_FRACTION = 0.80
DEFAULT_SEED_RUNS = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_PREFIX = "evaluation_upgrades"
RANKING_K = 10


@dataclass
class MethodArtifacts:
    method_id: str
    display_name: str
    results_path: Path
    artifact_path: Path
    frame: pd.DataFrame
    songs: np.ndarray
    genres: np.ndarray
    artists: np.ndarray
    titles: np.ndarray
    filenames: np.ndarray
    msd_track_ids: np.ndarray
    labels: np.ndarray
    prepared_features: np.ndarray
    coords: np.ndarray
    assignment_confidence: Optional[np.ndarray]
    posterior_probabilities: Optional[np.ndarray]
    distance_to_cluster: Optional[np.ndarray]
    log_likelihood: Optional[np.ndarray]
    artifact_version: int
    feature_subset_name: str
    selected_audio_feature_keys: List[str]
    equalization_method: str
    pca_components_per_group: Optional[int]
    raw_feature_dimension: Optional[int]
    prepared_feature_dimension: int
    profile_id: str


@dataclass
class StabilityRun:
    run_id: int
    sample_indices: np.ndarray
    labels: np.ndarray
    n_clusters: int
    noise_fraction: float
    reference_ari: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate clustering outputs with recommendation-centered metrics, "
            "stability protocols, and diagnostics."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Clustering methods to evaluate (default: kmeans gmm hdbscan).",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=list(DEFAULT_KS),
        help="Top-K cutoffs for recommendation metrics (default: 5 10 20).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="output/clustering_results",
        help="Directory containing saved clustering artifacts and results CSV files.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="output/metrics",
        help="Directory where evaluation outputs will be written.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix used for generated output files.",
    )
    parser.add_argument(
        "--ranking-method",
        type=str,
        default=DEFAULT_RANKING_METHOD,
        choices=("distance", "posterior_weighted"),
        help="Recommendation ranking method to mirror the UI behavior.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum assignment confidence filter for candidates.",
    )
    parser.add_argument(
        "--min-posterior",
        type=float,
        default=DEFAULT_MIN_POSTERIOR,
        help="Minimum selected-cluster posterior filter for GMM candidates.",
    )
    parser.add_argument(
        "--subsample-runs",
        type=int,
        default=DEFAULT_SUBSAMPLE_RUNS,
        help="Number of subsample runs for stability estimation (default: 50).",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=DEFAULT_SUBSAMPLE_FRACTION,
        help="Fraction of items kept in each stability subsample (default: 0.80).",
    )
    parser.add_argument(
        "--seed-runs",
        type=int,
        default=DEFAULT_SEED_RUNS,
        help="Number of full-data repeated-seed runs for seed-based methods.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base random seed used for resampling and reruns.",
    )
    parser.add_argument(
        "--stability-jobs",
        type=int,
        default=1,
        help="Parallel jobs for stability fits. Use 1 for deterministic serial execution.",
    )
    return parser.parse_args()


def _normalize_method_id(method_id: str) -> str:
    normalized = method_id.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in DEFAULT_METHOD_LABELS:
        raise ValueError(
            f"Unsupported method '{method_id}'. Expected one of {sorted(DEFAULT_METHOD_LABELS)}."
        )
    return normalized


def _split_song_display_name(song: str) -> Tuple[str, str]:
    text = str(song)
    if " - " in text:
        artist, title = text.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", text.strip()


def _normalize_metadata_text(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().strip()
    value = value.replace("&", " and ")
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^\]]*\]", " ", value)
    value = re.sub(r"[^a-z0-9']+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _artifact_scalar_str(
    artifact_payload: Dict[str, np.ndarray],
    key: str,
    default: str,
) -> str:
    values = artifact_payload.get(key)
    if values is None:
        return default
    arr = np.asarray(values)
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def _artifact_scalar_int(
    artifact_payload: Dict[str, np.ndarray],
    key: str,
    default: Optional[int],
) -> Optional[int]:
    values = artifact_payload.get(key)
    if values is None:
        return default
    arr = np.asarray(values)
    if arr.size == 0:
        return default
    value = int(arr.reshape(-1)[0])
    if value < 0:
        return default
    return value


def _artifact_string_list(
    artifact_payload: Dict[str, np.ndarray],
    key: str,
    default: List[str],
) -> List[str]:
    values = artifact_payload.get(key)
    if values is None:
        return list(default)
    arr = np.asarray(values)
    return [str(item) for item in arr.tolist()]


def _load_method_artifacts(method_id: str, artifact_dir: Path) -> MethodArtifacts:
    results_path = artifact_dir / f"audio_clustering_results_{method_id}.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing clustering results CSV: {results_path}")

    artifact_payload = load_retrieval_artifact(method_id, output_dir=str(artifact_dir))
    frame = pd.read_csv(results_path)

    songs = artifact_payload["songs"].astype(str)
    if "Song" not in frame.columns:
        raise ValueError(f"Results CSV is missing 'Song' column: {results_path}")
    frame_songs = frame["Song"].astype(str).to_numpy()
    if len(frame_songs) != len(songs) or not np.array_equal(frame_songs, songs):
        raise ValueError(
            "Results CSV row order does not match retrieval artifact order for "
            f"method '{method_id}'."
        )

    genres = (
        frame["Genre"].fillna("unknown").astype(str).to_numpy()
        if "Genre" in frame.columns
        else np.array(["unknown"] * len(songs), dtype=object)
    )

    if "Artist" in frame.columns:
        artists_list = frame["Artist"].fillna("").astype(str).tolist()
    elif artifact_payload.get("artists") is not None:
        artists_list = np.asarray(artifact_payload["artists"]).astype(str).tolist()
    else:
        artists_list = []
        for song in songs:
            artist, _ = _split_song_display_name(song)
            artists_list.append(artist)

    if "Title" in frame.columns:
        titles_list = frame["Title"].fillna("").astype(str).tolist()
    elif artifact_payload.get("titles") is not None:
        titles_list = np.asarray(artifact_payload["titles"]).astype(str).tolist()
    else:
        titles_list = []
        for song in songs:
            _, title = _split_song_display_name(song)
            titles_list.append(title)

    if "Filename" in frame.columns:
        filenames = frame["Filename"].fillna("").astype(str).to_numpy()
    elif artifact_payload.get("filenames") is not None:
        filenames = np.asarray(artifact_payload["filenames"]).astype(str)
    else:
        filenames = np.array([""] * len(songs), dtype=object)

    if "MSDTrackID" in frame.columns:
        msd_track_ids = frame["MSDTrackID"].fillna("").astype(str).to_numpy()
    elif artifact_payload.get("msd_track_ids") is not None:
        msd_track_ids = np.asarray(artifact_payload["msd_track_ids"]).astype(str)
    else:
        msd_track_ids = np.array([""] * len(songs), dtype=object)

    feature_subset_name = _artifact_scalar_str(
        artifact_payload,
        "feature_subset_name",
        str(getattr(fv, "clustering_feature_subset_name", "unknown")),
    )
    selected_audio_feature_keys = _artifact_string_list(
        artifact_payload,
        "selected_audio_feature_keys",
        list(getattr(fv, "clustering_audio_feature_keys", [])),
    )
    equalization_method = _artifact_scalar_str(
        artifact_payload,
        "feature_equalization_method",
        str(getattr(fv, "feature_equalization_method", "unknown")),
    )
    pca_components_per_group = _artifact_scalar_int(
        artifact_payload,
        "pca_components_per_group",
        None,
    )
    raw_feature_dimension = _artifact_scalar_int(
        artifact_payload,
        "raw_feature_dimension",
        None,
    )
    prepared_feature_dimension = _artifact_scalar_int(
        artifact_payload,
        "prepared_feature_dimension",
        int(np.asarray(artifact_payload["prepared_features"]).shape[1]),
    ) or int(np.asarray(artifact_payload["prepared_features"]).shape[1])
    artifact_version = _artifact_scalar_int(artifact_payload, "artifact_version", 1) or 1
    profile_id = _artifact_scalar_str(artifact_payload, "profile_id", "unspecified")

    return MethodArtifacts(
        method_id=method_id,
        display_name=DEFAULT_METHOD_LABELS[method_id],
        results_path=results_path,
        artifact_path=artifact_dir / f"audio_clustering_artifact_{method_id}.npz",
        frame=frame,
        songs=songs,
        genres=np.asarray(genres, dtype=object),
        artists=np.asarray(artists_list, dtype=object),
        titles=np.asarray(titles_list, dtype=object),
        filenames=np.asarray(filenames, dtype=object),
        msd_track_ids=np.asarray(msd_track_ids, dtype=object),
        labels=np.asarray(artifact_payload["labels"]).astype(int),
        prepared_features=np.asarray(
            artifact_payload["prepared_features"], dtype=np.float32
        ),
        coords=np.asarray(artifact_payload["coords"], dtype=np.float32),
        assignment_confidence=(
            None
            if artifact_payload.get("assignment_confidence") is None
            else np.asarray(artifact_payload["assignment_confidence"], dtype=np.float32)
        ),
        posterior_probabilities=(
            None
            if artifact_payload.get("posterior_probabilities") is None
            else np.asarray(
                artifact_payload["posterior_probabilities"], dtype=np.float32
            )
        ),
        distance_to_cluster=(
            None
            if artifact_payload.get("distance_to_cluster") is None
            else np.asarray(artifact_payload["distance_to_cluster"], dtype=np.float32)
        ),
        log_likelihood=(
            None
            if artifact_payload.get("log_likelihood") is None
            else np.asarray(artifact_payload["log_likelihood"], dtype=np.float32)
        ),
        artifact_version=int(artifact_version),
        feature_subset_name=feature_subset_name,
        selected_audio_feature_keys=selected_audio_feature_keys,
        equalization_method=equalization_method,
        pca_components_per_group=pca_components_per_group,
        raw_feature_dimension=raw_feature_dimension,
        prepared_feature_dimension=int(prepared_feature_dimension),
        profile_id=profile_id,
    )


def _compute_cluster_summary(labels: np.ndarray) -> Dict[str, Any]:
    counts = Counter(int(label) for label in labels)
    noise_count = int(counts.get(-1, 0))
    non_noise = {label: count for label, count in counts.items() if label != -1}
    occupied_counts = np.asarray(list(non_noise.values()), dtype=np.int64)

    if occupied_counts.size == 0:
        min_cluster_size = 0
        median_cluster_size = 0.0
        max_cluster_size = 0
        dominant_cluster_fraction = float("nan")
    else:
        min_cluster_size = int(occupied_counts.min())
        median_cluster_size = float(np.median(occupied_counts))
        max_cluster_size = int(occupied_counts.max())
        dominant_cluster_fraction = float(
            occupied_counts.max() / max(1, occupied_counts.sum())
        )

    return {
        "cluster_count": int(len(non_noise)),
        "noise_count": noise_count,
        "noise_fraction": float(noise_count / max(1, len(labels))),
        "min_cluster_size": min_cluster_size,
        "median_cluster_size": median_cluster_size,
        "max_cluster_size": max_cluster_size,
        "dominant_cluster_fraction": dominant_cluster_fraction,
        "cluster_size_distribution": json.dumps(
            {str(label): int(count) for label, count in sorted(counts.items())},
            sort_keys=True,
        ),
    }


def _compute_internal_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    mask = np.ones(len(labels), dtype=bool)
    if np.any(labels == -1):
        mask = labels != -1

    filtered_features = features[mask]
    filtered_labels = labels[mask]
    unique_labels = np.unique(filtered_labels)

    if filtered_features.shape[0] < 3 or unique_labels.size < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
            "diagnostic_sample_count": int(filtered_features.shape[0]),
        }

    sample_size = min(5000, filtered_features.shape[0])
    silhouette_kwargs: Dict[str, Any] = {"random_state": 42}
    if sample_size < filtered_features.shape[0]:
        silhouette_kwargs["sample_size"] = sample_size

    return {
        "silhouette": float(
            silhouette_score(filtered_features, filtered_labels, **silhouette_kwargs)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(filtered_features, filtered_labels)
        ),
        "davies_bouldin": float(
            davies_bouldin_score(filtered_features, filtered_labels)
        ),
        "diagnostic_sample_count": int(filtered_features.shape[0]),
    }


def _get_recommendation_indices(
    artifacts: MethodArtifacts,
    query_idx: int,
    top_k: int,
    min_confidence: float,
    min_posterior: float,
    ranking_method: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = artifacts.labels
    selected_cluster = int(labels[query_idx])
    metadata = {
        "selected_cluster": selected_cluster,
        "initial_cluster_size": int(np.sum(labels == selected_cluster)),
        "candidate_count_before_filters": 0,
        "candidate_count_after_filters": 0,
        "noise_disabled": False,
        "ranking_method": ranking_method,
    }

    if selected_cluster == -1:
        metadata["noise_disabled"] = True
        return np.array([], dtype=np.int32), metadata

    candidate_indices = np.where(labels == selected_cluster)[0]
    candidate_indices = candidate_indices[candidate_indices != query_idx]
    metadata["candidate_count_before_filters"] = int(len(candidate_indices))

    candidate_confidence = None
    if artifacts.assignment_confidence is not None:
        candidate_confidence = artifacts.assignment_confidence[candidate_indices]
        keep = candidate_confidence >= float(min_confidence)
        candidate_indices = candidate_indices[keep]
        candidate_confidence = candidate_confidence[keep]

    candidate_cluster_posterior = None
    if (
        artifacts.posterior_probabilities is not None
        and 0 <= selected_cluster < artifacts.posterior_probabilities.shape[1]
    ):
        candidate_cluster_posterior = artifacts.posterior_probabilities[
            candidate_indices, selected_cluster
        ]
        keep = candidate_cluster_posterior >= float(min_posterior)
        candidate_indices = candidate_indices[keep]
        if candidate_confidence is not None:
            candidate_confidence = candidate_confidence[keep]
        candidate_cluster_posterior = candidate_cluster_posterior[keep]

    metadata["candidate_count_after_filters"] = int(len(candidate_indices))
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.int32), metadata

    selected_vector = artifacts.prepared_features[query_idx]
    candidate_vectors = artifacts.prepared_features[candidate_indices]
    distances = np.linalg.norm(candidate_vectors - selected_vector, axis=1)

    ranking_scores = distances.copy()
    active_ranking_method = ranking_method
    if ranking_method == "posterior_weighted" and candidate_cluster_posterior is not None:
        ranking_scores = distances / np.clip(candidate_cluster_posterior, 1e-3, 1.0)
    else:
        active_ranking_method = "distance"

    order = np.lexsort((distances, ranking_scores))[:top_k]
    metadata["ranking_method"] = active_ranking_method
    return candidate_indices[order].astype(np.int32), metadata


def _compute_entropy_from_counts(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return float("nan")
    probs = values / total
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
    max_entropy = math.log(len(values)) if len(values) > 1 else 0.0
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def _evaluate_recommendations(
    artifacts: MethodArtifacts,
    ks: Sequence[int],
    min_confidence: float,
    min_posterior: float,
    ranking_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ks_sorted = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_sorted:
        raise ValueError("At least one positive K value is required.")

    max_k = max(ks_sorted)
    rows: List[Dict[str, Any]] = []
    exposure_counts: Dict[int, Counter] = {k: Counter() for k in ks_sorted}
    cluster_exposure_counts: Dict[int, Counter] = {k: Counter() for k in ks_sorted}

    for query_idx in range(len(artifacts.songs)):
        chosen_indices, metadata = _get_recommendation_indices(
            artifacts=artifacts,
            query_idx=query_idx,
            top_k=max_k,
            min_confidence=min_confidence,
            min_posterior=min_posterior,
            ranking_method=ranking_method,
        )

        row: Dict[str, Any] = {
            "Method": artifacts.display_name,
            "MethodId": artifacts.method_id,
            "QueryIndex": int(query_idx),
            "Song": str(artifacts.songs[query_idx]),
            "QueryArtist": str(artifacts.artists[query_idx]),
            "QueryTitle": str(artifacts.titles[query_idx]),
            "QueryFilename": str(artifacts.filenames[query_idx]),
            "QueryMSDTrackID": str(artifacts.msd_track_ids[query_idx]),
            "QueryGenre": str(artifacts.genres[query_idx]),
            "Cluster": int(artifacts.labels[query_idx]),
            "NoiseRecommendationsDisabled": bool(metadata["noise_disabled"]),
            "InitialClusterSize": int(metadata["initial_cluster_size"]),
            "CandidateCountBeforeFilters": int(
                metadata["candidate_count_before_filters"]
            ),
            "CandidateCountAfterFilters": int(metadata["candidate_count_after_filters"]),
            "RankingMethod": str(metadata["ranking_method"]),
        }

        for k in ks_sorted:
            top_indices = chosen_indices[:k]
            returned = int(len(top_indices))
            genre_hits = int(
                np.sum(artifacts.genres[top_indices] == artifacts.genres[query_idx])
            )
            artist_hits = int(
                np.sum(artifacts.artists[top_indices] == artifacts.artists[query_idx])
            )

            row[f"Returned@{k}"] = returned
            row[f"GenreHits@{k}"] = genre_hits
            row[f"GenrePrecision@{k}"] = float(genre_hits / k)
            row[f"GenreHitRate@{k}"] = float(1.0 if genre_hits > 0 else 0.0)
            row[f"ArtistHits@{k}"] = artist_hits
            row[f"ArtistPrecision@{k}"] = float(artist_hits / k)
            row[f"ArtistHitRate@{k}"] = float(1.0 if artist_hits > 0 else 0.0)
            row[f"FullList@{k}"] = float(1.0 if returned == k else 0.0)
            row[f"RecommendationIndices@{k}"] = json.dumps(
                [int(idx) for idx in top_indices]
            )

            for idx in top_indices:
                exposure_counts[k][int(idx)] += 1
                cluster_exposure_counts[k][int(artifacts.labels[idx])] += 1

        rows.append(row)

    per_query = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []

    for k in ks_sorted:
        exposures = exposure_counts[k]
        cluster_exposures = cluster_exposure_counts[k]
        total_slots = int(sum(exposures.values()))
        coverage = float(len(exposures) / len(artifacts.songs))
        item_hhi = (
            float(
                np.sum(
                    (
                        np.asarray(list(exposures.values()), dtype=np.float64)
                        / max(1, total_slots)
                    )
                    ** 2
                )
            )
            if total_slots > 0
            else float("nan")
        )
        cluster_top_share = (
            float(max(cluster_exposures.values()) / max(1, total_slots))
            if cluster_exposures
            else float("nan")
        )

        summary_rows.append(
            {
                "Method": artifacts.display_name,
                "MethodId": artifacts.method_id,
                "K": int(k),
                "QueryCount": int(len(per_query)),
                "SupportedQueryFraction": float((per_query[f"Returned@{k}"] > 0).mean()),
                "FullListFraction": float(per_query[f"FullList@{k}"].mean()),
                "MeanReturned": float(per_query[f"Returned@{k}"].mean()),
                "GenrePrecision@K": float(per_query[f"GenrePrecision@{k}"].mean()),
                "GenreHitRate@K": float(per_query[f"GenreHitRate@{k}"].mean()),
                "ArtistPrecision@K": float(per_query[f"ArtistPrecision@{k}"].mean()),
                "ArtistHitRate@K": float(per_query[f"ArtistHitRate@{k}"].mean()),
                "CatalogCoverage": coverage,
                "RecommendationSlotCount": total_slots,
                "ItemExposureHHI": item_hhi,
                "ItemExposureDiversity": (
                    float(1.0 - item_hhi) if np.isfinite(item_hhi) else float("nan")
                ),
                "ClusterExposureTopShare": cluster_top_share,
                "ClusterExposureEntropy": _compute_entropy_from_counts(
                    cluster_exposures.values()
                ),
            }
        )

    return per_query, pd.DataFrame(summary_rows)


def _load_kmeans_params(metrics_dir: Path, labels: np.ndarray) -> Dict[str, Any]:
    criteria_path = metrics_dir / "kmeans_selection_criteria.csv"
    if criteria_path.exists():
        criteria = pd.read_csv(criteria_path)
        if (
            not criteria.empty
            and "Silhouette" in criteria.columns
            and "K" in criteria.columns
        ):
            criteria["K"] = criteria["K"].astype(int)
            criteria["Silhouette"] = pd.to_numeric(
                criteria["Silhouette"], errors="coerce"
            )
            best_row = criteria.sort_values(
                by=["Silhouette", "K"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            return {"n_clusters": int(best_row["K"]), "n_init": 20}

    return {"n_clusters": int(len(np.unique(labels))), "n_init": 20}


def _load_gmm_params(metrics_dir: Path, labels: np.ndarray) -> Dict[str, Any]:
    summary_path = metrics_dir / "gmm_selection_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        selected = payload.get("selected_candidate", {})
        return {
            "n_components": int(selected.get("components", len(np.unique(labels)))),
            "covariance_type": str(selected.get("covariance_type", "diag")),
            "reg_covar": float(selected.get("reg_covar", 1e-5)),
            "n_init": int(selected.get("n_init", 10)),
            "max_iter": int(selected.get("effective_max_iter", 300)),
            "tol": 1e-3,
            "init_params": "kmeans",
        }

    return {
        "n_components": int(len(np.unique(labels))),
        "covariance_type": "diag",
        "reg_covar": 1e-5,
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-3,
        "init_params": "kmeans",
    }


def _load_hdbscan_params(metrics_dir: Path) -> Dict[str, Any]:
    criteria_path = metrics_dir / "hdbscan_selection_criteria.csv"
    if criteria_path.exists():
        criteria = pd.read_csv(criteria_path)
        if not criteria.empty:
            numeric_columns = [
                "min_cluster_size",
                "min_samples",
                "noise_fraction",
                "silhouette",
                "score",
            ]
            for column in numeric_columns:
                if column in criteria.columns:
                    criteria[column] = pd.to_numeric(criteria[column], errors="coerce")
            criteria["score_filled"] = criteria["score"].fillna(float("-inf"))
            criteria["silhouette_filled"] = criteria["silhouette"].fillna(float("-inf"))
            ranked = criteria.sort_values(
                by=[
                    "score_filled",
                    "noise_fraction",
                    "silhouette_filled",
                    "min_cluster_size",
                    "min_samples",
                ],
                ascending=[False, True, False, True, True],
                kind="stable",
            )
            best_row = ranked.iloc[0]
            return {
                "min_cluster_size": int(best_row["min_cluster_size"]),
                "min_samples": int(best_row["min_samples"]),
                "cluster_selection_epsilon": 0.0,
                "allow_single_cluster": False,
            }

    return {
        "min_cluster_size": 10,
        "min_samples": 5,
        "cluster_selection_epsilon": 0.0,
        "allow_single_cluster": False,
    }


def _fit_labels_for_method(
    method_id: str,
    features: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    if method_id == "kmeans":
        estimator = KMeans(
            n_clusters=int(params["n_clusters"]),
            random_state=int(seed),
            n_init=int(params.get("n_init", 20)),
        )
        return estimator.fit_predict(features).astype(int)

    if method_id == "gmm":
        estimator = GaussianMixture(
            n_components=int(params["n_components"]),
            covariance_type=str(params["covariance_type"]),
            reg_covar=float(params["reg_covar"]),
            n_init=int(params.get("n_init", 10)),
            max_iter=int(params.get("max_iter", 300)),
            tol=float(params.get("tol", 1e-3)),
            init_params=str(params.get("init_params", "kmeans")),
            random_state=int(seed),
        )
        return estimator.fit_predict(features).astype(int)

    if method_id == "hdbscan":
        estimator = hdbscan.HDBSCAN(
            min_cluster_size=int(params["min_cluster_size"]),
            min_samples=int(params["min_samples"]),
            cluster_selection_epsilon=float(
                params.get("cluster_selection_epsilon", 0.0)
            ),
            allow_single_cluster=bool(params.get("allow_single_cluster", False)),
            prediction_data=False,
        )
        return estimator.fit_predict(features).astype(int)

    raise ValueError(f"Unsupported method '{method_id}'.")


def _build_subsample_indices(
    n_samples: int,
    sample_fraction: float,
    runs: int,
    base_seed: int,
) -> List[np.ndarray]:
    sample_size = max(2, int(round(n_samples * sample_fraction)))
    sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(base_seed)
    indices: List[np.ndarray] = []
    for _ in range(runs):
        sample = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
        indices.append(sample.astype(np.int32))
    return indices


def _run_one_subsample_fit(
    method_id: str,
    features: np.ndarray,
    reference_labels: np.ndarray,
    params: Dict[str, Any],
    run_id: int,
    sample_indices: np.ndarray,
    base_seed: int,
) -> StabilityRun:
    labels = _fit_labels_for_method(
        method_id=method_id,
        features=features[sample_indices],
        params=params,
        seed=base_seed + run_id,
    )
    non_noise_clusters = {int(label) for label in labels if int(label) != -1}
    noise_fraction = float(np.mean(labels == -1)) if len(labels) else float("nan")
    reference_ari = float(adjusted_rand_score(reference_labels[sample_indices], labels))
    return StabilityRun(
        run_id=int(run_id),
        sample_indices=sample_indices.astype(np.int32),
        labels=labels.astype(np.int32),
        n_clusters=int(len(non_noise_clusters)),
        noise_fraction=noise_fraction,
        reference_ari=reference_ari,
    )


def _pairwise_overlap_ari(run_a: StabilityRun, run_b: StabilityRun) -> Tuple[int, int, float, int]:
    overlap, pos_a, pos_b = np.intersect1d(
        run_a.sample_indices,
        run_b.sample_indices,
        assume_unique=True,
        return_indices=True,
    )
    if len(overlap) < 2:
        return run_a.run_id, run_b.run_id, float("nan"), int(len(overlap))
    score = float(adjusted_rand_score(run_a.labels[pos_a], run_b.labels[pos_b]))
    return run_a.run_id, run_b.run_id, score, int(len(overlap))


def _compute_binary_cluster_match(
    reference_members: np.ndarray,
    run_indices: np.ndarray,
    run_labels: np.ndarray,
) -> Tuple[float, float, float]:
    overlap_members = np.intersect1d(reference_members, run_indices, assume_unique=False)
    if len(overlap_members) == 0:
        return float("nan"), float("nan"), float("nan")

    best_jaccard = 0.0
    best_precision = 0.0
    best_recall = 0.0
    candidate_labels = [int(label) for label in np.unique(run_labels) if int(label) != -1]

    for label in candidate_labels:
        candidate_members = run_indices[run_labels == label]
        intersection = int(
            np.intersect1d(overlap_members, candidate_members, assume_unique=False).size
        )
        if intersection == 0:
            continue
        union = int(np.union1d(overlap_members, candidate_members).size)
        jaccard = float(intersection / max(1, union))
        if jaccard > best_jaccard:
            best_jaccard = jaccard
            best_precision = float(intersection / max(1, len(candidate_members)))
            best_recall = float(intersection / max(1, len(overlap_members)))

    return best_jaccard, best_precision, best_recall


def _evaluate_subsample_stability(
    artifacts: MethodArtifacts,
    metrics_dir: Path,
    subsample_runs: int,
    subsample_fraction: float,
    seed_runs: int,
    base_seed: int,
    n_jobs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    method_id = artifacts.method_id
    if method_id == "kmeans":
        params = _load_kmeans_params(metrics_dir, artifacts.labels)
    elif method_id == "gmm":
        params = _load_gmm_params(metrics_dir, artifacts.labels)
    elif method_id == "hdbscan":
        params = _load_hdbscan_params(metrics_dir)
    else:
        raise ValueError(f"Unsupported method '{method_id}'.")

    sample_indices_list = _build_subsample_indices(
        n_samples=len(artifacts.labels),
        sample_fraction=subsample_fraction,
        runs=subsample_runs,
        base_seed=base_seed,
    )
    run_ids = list(range(1, subsample_runs + 1))

    def _run_fit(run_id: int, sample_indices: np.ndarray) -> StabilityRun:
        return _run_one_subsample_fit(
            method_id=method_id,
            features=artifacts.prepared_features,
            reference_labels=artifacts.labels,
            params=params,
            run_id=run_id,
            sample_indices=sample_indices,
            base_seed=base_seed,
        )

    if n_jobs == 1:
        runs = [
            _run_fit(run_id, sample_indices)
            for run_id, sample_indices in zip(run_ids, sample_indices_list)
        ]
    else:
        runs = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            delayed(_run_fit)(run_id, sample_indices)
            for run_id, sample_indices in zip(run_ids, sample_indices_list)
        )

    subsample_runs_df = pd.DataFrame(
        [
            {
                "Method": artifacts.display_name,
                "MethodId": method_id,
                "RunId": int(run.run_id),
                "SampleSize": int(len(run.sample_indices)),
                "SampleFraction": float(len(run.sample_indices) / len(artifacts.labels)),
                "Clusters": int(run.n_clusters),
                "NoiseFraction": float(run.noise_fraction),
                "ReferenceARI": float(run.reference_ari),
            }
            for run in runs
        ]
    )

    pairwise_rows = []
    pairwise_scores: List[float] = []
    for idx in range(len(runs)):
        for jdx in range(idx + 1, len(runs)):
            run_id_a, run_id_b, score, overlap_size = _pairwise_overlap_ari(
                runs[idx], runs[jdx]
            )
            pairwise_rows.append(
                {
                    "Method": artifacts.display_name,
                    "MethodId": method_id,
                    "RunIdA": int(run_id_a),
                    "RunIdB": int(run_id_b),
                    "OverlapSize": int(overlap_size),
                    "PairwiseARI": float(score),
                }
            )
            if np.isfinite(score):
                pairwise_scores.append(float(score))

    pairwise_df = pd.DataFrame(pairwise_rows)

    cluster_rows = []
    unique_clusters = [int(label) for label in np.unique(artifacts.labels) if int(label) != -1]
    for cluster_label in unique_clusters:
        reference_members = np.flatnonzero(artifacts.labels == cluster_label).astype(np.int32)
        jaccard_scores: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        overlap_runs = 0

        for run in runs:
            jaccard, precision, recall = _compute_binary_cluster_match(
                reference_members=reference_members,
                run_indices=run.sample_indices,
                run_labels=run.labels,
            )
            if np.isfinite(jaccard):
                overlap_runs += 1
                jaccard_scores.append(float(jaccard))
                precision_scores.append(float(precision))
                recall_scores.append(float(recall))

        cluster_rows.append(
            {
                "Method": artifacts.display_name,
                "MethodId": method_id,
                "Cluster": int(cluster_label),
                "ClusterSize": int(len(reference_members)),
                "RunsWithOverlap": int(overlap_runs),
                "MeanBestMatchJaccard": (
                    float(np.mean(jaccard_scores)) if jaccard_scores else float("nan")
                ),
                "MedianBestMatchJaccard": (
                    float(np.median(jaccard_scores))
                    if jaccard_scores
                    else float("nan")
                ),
                "MinBestMatchJaccard": (
                    float(np.min(jaccard_scores)) if jaccard_scores else float("nan")
                ),
                "MeanBestMatchPrecision": (
                    float(np.mean(precision_scores))
                    if precision_scores
                    else float("nan")
                ),
                "MeanBestMatchRecall": (
                    float(np.mean(recall_scores)) if recall_scores else float("nan")
                ),
            }
        )

    cluster_df = pd.DataFrame(cluster_rows)

    seed_df = pd.DataFrame()
    if method_id in {"kmeans", "gmm"} and seed_runs > 1:
        seed_rows = []
        seed_labels: List[np.ndarray] = []
        for rerun_index in range(seed_runs):
            seed_value = base_seed + 1000 + rerun_index
            labels = _fit_labels_for_method(
                method_id=method_id,
                features=artifacts.prepared_features,
                params=params,
                seed=seed_value,
            )
            seed_labels.append(labels.astype(np.int32))
            seed_rows.append(
                {
                    "Method": artifacts.display_name,
                    "MethodId": method_id,
                    "SeedRunId": int(rerun_index + 1),
                    "RandomSeed": int(seed_value),
                    "ReferenceARI": float(adjusted_rand_score(artifacts.labels, labels)),
                    "Clusters": int(
                        len({int(label) for label in labels if int(label) != -1})
                    ),
                    "NoiseFraction": (
                        float(np.mean(labels == -1)) if len(labels) else float("nan")
                    ),
                }
            )

        seed_pairwise: List[float] = []
        for idx in range(len(seed_labels)):
            for jdx in range(idx + 1, len(seed_labels)):
                seed_pairwise.append(
                    float(adjusted_rand_score(seed_labels[idx], seed_labels[jdx]))
                )
        seed_df = pd.DataFrame(seed_rows)
        seed_summary = {
            "seed_runs": int(seed_runs),
            "mean_pairwise_ari": (
                float(np.mean(seed_pairwise)) if seed_pairwise else float("nan")
            ),
            "median_pairwise_ari": (
                float(np.median(seed_pairwise)) if seed_pairwise else float("nan")
            ),
            "mean_reference_ari": (
                float(seed_df["ReferenceARI"].mean()) if not seed_df.empty else float("nan")
            ),
            "median_reference_ari": (
                float(seed_df["ReferenceARI"].median())
                if not seed_df.empty
                else float("nan")
            ),
        }
    else:
        seed_summary = {
            "seed_runs": 0,
            "mean_pairwise_ari": float("nan"),
            "median_pairwise_ari": float("nan"),
            "mean_reference_ari": float("nan"),
            "median_reference_ari": float("nan"),
        }

    summary = {
        "stability_protocol": {
            "subsample_runs": int(subsample_runs),
            "subsample_fraction": float(subsample_fraction),
            "pairwise_overlap_metric": "Adjusted Rand Index",
            "per_cluster_metric": "Best-match Jaccard against reference clusters",
            "seed_runs": int(seed_summary["seed_runs"]),
        },
        "subsample_pairwise_mean_ari": (
            float(np.mean(pairwise_scores)) if pairwise_scores else float("nan")
        ),
        "subsample_pairwise_median_ari": (
            float(np.median(pairwise_scores)) if pairwise_scores else float("nan")
        ),
        "subsample_reference_mean_ari": (
            float(subsample_runs_df["ReferenceARI"].mean())
            if not subsample_runs_df.empty
            else float("nan")
        ),
        "subsample_reference_median_ari": (
            float(subsample_runs_df["ReferenceARI"].median())
            if not subsample_runs_df.empty
            else float("nan")
        ),
        "subsample_mean_clusters": (
            float(subsample_runs_df["Clusters"].mean())
            if not subsample_runs_df.empty
            else float("nan")
        ),
        "subsample_mean_noise_fraction": (
            float(subsample_runs_df["NoiseFraction"].mean())
            if not subsample_runs_df.empty
            else float("nan")
        ),
        "per_cluster_mean_best_match_jaccard": (
            float(cluster_df["MeanBestMatchJaccard"].mean())
            if not cluster_df.empty
            else float("nan")
        ),
        "per_cluster_median_best_match_jaccard": (
            float(cluster_df["MedianBestMatchJaccard"].median())
            if not cluster_df.empty
            else float("nan")
        ),
        "seed_summary": seed_summary,
        "seed_df": seed_df,
    }

    return subsample_runs_df, pairwise_df, cluster_df, summary


def _build_method_summary_row(
    artifacts: MethodArtifacts,
    recommendation_summary: pd.DataFrame,
    stability_summary: Dict[str, Any],
    internal_metrics: Dict[str, float],
    cluster_summary: Dict[str, Any],
) -> Dict[str, Any]:
    ranking_row = recommendation_summary.loc[recommendation_summary["K"] == RANKING_K]
    if ranking_row.empty:
        ranking_row = recommendation_summary.sort_values("K", kind="stable").tail(1)
    ranking = ranking_row.iloc[0]

    avg_confidence = (
        float(np.mean(artifacts.assignment_confidence))
        if artifacts.assignment_confidence is not None
        else float("nan")
    )
    avg_log_likelihood = (
        float(np.mean(artifacts.log_likelihood))
        if artifacts.log_likelihood is not None
        else float("nan")
    )

    return {
        "Method": artifacts.display_name,
        "MethodId": artifacts.method_id,
        "RecommendationRankingK": int(ranking["K"]),
        "GenrePrecision@K": float(ranking["GenrePrecision@K"]),
        "GenreHitRate@K": float(ranking["GenreHitRate@K"]),
        "ArtistPrecision@K": float(ranking["ArtistPrecision@K"]),
        "ArtistHitRate@K": float(ranking["ArtistHitRate@K"]),
        "CatalogCoverage": float(ranking["CatalogCoverage"]),
        "ItemExposureDiversity": float(ranking["ItemExposureDiversity"]),
        "ClusterExposureTopShare": float(ranking["ClusterExposureTopShare"]),
        "SupportedQueryFraction": float(ranking["SupportedQueryFraction"]),
        "FullListFraction": float(ranking["FullListFraction"]),
        "MeanReturned": float(ranking["MeanReturned"]),
        "SubsampleMeanARI": float(stability_summary["subsample_pairwise_mean_ari"]),
        "SubsampleMedianARI": float(stability_summary["subsample_pairwise_median_ari"]),
        "ReferenceMeanARI": float(stability_summary["subsample_reference_mean_ari"]),
        "ReferenceMedianARI": float(stability_summary["subsample_reference_median_ari"]),
        "PerClusterMeanJaccard": float(
            stability_summary["per_cluster_mean_best_match_jaccard"]
        ),
        "PerClusterMedianJaccard": float(
            stability_summary["per_cluster_median_best_match_jaccard"]
        ),
        "SeedMeanARI": float(stability_summary["seed_summary"]["mean_pairwise_ari"]),
        "SeedMedianARI": float(
            stability_summary["seed_summary"]["median_pairwise_ari"]
        ),
        "SilhouetteDiagnostic": float(internal_metrics["silhouette"]),
        "CalinskiHarabaszDiagnostic": float(
            internal_metrics["calinski_harabasz"]
        ),
        "DaviesBouldinDiagnostic": float(internal_metrics["davies_bouldin"]),
        "ClusterCount": int(cluster_summary["cluster_count"]),
        "NoiseFraction": float(cluster_summary["noise_fraction"]),
        "MinClusterSize": int(cluster_summary["min_cluster_size"]),
        "MedianClusterSize": float(cluster_summary["median_cluster_size"]),
        "MaxClusterSize": int(cluster_summary["max_cluster_size"]),
        "DominantClusterFraction": float(
            cluster_summary["dominant_cluster_fraction"]
        ),
        "AvgAssignmentConfidence": avg_confidence,
        "AvgLogLikelihood": avg_log_likelihood,
        "ClusterSizeDistribution": cluster_summary["cluster_size_distribution"],
    }


def _rank_methods(comparison_df: pd.DataFrame) -> pd.DataFrame:
    ranked = comparison_df.sort_values(
        by=[
            "GenrePrecision@K",
            "GenreHitRate@K",
            "ArtistPrecision@K",
            "ArtistHitRate@K",
            "CatalogCoverage",
            "ItemExposureDiversity",
            "SubsampleMedianARI",
            "SubsampleMeanARI",
        ],
        ascending=[False, False, False, False, False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "OverallRank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def _write_markdown_report(
    output_path: Path,
    ranked_df: pd.DataFrame,
    recommendation_summaries: Dict[str, pd.DataFrame],
    stability_summaries: Dict[str, Dict[str, Any]],
    cluster_summaries: Dict[str, Dict[str, Any]],
    representation_contract: Dict[str, Any],
    decision_policy: Dict[str, Any],
    decision_assessment: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Clustering Evaluation Comparison")
    lines.append("")
    lines.append(
        "This report ranks candidate models by recommendation quality first and stability second."
    )
    lines.append("")
    lines.append("## Ranking basis")
    lines.append("")
    lines.append(
        "- Primary metrics at K=10: genre precision, genre hit rate, artist precision, artist hit rate."
    )
    lines.append(
        "- Recommendation breadth checks: catalog coverage and item exposure diversity."
    )
    lines.append(
        "- Stability tiebreakers: median pairwise subsample ARI, then mean pairwise subsample ARI."
    )
    lines.append("- Internal metrics are included below as diagnostics only.")
    lines.append("")
    lines.append("## Representation contract")
    lines.append("")
    lines.append(
        f"- Profile id: `{representation_contract['profile_id']}`"
    )
    lines.append(
        f"- Feature subset: `{representation_contract['feature_subset_name']}` "
        f"({representation_contract['selected_audio_feature_keys']})"
    )
    lines.append(
        f"- Equalization: `{representation_contract['equalization_method']}`"
    )
    if representation_contract["pca_components_per_group"] is not None:
        lines.append(
            f"- PCA components per group: {representation_contract['pca_components_per_group']}"
        )
    lines.append(
        f"- Raw/prepared dimensions: {representation_contract['raw_feature_dimension']} / "
        f"{representation_contract['prepared_feature_dimension']}"
    )
    lines.append("")
    lines.append("## Explicit decisions")
    lines.append("")
    lines.append(
        f"- Cluster granularity: `{decision_policy['cluster_granularity']['policy']}` "
        f"with target range {decision_policy['cluster_granularity']['target_cluster_range']}"
    )
    lines.append(
        "- GMM stability gate: "
        f"median ARI >= {decision_policy['gmm_stability_gate']['minimum_subsample_median_ari']:.2f}, "
        f"mean ARI >= {decision_policy['gmm_stability_gate']['minimum_subsample_mean_ari']:.2f}, "
        f"reference median ARI >= {decision_policy['gmm_stability_gate']['minimum_reference_median_ari']:.2f}, "
        f"per-cluster median Jaccard >= {decision_policy['gmm_stability_gate']['minimum_per_cluster_median_jaccard']:.2f}"
    )
    lines.append(
        f"- Uncertain assignments: `{decision_policy['uncertain_assignments']['policy']}` "
        f"with default ranking `{decision_policy['uncertain_assignments']['default_ranking_method']}`"
    )
    lines.append(
        "- MSD restore gate: coverage >= "
        f"{decision_policy['msd_restore_gate']['minimum_live_audio_coverage_fraction']:.0%}, "
        f"missing rows <= {decision_policy['msd_restore_gate']['maximum_missing_audio_rows']}, "
        "plus clean-audit and explicit-experiment requirements"
    )
    lines.append("")
    lines.append("## Decision status")
    lines.append("")
    granularity = decision_assessment.get("cluster_granularity", {})
    lines.append(
        f"- Cluster granularity status: {granularity.get('status', 'unknown')} "
        f"(selected GMM clusters={granularity.get('selected_cluster_count', 'n/a')})"
    )
    stability_gate = decision_assessment.get("gmm_stability_gate", {})
    lines.append(
        f"- GMM stability gate: {stability_gate.get('status', 'unknown')} "
        f"(median={stability_gate.get('subsample_median_ari', float('nan')):.4f}, "
        f"mean={stability_gate.get('subsample_mean_ari', float('nan')):.4f})"
    )
    uncertain_assignments = decision_assessment.get("uncertain_assignments", {})
    lines.append(
        f"- Uncertain-assignment policy alignment: {uncertain_assignments.get('status', 'unknown')} "
        f"(ranking={uncertain_assignments.get('active_ranking_method', 'n/a')}, "
        f"min_conf={uncertain_assignments.get('active_min_confidence', float('nan')):.2f}, "
        f"min_p(cluster)={uncertain_assignments.get('active_min_posterior', float('nan')):.2f})"
    )
    msd_gate = decision_assessment.get("msd_restore_gate", {})
    if msd_gate:
        lines.append(
            f"- MSD restore gate: {msd_gate.get('status', 'unknown')} "
            f"(coverage={msd_gate.get('coverage_fraction', float('nan')):.2%}, "
            f"missing_rows={msd_gate.get('missing_audio_rows', 'n/a')})"
        )
    lines.append("")
    lines.append("## Overall ranking")
    lines.append("")
    lines.append(ranked_df.to_markdown(index=False))
    lines.append("")

    for _, row in ranked_df.iterrows():
        method_id = str(row["MethodId"])
        lines.append(f"## {row['Method']}")
        lines.append("")
        lines.append(
            f"- Recommendation@10: genre precision {row['GenrePrecision@K']:.4f}, "
            f"genre hit rate {row['GenreHitRate@K']:.4f}, artist precision {row['ArtistPrecision@K']:.4f}, "
            f"artist hit rate {row['ArtistHitRate@K']:.4f}"
        )
        lines.append(
            f"- Breadth: catalog coverage {row['CatalogCoverage']:.4f}, "
            f"item exposure diversity {row['ItemExposureDiversity']:.4f}, "
            f"cluster exposure top share {row['ClusterExposureTopShare']:.4f}"
        )
        lines.append(
            f"- Stability: subsample mean ARI {row['SubsampleMeanARI']:.4f}, "
            f"subsample median ARI {row['SubsampleMedianARI']:.4f}, "
            f"per-cluster mean Jaccard {row['PerClusterMeanJaccard']:.4f}"
        )
        lines.append(
            f"- Diagnostics: silhouette {row['SilhouetteDiagnostic']:.4f}, "
            f"Calinski-Harabasz {row['CalinskiHarabaszDiagnostic']:.2f}, "
            f"Davies-Bouldin {row['DaviesBouldinDiagnostic']:.4f}"
        )
        lines.append(
            f"- Cluster sanity check: clusters {int(row['ClusterCount'])}, noise fraction {row['NoiseFraction']:.4f}, "
            f"min/median/max cluster size {int(row['MinClusterSize'])}/{row['MedianClusterSize']:.1f}/{int(row['MaxClusterSize'])}"
        )
        lines.append("")

        rec_summary = recommendation_summaries[method_id].copy()
        rec_summary = rec_summary[
            [
                "K",
                "GenrePrecision@K",
                "GenreHitRate@K",
                "ArtistPrecision@K",
                "ArtistHitRate@K",
                "CatalogCoverage",
                "SupportedQueryFraction",
                "FullListFraction",
                "MeanReturned",
            ]
        ]
        lines.append("Recommendation summary:")
        lines.append("")
        lines.append(rec_summary.to_markdown(index=False))
        lines.append("")

        stability = stability_summaries[method_id]
        cluster_summary = cluster_summaries[method_id]
        lines.append("Stability protocol:")
        lines.append("")
        lines.append(
            f"- {stability['stability_protocol']['subsample_runs']} runs at "
            f"{stability['stability_protocol']['subsample_fraction']:.2f} subsample fraction"
        )
        lines.append(
            f"- Pairwise overlap ARI mean/median: "
            f"{stability['subsample_pairwise_mean_ari']:.4f} / "
            f"{stability['subsample_pairwise_median_ari']:.4f}"
        )
        if stability["seed_summary"]["seed_runs"] > 0:
            lines.append(
                f"- Repeated-seed ARI mean/median: "
                f"{stability['seed_summary']['mean_pairwise_ari']:.4f} / "
                f"{stability['seed_summary']['median_pairwise_ari']:.4f}"
            )
        lines.append(
            f"- Cluster size distribution: {cluster_summary['cluster_size_distribution']}"
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(f"- {PROXY_EVALUATION_LIMITATION}")
    lines.append("")
    lines.append("## Future validation backlog")
    lines.append("")
    for item in FUTURE_VALIDATION_BACKLOG:
        lines.append(f"- {item}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_representation_contract(
    method_artifacts: Dict[str, MethodArtifacts],
) -> Dict[str, Any]:
    first = next(iter(method_artifacts.values()))
    mismatch_methods: List[str] = []
    for method_id, artifacts in method_artifacts.items():
        if (
            artifacts.feature_subset_name != first.feature_subset_name
            or artifacts.selected_audio_feature_keys != first.selected_audio_feature_keys
            or artifacts.equalization_method != first.equalization_method
            or artifacts.pca_components_per_group != first.pca_components_per_group
            or artifacts.prepared_feature_dimension != first.prepared_feature_dimension
            or artifacts.profile_id != first.profile_id
        ):
            mismatch_methods.append(method_id)

    contract = {
        "profile_id": first.profile_id,
        "feature_subset_name": first.feature_subset_name,
        "selected_audio_feature_keys": list(first.selected_audio_feature_keys),
        "equalization_method": first.equalization_method,
        "pca_components_per_group": first.pca_components_per_group,
        "raw_feature_dimension": first.raw_feature_dimension,
        "prepared_feature_dimension": first.prepared_feature_dimension,
        "artifact_version": first.artifact_version,
        "mismatch_methods": mismatch_methods,
    }
    return contract


def _load_msd_restore_gate_status(decision_policy: Dict[str, Any]) -> Dict[str, Any]:
    summary_path = Path("data/songs_schema_summary.json")
    if not summary_path.exists():
        return {
            "status": "unknown_missing_summary",
            "coverage_fraction": float("nan"),
            "missing_audio_rows": None,
            "summary_path": str(summary_path),
        }

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    current_audio_rows = int(summary.get("current_audio_rows", 0))
    audio_with_numeric = int(summary.get("audio_rows_with_numeric_msd_features", 0))
    missing_audio_rows = int(summary.get("audio_rows_without_numeric_msd_features", 0))
    coverage_fraction = (
        float(audio_with_numeric / current_audio_rows) if current_audio_rows > 0 else 0.0
    )
    gate = decision_policy["msd_restore_gate"]
    coverage_ok = coverage_fraction >= float(gate["minimum_live_audio_coverage_fraction"])
    missing_ok = missing_audio_rows <= int(gate["maximum_missing_audio_rows"])
    passed = coverage_ok and missing_ok
    return {
        "status": "ready_for_experiment_only" if passed else "not_ready",
        "coverage_fraction": coverage_fraction,
        "missing_audio_rows": missing_audio_rows,
        "audio_rows_with_numeric_msd_features": audio_with_numeric,
        "current_audio_rows": current_audio_rows,
        "required_coverage_fraction": float(gate["minimum_live_audio_coverage_fraction"]),
        "required_max_missing_audio_rows": int(gate["maximum_missing_audio_rows"]),
        "summary_path": str(summary_path),
    }


def _build_decision_assessment(
    decision_policy: Dict[str, Any],
    cluster_summaries: Dict[str, Dict[str, Any]],
    stability_summaries: Dict[str, Dict[str, Any]],
    ranking_method: str,
    min_confidence: float,
    min_posterior: float,
) -> Dict[str, Any]:
    assessment: Dict[str, Any] = {
        "uncertain_assignments": {
            "status": (
                "aligned"
                if str(ranking_method)
                == str(decision_policy["uncertain_assignments"]["default_ranking_method"])
                and float(min_confidence)
                == float(
                    decision_policy["uncertain_assignments"][
                        "default_min_assignment_confidence"
                    ]
                )
                and float(min_posterior)
                == float(
                    decision_policy["uncertain_assignments"][
                        "default_min_selected_cluster_posterior"
                    ]
                )
                else "override_active"
            ),
            "policy": decision_policy["uncertain_assignments"]["policy"],
            "active_ranking_method": str(ranking_method),
            "active_min_confidence": float(min_confidence),
            "active_min_posterior": float(min_posterior),
        },
        "msd_restore_gate": _load_msd_restore_gate_status(decision_policy),
    }

    if "gmm" in cluster_summaries:
        granularity_policy = decision_policy["cluster_granularity"]
        selected_cluster_count = int(cluster_summaries["gmm"]["cluster_count"])
        in_range = int(granularity_policy["target_cluster_range"][0]) <= selected_cluster_count <= int(
            granularity_policy["target_cluster_range"][1]
        )
        assessment["cluster_granularity"] = {
            "status": "pass" if in_range else "out_of_range",
            "policy": granularity_policy["policy"],
            "selected_cluster_count": selected_cluster_count,
            "target_cluster_range": list(granularity_policy["target_cluster_range"]),
        }

    if "gmm" in stability_summaries:
        gate = decision_policy["gmm_stability_gate"]
        stability = stability_summaries["gmm"]
        median_ari = float(stability["subsample_pairwise_median_ari"])
        mean_ari = float(stability["subsample_pairwise_mean_ari"])
        reference_median_ari = float(stability["subsample_reference_median_ari"])
        per_cluster_median_jaccard = float(
            stability["per_cluster_median_best_match_jaccard"]
        )
        passed = (
            median_ari >= float(gate["minimum_subsample_median_ari"])
            and mean_ari >= float(gate["minimum_subsample_mean_ari"])
            and reference_median_ari >= float(gate["minimum_reference_median_ari"])
            and per_cluster_median_jaccard
            >= float(gate["minimum_per_cluster_median_jaccard"])
        )
        assessment["gmm_stability_gate"] = {
            "status": "pass" if passed else "fail",
            "subsample_median_ari": median_ari,
            "subsample_mean_ari": mean_ari,
            "reference_median_ari": reference_median_ari,
            "per_cluster_median_jaccard": per_cluster_median_jaccard,
            "required_subsample_median_ari": float(
                gate["minimum_subsample_median_ari"]
            ),
            "required_subsample_mean_ari": float(
                gate["minimum_subsample_mean_ari"]
            ),
            "required_reference_median_ari": float(
                gate["minimum_reference_median_ari"]
            ),
            "required_per_cluster_median_jaccard": float(
                gate["minimum_per_cluster_median_jaccard"]
            ),
        }

    return assessment


def run_evaluation(
    methods: Sequence[str],
    ks: Sequence[int] = DEFAULT_KS,
    artifact_dir: str | Path = "output/clustering_results",
    metrics_dir: str | Path = "output/metrics",
    prefix: str = DEFAULT_OUTPUT_PREFIX,
    ranking_method: str = DEFAULT_RANKING_METHOD,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_posterior: float = DEFAULT_MIN_POSTERIOR,
    subsample_runs: int = DEFAULT_SUBSAMPLE_RUNS,
    subsample_fraction: float = DEFAULT_SUBSAMPLE_FRACTION,
    seed_runs: int = DEFAULT_SEED_RUNS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    stability_jobs: int = 1,
) -> Dict[str, Any]:
    methods = [_normalize_method_id(method_id) for method_id in methods]
    ks = sorted({int(k) for k in ks if int(k) > 0})
    artifact_dir = Path(artifact_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix.strip() or DEFAULT_OUTPUT_PREFIX

    print("Loading saved clustering artifacts...")
    method_artifacts = {
        method_id: _load_method_artifacts(method_id, artifact_dir)
        for method_id in methods
    }
    representation_contract = _build_representation_contract(method_artifacts)
    decision_policy = build_decision_policy_contract()

    evaluation_payload: Dict[str, Any] = {
        "config": {
            "methods": methods,
            "ks": ks,
            "ranking_method": ranking_method,
            "min_confidence": float(min_confidence),
            "min_posterior": float(min_posterior),
            "subsample_runs": int(subsample_runs),
            "subsample_fraction": float(subsample_fraction),
            "seed_runs": int(seed_runs),
            "random_seed": int(random_seed),
            "stability_jobs": int(stability_jobs),
        },
        "representation_contract": representation_contract,
        "decision_policy": decision_policy,
        "evaluation_limitations": [PROXY_EVALUATION_LIMITATION],
        "future_validation_backlog": list(FUTURE_VALIDATION_BACKLOG),
        "methods": {},
    }

    recommendation_summaries: Dict[str, pd.DataFrame] = {}
    stability_summaries: Dict[str, Dict[str, Any]] = {}
    cluster_summaries: Dict[str, Dict[str, Any]] = {}
    comparison_rows: List[Dict[str, Any]] = []

    for method_id, artifacts in method_artifacts.items():
        print(f"Evaluating {artifacts.display_name}...")

        cluster_summary = _compute_cluster_summary(artifacts.labels)
        cluster_summaries[method_id] = cluster_summary
        internal_metrics = _compute_internal_metrics(
            artifacts.prepared_features,
            artifacts.labels,
        )

        per_query_df, recommendation_summary_df = _evaluate_recommendations(
            artifacts=artifacts,
            ks=ks,
            min_confidence=float(min_confidence),
            min_posterior=float(min_posterior),
            ranking_method=str(ranking_method),
        )
        recommendation_summaries[method_id] = recommendation_summary_df

        per_query_path = metrics_dir / f"{prefix}_per_query_{method_id}.csv"
        per_query_df.to_csv(per_query_path, index=False)

        recommendation_summary_path = (
            metrics_dir / f"{prefix}_recommendation_summary_{method_id}.csv"
        )
        recommendation_summary_df.to_csv(recommendation_summary_path, index=False)

        (
            subsample_runs_df,
            pairwise_df,
            cluster_df,
            stability_summary,
        ) = _evaluate_subsample_stability(
            artifacts=artifacts,
            metrics_dir=metrics_dir,
            subsample_runs=int(subsample_runs),
            subsample_fraction=float(subsample_fraction),
            seed_runs=int(seed_runs),
            base_seed=int(random_seed),
            n_jobs=int(stability_jobs),
        )
        seed_df = stability_summary.pop("seed_df")
        stability_summaries[method_id] = stability_summary

        subsample_runs_path = metrics_dir / f"{prefix}_stability_runs_{method_id}.csv"
        pairwise_path = metrics_dir / f"{prefix}_stability_pairwise_{method_id}.csv"
        cluster_stability_path = (
            metrics_dir / f"{prefix}_cluster_stability_{method_id}.csv"
        )
        seed_path = metrics_dir / f"{prefix}_seed_stability_{method_id}.csv"

        subsample_runs_df.to_csv(subsample_runs_path, index=False)
        pairwise_df.to_csv(pairwise_path, index=False)
        cluster_df.to_csv(cluster_stability_path, index=False)
        if not seed_df.empty:
            seed_df.to_csv(seed_path, index=False)

        comparison_rows.append(
            _build_method_summary_row(
                artifacts=artifacts,
                recommendation_summary=recommendation_summary_df,
                stability_summary=stability_summary,
                internal_metrics=internal_metrics,
                cluster_summary=cluster_summary,
            )
        )

        evaluation_payload["methods"][method_id] = {
            "display_name": artifacts.display_name,
            "results_csv": str(artifacts.results_path),
            "artifact_path": str(artifacts.artifact_path),
            "representation": {
                "artifact_version": int(artifacts.artifact_version),
                "profile_id": artifacts.profile_id,
                "feature_subset_name": artifacts.feature_subset_name,
                "selected_audio_feature_keys": list(
                    artifacts.selected_audio_feature_keys
                ),
                "equalization_method": artifacts.equalization_method,
                "pca_components_per_group": artifacts.pca_components_per_group,
                "raw_feature_dimension": artifacts.raw_feature_dimension,
                "prepared_feature_dimension": artifacts.prepared_feature_dimension,
            },
            "cluster_summary": cluster_summary,
            "internal_metrics": internal_metrics,
            "recommendation_summary_csv": str(recommendation_summary_path),
            "per_query_csv": str(per_query_path),
            "stability_summary": stability_summary,
            "stability_runs_csv": str(subsample_runs_path),
            "stability_pairwise_csv": str(pairwise_path),
            "cluster_stability_csv": str(cluster_stability_path),
            "seed_stability_csv": str(seed_path) if not seed_df.empty else None,
        }

    comparison_df = pd.DataFrame(comparison_rows)
    ranked_df = _rank_methods(comparison_df)
    decision_assessment = _build_decision_assessment(
        decision_policy=decision_policy,
        cluster_summaries=cluster_summaries,
        stability_summaries=stability_summaries,
        ranking_method=str(ranking_method),
        min_confidence=float(min_confidence),
        min_posterior=float(min_posterior),
    )

    comparison_path = metrics_dir / f"{prefix}_comparison.csv"
    ranked_df.to_csv(comparison_path, index=False)

    report_path = metrics_dir / f"{prefix}_comparison_report.md"
    _write_markdown_report(
        output_path=report_path,
        ranked_df=ranked_df,
        recommendation_summaries=recommendation_summaries,
        stability_summaries=stability_summaries,
        cluster_summaries=cluster_summaries,
        representation_contract=representation_contract,
        decision_policy=decision_policy,
        decision_assessment=decision_assessment,
    )

    evaluation_payload["comparison_csv"] = str(comparison_path)
    evaluation_payload["comparison_report"] = str(report_path)
    evaluation_payload["ranked_methods"] = ranked_df.to_dict(orient="records")
    evaluation_payload["decision_assessment"] = decision_assessment

    summary_path = metrics_dir / f"{prefix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_payload, handle, indent=2)

    print("\nEvaluation complete.")
    print(f"Comparison CSV -> {comparison_path}")
    print(f"Comparison report -> {report_path}")
    print(f"Summary JSON -> {summary_path}")
    return {
        "summary_path": str(summary_path),
        "comparison_csv": str(comparison_path),
        "comparison_report": str(report_path),
        "payload": evaluation_payload,
    }


def main() -> None:
    args = _parse_args()
    run_evaluation(
        methods=args.methods,
        ks=args.ks,
        artifact_dir=args.artifact_dir,
        metrics_dir=args.metrics_dir,
        prefix=args.prefix,
        ranking_method=args.ranking_method,
        min_confidence=args.min_confidence,
        min_posterior=args.min_posterior,
        subsample_runs=args.subsample_runs,
        subsample_fraction=args.subsample_fraction,
        seed_runs=args.seed_runs,
        random_seed=args.random_seed,
        stability_jobs=args.stability_jobs,
    )


if __name__ == "__main__":
    main()
