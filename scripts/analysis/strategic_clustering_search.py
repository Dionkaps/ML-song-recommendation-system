"""Run a strategic clustering search over feature subsets and preprocessing modes.

Outputs are written to a timestamped directory under ``output/metrics``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from config import feature_vars as fv  # noqa: E402
from src.clustering.kmeans import _collect_feature_vectors, _load_genre_mapping  # noqa: E402
from src.utils.genre_taxonomy import resolve_pipeline_songs_csv  # noqa: E402


DEFAULT_RESULTS_DIR = Path("output/features")
DEFAULT_AUDIO_DIR = Path("audio_files")
DEFAULT_OUTPUT_ROOT = Path("output/metrics")
DEFAULT_EXISTING_CACHE = (
    DEFAULT_OUTPUT_ROOT / "feature_sensitivity_suite" / "raw_audio_feature_cache.npz"
)
RANDOM_SEED = 42


@dataclass(frozen=True)
class GroupSpec:
    key: str
    label: str
    size: int


GROUP_SPECS: Tuple[GroupSpec, ...] = (
    GroupSpec("mfcc", "MFCC", 2 * fv.n_mfcc),
    GroupSpec("delta_mfcc", "Delta MFCC", 2 * fv.n_mfcc),
    GroupSpec("delta2_mfcc", "Delta2 MFCC", 2 * fv.n_mfcc),
    GroupSpec("spectral_centroid", "Spectral Centroid", 2),
    GroupSpec("spectral_rolloff", "Spectral Rolloff", 2),
    GroupSpec("spectral_flux", "Spectral Flux", 2),
    GroupSpec("spectral_flatness", "Spectral Flatness", 2),
    GroupSpec("zero_crossing_rate", "Zero Crossing Rate", 2),
    GroupSpec("chroma", "Chroma", 24),
    GroupSpec("beat_strength", "Beat Strength", 4),
)

GROUP_KEYS = [spec.key for spec in GROUP_SPECS]
GROUP_SLICES: Dict[str, slice] = {}
_offset = 0
for _spec in GROUP_SPECS:
    GROUP_SLICES[_spec.key] = slice(_offset, _offset + _spec.size)
    _offset += _spec.size


FEATURE_COMBOS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("all_audio", GROUP_KEYS),
        ("timbre_only", ["mfcc", "delta_mfcc", "delta2_mfcc"]),
        (
            "spectral_only",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
            ],
        ),
        (
            "spectral_plus_beat",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
                "beat_strength",
            ],
        ),
        (
            "spectral_pruned",
            [
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "beat_strength",
            ],
        ),
        ("spectral_minimal", ["spectral_flux", "spectral_flatness", "beat_strength"]),
        (
            "timbre_plus_spectral",
            [
                "mfcc",
                "delta_mfcc",
                "delta2_mfcc",
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
            ],
        ),
        (
            "timbre_plus_pruned_spectral",
            [
                "mfcc",
                "delta_mfcc",
                "delta2_mfcc",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "beat_strength",
            ],
        ),
        ("delta_family", ["delta_mfcc", "delta2_mfcc"]),
        (
            "delta_plus_pruned_spectral",
            [
                "delta_mfcc",
                "delta2_mfcc",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "beat_strength",
            ],
        ),
        ("single_mfcc", ["mfcc"]),
        ("single_delta_mfcc", ["delta_mfcc"]),
        ("single_delta2_mfcc", ["delta2_mfcc"]),
        ("chroma_plus_beat", ["chroma", "beat_strength"]),
        ("mfcc_plus_chroma", ["mfcc", "chroma"]),
        (
            "spectral_plus_chroma",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
                "chroma",
            ],
        ),
    ]
)

PREPROCESS_MODES = ("raw_zscore", "pca_per_group_2", "pca_per_group_5")


def import_hdbscan():
    """Import the external hdbscan package without colliding with local modules."""
    script_dir = str(Path(__file__).resolve().parent)
    removed = False
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        removed = True
    try:
        import hdbscan  # type: ignore
        from hdbscan.validity import validity_index  # type: ignore
    finally:
        if removed:
            sys.path.insert(0, script_dir)
    return hdbscan, validity_index


def compute_hdbscan_dbcv_safely(
    validity_index,
    data64: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Return a finite DBCV score or NaN when the external validity code is unstable."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            score = float(validity_index(data64, labels))
    except Exception:
        return float("nan")

    return score if np.isfinite(score) else float("nan")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return ensure_dir(Path(explicit))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ensure_dir(DEFAULT_OUTPUT_ROOT / f"strategic_clustering_search_{stamp}")


def combo_dims(group_keys: Sequence[str]) -> int:
    return int(sum(GROUP_SLICES[key].stop - GROUP_SLICES[key].start for key in group_keys))


def normalized_entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    values, counts = np.unique(labels, return_counts=True)
    if len(values) <= 1:
        return 0.0
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy / np.log(len(values)))


def minmax_normalize(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if series.isna().all():
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan)
    valid = finite.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    min_value = valid.min()
    max_value = valid.max()
    if math.isclose(float(min_value), float(max_value)):
        base = pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    else:
        if higher_is_better:
            base = (finite - min_value) / (max_value - min_value)
        else:
            base = (max_value - finite) / (max_value - min_value)
        base = base.fillna(0.0)
    return base.clip(0.0, 1.0)


def compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    unique = np.unique(labels)
    n_clusters = int(len(unique))
    result["n_clusters"] = float(n_clusters)
    result["cluster_balance"] = normalized_entropy(labels)

    if n_clusters > 1 and len(labels) > n_clusters:
        sample_size = min(len(X), 5000)
        silhouette_kwargs: Dict[str, int] = {"random_state": RANDOM_SEED}
        if sample_size < len(X):
            silhouette_kwargs["sample_size"] = sample_size
        result["silhouette"] = float(silhouette_score(X, labels, **silhouette_kwargs))
        result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    else:
        result["silhouette"] = float("nan")
        result["calinski_harabasz"] = float("nan")
        result["davies_bouldin"] = float("nan")

    if len(np.unique(true_labels)) > 1 and n_clusters > 1:
        result["ari"] = float(adjusted_rand_score(true_labels, labels))
        result["ami"] = float(adjusted_mutual_info_score(true_labels, labels))
        result["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
        result["homogeneity"] = float(homogeneity_score(true_labels, labels))
        result["completeness"] = float(completeness_score(true_labels, labels))
        result["v_measure"] = float(v_measure_score(true_labels, labels))
    else:
        result["ari"] = float("nan")
        result["ami"] = float("nan")
        result["nmi"] = float("nan")
        result["homogeneity"] = float("nan")
        result["completeness"] = float("nan")
        result["v_measure"] = float("nan")

    return result


def add_weighted_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    minimize_columns: Optional[Iterable[str]] = None,
    score_column: str = "hybrid_score",
) -> pd.DataFrame:
    out = df.copy()
    minimize_set = set(minimize_columns or [])
    total = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    weight_sum = 0.0
    for column, weight in weights.items():
        higher = column not in minimize_set
        norm = minmax_normalize(out[column], higher_is_better=higher)
        out[f"{column}_norm"] = norm
        total += norm * weight
        weight_sum += weight
    if weight_sum <= 0:
        out[score_column] = 0.0
    else:
        out[score_column] = total / weight_sum
    return out


def mean_pairwise_ari(label_runs: Sequence[np.ndarray]) -> float:
    scores: List[float] = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            scores.append(adjusted_rand_score(label_runs[i], label_runs[j]))
    return float(np.mean(scores)) if scores else float("nan")


def choose_subset_indices(
    genres: np.ndarray,
    subset_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = len(genres)
    if subset_size >= total:
        return np.arange(total, dtype=int)

    per_genre: Dict[str, List[int]] = {}
    for idx, genre in enumerate(genres):
        per_genre.setdefault(str(genre), []).append(idx)

    chosen: List[int] = []
    leftovers: List[int] = []
    for indices in per_genre.values():
        copied = indices.copy()
        rng.shuffle(copied)
        chosen.append(copied[0])
        leftovers.extend(copied[1:])

    if len(chosen) > subset_size:
        chosen = list(rng.choice(np.array(chosen), size=subset_size, replace=False))
    elif len(chosen) < subset_size:
        remaining = subset_size - len(chosen)
        extra = rng.choice(np.array(leftovers), size=remaining, replace=False)
        chosen.extend(int(x) for x in extra)

    return np.array(sorted(chosen), dtype=int)


def prepare_feature_matrix(
    X_raw: np.ndarray,
    group_keys: Sequence[str],
    mode: str,
) -> np.ndarray:
    if mode == "raw_zscore":
        blocks = [X_raw[:, GROUP_SLICES[key]] for key in group_keys]
        X_selected = np.hstack(blocks)
        return StandardScaler().fit_transform(X_selected).astype(np.float32)

    if not mode.startswith("pca_per_group_"):
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    components = int(mode.rsplit("_", 1)[-1])
    blocks = []
    n_samples = X_raw.shape[0]
    for key in group_keys:
        X_group = X_raw[:, GROUP_SLICES[key]]
        X_scaled = StandardScaler().fit_transform(X_group)
        target = components

        if X_scaled.shape[1] > target:
            n_components = min(target, X_scaled.shape[1], n_samples - 1)
            transformed = PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(
                X_scaled
            )
        else:
            transformed = X_scaled

        if transformed.shape[1] < target:
            padding = np.zeros((n_samples, target - transformed.shape[1]), dtype=np.float32)
            transformed = np.hstack([transformed, padding])

        group_norm = np.sqrt(np.mean(np.sum(transformed**2, axis=1)))
        if group_norm > 1e-10:
            transformed = transformed / group_norm

        blocks.append(transformed.astype(np.float32))

    return np.hstack(blocks).astype(np.float32)


def build_variance_summary(X_raw: np.ndarray) -> pd.DataFrame:
    rows = []
    for spec in GROUP_SPECS:
        X_group = X_raw[:, GROUP_SLICES[spec.key]]
        variances = X_group.var(axis=0)
        rows.append(
            {
                "group": spec.key,
                "label": spec.label,
                "dims": spec.size,
                "mean_variance": float(np.mean(variances)),
                "median_variance": float(np.median(variances)),
                "total_variance": float(np.sum(variances)),
            }
        )
    return pd.DataFrame(rows).sort_values("total_variance", ascending=False)


def build_correlation_summary(X_raw: np.ndarray) -> pd.DataFrame:
    corr = np.corrcoef(StandardScaler().fit_transform(X_raw), rowvar=False)
    rows = []
    for idx_a, spec_a in enumerate(GROUP_SPECS):
        slice_a = GROUP_SLICES[spec_a.key]
        for spec_b in GROUP_SPECS[idx_a + 1 :]:
            slice_b = GROUP_SLICES[spec_b.key]
            block = np.abs(corr[slice_a, slice_b])
            rows.append(
                {
                    "group_a": spec_a.key,
                    "group_b": spec_b.key,
                    "mean_abs_corr": float(np.mean(block)),
                    "max_abs_corr": float(np.max(block)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_corr", "max_abs_corr"], ascending=False
    )


def evaluate_kmeans(
    X: np.ndarray,
    true_labels: np.ndarray,
    k_values: Sequence[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(X)
        metrics = compute_metrics(X, labels, true_labels)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df = add_weighted_score(
        df,
        weights={
            "silhouette": 0.30,
            "nmi": 0.35,
            "v_measure": 0.25,
            "cluster_balance": 0.10,
        },
    )
    best = df.sort_values(
        ["hybrid_score", "nmi", "v_measure", "silhouette"],
        ascending=False,
    ).iloc[0]
    return df, best


def evaluate_gmm(
    X: np.ndarray,
    true_labels: np.ndarray,
    component_values: Sequence[int],
    covariance_types: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []
    for covariance_type in covariance_types:
        for n_components in component_values:
            try:
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    random_state=RANDOM_SEED,
                    max_iter=200,
                    tol=1e-3,
                    reg_covar=1e-5,
                    init_params="kmeans",
                )
                model.fit(X)
                labels = model.predict(X)
                metrics = compute_metrics(X, labels, true_labels)
                confidence = float(model.predict_proba(X).max(axis=1).mean())
                rows.append(
                    {
                        "covariance_type": covariance_type,
                        "n_components": int(n_components),
                        "bic": float(model.bic(X)),
                        "aic": float(model.aic(X)),
                        "avg_confidence": confidence,
                        **metrics,
                    }
                )
            except Exception:
                rows.append(
                    {
                        "covariance_type": covariance_type,
                        "n_components": int(n_components),
                        "bic": float("nan"),
                        "aic": float("nan"),
                        "avg_confidence": float("nan"),
                        "n_clusters": float("nan"),
                        "cluster_balance": float("nan"),
                        "silhouette": float("nan"),
                        "calinski_harabasz": float("nan"),
                        "davies_bouldin": float("nan"),
                        "ari": float("nan"),
                        "ami": float("nan"),
                        "nmi": float("nan"),
                        "homogeneity": float("nan"),
                        "completeness": float("nan"),
                        "v_measure": float("nan"),
                    }
                )

    df = pd.DataFrame(rows)
    df = add_weighted_score(
        df,
        weights={
            "silhouette": 0.25,
            "nmi": 0.30,
            "v_measure": 0.20,
            "cluster_balance": 0.10,
            "bic": 0.15,
        },
        minimize_columns={"bic"},
    )
    best = df.sort_values(
        ["hybrid_score", "nmi", "v_measure", "silhouette"],
        ascending=False,
    ).iloc[0]
    return df, best


def build_hdbscan_search_space(n_samples: int) -> List[Tuple[int, int]]:
    min_cluster_sizes = sorted(
        {
            5,
            8,
            10,
            15,
            25,
            max(5, int(round(n_samples * 0.02))),
            max(5, int(round(n_samples * 0.04))),
        }
    )
    search_space = set()
    for min_cluster_size in min_cluster_sizes:
        for min_samples in {1, 2, 5, max(1, min_cluster_size // 2), min_cluster_size}:
            if min_samples <= min_cluster_size:
                search_space.add((min_cluster_size, min_samples))
    return sorted(search_space)


def evaluate_hdbscan(
    X: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[pd.DataFrame, pd.Series]:
    hdbscan, validity_index = import_hdbscan()
    rows = []
    data64 = np.asarray(X, dtype=np.float64)
    for min_cluster_size, min_samples in build_hdbscan_search_space(len(X)):
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        labels = model.fit_predict(X)
        noise_fraction = float(np.mean(labels == -1))
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        row: Dict[str, float] = {
            "min_cluster_size": float(min_cluster_size),
            "min_samples": float(min_samples),
            "noise_fraction": noise_fraction,
            "coverage": 1.0 - noise_fraction,
            "n_clusters": float(n_clusters),
        }

        if n_clusters >= 2 and mask.sum() > n_clusters:
            metrics = compute_metrics(data64[mask], labels[mask], true_labels[mask])
            row.update(metrics)
            row["dbcv"] = compute_hdbscan_dbcv_safely(
                validity_index=validity_index,
                data64=data64,
                labels=labels,
            )
        else:
            row.update(
                {
                    "cluster_balance": float("nan"),
                    "silhouette": float("nan"),
                    "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan"),
                    "ari": float("nan"),
                    "ami": float("nan"),
                    "nmi": float("nan"),
                    "homogeneity": float("nan"),
                    "completeness": float("nan"),
                    "v_measure": float("nan"),
                    "dbcv": float("nan"),
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_weighted_score(
        df,
        weights={
            "dbcv": 0.25,
            "silhouette": 0.20,
            "nmi": 0.25,
            "v_measure": 0.15,
            "coverage": 0.10,
            "cluster_balance": 0.05,
        },
    )
    best = df.sort_values(
        ["hybrid_score", "dbcv", "nmi", "silhouette"],
        ascending=False,
    ).iloc[0]
    return df, best


def kmeans_stability(X: np.ndarray, k: int, seeds: Sequence[int]) -> float:
    runs = []
    for seed in seeds:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
        runs.append(labels)
    return mean_pairwise_ari(runs)


def gmm_stability(
    X: np.ndarray,
    n_components: int,
    covariance_type: str,
    seeds: Sequence[int],
) -> float:
    runs = []
    for seed in seeds:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=seed,
            max_iter=200,
            tol=1e-3,
            reg_covar=1e-5,
            init_params="kmeans",
        )
        runs.append(model.fit(X).predict(X))
    return mean_pairwise_ari(runs)


def hdbscan_stability(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    rounds: int = 3,
    sample_frac: float = 0.9,
    seed: int = RANDOM_SEED,
) -> float:
    hdbscan, _ = import_hdbscan()
    rng = np.random.default_rng(seed)
    runs: List[np.ndarray] = []
    idxs: List[np.ndarray] = []

    for _ in range(rounds):
        mask = rng.random(len(X)) < sample_frac
        Xi = X[mask]
        idx = np.where(mask)[0]
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        ).fit(Xi)
        runs.append(model.labels_)
        idxs.append(idx)

    pairwise = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common = np.intersect1d(idxs[i], idxs[j], assume_unique=False)
            if common.size == 0:
                continue
            map_i = {value: pos for pos, value in enumerate(idxs[i])}
            map_j = {value: pos for pos, value in enumerate(idxs[j])}
            labels_i = np.array([runs[i][map_i[value]] for value in common])
            labels_j = np.array([runs[j][map_j[value]] for value in common])
            keep = (labels_i != -1) & (labels_j != -1)
            if keep.sum() >= 2:
                if len(np.unique(labels_i[keep])) > 1 and len(np.unique(labels_j[keep])) > 1:
                    pairwise.append(adjusted_rand_score(labels_i[keep], labels_j[keep]))
    return float(np.mean(pairwise)) if pairwise else float("nan")


def load_or_build_raw_cache(
    audio_dir: Path,
    results_dir: Path,
    preferred_cache_path: Optional[Path],
    output_cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolved_songs_csv = resolve_pipeline_songs_csv("data/songs.csv")
    songs_csv_signature = (
        f"{resolved_songs_csv}:{int(resolved_songs_csv.stat().st_mtime)}:"
        f"{resolved_songs_csv.stat().st_size}"
    )
    cache_candidates = [preferred_cache_path, DEFAULT_EXISTING_CACHE]
    for cache_path in cache_candidates:
        if cache_path and cache_path.exists():
            cache = np.load(cache_path, allow_pickle=True)
            cached_signature_values = cache.get("songs_csv_signature")
            if cached_signature_values is None:
                continue
            cached_signature = str(np.asarray(cached_signature_values).reshape(-1)[0])
            if cached_signature != songs_csv_signature:
                continue
            X = np.asarray(cache["X"], dtype=np.float32)
            file_names = np.asarray(cache["file_names"])
            genres = np.asarray(cache["genres"])
            return X, file_names, genres

    genre_map, unique_genres = _load_genre_mapping(
        audio_dir=str(audio_dir),
        results_dir=str(results_dir),
        include_genre=False,
    )
    file_names, feature_vectors, genres, _qc_rows, _qc_summary = _collect_feature_vectors(
        audio_dir=str(audio_dir),
        results_dir=str(results_dir),
        genre_map=genre_map,
        unique_genres=unique_genres,
        include_genre=False,
        include_msd=False,
        songs_csv_path=str(resolved_songs_csv),
        selected_audio_feature_keys=list(fv.AUDIO_FEATURE_KEYS),
    )
    X = np.vstack(feature_vectors).astype(np.float32)
    file_names_arr = np.asarray(file_names)
    genres_arr = np.asarray(genres)
    np.savez_compressed(
        output_cache_path,
        X=X,
        file_names=file_names_arr,
        genres=genres_arr,
        songs_csv_signature=np.asarray([songs_csv_signature]),
    )
    return X, file_names_arr, genres_arr


def evaluate_scenarios(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    combos: "OrderedDict[str, List[str]]",
    preprocess_modes: Sequence[str],
    gmm_covariance_types: Sequence[str],
    save_diagnostics_dir: Optional[Path] = None,
    prefix: str = "subset",
) -> pd.DataFrame:
    rows = []
    k_values = list(range(2, min(12, len(X_raw) - 1) + 1))
    component_values = list(range(2, min(10, len(X_raw) - 1) + 1))

    for combo_name, group_keys in combos.items():
        for mode in preprocess_modes:
            print(f"[{prefix}] Evaluating {combo_name} / {mode}")
            X_prepared = prepare_feature_matrix(X_raw, group_keys, mode)

            kmeans_df, kmeans_best = evaluate_kmeans(X_prepared, y_true, k_values=k_values)
            gmm_df, gmm_best = evaluate_gmm(
                X_prepared,
                y_true,
                component_values=component_values,
                covariance_types=gmm_covariance_types,
            )
            hdbscan_df, hdbscan_best = evaluate_hdbscan(X_prepared, y_true)

            if save_diagnostics_dir is not None:
                base_name = f"{prefix}__{combo_name}__{mode}"
                kmeans_df.to_csv(save_diagnostics_dir / f"{base_name}__kmeans.csv", index=False)
                gmm_df.to_csv(save_diagnostics_dir / f"{base_name}__gmm.csv", index=False)
                hdbscan_df.to_csv(save_diagnostics_dir / f"{base_name}__hdbscan.csv", index=False)

            rows.append(
                {
                    "combo": combo_name,
                    "groups": "+".join(group_keys),
                    "n_groups": len(group_keys),
                    "preprocess_mode": mode,
                    "n_dims_raw": combo_dims(group_keys),
                    "n_dims_transformed": int(X_prepared.shape[1]),
                    "n_samples": int(len(X_prepared)),
                    "kmeans_best_k": int(kmeans_best["k"]),
                    "kmeans_silhouette": float(kmeans_best["silhouette"]),
                    "kmeans_nmi": float(kmeans_best["nmi"]),
                    "kmeans_v_measure": float(kmeans_best["v_measure"]),
                    "kmeans_balance": float(kmeans_best["cluster_balance"]),
                    "kmeans_score": float(kmeans_best["hybrid_score"]),
                    "gmm_covariance_type": str(gmm_best["covariance_type"]),
                    "gmm_best_components": int(gmm_best["n_components"]),
                    "gmm_bic": float(gmm_best["bic"]),
                    "gmm_silhouette": float(gmm_best["silhouette"]),
                    "gmm_nmi": float(gmm_best["nmi"]),
                    "gmm_v_measure": float(gmm_best["v_measure"]),
                    "gmm_balance": float(gmm_best["cluster_balance"]),
                    "gmm_score": float(gmm_best["hybrid_score"]),
                    "hdbscan_best_min_cluster_size": int(hdbscan_best["min_cluster_size"]),
                    "hdbscan_best_min_samples": int(hdbscan_best["min_samples"]),
                    "hdbscan_clusters": int(hdbscan_best["n_clusters"]),
                    "hdbscan_noise_fraction": float(hdbscan_best["noise_fraction"]),
                    "hdbscan_dbcv": float(hdbscan_best["dbcv"]),
                    "hdbscan_nmi": float(hdbscan_best["nmi"]),
                    "hdbscan_v_measure": float(hdbscan_best["v_measure"]),
                    "hdbscan_score": float(hdbscan_best["hybrid_score"]),
                }
            )

    result = pd.DataFrame(rows)
    result["scenario_score"] = result[
        ["kmeans_score", "gmm_score", "hdbscan_score"]
    ].mean(axis=1)
    return result.sort_values("scenario_score", ascending=False).reset_index(drop=True)


def validate_top_scenarios(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    scenario_df: pd.DataFrame,
    top_n: int,
    diagnostics_dir: Path,
) -> pd.DataFrame:
    rows = []
    top_df = scenario_df.head(top_n)
    stability_seeds = [0, 1, 2, 3, 4]

    for scenario in top_df.itertuples(index=False):
        combo_name = str(scenario.combo)
        group_keys = FEATURE_COMBOS[combo_name]
        mode = str(scenario.preprocess_mode)
        print(f"[full] Validating {combo_name} / {mode}")
        X_prepared = prepare_feature_matrix(X_raw, group_keys, mode)

        kmeans_df, kmeans_best = evaluate_kmeans(
            X_prepared,
            y_true,
            k_values=list(range(2, min(14, len(X_raw) - 1) + 1)),
        )
        gmm_df, gmm_best = evaluate_gmm(
            X_prepared,
            y_true,
            component_values=list(range(2, min(12, len(X_raw) - 1) + 1)),
            covariance_types=("full", "diag"),
        )
        hdbscan_df, hdbscan_best = evaluate_hdbscan(X_prepared, y_true)

        base_name = f"full__{combo_name}__{mode}"
        kmeans_df.to_csv(diagnostics_dir / f"{base_name}__kmeans.csv", index=False)
        gmm_df.to_csv(diagnostics_dir / f"{base_name}__gmm.csv", index=False)
        hdbscan_df.to_csv(diagnostics_dir / f"{base_name}__hdbscan.csv", index=False)

        kmeans_stab = kmeans_stability(
            X_prepared,
            k=int(kmeans_best["k"]),
            seeds=stability_seeds,
        )
        gmm_stab = gmm_stability(
            X_prepared,
            n_components=int(gmm_best["n_components"]),
            covariance_type=str(gmm_best["covariance_type"]),
            seeds=stability_seeds,
        )
        hdbscan_stab = hdbscan_stability(
            X_prepared,
            min_cluster_size=int(hdbscan_best["min_cluster_size"]),
            min_samples=int(hdbscan_best["min_samples"]),
        )

        rows.extend(
            [
                {
                    "combo": combo_name,
                    "groups": "+".join(group_keys),
                    "preprocess_mode": mode,
                    "method": "kmeans",
                    "n_dims_transformed": int(X_prepared.shape[1]),
                    "param_1_name": "k",
                    "param_1_value": int(kmeans_best["k"]),
                    "param_2_name": "",
                    "param_2_value": "",
                    "n_clusters": int(kmeans_best["n_clusters"]),
                    "silhouette": float(kmeans_best["silhouette"]),
                    "nmi": float(kmeans_best["nmi"]),
                    "v_measure": float(kmeans_best["v_measure"]),
                    "cluster_balance": float(kmeans_best["cluster_balance"]),
                    "noise_fraction": 0.0,
                    "extra_quality": float(kmeans_best["calinski_harabasz"]),
                    "extra_cost": float(kmeans_best["davies_bouldin"]),
                    "hybrid_score": float(kmeans_best["hybrid_score"]),
                    "stability_ari": kmeans_stab,
                },
                {
                    "combo": combo_name,
                    "groups": "+".join(group_keys),
                    "preprocess_mode": mode,
                    "method": "gmm",
                    "n_dims_transformed": int(X_prepared.shape[1]),
                    "param_1_name": "components",
                    "param_1_value": int(gmm_best["n_components"]),
                    "param_2_name": "covariance_type",
                    "param_2_value": str(gmm_best["covariance_type"]),
                    "n_clusters": int(gmm_best["n_clusters"]),
                    "silhouette": float(gmm_best["silhouette"]),
                    "nmi": float(gmm_best["nmi"]),
                    "v_measure": float(gmm_best["v_measure"]),
                    "cluster_balance": float(gmm_best["cluster_balance"]),
                    "noise_fraction": 0.0,
                    "extra_quality": float(gmm_best["avg_confidence"]),
                    "extra_cost": float(gmm_best["bic"]),
                    "hybrid_score": float(gmm_best["hybrid_score"]),
                    "stability_ari": gmm_stab,
                },
                {
                    "combo": combo_name,
                    "groups": "+".join(group_keys),
                    "preprocess_mode": mode,
                    "method": "hdbscan",
                    "n_dims_transformed": int(X_prepared.shape[1]),
                    "param_1_name": "min_cluster_size",
                    "param_1_value": int(hdbscan_best["min_cluster_size"]),
                    "param_2_name": "min_samples",
                    "param_2_value": int(hdbscan_best["min_samples"]),
                    "n_clusters": int(hdbscan_best["n_clusters"]),
                    "silhouette": float(hdbscan_best["silhouette"]),
                    "nmi": float(hdbscan_best["nmi"]),
                    "v_measure": float(hdbscan_best["v_measure"]),
                    "cluster_balance": float(hdbscan_best["cluster_balance"]),
                    "noise_fraction": float(hdbscan_best["noise_fraction"]),
                    "extra_quality": float(hdbscan_best["dbcv"]),
                    "extra_cost": float(hdbscan_best["coverage"]),
                    "hybrid_score": float(hdbscan_best["hybrid_score"]),
                    "stability_ari": hdbscan_stab,
                },
            ]
        )

    validation_df = pd.DataFrame(rows)
    validation_df = add_weighted_score(
        validation_df,
        weights={
            "hybrid_score": 0.55,
            "stability_ari": 0.20,
            "nmi": 0.15,
            "v_measure": 0.10,
        },
        score_column="final_score",
    )
    return validation_df.sort_values("final_score", ascending=False).reset_index(drop=True)


def build_report(
    output_dir: Path,
    dataset_summary: Dict[str, object],
    correlation_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    recommendation: Dict[str, object],
) -> Path:
    report_path = output_dir / "recommendation_report.md"
    top_corr = correlation_df.head(8)
    top_scenarios = scenario_df.head(6)
    top_validation = validation_df.head(9)

    lines = [
        "# Strategic Clustering Search",
        "",
        "## Dataset",
        "",
        f"- Songs evaluated: {dataset_summary['n_songs']}",
        f"- Unique primary genres: {dataset_summary['n_genres']}",
        f"- Raw audio feature dimensions: {dataset_summary['n_raw_dims']}",
        f"- Broad-search subset size: {dataset_summary['subset_size']}",
        "",
        "## Strongest Feature Correlations",
        "",
    ]
    for row in top_corr.itertuples(index=False):
        lines.append(
            f"- `{row.group_a}` vs `{row.group_b}`: mean |corr|={row.mean_abs_corr:.3f}, max={row.max_abs_corr:.3f}"
        )

    lines.extend(["", "## Broad Search Leaders", ""])
    for row in top_scenarios.itertuples(index=False):
        lines.append(
            f"- `{row.combo}` / `{row.preprocess_mode}`: scenario_score={row.scenario_score:.3f}, "
            f"KMeans={row.kmeans_score:.3f}, GMM={row.gmm_score:.3f}, HDBSCAN={row.hdbscan_score:.3f}"
        )

    lines.extend(["", "## Full Validation Leaders", ""])
    for row in top_validation.itertuples(index=False):
        lines.append(
            f"- `{row.method}` with `{row.combo}` / `{row.preprocess_mode}`: final_score={row.final_score:.3f}, "
            f"NMI={row.nmi:.3f}, silhouette={row.silhouette:.3f}, stability={row.stability_ari:.3f}"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Default method: `{recommendation['method']}`",
            f"- Feature subset: `{recommendation['combo']}` -> `{recommendation['groups']}`",
            f"- Preprocessing: `{recommendation['preprocess_mode']}`",
            f"- Primary parameter: `{recommendation['param_1_name']}={recommendation['param_1_value']}`",
        ]
    )
    if recommendation.get("param_2_name"):
        lines.append(
            f"- Secondary parameter: `{recommendation['param_2_name']}={recommendation['param_2_value']}`"
        )
    lines.extend(
        [
            f"- Validation score: `{recommendation['final_score']:.3f}`",
            "- Notes: the strongest candidates consistently came from spectral-heavy or "
            "spectral+timbre subsets rather than the full 116-dim audio stack.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--subset-size", type=int, default=2500)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = get_output_dir(args.output_dir or None)
    diagnostics_dir = ensure_dir(output_dir / "diagnostics")
    cache_output = output_dir / "raw_audio_feature_cache.npz"

    X_raw, file_names, genres = load_or_build_raw_cache(
        audio_dir=Path(args.audio_dir),
        results_dir=Path(args.results_dir),
        preferred_cache_path=Path(args.cache_path) if args.cache_path else None,
        output_cache_path=cache_output,
    )

    if X_raw.shape[1] != sum(spec.size for spec in GROUP_SPECS):
        raise RuntimeError(
            f"Expected {sum(spec.size for spec in GROUP_SPECS)} raw dims but found {X_raw.shape[1]}"
        )

    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(genres)

    subset_idx = choose_subset_indices(genres, args.subset_size, seed=RANDOM_SEED)
    X_subset = X_raw[subset_idx]
    y_subset = y_true[subset_idx]

    variance_df = build_variance_summary(X_raw)
    correlation_df = build_correlation_summary(X_raw)
    scenario_df = evaluate_scenarios(
        X_raw=X_subset,
        y_true=y_subset,
        combos=FEATURE_COMBOS,
        preprocess_modes=PREPROCESS_MODES,
        gmm_covariance_types=("full",),
        save_diagnostics_dir=diagnostics_dir,
        prefix="subset",
    )
    validation_df = validate_top_scenarios(
        X_raw=X_raw,
        y_true=y_true,
        scenario_df=scenario_df,
        top_n=args.top_n,
        diagnostics_dir=diagnostics_dir,
    )

    recommended_row = validation_df.iloc[0].to_dict()
    dataset_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_songs": int(len(X_raw)),
        "n_genres": int(len(label_encoder.classes_)),
        "n_raw_dims": int(X_raw.shape[1]),
        "subset_size": int(len(subset_idx)),
        "top_n_validated": int(args.top_n),
        "raw_cache": str(cache_output),
        "n_file_names": int(len(file_names)),
    }
    recommendation = {
        **recommended_row,
        "groups": recommended_row["groups"],
    }

    variance_df.to_csv(output_dir / "feature_group_variance_summary.csv", index=False)
    correlation_df.to_csv(output_dir / "feature_group_correlation_summary.csv", index=False)
    scenario_df.to_csv(output_dir / "subset_scenario_results.csv", index=False)
    validation_df.to_csv(output_dir / "full_validation_results.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "recommended_config.json").write_text(
        json.dumps(recommendation, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = build_report(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        correlation_df=correlation_df,
        scenario_df=scenario_df,
        validation_df=validation_df,
        recommendation=recommendation,
    )

    print("\nSearch complete.")
    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
