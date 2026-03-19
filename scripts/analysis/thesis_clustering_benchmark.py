"""Run a pilot-free thesis clustering benchmark over a prespecified feature space.

This script treats the thesis benchmark as the main experiment rather than an
extension of any earlier exploratory search. The representation space is
defined up front from standard MIR feature families, then evaluated directly on
the full dataset with:

- native operating-point comparison per method
- matched-granularity comparison across methods
- full metric logging across all representation / preprocessing settings
- runtime and stability reporting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scripts.analysis.strategic_clustering_search import (  # noqa: E402
    DEFAULT_AUDIO_DIR,
    DEFAULT_EXISTING_CACHE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RESULTS_DIR,
    FEATURE_COMBOS,
    GROUP_SPECS,
    PREPROCESS_MODES,
    RANDOM_SEED,
    add_weighted_score,
    build_correlation_summary,
    build_hdbscan_search_space,
    build_variance_summary,
    compute_metrics,
    ensure_dir,
    gmm_stability,
    hdbscan_stability,
    import_hdbscan,
    kmeans_stability,
    load_or_build_raw_cache,
    prepare_feature_matrix,
)
from src.utils.song_metadata import build_audio_metadata_frame  # noqa: E402

DEFAULT_MATCHED_TARGETS: Tuple[int, ...] = (4, 8, 12, 16, 20)
DEFAULT_MATCHED_BAND_FRACTION = 0.25
DEFAULT_STABILITY_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


THESIS_FEATURE_COMBOS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("mfcc_only", ["mfcc"]),
        ("delta_mfcc_only", ["delta_mfcc"]),
        ("delta2_mfcc_only", ["delta2_mfcc"]),
        ("timbre_full", ["mfcc", "delta_mfcc", "delta2_mfcc"]),
        (
            "spectral_shape",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
            ],
        ),
        ("pitch_only", ["chroma"]),
        ("rhythm_only", ["beat_strength"]),
        ("pitch_rhythm", ["chroma", "beat_strength"]),
        ("timbre_pitch", ["mfcc", "delta_mfcc", "delta2_mfcc", "chroma"]),
        (
            "timbre_spectral",
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
            "spectral_pitch",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
                "chroma",
            ],
        ),
        (
            "spectral_rhythm",
            [
                "spectral_centroid",
                "spectral_rolloff",
                "spectral_flux",
                "spectral_flatness",
                "zero_crossing_rate",
                "beat_strength",
            ],
        ),
        ("timbre_pitch_rhythm", ["mfcc", "delta_mfcc", "delta2_mfcc", "chroma", "beat_strength"]),
        ("all_audio", list(FEATURE_COMBOS["all_audio"])),
    ]
)


REPRESENTATION_RATIONALES: Dict[str, str] = {
    "mfcc_only": "Canonical timbre baseline built from static cepstral coefficients.",
    "delta_mfcc_only": "First-order timbral dynamics without static cepstral context.",
    "delta2_mfcc_only": "Second-order timbral dynamics to test acceleration-style temporal structure.",
    "timbre_full": "Full classical timbre stack combining static, delta, and delta-delta MFCC information.",
    "spectral_shape": "Brightness and spectral-shape family using scalar spectral descriptors.",
    "pitch_only": "Pitch-class profile baseline using chroma only.",
    "rhythm_only": "Pulse-strength baseline using beat-strength descriptors only.",
    "pitch_rhythm": "Joint harmonic-rhythmic view without timbral descriptors.",
    "timbre_pitch": "Timbre with pitch information to test whether harmonic context strengthens timbral organization.",
    "timbre_spectral": "Timbre plus spectral-shape descriptors for a broad timbral-textural representation.",
    "spectral_pitch": "Spectral-shape plus pitch-class energy for brightness-harmonic interactions.",
    "spectral_rhythm": "Spectral-shape plus rhythm to test textural-pulse structure.",
    "timbre_pitch_rhythm": "Multifamily combination covering timbre, pitch, and rhythm without explicit spectral scalars.",
    "all_audio": "Full engineered audio stack used as the omnibus reference representation.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--songs-csv", default="data/songs.csv")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--matched-targets",
        nargs="+",
        type=int,
        default=list(DEFAULT_MATCHED_TARGETS),
    )
    parser.add_argument(
        "--matched-band-fraction",
        type=float,
        default=DEFAULT_MATCHED_BAND_FRACTION,
    )
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--max-components", type=int, default=20)
    return parser.parse_args()


def get_output_dir(explicit: Optional[str]) -> Path:
    if explicit:
        return ensure_dir(Path(explicit))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ensure_dir(DEFAULT_OUTPUT_ROOT / f"thesis_clustering_benchmark_{stamp}")


def normalize_artist_name(value: object, idx: int) -> str:
    text = str(value or "").strip().lower()
    if text:
        return text
    return f"__unknown_artist_{idx}"


def build_aligned_metadata(file_names: np.ndarray, songs_csv_path: str) -> pd.DataFrame:
    metadata = build_audio_metadata_frame(file_names, songs_csv_path=songs_csv_path)
    metadata = metadata.copy()
    metadata["Artist"] = metadata["Artist"].fillna("").astype(str)
    metadata["PrimaryGenre"] = metadata["PrimaryGenre"].fillna("unknown").astype(str)
    metadata["GenreList"] = metadata["GenreList"].fillna("").astype(str)
    metadata["artist_key"] = [
        normalize_artist_name(value, idx)
        for idx, value in enumerate(metadata["Artist"].tolist())
    ]
    return metadata


def cluster_size_diagnostics(labels: np.ndarray) -> Tuple[float, float]:
    mask = labels != -1
    if not mask.any():
        return float("nan"), float("nan")
    _, counts = np.unique(labels[mask], return_counts=True)
    singleton_fraction = float(np.mean(counts == 1))
    largest_cluster_fraction = float(counts.max() / counts.sum())
    return singleton_fraction, largest_cluster_fraction


def build_representation_catalog() -> pd.DataFrame:
    rows = []
    for combo_name, group_keys in THESIS_FEATURE_COMBOS.items():
        rows.append(
            {
                "combo": combo_name,
                "groups": "+".join(group_keys),
                "n_groups": int(len(group_keys)),
                "n_dims_raw": int(sum(spec.size for spec in GROUP_SPECS if spec.key in group_keys)),
                "rationale": REPRESENTATION_RATIONALES[combo_name],
            }
        )
    return pd.DataFrame(rows)


def evaluate_kmeans_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    k_values: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for k in k_values:
        start = perf_counter()
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(X)
        fit_time_sec = perf_counter() - start
        metrics = compute_metrics(X, labels, y_true)
        singleton_fraction, largest_cluster_fraction = cluster_size_diagnostics(labels)
        rows.append(
            {
                "method": "kmeans",
                "param_1_name": "k",
                "param_1_value": int(k),
                "param_2_name": "",
                "param_2_value": "",
                "fit_time_sec": fit_time_sec,
                "coverage": 1.0,
                "noise_fraction": 0.0,
                "dbcv": float("nan"),
                "bic": float("nan"),
                "aic": float("nan"),
                "avg_confidence": float("nan"),
                "singleton_fraction": singleton_fraction,
                "largest_cluster_fraction": largest_cluster_fraction,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_gmm_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    component_values: Sequence[int],
    covariance_types: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for covariance_type in covariance_types:
        for n_components in component_values:
            row = {
                "method": "gmm",
                "param_1_name": "components",
                "param_1_value": int(n_components),
                "param_2_name": "covariance_type",
                "param_2_value": str(covariance_type),
            }
            try:
                start = perf_counter()
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
                fit_time_sec = perf_counter() - start
                metrics = compute_metrics(X, labels, y_true)
                confidence = float(model.predict_proba(X).max(axis=1).mean())
                singleton_fraction, largest_cluster_fraction = cluster_size_diagnostics(labels)
                row.update(
                    {
                        "fit_time_sec": fit_time_sec,
                        "coverage": 1.0,
                        "noise_fraction": 0.0,
                        "dbcv": float("nan"),
                        "bic": float(model.bic(X)),
                        "aic": float(model.aic(X)),
                        "avg_confidence": confidence,
                        "singleton_fraction": singleton_fraction,
                        "largest_cluster_fraction": largest_cluster_fraction,
                        **metrics,
                    }
                )
            except Exception:
                row.update(
                    {
                        "fit_time_sec": float("nan"),
                        "coverage": 1.0,
                        "noise_fraction": 0.0,
                        "dbcv": float("nan"),
                        "bic": float("nan"),
                        "aic": float("nan"),
                        "avg_confidence": float("nan"),
                        "singleton_fraction": float("nan"),
                        "largest_cluster_fraction": float("nan"),
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
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_hdbscan_grid(
    X: np.ndarray,
    y_true: np.ndarray,
) -> pd.DataFrame:
    hdbscan, validity_index = import_hdbscan()
    data64 = np.asarray(X, dtype=np.float64)
    rows = []
    for min_cluster_size, min_samples in build_hdbscan_search_space(len(X)):
        start = perf_counter()
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        labels = model.fit_predict(X)
        fit_time_sec = perf_counter() - start
        noise_fraction = float(np.mean(labels == -1))
        coverage = 1.0 - noise_fraction
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        singleton_fraction, largest_cluster_fraction = cluster_size_diagnostics(labels)

        row = {
            "method": "hdbscan",
            "param_1_name": "min_cluster_size",
            "param_1_value": int(min_cluster_size),
            "param_2_name": "min_samples",
            "param_2_value": int(min_samples),
            "fit_time_sec": fit_time_sec,
            "coverage": coverage,
            "noise_fraction": noise_fraction,
            "bic": float("nan"),
            "aic": float("nan"),
            "avg_confidence": float("nan"),
            "singleton_fraction": singleton_fraction,
            "largest_cluster_fraction": largest_cluster_fraction,
            "n_clusters": float(n_clusters),
        }

        if n_clusters >= 2 and mask.sum() > n_clusters:
            metrics = compute_metrics(data64[mask], labels[mask], y_true[mask])
            row.update(metrics)
            try:
                row["dbcv"] = float(validity_index(data64, labels))
            except Exception:
                row["dbcv"] = float("nan")
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

    return pd.DataFrame(rows)


def add_internal_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (combo, preprocess_mode, method), group in df.groupby(
        ["combo", "preprocess_mode", "method"],
        dropna=False,
        sort=False,
    ):
        scored = group.copy()
        if method == "kmeans":
            scored = add_weighted_score(
                scored,
                weights={
                    "silhouette": 0.40,
                    "calinski_harabasz": 0.25,
                    "davies_bouldin": 0.20,
                    "cluster_balance": 0.15,
                },
                minimize_columns={"davies_bouldin"},
                score_column="internal_selection_score",
            )
        elif method == "gmm":
            scored = add_weighted_score(
                scored,
                weights={
                    "silhouette": 0.30,
                    "calinski_harabasz": 0.20,
                    "davies_bouldin": 0.15,
                    "cluster_balance": 0.15,
                    "bic": 0.20,
                },
                minimize_columns={"davies_bouldin", "bic"},
                score_column="internal_selection_score",
            )
        elif method == "hdbscan":
            scored = add_weighted_score(
                scored,
                weights={
                    "dbcv": 0.40,
                    "coverage": 0.25,
                    "silhouette": 0.20,
                    "cluster_balance": 0.15,
                },
                score_column="internal_selection_score",
            )
        else:
            scored["internal_selection_score"] = float("nan")
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def sort_for_native_selection(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "kmeans":
        return df.sort_values(
            ["internal_selection_score", "silhouette", "calinski_harabasz", "cluster_balance"],
            ascending=[False, False, False, False],
        )
    if method == "gmm":
        return df.sort_values(
            ["internal_selection_score", "silhouette", "avg_confidence", "bic"],
            ascending=[False, False, False, True],
        )
    if method == "hdbscan":
        return df.sort_values(
            ["internal_selection_score", "dbcv", "coverage", "silhouette"],
            ascending=[False, False, False, False],
        )
    return df.sort_values("internal_selection_score", ascending=False)


def select_native_best(grid_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (combo, preprocess_mode, method), group in grid_df.groupby(
        ["combo", "preprocess_mode", "method"],
        dropna=False,
        sort=False,
    ):
        ordered = sort_for_native_selection(group, method)
        best = ordered.iloc[0].to_dict()
        best["benchmark_mode"] = "native"
        rows.append(best)
    return pd.DataFrame(rows)


def select_matched_best(
    grid_df: pd.DataFrame,
    matched_targets: Sequence[int],
    band_fraction: float,
) -> pd.DataFrame:
    rows = []
    for (combo, preprocess_mode, method), group in grid_df.groupby(
        ["combo", "preprocess_mode", "method"],
        dropna=False,
        sort=False,
    ):
        for target in matched_targets:
            tolerance = max(1, int(round(target * band_fraction)))
            band = group.copy()
            band["cluster_gap"] = (band["n_clusters"] - target).abs()
            band = band[band["cluster_gap"] <= tolerance]
            if band.empty:
                continue
            band = band.sort_values(
                ["cluster_gap", "internal_selection_score", "silhouette"],
                ascending=[True, False, False],
            )
            best = band.iloc[0].to_dict()
            best["benchmark_mode"] = "matched_granularity"
            best["matched_target_clusters"] = int(target)
            best["matched_tolerance"] = int(tolerance)
            rows.append(best)
    return pd.DataFrame(rows)


def select_global_native_leaders(native_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if native_df.empty:
        return pd.DataFrame()
    for method, group in native_df.groupby("method", sort=False):
        ordered = sort_for_native_selection(group, str(method))
        rows.append(ordered.iloc[0].to_dict())
    return pd.DataFrame(rows)


def select_global_matched_leaders(matched_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if matched_df.empty:
        return pd.DataFrame()
    for (target, method), group in matched_df.groupby(
        ["matched_target_clusters", "method"],
        sort=False,
        dropna=False,
    ):
        ordered = group.sort_values(
            ["cluster_gap", "internal_selection_score", "silhouette", "stability_ari"],
            ascending=[True, False, False, False],
        )
        best = ordered.iloc[0].to_dict()
        best["matched_target_clusters"] = int(target)
        rows.append(best)
    return pd.DataFrame(rows)


def stability_cache_key(row: pd.Series) -> Tuple[object, ...]:
    return (
        row["combo"],
        row["preprocess_mode"],
        row["method"],
        row["param_1_name"],
        row["param_1_value"],
        row["param_2_name"],
        row["param_2_value"],
    )


def compute_stability_for_row(row: pd.Series, X: np.ndarray) -> Tuple[float, float]:
    start = perf_counter()
    if row["method"] == "kmeans":
        score = kmeans_stability(
            X,
            k=int(row["param_1_value"]),
            seeds=DEFAULT_STABILITY_SEEDS,
        )
    elif row["method"] == "gmm":
        score = gmm_stability(
            X,
            n_components=int(row["param_1_value"]),
            covariance_type=str(row["param_2_value"]),
            seeds=DEFAULT_STABILITY_SEEDS,
        )
    elif row["method"] == "hdbscan":
        score = hdbscan_stability(
            X,
            min_cluster_size=int(row["param_1_value"]),
            min_samples=int(row["param_2_value"]),
        )
    else:
        score = float("nan")
    return score, perf_counter() - start


def attach_stability(
    selected_df: pd.DataFrame,
    prepared_lookup: Dict[Tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    if selected_df.empty:
        return selected_df.copy()

    cache: Dict[Tuple[object, ...], Tuple[float, float]] = {}
    rows = []
    for _, row in selected_df.iterrows():
        key = stability_cache_key(row)
        if key not in cache:
            prepared = prepared_lookup[(str(row["combo"]), str(row["preprocess_mode"]))]
            cache[key] = compute_stability_for_row(row, prepared)
        stability_ari, stability_time_sec = cache[key]
        updated = row.to_dict()
        updated["stability_ari"] = stability_ari
        updated["stability_time_sec"] = stability_time_sec
        rows.append(updated)
    return pd.DataFrame(rows)


def evaluate_full_benchmark(
    X_raw: np.ndarray,
    y_true: np.ndarray,
    representation_catalog: pd.DataFrame,
    preprocess_modes: Sequence[str],
    matched_targets: Sequence[int],
    band_fraction: float,
    max_k: int,
    max_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = []
    prepared_lookup: Dict[Tuple[str, str], np.ndarray] = {}

    matched_ceiling = max(int(value) for value in matched_targets) if matched_targets else 2
    effective_max_k = max(int(max_k), matched_ceiling)
    effective_max_components = max(int(max_components), matched_ceiling)

    k_values = list(range(2, min(effective_max_k, len(X_raw) - 1) + 1))
    component_values = list(range(2, min(effective_max_components, len(X_raw) - 1) + 1))

    for scenario in representation_catalog.itertuples(index=False):
        combo = str(scenario.combo)
        group_keys = THESIS_FEATURE_COMBOS[combo]
        for preprocess_mode in preprocess_modes:
            print(f"[benchmark] Evaluating {combo} / {preprocess_mode}")
            X_prepared = prepare_feature_matrix(X_raw, group_keys, preprocess_mode)
            prepared_lookup[(combo, preprocess_mode)] = X_prepared

            method_frames = [
                evaluate_kmeans_grid(X_prepared, y_true, k_values),
                evaluate_gmm_grid(X_prepared, y_true, component_values, ("full", "diag")),
                evaluate_hdbscan_grid(X_prepared, y_true),
            ]
            for method_df in method_frames:
                method_df = method_df.copy()
                method_df["combo"] = combo
                method_df["groups"] = "+".join(group_keys)
                method_df["preprocess_mode"] = preprocess_mode
                method_df["representation_rationale"] = REPRESENTATION_RATIONALES[combo]
                method_df["n_dims_raw"] = int(
                    sum(spec.size for spec in GROUP_SPECS if spec.key in group_keys)
                )
                method_df["n_dims_transformed"] = int(X_prepared.shape[1])
                method_df["n_samples"] = int(len(X_prepared))
                all_rows.append(method_df)

    grid_df = pd.concat(all_rows, ignore_index=True)
    grid_df = add_internal_selection_scores(grid_df)
    native_df = attach_stability(select_native_best(grid_df), prepared_lookup)
    matched_df = attach_stability(
        select_matched_best(grid_df, matched_targets=matched_targets, band_fraction=band_fraction),
        prepared_lookup,
    )
    return grid_df, native_df, matched_df


def build_benchmark_report(
    output_dir: Path,
    dataset_summary: Dict[str, object],
    representation_catalog: pd.DataFrame,
    native_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    global_native_df: pd.DataFrame,
    global_matched_df: pd.DataFrame,
) -> Path:
    report_path = output_dir / "benchmark_report.md"
    lines = [
        "# Thesis Clustering Benchmark",
        "",
        "## Dataset",
        "",
        f"- Songs evaluated: {dataset_summary['n_songs']}",
        f"- Unique primary genres: {dataset_summary['n_genres']}",
        f"- Unique artists: {dataset_summary['n_artists']}",
        f"- Raw audio feature dimensions: {dataset_summary['n_raw_dims']}",
        f"- Representation families evaluated: {dataset_summary['n_representations']}",
        f"- Preprocessing modes evaluated: {dataset_summary['preprocess_modes']}",
        f"- Matched targets: {dataset_summary['matched_targets']}",
        "",
        "## Representation Space",
        "",
    ]

    for row in representation_catalog.itertuples(index=False):
        lines.append(
            f"- `{row.combo}` ({row.n_dims_raw} raw dims): {row.rationale}"
        )

    lines.extend(["", "## Global Native Leaders", ""])
    for row in global_native_df.sort_values("method").itertuples(index=False):
        lines.append(
            f"- `{row.method}` with `{row.combo}` / `{row.preprocess_mode}`: "
            f"n_clusters={int(row.n_clusters)}, silhouette={row.silhouette:.3f}, "
            f"NMI={row.nmi:.3f}, stability={row.stability_ari:.3f}, "
            f"fit_time={row.fit_time_sec:.3f}s"
        )

    lines.extend(["", "## Global Matched-Granularity Leaders", ""])
    if global_matched_df.empty:
        lines.append("- No matched-granularity candidates satisfied the requested cluster-count bands.")
    else:
        matched_sorted = global_matched_df.sort_values(
            ["matched_target_clusters", "method", "cluster_gap", "internal_selection_score"],
            ascending=[True, True, True, False],
        )
        for row in matched_sorted.itertuples(index=False):
            lines.append(
                f"- target={int(row.matched_target_clusters)} | `{row.method}` with "
                f"`{row.combo}` / `{row.preprocess_mode}`: "
                f"n_clusters={int(row.n_clusters)}, gap={int(row.cluster_gap)}, "
                f"silhouette={row.silhouette:.3f}, NMI={row.nmi:.3f}, "
                f"stability={row.stability_ari:.3f}"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This benchmark does not use any pilot shortlist or subset-stage promotion logic.",
            "- Every prespecified representation was evaluated directly on the full dataset.",
            "- Native operating-point selection uses internal-only method-specific criteria.",
            "- External genre-alignment metrics are reported as evaluation outputs, not as the selection target.",
            "- Full per-representation native and matched outputs are stored in CSV artifacts for later analysis.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = get_output_dir(args.output_dir or None)
    cache_output = output_dir / "raw_audio_feature_cache.npz"

    X_raw, file_names, genres = load_or_build_raw_cache(
        audio_dir=Path(args.audio_dir),
        results_dir=Path(args.results_dir),
        preferred_cache_path=Path(args.cache_path) if args.cache_path else DEFAULT_EXISTING_CACHE,
        output_cache_path=cache_output,
    )
    if not cache_output.exists():
        np.savez_compressed(cache_output, X=X_raw, file_names=file_names, genres=genres)

    metadata = build_aligned_metadata(file_names, songs_csv_path=args.songs_csv)
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(genres)

    variance_df = build_variance_summary(X_raw)
    correlation_df = build_correlation_summary(X_raw)
    representation_catalog = build_representation_catalog()

    grid_df, native_df, matched_df = evaluate_full_benchmark(
        X_raw=X_raw,
        y_true=y_true,
        representation_catalog=representation_catalog,
        preprocess_modes=PREPROCESS_MODES,
        matched_targets=args.matched_targets,
        band_fraction=args.matched_band_fraction,
        max_k=args.max_k,
        max_components=args.max_components,
    )
    global_native_df = select_global_native_leaders(native_df)
    global_matched_df = select_global_matched_leaders(matched_df)

    dataset_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_songs": int(len(X_raw)),
        "n_genres": int(len(label_encoder.classes_)),
        "n_artists": int(metadata["artist_key"].nunique()),
        "n_raw_dims": int(X_raw.shape[1]),
        "n_representations": int(len(representation_catalog)),
        "preprocess_modes": list(PREPROCESS_MODES),
        "matched_targets": [int(value) for value in args.matched_targets],
        "matched_band_fraction": float(args.matched_band_fraction),
        "raw_cache": str(cache_output),
        "songs_csv": str(args.songs_csv),
    }

    variance_df.to_csv(output_dir / "feature_group_variance_summary.csv", index=False)
    correlation_df.to_csv(output_dir / "feature_group_correlation_summary.csv", index=False)
    representation_catalog.to_csv(output_dir / "representation_catalog.csv", index=False)
    grid_df.to_csv(output_dir / "full_grid_results.csv", index=False)
    native_df.to_csv(output_dir / "native_best_results.csv", index=False)
    matched_df.to_csv(output_dir / "matched_granularity_results.csv", index=False)
    global_native_df.to_csv(output_dir / "global_native_leaders.csv", index=False)
    global_matched_df.to_csv(output_dir / "global_matched_leaders.csv", index=False)
    metadata.to_csv(output_dir / "aligned_metadata.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    report_path = build_benchmark_report(
        output_dir=output_dir,
        dataset_summary=dataset_summary,
        representation_catalog=representation_catalog,
        native_df=native_df,
        matched_df=matched_df,
        global_native_df=global_native_df,
        global_matched_df=global_matched_df,
    )

    print("\nThesis benchmark complete.")
    print(json.dumps({"output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
