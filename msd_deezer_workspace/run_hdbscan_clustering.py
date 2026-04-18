from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

try:
    from tqdm.auto import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    def tqdm(iterable, total=None, desc=None, **_kwargs):
        return iterable

from clustering.shared import (
    DEFAULT_FEATURES_DIR,
    PreparedDataset,
    default_algorithm_output_dir,
    ensure_output_dir,
    prepare_dataset,
    write_assignments,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic HDBSCAN clustering on extracted audio features.")
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES_DIR), help="Path to the features directory or feature summary CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for HDBSCAN outputs. Defaults to cluster_results/<feature_source>/hdbscan so pretrained-embedding and audio-feature runs stay separate.")
    parser.add_argument("--limit", type=int, help="Optional cap on how many songs to cluster.")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99, help="Explained variance target used for PCA reduction.")
    parser.add_argument("--max-pca-components", type=int, default=100, help="Maximum PCA dimensions kept for clustering.")
    parser.add_argument("--umap-n-components", type=int, default=15, help="UMAP target dimensions for clustering.")
    parser.add_argument("--umap-n-neighbors", type=int, default=40, help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap-min-dist", type=float, default=0.01, help="UMAP min_dist parameter.")
    parser.add_argument("--disable-umap", action="store_true", help="Skip UMAP and cluster on PCA output directly.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for the HDBSCAN grid search (default: 8).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Item 14 / 16: Build candidate hyperparameters
#   - min_cluster_size focused on 10-40 range (paper-recommended)
#   - explicit min_samples 3-10 (with proportional scaling for larger sizes)
#   - both "eom" and "leaf", with eom preferred at selection time
# ---------------------------------------------------------------------------

def build_hdbscan_candidates(n_samples: int) -> list[tuple[int, int, str]]:
    size_candidates = {
        10, 15, 20, 25, 30, 35, 40,
        max(10, n_samples // 200),
        max(10, n_samples // 100),
        max(10, int(np.sqrt(n_samples) / 2)),
    }
    size_candidates = {v for v in size_candidates if 2 <= v < n_samples}

    candidates: set[tuple[int, int, str]] = set()
    for min_cluster_size in sorted(size_candidates):
        min_samples_options = {
            3, 5, 7, 10,
            max(3, min_cluster_size // 3),
            max(3, min_cluster_size // 5),
        }
        for min_samples in min_samples_options:
            candidates.add((min_cluster_size, min_samples, "eom"))
            candidates.add((min_cluster_size, min_samples, "leaf"))

    return sorted(candidates, key=lambda t: (t[0], t[1], 0 if t[2] == "eom" else 1))


def _fit_hdbscan_candidate(
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
    x: np.ndarray,
) -> tuple[HDBSCAN, np.ndarray, dict[str, float | int | None]]:
    """Fit one HDBSCAN candidate and score it. Module-level so joblib can pickle it."""
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        copy=True,
    )
    labels = model.fit_predict(x)
    clustered_mask = labels != -1
    cluster_count = int(len(set(labels[clustered_mask]))) if clustered_mask.any() else 0
    clustered_fraction = float(clustered_mask.mean())
    noise_fraction = 1.0 - clustered_fraction
    sil_score = np.nan

    if cluster_count >= 2 and clustered_mask.sum() > cluster_count:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                sil_score = float(silhouette_score(x[clustered_mask], labels[clustered_mask]))
        except Exception:
            sil_score = np.nan

    record = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_selection_method": cluster_selection_method,
        "cluster_count": cluster_count,
        "clustered_fraction": clustered_fraction,
        "noise_fraction": noise_fraction,
        "silhouette_score": sil_score,
    }
    return model, labels, record


def select_best_hdbscan(
    dataset: PreparedDataset, workers: int = 8,
) -> tuple[HDBSCAN, np.ndarray, pd.DataFrame]:
    x = dataset.reduced_matrix
    candidates = build_hdbscan_candidates(len(x))
    records: list[dict[str, float | int | None]] = []
    best_model: HDBSCAN | None = None
    best_labels: np.ndarray | None = None
    best_key: tuple[float, float, float] | None = None
    fallback_model: HDBSCAN | None = None
    fallback_labels: np.ndarray | None = None
    fallback_key: tuple[float,] | None = None

    print(
        f"[HDBSCAN] Evaluating {len(candidates)} hyperparameter combinations "
        f"using {workers} worker(s)"
    )

    jobs = [
        delayed(_fit_hdbscan_candidate)(mcs, ms, method, x)
        for (mcs, ms, method) in candidates
    ]
    try:
        results_iter = Parallel(
            n_jobs=workers, return_as="generator_unordered",
        )(jobs)
    except TypeError:
        results_iter = Parallel(n_jobs=workers)(jobs)

    for model, labels, record in tqdm(
        results_iter, total=len(jobs), desc="[HDBSCAN] Grid search",
    ):
        records.append(record)

        clustered_fraction = record["clustered_fraction"]
        noise_fraction = record["noise_fraction"]
        cluster_count = record["cluster_count"]
        sil_score = record["silhouette_score"]

        fallback_score = (clustered_fraction,)
        if fallback_key is None or fallback_score > fallback_key:
            fallback_key = fallback_score
            fallback_model = model
            fallback_labels = labels

        if cluster_count < 2:
            continue

        selection_validity = sil_score if np.isfinite(sil_score) else -1.0

        # Item 16 – small bonus for "eom" (Excess of Mass) method
        eom_bonus = 0.01 if record["cluster_selection_method"] == "eom" else 0.0

        selection_key = (selection_validity + eom_bonus, clustered_fraction, -noise_fraction)
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_model = model
            best_labels = labels

    if best_model is None or best_labels is None:
        if fallback_model is None or fallback_labels is None:
            raise RuntimeError("HDBSCAN automatic selection could not produce a clustering model.")
        best_model = fallback_model
        best_labels = fallback_labels

    n_noise = int((best_labels == -1).sum())
    n_clusters = len(set(best_labels[best_labels != -1]))
    print(f"[HDBSCAN] Selected: min_cluster_size={best_model.min_cluster_size}, "
          f"min_samples={best_model.min_samples}, method={best_model.cluster_selection_method}, "
          f"clusters={n_clusters}, noise={n_noise}/{len(best_labels)}")

    metrics_frame = pd.DataFrame(records).sort_values(
        by=["silhouette_score", "clustered_fraction", "cluster_count"],
        ascending=[False, False, False],
        na_position="last",
    )
    return best_model, best_labels, metrics_frame


# ---------------------------------------------------------------------------
# Item 15: GLOSH-inspired soft noise reassignment
# ---------------------------------------------------------------------------

def reassign_noise_soft(
    x: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign noise points to the nearest cluster centroid, with a continuous
    confidence score based on how close the noise point is relative to the
    typical intra-cluster distances (inspired by GLOSH outlier scoring).

    Returns (reassigned_labels, confidence_array).
    Clustered points get confidence = 1.0; noise points get a score in [0, 1].
    """
    reassigned = labels.copy()
    full_confidence = np.ones(len(labels), dtype=np.float64)
    noise_mask = labels == -1

    if not noise_mask.any():
        return reassigned, full_confidence

    cluster_ids = sorted(set(labels[~noise_mask]))
    if not cluster_ids:
        return reassigned, full_confidence

    # Compute cluster centroids
    centroids = np.array([x[labels == cid].mean(axis=0) for cid in cluster_ids])

    # Reference distribution: intra-cluster distances
    clustered_dists: list[float] = []
    for cid_idx, cid in enumerate(cluster_ids):
        pts = x[labels == cid]
        d = np.linalg.norm(pts - centroids[cid_idx], axis=1)
        clustered_dists.extend(d.tolist())
    dist_p95 = float(np.percentile(clustered_dists, 95))

    # Process noise points
    noise_points = x[noise_mask]
    distances = np.linalg.norm(
        noise_points[:, np.newaxis] - centroids[np.newaxis, :], axis=2,
    )
    min_distances = distances.min(axis=1)
    nearest_idx = distances.argmin(axis=1)

    # Confidence: 1 when distance=0, → 0 as distance approaches / exceeds p95
    noise_confidence = np.clip(1.0 - (min_distances / (dist_p95 + 1e-12)), 0.0, 1.0)

    # Always assign to nearest centroid (for complete coverage)
    noise_indices = np.where(noise_mask)[0]
    for i, idx in enumerate(noise_indices):
        reassigned[idx] = cluster_ids[nearest_idx[i]]
    full_confidence[noise_mask] = noise_confidence

    return reassigned, full_confidence


def build_outputs(
    dataset: PreparedDataset,
    model: HDBSCAN,
    original_labels: np.ndarray,
    metrics_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    probabilities = np.asarray(model.probabilities_, dtype=np.float64)
    is_noise = original_labels == -1

    # Item 15 – soft noise reassignment
    reassigned_labels, reassignment_confidence = reassign_noise_soft(
        dataset.reduced_matrix, original_labels,
    )

    assignments = dataset.metadata.copy()
    assignments["cluster_label"] = reassigned_labels
    assignments["original_cluster_label"] = original_labels
    assignments["membership_probability"] = probabilities
    assignments["reassignment_confidence"] = reassignment_confidence
    assignments["outlier_score"] = 1.0 - reassignment_confidence
    assignments["is_noise"] = is_noise

    cluster_sizes = assignments["cluster_label"].value_counts().sort_index()
    assignments["cluster_size"] = assignments["cluster_label"].map(cluster_sizes.to_dict())

    write_assignments(assignments, output_dir / "cluster_assignments.csv")
    metrics_frame.to_csv(output_dir / "selection_metrics.csv", index=False)
    cluster_sizes.rename_axis("cluster_label").reset_index(name="size").to_csv(
        output_dir / "cluster_summary.csv",
        index=False,
    )

    core_cluster_labels = {int(label) for label in np.unique(original_labels) if int(label) != -1}
    selected_rows = metrics_frame.loc[
        (metrics_frame["min_cluster_size"] == int(model.min_cluster_size))
        & (
            metrics_frame["min_samples"].fillna(-1)
            == (-1 if model.min_samples is None else int(model.min_samples))
        )
        & (metrics_frame["cluster_selection_method"] == model.cluster_selection_method)
        & (metrics_frame["cluster_count"] == int(len(core_cluster_labels)))
    ]
    selected_row = selected_rows.iloc[0] if not selected_rows.empty else None
    payload = {
        "algorithm": "hdbscan",
        "cluster_count": int(len(core_cluster_labels)),
        "samples": int(len(assignments)),
        "original_noise_points": int(is_noise.sum()),
        "noise_reassigned": True,
        "reassignment_method": "glosh_inspired_soft",
        "feature_source": str(dataset.source_path),
        "summary_csv_path": str(dataset.summary_csv_path) if dataset.summary_csv_path else None,
        "pca_components": int(dataset.pca_components),
        "pca_explained_variance_ratio": float(dataset.pca_explained_variance_ratio),
        "umap_components": dataset.umap_components,
        "umap_n_neighbors": dataset.umap_n_neighbors,
        "umap_min_dist": dataset.umap_min_dist,
        "clustering_dimensions": int(dataset.reduced_matrix.shape[1]),
        "min_cluster_size": int(model.min_cluster_size),
        "min_samples": None if model.min_samples is None else int(model.min_samples),
        "cluster_selection_method": model.cluster_selection_method,
        "silhouette_score_clustered_only": float(selected_row["silhouette_score"]) if selected_row is not None else float("nan"),
    }
    write_json(payload, output_dir / "run_metadata.json")
    return payload


def main() -> None:
    args = parse_args()
    resolved_output_dir = args.output_dir or default_algorithm_output_dir(args.features_path, "hdbscan")
    output_dir = ensure_output_dir(resolved_output_dir)
    dataset = prepare_dataset(
        features_path=args.features_path,
        limit=args.limit,
        pca_variance_threshold=args.pca_variance_threshold,
        max_pca_components=args.max_pca_components,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        disable_umap=args.disable_umap,
    )
    model, labels, metrics_frame = select_best_hdbscan(dataset, workers=args.workers)
    payload = build_outputs(dataset, model, labels, metrics_frame, output_dir)
    print(payload)


if __name__ == "__main__":
    main()
