from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

try:
    from tqdm.auto import tqdm
except ImportError:  # graceful fallback if tqdm is not installed
    def tqdm(iterable, total=None, desc=None, **_kwargs):
        return iterable

from clustering.shared import (
    DEFAULT_CLUSTER_OUTPUT_DIR,
    DEFAULT_FEATURES_DIR,
    PreparedDataset,
    candidate_cluster_counts,
    ensure_output_dir,
    prepare_dataset,
    write_assignments,
    write_json,
)


DEFAULT_OUTPUT_DIR = DEFAULT_CLUSTER_OUTPUT_DIR / "kmeans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic KMeans clustering on extracted audio features.")
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES_DIR), help="Path to the features directory or feature summary CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where KMeans outputs will be stored.")
    parser.add_argument("--limit", type=int, help="Optional cap on how many songs to cluster.")
    parser.add_argument("--max-clusters", type=int, default=60, help="Maximum cluster count to consider during automatic selection.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99, help="Explained variance target used for PCA reduction.")
    parser.add_argument("--max-pca-components", type=int, default=100, help="Maximum PCA dimensions kept for clustering.")
    parser.add_argument("--umap-n-components", type=int, default=15, help="UMAP target dimensions for clustering.")
    parser.add_argument("--umap-n-neighbors", type=int, default=40, help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap-min-dist", type=float, default=0.01, help="UMAP min_dist parameter.")
    parser.add_argument("--disable-umap", action="store_true", help="Skip UMAP and cluster on PCA output directly.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for the KMeans grid search (default: 8).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Item 8: Outlier removal using Local Outlier Factor
# ---------------------------------------------------------------------------

def remove_outliers(x: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    """Return boolean mask where True = inlier, using LOF."""
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    predictions = lof.fit_predict(x)
    return predictions == 1


# ---------------------------------------------------------------------------
# Item 5: BIC-like criterion for multi-criterion dynamic cluster selection
# ---------------------------------------------------------------------------

def compute_bic(x: np.ndarray, model: KMeans) -> float:
    """BIC approximation for KMeans: n*log(RSS/n) + k*d*log(n)."""
    n, d = x.shape
    k = model.n_clusters
    rss = model.inertia_
    return float(n * np.log(rss / n + 1e-12) + k * d * np.log(n))


def _fit_kmeans_candidate(
    cluster_count: int,
    x: np.ndarray,
    random_state: int,
    silhouette_sample_size: int,
) -> tuple[int, KMeans | None, dict[str, float | int] | None]:
    """Fit one KMeans candidate and score it. Module-level so joblib can pickle it."""
    model = KMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init=20,
        max_iter=500,
    )
    labels = model.fit_predict(x)
    if len(np.unique(labels)) < 2:
        return cluster_count, None, None

    cluster_sizes = np.bincount(labels, minlength=cluster_count)
    silhouette = float(
        silhouette_score(
            x, labels,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    )
    calinski = float(calinski_harabasz_score(x, labels))
    davies = float(davies_bouldin_score(x, labels))
    inertia = float(model.inertia_)
    bic = compute_bic(x, model)

    record = {
        "cluster_count": int(cluster_count),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski,
        "davies_bouldin_score": davies,
        "inertia": inertia,
        "bic": bic,
        "min_cluster_size": int(cluster_sizes.min()),
    }
    return cluster_count, model, record


def select_best_kmeans(
    dataset: PreparedDataset,
    max_clusters: int,
    random_state: int,
    workers: int = 8,
) -> tuple[KMeans, pd.DataFrame, np.ndarray]:
    raw_x = dataset.reduced_matrix

    # Item 8 – remove outliers with LOF before clustering
    inlier_mask = remove_outliers(raw_x)
    x_clean = raw_x[inlier_mask]
    if len(x_clean) < 10:
        x_clean = raw_x
        inlier_mask = np.ones(len(raw_x), dtype=bool)

    n_outliers = int((~inlier_mask).sum())
    print(f"[KMeans] LOF outlier removal: {n_outliers} outliers detected, "
          f"{len(x_clean)} inliers retained")

    # Item 7 – L2 normalise → spherical K-Means (cosine similarity)
    x = normalize(x_clean, norm="l2")

    candidates = candidate_cluster_counts(len(x), max_clusters=max_clusters)
    records: list[dict[str, float | int]] = []
    models: dict[int, KMeans] = {}
    silhouette_sample_size = min(len(x), 2000)

    jobs = [
        delayed(_fit_kmeans_candidate)(k, x, random_state, silhouette_sample_size)
        for k in candidates
    ]
    print(f"[KMeans] Fitting {len(jobs)} candidate(s) using {workers} worker(s)")
    try:
        results_iter = Parallel(
            n_jobs=workers, return_as="generator_unordered",
        )(jobs)
    except TypeError:
        # joblib < 1.3: no generator return mode, fall back to list
        results_iter = Parallel(n_jobs=workers)(jobs)

    for cluster_count, model, record in tqdm(
        results_iter, total=len(jobs), desc="[KMeans] Grid search",
    ):
        if model is None or record is None:
            continue
        records.append(record)
        models[int(cluster_count)] = model

    if not models:
        raise RuntimeError("KMeans automatic selection could not find a valid clustering solution.")

    # ------------------------------------------------------------------
    # Item 5 – multi-criterion consensus (dynamic cluster selection)
    # ------------------------------------------------------------------
    ks = sorted(models.keys())

    # Criterion 1: Kneedle on inertia
    inertias = [next(r["inertia"] for r in records if r["cluster_count"] == k) for k in ks]
    knee_locator = KneeLocator(ks, inertias, curve="convex", direction="decreasing", S=3.0)
    knee_k = knee_locator.knee

    # Criterion 2: best silhouette
    sil_k = int(max(records, key=lambda r: r["silhouette_score"])["cluster_count"])

    # Criterion 3: best Calinski-Harabasz
    ch_k = int(max(records, key=lambda r: r["calinski_harabasz_score"])["cluster_count"])

    # Criterion 4: best (lowest) Davies-Bouldin
    db_k = int(min(records, key=lambda r: r["davies_bouldin_score"])["cluster_count"])

    # Criterion 5: best (lowest) BIC
    bic_k = int(min(records, key=lambda r: r["bic"])["cluster_count"])

    # Weighted vote
    votes: dict[int, float] = {k: 0.0 for k in ks}
    if knee_k is not None and knee_k in models:
        votes[int(knee_k)] += 2.0
    votes[sil_k] += 2.0
    votes[ch_k] += 1.0
    votes[db_k] += 1.0
    votes[bic_k] += 1.5

    selected_k = max(
        votes.keys(),
        key=lambda k: (
            votes[k],
            next(r["silhouette_score"] for r in records if r["cluster_count"] == k),
        ),
    )

    print(f"[KMeans] Criterion votes - Knee={knee_k}, Silhouette={sil_k}, "
          f"CH={ch_k}, DB={db_k}, BIC={bic_k} -> selected k={selected_k}")

    best_model = models[selected_k]
    metrics_frame = pd.DataFrame(records).sort_values(
        by=["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"],
        ascending=[False, False, True],
    )
    return best_model, metrics_frame, inlier_mask


def build_outputs(
    dataset: PreparedDataset,
    model: KMeans,
    metrics_frame: pd.DataFrame,
    inlier_mask: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    # Predict on all points (normalised) – outliers get assigned to nearest centroid
    x_all_norm = normalize(dataset.reduced_matrix, norm="l2")
    all_labels = model.predict(x_all_norm)
    distances = model.transform(x_all_norm)
    assigned_distance = distances[np.arange(len(all_labels)), all_labels]

    outlier_mask = ~inlier_mask

    assignments = dataset.metadata.copy()
    assignments["cluster_label"] = all_labels
    assignments["distance_to_centroid"] = assigned_distance
    assignments["is_outlier"] = outlier_mask

    cluster_sizes = assignments["cluster_label"].value_counts().sort_index()
    assignments["cluster_size"] = assignments["cluster_label"].map(cluster_sizes.to_dict())

    write_assignments(assignments, output_dir / "cluster_assignments.csv")
    metrics_frame.to_csv(output_dir / "selection_metrics.csv", index=False)
    cluster_sizes.rename_axis("cluster_label").reset_index(name="size").to_csv(
        output_dir / "cluster_summary.csv",
        index=False,
    )
    np.save(output_dir / "cluster_centers_reduced.npy", model.cluster_centers_)

    selected_row = metrics_frame.loc[metrics_frame["cluster_count"] == int(model.n_clusters)]
    selected_silhouette = float(selected_row.iloc[0]["silhouette_score"]) if not selected_row.empty else float("nan")
    payload = {
        "algorithm": "kmeans",
        "cluster_count": int(model.n_clusters),
        "samples": int(len(assignments)),
        "outliers_detected": int(outlier_mask.sum()),
        "outliers_reassigned": True,
        "feature_source": str(dataset.source_path),
        "summary_csv_path": str(dataset.summary_csv_path) if dataset.summary_csv_path else None,
        "pca_components": int(dataset.pca_components),
        "pca_explained_variance_ratio": float(dataset.pca_explained_variance_ratio),
        "umap_components": dataset.umap_components,
        "umap_n_neighbors": dataset.umap_n_neighbors,
        "umap_min_dist": dataset.umap_min_dist,
        "clustering_dimensions": int(dataset.reduced_matrix.shape[1]),
        "inertia": float(model.inertia_),
        "silhouette_score": selected_silhouette,
        "distance_metric": "cosine_via_l2_normalization",
        "selection_method": "multi_criterion_consensus",
    }
    write_json(payload, output_dir / "run_metadata.json")
    return payload


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    dataset = prepare_dataset(
        features_path=args.features_path,
        limit=args.limit,
        pca_variance_threshold=args.pca_variance_threshold,
        max_pca_components=args.max_pca_components,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.random_state,
        disable_umap=args.disable_umap,
    )
    model, metrics_frame, inlier_mask = select_best_kmeans(
        dataset=dataset,
        max_clusters=args.max_clusters,
        random_state=args.random_state,
        workers=args.workers,
    )
    payload = build_outputs(dataset, model, metrics_frame, inlier_mask, output_dir)
    print(payload)


if __name__ == "__main__":
    main()
