from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import normalize

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, total=None, desc=None, **_kwargs):
        return iterable

from clustering.benchmark_metrics import bootstrap_stability_ari
from clustering.shared import (
    DEFAULT_FEATURES_DIR,
    REDUCTION_MODES,
    assert_inside_test_root,
    ensure_output_dir,
    prepare_dataset,
    write_json,
)


DEFAULT_CANDIDATES = "5,8,10,12,15,20,30,40,50,60,80,100"


def parse_candidate_ks(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError("At least one candidate K is required.")
    if min(values) < 2:
        raise argparse.ArgumentTypeError("Candidate K values must be >= 2.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a short list of candidate K values on one copied sample. "
            "This is the full-dataset decision stage: after repeated samples "
            "estimate a stable range, compare practical K candidates directly."
        ),
    )
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--candidate-ks", type=parse_candidate_ks, default=parse_candidate_ks(DEFAULT_CANDIDATES))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--stability-bootstraps", type=int, default=5)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.80)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99)
    parser.add_argument("--max-pca-components", type=int, default=100)
    parser.add_argument("--umap-n-components", type=int, default=15)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--reduction-mode", choices=list(REDUCTION_MODES), default="pca_then_umap")
    parser.add_argument("--disable-outlier-removal", action="store_true")
    return parser.parse_args()


def feature_source_key(features_path: str | Path) -> str:
    path = Path(features_path).resolve()
    return path.parent.name if path.is_file() else path.name


def default_output_dir(features_path: str | Path, reduction_mode: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "cluster_results"
        / feature_source_key(features_path)
        / reduction_mode
        / "candidate_k_validation"
    )


def remove_outliers(x: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    if len(x) < 25:
        return np.ones(len(x), dtype=bool)
    n_neighbors = max(2, min(20, len(x) - 1))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return lof.fit_predict(x) == 1


def _fit_kmeans(x: np.ndarray, k: int, seed: int, n_init: int, max_iter: int) -> KMeans:
    model = KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    model.fit(x)
    return model


def _evaluate_candidate(
    k: int,
    x: np.ndarray,
    random_state: int,
    n_init: int,
    max_iter: int,
    silhouette_sample_size: int,
    stability_bootstraps: int,
    stability_sample_fraction: float,
) -> dict[str, Any]:
    if k >= len(x):
        return {
            "candidate_k": int(k),
            "status": "skipped_too_large",
        }

    model = _fit_kmeans(x, k, random_state + k, n_init, max_iter)
    labels = model.labels_
    sizes = np.bincount(labels, minlength=k)
    sil_size = min(len(x), int(silhouette_sample_size))

    def cluster_fn(x_sub: np.ndarray) -> np.ndarray:
        m = KMeans(
            n_clusters=int(k),
            random_state=random_state + k,
            n_init=max(5, min(n_init, 10)),
            max_iter=max_iter,
        )
        return m.fit_predict(x_sub)

    stability = bootstrap_stability_ari(
        x,
        cluster_fn,
        n_bootstraps=max(2, int(stability_bootstraps)),
        sample_fraction=float(stability_sample_fraction),
        random_state=random_state + k,
    )

    return {
        "candidate_k": int(k),
        "status": "ok",
        "inertia": float(model.inertia_),
        "silhouette_score": float(
            silhouette_score(
                x,
                labels,
                sample_size=sil_size,
                random_state=random_state,
            )
        ),
        "calinski_harabasz_score": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(x, labels)),
        "stability_mean_ari": stability["mean_ari"],
        "stability_std_ari": stability["std_ari"],
        "stability_n_pairs": stability["n_pairs"],
        "stability_n_successful_runs": stability["n_successful_runs"],
        "min_cluster_size": int(sizes.min()),
        "p10_cluster_size": int(np.percentile(sizes, 10)),
        "median_cluster_size": int(np.median(sizes)),
        "max_cluster_size": int(sizes.max()),
        "clusters_under_25": int((sizes < 25).sum()),
        "clusters_under_50": int((sizes < 50).sum()),
        "clusters_under_100": int((sizes < 100).sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(
        args.output_dir or default_output_dir(args.features_path, args.reduction_mode)
    )

    dataset = prepare_dataset(
        features_path=args.features_path,
        pca_variance_threshold=args.pca_variance_threshold,
        max_pca_components=args.max_pca_components,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.random_state,
        reduction_mode=args.reduction_mode,
    )

    if args.disable_outlier_removal:
        inlier_mask = np.ones(len(dataset.reduced_matrix), dtype=bool)
    else:
        inlier_mask = remove_outliers(dataset.reduced_matrix)
        if inlier_mask.sum() < 10:
            inlier_mask = np.ones(len(dataset.reduced_matrix), dtype=bool)

    x = normalize(dataset.reduced_matrix[inlier_mask], norm="l2")
    candidates = [k for k in args.candidate_ks if k < len(x)]
    if not candidates:
        raise SystemExit("No candidate K values are valid for this sample size.")

    print(
        f"[CandidateK] Evaluating {len(candidates)} K values on "
        f"{len(x)} inlier row(s) with {args.workers} worker(s)"
    )
    jobs = [
        delayed(_evaluate_candidate)(
            k,
            x,
            args.random_state,
            args.n_init,
            args.max_iter,
            args.silhouette_sample_size,
            args.stability_bootstraps,
            args.stability_sample_fraction,
        )
        for k in candidates
    ]
    rows = Parallel(n_jobs=max(1, int(args.workers)))(
        tqdm(jobs, total=len(jobs), desc="[CandidateK] Grid")
    )
    frame = pd.DataFrame(rows).sort_values("candidate_k")

    csv_path = output_dir / "candidate_validation_metrics.csv"
    assert_inside_test_root(csv_path)
    frame.to_csv(csv_path, index=False)

    ok = frame[frame["status"] == "ok"].copy()
    best_stability_k = None
    best_silhouette_k = None
    if not ok.empty:
        best_stability_k = int(ok.loc[ok["stability_mean_ari"].idxmax(), "candidate_k"])
        best_silhouette_k = int(ok.loc[ok["silhouette_score"].idxmax(), "candidate_k"])

    payload = {
        "feature_source": str(dataset.source_path),
        "summary_csv_path": str(dataset.summary_csv_path) if dataset.summary_csv_path else None,
        "reduction_mode": args.reduction_mode,
        "samples": int(len(dataset.reduced_matrix)),
        "inlier_samples": int(inlier_mask.sum()),
        "outliers_detected": int((~inlier_mask).sum()),
        "candidate_ks": candidates,
        "best_stability_k": best_stability_k,
        "best_silhouette_k": best_silhouette_k,
        "pca_components": int(dataset.pca_components),
        "pca_explained_variance_ratio": float(dataset.pca_explained_variance_ratio),
        "umap_components": dataset.umap_components,
        "clustering_dimensions": int(dataset.reduced_matrix.shape[1]),
        "output_csv": str(csv_path),
    }
    write_json(payload, output_dir / "run_metadata.json")
    print(payload)


if __name__ == "__main__":
    main()
