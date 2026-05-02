from __future__ import annotations

import argparse
import math
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

from clustering.benchmark_metrics import bootstrap_stability_ari, dunn_index, trustworthiness_score
from clustering.shared import (
    DEFAULT_FEATURES_DIR,
    REDUCTION_MODES,
    PreparedDataset,
    assert_inside_test_root,
    candidate_cluster_counts,
    default_algorithm_output_dir,
    ensure_output_dir,
    prepare_dataset,
    write_assignments,
    write_json,
)


ALGORITHM_NAME = "kmeans_gap_stability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run KMeans with thesis-oriented K selection: gap statistic primary, "
            "prediction strength / bootstrap ARI stability checks, and internal "
            "validity indices as supporting evidence."
        ),
    )
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-clusters", type=int, default=60)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--silhouette-sample-size", type=int, default=2000)
    parser.add_argument(
        "--selection-sample-size",
        type=int,
        default=None,
        help=(
            "Optional uniform subsample size for the expensive K-selection "
            "calculations. Default uses all inlier rows."
        ),
    )
    parser.add_argument(
        "--gap-reference-samples",
        type=int,
        default=10,
        help="Number of uniform-reference datasets for the gap statistic.",
    )
    parser.add_argument(
        "--prediction-strength-repeats",
        type=int,
        default=5,
        help="Repeated half-split evaluations per candidate K.",
    )
    parser.add_argument(
        "--prediction-strength-threshold",
        type=float,
        default=0.80,
        help="Common reproducibility threshold from prediction-strength literature.",
    )
    parser.add_argument("--stability-bootstraps", type=int, default=5)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.80)
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99)
    parser.add_argument("--max-pca-components", type=int, default=100)
    parser.add_argument("--umap-n-components", type=int, default=15)
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument(
        "--reduction-mode",
        choices=list(REDUCTION_MODES),
        default="pca_then_umap",
    )
    parser.add_argument("--disable-umap", action="store_true")
    parser.add_argument(
        "--disable-outlier-removal",
        action="store_true",
        help="Skip the LOF inlier filter and run selection on every row.",
    )
    return parser.parse_args()


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


def _fit_data_candidate(
    k: int,
    x: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
    silhouette_sample_size: int,
) -> dict[str, Any]:
    model = _fit_kmeans(x, k, seed, n_init, max_iter)
    labels = model.labels_
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise RuntimeError(f"KMeans produced a degenerate one-cluster result for k={k}")

    cluster_sizes = np.bincount(labels, minlength=k)
    sil_sample = min(len(x), int(silhouette_sample_size))
    silhouette = float(
        silhouette_score(
            x,
            labels,
            sample_size=sil_sample,
            random_state=seed,
        )
    )
    return {
        "cluster_count": int(k),
        "inertia": float(model.inertia_),
        "log_inertia": float(math.log(model.inertia_ + 1e-12)),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(x, labels)),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
    }


def _fit_reference_log_inertia(
    k: int,
    n_samples: int,
    mins: np.ndarray,
    maxs: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
) -> tuple[int, float]:
    rng = np.random.RandomState(seed)
    ref = rng.uniform(mins, maxs, size=(n_samples, len(mins)))
    model = _fit_kmeans(ref, k, seed, n_init, max_iter)
    return int(k), float(math.log(model.inertia_ + 1e-12))


def _prediction_strength_once(
    k: int,
    x: np.ndarray,
    seed: int,
    n_init: int,
    max_iter: int,
) -> float:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x))
    split = len(order) // 2
    train_idx = order[:split]
    test_idx = order[split:]
    if len(train_idx) <= k or len(test_idx) <= k:
        return float("nan")

    train = x[train_idx]
    test = x[test_idx]
    train_model = _fit_kmeans(train, k, seed, n_init, max_iter)
    test_model = _fit_kmeans(test, k, seed + 1, n_init, max_iter)
    labels_by_train = train_model.predict(test)
    labels_by_test = test_model.labels_

    strengths: list[float] = []
    for cluster_id in range(k):
        members = np.where(labels_by_test == cluster_id)[0]
        n_members = len(members)
        if n_members < 2:
            continue
        counts = np.bincount(labels_by_train[members], minlength=k)
        same_pairs = float(np.sum(counts * (counts - 1) / 2.0))
        total_pairs = float(n_members * (n_members - 1) / 2.0)
        strengths.append(same_pairs / total_pairs if total_pairs else float("nan"))

    if not strengths:
        return float("nan")
    return float(np.nanmin(strengths))


def _prediction_strength_for_k(
    k: int,
    x: np.ndarray,
    base_seed: int,
    repeats: int,
    n_init: int,
    max_iter: int,
) -> dict[str, Any]:
    values = [
        _prediction_strength_once(
            k=k,
            x=x,
            seed=base_seed + 10_000 * k + repeat,
            n_init=n_init,
            max_iter=max_iter,
        )
        for repeat in range(repeats)
    ]
    finite = [v for v in values if np.isfinite(v)]
    return {
        "cluster_count": int(k),
        "prediction_strength_mean": float(np.mean(finite)) if finite else float("nan"),
        "prediction_strength_std": float(np.std(finite)) if finite else float("nan"),
        "prediction_strength_repeats": len(finite),
    }


def choose_gap_one_se(metrics: pd.DataFrame) -> int:
    ordered = metrics.sort_values("cluster_count").reset_index(drop=True)
    for idx in range(len(ordered) - 1):
        cur = ordered.iloc[idx]
        nxt = ordered.iloc[idx + 1]
        if float(cur["gap"]) >= float(nxt["gap"]) - float(nxt["gap_sk"]):
            return int(cur["cluster_count"])
    best = ordered.loc[ordered["gap"].idxmax()]
    return int(best["cluster_count"])


def sample_for_selection(
    x: np.ndarray,
    sample_size: int | None,
    random_state: int,
) -> tuple[np.ndarray, int]:
    if sample_size is None or sample_size <= 0 or len(x) <= sample_size:
        return x, len(x)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(x), int(sample_size), replace=False)
    idx.sort()
    return x[idx], len(idx)


def evaluate_candidates(
    x: np.ndarray,
    candidates: list[int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    workers = max(1, int(args.workers))
    print(f"[KSelection] Evaluating {len(candidates)} candidate K values on {len(x)} row(s)")
    data_jobs = [
        delayed(_fit_data_candidate)(
            k,
            x,
            args.random_state + k,
            args.n_init,
            args.max_iter,
            args.silhouette_sample_size,
        )
        for k in candidates
    ]
    records = Parallel(n_jobs=workers)(
        tqdm(data_jobs, total=len(data_jobs), desc="[KSelection] Internal CVIs")
    )
    frame = pd.DataFrame(records).sort_values("cluster_count").reset_index(drop=True)

    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    # Avoid a zero-width reference box on constant dimensions.
    maxs = np.where(maxs <= mins, mins + 1e-9, maxs)

    reference_samples = max(1, int(args.gap_reference_samples))
    ref_jobs = []
    for k in candidates:
        for b in range(reference_samples):
            seed = args.random_state + 1_000_000 + 10_000 * k + b
            ref_jobs.append(
                delayed(_fit_reference_log_inertia)(
                    k,
                    len(x),
                    mins,
                    maxs,
                    seed,
                    args.n_init,
                    args.max_iter,
                )
            )

    print(
        f"[KSelection] Gap statistic: {reference_samples} reference sample(s) "
        f"x {len(candidates)} K values"
    )
    ref_results = Parallel(n_jobs=workers)(
        tqdm(ref_jobs, total=len(ref_jobs), desc="[KSelection] Gap references")
    )
    ref_logs: dict[int, list[float]] = {k: [] for k in candidates}
    for k, log_inertia in ref_results:
        ref_logs[int(k)].append(float(log_inertia))

    gaps = []
    for _, row in frame.iterrows():
        k = int(row["cluster_count"])
        logs = np.asarray(ref_logs[k], dtype=np.float64)
        std = float(np.std(logs, ddof=1)) if len(logs) > 1 else 0.0
        gaps.append({
            "cluster_count": k,
            "gap": float(np.mean(logs) - float(row["log_inertia"])),
            "gap_ref_log_inertia_mean": float(np.mean(logs)),
            "gap_ref_log_inertia_std": std,
            "gap_sk": float(math.sqrt(1.0 + 1.0 / len(logs)) * std),
        })
    frame = frame.merge(pd.DataFrame(gaps), on="cluster_count", how="left")

    repeats = max(0, int(args.prediction_strength_repeats))
    if repeats > 0:
        print(
            f"[KSelection] Prediction strength: {repeats} half-split repeat(s) "
            f"x {len(candidates)} K values"
        )
        ps_jobs = [
            delayed(_prediction_strength_for_k)(
                k,
                x,
                args.random_state + 2_000_000,
                repeats,
                args.n_init,
                args.max_iter,
            )
            for k in candidates
        ]
        ps_records = Parallel(n_jobs=workers)(
            tqdm(ps_jobs, total=len(ps_jobs), desc="[KSelection] Prediction strength")
        )
        frame = frame.merge(pd.DataFrame(ps_records), on="cluster_count", how="left")
    else:
        frame["prediction_strength_mean"] = np.nan
        frame["prediction_strength_std"] = np.nan
        frame["prediction_strength_repeats"] = 0

    return frame.sort_values("cluster_count").reset_index(drop=True)


def decision_payload(metrics: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    gap_k = choose_gap_one_se(metrics)
    metrics["gap_one_se_selected"] = metrics["cluster_count"] == gap_k

    sil_k = int(metrics.loc[metrics["silhouette_score"].idxmax(), "cluster_count"])
    ch_k = int(metrics.loc[metrics["calinski_harabasz_score"].idxmax(), "cluster_count"])
    db_k = int(metrics.loc[metrics["davies_bouldin_score"].idxmin(), "cluster_count"])

    ps_frame = metrics[np.isfinite(metrics["prediction_strength_mean"])].copy()
    threshold = float(args.prediction_strength_threshold)
    ps_supported = ps_frame[ps_frame["prediction_strength_mean"] >= threshold]
    if not ps_supported.empty:
        ps_k = int(ps_supported["cluster_count"].max())
        ps_rule = f"largest_k_with_prediction_strength_ge_{threshold:.2f}"
    elif not ps_frame.empty:
        ps_k = int(ps_frame.loc[ps_frame["prediction_strength_mean"].idxmax(), "cluster_count"])
        ps_rule = "no_k_reached_threshold_using_best_prediction_strength"
    else:
        ps_k = None
        ps_rule = "prediction_strength_not_computed"

    if ps_k is None:
        support = "gap_selected_without_prediction_strength"
    elif abs(ps_k - gap_k) <= 1:
        support = "gap_and_prediction_strength_agree_or_are_adjacent"
    else:
        support = "gap_and_prediction_strength_disagree"

    return {
        "selected_k": int(gap_k),
        "primary_rule": "gap_statistic_one_standard_error",
        "gap_selected_k": int(gap_k),
        "prediction_strength_selected_k": ps_k,
        "prediction_strength_rule": ps_rule,
        "prediction_strength_threshold": threshold,
        "support_status": support,
        "requires_interpretation_review": support == "gap_and_prediction_strength_disagree",
        "best_silhouette_k": sil_k,
        "best_calinski_harabasz_k": ch_k,
        "best_davies_bouldin_k": db_k,
    }


def write_selection_report(
    metrics: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    report_path = output_dir / "k_selection_report.md"
    assert_inside_test_root(report_path)

    selected = metrics[metrics["cluster_count"] == metadata["selected_k"]].iloc[0]
    top = metrics.copy()
    top["gap_rank"] = top["gap"].rank(ascending=False, method="min")
    top["silhouette_rank"] = top["silhouette_score"].rank(ascending=False, method="min")
    top["prediction_strength_rank"] = top["prediction_strength_mean"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    display_cols = [
        "cluster_count",
        "gap",
        "gap_sk",
        "prediction_strength_mean",
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ]
    display = top.sort_values(["gap_rank", "prediction_strength_rank"]).head(12)[display_cols]

    def table(frame: pd.DataFrame) -> str:
        lines = ["| " + " | ".join(frame.columns) + " |"]
        lines.append("| " + " | ".join("---" for _ in frame.columns) + " |")
        for _, row in frame.iterrows():
            cells = []
            for col in frame.columns:
                value = row[col]
                if isinstance(value, float):
                    cells.append("" if pd.isna(value) else f"{value:.5f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    lines = [
        "# KMeans K-Selection Report",
        "",
        f"Selected K: {metadata['selected_k']}",
        f"Primary rule: {metadata['primary_rule']}",
        f"Support status: {metadata['support_status']}",
        f"Prediction-strength selected K: {metadata['prediction_strength_selected_k']}",
        f"Bootstrap stability ARI: {metadata.get('stability_mean_ari')}",
        "",
        "## Selected Row",
        "",
        table(pd.DataFrame([selected[display_cols].to_dict()])),
        "",
        "## Highest Gap / Stability Candidates",
        "",
        table(display),
        "",
        "## Decision Notes",
        "",
        (
            "The selected K is the smallest candidate satisfying the gap "
            "statistic one-standard-error rule. Prediction strength and "
            "bootstrap ARI are reproducibility checks; silhouette, "
            "Calinski-Harabasz, and Davies-Bouldin are supporting internal "
            "validity evidence."
        ),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(
    dataset: PreparedDataset,
    model: KMeans,
    metrics_frame: pd.DataFrame,
    inlier_mask: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    x_all_norm = normalize(dataset.reduced_matrix, norm="l2")
    all_labels = model.predict(x_all_norm)
    distances = model.transform(x_all_norm)
    assigned_distance = distances[np.arange(len(all_labels)), all_labels]

    assignments = dataset.metadata.copy()
    assignments["cluster_label"] = all_labels
    assignments["distance_to_centroid"] = assigned_distance
    assignments["is_outlier"] = ~inlier_mask
    cluster_sizes = assignments["cluster_label"].value_counts().sort_index()
    assignments["cluster_size"] = assignments["cluster_label"].map(cluster_sizes.to_dict())

    write_assignments(assignments, output_dir / "cluster_assignments.csv")

    metrics_path = output_dir / "selection_metrics.csv"
    assert_inside_test_root(metrics_path)
    metrics_frame.to_csv(metrics_path, index=False)

    summary_path = output_dir / "cluster_summary.csv"
    assert_inside_test_root(summary_path)
    cluster_sizes.rename_axis("cluster_label").reset_index(name="size").to_csv(
        summary_path,
        index=False,
    )

    centers_path = output_dir / "cluster_centers_reduced.npy"
    assert_inside_test_root(centers_path)
    np.save(centers_path, model.cluster_centers_)

    def _jsonable(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    selected_row = metrics_frame.loc[metrics_frame["cluster_count"] == int(model.n_clusters)]
    selected_metrics = (
        {key: _jsonable(value) for key, value in selected_row.iloc[0].to_dict().items()}
        if not selected_row.empty
        else {}
    )

    payload = {
        "algorithm": ALGORITHM_NAME,
        "cluster_count": int(model.n_clusters),
        "samples": int(len(assignments)),
        "inlier_samples": int(inlier_mask.sum()),
        "outliers_detected": int((~inlier_mask).sum()),
        "outliers_reassigned": True,
        "feature_source": str(dataset.source_path),
        "summary_csv_path": str(dataset.summary_csv_path) if dataset.summary_csv_path else None,
        "pca_components": int(dataset.pca_components),
        "pca_explained_variance_ratio": float(dataset.pca_explained_variance_ratio),
        "umap_components": dataset.umap_components,
        "umap_n_neighbors": dataset.umap_n_neighbors,
        "umap_min_dist": dataset.umap_min_dist,
        "clustering_dimensions": int(dataset.reduced_matrix.shape[1]),
        "distance_metric": "cosine_via_l2_normalization",
        "selection_method": "gap_statistic_primary_with_stability_confirmation",
        "selected_row": selected_metrics,
    }
    payload.update(metadata)
    return payload


def compute_final_stability(
    dataset: PreparedDataset,
    model: KMeans,
    inlier_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_red = dataset.reduced_matrix
    x_norm = normalize(x_red, norm="l2")
    labels = model.predict(x_norm)

    selected_k = int(model.n_clusters)

    def cluster_fn(x_sub: np.ndarray) -> np.ndarray:
        sub_norm = normalize(x_sub, norm="l2")
        m = KMeans(
            n_clusters=selected_k,
            random_state=args.random_state,
            n_init=max(5, min(args.n_init, 10)),
            max_iter=args.max_iter,
        )
        return m.fit_predict(sub_norm)

    stability = bootstrap_stability_ari(
        x_red,
        cluster_fn,
        n_bootstraps=max(2, int(args.stability_bootstraps)),
        sample_fraction=float(args.stability_sample_fraction),
        random_state=args.random_state,
    )

    return {
        "silhouette_score": float(
            silhouette_score(
                x_norm,
                labels,
                sample_size=min(len(x_norm), args.silhouette_sample_size),
                random_state=args.random_state,
            )
        ),
        "calinski_harabasz_score": float(calinski_harabasz_score(x_norm, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(x_norm, labels)),
        "dunn_index": dunn_index(x_red[inlier_mask], labels[inlier_mask]),
        "trustworthiness": trustworthiness_score(
            dataset.scaled_matrix,
            x_red,
            n_neighbors=10,
            sample_size=5000,
            random_state=args.random_state,
        ),
        "stability_mean_ari": stability["mean_ari"],
        "stability_std_ari": stability["std_ari"],
        "stability_n_pairs": stability["n_pairs"],
        "stability_n_successful_runs": stability["n_successful_runs"],
    }


def main() -> None:
    args = parse_args()
    reduction_mode = args.reduction_mode
    if args.disable_umap:
        if reduction_mode not in ("pca_then_umap", "pca_only"):
            raise SystemExit(
                f"--disable-umap conflicts with --reduction-mode {reduction_mode}."
            )
        reduction_mode = "pca_only"

    resolved_output_dir = args.output_dir or default_algorithm_output_dir(
        args.features_path,
        ALGORITHM_NAME,
        reduction_mode=reduction_mode,
    )
    output_dir = ensure_output_dir(resolved_output_dir)

    dataset = prepare_dataset(
        features_path=args.features_path,
        limit=args.limit,
        pca_variance_threshold=args.pca_variance_threshold,
        max_pca_components=args.max_pca_components,
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_random_state=args.random_state,
        reduction_mode=reduction_mode,
    )

    raw_x = dataset.reduced_matrix
    if args.disable_outlier_removal:
        inlier_mask = np.ones(len(raw_x), dtype=bool)
    else:
        inlier_mask = remove_outliers(raw_x)
        if inlier_mask.sum() < 10:
            inlier_mask = np.ones(len(raw_x), dtype=bool)

    x_clean = normalize(raw_x[inlier_mask], norm="l2")
    n_outliers = int((~inlier_mask).sum())
    print(
        f"[KSelection] LOF outlier removal: {n_outliers} outlier(s), "
        f"{len(x_clean)} inlier row(s)"
    )

    selection_x, selection_n = sample_for_selection(
        x_clean,
        args.selection_sample_size,
        args.random_state,
    )
    candidates = candidate_cluster_counts(selection_n, max_clusters=args.max_clusters)
    metrics = evaluate_candidates(selection_x, candidates, args)
    metadata = decision_payload(metrics, args)

    selected_k = int(metadata["selected_k"])
    print(f"[KSelection] Selected K={selected_k} via {metadata['primary_rule']}")
    print(f"[KSelection] Support status: {metadata['support_status']}")

    final_model = _fit_kmeans(
        x_clean,
        selected_k,
        args.random_state,
        args.n_init,
        args.max_iter,
    )

    metadata.update({
        "selection_samples": int(selection_n),
        "candidate_min_k": int(min(candidates)),
        "candidate_max_k": int(max(candidates)),
        "gap_reference_samples": int(args.gap_reference_samples),
        "prediction_strength_repeats": int(args.prediction_strength_repeats),
        "reduction_mode": reduction_mode,
    })

    payload = build_outputs(dataset, final_model, metrics, inlier_mask, output_dir, metadata)
    payload.update(compute_final_stability(dataset, final_model, inlier_mask, args))
    write_json(payload, output_dir / "run_metadata.json")
    write_selection_report(metrics, payload, output_dir)
    print(payload)


if __name__ == "__main__":
    main()
