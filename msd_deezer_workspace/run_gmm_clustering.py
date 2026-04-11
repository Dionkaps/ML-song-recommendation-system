from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import BayesianGaussianMixture

from clustering.shared import (
    DEFAULT_CLUSTER_OUTPUT_DIR,
    DEFAULT_FEATURES_DIR,
    PreparedDataset,
    ensure_output_dir,
    prepare_dataset,
    write_assignments,
    write_json,
)


DEFAULT_OUTPUT_DIR = DEFAULT_CLUSTER_OUTPUT_DIR / "gmm"

WEIGHT_THRESHOLD = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic Bayesian Gaussian Mixture clustering on extracted audio features.")
    parser.add_argument("--features-path", default=str(DEFAULT_FEATURES_DIR), help="Path to the features directory or feature summary CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory where GMM outputs will be stored.")
    parser.add_argument("--limit", type=int, help="Optional cap on how many songs to cluster.")
    parser.add_argument("--max-components", type=int, default=80, help="Upper bound on component count; the Dirichlet Process determines the effective number.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--pca-variance-threshold", type=float, default=0.99, help="Explained variance target used for PCA reduction.")
    parser.add_argument("--max-pca-components", type=int, default=100, help="Maximum PCA dimensions kept for clustering.")
    parser.add_argument("--umap-n-components", type=int, default=15, help="UMAP target dimensions for clustering.")
    parser.add_argument("--umap-n-neighbors", type=int, default=40, help="UMAP n_neighbors parameter.")
    parser.add_argument("--umap-min-dist", type=float, default=0.01, help="UMAP min_dist parameter.")
    parser.add_argument("--disable-umap", action="store_true", help="Skip UMAP and cluster on PCA output directly.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Item 9: Data-driven covariance regularisation
# ---------------------------------------------------------------------------

def compute_data_driven_reg_covar(x: np.ndarray) -> float:
    """Regularisation proportional to mean feature variance (with a floor)."""
    feature_variances = np.var(x, axis=0)
    reg = float(1e-3 * np.mean(feature_variances))
    return max(reg, 1e-6)


# ---------------------------------------------------------------------------
# Item 12: Confidence-weighted (soft) silhouette score
# ---------------------------------------------------------------------------

def soft_silhouette_score(
    x: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    sample_size: int = 2000,
    random_state: int = 42,
) -> float:
    """Silhouette weighted by per-point membership confidence."""
    if len(np.unique(labels)) < 2:
        return -1.0

    n = len(x)
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        sil = silhouette_samples(x[idx], labels[idx])
        conf = confidences[idx]
    else:
        sil = silhouette_samples(x, labels)
        conf = confidences

    total_weight = conf.sum()
    if total_weight <= 0:
        return -1.0
    return float(np.average(sil, weights=conf))


def select_best_bgmm(
    dataset: PreparedDataset,
    max_components: int,
    random_state: int,
) -> tuple[BayesianGaussianMixture, np.ndarray, np.ndarray, pd.DataFrame]:
    x = dataset.reduced_matrix
    silhouette_sample_size = min(len(x), 2000)
    records: list[dict[str, float | int | str]] = []
    best_model: BayesianGaussianMixture | None = None
    best_labels: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None
    best_key: float | None = None

    # Item 9 – adaptive regularisation
    reg_covar = compute_data_driven_reg_covar(x)
    print(f"[GMM] Data-driven reg_covar = {reg_covar:.6f}")

    # Item 10 – add "tied" covariance type
    covariance_types = ("diag", "full", "tied")
    concentration_priors = (0.01, 1.0, 100.0, 500.0)
    component_caps = (10, 15, 20, 25, 30, 40, 50, 60)

    total = len(covariance_types) * len(component_caps) * len(concentration_priors)
    progress = 0

    for covariance_type in covariance_types:
        for n_comp in component_caps:
            for concentration_prior in concentration_priors:
                progress += 1
                if progress % 12 == 0:
                    print(f"[GMM] Grid search progress: {progress}/{total}")

                model = BayesianGaussianMixture(
                    n_components=n_comp,
                    covariance_type=covariance_type,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=concentration_prior,
                    n_init=3,
                    max_iter=500,
                    reg_covar=reg_covar,
                    random_state=random_state,
                )
                try:
                    model.fit(x)
                except ValueError:
                    continue

                labels = model.predict(x)
                effective_count = int(len(np.unique(labels)))
                if effective_count < 2:
                    continue

                active_mask = model.weights_ > WEIGHT_THRESHOLD
                active_components = int(active_mask.sum())

                probabilities = model.predict_proba(x)
                lower_bound = float(np.asarray(model.lower_bound_).item())

                silhouette = float(
                    silhouette_score(
                        x, labels,
                        sample_size=silhouette_sample_size,
                        random_state=random_state,
                    )
                )

                # Item 12 – soft silhouette for selection
                confidences = probabilities.max(axis=1)
                soft_sil = soft_silhouette_score(
                    x, labels, confidences,
                    silhouette_sample_size, random_state,
                )
                mean_confidence = float(confidences.mean())

                records.append(
                    {
                        "covariance_type": covariance_type,
                        "weight_concentration_prior": float(concentration_prior),
                        "max_components": n_comp,
                        "effective_components": effective_count,
                        "active_components": active_components,
                        "lower_bound": lower_bound,
                        "silhouette_score": silhouette,
                        "soft_silhouette_score": soft_sil,
                        "mean_membership_confidence": mean_confidence,
                        "reg_covar": reg_covar,
                    }
                )

                # Item 12 – select by soft silhouette
                if best_key is None or soft_sil > best_key:
                    best_key = soft_sil
                    best_model = model
                    best_labels = labels
                    best_probabilities = probabilities

    if best_model is None or best_labels is None or best_probabilities is None:
        raise RuntimeError("Bayesian GMM automatic selection could not find a valid clustering solution.")

    print(f"[GMM] Best grid result: soft_sil={best_key:.4f}, "
          f"k={len(np.unique(best_labels))}, cov={best_model.covariance_type}")

    # ------------------------------------------------------------------
    # Item 11 – hierarchical refinement: try larger component counts
    #           with the winning covariance type to discover sub-structure
    # ------------------------------------------------------------------
    if best_key is not None and best_key < 0.3:
        best_cov = best_model.covariance_type
        best_conc = float(best_model.weight_concentration_prior_)
        effective_n = len(np.unique(best_labels))
        print(f"[GMM] Soft silhouette {best_key:.4f} < 0.3 -> trying hierarchical refinement")

        for multiplier in (1.5, 2.0, 3.0):
            n_extra = int(effective_n * multiplier)
            if n_extra <= effective_n or n_extra > max_components:
                continue

            ref_model = BayesianGaussianMixture(
                n_components=n_extra,
                covariance_type=best_cov,
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=best_conc,
                n_init=5,
                max_iter=500,
                reg_covar=reg_covar,
                random_state=random_state,
            )
            try:
                ref_model.fit(x)
            except ValueError:
                continue

            ref_labels = ref_model.predict(x)
            if len(np.unique(ref_labels)) < 2:
                continue

            ref_probs = ref_model.predict_proba(x)
            ref_conf = ref_probs.max(axis=1)
            ref_soft_sil = soft_silhouette_score(
                x, ref_labels, ref_conf,
                silhouette_sample_size, random_state,
            )

            ref_sil = float(silhouette_score(
                x, ref_labels,
                sample_size=silhouette_sample_size,
                random_state=random_state,
            ))
            ref_eff = int(len(np.unique(ref_labels)))
            ref_active = int((ref_model.weights_ > WEIGHT_THRESHOLD).sum())
            records.append(
                {
                    "covariance_type": best_cov,
                    "weight_concentration_prior": best_conc,
                    "max_components": n_extra,
                    "effective_components": ref_eff,
                    "active_components": ref_active,
                    "lower_bound": float(np.asarray(ref_model.lower_bound_).item()),
                    "silhouette_score": ref_sil,
                    "soft_silhouette_score": ref_soft_sil,
                    "mean_membership_confidence": float(ref_conf.mean()),
                    "reg_covar": reg_covar,
                }
            )

            if ref_soft_sil > best_key:
                print(f"[GMM] Refinement improved: soft_sil {best_key:.4f} -> {ref_soft_sil:.4f} "
                      f"(k={ref_eff}, max_comp={n_extra})")
                best_key = ref_soft_sil
                best_model = ref_model
                best_labels = ref_labels
                best_probabilities = ref_probs

    metrics_frame = pd.DataFrame(records).sort_values(
        by=["soft_silhouette_score", "silhouette_score", "effective_components", "lower_bound"],
        ascending=[False, False, False, False],
    )
    return best_model, best_labels, best_probabilities, metrics_frame


def build_outputs(
    dataset: PreparedDataset,
    model: BayesianGaussianMixture,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metrics_frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    max_probability = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)

    assignments = dataset.metadata.copy()
    assignments["cluster_label"] = labels
    assignments["membership_confidence"] = max_probability
    assignments["membership_entropy"] = entropy

    cluster_sizes = assignments["cluster_label"].value_counts().sort_index()
    assignments["cluster_size"] = assignments["cluster_label"].map(cluster_sizes.to_dict())

    write_assignments(assignments, output_dir / "cluster_assignments.csv")
    metrics_frame.to_csv(output_dir / "selection_metrics.csv", index=False)
    cluster_sizes.rename_axis("cluster_label").reset_index(name="size").to_csv(
        output_dir / "cluster_summary.csv",
        index=False,
    )
    np.save(output_dir / "gmm_means_reduced.npy", model.means_)
    np.save(output_dir / "gmm_weights.npy", model.weights_)

    effective_count = int(len(np.unique(labels)))
    active_mask = model.weights_ > WEIGHT_THRESHOLD
    active_components = int(active_mask.sum())
    selected_row = metrics_frame.iloc[0]

    confidences = probabilities.max(axis=1)
    soft_sil = soft_silhouette_score(
        dataset.reduced_matrix, labels, confidences,
        sample_size=min(len(labels), 2000),
    )

    payload = {
        "algorithm": "bayesian_gmm",
        "effective_cluster_count": effective_count,
        "active_components": active_components,
        "max_components": int(model.n_components),
        "covariance_type": model.covariance_type,
        "weight_concentration_prior": float(model.weight_concentration_prior_),
        "samples": int(len(assignments)),
        "feature_source": str(dataset.source_path),
        "summary_csv_path": str(dataset.summary_csv_path) if dataset.summary_csv_path else None,
        "pca_components": int(dataset.pca_components),
        "pca_explained_variance_ratio": float(dataset.pca_explained_variance_ratio),
        "umap_components": dataset.umap_components,
        "umap_n_neighbors": dataset.umap_n_neighbors,
        "umap_min_dist": dataset.umap_min_dist,
        "clustering_dimensions": int(dataset.reduced_matrix.shape[1]),
        "lower_bound": float(np.asarray(model.lower_bound_).item()),
        "silhouette_score": float(selected_row["silhouette_score"]),
        "soft_silhouette_score": soft_sil,
        "reg_covar": float(model.reg_covar),
        "selection_method": "soft_silhouette_with_hierarchical_refinement",
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
    model, labels, probabilities, metrics_frame = select_best_bgmm(
        dataset=dataset,
        max_components=args.max_components,
        random_state=args.random_state,
    )
    payload = build_outputs(dataset, model, labels, probabilities, metrics_frame, output_dir)
    print(payload)


if __name__ == "__main__":
    main()
