import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv  # noqa: E402
from src.ui.modern_ui import launch_ui  # noqa: E402
from src.clustering.kmeans import (  # noqa: E402
    _collect_feature_vectors,
    _load_genre_mapping,
    build_group_weights,
    compute_cluster_range,
    prepare_features,
)


def _select_components(
    data: np.ndarray,
    min_components: int,
    max_components: int,
    covariance_type: str,
    max_iter: int,
    tol: float,
    init_params: str,
    reg_covar: float = 1e-6,
) -> Tuple[int, GaussianMixture, List[float], List[float]]:
    """Pick the component count that minimises BIC (also tracks AIC)."""
    best_n = min_components
    best_model: Optional[GaussianMixture] = None
    best_bic = float('inf')
    bic_scores: List[float] = []
    aic_scores: List[float] = []

    for n in range(min_components, max_components + 1):
        try:
            model = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                max_iter=max_iter,
                tol=tol,
                init_params=init_params,
                reg_covar=reg_covar,
                random_state=42,
            )
            model.fit(data)
            bic = model.bic(data)
            aic = model.aic(data)
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            # Only accept model if it converged
            if not model.converged_:
                print(f"  GMM with {n} components did not converge, skipping...")
                bic_scores.append(float('inf'))
                aic_scores.append(float('inf'))
                continue
                
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_model = model
        except ValueError as e:
            # Handle ill-conditioned covariance matrices
            print(f"  GMM with {n} components failed: {str(e)[:60]}...")
            bic_scores.append(float('inf'))
            aic_scores.append(float('inf'))
            continue

    if best_model is None:
        raise RuntimeError("Failed to fit any GaussianMixture models during selection.")

    return best_n, best_model, bic_scores, aic_scores


def run_gmm_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/results",
    n_components: int = 5,
    covariance_type: str = "full",
    max_iter: int = 200,
    tol: float = 1e-3,
    init_params: str = "kmeans",
    reg_covar: float = 1e-5,
    dynamic_component_selection: bool = True,
    dynamic_min_components: Optional[int] = None,
    dynamic_max_components: Optional[int] = None,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
):
    os.makedirs(results_dir, exist_ok=True)

    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        results_dir, genre_map, unique_genres, include_genre
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors)
    
    # Use the unified feature preparation (PCA per group or weighted)
    X_prepared = prepare_features(
        X_all,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_genres=len(unique_genres),
        include_genre=include_genre,
    )

    model: Optional[GaussianMixture] = None
    selected_components = n_components
    bic_scores: Optional[List[float]] = None
    aic_scores: Optional[List[float]] = None

    if dynamic_component_selection:
        n_samples = X_prepared.shape[0]
        
        # Compute data-driven component range if not explicitly provided
        auto_min, auto_max = compute_cluster_range(n_samples, len(unique_genres))
        min_comp = dynamic_min_components if dynamic_min_components is not None else auto_min
        max_comp = dynamic_max_components if dynamic_max_components is not None else auto_max
        
        # Additional cap for GMM stability with full covariance
        if covariance_type == 'full':
            max_stable = X_prepared.shape[0] // 20
            max_comp = min(max_comp, max_stable)
        
        print(f"Dynamic component selection: searching n in [{min_comp}, {max_comp}]")
        print(f"  (Based on {n_samples} samples and {len(unique_genres)} genres)")

        selected_components, model, bic_scores, aic_scores = _select_components(
            X_prepared,
            min_comp,
            max_comp,
            covariance_type,
            max_iter,
            tol,
            init_params,
            reg_covar,
        )
        print(
            f"BIC-driven selection picked {selected_components} components "
            f"(min BIC={min(bic_scores):.2f}, min AIC={min(aic_scores):.2f})"
        )

    if model is None:
        model = GaussianMixture(
            n_components=selected_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            random_state=42,
        )
        model.fit(X_prepared)

    labels = model.predict(X_prepared)
    probabilities = model.predict_proba(X_prepared).max(axis=1)
    log_probs = model.score_samples(X_prepared)

    coords = PCA(n_components=2, random_state=42).fit_transform(X_prepared)

    df = pd.DataFrame(
        {
            "Song": file_names,
            "Genre": genres,
            "Cluster": labels,
            "Confidence": probabilities,
            "LogLikelihood": log_probs,
            "PCA1": coords[:, 0],
            "PCA2": coords[:, 1],
        }
    )

    output_dir = Path("output/clustering_results")
    metrics_dir = Path("output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if bic_scores is not None and aic_scores is not None:
        min_comp = dynamic_min_components if dynamic_min_components is not None else 2
        selection_df = pd.DataFrame(
            {
                "Components": list(range(min_comp, min_comp + len(bic_scores))),
                "BIC": bic_scores,
                "AIC": aic_scores,
            }
        )
        selection_path = metrics_dir / "gmm_selection_criteria.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"Stored BIC/AIC diagnostics -> {selection_path}")

    csv_path = output_dir / "audio_clustering_results_gmm.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    print(
        f"GMM formed {len(np.unique(labels))} components; "
        f"avg confidence: {probabilities.mean():.2f}"
    )

    return df, coords, labels


if __name__ == "__main__":
    DF, COORDS, LABELS = run_gmm_clustering(
        audio_dir="audio_files",
        results_dir="output/features",
        n_components=5,
        dynamic_component_selection=True,
        include_genre=fv.include_genre,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="audio_files", clustering_method="GMM")
