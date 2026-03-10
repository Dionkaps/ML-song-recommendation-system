import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv  # noqa: E402
from src.ui.modern_ui import launch_ui  # noqa: E402
from src.clustering.kmeans import (  # noqa: E402
    compute_cluster_range,
    compute_visualization_coords,
    load_clustering_dataset,
)
from joblib import Parallel, delayed


def _select_components(
    data: np.ndarray,
    min_components: int,
    max_components: int,
    covariance_type: str,
    max_iter: int,
    tol: float,
    init_params: str,
    reg_covar: float = 1e-6,
    n_jobs: int = 1,
) -> Tuple[int, GaussianMixture, List[float], List[float], List[float]]:
    """Pick the component count with BIC as the primary criterion.

    BIC is the correct model-selection metric for Gaussian mixtures.
    When multiple models are within a weak-evidence BIC band, prefer the one
    with the best silhouette score to avoid selecting needlessly fragmented
    mixtures. Uses parallel processing across candidate counts.
    """

    sample_size = min(5000, data.shape[0])

    def evaluate_n(
        n: int,
    ) -> Tuple[int, float, float, float, bool, Optional[GaussianMixture]]:
        """Evaluate BIC/AIC and cluster separation for a single n value."""
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
            labels = model.predict(data)
            unique_labels = np.unique(labels)
            silhouette = float("nan")
            if len(unique_labels) > 1 and len(unique_labels) < len(data):
                silhouette_kwargs = {"random_state": 42}
                if sample_size < len(data):
                    silhouette_kwargs["sample_size"] = sample_size
                silhouette = float(silhouette_score(data, labels, **silhouette_kwargs))
            return n, bic, aic, silhouette, model.converged_, model
        except ValueError:
            return n, float("inf"), float("inf"), float("nan"), False, None
    
    print(f"Finding optimal GMM components ({min_components}-{max_components})...")
    
    # Parallel evaluation of all component counts
    n_values = list(range(min_components, max_components + 1))
    if n_jobs == 1:
        results = [evaluate_n(n) for n in n_values]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
            delayed(evaluate_n)(n) for n in n_values
        )
    
    # Process results
    best_n = min_components
    best_model: Optional[GaussianMixture] = None
    bic_scores: List[float] = []
    aic_scores: List[float] = []
    silhouette_scores: List[float] = []
    converged_rows = []

    for n, bic, aic, silhouette, converged, model in sorted(results, key=lambda x: x[0]):
        bic_scores.append(bic)
        aic_scores.append(aic)
        silhouette_scores.append(silhouette)
        if converged and model is not None:
            converged_rows.append((n, bic, aic, silhouette, model))

    if not converged_rows:
        raise RuntimeError("Failed to fit any GaussianMixture models during selection.")

    best_bic = min(row[1] for row in converged_rows)
    bic_tolerance = 10.0
    candidate_rows = [row for row in converged_rows if row[1] <= best_bic + bic_tolerance]
    best_n, _, _, _, best_model = min(
        candidate_rows,
        key=lambda row: (
            -np.nan_to_num(row[3], nan=-1.0),
            row[1],
            row[0],
        ),
    )

    return best_n, best_model, bic_scores, aic_scores, silhouette_scores


def run_gmm_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
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
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
):
    os.makedirs(results_dir, exist_ok=True)
    
    file_names, genres, unique_genres, X_prepared = load_clustering_dataset(
        audio_dir=audio_dir,
        results_dir=results_dir,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
    )

    model: Optional[GaussianMixture] = None
    selected_components = n_components
    bic_scores: Optional[List[float]] = None
    aic_scores: Optional[List[float]] = None
    silhouette_scores: Optional[List[float]] = None

    if dynamic_component_selection:
        n_samples = X_prepared.shape[0]
        
        # Compute data-driven component range if not explicitly provided
        genre_count_hint = len(unique_genres) if include_genre else 0
        auto_min, auto_max = compute_cluster_range(n_samples, genre_count_hint)
        min_comp = dynamic_min_components if dynamic_min_components is not None else auto_min
        max_comp = dynamic_max_components if dynamic_max_components is not None else auto_max
        
        # Additional cap for GMM stability with full covariance
        if covariance_type == 'full':
            max_stable = X_prepared.shape[0] // 20
            max_comp = max(min_comp, min(max_comp, max_stable))
        
        print(f"Dynamic component selection: searching n in [{min_comp}, {max_comp}]")
        print(f"  (Based on {n_samples} samples)")

        selected_components, model, bic_scores, aic_scores, silhouette_scores = _select_components(
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

    coords = compute_visualization_coords(X_prepared)

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

    if bic_scores is not None and aic_scores is not None and silhouette_scores is not None:
        min_comp = dynamic_min_components if dynamic_min_components is not None else auto_min
        selection_df = pd.DataFrame(
            {
                "Components": list(range(min_comp, min_comp + len(bic_scores))),
                "BIC": bic_scores,
                "AIC": aic_scores,
                "Silhouette": silhouette_scores,
            }
        )
        selection_path = metrics_dir / "gmm_selection_criteria.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"Stored GMM selection diagnostics -> {selection_path}")

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
