import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv  # noqa: E402
from src.clustering.kmeans import (  # noqa: E402
    compute_visualization_coords,
    load_clustering_dataset_bundle,
    save_retrieval_artifact,
    snapshot_dataset_qc_artifacts,
)
DEFAULT_COVARIANCE_TYPES: Tuple[str, ...] = ("diag", "full")
DEFAULT_REG_COVAR_GRID: Tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3)
DEFAULT_STABILITY_SEEDS: Tuple[int, ...] = (42, 43, 44)
DEFAULT_BIC_TOLERANCE = 10.0
SHORTLIST_PER_COVARIANCE = 3


@dataclass
class GMMCandidateResult:
    """Container for one fitted GMM candidate and its diagnostics."""

    stage: str
    components: int
    covariance_type: str
    reg_covar: float
    n_init: int
    requested_max_iter: int
    effective_max_iter: int
    used_extended_iter: bool
    converged: bool
    warning_summary: str
    bic: float
    aic: float
    avg_log_likelihood: float
    silhouette: float
    avg_confidence: float
    occupied_components: int
    min_cluster_size: int
    median_cluster_size: float
    max_cluster_size: int
    tiny_cluster_min_size: int
    tiny_cluster_count: int
    dominant_cluster_fraction: float
    cluster_size_distribution: str
    covariance_min_eigen: float
    covariance_max_eigen: float
    covariance_condition_number: float
    covariance_floor_fraction: float
    unstable_covariance: bool
    is_degenerate: bool
    passed_validity_filters: bool
    stability_ari: float = float("nan")
    near_best_bic: bool = False
    selected: bool = False
    selection_rank: int = 0
    error_message: str = ""
    model: Optional[GaussianMixture] = None

    def to_row(self) -> Dict[str, Any]:
        """Serialize diagnostics for CSV/JSON output."""

        return {
            "Stage": self.stage,
            "Components": self.components,
            "CovarianceType": self.covariance_type,
            "RegCovar": self.reg_covar,
            "NInit": self.n_init,
            "RequestedMaxIter": self.requested_max_iter,
            "EffectiveMaxIter": self.effective_max_iter,
            "UsedExtendedIter": self.used_extended_iter,
            "Converged": self.converged,
            "WarningSummary": self.warning_summary,
            "BIC": self.bic,
            "AIC": self.aic,
            "AvgLogLikelihood": self.avg_log_likelihood,
            "Silhouette": self.silhouette,
            "AvgConfidence": self.avg_confidence,
            "OccupiedComponents": self.occupied_components,
            "MinClusterSize": self.min_cluster_size,
            "MedianClusterSize": self.median_cluster_size,
            "MaxClusterSize": self.max_cluster_size,
            "TinyClusterMinSize": self.tiny_cluster_min_size,
            "TinyClusterCount": self.tiny_cluster_count,
            "DominantClusterFraction": self.dominant_cluster_fraction,
            "ClusterSizeDistribution": self.cluster_size_distribution,
            "CovarianceMinEigen": self.covariance_min_eigen,
            "CovarianceMaxEigen": self.covariance_max_eigen,
            "CovarianceConditionNumber": self.covariance_condition_number,
            "CovarianceFloorFraction": self.covariance_floor_fraction,
            "UnstableCovariance": self.unstable_covariance,
            "IsDegenerate": self.is_degenerate,
            "PassedValidityFilters": self.passed_validity_filters,
            "StabilityARI": self.stability_ari,
            "NearBestBIC": self.near_best_bic,
            "Selected": self.selected,
            "SelectionRank": self.selection_rank,
            "ErrorMessage": self.error_message,
        }


def _resolve_component_search_range(
    n_samples: int,
    dynamic_min_components: Optional[int],
    dynamic_max_components: Optional[int],
) -> Tuple[int, int]:
    """Resolve the explicit recommended component search range."""

    min_components = dynamic_min_components if dynamic_min_components is not None else 4
    max_components = dynamic_max_components if dynamic_max_components is not None else 40
    max_components = min(max_components, max(2, n_samples - 1))
    min_components = max(2, min(min_components, max_components))
    return min_components, max_components


def _fit_model_with_retry(
    data: np.ndarray,
    n_components: int,
    covariance_type: str,
    max_iter: int,
    extended_max_iter: int,
    tol: float,
    init_params: str,
    reg_covar: float,
    n_init: int,
    random_state: int,
) -> Tuple[GaussianMixture, int, bool, List[str]]:
    """Fit a GaussianMixture and retry with a larger iteration budget if needed."""

    warning_names: List[str] = []

    def fit_once(iter_budget: int) -> Tuple[GaussianMixture, bool, List[str]]:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=iter_budget,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(data)
        names = [warning.category.__name__ for warning in caught]
        has_convergence_warning = any(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        return model, has_convergence_warning, names

    model, warned, names = fit_once(max_iter)
    warning_names.extend(names)
    effective_max_iter = max_iter
    used_extended_iter = False

    if (warned or not model.converged_) and extended_max_iter > max_iter:
        used_extended_iter = True
        effective_max_iter = extended_max_iter
        warning_names.append("ExtendedMaxIterRetry")
        model, warned, names = fit_once(extended_max_iter)
        warning_names.extend(names)

    deduped_warnings = sorted(set(warning_names))
    return model, effective_max_iter, used_extended_iter, deduped_warnings


def _cluster_stats(
    labels: np.ndarray,
    n_components: int,
    n_samples: int,
) -> Dict[str, Any]:
    """Summarize the hard cluster usage pattern for degeneracy checks."""

    cluster_counts = np.bincount(labels, minlength=n_components)
    occupied_counts = cluster_counts[cluster_counts > 0]
    tiny_cluster_min_size = max(5, int(np.ceil(n_samples * 0.002)))
    tiny_cluster_count = int(
        np.sum((cluster_counts > 0) & (cluster_counts < tiny_cluster_min_size))
    )
    dominant_cluster_fraction = float(cluster_counts.max() / n_samples)
    cluster_size_distribution = json.dumps(
        {str(idx): int(count) for idx, count in enumerate(cluster_counts)}
    )

    if occupied_counts.size == 0:
        min_cluster_size = 0
        median_cluster_size = 0.0
        max_cluster_size = 0
        occupied_components = 0
    else:
        min_cluster_size = int(occupied_counts.min())
        median_cluster_size = float(np.median(occupied_counts))
        max_cluster_size = int(occupied_counts.max())
        occupied_components = int(occupied_counts.size)

    return {
        "occupied_components": occupied_components,
        "min_cluster_size": min_cluster_size,
        "median_cluster_size": median_cluster_size,
        "max_cluster_size": max_cluster_size,
        "tiny_cluster_min_size": tiny_cluster_min_size,
        "tiny_cluster_count": tiny_cluster_count,
        "dominant_cluster_fraction": dominant_cluster_fraction,
        "cluster_size_distribution": cluster_size_distribution,
    }


def _covariance_health_stats(
    model: GaussianMixture,
    reg_covar: float,
) -> Dict[str, Any]:
    """Estimate whether the fitted covariance structures look numerically healthy."""

    min_eigen = float("inf")
    max_eigen = float("-inf")
    floor_hits = 0
    total_entries = 0
    condition_numbers: List[float] = []

    if model.covariance_type == "diag":
        diag_values = np.asarray(model.covariances_, dtype=np.float64).reshape(-1)
        total_entries = int(diag_values.size)
        floor_hits = int(np.sum(diag_values <= (reg_covar * 1.05)))
        min_eigen = float(np.min(diag_values))
        max_eigen = float(np.max(diag_values))
        condition_numbers.append(max_eigen / max(min_eigen, 1e-12))
    else:
        for covariance in np.asarray(model.covariances_, dtype=np.float64):
            eigenvalues = np.linalg.eigvalsh(covariance)
            total_entries += int(eigenvalues.size)
            floor_hits += int(np.sum(eigenvalues <= (reg_covar * 1.05)))
            current_min = float(np.min(eigenvalues))
            current_max = float(np.max(eigenvalues))
            min_eigen = min(min_eigen, current_min)
            max_eigen = max(max_eigen, current_max)
            condition_numbers.append(current_max / max(current_min, 1e-12))

    if not np.isfinite(min_eigen):
        min_eigen = float("nan")
    if not np.isfinite(max_eigen):
        max_eigen = float("nan")

    floor_fraction = (
        float(floor_hits / total_entries) if total_entries > 0 else float("nan")
    )
    covariance_condition_number = (
        float(np.max(condition_numbers)) if condition_numbers else float("nan")
    )
    unstable_covariance = bool(
        not np.isfinite(min_eigen)
        or min_eigen <= 0.0
        or not np.isfinite(covariance_condition_number)
        or covariance_condition_number > 1e8
        or (
            np.isfinite(floor_fraction)
            and floor_fraction > 0.90
            and covariance_condition_number > 1e7
        )
    )

    return {
        "covariance_min_eigen": min_eigen,
        "covariance_max_eigen": max_eigen,
        "covariance_condition_number": covariance_condition_number,
        "covariance_floor_fraction": floor_fraction,
        "unstable_covariance": unstable_covariance,
    }


def _fit_candidate(
    data: np.ndarray,
    stage: str,
    n_components: int,
    covariance_type: str,
    max_iter: int,
    extended_max_iter: int,
    tol: float,
    init_params: str,
    reg_covar: float,
    n_init: int,
    random_state: int = 42,
) -> GMMCandidateResult:
    """Fit one candidate model and collect diagnostics."""

    try:
        model, effective_max_iter, used_extended_iter, warning_names = _fit_model_with_retry(
            data=data,
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=random_state,
        )

        posterior_probabilities = model.predict_proba(data)
        labels = posterior_probabilities.argmax(axis=1)
        cluster_stats = _cluster_stats(labels, n_components, data.shape[0])
        covariance_stats = _covariance_health_stats(model, reg_covar)

        occupied_components = cluster_stats["occupied_components"]
        silhouette = float("nan")
        if 1 < occupied_components < len(data):
            silhouette_kwargs: Dict[str, Any] = {"random_state": 42}
            sample_size = min(5000, len(data))
            if sample_size < len(data):
                silhouette_kwargs["sample_size"] = sample_size
            silhouette = float(silhouette_score(data, labels, **silhouette_kwargs))

        tiny_cluster_limit = max(1, n_components // 4)
        is_degenerate = bool(
            occupied_components < 2
            or cluster_stats["dominant_cluster_fraction"] >= 0.90
            or cluster_stats["tiny_cluster_count"] > tiny_cluster_limit
            or covariance_stats["unstable_covariance"]
        )

        return GMMCandidateResult(
            stage=stage,
            components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            requested_max_iter=max_iter,
            effective_max_iter=effective_max_iter,
            used_extended_iter=used_extended_iter,
            converged=bool(model.converged_),
            warning_summary="; ".join(warning_names),
            bic=float(model.bic(data)),
            aic=float(model.aic(data)),
            avg_log_likelihood=float(model.score(data)),
            silhouette=silhouette,
            avg_confidence=float(posterior_probabilities.max(axis=1).mean()),
            occupied_components=occupied_components,
            min_cluster_size=cluster_stats["min_cluster_size"],
            median_cluster_size=cluster_stats["median_cluster_size"],
            max_cluster_size=cluster_stats["max_cluster_size"],
            tiny_cluster_min_size=cluster_stats["tiny_cluster_min_size"],
            tiny_cluster_count=cluster_stats["tiny_cluster_count"],
            dominant_cluster_fraction=cluster_stats["dominant_cluster_fraction"],
            cluster_size_distribution=cluster_stats["cluster_size_distribution"],
            covariance_min_eigen=covariance_stats["covariance_min_eigen"],
            covariance_max_eigen=covariance_stats["covariance_max_eigen"],
            covariance_condition_number=covariance_stats["covariance_condition_number"],
            covariance_floor_fraction=covariance_stats["covariance_floor_fraction"],
            unstable_covariance=covariance_stats["unstable_covariance"],
            is_degenerate=is_degenerate,
            passed_validity_filters=bool(model.converged_ and not is_degenerate),
            model=model,
        )
    except Exception as exc:
        return GMMCandidateResult(
            stage=stage,
            components=n_components,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            requested_max_iter=max_iter,
            effective_max_iter=max_iter,
            used_extended_iter=False,
            converged=False,
            warning_summary="",
            bic=float("inf"),
            aic=float("inf"),
            avg_log_likelihood=float("nan"),
            silhouette=float("nan"),
            avg_confidence=float("nan"),
            occupied_components=0,
            min_cluster_size=0,
            median_cluster_size=0.0,
            max_cluster_size=0,
            tiny_cluster_min_size=max(5, int(np.ceil(data.shape[0] * 0.002))),
            tiny_cluster_count=0,
            dominant_cluster_fraction=float("nan"),
            cluster_size_distribution="{}",
            covariance_min_eigen=float("nan"),
            covariance_max_eigen=float("nan"),
            covariance_condition_number=float("nan"),
            covariance_floor_fraction=float("nan"),
            unstable_covariance=True,
            is_degenerate=True,
            passed_validity_filters=False,
            error_message=str(exc),
            model=None,
        )


def _run_candidate_grid(
    data: np.ndarray,
    stage: str,
    component_values: Sequence[int],
    covariance_types: Sequence[str],
    reg_covar_values: Sequence[float],
    max_iter: int,
    extended_max_iter: int,
    tol: float,
    init_params: str,
    n_init: int,
    n_jobs: int = 1,
) -> List[GMMCandidateResult]:
    """Evaluate a grid of GMM candidates."""

    tasks = [
        (components, covariance_type, reg_covar)
        for covariance_type in covariance_types
        for components in component_values
        for reg_covar in reg_covar_values
    ]

    def evaluate(task: Tuple[int, str, float]) -> GMMCandidateResult:
        components, covariance_type, reg_covar = task
        return _fit_candidate(
            data=data,
            stage=stage,
            n_components=components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            n_init=n_init,
        )

    if n_jobs == 1:
        results = [evaluate(task) for task in tasks]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
            delayed(evaluate)(task) for task in tasks
        )

    return results


def _shortlist_component_pairs(
    stage_one_results: Sequence[GMMCandidateResult],
    per_covariance: int = SHORTLIST_PER_COVARIANCE,
) -> List[Tuple[int, str]]:
    """Shortlist the most promising component counts before the reg sweep."""

    shortlist: List[Tuple[int, str]] = []
    seen = set()

    for covariance_type in DEFAULT_COVARIANCE_TYPES:
        rows = [
            row
            for row in stage_one_results
            if row.covariance_type == covariance_type and row.passed_validity_filters
        ]
        rows.sort(key=lambda row: (row.bic, row.components))
        for row in rows[:per_covariance]:
            key = (row.components, row.covariance_type)
            if key not in seen:
                shortlist.append(key)
                seen.add(key)

    if not shortlist:
        fallback_rows = [
            row for row in stage_one_results if row.converged and np.isfinite(row.bic)
        ]
        fallback_rows.sort(key=lambda row: (row.bic, row.components))
        for row in fallback_rows[: max(1, per_covariance)]:
            key = (row.components, row.covariance_type)
            if key not in seen:
                shortlist.append(key)
                seen.add(key)

    return shortlist


def _estimate_stability_ari(
    data: np.ndarray,
    candidate: GMMCandidateResult,
    max_iter: int,
    extended_max_iter: int,
    tol: float,
    init_params: str,
    seeds: Sequence[int],
) -> float:
    """Estimate repeated-seed stability for a near-best candidate."""

    label_runs: List[np.ndarray] = []
    for seed in seeds:
        fitted = _fit_candidate(
            data=data,
            stage="stability",
            n_components=candidate.components,
            covariance_type=candidate.covariance_type,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=candidate.reg_covar,
            n_init=candidate.n_init,
            random_state=int(seed),
        )
        if not fitted.passed_validity_filters or fitted.model is None:
            return float("nan")
        label_runs.append(fitted.model.predict(data))

    pairwise_scores: List[float] = []
    for idx in range(len(label_runs)):
        for jdx in range(idx + 1, len(label_runs)):
            pairwise_scores.append(
                float(adjusted_rand_score(label_runs[idx], label_runs[jdx]))
            )

    return float(np.mean(pairwise_scores)) if pairwise_scores else float("nan")


def _selection_key(candidate: GMMCandidateResult) -> Tuple[float, float, float, float, int, int]:
    """Order near-best candidates by quality proxy, then stability, then parsimony."""

    covariance_penalty = 0 if candidate.covariance_type == "diag" else 1
    return (
        -np.nan_to_num(candidate.avg_confidence, nan=-1.0),
        -np.nan_to_num(candidate.silhouette, nan=-1.0),
        -np.nan_to_num(candidate.stability_ari, nan=-1.0),
        candidate.bic,
        candidate.components,
        covariance_penalty,
    )


def _select_model(
    data: np.ndarray,
    stage_one_results: Sequence[GMMCandidateResult],
    stage_two_results: Sequence[GMMCandidateResult],
    max_iter: int,
    extended_max_iter: int,
    tol: float,
    init_params: str,
    bic_tolerance: float,
    stability_seeds: Sequence[int],
) -> Tuple[GMMCandidateResult, List[GMMCandidateResult], Dict[str, Any]]:
    """Choose the final GMM using the explicit staged selection procedure."""

    final_pool = [row for row in stage_two_results if row.passed_validity_filters]
    if not final_pool:
        final_pool = [row for row in stage_one_results if row.passed_validity_filters]
    if not final_pool:
        final_pool = [row for row in stage_two_results if row.converged]
    if not final_pool:
        final_pool = [row for row in stage_one_results if row.converged]
    if not final_pool:
        raise RuntimeError("No GMM candidates converged successfully.")

    best_bic = min(row.bic for row in final_pool)
    near_best_candidates = [
        row for row in final_pool if row.bic <= best_bic + bic_tolerance
    ]
    near_best_candidates.sort(key=lambda row: (row.bic, row.components))

    for row in near_best_candidates:
        row.near_best_bic = True
        row.stability_ari = _estimate_stability_ari(
            data=data,
            candidate=row,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            seeds=stability_seeds,
        )

    ranked_candidates = sorted(near_best_candidates, key=_selection_key)
    for rank, row in enumerate(ranked_candidates, start=1):
        row.selection_rank = rank

    selected = ranked_candidates[0]
    selected.selected = True
    if selected.model is None:
        raise RuntimeError("Selected GMM candidate does not have a fitted model.")

    summary = {
        "selection_procedure": {
            "stage_one": (
                "Search components in the recommended 4..40 working range using "
                "baseline reg_covar=1e-5 while comparing covariance_type in "
                "['diag', 'full']."
            ),
            "stage_two": (
                "For the top component counts per covariance type, sweep reg_covar "
                "across [1e-6, 1e-5, 1e-4, 1e-3]."
            ),
            "first_pass_filter": (
                f"Keep candidates within {bic_tolerance:.1f} BIC points of the best valid BIC."
            ),
            "final_ranking": [
                "higher avg_confidence",
                "higher silhouette",
                "higher repeated-seed stability ARI",
                "lower BIC",
                "fewer components",
                "prefer diag on exact tie",
            ],
            "stability_seeds": [int(seed) for seed in stability_seeds],
        },
        "selected_candidate": {
            "components": int(selected.components),
            "covariance_type": selected.covariance_type,
            "reg_covar": float(selected.reg_covar),
            "n_init": int(selected.n_init),
            "effective_max_iter": int(selected.effective_max_iter),
            "passed_validity_filters": bool(selected.passed_validity_filters),
            "bic": float(selected.bic),
            "aic": float(selected.aic),
            "avg_log_likelihood": float(selected.avg_log_likelihood),
            "silhouette": float(selected.silhouette),
            "avg_confidence": float(selected.avg_confidence),
            "stability_ari": float(selected.stability_ari),
            "cluster_size_distribution": json.loads(selected.cluster_size_distribution),
        },
        "candidate_counts": {
            "stage_one_total": int(len(stage_one_results)),
            "stage_two_total": int(len(stage_two_results)),
            "near_best_bic": int(len(near_best_candidates)),
        },
        "used_validity_fallback": bool(not selected.passed_validity_filters),
    }

    combined_results = list(stage_one_results) + list(stage_two_results)
    return selected, combined_results, summary


def _selection_diagnostics_frame(results: Sequence[GMMCandidateResult]) -> pd.DataFrame:
    """Convert candidate diagnostics into a stable DataFrame layout."""

    diagnostics = pd.DataFrame([row.to_row() for row in results])
    if diagnostics.empty:
        return diagnostics

    stage_order = {"component_search": 0, "reg_covar_sweep": 1, "fixed_fit": 2}
    diagnostics["_StageOrder"] = diagnostics["Stage"].map(stage_order).fillna(99)
    diagnostics = diagnostics.sort_values(
        by=[
            "_StageOrder",
            "CovarianceType",
            "Components",
            "RegCovar",
            "SelectionRank",
        ],
        kind="stable",
    ).drop(columns="_StageOrder")
    diagnostics.reset_index(drop=True, inplace=True)
    return diagnostics


def run_gmm_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
    output_dir: str = "output/clustering_results",
    metrics_dir: str = "output/metrics",
    n_components: int = 5,
    covariance_type: str = "diag",
    max_iter: int = 300,
    tol: float = 1e-3,
    init_params: str = "kmeans",
    reg_covar: float = 1e-5,
    n_init: int = 10,
    dynamic_component_selection: bool = True,
    dynamic_min_components: Optional[int] = None,
    dynamic_max_components: Optional[int] = None,
    dynamic_covariance_types: Optional[Sequence[str]] = None,
    reg_covar_grid: Optional[Sequence[float]] = None,
    bic_tolerance: float = DEFAULT_BIC_TOLERANCE,
    selection_n_jobs: int = 1,
    stability_seeds: Sequence[int] = DEFAULT_STABILITY_SEEDS,
    extended_max_iter: int = 500,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
    profile_id: Optional[str] = None,
):
    os.makedirs(results_dir, exist_ok=True)

    dataset_bundle = load_clustering_dataset_bundle(
        audio_dir=audio_dir,
        results_dir=results_dir,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
        selected_audio_feature_keys=selected_audio_feature_keys,
        equalization_method=equalization_method,
        pca_components=pca_components,
    )
    file_names = dataset_bundle["file_names"]
    genres = dataset_bundle["genres"]
    X_prepared = dataset_bundle["prepared_features"]
    metadata_frame = dataset_bundle["metadata_frame"]

    output_dir_path = Path(output_dir)
    metrics_dir_path = Path(metrics_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    metrics_dir_path.mkdir(parents=True, exist_ok=True)

    diagnostics_results: List[GMMCandidateResult] = []
    selection_summary: Optional[Dict[str, Any]] = None

    if dynamic_component_selection:
        min_components, max_components = _resolve_component_search_range(
            n_samples=X_prepared.shape[0],
            dynamic_min_components=dynamic_min_components,
            dynamic_max_components=dynamic_max_components,
        )
        covariance_types = tuple(dynamic_covariance_types or DEFAULT_COVARIANCE_TYPES)
        reg_values = tuple(reg_covar_grid or DEFAULT_REG_COVAR_GRID)

        print(
            f"GMM stage 1: searching components in [{min_components}, {max_components}] "
            f"for covariance types {list(covariance_types)} using reg_covar={reg_covar}"
        )
        stage_one_results = _run_candidate_grid(
            data=X_prepared,
            stage="component_search",
            component_values=list(range(min_components, max_components + 1)),
            covariance_types=covariance_types,
            reg_covar_values=[reg_covar],
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            n_init=max(10, n_init),
            n_jobs=selection_n_jobs,
        )

        shortlist = _shortlist_component_pairs(stage_one_results)
        if not shortlist:
            raise RuntimeError("GMM stage 1 did not produce any viable shortlist.")

        print(
            "GMM stage 2: sweeping reg_covar for shortlisted candidates -> "
            f"{shortlist} with reg grid {list(reg_values)}"
        )
        stage_two_results: List[GMMCandidateResult] = []
        for components, covariance in shortlist:
            stage_two_results.extend(
                _run_candidate_grid(
                    data=X_prepared,
                    stage="reg_covar_sweep",
                    component_values=[components],
                    covariance_types=[covariance],
                    reg_covar_values=reg_values,
                    max_iter=max_iter,
                    extended_max_iter=extended_max_iter,
                    tol=tol,
                    init_params=init_params,
                    n_init=max(10, n_init),
                    n_jobs=selection_n_jobs,
                )
            )

        selected_candidate, diagnostics_results, selection_summary = _select_model(
            data=X_prepared,
            stage_one_results=stage_one_results,
            stage_two_results=stage_two_results,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            bic_tolerance=bic_tolerance,
            stability_seeds=stability_seeds,
        )
    else:
        selected_candidate = _fit_candidate(
            data=X_prepared,
            stage="fixed_fit",
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            extended_max_iter=extended_max_iter,
            tol=tol,
            init_params=init_params,
            reg_covar=reg_covar,
            n_init=max(10, n_init),
        )
        if not selected_candidate.passed_validity_filters or selected_candidate.model is None:
            raise RuntimeError(
                "Fixed GMM fit did not converge to a healthy solution. "
                f"Details: {selected_candidate.error_message or selected_candidate.warning_summary}"
            )
        selected_candidate.selected = True
        selected_candidate.selection_rank = 1
        diagnostics_results = [selected_candidate]
        selection_summary = {
            "selection_procedure": {
                "stage_one": "Dynamic selection disabled; fixed candidate fit only.",
                "stage_two": "Not used.",
                "first_pass_filter": "Not used.",
                "final_ranking": ["fixed candidate only"],
                "stability_seeds": [],
            },
            "selected_candidate": {
                "components": int(selected_candidate.components),
                "covariance_type": selected_candidate.covariance_type,
                "reg_covar": float(selected_candidate.reg_covar),
                "n_init": int(selected_candidate.n_init),
                "effective_max_iter": int(selected_candidate.effective_max_iter),
                "passed_validity_filters": bool(
                    selected_candidate.passed_validity_filters
                ),
                "bic": float(selected_candidate.bic),
                "aic": float(selected_candidate.aic),
                "avg_log_likelihood": float(selected_candidate.avg_log_likelihood),
                "silhouette": float(selected_candidate.silhouette),
                "avg_confidence": float(selected_candidate.avg_confidence),
                "stability_ari": float("nan"),
                "cluster_size_distribution": json.loads(
                    selected_candidate.cluster_size_distribution
                ),
            },
            "candidate_counts": {
                "stage_one_total": 1,
                "stage_two_total": 0,
                "near_best_bic": 1,
            },
            "used_validity_fallback": False,
        }

    model = selected_candidate.model
    if model is None:
        raise RuntimeError("Selected GMM candidate is missing its fitted model.")

    posterior_probabilities = model.predict_proba(X_prepared)
    labels = posterior_probabilities.argmax(axis=1)
    probabilities = posterior_probabilities.max(axis=1)
    log_probs = model.score_samples(X_prepared)
    coords = compute_visualization_coords(X_prepared)

    df = pd.DataFrame(
        {
            "Song": file_names,
            "Artist": metadata_frame["Artist"].astype(str).to_numpy(),
            "Title": metadata_frame["Title"].astype(str).to_numpy(),
            "Filename": metadata_frame["Filename"].astype(str).to_numpy(),
            "MSDTrackID": metadata_frame["MSDTrackID"].astype(str).to_numpy(),
            "GenreList": metadata_frame["GenreList"].astype(str).to_numpy(),
            "PrimaryGenres": metadata_frame["PrimaryGenres"].astype(str).to_numpy(),
            "SecondaryTags": metadata_frame["SecondaryTags"].astype(str).to_numpy(),
            "AllGenreTags": metadata_frame["AllGenreTags"].astype(str).to_numpy(),
            "OriginalGenreList": metadata_frame["OriginalGenreList"].astype(str).to_numpy(),
            "OriginalPrimaryGenre": metadata_frame["OriginalPrimaryGenre"].astype(str).to_numpy(),
            "Genre": genres,
            "Cluster": labels,
            "Confidence": probabilities,
            "LogLikelihood": log_probs,
            "PCA1": coords[:, 0],
            "PCA2": coords[:, 1],
        }
    )

    diagnostics_df = _selection_diagnostics_frame(diagnostics_results)
    diagnostics_path = metrics_dir_path / "gmm_selection_criteria.csv"
    diagnostics_df.to_csv(diagnostics_path, index=False)
    print(f"Stored GMM selection diagnostics -> {diagnostics_path}")

    selection_summary = dict(selection_summary or {})
    selection_summary.update(
        {
            "method_id": "gmm",
            "profile_id": profile_id or "unspecified",
            "representation": {
                "feature_subset_name": dataset_bundle["qc_summary"].get(
                    "feature_subset_name"
                ),
                "selected_audio_feature_keys": dataset_bundle["qc_summary"].get(
                    "selected_audio_feature_keys"
                ),
                "equalization_method": dataset_bundle["qc_summary"].get(
                    "equalization_method"
                ),
                "pca_components_per_group": dataset_bundle["qc_summary"].get(
                    "pca_components_per_group"
                ),
                "raw_feature_dimension": dataset_bundle["qc_summary"].get(
                    "raw_feature_dimension"
                ),
                "prepared_feature_dimension": dataset_bundle["qc_summary"].get(
                    "prepared_feature_dimension"
                ),
                "include_genre": bool(include_genre),
                "include_msd_requested": bool(include_msd),
                "include_msd_effective": dataset_bundle["qc_summary"].get(
                    "include_msd_effective"
                ),
            },
            "search_config": {
                "dynamic_component_selection": bool(dynamic_component_selection),
                "fixed_n_components": int(n_components),
                "fixed_covariance_type": covariance_type,
                "max_iter": int(max_iter),
                "extended_max_iter": int(extended_max_iter),
                "tol": float(tol),
                "init_params": init_params,
                "n_init": int(max(10, n_init)),
                "reg_covar": float(reg_covar),
                "bic_tolerance": float(bic_tolerance),
                "selection_n_jobs": int(selection_n_jobs),
                "stability_seeds": [int(seed) for seed in stability_seeds],
            },
        }
    )
    csv_path = output_dir_path / "audio_clustering_results_gmm.csv"
    run_qc_csv_path, run_qc_json_path = snapshot_dataset_qc_artifacts(
        dataset_bundle["qc_csv_path"],
        dataset_bundle["qc_json_path"],
        str(metrics_dir_path),
    )
    selection_summary["outputs"] = {
        "results_csv": str(csv_path),
        "selection_criteria_csv": str(diagnostics_path),
        "dataset_qc_summary_json": run_qc_json_path,
        "dataset_qc_csv": run_qc_csv_path,
    }
    summary_path = metrics_dir_path / "gmm_selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(selection_summary, handle, indent=2)
    print(f"Stored GMM selection summary -> {summary_path}")

    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    artifact_path = save_retrieval_artifact(
        method_id="gmm",
        file_names=file_names,
        prepared_features=X_prepared,
        labels=labels,
        coords=coords,
        output_dir=str(output_dir_path),
        artists=metadata_frame["Artist"].astype(str).to_numpy(),
        titles=metadata_frame["Title"].astype(str).to_numpy(),
        filenames=metadata_frame["Filename"].astype(str).to_numpy(),
        msd_track_ids=metadata_frame["MSDTrackID"].astype(str).to_numpy(),
        assignment_confidence=probabilities,
        posterior_probabilities=posterior_probabilities,
        log_likelihood=log_probs,
        feature_subset_name=dataset_bundle["qc_summary"].get("feature_subset_name"),
        selected_audio_feature_keys=dataset_bundle["qc_summary"].get(
            "selected_audio_feature_keys"
        ),
        feature_equalization_method=dataset_bundle["qc_summary"].get(
            "equalization_method"
        ),
        pca_components_per_group=dataset_bundle["qc_summary"].get(
            "pca_components_per_group"
        ),
        raw_feature_dimension=dataset_bundle["qc_summary"].get("raw_feature_dimension"),
        prepared_feature_dimension=dataset_bundle["qc_summary"].get(
            "prepared_feature_dimension"
        ),
        profile_id=profile_id,
    )
    selection_summary["outputs"]["retrieval_artifact"] = str(artifact_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(selection_summary, handle, indent=2)

    cluster_counts = np.bincount(labels, minlength=selected_candidate.components)
    print(
        "Selected GMM -> "
        f"components={selected_candidate.components}, "
        f"covariance_type={selected_candidate.covariance_type}, "
        f"reg_covar={selected_candidate.reg_covar}, "
        f"n_init={selected_candidate.n_init}, "
        f"effective_max_iter={selected_candidate.effective_max_iter}"
    )
    print(
        "Selection metrics -> "
        f"BIC={selected_candidate.bic:.2f}, "
        f"AIC={selected_candidate.aic:.2f}, "
        f"avg_log_likelihood={selected_candidate.avg_log_likelihood:.4f}, "
        f"silhouette={selected_candidate.silhouette:.4f}, "
        f"avg_confidence={selected_candidate.avg_confidence:.4f}, "
        f"stability_ari={selected_candidate.stability_ari:.4f}"
    )
    print(
        "Cluster sizes -> "
        + ", ".join(
            f"{idx}:{int(count)}"
            for idx, count in enumerate(cluster_counts)
            if int(count) > 0
        )
    )
    if not selected_candidate.passed_validity_filters:
        print(
            "Warning: no candidate passed the full degeneracy filters; "
            "using the best converged fallback candidate."
        )

    return df, coords, labels


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GMM clustering and optionally launch the interactive clustering UI."
        )
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive clustering UI after clustering finishes.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help=(
            "Deprecated compatibility flag. The UI is skipped by default unless "
            "--ui is provided."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    DF, COORDS, LABELS = run_gmm_clustering(
        audio_dir="audio_files",
        results_dir="output/features",
        n_components=5,
        dynamic_component_selection=True,
        include_genre=fv.include_genre,
    )

    if args.ui and args.no_ui:
        raise SystemExit("Use either --ui or --no-ui, not both.")

    if args.ui:
        from src.ui.modern_ui import launch_ui  # noqa: E402

        launch_ui(
            DF,
            COORDS,
            LABELS,
            audio_dir="audio_files",
            clustering_method="GMM",
            retrieval_method_id="gmm",
        )
    else:
        print(
            "Skipping interactive clustering UI. Run 'python src/ui/modern_ui.py' "
            "to open the latest benchmark-linked UI snapshot."
        )
