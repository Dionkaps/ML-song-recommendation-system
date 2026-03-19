from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from config import feature_vars as fv


CANONICAL_BASELINE_VERSION = 1
RECOMMENDED_PRODUCTION_PROFILE_ID = "recommended_production"
ALL_AUDIO_PCA_COMPARISON_PROFILE_ID = "all_audio_pca_comparison"
ALL_AUDIO_ZSCORE_COMPARISON_PROFILE_ID = "all_audio_zscore_comparison"

DEFAULT_COMPARISON_METHODS = ["kmeans", "gmm", "hdbscan"]
OPTIONAL_COMPARISON_METHODS = ["vade"]
DECISION_POLICY_DOC = "docs/DECISION_POLICY.md"

PROXY_EVALUATION_LIMITATION = (
    "Recommendation quality is still being estimated with metadata proxies "
    "(genre and artist overlap plus coverage/diversity checks) rather than "
    "human similarity judgments or listening tests."
)

FUTURE_VALIDATION_BACKLOG = [
    "Run a small blind listening study on top-K recommendations from the recommended production profile.",
    "Collect human similarity judgments for difficult cases such as uncertain GMM assignments and cross-genre neighbors.",
    "Revisit MSD numeric metadata only as an explicitly logged experiment after coverage gaps are resolved.",
]


def build_decision_policy_contract() -> Dict[str, Any]:
    return {
        "cluster_granularity": {
            "policy": str(fv.product_cluster_granularity_policy),
            "target_cluster_range": [
                int(fv.product_cluster_target_min),
                int(fv.product_cluster_target_max),
            ],
            "current_reference_target": int(fv.product_cluster_target_default),
            "decision": (
                "Prefer a few broad macro-style clusters for product navigation and "
                "retrieval, not many micro-style clusters."
            ),
        },
        "gmm_stability_gate": {
            "minimum_subsample_median_ari": float(fv.gmm_min_subsample_median_ari),
            "minimum_subsample_mean_ari": float(fv.gmm_min_subsample_mean_ari),
            "minimum_reference_median_ari": float(fv.gmm_min_reference_median_ari),
            "minimum_per_cluster_median_jaccard": float(
                fv.gmm_min_per_cluster_median_jaccard
            ),
            "decision": (
                "A GMM may remain the production default only if it clears the explicit "
                "stability gate under the final evaluation protocol."
            ),
        },
        "uncertain_assignments": {
            "policy": str(fv.uncertain_gmm_assignment_policy),
            "default_ranking_method": str(fv.default_recommendation_ranking_method),
            "default_min_assignment_confidence": float(
                fv.default_min_assignment_confidence
            ),
            "default_min_selected_cluster_posterior": float(
                fv.default_min_selected_cluster_posterior
            ),
            "decision": (
                "Keep uncertain GMM assignments visible by default and expose "
                "posterior-weighted ranking plus hard thresholds as optional controls."
            ),
        },
        "msd_restore_gate": {
            "minimum_live_audio_coverage_fraction": float(
                fv.msd_restore_min_live_audio_coverage
            ),
            "maximum_missing_audio_rows": int(fv.msd_restore_max_missing_audio_rows),
            "require_clean_schema_audit": bool(
                fv.msd_restore_require_clean_schema_audit
            ),
            "require_explicit_experiment_profile": bool(
                fv.msd_restore_require_explicit_experiment_profile
            ),
            "require_fresh_comparison_run": bool(
                fv.msd_restore_require_fresh_comparison_run
            ),
            "require_no_silent_fallback": bool(
                fv.msd_restore_require_no_silent_fallback
            ),
            "decision": (
                "Restore MSD numeric metadata only after near-complete live-audio "
                "coverage and only as an explicitly logged comparison experiment."
            ),
        },
    }


def _audio_group_dimension_map() -> Dict[str, int]:
    return {
        "mfcc": 2 * int(fv.n_mfcc),
        "delta_mfcc": 2 * int(fv.n_mfcc),
        "delta2_mfcc": 2 * int(fv.n_mfcc),
        "spectral_centroid": 2,
        "spectral_rolloff": 2,
        "spectral_flux": 2,
        "spectral_flatness": 2,
        "zero_crossing_rate": 2,
        "chroma": 2 * int(fv.n_chroma),
        "beat_strength": 4,
    }


def _raw_audio_dimension(selected_audio_feature_keys: List[str]) -> int:
    dim_map = _audio_group_dimension_map()
    return int(sum(dim_map[key] for key in selected_audio_feature_keys))


def _prepared_audio_dimension(
    selected_audio_feature_keys: List[str],
    equalization_method: str,
    pca_components_per_group: int | None,
) -> int:
    if equalization_method == "pca_per_group":
        if pca_components_per_group is None:
            raise ValueError("pca_components_per_group must be provided for pca_per_group")
        return int(len(selected_audio_feature_keys) * pca_components_per_group)
    return _raw_audio_dimension(selected_audio_feature_keys)


def _build_profile(
    profile_id: str,
    title: str,
    description: str,
    selected_audio_feature_keys: List[str],
    equalization_method: str,
    pca_components_per_group: int | None,
    status: str,
) -> Dict[str, Any]:
    return {
        "profile_id": profile_id,
        "title": title,
        "status": status,
        "description": description,
        "selected_audio_feature_keys": list(selected_audio_feature_keys),
        "feature_subset_name": (
            fv.clustering_feature_subset_name
            if list(selected_audio_feature_keys) == list(fv.clustering_audio_feature_keys)
            else profile_id
        ),
        "equalization_method": str(equalization_method),
        "pca_components_per_group": (
            None if pca_components_per_group is None else int(pca_components_per_group)
        ),
        "expected_raw_dimension": _raw_audio_dimension(selected_audio_feature_keys),
        "expected_prepared_dimension": _prepared_audio_dimension(
            selected_audio_feature_keys,
            equalization_method,
            pca_components_per_group,
        ),
        "include_genre": bool(False),
        "include_msd_features": bool(False),
        "default_comparison_methods": list(DEFAULT_COMPARISON_METHODS),
        "optional_comparison_methods": list(OPTIONAL_COMPARISON_METHODS),
    }


PROFILE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    RECOMMENDED_PRODUCTION_PROFILE_ID: _build_profile(
        profile_id=RECOMMENDED_PRODUCTION_PROFILE_ID,
        title="Recommended Production Baseline",
        description=(
            "Supported audio-only baseline using spectral_plus_beat, per-group scaling, "
            "and pca_per_group_5. This is the profile that should feed the default UI and reports."
        ),
        selected_audio_feature_keys=list(fv.clustering_audio_feature_keys),
        equalization_method=str(fv.feature_equalization_method),
        pca_components_per_group=int(fv.pca_components_per_group),
        status="recommended_production",
    ),
    ALL_AUDIO_PCA_COMPARISON_PROFILE_ID: _build_profile(
        profile_id=ALL_AUDIO_PCA_COMPARISON_PROFILE_ID,
        title="All Audio PCA Comparison",
        description=(
            "Explicit comparison baseline using the full handcrafted audio set with the same "
            "per-group PCA equalization policy."
        ),
        selected_audio_feature_keys=list(fv.AUDIO_FEATURE_KEYS),
        equalization_method="pca_per_group",
        pca_components_per_group=int(fv.pca_components_per_group),
        status="comparison_only",
    ),
    ALL_AUDIO_ZSCORE_COMPARISON_PROFILE_ID: _build_profile(
        profile_id=ALL_AUDIO_ZSCORE_COMPARISON_PROFILE_ID,
        title="All Audio Raw Z-Score Comparison",
        description=(
            "Explicit comparison baseline using the full handcrafted audio set after plain "
            "z-score standardization, without per-group PCA equalization."
        ),
        selected_audio_feature_keys=list(fv.AUDIO_FEATURE_KEYS),
        equalization_method="zscore",
        pca_components_per_group=None,
        status="comparison_only",
    ),
}


def list_profile_ids() -> List[str]:
    return list(PROFILE_DEFINITIONS.keys())


def get_profile(profile_id: str) -> Dict[str, Any]:
    normalized = profile_id.strip().lower()
    if normalized not in PROFILE_DEFINITIONS:
        raise KeyError(
            f"Unknown experiment profile '{profile_id}'. "
            f"Expected one of {sorted(PROFILE_DEFINITIONS)}."
        )
    return deepcopy(PROFILE_DEFINITIONS[normalized])


def get_profiles(profile_ids: List[str] | None = None) -> List[Dict[str, Any]]:
    ids = profile_ids or [RECOMMENDED_PRODUCTION_PROFILE_ID]
    return [get_profile(profile_id) for profile_id in ids]


def build_canonical_baseline_contract() -> Dict[str, Any]:
    production_profile = get_profile(RECOMMENDED_PRODUCTION_PROFILE_ID)
    comparison_profiles = [
        get_profile(ALL_AUDIO_PCA_COMPARISON_PROFILE_ID),
        get_profile(ALL_AUDIO_ZSCORE_COMPARISON_PROFILE_ID),
    ]
    return {
        "contract_version": int(CANONICAL_BASELINE_VERSION),
        "supported_clustering_mode": str(fv.supported_clustering_mode),
        "default_clustering_method": str(fv.default_clustering_method),
        "recommended_production_profile_id": RECOMMENDED_PRODUCTION_PROFILE_ID,
        "recommended_production_profile": production_profile,
        "documented_comparison_profiles": comparison_profiles,
        "msd_metadata_policy": str(getattr(fv, "msd_metadata_policy", "unknown")),
        "msd_metadata_restore_policy": str(
            getattr(fv, "msd_metadata_restore_policy", "unknown")
        ),
        "audio_preprocessing_contract": {
            "target_duration_seconds": float(fv.baseline_target_duration_seconds),
            "target_lufs": float(fv.baseline_target_lufs),
            "max_peak_dbfs_sample_ceiling": float(fv.baseline_max_true_peak_dbtp),
            "sample_rate_hz": int(fv.baseline_sample_rate),
            "output_subtype": str(fv.baseline_output_subtype),
            "force_mono": bool(fv.baseline_force_mono),
        },
        "decision_policy": build_decision_policy_contract(),
        "evaluation_limitations": [PROXY_EVALUATION_LIMITATION],
        "future_validation_backlog": list(FUTURE_VALIDATION_BACKLOG),
        "docs": {
            "supported_baseline": "docs/SUPPORTED_BASELINE.md",
            "recommended_production_baseline": "docs/RECOMMENDED_PRODUCTION_BASELINE.md",
            "decision_policy": DECISION_POLICY_DOC,
        },
    }
