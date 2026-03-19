import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment_profiles import (  # noqa: E402
    RECOMMENDED_PRODUCTION_PROFILE_ID,
    build_canonical_baseline_contract,
    get_profile,
    list_profile_ids,
)
from scripts.analysis.evaluate_clustering import run_evaluation  # noqa: E402
from src.clustering.gmm import run_gmm_clustering  # noqa: E402
from src.clustering.hdbscan import run_hdbscan_clustering  # noqa: E402
from src.clustering.kmeans import run_kmeans_clustering  # noqa: E402
from src.clustering.vade import run_vade_clustering  # noqa: E402


METHOD_CHOICES = ("kmeans", "gmm", "hdbscan", "vade")
METHOD_SUMMARY_FILENAMES = {
    "kmeans": "kmeans_run_summary.json",
    "gmm": "gmm_selection_summary.json",
    "hdbscan": "hdbscan_run_summary.json",
    "vade": "vade_run_summary.json",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more explicit clustering comparison profiles and store "
            "timestamped experiment manifests."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list_profile_ids(),
        default=[RECOMMENDED_PRODUCTION_PROFILE_ID],
        help="Experiment profiles to run (default: recommended_production).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=None,
        help="Optional method override for every selected profile.",
    )
    parser.add_argument(
        "--include-vade",
        action="store_true",
        help="Append VaDE as an explicit optional comparison method.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip the recommendation/stability evaluation stage.",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="output/experiment_runs",
        help="Root directory for timestamped experiment manifests.",
    )
    parser.add_argument(
        "--stability-jobs",
        type=int,
        default=1,
        help="Parallel jobs passed through to evaluation stability checks.",
    )
    return parser.parse_args()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_methods(
    profile: Dict[str, Any],
    methods_override: Optional[Sequence[str]],
    include_vade: bool,
) -> List[str]:
    if methods_override:
        methods = [str(method).strip().lower() for method in methods_override]
    else:
        methods = [
            str(method).strip().lower()
            for method in profile["default_comparison_methods"]
        ]

    if include_vade and "vade" not in methods:
        methods.append("vade")

    deduped: List[str] = []
    for method in methods:
        if method not in deduped:
            deduped.append(method)
    return deduped


def _build_common_kwargs(
    profile: Dict[str, Any],
    artifact_dir: Path,
    metrics_dir: Path,
) -> Dict[str, Any]:
    include_msd = bool(profile["include_msd_features"])
    return {
        "audio_dir": "audio_files",
        "results_dir": "output/features",
        "output_dir": str(artifact_dir),
        "metrics_dir": str(metrics_dir),
        "include_genre": bool(profile["include_genre"]),
        "include_msd": include_msd,
        "songs_csv_path": "data/songs.csv" if include_msd else None,
        "selected_audio_feature_keys": list(profile["selected_audio_feature_keys"]),
        "equalization_method": str(profile["equalization_method"]),
        "pca_components": profile.get("pca_components_per_group"),
        "profile_id": str(profile["profile_id"]),
    }


def _run_method(
    method_id: str,
    common_kwargs: Dict[str, Any],
) -> tuple[Any, Any, Any]:
    if method_id == "kmeans":
        return run_kmeans_clustering(
            dynamic_cluster_selection=True,
            **common_kwargs,
        )
    if method_id == "gmm":
        return run_gmm_clustering(
            dynamic_component_selection=True,
            selection_n_jobs=max(1, min(4, os.cpu_count() or 1)),
            **common_kwargs,
        )
    if method_id == "hdbscan":
        return run_hdbscan_clustering(
            dynamic_parameter_selection=True,
            **common_kwargs,
        )
    if method_id == "vade":
        return run_vade_clustering(
            dynamic_component_selection=True,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported method: {method_id}")


def _build_method_row(
    method_id: str,
    df,
    labels,
    metrics_dir: Path,
) -> Dict[str, Any]:
    summary_path = metrics_dir / METHOD_SUMMARY_FILENAMES[method_id]
    method_summary = _load_json(summary_path) if summary_path.exists() else {}
    row = {
        "rows": int(len(df)),
        "method_summary_path": str(summary_path),
        "method_summary": method_summary,
    }
    if method_id == "hdbscan":
        row["clusters"] = int(len({int(x) for x in labels if int(x) != -1}))
        row["noise"] = int(sum(1 for x in labels if int(x) == -1))
    else:
        row["clusters"] = int(len(set(map(int, labels))))
    return row


def _copy_profile_outputs(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, destination_dir / path.name)


def _promote_profile_to_default_outputs(profile_root: Path) -> Dict[str, str]:
    source_artifacts = profile_root / "clustering_results"
    source_metrics = profile_root / "metrics"
    default_artifacts = Path("output/clustering_results")
    default_metrics = Path("output/metrics")
    default_artifacts.mkdir(parents=True, exist_ok=True)
    default_metrics.mkdir(parents=True, exist_ok=True)
    _copy_profile_outputs(source_artifacts, default_artifacts)
    _copy_profile_outputs(source_metrics, default_metrics)
    return {
        "artifact_dir": str(default_artifacts),
        "metrics_dir": str(default_metrics),
    }


def main() -> None:
    args = _parse_args()

    run_root = Path(args.run_root) / f"run_{_timestamp_slug()}"
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_contract = build_canonical_baseline_contract()
    baseline_contract_path = _write_json(
        run_root / "canonical_baseline_contract.json",
        baseline_contract,
    )

    run_manifest: Dict[str, Any] = {
        "run_id": run_root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_baseline_contract_path": str(baseline_contract_path),
        "requested_profiles": list(args.profiles),
        "methods_override": list(args.methods) if args.methods else None,
        "include_vade": bool(args.include_vade),
        "skip_evaluation": bool(args.skip_evaluation),
        "profiles": {},
    }

    for profile_id in args.profiles:
        profile = get_profile(profile_id)
        methods = _normalize_methods(profile, args.methods, args.include_vade)
        profile_root = run_root / profile_id
        artifact_dir = profile_root / "clustering_results"
        metrics_dir = profile_root / "metrics"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"PROFILE: {profile['title']} ({profile_id})")
        print("=" * 80)
        print(
            f"Methods: {methods} | subset={profile['feature_subset_name']} | "
            f"equalization={profile['equalization_method']}"
        )

        common_kwargs = _build_common_kwargs(profile, artifact_dir, metrics_dir)
        profile_manifest: Dict[str, Any] = {
            "profile": profile,
            "artifact_dir": str(artifact_dir),
            "metrics_dir": str(metrics_dir),
            "methods": {},
            "evaluation": None,
            "promoted_to_default_outputs": False,
        }

        for method_id in methods:
            print(f"\nRunning {method_id} for profile {profile_id}...")
            df, _, labels = _run_method(method_id, common_kwargs)
            profile_manifest["methods"][method_id] = _build_method_row(
                method_id=method_id,
                df=df,
                labels=labels,
                metrics_dir=metrics_dir,
            )

        if not args.skip_evaluation:
            evaluation_result = run_evaluation(
                methods=methods,
                artifact_dir=artifact_dir,
                metrics_dir=metrics_dir,
                prefix=profile_id,
                stability_jobs=int(args.stability_jobs),
            )
            profile_manifest["evaluation"] = {
                "summary_path": evaluation_result["summary_path"],
                "comparison_csv": evaluation_result["comparison_csv"],
                "comparison_report": evaluation_result["comparison_report"],
            }

        profile_manifest["profile_manifest_path"] = str(
            profile_root / "profile_manifest.json"
        )

        if profile_id == RECOMMENDED_PRODUCTION_PROFILE_ID:
            promoted_paths = _promote_profile_to_default_outputs(profile_root)
            profile_manifest["promoted_to_default_outputs"] = True
            profile_manifest["default_output_paths"] = promoted_paths
        profile_manifest_path = _write_json(
            Path(profile_manifest["profile_manifest_path"]),
            profile_manifest,
        )

        run_manifest["profiles"][profile_id] = profile_manifest

    run_manifest_path = _write_json(run_root / "run_manifest.json", run_manifest)
    print("\nExperiment suite finished.")
    print(f"Run manifest -> {run_manifest_path}")


if __name__ == "__main__":
    main()
