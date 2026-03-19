import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.clustering import kmeans as clustering_kmeans
from src.features.extract_features import process_file
from src.features.feature_qc import (
    FEATURE_KEYS,
    collect_feature_bundle_inventory,
    load_validated_feature_bundle,
    remove_feature_bundle,
)


def build_audio_path_map(audio_dir: str) -> Dict[str, Path]:
    """Map each audio basename to its concrete path in the library."""

    audio_path_map: Dict[str, Path] = {}
    audio_root = Path(audio_dir)
    for pattern in ("*.wav", "*.mp3", "*.flac", "*.m4a"):
        for path in sorted(audio_root.glob(pattern)):
            audio_path_map[path.stem] = path
    return dict(sorted(audio_path_map.items()))


def clean_stale_genre_cache(results_dir: str, audio_basenames: List[str]) -> List[str]:
    """Remove cached genre entries that no longer correspond to real audio files."""

    genre_map_path = Path(results_dir) / "genre_map.npy"
    if not genre_map_path.exists():
        return []

    genre_map = np.load(genre_map_path, allow_pickle=True).item()
    audio_base_set = set(audio_basenames)
    stale_entries = sorted(set(genre_map) - audio_base_set)
    if not stale_entries:
        return []

    cleaned = {
        base_name: genre
        for base_name, genre in genre_map.items()
        if base_name in audio_base_set
    }
    np.save(genre_map_path, cleaned)
    return stale_entries


def inspect_and_repair_feature_library(
    audio_dir: str,
    results_dir: str,
    repair: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Validate complete feature bundles, optionally re-extracting broken tracks."""

    audio_path_map = build_audio_path_map(audio_dir)
    audio_basenames = list(audio_path_map.keys())
    bundle_inventory = collect_feature_bundle_inventory(results_dir, FEATURE_KEYS)

    track_rows: List[Dict[str, Any]] = []
    complete_before_repair = 0
    reextract_attempted = 0
    reextract_succeeded = 0
    reextract_failed = 0
    incomplete_before_repair = 0
    invalid_before_repair = 0

    for base_name in audio_basenames:
        _, initial_validation = load_validated_feature_bundle(
            base_name,
            results_dir,
            feature_keys=FEATURE_KEYS,
            n_mfcc=fv.n_mfcc,
            n_chroma=fv.n_chroma,
        )
        final_validation = initial_validation
        action = "kept"
        removed_files: List[str] = []
        reextract_result: Dict[str, Any] = {}

        if initial_validation.is_valid:
            complete_before_repair += 1
        else:
            if initial_validation.status == "incomplete":
                incomplete_before_repair += 1
            else:
                invalid_before_repair += 1

            if repair:
                reextract_attempted += 1
                action = "re_extract_attempted"
                removed_files = remove_feature_bundle(
                    base_name,
                    results_dir,
                    FEATURE_KEYS,
                )
                reextract_result = process_file(
                    str(audio_path_map[base_name]),
                    results_dir,
                    fv.n_mfcc,
                    fv.n_fft,
                    fv.hop_length,
                    fv.n_mels,
                )
                _, final_validation = load_validated_feature_bundle(
                    base_name,
                    results_dir,
                    feature_keys=FEATURE_KEYS,
                    n_mfcc=fv.n_mfcc,
                    n_chroma=fv.n_chroma,
                )
                if reextract_result.get("status") == "processed" and final_validation.is_valid:
                    action = "re_extracted_ok"
                    reextract_succeeded += 1
                else:
                    action = "re_extract_failed"
                    reextract_failed += 1

        track_rows.append(
            {
                "Category": "audio_track",
                "BaseName": base_name,
                "AudioPath": str(audio_path_map[base_name]),
                "InitialStatus": initial_validation.status,
                "InitialMissingKeys": ",".join(initial_validation.missing_keys),
                "InitialInvalidKeys": ",".join(initial_validation.invalid_keys),
                "InitialPresentKeys": ",".join(initial_validation.present_keys),
                "InitialIssueDetails": json.dumps(
                    initial_validation.issue_details,
                    sort_keys=True,
                ),
                "Action": action,
                "ReextractAttempted": bool(action != "kept"),
                "RemovedBundleFileCount": int(len(removed_files)),
                "RemovedBundleFiles": json.dumps(removed_files, sort_keys=True),
                "ReextractResult": json.dumps(reextract_result, sort_keys=True),
                "FinalValidationStatus": final_validation.status,
                "FinalMissingKeys": ",".join(final_validation.missing_keys),
                "FinalInvalidKeys": ",".join(final_validation.invalid_keys),
                "FinalPresentKeys": ",".join(final_validation.present_keys),
                "FinalIssueDetails": json.dumps(
                    final_validation.issue_details,
                    sort_keys=True,
                ),
            }
        )

    stale_feature_bases = sorted(set(bundle_inventory) - set(audio_basenames))
    for base_name in stale_feature_bases:
        track_rows.append(
            {
                "Category": "stale_feature_bundle",
                "BaseName": base_name,
                "AudioPath": "",
                "InitialStatus": "stale_feature_bundle",
                "InitialMissingKeys": "",
                "InitialInvalidKeys": "",
                "InitialPresentKeys": ",".join(bundle_inventory.get(base_name, [])),
                "InitialIssueDetails": json.dumps(
                    {"stale_feature_bundle": "no_matching_audio_file"},
                    sort_keys=True,
                ),
                "Action": "reported_only",
                "ReextractAttempted": False,
                "RemovedBundleFileCount": 0,
                "RemovedBundleFiles": "[]",
                "ReextractResult": "{}",
                "FinalValidationStatus": "stale_feature_bundle",
                "FinalMissingKeys": "",
                "FinalInvalidKeys": "",
                "FinalPresentKeys": ",".join(bundle_inventory.get(base_name, [])),
                "FinalIssueDetails": json.dumps(
                    {"stale_feature_bundle": "no_matching_audio_file"},
                    sort_keys=True,
                ),
            }
        )

    summary = {
        "audio_tracks_scanned": int(len(audio_basenames)),
        "complete_before_repair": int(complete_before_repair),
        "incomplete_before_repair": int(incomplete_before_repair),
        "invalid_before_repair": int(invalid_before_repair),
        "reextract_attempted": int(reextract_attempted),
        "reextract_succeeded": int(reextract_succeeded),
        "reextract_failed": int(reextract_failed),
        "stale_feature_bundles": int(len(stale_feature_bases)),
    }
    return track_rows, summary, audio_basenames


def _summarize_block_variance(block: np.ndarray) -> Dict[str, Any]:
    """Return compact variance diagnostics for a feature block."""

    variances = np.var(block, axis=0)
    active_mask = variances > 1e-10
    active_variances = variances[active_mask]

    return {
        "mean_variance_all_dims": float(np.mean(variances)),
        "median_variance_all_dims": float(np.median(variances)),
        "min_variance_all_dims": float(np.min(variances)),
        "max_variance_all_dims": float(np.max(variances)),
        "active_dims_observed": int(np.count_nonzero(active_mask)),
        "near_zero_dims_observed": int(np.count_nonzero(~active_mask)),
        "mean_variance_active_dims": (
            float(np.mean(active_variances)) if active_variances.size else 0.0
        ),
        "median_variance_active_dims": (
            float(np.median(active_variances)) if active_variances.size else 0.0
        ),
        "mean_sample_l2_norm": float(np.mean(np.linalg.norm(block, axis=1))),
        "std_sample_l2_norm": float(np.std(np.linalg.norm(block, axis=1))),
    }


def build_group_variance_rows(
    raw_features: np.ndarray,
    prepared_features: np.ndarray,
    selected_audio_feature_keys: List[str],
    pca_components: int,
) -> List[Dict[str, Any]]:
    """Measure raw, scaled, and prepared variance for the active clustering groups."""

    group_specs = clustering_kmeans._get_audio_group_specs(
        n_mfcc=fv.n_mfcc,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )

    rows: List[Dict[str, Any]] = []
    raw_start = 0
    prepared_start = 0

    for key, display_name, raw_size in group_specs:
        raw_end = raw_start + raw_size
        prepared_end = prepared_start + pca_components

        raw_block = raw_features[:, raw_start:raw_end]
        scaled_block = StandardScaler().fit_transform(raw_block)

        if raw_size > pca_components:
            n_components = min(pca_components, raw_size, raw_features.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=42)
            transformed = pca.fit_transform(scaled_block)
            explained_variance_ratio_sum = float(np.sum(pca.explained_variance_ratio_))
            if transformed.shape[1] < pca_components:
                padding = np.zeros(
                    (raw_features.shape[0], pca_components - transformed.shape[1])
                )
                transformed = np.hstack([transformed, padding])
        elif raw_size < pca_components:
            explained_variance_ratio_sum = 1.0
            padding = np.zeros((raw_features.shape[0], pca_components - raw_size))
            transformed = np.hstack([scaled_block, padding])
        else:
            explained_variance_ratio_sum = 1.0
            transformed = scaled_block

        group_norm = np.sqrt(np.mean(np.sum(transformed ** 2, axis=1)))
        if group_norm > 1e-10:
            transformed = transformed / group_norm

        prepared_block = prepared_features[:, prepared_start:prepared_end]

        expected_active_dims = min(raw_size, pca_components)
        expected_zero_padded_dims = max(0, pca_components - raw_size)

        for stage_name, block in (
            ("raw_summary", raw_block),
            ("scaled_pre_equalization", scaled_block),
            ("prepared_equalized_rebuilt", transformed),
            ("prepared_equalized_runtime", prepared_block),
        ):
            row = {
                "FeatureKey": key,
                "FeatureGroup": display_name,
                "Stage": stage_name,
                "InputDims": int(raw_size),
                "PreparedDims": int(pca_components),
                "ExpectedActiveDims": int(expected_active_dims),
                "ExpectedZeroPaddedDims": int(expected_zero_padded_dims),
                "ExplainedVarianceRatioSum": (
                    explained_variance_ratio_sum
                    if stage_name.startswith("prepared_equalized")
                    else np.nan
                ),
            }
            row.update(_summarize_block_variance(block))
            rows.append(row)

        rebuilt_matches_runtime = bool(
            np.allclose(transformed, prepared_block, atol=1e-5, rtol=1e-5)
        )
        rows.append(
            {
                "FeatureKey": key,
                "FeatureGroup": display_name,
                "Stage": "prepared_consistency_check",
                "InputDims": int(raw_size),
                "PreparedDims": int(pca_components),
                "ExpectedActiveDims": int(expected_active_dims),
                "ExpectedZeroPaddedDims": int(expected_zero_padded_dims),
                "ExplainedVarianceRatioSum": explained_variance_ratio_sum,
                "mean_variance_all_dims": np.nan,
                "median_variance_all_dims": np.nan,
                "min_variance_all_dims": np.nan,
                "max_variance_all_dims": np.nan,
                "active_dims_observed": np.nan,
                "near_zero_dims_observed": np.nan,
                "mean_variance_active_dims": np.nan,
                "median_variance_active_dims": np.nan,
                "mean_sample_l2_norm": np.nan,
                "std_sample_l2_norm": np.nan,
                "PreparedMatchesRuntime": rebuilt_matches_runtime,
            }
        )

        raw_start = raw_end
        prepared_start = prepared_end

    return rows


def write_markdown_summary(
    output_path: Path,
    summary: Dict[str, Any],
    track_rows: List[Dict[str, Any]],
    variance_rows: List[Dict[str, Any]],
) -> None:
    """Write a readable QC snapshot next to the CSV and JSON artifacts."""

    df_tracks = pd.DataFrame(track_rows)
    df_variance = pd.DataFrame(variance_rows)

    dropped_tracks = 0
    if not df_tracks.empty and "IncludedInClustering" in df_tracks.columns:
        audio_rows = df_tracks[df_tracks["Category"] == "audio_track"]
        dropped_tracks = int((~audio_rows["IncludedInClustering"]).sum())

    runtime_rows = df_variance[df_variance["Stage"] == "prepared_equalized_runtime"]

    lines = [
        "# Feature QC Summary",
        "",
        f"- Run date: {summary['run_date']}",
        f"- Audio tracks scanned: {summary['audio_tracks_scanned']}",
        f"- Re-extraction attempted: {summary['reextract_attempted']}",
        f"- Re-extraction succeeded: {summary['reextract_succeeded']}",
        f"- Re-extraction failed: {summary['reextract_failed']}",
        f"- Stale feature bundles reported: {summary['stale_feature_bundles']}",
        f"- Stale genre cache entries removed: {summary['stale_genre_cache_entries_removed']}",
        f"- Tracks loaded into clustering dataset: {summary['clustering_loaded_tracks']}",
        f"- Tracks dropped from clustering dataset: {dropped_tracks}",
        "",
        "## Prepared-space variance snapshot",
        "",
    ]

    for _, row in runtime_rows.iterrows():
        lines.append(
            f"- {row['FeatureGroup']}: active dims {int(row['active_dims_observed'])}, "
            f"mean active variance {row['mean_variance_active_dims']:.6f}, "
            f"near-zero dims {int(row['near_zero_dims_observed'])}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate feature bundles, repair broken tracks, and write QC reports."
    )
    parser.add_argument("--audio-dir", default="audio_files")
    parser.add_argument("--results-dir", default="output/features")
    parser.add_argument("--songs-csv", default="data/songs.csv")
    parser.add_argument(
        "--repair",
        dest="repair",
        action="store_true",
        help="Re-extract incomplete or invalid feature bundles before clustering.",
    )
    parser.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="Only report broken bundles without re-extracting them.",
    )
    parser.set_defaults(repair=True)
    args = parser.parse_args()

    start_time = time.time()
    metrics_dir = PROJECT_ROOT / "output" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")

    track_rows, repair_summary, audio_basenames = inspect_and_repair_feature_library(
        audio_dir=args.audio_dir,
        results_dir=args.results_dir,
        repair=args.repair,
    )

    stale_genre_entries = clean_stale_genre_cache(args.results_dir, audio_basenames)
    for base_name in stale_genre_entries:
        track_rows.append(
            {
                "Category": "stale_genre_cache_entry",
                "BaseName": base_name,
                "AudioPath": "",
                "InitialStatus": "stale_genre_cache_entry",
                "InitialMissingKeys": "",
                "InitialInvalidKeys": "",
                "InitialPresentKeys": "",
                "InitialIssueDetails": json.dumps(
                    {"stale_genre_cache_entry": "removed_from_genre_map_cache"},
                    sort_keys=True,
                ),
                "Action": "removed_from_genre_cache",
                "ReextractAttempted": False,
                "RemovedBundleFileCount": 0,
                "RemovedBundleFiles": "[]",
                "ReextractResult": "{}",
                "FinalValidationStatus": "stale_genre_cache_removed",
                "FinalMissingKeys": "",
                "FinalInvalidKeys": "",
                "FinalPresentKeys": "",
                "FinalIssueDetails": json.dumps(
                    {"stale_genre_cache_entry": "removed_from_genre_map_cache"},
                    sort_keys=True,
                ),
            }
        )

    dataset_bundle = clustering_kmeans.load_clustering_dataset_bundle(
        audio_dir=args.audio_dir,
        results_dir=args.results_dir,
        include_genre=fv.include_genre,
        include_msd=fv.include_msd_features,
        songs_csv_path=args.songs_csv,
        selected_audio_feature_keys=fv.clustering_audio_feature_keys,
    )

    loaded_bases = set(dataset_bundle["file_names"])
    loader_qc_status = {
        row["BaseName"]: row["Status"]
        for row in dataset_bundle["qc_rows"]
        if row.get("BaseName")
    }
    for row in track_rows:
        if row["Category"] == "audio_track":
            included = row["BaseName"] in loaded_bases
            row["IncludedInClustering"] = included
            row["ClusteringDatasetStatus"] = (
                "loaded" if included else loader_qc_status.get(row["BaseName"], "dropped")
            )
        else:
            row["IncludedInClustering"] = False
            row["ClusteringDatasetStatus"] = loader_qc_status.get(
                row["BaseName"],
                row["FinalValidationStatus"],
            )

    variance_rows = build_group_variance_rows(
        raw_features=dataset_bundle["raw_features"],
        prepared_features=dataset_bundle["prepared_features"],
        selected_audio_feature_keys=fv.clustering_audio_feature_keys,
        pca_components=fv.pca_components_per_group,
    )

    summary = {
        "run_date": stamp,
        "audio_dir": args.audio_dir,
        "results_dir": args.results_dir,
        "repair_enabled": bool(args.repair),
        "feature_manifest_path": str(
            Path(args.results_dir) / "feature_extraction_manifest.json"
        ),
        "selected_audio_feature_keys": list(fv.clustering_audio_feature_keys),
        "clustering_feature_subset_name": str(fv.clustering_feature_subset_name),
        "feature_equalization_method": str(fv.feature_equalization_method),
        "pca_components_per_group": int(fv.pca_components_per_group),
        "stale_genre_cache_entries_removed": int(len(stale_genre_entries)),
        "clustering_loaded_tracks": int(len(dataset_bundle["file_names"])),
        "clustering_qc_csv_path": dataset_bundle["qc_csv_path"],
        "clustering_qc_json_path": dataset_bundle["qc_json_path"],
        "runtime_seconds": float(time.time() - start_time),
    }
    summary.update(repair_summary)

    track_output_path = metrics_dir / f"feature_qc_{stamp}_track_status.csv"
    summary_output_path = metrics_dir / f"feature_qc_{stamp}_summary.json"
    variance_output_path = metrics_dir / f"feature_qc_{stamp}_group_variance.csv"
    markdown_output_path = metrics_dir / f"feature_qc_{stamp}_summary.md"

    pd.DataFrame(track_rows).to_csv(track_output_path, index=False)
    pd.DataFrame(variance_rows).to_csv(variance_output_path, index=False)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(markdown_output_path, summary, track_rows, variance_rows)

    print(f"Track QC report written to: {track_output_path}")
    print(f"Variance QC report written to: {variance_output_path}")
    print(f"QC summary JSON written to: {summary_output_path}")
    print(f"QC summary markdown written to: {markdown_output_path}")


if __name__ == "__main__":
    main()
