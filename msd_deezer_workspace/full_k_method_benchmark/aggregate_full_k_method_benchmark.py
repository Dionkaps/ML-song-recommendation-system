from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from safety import WORKSPACE_DIR, assert_inside_workspace


DEFAULT_FEATURE_SOURCE = "pretrained_embeddings"
DEFAULT_REDUCTION = "pca_then_umap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated-sample full-dataset K-method benchmark results.",
    )
    parser.add_argument("--samples-root", default=str(WORKSPACE_DIR / "samples"))
    parser.add_argument("--feature-source", default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--reduction-mode", default=DEFAULT_REDUCTION)
    parser.add_argument("--stability-threshold", type=float, default=0.75)
    parser.add_argument("--min-cluster-size-threshold", type=int, default=50)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARN: cannot read {path}: {exc}")
        return None


def collect_selection_rows(samples_root: Path, feature_source: str, reduction: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(samples_root.glob("sample_*")):
        meta = (
            sample_dir
            / "cluster_results"
            / feature_source
            / reduction
            / "kmeans_gap_stability"
            / "run_metadata.json"
        )
        if not meta.exists():
            print(f"WARN: missing selection metadata for {sample_dir.name}: {meta}")
            continue
        payload = _read_json(meta)
        if payload is None:
            continue
        selected = payload.get("selected_row") or {}
        rows.append({
            "sample_id": sample_dir.name,
            "samples": payload.get("samples"),
            "inlier_samples": payload.get("inlier_samples"),
            "gap_selected_k": payload.get("gap_selected_k"),
            "prediction_strength_selected_k": payload.get("prediction_strength_selected_k"),
            "support_status": payload.get("support_status"),
            "requires_interpretation_review": payload.get("requires_interpretation_review"),
            "selected_gap": selected.get("gap"),
            "selected_prediction_strength": selected.get("prediction_strength_mean"),
            "final_silhouette": payload.get("silhouette_score"),
            "final_davies_bouldin": payload.get("davies_bouldin_score"),
            "final_calinski_harabasz": payload.get("calinski_harabasz_score"),
            "final_stability_ari": payload.get("stability_mean_ari"),
            "best_silhouette_k": payload.get("best_silhouette_k"),
            "best_calinski_harabasz_k": payload.get("best_calinski_harabasz_k"),
            "best_davies_bouldin_k": payload.get("best_davies_bouldin_k"),
        })
    return pd.DataFrame(rows)


def collect_candidate_rows(samples_root: Path, feature_source: str, reduction: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sample_dir in sorted(samples_root.glob("sample_*")):
        csv_path = (
            sample_dir
            / "cluster_results"
            / feature_source
            / reduction
            / "candidate_k_validation"
            / "candidate_validation_metrics.csv"
        )
        if not csv_path.exists():
            print(f"WARN: missing candidate validation for {sample_dir.name}: {csv_path}")
            continue
        frame = pd.read_csv(csv_path)
        frame.insert(0, "sample_id", sample_dir.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    numeric_cols = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "stability_mean_ari",
        "min_cluster_size",
        "p10_cluster_size",
        "median_cluster_size",
        "max_cluster_size",
        "clusters_under_50",
        "clusters_under_100",
    ]
    grouped = ok.groupby("candidate_k")
    summary = grouped[numeric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    summary["sample_count"] = grouped.size().values
    return summary


def _median_int(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return int(round(float(values.median())))


def choose_recommended_candidate(
    summary: pd.DataFrame,
    stability_threshold: float,
    min_cluster_size_threshold: int,
) -> dict[str, Any]:
    if summary.empty:
        return {
            "recommended_candidate_k": None,
            "recommendation_rule": "no_candidate_validation_results",
        }

    eligible = summary[
        (summary["stability_mean_ari_mean"] >= stability_threshold)
        & (summary["min_cluster_size_min"] >= min_cluster_size_threshold)
    ].copy()
    if not eligible.empty:
        chosen = eligible.sort_values(
            ["candidate_k"],
            ascending=[False],
        ).iloc[0]
        return {
            "recommended_candidate_k": int(chosen["candidate_k"]),
            "recommendation_rule": (
                f"largest_candidate_with_mean_ari_ge_{stability_threshold:.2f}_"
                f"and_min_cluster_size_ge_{min_cluster_size_threshold}"
            ),
            "recommended_candidate_stability_ari_mean": float(chosen["stability_mean_ari_mean"]),
            "recommended_candidate_silhouette_mean": float(chosen["silhouette_score_mean"]),
            "recommended_candidate_davies_bouldin_mean": float(chosen["davies_bouldin_score_mean"]),
        }

    ranked = summary.copy()
    ranked["rank_stability"] = ranked["stability_mean_ari_mean"].rank(ascending=False, method="min")
    ranked["rank_silhouette"] = ranked["silhouette_score_mean"].rank(ascending=False, method="min")
    ranked["rank_db"] = ranked["davies_bouldin_score_mean"].rank(ascending=True, method="min")
    ranked["rank_size"] = ranked["min_cluster_size_min"].rank(ascending=False, method="min")
    ranked["overall_rank"] = ranked[["rank_stability", "rank_silhouette", "rank_db", "rank_size"]].mean(axis=1)
    chosen = ranked.sort_values(["overall_rank", "candidate_k"], ascending=[True, False]).iloc[0]
    return {
        "recommended_candidate_k": int(chosen["candidate_k"]),
        "recommendation_rule": "no_candidate_passed_thresholds_using_best_average_rank",
        "recommended_candidate_stability_ari_mean": float(chosen["stability_mean_ari_mean"]),
        "recommended_candidate_silhouette_mean": float(chosen["silhouette_score_mean"]),
        "recommended_candidate_davies_bouldin_mean": float(chosen["davies_bouldin_score_mean"]),
    }


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                cells.append("" if pd.isna(value) else format(float(value), floatfmt))
            else:
                cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    selection: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    recommendation: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path = assert_inside_workspace(
        WORKSPACE_DIR / "full_dataset_k_method_recommendation.md",
        "recommendation_report",
    )
    broad_k = _median_int(selection.get("prediction_strength_selected_k", pd.Series(dtype=float)))
    gap_k = _median_int(selection.get("gap_selected_k", pd.Series(dtype=float)))

    lines: list[str] = []
    lines.append("# Full-Dataset K-Method Benchmark Recommendation\n")
    lines.append("## Decision\n")
    lines.append(f"- Representation tested: `{args.feature_source} + {args.reduction_mode}`")
    lines.append(f"- Median prediction-strength K across samples: `{broad_k}`")
    lines.append(f"- Median gap-statistic K across samples: `{gap_k}`")
    lines.append(f"- Recommended candidate K for full-data validation: `{recommendation['recommended_candidate_k']}`")
    lines.append(f"- Rule: `{recommendation['recommendation_rule']}`")
    lines.append("")
    lines.append(
        "Interpretation: prediction strength estimates the broad stable structure, "
        "while gap statistic estimates fine-grained structure. The final candidate "
        "K should be the largest tested value that remains stable and avoids tiny "
        "clusters."
    )
    lines.append("")

    lines.append("## Repeated Sample K-Selection\n")
    keep_selection = [
        "sample_id",
        "gap_selected_k",
        "prediction_strength_selected_k",
        "support_status",
        "final_stability_ari",
        "final_silhouette",
        "final_davies_bouldin",
    ]
    keep_selection = [c for c in keep_selection if c in selection.columns]
    lines.append(_markdown_table(selection[keep_selection] if keep_selection else selection))
    lines.append("")

    lines.append("## Candidate K Validation Summary\n")
    keep_candidates = [
        "candidate_k",
        "sample_count",
        "stability_mean_ari_mean",
        "stability_mean_ari_std",
        "silhouette_score_mean",
        "davies_bouldin_score_mean",
        "min_cluster_size_min",
        "p10_cluster_size_mean",
        "clusters_under_50_mean",
    ]
    keep_candidates = [c for c in keep_candidates if c in candidate_summary.columns]
    display = candidate_summary[keep_candidates].sort_values("candidate_k") if keep_candidates else candidate_summary
    lines.append(_markdown_table(display))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    samples_root = Path(args.samples_root).resolve()
    try:
        samples_root.relative_to(WORKSPACE_DIR.resolve())
    except ValueError as exc:
        raise SystemExit(f"samples-root must stay inside {WORKSPACE_DIR}: {samples_root}") from exc

    selection = collect_selection_rows(samples_root, args.feature_source, args.reduction_mode)
    candidates = collect_candidate_rows(samples_root, args.feature_source, args.reduction_mode)
    candidate_summary = summarize_candidate_rows(candidates)
    recommendation = choose_recommended_candidate(
        candidate_summary,
        args.stability_threshold,
        args.min_cluster_size_threshold,
    )

    selection_path = assert_inside_workspace(
        WORKSPACE_DIR / "method_selection_repeated_samples.csv",
        "selection_summary",
    )
    all_candidates_path = assert_inside_workspace(
        WORKSPACE_DIR / "candidate_k_validation_all_samples.csv",
        "candidate_validation_all",
    )
    candidate_summary_path = assert_inside_workspace(
        WORKSPACE_DIR / "candidate_k_validation_summary.csv",
        "candidate_validation_summary",
    )

    selection.to_csv(selection_path, index=False)
    candidates.to_csv(all_candidates_path, index=False)
    candidate_summary.to_csv(candidate_summary_path, index=False)
    write_report(selection, candidate_summary, recommendation, args)

    print(f"Wrote {selection_path}")
    print(f"Wrote {all_candidates_path}")
    print(f"Wrote {candidate_summary_path}")
    print(WORKSPACE_DIR / "full_dataset_k_method_recommendation.md")


if __name__ == "__main__":
    main()
