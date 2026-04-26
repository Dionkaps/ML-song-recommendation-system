"""
Walk dimred_benchmark/cluster_results/<feature>/<reduction>/<algorithm>/run_metadata.json
and produce a side-by-side comparison table + a human-readable markdown report.

Lower-is-better metrics (Davies-Bouldin) are inverted in the rank computation
so that "higher rank = better" holds uniformly across columns. The
`overall_rank` column is the mean of the per-metric ranks for that row's
algorithm; we rank within each algorithm separately because cross-algorithm
silhouette comparison is not meaningful (different algorithms make different
shape assumptions about clusters). The comparison we actually care about is:
"for a fixed algorithm and feature type, which dim-reduction wins?".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = THIS_DIR / "cluster_results"
CSV_OUT = THIS_DIR / "comparison_report.csv"
MD_OUT = THIS_DIR / "comparison_report.md"


def _to_markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as GFM-flavoured markdown without the tabulate dep."""
    cols = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else format(v, floatfmt))
            else:
                cells.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


# (column_name, higher_is_better). Columns absent from a row are tolerated.
METRIC_COLS: list[tuple[str, bool]] = [
    ("silhouette_score", True),
    ("calinski_harabasz_score", True),
    ("davies_bouldin_score", False),
    ("dunn_index", True),
    ("trustworthiness", True),
    ("stability_mean_ari", True),
]


def collect_runs() -> pd.DataFrame:
    rows: list[dict] = []
    if not RESULTS_ROOT.exists():
        raise SystemExit(f"No cluster_results/ directory found at {RESULTS_ROOT}")

    for meta in RESULTS_ROOT.glob("*/*/*/run_metadata.json"):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  WARN: skip {meta}: {exc}")
            continue

        # Path layout: cluster_results/<feature_source>/<reduction>/<algorithm>/run_metadata.json
        algorithm = meta.parent.name
        reduction = meta.parent.parent.name
        feature_source = meta.parent.parent.parent.name

        row = {
            "feature_source": feature_source,
            "reduction": reduction,
            "algorithm": algorithm,
            "cluster_count": payload.get(
                "cluster_count",
                payload.get("effective_cluster_count"),
            ),
            "samples": payload.get("samples"),
            "pca_components": payload.get("pca_components"),
            "umap_components": payload.get("umap_components"),
            "clustering_dimensions": payload.get("clustering_dimensions"),
        }
        # Pull the metric columns; HDBSCAN-only columns (silhouette_score
        # under "silhouette_score_clustered_only") get aliased into the
        # canonical name so the column lines up with KMeans/GMM rows.
        sil = payload.get("silhouette_score")
        if sil is None:
            sil = payload.get("silhouette_score_clustered_only")
        row["silhouette_score"] = sil

        for col, _ in METRIC_COLS:
            if col == "silhouette_score":
                continue
            row[col] = payload.get(col)

        rows.append(row)

    if not rows:
        raise SystemExit(f"No run_metadata.json files found under {RESULTS_ROOT}")

    return pd.DataFrame(rows)


def add_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """For each (feature_source, algorithm) group, rank the 3 reductions per metric."""
    frame = frame.copy()

    rank_cols: list[str] = []
    for col, higher_is_better in METRIC_COLS:
        if col not in frame.columns:
            continue
        rank_col = f"rank_{col}"
        # rank within each (feature_source, algorithm) so we compare apples
        # to apples -- only the reduction varies.
        # ascending=False means "biggest value gets rank 1"; we then invert
        # for lower-is-better metrics so rank 1 always = best.
        ranked = frame.groupby(["feature_source", "algorithm"])[col].rank(
            ascending=not higher_is_better, method="min", na_option="bottom",
        )
        frame[rank_col] = ranked
        rank_cols.append(rank_col)

    if rank_cols:
        # Average rank across metrics. Lower mean rank = better.
        frame["overall_rank"] = frame[rank_cols].mean(axis=1)
    return frame


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Dim-Reduction Benchmark — Comparison Report\n")
    lines.append(
        "Each row is one (feature source, reduction, algorithm) clustering run.\n"
        "Within every (feature source, algorithm) group the three reductions "
        "are ranked per metric; `overall_rank` is the mean of those per-metric "
        "ranks (lower is better).\n"
    )

    # Per-(feature, algo) winner section
    lines.append("## Winners per algorithm (lower overall_rank = better)\n")
    winners_rows: list[dict] = []
    for (src, algo), group in frame.groupby(["feature_source", "algorithm"]):
        if "overall_rank" not in group.columns:
            continue
        winner = group.loc[group["overall_rank"].idxmin()]
        winners_rows.append({
            "feature_source": src,
            "algorithm": algo,
            "best_reduction": winner["reduction"],
            "overall_rank": round(float(winner["overall_rank"]), 3),
            "silhouette": winner.get("silhouette_score"),
            "calinski_harabasz": winner.get("calinski_harabasz_score"),
            "davies_bouldin": winner.get("davies_bouldin_score"),
            "dunn": winner.get("dunn_index"),
            "trustworthiness": winner.get("trustworthiness"),
            "stability_ari": winner.get("stability_mean_ari"),
            "k": winner.get("cluster_count"),
        })
    winners = pd.DataFrame(winners_rows)
    if not winners.empty:
        lines.append(_to_markdown_table(winners, floatfmt=".4f"))
        lines.append("")

    # Full table
    lines.append("\n## Full results\n")
    sort_cols = ["feature_source", "algorithm", "reduction"]
    display = frame.sort_values(sort_cols)
    keep = [
        "feature_source", "reduction", "algorithm", "cluster_count",
        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score",
        "dunn_index", "trustworthiness", "stability_mean_ari",
        "overall_rank",
    ]
    keep = [c for c in keep if c in display.columns]
    lines.append(_to_markdown_table(display[keep], floatfmt=".4f"))
    lines.append("")

    # Disagreement diagnostic: highlight cases where trustworthiness ranking
    # disagrees with internal-CVI ranking. That's the "good silhouette but
    # the reduction lost the original neighbourhood structure" failure mode.
    lines.append("\n## Diagnostic: trustworthiness vs internal-CVI agreement\n")
    diag_rows: list[dict] = []
    for (src, algo), group in frame.groupby(["feature_source", "algorithm"]):
        if "rank_trustworthiness" not in group.columns:
            continue
        if "rank_silhouette_score" not in group.columns:
            continue
        for _, row in group.iterrows():
            diag_rows.append({
                "feature_source": src,
                "algorithm": algo,
                "reduction": row["reduction"],
                "rank_silhouette": row.get("rank_silhouette_score"),
                "rank_trustworthiness": row.get("rank_trustworthiness"),
                "agreement": (
                    "agree" if row.get("rank_silhouette_score")
                    == row.get("rank_trustworthiness") else "disagree"
                ),
            })
    if diag_rows:
        diag = pd.DataFrame(diag_rows)
        lines.append(_to_markdown_table(diag, floatfmt=".0f"))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Scanning {RESULTS_ROOT}...")
    frame = collect_runs()
    print(f"Found {len(frame)} clustering runs across "
          f"{frame['feature_source'].nunique()} feature source(s) x "
          f"{frame['reduction'].nunique()} reduction(s) x "
          f"{frame['algorithm'].nunique()} algorithm(s).")

    frame = add_ranks(frame)
    frame.sort_values(["feature_source", "algorithm", "overall_rank"]).to_csv(
        CSV_OUT, index=False,
    )
    write_markdown(frame, MD_OUT)
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
