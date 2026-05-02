from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from safety import assert_inside_workspace


THIS_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = THIS_DIR / "cluster_results"
CSV_OUT = THIS_DIR / "k_selection_summary.csv"
MD_OUT = THIS_DIR / "k_selection_summary.md"


def _to_markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else format(value, floatfmt))
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect_runs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not RESULTS_ROOT.exists():
        raise SystemExit(f"No cluster_results/ directory found at {RESULTS_ROOT}")

    for meta in RESULTS_ROOT.glob("*/*/kmeans_gap_stability/run_metadata.json"):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: skipping {meta}: {exc}")
            continue

        feature_source = meta.parent.parent.parent.name
        reduction = meta.parent.parent.name
        selected_row = payload.get("selected_row") or {}
        rows.append({
            "feature_source": feature_source,
            "reduction": reduction,
            "selected_k": payload.get("cluster_count"),
            "gap_selected_k": payload.get("gap_selected_k"),
            "prediction_strength_selected_k": payload.get("prediction_strength_selected_k"),
            "support_status": payload.get("support_status"),
            "requires_review": payload.get("requires_interpretation_review"),
            "selection_samples": payload.get("selection_samples"),
            "samples": payload.get("samples"),
            "gap": selected_row.get("gap"),
            "gap_sk": selected_row.get("gap_sk"),
            "prediction_strength": selected_row.get("prediction_strength_mean"),
            "silhouette": payload.get("silhouette_score"),
            "calinski_harabasz": payload.get("calinski_harabasz_score"),
            "davies_bouldin": payload.get("davies_bouldin_score"),
            "dunn": payload.get("dunn_index"),
            "trustworthiness": payload.get("trustworthiness"),
            "stability_ari": payload.get("stability_mean_ari"),
            "best_silhouette_k": payload.get("best_silhouette_k"),
            "best_calinski_harabasz_k": payload.get("best_calinski_harabasz_k"),
            "best_davies_bouldin_k": payload.get("best_davies_bouldin_k"),
        })

    if not rows:
        raise SystemExit(f"No run_metadata.json files found under {RESULTS_ROOT}")
    return pd.DataFrame(rows)


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# KMeans K-Selection Benchmark Summary\n")
    lines.append(
        "Each row is one feature source and dimensionality-reduction recipe. "
        "The selected K comes from the gap statistic one-standard-error rule; "
        "prediction strength and bootstrap ARI report stability support.\n"
    )

    reviewed = frame[frame["requires_review"] == True]  # noqa: E712
    if not reviewed.empty:
        lines.append("## Runs Needing Interpretation Review\n")
        lines.append(_to_markdown_table(
            reviewed[[
                "feature_source",
                "reduction",
                "selected_k",
                "prediction_strength_selected_k",
                "support_status",
                "prediction_strength",
                "stability_ari",
            ]],
        ))
        lines.append("")

    lines.append("## Full Results\n")
    display_cols = [
        "feature_source",
        "reduction",
        "selected_k",
        "prediction_strength_selected_k",
        "support_status",
        "gap",
        "prediction_strength",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "dunn",
        "trustworthiness",
        "stability_ari",
    ]
    lines.append(_to_markdown_table(frame.sort_values(["feature_source", "reduction"])[display_cols]))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame = collect_runs()
    csv_out = assert_inside_workspace(CSV_OUT, "summary_csv")
    md_out = assert_inside_workspace(MD_OUT, "summary_markdown")
    frame.sort_values(["feature_source", "reduction"]).to_csv(csv_out, index=False)
    write_markdown(frame, md_out)
    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
