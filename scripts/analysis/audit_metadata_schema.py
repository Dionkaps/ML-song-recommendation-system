import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.utils.song_metadata import load_legacy_msd_catalog, load_unified_songs


def _gini(values: Iterable[int]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.allclose(array.sum(), 0.0):
        return float("nan")
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def _distribution_metrics(counts: pd.Series) -> Dict[str, float]:
    values = counts.to_numpy(dtype=np.float64)
    total = values.sum()
    if total <= 0 or values.size == 0:
        return {
            "gini": float("nan"),
            "hhi": float("nan"),
            "effective_genre_count": float("nan"),
            "top_10_share": float("nan"),
            "singleton_genre_fraction": float("nan"),
            "genres_with_5_or_fewer_fraction": float("nan"),
        }
    shares = values / total
    hhi = float(np.sum(shares ** 2))
    return {
        "gini": _gini(values),
        "hhi": hhi,
        "effective_genre_count": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "top_10_share": float(values[:10].sum() / total),
        "singleton_genre_fraction": float(np.mean(values == 1)),
        "genres_with_5_or_fewer_fraction": float(np.mean(values <= 5)),
    }


def _write_markdown_summary(
    output_path: Path,
    summary: Dict[str, object],
    primary_distribution: pd.DataFrame,
    label_distribution: pd.DataFrame,
) -> None:
    top_primary = primary_distribution.head(15)
    top_labels = label_distribution.head(15)

    lines: List[str] = [
        "# Metadata Schema Audit",
        "",
        f"- Run date: {summary['run_date']}",
        f"- Unified rows: {summary['unified_rows']}",
        f"- Audio-backed rows: {summary['audio_backed_rows']}",
        f"- Audio-backed rows with MSD track id: {summary['audio_rows_with_msd_track_id']}",
        f"- Audio-backed rows with numeric MSD features: {summary['audio_rows_with_numeric_msd_features']}",
        f"- Current audio library size: {summary['current_audio_files']}",
        f"- Audio coverage gap: {summary['audio_library_minus_unified_audio_rows']}",
        f"- Primary genres in audio-backed subset: {summary['audio_primary_genre_unique_count']}",
        f"- MSD metadata policy: {summary['msd_metadata_policy']}",
        "",
        "## Audio-backed primary genre imbalance",
        "",
        f"- Gini: {summary['audio_primary_genre_gini']:.6f}",
        f"- HHI: {summary['audio_primary_genre_hhi']:.6f}",
        f"- Effective genre count: {summary['audio_primary_genre_effective_count']:.2f}",
        f"- Top-10 primary genre share: {summary['audio_primary_genre_top_10_share']:.6f}",
        f"- Genres with 5 or fewer tracks: {summary['audio_primary_genre_tail_fraction_le_5']:.6f}",
        "",
        "## Top primary genres",
        "",
    ]

    for _, row in top_primary.iterrows():
        lines.append(
            f"- {row['PrimaryGenre']}: {int(row['TrackCount'])} tracks "
            f"({float(row['TrackFraction']):.6f} share)"
        )

    lines.extend(["", "## Top exploded genre labels", ""])
    for _, row in top_labels.iterrows():
        lines.append(
            f"- {row['GenreLabel']}: {int(row['TrackCount'])} tracks "
            f"({float(row['TrackFraction']):.6f} share)"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit unified songs metadata, genre imbalance, and MSD coverage."
    )
    parser.add_argument("--songs-csv", default="data/songs.csv")
    parser.add_argument("--msd-catalog", default="data/millionsong_dataset.csv")
    parser.add_argument("--audio-dir", default="audio_files")
    parser.add_argument("--output-dir", default="output/metrics")
    args = parser.parse_args()

    start_time = time.time()
    stamp = time.strftime("%Y%m%d")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unified = load_unified_songs(str(PROJECT_ROOT / args.songs_csv))
    msd_catalog = load_legacy_msd_catalog(str(PROJECT_ROOT / args.msd_catalog))
    audio_files = [
        path
        for path in (PROJECT_ROOT / args.audio_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}
    ]

    audio_rows = unified[unified["has_audio"]].copy()
    audio_rows["PrimaryGenre"] = audio_rows["primary_genre"].replace("", "unknown")

    primary_counts = (
        audio_rows["PrimaryGenre"].value_counts().rename_axis("PrimaryGenre").reset_index(name="TrackCount")
    )
    total_audio_rows = int(max(1, audio_rows.shape[0]))
    primary_counts["TrackFraction"] = primary_counts["TrackCount"] / total_audio_rows

    exploded_labels = []
    for genre_text in audio_rows["genre"].astype(str):
        tokens = [token.strip() for token in genre_text.split(",") if token.strip()]
        if not tokens:
            exploded_labels.append("unknown")
        else:
            exploded_labels.extend(tokens)
    label_counts_series = pd.Series(exploded_labels, dtype=object).value_counts()
    label_counts = label_counts_series.rename_axis("GenreLabel").reset_index(name="TrackCount")
    label_counts["TrackFraction"] = label_counts["TrackCount"] / max(1, len(exploded_labels))

    primary_metrics = _distribution_metrics(primary_counts["TrackCount"])

    conflicting_catalog_track_ids = 0
    nonempty_catalog = msd_catalog[msd_catalog["msd_track_id"].ne("")]
    for _, group in nonempty_catalog.groupby("msd_track_id"):
        if len(group[["msd_artist", "msd_title", "genre"]].drop_duplicates()) > 1:
            conflicting_catalog_track_ids += 1

    summary = {
        "run_date": stamp,
        "songs_csv_path": args.songs_csv,
        "msd_catalog_path": args.msd_catalog,
        "unified_rows": int(len(unified)),
        "audio_backed_rows": int(audio_rows.shape[0]),
        "current_audio_files": int(len(audio_files)),
        "audio_library_minus_unified_audio_rows": int(len(audio_files) - audio_rows.shape[0]),
        "unified_rows_with_msd_track_id": int(unified["msd_track_id"].astype(str).str.strip().ne("").sum()),
        "audio_rows_with_msd_track_id": int(
            (audio_rows["msd_track_id"].astype(str).str.strip().ne("")).sum()
        ),
        "audio_rows_with_numeric_msd_features": int(
            (
                audio_rows["key"].notna()
                & audio_rows["mode"].notna()
                & audio_rows["loudness"].notna()
                & audio_rows["tempo"].notna()
            ).sum()
        ),
        "audio_rows_without_numeric_msd_features": int(
            (
                ~(
                    audio_rows["key"].notna()
                    & audio_rows["mode"].notna()
                    & audio_rows["loudness"].notna()
                    & audio_rows["tempo"].notna()
                )
            ).sum()
        ),
        "audio_primary_genre_unique_count": int(primary_counts.shape[0]),
        "audio_exploded_genre_label_unique_count": int(label_counts.shape[0]),
        "audio_primary_genre_gini": primary_metrics["gini"],
        "audio_primary_genre_hhi": primary_metrics["hhi"],
        "audio_primary_genre_effective_count": primary_metrics["effective_genre_count"],
        "audio_primary_genre_top_10_share": primary_metrics["top_10_share"],
        "audio_primary_genre_singleton_fraction": primary_metrics["singleton_genre_fraction"],
        "audio_primary_genre_tail_fraction_le_5": primary_metrics["genres_with_5_or_fewer_fraction"],
        "audio_rows_with_unknown_primary_genre": int((audio_rows["PrimaryGenre"] == "unknown").sum()),
        "audio_rows_with_multilabel_genre": int((audio_rows["genre_count"] > 1).sum()),
        "audio_rows_mean_genre_count": float(audio_rows["genre_count"].mean()),
        "msd_catalog_rows": int(len(msd_catalog)),
        "msd_catalog_missing_track_id_rows": int((msd_catalog["msd_track_id"] == "").sum()),
        "msd_catalog_duplicate_track_id_rows": int(msd_catalog["msd_track_id"].duplicated().sum()),
        "msd_catalog_conflicting_track_id_groups": int(conflicting_catalog_track_ids),
        "msd_metadata_policy": fv.msd_metadata_policy,
        "msd_metadata_restore_policy": fv.msd_metadata_restore_policy,
        "runtime_seconds": float(time.time() - start_time),
    }

    summary_path = output_dir / f"metadata_schema_audit_{stamp}_summary.json"
    primary_path = output_dir / f"metadata_primary_genre_distribution_{stamp}.csv"
    label_path = output_dir / f"metadata_label_genre_distribution_{stamp}.csv"
    markdown_path = output_dir / f"metadata_schema_audit_{stamp}.md"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    primary_counts.to_csv(primary_path, index=False)
    label_counts.to_csv(label_path, index=False)
    _write_markdown_summary(markdown_path, summary, primary_counts, label_counts)

    print(f"Metadata audit summary written to: {summary_path}")
    print(f"Primary genre distribution written to: {primary_path}")
    print(f"Exploded genre label distribution written to: {label_path}")
    print(f"Metadata audit markdown written to: {markdown_path}")


if __name__ == "__main__":
    main()
