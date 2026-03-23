from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import sleep
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove generated pipeline artifacts so the full download-to-benchmark "
            "process can be rerun from a clean state."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the listed targets. Without this flag the script only previews.",
    )
    parser.add_argument(
        "--keep-download-state",
        action="store_true",
        help=(
            "Preserve downloaded audio, checkpoint/cache files, and download-derived "
            "metadata so you can restart from the post-download stages."
        ),
    )
    parser.add_argument(
        "--keep-run-logs",
        action="store_true",
        help="Preserve docs/reports/run_logs instead of deleting them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final summary as JSON.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for writing the final summary JSON.",
    )
    return parser.parse_args()


def unique_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    result: List[Path] = []
    for path in paths:
        normalized = str(path.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(path)
    return result


def build_target_groups(
    project_root: Path,
    *,
    keep_download_state: bool,
    keep_run_logs: bool,
) -> List[Tuple[str, List[Path]]]:
    data_root = project_root / "data"
    backup_root = data_root / "backup_old_csvs"

    groups: List[Tuple[str, List[Path]]] = [
        (
            "pipeline_outputs",
            [
                project_root / "output",
                project_root / "metrics",
                project_root / "src" / "data_collection" / "download_stats",
            ],
        ),
        (
            "core_derived_metadata",
            [
                data_root / "songs.csv",
                data_root / "songs_schema_summary.json",
                data_root / "songs_with_merged_genres.csv",
                data_root / "songs_genre_list.csv",
                data_root / "unique_genres.csv",
                data_root / "_tmp_rebuilt_songs.csv",
            ],
        ),
    ]

    if not keep_download_state:
        groups.append(
            (
                "download_derived_metadata",
                [
                    data_root / "songs_data_with_genre.csv",
                    data_root / "msd_matches.csv",
                    data_root / "audio_msd_mapping.csv",
                    data_root / "msd_unmatched.csv",
                    backup_root / "songs_data_with_genre.csv",
                    backup_root / "audio_msd_mapping.csv",
                    backup_root / "msd_matches.csv",
                    backup_root / "msd_unmatched.csv",
                ],
            )
        )

    if not keep_download_state:
        groups.append(
            (
                "download_state",
                [
                    project_root / "audio_files",
                    project_root / "download_checkpoint_with_genre.json",
                    project_root / "deezer_search_cache.json",
                ],
            )
        )

    if not keep_run_logs:
        groups.append(
            (
                "run_logs",
                [
                    project_root / "docs" / "reports" / "run_logs",
                ],
            )
        )

    return [(category, unique_paths(paths)) for category, paths in groups]


def build_preserved_reference_inputs(project_root: Path) -> List[Path]:
    return [
        project_root / "data" / "millionsong_dataset.csv",
        project_root / "data" / "acoustically_coherent_merged_genres_corrected.csv",
        project_root / "data" / "backup_old_csvs" / "msd_features.csv",
        project_root / "data" / "backup_old_csvs" / "msd_feature_vectors.npz",
        project_root / "README.md",
        project_root / "requirements.txt",
    ]


def remove_path(path: Path) -> Dict[str, object]:
    existed = path.exists()
    item_type = "missing"
    if existed:
        item_type = "directory" if path.is_dir() else "file"
        for attempt in range(1, 6):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt >= 5:
                    raise
                sleep(float(attempt))
    return {
        "path": str(path),
        "existed": bool(existed),
        "type": item_type,
        "removed": bool(existed and not path.exists()),
    }


def preview_target(category: str, path: Path) -> Dict[str, object]:
    return {
        "category": category,
        "path": str(path),
        "exists": bool(path.exists()),
        "type": "directory" if path.is_dir() else ("file" if path.exists() else "missing"),
    }


def iter_target_entries(
    groups: Sequence[Tuple[str, Sequence[Path]]],
) -> Iterable[Tuple[str, Path]]:
    for category, paths in groups:
        for path in paths:
            yield category, path


def build_payload(
    *,
    project_root: Path,
    apply: bool,
    keep_download_state: bool,
    keep_run_logs: bool,
    entries: List[Dict[str, object]],
) -> Dict[str, object]:
    existing_entries = [entry for entry in entries if bool(entry.get("exists") or entry.get("existed"))]
    removed_entries = [entry for entry in entries if bool(entry.get("removed"))]
    category_counts: Dict[str, int] = {}
    for entry in entries:
        category = str(entry["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "project_root": str(project_root),
        "apply": bool(apply),
        "keep_download_state": bool(keep_download_state),
        "keep_run_logs": bool(keep_run_logs),
        "target_count": int(len(entries)),
        "existing_target_count": int(len(existing_entries)),
        "removed_count": int(len(removed_entries)),
        "category_counts": dict(sorted(category_counts.items())),
        "targets": entries,
        "preserved_reference_inputs": [
            str(path)
            for path in build_preserved_reference_inputs(project_root)
            if path.exists()
        ],
    }


def print_human_summary(payload: Dict[str, object]) -> None:
    mode = "APPLY" if payload["apply"] else "DRY RUN"
    print(f"Full Pipeline Reset [{mode}]")
    print(f"Project root: {payload['project_root']}")
    print(f"Targets listed: {payload['target_count']}")
    print(f"Targets currently present: {payload['existing_target_count']}")
    if payload["apply"]:
        print(f"Targets removed: {payload['removed_count']}")
    else:
        print("No files were deleted. Re-run with --apply to remove these targets.")

    print("\nCategories:")
    for category, count in payload["category_counts"].items():
        print(f"- {category}: {count}")

    print("\nTargets:")
    state_key = "removed" if payload["apply"] else "exists"
    for entry in payload["targets"]:
        state = "present" if entry.get(state_key) else "missing"
        print(f"- [{entry['category']}] {state}: {entry['path']}")

    preserved = payload.get("preserved_reference_inputs", [])
    if preserved:
        print("\nPreserved reference inputs:")
        for path_text in preserved:
            print(f"- {path_text}")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    groups = build_target_groups(
        project_root,
        keep_download_state=bool(args.keep_download_state),
        keep_run_logs=bool(args.keep_run_logs),
    )

    entries: List[Dict[str, object]] = []
    for category, path in iter_target_entries(groups):
        if args.apply:
            result = remove_path(path)
            result["category"] = category
            entries.append(result)
        else:
            entries.append(preview_target(category, path))

    payload = build_payload(
        project_root=project_root,
        apply=bool(args.apply),
        keep_download_state=bool(args.keep_download_state),
        keep_run_logs=bool(args.keep_run_logs),
        entries=entries,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_human_summary(payload)


if __name__ == "__main__":
    main()
