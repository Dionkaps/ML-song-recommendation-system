from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / "data"
DEFAULT_AUDIO_DIR = WORKSPACE_DIR / "audio"
DEFAULT_FEATURES_DIR = WORKSPACE_DIR / "features"
DEFAULT_CLUSTER_RESULTS_DIR = WORKSPACE_DIR / "cluster_results"
DEFAULT_CACHE_DIR = WORKSPACE_DIR / "cache"
DEFAULT_LOGS_DIR = WORKSPACE_DIR / "logs"
DEFAULT_PRETRAINED_DIR = WORKSPACE_DIR / "pretrained_embeddings"
DEFAULT_PRETRAINED_SHARD_DIRS = (
    WORKSPACE_DIR / "pretrained_embeddings_musicnn",
    WORKSPACE_DIR / "pretrained_embeddings_mert",
    WORKSPACE_DIR / "pretrained_embeddings_encodecmae",
)
DEFAULT_PRETRAINED_LOGS_DIR = WORKSPACE_DIR / "pretrained_embeddings_logs"
DEFAULT_MATCHES_CSV = DEFAULT_DATA_DIR / "msd_deezer_matches.csv"
DEFAULT_CATALOG_CSV = DEFAULT_DATA_DIR / "msd_deezer_catalog.csv"
DEFAULT_PENDING_PATTERNS = ("*.pending.csv", "*.pending.json", "*.pending")


@dataclass(frozen=True)
class CleanupGroup:
    key: str
    title: str
    description: str
    paths: tuple[Path, ...]

    @property
    def existing_paths(self) -> tuple[Path, ...]:
        return tuple(path for path in self.paths if path.exists())

    @property
    def item_count(self) -> int:
        return len(self.existing_paths)

    @property
    def total_bytes(self) -> int:
        return sum(path_size(path) for path in self.existing_paths)


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def human_size(total_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(total_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{int(total_bytes)} B"


def unique_existing_paths(paths: list[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return tuple(sorted(unique_paths, key=lambda item: (len(item.parts), str(item).lower())))


def discover_data_outputs() -> tuple[Path, ...]:
    # Excludes DEFAULT_CATALOG_CSV: that is the pristine MSD-only catalog
    # built once from the MillionSongSubset HDF5 (which lives on the laptop
    # only). Deleting it would strand a DGX reset. The working CSV
    # (msd_deezer_matches.csv) is handled by the dedicated "download_state"
    # cleanup group -- deleting it causes the next pipeline run to reseed
    # from the catalog, giving a true fresh download start.
    candidates: list[Path] = []
    if DEFAULT_DATA_DIR.exists():
        for pattern in DEFAULT_PENDING_PATTERNS:
            candidates.extend(DEFAULT_DATA_DIR.glob(pattern))
    return unique_existing_paths(candidates)


def discover_python_cache_outputs() -> tuple[Path, ...]:
    candidates: list[Path] = list(WORKSPACE_DIR.rglob("__pycache__"))
    candidates.extend(WORKSPACE_DIR.rglob("*.pyc"))
    return unique_existing_paths(candidates)


def build_cleanup_groups() -> list[CleanupGroup]:
    groups = [
        CleanupGroup(
            key="audio",
            title="Audio previews",
            description="Downloaded audio preview files in audio/.",
            paths=(DEFAULT_AUDIO_DIR,),
        ),
        CleanupGroup(
            key="features",
            title="Extracted features",
            description="Feature CSVs, raw NPZ feature files, and extraction summaries in features/.",
            paths=(DEFAULT_FEATURES_DIR,),
        ),
        CleanupGroup(
            key="cluster_results",
            title="Cluster results",
            description="All KMeans, GMM, and HDBSCAN outputs in cluster_results/.",
            paths=(DEFAULT_CLUSTER_RESULTS_DIR,),
        ),
        CleanupGroup(
            key="cache",
            title="Deezer cache",
            description="Cached Deezer API responses in cache/.",
            paths=(DEFAULT_CACHE_DIR,),
        ),
        CleanupGroup(
            key="download_state",
            title="Deezer download progress",
            description=(
                "Working CSV (data/msd_deezer_matches.csv) tracking match + download state. "
                "Deleting triggers a fresh download run reseeded from msd_deezer_catalog.csv."
            ),
            paths=(DEFAULT_MATCHES_CSV,),
        ),
        CleanupGroup(
            key="data",
            title="Pending data snapshots",
            description="Temporary *.pending.* snapshots in data/ (catalog CSV is always preserved).",
            paths=discover_data_outputs(),
        ),
        CleanupGroup(
            key="logs",
            title="Run logs",
            description="Per-session resilient download logs and summaries in logs/.",
            paths=(DEFAULT_LOGS_DIR,),
        ),
        CleanupGroup(
            key="pretrained_embeddings",
            title="Pretrained embeddings",
            description="MusicNN/MERT/EnCodecMAE fused + per-model CSVs and per-song NPZs in pretrained_embeddings/ (tens of GB).",
            paths=(DEFAULT_PRETRAINED_DIR,),
        ),
        CleanupGroup(
            key="pretrained_shards",
            title="Pretrained-embedding shard intermediates",
            description="Per-model shard outputs from run_parallel_extraction.sh (musicnn/mert/encodecmae).",
            paths=DEFAULT_PRETRAINED_SHARD_DIRS,
        ),
        CleanupGroup(
            key="pretrained_logs",
            title="Pretrained-embedding run logs",
            description="Timestamped parallel-extraction logs in pretrained_embeddings_logs/.",
            paths=(DEFAULT_PRETRAINED_LOGS_DIR,),
        ),
        CleanupGroup(
            key="pycache",
            title="Python cache files",
            description="All __pycache__ folders and stray .pyc files inside the workspace.",
            paths=discover_python_cache_outputs(),
        ),
    ]
    return [group for group in groups if group.existing_paths]


def _rmtree_nfs_safe(path: Path, max_retries: int = 5, delay_sec: float = 1.0) -> None:
    # On NFS-backed storage (e.g. the DGX /storage/ mounts) unlinking a file
    # that another process still has open leaves a hidden .nfs* sidecar behind,
    # so the final rmdir in shutil.rmtree fails with ENOTEMPTY. Retry a few
    # times, sweeping any .nfs* leftovers between attempts.
    last_error: OSError | None = None
    for _ in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            if not path.exists():
                return
            last_error = exc
            for leftover in path.rglob(".nfs*"):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            time.sleep(delay_sec)

    if path.exists():
        remaining = sorted(path.rglob("*"))
        print(
            f"Warning: could not fully remove {path} ({last_error}). "
            f"{len(remaining)} entr(ies) remain; likely NFS-held .nfs* sidecars "
            "that will clear once the holding process exits."
        )


def delete_path(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False

    if dry_run:
        print(f"Would remove: {path}")
        return True

    if path.is_dir():
        _rmtree_nfs_safe(path)
    else:
        path.unlink()

    print(f"Removed: {path}")
    return True


def prune_empty_parent_dirs(path: Path) -> None:
    current = path.parent
    while current != WORKSPACE_DIR and current.is_dir():
        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent


def choose_deletion_paths(selected_groups: list[CleanupGroup]) -> tuple[Path, ...]:
    selected_paths = [path for group in selected_groups for path in group.existing_paths]
    ordered_paths = sorted(unique_existing_paths(selected_paths), key=lambda item: (len(item.parts), str(item).lower()))

    compact_paths: list[Path] = []
    for path in ordered_paths:
        if any(parent in path.parents for parent in compact_paths if parent.is_dir()):
            continue
        compact_paths.append(path)
    return tuple(compact_paths)


def remove_generated_files(selected_groups: list[CleanupGroup], dry_run: bool) -> int:
    removed_count = 0
    deletion_paths = choose_deletion_paths(selected_groups)

    for path in deletion_paths:
        if delete_path(path, dry_run=dry_run):
            removed_count += 1
            if not dry_run and path.is_file():
                prune_empty_parent_dirs(path)

    if removed_count == 0:
        print("Nothing to remove.")
    else:
        message = "Would remove" if dry_run else "Removed"
        print(f"{message} {removed_count} selected item(s).")

    return removed_count


def print_group_summary(groups: list[CleanupGroup]) -> None:
    print("Generated artifacts currently present:")
    for group in groups:
        print(
            f"- {group.title}: {group.item_count} path(s), {human_size(group.total_bytes)}"
            f" | {group.description}"
        )
        for path in group.existing_paths:
            print(f"    {path}")


def select_groups_cli(groups: list[CleanupGroup], requested_keys: list[str] | None) -> list[CleanupGroup]:
    if not requested_keys:
        return groups

    key_set = set(requested_keys)
    selected = [group for group in groups if group.key in key_set]
    missing = sorted(key_set - {group.key for group in groups})
    if missing:
        raise ValueError(f"Unknown or unavailable cleanup target(s): {', '.join(missing)}")
    return selected


def select_groups_terminal(groups: list[CleanupGroup]) -> list[CleanupGroup] | None:
    print("\nAvailable cleanup targets:\n")
    for index, group in enumerate(groups, 1):
        print(f"  [{index}] {group.title} -- {group.item_count} path(s), {human_size(group.total_bytes)}")
        print(f"      {group.description}")
        for path in group.existing_paths[:3]:
            print(f"        {path}")
        remaining = group.item_count - min(3, group.item_count)
        if remaining > 0:
            print(f"        ... and {remaining} more path(s)")
        print()

    print("Enter numbers to delete (e.g. 1,3), 'all' for everything, or 'q' to cancel.")
    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not choice or choice == "q":
        return None

    if choice == "all":
        return groups

    selected: list[CleanupGroup] = []
    for token in choice.replace(",", " ").split():
        try:
            idx = int(token)
        except ValueError:
            print(f"Ignoring invalid input: {token}")
            continue
        if 1 <= idx <= len(groups):
            if groups[idx - 1] not in selected:
                selected.append(groups[idx - 1])
        else:
            print(f"Ignoring out-of-range number: {idx}")

    return selected if selected else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete generated audio, features, cluster results, logs, caches, and other workspace artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting anything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the generated artifact groups that currently exist and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete every discovered generated artifact group without prompting.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=(
            "audio",
            "features",
            "cluster_results",
            "cache",
            "download_state",
            "data",
            "logs",
            "pretrained_embeddings",
            "pretrained_shards",
            "pretrained_logs",
            "pycache",
        ),
        help="Delete only the named cleanup groups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = build_cleanup_groups()

    if not groups:
        print("Nothing to remove.")
        return 0

    if args.list:
        print_group_summary(groups)
        return 0

    if args.all:
        selected_groups = groups
    elif args.targets:
        selected_groups = select_groups_cli(groups, args.targets)
    else:
        selected_groups = select_groups_terminal(groups)
        if selected_groups is None:
            print("Reset cancelled.")
            return 1

    if not selected_groups:
        print("Nothing selected.")
        return 0

    print_group_summary(selected_groups)
    remove_generated_files(selected_groups, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
