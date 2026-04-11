from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:  # pragma: no cover
    tk = None
    messagebox = None


WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = WORKSPACE_DIR / "data"
DEFAULT_AUDIO_DIR = WORKSPACE_DIR / "audio"
DEFAULT_FEATURES_DIR = WORKSPACE_DIR / "features"
DEFAULT_CLUSTER_RESULTS_DIR = WORKSPACE_DIR / "cluster_results"
DEFAULT_CACHE_DIR = WORKSPACE_DIR / "cache"
DEFAULT_LOGS_DIR = WORKSPACE_DIR / "logs"
DEFAULT_MATCHES_CSV = DEFAULT_DATA_DIR / "msd_deezer_matches.csv"
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
    candidates: list[Path] = []
    if DEFAULT_MATCHES_CSV.exists():
        candidates.append(DEFAULT_MATCHES_CSV)
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
            key="data",
            title="Generated match data",
            description="Pipeline CSV outputs and pending snapshots in data/.",
            paths=discover_data_outputs(),
        ),
        CleanupGroup(
            key="logs",
            title="Run logs",
            description="Per-session resilient download logs and summaries in logs/.",
            paths=(DEFAULT_LOGS_DIR,),
        ),
        CleanupGroup(
            key="pycache",
            title="Python cache files",
            description="All __pycache__ folders and stray .pyc files inside the workspace.",
            paths=discover_python_cache_outputs(),
        ),
    ]
    return [group for group in groups if group.existing_paths]


def delete_path(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False

    if dry_run:
        print(f"Would remove: {path}")
        return True

    if path.is_dir():
        shutil.rmtree(path)
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


def select_groups_gui(groups: list[CleanupGroup], dry_run: bool) -> list[CleanupGroup] | None:
    if tk is None:
        return None

    root = tk.Tk()
    root.title("Reset MSD Deezer Workspace")
    root.geometry("760x520")
    root.minsize(680, 420)
    root.resizable(True, True)

    result: dict[str, list[CleanupGroup] | None] = {"selection": None}
    variables: dict[str, tk.BooleanVar] = {
        group.key: tk.BooleanVar(value=True) for group in groups
    }
    select_all_var = tk.BooleanVar(value=True)

    def sync_select_all(*_: object) -> None:
        all_checked = all(var.get() for var in variables.values()) if variables else False
        if select_all_var.get() != all_checked:
            select_all_var.set(all_checked)

    def on_toggle_all() -> None:
        target_value = select_all_var.get()
        for var in variables.values():
            var.set(target_value)

    def on_confirm() -> None:
        selected = [group for group in groups if variables[group.key].get()]
        if not selected:
            if messagebox is not None:
                messagebox.showwarning("Nothing selected", "Choose at least one cleanup target.")
            return
        result["selection"] = selected
        root.destroy()

    def on_cancel() -> None:
        result["selection"] = None
        root.destroy()

    for var in variables.values():
        var.trace_add("write", sync_select_all)

    container = tk.Frame(root, padx=14, pady=14)
    container.pack(fill="both", expand=True)

    heading = "Choose which generated artifacts to remove"
    if dry_run:
        heading += " (dry run)"
    tk.Label(container, text=heading, font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(
        container,
        text="Only existing generated outputs are listed below. Select all or pick specific groups.",
        wraplength=710,
        justify="left",
    ).pack(anchor="w", pady=(6, 12))

    select_all_row = tk.Frame(container)
    select_all_row.pack(fill="x", pady=(0, 8))
    tk.Checkbutton(
        select_all_row,
        text="Select all",
        variable=select_all_var,
        command=on_toggle_all,
        font=("Segoe UI", 10, "bold"),
    ).pack(anchor="w")

    list_frame = tk.Frame(container)
    list_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0)
    scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
    content = tk.Frame(canvas)

    content.bind(
        "<Configure>",
        lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.create_window((0, 0), window=content, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for group in groups:
        row = tk.Frame(content, padx=8, pady=8, bd=1, relief="groove")
        row.pack(fill="x", pady=(0, 8))

        label = f"{group.title} ({group.item_count} path(s), {human_size(group.total_bytes)})"
        tk.Checkbutton(
            row,
            text=label,
            variable=variables[group.key],
            anchor="w",
            justify="left",
            wraplength=650,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        tk.Label(
            row,
            text=group.description,
            anchor="w",
            justify="left",
            wraplength=670,
            fg="#444444",
        ).pack(anchor="w", pady=(2, 4))

        preview_paths = list(group.existing_paths[:3])
        preview_text = "\n".join(str(path) for path in preview_paths)
        remaining = group.item_count - len(preview_paths)
        if remaining > 0:
            preview_text += f"\n... and {remaining} more path(s)"
        tk.Label(
            row,
            text=preview_text,
            anchor="w",
            justify="left",
            wraplength=670,
            fg="#1f1f1f",
        ).pack(anchor="w")

    button_row = tk.Frame(container)
    button_row.pack(fill="x", pady=(12, 0))
    tk.Button(button_row, text="Cancel", width=12, command=on_cancel).pack(side="right", padx=(8, 0))
    tk.Button(button_row, text="Delete selected", width=14, command=on_confirm).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    return result["selection"]


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
        "--no-gui",
        action="store_true",
        help="Skip the checkbox popup and use terminal selection instead.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Delete every discovered generated artifact group without prompting.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("audio", "features", "cluster_results", "cache", "data", "logs", "pycache"),
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
    elif not args.no_gui:
        selected_groups = select_groups_gui(groups, dry_run=args.dry_run)
        if selected_groups is None:
            print("Reset cancelled.")
            return 1
    else:
        print_group_summary(groups)
        print("\nNo GUI requested, so all listed groups will be selected.")
        selected_groups = groups

    if not selected_groups:
        print("Nothing selected.")
        return 0

    print_group_summary(selected_groups)
    remove_generated_files(selected_groups, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
