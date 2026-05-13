#!/usr/bin/env python3
"""Copy a deterministic 10k sample from DGX production audio into this run.

The production audio directory is read-only from this script's point of view:
files are copied with shutil.copy2(), never moved, renamed, or deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = (SCRIPT_DIR.parent / "audio").resolve()
TARGET_AUDIO = (SCRIPT_DIR / "audio_10000").resolve()
DATA_DIR = (SCRIPT_DIR / "data").resolve()
MANIFEST = DATA_DIR / "copied_audio_manifest.csv"
METADATA = DATA_DIR / "copy_metadata.json"
EXTENSIONS = (".mp3", ".m4a", ".flac", ".wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy N downloaded audio files into sample_10000_storage_run/audio_10000. "
            "Default source is ../audio, i.e. msd_deezer_workspace/audio on the DGX."
        )
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only this run's audio_10000 directory if it is non-empty.",
    )
    return parser.parse_args()


def assert_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Refusing path outside benchmark root: {path}") from exc


def reject_symlink(path: Path, label: str) -> None:
    if path.exists() and path.is_symlink():
        raise SystemExit(f"Refusing {label} symlink: {path}")


def list_audio(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise SystemExit(f"Source audio directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise SystemExit(f"Source audio path is not a directory: {source_dir}")

    files: list[Path] = []
    for ext in EXTENSIONS:
        files.extend(source_dir.glob(f"*{ext}"))
    files.sort(key=lambda p: p.name)
    return files


def prepare_target(force: bool) -> None:
    if SCRIPT_DIR.name != "sample_10000_storage_run":
        raise SystemExit(f"Unexpected benchmark root name: {SCRIPT_DIR}")
    assert_inside(TARGET_AUDIO, SCRIPT_DIR)
    assert_inside(DATA_DIR, SCRIPT_DIR)
    assert_inside(MANIFEST, SCRIPT_DIR)
    assert_inside(METADATA, SCRIPT_DIR)
    reject_symlink(TARGET_AUDIO, "target audio directory")
    reject_symlink(DATA_DIR, "data directory")
    if TARGET_AUDIO.exists():
        existing = list(TARGET_AUDIO.iterdir())
        if existing and not force:
            raise SystemExit(
                f"{TARGET_AUDIO} is not empty ({len(existing)} entries). "
                "Use --force only if you intentionally want to replace this run's sample."
            )
        if existing and force:
            shutil.rmtree(TARGET_AUDIO)
    TARGET_AUDIO.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reject_symlink(TARGET_AUDIO, "target audio directory")
    reject_symlink(DATA_DIR, "data directory")


def copy_one(index: int, src: Path) -> tuple[int, Path, Path, int]:
    dest = TARGET_AUDIO / src.name
    assert_inside(dest, SCRIPT_DIR)
    shutil.copy2(src, dest)
    return index, src, dest, src.stat().st_size


def main() -> int:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")

    source_dir = Path(args.source_dir).expanduser().resolve()
    if source_dir == TARGET_AUDIO:
        raise SystemExit("Source and target directories are identical.")
    try:
        source_dir.relative_to(SCRIPT_DIR)
        raise SystemExit("Source must be outside sample_10000_storage_run.")
    except ValueError:
        pass

    candidates = list_audio(source_dir)
    print(f"Source     : {source_dir}")
    print(f"Target     : {TARGET_AUDIO}")
    print(f"Candidates : {len(candidates)} audio files")
    print(f"Requested  : {args.n}")
    print(f"Seed       : {args.seed}")

    if len(candidates) < args.n:
        raise SystemExit(f"Need {args.n} audio files, found only {len(candidates)}.")

    names = [p.name for p in candidates]
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate filenames in source would collide in target.")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.n)
    selected.sort(key=lambda p: p.name)

    prepare_target(args.force)

    workers = max(1, min(args.workers, len(selected)))
    rows: list[tuple[int, Path, Path, int]] = []
    failures: list[tuple[str, str]] = []
    print(f"Copying with {workers} worker(s)...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(copy_one, idx, src): src
            for idx, src in enumerate(selected, start=1)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            src = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report and abort after pool
                failures.append((src.name, repr(exc)))
            if done % 500 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} done")

    if failures:
        print(f"ERROR: {len(failures)} copy failures", file=sys.stderr)
        for name, err in failures[:10]:
            print(f"  {name}: {err}", file=sys.stderr)
        return 2

    rows.sort(key=lambda r: r[0])
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "source_path", "dest_path", "bytes"])
        for index, src, dest, byte_count in rows:
            writer.writerow([index, str(src), str(dest), byte_count])

    metadata = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_dir": str(source_dir),
        "target_dir": str(TARGET_AUDIO),
        "candidate_count": len(candidates),
        "sample_count": args.n,
        "seed": args.seed,
        "workers": workers,
        "extensions": list(EXTENSIONS),
        "manifest": str(MANIFEST),
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    copied = len(list(TARGET_AUDIO.glob("*")))
    if copied != args.n:
        raise SystemExit(f"Post-copy count mismatch: expected {args.n}, found {copied}.")

    print(f"Done. Copied {copied} files.")
    print(f"Manifest: {MANIFEST}")
    print(f"Metadata: {METADATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
