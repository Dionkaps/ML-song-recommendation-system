"""
Pick a deterministic 10k random subset of the production `audio/` catalogue
and materialise it inside `dimred_benchmark/audio/` as hard copies
(default) or symlinks (`--mode symlink`).

This is the only step in the benchmark that touches anything outside the
test root, and it is strictly read-only with respect to the production
catalogue: it inspects file names under `../audio/` and creates entries
under `./audio/`. Nothing in `../audio/` is moved, modified, or deleted.

Reproducibility: the subset is selected with a fixed `random.Random(seed)`
and an alphabetically sorted candidate list, so two invocations on the
same source directory produce identical samples.

Default is `--mode copy` because the user spec calls for true copies of
the test sample. `--mode symlink` is available for disk-constrained
hosts (saves ~30 GB on a 10k sample); on the DGX `/storage/data4` has
no quota so copies are the safer default.

Safety:
  * Refuses to write to any path outside `dimred_benchmark/audio/`.
  * Refuses to run if `dimred_benchmark/audio/` already exists and is
    non-empty unless `--force` is passed (avoids silently extending a
    previous sample with a different seed).
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent  # .../msd_deezer_workspace/dimred_benchmark
PRODUCTION_AUDIO = (THIS_DIR.parent / "audio").resolve()
DEST_AUDIO = (THIS_DIR / "audio").resolve()

# Mirrors audio_preprocessing.processor.SUPPORTED_AUDIO_EXTENSIONS so the
# sample matches what the preprocessor will actually accept. Ordering
# matters: mp3 first (the pristine Deezer download format).
SOURCE_EXTENSIONS = (".mp3", ".m4a", ".flac", ".wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample N random songs from ../audio/ into dimred_benchmark/audio/. "
            "Hard copies by default (per spec); --mode symlink saves ~30 GB on "
            "a 10k sample if disk is tight."
        ),
    )
    parser.add_argument("--n", type=int, default=10_000, help="Sample size (default: 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--mode", choices=["copy", "symlink"], default="copy",
        help="copy: duplicate files (default, ~30 GB for 10k songs). symlink: zero extra disk.",
    )
    parser.add_argument(
        "--source-audio-dir", default=str(PRODUCTION_AUDIO),
        help="Path to the production audio/ directory (default: ../audio).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing non-empty dimred_benchmark/audio/ directory.",
    )
    parser.add_argument(
        "--manifest", default=str(THIS_DIR / "sampled_manifest.txt"),
        help="Where to write the list of sampled file basenames.",
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel I/O workers for the materialise step (default: 16). "
             "Copies to /storage/data4 NFS run ~10-20x faster threaded.",
    )
    return parser.parse_args()


def list_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise SystemExit(f"Source audio directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise SystemExit(f"Source audio path is not a directory: {source_dir}")

    files: list[Path] = []
    for ext in SOURCE_EXTENSIONS:
        files.extend(source_dir.glob(f"*{ext}"))
    # Deterministic sort so the random.sample selection is repeatable.
    files.sort(key=lambda p: p.name)
    if not files:
        raise SystemExit(
            f"No audio files (extensions {SOURCE_EXTENSIONS}) under {source_dir}"
        )
    return files


def reset_dest(dest: Path, force: bool) -> None:
    if dest.exists():
        existing = list(dest.iterdir()) if dest.is_dir() else []
        if existing and not force:
            raise SystemExit(
                f"Destination {dest} already contains {len(existing)} entries. "
                f"Re-run with --force to wipe and resample."
            )
        if force:
            shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)


def materialise(src: Path, dest: Path, mode: str) -> None:
    # Defensive: refuse any destination outside DEST_AUDIO. Belt-and-braces
    # against a malicious filename containing '..' on case-insensitive FS.
    resolved_dest = dest.resolve()
    try:
        resolved_dest.relative_to(DEST_AUDIO)
    except ValueError:
        raise RuntimeError(
            f"Refusing to materialise outside test sample dir.\n"
            f"  attempted: {resolved_dest}\n"
            f"  test root: {DEST_AUDIO}"
        )

    if mode == "symlink":
        try:
            os.symlink(src, dest)
        except OSError as exc:
            # Windows without Developer Mode: WinError 1314 (privilege not held).
            raise SystemExit(
                f"symlink({src} -> {dest}) failed: {exc}\n"
                f"On Windows, enable Developer Mode or rerun with --mode copy."
            )
    else:  # copy
        # `copy2` preserves mtime, which keeps the per-file ordering stable
        # across re-runs in case downstream code sorts by modification time.
        shutil.copy2(src, dest)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_audio_dir).resolve()
    if source_dir == DEST_AUDIO:
        raise SystemExit("Source and destination directories are identical.")

    print(f"Source     : {source_dir}")
    print(f"Destination: {DEST_AUDIO}")
    print(f"Mode       : {args.mode}")
    print(f"N          : {args.n}")
    print(f"Seed       : {args.seed}")

    candidates = list_source_files(source_dir)
    print(f"Candidates : {len(candidates)} audio files in source")

    if args.n > len(candidates):
        raise SystemExit(
            f"Requested {args.n} samples but only {len(candidates)} candidates exist."
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(candidates, args.n)
    sampled.sort(key=lambda p: p.name)  # deterministic write order

    reset_dest(DEST_AUDIO, args.force)

    # Threaded materialise. shutil.copy2 and os.symlink both release the GIL
    # in the underlying syscalls (read/write/symlinkat), so threads scale on
    # I/O even though Python is single-threaded for CPU. NFS metadata ops
    # are the bottleneck; 16 workers saturate /storage/data4 well.
    workers = max(1, min(int(args.workers), len(sampled)))
    print(f"\nMaterialising {len(sampled)} entries with {workers} worker(s)...")

    failures: list[tuple[str, str]] = []
    completed = 0

    def _one(src_path: Path) -> tuple[Path, Exception | None]:
        try:
            materialise(src_path, DEST_AUDIO / src_path.name, args.mode)
            return src_path, None
        except SystemExit as exc:
            # materialise() raises SystemExit on Windows symlink-permission
            # errors; convert to a regular exception so the pool keeps going.
            return src_path, RuntimeError(str(exc))
        except Exception as exc:
            return src_path, exc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, src) for src in sampled]
        for fut in as_completed(futures):
            src_path, exc = fut.result()
            completed += 1
            if exc is not None:
                failures.append((src_path.name, repr(exc)))
            # Lightweight progress without bringing in tqdm:
            if completed % 500 == 0 or completed == len(sampled):
                print(f"  {completed}/{len(sampled)} done"
                      f"{'  (' + str(len(failures)) + ' failed)' if failures else ''}")

    if failures:
        print(f"\nWARNING: {len(failures)} entries failed to materialise:")
        for name, repr_exc in failures[:5]:
            print(f"  {name}: {repr_exc}")
        if len(failures) > 5:
            print(f"  ... ({len(failures) - 5} more)")
        # Don't write the manifest if we didn't end up with the requested set.
        raise SystemExit(2)

    manifest_path = Path(args.manifest)
    manifest_path.write_text(
        "\n".join(p.name for p in sampled) + "\n", encoding="utf-8",
    )

    print(f"\nDone. {len(sampled)} entries materialised in {DEST_AUDIO}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
