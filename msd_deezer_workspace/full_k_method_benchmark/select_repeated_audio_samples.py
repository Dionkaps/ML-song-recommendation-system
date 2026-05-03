from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from safety import WORKSPACE_DIR, assert_inside_workspace


PRODUCTION_AUDIO = (WORKSPACE_DIR.parent / "audio").resolve()
SAMPLES_ROOT = (WORKSPACE_DIR / "samples").resolve()
SOURCE_EXTENSIONS = (".mp3", ".m4a", ".flac", ".wav")

PIPELINE_DIRS = ("audio_preprocessing", "pretrained_models", "clustering")
PIPELINE_FILES = (
    "safety.py",
    "preprocess_downloaded_audio.py",
    "extract_audio_features.py",
    "extract_pretrained_embeddings.py",
    "merge_sharded_embeddings.py",
    "run_parallel_extraction.sh",
    "run_kmeans_gap_stability.py",
    "run_candidate_k_validation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create repeated copied-audio sample workspaces for the full-dataset "
            "K-method benchmark. Production audio is read-only; every copied "
            "file and every generated script/output stays under this benchmark."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[101, 202, 303, 404, 505],
        help="One sample workspace is created per seed.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--source-audio-dir",
        default=str(PRODUCTION_AUDIO),
        help="Read-only production audio directory, default ../audio.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Default copy keeps the benchmark independent of production audio.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing sample audio folders for the requested seeds.",
    )
    parser.add_argument(
        "--refresh-scripts",
        action="store_true",
        help="Refresh copied pipeline scripts in existing sample workspaces.",
    )
    return parser.parse_args()


def list_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise SystemExit(f"Source audio directory not found: {source_dir}")
    files: list[Path] = []
    for ext in SOURCE_EXTENSIONS:
        files.extend(source_dir.glob(f"*{ext}"))
    files.sort(key=lambda p: p.name)
    if not files:
        raise SystemExit(f"No audio files with extensions {SOURCE_EXTENSIONS} under {source_dir}")
    return files


def copy_pipeline_template(sample_dir: Path, refresh: bool = False) -> None:
    sample_dir = assert_inside_workspace(sample_dir, "sample_workspace")
    for dirname in PIPELINE_DIRS:
        src = WORKSPACE_DIR / dirname
        dst = sample_dir / dirname
        if dst.exists() and not refresh:
            continue
        shutil.copytree(src, dst, dirs_exist_ok=True)
    for filename in PIPELINE_FILES:
        src = WORKSPACE_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"Template file missing: {src}")
        dst = sample_dir / filename
        if dst.exists() and not refresh:
            continue
        shutil.copy2(src, dst)


def reset_audio_dir(audio_dir: Path, force: bool) -> None:
    audio_dir = assert_inside_workspace(audio_dir, "sample_audio_dir")
    if audio_dir.exists():
        existing = list(audio_dir.iterdir()) if audio_dir.is_dir() else []
        if existing and not force:
            raise SystemExit(
                f"{audio_dir} already contains {len(existing)} file(s). "
                "Use --force to replace this sample."
            )
        if force:
            shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)


def materialise_one(src: Path, dest: Path, mode: str, audio_dir: Path) -> None:
    dest = assert_inside_workspace(dest, "sample_audio_file")
    try:
        dest.relative_to(audio_dir)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to write outside sample audio dir: {dest}") from exc
    if mode == "symlink":
        os.symlink(src, dest)
    else:
        shutil.copy2(src, dest)


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest = assert_inside_workspace(WORKSPACE_DIR / "sample_manifest.csv", "sample_manifest")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "seed", "sample_size", "file", "source_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_audio_dir).resolve()
    candidates = list_source_files(source_dir)
    if args.sample_size > len(candidates):
        raise SystemExit(
            f"Requested sample size {args.sample_size}, but only {len(candidates)} files exist."
        )

    samples_root = assert_inside_workspace(SAMPLES_ROOT, "samples_root")
    samples_root.mkdir(parents=True, exist_ok=True)

    print(f"Source audio : {source_dir}")
    print(f"Samples root : {samples_root}")
    print(f"Sample size  : {args.sample_size}")
    print(f"Seeds        : {', '.join(str(s) for s in args.seeds)}")
    print(f"Mode         : {args.mode}")
    print(f"Workers      : {args.workers}")

    manifest_rows: list[dict[str, str]] = []
    for seed in args.seeds:
        sample_id = f"sample_{args.sample_size // 1000}k_seed{seed}"
        sample_dir = assert_inside_workspace(samples_root / sample_id, "sample_workspace")
        audio_dir = assert_inside_workspace(sample_dir / "audio", "sample_audio_dir")
        sample_dir.mkdir(parents=True, exist_ok=True)
        copy_pipeline_template(sample_dir, refresh=args.refresh_scripts)

        if audio_dir.exists() and any(audio_dir.iterdir()) and not args.force:
            print(
                f"\n[{sample_id}] audio already exists; leaving copied audio untouched. "
                "Use --force to replace it."
            )
            continue

        reset_audio_dir(audio_dir, force=args.force)
        rng = random.Random(seed)
        selected = rng.sample(candidates, args.sample_size)
        selected.sort(key=lambda p: p.name)

        print(f"\n[{sample_id}] materialising {len(selected)} audio files...")
        workers = max(1, min(int(args.workers), len(selected)))
        failures: list[tuple[str, str]] = []
        completed = 0

        def _one(src: Path) -> tuple[Path, Exception | None]:
            try:
                materialise_one(src, audio_dir / src.name, args.mode, audio_dir)
                return src, None
            except Exception as exc:  # noqa: BLE001 - collect copy failures
                return src, exc

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_one, src) for src in selected]
            for future in as_completed(futures):
                src, exc = future.result()
                completed += 1
                if exc is not None:
                    failures.append((src.name, repr(exc)))
                if completed % 500 == 0 or completed == len(selected):
                    print(f"  {completed}/{len(selected)} done")

        if failures:
            for name, error in failures[:5]:
                print(f"  FAILED {name}: {error}")
            raise SystemExit(f"{len(failures)} file(s) failed in {sample_id}")

        sample_manifest = assert_inside_workspace(sample_dir / "sample_manifest.txt", "sample_manifest")
        sample_manifest.write_text(
            "\n".join(p.name for p in selected) + "\n",
            encoding="utf-8",
        )
        for src in selected:
            manifest_rows.append({
                "sample_id": sample_id,
                "seed": str(seed),
                "sample_size": str(args.sample_size),
                "file": src.name,
                "source_path": str(src),
            })

    if manifest_rows:
        write_manifest(manifest_rows)
    print(f"\nDone. Created {len(args.seeds)} sample workspace(s) under {samples_root}")


if __name__ == "__main__":
    main()
