from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ZERO_BYTE_SKIP_PARTS = {".git", ".venv", "node_modules"}


def _is_zero_byte_csv_candidate(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".csv"
        and path.stat().st_size == 0
        and not any(part in ZERO_BYTE_SKIP_PARTS for part in path.parts)
    )


def find_junk_csvs(project_root: Path) -> List[Path]:
    candidates: List[Path] = []

    for path in sorted((project_root / "data").glob("_tmp_*.csv")):
        if path.is_file():
            candidates.append(path)

    for relative in (
        Path("data") / "songs_genre_list.csv",
        Path("data") / "unique_genres.csv",
    ):
        path = project_root / relative
        if path.exists():
            candidates.append(path)

    for path in project_root.rglob("*.csv"):
        if _is_zero_byte_csv_candidate(path) and path not in candidates:
            candidates.append(path)

    return sorted(set(candidates))


def cleanup_junk_csvs(project_root: Path, apply: bool) -> Dict[str, object]:
    removed: List[str] = []
    found = find_junk_csvs(project_root)

    for path in found:
        if apply and path.exists():
            path.unlink()
            removed.append(str(path))

    return {
        "project_root": str(project_root),
        "apply": bool(apply),
        "junk_csv_candidates": [str(path) for path in found],
        "removed": removed,
        "removed_count": int(len(removed)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove known temporary/obsolete CSV artifacts from the project."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the detected junk CSV files.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for the cleanup summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    if not project_root.is_absolute():
        project_root = (PROJECT_ROOT / project_root).resolve()

    payload = cleanup_junk_csvs(project_root, apply=bool(args.apply))
    text = json.dumps(payload, indent=2)
    print(text)

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = (project_root / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
