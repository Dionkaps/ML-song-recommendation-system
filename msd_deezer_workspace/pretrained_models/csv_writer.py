"""
CSV generation for pretrained embedding extraction.

Produces:
  - One per-model CSV (metadata + dense embedding columns)
  - One fused CSV with L2-normalized-per-model concatenation
    (directly compatible with clustering/shared.py)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    METADATA_COLUMNS,
    SUPPORTED_AUDIO_EXTENSIONS,
    extract_ids_from_filename,
    l2_normalize,
)


logger = logging.getLogger(__name__)


DURATION_KEY = "__duration_sec__"
SAMPLE_RATE_KEY = "__sample_rate__"


def _npz_embeddings(npz_path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ and return a regular dict (copies arrays so the file handle can close)."""
    with np.load(npz_path) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _find_audio_path(audio_dir: Path, stem: str) -> str:
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _metadata_row(
    npz_path: Path,
    audio_path: str,
    duration_sec: float,
    sample_rate: int,
) -> dict[str, Any]:
    ids = extract_ids_from_filename(npz_path.name)
    return {
        "file": npz_path.stem + ".wav",
        "audio_path": audio_path,
        "raw_feature_path": str(npz_path.resolve()),
        "msd_track_id": ids["msd_track_id"],
        "deezer_track_id": ids["deezer_track_id"],
        "sample_rate": int(sample_rate) if sample_rate > 0 else 22050,
        "duration_sec": round(float(duration_sec), 6),
        "frames": 0,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["file"]):
            writer.writerow(row)


def _gather_records(
    raw_dir: Path,
    audio_dir: Path,
) -> list[dict[str, Any]]:
    """Load all NPZ files into a list of records ready for CSV emission."""
    records: list[dict[str, Any]] = []
    for npz_path in sorted(raw_dir.glob("*.npz")):
        try:
            blob = _npz_embeddings(npz_path)
        except Exception as exc:
            logger.warning("Could not load %s: %s", npz_path.name, exc)
            continue

        duration_sec = float(blob.pop(DURATION_KEY, np.float32(0.0)))
        sample_rate = int(blob.pop(SAMPLE_RATE_KEY, np.int32(22050)))
        audio_path = _find_audio_path(audio_dir, npz_path.stem)

        records.append({
            "npz_path": npz_path,
            "audio_path": audio_path,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "embeddings": blob,
        })
    return records


def write_per_model_csv(
    model_name: str,
    output_path: Path,
    records: list[dict[str, Any]],
) -> int:
    """Write a single-model CSV (no normalization needed).

    Returns the number of rows written.
    """
    dim = 0
    for rec in records:
        if model_name in rec["embeddings"]:
            dim = int(rec["embeddings"][model_name].shape[0])
            break
    if dim == 0:
        logger.warning("No embeddings found for model '%s' -- skipping CSV", model_name)
        return 0

    fieldnames = list(METADATA_COLUMNS) + [
        f"{model_name}_{i + 1:04d}" for i in range(dim)
    ]

    rows: list[dict[str, Any]] = []
    for rec in records:
        emb = rec["embeddings"].get(model_name)
        if emb is None:
            continue
        row = _metadata_row(
            rec["npz_path"], rec["audio_path"],
            rec["duration_sec"], rec["sample_rate"],
        )
        emb = emb.astype(np.float32)
        for i in range(dim):
            row[f"{model_name}_{i + 1:04d}"] = round(float(emb[i]), 8)
        rows.append(row)

    _write_csv(output_path, fieldnames, rows)
    logger.info(
        "Wrote %s (%d songs, %d dims)", output_path.name, len(rows), dim,
    )
    return len(rows)


def write_fused_csv(
    output_path: Path,
    records: list[dict[str, Any]],
    model_order: list[str],
) -> tuple[int, dict[str, int]]:
    """Write the fused CSV with L2-normalized concatenation.

    Only songs that have embeddings from *every* model in `model_order`
    are included. Returns (row count, dim-per-model map).
    """
    fieldnames = list(METADATA_COLUMNS)
    dims: dict[str, int] = {}
    for model_name in model_order:
        for rec in records:
            if model_name in rec["embeddings"]:
                d = int(rec["embeddings"][model_name].shape[0])
                dims[model_name] = d
                fieldnames.extend(f"{model_name}_{i + 1:04d}" for i in range(d))
                break

    rows: list[dict[str, Any]] = []
    for rec in records:
        if not all(m in rec["embeddings"] for m in model_order):
            continue

        row = _metadata_row(
            rec["npz_path"], rec["audio_path"],
            rec["duration_sec"], rec["sample_rate"],
        )

        for model_name in model_order:
            emb = l2_normalize(rec["embeddings"][model_name].astype(np.float32))
            d = dims[model_name]
            for i in range(d):
                row[f"{model_name}_{i + 1:04d}"] = round(float(emb[i]), 8)

        rows.append(row)

    _write_csv(output_path, fieldnames, rows)
    total_dims = sum(dims.values())
    logger.info(
        "Wrote %s (%d songs, %d dims: %s)",
        output_path.name, len(rows), total_dims,
        " + ".join(f"{m}={d}" for m, d in dims.items()),
    )
    return len(rows), dims


def generate_all_csvs(
    raw_dir: Path,
    output_dir: Path,
    audio_dir: Path,
    active_models: list[str],
) -> dict[str, str]:
    """Build all CSV files from existing per-song NPZ embeddings.

    Returns a dict mapping labels to output paths.
    """
    records = _gather_records(raw_dir, audio_dir)
    if not records:
        logger.warning("No NPZ files found in %s", raw_dir)
        return {}

    paths: dict[str, str] = {}

    for model_name in active_models:
        csv_path = output_dir / f"{model_name}_vectors.csv"
        n = write_per_model_csv(model_name, csv_path, records)
        if n > 0:
            paths[model_name] = str(csv_path)

    fused_path = output_dir / "feature_vectors.csv"
    n_fused, _ = write_fused_csv(fused_path, records, active_models)
    if n_fused > 0:
        paths["fused"] = str(fused_path)

    return paths
