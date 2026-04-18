from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_DIR = WORKSPACE_DIR / "features"
DEFAULT_CLUSTER_OUTPUT_DIR = WORKSPACE_DIR / "cluster_results"
DEFAULT_FEATURE_SUMMARY_CSV = DEFAULT_FEATURES_DIR / "feature_vectors.csv"
DEFAULT_RAW_FEATURE_DIR = DEFAULT_FEATURES_DIR / "raw"
METADATA_COLUMNS = [
    "file",
    "audio_path",
    "raw_feature_path",
    "msd_track_id",
    "deezer_track_id",
    "sample_rate",
    "duration_sec",
    "frames",
]
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a")


@dataclass
class PreparedDataset:
    data_frame: pd.DataFrame
    metadata: pd.DataFrame
    feature_names: list[str]
    raw_matrix: np.ndarray
    scaled_matrix: np.ndarray
    reduced_matrix: np.ndarray
    scaler: StandardScaler
    pca_components: int
    pca_explained_variance_ratio: float
    umap_components: int | None
    umap_n_neighbors: int | None
    umap_min_dist: float | None
    source_path: Path
    summary_csv_path: Path | None


def summary_fieldnames(n_mfcc: int = 20) -> list[str]:
    fields = list(METADATA_COLUMNS)

    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        for index in range(1, n_mfcc + 1):
            fields.append(f"{prefix}_{index:02d}_mean")
            fields.append(f"{prefix}_{index:02d}_std")

    for index in range(1, 13):
        fields.append(f"chroma_{index:02d}_mean")
        fields.append(f"chroma_{index:02d}_std")

    for prefix in (
        "spectral_centroid",
        "spectral_rolloff",
        "spectral_flux",
        "spectral_flatness",
        "spectral_bandwidth",
        "zero_crossing_rate",
        "beat_strength",
    ):
        fields.append(f"{prefix}_mean")
        fields.append(f"{prefix}_std")

    # Tzanetakis rhythm features (track-level scalars)
    fields.append("tempo_bpm")
    fields.append("tempogram_peak1_bpm")
    fields.append("tempogram_peak1_amp")
    fields.append("tempogram_peak2_bpm")
    fields.append("tempogram_peak2_amp")
    fields.append("tempogram_peak_ratio")
    fields.append("tempogram_sum")

    # Tzanetakis low-energy rate
    fields.append("low_energy_rate")

    return fields


def extract_ids_from_filename(filename: str) -> dict[str, str]:
    msd_match = re.search(r"\[(TR[A-Z0-9]+)\]", filename)
    deezer_match = re.search(r"\[deezer-(\d+)\]", filename, flags=re.IGNORECASE)
    return {
        "msd_track_id": msd_match.group(1) if msd_match else "",
        "deezer_track_id": deezer_match.group(1) if deezer_match else "",
    }


def find_audio_path_for_feature(npz_path: Path) -> str:
    audio_dir = WORKSPACE_DIR / "audio"
    stem = npz_path.stem
    for extension in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{stem}{extension}"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def summarize_feature_npz(npz_path: Path) -> dict[str, Any]:
    data = np.load(npz_path)
    ids = extract_ids_from_filename(npz_path.name)
    audio_path = find_audio_path_for_feature(npz_path)

    row: dict[str, Any] = {
        "file": Path(audio_path).name if audio_path else npz_path.stem,
        "audio_path": audio_path,
        "raw_feature_path": str(npz_path.resolve()),
        "msd_track_id": ids["msd_track_id"],
        "deezer_track_id": ids["deezer_track_id"],
        "sample_rate": int(np.asarray(data["sample_rate"]).item()),
        "duration_sec": round(float(np.asarray(data["duration_sec"]).item()), 6),
        "frames": int(np.asarray(data["mfcc"]).shape[1]),
    }

    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        matrix = np.asarray(data[prefix], dtype=np.float32)
        means = matrix.mean(axis=1)
        stds = matrix.std(axis=1)
        for index, (mean_value, std_value) in enumerate(zip(means, stds), start=1):
            row[f"{prefix}_{index:02d}_mean"] = round(float(mean_value), 8)
            row[f"{prefix}_{index:02d}_std"] = round(float(std_value), 8)

    chroma = np.asarray(data["chroma"], dtype=np.float32)
    chroma_means = chroma.mean(axis=1)
    chroma_stds = chroma.std(axis=1)
    for index, (mean_value, std_value) in enumerate(zip(chroma_means, chroma_stds), start=1):
        row[f"chroma_{index:02d}_mean"] = round(float(mean_value), 8)
        row[f"chroma_{index:02d}_std"] = round(float(std_value), 8)

    for prefix in (
        "spectral_centroid",
        "spectral_rolloff",
        "spectral_flux",
        "spectral_flatness",
        "spectral_bandwidth",
        "zero_crossing_rate",
        "beat_strength",
    ):
        values = np.asarray(data[prefix], dtype=np.float32)
        row[f"{prefix}_mean"] = round(float(values.mean()), 8)
        row[f"{prefix}_std"] = round(float(values.std()), 8)

    # Tzanetakis rhythm scalars
    for scalar_key in (
        "tempo_bpm",
        "tempogram_peak1_bpm",
        "tempogram_peak1_amp",
        "tempogram_peak2_bpm",
        "tempogram_peak2_amp",
        "tempogram_peak_ratio",
        "tempogram_sum",
        "low_energy_rate",
    ):
        row[scalar_key] = round(float(np.asarray(data[scalar_key]).item()), 8)

    return row


def build_summary_csv_from_raw(raw_dir: Path, summary_csv_path: Path) -> pd.DataFrame:
    npz_files = sorted(raw_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No raw feature files found in {raw_dir}")

    rows = []
    for npz_path in tqdm(npz_files, desc="Building feature summary", unit="song"):
        rows.append(summarize_feature_npz(npz_path))

    fieldnames = summary_fieldnames()
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return pd.DataFrame(rows)


def resolve_feature_sources(features_path: str | Path) -> tuple[Path, Path | None, Path | None]:
    source_path = Path(features_path).resolve()
    if source_path.is_file():
        if source_path.suffix.lower() != ".csv":
            raise ValueError(f"Unsupported feature file: {source_path}")
        return source_path, source_path, None

    if not source_path.exists():
        raise FileNotFoundError(f"Features path not found: {source_path}")

    summary_csv_path = source_path / "feature_vectors.csv"
    raw_dir = source_path / "raw"

    if summary_csv_path.exists():
        return source_path, summary_csv_path, raw_dir if raw_dir.exists() else None
    if raw_dir.exists():
        return source_path, summary_csv_path, raw_dir
    if list(source_path.glob("*.npz")):
        return source_path, None, source_path

    raise FileNotFoundError(f"Could not locate feature_vectors.csv or raw .npz files under {source_path}")


def load_feature_table(features_path: str | Path) -> tuple[pd.DataFrame, Path, Path | None]:
    source_path, summary_csv_path, raw_dir = resolve_feature_sources(features_path)

    if summary_csv_path is not None and summary_csv_path.exists():
        data_frame = pd.read_csv(summary_csv_path)
        return data_frame, source_path, summary_csv_path

    if raw_dir is None:
        raise FileNotFoundError(f"No usable features found under {source_path}")

    if summary_csv_path is None:
        summary_csv_path = raw_dir.parent / "feature_vectors.csv"

    data_frame = build_summary_csv_from_raw(raw_dir, summary_csv_path)
    return data_frame, source_path, summary_csv_path


def select_feature_columns(data_frame: pd.DataFrame) -> list[str]:
    return [column for column in data_frame.columns if column not in METADATA_COLUMNS]


def candidate_cluster_counts(n_samples: int, max_clusters: int = 40) -> list[int]:
    if n_samples < 3:
        raise ValueError("At least 3 samples are required for automatic cluster selection.")

    upper = min(max_clusters, n_samples - 1)
    upper = max(2, upper)
    return list(range(2, upper + 1))


def prepare_dataset(
    features_path: str | Path = DEFAULT_FEATURES_DIR,
    limit: int | None = None,
    pca_variance_threshold: float = 0.99,
    max_pca_components: int = 100,
    umap_n_components: int = 15,
    umap_n_neighbors: int = 40,
    umap_min_dist: float = 0.01,
    umap_random_state: int = 42,
    disable_umap: bool = False,
) -> PreparedDataset:
    if not (0 < pca_variance_threshold <= 1.0):
        raise ValueError("pca_variance_threshold must be within (0, 1].")

    data_frame, source_path, summary_csv_path = load_feature_table(features_path)
    if data_frame.empty:
        raise ValueError(f"No feature rows available from {source_path}")

    if limit is not None:
        data_frame = data_frame.head(max(1, int(limit))).copy()
    else:
        data_frame = data_frame.copy()

    feature_names = select_feature_columns(data_frame)
    if not feature_names:
        raise ValueError("No numeric feature columns found for clustering.")

    numeric_frame = data_frame[feature_names].apply(pd.to_numeric, errors="coerce")
    valid_mask = np.isfinite(numeric_frame.to_numpy(dtype=np.float64)).all(axis=1)
    if not valid_mask.all():
        data_frame = data_frame.loc[valid_mask].reset_index(drop=True)
        numeric_frame = numeric_frame.loc[valid_mask].reset_index(drop=True)

    if len(data_frame) < 3:
        raise ValueError("At least 3 valid feature rows are required for clustering.")

    raw_matrix = numeric_frame.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(raw_matrix)

    max_components = min(max_pca_components, scaled_matrix.shape[0] - 1, scaled_matrix.shape[1])
    if max_components < 2:
        pca_reduced = scaled_matrix.copy()
        pca_components = pca_reduced.shape[1]
        explained_variance = 1.0
    else:
        exploratory_pca = PCA(n_components=max_components, svd_solver="full")
        exploratory_reduced = exploratory_pca.fit_transform(scaled_matrix)
        cumulative = np.cumsum(exploratory_pca.explained_variance_ratio_)
        chosen_components = int(np.searchsorted(cumulative, pca_variance_threshold) + 1)
        chosen_components = max(2, min(chosen_components, max_components))
        pca_reduced = exploratory_reduced[:, :chosen_components]
        pca_components = chosen_components
        explained_variance = float(cumulative[chosen_components - 1])

    umap_dims: int | None = None
    umap_neighbors: int | None = None
    umap_dist: float | None = None

    if not disable_umap and pca_reduced.shape[1] > umap_n_components:
        umap_model = UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=umap_random_state,
        )
        reduced_matrix = umap_model.fit_transform(pca_reduced)
        umap_dims = umap_n_components
        umap_neighbors = umap_n_neighbors
        umap_dist = umap_min_dist
    else:
        reduced_matrix = pca_reduced

    metadata = data_frame[METADATA_COLUMNS].copy()
    return PreparedDataset(
        data_frame=data_frame,
        metadata=metadata,
        feature_names=feature_names,
        raw_matrix=raw_matrix,
        scaled_matrix=scaled_matrix,
        reduced_matrix=reduced_matrix,
        scaler=scaler,
        pca_components=pca_components,
        pca_explained_variance_ratio=explained_variance,
        umap_components=umap_dims,
        umap_n_neighbors=umap_neighbors,
        umap_min_dist=umap_dist,
        source_path=source_path,
        summary_csv_path=summary_csv_path,
    )


def feature_source_key(features_path: str | Path) -> str:
    # Derives a filesystem-safe slug identifying which feature pipeline a run
    # came from (e.g. "features", "pretrained_embeddings"). Used to route
    # clustering outputs into separate per-source subtrees so pretrained-
    # embedding results don't overwrite hand-crafted audio-feature results.
    path = Path(features_path).resolve()
    if path.is_file():
        raw = path.parent.name
    else:
        raw = path.name
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip()).strip("_")
    return cleaned or "features"


def default_algorithm_output_dir(features_path: str | Path, algorithm: str) -> Path:
    return DEFAULT_CLUSTER_OUTPUT_DIR / feature_source_key(features_path) / algorithm


def default_cluster_results_dir(features_path: str | Path) -> Path:
    return DEFAULT_CLUSTER_OUTPUT_DIR / feature_source_key(features_path)


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_assignments(assignments: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
