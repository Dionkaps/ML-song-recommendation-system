import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv
from src.features.feature_qc import (
    FEATURE_KEYS as EXTRACTED_FEATURE_KEYS,
    audio_library_basenames,
    collect_feature_bundle_inventory,
    load_validated_feature_bundle,
)
from src.utils import genre_mapper
from src.utils.genre_taxonomy import resolve_pipeline_songs_csv
from src.utils.song_metadata import build_audio_metadata_frame


_AUDIO_FEATURE_DISPLAY_NAMES = {
    "mfcc": "MFCC",
    "delta_mfcc": "DeltaMFCC",
    "delta2_mfcc": "Delta2MFCC",
    "spectral_centroid": "SpectralCentroid",
    "spectral_rolloff": "SpectralRolloff",
    "spectral_flux": "SpectralFlux",
    "spectral_flatness": "SpectralFlatness",
    "zero_crossing_rate": "ZCR",
    "chroma": "Chroma",
    "beat_strength": "BeatStrength",
}


def _resolve_selected_audio_feature_keys(
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> List[str]:
    """Resolve the clustering feature subset from config or an explicit override."""

    if selected_audio_feature_keys is None:
        selected_audio_feature_keys = list(
            getattr(fv, "clustering_audio_feature_keys", fv.AUDIO_FEATURE_KEYS)
        )

    resolved: List[str] = []
    seen = set()
    for key in selected_audio_feature_keys:
        if key not in seen:
            resolved.append(key)
            seen.add(key)

    unknown = [key for key in resolved if key not in fv.AUDIO_FEATURE_KEYS]
    if unknown:
        raise ValueError(
            f"Unknown audio feature keys in clustering subset: {unknown}. "
            f"Available keys: {fv.AUDIO_FEATURE_KEYS}"
        )

    if not resolved:
        raise ValueError("selected_audio_feature_keys must not be empty")

    return resolved


def _get_audio_group_specs(
    n_mfcc: int = fv.n_mfcc,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> List[Tuple[str, str, int]]:
    """Return the summarized feature groups used for clustering."""

    feature_sizes = {
        "mfcc": 2 * n_mfcc,
        "delta_mfcc": 2 * n_mfcc,
        "delta2_mfcc": 2 * n_mfcc,
        "spectral_centroid": 2,
        "spectral_rolloff": 2,
        "spectral_flux": 2,
        "spectral_flatness": 2,
        "zero_crossing_rate": 2,
        "chroma": 2 * fv.n_chroma,
        "beat_strength": 4,
    }

    return [
        (key, _AUDIO_FEATURE_DISPLAY_NAMES[key], feature_sizes[key])
        for key in _resolve_selected_audio_feature_keys(selected_audio_feature_keys)
    ]


def _get_full_group_specs(
    n_mfcc: int = fv.n_mfcc,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> List[Tuple[str, str, int]]:
    """Return all clustering groups in the order they are concatenated."""

    specs = list(_get_audio_group_specs(n_mfcc, selected_audio_feature_keys))
    if include_msd:
        specs.extend(
            [
                ("key", "Key", 12),
                ("mode", "Mode", 2),
                ("loudness", "Loudness", 1),
                ("msd_tempo", "MSD_Tempo", 1),
            ]
        )
    if include_genre:
        specs.append(("genre", "Genre", n_genres))
    return specs


def infer_feature_subset_name(
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> str:
    """Infer a readable subset identifier for manifests and retrieval artifacts."""

    resolved = _resolve_selected_audio_feature_keys(selected_audio_feature_keys)
    if resolved == list(getattr(fv, "clustering_audio_feature_keys", [])):
        return str(getattr(fv, "clustering_feature_subset_name", "unknown"))
    if resolved == list(getattr(fv, "AUDIO_FEATURE_KEYS", [])):
        return "all_audio"
    return f"custom_audio_{len(resolved)}groups"


def _summarize_feature_array(
    key: str,
    arr: np.ndarray,
    n_mfcc: int,
) -> np.ndarray:
    """Summarize a stored feature array to the fixed-length clustering vector."""

    if arr.size == 0:
        raise ValueError(f"{key} array is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{key} contains NaN or Inf values")

    expected_dims = {
        "mfcc": n_mfcc,
        "delta_mfcc": n_mfcc,
        "delta2_mfcc": n_mfcc,
        "spectral_centroid": 1,
        "spectral_rolloff": 1,
        "spectral_flux": 1,
        "spectral_flatness": 1,
        "zero_crossing_rate": 1,
        "chroma": fv.n_chroma,
    }

    if key == "beat_strength":
        flattened = arr.reshape(-1).astype(np.float32)
        if flattened.size != 4:
            raise ValueError(
                f"beat_strength must flatten to 4 scalars, got shape {arr.shape}"
            )
        return flattened

    if arr.ndim != 2:
        raise ValueError(f"{key} must be 2D, got shape {arr.shape}")
    if arr.shape[1] == 0:
        raise ValueError(f"{key} has zero frames")

    expected_first_dim = expected_dims.get(key)
    if expected_first_dim is not None and arr.shape[0] != expected_first_dim:
        raise ValueError(
            f"{key} expected first dimension {expected_first_dim}, got {arr.shape[0]}"
        )

    mean = arr.mean(axis=1)
    std = arr.std(axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def build_group_weights(
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    selected_audio_feature_keys: Optional[List[str]] = None,
    group_multipliers: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Create group weights with optional multipliers for adjusting feature importance.
    
    Args:
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands (kept for compatibility, not used)
        n_genres: Number of genre categories
        include_genre: Whether to include genre as a feature
        include_msd: Whether to include MSD metadata features (key, mode, loudness, tempo)
        group_multipliers: List of multipliers for each feature group.
                          If None, uses fv.feature_group_weights from config.
                          Order: [MFCC, ΔMFCC, ΔΔMFCC, SpectralCentroid, SpectralRolloff, 
                                  SpectralFlux, SpectralFlatness, ZCR, Chroma, BeatStrength,
                                  Key, Mode, Loudness, Tempo, Genre]
    
    Returns:
        Weight array matching the concatenated feature vector dimensions
    
    Feature dimensions (mean + std for time-varying features):
        - MFCC: 2 * n_mfcc (e.g., 26 for n_mfcc=13)
        - ΔMFCC: 2 * n_mfcc
        - ΔΔMFCC: 2 * n_mfcc
        - Spectral Centroid: 2 (1-dim feature, mean + std)
        - Spectral Rolloff: 2
        - Spectral Flux: 2
        - Spectral Flatness: 2
        - ZCR: 2
        - Chroma: 24 (12-dim feature, mean + std)
        - Beat Strength: 4 (tempo, mean_onset, std_onset, onset_rate - no mean/std)
        - Key: 12 (one-hot encoded)
        - Mode: 2 (one-hot encoded: major/minor)
        - Loudness: 1 (scalar)
        - Tempo (MSD): 1 (scalar)
        - Genre: n_genres (one-hot encoded)
    """
    group_sizes = [
        size
        for _, _, size in _get_full_group_specs(
            n_mfcc=n_mfcc,
            n_genres=n_genres,
            include_genre=include_genre,
            include_msd=include_msd,
            selected_audio_feature_keys=selected_audio_feature_keys,
        )
    ]
    
    # Base audio features
    group_sizes = [
        2 * n_mfcc,     # MFCC
        2 * n_mfcc,     # ΔMFCC
        2 * n_mfcc,     # ΔΔMFCC
        2,              # Spectral Centroid
        2,              # Spectral Rolloff
        2,              # Spectral Flux
        2,              # Spectral Flatness
        2,              # ZCR
        2 * n_chroma,   # Chroma (24 dims)
        4,              # Beat Strength (4 scalar values)
    ]
    
    # Add MSD features if enabled
    if include_msd:
        group_sizes.extend([
            12,             # Key (one-hot)
            2,              # Mode (one-hot)
            1,              # Loudness
            1,              # Tempo (MSD)
        ])
    
    # Add genre if enabled
    if include_genre:
        group_sizes.append(n_genres)  # Genre
    
    # Use global config if no multipliers provided
    if group_multipliers is None:
        group_multipliers = fv.feature_group_weights[:len(group_sizes)]
    
    if len(group_multipliers) != len(group_sizes):
        raise ValueError(
            f"group_multipliers length ({len(group_multipliers)}) must match "
            f"number of feature groups ({len(group_sizes)})"
        )
    
    total_dims = sum(group_sizes)
    weights = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for i, size in enumerate(group_sizes):
        # Apply multiplier and normalize by sqrt(size) to equalize within group
        weights[idx : idx + size] = group_multipliers[i] / np.sqrt(size)
        idx += size
    return weights


def equalize_features_pca(
    X_all: np.ndarray,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    pca_components: int = fv.pca_components_per_group,
) -> np.ndarray:
    """
    Equalize feature contributions by applying PCA to each feature group separately.
    
    Each group is standardized and transformed to EXACTLY the same number of dimensions,
    ensuring truly equal contribution regardless of original dimensionality.
    
    For groups with fewer dimensions than pca_components, we pad with zeros after
    standardization to ensure all groups contribute equally.
    
    Additionally, each group is normalized to have unit Frobenius norm (per sample),
    ensuring equal total variance contribution.
    
    Args:
        X_all: Raw feature matrix [n_samples, n_features]
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands (kept for compatibility, not used)
        n_genres: Number of genre categories
        include_genre: Whether genre features are included
        include_msd: Whether MSD features (key, mode, loudness, tempo) are included
        pca_components: Number of PCA components per group (target dimensions)
    
    Returns:
        Equalized feature matrix [n_samples, n_groups * pca_components]
    
    Feature groups (AUDIO):
        1. MFCC (2 * n_mfcc dims)
        2. ΔMFCC (2 * n_mfcc dims)
        3. ΔΔMFCC (2 * n_mfcc dims)
        4. Spectral Centroid (2 dims)
        5. Spectral Rolloff (2 dims)
        6. Spectral Flux (2 dims)
        7. Spectral Flatness (2 dims)
        8. ZCR (2 dims)
        9. Chroma (24 dims)
        10. Beat Strength (4 dims)
    
    Feature groups (MSD, optional):
        11. Key (12 dims one-hot)
        12. Mode (2 dims one-hot)
        13. Loudness (1 dim)
        14. Tempo MSD (1 dim)
    
    Feature groups (optional):
        15. Genre (n_genres dims)
    """
    n_chroma = 12
    
    # Define group boundaries - base audio features
    group_sizes = [
        2 * n_mfcc,     # MFCC
        2 * n_mfcc,     # ΔMFCC
        2 * n_mfcc,     # ΔΔMFCC
        2,              # Spectral Centroid
        2,              # Spectral Rolloff
        2,              # Spectral Flux
        2,              # Spectral Flatness
        2,              # ZCR
        2 * n_chroma,   # Chroma
        4,              # Beat Strength
    ]
    group_names = [
        "MFCC", "DeltaMFCC", "Delta2MFCC", "SpectralCentroid", "SpectralRolloff",
        "SpectralFlux", "SpectralFlatness", "ZCR", "Chroma", "BeatStrength"
    ]
    
    # Add MSD feature groups if enabled
    if include_msd:
        group_sizes.extend([12, 2, 1, 1])  # Key, Mode, Loudness, Tempo
        group_names.extend(["Key", "Mode", "Loudness", "MSD_Tempo"])
    
    # Add genre if enabled
    if include_genre:
        group_sizes.append(n_genres)
        group_names.append("Genre")
    
    # Validate input dimensions match expected
    expected_dims = sum(group_sizes)
    if X_all.shape[1] != expected_dims:
        raise ValueError(
            f"Feature dimension mismatch: got {X_all.shape[1]} features, "
            f"expected {expected_dims} (group_sizes={group_sizes}). "
            f"Check n_mfcc={n_mfcc}, n_genres={n_genres}, include_genre={include_genre}"
        )
    
    # Compute group start indices
    group_starts = [0]
    for size in group_sizes[:-1]:
        group_starts.append(group_starts[-1] + size)
    
    equalized_groups = []
    n_samples = X_all.shape[0]
    
    for i, (start, size, name) in enumerate(zip(group_starts, group_sizes, group_names)):
        end = start + size
        X_group = X_all[:, start:end]
        
        # Standardize the group
        scaler = StandardScaler()
        X_group_scaled = scaler.fit_transform(X_group)
        
        # Target: exactly pca_components dimensions for ALL groups
        if size > pca_components:
            # Reduce via PCA
            n_components = min(pca_components, n_samples - 1)  # PCA constraint
            pca = PCA(n_components=n_components, random_state=42)
            X_group_transformed = pca.fit_transform(X_group_scaled)
            variance_retained = sum(pca.explained_variance_ratio_) * 100
            
            # Pad if PCA gave fewer components than target (due to sample constraint)
            if X_group_transformed.shape[1] < pca_components:
                padding = np.zeros((n_samples, pca_components - X_group_transformed.shape[1]))
                X_group_transformed = np.hstack([X_group_transformed, padding])
            
            status = "OK" if variance_retained >= 80 else "WARN"
            print(f"  {status} {name}: {size} dims -> {pca_components} dims (PCA, {variance_retained:.1f}% variance)")
        elif size < pca_components:
            # Pad with zeros to reach target dimensions
            padding = np.zeros((n_samples, pca_components - size))
            X_group_transformed = np.hstack([X_group_scaled, padding])
            print(f"  {name}: {size} dims -> {pca_components} dims (padded with {pca_components - size} zeros)")
        else:
            # Exact match
            X_group_transformed = X_group_scaled
            print(f"  {name}: {size} dims -> {pca_components} dims (exact match)")
        
        # Normalize each group to have equal contribution (unit Frobenius norm per sample on average)
        # This ensures groups with different variance scales contribute equally
        group_norm = np.sqrt(np.mean(np.sum(X_group_transformed ** 2, axis=1)))
        if group_norm > 1e-10:
            X_group_transformed = X_group_transformed / group_norm
        
        equalized_groups.append(X_group_transformed)
    
    # Concatenate all equalized groups
    X_equalized = np.hstack(equalized_groups)
    print(f"  Total: {X_all.shape[1]} dims -> {X_equalized.shape[1]} dims ({len(group_names)} groups x {pca_components} dims)")
    
    return X_equalized.astype(np.float32)


def prepare_features(
    X_all: np.ndarray,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare features for clustering using the configured equalization method.
    
    Args:
        X_all: Raw feature matrix
        n_mfcc, n_mels, n_genres: Feature dimensions
        include_genre: Whether genre is included
        include_msd: Whether MSD features are included
        equalization_method: Override config method ("pca_per_group" or "weighted")
        pca_components: Override config PCA components per group
    
    Returns:
        Prepared feature matrix ready for clustering
    """
    method = equalization_method or getattr(fv, 'feature_equalization_method', 'weighted')
    pca_comp = pca_components or getattr(fv, 'pca_components_per_group', 5)
    
    print(f"Feature equalization method: {method}")
    
    if method == "pca_per_group":
        return equalize_features_pca(
            X_all, n_mfcc, n_mels, n_genres, include_genre, include_msd, pca_comp
        )
    if method in {"zscore", "raw_zscore", "zscore_only"}:
        scaler = StandardScaler()
        return scaler.fit_transform(X_all).astype(np.float32)

    # Legacy weighted method
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    weights = build_group_weights(n_mfcc, n_mels, n_genres, include_genre, include_msd)
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
        )
    return (X_scaled * weights).astype(np.float32)


def _clean_genre_mapping_for_audio(
    genre_map: Dict[str, str],
    audio_basenames: Optional[set],
    genre_map_path: str,
) -> Tuple[Dict[str, str], int]:
    """Remove cached genre-map entries that no longer have matching audio."""

    if not genre_map or audio_basenames is None:
        return genre_map, 0

    stale_keys = sorted(set(genre_map) - set(audio_basenames))
    if not stale_keys:
        return genre_map, 0

    cleaned = {base: genre for base, genre in genre_map.items() if base in audio_basenames}
    np.save(genre_map_path, cleaned)
    print(
        f"Removed {len(stale_keys)} stale genre-map entries from {genre_map_path}"
    )
    return cleaned, len(stale_keys)


def _load_genre_mapping(
    audio_dir: str,
    results_dir: str,
    include_genre: bool,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Load genre mapping from unified songs.csv or fallback to legacy methods.
    
    Genre metadata is always loaded when available so clustering outputs can still
    include reference genres for evaluation/reporting. The `include_genre` flag is
    kept for backward compatibility, but no longer controls whether the mapping
    itself is loaded.

    NOTE: When using songs.csv, songs have MULTIPLE genres (multi-label).
    For reporting/evaluation we use the PRIMARY genre (first listed) to maintain
    compatibility with single-label metrics.
    """
    genre_map_path = os.path.join(results_dir, "genre_map.npy")
    genre_list_path = os.path.join(results_dir, "genre_list.npy")
    audio_basenames = _load_audio_basenames(audio_dir)

    # Try loading from the taxonomy-aware songs CSV first, then fallback to legacy
    csv_path = str(resolve_pipeline_songs_csv(os.path.join("data", "songs.csv")))
    legacy_csv_path = os.path.join("data", "songs_data_with_genre.csv")
    
    active_csv = (
        csv_path
        if os.path.exists(csv_path)
        else (legacy_csv_path if os.path.exists(legacy_csv_path) else None)
    )
    
    if active_csv:
        print(f"Loading genre mapping from {active_csv}...")
        multi_label_mapping = genre_mapper.load_genre_mapping(active_csv)
        
        # Convert to single-label using PRIMARY genre (first one listed)
        # NOTE: This is intentional for clustering - we use the most prominent genre
        genre_map = {}
        for filename, genres_list in multi_label_mapping.items():
            # Remove .mp3 extension if present to match feature file basenames
            base = Path(filename).stem
            primary_genre = genre_mapper.get_primary_genre(filename, multi_label_mapping)
            genre_map[base] = primary_genre

        genre_map, _ = _clean_genre_mapping_for_audio(
            genre_map,
            audio_basenames,
            genre_map_path,
        )
        if genre_map:
            np.save(genre_map_path, genre_map)
            print(f"Loaded genre mapping for {len(genre_map)} songs from CSV")
    elif os.path.exists(genre_map_path):
        # Load existing cached mapping
        genre_map: Dict[str, str] = np.load(genre_map_path, allow_pickle=True).item()
        genre_map, _ = _clean_genre_mapping_for_audio(
            genre_map,
            audio_basenames,
            genre_map_path,
        )
        print(f"Loaded cached genre mapping for {len(genre_map)} files")
    else:
        # Fallback: Try to load from genre-based directory structure
        print("Attempting to load genre mapping from directory structure...")
        genre_map = {}
        genre_dirs = [d for d in glob.glob(os.path.join(audio_dir, "*")) if os.path.isdir(d)]
        
        if genre_dirs:
            for genre_dir in genre_dirs:
                genre = os.path.basename(genre_dir)
                wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
                for wav_path in wav_files:
                    base = Path(wav_path).stem
                    genre_map[base] = genre
            genre_map, _ = _clean_genre_mapping_for_audio(
                genre_map,
                audio_basenames,
                genre_map_path,
            )
            np.save(genre_map_path, genre_map)
            print(f"Created genre mapping for {len(genre_map)} files from directory structure")
        else:
            print(f"Warning: No genre directories found in {audio_dir} and no CSV mapping available")
            print("   Songs will be assigned 'unknown' genre for metadata only")

    unique_genres = sorted(set(genre_map.values())) if genre_map else ['unknown']
    np.save(genre_list_path, unique_genres)
    print(f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres[:10])}{'...' if len(unique_genres) > 10 else ''}")
    return genre_map, unique_genres


def _load_audio_basenames(audio_dir: Optional[str]) -> Optional[set]:
    """Return the basenames that currently exist in the audio library."""

    if not audio_dir or not os.path.isdir(audio_dir):
        return None

    basenames = set(audio_library_basenames(audio_dir))

    if not basenames:
        print(f"Warning: No audio files found in {audio_dir}; using all feature bundles.")
        return None

    print(f"Restricting clustering dataset to {len(basenames)} audio files from {audio_dir}")
    return basenames


def _collect_feature_vectors(
    audio_dir: Optional[str],
    results_dir: str,
    genre_map: Dict[str, str],
    unique_genres: List[str],
    include_genre: bool,
    include_msd: bool = True,
    songs_csv_path: Optional[str] = None,
) -> Tuple[List[str], List[np.ndarray], List[str]]:
    """Assemble feature vectors for each track.
    
    Loads the following features for each track:
    - MFCC, ΔMFCC, ΔΔMFCC (timbre features)
    - Spectral centroid, rolloff, flux, flatness (spectral features)
    - Zero Crossing Rate (ZCR)
    - Chroma (12-dim pitch class profile)
    - Beat strength / onset rate
    
    If include_msd=True, also adds MSD metadata features:
    - Key (12-dim one-hot encoded)
    - Mode (2-dim one-hot: major/minor)
    - Loudness (1-dim, normalized)
    - Tempo (1-dim, normalized)
    
    Each time-varying feature is summarized as mean + std.
    Beat strength features are single values (no mean/std needed).

    Genre metadata is attached to the returned rows when available, but is only
    appended to the feature vector when `include_genre=True`.
    """
    import pandas as pd
    
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    
    # Load MSD features from songs.csv if needed
    msd_features_map = {}
    if include_msd and songs_csv_path and os.path.exists(songs_csv_path):
        try:
            songs_df = pd.read_csv(songs_csv_path)
            # Create mapping from filename (without extension) to MSD features
            for _, row in songs_df.iterrows():
                if pd.notna(row.get('filename')) and pd.notna(row.get('key')):
                    # Strip .mp3 extension to match base names
                    base_name = str(row['filename']).replace('.mp3', '')
                    msd_features_map[base_name] = {
                        'key': int(row['key']) if pd.notna(row['key']) else 0,
                        'mode': int(row['mode']) if pd.notna(row['mode']) else 0,
                        'loudness': float(row['loudness']) if pd.notna(row['loudness']) else -10.0,
                        'tempo': float(row['tempo']) if pd.notna(row['tempo']) else 120.0,
                    }
            print(f"Loaded MSD features for {len(msd_features_map)} songs from {songs_csv_path}")
        except Exception as e:
            print(f"Warning: Could not load MSD features from {songs_csv_path}: {e}")
            include_msd = False
    elif include_msd:
        print("Warning: No songs_csv_path provided or file not found. MSD features will not be included.")
        include_msd = False

    feature_files = glob.glob(os.path.join(results_dir, "*_mfcc.npy"))
    base_names = sorted(
        os.path.basename(f).replace("_mfcc.npy", "")
        for f in feature_files
    )

    audio_basenames = _load_audio_basenames(audio_dir)
    if audio_basenames is not None:
        total_feature_bases = len(base_names)
        base_names = [base for base in base_names if base in audio_basenames]
        dropped_feature_bases = total_feature_bases - len(base_names)
        if dropped_feature_bases > 0:
            print(
                f"Skipped {dropped_feature_bases} stale feature bundles that do not "
                f"have a matching file in {audio_dir}"
            )

    file_names: List[str] = []
    feature_vectors: List[np.ndarray] = []
    genres: List[str] = []
    
    # Collect loudness and tempo values for normalization
    loudness_values = []
    tempo_values = []
    if include_msd:
        for base in base_names:
            if base in msd_features_map:
                loudness_values.append(msd_features_map[base]['loudness'])
                tempo_values.append(msd_features_map[base]['tempo'])
        
        # Compute normalization stats
        if loudness_values and tempo_values:
            loudness_mean, loudness_std = np.mean(loudness_values), np.std(loudness_values)
            tempo_mean, tempo_std = np.mean(tempo_values), np.std(tempo_values)
            if loudness_std < 1e-6:
                loudness_std = 1.0
            if tempo_std < 1e-6:
                tempo_std = 1.0
        else:
            loudness_mean, loudness_std = -10.0, 5.0
            tempo_mean, tempo_std = 120.0, 30.0
    
    # Define feature keys - must match what extract_features.py produces
    feature_keys = [
        "mfcc",
        "delta_mfcc",
        "delta2_mfcc",
        "spectral_centroid",
        "spectral_rolloff",
        "spectral_flux",
        "spectral_flatness",
        "zero_crossing_rate",
        "chroma",
        "beat_strength",
    ]

    for base in tqdm(base_names, desc="Loading features", unit="song"):
        feats = {
            key: os.path.join(results_dir, f"{base}_{key}.npy")
            for key in feature_keys
        }
        if not all(os.path.isfile(path) for path in feats.values()):
            continue

        if base in genre_map:
            genre = genre_map[base]
        else:
            parts = base.split(".")
            if parts and parts[0] in genre_to_idx:
                genre = parts[0]
            else:
                genre = "unknown"
                if genre not in genre_to_idx:
                    genre_to_idx[genre] = len(genre_to_idx)
                    unique_genres.append(genre)
        genres.append(genre)

        # Load and process audio features
        feature_parts = []
        for key in feature_keys:
            arr = np.load(feats[key])
            if key == "beat_strength":
                # Beat strength is [4, 1] - flatten it directly
                feature_parts.append(arr.flatten())
            else:
                # Time-varying features: compute mean and std across time axis
                feature_parts.append(np.concatenate([arr.mean(axis=1), arr.std(axis=1)]))
        
        vec = np.concatenate(feature_parts)
        
        # Add MSD features if enabled
        if include_msd:
            if base in msd_features_map:
                msd_data = msd_features_map[base]
                
                # Key: one-hot encode (12 dimensions, keys 0-11)
                key_vec = np.zeros(12, dtype=float)
                key_idx = msd_data['key'] % 12  # Ensure valid range
                key_vec[key_idx] = 1.0
                
                # Mode: one-hot encode (2 dimensions: 0=minor, 1=major)
                mode_vec = np.zeros(2, dtype=float)
                mode_idx = msd_data['mode'] % 2  # Ensure valid range
                mode_vec[mode_idx] = 1.0
                
                # Loudness: normalize
                loudness_normalized = (msd_data['loudness'] - loudness_mean) / loudness_std
                
                # Tempo: normalize
                tempo_normalized = (msd_data['tempo'] - tempo_mean) / tempo_std
                
                msd_vec = np.concatenate([
                    key_vec,           # 12 dims
                    mode_vec,          # 2 dims
                    [loudness_normalized],  # 1 dim
                    [tempo_normalized],     # 1 dim
                ])
            else:
                # Default MSD features for songs not in CSV
                msd_vec = np.zeros(16, dtype=float)  # 12 + 2 + 1 + 1
                msd_vec[0] = 1.0  # Default key = 0 (C)
                msd_vec[12] = 1.0  # Default mode = 0 (minor)
                # Loudness and tempo remain at 0 (mean after normalization)
            
            vec = np.concatenate([vec, msd_vec])

        if include_genre:
            genre_vec = np.zeros(len(unique_genres), dtype=float)
            genre_vec[genre_to_idx[genre]] = 1.0
            vec = np.concatenate([vec, genre_vec])

        file_names.append(base)
        feature_vectors.append(vec)

    return file_names, feature_vectors, genres


def load_clustering_dataset(
    audio_dir: str,
    results_dir: str,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """
    Build the shared feature matrix used by all clustering algorithms.

    Genre remains available as metadata for outputs/evaluation, but is only
    appended to the feature vector when `include_genre=True`.
    """
    if songs_csv_path is None:
        songs_csv_path = "data/songs.csv"

    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        audio_dir,
        results_dir,
        genre_map,
        unique_genres,
        include_genre,
        include_msd,
        songs_csv_path,
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors).astype(np.float32)
    X_prepared = prepare_features(
        X_all,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_genres=len(unique_genres),
        include_genre=include_genre,
        include_msd=include_msd,
    )
    return file_names, genres, unique_genres, X_prepared


def build_group_weights(
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    selected_audio_feature_keys: Optional[List[str]] = None,
    group_multipliers: Optional[List[float]] = None,
) -> np.ndarray:
    """Create per-dimension weights that respect the active feature subset."""

    del n_mels  # Kept for backward compatibility with older call sites.

    audio_specs = _get_audio_group_specs(n_mfcc, selected_audio_feature_keys)
    group_sizes = [size for _, _, size in audio_specs]
    if include_msd:
        group_sizes.extend([12, 2, 1, 1])
    if include_genre:
        group_sizes.append(n_genres)

    if group_multipliers is None:
        audio_weight_map = {
            key: fv.audio_feature_weights[idx]
            for idx, key in enumerate(fv.AUDIO_FEATURE_KEYS)
        }
        group_multipliers = [audio_weight_map[key] for key, _, _ in audio_specs]
        if include_msd:
            group_multipliers.extend(fv.msd_feature_weights)
        if include_genre:
            group_multipliers.append(fv.genre_weight)

    if len(group_multipliers) != len(group_sizes):
        raise ValueError(
            f"group_multipliers length ({len(group_multipliers)}) must match "
            f"number of feature groups ({len(group_sizes)})"
        )

    weights = np.ones(sum(group_sizes), dtype=np.float32)
    idx = 0
    for i, size in enumerate(group_sizes):
        weights[idx : idx + size] = group_multipliers[i] / np.sqrt(size)
        idx += size
    return weights


def equalize_features_pca(
    X_all: np.ndarray,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    pca_components: int = fv.pca_components_per_group,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """Equalize feature contributions with per-group scaling and PCA."""

    del n_mels  # Kept for backward compatibility with older call sites.

    group_specs = _get_full_group_specs(
        n_mfcc=n_mfcc,
        n_genres=n_genres,
        include_genre=include_genre,
        include_msd=include_msd,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )
    expected_dims = sum(size for _, _, size in group_specs)
    if X_all.shape[1] != expected_dims:
        raise ValueError(
            f"Feature dimension mismatch: got {X_all.shape[1]} features, "
            f"expected {expected_dims} for group specs {group_specs}"
        )

    group_starts = [0]
    for _, _, size in group_specs[:-1]:
        group_starts.append(group_starts[-1] + size)

    equalized_groups = []
    n_samples = X_all.shape[0]

    for start, (_, name, size) in zip(group_starts, group_specs):
        end = start + size
        X_group = X_all[:, start:end]

        scaler = StandardScaler()
        X_group_scaled = scaler.fit_transform(X_group)

        if size > pca_components:
            n_components = min(pca_components, n_samples - 1)
            pca = PCA(n_components=n_components, random_state=42)
            X_group_transformed = pca.fit_transform(X_group_scaled)
            variance_retained = sum(pca.explained_variance_ratio_) * 100
            if X_group_transformed.shape[1] < pca_components:
                padding = np.zeros(
                    (n_samples, pca_components - X_group_transformed.shape[1])
                )
                X_group_transformed = np.hstack([X_group_transformed, padding])
            status = "OK" if variance_retained >= 80 else "WARN"
            print(
                f"  {status} {name}: {size} dims -> {pca_components} dims "
                f"(PCA, {variance_retained:.1f}% variance)"
            )
        elif size < pca_components:
            padding = np.zeros((n_samples, pca_components - size))
            X_group_transformed = np.hstack([X_group_scaled, padding])
            print(
                f"  {name}: {size} dims -> {pca_components} dims "
                f"(padded with {pca_components - size} zeros)"
            )
        else:
            X_group_transformed = X_group_scaled
            print(f"  {name}: {size} dims -> {pca_components} dims (exact match)")

        group_norm = np.sqrt(np.mean(np.sum(X_group_transformed ** 2, axis=1)))
        if group_norm > 1e-10:
            X_group_transformed = X_group_transformed / group_norm

        equalized_groups.append(X_group_transformed)

    X_equalized = np.hstack(equalized_groups)
    print(
        f"  Total: {X_all.shape[1]} dims -> {X_equalized.shape[1]} dims "
        f"({len(group_specs)} groups x {pca_components} dims)"
    )
    return X_equalized.astype(np.float32)


def prepare_features(
    X_all: np.ndarray,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """Prepare the raw feature matrix for clustering."""

    method = equalization_method or getattr(fv, "feature_equalization_method", "weighted")
    pca_comp = pca_components or getattr(fv, "pca_components_per_group", 5)

    print(f"Feature equalization method: {method}")
    print(
        "Active clustering feature subset: "
        f"{_resolve_selected_audio_feature_keys(selected_audio_feature_keys)}"
    )

    if method == "pca_per_group":
        return equalize_features_pca(
            X_all,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_genres=n_genres,
            include_genre=include_genre,
            include_msd=include_msd,
            pca_components=pca_comp,
            selected_audio_feature_keys=selected_audio_feature_keys,
        )

    if method in {"zscore", "raw_zscore", "zscore_only"}:
        scaler = StandardScaler()
        return scaler.fit_transform(X_all).astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    weights = build_group_weights(
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_genres=n_genres,
        include_genre=include_genre,
        include_msd=include_msd,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
        )
    return (X_scaled * weights).astype(np.float32)


def _load_msd_feature_map(
    base_names: List[str],
    songs_csv_path: Optional[str],
) -> Tuple[Dict[str, Dict[str, float]], bool]:
    """Load normalized MSD feature source data if the unified CSV is available."""

    resolved_songs_csv = (
        str(resolve_pipeline_songs_csv(songs_csv_path))
        if songs_csv_path
        else None
    )
    if not resolved_songs_csv or not os.path.exists(resolved_songs_csv):
        return {}, False

    msd_features_map: Dict[str, Dict[str, float]] = {}
    try:
        songs_df = pd.read_csv(resolved_songs_csv)
        for _, row in songs_df.iterrows():
            if pd.notna(row.get("filename")) and pd.notna(row.get("key")):
                base_name = str(row["filename"]).replace(".mp3", "")
                msd_features_map[base_name] = {
                    "key": int(row["key"]) if pd.notna(row["key"]) else 0,
                    "mode": int(row["mode"]) if pd.notna(row["mode"]) else 0,
                    "loudness": float(row["loudness"]) if pd.notna(row["loudness"]) else -10.0,
                    "tempo": float(row["tempo"]) if pd.notna(row["tempo"]) else 120.0,
                }
    except Exception as exc:
        print(f"Warning: Could not load MSD features from {resolved_songs_csv}: {exc}")
        return {}, False

    covered = sum(1 for base in base_names if base in msd_features_map)
    print(f"Loaded MSD features for {covered} songs from {resolved_songs_csv}")
    return msd_features_map, True


def _write_dataset_qc_report(
    results_dir: str,
    qc_rows: List[Dict[str, Any]],
    qc_summary: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Persist the latest clustering-loader QC summary for debugging and reports."""

    metrics_dir = Path(results_dir).parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "clustering_dataset_qc_latest.csv"
    json_path = metrics_dir / "clustering_dataset_qc_summary_latest.json"

    columns = [
        "BaseName",
        "Status",
        "MissingKeys",
        "InvalidKeys",
        "PresentKeys",
        "IssueDetails",
    ]
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(qc_summary, handle, indent=2)

    return csv_path, json_path


def _collect_feature_vectors(
    audio_dir: Optional[str],
    results_dir: str,
    genre_map: Dict[str, str],
    unique_genres: List[str],
    include_genre: bool,
    n_mfcc: int = fv.n_mfcc,
    include_msd: bool = True,
    songs_csv_path: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> Tuple[List[str], List[np.ndarray], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Assemble summarized feature vectors for the active clustering subset."""

    selected_audio_feature_keys = _resolve_selected_audio_feature_keys(
        selected_audio_feature_keys
    )
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
    bundle_inventory = collect_feature_bundle_inventory(
        results_dir,
        EXTRACTED_FEATURE_KEYS,
    )

    audio_basenames = _load_audio_basenames(audio_dir)
    if audio_basenames is not None:
        candidate_bases = sorted(audio_basenames)
        stale_feature_bases = sorted(set(bundle_inventory) - set(audio_basenames))
    else:
        candidate_bases = sorted(bundle_inventory)
        stale_feature_bases = []

    qc_rows: List[Dict[str, Any]] = []
    if genre_map:
        eligible_bases = set(genre_map)
        excluded_bases = sorted(set(candidate_bases) - eligible_bases)
        if excluded_bases:
            for base in excluded_bases:
                qc_rows.append(
                    {
                        "BaseName": base,
                        "Status": "excluded_by_taxonomy",
                        "MissingKeys": "",
                        "InvalidKeys": "",
                        "PresentKeys": ",".join(bundle_inventory.get(base, [])),
                        "IssueDetails": json.dumps(
                            {
                                "excluded_by_taxonomy": (
                                    "missing taxonomy-backed primary genre"
                                )
                            },
                            sort_keys=True,
                        ),
                    }
                )
            print(
                f"Excluded {len(excluded_bases)} audio tracks without a taxonomy-backed "
                "primary genre."
            )
        candidate_bases = [base for base in candidate_bases if base in eligible_bases]

    for base in stale_feature_bases:
        qc_rows.append(
            {
                "BaseName": base,
                "Status": "stale_feature_bundle",
                "MissingKeys": "",
                "InvalidKeys": "",
                "PresentKeys": ",".join(bundle_inventory.get(base, [])),
                "IssueDetails": json.dumps(
                    {"stale_feature_bundle": "no_matching_audio_file"},
                    sort_keys=True,
                ),
            }
        )

    if stale_feature_bases:
        print(
            f"Found {len(stale_feature_bases)} stale feature bundles without a "
            f"matching file in {audio_dir}"
        )

    msd_features_map, msd_enabled = _load_msd_feature_map(
        candidate_bases,
        songs_csv_path,
    )
    include_msd = include_msd and msd_enabled
    if include_msd:
        loudness_values = [
            msd_features_map[base]["loudness"]
            for base in candidate_bases
            if base in msd_features_map
        ]
        tempo_values = [
            msd_features_map[base]["tempo"]
            for base in candidate_bases
            if base in msd_features_map
        ]
        if loudness_values and tempo_values:
            loudness_mean = float(np.mean(loudness_values))
            loudness_std = float(np.std(loudness_values)) or 1.0
            tempo_mean = float(np.mean(tempo_values))
            tempo_std = float(np.std(tempo_values)) or 1.0
        else:
            loudness_mean, loudness_std = -10.0, 5.0
            tempo_mean, tempo_std = 120.0, 30.0
    else:
        loudness_mean, loudness_std = -10.0, 5.0
        tempo_mean, tempo_std = 120.0, 30.0

    file_names: List[str] = []
    feature_vectors: List[np.ndarray] = []
    genres: List[str] = []
    dropped_incomplete = 0
    dropped_invalid = 0
    missing_all_feature_bundles = 0

    for base in tqdm(candidate_bases, desc="Loading features", unit="song"):
        arrays, validation = load_validated_feature_bundle(
            base,
            results_dir,
            feature_keys=selected_audio_feature_keys,
            n_mfcc=n_mfcc,
            n_chroma=fv.n_chroma,
        )
        if not validation.is_valid or arrays is None:
            qc_rows.append(validation.to_row())
            if validation.status == "incomplete":
                dropped_incomplete += 1
                if not validation.present_keys:
                    missing_all_feature_bundles += 1
            else:
                dropped_invalid += 1
            continue

        if base in genre_map:
            genre = genre_map[base]
        else:
            parts = base.split(".")
            if parts and parts[0] in genre_to_idx:
                genre = parts[0]
            else:
                genre = "unknown"
                if genre not in genre_to_idx:
                    genre_to_idx[genre] = len(genre_to_idx)
                    unique_genres.append(genre)

        feature_parts = [
            _summarize_feature_array(
                key,
                arrays[key],
                n_mfcc=n_mfcc,
            )
            for key in selected_audio_feature_keys
        ]

        vec = np.concatenate(feature_parts).astype(np.float32)

        if include_msd:
            if base in msd_features_map:
                msd_data = msd_features_map[base]
                key_vec = np.zeros(12, dtype=np.float32)
                key_vec[msd_data["key"] % 12] = 1.0

                mode_vec = np.zeros(2, dtype=np.float32)
                mode_vec[msd_data["mode"] % 2] = 1.0

                msd_vec = np.concatenate(
                    [
                        key_vec,
                        mode_vec,
                        np.array(
                            [(msd_data["loudness"] - loudness_mean) / loudness_std],
                            dtype=np.float32,
                        ),
                        np.array(
                            [(msd_data["tempo"] - tempo_mean) / tempo_std],
                            dtype=np.float32,
                        ),
                    ]
                )
            else:
                msd_vec = np.zeros(16, dtype=np.float32)
                msd_vec[0] = 1.0
                msd_vec[12] = 1.0
            vec = np.concatenate([vec, msd_vec]).astype(np.float32)

        if include_genre:
            genre_vec = np.zeros(len(unique_genres), dtype=np.float32)
            genre_vec[genre_to_idx[genre]] = 1.0
            vec = np.concatenate([vec, genre_vec]).astype(np.float32)

        file_names.append(base)
        feature_vectors.append(vec)
        genres.append(genre)

    qc_summary = {
        "candidate_audio_tracks": int(len(candidate_bases)),
        "excluded_by_taxonomy_tracks": int(len(excluded_bases) if genre_map else 0),
        "loaded_tracks": int(len(file_names)),
        "dropped_incomplete_tracks": int(dropped_incomplete),
        "dropped_invalid_tracks": int(dropped_invalid),
        "missing_all_feature_bundles": int(missing_all_feature_bundles),
        "stale_feature_bundles": int(len(stale_feature_bases)),
        "selected_audio_feature_keys": list(selected_audio_feature_keys),
    }

    if dropped_incomplete > 0:
        print(f"Dropped {dropped_incomplete} incomplete feature bundles.")
    if dropped_invalid > 0:
        print(f"Dropped {dropped_invalid} invalid feature bundles.")

    return file_names, feature_vectors, genres, qc_rows, qc_summary


def _expected_prepared_dimension(
    n_mfcc: int,
    n_genres: int,
    include_genre: bool,
    include_msd: bool,
    equalization_method: str,
    pca_components: int,
    selected_audio_feature_keys: Optional[List[str]] = None,
) -> int:
    """Return the expected prepared dimension for the active configuration."""

    group_specs = _get_full_group_specs(
        n_mfcc=n_mfcc,
        n_genres=n_genres,
        include_genre=include_genre,
        include_msd=include_msd,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )
    if equalization_method == "pca_per_group":
        return len(group_specs) * pca_components
    return sum(size for _, _, size in group_specs)


def load_clustering_dataset(
    audio_dir: str,
    results_dir: str,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], np.ndarray]:
    """Build the shared feature matrix used by all clustering algorithms."""

    dataset_bundle = load_clustering_dataset_bundle(
        audio_dir=audio_dir,
        results_dir=results_dir,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
        selected_audio_feature_keys=selected_audio_feature_keys,
        equalization_method=equalization_method,
        pca_components=pca_components,
    )
    return (
        dataset_bundle["file_names"],
        dataset_bundle["genres"],
        dataset_bundle["unique_genres"],
        dataset_bundle["prepared_features"],
    )


def load_clustering_dataset_bundle(
    audio_dir: str,
    results_dir: str,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the shared clustering dataset and return QC/reporting artifacts."""

    if songs_csv_path is None and include_msd:
        songs_csv_path = "data/songs.csv"
    if songs_csv_path:
        songs_csv_path = str(resolve_pipeline_songs_csv(songs_csv_path))

    selected_audio_feature_keys = _resolve_selected_audio_feature_keys(
        selected_audio_feature_keys
    )
    resolved_equalization_method = equalization_method or str(
        getattr(fv, "feature_equalization_method", "weighted")
    )
    resolved_pca_components = (
        int(pca_components)
        if pca_components is not None
        else int(getattr(fv, "pca_components_per_group", 5))
    )
    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    (
        file_names,
        feature_vectors,
        genres,
        qc_rows,
        qc_summary,
    ) = _collect_feature_vectors(
        audio_dir=audio_dir,
        results_dir=results_dir,
        genre_map=genre_map,
        unique_genres=unique_genres,
        n_mfcc=n_mfcc,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors).astype(np.float32)
    effective_include_msd = include_msd and bool(
        songs_csv_path and os.path.exists(songs_csv_path)
    )
    X_prepared = prepare_features(
        X_all,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_genres=len(unique_genres),
        include_genre=include_genre,
        include_msd=effective_include_msd,
        equalization_method=resolved_equalization_method,
        pca_components=resolved_pca_components,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )

    expected_dim = _expected_prepared_dimension(
        n_mfcc=n_mfcc,
        n_genres=len(unique_genres),
        include_genre=include_genre,
        include_msd=effective_include_msd,
        equalization_method=resolved_equalization_method,
        pca_components=resolved_pca_components,
        selected_audio_feature_keys=selected_audio_feature_keys,
    )
    if X_prepared.shape[1] != expected_dim:
        raise RuntimeError(
            f"Prepared feature dimension mismatch: got {X_prepared.shape[1]}, "
            f"expected {expected_dim}"
        )

    reported_pca_components = (
        int(resolved_pca_components)
        if resolved_equalization_method == "pca_per_group"
        else None
    )
    qc_summary = dict(qc_summary)
    qc_summary.update(
        {
            "raw_feature_dimension": int(X_all.shape[1]),
            "prepared_feature_dimension": int(X_prepared.shape[1]),
            "feature_subset_name": infer_feature_subset_name(
                selected_audio_feature_keys
            ),
            "equalization_method": str(resolved_equalization_method),
            "pca_components_per_group": reported_pca_components,
            "include_genre": bool(include_genre),
            "include_msd_requested": bool(include_msd),
            "include_msd_effective": bool(effective_include_msd),
            "unique_genre_count": int(len(unique_genres)),
            "audio_dir": str(audio_dir),
            "results_dir": str(results_dir),
        }
    )
    qc_csv_path, qc_json_path = _write_dataset_qc_report(results_dir, qc_rows, qc_summary)
    metadata_csv_path = songs_csv_path or str(resolve_pipeline_songs_csv("data/songs.csv"))
    metadata_frame = build_audio_metadata_frame(
        file_names,
        songs_csv_path=metadata_csv_path,
    )

    print(
        f"Prepared clustering dataset: {len(file_names)} tracks, "
        f"raw dims={X_all.shape[1]}, prepared dims={X_prepared.shape[1]}"
    )
    print(f"Dataset QC summary written to: {qc_csv_path} and {qc_json_path}")

    return {
        "file_names": file_names,
        "genres": genres,
        "unique_genres": unique_genres,
        "raw_features": X_all,
        "prepared_features": X_prepared,
        "metadata_frame": metadata_frame,
        "qc_rows": qc_rows,
        "qc_summary": qc_summary,
        "qc_csv_path": str(qc_csv_path),
        "qc_json_path": str(qc_json_path),
    }


def get_retrieval_artifact_path(
    method_id: str,
    output_dir: str = "output/clustering_results",
) -> Path:
    """Return the saved prepared-space retrieval artifact path for a method."""

    normalized = method_id.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("method_id must not be empty")
    return Path(output_dir) / f"audio_clustering_artifact_{normalized}.npz"


def snapshot_dataset_qc_artifacts(
    qc_csv_path: str,
    qc_json_path: str,
    metrics_dir: str,
) -> Tuple[str, str]:
    """Copy the latest QC outputs into a run-scoped metrics directory."""

    metrics_dir_path = Path(metrics_dir)
    metrics_dir_path.mkdir(parents=True, exist_ok=True)
    destination_csv = metrics_dir_path / "clustering_dataset_qc.csv"
    destination_json = metrics_dir_path / "clustering_dataset_qc_summary.json"
    shutil.copy2(qc_csv_path, destination_csv)
    shutil.copy2(qc_json_path, destination_json)
    return str(destination_csv), str(destination_json)


def save_retrieval_artifact(
    method_id: str,
    file_names: List[str],
    prepared_features: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    output_dir: str = "output/clustering_results",
    artists: Optional[np.ndarray] = None,
    titles: Optional[np.ndarray] = None,
    filenames: Optional[np.ndarray] = None,
    msd_track_ids: Optional[np.ndarray] = None,
    assignment_confidence: Optional[np.ndarray] = None,
    posterior_probabilities: Optional[np.ndarray] = None,
    distance_to_cluster: Optional[np.ndarray] = None,
    log_likelihood: Optional[np.ndarray] = None,
    feature_subset_name: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
    feature_equalization_method: Optional[str] = None,
    pca_components_per_group: Optional[int] = None,
    raw_feature_dimension: Optional[int] = None,
    prepared_feature_dimension: Optional[int] = None,
    profile_id: Optional[str] = None,
) -> Path:
    """Persist the true retrieval space used by the UI and offline evaluation."""

    n_samples = len(file_names)
    prepared_features = np.asarray(prepared_features, dtype=np.float32)
    labels = np.asarray(labels)
    coords = np.asarray(coords, dtype=np.float32)

    if prepared_features.ndim != 2:
        raise ValueError(
            f"prepared_features must be 2D, got shape {prepared_features.shape}"
        )
    if prepared_features.shape[0] != n_samples:
        raise ValueError(
            "prepared_features row count must match file_names length: "
            f"{prepared_features.shape[0]} != {n_samples}"
        )
    if labels.shape[0] != n_samples:
        raise ValueError(
            f"labels length must match file_names length: {labels.shape[0]} != {n_samples}"
        )
    if coords.shape != (n_samples, 2):
        raise ValueError(
            f"coords must have shape ({n_samples}, 2), got {coords.shape}"
        )

    payload: Dict[str, np.ndarray] = {
        "artifact_version": np.array([2], dtype=np.int32),
        "method_id": np.array([method_id]),
        "songs": np.asarray(file_names),
        "prepared_features": prepared_features,
        "labels": labels,
        "coords": coords,
        "feature_subset_name": np.array(
            [feature_subset_name or infer_feature_subset_name(selected_audio_feature_keys)]
        ),
        "feature_equalization_method": np.array(
            [feature_equalization_method or fv.feature_equalization_method]
        ),
        "pca_components_per_group": np.array(
            [
                -1
                if pca_components_per_group is None
                else int(pca_components_per_group)
            ],
            dtype=np.int32,
        ),
        "raw_feature_dimension": np.array(
            [
                -1
                if raw_feature_dimension is None
                else int(raw_feature_dimension)
            ],
            dtype=np.int32,
        ),
        "prepared_feature_dimension": np.array(
            [
                int(prepared_feature_dimension)
                if prepared_feature_dimension is not None
                else int(prepared_features.shape[1])
            ],
            dtype=np.int32,
        ),
        "profile_id": np.array([profile_id or "unspecified"]),
    }

    if selected_audio_feature_keys is not None:
        payload["selected_audio_feature_keys"] = np.asarray(
            _resolve_selected_audio_feature_keys(selected_audio_feature_keys),
            dtype=np.str_,
        )

    optional_metadata_vectors = {
        "artists": artists,
        "titles": titles,
        "filenames": filenames,
        "msd_track_ids": msd_track_ids,
    }
    for key, values in optional_metadata_vectors.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=np.str_)
        if values.shape[0] != n_samples:
            raise ValueError(
                f"{key} length must match file_names length: {values.shape[0]} != {n_samples}"
            )
        payload[key] = values

    optional_vectors = {
        "assignment_confidence": assignment_confidence,
        "distance_to_cluster": distance_to_cluster,
        "log_likelihood": log_likelihood,
    }
    for key, values in optional_vectors.items():
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != n_samples:
            raise ValueError(
                f"{key} length must match file_names length: {values.shape[0]} != {n_samples}"
            )
        payload[key] = values

    if posterior_probabilities is not None:
        posterior_probabilities = np.asarray(posterior_probabilities, dtype=np.float32)
        if posterior_probabilities.ndim != 2:
            raise ValueError(
                "posterior_probabilities must be 2D, got shape "
                f"{posterior_probabilities.shape}"
            )
        if posterior_probabilities.shape[0] != n_samples:
            raise ValueError(
                "posterior_probabilities row count must match file_names length: "
                f"{posterior_probabilities.shape[0]} != {n_samples}"
            )
        payload["posterior_probabilities"] = posterior_probabilities

    artifact_path = get_retrieval_artifact_path(method_id, output_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **payload)
    print(f"Saved retrieval artifact -> {artifact_path}")
    return artifact_path


def load_retrieval_artifact(
    method_id: str,
    output_dir: str = "output/clustering_results",
) -> Dict[str, np.ndarray]:
    """Load the saved prepared-space retrieval artifact for a method."""

    artifact_path = get_retrieval_artifact_path(method_id, output_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Retrieval artifact not found: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as artifact:
        return {key: artifact[key] for key in artifact.files}


def compute_visualization_coords(X_prepared: np.ndarray) -> np.ndarray:
    """Project the shared prepared feature space to 2D for comparable plots."""
    return PCA(n_components=2, random_state=42).fit_transform(X_prepared)


def compute_cluster_range(n_samples: int, n_genres: int = 0) -> Tuple[int, int]:
    """
    Compute a sensible cluster search range based on data characteristics.
    
    Uses the rule of thumb: k_max ~ sqrt(n/2), with adjustments for genre count.
    
    Args:
        n_samples: Number of data points
        n_genres: Number of unique genres (0 if unknown or not using genre)
    
    Returns:
        Tuple of (k_min, k_max)
    """
    k_min = 2
    
    # Rule of thumb: sqrt(n/2)
    k_max_data = int(np.sqrt(n_samples / 2))
    
    # Adjust based on genre count if available
    if n_genres > 0:
        # Allow up to 1.5x the genre count, but cap reasonably
        k_max_genre = int(n_genres * 1.5)
        k_max = min(k_max_data, k_max_genre, 50)
    else:
        k_max = min(k_max_data, 30)
    
    # Ensure sensible bounds
    k_max = max(k_max, k_min + 1)
    
    return k_min, k_max


def _select_optimal_k_silhouette(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    n_jobs: int = 1,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[float], List[float]]:
    """
    Pick the number of clusters for K-Means using silhouette score.

    This matches the geometry that K-Means actually optimises: compact,
    well-separated partitions in the prepared feature space.
    Uses parallel processing across candidate k values.
    
    Args:
        X: Feature matrix
        k_min: Minimum number of clusters
        k_max: Maximum number of clusters  
        n_jobs: Number of parallel jobs (-1 = all cores)
    """
    k_max = min(k_max, max(k_min, X.shape[0] - 1))
    sample_size = min(5000, X.shape[0])

    def evaluate_k(k: int) -> Tuple[int, float, float, bool]:
        """Evaluate silhouette score and inertia for a single k value."""
        try:
            estimator = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=20,
            )
            labels = estimator.fit_predict(X)
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2 or len(unique_labels) >= len(X):
                return k, float("-inf"), float("inf"), False

            silhouette_kwargs = {"random_state": 42}
            if sample_size < len(X):
                silhouette_kwargs["sample_size"] = sample_size
            score = silhouette_score(X, labels, **silhouette_kwargs)
            return k, score, estimator.inertia_, True
        except Exception:
            return k, float("-inf"), float("inf"), False

    print(f"Searching for optimal k using silhouette ({k_min}-{k_max})...")

    k_values = list(range(k_min, k_max + 1))
    if n_jobs == 1:
        results = [evaluate_k(k) for k in k_values]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
            delayed(evaluate_k)(k) for k in k_values
        )

    best_k = k_min
    best_score = float("-inf")
    silhouette_scores: List[float] = []
    inertia_scores: List[float] = []

    for k, score, inertia, converged in sorted(results, key=lambda x: x[0]):
        silhouette_scores.append(score)
        inertia_scores.append(inertia)
        if converged and score > best_score:
            best_score = score
            best_k = k

    selected_score = best_score

    if best_score > float("-inf"):
        candidate_rows = [
            (k, score, inertia)
            for k, score, inertia, converged in sorted(results, key=lambda x: x[0])
            if converged and score >= best_score - 0.01
        ]
        if candidate_rows:
            best_k, selected_score, _ = min(
                candidate_rows,
                key=lambda row: (row[0], row[2]),
            )

    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None

    if best_score > float("-inf"):
        print(f"Optimal k (silhouette) -> {best_k} (score={selected_score:.4f})")
        estimator = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        best_labels = estimator.fit_predict(X)
        best_centers = estimator.cluster_centers_
    else:
        print("Silhouette-based search failed, using fallback")

    return best_k, best_labels, best_centers, silhouette_scores, inertia_scores


def run_kmeans_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/features",
    output_dir: str = "output/clustering_results",
    metrics_dir: str = "output/metrics",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = True,
    dynamic_k_min: Optional[int] = None,
    dynamic_k_max: Optional[int] = None,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
    include_msd: bool = fv.include_msd_features,
    songs_csv_path: Optional[str] = None,
    selected_audio_feature_keys: Optional[List[str]] = None,
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
    profile_id: Optional[str] = None,
):
    os.makedirs(results_dir, exist_ok=True)
    requested_n_clusters = int(n_clusters)

    dataset_bundle = load_clustering_dataset_bundle(
        audio_dir=audio_dir,
        results_dir=results_dir,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        include_genre=include_genre,
        include_msd=include_msd,
        songs_csv_path=songs_csv_path,
        selected_audio_feature_keys=selected_audio_feature_keys,
        equalization_method=equalization_method,
        pca_components=pca_components,
    )
    file_names = dataset_bundle["file_names"]
    genres = dataset_bundle["genres"]
    unique_genres = dataset_bundle["unique_genres"]
    X_prepared = dataset_bundle["prepared_features"]
    metadata_frame = dataset_bundle["metadata_frame"]

    labels: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = None
    silhouette_scores: Optional[List[float]] = None
    inertia_scores: Optional[List[float]] = None
    resolved_search_k_min: Optional[int] = None
    resolved_search_k_max: Optional[int] = None

    if dynamic_cluster_selection:
        n_samples = X_prepared.shape[0]
        
        # Compute data-driven cluster range if not explicitly provided
        genre_count_hint = len(unique_genres) if include_genre else 0
        auto_k_min, auto_k_max = compute_cluster_range(n_samples, genre_count_hint)
        k_min = dynamic_k_min if dynamic_k_min is not None else auto_k_min
        k_max = dynamic_k_max if dynamic_k_max is not None else auto_k_max
        resolved_search_k_min = int(k_min)
        resolved_search_k_max = int(k_max)
        
        print(f"Dynamic cluster selection: searching k in [{k_min}, {k_max}]")
        print(f"  (Based on {n_samples} samples)")

        best_k, best_labels, best_centers, silhouette_scores, inertia_scores = _select_optimal_k_silhouette(
            X_prepared, k_min, k_max
        )
        n_clusters = best_k
        if best_labels is not None and best_centers is not None:
            labels = best_labels
            centers = best_centers

    if labels is None or centers is None:
        estimator = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = estimator.fit_predict(X_prepared)
        centers = estimator.cluster_centers_

    coords = compute_visualization_coords(X_prepared)
    distances = np.linalg.norm(X_prepared - centers[labels], axis=1)

    df = pd.DataFrame(
        {
            "Song": file_names,
            "Artist": metadata_frame["Artist"].astype(str).to_numpy(),
            "Title": metadata_frame["Title"].astype(str).to_numpy(),
            "Filename": metadata_frame["Filename"].astype(str).to_numpy(),
            "MSDTrackID": metadata_frame["MSDTrackID"].astype(str).to_numpy(),
            "GenreList": metadata_frame["GenreList"].astype(str).to_numpy(),
            "PrimaryGenres": metadata_frame["PrimaryGenres"].astype(str).to_numpy(),
            "SecondaryTags": metadata_frame["SecondaryTags"].astype(str).to_numpy(),
            "AllGenreTags": metadata_frame["AllGenreTags"].astype(str).to_numpy(),
            "OriginalGenreList": metadata_frame["OriginalGenreList"].astype(str).to_numpy(),
            "OriginalPrimaryGenre": metadata_frame["OriginalPrimaryGenre"].astype(str).to_numpy(),
            "Genre": genres,
            "Cluster": labels,
            "Distance": distances,
            "PCA1": coords[:, 0],
            "PCA2": coords[:, 1],
        }
    )

    output_dir_path = Path(output_dir)
    metrics_dir_path = Path(metrics_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    metrics_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save selection diagnostics if available
    if silhouette_scores is not None and inertia_scores is not None:
        k_min = dynamic_k_min if dynamic_k_min is not None else auto_k_min
        selection_df = pd.DataFrame({
            "K": list(range(k_min, k_min + len(silhouette_scores))),
            "Silhouette": silhouette_scores,
            "Inertia": inertia_scores,
        })
        selection_path = metrics_dir_path / "kmeans_selection_criteria.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"Stored silhouette diagnostics -> {selection_path}")
    else:
        selection_path = None
    
    csv_path = output_dir_path / "audio_clustering_results_kmeans.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    run_qc_csv_path, run_qc_json_path = snapshot_dataset_qc_artifacts(
        dataset_bundle["qc_csv_path"],
        dataset_bundle["qc_json_path"],
        str(metrics_dir_path),
    )

    save_retrieval_artifact(
        method_id="kmeans",
        file_names=file_names,
        prepared_features=X_prepared,
        labels=labels,
        coords=coords,
        output_dir=str(output_dir_path),
        artists=metadata_frame["Artist"].astype(str).to_numpy(),
        titles=metadata_frame["Title"].astype(str).to_numpy(),
        filenames=metadata_frame["Filename"].astype(str).to_numpy(),
        msd_track_ids=metadata_frame["MSDTrackID"].astype(str).to_numpy(),
        distance_to_cluster=distances,
        feature_subset_name=dataset_bundle["qc_summary"].get("feature_subset_name"),
        selected_audio_feature_keys=dataset_bundle["qc_summary"].get(
            "selected_audio_feature_keys"
        ),
        feature_equalization_method=dataset_bundle["qc_summary"].get(
            "equalization_method"
        ),
        pca_components_per_group=dataset_bundle["qc_summary"].get(
            "pca_components_per_group"
        ),
        raw_feature_dimension=dataset_bundle["qc_summary"].get("raw_feature_dimension"),
        prepared_feature_dimension=dataset_bundle["qc_summary"].get(
            "prepared_feature_dimension"
        ),
        profile_id=profile_id,
    )

    method_summary = {
        "method_id": "kmeans",
        "profile_id": profile_id or "unspecified",
        "representation": {
            "feature_subset_name": dataset_bundle["qc_summary"].get("feature_subset_name"),
            "selected_audio_feature_keys": dataset_bundle["qc_summary"].get(
                "selected_audio_feature_keys"
            ),
            "equalization_method": dataset_bundle["qc_summary"].get(
                "equalization_method"
            ),
            "pca_components_per_group": dataset_bundle["qc_summary"].get(
                "pca_components_per_group"
            ),
            "raw_feature_dimension": dataset_bundle["qc_summary"].get(
                "raw_feature_dimension"
            ),
            "prepared_feature_dimension": dataset_bundle["qc_summary"].get(
                "prepared_feature_dimension"
            ),
            "include_genre": bool(include_genre),
            "include_msd_requested": bool(include_msd),
            "include_msd_effective": dataset_bundle["qc_summary"].get(
                "include_msd_effective"
            ),
        },
        "hyperparameters": {
            "dynamic_cluster_selection": bool(dynamic_cluster_selection),
            "requested_n_clusters": requested_n_clusters,
            "dynamic_k_min": resolved_search_k_min,
            "dynamic_k_max": resolved_search_k_max,
            "random_state": 42,
            "n_init": 10,
            "selected_n_clusters": int(len(np.unique(labels))),
        },
        "outputs": {
            "results_csv": str(csv_path),
            "retrieval_artifact": str(
                get_retrieval_artifact_path("kmeans", output_dir=str(output_dir_path))
            ),
            "selection_criteria_csv": (
                str(selection_path) if selection_path is not None else None
            ),
            "dataset_qc_summary_json": run_qc_json_path,
            "dataset_qc_csv": run_qc_csv_path,
        },
    }
    summary_path = metrics_dir_path / "kmeans_run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(method_summary, handle, indent=2)
    print(f"Stored K-Means run summary -> {summary_path}")

    return df, coords, labels


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run K-Means clustering and optionally launch the interactive UI."
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive clustering UI after clustering finishes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    DF, COORDS, LABELS = run_kmeans_clustering(
        audio_dir="audio_files",
        results_dir="output/features",
        n_clusters=5,
        dynamic_cluster_selection=True,
        include_genre=fv.include_genre,
    )

    if args.ui:
        from src.ui.modern_ui import launch_ui

        launch_ui(
            DF,
            COORDS,
            LABELS,
            audio_dir="audio_files",
            clustering_method="K-means",
            retrieval_method_id="kmeans",
        )
    else:
        print(
            "Skipping interactive clustering UI. Run 'python src/ui/modern_ui.py' "
            "to open the latest benchmark-linked UI snapshot."
        )
