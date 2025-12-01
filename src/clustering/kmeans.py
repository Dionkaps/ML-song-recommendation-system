import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv
from src.ui.modern_ui import launch_ui
from src.utils import genre_mapper


def build_group_weights(
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
    group_multipliers: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Create group weights with optional multipliers for adjusting feature importance.
    
    Args:
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands
        n_genres: Number of genre categories
        include_genre: Whether to include genre as a feature
        group_multipliers: List of multipliers for each feature group.
                          If None, uses fv.feature_group_weights from config.
                          Order: [MFCC, MelSpec, SpectralCentroid, SpectralFlatness, ZCR, Genre]
    
    Returns:
        Weight array matching the concatenated feature vector dimensions
    """
    if include_genre:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 2, n_genres]
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 2]
    
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
        n_mels: Number of mel bands
        n_genres: Number of genre categories
        include_genre: Whether genre features are included
        pca_components: Number of PCA components per group (target dimensions)
    
    Returns:
        Equalized feature matrix [n_samples, n_groups * pca_components]
    """
    # Define group boundaries
    if include_genre:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 2, n_genres]
        group_names = ["MFCC", "MelSpec", "SpectralCentroid", "SpectralFlatness", "ZCR", "Genre"]
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 2]
        group_names = ["MFCC", "MelSpec", "SpectralCentroid", "SpectralFlatness", "ZCR"]
    
    # Validate input dimensions match expected
    expected_dims = sum(group_sizes)
    if X_all.shape[1] != expected_dims:
        raise ValueError(
            f"Feature dimension mismatch: got {X_all.shape[1]} features, "
            f"expected {expected_dims} (group_sizes={group_sizes}). "
            f"Check n_mfcc={n_mfcc}, n_mels={n_mels}, n_genres={n_genres}, include_genre={include_genre}"
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
            
            status = "\u2713" if variance_retained >= 80 else "\u26a0"
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
    equalization_method: Optional[str] = None,
    pca_components: Optional[int] = None,
) -> np.ndarray:
    """
    Prepare features for clustering using the configured equalization method.
    
    Args:
        X_all: Raw feature matrix
        n_mfcc, n_mels, n_genres: Feature dimensions
        include_genre: Whether genre is included
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
            X_all, n_mfcc, n_mels, n_genres, include_genre, pca_comp
        )
    else:
        # Legacy weighted method
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        weights = build_group_weights(n_mfcc, n_mels, n_genres, include_genre)
        if X_scaled.shape[1] != len(weights):
            raise ValueError(
                f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
            )
        return (X_scaled * weights).astype(np.float32)


def _load_genre_mapping(
    audio_dir: str,
    results_dir: str,
    include_genre: bool,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Load genre mapping from songs_data_with_genre.csv or fallback to directory structure.
    
    NOTE: When using songs_data_with_genre.csv, songs have MULTIPLE genres (multi-label).
    For clustering purposes, we use the PRIMARY genre (first listed) to maintain
    compatibility with single-label clustering algorithms.
    """
    genre_map_path = os.path.join(results_dir, "genre_map.npy")
    genre_list_path = os.path.join(results_dir, "genre_list.npy")

    # Try loading from songs_data_with_genre.csv first
    csv_path = os.path.join("data", "songs_data_with_genre.csv")
    if os.path.exists(csv_path) and include_genre:
        print(f"Loading genre mapping from {csv_path}...")
        multi_label_mapping = genre_mapper.load_genre_mapping(csv_path)
        
        # Convert to single-label using PRIMARY genre (first one listed)
        # NOTE: This is intentional for clustering - we use the most prominent genre
        genre_map = {}
        for filename, genres_list in multi_label_mapping.items():
            # Remove .mp3 extension if present to match feature file basenames
            base = Path(filename).stem
            primary_genre = genre_mapper.get_primary_genre(filename, multi_label_mapping)
            genre_map[base] = primary_genre
        
        if genre_map:
            np.save(genre_map_path, genre_map)
            print(f"✓ Loaded genre mapping for {len(genre_map)} songs from CSV")
    elif os.path.exists(genre_map_path) and include_genre:
        # Load existing cached mapping
        genre_map: Dict[str, str] = np.load(genre_map_path, allow_pickle=True).item()
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
            if include_genre:
                np.save(genre_map_path, genre_map)
            print(f"Created genre mapping for {len(genre_map)} files from directory structure")
        else:
            print(f"⚠️ Warning: No genre directories found in {audio_dir} and no CSV mapping available")
            print("   Songs will be assigned 'unknown' genre")

    unique_genres = sorted(set(genre_map.values())) if genre_map else ['unknown']
    np.save(genre_list_path, unique_genres)
    print(f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres[:10])}{'...' if len(unique_genres) > 10 else ''}")
    return genre_map, unique_genres


def _collect_feature_vectors(
    results_dir: str,
    genre_map: Dict[str, str],
    unique_genres: List[str],
    include_genre: bool,
) -> Tuple[List[str], List[np.ndarray], List[str]]:
    """Assemble feature vectors for each track."""
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}

    feature_files = glob.glob(os.path.join(results_dir, "*_mfcc.npy"))
    base_names = [os.path.basename(f).replace("_mfcc.npy", "") for f in feature_files]

    file_names: List[str] = []
    feature_vectors: List[np.ndarray] = []
    genres: List[str] = []

    for base in base_names:
        feats = {
            key: os.path.join(results_dir, f"{base}_{key}.npy")
            for key in ["mfcc", "melspectrogram", "spectral_centroid", "spectral_flatness", "zero_crossing_rate"]
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
                print(f"Warning: Could not determine genre for {base}, skipping")
                continue
        genres.append(genre)

        arrays = [np.load(path) for path in feats.values()]
        vec = np.concatenate([
            np.concatenate([arr.mean(axis=1), arr.std(axis=1)])
            for arr in arrays
        ])

        if include_genre:
            genre_vec = np.zeros(len(unique_genres), dtype=float)
            genre_vec[genre_to_idx[genre]] = 1.0
            vec = np.concatenate([vec, genre_vec])

        file_names.append(base)
        feature_vectors.append(vec)

    return file_names, feature_vectors, genres


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


def _select_optimal_k_bic(
    X: np.ndarray,
    k_min: int,
    k_max: int,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], List[float]]:
    """
    Pick the number of clusters using BIC (consistent with GMM/VaDE).
    
    Uses a GMM with spherical covariance to approximate K-Means BIC.
    This provides consistency with GMM's selection criterion.
    """
    from sklearn.mixture import GaussianMixture
    
    best_k = k_min
    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None
    best_bic = float('inf')
    bic_scores: List[float] = []
    
    print(f"Searching for optimal k using BIC ({k_min}-{k_max})...")
    
    for k in range(k_min, k_max + 1):
        try:
            # Use GMM with spherical covariance (equivalent to K-Means assumption)
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='spherical',
                n_init=3,
                max_iter=100,
                random_state=42,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores.append(bic)
            
            if not gmm.converged_:
                print(f"  K={k}: BIC={bic:.2f} (did not converge)")
                continue
            
            print(f"  K={k}: BIC={bic:.2f}")
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
                # Get K-Means solution for this k
                estimator = KMeans(n_clusters=k, random_state=42, n_init=10)
                best_labels = estimator.fit_predict(X)
                best_centers = estimator.cluster_centers_
                
        except Exception as e:
            print(f"  K={k}: Failed ({str(e)[:50]}...)")
            bic_scores.append(float('inf'))
            continue
    
    if best_labels is not None:
        print(f"Optimal k (BIC) -> {best_k} (BIC={best_bic:.2f})")
    else:
        print("BIC-based search failed, using fallback")
    
    return best_k, best_labels, best_centers, bic_scores


def run_kmeans_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "output/results",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = True,
    dynamic_k_min: Optional[int] = None,
    dynamic_k_max: Optional[int] = None,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = fv.include_genre,
):
    os.makedirs(results_dir, exist_ok=True)

    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        results_dir, genre_map, unique_genres, include_genre
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors)
    
    # Use the unified feature preparation (PCA per group or weighted)
    X_prepared = prepare_features(
        X_all,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_genres=len(unique_genres),
        include_genre=include_genre,
    )

    labels: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = None
    bic_scores: Optional[List[float]] = None

    if dynamic_cluster_selection:
        n_samples = X_prepared.shape[0]
        
        # Compute data-driven cluster range if not explicitly provided
        auto_k_min, auto_k_max = compute_cluster_range(n_samples, len(unique_genres))
        k_min = dynamic_k_min if dynamic_k_min is not None else auto_k_min
        k_max = dynamic_k_max if dynamic_k_max is not None else auto_k_max
        
        print(f"Dynamic cluster selection: searching k in [{k_min}, {k_max}]")
        print(f"  (Based on {n_samples} samples and {len(unique_genres)} genres)")

        best_k, best_labels, best_centers, bic_scores = _select_optimal_k_bic(
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

    coords = PCA(n_components=2, random_state=42).fit_transform(X_prepared)
    distances = np.linalg.norm(X_prepared - centers[labels], axis=1)

    df = pd.DataFrame(
        {
            "Song": file_names,
            "Genre": genres,
            "Cluster": labels,
            "Distance": distances,
            "PCA1": coords[:, 0],
            "PCA2": coords[:, 1],
        }
    )

    output_dir = Path("output/clustering_results")
    metrics_dir = Path("output/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save BIC scores if available
    if bic_scores is not None:
        k_min = dynamic_k_min if dynamic_k_min is not None else 2
        selection_df = pd.DataFrame({
            "K": list(range(k_min, k_min + len(bic_scores))),
            "BIC": bic_scores,
        })
        selection_path = metrics_dir / "kmeans_selection_criteria.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"Stored BIC diagnostics -> {selection_path}")
    
    csv_path = output_dir / "audio_clustering_results_kmeans.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    return df, coords, labels


if __name__ == "__main__":
    DF, COORDS, LABELS = run_kmeans_clustering(
        audio_dir="audio_files",
        results_dir="output/features",
        n_clusters=5,
        dynamic_cluster_selection=True,
        include_genre=fv.include_genre,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="audio_files", clustering_method="K-means")
