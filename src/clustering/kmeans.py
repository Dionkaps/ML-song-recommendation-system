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


def build_group_weights(
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    n_genres: int = fv.n_genres,
    include_genre: bool = fv.include_genre,
) -> np.ndarray:
    """Create static group weights so each feature family contributes roughly equally."""
    if include_genre:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, n_genres]
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2]

    total_dims = sum(group_sizes)
    weights = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for size in group_sizes:
        weights[idx : idx + size] /= np.sqrt(size)
        idx += size
    return weights


def _load_genre_mapping(
    audio_dir: str,
    results_dir: str,
    include_genre: bool,
) -> Tuple[Dict[str, str], List[str]]:
    """Load existing genre mapping or rebuild it from the audio directory."""
    genre_map_path = os.path.join(results_dir, "genre_map.npy")
    genre_list_path = os.path.join(results_dir, "genre_list.npy")

    if os.path.exists(genre_map_path) and include_genre:
        genre_map: Dict[str, str] = np.load(genre_map_path, allow_pickle=True).item()
        print(f"Loaded genre mapping for {len(genre_map)} files")
    else:
        genre_map = {}
        genre_dirs = [d for d in glob.glob(os.path.join(audio_dir, "*")) if os.path.isdir(d)]
        for genre_dir in genre_dirs:
            genre = os.path.basename(genre_dir)
            wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
            for wav_path in wav_files:
                base = Path(wav_path).stem
                genre_map[base] = genre
        if include_genre:
            np.save(genre_map_path, genre_map)
        print(f"Created genre mapping for {len(genre_map)} files")

    unique_genres = sorted(set(genre_map.values()))
    np.save(genre_list_path, unique_genres)
    print(f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres)}")
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
            for key in ["mfcc", "melspectrogram", "spectral_centroid", "zero_crossing_rate"]
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


def _select_optimal_k(
    X: np.ndarray,
    initial_k: int,
    k_min: int,
    k_max: int,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """Pick the number of clusters that maximizes the silhouette score."""
    best_k = initial_k
    best_labels: Optional[np.ndarray] = None
    best_centers: Optional[np.ndarray] = None
    best_score = -np.inf

    for k in range(k_min, k_max + 1):
        estimator = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = estimator.fit_predict(X)
        unique_count = len(np.unique(labels_tmp))
        if unique_count < 2 or unique_count >= len(labels_tmp):
            continue
        silhouette = silhouette_score(X, labels_tmp)
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
            best_labels = labels_tmp
            best_centers = estimator.cluster_centers_

    if best_labels is not None:
        print(f"Optimal k (silhouette) -> {best_k}")
    else:
        print("Silhouette-based search did not improve on the initial cluster count")

    return best_k, best_labels, best_centers


def run_kmeans_clustering(
    audio_dir: str = "genres_original",
    results_dir: str = "output/results",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = False,
    dynamic_k_min: int = 2,
    dynamic_k_max: int = 10,
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    weights = build_group_weights(n_mfcc=n_mfcc, n_mels=n_mels, include_genre=include_genre)
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
        )
    X_weighted = X_scaled * weights

    labels: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = None

    if dynamic_cluster_selection:
        best_k, best_labels, best_centers = _select_optimal_k(
            X_weighted, n_clusters, dynamic_k_min, dynamic_k_max
        )
        n_clusters = best_k
        if best_labels is not None and best_centers is not None:
            labels = best_labels
            centers = best_centers

    if labels is None or centers is None:
        estimator = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = estimator.fit_predict(X_weighted)
        centers = estimator.cluster_centers_

    coords = PCA(n_components=2, random_state=42).fit_transform(X_weighted)
    distances = np.linalg.norm(X_weighted - centers[labels], axis=1)

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

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "audio_clustering_results_kmeans.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results written to -> {csv_path}")

    return df, coords, labels


if __name__ == "__main__":
    DF, COORDS, LABELS = run_kmeans_clustering(
        audio_dir="genres_original",
        results_dir="output/results",
        n_clusters=5,
        dynamic_cluster_selection=True,
        include_genre=fv.include_genre,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="genres_original", clustering_method="K-means")
