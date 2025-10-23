import os
import glob
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import matplotlib
import matplotlib.pyplot as plt

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import feature_vars as fv
from src.ui.modern_ui import launch_ui


def build_group_weights(n_mfcc: int = fv.n_mfcc, n_mels: int = fv.n_mels, include_genre: bool = True) -> np.ndarray:
    """
    Create weights to balance different feature groups during clustering.
    """
    # If we include genre as one-hot encoding, we need to account for that in the weights
    if include_genre:
        # We'll add a group for genre (assuming 10 genres, one-hot encoded)
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 10]  # 10 for one-hot encoded genres
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2]
    
    total_dims = sum(group_sizes)
    w = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for g in group_sizes:
        w[idx:idx + g] /= np.sqrt(g)
        idx += g
    return w


def find_optimal_eps(X, min_pts=5, n_neighbors=10):
    """
    Use k-distance graph to find optimal eps value for DBSCAN.
    """
    # Calculate distances to the nth nearest neighbor
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)
    
    # Sort distances to the nth nearest neighbor
    distances = np.sort(distances[:, n_neighbors-1])
    
    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'Distance to {n_neighbors}th nearest neighbor')
    plt.title('K-distance Graph for DBSCAN Epsilon Parameter Selection')
    plt.grid(True)
    
    # Save the plot
    dbscan_dir = "dbscan"
    os.makedirs(dbscan_dir, exist_ok=True)
    kdist_plot_path = os.path.join(dbscan_dir, "kdistance_plot.png")
    plt.savefig(kdist_plot_path)
    plt.close()
    print(f"K-distance plot saved to → {kdist_plot_path}")
    
    # Find the "elbow" point where the slope changes dramatically
    # This is a heuristic approach to find the optimal epsilon
    diff = np.diff(distances)
    diff_r = diff[1:] / diff[:-1]
    elbow_idx = np.argmax(diff_r) + 1
    eps = distances[elbow_idx]
    
    print(f"Suggested optimal eps value: {eps:.4f} at point {elbow_idx}")
    return eps


def run_dbscan_clustering(
    audio_dir: str = "genres_original",
    results_dir: str = "results",
    eps: float = None,  # DBSCAN epsilon parameter, if None it will be automatically determined
    min_samples: int = 5,  # Min points to form a core point
    auto_eps: bool = True,  # Whether to automatically determine eps
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = True,
):
    """
    Run DBSCAN clustering on audio features.
    
    Parameters:
    -----------
    audio_dir : str
        Directory containing audio files organized by genre
    results_dir : str
        Directory containing extracted features
    eps : float or None
        DBSCAN epsilon parameter (max distance between points to be considered neighbors)
    min_samples : int
        Minimum number of samples in a neighborhood to be considered a core point
    auto_eps : bool
        Whether to automatically determine epsilon using k-distance graph
    n_mfcc, n_mels : int
        Feature extraction parameters
    include_genre : bool
        Whether to include genre information in clustering
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing clustering results
    coords : numpy.ndarray
        2D coordinates for visualization (PCA)
    labels : numpy.ndarray
        Cluster labels for each data point (-1 represents noise)
    """
    # Create a dedicated folder for DBSCAN clustering results
    dbscan_dir = "dbscan"
    os.makedirs(dbscan_dir, exist_ok=True)
    
    # Make sure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Load genre mapping if it exists
    genre_map_path = os.path.join(results_dir, "genre_map.npy")
    genre_list_path = os.path.join(results_dir, "genre_list.npy")
    
    if os.path.exists(genre_map_path) and include_genre:
        genre_map = np.load(genre_map_path, allow_pickle=True).item()
        print(f"Loaded genre mapping for {len(genre_map)} files")
    else:
        # Create genre mapping from directory structure if it doesn't exist
        genre_map = {}
        genre_dirs = [d for d in glob.glob(os.path.join(audio_dir, "*")) if os.path.isdir(d)]
        for genre_dir in genre_dirs:
            genre = os.path.basename(genre_dir)
            wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
            for wav_path in wav_files:
                base = Path(wav_path).stem
                genre_map[base] = genre
        print(f"Created genre mapping for {len(genre_map)} files")
        
    # Get unique genres and create a mapping from genre to index
    unique_genres = sorted(set(genre_map.values()))
    genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
    
    # Save the list of genres for future reference
    np.save(genre_list_path, unique_genres)
    print(f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres)}")
    
    # Get all files with extracted features
    feature_files = glob.glob(os.path.join(results_dir, "*_mfcc.npy"))
    base_names = [os.path.basename(f).replace("_mfcc.npy", "") for f in feature_files]
    
    file_names, feature_vectors, genres = [], [], []
    for base in base_names:
        feats = {k: os.path.join(results_dir, f"{base}_{k}.npy") for k in [
            "mfcc", "melspectrogram", "spectral_centroid",
            "zero_crossing_rate",
        ]}
        if not all(os.path.isfile(p) for p in feats.values()):
            continue
            
        # Extract genre for this file
        if base in genre_map:
            genre = genre_map[base]
        else:
            # Try to extract genre from filename
            parts = base.split('.')
            if len(parts) > 0:
                potential_genre = parts[0]
                if potential_genre in unique_genres:
                    genre = potential_genre
                else:
                    print(f"Warning: Could not determine genre for {base}, skipping")
                    continue
            else:
                print(f"Warning: Could not determine genre for {base}, skipping")
                continue
                
        genres.append(genre)
        
        # Load and process acoustic features
        arrays = [np.load(p) for p in feats.values()]
        vec = np.concatenate([
            np.concatenate([arr.mean(axis=1), arr.std(axis=1)])
            for arr in arrays
        ])
        
        # Create one-hot encoding for genre if needed
        if include_genre:
            genre_vec = np.zeros(len(unique_genres))
            genre_vec[genre_to_idx[genre]] = 1
            vec = np.concatenate([vec, genre_vec])
            
        file_names.append(base)
        feature_vectors.append(vec)

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    # Scale features and apply weights
    X_scaled = StandardScaler().fit_transform(np.vstack(feature_vectors))
    weights = build_group_weights(n_mfcc=n_mfcc, n_mels=n_mels, include_genre=include_genre)
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}")
    X = X_scaled * weights

    # Find optimal epsilon parameter if auto_eps is True
    if auto_eps or eps is None:
        eps = find_optimal_eps(X, min_pts=min_samples)
    
    # Run DBSCAN clustering
    print(f"Running DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Get number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate cluster centers (excluding noise points)
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    centers = np.zeros((len(unique_labels), X.shape[1]))
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        centers[i] = np.mean(cluster_points, axis=0)

    # Apply PCA for visualization
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    # Create DataFrame with results
    df = pd.DataFrame({
        "Song": file_names,
        "Genre": genres,
        "Cluster": labels,
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
    })
    
    # Add distance to cluster center (except for noise points)
    distances = np.zeros(len(labels))
    for i, label in enumerate(labels):
        if label != -1:  # Not a noise point
            center_idx = unique_labels.index(label)
            distances[i] = np.linalg.norm(X[i] - centers[center_idx])
        else:
            # For noise points, use a large distance
            distances[i] = np.nan
    
    df["Distance"] = distances
    
    # Save results to CSV
    csv_path = os.path.join(dbscan_dir, "dbscan_clustering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results written to → {csv_path}")

    # Plot clusters in 2D space
    plt.figure(figsize=(12, 10))
    
    # Plot noise points first (black)
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(coords[noise_mask, 0], coords[noise_mask, 1], 
                   color='black', marker='x', label='Noise', alpha=0.7)
    
    # Plot clusters with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i, label in enumerate(unique_labels):
        cluster_points = coords[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i % len(colors)], label=f'Cluster {label}', alpha=0.7)
    
    plt.title('DBSCAN Clustering Result (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    clusters_plot_path = os.path.join(dbscan_dir, "dbscan_clusters.png")
    plt.savefig(clusters_plot_path)
    plt.close()
    print(f"Clusters plot saved to → {clusters_plot_path}")

    return df, coords, labels


if __name__ == "__main__":
    try:
        # For UI to properly work with DBSCAN's noise points (-1),
        # shift all labels up by 1, making noise points = cluster 0
        df, coords, labels_original = run_dbscan_clustering(
            audio_dir="genres_original",
            results_dir="output/results",
            auto_eps=True,
            min_samples=4,
            include_genre=True,
        )
        
        # Create a copy of the labels with shifted values (no negative values)
        # This is needed for the UI to work properly
        labels_shifted = labels_original + 1
        
        # Launch the UI with the shifted labels
        launch_ui(df, coords, labels_shifted, top_n=5, audio_dir="genres_original", clustering_method="DBSCAN")
    except Exception as e:
        print(f"Error launching UI: {e}")
        print("Running clustering without UI...")
        run_dbscan_clustering(audio_dir="genres_original")
