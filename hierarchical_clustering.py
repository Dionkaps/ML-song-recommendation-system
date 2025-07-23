import os
import glob
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import matplotlib
import matplotlib.pyplot as plt

import feature_vars as fv


def build_group_weights(n_mfcc: int = fv.n_mfcc, n_mels: int = fv.n_mels, include_genre: bool = True) -> np.ndarray:
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


def run_hierarchical_clustering(
    audio_dir: str = "genres_original",
    results_dir: str = "results",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = False,
    dynamic_k_min: int = 2,
    dynamic_k_max: int = 10,
    linkage_method: str = "ward",  # Options: 'ward', 'complete', 'average', 'single'
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = True,
):
    # Create a dedicated folder for hierarchical clustering results
    hierarchical_dir = "hierarchical"
    os.makedirs(hierarchical_dir, exist_ok=True)
    
    # Original results_dir is still needed for finding feature files
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

    X_scaled = StandardScaler().fit_transform(np.vstack(feature_vectors))
    weights = build_group_weights(n_mfcc=n_mfcc, n_mels=n_mels, include_genre=include_genre)
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}")
    X = X_scaled * weights

    # Perform hierarchical clustering
    if dynamic_cluster_selection:
        sil = {}
        Z = linkage(X, method=linkage_method)
        for k in range(dynamic_k_min, dynamic_k_max + 1):
            lbls_tmp = fcluster(Z, k, criterion='maxclust')
            sil[k] = silhouette_score(X, lbls_tmp)
        n_clusters = max(sil, key=sil.get)
        print(f"Optimal k (silhouette) → {n_clusters}")

    # Perform final clustering with optimal or specified number of clusters
    Z = linkage(X, method=linkage_method)
    labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # Zero-based indexing
    
    # Calculate cluster centers
    centers = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centers[i] = np.mean(cluster_points, axis=0)

    # Apply PCA for visualization
    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    # Create DataFrame with results
    df = pd.DataFrame({
        "Song": file_names,
        "Genre": genres,
        "Cluster": labels,
        "Distance": [np.linalg.norm(X[i] - centers[labels[i]]) for i in range(len(X))],
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
    })
    csv_path = os.path.join(hierarchical_dir, "hierarchical_clustering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results written to → {csv_path}")

    # Optional: Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram_path = os.path.join(hierarchical_dir, "dendrogram.png")
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Dendrogram saved to → {dendrogram_path}")

    # Plot clusters in 2D space
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = coords[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], label=f'Cluster {i+1}', alpha=0.7)
    
    plt.title('Hierarchical Clustering Result (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    clusters_plot_path = os.path.join(hierarchical_dir, "hierarchical_clusters.png")
    plt.savefig(clusters_plot_path)
    plt.close()
    print(f"Clusters plot saved to → {clusters_plot_path}")

    return df, coords, labels


if __name__ == "__main__":
    try:
        from modern_ui import launch_ui
        df, coords, labels = run_hierarchical_clustering(
            audio_dir="genres_original",
            results_dir="results",
            n_clusters=5,  # More clusters for genre data
            include_genre=True,
        )
        launch_ui(df, coords, labels, top_n=5, audio_dir="genres_original", clustering_method="Hierarchical")
    except Exception as e:
        print(f"Error launching UI: {e}")
        print("Running clustering without UI...")
        run_hierarchical_clustering(audio_dir="genres_original")
