"""
Evaluate clustering performance using multiple metrics:
- Silhouette Score
- Calinski-Harabasz Index (CH)
- Davies-Bouldin Index (DB)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Stability_ARI (multi-run robustness)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------
# Path setup (aligned with your other clustering scripts)
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # ML-song-recommendation-system/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv  # noqa: E402
from src.clustering.kmeans import (   # noqa: E402
    _load_genre_mapping,
    _collect_feature_vectors,
    build_group_weights,
)

# ---------------------------------------------------------------------
# Stability configuration (tweak if you want)
# ---------------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 4]          # KMeans/GMM: random_state seeds
HDBSCAN_BOOT_ROUNDS = 5          # HDBSCAN: bootstrap repetitions
HDBSCAN_SAMPLE_FRAC = 0.90       # HDBSCAN: subsample fraction
HDBSCAN_RANDOM_SEED = 0          # HDBSCAN: RNG seed for subsampling

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def load_clustering_results(method: str) -> pd.DataFrame:
    """Load clustering results CSV for a given method."""
    csv_path = f"output/audio_clustering_results_{method}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_feature_data(
    audio_dir: str = "genres_original",
    results_dir: str = "output/results",
    include_genre: bool = fv.include_genre,
):
    """Prepare standardized and weighted feature data."""
    genre_map, unique_genres = _load_genre_mapping(
        audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        results_dir, genre_map, unique_genres, include_genre
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    weights = build_group_weights(
        n_mfcc=fv.n_mfcc,
        n_mels=fv.n_mels,
        include_genre=include_genre,  # n_genres comes from fv by default
    )

    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
        )

    X_weighted = X_scaled * weights

    # Create a mapping from song name to index
    song_to_idx = {name: idx for idx, name in enumerate(file_names)}
    return X_weighted, song_to_idx, genres, unique_genres


# ---------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------
def stability_ari_kmeans(
    X: np.ndarray,
    n_clusters: int,
    seeds: List[int] = SEEDS,
    n_init: int = 10,
) -> float:
    """Mean pairwise ARI across multiple KMeans runs with different seeds."""
    if n_clusters < 2:
        return np.nan
    runs = []
    for s in seeds:
        km = KMeans(n_clusters=n_clusters,
                    random_state=s, n_init=n_init).fit(X)
        runs.append(km.labels_)
    pairwise = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairwise.append(adjusted_rand_score(runs[i], runs[j]))
    return float(np.mean(pairwise)) if pairwise else np.nan


def stability_ari_gmm(
    X: np.ndarray,
    n_components: int,
    seeds: List[int] = SEEDS,
    covariance_type: str = "full",
    max_iter: int = 200,
    tol: float = 1e-3,
    init_params: str = "kmeans",
) -> float:
    """Mean pairwise ARI across multiple GMM runs with different seeds."""
    if n_components < 2:
        return np.nan
    runs = []
    for s in seeds:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            init_params=init_params,
            random_state=s,
        ).fit(X)
        runs.append(gmm.predict(X))
    pairwise = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairwise.append(adjusted_rand_score(runs[i], runs[j]))
    return float(np.mean(pairwise)) if pairwise else np.nan


def stability_ari_hdbscan(
    X: np.ndarray,
    n_rounds: int = HDBSCAN_BOOT_ROUNDS,
    sample_frac: float = HDBSCAN_SAMPLE_FRAC,
    seed: int = HDBSCAN_RANDOM_SEED,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None,
    cluster_selection_epsilon: float = 0.0,
    allow_single_cluster: bool = False,
) -> float:
    """
    Bootstrap stability for HDBSCAN:
    - Refit on random subsamples.
    - Compute ARI on the intersection of indices that are non-noise in both runs.
    """
    # Avoid module-name collision with your src/clustering/hdbscan.py
    removed_script_dir = False
    if str(SCRIPT_DIR) in sys.path:
        sys.path.remove(str(SCRIPT_DIR))
        removed_script_dir = True
    try:
        import hdbscan  # type: ignore
    finally:
        if removed_script_dir:
            sys.path.insert(0, str(SCRIPT_DIR))

    rng = np.random.default_rng(seed)
    runs: List[np.ndarray] = []
    idxs: List[np.ndarray] = []

    for _ in range(n_rounds):
        mask = rng.random(len(X)) < sample_frac
        Xi = X[mask]
        idx = np.where(mask)[0]
        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            allow_single_cluster=allow_single_cluster,
            prediction_data=True,
        ).fit(Xi)
        runs.append(model.labels_)
        idxs.append(idx)

    pairwise = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            common = np.intersect1d(idxs[i], idxs[j], assume_unique=False)
            if common.size == 0:
                continue
            map_i = {v: k for k, v in enumerate(idxs[i])}
            map_j = {v: k for k, v in enumerate(idxs[j])}
            li = np.array([runs[i][map_i[v]] for v in common])
            lj = np.array([runs[j][map_j[v]] for v in common])
            # Only compare points that are NON-NOISE in both runs and have ≥2 clusters
            keep = (li != -1) & (lj != -1)
            if keep.sum() >= 2:
                if len(np.unique(li[keep])) > 1 and len(np.unique(lj[keep])) > 1:
                    pairwise.append(adjusted_rand_score(li[keep], lj[keep]))
    return float(np.mean(pairwise)) if pairwise else np.nan


def stability_ari_vade(
    X: np.ndarray,
    n_components: int,
    latent_dim: int = 10,
    seeds: List[int] = SEEDS,
    pretrain_epochs: int = 10,  # Reduced for speed during evaluation
    train_epochs: int = 40,     # Reduced for speed during evaluation
    batch_size: int = 128,
    lr: float = 1e-3,
) -> float:
    """
    Mean pairwise ARI across multiple VaDE runs with different random seeds.
    Note: VaDE training can be slow, so we use reduced epochs for stability testing.
    """
    try:
        import torch
        from src.clustering.vade import VaDE, pretrain_autoencoder, init_gmm_prior, train_vade, TrainConfig
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("Warning: PyTorch not available, cannot compute VaDE stability")
        return np.nan
    
    if n_components < 2:
        return np.nan
    
    runs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for s in seeds:
        # Set seed for reproducibility
        torch.manual_seed(s)
        np.random.seed(s)
        
        # Prepare data
        X_tensor = torch.from_numpy(X.astype(np.float32))
        ds = TensorDataset(X_tensor)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # Create and train model
        cfg = TrainConfig(
            latent_dim=latent_dim,
            n_components=n_components,
            enc_hidden=[512, 256],
            dec_hidden=[256, 512],
            batch_size=batch_size,
            lr=lr,
            pretrain_epochs=pretrain_epochs,
            train_epochs=train_epochs,
            device=str(device),
        )
        
        model = VaDE(
            input_dim=X.shape[1],
            latent_dim=cfg.latent_dim,
            enc_hidden=cfg.enc_hidden,
            dec_hidden=cfg.dec_hidden,
            n_components=cfg.n_components,
        ).to(device)
        
        # Train (suppressing output)
        pretrain_autoencoder(model, loader, device, cfg)
        init_gmm_prior(model, X, cfg)
        train_vade(model, loader, cfg)
        
        # Get cluster assignments
        model.eval()
        with torch.no_grad():
            all_gamma = []
            for (x_batch,) in DataLoader(ds, batch_size=2048, shuffle=False):
                x_batch = x_batch.to(device)
                _, _, _, _, gamma = model(x_batch)
                all_gamma.append(gamma.cpu().numpy())
            GAMMA = np.vstack(all_gamma)
        
        labels = GAMMA.argmax(axis=1)
        runs.append(labels)
    
    # Calculate pairwise ARI
    pairwise = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pairwise.append(adjusted_rand_score(runs[i], runs[j]))
    
    return float(np.mean(pairwise)) if pairwise else np.nan


# ---------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------
def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: np.ndarray,
    method_name: str,
) -> dict:
    """Calculate all clustering evaluation metrics."""
    results = {"Method": method_name}

    # Filter out noise points (label -1) for HDBSCAN
    noise_mask = labels != -1
    X_filtered = X[noise_mask]
    labels_filtered = labels[noise_mask]
    true_labels_filtered = true_labels[noise_mask]

    # Calculate number of clusters
    n_clusters = len(np.unique(labels_filtered))
    n_noise = np.sum(~noise_mask)
    results["N_Clusters"] = n_clusters
    results["N_Noise_Points"] = n_noise
    results["Noise_Percentage"] = (n_noise / len(labels)) * 100.0

    # Internal validation metrics (no ground truth needed)
    if n_clusters > 1 and len(labels_filtered) > n_clusters:
        try:
            results["Silhouette_Score"] = silhouette_score(
                X_filtered, labels_filtered)
        except Exception as e:
            results["Silhouette_Score"] = np.nan
            print(
                f"Warning: Could not calculate Silhouette Score for {method_name}: {e}")

        try:
            results["Calinski_Harabasz_Score"] = calinski_harabasz_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            results["Calinski_Harabasz_Score"] = np.nan
            print(
                f"Warning: Could not calculate CH Score for {method_name}: {e}")

        try:
            results["Davies_Bouldin_Score"] = davies_bouldin_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            results["Davies_Bouldin_Score"] = np.nan
            print(
                f"Warning: Could not calculate DB Score for {method_name}: {e}")
    else:
        results["Silhouette_Score"] = np.nan
        results["Calinski_Harabasz_Score"] = np.nan
        results["Davies_Bouldin_Score"] = np.nan
        print(
            f"Warning: Not enough clusters or samples for internal metrics in {method_name}")

    # External validation metrics (need true labels)
    if len(np.unique(true_labels_filtered)) > 1 and n_clusters > 1:
        try:
            results["Adjusted_Rand_Index"] = adjusted_rand_score(
                true_labels_filtered, labels_filtered
            )
        except Exception as e:
            results["Adjusted_Rand_Index"] = np.nan
            print(f"Warning: Could not calculate ARI for {method_name}: {e}")

        try:
            results["Normalized_Mutual_Info"] = normalized_mutual_info_score(
                true_labels_filtered, labels_filtered
            )
        except Exception as e:
            results["Normalized_Mutual_Info"] = np.nan
            print(f"Warning: Could not calculate NMI for {method_name}: {e}")
    else:
        results["Adjusted_Rand_Index"] = np.nan
        results["Normalized_Mutual_Info"] = np.nan
        print(
            f"Warning: Not enough unique labels for external metrics in {method_name}")

    results["Stability_ARI"] = np.nan  # filled below in main()
    return results


# ---------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------
def main():
    """Evaluate all clustering methods and include stability."""
    print("=" * 70)
    print("CLUSTERING EVALUATION - COMPARING ALL METHODS (with Stability)")
    print("=" * 70)

    # Prepare feature data
    print("\nLoading and preparing feature data...")
    X_weighted, song_to_idx, genre_labels, unique_genres = prepare_feature_data()
    print(
        f"Loaded {len(X_weighted)} songs with {X_weighted.shape[1]} features")
    print(
        f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres)}")

    # Create integer labels for genres
    genre_to_int = {genre: idx for idx, genre in enumerate(unique_genres)}
    true_labels_full = np.array([genre_to_int[genre]
                                for genre in genre_labels])

    methods = ["kmeans", "hdbscan", "gmm", "vade"]
    all_results = []

    for method in methods:
        print(f"\n{'-' * 70}")
        print(f"Evaluating {method.upper()} clustering...")
        print(f"{'-' * 70}")

        try:
            df = load_clustering_results(method)
            print(f"Loaded {len(df)} results from {method}")

            # Align labels with feature data
            aligned_labels: List[int] = []
            aligned_features: List[np.ndarray] = []
            aligned_true_labels: List[int] = []

            for _, row in df.iterrows():
                song_name = row["Song"]
                if song_name in song_to_idx:
                    idx = song_to_idx[song_name]
                    aligned_labels.append(int(row["Cluster"]))
                    aligned_features.append(X_weighted[idx])
                    aligned_true_labels.append(true_labels_full[idx])

            aligned_labels = np.array(aligned_labels)
            aligned_features = np.array(aligned_features)
            aligned_true_labels = np.array(aligned_true_labels)

            print(f"Aligned {len(aligned_labels)} songs for evaluation")

            # Evaluate base metrics
            results = evaluate_clustering(
                aligned_features, aligned_labels, aligned_true_labels, method.upper()
            )

            # -------- Stability (mean ARI across runs) --------
            stab = np.nan
            if method == "kmeans":
                k_est = results["N_Clusters"]
                if isinstance(k_est, (int, np.integer)) and k_est >= 2:
                    stab = stability_ari_kmeans(
                        X_weighted, n_clusters=int(k_est), seeds=SEEDS)
            elif method == "gmm":
                c_est = results["N_Clusters"]
                if isinstance(c_est, (int, np.integer)) and c_est >= 2:
                    stab = stability_ari_gmm(
                        X_weighted,
                        n_components=int(c_est),
                        seeds=SEEDS,
                        covariance_type="full",
                        max_iter=200,
                        tol=1e-3,
                        init_params="kmeans",
                    )
            elif method == "hdbscan":
                # Parameters mirror your defaults in run_hdbscan_clustering()
                stab = stability_ari_hdbscan(
                    X_weighted,
                    n_rounds=HDBSCAN_BOOT_ROUNDS,
                    sample_frac=HDBSCAN_SAMPLE_FRAC,
                    seed=HDBSCAN_RANDOM_SEED,
                    min_cluster_size=10,
                    min_samples=None,
                    cluster_selection_epsilon=0.0,
                    allow_single_cluster=False,
                )
            elif method == "vade":
                # VaDE stability (can be slow, using reduced epochs)
                c_est = results["N_Clusters"]
                if isinstance(c_est, (int, np.integer)) and c_est >= 2:
                    print("  Computing VaDE stability (this may take a few minutes)...")
                    stab = stability_ari_vade(
                        X_weighted,
                        n_components=int(c_est),
                        latent_dim=10,
                        seeds=SEEDS,
                        pretrain_epochs=10,  # Reduced for evaluation
                        train_epochs=40,     # Reduced for evaluation
                        batch_size=128,
                        lr=1e-3,
                    )

            results["Stability_ARI"] = stab
            all_results.append(results)

            # Print results
            print(
                f"\nResults for {method.UPPER() if hasattr(method,'UPPER') else method.upper()}:")
            print(f"  Number of Clusters: {results['N_Clusters']}")
            print(
                f"  Noise Points: {results['N_Noise_Points']} ({results['Noise_Percentage']:.2f}%)")
            print(f"  Silhouette Score: {results['Silhouette_Score']:.4f}")
            print(
                f"  Calinski-Harabasz Index: {results['Calinski_Harabasz_Score']:.2f}")
            print(
                f"  Davies-Bouldin Index: {results['Davies_Bouldin_Score']:.4f}")
            print(
                f"  Adjusted Rand Index: {results['Adjusted_Rand_Index']:.4f}")
            print(
                f"  Normalized Mutual Info: {results['Normalized_Mutual_Info']:.4f}")
            print(
                f"  Stability (mean ARI): {results['Stability_ARI'] if pd.notna(results['Stability_ARI']) else 'nan'}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {method}. Please run the clustering first.")
        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY - COMPARISON OF ALL METHODS")
        print("=" * 70)

        comparison_df = pd.DataFrame(all_results)

        column_order = [
            "Method",
            "N_Clusters",
            "N_Noise_Points",
            "Noise_Percentage",
            "Silhouette_Score",
            "Calinski_Harabasz_Score",
            "Davies_Bouldin_Score",
            "Adjusted_Rand_Index",
            "Normalized_Mutual_Info",
            "Stability_ARI",
        ]
        comparison_df = comparison_df[column_order]

        print("\n" + comparison_df.to_string(index=False))

        # Save results
        output_path = "output/clustering_evaluation_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")

        # Interpretation guide
        print("\n" + "=" * 70)
        print("INTERPRETATION GUIDE")
        print("=" * 70)
        print("\nInternal Metrics (no ground truth needed):")
        print("  • Silhouette Score [-1, 1]: HIGHER is better (>0.5 = good)")
        print("  • Calinski-Harabasz Index: HIGHER is better")
        print("  • Davies-Bouldin Index: LOWER is better (closer to 0)")
        print("\nExternal Metrics (compared to true genres):")
        print(
            "  • Adjusted Rand Index [-1, 1]: HIGHER is better (1 = perfect)")
        print(
            "  • Normalized Mutual Info [0, 1]: HIGHER is better (1 = perfect)")
        print("\nRobustness:")
        print(
            "  • Stability_ARI [0, 1]: mean ARI across multiple runs; HIGHER = more reproducible")
        print(
            "\nNote: For HDBSCAN, noise points (-1) are excluded from metric calculations.")
        print("=" * 70)
    else:
        print("\n❌ No results to compare. Please run clustering methods first.")


if __name__ == "__main__":
    main()
