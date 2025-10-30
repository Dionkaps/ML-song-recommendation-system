"""
Evaluate clustering performance using multiple metrics:
- Silhouette Score
- Calinski-Harabasz Index (CH)
- Davies-Bouldin Index (DB)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
"""

import os
import sys
from pathlib import Path

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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import feature_vars as fv
from src.clustering.kmeans import (
    _load_genre_mapping,
    _collect_feature_vectors,
    build_group_weights,
)


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
    genre_map, unique_genres = _load_genre_mapping(audio_dir, results_dir, include_genre)
    file_names, feature_vectors, genres = _collect_feature_vectors(
        results_dir, genre_map, unique_genres, include_genre
    )

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    weights = build_group_weights(
        n_mfcc=fv.n_mfcc, n_mels=fv.n_mels, include_genre=include_genre
    )
    X_weighted = X_scaled * weights

    # Create a mapping from song name to index
    song_to_idx = {name: idx for idx, name in enumerate(file_names)}

    return X_weighted, song_to_idx, genres, unique_genres


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
    results["Noise_Percentage"] = (n_noise / len(labels)) * 100

    # Internal validation metrics (don't need true labels)
    if n_clusters > 1 and len(labels_filtered) > n_clusters:
        try:
            results["Silhouette_Score"] = silhouette_score(X_filtered, labels_filtered)
        except Exception as e:
            results["Silhouette_Score"] = np.nan
            print(f"Warning: Could not calculate Silhouette Score for {method_name}: {e}")

        try:
            results["Calinski_Harabasz_Score"] = calinski_harabasz_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            results["Calinski_Harabasz_Score"] = np.nan
            print(f"Warning: Could not calculate CH Score for {method_name}: {e}")

        try:
            results["Davies_Bouldin_Score"] = davies_bouldin_score(
                X_filtered, labels_filtered
            )
        except Exception as e:
            results["Davies_Bouldin_Score"] = np.nan
            print(f"Warning: Could not calculate DB Score for {method_name}: {e}")
    else:
        results["Silhouette_Score"] = np.nan
        results["Calinski_Harabasz_Score"] = np.nan
        results["Davies_Bouldin_Score"] = np.nan
        print(f"Warning: Not enough clusters or samples for internal metrics in {method_name}")

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
        print(f"Warning: Not enough unique labels for external metrics in {method_name}")

    return results


def main():
    """Evaluate all three clustering methods."""
    print("=" * 70)
    print("CLUSTERING EVALUATION - COMPARING ALL METHODS")
    print("=" * 70)

    # Prepare feature data
    print("\nLoading and preparing feature data...")
    X_weighted, song_to_idx, genre_labels, unique_genres = prepare_feature_data()
    print(f"Loaded {len(X_weighted)} songs with {X_weighted.shape[1]} features")
    print(f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres)}")

    # Create genre label encoder
    genre_to_int = {genre: idx for idx, genre in enumerate(unique_genres)}
    true_labels = np.array([genre_to_int[genre] for genre in genre_labels])

    # Evaluate each clustering method
    methods = ["kmeans", "hdbscan", "gmm"]
    all_results = []

    for method in methods:
        print(f"\n{'-' * 70}")
        print(f"Evaluating {method.upper()} clustering...")
        print(f"{'-' * 70}")

        try:
            # Load clustering results
            df = load_clustering_results(method)
            print(f"Loaded {len(df)} results from {method}")

            # Align labels with feature data
            aligned_labels = []
            aligned_features = []
            aligned_true_labels = []

            for idx, row in df.iterrows():
                song_name = row["Song"]
                if song_name in song_to_idx:
                    feature_idx = song_to_idx[song_name]
                    aligned_labels.append(row["Cluster"])
                    aligned_features.append(X_weighted[feature_idx])
                    aligned_true_labels.append(true_labels[feature_idx])

            aligned_labels = np.array(aligned_labels)
            aligned_features = np.array(aligned_features)
            aligned_true_labels = np.array(aligned_true_labels)

            print(f"Aligned {len(aligned_labels)} songs for evaluation")

            # Evaluate
            results = evaluate_clustering(
                aligned_features, aligned_labels, aligned_true_labels, method.upper()
            )
            all_results.append(results)

            # Print results for this method
            print(f"\nResults for {method.upper()}:")
            print(f"  Number of Clusters: {results['N_Clusters']}")
            print(f"  Noise Points: {results['N_Noise_Points']} ({results['Noise_Percentage']:.2f}%)")
            print(f"  Silhouette Score: {results['Silhouette_Score']:.4f}")
            print(f"  Calinski-Harabasz Index: {results['Calinski_Harabasz_Score']:.2f}")
            print(f"  Davies-Bouldin Index: {results['Davies_Bouldin_Score']:.4f}")
            print(f"  Adjusted Rand Index: {results['Adjusted_Rand_Index']:.4f}")
            print(f"  Normalized Mutual Info: {results['Normalized_Mutual_Info']:.4f}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {method}. Please run the clustering first.")
        except Exception as e:
            print(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison DataFrame
    if all_results:
        print("\n" + "=" * 70)
        print("SUMMARY - COMPARISON OF ALL METHODS")
        print("=" * 70)

        comparison_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
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
        ]
        comparison_df = comparison_df[column_order]

        print("\n" + comparison_df.to_string(index=False))

        # Save results
        output_path = "output/clustering_evaluation_comparison.csv"
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")

        # Print interpretation guide
        print("\n" + "=" * 70)
        print("INTERPRETATION GUIDE")
        print("=" * 70)
        print("\nInternal Metrics (no ground truth needed):")
        print("  • Silhouette Score [-1, 1]: HIGHER is better (>0.5 = good)")
        print("  • Calinski-Harabasz Index: HIGHER is better")
        print("  • Davies-Bouldin Index: LOWER is better (closer to 0)")
        print("\nExternal Metrics (compared to true genres):")
        print("  • Adjusted Rand Index [-1, 1]: HIGHER is better (1 = perfect)")
        print("  • Normalized Mutual Info [0, 1]: HIGHER is better (1 = perfect)")
        print("\nNote: Noise points are excluded from metric calculations.")
        print("=" * 70)
    else:
        print("\n❌ No results to compare. Please run clustering methods first.")


if __name__ == "__main__":
    main()
