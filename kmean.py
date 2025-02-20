import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results',
                          n_clusters=3,
                          elbow_max_k=10,
                          show_elbow=True,
                          reduce_dim=False,
                          n_components=50,
                          dynamic_cluster_selection=False,
                          dynamic_k_min=2,
                          dynamic_k_max=10):
    """
    Creates a feature vector for each audio file by summarizing its extracted features,
    optionally reduces dimensionality with PCA, uses the elbow method for visualization,
    and then dynamically determines the best number of clusters using silhouette score
    if enabled. Finally, it runs optimized KMeans clustering, saves the clustering results
    to a CSV, and visualizes the clusters with their perimeters. Clicking on a dot in the
    final 2D scatter plot will show the corresponding song name.

    Parameters:
        audio_dir (str): Directory where the original audio (.wav) files are stored.
        results_dir (str): Directory where the extracted feature .npy files are stored.
        n_clusters (int): Number of clusters for final clustering (overridden if dynamic_cluster_selection is True).
        elbow_max_k (int): Maximum number of clusters to try in the elbow method.
        show_elbow (bool): If True, display the elbow method plot.
        reduce_dim (bool): If True, apply PCA to reduce dimensionality before clustering.
        n_components (int): Number of PCA components to retain if reduce_dim is True.
        dynamic_cluster_selection (bool): If True, automatically select the best number of clusters using silhouette scores.
        dynamic_k_min (int): Minimum number of clusters to try for dynamic selection (should be >=2).
        dynamic_k_max (int): Maximum number of clusters to try for dynamic selection.
    """
    # List all .wav files
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print("No audio files found in the audio directory.")
        return

    feature_vectors = []
    file_names = []

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        # Feature file paths
        mfcc_path = os.path.join(results_dir, f"{base_filename}_mfcc.npy")
        mel_path = os.path.join(
            results_dir, f"{base_filename}_melspectrogram.npy")
        spectral_centroid_path = os.path.join(
            results_dir, f"{base_filename}_spectral_centroid.npy")
        spectral_flatness_path = os.path.join(
            results_dir, f"{base_filename}_spectral_flatness.npy")
        zcr_path = os.path.join(
            results_dir, f"{base_filename}_zero_crossing_rate.npy")

        # Ensure all required feature files exist
        if not all(os.path.isfile(p) for p in [mfcc_path, mel_path, spectral_centroid_path, spectral_flatness_path, zcr_path]):
            print(f"Missing feature files for {base_filename}, skipping.")
            continue

        # Load features
        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        spectral_centroid = np.load(spectral_centroid_path)
        spectral_flatness = np.load(spectral_flatness_path)
        zcr = np.load(zcr_path)

        # Compute summary statistics (mean and std)
        mfcc_features = np.concatenate(
            [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        mel_features = np.concatenate(
            [np.mean(mel, axis=1), np.std(mel, axis=1)])
        spectral_centroid_features = np.concatenate(
            [np.mean(spectral_centroid, axis=1),
             np.std(spectral_centroid, axis=1)]
        )
        spectral_flatness_features = np.concatenate(
            [np.mean(spectral_flatness, axis=1),
             np.std(spectral_flatness, axis=1)]
        )
        zcr_features = np.concatenate(
            [np.mean(zcr, axis=1), np.std(zcr, axis=1)])

        # Combine all features into a single feature vector per audio file
        feature_vector = np.concatenate([
            mfcc_features,
            mel_features,
            spectral_centroid_features,
            spectral_flatness_features,
            zcr_features
        ])
        feature_vectors.append(feature_vector)
        file_names.append(base_filename)

    if not feature_vectors:
        print("No audio files with complete features found.")
        return

    # Create feature matrix and standardize it
    X = np.array(feature_vectors)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Optionally reduce dimensionality with PCA before clustering
    if reduce_dim:
        pca_reducer = PCA(n_components=n_components, random_state=42)
        X = pca_reducer.fit_transform(X)

    # Always use KMeans
    ClusterModel = KMeans

    # Optional: Display elbow method plot for visualization
    if show_elbow:
        inertia = []
        k_range = range(1, elbow_max_k + 1)
        for k in k_range:
            model = ClusterModel(
                n_clusters=k, init='k-means++', n_init=10, random_state=42)
            model.fit(X)
            inertia.append(model.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertia, 'bo-', markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.xticks(k_range)
        plt.show()

    # Dynamic cluster selection using silhouette score
    if dynamic_cluster_selection:
        silhouette_scores = {}
        for k in range(dynamic_k_min, dynamic_k_max + 1):
            model = ClusterModel(
                n_clusters=k, init='k-means++', n_init=10, random_state=42)
            cluster_labels = model.fit_predict(X)
            score = silhouette_score(X, cluster_labels)
            silhouette_scores[k] = score
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        print("Silhouette scores:")
        for k, score in silhouette_scores.items():
            print(f"  k = {k}: silhouette score = {score:.4f}")
        print(
            f"Optimal number of clusters based on silhouette score: {best_k}")
        n_clusters = best_k

    # Run final clustering
    clustering_params = {
        'n_clusters': n_clusters,
        'init': 'k-means++',
        'n_init': 10,
        'random_state': 42
    }
    cluster_model = ClusterModel(**clustering_params)
    clusters = cluster_model.fit_predict(X)

    # Compute distance to cluster center for each sample
    distances = np.linalg.norm(
        X - cluster_model.cluster_centers_[clusters], axis=1)

    # Display cluster assignments
    print("\nCluster assignments:")
    for fname, cluster in zip(file_names, clusters):
        print(f"  {fname}: Cluster {cluster}")

    # For visualization, project the features into 2D using PCA
    pca_vis = PCA(n_components=2, random_state=42)
    X_pca = pca_vis.fit_transform(X)

    # Create a DataFrame with clustering information
    df = pd.DataFrame({
        'Song': file_names,
        'Cluster': clusters,
        'Distance_to_Center': distances,
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1]
    })

    # Save the CSV in the same directory as this Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'audio_clustering_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nSaved clustering results to CSV: {csv_path}")

    # Visualize clusters and define cluster perimeters using convex hulls
    fig, ax = plt.subplots(figsize=(8, 6))

    # Enable picking by setting picker=True (and optional pickradius)
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap='viridis',
        s=100, alpha=0.7,
        picker=True  # or picker=5 for a 5-point radius around mouse
    )

    # Draw convex hulls for each cluster
    unique_clusters = np.unique(clusters)
    for cl in unique_clusters:
        indices = np.where(clusters == cl)[0]
        if len(indices) >= 3:  # need at least 3 points to form a hull
            points = X_pca[indices]
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.concatenate(
                (hull_points, hull_points[0:1]), axis=0)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'k-', lw=2)
        else:
            # If there are fewer than 3 points, just draw circles around them
            for idx in indices:
                ax.plot(X_pca[idx, 0], X_pca[idx, 1], 'ko',
                        markersize=12, fillstyle='none')

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Optimized K-Means Clustering of Audio Features")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    # Create an Annotation object for displaying song names on click
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    # Define a function to update the annotation text and position
    def update_annotation(event_ind):
        """Update annotation with the first picked point's info."""
        idx = event_ind[0]
        x, y = X_pca[idx, 0], X_pca[idx, 1]
        song_name = df['Song'].iloc[idx]

        annot.xy = (x, y)
        annot.set_text(song_name)
        annot.set_visible(True)
        fig.canvas.draw_idle()

    # Define a callback for pick events
    def on_pick(event):
        # event.ind is a list of data point indices picked
        if len(event.ind) > 0:
            update_annotation(event.ind)

    # Connect the pick_event to the callback
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results',
                          n_clusters=3,
                          reduce_dim=True,
                          n_components=50,
                          dynamic_cluster_selection=True,
                          dynamic_k_min=2,
                          dynamic_k_max=10)
