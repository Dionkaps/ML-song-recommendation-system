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
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print("No audio files found in the audio directory.")
        return

    feature_vectors = []
    file_names = []

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        mfcc_path = os.path.join(results_dir, f"{base_filename}_mfcc.npy")
        mel_path = os.path.join(
            results_dir, f"{base_filename}_melspectrogram.npy")
        spectral_centroid_path = os.path.join(
            results_dir, f"{base_filename}_spectral_centroid.npy")
        spectral_flatness_path = os.path.join(
            results_dir, f"{base_filename}_spectral_flatness.npy")
        zcr_path = os.path.join(
            results_dir, f"{base_filename}_zero_crossing_rate.npy")
        if not all(os.path.isfile(p) for p in [mfcc_path, mel_path, spectral_centroid_path, spectral_flatness_path, zcr_path]):
            print(f"Missing feature files for {base_filename}, skipping.")
            continue

        mfcc = np.load(mfcc_path)
        mel = np.load(mel_path)
        spectral_centroid = np.load(spectral_centroid_path)
        spectral_flatness = np.load(spectral_flatness_path)
        zcr = np.load(zcr_path)

        mfcc_features = np.concatenate(
            [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        mel_features = np.concatenate(
            [np.mean(mel, axis=1), np.std(mel, axis=1)])
        spectral_centroid_features = np.concatenate(
            [np.mean(spectral_centroid, axis=1), np.std(spectral_centroid, axis=1)])
        spectral_flatness_features = np.concatenate(
            [np.mean(spectral_flatness, axis=1), np.std(spectral_flatness, axis=1)])
        zcr_features = np.concatenate(
            [np.mean(zcr, axis=1), np.std(zcr, axis=1)])

        feature_vector = np.concatenate(
            [mfcc_features, mel_features, spectral_centroid_features, spectral_flatness_features, zcr_features])
        feature_vectors.append(feature_vector)
        file_names.append(base_filename)

    if not feature_vectors:
        print("No audio files with complete features found.")
        return

    X = np.array(feature_vectors)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if reduce_dim:
        pca_reducer = PCA(n_components=n_components, random_state=42)
        X = pca_reducer.fit_transform(X)

    ClusterModel = KMeans

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

    clustering_params = {
        'n_clusters': n_clusters,
        'init': 'k-means++',
        'n_init': 10,
        'random_state': 42
    }
    cluster_model = ClusterModel(**clustering_params)
    clusters = cluster_model.fit_predict(X)
    distances = np.linalg.norm(
        X - cluster_model.cluster_centers_[clusters], axis=1)
    print("\nCluster assignments:")
    for fname, cluster in zip(file_names, clusters):
        print(f"  {fname}: Cluster {cluster}")

    pca_vis = PCA(n_components=2, random_state=42)
    X_pca = pca_vis.fit_transform(X)
    df = pd.DataFrame({
        'Song': file_names,
        'Cluster': clusters,
        'Distance_to_Center': distances,
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1]
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'audio_clustering_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nSaved clustering results to CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                         cmap='viridis', s=100, alpha=0.7, picker=True)
    unique_clusters = np.unique(clusters)
    for cl in unique_clusters:
        indices = np.where(clusters == cl)[0]
        if len(indices) >= 3:
            points = X_pca[indices]
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.concatenate(
                (hull_points, hull_points[0:1]), axis=0)
            ax.plot(hull_points[:, 0], hull_points[:, 1], 'k-', lw=2)
        else:
            for idx in indices:
                ax.plot(X_pca[idx, 0], X_pca[idx, 1], 'ko',
                        markersize=12, fillstyle='none')

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("K-Means Clustering of Audio Features")
    plt.colorbar(scatter, ax=ax, label="Cluster")

    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points", bbox=dict(
        boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annotation(event_ind):
        idx = event_ind[0]
        x, y = X_pca[idx, 0], X_pca[idx, 1]
        song_name = df['Song'].iloc[idx]
        annot.xy = (x, y)
        annot.set_text(song_name)
        annot.set_visible(True)
        fig.canvas.draw_idle()

    def on_pick(event):
        if len(event.ind) > 0:
            update_annotation(event.ind)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results',
                          n_clusters=3,
                          reduce_dim=True,
                          n_components=50,
                          dynamic_cluster_selection=True,
                          dynamic_k_min=2,
                          dynamic_k_max=10)
