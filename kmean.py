import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # optional


def run_kmeans_clustering(audio_dir='audio_files', results_dir='results', n_clusters=3):
    """
    Creates a feature vector for each audio file by summarizing its extracted features,
    and then runs KMeans clustering on the collection of feature vectors.

    Parameters:
        audio_dir (str): Directory where the original audio (.wav) files are stored.
        results_dir (str): Directory where the extracted feature .npy files are stored.
        n_clusters (int): Number of clusters to form.
    """
    # List all .wav files to get the base filenames
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print("No audio files found in the audio directory.")
        return

    feature_vectors = []
    file_names = []

    for wav_path in wav_files:
        base_filename = os.path.splitext(os.path.basename(wav_path))[0]
        # Paths to the feature files
        mfcc_path = os.path.join(results_dir, f"{base_filename}_mfcc.npy")
        mel_path = os.path.join(
            results_dir, f"{base_filename}_melspectrogram.npy")
        spectral_centroid_path = os.path.join(
            results_dir, f"{base_filename}_spectral_centroid.npy")
        spectral_flatness_path = os.path.join(
            results_dir, f"{base_filename}_spectral_flatness.npy")
        zcr_path = os.path.join(
            results_dir, f"{base_filename}_zero_crossing_rate.npy")

        # Skip files that don't have complete features
        if not all(os.path.isfile(p) for p in [mfcc_path, mel_path, spectral_centroid_path, spectral_flatness_path, zcr_path]):
            print(f"Missing feature files for {base_filename}, skipping.")
            continue

        # Load features
        mfcc = np.load(mfcc_path)                   # shape: (n_mfcc, T)
        mel = np.load(mel_path)                     # shape: (n_mels, T)
        spectral_centroid = np.load(spectral_centroid_path)  # shape: (1, T)
        spectral_flatness = np.load(spectral_flatness_path)  # shape: (1, T)
        zcr = np.load(zcr_path)                     # shape: (1, T)

        # Compute summary statistics (mean and std) along the time axis (axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_features = np.concatenate([mfcc_mean, mfcc_std])

        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        mel_features = np.concatenate([mel_mean, mel_std])

        spectral_centroid_mean = np.mean(spectral_centroid, axis=1)
        spectral_centroid_std = np.std(spectral_centroid, axis=1)
        spectral_centroid_features = np.concatenate(
            [spectral_centroid_mean, spectral_centroid_std])

        spectral_flatness_mean = np.mean(spectral_flatness, axis=1)
        spectral_flatness_std = np.std(spectral_flatness, axis=1)
        spectral_flatness_features = np.concatenate(
            [spectral_flatness_mean, spectral_flatness_std])

        zcr_mean = np.mean(zcr, axis=1)
        zcr_std = np.std(zcr, axis=1)
        zcr_features = np.concatenate([zcr_mean, zcr_std])

        # Combine all features into a single feature vector
        feature_vector = np.concatenate([
            mfcc_features,
            mel_features,
            spectral_centroid_features,
            spectral_flatness_features,
            zcr_features
        ])

        feature_vectors.append(feature_vector)
        file_names.append(base_filename)

    if len(feature_vectors) == 0:
        print("No audio files with complete features found.")
        return

    # Stack the feature vectors into a 2D array (samples x features)
    X = np.array(feature_vectors)

    # (Optional) Standardize features if needed:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Display cluster assignments
    print("\nCluster assignments:")
    for fname, cluster in zip(file_names, clusters):
        print(f"  {fname}: Cluster {cluster}")

    # Optional: Visualize clusters using PCA (reduce to 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=clusters, cmap='viridis', s=100, alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clustering of Audio Features")
    plt.colorbar(scatter, label="Cluster")
    # Optionally, annotate points with filenames
    for i, txt in enumerate(file_names):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.8)
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results', n_clusters=3)
