import os
import glob
import warnings
import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import feature_vars as fv
from src.ui.modern_ui import launch_ui


# -----------------------------
# Existing helper (unchanged)
# -----------------------------
def build_group_weights(n_mfcc: int = fv.n_mfcc, n_mels: int = fv.n_mels, include_genre: bool = True) -> np.ndarray:
    """Build static group weights so that each feature group contributes ~equally.
    NOTE: When using WKBSC (learned per-dimension weights), we skip these to avoid double-weighting.
    """
    if include_genre:
        # Assume 10 one-hot genres (e.g., GTZAN). If your dataset has a different count,
        # this function is only used by the classic sklearn KMeans path below anyway.
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, 10]
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2]

    total_dims = sum(group_sizes)
    w = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for g in group_sizes:
        w[idx:idx + g] /= np.sqrt(g)
        idx += g
    return w


# -----------------------------
# WKBSC implementation (parallel + correct)
# -----------------------------

def _weighted_dist_and_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-12):
    """
    Distance in weighted space: d = || w ⊙ (x - y) ||_2
    Gradient wrt w_d: ∂d/∂w_d = (w_d * (x_d - y_d)^2) / d
    """
    diff = x - y
    wd = w * diff
    dist = np.linalg.norm(wd)
    if dist < eps:
        return 0.0, np.zeros_like(w)
    grad = (w * (diff ** 2)) / dist
    return float(dist), grad


def _ai_bi_for_point(i: int, labels: np.ndarray, clusters: dict, X: np.ndarray, w: np.ndarray, unique_labels: np.ndarray, D: int):
    """Compute a(i), b(i) and their gradients wrt w for a single point i."""
    ai_i = 0.0
    bi_i = 0.0
    dai_i = np.zeros(D, dtype=float)
    dbi_i = np.zeros(D, dtype=float)

    c = labels[i]
    idxs = clusters[c]
    xi = X[i]

    # a(i): same-cluster mean distance (exclude i)
    if len(idxs) > 1:
        d_sum = 0.0
        g_sum = np.zeros(D, dtype=float)
        count = 0
        for j in idxs:
            if j == i:
                continue
            d_ij, g_ij = _weighted_dist_and_grad(xi, X[j], w)
            d_sum += d_ij
            g_sum += g_ij
            count += 1
        if count > 0:
            ai_i = d_sum / count
            dai_i = g_sum / count

    # b(i): nearest other cluster by mean distance
    best_mean = None
    best_grad = None
    for c2 in unique_labels:
        if c2 == c:
            continue
        idxs2 = clusters[c2]
        if len(idxs2) == 0:
            continue
        d_sum2 = 0.0
        g_sum2 = np.zeros(D, dtype=float)
        count2 = 0
        for j in idxs2:
            d_ij, g_ij = _weighted_dist_and_grad(xi, X[j], w)
            d_sum2 += d_ij
            g_sum2 += g_ij
            count2 += 1
        if count2 > 0:
            mean2 = d_sum2 / count2
            if best_mean is None or mean2 < best_mean:
                best_mean = mean2
                best_grad = g_sum2 / count2

    if best_mean is not None:
        bi_i = best_mean
        dbi_i = best_grad

    return ai_i, bi_i, dai_i, dbi_i


def _compute_ai_bi_and_grads_parallel(X: np.ndarray, labels: np.ndarray, w: np.ndarray, n_jobs: int = -1, prefer_threads: bool = True):
    """
    Parallel over points to compute:
      - ai, bi vectors
      - dai, dbi gradients wrt w
    """
    N, D = X.shape
    unique_labels = np.unique(labels)
    clusters = {c: np.where(labels == c)[0] for c in unique_labels}

    prefer = "threads" if prefer_threads else "processes"
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_ai_bi_for_point)(i, labels, clusters, X, w, unique_labels, D)
        for i in range(N)
    )

    ai = np.empty(N, dtype=float)
    bi = np.empty(N, dtype=float)
    dai = np.empty((N, D), dtype=float)
    dbi = np.empty((N, D), dtype=float)

    for i, (ai_i, bi_i, dai_i, dbi_i) in enumerate(results):
        ai[i] = ai_i
        bi[i] = bi_i
        dai[i] = dai_i
        dbi[i] = dbi_i

    return ai, bi, dai, dbi


def _silhouette_and_grad(ai: np.ndarray, bi: np.ndarray, dai: np.ndarray, dbi: np.ndarray, eps: float = 1e-12):
    """
    s_i = (b - a) / max(a, b)
    Piecewise gradient:
      if a < b: s = 1 - a/b → ds = -(1/b)*da + (a/b^2)*db
      else:     s = b/a - 1 → ds = (1/a)*db - (b/a^2)*da
    Returns mean silhouette (scalar) and mean gradient wrt w (vector D).
    """
    N, D = dai.shape
    s = np.zeros(N, dtype=float)
    ds = np.zeros((N, D), dtype=float)

    a = ai
    b = bi

    # a < b branch
    mask = a + eps < b
    if np.any(mask):
        b_m = np.clip(b[mask], eps, None)
        s[mask] = 1.0 - (a[mask] / b_m)
        ds[mask] = -(dai[mask] / b_m[:, None]) + \
            ((a[mask] / (b_m ** 2))[:, None]) * dbi[mask]

    # a >= b branch
    mask2 = ~mask
    if np.any(mask2):
        a_m = np.clip(a[mask2], eps, None)
        s[mask2] = (b[mask2] / a_m) - 1.0
        ds[mask2] = (dbi[mask2] / a_m[:, None]) - \
            ((b[mask2] / (a_m ** 2))[:, None]) * dai[mask2]

    return float(np.mean(s)), np.mean(ds, axis=0)


def WKBSC(
    X: np.ndarray,
    K: int,
    n: float = 0.01,
    floss_min: float = 0.999,
    floss_max: float = 1.001,
    max_iters: int = 100,
    max_weight_updates: int = 200,
    random_state: int | None = 42,
    n_jobs: int = -1,                 # <— use all cores by default
    # <— limit underlying BLAS/OpenMP threads (None = leave as-is)
    blas_threads: int | None = None,
    prefer_threads: bool = True,      # <— joblib backend preference
):
    """
    Weighted K-Means with per-dimension weights learned by maximizing mean silhouette.

    Parallelism:
      - Python-level parallelism over points via joblib (n_jobs)
      - Optional control of BLAS/OpenMP threads via threadpoolctl (blas_threads)

    Returns
    -------
    labels : np.ndarray [N]
    centers : np.ndarray [K, D]   (centroids in the *weighted* space)
    weights : np.ndarray [D]
    Xw : np.ndarray [N, D]        (weighted data used for clustering)
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    N, D = X.shape

    # init non-negative weights that sum to 1
    w = np.ones(D, dtype=float) / D

    # init centers from the (initially same) weighted space
    init_idx = rng.choice(N, K, replace=False)
    centers = X[init_idx] * w  # store centers in weighted space
    labels = np.zeros(N, dtype=int)

    S_hist: list[float] = []
    outer = 0

    # Manage BLAS/OpenMP threads to avoid oversubscription
    tp_ctx = threadpool_limits(
        blas_threads) if blas_threads is not None else nullcontext()

    with tp_ctx:
        while True:
            # weighted data
            Xw = X * w

            # ------ Lloyd steps in weighted space ------
            for _ in range(max_iters):
                # assign
                distances = np.linalg.norm(
                    Xw[:, None, :] - centers[None, :, :], axis=-1)
                new_labels = distances.argmin(axis=1)
                if np.array_equal(new_labels, labels):
                    labels = new_labels
                    break
                labels = new_labels
                # update centers (protect empty clusters)
                new_centers = []
                for k in range(K):
                    mask = labels == k
                    if np.any(mask):
                        new_centers.append(Xw[mask].mean(axis=0))
                    else:
                        new_centers.append(centers[k])
                centers = np.vstack(new_centers)

            # ------ silhouette + gradient wrt weights (PARALLEL) ------
            ai, bi, dai, dbi = _compute_ai_bi_and_grads_parallel(
                X, labels, w, n_jobs=n_jobs, prefer_threads=prefer_threads
            )
            S_mean, dS_dw = _silhouette_and_grad(ai, bi, dai, dbi)

            # gradient ascent on mean silhouette
            w = w + n * dS_dw

            # keep weights non-negative and normalized
            w = np.maximum(w, 1e-12)
            w = w / w.sum()

            # convergence check (ratio window on mean silhouette)
            S_hist.append(S_mean)
            if len(S_hist) > 1:
                prev = S_hist[-2]
                curr = S_hist[-1]
                if prev != 0 and np.sign(prev) == np.sign(curr):
                    ratio = curr / prev
                    if floss_min < ratio < floss_max:
                        return labels, centers, w, Xw

            outer += 1
            if outer >= max_weight_updates:
                warnings.warn(
                    "WKBSC: Reached max_weight_updates without hitting silhouette ratio window; returning current state.",
                    RuntimeWarning,
                )
                return labels, centers, w, Xw


# -----------------------------
# Pipeline entry used by run_pipeline.py
# -----------------------------

def run_kmeans_clustering(
    audio_dir: str = "genres_original",
    results_dir: str = "results",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = False,
    dynamic_k_min: int = 2,
    dynamic_k_max: int = 10,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
    include_genre: bool = True,
    algorithm: str = "wkbsc",  # "wkbsc" or "sklearn"
    wkbsc_step: float = 0.01,
    wkbsc_floss_min: float = 0.999,
    wkbsc_floss_max: float = 1.001,
    wkbsc_max_lloyd: int = 100,
    wkbsc_max_weight_updates: int = 200,
    n_jobs: int = -1,                # <— parallel over points
    # <— control BLAS threads; e.g., set to 1 when n_jobs != 1
    blas_threads: int | None = None,
    prefer_threads: bool = True,     # <— prefer "threads" for shared-memory numpy ops
):
    """Run clustering and produce a results CSV + coordinates for plotting/UI.

    algorithm = "wkbsc" uses the learned-weight K-Means; "sklearn" keeps the old behavior.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Load or derive genre mapping
    genre_map_path = os.path.join(results_dir, "genre_map.npy")
    genre_list_path = os.path.join(results_dir, "genre_list.npy")

    if os.path.exists(genre_map_path) and include_genre:
        genre_map = np.load(genre_map_path, allow_pickle=True).item()
        print(f"Loaded genre mapping for {len(genre_map)} files")
    else:
        genre_map = {}
        genre_dirs = [d for d in glob.glob(
            os.path.join(audio_dir, "*")) if os.path.isdir(d)]
        for genre_dir in genre_dirs:
            genre = os.path.basename(genre_dir)
            wav_files = glob.glob(os.path.join(genre_dir, "*.wav"))
            for wav_path in wav_files:
                base = Path(wav_path).stem
                genre_map[base] = genre
        print(f"Created genre mapping for {len(genre_map)} files")

    unique_genres = sorted(set(genre_map.values()))
    genre_to_idx = {genre: i for i, genre in enumerate(unique_genres)}
    np.save(genre_list_path, unique_genres)
    print(
        f"Found {len(unique_genres)} unique genres: {', '.join(unique_genres)}")

    # Collect feature vectors
    feature_files = glob.glob(os.path.join(results_dir, "*_mfcc.npy"))
    base_names = [os.path.basename(f).replace(
        "_mfcc.npy", "") for f in feature_files]

    file_names, feature_vectors, genres = [], [], []
    for base in base_names:
        feats = {k: os.path.join(results_dir, f"{base}_{k}.npy") for k in [
            "mfcc", "melspectrogram", "spectral_centroid",
            "zero_crossing_rate",
        ]}
        if not all(os.path.isfile(p) for p in feats.values()):
            continue

        # Determine genre for this file
        if base in genre_map:
            genre = genre_map[base]
        else:
            parts = base.split('.')
            if len(parts) > 0 and parts[0] in unique_genres:
                genre = parts[0]
            else:
                print(
                    f"Warning: Could not determine genre for {base}, skipping")
                continue
        genres.append(genre)

        arrays = [np.load(p) for p in feats.values()]
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

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_all = np.vstack(feature_vectors)
    X_scaled = StandardScaler().fit_transform(X_all)

    if algorithm == "sklearn":
        # Old behavior: apply static group weights to equalize contributions
        weights = build_group_weights(
            n_mfcc=n_mfcc, n_mels=n_mels, include_genre=include_genre)
        if X_scaled.shape[1] != len(weights):
            raise ValueError(
                f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}")
        X_for_cluster = X_scaled * weights
    else:
        # WKBSC learns per-dimension weights, so we feed only standardized data
        X_for_cluster = X_scaled

    # Optionally choose k dynamically
    best_pack = None
    if dynamic_cluster_selection:
        best_k = None
        best_sil = -np.inf
        for k in range(dynamic_k_min, dynamic_k_max + 1):
            if algorithm == "sklearn":
                lbls_tmp = KMeans(n_clusters=k, random_state=42,
                                  n_init=10).fit_predict(X_for_cluster)
                n_labels = len(np.unique(lbls_tmp))
                sil = silhouette_score(X_for_cluster, lbls_tmp) if (
                    2 <= n_labels <= X_for_cluster.shape[0] - 1
                ) else -np.inf
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_pack = (lbls_tmp, None, None, X_for_cluster)
            else:
                lbls_tmp, centers_tmp, w_tmp, Xw_tmp = WKBSC(
                    X_for_cluster,
                    k,
                    n=wkbsc_step,
                    floss_min=wkbsc_floss_min,
                    floss_max=wkbsc_floss_max,
                    max_iters=wkbsc_max_lloyd,
                    max_weight_updates=wkbsc_max_weight_updates,
                    random_state=42,
                    n_jobs=n_jobs,
                    blas_threads=blas_threads,
                    prefer_threads=prefer_threads,
                )
                n_labels = len(np.unique(lbls_tmp))
                sil = silhouette_score(Xw_tmp, lbls_tmp) if (
                    2 <= n_labels <= Xw_tmp.shape[0] - 1
                ) else -np.inf
                if sil > best_sil:
                    best_sil = sil
                    best_k = k
                    best_pack = (lbls_tmp, centers_tmp, w_tmp, Xw_tmp)
        n_clusters = best_k if best_k is not None else n_clusters
        print(f"Optimal k (silhouette) → {n_clusters}")

    # Final clustering with chosen k
    if algorithm == "sklearn":
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_for_cluster)
        centers = km.cluster_centers_
        X_weighted_for_metrics = X_for_cluster
        learned_weights = None
    else:
        if best_pack is not None and best_pack[0] is not None and len(best_pack[0]) == X_for_cluster.shape[0]:
            labels, centers, learned_weights, X_weighted_for_metrics = best_pack
        else:
            labels, centers, learned_weights, X_weighted_for_metrics = WKBSC(
                X_for_cluster,
                n_clusters,
                n=wkbsc_step,
                floss_min=wkbsc_floss_min,
                floss_max=wkbsc_floss_max,
                max_iters=wkbsc_max_lloyd,
                max_weight_updates=wkbsc_max_weight_updates,
                random_state=42,
                n_jobs=n_jobs,
                blas_threads=blas_threads,
                prefer_threads=prefer_threads,
            )

    # 2D coordinates for plotting/UI (computed in the space used to cluster)
    coords = PCA(n_components=2, random_state=42).fit_transform(
        X_weighted_for_metrics)

    # Build output table
    distances = np.linalg.norm(
        X_weighted_for_metrics - centers[labels], axis=1)
    df = pd.DataFrame({
        "Song": file_names,
        "Genre": genres,
        "Cluster": labels,
        "Distance": distances,
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
    })

    csv_path = os.path.join("audio_clustering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results written to → {csv_path}")

    if algorithm == "wkbsc" and learned_weights is not None:
        # Persist learned weights for inspection
        np.save(os.path.join(results_dir, "wkbsc_feature_weights.npy"),
                learned_weights)
        print("Saved learned per-dimension weights to results/wkbsc_feature_weights.npy")

    return df, coords, labels


if __name__ == "__main__":
    # Default to your WKBSC algorithm. Switch algorithm="sklearn" to use classic KMeans.
    DF, COORDS, LABELS = run_kmeans_clustering(
        audio_dir="genres_original",
        results_dir="output/results",
        n_clusters=5,  # More clusters for typical 10-genre datasets
        dynamic_cluster_selection=True,
        include_genre=True,
        algorithm="wkbsc",
        wkbsc_step=0.01,
        wkbsc_floss_min=0.999,
        wkbsc_floss_max=1.001,
        wkbsc_max_lloyd=100,
        wkbsc_max_weight_updates=200,
        n_jobs=-1,          # <— use all cores
        blas_threads=1,     # <— good default to avoid nested oversubscription when using threads
        prefer_threads=True
    )

    # Launch the UI
    launch_ui(DF, COORDS, LABELS, audio_dir="genres_original",
              clustering_method="K-means (WKBSC, parallel)")
