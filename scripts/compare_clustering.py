import os
import itertools
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import feature_vars as fv
except Exception:
    class _FV:
        n_mfcc = 13
        n_mels = 128
    fv = _FV()


RESULTS_DIR = "output/results"
KMEANS_PATH = "output/audio_clustering_results.csv"
DBSCAN_PATH = os.path.join("output/dbscan", "dbscan_clustering_results.csv")
HIER_PATH = os.path.join("output/hierarchical", "hierarchical_clustering_results.csv")


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_clustering_csv(path: str, method_name: str) -> pd.DataFrame:
    """Load a clustering results CSV and normalize columns.

    Expected columns: Song, Genre, Cluster, PCA1, PCA2 (Distance optional).
    Returns a DataFrame with required columns and an added Method column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results for {method_name}: {path}")
    df = pd.read_csv(path)

    required = {"Song", "Genre", "Cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{method_name} CSV missing required columns: {', '.join(sorted(missing))}"
        )

    # Normalize types
    # Keep cluster labels as they are (DBSCAN may have -1 for noise)
    # Some CSVs might store as float; cast to int when safe, else leave as object
    try:
        df["Cluster"] = df["Cluster"].astype(int)
    except Exception:
        # Leave as-is; sklearn can handle mixed labels as strings
        df["Cluster"] = df["Cluster"].astype(str)

    # Ensure PCA columns exist (optional)
    for col in ("PCA1", "PCA2"):
        if col not in df.columns:
            df[col] = np.nan

    # Deduplicate by song, keep first
    if df["Song"].duplicated().any():
        df = df.drop_duplicates(subset=["Song"], keep="first").reset_index(drop=True)

    df["Method"] = method_name
    return df


def cluster_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of cluster sizes for a method."""
    sizes = (
        df.groupby("Cluster")["Song"].count().rename("Count").reset_index().sort_values("Cluster")
    )
    return sizes


def genre_alignment_metrics(df: pd.DataFrame) -> dict:
    """Compute genre alignment metrics between cluster labels and provided Genre.

    Returns homogeneity, completeness, v_measure, ARI, NMI, and purity metrics.
    """
    y_true = df["Genre"].astype(str).to_numpy()
    y_pred = df["Cluster"].to_numpy()

    h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Purity: sum over clusters of max genre count / total
    contingency = pd.crosstab(df["Cluster"], df["Genre"])
    majority_sum = contingency.max(axis=1).sum()
    purity_micro = majority_sum / len(df) if len(df) else np.nan
    purity_macro = (contingency.max(axis=1) / contingency.sum(axis=1)).mean()

    # Silhouette on PCA space if available (rough proxy)
    sil = np.nan
    if df[["PCA1", "PCA2"]].notna().all().all():
        try:
            X = df[["PCA1", "PCA2"]].to_numpy()
            labels = df["Cluster"].to_numpy()
            # Need at least 2 clusters present
            if len(np.unique(labels)) >= 2:
                sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan

    return {
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
        "ARI_vs_genre": ari,
        "NMI_vs_genre": nmi,
        "purity_micro": purity_micro,
        "purity_macro": purity_macro,
        "silhouette_on_PCA": sil,
    }


def pairwise_agreement(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> Tuple[pd.DataFrame, dict]:
    """Compute pairwise agreement metrics on intersection of Songs between two methods.

    Returns contingency table and a metrics dict.
    """
    merged = a.merge(b, on=["Song"], suffixes=(f"_{name_a}", f"_{name_b}"))
    if merged.empty:
        return pd.DataFrame(), {
            "intersection": 0,
            "ARI": np.nan,
            "NMI": np.nan,
            "v_measure": np.nan,
            "homogeneity(A|B)": np.nan,
            "homogeneity(B|A)": np.nan,
        }

    ya = merged[f"Cluster_{name_a}"].to_numpy()
    yb = merged[f"Cluster_{name_b}"].to_numpy()

    ari = adjusted_rand_score(ya, yb)
    nmi = normalized_mutual_info_score(ya, yb)
    h_ab, c_ab, v_ab = homogeneity_completeness_v_measure(ya, yb)
    h_ba, c_ba, v_ba = homogeneity_completeness_v_measure(yb, ya)

    # Contingency table
    contingency = pd.crosstab(merged[f"Cluster_{name_a}"], merged[f"Cluster_{name_b}"])
    return contingency, {
        "intersection": len(merged),
        "ARI": ari,
        "NMI": nmi,
        "v_measure": v_ab,  # symmetric, equals v_ba
        "homogeneity(A|B)": h_ab,
        "homogeneity(B|A)": h_ba,
    }


def dbscan_exclude_noise(df: pd.DataFrame) -> pd.DataFrame:
    """If the method is DBSCAN (label may include -1), return a view excluding noise."""
    if df["Method"].iloc[0].lower() != "dbscan":
        return df
    # If clusters are not integers (casting failed), try to coerce
    labels = df["Cluster"]
    try:
        mask = labels.astype(int) != -1
    except Exception:
        # If not convertible, treat all as non-noise
        mask = pd.Series([True] * len(df), index=df.index)
    return df.loc[mask].reset_index(drop=True)


def build_group_weights(n_mfcc: int, n_mels: int, include_genre: bool, n_genres: int) -> np.ndarray:
    """Replicate weighting used in clustering scripts."""
    if include_genre:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2, n_genres]
    else:
        group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2]
    total_dims = sum(group_sizes)
    w = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for g in group_sizes:
        w[idx:idx + g] /= np.sqrt(g)
        idx += g
    return w


def reconstruct_feature_matrix(df: pd.DataFrame, results_dir: str = RESULTS_DIR, include_genre: bool = True,
                               n_mfcc: int = fv.n_mfcc, n_mels: int = fv.n_mels) -> np.ndarray:
    """Reconstruct feature vectors (mean+std pooling) like in clustering scripts, ordered as df."""
    # Build genre mapping from the DataFrame itself
    genres = sorted(df["Genre"].astype(str).unique().tolist())
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    feats_keys = ["mfcc", "melspectrogram", "spectral_centroid", "zero_crossing_rate"]
    vectors: List[np.ndarray] = []
    kept_rows = []
    for _, row in df.iterrows():
        base = row["Song"]
        paths = {k: os.path.join(results_dir, f"{base}_{k}.npy") for k in feats_keys}
        if not all(os.path.isfile(p) for p in paths.values()):
            # skip if any missing
            vectors.append(None)
            kept_rows.append(False)
            continue
        arrays = [np.load(paths[k]) for k in feats_keys]
        vec = np.concatenate([np.concatenate([arr.mean(axis=1), arr.std(axis=1)]) for arr in arrays])
        if include_genre:
            gvec = np.zeros(len(genres), dtype=np.float32)
            g = str(row["Genre"])
            if g in genre_to_idx:
                gvec[genre_to_idx[g]] = 1.0
            vec = np.concatenate([vec, gvec])
        vectors.append(vec.astype(np.float32))
        kept_rows.append(True)

    # Filter to rows with complete vectors
    mask = np.array(kept_rows, dtype=bool)
    if not mask.any():
        return np.empty((0, 0), dtype=np.float32)
    X = np.vstack([v for v, m in zip(vectors, kept_rows) if m])
    # Scale and apply weights similar to clustering
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    w = build_group_weights(n_mfcc=n_mfcc, n_mels=n_mels, include_genre=include_genre, n_genres=len(genres))
    if len(w) != Xs.shape[1]:
        # fallback without weights
        return Xs
    return Xs * w


def df_to_markdown_or_text(df: pd.DataFrame, index: bool = False) -> str:
    """Return a markdown table if supported, otherwise a plain text table in a code block."""
    try:
        # This requires 'tabulate' optional dependency
        return df.to_markdown(index=index)
    except Exception:
        return "\n".join([
            "````",
            df.to_string(index=index),
            "````",
        ])


def main() -> None:
    _ensure_results_dir()

    methods: Dict[str, str] = {
        "KMeans": KMEANS_PATH,
        "DBSCAN": DBSCAN_PATH,
        "Hierarchical": HIER_PATH,
    }

    loaded: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []
    for name, path in methods.items():
        try:
            df = load_clustering_csv(path, name)
            loaded[name] = df
        except Exception as e:
            errors.append(f"{name}: {e}")

    if not loaded:
        raise SystemExit(
            "No clustering results found. Run the clustering scripts first.\n" +
            "\n".join(errors)
        )

    # Per-method summaries and metrics
    per_method_rows = []
    for name, df in loaded.items():
        sizes = cluster_size_summary(df)
        sizes_path = os.path.join(RESULTS_DIR, f"cluster_sizes_{name.lower()}.csv")
        sizes.to_csv(sizes_path, index=False)

        # Compute metrics including DBSCAN noise
        metrics_all = genre_alignment_metrics(df)

        # For DBSCAN, also compute metrics excluding noise points
        metrics_excl = {}
        if name.lower() == "dbscan":
            df_wo = dbscan_exclude_noise(df)
            metrics_excl = genre_alignment_metrics(df_wo) if not df_wo.empty else {}
            # Count noise points
            try:
                n_noise = int((df["Cluster"].astype(int) == -1).sum())
            except Exception:
                n_noise = int((df["Cluster"].astype(str) == "-1").sum())
        else:
            n_noise = 0

        # Silhouette on reconstructed features (more faithful than PCA), with and without DBSCAN noise
        sil_feat = np.nan
        sil_feat_no_noise = np.nan
        try:
            Xf = reconstruct_feature_matrix(df, results_dir=RESULTS_DIR, include_genre=True,
                                            n_mfcc=fv.n_mfcc, n_mels=fv.n_mels)
            if Xf.size and len(np.unique(df["Cluster"])) >= 2:
                sil_feat = silhouette_score(Xf, df["Cluster"])  # may include -1 as a cluster
            if name.lower() == "dbscan":
                df_wo = dbscan_exclude_noise(df)
                if not df_wo.empty:
                    Xf_wo = reconstruct_feature_matrix(df_wo, results_dir=RESULTS_DIR, include_genre=True,
                                                       n_mfcc=fv.n_mfcc, n_mels=fv.n_mels)
                    if Xf_wo.size and len(np.unique(df_wo["Cluster"])) >= 2:
                        sil_feat_no_noise = silhouette_score(Xf_wo, df_wo["Cluster"])        
        except Exception:
            pass

        per_method_rows.append({
            "method": name,
            "n_items": len(df),
            "n_clusters": int(df["Cluster"].nunique()),
            "n_noise": n_noise,
            **{f"{k}": v for k, v in metrics_all.items()},
            **{f"{k}_no_noise": v for k, v in metrics_excl.items()},
            "silhouette_on_features": sil_feat,
            "silhouette_on_features_no_noise": sil_feat_no_noise,
        })

    per_method_df = pd.DataFrame(per_method_rows)
    per_method_csv = os.path.join(RESULTS_DIR, "clustering_per_method_metrics.csv")
    per_method_df.to_csv(per_method_csv, index=False)

    # Pairwise comparisons
    pair_rows = []
    for (name_a, df_a), (name_b, df_b) in itertools.combinations(loaded.items(), 2):
        contingency, metrics = pairwise_agreement(df_a, df_b, name_a, name_b)
        if not contingency.empty:
            cont_path = os.path.join(
                RESULTS_DIR, f"contingency_{name_a.lower()}_vs_{name_b.lower()}.csv"
            )
            contingency.to_csv(cont_path)
        pair_rows.append({"A": name_a, "B": name_b, **metrics})

    pair_df = pd.DataFrame(pair_rows)
    pair_csv = os.path.join(RESULTS_DIR, "clustering_pairwise_metrics.csv")
    pair_df.to_csv(pair_csv, index=False)

    # Markdown report
    report_lines = []
    report_lines.append("# Clustering Comparison Report\n")
    report_lines.append("\n## Available methods\n")
    report_lines.append("- " + "\n- ".join(sorted(loaded.keys())) + "\n")

    if errors:
        report_lines.append("\n## Missing/Errors\n")
        for e in errors:
            report_lines.append(f"- {e}")

    report_lines.append("\n## Per-method metrics (vs Genre)\n")
    report_lines.append(df_to_markdown_or_text(per_method_df, index=False))

    report_lines.append("\n## Pairwise agreement between methods\n")
    if not pair_df.empty:
        report_lines.append(df_to_markdown_or_text(pair_df, index=False))
    else:
        report_lines.append("No pairwise intersections found.")

    report_path = os.path.join(RESULTS_DIR, "clustering_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Console summary
    print("Saved per-method metrics to:", per_method_csv)
    print("Saved pairwise metrics to:", pair_csv)
    print("Saved markdown report to:", report_path)
    for name in loaded:
        print(f"Saved cluster size breakdown for {name}.")


if __name__ == "__main__":
    main()
