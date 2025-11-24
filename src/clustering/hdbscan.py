import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

removed_script_dir = False
if str(SCRIPT_DIR) in sys.path:
	sys.path.remove(str(SCRIPT_DIR))
	removed_script_dir = True

import hdbscan  # type: ignore  # noqa: E402

if removed_script_dir:
	sys.path.insert(0, str(SCRIPT_DIR))

from config import feature_vars as fv
from src.ui.modern_ui import launch_ui

from src.clustering.kmeans import _collect_feature_vectors, _load_genre_mapping, build_group_weights


def _compute_cluster_centers(
	data: np.ndarray, labels: np.ndarray
) -> Dict[int, np.ndarray]:
	"""Calculate centroids for each HDBSCAN cluster (excluding noise)."""

	centers: Dict[int, np.ndarray] = {}
	unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
	for lbl in unique_labels:
		members = data[labels == lbl]
		if members.size:
			centers[lbl] = members.mean(axis=0)
	return centers


def _compute_distances(
	data: np.ndarray,
	labels: np.ndarray,
	centers: Dict[int, np.ndarray],
) -> np.ndarray:
	"""Distance to assigned centroid; NaN for noise points."""

	distances = np.full(len(data), np.nan, dtype=float)
	for idx, lbl in enumerate(labels):
		if lbl == -1:
			continue
		center = centers.get(lbl)
		if center is not None:
			distances[idx] = np.linalg.norm(data[idx] - center)
	return distances


def run_hdbscan_clustering(
	audio_dir: str = "audio_files",
	results_dir: str = "output/results",
	min_cluster_size: int = 10,
	min_samples: Optional[int] = None,
	cluster_selection_epsilon: float = 0.0,
	allow_single_cluster: bool = False,
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

	weights = build_group_weights(
		n_mfcc=n_mfcc, 
		n_mels=n_mels, 
		n_genres=len(unique_genres),
		include_genre=include_genre
	)
	if X_scaled.shape[1] != len(weights):
		raise ValueError(
			f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}"
		)
	X_weighted = X_scaled * weights

	clusterer = hdbscan.HDBSCAN(
		min_cluster_size=min_cluster_size,
		min_samples=min_samples,
		cluster_selection_epsilon=cluster_selection_epsilon,
		allow_single_cluster=allow_single_cluster,
		prediction_data=True,
	)

	labels = clusterer.fit_predict(X_weighted)
	probabilities = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

	centers = _compute_cluster_centers(X_weighted, labels)
	distances = _compute_distances(X_weighted, labels, centers)

	coords = PCA(n_components=2, random_state=42).fit_transform(X_weighted)

	df = pd.DataFrame(
		{
			"Song": file_names,
			"Genre": genres,
			"Cluster": labels,
			"Probability": probabilities,
			"Distance": distances,
			"PCA1": coords[:, 0],
			"PCA2": coords[:, 1],
		}
	)

	output_dir = Path("output")
	output_dir.mkdir(exist_ok=True)
	csv_path = output_dir / "audio_clustering_results_hdbscan.csv"
	df.to_csv(csv_path, index=False)
	print(f"Results written to -> {csv_path}")

	noise_pct = (labels == -1).mean() * 100.0
	print(
		f"HDBSCAN formed {len(set(labels)) - (1 if -1 in labels else 0)} clusters; "
		f"noise points: {noise_pct:.1f}%"
	)

	return df, coords, labels
if __name__ == "__main__":
	DF, COORDS, LABELS = run_hdbscan_clustering(
		audio_dir="audio_files",
		results_dir="output/results",
		min_cluster_size=10,
		include_genre=fv.include_genre,
	)

	launch_ui(DF, COORDS, LABELS, audio_dir="audio_files", clustering_method="HDBSCAN")
