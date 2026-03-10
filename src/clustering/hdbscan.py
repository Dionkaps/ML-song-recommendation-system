import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

removed_script_dir = False
if str(SCRIPT_DIR) in sys.path:
	sys.path.remove(str(SCRIPT_DIR))
	removed_script_dir = True

import hdbscan  # type: ignore  # noqa: E402
from hdbscan.validity import validity_index  # type: ignore  # noqa: E402

if removed_script_dir:
	sys.path.insert(0, str(SCRIPT_DIR))

from config import feature_vars as fv
from src.ui.modern_ui import launch_ui

from src.clustering.kmeans import compute_visualization_coords, load_clustering_dataset


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


def _build_min_cluster_size_candidates(n_samples: int) -> List[int]:
	"""Build a practical HDBSCAN search space from dataset size."""

	candidates = {5, 8, 10, 15, 20, 30}
	for frac in (0.01, 0.015, 0.02, 0.03, 0.05, 0.08):
		candidates.add(max(2, int(round(n_samples * frac))))

	max_candidate = max(2, n_samples // 2)
	valid = sorted(value for value in candidates if 2 <= value <= max_candidate)
	if valid:
		return valid
	return [max(2, min(5, n_samples // 2))]


def _build_min_samples_candidates(min_cluster_size: int) -> List[int]:
	"""Search a few conservative and permissive density settings per cluster size."""

	candidates = {
		1,
		2,
		5,
		max(1, min_cluster_size // 4),
		max(1, min_cluster_size // 2),
		min_cluster_size,
	}
	return sorted(value for value in candidates if 1 <= value <= min_cluster_size)


def _select_hdbscan_model(
	data: np.ndarray,
	min_cluster_sizes: List[int],
	cluster_selection_epsilon: float,
	allow_single_cluster: bool,
) -> Tuple[hdbscan.HDBSCAN, List[Dict[str, float]], int, int]:
	"""Pick HDBSCAN density parameters using DBCV with a light noise penalty."""

	sample_size = min(5000, len(data))
	data64 = np.asarray(data, dtype=np.float64)
	best_model: Optional[hdbscan.HDBSCAN] = None
	best_params = (min_cluster_sizes[0], _build_min_samples_candidates(min_cluster_sizes[0])[0])
	best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
	diagnostics: List[Dict[str, float]] = []

	for min_cluster_size in min_cluster_sizes:
		for min_samples in _build_min_samples_candidates(min_cluster_size):
			try:
				model = hdbscan.HDBSCAN(
					min_cluster_size=min_cluster_size,
					min_samples=min_samples,
					cluster_selection_epsilon=cluster_selection_epsilon,
					allow_single_cluster=allow_single_cluster,
					prediction_data=True,
				)
				labels = model.fit_predict(data)
				n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
				noise_fraction = float(np.mean(labels == -1))
				dbcv = float("nan")
				silhouette = float("nan")
				score = float("-inf")

				if n_clusters >= 2:
					dbcv = float(validity_index(data64, labels))
					mask = labels != -1
					if mask.sum() > n_clusters:
						silhouette_kwargs = {"random_state": 42}
						if sample_size < mask.sum():
							silhouette_kwargs["sample_size"] = sample_size
						silhouette = float(
							silhouette_score(data64[mask], labels[mask], **silhouette_kwargs)
						)
					score = dbcv - (0.20 * noise_fraction) + (0.05 * np.nan_to_num(silhouette, nan=0.0))

				diagnostics.append(
					{
						"min_cluster_size": float(min_cluster_size),
						"min_samples": float(min_samples),
						"clusters": float(n_clusters),
						"noise_fraction": noise_fraction,
						"dbcv": dbcv,
						"silhouette": silhouette,
						"score": score,
					}
				)

				selection_key = (
					score,
					-noise_fraction,
					np.nan_to_num(silhouette, nan=-1.0),
					-float(min_cluster_size),
					-float(min_samples),
				)
				if selection_key > best_key:
					best_key = selection_key
					best_model = model
					best_params = (min_cluster_size, min_samples)
			except Exception:
				diagnostics.append(
					{
						"min_cluster_size": float(min_cluster_size),
						"min_samples": float(min_samples),
						"clusters": float("nan"),
						"noise_fraction": float("nan"),
						"dbcv": float("nan"),
						"silhouette": float("nan"),
						"score": float("-inf"),
					}
				)

	if best_model is None:
		fallback_size, fallback_samples = best_params
		best_model = hdbscan.HDBSCAN(
			min_cluster_size=fallback_size,
			min_samples=fallback_samples,
			cluster_selection_epsilon=cluster_selection_epsilon,
			allow_single_cluster=allow_single_cluster,
			prediction_data=True,
		).fit(data)
		return best_model, diagnostics, fallback_size, fallback_samples

	return best_model, diagnostics, best_params[0], best_params[1]


def run_hdbscan_clustering(
	audio_dir: str = "audio_files",
	results_dir: str = "output/features",
	min_cluster_size: int = 10,
	min_samples: Optional[int] = None,
	dynamic_parameter_selection: bool = True,
	dynamic_min_cluster_sizes: Optional[List[int]] = None,
	cluster_selection_epsilon: float = 0.0,
	allow_single_cluster: bool = False,
	n_mfcc: int = fv.n_mfcc,
	n_mels: int = fv.n_mels,
	include_genre: bool = fv.include_genre,
	include_msd: bool = fv.include_msd_features,
	songs_csv_path: Optional[str] = None,
):
	os.makedirs(results_dir, exist_ok=True)

	file_names, genres, unique_genres, X_prepared = load_clustering_dataset(
		audio_dir=audio_dir,
		results_dir=results_dir,
		n_mfcc=n_mfcc,
		n_mels=n_mels,
		include_genre=include_genre,
		include_msd=include_msd,
		songs_csv_path=songs_csv_path,
	)

	selection_rows: Optional[List[Dict[str, float]]] = None

	if dynamic_parameter_selection:
		n_samples = X_prepared.shape[0]
		min_cluster_sizes = (
			dynamic_min_cluster_sizes
			if dynamic_min_cluster_sizes is not None
			else _build_min_cluster_size_candidates(n_samples)
		)
		print(
			f"Dynamic HDBSCAN selection: searching min_cluster_size in {min_cluster_sizes}"
		)
		clusterer, selection_rows, min_cluster_size, min_samples = _select_hdbscan_model(
			X_prepared,
			min_cluster_sizes=min_cluster_sizes,
			cluster_selection_epsilon=cluster_selection_epsilon,
			allow_single_cluster=allow_single_cluster,
		)
		print(
			f"Selected HDBSCAN parameters -> min_cluster_size={min_cluster_size}, "
			f"min_samples={min_samples}"
		)
	else:
		clusterer = hdbscan.HDBSCAN(
			min_cluster_size=min_cluster_size,
			min_samples=min_samples,
			cluster_selection_epsilon=cluster_selection_epsilon,
			allow_single_cluster=allow_single_cluster,
			prediction_data=True,
		)
		clusterer.fit(X_prepared)

	labels = clusterer.labels_
	probabilities = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

	centers = _compute_cluster_centers(X_prepared, labels)
	distances = _compute_distances(X_prepared, labels, centers)

	coords = compute_visualization_coords(X_prepared)

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

	output_dir = Path("output/clustering_results")
	metrics_dir = Path("output/metrics")
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_dir.mkdir(parents=True, exist_ok=True)
	if selection_rows is not None:
		selection_df = pd.DataFrame(selection_rows)
		selection_path = metrics_dir / "hdbscan_selection_criteria.csv"
		selection_df.to_csv(selection_path, index=False)
		print(f"Stored HDBSCAN selection diagnostics -> {selection_path}")
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
		results_dir="output/features",
		min_cluster_size=10,
		dynamic_parameter_selection=True,
		include_genre=fv.include_genre,
	)

	launch_ui(DF, COORDS, LABELS, audio_dir="audio_files", clustering_method="HDBSCAN")
