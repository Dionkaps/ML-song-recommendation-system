import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import feature_vars as fv
from src.clustering.gmm import run_gmm_clustering
from src.clustering.hdbscan import run_hdbscan_clustering
from src.clustering.kmeans import run_kmeans_clustering
from src.clustering.vade import run_vade_clustering


def main():
    common = {
        "audio_dir": "audio_files",
        "results_dir": "output/features",
        "include_genre": fv.include_genre,
        "include_msd": fv.include_msd_features,
        "songs_csv_path": "data/songs.csv",
    }

    summary = {}

    kmeans_df, _, kmeans_labels = run_kmeans_clustering(
        dynamic_cluster_selection=True,
        **common,
    )
    summary["kmeans"] = {
        "rows": int(len(kmeans_df)),
        "clusters": int(len(set(map(int, kmeans_labels)))),
    }

    gmm_df, _, gmm_labels = run_gmm_clustering(
        dynamic_component_selection=True,
        **common,
    )
    summary["gmm"] = {
        "rows": int(len(gmm_df)),
        "clusters": int(len(set(map(int, gmm_labels)))),
    }

    hdbscan_df, _, hdbscan_labels = run_hdbscan_clustering(
        dynamic_parameter_selection=True,
        **common,
    )
    summary["hdbscan"] = {
        "rows": int(len(hdbscan_df)),
        "clusters": int(len({int(x) for x in hdbscan_labels if int(x) != -1})),
        "noise": int(sum(1 for x in hdbscan_labels if int(x) == -1)),
    }

    vade_df, _, vade_labels = run_vade_clustering(
        dynamic_component_selection=True,
        **common,
    )
    summary["vade"] = {
        "rows": int(len(vade_df)),
        "clusters": int(len(set(map(int, vade_labels)))),
    }

    print("\nClustering stage finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
