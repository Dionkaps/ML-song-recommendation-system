import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from config import feature_vars as fv

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk
except ImportError as exc:
    raise RuntimeError("Tkinter isn't available on your Python installation.") from exc

try:
    import pygame

    pygame.mixer.init()
except ImportError as exc:
    raise RuntimeError("Pygame isn't available. Install with 'pip install pygame'") from exc
except Exception as exc:
    raise RuntimeError(f"Failed to initialize pygame mixer: {exc}") from exc


def _parse_bounded_int(value: str, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer control value and clamp it to a safe range."""

    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _parse_bounded_float(
    value: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Parse a float control value and clamp it to a safe range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _align_artifact_rows(
    df: pd.DataFrame,
    labels: np.ndarray,
    payload: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Align a saved retrieval artifact to the current DataFrame row order."""

    artifact_songs = [str(song) for song in payload["songs"].tolist()]
    song_to_idx = {song: idx for idx, song in enumerate(artifact_songs)}
    missing = [song for song in df["Song"] if song not in song_to_idx]
    if missing:
        raise ValueError(
            "Retrieval artifact is missing songs required by the UI. "
            f"First missing song: {missing[0]}"
        )

    row_positions = np.array([song_to_idx[song] for song in df["Song"]], dtype=np.int32)
    aligned: Dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if values.ndim >= 1 and values.shape[0] == len(artifact_songs):
            aligned[key] = values[row_positions]
        else:
            aligned[key] = values

    artifact_labels = aligned.get("labels")
    if artifact_labels is not None and not np.array_equal(artifact_labels, labels):
        raise ValueError(
            "Saved retrieval artifact labels do not match the currently loaded "
            "clustering results. Re-run the clustering method to refresh artifacts."
        )

    return aligned


def _resolve_retrieval_payload(
    df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    retrieval_features: Optional[np.ndarray],
    retrieval_method_id: Optional[str],
    assignment_confidence: Optional[np.ndarray],
    posterior_probabilities: Optional[np.ndarray],
) -> Dict[str, Optional[np.ndarray]]:
    """Resolve the prepared-space retrieval payload used by the UI."""

    if retrieval_features is None:
        if retrieval_method_id is None:
            raise ValueError(
                "launch_ui requires retrieval_features or retrieval_method_id so "
                "recommendations can run in the prepared feature space."
            )
        from src.clustering.kmeans import load_retrieval_artifact

        artifact = load_retrieval_artifact(retrieval_method_id)
        aligned = _align_artifact_rows(df, labels, artifact)
        retrieval_features = np.asarray(aligned["prepared_features"], dtype=np.float32)
        if assignment_confidence is None and "assignment_confidence" in aligned:
            assignment_confidence = np.asarray(
                aligned["assignment_confidence"], dtype=np.float32
            )
        if posterior_probabilities is None and "posterior_probabilities" in aligned:
            posterior_probabilities = np.asarray(
                aligned["posterior_probabilities"], dtype=np.float32
            )

    retrieval_features = np.asarray(retrieval_features, dtype=np.float32)
    if retrieval_features.ndim != 2 or retrieval_features.shape[0] != len(df):
        raise ValueError(
            "retrieval_features must have one row per song. "
            f"Got shape {retrieval_features.shape} for {len(df)} songs."
        )

    if assignment_confidence is not None:
        assignment_confidence = np.asarray(assignment_confidence, dtype=np.float32)
        if assignment_confidence.shape[0] != len(df):
            raise ValueError(
                "assignment_confidence must have one value per song. "
                f"Got shape {assignment_confidence.shape} for {len(df)} songs."
            )

    if posterior_probabilities is not None:
        posterior_probabilities = np.asarray(posterior_probabilities, dtype=np.float32)
        if (
            posterior_probabilities.ndim != 2
            or posterior_probabilities.shape[0] != len(df)
        ):
            raise ValueError(
                "posterior_probabilities must have one row per song. "
                f"Got shape {posterior_probabilities.shape} for {len(df)} songs."
            )

    return {
        "retrieval_features": retrieval_features,
        "assignment_confidence": assignment_confidence,
        "posterior_probabilities": posterior_probabilities,
        "coords": coords,
        "labels": labels,
    }


def launch_ui(
    df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    top_n: int = 5,
    audio_dir: str = "genres_original",
    clustering_method: str = "K-means",
    retrieval_features: Optional[np.ndarray] = None,
    retrieval_method_id: Optional[str] = None,
    assignment_confidence: Optional[np.ndarray] = None,
    posterior_probabilities: Optional[np.ndarray] = None,
):
    df = df.reset_index(drop=True).copy()
    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels)

    if coords.ndim != 2 or coords.shape[0] != len(df) or coords.shape[1] != 2:
        raise ValueError(
            f"coords must have shape ({len(df)}, 2), got {coords.shape}"
        )
    if labels.shape[0] != len(df):
        raise ValueError(f"labels length must match df length: {labels.shape[0]} != {len(df)}")

    payload = _resolve_retrieval_payload(
        df=df,
        coords=coords,
        labels=labels,
        retrieval_features=retrieval_features,
        retrieval_method_id=retrieval_method_id,
        assignment_confidence=assignment_confidence,
        posterior_probabilities=posterior_probabilities,
    )
    retrieval_features = payload["retrieval_features"]
    assignment_confidence = payload["assignment_confidence"]
    posterior_probabilities = payload["posterior_probabilities"]

    supports_confidence = assignment_confidence is not None
    supports_posteriors = (
        posterior_probabilities is not None and posterior_probabilities.shape[1] > 1
    )

    root = tk.Tk()
    root.title(f"Audio Recommendation System - {clustering_method}")
    root.geometry("1120x720")
    root.minsize(860, 560)

    audio_folder = audio_dir
    current_song = {"name": None, "playing": False}
    recommendation_rows: List[Dict[str, object]] = []
    selected_song_idx = {"value": None}
    song_to_index = {str(name): idx for idx, name in enumerate(df["Song"])}

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    header_font = tkfont.Font(size=14, weight="bold")
    normal_font = tkfont.Font(size=11)
    small_font = tkfont.Font(size=10)

    header = ttk.Frame(root, padding=(10, 6))
    header.pack(fill="x")
    ttk.Label(header, text="Audio Recommendation System", font=header_font).pack(
        side="left"
    )

    paned = ttk.Panedwindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=10, pady=6)

    left = ttk.Frame(paned, width=280)
    paned.add(left, weight=1)

    ttk.Label(left, text="Search songs", font=normal_font).pack(anchor="w", pady=(4, 0))
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.pack(fill="x", pady=5)

    ttk.Label(left, text="All songs", font=normal_font).pack(anchor="w")
    song_list_frame = ttk.Frame(left)
    song_list_frame.pack(fill="both", expand=True)
    song_list = tk.Listbox(song_list_frame, font=normal_font, activestyle="none")
    scroll_songs = ttk.Scrollbar(song_list_frame, orient="vertical", command=song_list.yview)
    song_list.config(yscrollcommand=scroll_songs.set)
    song_list.pack(side="left", fill="both", expand=True)
    scroll_songs.pack(side="right", fill="y")

    play_button = ttk.Button(left, text="Play selected")
    play_button.pack(fill="x", pady=6)

    for name in df["Song"]:
        song_list.insert("end", name)

    right = ttk.Frame(paned)
    paned.add(right, weight=3)

    plot_frame = ttk.LabelFrame(right, text="Cluster map (PCA visualization only)")
    plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 6))

    fig = plt.Figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(111)

    def full_scatter(alpha: float = 1.0):
        return ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="tab10",
            alpha=alpha,
            edgecolors="none",
            s=50,
        )

    base_scatter = full_scatter()
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title(f"{clustering_method} clusters (PCA projection)")
    ax.grid(True, linestyle=":", linewidth=0.4)

    handles, _ = base_scatter.legend_elements(prop="colors")
    cluster_names = [f"Cluster {cluster_id}" for cluster_id in sorted(df["Cluster"].unique())]
    ax.legend(handles, cluster_names, title="Clusters", frameon=False, loc="best")

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    recommendation_header = ttk.Frame(right)
    recommendation_header.pack(fill="x", pady=(4, 0))
    ttk.Label(recommendation_header, text="Recommendations", font=header_font).pack(
        side="left"
    )

    controls_frame = ttk.Frame(right)
    controls_frame.pack(fill="x", pady=(4, 6))

    default_ranking_mode = "Distance"
    if (
        supports_posteriors
        and str(getattr(fv, "default_recommendation_ranking_method", "distance")).lower()
        == "posterior_weighted"
    ):
        default_ranking_mode = "Posterior-weighted"

    top_n_var = tk.StringVar(value=str(top_n))
    ranking_mode_var = tk.StringVar(value=default_ranking_mode)
    min_confidence_var = tk.StringVar(
        value=f"{float(getattr(fv, 'default_min_assignment_confidence', 0.0)):.2f}"
    )
    min_posterior_var = tk.StringVar(
        value=f"{float(getattr(fv, 'default_min_selected_cluster_posterior', 0.0)):.2f}"
    )

    ttk.Label(controls_frame, text="Top N").pack(side="left")
    top_n_spinbox = ttk.Spinbox(
        controls_frame,
        from_=1,
        to=20,
        increment=1,
        textvariable=top_n_var,
        width=4,
    )
    top_n_spinbox.pack(side="left", padx=(6, 12))

    ttk.Label(controls_frame, text="Ranking").pack(side="left")
    ranking_values = ["Distance"]
    if supports_posteriors:
        ranking_values.append("Posterior-weighted")
    ranking_combo = ttk.Combobox(
        controls_frame,
        textvariable=ranking_mode_var,
        values=ranking_values,
        width=18,
        state="readonly",
    )
    ranking_combo.pack(side="left", padx=(6, 12))
    if not supports_posteriors:
        ranking_combo.state(["disabled"])

    ttk.Label(controls_frame, text="Min confidence").pack(side="left")
    min_confidence_box = ttk.Spinbox(
        controls_frame,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=min_confidence_var,
        width=6,
    )
    min_confidence_box.pack(side="left", padx=(6, 12))
    if not supports_confidence:
        min_confidence_box.state(["disabled"])

    ttk.Label(controls_frame, text="Min selected-cluster posterior").pack(side="left")
    min_posterior_box = ttk.Spinbox(
        controls_frame,
        from_=0.0,
        to=1.0,
        increment=0.05,
        textvariable=min_posterior_var,
        width=6,
    )
    min_posterior_box.pack(side="left", padx=(6, 0))
    if not supports_posteriors:
        min_posterior_box.state(["disabled"])

    recommendation_summary_var = tk.StringVar(
        value=(
            "Retrieval uses the full prepared feature space inside the selected "
            "cluster. PCA-2 is kept for visualization only. Probabilistic methods "
            "default to posterior-weighted ranking unless you override it."
        )
    )
    recommendation_summary = ttk.Label(
        right,
        textvariable=recommendation_summary_var,
        font=small_font,
        wraplength=760,
        justify="left",
    )
    recommendation_summary.pack(fill="x", pady=(0, 6))

    recommendation_list_frame = ttk.Frame(right)
    recommendation_list_frame.pack(fill="both", expand=True, pady=(0, 6))
    rec_list = tk.Listbox(recommendation_list_frame, font=small_font, activestyle="none")
    scroll_rec = ttk.Scrollbar(
        recommendation_list_frame,
        orient="vertical",
        command=rec_list.yview,
    )
    rec_list.config(yscrollcommand=scroll_rec.set)
    rec_list.pack(side="left", fill="both", expand=True)
    scroll_rec.pack(side="right", fill="y")

    rec_play_button = ttk.Button(right, text="Play recommendation")
    rec_play_button.pack(fill="x", pady=(0, 6))

    now_playing_var = tk.StringVar(value="Now playing: None")
    now_playing_label = ttk.Label(root, textvariable=now_playing_var, font=normal_font)
    now_playing_label.pack(side="bottom", fill="x", padx=10, pady=6)

    def filter_songs(*_):
        query = search_var.get().lower()
        song_list.delete(0, "end")
        for name in df["Song"]:
            if query in name.lower():
                song_list.insert("end", name)

    search_var.trace_add("write", filter_songs)

    def redraw_plot(sel_idx: int, neighbour_indices: np.ndarray):
        ax.clear()
        full_scatter(alpha=1.0)
        ax.scatter(
            coords[sel_idx, 0],
            coords[sel_idx, 1],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=3,
        )
        for i in neighbour_indices:
            ax.scatter(
                coords[i, 0],
                coords[i, 1],
                marker="D",
                s=90,
                facecolors="yellow",
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )
            ax.plot(
                [coords[sel_idx, 0], coords[i, 0]],
                [coords[sel_idx, 1], coords[i, 1]],
                linestyle="--",
                linewidth=1.0,
                color="gray",
                zorder=2,
            )

        pts = (
            np.vstack([coords[sel_idx], coords[neighbour_indices]])
            if len(neighbour_indices)
            else coords[[sel_idx]]
        )
        margin = 0.5
        xmin, ymin = pts.min(axis=0) - margin
        xmax, ymax = pts.max(axis=0) + margin
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_title(f"{clustering_method} clusters (zoomed)")
        ax.grid(True, linestyle=":", linewidth=0.4)
        canvas.draw_idle()

    def _format_recommendation_row(row: Dict[str, object]) -> str:
        parts = [
            f"{row['song']}",
            f"dist={float(row['distance']):.4f}",
        ]
        if row.get("confidence") is not None:
            parts.append(f"conf={float(row['confidence']):.3f}")
        if row.get("cluster_posterior") is not None:
            parts.append(f"p(cluster)={float(row['cluster_posterior']):.3f}")
        if row.get("ranking_score") is not None and row.get("ranking_mode") == "Posterior-weighted":
            parts.append(f"weighted={float(row['ranking_score']):.4f}")
        return " | ".join(parts)

    def refresh_recommendations(*_):
        if selected_song_idx["value"] is None:
            return

        sel_idx = int(selected_song_idx["value"])
        selected_song = str(df.at[sel_idx, "Song"])
        selected_cluster = int(labels[sel_idx])

        rec_list.delete(0, "end")
        recommendation_rows.clear()

        if selected_cluster == -1:
            recommendation_summary_var.set(
                "The selected song is labeled as HDBSCAN noise, so within-cluster "
                "recommendations are intentionally disabled."
            )
            redraw_plot(sel_idx, np.array([], dtype=np.int32))
            return

        top_n_value = _parse_bounded_int(top_n_var.get(), top_n, 1, 20)
        ranking_mode = ranking_mode_var.get()
        min_confidence = _parse_bounded_float(min_confidence_var.get(), 0.0, 0.0, 1.0)
        min_posterior = _parse_bounded_float(min_posterior_var.get(), 0.0, 0.0, 1.0)

        candidate_indices = np.where(labels == selected_cluster)[0]
        candidate_indices = candidate_indices[candidate_indices != sel_idx]

        candidate_confidence = None
        if assignment_confidence is not None:
            candidate_confidence = assignment_confidence[candidate_indices]
            keep = candidate_confidence >= min_confidence
            candidate_indices = candidate_indices[keep]
            candidate_confidence = candidate_confidence[keep]

        candidate_cluster_posterior = None
        if (
            posterior_probabilities is not None
            and 0 <= selected_cluster < posterior_probabilities.shape[1]
        ):
            candidate_cluster_posterior = posterior_probabilities[
                candidate_indices, selected_cluster
            ]
            keep = candidate_cluster_posterior >= min_posterior
            candidate_indices = candidate_indices[keep]
            if candidate_confidence is not None:
                candidate_confidence = candidate_confidence[keep]
            candidate_cluster_posterior = candidate_cluster_posterior[keep]

        if len(candidate_indices) == 0:
            recommendation_summary_var.set(
                f"No candidates remain for '{selected_song}' after the active "
                "prepared-space filters."
            )
            redraw_plot(sel_idx, np.array([], dtype=np.int32))
            return

        selected_vector = retrieval_features[sel_idx]
        candidate_vectors = retrieval_features[candidate_indices]
        distances = np.linalg.norm(candidate_vectors - selected_vector, axis=1)

        ranking_scores = distances.copy()
        if ranking_mode == "Posterior-weighted" and candidate_cluster_posterior is not None:
            ranking_scores = distances / np.clip(candidate_cluster_posterior, 1e-3, 1.0)
        else:
            ranking_mode = "Distance"

        order = np.lexsort((distances, ranking_scores))[:top_n_value]
        chosen_indices = candidate_indices[order]

        for rank_position, order_idx in enumerate(order, start=1):
            candidate_idx = int(candidate_indices[order_idx])
            row = {
                "rank": rank_position,
                "song": str(df.at[candidate_idx, "Song"]),
                "song_index": candidate_idx,
                "distance": float(distances[order_idx]),
                "confidence": (
                    None
                    if candidate_confidence is None
                    else float(candidate_confidence[order_idx])
                ),
                "cluster_posterior": (
                    None
                    if candidate_cluster_posterior is None
                    else float(candidate_cluster_posterior[order_idx])
                ),
                "ranking_score": float(ranking_scores[order_idx]),
                "ranking_mode": ranking_mode,
            }
            recommendation_rows.append(row)
            rec_list.insert("end", _format_recommendation_row(row))

        summary_parts = [
            f"Selected: {selected_song}",
            f"cluster={selected_cluster}",
            f"retrieval=prepared-space {ranking_mode.lower()}",
            f"candidates={len(candidate_indices)}",
        ]
        if assignment_confidence is not None:
            summary_parts.append(f"min_conf={min_confidence:.2f}")
        if candidate_cluster_posterior is not None:
            summary_parts.append(f"min_p(cluster)={min_posterior:.2f}")
        recommendation_summary_var.set(" | ".join(summary_parts))

        redraw_plot(sel_idx, chosen_indices)

    def on_song_select(_event):
        selection = song_list.curselection()
        if not selection:
            return
        song_name = song_list.get(selection[0])
        sel_idx = song_to_index[song_name]
        selected_song_idx["value"] = sel_idx
        refresh_recommendations()

    def play_audio(song_name: str):
        pygame.mixer.music.stop()

        song_row = df[df["Song"] == song_name]
        audio_file = None

        candidates = [song_name]
        if not song_name.lower().endswith((".wav", ".mp3")):
            candidates.append(f"{song_name}.wav")
            candidates.append(f"{song_name}.mp3")

        for candidate in candidates:
            potential_path = os.path.join(audio_folder, candidate)
            if os.path.exists(potential_path):
                audio_file = potential_path
                break

        if audio_file is None and not song_row.empty and "Genre" in df.columns:
            genre = song_row["Genre"].values[0]
            for candidate in candidates:
                potential_path = os.path.join(audio_folder, genre, candidate)
                if os.path.exists(potential_path):
                    audio_file = potential_path
                    break

        if audio_file is None:
            parts = song_name.split(".")
            if parts:
                genre = parts[0]
                for candidate in candidates:
                    potential_path = os.path.join(audio_folder, genre, candidate)
                    if os.path.exists(potential_path):
                        audio_file = potential_path
                        break

        if audio_file is None:
            for root_dir, _, files in os.walk(audio_folder):
                for candidate in candidates:
                    if candidate in files:
                        audio_file = os.path.join(root_dir, candidate)
                        break
                if audio_file:
                    break

        if audio_file is None:
            messagebox.showerror(
                "Error",
                f"Could not find audio file for {song_name}\nSearched in {audio_folder}",
            )
            return

        if current_song["name"] == song_name and current_song["playing"]:
            current_song["playing"] = False
            now_playing_var.set("Now playing: None")
            return

        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            current_song["name"] = song_name
            current_song["playing"] = True
            now_playing_var.set(f"Now playing: {song_name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to play audio: {exc}")

    def play_selected_song():
        selection = song_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a song first")
            return
        play_audio(song_list.get(selection[0]))

    def play_recommended_song():
        selection = rec_list.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a recommendation first")
            return
        row_idx = int(selection[0])
        if not (0 <= row_idx < len(recommendation_rows)):
            messagebox.showinfo("Info", "That recommendation entry is not playable")
            return
        play_audio(str(recommendation_rows[row_idx]["song"]))

    song_list.bind("<Double-1>", lambda _event: play_selected_song())
    rec_list.bind("<Double-1>", lambda _event: play_recommended_song())
    play_button.config(command=play_selected_song)
    rec_play_button.config(command=play_recommended_song)
    song_list.bind("<<ListboxSelect>>", on_song_select)

    for control_var in (
        top_n_var,
        ranking_mode_var,
        min_confidence_var,
        min_posterior_var,
    ):
        control_var.trace_add("write", refresh_recommendations)

    def on_closing():
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
