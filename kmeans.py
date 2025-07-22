import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import feature_vars as fv

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont
    # Add messagebox for error handling
    from tkinter import messagebox
except ImportError as exc:
    raise RuntimeError(
        "Tkinter isn't available on your Python installation.") from exc

# Import pygame for audio playback
try:
    import pygame
except ImportError as exc:
    raise RuntimeError(
        "Pygame isn't available. Install with 'pip install pygame'") from exc

# Initialize pygame mixer
pygame.mixer.init()


def build_group_weights(n_mfcc: int = fv.n_mfcc, n_mels: int = fv.n_mels) -> np.ndarray:
    group_sizes = [2 * n_mfcc, 2 * n_mels, 2, 2]
    total_dims = sum(group_sizes)
    w = np.ones(total_dims, dtype=np.float32)
    idx = 0
    for g in group_sizes:
        w[idx:idx + g] /= np.sqrt(g)
        idx += g
    return w


def run_kmeans_clustering(
    audio_dir: str = "audio_files",
    results_dir: str = "results",
    n_clusters: int = 3,
    dynamic_cluster_selection: bool = False,
    dynamic_k_min: int = 2,
    dynamic_k_max: int = 10,
    n_mfcc: int = fv.n_mfcc,
    n_mels: int = fv.n_mels,
):
    os.makedirs(results_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3"))
    if not audio_files:
        raise FileNotFoundError(f"No *.mp3 under {audio_dir!r}.")

    file_names, feature_vectors = [], []
    for audio_path in audio_files:
        base = Path(audio_path).stem
        feats = {k: os.path.join(results_dir, f"{base}_{k}.npy") for k in [
            "mfcc", "melspectrogram", "spectral_centroid",
            "zero_crossing_rate",
        ]}
        if not all(os.path.isfile(p) for p in feats.values()):
            continue
        arrays = [np.load(p) for p in feats.values()]
        vec = np.concatenate([
            np.concatenate([arr.mean(axis=1), arr.std(axis=1)])
            for arr in arrays
        ])
        file_names.append(base)
        feature_vectors.append(vec)

    if not feature_vectors:
        raise RuntimeError("No songs with complete feature files were found.")

    X_scaled = StandardScaler().fit_transform(np.vstack(feature_vectors))
    weights = build_group_weights(n_mfcc=n_mfcc, n_mels=n_mels)
    if X_scaled.shape[1] != len(weights):
        raise ValueError(
            f"Expected {len(weights)} dims after feature concat, got {X_scaled.shape[1]}")
    X = X_scaled * weights

    if dynamic_cluster_selection:
        sil = {}
        for k in range(dynamic_k_min, dynamic_k_max + 1):
            lbls_tmp = KMeans(n_clusters=k, random_state=42,
                              n_init=10).fit_predict(X)
            sil[k] = silhouette_score(X, lbls_tmp)
        n_clusters = max(sil, key=sil.get)
        print(f"Optimal k (silhouette) → {n_clusters}")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    df = pd.DataFrame({
        "Song": file_names,
        "Cluster": labels,
        "Distance": np.linalg.norm(X - centers[labels], axis=1),
        "PCA1": coords[:, 0],
        "PCA2": coords[:, 1],
    })
    csv_path = os.path.join("audio_clustering_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results written to → {csv_path}")

    return df, coords, labels


def launch_ui(df: pd.DataFrame, coords: np.ndarray, labels: np.ndarray, top_n: int = 5, audio_dir: str = "audio_files"):
    root = tk.Tk()
    root.title("🎵 Audio Recommendation System")
    root.geometry("900x600")
    root.minsize(720, 480)

    # Store the audio directory
    audio_folder = audio_dir

    # Track currently playing song
    current_song = {"name": None, "playing": False}

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")

    header_font = tkfont.Font(size=14, weight="bold")
    normal_font = tkfont.Font(size=11)

    header = ttk.Frame(root, padding=(10, 5))
    header.pack(fill="x")
    ttk.Label(header, text="Audio Recommendation System",
              font=header_font).pack(side="left")

    paned = ttk.Panedwindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=10, pady=5)

    left = ttk.Frame(paned, width=260)
    paned.add(left, weight=1)

    ttk.Label(left, text="🔍  Search Songs:", font=normal_font).pack(
        anchor="w", pady=(5, 0))
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.pack(fill="x", pady=5)

    ttk.Label(left, text="🎶  All Songs:", font=normal_font).pack(anchor="w")
    song_list = tk.Listbox(left, font=normal_font, activestyle="none")
    scroll_songs = ttk.Scrollbar(
        left, orient="vertical", command=song_list.yview)
    song_list.config(yscrollcommand=scroll_songs.set)
    song_list.pack(side="left", fill="both", expand=True)
    scroll_songs.pack(side="right", fill="y")

    # Add play button for selected song
    play_button_frame = ttk.Frame(left)
    play_button_frame.pack(fill="x", pady=5)
    play_button = ttk.Button(play_button_frame, text="▶️ Play Selected")
    play_button.pack(fill="x")

    for name in df["Song"]:
        song_list.insert("end", name)

    right = ttk.Frame(paned)
    paned.add(right, weight=3)

    plot_frame = ttk.LabelFrame(right, text="Cluster map")
    plot_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    def full_scatter(alpha=1.0):
        return ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", alpha=alpha,
                          edgecolors="none", s=50)

    base_scatter = full_scatter()

    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("K-means clusters (PCA projection)")
    ax.grid(True, linestyle=":", linewidth=0.4)

    handles, _ = base_scatter.legend_elements(prop="colors")
    ax.legend(handles, [f"Cluster {c}" for c in sorted(df["Cluster"].unique())],
              title="Clusters", frameon=False, loc="best")

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    ttk.Label(right, text="Recommendations", font=header_font).pack(
        anchor="w", pady=(8, 0))
    rec_list = tk.Listbox(right, font=normal_font)
    scroll_rec = ttk.Scrollbar(
        right, orient="vertical", command=rec_list.yview)
    rec_list.config(yscrollcommand=scroll_rec.set)
    rec_list.pack(fill="both", expand=True, pady=(0, 5))
    scroll_rec.pack(fill="y", side="right")

    # Add play button for recommendations
    rec_play_frame = ttk.Frame(right)
    rec_play_frame.pack(fill="x", pady=5)
    rec_play_button = ttk.Button(rec_play_frame, text="▶️ Play Recommendation")
    rec_play_button.pack(fill="x")

    # Now playing indicator
    now_playing_var = tk.StringVar()
    now_playing_var.set("Now Playing: None")
    now_playing_label = ttk.Label(
        root, textvariable=now_playing_var, font=normal_font)
    now_playing_label.pack(side="bottom", fill="x", padx=10, pady=5)

    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)

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
        ax.scatter(coords[sel_idx, 0], coords[sel_idx, 1], s=160,
                   facecolors="none", edgecolors="red", linewidths=2, zorder=3)
        for i in neighbour_indices:
            ax.scatter(coords[i, 0], coords[i, 1], marker="D", s=100,
                       facecolors="yellow", edgecolors="k", linewidths=0.8, zorder=3)
            ax.plot([coords[sel_idx, 0], coords[i, 0]],
                    [coords[sel_idx, 1], coords[i, 1]],
                    linestyle="--", linewidth=1.0, color="gray", zorder=2)
        pts = np.vstack([coords[sel_idx], coords[neighbour_indices]]) if len(
            neighbour_indices) else coords[[sel_idx]]
        margin = 0.5
        xmin, ymin = pts.min(axis=0) - margin
        xmax, ymax = pts.max(axis=0) + margin
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_title("K-means clusters (zoomed)")
        ax.grid(True, linestyle=":", linewidth=0.4)
        canvas.draw_idle()

    def on_song_select(_event):
        sel = song_list.curselection()
        if not sel:
            return
        song_name = song_list.get(sel)
        sel_idx = df.index[df["Song"] == song_name][0]
        same = np.where(labels == labels[sel_idx])[0]
        same = same[same != sel_idx]
        dists = distances[sel_idx, same]
        order = same[np.argsort(dists)][:top_n]
        rec_list.delete(0, "end")
        for i in order:
            rec_list.insert("end", df.at[i, "Song"])
        redraw_plot(sel_idx, order)

    def play_audio(song_name):
        """Play the audio file associated with the song name."""
        # Stop any currently playing song
        pygame.mixer.music.stop()

        # Update the current song information
        if current_song["name"] == song_name and current_song["playing"]:
            # Toggle pause/play
            current_song["playing"] = False
            now_playing_var.set("Now Playing: None")
            return

        # Find the mp3 file
        song_file = os.path.join(audio_folder, f"{song_name}.mp3")
        if not os.path.exists(song_file):
            messagebox.showerror("Error", f"Audio file not found: {song_file}")
            return

        try:
            pygame.mixer.music.load(song_file)
            pygame.mixer.music.play()
            current_song["name"] = song_name
            current_song["playing"] = True
            now_playing_var.set(f"Now Playing: {song_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")

    def play_selected_song():
        """Play the selected song from the song list."""
        sel = song_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a song first")
            return
        song_name = song_list.get(sel)
        play_audio(song_name)

    def play_recommended_song():
        """Play the selected song from the recommendations list."""
        sel = rec_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "Please select a recommendation first")
            return
        song_name = rec_list.get(sel)
        play_audio(song_name)

    # Bind double-click on songs to play
    song_list.bind("<Double-1>", lambda event: play_selected_song())
    rec_list.bind("<Double-1>", lambda event: play_recommended_song())

    # Connect play buttons
    play_button.config(command=play_selected_song)
    rec_play_button.config(command=play_recommended_song)

    # Bind single-click for selection
    song_list.bind("<<ListboxSelect>>", on_song_select)

    # Clean up when closing
    def on_closing():
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()


if __name__ == "__main__":
    DF, COORDS, LABELS = run_kmeans_clustering(
        audio_dir="audio_files",
        results_dir="results",
        n_clusters=3,
        dynamic_cluster_selection=True,
    )

    launch_ui(DF, COORDS, LABELS, audio_dir="audio_files")
