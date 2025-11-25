import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from pathlib import Path

# Ensure we are running from project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

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
    # Initialize pygame mixer
    pygame.mixer.init()
except ImportError as exc:
    raise RuntimeError(
        "Pygame isn't available. Install with 'pip install pygame'") from exc
except Exception as exc:
    raise RuntimeError(
        f"Failed to initialize pygame mixer: {exc}") from exc

def launch_ui(df: pd.DataFrame, coords: np.ndarray, labels: np.ndarray, top_n: int = 5, audio_dir: str = "genres_original", clustering_method: str = "K-means"):
    root = tk.Tk()
    root.title(f"🎵 Audio Recommendation System - {clustering_method}")
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
    ax.set_title(f"{clustering_method} clusters (PCA projection)")
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
        ax.set_title(f"{clustering_method} clusters (zoomed)")
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

        # Find the genre from the DataFrame
        song_row = df[df["Song"] == song_name]
        audio_file = None
        
        # Method 1: Try to get genre from DataFrame
        if not song_row.empty and "Genre" in df.columns:
            genre = song_row["Genre"].values[0]
            # The file path should include the genre folder
            potential_file = os.path.join(audio_folder, genre, f"{song_name}.wav")
            if os.path.exists(potential_file):
                audio_file = potential_file
        
        # Method 2: Try to extract genre from song name (assuming format 'genre.xxxxxx')
        if audio_file is None:
            parts = song_name.split('.')
            if len(parts) > 0:
                genre = parts[0]
                potential_file = os.path.join(audio_folder, genre, f"{song_name}.wav")
                if os.path.exists(potential_file):
                    audio_file = potential_file
        
        # Method 3: Search in all genre folders
        if audio_file is None:
            for genre_folder in os.listdir(audio_folder):
                genre_dir = os.path.join(audio_folder, genre_folder)
                if os.path.isdir(genre_dir):
                    potential_path = os.path.join(genre_dir, f"{song_name}.wav")
                    if os.path.exists(potential_path):
                        audio_file = potential_path
                        break
        
        # If we still don't have a file, error out
        if audio_file is None:
            messagebox.showerror("Error", f"Could not find audio file for {song_name}")
            return

        # Update the current song information
        if current_song["name"] == song_name and current_song["playing"]:
            # Toggle pause/play
            current_song["playing"] = False
            now_playing_var.set("Now Playing: None")
            return

        try:
            pygame.mixer.music.load(audio_file)
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
