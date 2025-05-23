import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import font as tkfont
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None


def run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results',
                          n_clusters=3,
                          elbow_max_k=10,
                          show_elbow=True,
                          dynamic_cluster_selection=False,
                          dynamic_k_min=2,
                          dynamic_k_max=10,
                          ui=True):
    """Cluster pre-computed audio-feature vectors with K-means (no PCA)."""
    os.makedirs(results_dir, exist_ok=True)

    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not wav_files:
        print("No audio files found.")
        return

    file_names, feature_vectors = [], []
    for path in wav_files:
        base = os.path.splitext(os.path.basename(path))[0]
        feats = {k: os.path.join(results_dir, f"{base}_{k}.npy") for k in [
            'mfcc', 'melspectrogram', 'spectral_centroid',
            'spectral_flatness', 'zero_crossing_rate']}
        if not all(os.path.isfile(p) for p in feats.values()):
            continue

        arrays = [np.load(feats[k]) for k in feats]
        vec = np.concatenate(
            [np.concatenate([arr.mean(axis=1), arr.std(axis=1)]) for arr in arrays])
        file_names.append(base)
        feature_vectors.append(vec)

    if not feature_vectors:
        print("No complete feature files.")
        return

    X = StandardScaler().fit_transform(np.vstack(feature_vectors))

    # Optional automatic k selection with silhouette score
    if dynamic_cluster_selection:
        sil_scores = {}
        for k in range(dynamic_k_min, dynamic_k_max + 1):
            labels_tmp = KMeans(n_clusters=k, random_state=42,
                                n_init=10).fit_predict(X)
            sil_scores[k] = silhouette_score(X, labels_tmp)
        n_clusters = max(sil_scores, key=sil_scores.get)
        print(f"Optimal k selected by silhouette score: {n_clusters}")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_

    df = pd.DataFrame({
        'Song': file_names,
        'Cluster': labels,
        'Distance': np.linalg.norm(X - centers[labels], axis=1)
    })
    csv_path = os.path.join(results_dir, 'audio_clustering_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # 2-D coordinates for plotting – just take the first two scaled features
    if X.shape[1] >= 2:
        coords = X[:, :2]
    else:  # If only one feature dimension, add zeros for Y
        coords = np.column_stack((X[:, 0], np.zeros_like(X[:, 0])))
    df['X'], df['Y'] = coords[:, 0], coords[:, 1]

    if ui and tk:
        launch_ui(file_names, X, labels, coords)
    elif ui:
        print("UI unavailable (tkinter missing).")


def launch_ui(file_names, X, labels, coords, top_n=5):
    """Tkinter interface: left pane song list, right pane recs + scatter plot."""
    root = tk.Tk()
    root.title("🎵 Audio Recommendation System 🎵")
    root.geometry('800x600')
    root.minsize(700, 500)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use('clam')

    header_font = tkfont.Font(size=14, weight='bold')
    normal_font = tkfont.Font(size=11)

    # Header
    header = ttk.Frame(root, padding=(10, 5))
    header.pack(fill='x')
    ttk.Label(header, text="Audio Recommendation System",
              font=header_font).pack(side='left')

    # Paned window
    paned = ttk.Panedwindow(root, orient='horizontal')
    paned.pack(fill='both', expand=True, padx=10, pady=5)

    # -- Left pane: song list & search
    left = ttk.Frame(paned, width=250)
    paned.add(left, weight=1)

    ttk.Label(left, text="🔍 Search Songs:", font=normal_font).pack(
        anchor='w', pady=(5, 0))
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.pack(fill='x', pady=5)

    ttk.Label(left, text="🎶 All Songs:", font=normal_font).pack(anchor='w')
    song_list = tk.Listbox(left, font=normal_font, activestyle='none')
    scroll_songs = ttk.Scrollbar(
        left, orient='vertical', command=song_list.yview)
    song_list.config(yscrollcommand=scroll_songs.set)
    song_list.pack(side='left', fill='both', expand=True)
    scroll_songs.pack(side='right', fill='y')

    for name in file_names:
        song_list.insert('end', name)

    # -- Right pane: recommendations + plot
    right = ttk.Frame(paned)
    paned.add(right, weight=3)
    ttk.Label(right, text="Recommendations", font=header_font).pack(
        anchor='w', pady=(5, 0))

    rec_list = tk.Listbox(right, font=normal_font)
    scroll_rec = ttk.Scrollbar(
        right, orient='vertical', command=rec_list.yview)
    rec_list.config(yscrollcommand=scroll_rec.set)
    rec_list.pack(fill='both', expand=False, pady=(0, 5))
    scroll_rec.pack(fill='y', side='right')

    # Scatter plot
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(coords[:, 0], coords[:, 1],
                         c=labels, cmap='tab10', s=50)
    ax.set_title('Clusters (first two features)')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Filtering & recommendations
    def filter_songs(*_):
        query = search_var.get().lower()
        song_list.delete(0, 'end')
        for name in file_names:
            if query in name.lower():
                song_list.insert('end', name)
    search_var.trace_add('write', filter_songs)

    def on_song_select(_):
        sel = song_list.curselection()
        if not sel:
            return
        idx = file_names.index(song_list.get(sel))
        dists = np.linalg.norm(X - X[idx], axis=1)
        same_cluster = [(file_names[i], d)
                        for i, d in enumerate(dists)
                        if labels[i] == labels[idx] and i != idx]
        same_cluster.sort(key=lambda x: x[1])
        rec_list.delete(0, 'end')
        for name, _ in same_cluster[:top_n]:
            rec_list.insert('end', name)
    song_list.bind('<<ListboxSelect>>', on_song_select)

    root.mainloop()


if __name__ == '__main__':
    run_kmeans_clustering(audio_dir='audio_files',
                          results_dir='results',
                          n_clusters=3,
                          dynamic_cluster_selection=True)
