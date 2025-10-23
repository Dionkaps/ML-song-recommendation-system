import os
import glob
import argparse
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Tkinter UI
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except ImportError as exc:
    raise RuntimeError("Tkinter isn't available on your Python installation.") from exc


RESULTS_DIR = "output/results"

PER_METHOD_CSV = os.path.join(RESULTS_DIR, "clustering_per_method_metrics.csv")
PAIRWISE_CSV = os.path.join(RESULTS_DIR, "clustering_pairwise_metrics.csv")
CLUSTER_SIZES = {
    "KMeans": os.path.join(RESULTS_DIR, "cluster_sizes_kmeans.csv"),
    "DBSCAN": os.path.join(RESULTS_DIR, "cluster_sizes_dbscan.csv"),
    "Hierarchical": os.path.join(RESULTS_DIR, "cluster_sizes_hierarchical.csv"),
}
REPORT_MD = os.path.join(RESULTS_DIR, "clustering_comparison.md")


def _load_df(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _list_contingency_files() -> List[str]:
    pattern = os.path.join(RESULTS_DIR, "contingency_*_vs_*.csv")
    return sorted(glob.glob(pattern))


def _df_to_tree(tree: ttk.Treeview, df: pd.DataFrame) -> None:
    tree.delete(*tree.get_children())
    # Configure columns
    cols = list(df.columns)
    tree["columns"] = cols
    tree["show"] = "headings"
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=120, anchor="center")
    # Insert rows
    for _, row in df.iterrows():
        vals = [row[c] for c in cols]
        tree.insert("", "end", values=vals)


def _barh_from_series(ax: plt.Axes, s: pd.Series, title: str, xlabel: str = "Count") -> None:
    if s.empty:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return
    s_sorted = s.sort_index()
    ax.barh(s_sorted.index.astype(str), s_sorted.values, color="#4e79a7")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for i, v in enumerate(s_sorted.values):
        ax.text(v, i, f" {v}", va="center")


def _bar_from_df(ax: plt.Axes, df: pd.DataFrame, metrics: List[str], title: str) -> None:
    if df is None or df.empty:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.8 / max(1, len(metrics))
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        vals = df[m].astype(float).values
        ax.bar(x + i * width - (len(metrics)-1)*width/2, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)


def _heatmap(ax: plt.Axes, mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str) -> None:
    if mat.size == 0:
        ax.set_title(title + " (no data)")
        ax.axis("off")
        return
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("B")
    ax.set_ylabel("A")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


class ComparisonUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Clustering Comparison Dashboard")
        self.root.geometry("1200x850")
        self.root.minsize(980, 700)

        # Modern style
        style = ttk.Style(self.root)
        # Try clam theme for better visuals
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TNotebook", tabposition="n")
        style.configure("TNotebook.Tab", padding=(12, 6))
        style.configure("Card.TFrame", background="#f7f7fb", relief="flat")
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("KPI.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Desc.TLabel", foreground="#555")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        # Data
        self.per_method_df = _load_df(PER_METHOD_CSV)
        self.pairwise_df = _load_df(PAIRWISE_CSV)
        self.cluster_sizes = {k: _load_df(v) for k, v in CLUSTER_SIZES.items()}
        self.cont_files = _list_contingency_files()

        # Tabs
        self.tab_overview = ttk.Frame(nb)
        nb.add(self.tab_overview, text="Overview")
        self._build_overview()

        self.tab_pairwise = ttk.Frame(nb)
        nb.add(self.tab_pairwise, text="Pairwise")
        self._build_pairwise()

        self.tab_sizes = ttk.Frame(nb)
        nb.add(self.tab_sizes, text="Cluster Sizes")
        self._build_sizes()

        self.tab_report = ttk.Frame(nb)
        nb.add(self.tab_report, text="Report")
        self._build_report()

    def _build_overview(self):
        frame = self.tab_overview
        top = ttk.Frame(frame)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        # KPI cards
        cards = ttk.Frame(top)
        cards.pack(fill="x")
        if self.per_method_df is not None and not self.per_method_df.empty:
            # Derive some quick KPIs
            methods = ", ".join(self.per_method_df["method"].tolist())
            total_items = int(self.per_method_df["n_items"].max())
            try:
                noise = int(self.per_method_df.get("n_noise", pd.Series([0])).sum())
            except Exception:
                noise = 0

            for i, (title, value, desc) in enumerate([
                ("Methods", methods, "Available clustering methods"),
                ("Items", f"{total_items}", "Total items considered"),
                ("Noise", f"{noise}", "DBSCAN noise points"),
            ]):
                card = ttk.Frame(cards, style="Card.TFrame", padding=10)
                card.grid(row=0, column=i, padx=6, pady=6, sticky="ew")
                ttk.Label(card, text=title, style="Header.TLabel").pack(anchor="w")
                ttk.Label(card, text=value, style="KPI.TLabel").pack(anchor="w")
                ttk.Label(card, text=desc, style="Desc.TLabel").pack(anchor="w")
                cards.grid_columnconfigure(i, weight=1)

        # Table of per-method metrics
        mid = ttk.Frame(top)
        mid.pack(fill="both", expand=True, pady=(10, 0))
        top_bar = ttk.Frame(mid)
        top_bar.pack(fill="x")
        ttk.Label(top_bar, text="Per-method metrics vs Genre", style="Header.TLabel").pack(side="left")
        ttk.Label(top_bar, text="Metric set:").pack(side="left", padx=(10, 2))
        self.metric_var = tk.StringVar(value="Quality")
        self.metric_combo = ttk.Combobox(top_bar, width=18, state="readonly",
                                         values=["Quality", "Silhouette", "Noise-aware"])
        self.metric_combo.current(0)
        self.metric_combo.pack(side="left")
        self.metric_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_overview_chart())

        self.tree_per = ttk.Treeview(mid, height=14)
        self.tree_per.pack(fill="both", expand=True, pady=(6, 10))
        if self.per_method_df is not None:
            # Make tree columns sortable
            df_show = self.per_method_df.copy()
            _df_to_tree(self.tree_per, df_show.round(4))
            for c in df_show.columns:
                self.tree_per.heading(c, command=lambda col=c: self._sort_tree(self.tree_per, col))
        else:
            ttk.Label(mid, text="No per-method metrics found.").pack(anchor="w")

        # Chart of selected metrics
        chart = ttk.LabelFrame(top, text="Charts")
        chart.pack(fill="both", expand=True, padx=2, pady=2)
        self.fig_over = plt.Figure(figsize=(6, 4))
        self.ax_over = self.fig_over.add_subplot(111)
        self.canvas_over = FigureCanvasTkAgg(self.fig_over, master=chart)
        self.canvas_over.draw()
        self.canvas_over.get_tk_widget().pack(fill="both", expand=True)
        self._refresh_overview_chart()

    def _sort_tree(self, tree: ttk.Treeview, col: str, reverse: bool = False):
        try:
            data = [(float(tree.set(k, col)), k) for k in tree.get_children("")]
        except Exception:
            data = [(tree.set(k, col), k) for k in tree.get_children("")]
        data.sort(reverse=reverse)
        for index, (_, k) in enumerate(data):
            tree.move(k, "", index)
        tree.heading(col, command=lambda: self._sort_tree(tree, col, not reverse))

    def _refresh_overview_chart(self):
        self.ax_over.clear()
        if self.per_method_df is None or self.per_method_df.empty:
            self.ax_over.set_title("No data")
            self.canvas_over.draw_idle()
            return
        mode = self.metric_combo.get()
        if mode == "Quality":
            metrics = ["v_measure", "ARI_vs_genre", "purity_micro"]
        elif mode == "Silhouette":
            metrics = ["silhouette_on_features", "silhouette_on_features_no_noise", "silhouette_on_PCA"]
        else:  # Noise-aware
            metrics = ["n_clusters", "n_noise", "purity_macro"]
        _bar_from_df(self.ax_over, self.per_method_df, metrics, f"Metrics: {mode}")
        self.fig_over.tight_layout()
        self.canvas_over.draw_idle()

    def _build_pairwise(self):
        frame = self.tab_pairwise
        top = ttk.Frame(frame)
        top.pack(fill="x", padx=12, pady=12)

        ttk.Label(top, text="Pairwise metrics", font=("", 12, "bold")).pack(anchor="w")
        self.tree_pair = ttk.Treeview(frame, height=12)
        self.tree_pair.pack(fill="x", padx=10)
        if self.pairwise_df is not None:
            df_pair = self.pairwise_df.copy()
            _df_to_tree(self.tree_pair, df_pair.round(4))
            for c in df_pair.columns:
                self.tree_pair.heading(c, command=lambda col=c: self._sort_tree(self.tree_pair, col))
        else:
            ttk.Label(frame, text="No pairwise metrics found.").pack(anchor="w", padx=10)

        # Contingency selection and heatmap
        bottom = ttk.Frame(frame)
        bottom.pack(fill="both", expand=True, padx=10, pady=10)

        sel_frame = ttk.Frame(bottom)
        sel_frame.pack(fill="x")
        ttk.Label(sel_frame, text="Contingency:").pack(side="left")
        self.combo = ttk.Combobox(sel_frame, values=self.cont_files, state="readonly")
        self.combo.pack(side="left", fill="x", expand=True, padx=6)
        btn_load = ttk.Button(sel_frame, text="Load", command=self._load_contingency)
        btn_load.pack(side="left")

        fig = plt.Figure(figsize=(6, 4))
        self.ax_heat = fig.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(fig, master=bottom)
        self.canvas_heat.draw()
        self.canvas_heat.get_tk_widget().pack(fill="both", expand=True)

        if self.cont_files:
            self.combo.current(0)
            self._load_contingency()

    def _load_contingency(self):
        path = self.combo.get()
        if not path:
            return
        try:
            cont = pd.read_csv(path, index_col=0)
            self.ax_heat.clear()
            _heatmap(self.ax_heat, cont.values, cont.columns.astype(str).tolist(), cont.index.astype(str).tolist(), os.path.basename(path))
            self.canvas_heat.draw_idle()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load contingency: {e}")

    def _build_sizes(self):
        frame = self.tab_sizes
        top = ttk.Frame(frame)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        fig = plt.Figure(figsize=(10, 5))
        axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]
        titles = list(CLUSTER_SIZES.keys())
        for ax, name in zip(axes, titles):
            df = self.cluster_sizes.get(name)
            if df is not None and not df.empty and "Cluster" in df and "Count" in df:
                series = df.set_index("Cluster")["Count"]
            else:
                series = pd.Series(dtype=float)
            _barh_from_series(ax, series, f"{name} cluster sizes")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_report(self):
        frame = self.tab_report
        top = ttk.Frame(frame)
        top.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(top, text="Markdown report preview", style="Header.TLabel").pack(anchor="w")
        txt = tk.Text(top, wrap="word")
        txt.pack(fill="both", expand=True)
        if os.path.exists(REPORT_MD):
            try:
                with open(REPORT_MD, "r", encoding="utf-8") as f:
                    txt.insert("1.0", f.read())
            except Exception:
                txt.insert("1.0", "Failed to load report.")
        else:
            txt.insert("1.0", "Report not found. Run compare_clustering.py first.")


def launch_comparison_ui():
    root = tk.Tk()
    ComparisonUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Clustering comparison UI")
    parser.add_argument("--no-gui", action="store_true", help="Do not launch the GUI (just validate files)")
    args = parser.parse_args()

    # Quick validation
    per = _load_df(PER_METHOD_CSV)
    pair = _load_df(PAIRWISE_CSV)
    cont_list = _list_contingency_files()

    print("Check results directory:", os.path.abspath(RESULTS_DIR))
    print("Per-method metrics:", "OK" if per is not None else "MISSING")
    print("Pairwise metrics:", "OK" if pair is not None else "MISSING")
    print("Contingency files:", len(cont_list))

    if not args.no_gui:
        print("Launching comparison UI...")
        launch_comparison_ui()


if __name__ == "__main__":
    main()
