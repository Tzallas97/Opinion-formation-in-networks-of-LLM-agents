#!/usr/bin/env python3
"""Compare several finished runs on one figure (and a small GUI to drive it).

Pick 2-6 run folders that have an opinion_change CSV (either naming scheme).
The tool recomputes the population summaries from each run's trajectory and
overlays them so conditions can be read against each other:

  - mean opinion B(t)               one line per run
  - diversity D(t) = population std one line per run
  - fragmentation                   effective number of opinion clusters, 1/Sum p_i^2
  - final opinion distribution      grouped share per rating

It reads the raw trajectory only, so it does not need the per-run plots to have
been generated first. Run:  python scripts/compare_runs.py
"""
from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_naming  # noqa: E402

_PALETTE = ["#2f6bff", "#e8590c", "#2b8a3e", "#9c36b5", "#e03131", "#0b7285"]
_RATING_COLORS = {-2: "#b2182b", -1: "#ef8a62", 0: "#d9d9d9", 1: "#67a9cf", 2: "#2166ac"}


def _apply_style():
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": "#1b1e23",
        "axes.labelsize": 10, "axes.edgecolor": "#c9ccd1",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": "#e7e9ec", "grid.linewidth": 0.9,
        "xtick.color": "#5b5f66", "ytick.color": "#5b5f66",
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0,
    })


def find_run_csvs(root: str) -> list[str]:
    """Every *opinion_change*.csv under root (either naming scheme)."""
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if "opinion_change" in f.lower() and f.lower().endswith(".csv"):
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


def load_series(csv_path: str):
    """Return (steps, counts) where counts[r, t] = agents holding rating r
    (order -2..+2) at step index t, read straight from an opinion_change CSV."""
    df = pd.read_csv(csv_path)
    if df.empty or "time_step" not in df.columns:
        return None, None
    agent_cols = [c for c in df.columns if c != "time_step"]
    states = [-2, -1, 0, 1, 2]
    idx = {s: i for i, s in enumerate(states)}
    steps = df["time_step"].to_numpy()
    counts = np.zeros((5, len(df)), dtype=float)
    for t in range(len(df)):
        row = pd.to_numeric(df.iloc[t][agent_cols], errors="coerce").dropna()
        for v in row:
            iv = int(round(v))
            if iv in idx:
                counts[idx[iv], t] += 1
    return steps, counts


def _summaries(counts):
    states = np.array([-2, -1, 0, 1, 2], dtype=float)
    totals = counts.sum(axis=0)
    totals[totals == 0] = 1.0
    shares = counts / totals
    mean_b = (states[:, None] * shares).sum(axis=0)
    var = ((states[:, None] - mean_b[None, :]) ** 2 * shares).sum(axis=0)
    diversity = np.sqrt(var)
    simpson = (shares ** 2).sum(axis=0)
    simpson[simpson == 0] = np.nan
    eff_clusters = 1.0 / simpson
    final_share = shares[:, -1]
    return mean_b, diversity, eff_clusters, final_share


def compare_runs(specs, labels, out_path):
    """Build the 2x2 comparison figure for the given opinion_change CSVs."""
    _apply_style()
    runs = []
    for spec, lab in zip(specs, labels):
        steps, counts = load_series(spec)
        if counts is not None and counts.shape[1] >= 2:
            runs.append((lab, steps, _summaries(counts)))
    if not runs:
        raise SystemExit("no readable runs among the selected folders")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    (ax_b, ax_d), (ax_f, ax_dist) = axes

    for i, (lab, steps, (mean_b, div, eff, _fs)) in enumerate(runs):
        c = _PALETTE[i % len(_PALETTE)]
        ax_b.plot(steps, mean_b, color=c, label=lab)
        ax_d.plot(steps, div, color=c, label=lab)
        ax_f.plot(steps, eff, color=c, label=lab)
    ax_b.set_title("Mean opinion B(t)"); ax_b.set_xlabel("Time step"); ax_b.set_ylabel("B")
    ax_b.axhline(0, color="#adb5bd", linewidth=1.0)
    ax_b.set_ylim(-2.1, 2.1); ax_b.legend()
    ax_d.set_title("Diversity D(t) = population std"); ax_d.set_xlabel("Time step"); ax_d.set_ylabel("D")
    ax_f.set_title("Fragmentation (effective clusters, 1/Sum p^2)")
    ax_f.set_xlabel("Time step"); ax_f.set_ylabel("clusters")
    ax_f.axhline(1.0, color="#2166ac", linewidth=1.0, alpha=0.6)
    ax_f.axhline(2.0, color="#b2182b", linewidth=1.0, alpha=0.4)
    ax_f.set_ylim(0.8, 5.2)

    # grouped final distribution
    states = [-2, -1, 0, 1, 2]
    n = len(runs)
    w = 0.8 / max(1, n)
    x = np.arange(5)
    for i, (lab, _s, (_b, _d, _e, fs)) in enumerate(runs):
        ax_dist.bar(x + i * w - 0.4 + w / 2, fs, width=w,
                    color=_PALETTE[i % len(_PALETTE)], label=lab, edgecolor="white", linewidth=0.5)
    ax_dist.set_title("Final opinion distribution (share)")
    ax_dist.set_xticks(x); ax_dist.set_xticklabels([f"{s:+d}" for s in states])
    ax_dist.set_xlabel("Rating"); ax_dist.set_ylabel("share")
    ax_dist.grid(axis="x", alpha=0)

    fig.suptitle("Run comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path, [r[0] for r in runs]


# --------------------------------------------------------------------- GUI

def _launch_gui():
    import json
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    settings_path = os.path.join(HERE, ".compare_runs_settings.json")

    class CompareLauncher(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Compare runs")
            self.geometry("900x560")
            self.rows = []  # {"csv","label_var","frame"}
            try:
                self._settings = json.load(open(settings_path, encoding="utf-8"))
            except Exception:
                self._settings = {}
            top = ttk.Frame(self, padding=10); top.pack(fill="both", expand=True)
            head = ttk.Frame(top); head.pack(fill="x")
            ttk.Label(head, text="Runs to compare", font=("", 11, "bold")).pack(side="left")
            self.count_lbl = ttk.Label(head, text="(0)", foreground="#666"); self.count_lbl.pack(side="left", padx=8)
            ttk.Button(head, text="Add folder...", command=self.add_folder).pack(side="right", padx=4)
            ttk.Button(head, text="Clear", command=self.clear).pack(side="right", padx=4)
            ttk.Label(top, text="Pick 2-6 run folders (or a parent - every run inside is added). Each run's "
                               "condition label is editable and shown in the figure legend. The comparison is "
                               "recomputed from each run's opinion_change trajectory, so the runs do NOT need to "
                               "have been plotted first.", foreground="#666", wraplength=860, justify="left").pack(anchor="w", pady=(2, 6))
            wrap = ttk.Frame(top); wrap.pack(fill="both", expand=True)
            self.canvas = tk.Canvas(wrap, highlightthickness=0, height=260)
            vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
            self.rows_frame = ttk.Frame(self.canvas)
            self.rows_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=vsb.set)
            self.canvas.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")
            out = ttk.Frame(top); out.pack(fill="x", pady=(8, 0))
            ttk.Label(out, text="Output PNG:").pack(side="left")
            self.out_var = tk.StringVar(value=self._settings.get("out", os.path.join(HERE, "run_comparison.png")))
            ttk.Entry(out, textvariable=self.out_var, width=64).pack(side="left", padx=6, fill="x", expand=True)
            ttk.Button(out, text="...", command=self._pick_out).pack(side="left")
            bar = ttk.Frame(top); bar.pack(fill="x", pady=(10, 4))
            ttk.Button(bar, text="Build comparison", command=self.run).pack(side="left")
            ttk.Label(bar, text="Overlays B(t), D(t), fragmentation, and the final distribution.",
                      foreground="#666").pack(side="left", padx=8)
            self.status = tk.Text(top, height=6, state="disabled", background="#f4f4f4"); self.status.pack(fill="x", pady=(6, 0))
            for item in self._settings.get("rows", []):
                if os.path.exists(item.get("csv", "")):
                    self._add(item["csv"], item.get("label", ""))
            self._log("Ready. Add run folders to compare.")

        def _log(self, m):
            self.status.configure(state="normal"); self.status.insert("end", m + "\n")
            self.status.see("end"); self.status.configure(state="disabled"); self.update_idletasks()

        def _update_count(self):
            self.count_lbl.configure(text=f"({len(self.rows)})")

        def _add(self, csv_path, label=""):
            if any(r["csv"] == csv_path for r in self.rows):
                return
            fr = ttk.Frame(self.rows_frame); fr.pack(fill="x", pady=1)
            var = tk.StringVar(value=label or _auto_label(csv_path))
            ttk.Button(fr, text="X", width=2, command=lambda: self._remove(fr)).pack(side="left")
            ttk.Entry(fr, textvariable=var, width=26).pack(side="left", padx=(2, 6))
            ttk.Label(fr, text=os.path.basename(os.path.dirname(csv_path)), anchor="w").pack(side="left", fill="x", expand=True)
            self.rows.append({"csv": csv_path, "label_var": var, "frame": fr}); self._update_count()

        def _remove(self, fr):
            self.rows = [r for r in self.rows if r["frame"] is not fr]; fr.destroy(); self._update_count()

        def clear(self):
            for r in self.rows:
                r["frame"].destroy()
            self.rows = []; self._update_count()

        def add_folder(self):
            d = filedialog.askdirectory(title="Pick a run folder or a parent folder",
                                        initialdir=self._settings.get("last_dir", ""))
            if not d:
                return
            self._settings["last_dir"] = d
            found = find_run_csvs(d)
            if not found:
                messagebox.showwarning("No runs", "No opinion_change CSV found under that folder.")
                return
            for c in found:
                self._add(c)
            self._log(f"Added {len(found)} run(s) from {d}")

        def _pick_out(self):
            f = filedialog.asksaveasfilename(title="Comparison PNG", defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
            if f:
                self.out_var.set(f)

        def _save(self):
            self._settings.update({
                "rows": [{"csv": r["csv"], "label": r["label_var"].get().strip()} for r in self.rows],
                "out": self.out_var.get().strip(),
            })
            try:
                json.dump(self._settings, open(settings_path, "w", encoding="utf-8"), indent=2)
            except Exception:
                pass

        def run(self):
            if len(self.rows) < 2:
                messagebox.showwarning("Need 2+", "Add at least two runs to compare.")
                return
            self._save()
            specs = [r["csv"] for r in self.rows]
            labels = [r["label_var"].get().strip() or _auto_label(r["csv"]) for r in self.rows]
            out = self.out_var.get().strip() or os.path.join(HERE, "run_comparison.png")
            try:
                path, used = compare_runs(specs, labels, out)
                self._log(f"Wrote {path} ({len(used)} runs)")
                try:
                    if hasattr(os, "startfile"):
                        os.startfile(path)
                    else:
                        import webbrowser
                        webbrowser.open("file://" + os.path.abspath(path))
                except Exception:
                    pass
            except Exception as e:
                self._log(f"ERROR: {e}")

    CompareLauncher().mainloop()


def _auto_label(csv_path: str) -> str:
    """Distinguishing short label from a run's folder name: the config tag plus
    the abbreviated version, so two runs of the same topic but different settings
    (e.g. ...cragstrict vs ...crag) do not collapse to the same legend entry."""
    import re
    folder = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
    m = re.match(r"^seed\d+_(?P<cfg>.+?)_\d+_\d+_(?P<ver>v\d+_.*?)(?:_\d{6,8}_[a-z_]+)?$", folder)
    if m:
        return f"{m.group('cfg')}_{run_naming.abbrev_version(m.group('ver'))}"
    m = re.search(r"(v\d+_[a-z_]+?)(?:_\d{6,8}_[a-z_]+)?$", folder)
    return run_naming.abbrev_version(m.group(1)) if m else folder[:24]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare finished runs on one figure.")
    ap.add_argument("runs", nargs="*", help="run folders or opinion_change CSVs")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "run_comparison.png"))
    ap.add_argument("--gui", action="store_true", help="launch the picker GUI")
    a = ap.parse_args()
    if a.gui or not a.runs:
        _launch_gui()
        return
    specs = []
    for r in a.runs:
        if os.path.isdir(r):
            specs.extend(find_run_csvs(r))
        else:
            specs.append(r)
    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    if len(labels) != len(specs):
        labels = [_auto_label(s) for s in specs]
    path, used = compare_runs(specs, labels, a.out)
    print(f"wrote {path} ({len(used)} runs)")


if __name__ == "__main__":
    main()
