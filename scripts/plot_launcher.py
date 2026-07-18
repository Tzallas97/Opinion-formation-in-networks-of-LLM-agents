#!/usr/bin/env python3
"""GUI launcher for the plotting pipeline (scripts/main_plot_network.py).

Point it at a results folder; it scans recursively for finished runs (any
*opinion_change*.csv), auto-parses every argument main_plot_network needs from
the file name and folder layout, and runs the full plotting pipeline on each
run you tick. Nothing has to be typed by hand: agents, steps, seed, version,
date, distribution, and model are read back from the run itself.

Run:  python scripts/plot_launcher.py
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PLOT = os.path.join(HERE, "main_plot_network.py")
SETTINGS_PATH = os.path.join(HERE, ".plot_launcher_settings.json")
sys.path.insert(0, HERE)
import run_naming  # shared mode-abbreviation + run-stem naming

# Two opinion_change CSV name schemes, both must parse:
#   long (older runs): <out>_<agents>_<steps>_<version-full>_network_opinion_change_<date>_<dist>.csv
#   short (new/thesis): <out>_<agents>_<steps>_<version-short>_opinion_change.csv   (no date/dist/infix)
_CSV_RE_LONG = re.compile(
    r"^(?P<out>.+?)_(?P<agents>\d+)_(?P<steps>\d+)_(?P<version>v\d+.*?)"
    r"_(?:network_)?opinion_change_(?P<date>\d+)_(?P<dist>[A-Za-z0-9_]+?)\.csv$"
)
_CSV_RE_SHORT = re.compile(
    r"^(?P<out>.+?)_(?P<agents>\d+)_(?P<steps>\d+)_(?P<version>v\d+.*?)"
    r"_(?:network_)?opinion_change\.csv$"
)


def parse_run_csv(csv_path: str) -> dict | None:
    """Recover every main_plot_network argument from one opinion_change CSV path,
    in either the long (date+dist) or short (thesis) naming scheme.

    Returns None when the file name matches neither pattern (the run is then
    reported as unparseable rather than silently mis-plotted)."""
    base = os.path.basename(csv_path)
    m = _CSV_RE_LONG.match(base)
    date, dist = "20231212", "uniform"
    if m:
        g = m.groupdict()
        date, dist = g["date"], g["dist"]
    else:
        m = _CSV_RE_SHORT.match(base)
        if not m:
            return None
        g = m.groupdict()
    seed_m = re.search(r"seed(\d+)", g["out"])
    # model folder = the directory two levels above the CSV
    #   results/opinion_dynamics/<exp>/<model>/<run folder>/<csv>
    model = os.path.basename(os.path.dirname(os.path.dirname(csv_path))) or "qwen3_8b"
    # main_plot's -v selects the prompt folder, so hand it the FULL version;
    # it finds the file (either scheme) and writes outputs in the short scheme.
    return {
        "csv": csv_path,
        "out": g["out"],
        "agents": int(g["agents"]),
        "steps": int(g["steps"]),
        "version": run_naming.canonical_version(g["version"]),
        "date": date,
        "dist": dist,
        "seed": int(seed_m.group(1)) if seed_m else 1,
        "model": model,
    }


def find_run_csvs(root: str) -> list[str]:
    """Every *opinion_change*.csv under root (sorted, newest layout or old)."""
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if "opinion_change" in f.lower() and f.lower().endswith(".csv"):
                hits.append(os.path.join(dirpath, f))
    return sorted(hits)


class PlotLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plot launcher - main_plot_network")
        self.geometry("1000x680")
        self.minsize(820, 560)
        self._settings = self._load_settings()
        self.rows = []  # each: {"info": dict, "var": BooleanVar, "frame": Frame}
        self._proc = None

        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        # --- source ---------------------------------------------------------
        head = ttk.Frame(top)
        head.pack(fill="x")
        ttk.Label(head, text="Runs to plot", font=("", 11, "bold")).pack(side="left")
        self.count_lbl = ttk.Label(head, text="(0 runs)", foreground="#666")
        self.count_lbl.pack(side="left", padx=8)
        ttk.Button(head, text="Scan folder...", command=self.scan_folder).pack(side="right", padx=4)
        ttk.Button(head, text="Select all", command=lambda: self._set_all(True)).pack(side="right", padx=4)
        ttk.Button(head, text="Select none", command=lambda: self._set_all(False)).pack(side="right", padx=4)
        ttk.Label(top, text="Pick a results folder (a single run, a model folder, or the whole results/ tree). "
                            "Every finished run inside is found by its *opinion_change*.csv, and agents / steps / "
                            "seed / version / date / distribution / model are read back from each run automatically - "
                            "nothing is typed by hand. Untick any run you do not want plotted.",
                  foreground="#666", wraplength=960, justify="left").pack(anchor="w", pady=(2, 6))

        # filter
        frow = ttk.Frame(top)
        frow.pack(fill="x")
        ttk.Label(frow, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar()
        ttk.Entry(frow, textvariable=self.filter_var, width=44).pack(side="left", padx=6)
        self.filter_var.trace_add("write", lambda *_a: self._apply_filter())
        ttk.Label(frow, text="(substring match on the run name)", foreground="#666").pack(side="left")

        # scrollable run list
        wrap = ttk.Frame(top)
        wrap.pack(fill="both", expand=True, pady=(4, 0))
        self.canvas = tk.Canvas(wrap, highlightthickness=0, height=260)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.rows_frame = ttk.Frame(self.canvas)
        self.rows_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_wheel)
        self.canvas.bind_all("<Button-5>", self._on_wheel)

        # --- options --------------------------------------------------------
        opt = ttk.LabelFrame(top, text="Output options", padding=8)
        opt.pack(fill="x", pady=(8, 0))
        ttk.Label(opt, text="Figure type:").grid(row=0, column=0, sticky="w")
        self.fig_var = tk.StringVar(value=self._settings.get("figure_file_type", "png"))
        ttk.Combobox(opt, textvariable=self.fig_var, values=["png", "pdf"], width=6,
                     state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(opt, text="png = raster, best for quick viewing and the animated GIF frames; "
                            "pdf = vector, best for putting a figure straight into a paper (scales without blurring).",
                  foreground="#666", wraplength=900, justify="left").grid(row=1, column=0, columnspan=4, sticky="w")

        self.no_annot_var = tk.BooleanVar(value=bool(self._settings.get("no_annotation", False)))
        ttk.Checkbutton(opt, text="Clean figures (no titles / axis labels)",
                        variable=self.no_annot_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="On = strips titles and axis labels from the main trajectory figure for a bare, "
                            "presentation-ready image. Leave OFF while analysing - the labels tell you what you are looking at.",
                  foreground="#666", wraplength=900, justify="left").grid(row=3, column=0, columnspan=4, sticky="w")

        self.open_after_var = tk.BooleanVar(value=bool(self._settings.get("open_after", True)))
        ttk.Checkbutton(opt, text="Open the plots folder when done",
                        variable=self.open_after_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(opt, text="Plots are written under results/opinion_dynamics/<experiment>/<model>/plots/<version>/<run>/ - "
                            "the same tree the run lives in. This opens that folder for the first plotted run.",
                  foreground="#666", wraplength=900, justify="left").grid(row=5, column=0, columnspan=4, sticky="w")

        # --- what gets generated (reference) --------------------------------
        info = ttk.LabelFrame(top, text="What the pipeline generates per run", padding=8)
        info.pack(fill="x", pady=(8, 0))
        ttk.Label(info, text="Distribution: initial / final histograms, paired initial-vs-final, opinion x time heatmap, "
                            "fragmentation (effective number of opinion clusters over time), animated distribution GIF.  "
                            "Trajectories: all-agent overlay, per-agent small multiples, B/D/P time series, step movement, "
                            "cumulative drift.  Network: degree distribution, assortativity over time, BA hub assignment, "
                            "ego-network timelines, neighbourhood alignment, edge belief differential, animated + 3D network.  "
                            "Mechanisms: empirical Markov transition matrix, influence matrix, argument-theme effectiveness, "
                            "RAG evidence-direction breakdown, repair/cue diagnostics, exposure ecology, run dashboard.  "
                            "Each plot is skipped with a note if its source CSV is missing - the run never crashes.",
                  foreground="#666", wraplength=960, justify="left").pack(anchor="w")
        ttk.Label(info, text="Full guide: docs/main_plot_guide.md", foreground="#268bd2").pack(anchor="w", pady=(4, 0))

        # --- run bar --------------------------------------------------------
        runbar = ttk.Frame(top)
        runbar.pack(fill="x", pady=(10, 4))
        self.run_btn = ttk.Button(runbar, text="Generate plots", command=self.run)
        self.run_btn.pack(side="left")
        ttk.Label(runbar, text="Runs the pipeline on every ticked run, one after another. Output streams below.",
                  foreground="#666").pack(side="left", padx=8)

        self.status = tk.Text(top, height=9, state="disabled", background="#f4f4f4")
        self.status.pack(fill="both", expand=False, pady=(6, 0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        last = self._settings.get("last_dir", "")
        if last and os.path.isdir(last):
            self._scan_into_list(last, announce=False)
        self._log("Ready. Scan a results folder to begin.")

    # ------------------------------------------------------------------ list
    def _on_wheel(self, event):
        try:
            if getattr(event, "delta", 0):
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif getattr(event, "num", None) == 4:
                self.canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                self.canvas.yview_scroll(3, "units")
        except Exception:
            pass

    def _update_count(self):
        ticked = sum(1 for r in self.rows if r["var"].get())
        try:
            self.count_lbl.configure(text=f"({ticked}/{len(self.rows)} runs)")
        except Exception:
            pass

    def _set_all(self, val):
        q = self.filter_var.get().strip().lower()
        for r in self.rows:
            if not q or q in r["info"]["out"].lower():
                r["var"].set(val)
        self._update_count()

    def _apply_filter(self):
        q = self.filter_var.get().strip().lower()
        for r in self.rows:
            r["frame"].pack_forget()
            if not q or q in r["info"]["out"].lower():
                r["frame"].pack(fill="x", pady=1)

    def scan_folder(self):
        d = filedialog.askdirectory(title="Pick a results folder (run, model, or the whole tree)",
                                    initialdir=self._settings.get("last_dir", ""))
        if not d:
            return
        self._scan_into_list(d, announce=True)

    def _scan_into_list(self, d, announce=True):
        self._settings["last_dir"] = d
        for r in self.rows:
            r["frame"].destroy()
        self.rows = []
        csvs = find_run_csvs(d)
        parsed, bad = [], 0
        for c in csvs:
            info = parse_run_csv(c)
            if info:
                parsed.append(info)
            else:
                bad += 1
        parsed.sort(key=lambda i: i["out"])
        preticked = len(parsed) <= 12
        for info in parsed:
            self._add_row(info, preticked)
        self._update_count()
        if announce:
            msg = f"Found {len(parsed)} run(s) under {d}"
            if bad:
                msg += f" ({bad} file(s) skipped - name did not match the run pattern)"
            self._log(msg)

    def _add_row(self, info, ticked):
        fr = ttk.Frame(self.rows_frame)
        fr.pack(fill="x", pady=1)
        var = tk.BooleanVar(value=ticked)
        var.trace_add("write", lambda *_a: self._update_count())
        ttk.Checkbutton(fr, variable=var).pack(side="left")
        label = (f"{info['out']}  —  {info['agents']}a/{info['steps']}s  "
                 f"seed {info['seed']}  {info['version']}  {info['model']}  [{info['dist']}]")
        ttk.Label(fr, text=label, anchor="w").pack(side="left", fill="x", expand=True)
        self.rows.append({"info": info, "var": var, "frame": fr})

    # ------------------------------------------------------------------ run
    def _cmd_for(self, info):
        cmd = [sys.executable, MAIN_PLOT,
               "-agents", str(info["agents"]),
               "-steps", str(info["steps"]),
               "-seed", str(info["seed"]),
               "-v", info["version"],
               "-dist", info["dist"],
               "--date", info["date"],
               "-out", info["out"],
               "-m", info["model"],
               "--figure_file_type", self.fig_var.get().strip() or "png"]
        if self.no_annot_var.get():
            cmd.append("--no_annotation")
        return cmd

    def run(self):
        picked = [r["info"] for r in self.rows if r["var"].get()]
        if not picked:
            messagebox.showwarning("No runs", "Tick at least one run to plot.")
            return
        if self._proc is not None:
            messagebox.showinfo("Busy", "A plotting job is already running.")
            return
        self._save_settings()
        self.run_btn.configure(state="disabled")
        n = len(picked)

        def worker():
            done_dirs = []
            for i, info in enumerate(picked, 1):
                self._log(f"\n[{i}/{n}] {info['out']} ...")
                cmd = self._cmd_for(info)
                plot_dir = None
                try:
                    self._proc = subprocess.Popen(
                        cmd, cwd=os.path.dirname(HERE), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        encoding="utf-8", errors="replace")
                    for line in self._proc.stdout:
                        line = line.rstrip()
                        if line:
                            self._log(line)
                            if plot_dir is None and os.sep + "plots" + os.sep in line:
                                # capture this run's plot folder from a "Saved ... to: <path>" line
                                idx = line.find(os.sep + "plots" + os.sep)
                                start = line.rfind(" ", 0, idx) + 1
                                plot_dir = os.path.dirname(line[start:].strip())
                    code = self._proc.wait()
                    self._log(f"  [{i}/{n}] exit code {code}"
                              + (f" -> {plot_dir}" if plot_dir else ""))
                    if plot_dir:
                        done_dirs.append(plot_dir)
                except Exception as e:
                    self._log(f"  [{i}/{n}] ERROR: {e}")
                finally:
                    self._proc = None
            self._log(f"\nDone. {len(done_dirs)}/{n} run(s) produced a plot folder.")
            if self.open_after_var.get() and done_dirs:
                d0 = done_dirs[0]
                if os.path.isdir(d0):
                    try:
                        if hasattr(os, "startfile"):
                            os.startfile(d0)  # Windows
                        else:
                            import webbrowser
                            webbrowser.open("file://" + os.path.abspath(d0))
                    except Exception:
                        pass
            self._ui(lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------ misc
    def _ui(self, fn):
        """Marshal a Tk update onto the main thread. Calling widget methods from
        the worker thread directly is not safe and could stall the run loop after
        the first subprocess - every UI touch from the worker goes through here."""
        try:
            self.after(0, fn)
        except Exception:
            pass

    def _log(self, msg):
        def append():
            try:
                self.status.configure(state="normal")
                self.status.insert("end", msg + "\n")
                self.status.see("end")
                self.status.configure(state="disabled")
            except Exception:
                pass
        self._ui(append)

    def _load_settings(self):
        try:
            return json.load(open(SETTINGS_PATH, encoding="utf-8"))
        except Exception:
            return {}

    def _save_settings(self):
        self._settings.update({
            "last_dir": self._settings.get("last_dir", ""),
            "figure_file_type": self.fig_var.get().strip(),
            "no_annotation": bool(self.no_annot_var.get()),
            "open_after": bool(self.open_after_var.get()),
        })
        try:
            json.dump(self._settings, open(SETTINGS_PATH, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()


def main():
    PlotLauncher().mainloop()


if __name__ == "__main__":
    main()
