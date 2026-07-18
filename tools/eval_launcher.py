#!/usr/bin/env python3
"""Small GUI for the eval harness (tools/eval_runs.py).

Pick run folders (or any parent folder - it scans recursively and finds every
run inside), give each a condition label, choose where the report goes, and
optionally append the rows to a master CSV that accumulates across sessions.

Run:  python tools/eval_launcher.py
"""
from __future__ import annotations
import json, os, sys, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_runs  # noqa: E402  (find_run_dirs / infer_labels / run_eval)

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".eval_launcher_settings.json")


class EvalLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Eval runs - comparison launcher")
        self.geometry("980x640")
        self.rows = []  # each: {"path": str, "label": tk.StringVar, "frame": ttk.Frame}
        self._settings = self._load_settings()

        top = ttk.Frame(self, padding=10); top.pack(fill="both", expand=True)

        # --- run list -------------------------------------------------------
        head = ttk.Frame(top); head.pack(fill="x")
        ttk.Label(head, text="Runs to compare", font=("", 11, "bold")).pack(side="left")
        self.count_lbl = ttk.Label(head, text="(0 runs)", foreground="#666")
        self.count_lbl.pack(side="left", padx=8)
        ttk.Button(head, text="+ Add folder (run or parent)", command=self.add_folder).pack(side="right", padx=4)
        ttk.Button(head, text="+ Add via files (multi-select)", command=self.add_via_files).pack(side="right", padx=4)
        ttk.Button(head, text="Sort A-Z", command=self.sort_rows).pack(side="right", padx=4)
        ttk.Button(head, text="Clear all", command=self.clear_rows).pack(side="right", padx=4)
        ttk.Label(top, text="Pick a run folder directly, or any parent folder - every run found inside is added "
                           "(folders with an *opinion_change*.csv; loose CSVs dumped together are detected as separate runs).",
                  foreground="#666").pack(anchor="w", pady=(2, 6))

        cols = ttk.Frame(top); cols.pack(fill="x")
        ttk.Label(cols, text="condition label", width=24).pack(side="left", padx=(2, 4))
        ttk.Label(cols, text="run folder").pack(side="left")

        wrap = ttk.Frame(top); wrap.pack(fill="both", expand=True)
        canvas = tk.Canvas(wrap, highlightthickness=0, height=230)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        self.rows_frame = ttk.Frame(canvas)
        self.rows_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")

        # --- output ---------------------------------------------------------
        out_box = ttk.LabelFrame(top, text="Output", padding=8); out_box.pack(fill="x", pady=(8, 0))
        ttk.Label(out_box, text="Report folder:").grid(row=0, column=0, sticky="w")
        self.out_var = tk.StringVar(value=self._settings.get("out_dir", ""))
        ttk.Entry(out_box, textvariable=self.out_var, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(out_box, text="Browse", command=self.pick_out).grid(row=0, column=2)
        ttk.Button(out_box, text="Open folder", command=self.open_out).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(out_box, text="(empty = eval_report/ next to the first run)", foreground="#666").grid(row=1, column=1, sticky="w", padx=6)

        self.open_after_var = tk.BooleanVar(value=bool(self._settings.get("open_after", True)))
        ttk.Checkbutton(out_box, text="Open report when done", variable=self.open_after_var).grid(row=2, column=1, sticky="w", padx=6)

        # --- master CSV -----------------------------------------------------
        agg = ttk.LabelFrame(top, text="Aggregate CSV", padding=8); agg.pack(fill="x", pady=(8, 0))
        ttk.Label(agg, text="A fresh aggregate.csv is always written into the report folder. Optionally ALSO append to a master CSV that accumulates runs across sessions:",
                  foreground="#666", wraplength=900).grid(row=0, column=0, columnspan=3, sticky="w")
        self.append_var = tk.BooleanVar(value=bool(self._settings.get("append_enabled", False)))
        ttk.Checkbutton(agg, text="Append to master CSV:", variable=self.append_var).grid(row=1, column=0, sticky="w")
        self.master_var = tk.StringVar(value=self._settings.get("master_csv", ""))
        self.master_combo = ttk.Combobox(agg, textvariable=self.master_var, width=60,
                                         values=list(self._settings.get("recent_masters", [])))
        self.master_combo.grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(agg, text="Browse", command=self.pick_master).grid(row=1, column=2)
        ttk.Button(agg, text="Master report", command=self.run_master_report).grid(row=1, column=3, padx=(4, 0))
        self.skip_dup_var = tk.BooleanVar(value=bool(self._settings.get("skip_duplicates", True)))
        ttk.Checkbutton(agg, text="Skip runs already present in the master (by run name)", variable=self.skip_dup_var).grid(row=2, column=0, columnspan=2, sticky="w")

        # --- LLM judge --------------------------------------------------------
        jd = ttk.LabelFrame(top, text="LLM judge (read-only text scoring; needs Ollama running)", padding=8)
        jd.pack(fill="x", pady=(8, 0))
        ttk.Label(jd, text="Judge model:").grid(row=0, column=0, sticky="w")
        self.judge_model_var = tk.StringVar(value=self._settings.get("judge_model", "llama3.1:8b"))
        self.judge_model_combo = ttk.Combobox(jd, textvariable=self.judge_model_var, width=22,
                                              values=list(self._settings.get("ollama_models", [])))
        self.judge_model_combo.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Button(jd, text="Scan models", command=self.scan_models).grid(row=0, column=6, sticky="w", padx=4)
        ttk.Label(jd, text="Sample/run:").grid(row=0, column=2, sticky="e")
        self.judge_sample_var = tk.StringVar(value=str(self._settings.get("judge_sample", 120)))
        ttk.Entry(jd, textvariable=self.judge_sample_var, width=6).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(jd, text="Context:").grid(row=0, column=4, sticky="e")
        self.judge_context_var = tk.StringVar(value=self._settings.get("judge_context", "full"))
        ttk.Combobox(jd, textvariable=self.judge_context_var, values=["off", "light", "full"], width=6,
                     state="readonly").grid(row=0, column=5, sticky="w", padx=6)
        self.judge_dry_var = tk.BooleanVar(value=bool(self._settings.get("judge_dry", True)))
        ttk.Checkbutton(jd, text="Dry run (no LLM calls; writes example prompt)", variable=self.judge_dry_var).grid(row=1, column=0, columnspan=3, sticky="w")
        ttk.Label(jd, text="Calibration items:").grid(row=1, column=3, sticky="e")
        self.judge_calib_var = tk.StringVar(value=str(self._settings.get("judge_calib", 0)))
        ttk.Entry(jd, textvariable=self.judge_calib_var, width=6).grid(row=1, column=4, sticky="w", padx=6)
        self.judge_explain_var = tk.BooleanVar(value=bool(self._settings.get("judge_explain", True)))
        ttk.Checkbutton(jd, text="Explain flagged (two-pass rationale)", variable=self.judge_explain_var).grid(row=1, column=5, columnspan=2, sticky="w", padx=(12, 0))
        ttk.Label(jd, text="Judge model: which local LLM does the grading. Click 'Scan models' to list what is actually installed "
                          "(avoids name typos). Pick a DIFFERENT family than the graded runs (qwen runs -> llama judge).",
                  foreground="#666", wraplength=900).grid(row=2, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Explain flagged: a SECOND pass asks the judge to name, in plain text, exactly what outside material each "
                          "closed-world-leak item invokes. Runs only for flagged items (cheap) and only AFTER their scores are locked, "
                          "so the explanation cannot bias the verdict (CoT-judge failure mode). Rationales land in the 'rationale' "
                          "column of judge_scores_<label>.csv - read them while filling the calibration sheet.",
                  foreground="#666", wraplength=900).grid(row=8, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Sample/run: how many tweets+responses are graded per run (random but seeded, so re-runs pick the same items). "
                          "More = more reliable means, more time. 120 is a good default for a 100-step run.",
                  foreground="#666", wraplength=900).grid(row=3, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Context: how much of the run's OWN setup the judge is shown (auto-extracted from the files). "
                          "off = only the text; light = + run config and the author's persona; full = + the exact world-rules and bias wording the agents were given. "
                          "Use full for closed-world runs so leaks are judged against the real rules.",
                  foreground="#666", wraplength=900).grid(row=4, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Dry run: builds everything WITHOUT calling the LLM and saves example_prompt_<label>.txt so you can inspect exactly what the judge will see. Always do this first.",
                  foreground="#666", wraplength=900).grid(row=5, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Calibration items: exports N random items to calibration_<label>.csv with empty columns for YOUR hand scores. "
                          "Grade them yourself, compare with the judge; good agreement = the judge's numbers are trustworthy for the thesis.",
                  foreground="#666", wraplength=900).grid(row=6, column=0, columnspan=7, sticky="w")
        ttk.Label(jd, text="Full guide: docs/eval_judge_guide.md", foreground="#268bd2").grid(row=7, column=0, columnspan=7, sticky="w")

        # --- semantic analysis ----------------------------------------------
        sm = ttk.LabelFrame(top, text="Semantic (text-space) analysis - no LLM needed with TF-IDF", padding=8)
        sm.pack(fill="x", pady=(8, 0))
        ttk.Label(sm, text="Backend:").grid(row=0, column=0, sticky="w")
        self.sem_backend_var = tk.StringVar(value=self._settings.get("sem_backend", "tfidf"))
        ttk.Combobox(sm, textvariable=self.sem_backend_var, values=["tfidf", "ollama"], width=8,
                     state="readonly").grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(sm, text="Embed model (ollama):").grid(row=0, column=2, sticky="e")
        self.sem_model_var = tk.StringVar(value=self._settings.get("sem_model", "nomic-embed-text"))
        ttk.Entry(sm, textvariable=self.sem_model_var, width=20).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(sm, text="Measures what the ratings cannot: homogenisation (agents sounding alike), text convergence over time "
                          "(wording keeps converging even when integer ratings stall), stance separation (do FOR and AGAINST posts read differently?), "
                          "and self-repetition. Backend: tfidf = lexical, dependency-free, runs NOW (underestimates paraphrases); "
                          "ollama = true semantic embeddings via a local embed model, cached - use when Ollama is available.",
                  foreground="#666", wraplength=900).grid(row=1, column=0, columnspan=5, sticky="w")
        ttk.Label(sm, text="Multihop groundtruth:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.gt_path_var = tk.StringVar(value=self._settings.get("gt_path", ""))
        ttk.Entry(sm, textvariable=self.gt_path_var, width=64).grid(row=2, column=1, columnspan=3, sticky="we", padx=6, pady=(6, 0))
        ttk.Button(sm, text="Browse...", command=self._browse_gt).grid(row=2, column=4, sticky="w", pady=(6, 0))
        ttk.Label(sm, text="For the Inference emergence metric: the multihop corpus's .groundtruth.json (implied conclusions "
                          "+ null probes). Scores every tweet of the listed runs against conclusions no snippet ever states; "
                          "threshold auto-calibrated from the null probes. Uses the Backend above (tfidf now / ollama for real "
                          "semantics). Old runs that never saw the multihop corpus = negative control (should sit near the threshold).",
                  foreground="#666", wraplength=900).grid(row=3, column=0, columnspan=5, sticky="w")
        ttk.Label(sm, text="Scale info-loss: how much of the TEXT does the -2..+2 integer rating compress away? Per run: R2 "
                          "(text variance explained by the rating), leave-one-out text->rating recovery (are the 5 levels "
                          "textually real?), flat-rating drift (text moving while the integer stays put), distinct 'voices' "
                          "per rating. Needs only the run list + the Backend above; answers 'how much does the "
                          "integer scale compress away?'.",
                  foreground="#666", wraplength=900).grid(row=4, column=0, columnspan=5, sticky="w")

        # --- viewers ----------------------------------------------------------
        vw = ttk.LabelFrame(top, text="Viewers (interactive HTML, offline)", padding=8)
        vw.pack(fill="x", pady=(8, 0))
        ttk.Label(vw, text="Build single-run viewer(s): one HTML per listed run (network + scrubber + per-agent tweets/replies panel), "
                          "written as <run>_viewer.html next to each run.  Build A/B viewer: the FIRST TWO rows side-by-side with one "
                          "synchronized scrubber - ideal for condition comparisons. Viewers need run FOLDERS (loose CSV runs are skipped).",
                  foreground="#666", wraplength=900).grid(row=0, column=0, columnspan=4, sticky="w")

        # --- run ------------------------------------------------------------
        runbar = ttk.Frame(top); runbar.pack(fill="x", pady=(10, 4))
        self.run_btn = ttk.Button(runbar, text="Run comparison", command=self.run)
        self.run_btn.pack(side="left")
        self.judge_btn = ttk.Button(runbar, text="Run judge", command=self.run_judge)
        self.judge_btn.pack(side="left", padx=8)
        self.sem_btn = ttk.Button(runbar, text="Run semantic", command=self.run_semantic)
        self.sem_btn.pack(side="left")
        self.inf_btn = ttk.Button(runbar, text="Inference emergence", command=self.run_inference)
        self.inf_btn.pack(side="left", padx=8)
        self.scale_btn = ttk.Button(runbar, text="Scale info-loss", command=self.run_scale)
        self.scale_btn.pack(side="left")
        self.viewer_btn = ttk.Button(runbar, text="Build viewer(s)", command=self.build_viewers)
        self.viewer_btn.pack(side="left", padx=8)
        self.ab_btn = ttk.Button(runbar, text="Build A/B viewer", command=self.build_ab)
        self.ab_btn.pack(side="left")
        self.status = tk.Text(top, height=7, state="disabled", background="#f4f4f4")
        self.status.pack(fill="both", expand=False, pady=(6, 0))

        # restore previous session's rows
        for item in self._settings.get("rows", []):
            if os.path.exists(item.get("path", "")):
                self._add_row(item["path"], item.get("label", ""))
        self._log("Ready. Add run folders (or a parent folder) to begin.")

    # ------------------------------------------------------------------ rows
    def _update_count(self):
        try: self.count_lbl.configure(text=f"({len(self.rows)} runs)")
        except Exception: pass

    def sort_rows(self):
        """Re-pack rows alphabetically by folder name."""
        self.rows.sort(key=lambda r: eval_runs.spec_name(r["path"]).lower())
        for r in self.rows:
            r["frame"].pack_forget(); r["frame"].pack(fill="x", pady=1)

    def _add_row(self, path, label=""):
        if any(r["path"] == path for r in self.rows):
            return False
        fr = ttk.Frame(self.rows_frame)
        fr.pack(fill="x", pady=1)
        var = tk.StringVar(value=label)
        ttk.Button(fr, text="X", width=2, command=lambda: self._remove_row(fr)).pack(side="left")
        ttk.Entry(fr, textvariable=var, width=24).pack(side="left", padx=(2, 6))
        ttk.Label(fr, text=path, anchor="w").pack(side="left", fill="x", expand=True)
        self.rows.append({"path": path, "label": var, "frame": fr})
        self._update_count()
        return True

    def _remove_row(self, frame):
        self.rows = [r for r in self.rows if r["frame"] is not frame]
        frame.destroy()
        self._update_count()

    def clear_rows(self):
        for r in self.rows: r["frame"].destroy()
        self.rows = []
        self._update_count()

    def add_folder(self):
        d = filedialog.askdirectory(title="Pick a run folder OR a parent folder",
                                    initialdir=self._settings.get("last_dir", ""))
        if not d: return
        self._settings["last_dir"] = d
        found = eval_runs.find_run_specs(d)
        if not found:
            messagebox.showwarning("No runs found",
                                   "No run folders found there.\nA run folder contains an *opinion_change*.csv file.")
            return
        if len(found) > 1:
            self._pick_runs_dialog(d, found)   # tick exactly the runs you want
            return
        added = sum(1 for f in found if self._add_row(f))
        self._prefill_labels()
        self._log(f"Added {added} run(s) from {d}" + (f" ({len(found)-added} already listed)" if added < len(found) else ""))

    def add_via_files(self):
        """Native multi-select: pick one or more CSVs (Ctrl/Shift-click) inside the
        run folders you want; each selected file's parent folder is added as a run."""
        files = filedialog.askopenfilenames(title="Select CSV file(s) inside the run folders you want",
                                            initialdir=self._settings.get("last_dir", ""),
                                            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not files: return
        self._settings["last_dir"] = os.path.dirname(files[0])
        specs, skipped = [], 0
        import glob as _glob
        for f in files:
            d = os.path.dirname(f)
            base = os.path.basename(f).lower()
            ocs_here = _glob.glob(os.path.join(d, "*opinion_change*.csv"))
            if "opinion_change" in base and len(ocs_here) > 1:
                spec = f            # several runs share this folder -> the CSV itself is the run
            elif eval_runs.is_run_dir(d):
                spec = d            # normal case -> the parent folder is the run
            else:
                skipped += 1; continue
            if spec not in specs: specs.append(spec)
        added = sum(1 for s in specs if self._add_row(s))
        self._prefill_labels()
        msg = f"Added {added} run(s) via file selection"
        if skipped: msg += f" ({skipped} file(s) skipped - no opinion_change CSV found beside them)"
        self._log(msg)

    def _pick_runs_dialog(self, root_dir, found):
        """Checklist so a parent folder pick becomes a multi-select of its runs."""
        win = tk.Toplevel(self); win.title(f"Select runs - {len(found)} found"); win.geometry("760x480")
        ttk.Label(win, text=f"Runs found under {root_dir}:").pack(anchor="w", padx=10, pady=(8, 2))
        filt_var = tk.StringVar()
        frow = ttk.Frame(win); frow.pack(fill="x", padx=10)
        ttk.Label(frow, text="Filter:").pack(side="left")
        ttk.Entry(frow, textvariable=filt_var, width=40).pack(side="left", padx=6)
        wrap = ttk.Frame(win); wrap.pack(fill="both", expand=True, padx=10, pady=6)
        canvas = tk.Canvas(wrap, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")
        items = []  # (var, path, checkbutton, base)
        for f in found:
            base = os.path.relpath(f, root_dir)
            text = ("loose run: " + base) if os.path.isfile(f) else base
            v = tk.BooleanVar(value=(len(found) <= 12))  # preticked when the list is small
            cb = ttk.Checkbutton(inner, text=text, variable=v)
            cb.pack(anchor="w")
            items.append((v, f, cb, text.lower()))
        def apply_filter(*_a):
            q = filt_var.get().strip().lower()
            for v, f, cb, base in items:
                cb.pack_forget()
                if not q or q in base: cb.pack(anchor="w")
        filt_var.trace_add("write", apply_filter)
        btns = ttk.Frame(win); btns.pack(fill="x", padx=10, pady=(0, 10))
        def set_all(val):
            q = filt_var.get().strip().lower()
            for v, f, cb, base in items:
                if not q or q in base: v.set(val)
        ttk.Button(btns, text="Select all", command=lambda: set_all(True)).pack(side="left")
        ttk.Button(btns, text="Select none", command=lambda: set_all(False)).pack(side="left", padx=6)
        def ok():
            picked = [f for v, f, cb, base in items if v.get()]
            win.destroy()
            added = sum(1 for f in picked if self._add_row(f))
            self._prefill_labels()
            self._log(f"Added {added} run(s) from {root_dir} ({len(picked)-added} already listed)" if picked else "Nothing selected.")
        ttk.Button(btns, text="Add selected", command=ok).pack(side="right")
        win.transient(self); win.grab_set()

    def _prefill_labels(self):
        """Fill empty labels with the distinguishing part of the folder names."""
        if len(self.rows) < 2: return
        fake = [{"name": eval_runs.spec_name(r["path"])} for r in self.rows]
        auto = eval_runs.infer_labels(fake)
        for r in self.rows:
            if not r["label"].get().strip():
                r["label"].set(auto[eval_runs.spec_name(r["path"])])

    # ------------------------------------------------------------------ misc
    def pick_out(self):
        d = filedialog.askdirectory(title="Report output folder")
        if d: self.out_var.set(d)

    def open_out(self):
        d = self.out_var.get().strip()
        if not d and self.rows:
            d = os.path.join(os.path.dirname(self.rows[0]["path"]) or ".", "eval_report")
        if d and os.path.isdir(d):
            try:
                if hasattr(os, "startfile"): os.startfile(d)  # Windows
                else:
                    import webbrowser; webbrowser.open("file://" + os.path.abspath(d))
            except Exception: pass
        else:
            self._log("Output folder does not exist yet.")

    def pick_master(self):
        f = filedialog.asksaveasfilename(title="Master CSV", defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")], confirmoverwrite=False)
        if f: self.master_var.set(f)

    def _log(self, msg):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n"); self.status.see("end")
        self.status.configure(state="disabled")
        self.update_idletasks()

    def _load_settings(self):
        try: return json.load(open(SETTINGS_PATH, encoding="utf-8"))
        except Exception: return {}

    def _save_settings(self):
        self._settings.update({
            "rows": [{"path": r["path"], "label": r["label"].get().strip()} for r in self.rows],
            "out_dir": self.out_var.get().strip(),
            "open_after": bool(self.open_after_var.get()),
            "append_enabled": bool(self.append_var.get()),
            "master_csv": self.master_var.get().strip(),
            "skip_duplicates": bool(self.skip_dup_var.get()),
            "judge_model": self.judge_model_var.get().strip(),
            "judge_sample": self.judge_sample_var.get().strip(),
            "judge_context": self.judge_context_var.get().strip(),
            "judge_dry": bool(self.judge_dry_var.get()),
            "judge_calib": self.judge_calib_var.get().strip(),
            "judge_explain": bool(self.judge_explain_var.get()),
            "sem_backend": self.sem_backend_var.get().strip(),
            "gt_path": self.gt_path_var.get().strip(),
            "sem_model": self.sem_model_var.get().strip(),
        })
        master = self.master_var.get().strip()
        if master:
            rec = [master] + [m for m in self._settings.get("recent_masters", []) if m != master]
            self._settings["recent_masters"] = rec[:8]
            try: self.master_combo.configure(values=rec[:8])
            except Exception: pass
        try: json.dump(self._settings, open(SETTINGS_PATH, "w", encoding="utf-8"), indent=1)
        except Exception: pass

    # ------------------------------------------------------------------ run
    def run(self):
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        run_dirs = [r["path"] for r in self.rows]
        labels = [r["label"].get().strip() for r in self.rows]
        # duplicate labels would collapse conditions in the report -> auto-suffix _2, _3, ...
        seen = {}
        for i, l in enumerate(labels):
            if not l: continue
            if l in seen:
                seen[l] += 1
                labels[i] = f"{l}_{seen[l]}"
                self._log(f"Duplicate label '{l}' -> renamed to '{labels[i]}'")
            else:
                seen[l] = 1
        out = self.out_var.get().strip() or None
        master = self.master_var.get().strip() if self.append_var.get() else None
        if self.append_var.get() and not master:
            messagebox.showwarning("Master CSV", "Choose a master CSV path (or untick append)."); return
        self._save_settings()
        self.run_btn.configure(state="disabled")
        self._log(f"Running on {len(run_dirs)} run(s)...")
        try:
            res = eval_runs.run_eval(run_dirs, labels=labels, out=out,
                                     append_csv=master, skip_duplicates=bool(self.skip_dup_var.get()))
            for d, e in res.get("load_errors", []):
                self._log(f"SKIPPED (failed to load): {d} -> {e}")
            self._log(f"Report: {res['report']}")
            self._log(f"Aggregate: {res['aggregate']}")
            if master:
                self._log(f"Master CSV: +{res['appended']} appended, {res['skipped']} skipped (already present)")
            if self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(res["report"])  # Windows
                    else:
                        import webbrowser; webbrowser.open("file://" + os.path.abspath(res["report"]))
                except Exception: pass
            self._log("Done.")
        except Exception as e:
            self._log("ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.run_btn.configure(state="normal")

    # ------------------------------------------------------------------ judge
    def scan_models(self):
        """Query the local Ollama for installed models and fill the dropdown."""
        import judge_runs
        models = judge_runs.list_ollama_models()
        if models:
            self.judge_model_combo.configure(values=models)
            self._settings["ollama_models"] = models
            if self.judge_model_var.get().strip() not in models:
                self.judge_model_var.set(models[0])
            self._log(f"Ollama models found: {', '.join(models)}")
        else:
            self._log("Could not reach Ollama (is it running?). Type the model name manually.")

    def run_judge(self):
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        import judge_runs  # lazy import
        specs = [r["path"] for r in self.rows]
        labels = [r["label"].get().strip() or eval_runs.spec_name(r["path"]) for r in self.rows]
        out = self.out_var.get().strip() or None
        out = os.path.join(out, "judge") if out else None
        try: sample = max(1, int(self.judge_sample_var.get().strip() or "120"))
        except Exception: sample = 120
        try: calib = max(0, int(self.judge_calib_var.get().strip() or "0"))
        except Exception: calib = 0
        self._save_settings()
        self.judge_btn.configure(state="disabled")
        dry = bool(self.judge_dry_var.get())
        self._log(f"Judge: {len(specs)} run(s), model {self.judge_model_var.get()}, "
                  f"context {self.judge_context_var.get()}, sample {sample}{', DRY RUN' if dry else ''}")
        try:
            argv = ["judge_runs.py", *specs, "--labels", ",".join(labels),
                    "--model", self.judge_model_var.get().strip() or "llama3.1:8b",
                    "--sample", str(sample), "--context", self.judge_context_var.get()]
            if out: argv += ["--out", out]
            if dry: argv += ["--dry-run"]
            if calib: argv += ["--export-calibration", str(calib)]
            argv += ["--explain-flagged", "on" if self.judge_explain_var.get() else "off"]
            old_argv, old_print = sys.argv, None
            sys.argv = argv
            # capture judge prints into the status log
            import builtins
            old_print = builtins.print
            builtins.print = lambda *a, **k: self._log(" ".join(str(x) for x in a))
            try:
                judge_runs.main()
            finally:
                sys.argv = old_argv; builtins.print = old_print
            self._log("Judge done." + ("" if dry else " Check judge_comparison.md."))
        except SystemExit as e:
            self._log(f"Judge aborted: {e}")
        except Exception as e:
            self._log("JUDGE ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.judge_btn.configure(state="normal")

    def build_viewers(self):
        """One interactive HTML viewer per listed run (folders only)."""
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        import json as _json
        import build_viewer as BV
        self.viewer_btn.configure(state="disabled")
        built, skipped = [], 0
        try:
            for r in self.rows:
                path = r["path"]
                if not os.path.isdir(path):
                    skipped += 1; continue   # loose CSV runs: single viewer needs a folder
                try:
                    data = BV.build_data(path.rstrip("/\\"))
                    out = path.rstrip("/\\") + "_viewer.html"
                    open(out, "w", encoding="utf-8").write(
                        BV.TEMPLATE.replace("__DATA__", _json.dumps(data, ensure_ascii=False)))
                    built.append(out)
                    self._log(f"viewer -> {out}")
                except Exception as e:
                    self._log(f"SKIPPED {os.path.basename(path)}: {e}")
            if skipped:
                self._log(f"{skipped} loose-CSV run(s) skipped (viewers need run folders)")
            if built and self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(built[0])
                except Exception: pass
            self._log(f"Built {len(built)} viewer(s).")
        finally:
            self.viewer_btn.configure(state="normal")

    def build_ab(self):
        """Side-by-side A/B viewer from the FIRST TWO rows."""
        dirs = [r for r in self.rows if os.path.isdir(r["path"])]
        if len(dirs) < 2:
            messagebox.showwarning("Need two runs", "Add at least two run FOLDERS (the first two rows are used)."); return
        import json as _json
        import build_viewer as BV, build_ab_viewer as AB
        a, b = dirs[0], dirs[1]
        la = a["label"].get().strip() or eval_runs.spec_name(a["path"])
        lb = b["label"].get().strip() or eval_runs.spec_name(b["path"])
        self.ab_btn.configure(state="disabled")
        try:
            runs = [BV.build_data(a["path"].rstrip("/\\")), BV.build_data(b["path"].rstrip("/\\"))]
            out = os.path.join(os.path.dirname(a["path"].rstrip("/\\")) or ".",
                               f"ab_viewer_{la}_VS_{lb}.html".replace(" ", "_"))
            html = AB.TEMPLATE.replace("__DATA__", _json.dumps(runs, ensure_ascii=False)).replace("__LABELS__", _json.dumps([la, lb], ensure_ascii=False))
            open(out, "w", encoding="utf-8").write(html)
            self._log(f"A/B viewer -> {out}")
            if self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(out)
                except Exception: pass
        except Exception as e:
            self._log("A/B ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.ab_btn.configure(state="normal")

    def run_master_report(self):
        """Per-condition summary (mean±std across runs) over the master CSV."""
        master = self.master_var.get().strip()
        if not master or not os.path.exists(master):
            messagebox.showwarning("Master CSV", "Pick an existing master CSV first (Browse)."); return
        import master_report
        old_argv = sys.argv
        try:
            sys.argv = ["master_report.py", master]
            import builtins
            old_print = builtins.print
            builtins.print = lambda *a, **k: self._log(" ".join(str(x) for x in a))
            try:
                master_report.main()
            finally:
                builtins.print = old_print
            rep_path = os.path.join(os.path.dirname(os.path.abspath(master)) or ".", "master_report", "master_report.md")
            if self.open_after_var.get() and os.path.exists(rep_path):
                try:
                    if hasattr(os, "startfile"): os.startfile(rep_path)
                except Exception: pass
        except SystemExit as e:
            self._log(f"Master report aborted: {e}")
        except Exception as e:
            self._log("MASTER REPORT ERROR: " + str(e))
        finally:
            sys.argv = old_argv

    def _browse_gt(self):
        p = filedialog.askopenfilename(title="Pick the multihop .groundtruth.json",
                                       filetypes=[("groundtruth", "*.groundtruth.json"), ("json", "*.json"), ("all", "*.*")])
        if p:
            self.gt_path_var.set(p)

    def run_inference(self):
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        gt = self.gt_path_var.get().strip()
        if not gt or not os.path.exists(gt):
            messagebox.showwarning("No groundtruth", "Pick the multihop corpus .groundtruth.json first."); return
        import inference_emergence
        specs = [r["path"] for r in self.rows]
        labels = [r["label"].get().strip() or eval_runs.spec_name(r["path"]) for r in self.rows]
        out = self.out_var.get().strip() or None
        out = os.path.join(out, "inference") if out else None
        self._save_settings()
        self.inf_btn.configure(state="disabled")
        backend = self.sem_backend_var.get().strip() or "tfidf"
        self._log(f"Inference emergence: {len(specs)} run(s), backend {backend}")
        try:
            report = inference_emergence.run_inference(
                specs, labels, gt, backend=backend,
                embed_model=self.sem_model_var.get().strip() or "nomic-embed-text",
                out_dir=out, log=self._log)
            if self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(report)
                    else:
                        import webbrowser; webbrowser.open("file://" + os.path.abspath(report))
                except Exception: pass
            self._log("Inference emergence done.")
        except Exception as e:
            self._log("INFERENCE ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.inf_btn.configure(state="normal")

    def run_scale(self):
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        import scale_infoloss
        specs = [r["path"] for r in self.rows]
        labels = [r["label"].get().strip() or eval_runs.spec_name(r["path"]) for r in self.rows]
        out = self.out_var.get().strip() or None
        out = os.path.join(out, "scale_infoloss") if out else None
        self._save_settings()
        self.scale_btn.configure(state="disabled")
        backend = self.sem_backend_var.get().strip() or "tfidf"
        self._log(f"Scale info-loss: {len(specs)} run(s), backend {backend}")
        try:
            report, _ = scale_infoloss.run_scale(
                specs, labels, backend=backend,
                embed_model=self.sem_model_var.get().strip() or "nomic-embed-text",
                out=out, log=self._log)
            if self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(report)
                    else:
                        import webbrowser; webbrowser.open("file://" + os.path.abspath(report))
                except Exception: pass
            self._log("Scale info-loss done.")
        except Exception as e:
            self._log("SCALE ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.scale_btn.configure(state="normal")

    def run_semantic(self):
        if not self.rows:
            messagebox.showwarning("No runs", "Add at least one run folder first."); return
        import semantic_analysis
        specs = [r["path"] for r in self.rows]
        labels = [r["label"].get().strip() or eval_runs.spec_name(r["path"]) for r in self.rows]
        out = self.out_var.get().strip() or None
        out = os.path.join(out, "semantic") if out else None
        self._save_settings()
        self.sem_btn.configure(state="disabled")
        backend = self.sem_backend_var.get().strip() or "tfidf"
        self._log(f"Semantic: {len(specs)} run(s), backend {backend}")
        try:
            res = semantic_analysis.run_semantic(specs, labels=labels, backend=backend,
                                                 embed_model=self.sem_model_var.get().strip() or "nomic-embed-text",
                                                 out=out)
            self._log(f"Report: {res['report']}")
            for lab, r in res["results"].items():
                self._log(f"  [{lab}] tweets={r.get('n_tweets')} sim={r.get('mean_pairwise_sim', 0):.3f} "
                          f"conv={r.get('text_convergence', 0):+.3f} sep={r.get('stance_separation', 0):+.3f}")
            if self.open_after_var.get():
                try:
                    if hasattr(os, "startfile"): os.startfile(res["report"])
                    else:
                        import webbrowser; webbrowser.open("file://" + os.path.abspath(res["report"]))
                except Exception: pass
            self._log("Semantic done.")
        except Exception as e:
            self._log("SEMANTIC ERROR: " + str(e))
            self._log(traceback.format_exc(limit=3))
        finally:
            self.sem_btn.configure(state="normal")


if __name__ == "__main__":
    EvalLauncher().mainloop()
