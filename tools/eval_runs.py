#!/usr/bin/env python3
"""Eval harness (analysis layer): compare finished opinion-dynamics runs.

Usage:
  python tools/eval_runs.py RUN_DIR [RUN_DIR ...] [--out OUT_DIR] [--labels a,b]
  python tools/eval_runs.py PARENT_DIR --scan [--out OUT_DIR]
  python tools/eval_runs.py ... --append-csv MASTER.csv   # also append rows to a master CSV

A "run folder" is any folder containing *network_opinion_change*.csv.
From each run folder we read:
  *network_opinion_change*.csv   per-agent opinion series (the backbone)
  *network_interactions*.csv     per-event data: moves, repairs, cleanups, fallbacks
  metrics_*.json                 optional enrichment: holds, closed-world leaks, config

Outputs (in OUT_DIR):
  eval_report.md    human-readable comparison (thesis-ready tables + figures)
  aggregate.csv     one row per run (machine-readable)
  figures/*.png     trajectories / final distribution / interventions

GUI wrapper: tools/eval_launcher.py (folder picking, labels, append-to-master).
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, random, sys
from collections import Counter

# scripts/ holds the shared metric + naming modules used across the repo
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import opinion_metrics  # shared B/D/P definitions
import run_naming      # shared stem/sibling resolution + tolerant CSV open

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Thesis-ready figure style (applies to every figure this module makes)
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.prop_cycle": matplotlib.cycler(color=["#2f6bff", "#e8590c", "#2b8a3e", "#9c36b5", "#e03131", "#0b7285"]),
})

# ----------------------------------------------------------------------------- load

def _one(run_dir, pat):
    hits = glob.glob(os.path.join(run_dir, pat))
    return hits[0] if hits else None

def _pick(run_dir, *pats):
    """Return the first file matching any pattern, in priority order.
    Lets us support both naming schemes: *network_opinion_change*.csv (new)
    and <run>_opinion_change.csv (older/thesis runs)."""
    for pat in pats:
        f = _one(run_dir, pat)
        if f: return f
    return None

def _flag(v):
    s = str(v or "").strip().lower()
    return 0 if s in ("", "0", "0.0", "nan", "false", "none") else 1

def is_run_dir(d):
    return bool(_pick(d, "*network_opinion_change*.csv", "*opinion_change*.csv"))

def spec_name(spec):
    """Human name of a run spec (folder path OR a loose opinion_change CSV path)."""
    if os.path.isfile(spec):
        base = os.path.basename(spec)
        cut = base.lower().find("opinion_change")
        name = base[:cut].rstrip("_-") if cut > 0 else os.path.splitext(base)[0]
        return name or os.path.splitext(base)[0]
    return os.path.basename(os.path.normpath(spec))

def find_run_specs(root):
    """Collect run *specs* under root. A spec is either
      - a run FOLDER (contains exactly one opinion_change CSV), or
      - a loose opinion_change CSV FILE (folders where several runs' CSVs were
        dumped together, e.g. thesis/: each CSV counts as its own run).
    Mixed folders (loose CSVs + run subfolders) yield both kinds."""
    if os.path.isfile(root):
        return [root]
    specs = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        ocs = sorted(os.path.join(base, f) for f in files
                     if 'opinion_change' in f.lower() and f.lower().endswith('.csv'))
        sub_has_runs = any(is_run_dir(os.path.join(base, d)) for d in dirs)
        if len(ocs) == 1 and not sub_has_runs:
            specs.append(base)      # a normal single-run folder
            dirs[:] = []            # don't descend into its internals
        elif ocs:
            specs.extend(ocs)       # several runs share this folder -> one spec per CSV
    return sorted(set(specs))

def find_run_dirs(root):
    """Recursively collect run folders under root.

    The root itself is included as a candidate when it holds loose CSVs, but we
    do NOT stop there (a folder like thesis/ can contain loose top-level CSVs
    AND dozens of proper run folders - both are returned; the GUI checklist
    lets the user untick the ambiguous root)."""
    out = []
    if is_run_dir(root):
        out.append(root)
    for base, dirs, _files in os.walk(root):
        # skip hidden folders; do not descend into a run folder's own subfolders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for d in list(dirs):
            full = os.path.join(base, d)
            if is_run_dir(full):
                out.append(full)
                dirs.remove(d)
    return sorted(set(out))

def bdp(vals):
    '''B = mean belief, D = population std (ddof=0), P = 4*pos*neg bipolarization.
    Thin wrapper over scripts/opinion_metrics.py, which is the single definition
    shared by the simulator, the plots and the eval tools.'''
    return opinion_metrics.bdp(vals)

def load_run(spec):
    """Load one run from a spec: a run folder, or a loose opinion_change CSV
    (then sibling files sharing the same filename prefix are used)."""
    is_file = os.path.isfile(spec)
    run_dir = (os.path.dirname(spec) or ".") if is_file else spec
    prefix = None
    if is_file:
        b = os.path.basename(spec)
        cut = b.lower().find("opinion_change")
        prefix = b[:cut].rstrip("_-") if cut > 0 else None
    def pickf(*pats):
        # Old runs put the date on either side of the network infix depending on
        # the artifact, so a literal prefix can miss a sibling that is right
        # there. run_naming.sibling_patterns tries the literal prefix first, then
        # the core stem.
        if prefix:
            for pat in run_naming.sibling_patterns(os.path.basename(spec), *pats):
                f = _one(run_dir, pat)
                if f: return f
            return None
        return _pick(run_dir, *pats)
    r = {"dir": run_dir, "name": spec_name(spec), "spec": spec}
    # --- opinion series ---
    oc = spec if is_file else pickf("*network_opinion_change*.csv", "*opinion_change*.csv")
    rows = list(csv.reader(run_naming.open_text(oc)))
    names = rows[0][1:]
    series = {n: [] for n in names}
    steps = []
    for row in rows[1:]:
        steps.append(row[0])
        for i, n in enumerate(names):
            try: series[n].append(int(row[i + 1]))
            except Exception: series[n].append(0)
    n_ag = len(names) or 1
    r["agents"] = names; r["steps"] = steps; r["series"] = series
    r["traj"] = [bdp([series[n][i] for n in names]) for i in range(len(steps))]
    r["final_vals"] = [series[n][-1] for n in names]
    r["init_vals"] = [series[n][0] for n in names]
    # convergence: first step index after which B stays within eps of final B
    Bf = r["traj"][-1][0]; eps = 0.1; conv = len(steps) - 1
    for i in range(len(steps)):
        if all(abs(r["traj"][j][0] - Bf) <= eps for j in range(i, len(steps))):
            conv = i; break
    r["convergence_step"] = conv
    # population-movement metrics
    chg = 0; osc = 0
    for n in names:
        s = series[n]
        deltas = [1 if s[i] > s[i - 1] else -1 for i in range(1, len(s)) if s[i] != s[i - 1]]
        chg += len(deltas)
        osc += sum(1 for i in range(1, len(deltas)) if deltas[i] != deltas[i - 1])
    r["total_changes"] = chg
    r["oscillations"] = osc                       # direction reversals (change up then down, etc.)
    drift = [f - i0 for f, i0 in zip(r["final_vals"], r["init_vals"])]
    r["mean_drift"] = sum(drift) / n_ag           # signed net movement of the population
    r["mean_abs_drift"] = sum(abs(x) for x in drift) / n_ag
    r["changed_agents"] = sum(1 for x in drift if x != 0)
    r["flips"] = sum(1 for f, i0 in zip(r["final_vals"], r["init_vals"]) if f * i0 < 0)  # crossed sides
    r["extremity"] = sum(abs(x) for x in r["final_vals"]) / n_ag
    r["frac_extreme"] = sum(1 for x in r["final_vals"] if abs(x) == 2) / n_ag
    cnt = Counter(r["final_vals"])
    r["camp_share"] = max(cnt.values()) / n_ag    # size of the largest opinion camp
    ent = -sum((c / n_ag) * math.log(c / n_ag, 2) for c in cnt.values())
    r["entropy"] = ent / math.log(5, 2)           # 0 = consensus, 1 = uniform over 5 bins
    # --- interactions ---
    inter = {"listener_events": 0, "moves": 0, "moved_up": 0, "moved_down": 0,
             "hard_repairs": 0, "soft_cleanups": 0, "step2_fallbacks": 0,
             "empty_tweets": 0, "quality_bad": 0}
    ic = pickf("*network_interactions*.csv", "*interactions*.csv")
    meta = {}
    if ic:
        for row in csv.DictReader(run_naming.open_text(ic)):
            if not meta:
                meta = {"world": (row.get("World") or "").strip(),
                        "rag": (row.get("RAG Content Mode") or "").strip(),
                        "model": (row.get("Model Name Step2") or "").strip(),
                        "version": (row.get("Version Set") or "").strip(),
                        "seed": (row.get("Seed") or "").strip()}
            inter["listener_events"] += 1
            try: d = int(row.get("Agent_I Delta-Belief") or 0)
            except Exception: d = 0
            if d > 0: inter["moved_up"] += 1
            if d < 0: inter["moved_down"] += 1
            if d != 0: inter["moves"] += 1
            inter["hard_repairs"] += _flag(row.get("Agent_I Hard Repair"))
            inter["soft_cleanups"] += _flag(row.get("Agent_I Soft Cleanup"))
            inter["step2_fallbacks"] += _flag(row.get("Agent_J Step2 Fallback Used"))
            if not str(row.get("Agent_J Tweet") or "").strip(): inter["empty_tweets"] += 1
            q = (row.get("Agent_I Interaction Quality") or "ok").strip().lower()
            if q not in ("", "ok"): inter["quality_bad"] += 1
    r["inter"] = inter
    r["meta"] = meta
    # --- metrics json (optional enrichment) ---
    m = {}
    mj = _one(run_dir, (prefix + "*metrics*.json") if prefix else "metrics_*.json")
    if mj:
        try:
            data = json.load(run_naming.open_text(mj))
            m = dict(data.get("metrics", {}) or {})
            m.update({("summary::" + k): v for k, v in (data.get("summary", {}) or {}).items()})
            cfg = data.get("config", {}) or {}
            for k, v in cfg.items():
                if v and k not in r["meta"]: r["meta"][k] = v
        except Exception:
            pass
    else:
        # older/thesis runs store the same counters as <run>_run_metrics.csv (Metric,Count)
        rm = pickf("*run_metrics*.csv")
        if rm:
            try:
                for row in csv.DictReader(run_naming.open_text(rm)):
                    try: m[str(row.get("Metric", "")).strip()] = float(row.get("Count") or 0)
                    except Exception: pass
            except Exception:
                pass
    r["metrics"] = m
    # belief-transition matrix from the run's own counters (pre->post per listener event)
    trans = {}
    for k, v in m.items():
        # ONLY the allowed_set grouping: each listener event appears exactly once there.
        # (listener_prev/speaker_stance/step_bucket groupings also log transitions -
        # summing across all four would count every event 4x.)
        if str(k).startswith("allowed_set::") and "::transition::" in str(k):
            pair = str(k).split("::transition::")[-1]
            if "->" in pair:
                a, b2 = pair.split("->", 1)
                try:
                    trans[(int(a), int(b2))] = trans.get((int(a), int(b2)), 0) + int(float(v))
                except Exception:
                    pass
    r["transitions"] = trans
    r["holds_total"] = int(m.get("holds_total", 0) or 0)
    r["leak2"] = int(m.get("closed_world_leak_step2", m.get("closed_world_leak_step2_total", 0)) or 0)
    r["leak3"] = int(m.get("closed_world_leak_step3", m.get("closed_world_leak_step3_total", 0)) or 0)
    return r

# ----------------------------------------------------------------------------- stats

def bootstrap_bdp(final_vals, n_boot=2000, seed=7):
    """Agent-level bootstrap CIs for final B/D/P (resample agents with replacement)."""
    rng = random.Random(seed)
    n = len(final_vals)
    stats = {"B": [], "D": [], "P": []}
    for _ in range(n_boot):
        sample = [final_vals[rng.randrange(n)] for _ in range(n)]
        B, D, P = bdp(sample)
        stats["B"].append(B); stats["D"].append(D); stats["P"].append(P)
    out = {}
    for k, v in stats.items():
        v.sort()
        out[k] = (v[int(0.025 * n_boot)], v[int(0.975 * n_boot)])
    return out

def infer_labels(runs):
    """Condition label = distinguishing part of folder names (common prefix/suffix stripped)."""
    names = [r["name"] for r in runs]
    if len(names) == 1: return {names[0]: names[0]}
    pre = os.path.commonprefix(names)
    suf = os.path.commonprefix([n[::-1] for n in names])[::-1]
    labels = {}
    for n in names:
        core = n[len(pre):len(n) - len(suf)] if len(n) > len(pre) + len(suf) else ""
        labels[n] = core.strip("_-") or "base"
    return labels

# ----------------------------------------------------------------------------- figures

def fig_trajectories(runs, labels, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for r in runs:
        lab = labels[r["name"]]
        xs = list(range(len(r["steps"])))
        for ax, idx, ttl in ((axes[0], 0, "Mean belief B"), (axes[1], 1, "Diversity D"), (axes[2], 2, "Polarization P")):
            ys = [t[idx] for t in r["traj"]]
            line, = ax.plot(xs, ys, label=lab, linewidth=2.2)
            # annotate the final value at the curve end (readable comparison at a glance)
            ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=9, color=line.get_color(),
                        fontweight="bold", va="center")
            ax.set_title(ttl); ax.set_xlabel("step")
            ax.margins(x=0.06)
    axes[0].set_ylim(-2.15, 2.15); axes[0].axhline(0, color="#888", lw=0.8, ls="--")
    axes[0].set_yticks([-2, -1, 0, 1, 2]); axes[2].set_ylim(-0.05, 1.05)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_distribution(runs, labels, path):
    vals = [-2, -1, 0, 1, 2]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(runs)
    for i, r in enumerate(runs):
        cnt = Counter(r["final_vals"])
        xs = [v + (i - (len(runs) - 1) / 2) * w for v in vals]
        ys = [cnt.get(v, 0) for v in vals]
        bars = ax.bar(xs, ys, width=w * 0.94, label=labels[r["name"]], edgecolor="white", linewidth=0.5)
        for b, y in zip(bars, ys):
            if y:
                ax.annotate(str(y), xy=(b.get_x() + b.get_width() / 2, y), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_xticks(vals, ["-2\nstrongly\nagainst", "-1", "0\nmixed", "+1", "+2\nstrongly\nfor"])
    ax.set_ylabel("agents"); ax.set_title("Final opinion distribution")
    ax.grid(axis="y"); ax.grid(axis="x", alpha=0)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_interventions(runs, labels, path):
    keys = ["moves", "hard repairs", "soft cleanups", "step2 fallbacks", "holds"]
    fig, ax = plt.subplots(figsize=(8.5, 0.75 * len(keys) + 1.6))
    h = 0.8 / len(runs)
    for i, r in enumerate(runs):
        ev = max(1, r["inter"]["listener_events"])
        vals = [100.0 * r["inter"]["moves"] / ev, 100.0 * r["inter"]["hard_repairs"] / ev,
                100.0 * r["inter"]["soft_cleanups"] / ev, 100.0 * r["inter"]["step2_fallbacks"] / ev,
                100.0 * r["holds_total"] / ev]
        ys = [j + (i - (len(runs) - 1) / 2) * h for j in range(len(keys))]
        bars = ax.barh(ys, vals, height=h * 0.94, label=labels[r["name"]], edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", xy=(v, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
    ax.set_yticks(range(len(keys)), keys); ax.invert_yaxis()
    ax.set_xlabel("per 100 listener events"); ax.set_title("Dynamics & pipeline interventions")
    ax.grid(axis="x"); ax.grid(axis="y", alpha=0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_transitions(runs, labels, path):
    """Pre->post belief transition heatmap per run (from the run's own counters).
    Rows = pre rating, columns = post rating; colour = row-normalised share, label = raw count.
    Returns False when no run carries transition counters (old metrics)."""
    have = [r for r in runs if r.get("transitions")]
    if not have:
        return False
    vals = [-2, -1, 0, 1, 2]
    fig, axes = plt.subplots(1, len(have), figsize=(4.4 * len(have), 4.1), squeeze=False)
    for ax, r in zip(axes[0], have):
        M = [[r["transitions"].get((a, b), 0) for b in vals] for a in vals]
        norm = [[(c / s if (s := sum(row)) else 0.0) for c in row] for row in M]
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i, row in enumerate(M):
            for j, c in enumerate(row):
                if c:
                    ax.text(j, i, str(c), ha="center", va="center", fontsize=9,
                            color=("white" if norm[i][j] > 0.55 else "#1a1a2e"))
        ax.set_xticks(range(5), [f"{v:+d}" if v else "0" for v in vals])
        ax.set_yticks(range(5), [f"{v:+d}" if v else "0" for v in vals])
        ax.set_xlabel("post rating"); ax.set_ylabel("pre rating")
        ax.set_title(labels[r["name"]])
        ax.grid(False)
        for i in range(5):
            ax.axhline(i - 0.5, color="white", lw=1); ax.axvline(i - 0.5, color="white", lw=1)
    fig.suptitle("Belief transitions (listener events; colour = row share, number = count)", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return True


# ----------------------------------------------------------------------------- aggregate rows

AGG_COLS = ["condition", "run", "model", "version", "world", "rag", "seed", "steps", "agents",
            "B_init", "D_init", "P_init", "B_final", "D_final", "P_final", "delta_B",
            "extremity", "frac_extreme", "entropy", "camp_share",
            "convergence_step", "total_changes", "changed_agents", "flips", "oscillations",
            "mean_drift", "mean_abs_drift",
            "listener_events", "moves", "moved_up", "moved_down",
            "move_rate", "up_rate", "down_rate",
            "holds_total", "hold_rate", "hard_repairs", "repair_rate",
            "soft_cleanups", "cleanup_rate", "step2_fallbacks", "fallback_rate",
            "empty_tweets", "leak_step2", "leak_step3"]

def aggregate_rows(runs, labels):
    out = []
    for r in runs:
        B0, D0, P0 = r["traj"][0]; B, D, P = r["traj"][-1]
        i = r["inter"]; m = r["meta"]; ev = max(1, i["listener_events"])
        rate = lambda x: f"{x / ev:.3f}"
        out.append([labels[r["name"]], r["name"], m.get("model", ""), m.get("version", ""),
                    m.get("world", ""), m.get("rag", ""), m.get("seed", ""), len(r["steps"]), len(r["agents"]),
                    f"{B0:.3f}", f"{D0:.3f}", f"{P0:.3f}", f"{B:.3f}", f"{D:.3f}", f"{P:.3f}", f"{B - B0:+.3f}",
                    f"{r['extremity']:.3f}", f"{r['frac_extreme']:.3f}", f"{r['entropy']:.3f}", f"{r['camp_share']:.3f}",
                    r["convergence_step"], r["total_changes"], r["changed_agents"], r["flips"], r["oscillations"],
                    f"{r['mean_drift']:+.3f}", f"{r['mean_abs_drift']:.3f}",
                    i["listener_events"], i["moves"], i["moved_up"], i["moved_down"],
                    rate(i["moves"]), rate(i["moved_up"]), rate(i["moved_down"]),
                    r["holds_total"], rate(r["holds_total"]), i["hard_repairs"], rate(i["hard_repairs"]),
                    i["soft_cleanups"], rate(i["soft_cleanups"]), i["step2_fallbacks"], rate(i["step2_fallbacks"]),
                    i["empty_tweets"], r["leak2"], r["leak3"]])
    return AGG_COLS, out

# ----------------------------------------------------------------------------- report

def fmt_ci(x, ci):
    return f"{x:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"

def write_report(runs, labels, out_dir):
    figs = os.path.join(out_dir, "figures"); os.makedirs(figs, exist_ok=True)
    fig_trajectories(runs, labels, os.path.join(figs, "trajectories.png"))
    fig_distribution(runs, labels, os.path.join(figs, "final_distribution.png"))
    fig_interventions(runs, labels, os.path.join(figs, "interventions.png"))
    has_trans = fig_transitions(runs, labels, os.path.join(figs, "transitions.png"))

    L = []
    L.append("# Run comparison report\n")
    L.append("Generated by `tools/eval_runs.py` (analysis-only; no model calls).\n")
    L.append("## Runs\n")
    L.append("| condition | model | version | world | RAG | seed | steps | agents |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in runs:
        m = r["meta"]
        L.append(f"| **{labels[r['name']]}** | {m.get('model','?')} | {m.get('version','?')} | {m.get('world','?')} | {m.get('rag','?')} | {m.get('seed','?')} | {len(r['steps'])} | {len(r['agents'])} |")
    L.append("")
    if len({(len(r["agents"]), len(r["steps"])) for r in runs}) > 1:
        L.append("> **Note:** the compared runs differ in agents and/or steps - absolute counts are not directly comparable; prefer the rate columns.\n")
    L.append("## Outcomes (final step; 95% agent-bootstrap CIs)\n")
    L.append("| condition | B (mean belief) | D (diversity) | P (bipolarization, 4*pos*neg) | convergence step | opinion changes |")
    L.append("|---|---|---|---|---|---|")
    cis = {r["name"]: bootstrap_bdp(r["final_vals"]) for r in runs}   # computed once per run
    for r in runs:
        B, D, P = r["traj"][-1]
        ci = cis[r["name"]]
        L.append(f"| **{labels[r['name']]}** | {fmt_ci(B, ci['B'])} | {D:.2f} [{ci['D'][0]:.2f}, {ci['D'][1]:.2f}] | {P:.2f} [{ci['P'][0]:.2f}, {ci['P'][1]:.2f}] | {r['convergence_step']} / {len(r['steps'])-1} | {r['total_changes']} |")
    L.append("")
    L.append("## Population detail (final step)\n")
    L.append("| condition | ΔB init→final | extremity ⌀\\|o\\| | % at ±2 | entropy | largest camp | changed agents | side flips | oscillations | mean drift |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        B0 = r["traj"][0][0]; B = r["traj"][-1][0]; n_ag = len(r["agents"])
        L.append(f"| **{labels[r['name']]}** | {B-B0:+.2f} | {r['extremity']:.2f} | {100*r['frac_extreme']:.0f}% | {r['entropy']:.2f} | {100*r['camp_share']:.0f}% | {r['changed_agents']}/{n_ag} | {r['flips']} | {r['oscillations']} | {r['mean_drift']:+.2f} |")
    L.append("")
    L.append("![trajectories](figures/trajectories.png)\n")
    L.append("![final distribution](figures/final_distribution.png)\n")
    L.append("## Dynamics & pipeline interventions (per 100 listener events)\n")
    L.append("| condition | listener events | moves | ▲ up | ▼ down | holds | hard repairs | soft cleanups | step2 fallbacks | empty tweets | closed-world leaks (s2/s3) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        ev = max(1, r["inter"]["listener_events"]); i = r["inter"]
        pct = lambda x: f"{100.0*x/ev:.1f}"
        L.append(f"| **{labels[r['name']]}** | {i['listener_events']} | {pct(i['moves'])} | {pct(i['moved_up'])} | {pct(i['moved_down'])} | {pct(r['holds_total'])} | {pct(i['hard_repairs'])} | {pct(i['soft_cleanups'])} | {pct(i['step2_fallbacks'])} | {pct(i['empty_tweets'])} | {r['leak2']}/{r['leak3']} |")
    L.append("")
    L.append("![interventions](figures/interventions.png)\n")
    if has_trans:
        L.append("## Belief transitions\n")
        L.append("Pre-&gt;post rating ανά listener event (από τους counters του run). Η διαγώνιος = κρατήματα· πάνω/κάτω από τη διαγώνιο = κίνηση προς τα πάνω/κάτω.\n")
        L.append("![transitions](figures/transitions.png)\n")
    if 2 <= len(runs) <= 8:
        L.append("## Pairwise deltas (non-overlapping CIs => robust at agent level)\n")
        for a in range(len(runs)):
            for b in range(a + 1, len(runs)):
                ra, rb = runs[a], runs[b]
                la, lb = labels[ra["name"]], labels[rb["name"]]
                ca, cb = cis[ra["name"]], cis[rb["name"]]
                for k, idx in (("B", 0), ("D", 1), ("P", 2)):
                    va, vb = ra["traj"][-1][idx], rb["traj"][-1][idx]
                    sep = "non-overlapping" if (ca[k][1] < cb[k][0] or cb[k][1] < ca[k][0]) else "overlapping"
                    L.append(f"- **{k}**: {la} = {va:+.2f} vs {lb} = {vb:+.2f} (Delta = {va-vb:+.2f}; CIs {sep})")
        L.append("")
    elif len(runs) > 8:
        L.append("## Pairwise deltas\n")
        L.append(f"Skipped ({len(runs)} runs -> too many pairs); use aggregate.csv for cross-run analysis.\n")
    L.append("## Caveats\n")
    L.append("- One run per condition: CIs are **within-run** (agent-level bootstrap), they quantify population uncertainty, **not** seed-to-seed variance. Add replicate seeds per condition before drawing thesis-level conclusions; this report auto-extends when more runs are passed in.")
    L.append("- Conditions are inferred from folder-name differences; override labels in the GUI or with `--labels`.")
    L.append("")
    L.append("## Reproduce\n")
    L.append("```")
    L.append("python tools/eval_runs.py " + " ".join('"' + str(r.get("spec", r["dir"])) + '"' for r in runs))
    L.append("```")
    open(os.path.join(out_dir, "eval_report.md"), "w", encoding="utf-8").write("\n".join(L))

    header, rows = aggregate_rows(runs, labels)
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

# ----------------------------------------------------------------------------- public API

def append_to_master(master_csv, runs, labels, skip_duplicates=True):
    """Append aggregate rows to a master CSV (creates it if missing).
    Returns (appended, skipped). Refuses to append if an existing header differs."""
    header, rows = aggregate_rows(runs, labels)
    existing_runs = set()
    if os.path.exists(master_csv) and os.path.getsize(master_csv) > 0:
        with open(master_csv, encoding="utf-8") as f:
            rd = csv.reader(f)
            old_header = next(rd, None)
            if old_header != header:
                raise ValueError("master CSV has a different header; point to a new file or delete the old one")
            idx = header.index("run")
            for row in rd:
                if len(row) > idx: existing_runs.add(row[idx])
        mode = "a"
    else:
        mode = "w"
    appended = skipped = 0
    with open(master_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if mode == "w": w.writerow(header)
        for row in rows:
            if skip_duplicates and row[1] in existing_runs:
                skipped += 1; continue
            w.writerow(row); appended += 1
    return appended, skipped

def run_eval(run_dirs, labels=None, out=None, append_csv=None, skip_duplicates=True):
    """Programmatic entry point (used by the GUI). Returns a result dict.
    Runs that fail to load are skipped (reported in result["load_errors"])."""
    runs, load_errors, kept_labels = [], [], []
    for i, d in enumerate(run_dirs):
        try:
            runs.append(load_run(d))
            kept_labels.append(labels[i] if labels and i < len(labels) else "")
        except Exception as e:
            load_errors.append((str(d), f"{type(e).__name__}: {e}"))
    if not runs:
        raise RuntimeError("no runs could be loaded: " + "; ".join(f"{d} ({e})" for d, e in load_errors))
    labels = kept_labels
    lab = infer_labels(runs)
    if labels:
        for r, l in zip(runs, labels):
            if l: lab[r["name"]] = l
    out = out or os.path.join(runs[0]["dir"] if os.path.isfile(str(run_dirs[0])) else (os.path.dirname(str(run_dirs[0])) or "."), "eval_report")
    os.makedirs(out, exist_ok=True)
    write_report(runs, lab, out)
    res = {"out": out, "report": os.path.join(out, "eval_report.md"),
           "aggregate": os.path.join(out, "aggregate.csv"),
           "labels": [lab[r["name"]] for r in runs], "appended": 0, "skipped": 0,
           "load_errors": load_errors}
    if append_csv:
        res["appended"], res["skipped"] = append_to_master(append_csv, runs, lab, skip_duplicates)
    return res

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--scan", action="store_true", help="treat paths as parent dirs; analyze all run folders inside")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="comma-separated condition labels (same order as runs)")
    ap.add_argument("--append-csv", default=None, help="also append rows to this master CSV")
    a = ap.parse_args()

    run_dirs = []
    for p in a.paths:
        p = p.rstrip("/\\")
        run_dirs += [p] if (os.path.isfile(p) or (is_run_dir(p) and not a.scan)) else find_run_specs(p)
    if not run_dirs:
        sys.exit("no run folders found (need *network_opinion_change*.csv inside)")

    labels = [x.strip() for x in a.labels.split(",")] if a.labels else None
    res = run_eval(run_dirs, labels=labels, out=a.out, append_csv=a.append_csv)
    print("runs:", len(run_dirs), "->", ", ".join(res["labels"]))
    print("wrote", res["report"])
    if a.append_csv:
        print(f"master CSV: +{res['appended']} appended, {res['skipped']} skipped (already present)")

if __name__ == "__main__":
    main()
