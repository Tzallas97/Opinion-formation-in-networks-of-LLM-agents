#!/usr/bin/env python3
"""Master report: aggregate a master CSV (accumulated by eval_runs --append-csv)
into a per-condition summary - the "auto-report per batch" tool.

Groups rows by `condition`, reports mean +/- std (and n) for the key outcome and
intervention columns, plus a per-condition figure. This is where multi-seed
replication becomes readable: one row per condition instead of one per run.

Usage:  python tools/master_report.py MASTER.csv [--out DIR]
Outputs: OUT/master_report.md, OUT/figures/master_outcomes.png
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

NUMERIC = ["B_final", "D_final", "P_final", "delta_B", "extremity", "frac_extreme",
           "entropy", "camp_share", "convergence_step", "move_rate", "up_rate", "down_rate",
           "hold_rate", "repair_rate", "cleanup_rate", "fallback_rate", "flips", "oscillations"]


def mean_std(xs):
    n = len(xs)
    if not n: return 0.0, 0.0, 0
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n) if n > 1 else 0.0
    return mu, sd, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master_csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.master_csv, encoding="utf-8")))
    if not rows:
        sys.exit("master CSV is empty")
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.master_csv)) or ".", "master_report")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    groups = defaultdict(list)
    for r in rows:
        groups[(r.get("condition") or "?").strip()].append(r)

    stats = {}  # condition -> col -> (mu, sd, n)
    for cond, rs in groups.items():
        stats[cond] = {}
        for col in NUMERIC:
            xs = []
            for r in rs:
                try: xs.append(float(r.get(col, "") or 0))
                except Exception: pass
            stats[cond][col] = mean_std(xs)

    # --- figure: key outcomes as grouped bars with std error bars ---
    keys = ["B_final", "P_final", "entropy", "move_rate", "hold_rate"]
    conds = sorted(groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.6))
    for ax, k in zip(axes, keys):
        mus = [stats[c][k][0] for c in conds]
        sds = [stats[c][k][1] for c in conds]
        bars = ax.bar(range(len(conds)), mus, yerr=sds, capsize=4,
                      color=["#2f6bff", "#e8590c", "#2b8a3e", "#9c36b5", "#e03131"][:len(conds)],
                      edgecolor="white")
        for b, mu in zip(bars, mus):
            ax.annotate(f"{mu:.2f}", xy=(b.get_x() + b.get_width() / 2, mu), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(conds)), conds, rotation=20, ha="right")
        ax.set_title(k); ax.grid(axis="y"); ax.grid(axis="x", alpha=0)
    fig.suptitle(f"Per-condition means ± std (n runs per condition: " +
                 ", ".join(f"{c}={len(groups[c])}" for c in conds) + ")", y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "master_outcomes.png")); plt.close(fig)

    # --- markdown ---
    L = ["# Master report\n",
         f"Πηγή: `{os.path.basename(a.master_csv)}` — {len(rows)} runs σε {len(groups)} conditions.\n",
         "Μέσοι όροι ± τυπική απόκλιση **across runs** ανά condition (με πολλαπλά seeds ανά condition, "
         "αυτή η διασπορά είναι η πραγματική seed-to-seed αβεβαιότητα — ό,τι δεν έχουν τα within-run CIs).\n",
         "| condition | n | B_final | P_final | entropy | ΔB | move_rate | hold_rate | repair_rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        s = stats[c]
        def f(col, fmt="{:.2f}"):
            mu, sd, n = s[col]
            return (fmt.format(mu) + (f" ±{fmt.format(sd)}" if n > 1 else ""))
        L.append(f"| **{c}** | {len(groups[c])} | {f('B_final','{:+.2f}')} | {f('P_final')} | {f('entropy')} | "
                 f"{f('delta_B','{:+.2f}')} | {f('move_rate')} | {f('hold_rate')} | {f('repair_rate')} |")
    L += ["", "![outcomes](figures/master_outcomes.png)", "",
          "Runs ανά condition:", ""]
    for c in conds:
        for r in groups[c]:
            L.append(f"- `{c}` ← {r.get('run','?')} (seed {r.get('seed','?')}, {r.get('model','?')})")
    open(os.path.join(out, "master_report.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", os.path.join(out, "master_report.md"), f"({len(rows)} runs, {len(groups)} conditions)")


if __name__ == "__main__":
    main()
