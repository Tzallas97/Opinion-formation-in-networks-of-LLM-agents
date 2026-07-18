#!/usr/bin/env python3
"""Inference emergence - do agents articulate conclusions
that are written NOWHERE in the corpus?

Each multihop chain has an implied_conclusion in the groundtruth file that no
snippet states. This tool scores every tweet of a finished run against every
implied conclusion (similarity) and reports, per chain:
  max_sim         how close did ANY tweet ever get (+ the tweet itself, quoted)
  emergence_rate  fraction of tweets above the null-calibrated threshold
  first_step      earliest step crossing the threshold (emergence time)

Null calibration: the groundtruth's distractor null_probes are plausible-
sounding inferences the corpus does NOT support ("the baker was involved").
Threshold tau = mean + 2*std of all tweet-vs-null-probe similarities, per run
and backend - anything a tweet scores against the baker, it must beat.

Backends (as in semantic_analysis): tfidf (lexical, runs NOW, underestimates
paraphrase) / ollama (real embeddings, cached - use when Ollama is available).

Analysis-only: never touches run data. Runs that never saw the multihop corpus
(e.g. the old v119 runs) are a NEGATIVE CONTROL - they should sit at/below tau.

Usage:
  python tools/inference_emergence.py RUN [RUN ...] --groundtruth GT.json
         [--labels a,b] [--backend tfidf|ollama] [--embed-model nomic-embed-text]
         [--out DIR]
Outputs: inference_report.md, inference.csv, figures/inference_emergence.png
"""
from __future__ import annotations
import argparse, csv, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_runs
import semantic_analysis as sa


def load_groundtruth(path):
    gt = json.load(open(path, encoding="utf-8"))
    chains = [(c["chain_id"], c["direction"], c["implied_conclusion"])
              for c in gt.get("chains", []) if (c.get("implied_conclusion") or "").strip()]
    nulls = [(d["chain_id"], d["null_probe"]) for d in gt.get("distractors", [])
             if (d.get("null_probe") or "").strip()]
    return chains, nulls


def run_inference(specs, labels, gt_path, backend="tfidf", embed_model="nomic-embed-text",
                  out_dir=None, url=None, log=print):
    chains, nulls = load_groundtruth(gt_path)
    if not chains:
        raise SystemExit("groundtruth has no implied conclusions")
    if not nulls:
        log("WARNING: no null_probes in groundtruth - falling back to fixed tau=0.40")

    runs = []
    for spec, label in zip(specs, labels):
        _, items = sa.tweets_from_run(spec)
        runs.append({"label": label, "items": items})

    concl_texts = [c[2] for c in chains]
    null_texts = [n[1] for n in nulls]
    all_texts = [it["text"] for r in runs for it in r["items"]] + concl_texts + null_texts

    if backend == "ollama":
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"embed_cache_{embed_model.replace(':', '_')}.json")
        vecs = sa.ollama_embed(all_texts, embed_model, cache, url=url)
        vec_of = dict(zip(all_texts, vecs))
        sim = lambda a, b: sa.dense_cos(vec_of[a], vec_of[b])
    else:
        vec = sa.build_tfidf(all_texts)
        vcache = {t: vec(t) for t in set(all_texts)}
        sim = lambda a, b: sa.cosine(vcache[a], vcache[b])

    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(specs[0])) or ".",
                                      "inference_report")
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    rows_csv, results = [], []
    for r in runs:
        items = r["items"]
        if not items:
            results.append({"label": r["label"], "empty": True})
            continue
        null_sims = [sim(it["text"], nt) for it in items for nt in null_texts]
        if null_sims:
            mu = sum(null_sims) / len(null_sims)
            sd = (sum((x - mu) ** 2 for x in null_sims) / len(null_sims)) ** 0.5
            tau = mu + 2 * sd
        else:
            mu = sd = 0.0
            tau = 0.40
        per_chain = {}
        for cid, cdir, ctext in chains:
            sims = [sim(it["text"], ctext) for it in items]
            mx = max(sims)
            arg = items[sims.index(mx)]
            above = [(it, s) for it, s in zip(items, sims) if s > tau]
            first = min((int(it["step"]) for it, _ in above if str(it["step"]).isdigit()),
                        default=None)
            per_chain[cid] = {"direction": cdir, "max": mx, "top_tweet": arg,
                              "rate": len(above) / len(items), "first_step": first,
                              "sims": sims}
            rows_csv.append({"run": r["label"], "chain": cid, "direction": cdir,
                             "max_sim": round(mx, 4), "emergence_rate": round(len(above) / len(items), 4),
                             "first_step": first if first is not None else "",
                             "tau": round(tau, 4), "backend": backend})
        results.append({"label": r["label"], "tau": tau, "null_mu": mu, "null_sd": sd,
                        "per_chain": per_chain, "items": items})

    with open(os.path.join(out_dir, "inference.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "chain", "direction", "max_sim",
                                          "emergence_rate", "first_step", "tau", "backend"])
        w.writeheader()
        for row in rows_csv:
            w.writerow(row)

    # ---- figure: per run, per-step max similarity over chains vs tau ----
    live = [res for res in results if not res.get("empty")]
    if live:
        fig, axes = plt.subplots(1, len(live), figsize=(5.2 * len(live), 3.6), squeeze=False)
        palette = ["#2f6bff", "#e8590c", "#2b8a3e", "#9c36b5", "#e03131", "#0b7285"]
        for ax, res in zip(axes[0], live):
            items = res["items"]
            steps = sorted({int(it["step"]) for it in items if str(it["step"]).isdigit()})
            for ci, (cid, _, _) in enumerate(chains):
                sims = res["per_chain"][cid]["sims"]
                by_step = {}
                for it, s in zip(items, sims):
                    if str(it["step"]).isdigit():
                        st = int(it["step"])
                        by_step[st] = max(by_step.get(st, 0.0), s)
                ax.plot(steps, [by_step.get(st, 0.0) for st in steps],
                        color=palette[ci % len(palette)], lw=1.3, label=cid, alpha=0.85)
            ax.axhline(res["tau"], color="#555", ls="--", lw=1.1)
            ax.annotate(f"tau={res['tau']:.2f}", xy=(0.99, res["tau"]),
                        xycoords=("axes fraction", "data"), ha="right", va="bottom",
                        fontsize=8.5, color="#555")
            ax.set_title(res["label"])
            ax.set_xlabel("step")
            ax.set_ylabel("max sim to implied conclusion")
            ax.legend(fontsize=8, ncols=2, frameon=False)
        fig.suptitle(f"Inference emergence ({backend}) - πάνω από το tau = άρθρωση άγραφου συμπεράσματος",
                     y=1.03, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "inference_emergence.png"))
        plt.close(fig)

    # ---- report ----
    L = ["# Inference emergence report\n",
         f"Groundtruth: `{os.path.basename(gt_path)}` - {len(chains)} implied conclusions, "
         f"{len(nulls)} null probes. Backend: **{backend}**"
         + (" (lexical - υποεκτιμά παραφράσεις· ξανατρέξε με ollama για τα τελικά νούμερα)" if backend == "tfidf" else "") + ".\n",
         "Κατώφλι tau ανά run = mean + 2σ της ομοιότητας των tweets προς τα null probes "
         "(ψευτο-συμπεράσματα που το corpus ΔΕΝ στηρίζει). Emergence = tweets πάνω από το tau.\n",
         "| run | tau | chain | dir | max sim | emergence rate | first step |",
         "|---|---|---|---|---|---|---|"]
    for res in results:
        if res.get("empty"):
            L.append(f"| **{res['label']}** | - | (no tweets found) | | | | |")
            continue
        for cid, cdir, _ in chains:
            pc = res["per_chain"][cid]
            mark = " **↑**" if pc["max"] > res["tau"] else ""
            L.append(f"| **{res['label']}** | {res['tau']:.2f} | {cid} | {cdir[:4]} | "
                     f"{pc['max']:.2f}{mark} | {pc['rate']:.2%} | "
                     f"{pc['first_step'] if pc['first_step'] is not None else '-'} |")
    L += ["", "![emergence](figures/inference_emergence.png)", "",
          "## Πιο κοντινό tweet ανά αλυσίδα (ποιοτικός έλεγχος)", ""]
    for res in results:
        if res.get("empty"):
            continue
        L.append(f"**{res['label']}** (tau {res['tau']:.2f}, null μ={res['null_mu']:.2f} σ={res['null_sd']:.2f})")
        for cid, _, ctext in chains:
            pc = res["per_chain"][cid]
            t = pc["top_tweet"]
            L.append(f"- {cid} ({pc['max']:.2f}): step {t['step']}, {t['author']}: "
                     f"\"{t['text'][:150]}\"")
        L.append("")
    path = os.path.join(out_dir, "inference_report.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    log(f"wrote {path} ({len(rows_csv)} rows)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "ollama"])
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--url", default=None)
    a = ap.parse_args()
    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    if len(labels) != len(a.runs):
        labels = eval_runs.infer_labels(a.runs)
    run_inference(a.runs, labels, a.groundtruth, backend=a.backend,
                  embed_model=a.embed_model, out_dir=a.out, url=a.url)


if __name__ == "__main__":
    main()
