#!/usr/bin/env python3
"""Semantic (text-space) analysis of finished runs - analysis-only, no run data touched.

Answers questions the -2..+2 ratings cannot:
  homogenisation   are agents starting to SAY the same thing? (mean pairwise
                   similarity of tweets + near-duplicate rate)
  text convergence do tweets converge over time even when ratings stall?
                   (early-third vs late-third within-window similarity)
  stance separation do FOR and AGAINST tweets actually read differently?
                   (within-side vs cross-side similarity gap; a model whose
                   sides sound alike is not expressing stance in text)
  self-repetition  does an agent recycle its own wording? (mean similarity of
                   consecutive tweets by the same agent)

Backends:
  tfidf  (default) pure-python TF-IDF + cosine. Deterministic, dependency-free,
         runs anywhere NOW. Lexical - underestimates paraphrase similarity.
  ollama nomic-embed-text (or any embedding model) via the Ollama API, cached
         on disk like the judge. True semantic similarity - use when compute
         is available:  --backend ollama [--embed-model nomic-embed-text]

Usage:
  python tools/semantic_analysis.py RUN_DIR [RUN_DIR ...] [--labels a,b]
         [--backend tfidf|ollama] [--out DIR]
Outputs: semantic_report.md, semantic.csv, figures/semantic_similarity.png
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, re, sys, urllib.request
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_runs  # spec discovery + run loading

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")

# ----------------------------------------------------------------------------- vectors

def _tokens(text):
    return [w.lower() for w in _WORD.findall(text or "")]

def build_tfidf(all_texts):
    """Fit IDF over the union of all compared runs so similarities share one space."""
    df = Counter()
    toks = [ _tokens(t) for t in all_texts ]
    for tk in toks:
        df.update(set(tk))
    n = max(1, len(all_texts))
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
    def vec(text):
        tf = Counter(_tokens(text))
        total = sum(tf.values()) or 1
        return {w: (c / total) * idf.get(w, 0.0) for w, c in tf.items()}
    return vec

def cosine(a, b):
    if not a or not b: return 0.0
    if len(b) < len(a): a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0

def ollama_embed(texts, model, cache_path, url=None):
    """Embed texts via Ollama /api/embeddings with an on-disk cache (list of floats each)."""
    cache = {}
    if os.path.exists(cache_path):
        try: cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception: cache = {}
    out, new = [], 0
    for t in texts:
        key = hashlib.sha256((model + "||" + t).encode("utf-8")).hexdigest()
        if key not in cache:
            body = json.dumps({"model": model, "prompt": t}).encode("utf-8")
            req = urllib.request.Request((url or OLLAMA_URL) + "/api/embeddings",
                                         data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                cache[key] = json.loads(resp.read().decode("utf-8")).get("embedding", [])
            new += 1
            if new % 50 == 0: json.dump(cache, open(cache_path, "w"))
        out.append(cache[key])
    json.dump(cache, open(cache_path, "w"))
    return out

def dense_cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

# ----------------------------------------------------------------------------- metrics

def tweets_from_run(spec):
    """Pull (step, author, author_stance_at_step, text) from the interactions CSV."""
    run = eval_runs.load_run(spec)
    is_file = os.path.isfile(spec)
    run_dir = (os.path.dirname(spec) or ".") if is_file else spec
    prefix = None
    if is_file:
        b = os.path.basename(spec); cut = b.lower().find("opinion_change")
        prefix = b[:cut].rstrip("_-") if cut > 0 else None
    ic = None
    for pat in ("*network_interactions*.csv", "*interactions*.csv"):
        hit = eval_runs._one(run_dir, (prefix + "*" + pat.lstrip("*")) if prefix else pat)
        if hit: ic = hit; break
    items = []
    if not ic: return run, items
    seen = set()
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"^FINAL_RATING:\s*-?\d+\s*(?:\n|\\n|\s)*(?:TWEET:|EXPLANATION:)\s*", "", str(t or ""), flags=re.I)).strip()
    for row in csv.DictReader(open(ic, encoding="utf-8")):
        step = str(row.get("Time Step") or "").strip()
        author = (row.get("Agent_J Name") or "").strip()
        text = clean(row.get("Agent_J Tweet"))
        try: stance = int(row.get("Agent_J Belief") or 0)
        except Exception: stance = 0
        key = (step, author, text)
        if text and key not in seen:
            seen.add(key)
            items.append({"step": step, "author": author, "stance": stance, "text": text})
    return run, items

def analyze(run, items, vec=None, dense=None):
    """Compute the four metric families for one run."""
    n = len(items)
    res = {"n_tweets": n}
    if n < 4:
        return res
    if dense is not None:
        sim = lambda i, j: dense_cos(dense[i], dense[j])
    else:
        vs = [vec(it["text"]) for it in items]
        sim = lambda i, j: cosine(vs[i], vs[j])
    # 1) homogenisation: mean pairwise + near-dup rate. All O(n^2) pairs are
    #    computed - no cap; fine at current run sizes (<= ~1.5k tweets).
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sims = [sim(i, j) for i, j in pairs]
    res["mean_pairwise_sim"] = sum(sims) / len(sims)
    res["near_dup_rate"] = sum(1 for s in sims if s > 0.9) / len(sims)
    # 2) text convergence: within-window similarity, early third vs late third
    third = max(2, n // 3)
    def _win(seg):
        p = [(i, j) for i in seg for j in seg if i < j]
        return sum(sim(i, j) for i, j in p) / len(p) if p else 0.0
    early = _win(range(0, third)); late = _win(range(n - third, n))
    res["early_sim"] = early; res["late_sim"] = late
    res["text_convergence"] = late - early          # >0: texts converge over time
    # 3) stance separation: within-side vs cross-side similarity
    pos = [i for i, it in enumerate(items) if it["stance"] > 0]
    neg = [i for i, it in enumerate(items) if it["stance"] < 0]
    def _avg(pp):
        return sum(sim(i, j) for i, j in pp) / len(pp) if pp else 0.0
    within = [(i, j) for grp in (pos, neg) for i in grp for j in grp if i < j]
    cross = [(i, j) for i in pos for j in neg]
    res["within_side_sim"] = _avg(within); res["cross_side_sim"] = _avg(cross)
    res["stance_separation"] = res["within_side_sim"] - res["cross_side_sim"]  # >0: sides read differently
    # 4) self-repetition: consecutive tweets by the same author
    by_author = {}
    for idx, it in enumerate(items):
        by_author.setdefault(it["author"], []).append(idx)
    consec = []
    for idxs in by_author.values():
        consec += [sim(a, b) for a, b in zip(idxs, idxs[1:])]
    res["self_repetition"] = (sum(consec) / len(consec)) if consec else 0.0
    # timeline for the figure: rolling window similarity
    win = max(4, n // 10); timeline = []
    for start in range(0, n - win + 1, max(1, win // 2)):
        seg = range(start, start + win)
        timeline.append((start, _win(seg)))
    res["_timeline"] = timeline
    return res

# ----------------------------------------------------------------------------- report

def write_outputs(results, labels, out_dir, backend):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    # figure: rolling within-window similarity per run
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for lab, res in results.items():
        tl = res.get("_timeline") or []
        if tl: ax.plot([x for x, _ in tl], [y for _, y in tl], label=lab, linewidth=1.8)
    ax.set_xlabel("tweet # (chronological)"); ax.set_ylabel("similarity (rolling window)")
    ax.set_title(f"Text similarity over time ({backend})"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "semantic_similarity.png"), dpi=150)
    plt.close(fig)

    cols = ["condition", "backend", "n_tweets", "mean_pairwise_sim", "near_dup_rate",
            "early_sim", "late_sim", "text_convergence",
            "within_side_sim", "cross_side_sim", "stance_separation", "self_repetition"]
    with open(os.path.join(out_dir, "semantic.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for lab, res in results.items():
            w.writerow([lab, backend] + [f"{res.get(c, 0):.4f}" if isinstance(res.get(c), float) else res.get(c, "")
                                         for c in cols[2:]])

    L = ["# Semantic (text-space) report\n",
         f"Backend: **{backend}**" + (" (lexical TF-IDF - underestimates paraphrases; re-run with --backend ollama when compute is available)" if backend == "tfidf" else "") + "\n",
         "| condition | tweets | mean pairwise sim | near-dup rate | text convergence (late-early) | stance separation (within-cross) | self-repetition |",
         "|---|---|---|---|---|---|---|"]
    for lab, res in results.items():
        L.append(f"| **{lab}** | {res.get('n_tweets')} | {res.get('mean_pairwise_sim', 0):.3f} | {res.get('near_dup_rate', 0):.3f} | "
                 f"{res.get('text_convergence', 0):+.3f} | {res.get('stance_separation', 0):+.3f} | {res.get('self_repetition', 0):.3f} |")
    L += ["", "![similarity](figures/semantic_similarity.png)", "",
          "**How to read:** mean pairwise sim / near-dup rate = homogenisation (agents sounding alike - loss of discriminative power). "
          "text convergence > 0 = tweets keep converging in wording even when integer ratings stall. "
          "stance separation > 0 = FOR and AGAINST posts actually read differently (a good model expresses stance in text); near 0 = sides are indistinguishable. "
          "self-repetition = agents recycling their own wording (high = template-like output).", ""]
    open(os.path.join(out_dir, "semantic_report.md"), "w", encoding="utf-8").write("\n".join(L))

# ----------------------------------------------------------------------------- main

def run_semantic(specs, labels=None, backend="tfidf", embed_model="nomic-embed-text", out=None, url=None):
    runs_items = [tweets_from_run(s) for s in specs]
    labs = labels or [eval_runs.spec_name(s) for s in specs]
    out = out or os.path.join(os.path.dirname(specs[0].rstrip("/\\")) or ".", "semantic_report")
    os.makedirs(out, exist_ok=True)
    results = {}
    if backend == "ollama":
        for (run, items), lab in zip(runs_items, labs):
            texts = [it["text"] for it in items]
            dense = ollama_embed(texts, embed_model, os.path.join(out, f"embed_cache_{lab}.json"), url=url)
            results[lab] = analyze(run, items, dense=dense)
    else:
        all_texts = [it["text"] for (_r, items) in runs_items for it in items]
        vec = build_tfidf(all_texts)
        for (run, items), lab in zip(runs_items, labs):
            results[lab] = analyze(run, items, vec=vec)
    write_outputs(results, labs, out, backend)
    return {"out": out, "report": os.path.join(out, "semantic_report.md"), "results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in results.items()}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--backend", choices=["tfidf", "ollama"], default="tfidf")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=None)
    ap.add_argument("--url", default=None)
    a = ap.parse_args()
    specs = []
    for p in a.paths:
        p = p.rstrip("/\\")
        specs += [p] if (os.path.isfile(p) or eval_runs.is_run_dir(p)) else eval_runs.find_run_specs(p)
    if not specs: sys.exit("no runs found")
    labels = [x.strip() for x in a.labels.split(",")] if a.labels else None
    res = run_semantic(specs, labels=labels, backend=a.backend, embed_model=a.embed_model, out=a.out, url=a.url)
    print("wrote", res["report"])
    for lab, r in res["results"].items():
        print(f"  [{lab}] tweets={r.get('n_tweets')} sim={r.get('mean_pairwise_sim', 0):.3f} conv={r.get('text_convergence', 0):+.3f} sep={r.get('stance_separation', 0):+.3f}")

if __name__ == "__main__":
    main()
