#!/usr/bin/env python3
"""How much information does the -2..+2 integer opinion scale LOSE about what
agents actually say? Runs on EXISTING runs; no LLM needed with tfidf.

Five measurements per run, all in text-similarity space:
  R2 (headline)     variance in text space explained by the rating: 1 - within-
                    rating variance / total variance (distance = 1 - cosine to
                    centroid). Low R2 = the integer compresses away most of what
                    the texts distinguish.
  within vs between mean pairwise similarity of same-rating tweets vs different-
                    rating tweets. Small gap = ratings barely organise the text.
  text->rating      leave-one-out nearest-centroid recovery of the rating from
                    the text alone: accuracy + 5x5 confusion. Adjacent-rating
                    confusion (+1 vs +2) tells whether the 5 levels are textually
                    real; extreme confusion (-2 vs +2) whether even DIRECTION is
                    textually encoded.
  flat-rating drift how much the text moves between an agent's CONSECUTIVE
                    same-rating tweets (rating flat, words changing = hidden
                    dynamics the integer series misses).
  voices/rating     greedy clustering inside each rating bucket at sim>=thresh:
                    how many distinguishable "ways of holding rating r" collapse
                    into one integer.

Backends: tfidf (now; lexical - conservative, UNDERSTATES semantic diversity
loss) / ollama (real embeddings, cached - final numbers).

Usage:
  python tools/scale_infoloss.py RUN [RUN ...] [--labels a,b]
         [--backend tfidf|ollama] [--embed-model nomic-embed-text] [--out DIR]
Outputs: scale_infoloss.md, scale_infoloss.csv, figures/scale_infoloss.png
"""
from __future__ import annotations
import argparse, csv, os, sys
from collections import defaultdict

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

RATINGS = [-2, -1, 0, 1, 2]


def _pairs_mean(sim, idxs_a, idxs_b=None):
    """Mean similarity over pairs within idxs_a (if idxs_b None) or across a x b."""
    tot, n = 0.0, 0
    if idxs_b is None:
        for i in range(len(idxs_a)):
            for j in range(i + 1, len(idxs_a)):
                tot += sim(idxs_a[i], idxs_a[j]); n += 1
    else:
        for i in idxs_a:
            for j in idxs_b:
                tot += sim(i, j); n += 1
    return (tot / n) if n else float("nan"), n


def analyze_run(items, sim, vecs_dense=None, vecs_sparse=None, cluster_thresh=0.5):
    """All five measurements for one run's tweet list."""
    n = len(items)
    by_r = defaultdict(list)
    for i, it in enumerate(items):
        r = max(-2, min(2, int(it["stance"])))
        by_r[r].append(i)
    res = {"n_tweets": n, "n_ratings_used": sum(1 for r in RATINGS if by_r[r])}

    # --- within vs between ---
    within_all, wn = [], 0
    per_rating_within = {}
    for r in RATINGS:
        m, cnt = _pairs_mean(sim, by_r[r])
        per_rating_within[r] = m
        if cnt:
            within_all.append((m, cnt)); wn += cnt
    res["within_sim"] = sum(m * c for m, c in within_all) / wn if wn else float("nan")
    cross_tot, cross_n = 0.0, 0
    rs = [r for r in RATINGS if by_r[r]]
    for a_i in range(len(rs)):
        for b_i in range(a_i + 1, len(rs)):
            m, cnt = _pairs_mean(sim, by_r[rs[a_i]], by_r[rs[b_i]])
            cross_tot += m * cnt; cross_n += cnt
    res["between_sim"] = cross_tot / cross_n if cross_n else float("nan")
    res["separation"] = res["within_sim"] - res["between_sim"]
    res["per_rating_within"] = per_rating_within

    # --- R2: 1 - within-variance / total-variance, distance = 1 - cos(v, centroid)
    def centroid(idxs, exclude=None):
        acc = defaultdict(float); cnt = 0
        for i in idxs:
            if i == exclude: continue
            v = vecs_sparse[i] if vecs_sparse is not None else None
            if v is not None:
                for k, x in v.items(): acc[k] += x
            cnt += 1
        if vecs_sparse is not None:
            return {k: x / cnt for k, x in acc.items()} if cnt else {}
        dim = len(vecs_dense[idxs[0]])
        accd = [0.0] * dim; cnt = 0
        for i in idxs:
            if i == exclude: continue
            for d in range(dim): accd[d] += vecs_dense[i][d]
            cnt += 1
        return [x / cnt for x in accd] if cnt else accd

    def cos_to(i, cen):
        if vecs_sparse is not None:
            return sa.cosine(vecs_sparse[i], cen)
        return sa.dense_cos(vecs_dense[i], cen)

    all_idx = list(range(n))
    g_cen = centroid(all_idx)
    total_var = sum(1 - cos_to(i, g_cen) for i in all_idx) / n
    within_var_sum = 0.0
    for r in RATINGS:
        if not by_r[r]: continue
        cen = centroid(by_r[r])
        within_var_sum += sum(1 - cos_to(i, cen) for i in by_r[r])
    within_var = within_var_sum / n
    res["total_var"] = total_var
    res["within_var"] = within_var
    res["r2"] = 1 - within_var / total_var if total_var > 0 else float("nan")

    # --- leave-one-out nearest-centroid: text -> rating ---
    conf = {ra: {rb: 0 for rb in RATINGS} for ra in RATINGS}
    correct = 0; classified = 0
    for r in RATINGS:
        for i in by_r[r]:
            best, best_s = None, -2.0
            for r2 in RATINGS:
                idxs = by_r[r2]
                if not idxs or (r2 == r and len(idxs) < 2):
                    continue
                cen = centroid(idxs, exclude=i if r2 == r else None)
                s = cos_to(i, cen)
                if s > best_s:
                    best, best_s = r2, s
            if best is None: continue
            conf[r][best] += 1
            classified += 1
            correct += int(best == r)
    res["loo_accuracy"] = correct / classified if classified else float("nan")
    adj = sum(conf[r][r2] for r in RATINGS for r2 in RATINGS if abs(r - r2) == 1)
    res["adjacent_confusion"] = adj / classified if classified else float("nan")
    sign_ok = sum(conf[r][r2] for r in RATINGS for r2 in RATINGS
                  if (r > 0) == (r2 > 0) and (r < 0) == (r2 < 0))
    res["sign_accuracy"] = sign_ok / classified if classified else float("nan")
    res["confusion"] = conf

    # --- flat-rating drift (consecutive same-agent tweets) ---
    by_agent = defaultdict(list)
    for i, it in enumerate(items):
        by_agent[it["author"]].append(i)
    same_r, diff_r = [], []
    for idxs in by_agent.values():
        for a, b in zip(idxs, idxs[1:]):
            s = sim(a, b)
            (same_r if items[a]["stance"] == items[b]["stance"] else diff_r).append(s)
    res["flat_rating_consec_sim"] = sum(same_r) / len(same_r) if same_r else float("nan")
    res["moved_rating_consec_sim"] = sum(diff_r) / len(diff_r) if diff_r else float("nan")
    res["n_flat_pairs"] = len(same_r); res["n_moved_pairs"] = len(diff_r)

    # --- voices per rating: greedy leader clustering at sim >= thresh ---
    voices = {}
    for r in RATINGS:
        leaders = []
        for i in by_r[r]:
            if not any(sim(i, l) >= cluster_thresh for l in leaders):
                leaders.append(i)
        voices[r] = len(leaders) if by_r[r] else 0
    res["voices_per_rating"] = voices
    occupied = [r for r in RATINGS if by_r[r]]
    res["mean_voices"] = (sum(voices[r] for r in occupied) / len(occupied)) if occupied else 0
    res["bucket_sizes"] = {r: len(by_r[r]) for r in RATINGS}
    return res


def run_scale(specs, labels, backend="tfidf", embed_model="nomic-embed-text",
              out=None, url=None, cluster_thresh=None, log=print):
    runs = []
    for spec, label in zip(specs, labels):
        _, items = sa.tweets_from_run(spec)
        runs.append({"label": label, "items": items})
    thresh = cluster_thresh if cluster_thresh is not None else (0.5 if backend == "tfidf" else 0.8)
    all_texts = [it["text"] for r in runs for it in r["items"]]

    results = []
    for r in runs:
        items = r["items"]
        if len(items) < 10:
            results.append({"label": r["label"], "empty": True}); continue
        texts = [it["text"] for it in items]
        if backend == "ollama":
            cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"embed_cache_{embed_model.replace(':', '_')}.json")
            dense = sa.ollama_embed(texts, embed_model, cache, url=url)
            sim = lambda i, j: sa.dense_cos(dense[i], dense[j])
            res = analyze_run(items, sim, vecs_dense=dense, cluster_thresh=thresh)
        else:
            vec = sa.build_tfidf(all_texts)
            vs = [vec(t) for t in texts]
            sim = lambda i, j: sa.cosine(vs[i], vs[j])
            res = analyze_run(items, sim, vecs_sparse=vs, cluster_thresh=thresh)
        res["label"] = r["label"]
        results.append(res)

    out_dir = out or os.path.join(os.path.dirname(os.path.abspath(specs[0])) or ".",
                                  "scale_infoloss")
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # ---- csv ----
    cpath = os.path.join(out_dir, "scale_infoloss.csv")
    cols = ["run", "backend", "n_tweets", "r2", "within_sim", "between_sim", "separation",
            "loo_accuracy", "sign_accuracy", "adjacent_confusion",
            "flat_rating_consec_sim", "moved_rating_consec_sim", "mean_voices"]
    with open(cpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for res in results:
            if res.get("empty"): continue
            w.writerow([res["label"], backend] +
                       [round(res[c], 4) if isinstance(res.get(c), float) else res.get(c, "")
                        for c in cols[2:]])

    # ---- figure ----
    live = [res for res in results if not res.get("empty")]
    if live:
        fig, axes = plt.subplots(2, len(live), figsize=(5.4 * len(live), 7.2), squeeze=False)
        for col, res in enumerate(live):
            ax = axes[0][col]
            occupied = [r for r in RATINGS if res["bucket_sizes"][r]]
            xs = range(len(occupied))
            ax.bar([x - 0.2 for x in xs], [res["per_rating_within"][r] for r in occupied],
                   width=0.4, color="#2f6bff", label="within-rating sim")
            ax.axhline(res["between_sim"], color="#e8590c", ls="--", lw=1.4,
                       label=f"between-rating {res['between_sim']:.2f}")
            for x, r in zip(xs, occupied):
                ax.annotate(f"n={res['bucket_sizes'][r]}\nvoices={res['voices_per_rating'][r]}",
                            xy=(x - 0.2, 0.005), ha="center", va="bottom", fontsize=8, color="#333")
            ax.set_xticks(list(xs), [str(r) for r in occupied])
            ax.set_xlabel("rating"); ax.set_ylabel("mean pairwise sim")
            ax.set_title(f"{res['label']} - R²={res['r2']:.2f}")
            ax.legend(fontsize=8.5, frameon=False)
            ax2 = axes[1][col]
            mat = [[res["confusion"][ra][rb] for rb in RATINGS] for ra in RATINGS]
            im = ax2.imshow(mat, cmap="Blues")
            for yi, ra in enumerate(RATINGS):
                for xi, rb in enumerate(RATINGS):
                    v = mat[yi][xi]
                    if v:
                        ax2.text(xi, yi, str(v), ha="center", va="center", fontsize=9,
                                 color="white" if v > max(max(row) for row in mat) * 0.6 else "#1a1a1a")
            ax2.set_xticks(range(5), [str(r) for r in RATINGS])
            ax2.set_yticks(range(5), [str(r) for r in RATINGS])
            ax2.set_xlabel("predicted from text (LOO centroid)")
            ax2.set_ylabel("true rating")
            ax2.set_title(f"text→rating: acc {res['loo_accuracy']:.2f}, sign {res['sign_accuracy']:.2f}")
            ax2.grid(False)
        fig.suptitle(f"Scale information loss ({backend})", y=1.0, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "scale_infoloss.png")); plt.close(fig)

    # ---- report ----
    L = ["# Scale information-loss report\n",
         f"Backend: **{backend}**"
         + (" (lexical: υποεκτιμά την ομοιότητα των παραφράσεων → το within-rating variance φουσκώνει → "
            "το R² εδώ είναι μάλλον ΚΑΤΩ φράγμα· με πραγματικά embeddings περιμένουμε R² ίσο ή υψηλότερο)" if backend == "tfidf" else "")
         + f". Cluster threshold: {thresh}.\n",
         "Ερώτημα: πόση από την ποικιλία του ΚΕΙΜΕΝΟΥ εξηγεί το ακέραιο rating -2..+2;\n",
         "| run | tweets | R² (var. explained) | within | between | LOO acc | sign acc | adj. confusion | flat-drift sim | voices/rating |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        if res.get("empty"):
            L.append(f"| **{res['label']}** | (too few tweets) | | | | | | | | |"); continue
        L.append(f"| **{res['label']}** | {res['n_tweets']} | {res['r2']:.2f} | {res['within_sim']:.2f} | "
                 f"{res['between_sim']:.2f} | {res['loo_accuracy']:.2f} | {res['sign_accuracy']:.2f} | "
                 f"{res['adjacent_confusion']:.2f} | {res['flat_rating_consec_sim']:.2f} | {res['mean_voices']:.1f} |")
    L += ["", "![scale](figures/scale_infoloss.png)", "",
          "**Πώς διαβάζεται:** R² κοντά στο 0 = το rating εξηγεί ελάχιστη από τη διακύμανση του "
          "κειμένου (η κλίμακα συμπιέζει πραγματική πληροφορία). LOO acc = πόσο ανακτάται το rating "
          "ΜΟΝΟ από το κείμενο· sign acc = ανακτάται τουλάχιστον η ΠΛΕΥΡΑ; Υψηλό adjacent confusion "
          "με υψηλό sign acc = η κατεύθυνση είναι κειμενικά πραγματική αλλά τα 5 σκαλιά όχι (τα ±1/±2 "
          "δεν διαφέρουν στο πώς γράφονται). flat-drift: ομοιότητα διαδοχικών tweets ίδιου agent με "
          "ΑΜΕΤΑΒΛΗΤΟ rating — όσο πιο χαμηλή, τόσο περισσότερη κίνηση συμβαίνει στο κείμενο που η "
          "ακέραια χρονοσειρά δεν καταγράφει. voices/rating: πόσες διακριτές «φωνές» συνυπάρχουν στο "
          "ίδιο σκαλί.", ""]
    for res in results:
        if res.get("empty"): continue
        L.append(f"**{res['label']}** — bucket sizes: " +
                 ", ".join(f"{r}: {res['bucket_sizes'][r]}" for r in RATINGS) +
                 " · voices: " + ", ".join(f"{r}: {res['voices_per_rating'][r]}" for r in RATINGS) +
                 f" · flat pairs {res['n_flat_pairs']} vs moved pairs {res['n_moved_pairs']} "
                 f"({res['flat_rating_consec_sim']:.2f} vs {res['moved_rating_consec_sim']:.2f})")
    path = os.path.join(out_dir, "scale_infoloss.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    log(f"wrote {path}")
    return path, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", default="")
    ap.add_argument("--backend", default="tfidf", choices=["tfidf", "ollama"])
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--cluster-thresh", type=float, default=None,
                    help="voice-cluster similarity threshold (default 0.5 tfidf / 0.8 ollama)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--url", default=None)
    a = ap.parse_args()
    labels = [x.strip() for x in a.labels.split(",") if x.strip()]
    if len(labels) != len(a.runs):
        labels = eval_runs.infer_labels(a.runs)
    run_scale(a.runs, labels, backend=a.backend, embed_model=a.embed_model,
              out=a.out, url=a.url, cluster_thresh=a.cluster_thresh)


if __name__ == "__main__":
    main()
