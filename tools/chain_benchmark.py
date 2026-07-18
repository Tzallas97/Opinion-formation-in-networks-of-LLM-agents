#!/usr/bin/env python3
"""Chain-recovery benchmark - no LLM, runs anywhere, evaluates the three
retrieval architectures on the multihop corpus's ground truth.

This is the offline GATE before any simulation run uses these retrievers:
if graph does not clearly beat lexical/dense at recovering full chains
offline, the corpus (or the ranking rule) gets reworked before any compute
is spent.

For every test query x retriever it reports, per expected chain:
  recovery@k        |chain members in top-k| / |members|
  full_chain        1 if the ENTIRE chain is inside the top-k
and per query: distractor_share (distractor members in top-k / k).
Queries expecting many chains (e.g. the bare claim) cannot fit them all in
k slots, so the summary reports both the mean over (query, chain) pairs and
the best-chain-per-query mean (can the retriever nail at least ONE chain?).

Usage:
  python tools/chain_benchmark.py CORPUS.jsonl [--k 4] [--max-hops 3]
         [--backends lexical,graph,dense] [--embed-model nomic-embed-text]
         [--out DIR]
Dense needs Ollama once (embeddings cached); it is skipped with a note if
unreachable. Output: OUT/chain_benchmark.md (default: <corpus dir>/benchmark/).
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrievers as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--backends", default="lexical,graph,dense")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    snips, alias_map = R.load_corpus(a.corpus)
    stem = os.path.splitext(a.corpus)[0]
    gt = json.load(open(stem + ".groundtruth.json", encoding="utf-8"))
    chains = {c["chain_id"]: c for c in gt["chains"]}
    distractor_ids = {m for d in gt.get("distractors", []) for m in d["members"]}
    queries = gt.get("test_queries", [])
    backends = [b.strip() for b in a.backends.split(",") if b.strip()]

    out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(a.corpus)), "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    results, skipped = {}, {}
    for b in backends:
        rows = []
        try:
            for q in queries:
                dbg = {}
                if b == "lexical":
                    top = R.lexical_rank(q["text"], snips, a.k)
                elif b == "graph":
                    top = R.graph_rank(q["text"], snips, alias_map, a.k, a.max_hops, debug=dbg)
                elif b == "dense":
                    top = R.dense_rank(q["text"], snips, a.k, model=a.embed_model)
                else:
                    raise ValueError(f"unknown backend {b}")
                topset = set(top)
                per_chain = {}
                for cid in q.get("expected_chains", []):
                    members = chains[cid]["members"]
                    got = len(topset & set(members))
                    per_chain[cid] = {"recovery": got / len(members),
                                      "full": int(got == len(members))}
                rows.append({"qid": q["qid"], "top": top, "per_chain": per_chain,
                             "distractor_share": len(topset & distractor_ids) / a.k,
                             "seeds": dbg.get("seeds")})
            results[b] = rows
        except Exception as e:
            skipped[b] = f"{type(e).__name__}: {e}"

    # ---- summary ----
    def summarize(rows):
        pairs = [(r["qid"], cid, m) for r in rows for cid, m in r["per_chain"].items()]
        n = len(pairs) or 1
        mean_rec = sum(m["recovery"] for _, _, m in pairs) / n
        full_rate = sum(m["full"] for _, _, m in pairs) / n
        best = [max((m["recovery"] for m in r["per_chain"].values()), default=0) for r in rows]
        best_full = [max((m["full"] for m in r["per_chain"].values()), default=0) for r in rows]
        dis = sum(r["distractor_share"] for r in rows) / (len(rows) or 1)
        return {"mean_recovery": mean_rec, "full_chain_rate": full_rate,
                "best_chain_recovery": sum(best) / (len(best) or 1),
                "best_chain_full_rate": sum(best_full) / (len(best_full) or 1),
                "distractor_share": dis}

    summaries = {b: summarize(rows) for b, rows in results.items()}

    # ---- gate ----
    gate = ""
    if "graph" in summaries:
        g = summaries["graph"]["best_chain_full_rate"]
        others = [s["best_chain_full_rate"] for b, s in summaries.items() if b != "graph"]
        if others and g > max(others) and summaries["graph"]["mean_recovery"] >= max(
                s["mean_recovery"] for b, s in summaries.items() if b != "graph") + 0.15:
            gate = "PASS - ο graph υπερτερεί καθαρά στο full-chain recovery."
        elif others:
            gate = "FAIL/REWORK - ο graph ΔΕΝ διαχωρίζει αρκετά· χρειάζεται δουλειά στο corpus ή στο ranking πριν από κάθε simulation run."
        else:
            gate = "(μόνο graph έτρεξε - χρειάζονται και οι άλλοι για ετυμηγορία)"

    # ---- report ----
    L = ["# Chain-recovery benchmark (offline)\n",
         f"Corpus: `{os.path.basename(a.corpus)}` - {len(snips)} snippets, "
         f"{len(chains)} chains, {len(queries)} test queries, k={a.k}, max_hops={a.max_hops}.\n",
         "| retriever | mean recovery@k | full-chain rate | best-chain recovery | best-chain full | distractor share |",
         "|---|---|---|---|---|---|"]
    for b in backends:
        if b in summaries:
            s = summaries[b]
            L.append(f"| **{b}** | {s['mean_recovery']:.2f} | {s['full_chain_rate']:.2f} | "
                     f"{s['best_chain_recovery']:.2f} | {s['best_chain_full_rate']:.2f} | "
                     f"{s['distractor_share']:.2f} |")
        else:
            L.append(f"| **{b}** | skipped | | | | |")
    if skipped:
        L.append("")
        for b, msg in skipped.items():
            L.append(f"- `{b}` skipped: {msg}")
    L += ["", f"**Gate:** {gate}", "", "## Per query (τι επέστρεψε ο καθένας)", ""]
    for i, q in enumerate(queries):
        L.append(f"**{q['qid']}** - \"{q['text'][:100]}...\" (expects {', '.join(q['expected_chains'])})")
        for b in backends:
            if b not in results:
                continue
            r = results[b][i]
            per = ", ".join(f"{cid} {m['recovery']:.2f}{'✓' if m['full'] else ''}"
                            for cid, m in r["per_chain"].items())
            seeds = f" seeds={r['seeds']}" if b == "graph" and r.get("seeds") else ""
            L.append(f"- {b}: [{', '.join(r['top'])}] → {per}"
                     f"{' · distractors ' + format(r['distractor_share'], '.2f') if r['distractor_share'] else ''}{seeds}")
        L.append("")
    path = os.path.join(out_dir, "chain_benchmark.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:20]))
    print("...\nwrote", path)


if __name__ == "__main__":
    main()
