#!/usr/bin/env python3
"""Lint a multi-hop RAG corpus against the rules in docs/multihop_corpus_spec.md.

Checks (E = error, exits 1; W = warning, listed but non-fatal):
  E1  jsonl parses; required fields present; unique ids; valid directions; uniform topic_key
  E2  every tagged entity exists in the alias map AND one of its aliases appears
      verbatim in the snippet text (case-insensitive, word boundaries)
  E3  tag completeness: every alias occurrence in a text is tagged (longest-alias-first
      with span consumption, so 'P. Harris' does not force a 'D. Harris' tag)
  E4  ground truth: chain/distractor members exist; roles cover exactly all ids and
      agree with chain membership; consecutive chain members share >=1 entity;
      chains have implied_conclusion, distractors have null
  E5  reachability: every chain has >=1 member tagged with an expected seed of a
      test query that targets it
  W1  anti-leak: chain members sharing distinctive (non-stopword, non-entity) vocab
  W2  claim vocabulary inside non-head chain members
  W3  direction balance / chains per side / hop symmetry
  W4  entity degree table; hubs flagged above --hub-degree (default 12)

Usage: python tools/corpus_lint.py CORPUS.jsonl [--claim "..."] [--hub-degree 12]
Sidecars <stem>.graph.json and <stem>.groundtruth.json are found automatically.
"""
from __future__ import annotations
import argparse, json, os, re, sys, itertools
from collections import Counter

STOP = set("""a an the of for on in to and at by from with was were as that this his her its it
show shows lists listed list during before after under over any no not is are be been than then
per the a's s across against without within about had has have who which when where all one two
three same new old same""".split())

DEFAULT_CLAIM = ("twin towers were brought down by a conspiracy of government insiders "
                 "to justify wars and increase surveillance")


def load(path):
    snips = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    stem = os.path.splitext(path)[0]
    graph = json.load(open(stem + ".graph.json", encoding="utf-8"))
    gt = json.load(open(stem + ".groundtruth.json", encoding="utf-8"))
    return snips, graph, gt


def alias_spans(text, alias_map):
    """Return {canonical: [(start, end), ...]} using longest-alias-first span consumption."""
    low = text.lower()
    taken = [False] * len(low)
    pairs = sorted(((a, e) for e, al in alias_map.items() for a in al),
                   key=lambda p: -len(p[0]))
    found = {}
    for alias, ent in pairs:
        for m in re.finditer(r"(?<![\w'])" + re.escape(alias) + r"(?![\w-])", low):
            s, e = m.span()
            if any(taken[s:e]):
                continue
            for i in range(s, e):
                taken[i] = True
            found.setdefault(ent, []).append((s, e))
    return found


def words(text):
    return {re.sub(r"'s$", "", w) for w in re.findall(r"[a-z']+", text.lower())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--claim", default=DEFAULT_CLAIM)
    ap.add_argument("--hub-degree", type=int, default=12)
    a = ap.parse_args()
    snips, graph, gt = load(a.corpus)
    alias_map = {e: [x.lower() for x in v.get("aliases", [])]
                 for e, v in graph.get("entities", {}).items()}
    errs, warns = [], []

    # E1 basics
    ids = [s.get("id") for s in snips]
    if len(ids) != len(set(ids)):
        errs.append("E1 duplicate ids: " + str([i for i, c in Counter(ids).items() if c > 1]))
    for s in snips:
        for f in ("id", "text", "direction", "topic_key", "entities"):
            if f not in s:
                errs.append(f"E1 {s.get('id','?')}: missing field '{f}'")
        if s.get("direction") not in ("supportive", "criticism", "context"):
            errs.append(f"E1 {s.get('id')}: bad direction '{s.get('direction')}'")
    topics = {s.get("topic_key") for s in snips}
    if len(topics) != 1:
        errs.append(f"E1 mixed topic_key: {topics}")
    by_id = {s["id"]: s for s in snips}

    # E2 + E3 entity tagging both directions
    for s in snips:
        got = alias_spans(s["text"], alias_map)
        for e in s["entities"]:
            if e not in alias_map:
                errs.append(f"E2 {s['id']}: entity '{e}' not in alias map")
            elif e not in got:
                errs.append(f"E2 {s['id']}: no alias of '{e}' appears in text")
        for e in got:
            if e not in s["entities"]:
                errs.append(f"E3 {s['id']}: text mentions '{e}' but tag is missing")

    # E4 ground truth consistency
    chains = gt.get("chains", [])
    distractors = gt.get("distractors", [])
    roles = gt.get("roles", {})
    if set(roles) != set(ids):
        errs.append(f"E4 roles cover {len(roles)} ids, corpus has {len(ids)}; diff: "
                    f"{sorted(set(roles) ^ set(ids))[:6]}")
    member_role = {}
    for ch in chains:
        member_role.update({m: "chain" for m in ch["members"]})
    for d in distractors:
        member_role.update({m: "distractor" for m in d["members"]})
    for i in ids:
        want = member_role.get(i, "standalone")
        if roles.get(i) != want:
            errs.append(f"E4 {i}: role '{roles.get(i)}' but membership implies '{want}'")
    for ch in chains + distractors:
        for m in ch["members"]:
            if m not in by_id:
                errs.append(f"E4 {ch['chain_id']}: member {m} not in corpus")
        for x, y in zip(ch["members"], ch["members"][1:]):
            if x in by_id and y in by_id:
                if not set(by_id[x]["entities"]) & set(by_id[y]["entities"]):
                    errs.append(f"E4 {ch['chain_id']}: {x}->{y} share no entity (broken hop)")
    for ch in chains:
        if not (ch.get("implied_conclusion") or "").strip():
            errs.append(f"E4 {ch['chain_id']}: chain without implied_conclusion")
    for d in distractors:
        if d.get("implied_conclusion") is not None:
            errs.append(f"E4 {d['chain_id']}: distractor must have null conclusion")

    # E5 reachability via test queries
    tq = gt.get("test_queries", [])
    for ch in chains:
        seeds = set()
        for q in tq:
            if ch["chain_id"] in q.get("expected_chains", []):
                seeds |= set(q.get("expected_seeds", []))
        touch = any(set(by_id[m]["entities"]) & seeds for m in ch["members"] if m in by_id)
        if not touch:
            errs.append(f"E5 {ch['chain_id']}: no member tagged with any expected seed of its queries")

    # W1 anti-leak vocabulary inside chains
    for ch in chains:
        ms = [by_id[m] for m in ch["members"] if m in by_id]
        ent_words = set(w for m in ms for e in m["entities"]
                        for w in words(" ".join(alias_map.get(e, []))))
        for x, y in itertools.combinations(ms, 2):
            shared = (words(x["text"]) - STOP - ent_words) & (words(y["text"]) - STOP - ent_words)
            shared = {w for w in shared if len(w) > 3}
            if shared:
                warns.append(f"W1 {ch['chain_id']} {x['id']}~{y['id']} shared vocab: {sorted(shared)}")

    # W2 claim vocabulary in non-head members
    cw = words(a.claim) - STOP
    for ch in chains:
        for m in ch["members"][1:]:
            if m in by_id:
                hit = (words(by_id[m]["text"]) - STOP) & cw
                hit = {w for w in hit if len(w) > 3}
                if hit:
                    warns.append(f"W2 {m}: claim vocab {sorted(hit)}")

    # W3 balance
    dirs = Counter(s["direction"] for s in snips)
    side = Counter(ch["direction"] for ch in chains)
    hops = {d: sorted(ch.get("hops", len(ch["members"]) - 1)
                      for ch in chains if ch["direction"] == d) for d in side}
    if side.get("supportive") != side.get("criticism"):
        warns.append(f"W3 chains per side unbalanced: {dict(side)}")
    if hops.get("supportive") != hops.get("criticism"):
        warns.append(f"W3 hop profiles differ: {hops}")

    # W4 degree table
    deg = Counter(e for s in snips for e in s["entities"])
    hubs = [(e, d) for e, d in deg.most_common() if d > a.hub_degree]
    for e, d in hubs:
        warns.append(f"W4 hub entity '{e}' degree {d} > {a.hub_degree}")

    print(f"corpus: {a.corpus}")
    print(f"snippets: {len(snips)} | directions: {dict(dirs)} | chains: {dict(side)} "
          f"(hops {hops}) | distractors: {len(distractors)} | queries: {len(tq)}")
    print("entity degrees:", ", ".join(f"{e}={d}" for e, d in deg.most_common()))
    print(f"\n{len(errs)} errors, {len(warns)} warnings")
    for e in errs:
        print("  ERR ", e)
    for w in warns:
        print("  warn", w)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
