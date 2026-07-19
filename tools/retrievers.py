#!/usr/bin/env python3
"""Three retrieval architectures over the multihop corpus, as standalone deterministic functions.

All three rank the SAME corpus (multihop jsonl + sidecar files)
for a query and return the top-k snippet ids. They differ ONLY in what signal
they use - that difference IS the experimental variable:

  lexical  token overlap with the query (mirrors the main script's
           _rag_score_rows: >=3-char tokens, set overlap, sort by
           (-score, text length, id)). Reads text only.
  dense    cosine over Ollama embeddings (nomic-embed-text by default),
           on-disk cache; reads text only. Needs a running Ollama once
           per new corpus/query set - afterwards fully offline via cache.
  lexical+seeds
           CONTROL. lexical, but the query is first expanded with the same seed
           entities the graph derives from the alias map. Equalises the QUERY-side
           information between lexical and graph.
  lexical+tags
           CONTROL. lexical over "snippet text + that snippet's own entity tags",
           with the query expanded as above. Equalises the INDEX-side information:
           this backend reads exactly the bytes the graph reads.
  graph    entity graph from the hand-written tags: seed entities are
           matched in the query via the alias map (longest-alias-first,
           span consumption), then BFS over snippet<->entity links up to
           max_hops; selection prefers the LONGEST connected simple path
           (a chain), then fills with BFS order. Reads text + entity tags.
           Never reads chain_id / groundtruth (evaluation-only files).

The two controls exist to answer the obvious objection to any graph-vs-lexical
comparison: that the graph wins because it was given the entity tags, not because
traversal is a better access structure. If a control closes the gap, the result is
an information artefact; if it does not, the gap is attributable to structure.
If a control closes the gap the advantage was information; if not, it was structure.

Trust boundaries: retrievers may read snippet text and entity tags (graph and both
controls also read the alias map); the groundtruth sidecar (chains, implied
conclusions) is evaluation-only and is never read by any retriever or rendered into
any prompt.
"""
from __future__ import annotations
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_lint import alias_spans, STOP  # same matching + stopword rules as the linter


# ------------------------------------------------------------------ loading

def load_corpus(jsonl_path):
    snips = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    stem = os.path.splitext(jsonl_path)[0]
    graph = json.load(open(stem + ".graph.json", encoding="utf-8"))
    alias_map = {e: [a.lower() for a in v.get("aliases", [])]
                 for e, v in graph.get("entities", {}).items()}
    return snips, alias_map


# ------------------------------------------------------------------ lexical

def _norm(s):
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s):
    return {t for t in _norm(s).split() if len(t) >= 3}


def lexical_rank(query, snips, k=4):
    """Mirror of main's _rag_score_rows ordering (deterministic, no seeded pick)."""
    q = _tokens(query)
    scored = []
    for s in snips:
        ov = len(q & _tokens(s["text"]))
        if ov > 0:
            scored.append((-ov, len(s["text"]), s["id"]))
    scored.sort()
    return [sid for _, _, sid in scored[:k]]


# ------------------------------------------------------------------ dense

def dense_rank(query, snips, k=4, model="nomic-embed-text", cache_path=None, url=None):
    """Embedding cosine top-k. Raises URLError/OSError if Ollama is unreachable
    and the cache does not already hold every needed vector."""
    import semantic_analysis as sa
    cache_path = cache_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"embed_cache_{model.replace(':','_')}.json")
    texts = [s["text"] for s in snips] + [query]
    vecs = sa.ollama_embed(texts, model, cache_path, url=url)
    qv = vecs[-1]
    scored = sorted(
        ((-sa.dense_cos(v, qv), s["id"]) for v, s in zip(vecs, snips)),
        key=lambda t: (t[0], t[1]))
    return [sid for _, sid in scored[:k]]


# ------------------------------------------------------------------ graph

def build_graph(snips):
    ent2snips, snip2ents = {}, {}
    for s in snips:
        snip2ents[s["id"]] = list(s.get("entities") or [])
        for e in snip2ents[s["id"]]:
            ent2snips.setdefault(e, []).append(s["id"])
    for e in ent2snips:
        ent2snips[e].sort()
    return ent2snips, snip2ents


def seed_entities(query, alias_map):
    """Entities whose aliases occur in the query (longest-first, span consumption)."""
    return sorted(alias_spans(query, alias_map).keys())


def _bfs(seeds, ent2snips, snip2ents, max_hops):
    """Level-order visit: depth(snippet) = entity-hops from the nearest seed entity."""
    depth, order = {}, []
    frontier_ents = sorted(set(seeds))
    seen_ents = set(frontier_ents)
    d = 0
    while frontier_ents and d <= max_hops:
        next_ents = []
        for e in frontier_ents:
            for sid in ent2snips.get(e, []):
                if sid not in depth:
                    depth[sid] = d
                    order.append(sid)
                for e2 in snip2ents.get(sid, []):
                    if e2 not in seen_ents:
                        seen_ents.add(e2)
                        next_ents.append(e2)
        frontier_ents = sorted(set(next_ents))
        d += 1
    return depth, order


def _best_path(visited, snip2ents, ent2snips, texts, query, hub_max=6):
    """Best simple path over shared-entity edges within the visited set.

    v2 ranking (after the Stage A hub-clique pathology): (1) edges may only use
    SPECIFIC entities (corpus degree <= hub_max) - a topic anchor like 'North
    Tower' seeds the walk but is not a chain link, otherwise all chain heads
    form a clique and the 'longest path' snakes across heads instead of down a
    chain; (2) candidate paths are scored by (query token overlap of their
    texts, then length), so the QUERY chooses which chain, not id order;
    ties broken by the lexicographically smallest id sequence.
    Sizes are tiny (<=60 nodes), exhaustive DFS with a length cap is fine."""
    vis = sorted(visited)
    q = _tokens(query) - STOP
    ov = {v: len(q & (_tokens(texts.get(v, "")) - STOP)) for v in vis}
    adj = {v: [] for v in vis}
    for i, a in enumerate(vis):
        for b in vis[i + 1:]:
            link = set(snip2ents.get(a, [])) & set(snip2ents.get(b, []))
            if any(len(ent2snips.get(e, [])) <= hub_max for e in link):
                adj[a].append(b)
                adj[b].append(a)
    for v in adj:
        adj[v].sort()
    best, best_key = [], (-1, -1)

    def dfs(node, path, score):
        nonlocal best, best_key
        path = path + [node]
        score = score + ov[node]
        key = (score, len(path))
        if key > best_key or (key == best_key and path < best):
            best, best_key = path, key
        if len(path) >= 8:
            return
        for nb in adj[node]:
            if nb not in path:
                dfs(nb, path, score)

    for v in vis:
        dfs(v, [], 0)
    return best


def graph_rank(query, snips, alias_map, k=4, max_hops=3, hub_max=6, debug=None):
    """Chain-first selection: best connected path in the BFS neighbourhood
    (query-overlap first, then length), then remaining BFS order (depth, id).
    Falls back to lexical seeds."""
    ent2snips, snip2ents = build_graph(snips)
    texts = {s["id"]: s["text"] for s in snips}
    seeds = seed_entities(query, alias_map)
    if not seeds:
        top = lexical_rank(query, snips, k=1)
        seeds = sorted(snip2ents.get(top[0], [])) if top else []
    depth, order = _bfs(seeds, ent2snips, snip2ents, max_hops)
    if not order:
        return lexical_rank(query, snips, k)
    path = _best_path(set(order), snip2ents, ent2snips, texts, query, hub_max)
    picked = list(path[:k])
    if len(picked) < k:
        rest = sorted((depth[sid], sid) for sid in order if sid not in picked)
        picked += [sid for _, sid in rest[: k - len(picked)]]
    if debug is not None:
        debug.update({"seeds": seeds, "path": path,
                      "visited": {sid: depth[sid] for sid in order}})
    return picked[:k]


# ------------------------------------------------------------------ controls

def _expand_query(query, alias_map):
    """Query + the seed entity names the graph would derive from it.

    Same alias matching the graph uses, so the two see identical seeds.
    """
    seeds = seed_entities(query, alias_map)
    return (query + " " + " ".join(seeds)) if seeds else query


def tag_snippets(snips):
    """Snippets whose text has that snippet's own entity tags appended.

    This is what makes lexical+tags an index-side control: after this, a purely
    lexical ranker reads exactly the bytes the graph reads. Non-destructive - the
    caller's snippet dicts are left alone.
    """
    return [dict(s, text=s["text"] + " " + " ".join(s.get("entities") or []))
            for s in snips]


def lexical_seeds_rank(query, snips, alias_map, k=4):
    """CONTROL: query-side equalisation. lexical over the entity-expanded query."""
    return lexical_rank(_expand_query(query, alias_map), snips, k)


def lexical_tags_rank(query, snips, alias_map, k=4):
    """CONTROL: index-side equalisation. lexical over entity-expanded query AND
    tag-augmented snippet text - identical information to the graph, no traversal."""
    return lexical_rank(_expand_query(query, alias_map), tag_snippets(snips), k)
