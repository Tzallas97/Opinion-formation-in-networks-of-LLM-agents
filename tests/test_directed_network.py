#!/usr/bin/env python3
"""Regression tests for DirectedNetwork and coupled rewiring.

Run:  python tests/test_directed_network.py

The load-bearing test is `test_reciprocal_reproduces_undirected`: with
reciprocal=True the directed structure must be byte-identical to the legacy
undirected map the constructors return. If it ever diverges, the "reciprocity is
a condition, not a fork" property is broken and the directed path has
silently changed behaviour.
"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from network_models import (  # noqa: E402
    DirectedNetwork, build_small_world, build_erdos_renyi, build_barabasi_albert,
    evolve_once, _evolve_try_cut, opinion_only_pair_score,
)


def _legacy(builder, *a, **k):
    g = builder(*a, **k)
    return g[0] if isinstance(g, tuple) else g


def _fixture(n=25, seed=7):
    legacy = _legacy(build_barabasi_albert, n, 2, seed=seed)
    rng = random.Random(11)
    opinions = {i: rng.choice([-2, -1, 0, 1, 2]) for i in range(n)}
    attributes = {i: {"age": rng.randint(20, 60),
                      "political": rng.choice(["left", "right", "centre"]),
                      "education": rng.choice(["hs", "ba", "ms"])} for i in range(n)}
    return legacy, opinions, attributes


def test_reciprocal_reproduces_undirected():
    for name, builder, args in [
        ("small_world", build_small_world, (20, 4, 0.1)),
        ("erdos_renyi", build_erdos_renyi, (20, 0.2)),
        ("barabasi", build_barabasi_albert, (20, 2)),
    ]:
        legacy = _legacy(builder, *args, seed=7)
        net = DirectedNetwork.from_undirected(legacy, 20, reciprocal=True)
        net.check_invariants()
        for i in range(20):
            assert net.following[i] == set(legacy.get(i, set())), \
                f"{name}: following[{i}] != legacy"
            assert net.to_undirected_map()[i] == set(legacy.get(i, set())), \
                f"{name}: to_undirected_map[{i}] != legacy"
    print("OK: reciprocal=True reproduces all three constructors exactly")


def test_direction_separates_in_and_out():
    net = DirectedNetwork(5, reciprocal=False)
    for a, b in [(0, 3), (1, 3), (2, 3), (3, 4)]:
        net.add_edge(a, b)
    net.check_invariants()
    assert net.in_degree(3) == 3 and net.out_degree(3) == 1, "hub in/out wrong"
    assert net.in_degree(4) == 1 and net.out_degree(4) == 0, "sink wrong"
    print("OK: in-degree and out-degree are distinct under direction")


def test_edges_never_dropped_by_index_order():
    """The undirected export dedupes with `if i < j`; that silently drops every
    high-to-low edge once direction matters. edges() must not."""
    net = DirectedNetwork(5, reciprocal=False)
    for a, b in [(0, 3), (4, 1), (3, 0), (2, 4)]:
        net.add_edge(a, b)
    real = {(s, d) for s, d, _ in net.edges()}
    naive = {(s, d) for s, d, _ in net.edges() if s < d}
    assert real == {(0, 3), (4, 1), (3, 0), (2, 4)}
    assert (4, 1) in real and (3, 0) in real
    assert (4, 1) not in naive and (3, 0) not in naive
    print("OK: edges() keeps high-to-low edges (i<j dedupe would lose 2 of 4)")


def test_mutual_flag():
    net = DirectedNetwork(3, reciprocal=False)
    net.add_edge(0, 1)
    net.add_edge(1, 0)
    net.add_edge(0, 2)
    flags = {(s, d): m for s, d, m in net.edges()}
    assert flags[(0, 1)] and flags[(1, 0)], "reciprocated pair not marked mutual"
    assert not flags[(0, 2)], "one-way edge marked mutual"
    print("OK: mutual flag distinguishes friendship from double one-way")


def test_M_conserved_and_no_deaf_agents():
    legacy, opinions, attributes = _fixture()
    for reciprocal in (True, False):
        net = DirectedNetwork.from_undirected(legacy, 25, reciprocal=reciprocal)
        m0 = net.edge_count()
        r = random.Random(3)
        for step in range(60, 400):
            L = r.randrange(25)
            cands = list(net.following[L])
            if not cands:
                continue
            evolve_once(net, L, r.choice(cands), opinions, attributes, step,
                        burnin_steps=50, p_add=0.9, rng=r)
            assert net.edge_count() == m0, f"M drifted at step {step}"
        net.check_invariants()
        assert net.isolated() == [], f"deaf agent created (reciprocal={reciprocal})"
    print("OK: edge count conserved exactly; no agent left with out-degree 0")


def test_rollback_when_no_cut_available():
    legacy, opinions, attributes = _fixture()
    net = DirectedNetwork.from_undirected(legacy, 25, reciprocal=True)
    m0 = net.edge_count()
    r = random.Random(5)
    for step in range(60, 200):
        L = r.randrange(25)
        cands = list(net.following[L])
        if not cands:
            continue
        # soft_cut_distance=99 makes every cut fail its criterion
        evolve_once(net, L, r.choice(cands), opinions, attributes, step,
                    burnin_steps=50, p_add=1.0, soft_cut_distance=99, rng=r)
    assert net.edge_count() == m0, "add without a matching cut was not rolled back"
    assert any("rollback" in (c[4] or "") for c in net.changes), "no rollback logged"
    net.check_invariants()
    print("OK: an add with no available cut is rolled back; M holds")


def test_out_degree_guard_blocks_isolating_cut():
    net = DirectedNetwork(4, reciprocal=True)
    net.add_edge(0, 1)   # 0 and 1 now have out-degree 1 each
    opinions = {0: 2, 1: -2, 2: 0, 3: 0}
    attributes = {i: {"age": 30, "political": "left", "education": "ba"} for i in range(4)}
    cut = _evolve_try_cut(net, 0, 1, opinions, attributes, 100,
                          cut_score_threshold=10_000, soft_cut_distance=1,
                          min_out_degree=2, score_fn=opinion_only_pair_score,
                          rng=random.Random(0))
    assert cut is False, "cut proceeded despite out-degree at the floor"
    assert net.out_degree(0) == 1, "edge was removed"
    print("OK: out-degree guard refuses a cut that would deafen an agent")


def test_opinion_only_ignores_attributes():
    """The default scorer must not depend on the legacy attribute scheme, which
    real runs leave empty. Same opinions, empty vs populated attributes -> same
    rewiring outcome."""
    legacy, opinions, _ = _fixture()
    empty = {i: {} for i in range(25)}
    populated = {i: {"age": 40, "pol_score": 3, "edu_level": 2} for i in range(25)}
    counts = []
    for attrs in (empty, populated):
        net = DirectedNetwork.from_undirected(legacy, 25, reciprocal=True)
        r = random.Random(3)
        adds = 0
        for step in range(60, 400):
            L = r.randrange(25)
            c = list(net.following[L])
            if not c:
                continue
            a, _ = evolve_once(net, L, r.choice(c), opinions, attrs, step,
                               burnin_steps=50, p_add=0.9, rng=r)
            adds += a
        counts.append(adds)
    assert counts[0] == counts[1], f"attributes changed the outcome: {counts}"
    assert counts[0] > 0, "opinion-only scorer produced no rewires at all"
    print("OK: default scorer is opinion-only; attributes do not affect it")


def test_score_fn_is_injectable():
    """A custom scorer is honoured, so the profile-aware makeover needs no change
    to the evolution logic. A scorer that returns 0 for everything blocks adds
    (nothing reaches add_score_threshold)."""
    legacy, opinions, attributes = _fixture()
    net = DirectedNetwork.from_undirected(legacy, 25, reciprocal=True)
    m0 = net.edge_count()
    r = random.Random(3)
    zero = lambda i, j, op, at=None: 0.0
    for step in range(60, 400):
        L = r.randrange(25)
        c = list(net.following[L])
        if not c:
            continue
        a, _ = evolve_once(net, L, r.choice(c), opinions, attributes, step,
                           burnin_steps=50, p_add=1.0, score_fn=zero, rng=r)
        assert a == 0, "add fired despite a zero scorer"
    assert net.edge_count() == m0
    print("OK: score_fn is injectable and controls whether edges form")


def main():
    test_reciprocal_reproduces_undirected()
    test_direction_separates_in_and_out()
    test_edges_never_dropped_by_index_order()
    test_mutual_flag()
    test_M_conserved_and_no_deaf_agents()
    test_rollback_when_no_cut_available()
    test_out_degree_guard_blocks_isolating_cut()
    test_opinion_only_ignores_attributes()
    test_score_fn_is_injectable()


if __name__ == "__main__":
    main()
