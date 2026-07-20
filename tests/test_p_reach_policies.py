#!/usr/bin/env python3
"""ADR-006 Component 3: p_reach pluggable policy pattern.

Task 3.1 scope: pure ``assign_p_reach`` function with three policies
(uniform / homophilic / shadowban) + ``p_reach`` attribute + ``get_p_reach``
lookup on ``DirectedNetwork``. Task 3.2 wires this into the main sim's
listener sampling; Task 3.3 exports it in the edge CSV and updates docs.

Run:  python tests/test_p_reach_policies.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from network_models import (  # noqa: E402
    DirectedNetwork,
    assign_p_reach,
    build_barabasi_albert,
)


# ---------------------------------------------------------------- policy fn


def test_uniform_returns_value_from_params():
    v = assign_p_reach("uniform", {"uniform_value": 0.5}, 1, 2, {1: 0, 2: 0})
    assert v == 0.5


def test_uniform_defaults_to_one():
    v = assign_p_reach("uniform", {}, 1, 2, {1: 0, 2: 0})
    assert v == 1.0


def test_homophilic_high_for_identical_opinions():
    # Same opinion -> distance 0 -> sigmoid(k * 1) approaches 1 for k>0.
    v = assign_p_reach("homophilic", {"homophily_k": 2.0}, 1, 2, {1: 2, 2: 2})
    assert v > 0.5, f"expected p_reach > 0.5 for identical opinions, got {v}"
    assert 0.0 <= v <= 1.0


def test_homophilic_low_for_opposite_extremes():
    # Distance = 4 on the -2..+2 scale -> normalized 1.0 -> sigmoid(k*-1) < 0.5.
    v = assign_p_reach("homophilic", {"homophily_k": 2.0}, 1, 2, {1: -2, 2: 2})
    assert v < 0.5, f"expected p_reach < 0.5 for maximally dissimilar opinions, got {v}"
    assert 0.0 <= v <= 1.0


def test_homophilic_higher_k_sharpens_curve():
    # With higher k the same distance should push further from 0.5.
    low_k = assign_p_reach("homophilic", {"homophily_k": 1.0}, 1, 2, {1: -2, 2: 2})
    high_k = assign_p_reach("homophilic", {"homophily_k": 5.0}, 1, 2, {1: -2, 2: 2})
    assert high_k < low_k, f"higher k should push further from 0.5; got low_k={low_k}, high_k={high_k}"


def test_shadowban_reduces_output_only_for_banned_sources():
    params = {"shadowban_agents": {1}, "shadowban_value": 0.1}
    # Banned agent's outgoing edges get the reduced value.
    assert assign_p_reach("shadowban", params, 1, 2, {1: 0, 2: 0}) == 0.1
    # Non-banned agent's edges are unaffected.
    assert assign_p_reach("shadowban", params, 2, 3, {2: 0, 3: 0}) == 1.0
    # Ban applies to source only, not destination.
    assert assign_p_reach("shadowban", params, 3, 1, {1: 0, 3: 0}) == 1.0


def test_shadowban_defaults_when_no_agents_listed():
    # Empty ban set -> everyone gets 1.0 by default.
    v = assign_p_reach("shadowban", {}, 1, 2, {1: 0, 2: 0})
    assert v == 1.0


def test_unknown_policy_raises_value_error():
    try:
        assign_p_reach("popularity_weighted_TODO", {}, 1, 2, {1: 0, 2: 0})
    except ValueError as e:
        # Message should name the bad policy so debugging is fast.
        assert "popularity_weighted_TODO" in str(e), \
            f"error message should name the bad policy; got: {e}"
        return
    raise AssertionError("expected ValueError for unknown policy")


# ---------------------------------------------------------------- attribute


def test_directed_network_has_p_reach_attribute():
    net = DirectedNetwork(num_agents=5, reciprocal=True)
    assert hasattr(net, "p_reach"), "DirectedNetwork must expose a p_reach attribute"
    assert isinstance(net.p_reach, dict), "p_reach should be a dict"
    assert net.p_reach == {}, "p_reach should start empty; populated by sim at edge init"


def test_get_p_reach_returns_one_for_missing_edge():
    """The default 1.0 for a missing (src,dst) is the load-bearing baseline:
    it means every existing consumer that doesn't set p_reach gets pre-ADR-006
    behaviour by construction."""
    net = DirectedNetwork(num_agents=5, reciprocal=True)
    assert net.get_p_reach(0, 1) == 1.0
    assert net.get_p_reach(2, 3) == 1.0


def test_get_p_reach_returns_stored_value_for_present_edge():
    net = DirectedNetwork(num_agents=5, reciprocal=True)
    net.p_reach[(1, 2)] = 0.3
    assert net.get_p_reach(1, 2) == 0.3
    # A different edge still returns the default.
    assert net.get_p_reach(2, 1) == 1.0


def test_reciprocal_baseline_still_reproduces_undirected_after_p_reach_added():
    """Load-bearing regression: adding p_reach must not disturb ADR-004's
    core invariant that reciprocal=True reproduces the undirected map exactly.
    If this fails, ADR-006 §Component 3 broke ADR-004 §load-bearing test."""
    legacy = build_barabasi_albert(20, 2, seed=7)
    legacy = legacy[0] if isinstance(legacy, tuple) else legacy
    net = DirectedNetwork.from_undirected(legacy, 20, reciprocal=True)
    for i in range(20):
        assert net.following[i] == set(legacy.get(i, set())), \
            f"following[{i}] diverged from undirected map"


if __name__ == "__main__":
    # policy function tests
    test_uniform_returns_value_from_params()
    test_uniform_defaults_to_one()
    test_homophilic_high_for_identical_opinions()
    test_homophilic_low_for_opposite_extremes()
    test_homophilic_higher_k_sharpens_curve()
    test_shadowban_reduces_output_only_for_banned_sources()
    test_shadowban_defaults_when_no_agents_listed()
    test_unknown_policy_raises_value_error()
    # attribute + baseline tests
    test_directed_network_has_p_reach_attribute()
    test_get_p_reach_returns_one_for_missing_edge()
    test_get_p_reach_returns_stored_value_for_present_edge()
    test_reciprocal_baseline_still_reproduces_undirected_after_p_reach_added()
    print("OK")
