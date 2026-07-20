#!/usr/bin/env python3
"""ADR-006 Component 4: Bian 5-dim diagnostic tests.

Runs fully offline with a fake chat function (no Ollama). The fake returns a
constant occupation, which lets us assert the *direction* of each probe:
identical outputs -> maximal homogeneity/persistence, neutral text -> zero
positivity, single-role outputs -> non-zero divergence from the ILO baseline.

Run:  python tests/test_bian_diagnostic.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

SCORE_KEYS = {
    "social_role_kl", "inter_agent_sim", "intra_agent_sim",
    "keyword_persistence", "positivity",
}


def test_bian_diagnostic_module_imports():
    from bian_diagnostic import (  # noqa: F401
        probe_social_roles, probe_inter_agent_sim, probe_intra_agent_sim,
        probe_keyword_persistence, probe_positivity, run_all_probes,
    )
    assert callable(probe_social_roles)


def test_bian_diagnostic_writes_scores_json():
    from bian_diagnostic import run_all_probes

    def fake_call(prompt, model=None):
        return "software engineer"

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "bian_scores.json")
        data = run_all_probes(model="fake", chat_fn=fake_call,
                              out_path=out_path, n_samples=4)
        on_disk = json.load(open(out_path, encoding="utf-8"))
    assert set(on_disk["scores"].keys()) == SCORE_KEYS
    assert on_disk["model"] == "fake"
    assert "timestamp" in on_disk
    # return value and file agree
    assert data["scores"] == on_disk["scores"]


def test_probe_directions_with_constant_output():
    """Constant 'software engineer' fixes the expected direction of each score."""
    from bian_diagnostic import run_all_probes

    def fake_call(prompt, model=None):
        return "software engineer"

    with tempfile.TemporaryDirectory() as tmp:
        s = run_all_probes(model="fake", chat_fn=fake_call,
                           out_path=os.path.join(tmp, "x.json"), n_samples=5)["scores"]
    # identical utterances -> max similarity (tf-idf idf-smoothing gives 1.0)
    assert s["inter_agent_sim"] > 0.99, s["inter_agent_sim"]
    assert s["intra_agent_sim"] > 0.99, s["intra_agent_sim"]
    # first keyword recurs in every later utterance
    assert s["keyword_persistence"] > 0.99, s["keyword_persistence"]
    # no sentiment words -> neutral
    assert abs(s["positivity"]) < 1e-9, s["positivity"]
    # everyone is a "professional" -> far from the ILO baseline (~0.11 there)
    assert s["social_role_kl"] > 0.0, s["social_role_kl"]
    # all scores are plain floats (JSON-serialisable)
    assert all(isinstance(v, float) for v in s.values())


def test_positivity_sign_tracks_sentiment():
    """A positive-heavy generator scores > 0; a negative-heavy one scores < 0."""
    from bian_diagnostic import probe_positivity

    pos = probe_positivity(lambda p, model=None: "This is a wonderful, safe success.", n=6)
    neg = probe_positivity(lambda p, model=None: "This is a terrible, dangerous failure.", n=6)
    assert pos > 0.0, pos
    assert neg < 0.0, neg


if __name__ == "__main__":
    test_bian_diagnostic_module_imports()
    test_bian_diagnostic_writes_scores_json()
    test_probe_directions_with_constant_output()
    test_positivity_sign_tracks_sentiment()
    print("OK")
