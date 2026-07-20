#!/usr/bin/env python3
"""ADR-006 Task 5.1: byte-identical baseline regression (the load-bearing safety test).

Runs the simulator OFFLINE (FAKE_LLM=1, PYTHONHASHSEED=0) with all four ADR-006
components at their DEFAULT (no-op) flags and asserts the trajectory CSVs are
byte-identical to a committed snapshot.

That snapshot was verified byte-identical to the pre-ADR-006 commit 86d22f2 at the
moment it was created (a worktree of 86d22f2 produced the exact same
opinion_change / interactions / *_edges CSVs under the same seed+FAKE_LLM+
PYTHONHASHSEED). See docs/research_log.md 2026-07-21 (Component 5.1). So this guard
catches ANY future silent behaviour change introduced by the scaffold or later work.

Requires the v119_default prompt assets + RAG corpus (tracked in-repo). Skips
gracefully if they are absent, so it never false-fails on a bare checkout.

Run:  python tests/test_adr006_baseline_regression.py   (needs no LLM / no display)
"""
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(ROOT, "scripts", "opinion_dynamics_test_network_qwen.py")
FIX = os.path.join(HERE, "fixtures", "adr006_baseline")

TRAJECTORY_CSVS = [
    "regr_probe_5_3_v119_no_opinion_change.csv",
    "regr_probe_5_3_v119_no_interactions.csv",
    "regr_probe_5_3_v119_no_step_000_edges.csv",
]


def _find_produced(basename):
    matches = glob.glob(os.path.join(ROOT, "results", "**", basename), recursive=True)
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0] if matches else None


def _read_norm(path):
    """Read bytes with line endings normalised to LF, so the comparison is about
    the data, not the CRLF/LF artifact (git normalises the committed fixture to LF
    while the Windows sim emits CRLF -- both collapse to the same content here)."""
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


def test_default_flags_match_pre_adr006_baseline():
    args = json.load(open(os.path.join(FIX, "cprime_args.json"), encoding="utf-8"))

    # Asset guards -> SKIP (not FAIL) when the prompt set / corpus is not present.
    tmpl = os.path.join(ROOT, "prompts", "opinion_dynamics", "Flache_2017",
                        "v119", "v119_default", "step2_produce_tweet_prev_none.md")
    corpus_rel = next((args[i + 1] for i, a in enumerate(args)
                       if a == "--rag_corpus_path"), None)
    corpus = os.path.join(ROOT, corpus_rel) if corpus_rel else None
    if not os.path.exists(tmpl) or (corpus and not os.path.exists(corpus)):
        print("SKIP: v119_default prompt assets / RAG corpus not present")
        return

    env = dict(os.environ)
    env["FAKE_LLM"] = "1"           # deterministic dummy LLM (no Ollama)
    env["PYTHONHASHSEED"] = "0"     # pin set iteration order for reproducibility

    r = subprocess.run([sys.executable, SIM] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, "sim run failed:\n" + (r.stderr or "")[-2000:]

    produced_dir = None
    try:
        for name in TRAJECTORY_CSVS:
            produced = _find_produced(name)
            assert produced is not None, f"sim did not produce {name}"
            produced_dir = os.path.dirname(produced)
            assert _read_norm(os.path.join(FIX, name)) == _read_norm(produced), (
                f"REGRESSION: {name} differs from the committed pre-ADR-006 baseline "
                f"-> a default-flags run is no longer byte-identical")
    finally:
        # results/ is gitignored; still, keep the tree tidy after the test.
        if produced_dir and os.path.isdir(produced_dir):
            shutil.rmtree(produced_dir, ignore_errors=True)

    print("OK: default-flags trajectory byte-identical to pre-ADR-006 baseline (86d22f2)")


if __name__ == "__main__":
    test_default_flags_match_pre_adr006_baseline()
