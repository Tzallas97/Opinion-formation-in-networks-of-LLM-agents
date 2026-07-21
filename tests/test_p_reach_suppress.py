#!/usr/bin/env python3
"""ADR-006 roads_not_taken 3.11 fix gamma: skip-preserving shadowban regression.

Runs the sim OFFLINE (FAKE_LLM=1, PYTHONHASHSEED=0) as a MATCHED-BASELINE contrast:
  control   = shadowban, shadowban_value=1.0, enforcement=suppress  (nothing throttled)
  treatment = shadowban, shadowban_value=0.1, enforcement=suppress  (2/10 throttled)

Because `suppress` does NOT filter candidate speakers and draws its Bernoulli from a
DEDICATED rng, the two runs must share the EXACT interaction sequence (identical
listeners AND speakers under random selection); the sole difference is the suppressed
belief updates on throttled speakers. This guards the clean single-seed causal
isolation that fix gamma provides (contrast with the filter-mode RNG-path confound).

Skips gracefully if the v119_default prompt assets / RAG corpus are absent.
Run:  python tests/test_p_reach_suppress.py   (needs no LLM / no display)
"""
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(ROOT, "scripts", "opinion_dynamics_test_network_qwen.py")
FIX = os.path.join(HERE, "fixtures", "adr006_baseline")
AGENTS, STEPS = "10", "20"


def _override(args, flag, value):
    """Replace flag's value in place if present, else append -> unambiguous override."""
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def _run(base_args, out_name, sb_value):
    a = base_args
    for flag, val in (("-agents", AGENTS), ("-steps", STEPS), ("--out", out_name),
                      ("--interaction_selection", "random"),
                      ("--p_reach_policy", "shadowban"),
                      ("--p_reach_shadowban_fraction", "0.2"),
                      ("--p_reach_shadowban_value", sb_value),
                      ("--p_reach_enforcement", "suppress")):
        a = _override(a, flag, val)
    env = dict(os.environ)
    env["FAKE_LLM"] = "1"
    env["PYTHONHASHSEED"] = "0"
    r = subprocess.run([sys.executable, SIM] + a, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, "sim run failed:\n" + (r.stderr or "")[-2000:]
    dirs = glob.glob(os.path.join(ROOT, "results", "**",
                                  "%s_%s_%s_v119_no" % (out_name, AGENTS, STEPS)), recursive=True)
    assert dirs, "no run dir for %s" % out_name
    d = dirs[0]
    ic = glob.glob(os.path.join(d, "*interactions*.csv"))[0]
    listeners, speakers = [], []
    with open(ic, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        lcol = next(c for c in rd.fieldnames if "Agent_I" in c and "ID" in c)
        scol = next(c for c in rd.fieldnames if "Agent_J" in c and "ID" in c)
        for row in rd:
            listeners.append(row[lcol])
            speakers.append(row[scol])
    supp = 0
    for mj in glob.glob(os.path.join(d, "*metrics*.json")):
        m = re.search(r'"p_reach_suppressed_interactions"\s*:\s*(\d+)',
                      open(mj, encoding="utf-8", errors="ignore").read())
        if m:
            supp = int(m.group(1))
            break
    return d, listeners, speakers, supp


def test_suppress_clean_matched_contrast():
    tmpl = os.path.join(ROOT, "prompts", "opinion_dynamics", "Flache_2017",
                        "v119", "v119_default", "step2_produce_tweet_prev_none.md")
    base = json.load(open(os.path.join(FIX, "cprime_args.json"), encoding="utf-8"))
    corpus_rel = next((base[i + 1] for i, a in enumerate(base) if a == "--rag_corpus_path"), None)
    corpus = os.path.join(ROOT, corpus_rel) if corpus_rel else None
    if not os.path.exists(tmpl) or (corpus and not os.path.exists(corpus)):
        print("SKIP: v119_default prompt assets / RAG corpus not present")
        return

    dirs = []
    try:
        dc, lc, sc, suppc = _run(base, "suppress_ctl", "1.0")
        dirs.append(dc)
        dt, lt, st, suppt = _run(base, "suppress_trt", "0.1")
        dirs.append(dt)
        assert lc == lt, "listeners differ -> suppress perturbed the main RNG stream"
        assert sc == st, "speakers differ -> suppress perturbed the main RNG stream"
        assert suppc == 0, "matched baseline (value=1.0) must suppress nothing, got %d" % suppc
        assert suppt > 0, "treatment (value=0.1) must suppress >=1 throttled tweet, got 0"
    finally:
        for d in dirs:
            if d and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

    print("OK: suppress yields a clean matched contrast "
          "(identical listeners+speakers, %d suppressed update(s) in treatment)" % suppt)


if __name__ == "__main__":
    test_suppress_clean_matched_contrast()
