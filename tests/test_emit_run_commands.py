#!/usr/bin/env python3
"""Tests for the run-command generator.

The load-bearing guarantee: across the arms of an experiment, every flag is
identical except the declared variable, the seed, and the output name. If that
breaks, the comparison is confounded - exactly what the generator exists to
prevent.

Run:  python tests/test_emit_run_commands.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import emit_run_commands as G  # noqa: E402


def _flags(cmd):
    """Parse a command line into {flag: value}."""
    toks = cmd.split(" ")
    out = {}
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            val = toks[i + 1] if i + 1 < len(toks) else ""
            out[toks[i]] = val
            i += 2
        else:
            i += 1
    return out


def test_output_file_is_set_not_defaulted():
    """The bug that started this: a hand-built command emitted the output-file
    default and every run collided. Every run must have its own output name."""
    specs = G.parse_simulator_args()
    runs = list(G.expand_spec(G.DEFAULT_SPEC, specs))
    names = [_flags(G.command_line(r))["--output_file"] for _, r in runs]
    assert len(names) == len(set(names)), f"output names collide: {names}"
    assert all(n != "test1.csv" for n in names), "a run kept the output default"
    assert names[0] == "p1_strict_s1.csv", names[0]
    print("OK: every run has a unique, intended --output_file (no default collision)")


def test_fixed_block_identical_across_arms():
    specs = G.parse_simulator_args()
    runs = {name: G.command_line(r) for name, r in G.expand_spec(G.DEFAULT_SPEC, specs)}
    # compare the two seed-1 arms; they may differ ONLY in the declared vary
    # flag, the seed, and the output file.
    a = _flags(runs["p1_strict_s1"])
    b = _flags(runs["p1_warn_only_s1"])
    allowed = {"--validation_strictness", "--seed", "--output_file"}
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert diff <= allowed, f"arms differ in more than the variable: {diff - allowed}"
    assert a["--validation_strictness"] == "strict"
    assert b["--validation_strictness"] == "warn_only"
    # at the same seed the seed matches; output_file differs because it encodes
    # the arm, and the declared variable differs. Nothing else may.
    assert diff == {"--validation_strictness", "--output_file"}, \
        f"arms differ in more than variable+output at a fixed seed: {diff}"
    print("OK: the two arms differ ONLY in the declared variable (+ the output name)")


def test_completeness_every_value_flag_present():
    specs = G.parse_simulator_args()
    _, resolved = next(iter(G.expand_spec(G.DEFAULT_SPEC, specs)))
    cmd = _flags(G.command_line(resolved))
    # empty-string defaults are intentionally omitted (see the empty-string test);
    # every OTHER value flag must be pinned.
    missing = [a["canonical"] for a in specs
               if a["has_default"] and not a["store_true"]
               and a["default"] != "" and a["canonical"] not in cmd]
    assert not missing, f"value flags left on hidden defaults: {missing}"
    print("OK: every non-empty value flag is pinned - nothing runs on an unseen default")


def test_alias_override_resolves_to_canonical():
    """A spec that uses a short alias (or dest) must still land on the canonical
    flag, so `--out` cannot silently coexist with `--output_file`."""
    specs = G.parse_simulator_args()
    resolved = G.resolve_run(specs, {"--out": "x.csv"})
    assert resolved.get("--output_file") == "x.csv", resolved.get("--output_file")
    assert "--out" not in resolved, "alias leaked alongside the canonical flag"
    print("OK: aliases resolve to the canonical flag (no --out/--output_file split)")


def test_empty_string_flags_are_not_emitted():
    """Empty-string values (their argparse default) must not appear in the command:
    an empty quoted token is dropped by PowerShell re-parsing and argparse then
    fails with 'expected one argument'. They are still in the resolved config."""
    specs = G.parse_simulator_args()
    _, resolved = next(iter(G.expand_spec(G.DEFAULT_SPEC, specs)))
    cmd = G.command_line(resolved)
    for flag in ("--ba_hub_custom", "--rag_corpus_path", "--rag_topic_override", "--event_text"):
        assert f"{flag} " not in cmd + " ", f"{flag} emitted with empty value"
        assert flag in resolved, f"{flag} missing from resolved config"
    assert '""' not in cmd, "an empty quoted token is in the command"
    print("OK: empty-string flags are omitted from the command (kept in the config)")


def test_unknown_flag_raises():
    specs = G.parse_simulator_args()
    try:
        G.resolve_run(specs, {"--not_a_real_flag": 1})
    except KeyError as e:
        assert "not_a_real_flag" in str(e)
        print("OK: an unknown flag in a spec is caught, not silently emitted")
        return
    raise AssertionError("typo flag was not caught")


def main():
    test_output_file_is_set_not_defaulted()
    test_fixed_block_identical_across_arms()
    test_completeness_every_value_flag_present()
    test_alias_override_resolves_to_canonical()
    test_unknown_flag_raises()
    test_empty_string_flags_are_not_emitted()


if __name__ == "__main__":
    main()
