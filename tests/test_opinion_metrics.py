#!/usr/bin/env python3
"""Regression tests for the opinion metrics.

Run:  python tests/test_opinion_metrics.py

These guard against a past bug: the repo once carried six independent
implementations of "P" split across two incompatible formulas, and the one used
by the simulator and the plots scored a centred population as maximally
polarized. Every consumer must now agree, and that pathology must stay dead.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import opinion_metrics as om  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Cases whose expected P is known by hand. The third entry is the one that
# killed the old formula: 16 agents at dead centre with a single mild dissenter
# on each side scored 1.000 (maximum polarization) under the old formula.
CASES = [
    ("perfect split at +-2", [2] * 10 + [-2] * 10, 1.0),
    ("perfect split at +-1", [1] * 10 + [-1] * 10, 1.0),
    ("centred, one dissenter each way", [0] * 16 + [1, -1], 4 * (1 / 18) * (1 / 18)),
    ("consensus at +2", [2] * 20, 0.0),
    ("consensus at 0", [0] * 20, 0.0),
    ("one-sided mild", [1] * 18 + [0] * 2, 0.0),
    ("80/20 split at +-2", [2] * 16 + [-2] * 4, 4 * 0.8 * 0.2),
    ("empty population", [], 0.0),
]


def test_known_values():
    for label, values, expected in CASES:
        got = om.polarization(values)
        assert abs(got - expected) < 1e-12, f"{label}: P={got!r}, expected {expected!r}"
    print("OK: P matches hand-computed values on %d populations" % len(CASES))


def test_pathology_stays_dead():
    """A centred population must not read as polarized.

    The deleted formula (var - B^2)/(var + B^2) returns exactly 1.0 for every
    zero-mean population. If someone reintroduces it, this fails.
    """
    centred = [0] * 16 + [1, -1]
    assert om.polarization(centred) < 0.05, "centred population scored as polarized"
    # ... while a real split still saturates
    assert om.polarization([2] * 10 + [-2] * 10) == 1.0
    # both have mean zero, so any mean-based index cannot tell them apart
    assert abs(om.bdp(centred)[0]) < 1e-12
    assert abs(om.bdp([2] * 10 + [-2] * 10)[0]) < 1e-12
    print("OK: zero-mean pathology stays dead (centred != split)")


def test_scalar_matches_vectorised():
    # bdp_series needs a rectangular matrix (rows = time steps), so use the
    # equal-length cases; ragged input is covered by its own test below.
    width = 20
    matrix = [values for _l, values, _e in CASES if len(values) == width]
    assert len(matrix) >= 5, "expected several width-%d cases" % width
    B, D, P = om.bdp_series(matrix)
    for i, row in enumerate(matrix):
        b, d, p = om.bdp(row)
        assert abs(b - float(B[i])) < 1e-12
        assert abs(d - float(D[i])) < 1e-12
        assert abs(p - float(P[i])) < 1e-12
    print("OK: bdp and bdp_series agree on every case")


def test_all_consumers_agree():
    """The simulator, the plots and the eval tools must return the same P.

    The point: one definition, four consumers, all in agreement.
    """
    import pandas as pd

    values = [2] * 7 + [-2] * 5 + [0] * 3
    expected = om.polarization(values)

    evalr = _load("eval_runs_probe", "tools/eval_runs.py")
    assert abs(evalr.bdp(values)[2] - expected) < 1e-12, "tools/eval_runs.py disagrees"

    plots = _load("main_plot_probe", "scripts/main_plot_network.py")
    cols = ["a%d" % i for i in range(len(values))]
    frame = pd.DataFrame([values], columns=cols)
    assert abs(plots.compute_B_D_P(frame, cols, 0)[3] - expected) < 1e-12, \
        "main_plot_network.compute_B_D_P disagrees"
    assert abs(plots._belief_stats_from_values(values)["P"] - expected) < 1e-12, \
        "main_plot_network._belief_stats_from_values disagrees"

    print("OK: eval_runs and both main_plot entry points agree with the shared module")


def test_ragged_input_is_readable():
    try:
        om.bdp_series([[1, 2], [3]])
    except ValueError as exc:
        assert "rectangular" in str(exc), exc
        print("OK: ragged matrix raises a readable error")
        return
    raise AssertionError("ragged matrix did not raise")


def main():
    test_known_values()
    test_pathology_stays_dead()
    test_scalar_matches_vectorised()
    test_all_consumers_agree()
    test_ragged_input_is_readable()


if __name__ == "__main__":
    main()
