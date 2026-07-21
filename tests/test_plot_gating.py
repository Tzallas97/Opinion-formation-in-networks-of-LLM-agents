#!/usr/bin/env python3
"""ADR-006 plotting: applicability-gating + csv/ routing (main_plot_network).

Locks the two load-bearing pieces of the gated-plotting refactor without needing
Ollama or a full render:
- `_build_run_context` decides NETWORK vs SOLO/no-network from the CSVs a run
  produced -> drives which plot families are attempted (a solo_check run skips
  network/interaction/persona plots by design).
- `_out_path` routes derived CSVs into a csv/ subfolder while images stay at the
  plot-folder top level (declutter).

Run:  python tests/test_plot_gating.py
"""
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
except Exception:
    pass
import main_plot_network as M  # noqa: E402


def _mk_run(dirpath, files):
    os.makedirs(dirpath, exist_ok=True)
    for name in files:
        with open(os.path.join(dirpath, name), "w", encoding="utf-8") as f:
            f.write("time_step,A\n0,0\n")
    oc = [f for f in files if "opinion_change" in f][0]
    return os.path.join(dirpath, oc)


def test_run_context_network_vs_solo():
    with tempfile.TemporaryDirectory() as tmp:
        net = _mk_run(os.path.join(tmp, "net"), [
            "run_opinion_change.csv", "run_interactions.csv",
            "run_step_000_edges.csv", "run_step_summary.csv", "run_agent_summary.csv"])
        ctx = M._build_run_context(types.SimpleNamespace(), net)
        assert ctx["is_network"] and ctx["has_edges"] and ctx["has_step_summary"] \
            and ctx["has_agent_summary"], ctx

        solo = _mk_run(os.path.join(tmp, "solo"), ["solo_opinion_change.csv"])
        ctx2 = M._build_run_context(types.SimpleNamespace(), solo)
        assert not ctx2["is_network"] and not ctx2["has_edges"] \
            and not ctx2["has_interactions"] and not ctx2["has_step_summary"], ctx2
    print("OK: run-context detects NETWORK vs SOLO (drives which plot families run)")


def test_out_path_routes_csv_to_subfolder():
    with tempfile.TemporaryDirectory() as tmp:
        csv = M._out_path(tmp, "x_BDP_timeseries.csv")
        png = M._out_path(tmp, "x_final_distribution.png")
        gif = M._out_path(tmp, "x_distribution.gif")
        assert os.path.basename(os.path.dirname(csv)) == "csv", csv
        assert os.path.dirname(png) == tmp and os.path.dirname(gif) == tmp, (png, gif)
        assert os.path.isdir(os.path.join(tmp, "csv")), "csv/ subdir was not created"
    print("OK: derived CSVs -> csv/ subfolder; images stay top-level")


if __name__ == "__main__":
    test_run_context_network_vs_solo()
    test_out_path_routes_csv_to_subfolder()
    print("OK")
