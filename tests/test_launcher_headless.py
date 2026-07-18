#!/usr/bin/env python3
"""Headless smoke test for the Tkinter GUIs: instantiate them against the
tkinter stub in tests/tkstub and exercise their core methods.

Run:  python tests/test_launcher_headless.py     (no display needed)

Stub semantics that make this meaningful:
- Private attrs and app-state names (*_var, *_combo, ...) raise AttributeError
  until assigned, like real instances - so hasattr()-guarded lazy init and
  getattr(self, name, default) behave as in production. A catch-all stub would
  make hasattr() always true and silently skip both patterns.
- Therefore: only instantiation-without-exception and 'name' in Class.__dict__
  are meaningful checks; hasattr(instance, name) is NOT.
"""
import os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tkstub"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))


def test_main_launcher():
    import importlib
    ui = importlib.import_module("ui_lancher11")
    app = ui.LauncherUI()
    settings = app._collect_settings()
    defaults = set(ui.DEFAULT_SETTINGS)
    collected = set(settings)
    missing = defaults - collected
    extra = collected - defaults
    assert not missing, f"settings in DEFAULT_SETTINGS but never saved: {sorted(missing)}"
    assert not extra, f"settings saved but never loaded back (not in DEFAULT_SETTINGS): {sorted(extra)}"
    print(f"OK: LauncherUI instantiated, {len(settings)} settings keys, defaults<->collect symmetric")


def test_plot_launcher():
    import importlib
    pl = importlib.import_module("plot_launcher")
    # parser is pure (no Tk): check it recovers args from a canonical run name
    name = ("results/opinion_dynamics/Flache_2017/qwen3_8b/run/"
            "seed68_x_20_100_v119_strong_confirmation_bias_reverse"
            "_network_opinion_change_20231212_uniform.csv")
    info = pl.parse_run_csv(name)
    assert info is not None, "parser failed on a canonical run name"
    assert info["agents"] == 20 and info["steps"] == 100 and info["seed"] == 68
    assert info["version"] == "v119_strong_confirmation_bias_reverse"
    assert info["dist"] == "uniform" and info["model"] == "qwen3_8b"
    app = pl.PlotLauncher()
    cmd = app._cmd_for(info)
    assert "-agents" in cmd and "v119_strong_confirmation_bias_reverse" in cmd
    print("OK: PlotLauncher instantiated, run-name parser + command builder correct")


def test_run_naming():
    import importlib
    N = importlib.import_module("run_naming")
    # abbreviations taken from the thesis runs on disk
    assert N.abbrev_version("v52_strong_confirmation_bias_reverse") == "v52_strong_r"
    assert N.abbrev_version("v119_confirmation_bias_reverse") == "v119_weak_r"
    assert N.abbrev_version("v130_default_reverse") == "v130_no_r"
    # canonical makes the short and long forms compare equal (backward compat)
    assert N.canonical_version("v52_strong_r") == N.canonical_version("v52_strong_confirmation_bias_reverse")
    # abbreviation is idempotent, canonicalisation is its inverse
    assert N.abbrev_version(N.abbrev_version("v52_strong_confirmation_bias_reverse")) == "v52_strong_r"
    assert N.run_stem("seed66_tscw31supragbaneghub", 30, 501, "v52_strong_confirmation_bias_reverse") \
        == "seed66_tscw31supragbaneghub_30_501_v52_strong_r"
    print("OK: run_naming abbreviations match thesis scheme, round-trips hold")


def test_compare_runs():
    import importlib
    C = importlib.import_module("compare_runs")
    # auto-labels must distinguish two runs of the same topic but different config
    a = C._auto_label("r/seed68_cfgA_20_100_v119_strong_confirmation_bias_reverse_20231212_uniform/x_opinion_change.csv")
    b = C._auto_label("r/seed68_cfgBstrict_20_100_v119_strong_confirmation_bias_reverse_20231212_uniform/x_opinion_change.csv")
    assert a != b, f"comparison labels collapsed: {a!r} == {b!r}"
    assert a.endswith("v119_strong_r"), a
    print("OK: compare_runs auto-labels are distinct and use the short scheme")


def main():
    test_run_naming()
    test_compare_runs()
    test_main_launcher()
    test_plot_launcher()


if __name__ == "__main__":
    main()
