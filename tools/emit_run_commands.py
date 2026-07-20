#!/usr/bin/env python3
"""Generate complete, reproducible simulator commands from a compact spec.

Why this exists: a serious run must pin EVERY flag, because any flag left off
runs on a default the operator never saw, and two runs that differ in a hidden
default are not a clean comparison. Assembling ~70 flags by hand is itself
error-prone (it is easy to emit the output-file default and have every run
collide). This reads the simulator's argparse defaults directly, so the emitted
command is always complete and current, and it guarantees the fixed block is
byte-identical across the arms of an experiment.

Spec (JSON, or the built-in P1 default):
  {
    "name": "p1",
    "size": {"--num_agents": 10, "--num_steps": 100},
    "fixed": {"--version_set": "v119_confirmation_bias", "--network_type": "ba", ...},
    "vary":  {"--validation_strictness": ["strict", "warn_only"]},
    "seeds": [1, 2, 3, 4, 5]
  }

Usage:
  python tools/emit_run_commands.py [spec.json] [--ps1-dir DIR] [--config-dir DIR]

Every emitted run also gets a resolved-config JSON (every flag = its value) so
the exact configuration of a run is recorded, not reconstructed later.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMULATOR = os.path.join(os.path.dirname(HERE), "scripts",
                         "opinion_dynamics_test_network_qwen.py")


def parse_simulator_args(sim_path=SIMULATOR):
    """Return the simulator's argparse arguments by static analysis.

    Static (ast) rather than importing the simulator, which pulls heavy deps.
    Each entry: dict(canonical, aliases, default, store_true, has_default).
    'canonical' is the longest --option, so the descriptive name is used (e.g.
    --output_file, never the short --out that carries the colliding default).
    """
    tree = ast.parse(open(sim_path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"):
            continue
        opts = [a.value for a in node.args
                if isinstance(a, ast.Constant) and str(a.value).startswith("-")]
        longs = [o for o in opts if o.startswith("--")]
        if not longs:
            continue
        kw = {}
        for k in node.keywords:
            if isinstance(k.value, ast.Constant):
                kw[k.arg] = k.value.value
            elif isinstance(k.value, ast.List):
                kw[k.arg] = [e.value for e in k.value.elts if isinstance(e, ast.Constant)]
        # Prefer the dest-derived name (what the code calls the arg internally,
        # e.g. --world over the longer alias --world_mode), falling back to the
        # longest long option. Both are valid; this just matches the familiar name.
        dest = kw.get("dest") or longs[0].lstrip("-").replace("-", "_")
        canonical = ("--" + dest) if ("--" + dest) in longs else max(longs, key=len)
        out.append({
            "canonical": canonical,
            "aliases": opts + (["--" + kw["dest"]] if "dest" in kw else []),
            "default": kw.get("default"),
            "store_true": kw.get("action") == "store_true",
            "has_default": "default" in kw and kw["default"] is not None,
        })
    return out


def _alias_map(specs):
    """Map every alias/dest of an argument to its canonical --option."""
    m = {}
    for a in specs:
        for alias in a["aliases"] + [a["canonical"]]:
            m[alias] = a["canonical"]
    return m


def resolve_run(specs, overrides):
    """Full {canonical_flag: value} for one run: defaults, then overrides.

    store_true flags are included only when an override sets them truthy;
    None-default flags only when overridden. Everything else is pinned.
    """
    amap = _alias_map(specs)
    norm = {}
    for k, v in overrides.items():
        key = amap.get(k, k if k.startswith("--") else "--" + k)
        norm[key] = v
    resolved = {}
    for a in specs:
        c = a["canonical"]
        if c in norm:
            resolved[c] = norm[c]
        elif a["store_true"]:
            continue
        elif a["has_default"]:
            resolved[c] = a["default"]
    # any override the parser did not declare (typo guard)
    unknown = [k for k in norm if k not in {a["canonical"] for a in specs}]
    if unknown:
        raise KeyError(f"unknown flag(s) in spec: {unknown}")
    return resolved


def _quote(v):
    s = str(v)
    return f'"{s}"' if (s == "" or " " in s or '"' in s) else s


def command_line(resolved, python="python", script="scripts\\opinion_dynamics_test_network_qwen.py"):
    parts = [python, script]
    for flag in sorted(resolved):          # sorted -> stable, comparable across runs
        val = resolved[flag]
        # Empty-string values are the argparse default ("" = the unset state), and
        # an empty quoted token does not survive PowerShell's re-parsing (the "" is
        # dropped and argparse then reports "expected one argument"). Omitting the
        # flag yields the identical value, so skip it in the command. The resolved
        # config JSON still records it, so provenance is unaffected.
        if str(val) == "":
            continue
        parts.append(f"{flag} {_quote(val)}")
    return " ".join(parts)


def expand_spec(spec, specs):
    """Yield (run_name, resolved_values) for every vary-combo x seed."""
    name = spec.get("name", "run")
    base = {}
    base.update(spec.get("size", {}))
    base.update(spec.get("fixed", {}))
    vary = spec.get("vary", {}) or {"__none__": [None]}
    seeds = spec.get("seeds", [1])
    vary_keys = [k for k in vary if k != "__none__"]
    combos = list(itertools.product(*[vary[k] for k in vary_keys])) if vary_keys else [()]
    for combo in combos:
        tag = "_".join(str(v) for v in combo) if combo else ""
        for seed in seeds:
            run_name = "_".join(x for x in [name, tag, f"s{seed}"] if x)
            over = dict(base)
            for k, v in zip(vary_keys, combo):
                over[k] = v
            over["--seed"] = seed
            over["--output_file"] = f"{run_name}.csv"
            yield run_name, resolve_run(specs, over)


DEFAULT_SPEC = {
    "name": "p1",
    "size": {"--num_agents": 10, "--num_steps": 100},
    "fixed": {
        "--version_set": "v119_confirmation_bias",
        "--network_type": "ba", "--ba_m_attach": 3,
        "--world": "closed", "--rag_backend": "off",
        "--model_name": "qwen3:8b",
    },
    "vary": {"--validation_strictness": ["strict", "warn_only"]},
    "seeds": [1, 2, 3, 4, 5],
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?", help="spec JSON path (default: built-in P1 spec)")
    ap.add_argument("--ps1-dir", default=None, help="also write one .ps1 per run here")
    ap.add_argument("--config-dir", default=None, help="also write the resolved config JSON per run here")
    a = ap.parse_args(argv)

    spec = json.load(open(a.spec, encoding="utf-8")) if a.spec else DEFAULT_SPEC
    specs = parse_simulator_args()

    runs = list(expand_spec(spec, specs))
    for run_name, resolved in runs:
        line = command_line(resolved)
        print(line)
        if a.ps1_dir:
            os.makedirs(a.ps1_dir, exist_ok=True)
            open(os.path.join(a.ps1_dir, run_name + ".ps1"), "w",
                 encoding="utf-8", newline="\r\n").write(line + "\r\n")
        if a.config_dir:
            os.makedirs(a.config_dir, exist_ok=True)
            json.dump(resolved, open(os.path.join(a.config_dir, run_name + ".json"),
                      "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(f"\n# {len(runs)} runs. Fixed block is identical across arms by construction.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
