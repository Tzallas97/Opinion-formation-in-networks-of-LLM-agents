#!/usr/bin/env python3
"""Canonical run naming: one place that knows the mode abbreviations and how a
run's folder / file stem is built, shared by the writer (the simulator and the
plotter) and every reader (eval, judge, plotting, launchers).

Two naming schemes coexist and BOTH must always be readable:

  new (short, this is what fresh runs write):
      <out>_<agents>_<steps>_v<ver>_<mode-abbrev>
      files: <stem>_opinion_change.csv, <stem>_interactions.csv, ...
      no date, no distribution suffix, no "network_" infix.

  old (long, thesis/earlier runs on disk):
      <out>_<agents>_<steps>_v<ver>_<mode-full>_<date>_<dist>
      files: <stem>_network_opinion_change_<date>_<dist>.csv, ...

The mode abbreviations are taken verbatim from the thesis runs already on disk
(no, no_r, weak_r, strong_r); the rest follow the same pattern.
"""
from __future__ import annotations
import re

# full prompt mode  ->  short abbreviation used in file/folder names
MODE_TO_ABBREV = {
    "default": "no",
    "default_reverse": "no_r",
    "confirmation_bias": "weak",
    "confirmation_bias_reverse": "weak_r",
    "strong_confirmation_bias": "strong",
    "strong_confirmation_bias_reverse": "strong_r",
    "control": "control",
    "control_reverse": "control_r",
    "llm_check_true": "check_t",
    "llm_check_false": "check_f",
}
ABBREV_TO_MODE = {v: k for k, v in MODE_TO_ABBREV.items()}

# artifact-suffix aliases: new short name -> the substring that also appears in
# the old long file names. Used so a reader can find either form by globbing.
ARTIFACT_ALIASES = {
    "opinion_change": ["opinion_change", "network_opinion_change"],
    "interactions": ["interactions", "network_interactions"],
    "run_metrics": ["run_metrics", "network_run_metrics"],
    "step_summary": ["step_summary", "network_step_summary"],
    "step2_events": ["step2_events", "network_step2_events"],
    "agent_summary": ["agent_summary", "network_agent_summary"],
    "hub_metrics": ["hub_metrics", "network_hub_metrics"],
}

_VER_RE = re.compile(r"^(v\d+)_(.*)$", re.IGNORECASE)


def split_version(version_set: str) -> tuple[str, str]:
    """'v52_strong_confirmation_bias_reverse' -> ('v52', 'strong_confirmation_bias_reverse').
    If there is no v-prefix, the whole string is treated as the mode."""
    m = _VER_RE.match(str(version_set or "").strip())
    if not m:
        return "", str(version_set or "").strip()
    return m.group(1), m.group(2)


def abbrev_mode(mode: str) -> str:
    """Full (or already-short) mode -> short abbreviation. Unknown modes pass through."""
    mode = str(mode or "").strip()
    if mode in MODE_TO_ABBREV:
        return MODE_TO_ABBREV[mode]
    if mode in ABBREV_TO_MODE:      # already short
        return mode
    return mode


def full_mode(mode: str) -> str:
    """Short (or already-full) mode -> full canonical mode. Unknown modes pass through."""
    mode = str(mode or "").strip()
    if mode in ABBREV_TO_MODE:
        return ABBREV_TO_MODE[mode]
    if mode in MODE_TO_ABBREV:      # already full
        return mode
    return mode


def abbrev_version(version_set: str) -> str:
    """'v52_strong_confirmation_bias_reverse' -> 'v52_strong_r' (idempotent)."""
    prefix, mode = split_version(version_set)
    ab = abbrev_mode(mode)
    return f"{prefix}_{ab}" if prefix else ab


def canonical_version(version_set: str) -> str:
    """Any form -> the full version string, so short and long runs compare equal.
    'v52_strong_r' and 'v52_strong_confirmation_bias_reverse' both -> the latter."""
    prefix, mode = split_version(version_set)
    fm = full_mode(mode)
    return f"{prefix}_{fm}" if prefix else fm


def run_stem(out_base: str, num_agents, num_steps, version_set: str) -> str:
    """The new-scheme folder name and file stem for a run (no date/dist)."""
    return f"{out_base}_{int(num_agents)}_{int(num_steps)}_{abbrev_version(version_set)}"
