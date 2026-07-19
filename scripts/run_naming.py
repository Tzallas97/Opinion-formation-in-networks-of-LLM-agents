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
import io
import os
import re
import warnings

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


_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_token(text: str) -> str:
    """Reduce arbitrary text to a token that is safe as a file/folder name.

    Path separators and '..' cannot survive this, so a stray output name can
    never escape the results directory. Runs of unsafe characters collapse to a
    single underscore.
    """
    token = _UNSAFE_RE.sub("_", str(text or "").strip()).strip("._-")
    return token or "run"


def is_known_mode(version_set: str) -> bool:
    """True when the mode part of a version string is in the abbreviation table."""
    _prefix, mode = split_version(version_set)
    return mode in MODE_TO_ABBREV or mode in ABBREV_TO_MODE


def run_stem(out_base: str, num_agents, num_steps, version_set: str,
             strict: bool = False) -> str:
    """The new-scheme folder name and file stem for a run (no date/dist).

    ``strict=True`` is for the WRITER path (the simulator): an unrecognised mode
    is almost always a typo in --version_set, and a typo silently produces a run
    folder no reader can parse. Failing here costs nothing; failing after the
    run costs the run. Readers keep strict=False so that old or hand-made names
    still load.
    """
    mode_ok = is_known_mode(version_set)
    if not mode_ok:
        message = (
            f"unrecognised version_set {version_set!r}: the mode is not in "
            f"MODE_TO_ABBREV. Known modes: {sorted(MODE_TO_ABBREV)}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return (
        f"{safe_token(out_base)}_{int(num_agents)}_{int(num_steps)}"
        f"_{abbrev_version(version_set)}"
    )


# ------------------------------------------------------------------ siblings

# The old long scheme is not internally consistent about where the date sits:
#   <stem>_<infix>_opinion_change_<date>_<dist>.csv     (date AFTER the artifact)
#   <stem>_<date>_<infix>_interactions_<dist>.csv       (date BEFORE the infix)
# So a prefix taken by cutting an opinion_change filename at the artifact name
# still carries the network infix and no date, and will not prefix-match its own
# siblings. core_stem() strips both so the two forms meet.
_NET_INFIX_RE = re.compile(r"_(evolving_network|network)$", re.IGNORECASE)
_TRAILING_DATE_RE = re.compile(r"_\d{8}$")

ARTIFACT_NAMES = (
    "opinion_change", "interactions", "run_metrics", "step_summary",
    "step2_events", "agent_summary", "hub_metrics", "neighbor_summary",
    "agent_tweet_history", "agent_response_history",
)


def core_stem(filename: str) -> str:
    """The part of a run's filename that ALL its artifacts share.

    Cuts at the artifact name, then removes a trailing network infix and a
    trailing 8-digit date. Returns "" when no artifact name is recognised, which
    callers should read as "do not prefix-match".

    >>> core_stem("seed51_x_25_201_v113_default_evolving_network_opinion_change_20231212_uniform.csv")
    'seed51_x_25_201_v113_default'
    """
    base = os.path.basename(str(filename or ""))
    low = base.lower()
    cut = -1
    for name in ARTIFACT_NAMES:
        i = low.find(name)
        if i > 0 and (cut < 0 or i < cut):
            cut = i
    if cut < 0:
        return ""
    stem = base[:cut].rstrip("_-")
    prev = None
    while prev != stem:                     # infix and date can appear in either order
        prev = stem
        stem = _NET_INFIX_RE.sub("", stem)
        stem = _TRAILING_DATE_RE.sub("", stem)
    return stem


def sibling_patterns(filename: str, *artifact_globs: str):
    """Glob patterns for a run's sibling artifacts, most specific first.

    Tries the literal prefix (correct for the new scheme and for well-formed old
    runs), then the core stem (rescues old runs whose date/infix order differs),
    then the bare artifact pattern as a last resort for single-run folders.
    """
    base = os.path.basename(str(filename or ""))
    low = base.lower()
    cut = min((low.find(n) for n in ARTIFACT_NAMES if low.find(n) > 0), default=-1)
    literal = base[:cut].rstrip("_-") if cut > 0 else ""
    stem = core_stem(filename)
    out = []
    for pref in (literal, stem):
        if pref:
            for pat in artifact_globs:
                cand = pref + "*" + pat.lstrip("*")
                if cand not in out:
                    out.append(cand)
    for pat in artifact_globs:
        if pat not in out:
            out.append(pat)
    return out


def open_text(path):
    """Return a text handle for a run file that may not be valid UTF-8.

    Runs written on Windows occasionally carry cp1252 bytes (smart quotes) that
    abort a strict utf-8 read PART-WAY THROUGH the file, so validating only the
    first character is not enough - the whole payload has to decode. Decode the
    bytes once (utf-8, then cp1252, then utf-8 with replacement) and hand back a
    StringIO. A garbled quote is recoverable; a lost run is not.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    for encoding in ("utf-8", "cp1252"):
        try:
            return io.StringIO(data.decode(encoding), newline="")
        except (UnicodeDecodeError, LookupError):
            continue
    warnings.warn(
        f"{os.path.basename(path)}: not valid utf-8 or cp1252; decoding with "
        "replacement characters", RuntimeWarning, stacklevel=2)
    return io.StringIO(data.decode("utf-8", errors="replace"), newline="")
