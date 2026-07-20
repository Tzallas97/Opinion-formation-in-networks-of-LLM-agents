#!/usr/bin/env python3
"""ADR-006 §Component 2: silence prompt block + parser guard tests.

Task 2.1 scope: verify that the three step2 templates carry an
`<!--ALLOW_SILENCE_BLOCK-->` marker at a stable insertion point. The marker
is used by the renderer (Task 2.2) to swap in a mechanism-agnostic silence
option when --allow_silence is on, and to strip cleanly when it is off.

Later tasks (2.2, 2.3, 2.4) extend this file with renderer, parser, log,
and mechanism-attribution tests.

Run:  python tests/test_silence_step2.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

TEMPLATE_ROOT = os.path.join(
    ROOT, "prompts", "opinion_dynamics", "Flache_2017", "template"
)

STEP2_TEMPLATES = [
    "step2_produce_tweet_prev_none.md",
    "step2_produce_tweet_prev_tweet.md",
    "step2_produce_tweet_prev_read.md",
]

MARKER = "<!--ALLOW_SILENCE_BLOCK-->"


def test_all_step2_templates_have_allow_silence_marker():
    """Every step2 template must contain the marker in exactly ONE spot so
    the renderer knows where to substitute."""
    for name in STEP2_TEMPLATES:
        path = os.path.join(TEMPLATE_ROOT, name)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert MARKER in text, f"marker missing in {name}"
        assert text.count(MARKER) == 1, \
            f"marker must appear exactly once in {name}; got {text.count(MARKER)}"


def test_marker_is_between_direction_and_format_blocks():
    """The insertion point matters: silence must read as an alternative to
    producing a directional tweet, not as an output-format instruction.
    Assert the marker sits AFTER 'Direction and strength' and BEFORE 'Format'."""
    for name in STEP2_TEMPLATES:
        with open(os.path.join(TEMPLATE_ROOT, name), encoding="utf-8") as f:
            text = f.read()
        i_dir = text.find("Direction and strength")
        i_marker = text.find(MARKER)
        i_fmt = text.find("Format")
        assert i_dir != -1, f"'Direction and strength' block not found in {name}"
        assert i_fmt != -1, f"'Format' block not found in {name}"
        assert i_dir < i_marker < i_fmt, \
            f"marker in wrong position in {name}: dir={i_dir}, marker={i_marker}, fmt={i_fmt}"


if __name__ == "__main__":
    test_all_step2_templates_have_allow_silence_marker()
    test_marker_is_between_direction_and_format_blocks()
    print("OK")
