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


# -----------------------------------------------------------------------------
# Task 2.2: renderer + parser tests
# -----------------------------------------------------------------------------


def _read_template(name):
    with open(os.path.join(TEMPLATE_ROOT, name), encoding="utf-8") as f:
        return f.read()


def _template_without_marker_manually(name):
    """Reference: what the template looks like when the marker + its
    surrounding blank lines are stripped by hand. This is the byte-identical
    baseline the renderer must reproduce with allow_silence=False."""
    raw = _read_template(name)
    return raw.replace(f"\n\n{MARKER}\n\n", "\n\n")


def test_render_off_is_byte_identical_to_manual_strip():
    from step2_io import render_step2_template
    for name in STEP2_TEMPLATES:
        raw = _read_template(name)
        rendered = render_step2_template(raw, allow_silence=False)
        expected = _template_without_marker_manually(name)
        assert rendered == expected, \
            f"allow_silence=False rendering diverges from manual strip in {name}"


def test_render_off_removes_marker_entirely():
    from step2_io import render_step2_template
    for name in STEP2_TEMPLATES:
        raw = _read_template(name)
        rendered = render_step2_template(raw, allow_silence=False)
        assert MARKER not in rendered, f"marker leaked in {name}"


def test_render_on_substitutes_silence_block_text():
    from step2_io import render_step2_template
    for name in STEP2_TEMPLATES:
        raw = _read_template(name)
        rendered = render_step2_template(raw, allow_silence=True)
        assert MARKER not in rendered, f"marker not consumed in {name}"
        assert "<silent>" in rendered.lower(), \
            f"expected '<silent>' token instruction in rendered prompt for {name}"


def test_render_on_is_mechanism_agnostic():
    """The silence block must NOT steer the agent toward any specific reason
    to be silent (extremity, topic, majority, etc). That would pre-decide the
    mechanism the paper is trying to disentangle."""
    from step2_io import render_step2_template
    raw = _read_template("step2_produce_tweet_prev_none.md")
    rendered = render_step2_template(raw, allow_silence=True).lower()
    forbidden = [
        "if you disagree",
        "if you feel isolated",
        "if you are in the minority",
        "if you are a minority",
        "if others disagree",
        "unpopular",
        "your view is uncommon",
    ]
    for phrase in forbidden:
        assert phrase not in rendered, \
            f"prompt contains mechanism-steering language: {phrase!r}"


# -----------------------------------------------------------------------------
# Parser tests: Noels 2025 (arXiv:2504.03803) 4-way taxonomy + silent + tweet
# -----------------------------------------------------------------------------


def test_parser_recognises_explicit_silent_token():
    from step2_io import parse_step2_output
    kind, payload = parse_step2_output("<silent>")
    assert kind == "silent"
    assert payload == ""


def test_parser_recognises_silent_with_surrounding_whitespace():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("   <silent>\n\n")
    assert kind == "silent"


def test_parser_recognises_silent_case_insensitive():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("<SILENT>")
    assert kind == "silent"


def test_parser_recognises_normal_tweet_output():
    from step2_io import parse_step2_output
    raw = "FINAL_RATING: 1\nTWEET: The Moon landing has strong evidence."
    kind, payload = parse_step2_output(raw)
    assert kind == "tweet"
    assert "Moon" in payload


def test_parser_recognises_error_refusal():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("[safety filter violated]")
    assert kind == "refusal_error"


def test_parser_recognises_content_filter_error():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("[content_filter] triggered")
    assert kind == "refusal_error"


def test_parser_recognises_generated_refusal_as_an_ai():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("As an AI, I cannot express political opinions on this topic.")
    assert kind == "refusal_generated"


def test_parser_recognises_generated_refusal_language_model():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("As a language model, I don't have personal views.")
    assert kind == "refusal_generated"


def test_parser_recognises_canned_refusal_many_sites():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("There are many sites on the Internet with information on such topics.")
    assert kind == "refusal_canned"


def test_parser_recognises_canned_refusal_no_info():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("I don't have information on this specific claim.")
    assert kind == "refusal_canned"


def test_parser_empty_output_is_parse_fail():
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("")
    assert kind == "parse_fail"
    kind, _ = parse_step2_output("   \n   ")
    assert kind == "parse_fail"


def test_parser_ambiguous_content_is_parse_fail():
    """Garbage that doesn't match any pattern must be parse_fail, not tweet."""
    from step2_io import parse_step2_output
    kind, _ = parse_step2_output("random noise without markers")
    assert kind == "parse_fail"


def test_parser_normal_tweet_starting_with_as_an_engineer_is_still_tweet():
    """Guard: refusal patterns must NOT fire on legitimate tweets that happen
    to start with 'As an <role>'. Priority: tweet-format markers win over
    self-referential prefix."""
    from step2_io import parse_step2_output
    raw = "FINAL_RATING: 1\nTWEET: As an engineer, I find the moon landing evidence convincing."
    kind, _ = parse_step2_output(raw)
    assert kind == "tweet"


if __name__ == "__main__":
    # Task 2.1 tests
    test_all_step2_templates_have_allow_silence_marker()
    test_marker_is_between_direction_and_format_blocks()
    # Task 2.2 renderer tests
    test_render_off_is_byte_identical_to_manual_strip()
    test_render_off_removes_marker_entirely()
    test_render_on_substitutes_silence_block_text()
    test_render_on_is_mechanism_agnostic()
    # Task 2.2 parser tests
    test_parser_recognises_explicit_silent_token()
    test_parser_recognises_silent_with_surrounding_whitespace()
    test_parser_recognises_silent_case_insensitive()
    test_parser_recognises_normal_tweet_output()
    test_parser_recognises_error_refusal()
    test_parser_recognises_content_filter_error()
    test_parser_recognises_generated_refusal_as_an_ai()
    test_parser_recognises_generated_refusal_language_model()
    test_parser_recognises_canned_refusal_many_sites()
    test_parser_recognises_canned_refusal_no_info()
    test_parser_empty_output_is_parse_fail()
    test_parser_ambiguous_content_is_parse_fail()
    test_parser_normal_tweet_starting_with_as_an_engineer_is_still_tweet()
    print("OK")
