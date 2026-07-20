#!/usr/bin/env python3
"""Regression tests for the diversification scaffold persona additions.

ADR-006 §Component 1: locus_of_control and contrarianism as new core_causal
traits. Tests operate directly on DEFAULT_SCHEMA and DEFAULT_PROFILES because
the persona schema is a dict template, not a class with helper functions.

Run: python tests/test_persona_scaffold.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from copy import deepcopy

from persona_profiles_support import (  # noqa: E402
    DEFAULT_SCHEMA,
    DEFAULT_PROFILES,
    render_persona_card,
)


EXPECTED_LOCUS_VALUES = ["internal", "mixed", "external"]
EXPECTED_CONTRARIANISM_VALUES = ["very_low", "low", "medium", "high", "very_high"]


def _fresh_profile():
    """Return a copy of the first preset profile so tests can mutate freely."""
    return deepcopy(DEFAULT_PROFILES["profiles"][0])


def test_locus_of_control_in_default_schema_core_causal():
    assert "locus_of_control" in DEFAULT_SCHEMA["core_causal"], \
        "locus_of_control missing from DEFAULT_SCHEMA[core_causal]"
    assert DEFAULT_SCHEMA["core_causal"]["locus_of_control"] == "mixed", \
        "default for locus_of_control must be 'mixed' (neutral, byte-identical baseline)"


def test_contrarianism_in_default_schema_core_causal():
    assert "contrarianism" in DEFAULT_SCHEMA["core_causal"], \
        "contrarianism missing from DEFAULT_SCHEMA[core_causal]"
    assert DEFAULT_SCHEMA["core_causal"]["contrarianism"] == "medium", \
        "default for contrarianism must be 'medium' (neutral, byte-identical baseline)"


def test_locus_of_control_allowed_values():
    assert "locus_of_control" in DEFAULT_SCHEMA["allowed_values"], \
        "locus_of_control missing from DEFAULT_SCHEMA[allowed_values]"
    assert DEFAULT_SCHEMA["allowed_values"]["locus_of_control"] == EXPECTED_LOCUS_VALUES, \
        f"expected {EXPECTED_LOCUS_VALUES}, got {DEFAULT_SCHEMA['allowed_values']['locus_of_control']}"


def test_contrarianism_allowed_values():
    assert "contrarianism" in DEFAULT_SCHEMA["allowed_values"], \
        "contrarianism missing from DEFAULT_SCHEMA[allowed_values]"
    assert DEFAULT_SCHEMA["allowed_values"]["contrarianism"] == EXPECTED_CONTRARIANISM_VALUES, \
        f"expected {EXPECTED_CONTRARIANISM_VALUES}, got {DEFAULT_SCHEMA['allowed_values']['contrarianism']}"


def test_all_preset_profiles_have_new_fields_with_allowed_values():
    """Every preset profile must have both new fields set to a legal value.
    If we forget one, downstream code that assumes every profile is complete
    will get KeyError."""
    profiles = DEFAULT_PROFILES.get("profiles", [])
    assert len(profiles) >= 1, "DEFAULT_PROFILES must contain at least one profile"
    for i, profile in enumerate(profiles):
        core = profile.get("core_causal", {})
        assert "locus_of_control" in core, \
            f"profile[{i}] ({profile.get('profile_meta', {}).get('profile_id', '?')}) missing locus_of_control"
        assert "contrarianism" in core, \
            f"profile[{i}] ({profile.get('profile_meta', {}).get('profile_id', '?')}) missing contrarianism"
        assert core["locus_of_control"] in EXPECTED_LOCUS_VALUES, \
            f"profile[{i}] locus_of_control={core['locus_of_control']!r} not in {EXPECTED_LOCUS_VALUES}"
        assert core["contrarianism"] in EXPECTED_CONTRARIANISM_VALUES, \
            f"profile[{i}] contrarianism={core['contrarianism']!r} not in {EXPECTED_CONTRARIANISM_VALUES}"


def test_default_profile_render_has_no_locus_sentence():
    """Baseline: with default 'mixed' value, the card must NOT gain a
    locus-of-control sentence. This guarantees byte-identical rendering for
    every persona that has not been customized."""
    profile = _fresh_profile()  # locus_of_control='mixed' from Task 1.1 defaults
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    lower = card.lower()
    assert "arguments and evidence" not in lower
    assert "what others" not in lower


def test_internal_locus_adds_argument_sentence():
    profile = _fresh_profile()
    profile["core_causal"]["locus_of_control"] = "internal"
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    assert "arguments and evidence" in card.lower(), \
        f"expected 'arguments and evidence' sentence in card; got:\n{card}"


def test_external_locus_adds_social_sentence():
    profile = _fresh_profile()
    profile["core_causal"]["locus_of_control"] = "external"
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    assert "what others" in card.lower(), \
        f"expected 'what others' sentence in card; got:\n{card}"


def test_default_profile_render_has_no_contrarianism_sentence():
    profile = _fresh_profile()  # contrarianism='medium' from Task 1.1 defaults
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    lower = card.lower()
    assert "differ from" not in lower and "differ with" not in lower
    assert "agree with the perceived" not in lower


def test_high_contrarianism_adds_differ_sentence():
    profile = _fresh_profile()
    profile["core_causal"]["contrarianism"] = "very_high"
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    assert "differ" in card.lower() and "majority" in card.lower(), \
        f"expected differ+majority sentence; got:\n{card}"


def test_low_contrarianism_adds_agree_sentence():
    profile = _fresh_profile()
    profile["core_causal"]["contrarianism"] = "very_low"
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    assert "agree" in card.lower() and "majority" in card.lower(), \
        f"expected agree+majority sentence; got:\n{card}"


def test_render_at_defaults_is_byte_identical_across_scaffold_change():
    """Load-bearing: with every field at its default, the card text must match
    what the same profile produced BEFORE ADR-006 (i.e. must contain none of
    the new sentence stems). This guards against silent behavior change."""
    profile = _fresh_profile()
    card, _ = render_persona_card(profile, agent_name="alice", opinion=0)
    # All 4 new sentence stems must be absent when both new fields are at default.
    forbidden = [
        "arguments and evidence",
        "what others",
        "differ from",
        "agree with the perceived",
    ]
    for stem in forbidden:
        assert stem not in card.lower(), \
            f"scaffold added baseline sentence {stem!r} to default card:\n{card}"


if __name__ == "__main__":
    test_locus_of_control_in_default_schema_core_causal()
    test_contrarianism_in_default_schema_core_causal()
    test_locus_of_control_allowed_values()
    test_contrarianism_allowed_values()
    test_all_preset_profiles_have_new_fields_with_allowed_values()
    test_default_profile_render_has_no_locus_sentence()
    test_internal_locus_adds_argument_sentence()
    test_external_locus_adds_social_sentence()
    test_default_profile_render_has_no_contrarianism_sentence()
    test_high_contrarianism_adds_differ_sentence()
    test_low_contrarianism_adds_agree_sentence()
    test_render_at_defaults_is_byte_identical_across_scaffold_change()
    print("OK")
