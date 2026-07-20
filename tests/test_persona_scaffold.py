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

from persona_profiles_support import DEFAULT_SCHEMA, DEFAULT_PROFILES  # noqa: E402


EXPECTED_LOCUS_VALUES = ["internal", "mixed", "external"]
EXPECTED_CONTRARIANISM_VALUES = ["very_low", "low", "medium", "high", "very_high"]


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


if __name__ == "__main__":
    test_locus_of_control_in_default_schema_core_causal()
    test_contrarianism_in_default_schema_core_causal()
    test_locus_of_control_allowed_values()
    test_contrarianism_allowed_values()
    test_all_preset_profiles_have_new_fields_with_allowed_values()
    print("OK")
