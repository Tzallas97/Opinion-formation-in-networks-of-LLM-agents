"""Topic-specific persona profile presets.

These presets change evidence weighting, not final belief.  Avoid profiles such
as 'conspiracy believer' or 'NASA defender' because those encode conclusions.
"""
from __future__ import annotations

import random
from typing import Any

TOPIC_PROFILE_MANUAL = "Manual"

TOPIC_PERSONA_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    # Redesigned after the A/B cleanup: profiles vary only the retained topic traits,
    # are mutually distinct, carry no removed keys, and are named after the trait
    # pattern (not a conclusion).
    "v130": {
        "Computational plausibility thinker": {
            "computational_worldview": "high",
            "anthropic_reasoning_comfort": "high",
            "future_technology_prior": "high",
            "consciousness_intuition": "computational",
        },
        "Physicalist realist": {
            "computational_worldview": "low",
            "anthropic_reasoning_comfort": "low",
            "future_technology_prior": "low",
            "consciousness_intuition": "physicalist",
        },
        "Anthropic-reasoning accepter": {
            "computational_worldview": "medium",
            "anthropic_reasoning_comfort": "high",
            "future_technology_prior": "medium",
            "consciousness_intuition": "uncertain",
        },
        "Consciousness-computation sympathizer": {
            "computational_worldview": "high",
            "anthropic_reasoning_comfort": "medium",
            "future_technology_prior": "medium",
            "consciousness_intuition": "computational",
        },
    },
    "v119": {
        "Motive-and-anomaly connector": {
            "geopolitical_motive_sensitivity": "high",
            "conspiracy_coordination_prior": "medium",
            "anomaly_sensitivity": "high",
        },
        "Coordination-skeptical": {
            "geopolitical_motive_sensitivity": "medium",
            "conspiracy_coordination_prior": "low",
            "anomaly_sensitivity": "medium",
        },
        "Anomaly-minimal institutionalist": {
            "geopolitical_motive_sensitivity": "low",
            "conspiracy_coordination_prior": "low",
            "anomaly_sensitivity": "low",
        },
    },
    "v52": {
        "Engineering-anchored": {
            "cold_war_motive_sensitivity": "low",
            "engineering_evidence_weight": "high",
            "visual_anomaly_sensitivity": "low",
        },
        "Motive-driven doubter": {
            "cold_war_motive_sensitivity": "high",
            "engineering_evidence_weight": "medium",
            "visual_anomaly_sensitivity": "medium",
        },
        "Anomaly-focused": {
            "cold_war_motive_sensitivity": "medium",
            "engineering_evidence_weight": "low",
            "visual_anomaly_sensitivity": "high",
        },
    },
}


def topic_profile_names(topic_root: str) -> list[str]:
    return [TOPIC_PROFILE_MANUAL, *list(TOPIC_PERSONA_PROFILES.get(topic_root, {}).keys())]


def get_topic_profile(topic_root: str, profile_name: str) -> dict[str, str]:
    if str(profile_name or "") == TOPIC_PROFILE_MANUAL:
        return {}
    return dict(TOPIC_PERSONA_PROFILES.get(topic_root, {}).get(str(profile_name or ""), {}))


def random_topic_profile(topic_root: str, rng: random.Random | None = None) -> tuple[str, dict[str, str]]:
    profiles = TOPIC_PERSONA_PROFILES.get(topic_root, {})
    if not profiles:
        return TOPIC_PROFILE_MANUAL, {}
    rng = rng or random.Random()
    name = rng.choice(list(profiles.keys()))
    return name, dict(profiles[name])
