"""Topic-specific persona profile presets.

These presets change evidence weighting, not final belief.  Avoid profiles such
as 'conspiracy believer' or 'NASA defender' because those encode conclusions.
"""
from __future__ import annotations

import random
from typing import Any

TOPIC_PROFILE_MANUAL = "Manual"

TOPIC_PERSONA_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "v130": {
        "Computational plausibility thinker": {
            "computational_worldview": "high",
            "testability_preference": "medium",
            "anthropic_reasoning_comfort": "high",
            "future_technology_prior": "high",
            "consciousness_intuition": "computational",
            "metaphysical_speculation_tolerance": "medium",
        },
        "Empirical testability skeptic": {
            "computational_worldview": "medium",
            "testability_preference": "high",
            "anthropic_reasoning_comfort": "low",
            "future_technology_prior": "medium",
            "consciousness_intuition": "physicalist",
            "metaphysical_speculation_tolerance": "low",
        },
        "Anthropic-reasoning accepter": {
            "computational_worldview": "medium",
            "testability_preference": "medium",
            "anthropic_reasoning_comfort": "high",
            "future_technology_prior": "medium",
            "consciousness_intuition": "uncertain",
            "metaphysical_speculation_tolerance": "high",
        },
        "Speculation-averse realist": {
            "computational_worldview": "low",
            "testability_preference": "high",
            "anthropic_reasoning_comfort": "low",
            "future_technology_prior": "low",
            "consciousness_intuition": "physicalist",
            "metaphysical_speculation_tolerance": "low",
        },
        "Consciousness-computation sympathizer": {
            "computational_worldview": "high",
            "testability_preference": "medium",
            "anthropic_reasoning_comfort": "medium",
            "future_technology_prior": "medium",
            "consciousness_intuition": "computational",
            "metaphysical_speculation_tolerance": "high",
        },
    },
    "v119": {
        "Security-state skeptic": {
            "security_state_suspicion": "high",
            "official_investigation_trust": "low",
            "geopolitical_motive_sensitivity": "high",
            "conspiracy_coordination_prior": "medium",
            "anomaly_sensitivity": "high",
            "source_skepticism": "medium",
        },
        "Official-investigation truster": {
            "security_state_suspicion": "low",
            "official_investigation_trust": "high",
            "geopolitical_motive_sensitivity": "medium",
            "conspiracy_coordination_prior": "low",
            "anomaly_sensitivity": "low",
            "source_skepticism": "medium",
        },
        "Motive-and-anomaly connector": {
            "security_state_suspicion": "medium",
            "official_investigation_trust": "low",
            "geopolitical_motive_sensitivity": "high",
            "conspiracy_coordination_prior": "medium",
            "anomaly_sensitivity": "high",
            "source_skepticism": "low",
        },
        "Coordination-skeptical realist": {
            "security_state_suspicion": "medium",
            "official_investigation_trust": "medium",
            "geopolitical_motive_sensitivity": "medium",
            "conspiracy_coordination_prior": "low",
            "anomaly_sensitivity": "medium",
            "source_skepticism": "high",
        },
        "Alternative-source skeptic": {
            "security_state_suspicion": "medium",
            "official_investigation_trust": "medium",
            "geopolitical_motive_sensitivity": "medium",
            "conspiracy_coordination_prior": "low",
            "anomaly_sensitivity": "medium",
            "source_skepticism": "high",
        },
    },
    "v52": {
        "Space-program truster": {
            "space_program_trust": "high",
            "cold_war_motive_sensitivity": "medium",
            "engineering_evidence_weight": "high",
            "institutional_science_trust": "high",
            "visual_anomaly_sensitivity": "low",
            "media_manipulation_suspicion": "low",
        },
        "Cold-War motive skeptic": {
            "space_program_trust": "medium",
            "cold_war_motive_sensitivity": "high",
            "engineering_evidence_weight": "medium",
            "institutional_science_trust": "medium",
            "visual_anomaly_sensitivity": "medium",
            "media_manipulation_suspicion": "high",
        },
        "Engineering-first evaluator": {
            "space_program_trust": "medium",
            "cold_war_motive_sensitivity": "medium",
            "engineering_evidence_weight": "high",
            "institutional_science_trust": "medium",
            "visual_anomaly_sensitivity": "low",
            "media_manipulation_suspicion": "low",
        },
        "Visual-anomaly-sensitive skeptic": {
            "space_program_trust": "medium",
            "cold_war_motive_sensitivity": "medium",
            "engineering_evidence_weight": "medium",
            "institutional_science_trust": "medium",
            "visual_anomaly_sensitivity": "high",
            "media_manipulation_suspicion": "medium",
        },
        "Institutional-science truster": {
            "space_program_trust": "high",
            "cold_war_motive_sensitivity": "low",
            "engineering_evidence_weight": "high",
            "institutional_science_trust": "high",
            "visual_anomaly_sensitivity": "low",
            "media_manipulation_suspicion": "low",
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
