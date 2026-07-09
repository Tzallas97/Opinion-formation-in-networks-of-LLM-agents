"""Topic-specific persona field definitions for opinion-dynamics UI.

This file intentionally contains data, not UI code.  The launcher reads it to
show the correct topic-causal variables for v52, v119, v130, etc.
"""
from __future__ import annotations

import re
from typing import Any

LEVEL_VALUES = ["", "low", "medium", "high"]
CONSCIOUSNESS_VALUES = ["", "physicalist", "uncertain", "computational"]

TOPIC_PERSONA_FIELDSETS: dict[str, dict[str, Any]] = {
    "v130": {
        "label": "Simulation hypothesis",
        "claim_short": "We live in a computer simulation created by an advanced civilization",
        "fields": [
            {"key": "computational_worldview", "label": "Computational worldview", "importance": "high", "values": LEVEL_VALUES, "description": "How naturally code/information analogies feel for reality."},
            {"key": "testability_preference", "label": "Testability preference", "importance": "high", "values": LEVEL_VALUES, "description": "How strongly direct testability is required before belief."},
            {"key": "anthropic_reasoning_comfort", "label": "Anthropic reasoning comfort", "importance": "high", "values": LEVEL_VALUES, "description": "Comfort with observer-count and reference-class arguments."},
            {"key": "future_technology_prior", "label": "Future-technology prior", "importance": "high", "values": LEVEL_VALUES, "description": "How plausible very advanced simulation-capable civilizations feel."},
            {"key": "consciousness_intuition", "label": "Consciousness intuition", "importance": "medium", "values": CONSCIOUSNESS_VALUES, "description": "Whether simulated/digital consciousness feels plausible."},
            {"key": "metaphysical_speculation_tolerance", "label": "Metaphysical speculation tolerance", "importance": "medium", "values": LEVEL_VALUES, "description": "Willingness to take coherent but speculative claims seriously."},
        ],
    },
    "v119": {
        "label": "Twin Towers / 9-11",
        "claim_short": "Twin Towers brought down by insider conspiracy vs terrorist attack account",
        "fields": [
            {"key": "security_state_suspicion", "label": "Security-state suspicion", "importance": "high", "values": LEVEL_VALUES, "description": "How plausible hidden security-state involvement feels."},
            {"key": "official_investigation_trust", "label": "Official-investigation trust", "importance": "high", "values": LEVEL_VALUES, "description": "How much official commissions and institutional explanations are trusted."},
            {"key": "geopolitical_motive_sensitivity", "label": "Geopolitical motive sensitivity", "importance": "high", "values": LEVEL_VALUES, "description": "How strongly war/surveillance motives affect plausibility judgments."},
            {"key": "conspiracy_coordination_prior", "label": "Coordination prior", "importance": "high", "values": LEVEL_VALUES, "description": "How plausible large coordinated insider plots feel."},
            {"key": "anomaly_sensitivity", "label": "Anomaly sensitivity", "importance": "medium", "values": LEVEL_VALUES, "description": "How much unresolved anomalies affect belief."},
            {"key": "source_skepticism", "label": "Source skepticism", "importance": "medium", "values": LEVEL_VALUES, "description": "How skeptical the agent is toward both official and alternative source claims."},
        ],
    },
    "v52": {
        "label": "Moon landing",
        "claim_short": "US astronauts have landed on the moon",
        "fields": [
            {"key": "space_program_trust", "label": "Space-program trust", "importance": "high", "values": LEVEL_VALUES, "description": "How much NASA/space-program capability is trusted."},
            {"key": "cold_war_motive_sensitivity", "label": "Cold-War motive sensitivity", "importance": "high", "values": LEVEL_VALUES, "description": "How strongly Cold War propaganda incentives matter."},
            {"key": "engineering_evidence_weight", "label": "Engineering evidence weight", "importance": "high", "values": LEVEL_VALUES, "description": "How much technical/engineering feasibility arguments matter."},
            {"key": "institutional_science_trust", "label": "Institutional science trust", "importance": "high", "values": LEVEL_VALUES, "description": "How much scientific/institutional consensus carries weight."},
            {"key": "visual_anomaly_sensitivity", "label": "Visual-anomaly sensitivity", "importance": "medium", "values": LEVEL_VALUES, "description": "How much photo/video anomalies affect belief."},
            {"key": "media_manipulation_suspicion", "label": "Media-manipulation suspicion", "importance": "medium", "values": LEVEL_VALUES, "description": "How plausible staged-media/public deception feels."},
        ],
    },
    "generic": {
        "label": "Generic topic",
        "claim_short": "Generic claim",
        "fields": [
            {"key": "topic_specific_trust", "label": "Topic-specific trust", "importance": "medium", "values": LEVEL_VALUES, "description": "How much relevant institutions or domain actors are trusted for this topic."},
            {"key": "topic_specific_suspicion", "label": "Topic-specific suspicion", "importance": "medium", "values": LEVEL_VALUES, "description": "How much hidden motives feel plausible for this topic."},
            {"key": "topic_specific_expertise_weight", "label": "Expertise weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly domain expertise affects the agent's judgment."},
            {"key": "topic_specific_anomaly_weight", "label": "Anomaly weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly anomalies or unresolved details affect belief."},
            {"key": "topic_specific_motive_weight", "label": "Motive weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly motive/incentive arguments affect belief."},
            {"key": "topic_specific_speculation_tolerance", "label": "Speculation tolerance", "importance": "medium", "values": LEVEL_VALUES, "description": "How much coherent-but-indirect arguments are tolerated."},
        ],
    },
}

TOPIC_SLOT_KEYS = [
    "computational_worldview",
    "testability_preference",
    "anthropic_reasoning_comfort",
    "future_technology_prior",
    "consciousness_intuition",
    "metaphysical_speculation_tolerance",
]


def version_root(version_set: str | None) -> str:
    m = re.match(r"^(v\d+)", str(version_set or "").strip().lower())
    return m.group(1) if m else "generic"


def get_topic_fieldset(version_set: str | None) -> dict[str, Any]:
    root = version_root(version_set)
    return TOPIC_PERSONA_FIELDSETS.get(root, TOPIC_PERSONA_FIELDSETS["generic"])


def get_topic_fields(version_set: str | None) -> list[dict[str, Any]]:
    return list(get_topic_fieldset(version_set).get("fields", []))


def get_topic_label(version_set: str | None) -> str:
    return str(get_topic_fieldset(version_set).get("label", "Generic topic"))


ALL_TOPIC_PERSONA_FIELD_KEYS = sorted({f["key"] for fs in TOPIC_PERSONA_FIELDSETS.values() for f in fs.get("fields", [])})
ALL_TOPIC_PERSONA_FIELD_DEFS = {f["key"]: f for fs in TOPIC_PERSONA_FIELDSETS.values() for f in fs.get("fields", [])}
