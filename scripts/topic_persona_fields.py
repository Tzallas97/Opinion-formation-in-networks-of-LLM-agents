"""Topic-specific persona field definitions for opinion-dynamics UI.

This file intentionally contains data, not UI code.  The launcher reads it to
show the correct topic-causal variables for v52, v119, v130, etc.
"""
from __future__ import annotations

import re
from typing import Any

LEVEL_VALUES = ["", "low", "medium", "high"]
CONSCIOUSNESS_VALUES = ["", "physicalist", "uncertain", "computational"]

# A/B cleanup (2026-07): each topic-causal layer keeps ONLY constructs the general
# core_causal layer cannot already express. Domain trust/suspicion fields were removed
# because they duplicate institutional_trust / official_narrative_suspicion (projected
# onto the topic via the claim framing). What stays: motive sensitivity, anomaly/pattern
# perception, coordination priors, topic worldview.
TOPIC_PERSONA_FIELDSETS: dict[str, dict[str, Any]] = {
    "v130": {
        "label": "Simulation hypothesis",
        "claim_short": "We live in a computer simulation created by an advanced civilization",
        # Removed as overlaps: testability_preference (~evidence_style), metaphysical_speculation_tolerance (~uncertainty_tolerance).
        "fields": [
            {"key": "computational_worldview", "label": "Computational worldview", "importance": "high", "values": LEVEL_VALUES, "description": "How naturally code/information analogies feel for reality."},
            {"key": "anthropic_reasoning_comfort", "label": "Anthropic reasoning comfort", "importance": "high", "values": LEVEL_VALUES, "description": "Comfort with observer-count and reference-class arguments."},
            {"key": "future_technology_prior", "label": "Future-technology prior", "importance": "high", "values": LEVEL_VALUES, "description": "How plausible very advanced simulation-capable civilizations feel."},
            {"key": "consciousness_intuition", "label": "Consciousness intuition", "importance": "medium", "values": CONSCIOUSNESS_VALUES, "description": "Whether simulated/digital consciousness feels plausible."},
        ],
    },
    "v119": {
        "label": "Twin Towers / 9-11",
        "claim_short": "Twin Towers brought down by insider conspiracy vs terrorist attack account",
        # Removed as overlaps: official_investigation_trust (~institutional_trust), security_state_suspicion & source_skepticism (~official_narrative_suspicion).
        "fields": [
            {"key": "geopolitical_motive_sensitivity", "label": "Geopolitical motive sensitivity", "importance": "high", "values": LEVEL_VALUES, "description": "How strongly war/surveillance motives affect plausibility judgments."},
            {"key": "conspiracy_coordination_prior", "label": "Coordination prior", "importance": "high", "values": LEVEL_VALUES, "description": "How plausible large coordinated insider plots feel."},
            {"key": "anomaly_sensitivity", "label": "Anomaly sensitivity", "importance": "medium", "values": LEVEL_VALUES, "description": "How much unresolved anomalies affect belief."},
        ],
    },
    "v52": {
        "label": "Moon landing",
        "claim_short": "US astronauts have landed on the moon",
        # Removed as overlaps: space_program_trust & institutional_science_trust (~institutional_trust), media_manipulation_suspicion (~official_narrative_suspicion).
        "fields": [
            {"key": "cold_war_motive_sensitivity", "label": "Cold-War motive sensitivity", "importance": "high", "values": LEVEL_VALUES, "description": "How strongly Cold War propaganda incentives matter."},
            {"key": "engineering_evidence_weight", "label": "Engineering evidence weight", "importance": "high", "values": LEVEL_VALUES, "description": "How much technical/engineering feasibility arguments matter."},
            {"key": "visual_anomaly_sensitivity", "label": "Visual-anomaly sensitivity", "importance": "medium", "values": LEVEL_VALUES, "description": "How much photo/video anomalies affect belief."},
        ],
    },
    "generic": {
        "label": "Generic topic",
        "claim_short": "Generic claim",
        # Removed as overlaps: topic_specific_trust (~institutional_trust), topic_specific_suspicion (~official_narrative_suspicion), topic_specific_speculation_tolerance (~uncertainty_tolerance).
        "fields": [
            {"key": "topic_specific_expertise_weight", "label": "Expertise weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly domain expertise affects the agent's judgment."},
            {"key": "topic_specific_anomaly_weight", "label": "Anomaly weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly anomalies or unresolved details affect belief."},
            {"key": "topic_specific_motive_weight", "label": "Motive weight", "importance": "medium", "values": LEVEL_VALUES, "description": "How strongly motive/incentive arguments affect belief."},
        ],
    },
}

# Fixed UI/CSV slot names. The launcher owns six physical topic-causal widgets
# (persona_<slot>_var) and each topic's fields map onto them POSITIONALLY; the
# real data key always comes from the field definition, never from the slot
# name. Two names below are retired v130 field keys kept only as stable slot
# labels so saved launcher settings and old CSV columns keep working.
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
