from __future__ import annotations

import csv
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


# Topic-specific persona fields/profiles are data-only modules used by the UI
# and by this CSV exporter.  They let v52/v119/v130 expose different causal
# traits without hard-coding those topic definitions into the launcher.
try:
    from topic_persona_fields import (
        ALL_TOPIC_PERSONA_FIELD_DEFS,
        ALL_TOPIC_PERSONA_FIELD_KEYS,
        TOPIC_PERSONA_FIELDSETS,
        TOPIC_SLOT_KEYS,
        get_topic_fieldset,
        get_topic_fields,
        get_topic_label,
        version_root as topic_version_root,
    )
except Exception:  # pragma: no cover - fallback keeps old launchers usable
    _LEVEL_VALUES = ["", "low", "medium", "high"]
    TOPIC_SLOT_KEYS = [
        "computational_worldview",
        "testability_preference",
        "anthropic_reasoning_comfort",
        "future_technology_prior",
        "consciousness_intuition",
        "metaphysical_speculation_tolerance",
    ]
    TOPIC_PERSONA_FIELDSETS = {
        "generic": {
            "label": "Generic topic",
            "fields": [
                {"key": "topic_specific_expertise_weight", "label": "Expertise weight", "importance": "medium", "values": _LEVEL_VALUES},
                {"key": "topic_specific_anomaly_weight", "label": "Anomaly weight", "importance": "medium", "values": _LEVEL_VALUES},
                {"key": "topic_specific_motive_weight", "label": "Motive weight", "importance": "medium", "values": _LEVEL_VALUES},
            ],
        },
    }
    def topic_version_root(version_set: str | None) -> str:
        m = re.match(r"^(v\d+)", str(version_set or "").strip().lower())
        return m.group(1) if m else "generic"
    def get_topic_fieldset(version_set: str | None) -> dict[str, Any]:
        return TOPIC_PERSONA_FIELDSETS.get(topic_version_root(version_set), TOPIC_PERSONA_FIELDSETS["generic"])
    def get_topic_fields(version_set: str | None) -> list[dict[str, Any]]:
        return list(get_topic_fieldset(version_set).get("fields", []))
    def get_topic_label(version_set: str | None) -> str:
        return str(get_topic_fieldset(version_set).get("label", "Generic topic"))
    ALL_TOPIC_PERSONA_FIELD_KEYS = sorted({f["key"] for fs in TOPIC_PERSONA_FIELDSETS.values() for f in fs.get("fields", [])})
    ALL_TOPIC_PERSONA_FIELD_DEFS = {f["key"]: f for fs in TOPIC_PERSONA_FIELDSETS.values() for f in fs.get("fields", [])}

try:
    from topic_persona_profiles import TOPIC_PROFILE_MANUAL, get_topic_profile
except Exception:  # pragma: no cover
    TOPIC_PROFILE_MANUAL = "Manual"
    def get_topic_profile(topic_root: str, profile_name: str) -> dict[str, str]:
        return {}

OPINION_VALUES = [-2, -1, 0, 1, 2]

DEFAULT_SCHEMA = {
    "schema_version": "1.0",
    "profile_meta": {
        "profile_id": "",
        "profile_name": "",
        "profile_source": "preset",
        "description": "",
        "tags": []
    },
    "core_causal": {
        "epistemic_profile_label": "",
        "institutional_trust": "medium",
        "uncertainty_tolerance": "medium",
        "evidence_style": "mixed",
        "official_narrative_suspicion": "medium",
        "openness_to_update": "medium",
        # ADR-006 Component 1 additions (defaults are neutral -> byte-identical baseline).
        # locus_of_control: Rotter 1966 - "internal" agents update from arguments/evidence,
        #                   "external" update from majority/social pressure, "mixed" = neutral.
        # contrarianism:    Flache 2017 (repulsive influence) - "very_high" strongly diverges
        #                   from perceived majority, "very_low" strongly conforms, "medium" = neutral.
        "locus_of_control": "mixed",
        "contrarianism": "medium"
    },
    "topic_linked": {
        "occupation": "",
        "education_level": "",
        "training_style": "",
        "domain_familiarity": "",
        "topic_interest": "",
        "prior_exposure": "",
        "custom_topic_notes": ""
    },
    "flavor_only": {
        "age_group": "",
        "gender": "",
        "ethnicity": "",
        "lifestyle_notes": "",
        "tone_hint": ""
    },
    "render_config": {
        "use_core_causal": True,
        "use_topic_causal": False,
        "use_topic_linked": False,
        "use_flavor_only": False,
        "topic_mode": "off",
        "flavor_mode": "off",
        "manual_topic_override": False,
        "manual_topic_causal_override": False,
        "manual_flavor_override": False,
        "show_profile_label": True,
        "render_style": "structured_card"
    },
    "allowed_values": {
        "institutional_trust": ["very_low", "low", "medium", "high", "very_high"],
        "uncertainty_tolerance": ["very_low", "low", "medium", "high", "very_high"],
        "evidence_style": ["concrete_first", "source_first", "coherence_first", "intuition_first", "mixed"],
        "official_narrative_suspicion": ["very_low", "low", "medium", "high", "very_high"],
        "openness_to_update": ["very_low", "low", "medium", "high", "very_high"],
        # ADR-006 Component 1 additions.
        "locus_of_control": ["internal", "mixed", "external"],
        "contrarianism": ["very_low", "low", "medium", "high", "very_high"],
        "education_level": ["high_school", "bachelor", "master", "phd", "other"],
        "training_style": ["technical", "humanities", "social_science", "mixed", "none"],
        "domain_familiarity": ["none", "low", "medium", "high"],
        "age_group": ["18_24", "25_29", "30_39", "40_49", "50_59", "60_plus"],
        "topic_mode": ["off", "auto", "manual"],
        "flavor_mode": ["off", "auto", "manual"],
        "render_style": ["structured_card", "compact_card"],
        "profile_distribution_mode": ["random", "equal", "all", "more", "most"]
    }
}

DEFAULT_PROFILES = {
    "profiles": [
        {
            "profile_meta": {
                "profile_id": "trusting_pragmatist_v1",
                "profile_name": "Institutionally trusting pragmatist",
                "profile_source": "preset",
                "description": "High institutional trust, concrete-first, moderate openness",
                "tags": ["preset", "trusting"]
            },
            "core_causal": {
                "epistemic_profile_label": "Institutionally trusting pragmatist",
                "institutional_trust": "high",
                "uncertainty_tolerance": "medium",
                "evidence_style": "concrete_first",
                "official_narrative_suspicion": "low",
                "openness_to_update": "medium",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        },
        {
            "profile_meta": {
                "profile_id": "suspicious_skeptic_v1",
                "profile_name": "Suspicious anti-institutional skeptic",
                "profile_source": "preset",
                "description": "Low institutional trust, high official-narrative suspicion, lower openness",
                "tags": ["preset", "skeptical"]
            },
            "core_causal": {
                "epistemic_profile_label": "Suspicious anti-institutional skeptic",
                "institutional_trust": "low",
                "uncertainty_tolerance": "low",
                "evidence_style": "coherence_first",
                "official_narrative_suspicion": "high",
                "openness_to_update": "low",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        },
        {
            "profile_meta": {
                "profile_id": "open_minded_skeptic_v1",
                "profile_name": "Open-minded skeptical reviser",
                "profile_source": "preset",
                "description": "Suspicious of official stories but open to changing view when a clear point appears",
                "tags": ["preset", "skeptical", "open"]
            },
            "core_causal": {
                "epistemic_profile_label": "Open-minded skeptical reviser",
                "institutional_trust": "low",
                "uncertainty_tolerance": "high",
                "evidence_style": "coherence_first",
                "official_narrative_suspicion": "high",
                "openness_to_update": "high",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        },
        {
            "profile_meta": {
                "profile_id": "authority_stabilizer_v1",
                "profile_name": "Authority-leaning stabilizer",
                "profile_source": "preset",
                "description": "Relies heavily on established records and does not move easily",
                "tags": ["preset", "authority"]
            },
            "core_causal": {
                "epistemic_profile_label": "Authority-leaning stabilizer",
                "institutional_trust": "high",
                "uncertainty_tolerance": "low",
                "evidence_style": "source_first",
                "official_narrative_suspicion": "low",
                "openness_to_update": "low",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        },
        {
            "profile_meta": {
                "profile_id": "uncertainty_tolerant_agnostic_v1",
                "profile_name": "Uncertainty-tolerant agnostic",
                "profile_source": "preset",
                "description": "Comfortable staying unsure when information is limited",
                "tags": ["preset", "agnostic"]
            },
            "core_causal": {
                "epistemic_profile_label": "Uncertainty-tolerant agnostic",
                "institutional_trust": "medium",
                "uncertainty_tolerance": "high",
                "evidence_style": "concrete_first",
                "official_narrative_suspicion": "medium",
                "openness_to_update": "medium",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        },
        {
            "profile_meta": {
                "profile_id": "practical_distruster_v1",
                "profile_name": "Low-trust practical realist",
                "profile_source": "preset",
                "description": "Wary of institutions, but concrete physical details matter more than vague suspicion",
                "tags": ["preset", "practical"]
            },
            "core_causal": {
                "epistemic_profile_label": "Low-trust practical realist",
                "institutional_trust": "low",
                "uncertainty_tolerance": "medium",
                "evidence_style": "concrete_first",
                "official_narrative_suspicion": "medium",
                "openness_to_update": "medium",
                "locus_of_control": "mixed",
                "contrarianism": "medium"
            },
            "topic_linked": {
                "occupation": "",
                "education_level": "",
                "training_style": "",
                "domain_familiarity": "",
                "topic_interest": "",
                "prior_exposure": "",
                "custom_topic_notes": ""
            },
            "flavor_only": {
                "age_group": "",
                "gender": "",
                "ethnicity": "",
                "lifestyle_notes": "",
                "tone_hint": ""
            },
            "render_config": {
                "use_core_causal": True,
                "use_topic_causal": False,
                "use_topic_linked": False,
                "use_flavor_only": False,
                "topic_mode": "off",
                "flavor_mode": "off",
                "manual_topic_override": False,
                "manual_topic_causal_override": False,
                "manual_flavor_override": False,
                "show_profile_label": True,
                "render_style": "structured_card",
                "profile_distribution_mode": "all"
            }
        }
    ]
}

AGE_GROUP_RANGES = {
    "18_24": (18, 24),
    "25_29": (25, 29),
    "30_39": (30, 39),
    "40_49": (40, 49),
    "50_59": (50, 59),
    "60_plus": (60, 70),
}


BASE_TOPIC_LINKED_KEYS = [
    "occupation", "education_level", "training_style", "domain_familiarity",
    "topic_interest", "prior_exposure", "custom_topic_notes",
]
TOPIC_META_KEYS = ["topic_version", "topic_fieldset_label", "topic_profile_choice", "topic_causal_profile_choice"]


def _all_topic_linked_keys() -> list[str]:
    out: list[str] = []
    for k in [*BASE_TOPIC_LINKED_KEYS, *TOPIC_META_KEYS, *TOPIC_SLOT_KEYS, *ALL_TOPIC_PERSONA_FIELD_KEYS]:
        if k not in out:
            out.append(k)
    return out


def _ensure_topic_keys(mapping: dict[str, Any] | None) -> dict[str, Any]:
    mapping = dict(mapping or {})
    for k in _all_topic_linked_keys():
        mapping.setdefault(k, "")
    return mapping


def _install_topic_fields_into_defaults() -> None:
    DEFAULT_SCHEMA.setdefault("topic_linked", {})
    DEFAULT_SCHEMA.setdefault("allowed_values", {})
    for k in _all_topic_linked_keys():
        DEFAULT_SCHEMA["topic_linked"].setdefault(k, "")
    DEFAULT_SCHEMA["allowed_values"].setdefault("topic_profile_choice", [TOPIC_PROFILE_MANUAL])
    DEFAULT_SCHEMA["allowed_values"].setdefault("topic_causal_profile_choice", [TOPIC_PROFILE_MANUAL])
    DEFAULT_SCHEMA["allowed_values"].setdefault("topic_version", ["generic", "v52", "v119", "v130"])
    for key, meta in (ALL_TOPIC_PERSONA_FIELD_DEFS or {}).items():
        vals = list(meta.get("values", ["", "low", "medium", "high"]))
        DEFAULT_SCHEMA["allowed_values"].setdefault(key, vals)
    for prof in DEFAULT_PROFILES.get("profiles", []):
        prof["topic_linked"] = _ensure_topic_keys(prof.get("topic_linked", {}))


_install_topic_fields_into_defaults()


def _topic_version_from_state(ui_state: dict[str, Any]) -> str:
    return str(
        ui_state.get("topic_version")
        or ui_state.get("prompt_version")
        or ui_state.get("version_set")
        or "generic"
    ).strip()


def _topic_linked_from_ui_state(ui_state: dict[str, Any]) -> dict[str, Any]:
    topic_version = _topic_version_from_state(ui_state)
    topic = {
        "occupation": str(ui_state.get("occupation", "") or "").strip(),
        "education_level": str(ui_state.get("education_level", "") or "").strip(),
        "training_style": str(ui_state.get("training_style", "") or "").strip(),
        "domain_familiarity": str(ui_state.get("domain_familiarity", "") or "").strip(),
        "topic_interest": str(ui_state.get("topic_interest", "") or "").strip(),
        "prior_exposure": str(ui_state.get("prior_exposure", "") or "").strip(),
        "custom_topic_notes": str(ui_state.get("custom_topic_notes", "") or "").strip(),
        "topic_version": topic_version,
        "topic_fieldset_label": str(ui_state.get("topic_fieldset_label") or get_topic_label(topic_version) or "").strip(),
        "topic_causal_profile_choice": str(ui_state.get("topic_causal_profile_choice") or ui_state.get("topic_profile_choice") or ui_state.get("topic_profile") or TOPIC_PROFILE_MANUAL).strip(),
        "topic_profile_choice": str(ui_state.get("topic_causal_profile_choice") or ui_state.get("topic_profile_choice") or ui_state.get("topic_profile") or TOPIC_PROFILE_MANUAL).strip(),
    }

    # Slot keys are the physical UI variables.  For v119/v52 the slot labels
    # change, so also preserve the actual topic-specific keys via
    # topic_specific_fields.
    for slot_key in TOPIC_SLOT_KEYS:
        topic[slot_key] = str(ui_state.get(slot_key, "") or "").strip()

    specific = ui_state.get("topic_specific_fields") or {}
    if isinstance(specific, dict):
        for key, value in specific.items():
            if str(key).strip():
                topic[str(key).strip()] = str(value or "").strip()

    for key in ALL_TOPIC_PERSONA_FIELD_KEYS:
        if key in ui_state and str(ui_state.get(key, "") or "").strip():
            topic[key] = str(ui_state.get(key, "") or "").strip()
        else:
            topic.setdefault(key, "")

    return _ensure_topic_keys(topic)


def _topic_field_meta_by_key(version_set: str | None = None) -> dict[str, dict[str, Any]]:
    meta = dict(ALL_TOPIC_PERSONA_FIELD_DEFS or {})
    try:
        for f in get_topic_fields(version_set):
            meta[str(f.get("key"))] = dict(f)
    except Exception:
        pass
    return meta


def _topic_values_for_render(topic: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    version = str(topic.get("topic_version", "") or "generic")
    meta_by_key = _topic_field_meta_by_key(version)
    rows = []
    active_fields = get_topic_fields(version)
    keys = [str(f.get("key")) for f in active_fields if str(f.get("key", "")).strip()]
    # Include active topic keys first, then any stored dynamic keys that still have values.
    for key in ALL_TOPIC_PERSONA_FIELD_KEYS:
        if key not in keys and str(topic.get(key, "") or "").strip():
            keys.append(key)
    for key in keys:
        val = str(topic.get(key, "") or "").strip()
        if not val:
            continue
        meta = meta_by_key.get(key, {})
        label = str(meta.get("label") or key.replace("_", " ").title())
        importance = str(meta.get("importance") or "medium")
        rows.append((key, label, importance, val))
    return rows


def ensure_config_files(config_dir: Path) -> tuple[Path, Path]:
    config_dir.mkdir(parents=True, exist_ok=True)
    schema_path = config_dir / "persona_schema.json"
    profiles_path = config_dir / "persona_profiles.json"
    if not schema_path.exists():
        schema_path.write_text(json.dumps(DEFAULT_SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    if not profiles_path.exists():
        profiles_path.write_text(json.dumps(DEFAULT_PROFILES, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema_path, profiles_path


def load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return deepcopy(fallback)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_schema(config_dir: Path) -> dict[str, Any]:
    schema_path, _ = ensure_config_files(config_dir)
    return load_json(schema_path, DEFAULT_SCHEMA)


def load_profiles(config_dir: Path) -> dict[str, Any]:
    _, profiles_path = ensure_config_files(config_dir)
    payload = load_json(profiles_path, DEFAULT_PROFILES)
    if not isinstance(payload.get("profiles"), list):
        payload = deepcopy(DEFAULT_PROFILES)
    return payload


def save_profiles(config_dir: Path, payload: dict[str, Any]) -> None:
    _, profiles_path = ensure_config_files(config_dir)
    save_json(profiles_path, payload)

def preset_profile_names() -> list[str]:
    return [str(p.get("profile_meta", {}).get("profile_name", "")).strip() for p in DEFAULT_PROFILES.get("profiles", []) if str(p.get("profile_meta", {}).get("profile_name", "")).strip()]


def get_preset_profile_by_name(profile_name: str) -> dict[str, Any] | None:
    target = str(profile_name or "").strip()
    for p in DEFAULT_PROFILES.get("profiles", []):
        name = str(p.get("profile_meta", {}).get("profile_name", "")).strip()
        if name == target:
            return deepcopy(p)
    return None


def _profile_pool_for_mode(selected_profile: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    '''Candidate profiles per distribution mode.

    all         every agent gets the selected profile (as configured in the UI).
    random      uniform draw from the PRESET library; the selected profile's
                custom edits are not part of the pool unless it is a preset.
    equal       equal split of the preset library (same pool note as random).
    more / most selected profile (with its edits) weighted 55% / 80%, the
                remaining presets share the rest.
    '''
    mode = (mode or "all").strip().lower()
    presets = [deepcopy(p) for p in DEFAULT_PROFILES.get("profiles", [])]
    if mode == "all":
        return [deepcopy(selected_profile)]
    if mode == "random":
        return presets
    if mode == "equal":
        return presets
    selected_id = str(selected_profile.get("profile_meta", {}).get("profile_id", "")).strip()
    others = [deepcopy(p) for p in presets if str(p.get("profile_meta", {}).get("profile_id", "")).strip() != selected_id]
    pool = [deepcopy(selected_profile)]
    pool.extend(others)
    return pool


def slugify_profile_id(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "custom_profile"


def list_profile_choices(profiles_payload: dict[str, Any]) -> list[str]:
    out = []
    for p in profiles_payload.get("profiles", []):
        meta = p.get("profile_meta", {})
        pid = str(meta.get("profile_id", "")).strip()
        name = str(meta.get("profile_name", pid)).strip() or pid
        if pid:
            out.append(f"{name} [{pid}]")
    return out


def extract_profile_id(choice_text: str) -> str:
    m = re.search(r"\[([^\]]+)\]\s*$", str(choice_text or ""))
    if m:
        return m.group(1).strip()
    return slugify_profile_id(choice_text)


def get_profile_by_id(profiles_payload: dict[str, Any], profile_id: str) -> dict[str, Any] | None:
    target = (profile_id or "").strip()
    for p in profiles_payload.get("profiles", []):
        pid = str(p.get("profile_meta", {}).get("profile_id", "")).strip()
        if pid == target:
            return deepcopy(p)
    return None


def upsert_profile(profiles_payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(profiles_payload)
    pid = str(profile.get("profile_meta", {}).get("profile_id", "")).strip()
    if not pid:
        raise ValueError("profile_id is required")
    replaced = False
    for i, p in enumerate(payload.get("profiles", [])):
        cur = str(p.get("profile_meta", {}).get("profile_id", "")).strip()
        if cur == pid:
            payload["profiles"][i] = deepcopy(profile)
            replaced = True
            break
    if not replaced:
        payload.setdefault("profiles", []).append(deepcopy(profile))
    return payload


def belief_label(op: int) -> str:
    if op <= -2:
        return "Strongly Negative"
    if op == -1:
        return "Negative"
    if op == 0:
        return "Neutral"
    if op == 1:
        return "Positive"
    return "Strongly Positive"


def _titleize_enum(value: str) -> str:
    v = str(value or "").strip()
    if not v:
        return ""
    return v.replace("_", " ").replace("-", " ").title()


def derive_topic_from_core(core: dict[str, Any]) -> dict[str, str]:
    inst = str(core.get("institutional_trust", "medium")).strip().lower()
    ev = str(core.get("evidence_style", "mixed")).strip().lower()
    susp = str(core.get("official_narrative_suspicion", "medium")).strip().lower()

    topic = {
        "occupation": "",
        "education_level": "",
        "training_style": "",
        "domain_familiarity": "",
        "topic_interest": "",
        "prior_exposure": "",
        "custom_topic_notes": "",
    }

    if ev == "concrete_first" and inst in {"high", "very_high"}:
        topic.update({
            "occupation": "engineer",
            "education_level": "bachelor",
            "training_style": "technical",
            "domain_familiarity": "medium",
            "topic_interest": "space_science",
            "prior_exposure": "follows science and institutional reporting"
        })
    elif susp in {"high", "very_high"}:
        topic.update({
            "occupation": "journalist",
            "education_level": "bachelor",
            "training_style": "humanities",
            "domain_familiarity": "medium",
            "topic_interest": "media_critique",
            "prior_exposure": "often notices inconsistencies in public narratives"
        })
    elif ev == "coherence_first":
        topic.update({
            "occupation": "analyst",
            "education_level": "master",
            "training_style": "mixed",
            "domain_familiarity": "medium",
            "topic_interest": "reasoning_and_argument_quality",
            "prior_exposure": "used to comparing arguments for internal consistency"
        })
    else:
        topic.update({
            "occupation": "teacher",
            "education_level": "bachelor",
            "training_style": "mixed",
            "domain_familiarity": "low",
            "topic_interest": "general_public_affairs",
            "prior_exposure": "engages with public issues at a general level"
        })
    return topic


def derive_flavor_from_core(core: dict[str, Any]) -> dict[str, str]:
    '''Neutral flavor defaults; only age_group gets a stable default. The core
    argument is accepted for symmetry with derive_topic_from_core.'''
    return {
        "age_group": "30_39",
        "gender": "",
        "ethnicity": "",
        "lifestyle_notes": "",
        "tone_hint": ""
    }


def build_profile_from_ui(ui_state: dict[str, Any]) -> dict[str, Any]:
    profile_choice = str(ui_state.get("profile_label_choice", "") or "").strip()
    custom_profile_label = str(ui_state.get("custom_profile_label", "") or "").strip()
    derived_label = custom_profile_label if profile_choice.lower() == "custom" else profile_choice
    profile_name = str(ui_state.get("profile_name", "") or "").strip() or derived_label or "Custom profile"
    profile_id = slugify_profile_id(str(ui_state.get("profile_id", "") or profile_name))
    profile = {
        "profile_meta": {
            "profile_id": profile_id,
            "profile_name": profile_name,
            "profile_source": str(ui_state.get("profile_source", "custom") or "custom").strip() or "custom",
            "description": str(ui_state.get("profile_description", "") or "").strip(),
            "tags": list(ui_state.get("profile_tags", []) or []),
        },
        "core_causal": {
            "epistemic_profile_label": derived_label or str(ui_state.get("epistemic_profile_label", "") or "").strip(),
            "institutional_trust": str(ui_state.get("institutional_trust", "medium") or "medium").strip(),
            "uncertainty_tolerance": str(ui_state.get("uncertainty_tolerance", "medium") or "medium").strip(),
            "evidence_style": str(ui_state.get("evidence_style", "mixed") or "mixed").strip(),
            "official_narrative_suspicion": str(ui_state.get("official_narrative_suspicion", "medium") or "medium").strip(),
            "openness_to_update": str(ui_state.get("openness_to_update", "medium") or "medium").strip(),
        },
        "topic_linked": _topic_linked_from_ui_state(ui_state),
        "flavor_only": {
            "age_group": str(ui_state.get("age_group", "") or "").strip(),
            "gender": str(ui_state.get("flavor_gender", "") or "").strip(),
            "ethnicity": str(ui_state.get("flavor_ethnicity", "") or "").strip(),
            "lifestyle_notes": str(ui_state.get("lifestyle_notes", "") or "").strip(),
            "tone_hint": str(ui_state.get("tone_hint", "") or "").strip(),
        },
        "render_config": {
            "use_core_causal": bool(ui_state.get("use_core_causal", True)),
            "use_topic_causal": bool(ui_state.get("use_topic_causal", False)),
            "use_topic_linked": bool(ui_state.get("use_topic_linked", False)),
            "use_flavor_only": bool(ui_state.get("use_flavor_only", False)),
            "topic_mode": str(ui_state.get("topic_mode", "off") or "off").strip(),
            "flavor_mode": str(ui_state.get("flavor_mode", "off") or "off").strip(),
            "manual_topic_override": bool(ui_state.get("manual_topic_override", False)),
            "manual_topic_causal_override": bool(ui_state.get("manual_topic_causal_override", False)),
            "manual_flavor_override": bool(ui_state.get("manual_flavor_override", False)),
            "show_profile_label": bool(ui_state.get("show_profile_label", True)),
            "render_style": str(ui_state.get("render_style", "structured_card") or "structured_card").strip(),
            "profile_distribution_mode": str(ui_state.get("profile_distribution_mode", "all") or "all").strip(),
        }
    }
    return profile


def profile_to_ui_state(profile: dict[str, Any]) -> dict[str, Any]:
    meta = profile.get("profile_meta", {})
    core = profile.get("core_causal", {})
    topic = _ensure_topic_keys(profile.get("topic_linked", {}))
    flavor = profile.get("flavor_only", {})
    render = profile.get("render_config", {})
    state = {
        "profile_id": str(meta.get("profile_id", "") or ""),
        "profile_name": str(meta.get("profile_name", "") or ""),
        "profile_description": str(meta.get("description", "") or ""),
        "profile_label_choice": str(core.get("epistemic_profile_label", "") or ""),
        "custom_profile_label": str(core.get("epistemic_profile_label", "") or ""),
        "epistemic_profile_label": str(core.get("epistemic_profile_label", "") or ""),
        "institutional_trust": str(core.get("institutional_trust", "medium") or "medium"),
        "uncertainty_tolerance": str(core.get("uncertainty_tolerance", "medium") or "medium"),
        "evidence_style": str(core.get("evidence_style", "mixed") or "mixed"),
        "official_narrative_suspicion": str(core.get("official_narrative_suspicion", "medium") or "medium"),
        "openness_to_update": str(core.get("openness_to_update", "medium") or "medium"),
        "occupation": str(topic.get("occupation", "") or ""),
        "education_level": str(topic.get("education_level", "") or ""),
        "training_style": str(topic.get("training_style", "") or ""),
        "domain_familiarity": str(topic.get("domain_familiarity", "") or ""),
        "topic_interest": str(topic.get("topic_interest", "") or ""),
        "prior_exposure": str(topic.get("prior_exposure", "") or ""),
        "custom_topic_notes": str(topic.get("custom_topic_notes", "") or ""),
        "topic_version": str(topic.get("topic_version", "") or ""),
        "topic_fieldset_label": str(topic.get("topic_fieldset_label", "") or ""),
        "topic_causal_profile_choice": str(topic.get("topic_causal_profile_choice", "") or topic.get("topic_profile_choice", "") or TOPIC_PROFILE_MANUAL),
        "topic_profile_choice": str(topic.get("topic_causal_profile_choice", "") or topic.get("topic_profile_choice", "") or TOPIC_PROFILE_MANUAL),
        "age_group": str(flavor.get("age_group", "") or ""),
        "flavor_gender": str(flavor.get("gender", "") or ""),
        "flavor_ethnicity": str(flavor.get("ethnicity", "") or ""),
        "lifestyle_notes": str(flavor.get("lifestyle_notes", "") or ""),
        "tone_hint": str(flavor.get("tone_hint", "") or ""),
        "use_core_causal": bool(render.get("use_core_causal", True)),
        "use_topic_causal": bool(render.get("use_topic_causal", False)),
        "use_topic_linked": bool(render.get("use_topic_linked", False)),
        "use_flavor_only": bool(render.get("use_flavor_only", False)),
        "topic_mode": str(render.get("topic_mode", "off") or "off"),
        "flavor_mode": str(render.get("flavor_mode", "off") or "off"),
        "show_profile_label": bool(render.get("show_profile_label", True)),
        "render_style": str(render.get("render_style", "structured_card") or "structured_card"),
        "profile_distribution_mode": str(render.get("profile_distribution_mode", "all") or "all"),
    }
    for key in [*TOPIC_SLOT_KEYS, *ALL_TOPIC_PERSONA_FIELD_KEYS]:
        state[key] = str(topic.get(key, "") or "")
    return state


def materialize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(profile)
    out["topic_linked"] = _ensure_topic_keys(out.get("topic_linked", {}))
    render = out.get("render_config", {})

    topic = out["topic_linked"]
    topic_version = str(topic.get("topic_version") or "generic")
    topic_root = topic_version_root(topic_version)

    if bool(render.get("use_topic_causal")):
        # Topic-causal profiles are selected deliberately, not sampled from the
        # topic-background auto generator.  Manual topic-causal edits only fill
        # blanks from the selected profile; otherwise the selected profile is authoritative.
        profile_choice = str(topic.get("topic_causal_profile_choice") or topic.get("topic_profile_choice") or TOPIC_PROFILE_MANUAL).strip()
        topic["topic_causal_profile_choice"] = profile_choice
        topic["topic_profile_choice"] = profile_choice  # legacy alias
        if profile_choice and profile_choice != TOPIC_PROFILE_MANUAL:
            profile_values = get_topic_profile(topic_root, profile_choice)
            manual = bool(render.get("manual_topic_causal_override"))
            for k, v in profile_values.items():
                if manual:
                    if not str(topic.get(k, "") or "").strip():
                        topic[k] = v
                else:
                    topic[k] = v
    else:
        for k in [*TOPIC_SLOT_KEYS, *ALL_TOPIC_PERSONA_FIELD_KEYS]:
            topic[k] = ""
        topic["topic_causal_profile_choice"] = ""
        topic["topic_profile_choice"] = ""

    if bool(render.get("use_topic_linked")):
        if str(render.get("topic_mode", "off")).strip().lower() == "auto":
            auto_topic = derive_topic_from_core(out.get("core_causal", {}))
            for k, v in auto_topic.items():
                if not out["topic_linked"].get(k):
                    out["topic_linked"][k] = v
    else:
        for k in [*BASE_TOPIC_LINKED_KEYS, "custom_topic_notes"]:
            topic[k] = ""

    if bool(render.get("use_flavor_only")):
        if str(render.get("flavor_mode", "off")).strip().lower() == "auto":
            auto_flavor = derive_flavor_from_core(out.get("core_causal", {}))
            for k, v in auto_flavor.items():
                if not out["flavor_only"].get(k):
                    out["flavor_only"][k] = v
    else:
        out["flavor_only"] = {k: "" for k in out.get("flavor_only", {})}

    if not bool(render.get("use_core_causal", True)):
        out["core_causal"] = {k: "" for k in out.get("core_causal", {})}
    return out


def age_from_age_group(rng: random.Random, age_group: str) -> int | str:
    grp = str(age_group or "").strip()
    if not grp:
        return ""
    lo, hi = AGE_GROUP_RANGES.get(grp, (30, 39))
    return rng.randint(lo, hi)


def _mini_background(name: str, core: dict[str, Any], topic: dict[str, Any], flavor: dict[str, Any]) -> str:
    occ = str(topic.get("occupation", "") or "").strip()
    edu = str(topic.get("education_level", "") or "").strip()
    inst = _titleize_enum(core.get("institutional_trust", ""))
    ev = _titleize_enum(core.get("evidence_style", ""))
    training = _titleize_enum(topic.get("training_style", ""))
    domain = _titleize_enum(topic.get("domain_familiarity", ""))
    prior = str(topic.get("prior_exposure", "") or "").strip()
    pieces = []
    if occ:
        desc = f"{name} works as a {occ}"
        if edu:
            desc += f" with {edu.replace('_', ' ')} training"
        pieces.append(desc + ".")
    else:
        pieces.append(f"{name} has a grounded everyday perspective.")
    lens = []
    if inst:
        lens.append(f"institutional trust is {inst.lower()}")
    if ev:
        lens.append(f"their evidence style is {ev.lower()}")
    if training:
        lens.append(f"their training style is {training.lower()}")
    if domain:
        lens.append(f"domain familiarity is {domain.lower()}")
    if lens:
        pieces.append("They usually reason through " + ", ".join(lens) + ".")
    if prior:
        pieces.append(prior[:1].upper() + prior[1:] + ("." if not prior.endswith(".") else ""))
    return " ".join(pieces)


def _norm_trait_value(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_low(value: Any) -> bool:
    return _norm_trait_value(value) in {"very_low", "low"}


def _is_high(value: Any) -> bool:
    return _norm_trait_value(value) in {"high", "very_high"}


def _topic_trait_sentence(topic_root: str, key: str, label: str, value: str) -> str:
    """Convert topic-causal variables into natural prompt-facing behavioral tendencies.

    Important:
    - Do not expose internal importance labels like [high]/[medium].
    - Do not say the agent already believes the claim.
    - Describe evidence-weighting tendencies only.
    """
    root = str(topic_root or "generic").strip().lower()
    k = str(key or "").strip()
    v = _norm_trait_value(value)
    pretty_label = str(label or k.replace("_", " ")).strip()

    if not v:
        return ""

    # -----------------------------
    # v130: simulation hypothesis
    # -----------------------------
    if root == "v130":
        if k == "computational_worldview":
            if _is_high(v):
                return "Naturally notices code, information, and system-design analogies, but still treats them as suggestive rather than proof."
            if _is_low(v):
                return "Does not automatically treat code or information analogies as evidence about reality."
            return "Finds computational analogies understandable, but not decisive by themselves."

        if k == "anthropic_reasoning_comfort":
            if _is_high(v):
                return "Is comfortable considering observer-count and reference-class arguments."
            if _is_low(v):
                return "Is skeptical of observer-count and reference-class arguments."
            return "Can follow anthropic reasoning, but does not give it automatic priority."

        if k == "future_technology_prior":
            if _is_high(v):
                return "Finds very advanced future technology plausible enough to consider in the argument."
            if _is_low(v):
                return "Is cautious about arguments that depend on very advanced future technology."
            return "Treats future-technology assumptions as plausible but uncertain."

        if k == "consciousness_intuition":
            if v == "computational":
                return "Is open to the idea that consciousness could be instantiated computationally."
            if v == "physicalist":
                return "Leans physicalist about consciousness and is cautious about simulated-mind claims."
            return "Is uncertain about consciousness, so consciousness-based arguments can move either way."

    # -----------------------------
    # v119: Twin Towers / 9-11
    # -----------------------------
    if root == "v119":
        if k == "geopolitical_motive_sensitivity":
            if _is_high(v):
                return "Pays close attention to war, surveillance, and geopolitical motive arguments."
            if _is_low(v):
                return "Does not treat motive arguments as enough without direct mechanism evidence."
            return "Considers geopolitical motives, but weighs them against evidence quality."

        if k == "conspiracy_coordination_prior":
            if _is_high(v):
                return "Finds coordinated insider plots plausible enough to evaluate if the mechanism is concrete."
            if _is_low(v):
                return "Is skeptical that large coordinated insider plots can stay coherent and hidden."
            return "Can consider coordination claims, but needs a plausible mechanism."

        if k == "anomaly_sensitivity":
            if _is_high(v):
                return "Unresolved anomalies carry significant weight in their reasoning."
            if _is_low(v):
                return "Does not treat isolated anomalies as enough to overturn a broader explanation."
            return "Notices anomalies, but checks whether they connect to the central claim."

    # -----------------------------
    # v52: moon landing
    # -----------------------------
    if root == "v52":
        if k == "cold_war_motive_sensitivity":
            if _is_high(v):
                return "Pays close attention to Cold War propaganda and prestige incentives."
            if _is_low(v):
                return "Does not treat Cold War motive arguments as enough without direct evidence."
            return "Can weigh Cold War motive arguments, but not as decisive alone."

        if k == "engineering_evidence_weight":
            if _is_high(v):
                return "Gives strong weight to engineering feasibility and technical consistency arguments."
            if _is_low(v):
                return "Does not rely heavily on technical-feasibility claims unless they are clearly explained."
            return "Considers engineering evidence important, but not the only factor."

        if k == "visual_anomaly_sensitivity":
            if _is_high(v):
                return "Photo and video anomalies strongly attract their attention."
            if _is_low(v):
                return "Does not treat visual anomalies as decisive unless they connect to a concrete mechanism."
            return "Notices visual anomalies, but checks whether they are central."

    # Generic fallback for unknown future topics.
    readable_value = _titleize_enum(str(value))
    return f"{pretty_label} is {readable_value.lower()}, so this shapes how strongly that evidence lens matters."


def _core_reasoning_lines(core: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    trust = _norm_trait_value(core.get("institutional_trust", ""))
    suspicion = _norm_trait_value(core.get("official_narrative_suspicion", ""))
    evidence = _norm_trait_value(core.get("evidence_style", ""))
    uncertainty = _norm_trait_value(core.get("uncertainty_tolerance", ""))
    openness = _norm_trait_value(core.get("openness_to_update", ""))

    if trust:
        if _is_high(trust):
            lines.append("Usually gives established institutions and official records some initial weight.")
        elif _is_low(trust):
            lines.append("Does not give institutions automatic credibility.")
        else:
            lines.append("Treats institutional sources as relevant but not final.")

    if suspicion:
        if _is_high(suspicion):
            lines.append("Is alert to polished or incomplete official narratives.")
        elif _is_low(suspicion):
            lines.append("Is less inclined to suspect hidden official deception without concrete evidence.")
        else:
            lines.append("Can consider official-narrative concerns when they are specific.")

    if evidence:
        if evidence == "concrete_first":
            lines.append("Prefers concrete details over broad claims.")
        elif evidence == "source_first":
            lines.append("Gives source reliability strong weight.")
        elif evidence == "coherence_first":
            lines.append("Focuses on whether the argument is internally coherent.")
        elif evidence == "intuition_first":
            lines.append("Often starts from what feels intuitively plausible, then checks the argument.")
        else:
            lines.append("Uses a mixed evidence style.")

    if uncertainty:
        if _is_high(uncertainty):
            lines.append("Can remain uncertain when the evidence is incomplete.")
        elif _is_low(uncertainty):
            lines.append("Prefers clear conclusions and dislikes unresolved ambiguity.")
        else:
            lines.append("Can tolerate some uncertainty, but still looks for a clear direction.")

    if openness:
        if _is_high(openness):
            lines.append("Can revise their view when a concrete point lands.")
        elif _is_low(openness):
            lines.append("Does not change view easily after a single point.")
        else:
            lines.append("Can update moderately when the reason is clear.")

    # ADR-006 Component 1: locus of control (Rotter 1966).
    # Only emit a sentence when the value is off the neutral default ("mixed"),
    # so unchanged personas produce byte-identical cards to pre-ADR-006.
    locus = _norm_trait_value(core.get("locus_of_control", ""))
    if locus == "internal":
        lines.append("Updates their view primarily in response to arguments and evidence.")
    elif locus == "external":
        lines.append("Updates their view primarily in response to what others around them are saying.")

    # ADR-006 Component 1: contrarianism (Flache 2017, repulsive influence).
    # Same principle: neutral "medium" default emits nothing.
    contra = _norm_trait_value(core.get("contrarianism", ""))
    if contra == "very_high":
        lines.append("Has a strong tendency to differ from the perceived majority.")
    elif contra == "high":
        lines.append("Has a mild tendency to differ from the perceived majority.")
    elif contra == "low":
        lines.append("Has a mild tendency to agree with the perceived majority.")
    elif contra == "very_low":
        lines.append("Has a strong tendency to agree with the perceived majority.")

    # Keep the card short. Bumped from 6 to 8 to accommodate the two ADR-006
    # additions when both are set to a non-default value; the card is still
    # capped so unlimited future additions cannot bloat the prompt.
    return lines[:8]


def render_persona_card(profile: dict[str, Any], agent_name: str, opinion: int) -> tuple[str, dict[str, Any]]:
    mat = materialize_profile(profile)
    core = mat.get("core_causal", {})
    topic = _ensure_topic_keys(mat.get("topic_linked", {}))
    flavor = mat.get("flavor_only", {})
    render = mat.get("render_config", {})

    use_core = bool(render.get("use_core_causal", True))
    use_topic_causal = bool(render.get("use_topic_causal", False))
    use_topic_background = bool(render.get("use_topic_linked", False))
    use_flavor = bool(render.get("use_flavor_only", False))

    lines = ["AGENT_PERSONA_CARD"]
    lines.append(f"Name: {agent_name}")
    lines.append(f"Belief: {belief_label(int(opinion))}")
    lines.append(f"Initial rating: {int(opinion)}")

    # -----------------------------
    # Core causal traits: prompt-facing, natural wording.
    # -----------------------------
    if use_core:
        profile_label = str(core.get("epistemic_profile_label", "") or "").strip()
        if render.get("show_profile_label") and profile_label:
            lines.append(f"Core profile: {profile_label}")

        core_lines = _core_reasoning_lines(core)
        if core_lines:
            lines.append("Core reasoning tendencies:")
            for item in core_lines:
                lines.append(f"- {item}")

    # -----------------------------
    # Topic-causal traits: natural behavioral lens, no internal field names,
    # no [high]/[medium] labels, no topic_version/fieldset metadata.
    # -----------------------------
    if use_topic_causal:
        topic_version = str(topic.get("topic_version") or "generic")
        topic_root = topic_version_root(topic_version)
        topic_profile = str(
            topic.get("topic_causal_profile_choice")
            or topic.get("topic_profile_choice")
            or ""
        ).strip()

        if topic_profile and topic_profile != TOPIC_PROFILE_MANUAL:
            lines.append(f"Topic lens: {topic_profile}")

        topic_rows = _topic_values_for_render(topic)
        tendency_lines: list[str] = []
        for key, label, _importance, value in topic_rows:
            sent = _topic_trait_sentence(topic_root, key, label, value)
            if sent and sent not in tendency_lines:
                tendency_lines.append(sent)

        if tendency_lines:
            lines.append("Topic-specific reasoning tendencies:")
            for sent in tendency_lines[:7]:
                lines.append(f"- {sent}")

    # -----------------------------
    # Topic-linked background: role/expertise/familiarity only.
    # -----------------------------
    if use_topic_background:
        role_parts = []
        if str(topic.get("occupation", "") or "").strip():
            role_parts.append(f"occupation: {topic['occupation']}")
        if str(topic.get("education_level", "") or "").strip():
            role_parts.append(f"education: {_titleize_enum(topic['education_level'])}")
        if str(topic.get("training_style", "") or "").strip():
            role_parts.append(f"training style: {_titleize_enum(topic['training_style'])}")

        familiarity_parts = []
        if str(topic.get("domain_familiarity", "") or "").strip():
            familiarity_parts.append(f"domain familiarity: {_titleize_enum(topic['domain_familiarity'])}")
        if str(topic.get("topic_interest", "") or "").strip():
            familiarity_parts.append(f"topic interest: {str(topic['topic_interest']).replace('_', ' ')}")
        if str(topic.get("prior_exposure", "") or "").strip():
            familiarity_parts.append(f"prior exposure: {topic['prior_exposure']}")

        if role_parts:
            lines.append("Relevant background:")
            lines.append("- " + "; ".join(role_parts) + ".")

        if familiarity_parts:
            if not role_parts:
                lines.append("Relevant background:")
            lines.append("- " + "; ".join(familiarity_parts) + ".")

        notes = str(topic.get("custom_topic_notes", "") or "").strip()
        if notes:
            lines.append(f"- Topic note: {notes}")

    # -----------------------------
    # Flavor: wording/tone only.
    # -----------------------------
    if use_flavor:
        flavor_parts = []
        if str(flavor.get("age_group", "") or "").strip():
            flavor_parts.append(f"age group: {flavor['age_group']}")
        if str(flavor.get("gender", "") or "").strip():
            flavor_parts.append(f"gender: {flavor['gender']}")
        if str(flavor.get("ethnicity", "") or "").strip():
            flavor_parts.append(f"ethnicity: {flavor['ethnicity']}")
        if str(flavor.get("tone_hint", "") or "").strip():
            flavor_parts.append(f"tone: {flavor['tone_hint']}")
        if str(flavor.get("lifestyle_notes", "") or "").strip():
            flavor_parts.append(f"lifestyle: {flavor['lifestyle_notes']}")

        if flavor_parts:
            lines.append("Presentation flavor:")
            lines.append("- " + "; ".join(flavor_parts) + ".")

    legacy = {
        "political_leaning": "",
        "age": "",
        "gender": str(flavor.get("gender", "") or "").strip() if use_flavor else "",
        "ethnicity": str(flavor.get("ethnicity", "") or "").strip() if use_flavor else "",
        "education": _titleize_enum(topic.get("education_level", "")) if use_topic_background else "",
        "occupation": str(topic.get("occupation", "") or "").strip() if use_topic_background else "",
        "early_life": "",
        "epistemic_profile": str(core.get("epistemic_profile_label", "") or "").strip() if use_core else "",
        "institutional_trust": _titleize_enum(core.get("institutional_trust", "")) if use_core else "",
        "uncertainty_tolerance": _titleize_enum(core.get("uncertainty_tolerance", "")) if use_core else "",
        "evidence_style": _titleize_enum(core.get("evidence_style", "")) if use_core else "",
        "official_narrative_suspicion": _titleize_enum(core.get("official_narrative_suspicion", "")) if use_core else "",
        "openness_to_update": _titleize_enum(core.get("openness_to_update", "")) if use_core else "",
    }

    return "\n".join(lines), legacy


def load_agent_names(names_source_csv: str, rng: random.Random, n_agents: int) -> list[str]:
    p = Path(names_source_csv)
    if not p.exists():
        return [f"agent_{i:02d}" for i in range(n_agents)]
    try:
        import pandas as pd
        df = pd.read_csv(p)
        cols = {c.strip().lower(): c for c in df.columns}
        name_col = None
        for cand in ["agent_name", "name", "agent", "id"]:
            if cand in cols:
                name_col = cols[cand]
                break
        if name_col is None:
            return [f"agent_{i:02d}" for i in range(n_agents)]
        names = [str(x).strip() for x in df[name_col].tolist() if str(x).strip()]
        if not names:
            return [f"agent_{i:02d}" for i in range(n_agents)]
        rng.shuffle(names)
        out = names[:n_agents]
        if len(out) < n_agents:
            out.extend([f"agent_{i:02d}" for i in range(len(out), n_agents)])
        return out
    except Exception:
        return [f"agent_{i:02d}" for i in range(n_agents)]


def _build_beliefs(n_agents: int, opinion_strategy: str, custom_counts: "dict[int, int] | None", rng: random.Random) -> list[int]:
    '''Initial opinions for n agents. custom_counts maps opinion value -> count
    (e.g. {-2: 5, ..., 2: 5}); a plain list would be misread (negative values
    would index from the end), so callers must pass the mapping form.'''
    opinion_strategy = (opinion_strategy or "uniform").strip().lower()
    if opinion_strategy == "uniform":
        base = n_agents // len(OPINION_VALUES)
        rem = n_agents % len(OPINION_VALUES)
        beliefs = []
        for v in OPINION_VALUES:
            beliefs.extend([v] * base)
        for i in range(rem):
            beliefs.append(OPINION_VALUES[i])
        rng.shuffle(beliefs)
        return beliefs
    if opinion_strategy == "skewed_positive":
        beliefs = [2] * (n_agents - n_agents // 5) + [-2] * (n_agents // 5)
        rng.shuffle(beliefs)
        return beliefs
    if opinion_strategy == "skewed_negative":
        beliefs = [-2] * (n_agents - n_agents // 5) + [2] * (n_agents // 5)
        rng.shuffle(beliefs)
        return beliefs
    if opinion_strategy == "positive":
        return [2] * n_agents
    if opinion_strategy == "negative":
        return [-2] * n_agents
    if opinion_strategy == "custom_counts":
        if not custom_counts:
            raise ValueError("custom_counts required")
        beliefs = []
        for v in OPINION_VALUES:
            beliefs.extend([v] * int(custom_counts[int(v)]))
        if len(beliefs) != n_agents:
            raise ValueError("Custom counts do not sum to n_agents")
        rng.shuffle(beliefs)
        return beliefs
    raise ValueError(f"Unknown opinion strategy: {opinion_strategy}")


def generate_csv_from_profile(
    out_csv_path: str,
    n_agents: int,
    seed: int,
    opinion_strategy: str,
    custom_counts,
    names_source_csv: str,
    profile: dict[str, Any],
    profile_distribution_mode: str = "all",
) -> None:
    rng = random.Random(seed)
    beliefs = _build_beliefs(n_agents, opinion_strategy, custom_counts, rng)
    names = load_agent_names(names_source_csv, rng, n_agents)
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = (profile_distribution_mode or materialize_profile(profile).get("render_config", {}).get("profile_distribution_mode", "all") or "all").strip().lower()
    pool = _profile_pool_for_mode(profile, mode)
    if mode == "equal":
        seq = []
        base = n_agents // max(1, len(pool))
        rem = n_agents % max(1, len(pool))
        for p_obj in pool:
            seq.extend([deepcopy(p_obj)] * base)
        for i in range(rem):
            seq.append(deepcopy(pool[i % len(pool)]))
        rng.shuffle(seq)
    elif mode == "random":
        seq = [deepcopy(rng.choice(pool)) for _ in range(n_agents)]
    elif mode in {"more", "most"}:
        selected = deepcopy(profile)
        others = [deepcopy(p) for p in pool[1:]] if len(pool) > 1 else [deepcopy(profile)]
        target_w = 0.55 if mode == "more" else 0.80
        other_w = (1.0 - target_w) / max(1, len(others))
        choices = [(selected, target_w)] + [(p_obj, other_w) for p_obj in others]
        seq = []
        for _ in range(n_agents):
            r = rng.random()
            acc = 0.0
            chosen = selected
            for p_obj, wgt in choices:
                acc += wgt
                if r <= acc:
                    chosen = deepcopy(p_obj)
                    break
            seq.append(chosen)
    else:
        seq = [deepcopy(profile) for _ in range(n_agents)]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        extra_topic_columns = _all_topic_linked_keys()
        usage_columns = ["use_core_causal", "use_topic_causal", "use_topic_linked", "use_flavor_only", "topic_mode", "flavor_mode", "manual_topic_override", "manual_topic_causal_override", "manual_flavor_override", "profile_distribution_mode"]
        w.writerow([
            "row_id", "agent_id", "agent_name", "belief", "opinion",
            "political_leaning", "age", "gender", "ethnicity", "education", "occupation", "early_life",
            "epistemic_profile", "institutional_trust", "uncertainty_tolerance", "evidence_style",
            "official_narrative_suspicion", "openness_to_update",
            *extra_topic_columns,
            *usage_columns,
            "background", "persona",
        ])
        for i in range(n_agents):
            opinion = int(beliefs[i])
            name = names[i]
            profile_i = seq[i]
            persona, legacy = render_persona_card(profile_i, name, opinion)
            mat = materialize_profile(profile_i)
            background = _mini_background(name, mat.get("core_causal", {}), mat.get("topic_linked", {}), mat.get("flavor_only", {}))
            age = age_from_age_group(rng, mat.get("flavor_only", {}).get("age_group", ""))
            topic_cols = _ensure_topic_keys(mat.get("topic_linked", {}))
            render_cfg = dict(mat.get("render_config", {}) or {})
            w.writerow([
                i + 1,
                i + 1,
                name,
                belief_label(opinion),
                opinion,
                legacy["political_leaning"],
                age,
                legacy["gender"],
                legacy["ethnicity"],
                legacy["education"],
                legacy["occupation"],
                legacy["early_life"],
                legacy["epistemic_profile"],
                legacy["institutional_trust"],
                legacy["uncertainty_tolerance"],
                legacy["evidence_style"],
                legacy["official_narrative_suspicion"],
                legacy["openness_to_update"],
                *[topic_cols.get(k, "") for k in extra_topic_columns],
                *[render_cfg.get(k, "") for k in usage_columns],
                background,
                persona,
            ])


def render_profile_preview(profile: dict[str, Any], example_name: str = "Example Agent", example_opinion: int = 0) -> str:
    persona, _ = render_persona_card(profile, example_name, example_opinion)
    return persona
