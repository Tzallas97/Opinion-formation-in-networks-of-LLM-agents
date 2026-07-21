"""Tkinter launcher for running and monitoring LLM opinion-dynamics experiments.

The launcher collects experiment settings, generates persona CSVs when requested,
builds the corresponding simulation/plotting command, runs it in a subprocess,
and displays logs and live progress. It intentionally keeps UI state, batch-run
configuration, persona-profile controls, RAG/web options, and network settings in
one file so local experiments can be launched without a separate configuration
front end.
"""

import os
# --- UTF-8 safe stdout/stderr on Windows (prevents UnicodeEncodeError) ---
import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import csv
import time
import json
import random
import threading
import subprocess
from pathlib import Path
import re

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def strip_ansi(s: str) -> str:
    """Remove terminal ANSI escape codes before displaying subprocess output in the Tkinter log view."""
    return ANSI_ESCAPE_RE.sub("", s)


import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from tkinter import filedialog, messagebox

from persona_profiles_support import (
    DEFAULT_PROFILES,
    DEFAULT_SCHEMA,
    build_profile_from_ui,
    ensure_config_files,
    extract_profile_id,
    generate_csv_from_profile,
    get_profile_by_id,
    get_preset_profile_by_name,
    list_profile_choices,
    load_profiles,
    load_schema,
    preset_profile_names,
    profile_to_ui_state,
    render_profile_preview,
    save_profiles,
    upsert_profile,

)

from topic_persona_fields import (
    TOPIC_SLOT_KEYS,
    ALL_TOPIC_PERSONA_FIELD_KEYS,
    ALL_TOPIC_PERSONA_FIELD_DEFS,
    get_topic_fieldset,
    get_topic_fields,
    get_topic_label,
    version_root as topic_version_root,
)
from topic_persona_profiles import (
    TOPIC_PROFILE_MANUAL,
    get_topic_profile,
    topic_profile_names,
)

# Keep a handle to the support-module generator so the launcher can post-process
# exported persona CSVs without breaking the existing profile sampling path.
_SUPPORT_GENERATE_CSV_FROM_PROFILE = generate_csv_from_profile

# Local UI-only schema extension. The topic field definitions/profiles now live
# in topic_persona_fields.py and topic_persona_profiles.py.
PERSONA_LEVEL_VALUES = ["", "low", "medium", "high"]
PERSONA_CONSCIOUSNESS_VALUES = ["", "physicalist", "uncertain", "computational"]
PERSONA_IMPORTANCE_RULE = (
    "Core causal and topic-causal traits guide what feels persuasive; "
    "topic-relevant role/expertise matters when it connects to the claim; "
    "flavor-only descriptors shape wording and tone, not belief by themselves."
)
TOPIC_ROLE_RULE = (
    "Occupation, education, and training are evidence-lens fields. Use them only "
    "when they are relevant to the current claim or evidence type."
)

def _topic_fields_from_state(state: dict) -> list[tuple[str, str, str, str]]:
    """Return (key,label,importance,value) for the selected topic fieldset/state."""
    topic_fields = state.get("topic_specific_fields") or {}
    topic_labels = state.get("topic_specific_labels") or {}
    topic_importance = state.get("topic_specific_importance") or {}
    rows = []
    for key, val in dict(topic_fields).items():
        v = _persona_state_value(val)
        if not v:
            continue
        meta = ALL_TOPIC_PERSONA_FIELD_DEFS.get(key, {})
        label = topic_labels.get(key) or meta.get("label") or key.replace("_", " ").title()
        imp = topic_importance.get(key) or meta.get("importance") or "medium"
        rows.append((key, label, imp, v))
    return rows

def _persona_state_value(value):
    """Normalize a persona UI value and treat blank/off as disabled."""
    s = str(value or '').strip()
    if not s or s.lower() == 'off':
        return ''
    return s


def _persona_state_label_lines(profile_state: dict | None) -> list[tuple[str, str, str]]:
    """Convert enabled persona-profile UI state into audit rows for generated persona CSVs."""
    state = dict(profile_state or {})
    rows: list[tuple[str, str, str]] = []

    def add(column: str, label: str, key: str, *, enabled: bool = True):
        """Small local helper used to update the surrounding function state."""
        if not enabled:
            return
        val = _persona_state_value(state.get(key, ''))
        if val:
            rows.append((column, label, val))

    use_core = bool(state.get('use_core_causal', False))
    use_topic_causal = bool(state.get('use_topic_causal', False))
    use_topic_background = bool(state.get('use_topic_linked', False))
    use_flavor = bool(state.get('use_flavor_only', False))
    use_expressive = bool(state.get('use_expressive_style', False))  # separate expressive/debate-style layer (validator permissions)
    show_label = bool(state.get('show_profile_label', True))

    if use_core or use_topic_causal or use_topic_background or use_flavor or use_expressive:
        rows.append(('persona_weighting_rule', 'Persona weighting rule', PERSONA_IMPORTANCE_RULE))
        rows.append(('persona_layers_enabled', 'Persona layers enabled', f"core={int(use_core)}; topic_causal={int(use_topic_causal)}; topic_background={int(use_topic_background)}; flavor={int(use_flavor)}; expressive={int(use_expressive)}"))

    add('epistemic_profile', 'Epistemic profile', 'epistemic_profile_label', enabled=(use_core and show_label))
    add('institutional_trust', 'Institutional trust', 'institutional_trust', enabled=use_core)
    add('uncertainty_tolerance', 'Uncertainty tolerance', 'uncertainty_tolerance', enabled=use_core)
    add('evidence_style', 'Evidence style', 'evidence_style', enabled=use_core)
    add('official_narrative_suspicion', 'Official-narrative suspicion', 'official_narrative_suspicion', enabled=use_core)
    add('openness_to_update', 'Openness to update', 'openness_to_update', enabled=use_core)
    # Expressive/debate-style layer (validator permissions only, not causal). social_conformity retired (inert).
    add('value_orientation', 'Value orientation', 'value_orientation', enabled=use_expressive)
    add('agency_vs_fatalism', 'Agency vs fatalism', 'agency_vs_fatalism', enabled=use_expressive)
    add('conflict_style', 'Conflict style', 'conflict_style', enabled=use_expressive)

    if use_topic_causal:
        fieldset_label = state.get('topic_fieldset_label') or get_topic_label(state.get('topic_version', 'generic'))
        if fieldset_label:
            rows.append(('topic_fieldset', 'Topic-causal fieldset', str(fieldset_label)))
        causal_profile = _persona_state_value(state.get('topic_causal_profile_choice') or state.get('topic_profile_choice'))
        if causal_profile and causal_profile != TOPIC_PROFILE_MANUAL:
            rows.append(('topic_causal_profile_choice', 'Topic-causal profile', causal_profile))
        for key, label, importance, value in _topic_fields_from_state(state):
            rows.append((key, f'Topic-causal trait ({importance}) - {label}', value))

    if use_topic_background:
        rows.append(('topic_role_rule', 'Topic role/expertise rule', TOPIC_ROLE_RULE))
    add('occupation', 'Occupation', 'occupation', enabled=use_topic_background)
    add('education_level', 'Education level', 'education_level', enabled=use_topic_background)
    add('training_style', 'Training style', 'training_style', enabled=use_topic_background)
    add('domain_familiarity', 'Domain familiarity', 'domain_familiarity', enabled=use_topic_background)
    add('topic_interest', 'Topic interest', 'topic_interest', enabled=use_topic_background)
    add('prior_exposure', 'Prior exposure', 'prior_exposure', enabled=use_topic_background)

    add('age_group', 'Age group', 'age_group', enabled=use_flavor)
    add('flavor_gender', 'Gender', 'flavor_gender', enabled=use_flavor)
    add('flavor_ethnicity', 'Ethnicity', 'flavor_ethnicity', enabled=use_flavor)
    add('lifestyle_notes', 'Lifestyle notes', 'lifestyle_notes', enabled=use_flavor)
    add('tone_hint', 'Tone hint', 'tone_hint', enabled=use_flavor)
    return rows



INTERNAL_PERSONA_AUDIT_LINE_PREFIXES = (
    # UI/audit metadata. Keep as CSV columns only.
    "persona weighting rule:",
    "persona layers enabled:",
    "topic-causal fieldset:",
    "topic fieldset:",
    "topic-causal profile:",
    "topic profile:",
    "topic version:",
    "topic fields:",
    "topic-causal trait",
    "topic causal trait",
    "topic-causal traits (",
    "topic role/expertise rule:",
    # Raw core-field audit rows. The clean card already renders these as
    # natural `Core reasoning tendencies`, so direct field labels are duplicate.
    "epistemic profile:",
    "institutional trust:",
    "uncertainty tolerance:",
    "evidence style:",
    "official-narrative suspicion:",
    "official narrative suspicion:",
    "openness to update:",
    "value orientation:",
    "agency vs fatalism:",
    "conflict style:",
    # Raw topic-background/flavor audit rows. Clean card uses sectioned natural lines.
    "occupation:",
    "education level:",
    "education:",
    "training style:",
    "domain familiarity:",
    "topic interest:",
    "prior exposure:",
    "age group:",
    "gender:",
    "ethnicity:",
    "lifestyle notes:",
    "tone hint:",
)


def _audit_field_labels_for_stripping() -> set[str]:
    """Known raw label names that should not be appended into AGENT_PERSONA_CARD.

    The actual values remain in CSV columns. This only strips duplicate/schema-like
    prompt lines such as `- Computational worldview: medium`.
    """
    labels = {
        "epistemic profile", "institutional trust", "uncertainty tolerance",
        "evidence style", "official-narrative suspicion", "official narrative suspicion",
        "openness to update", "value orientation",
        "agency vs fatalism", "conflict style", "topic version", "topic fields",
        "topic fieldset", "topic-causal fieldset", "topic-causal profile",
        "topic profile", "occupation", "education", "education level",
        "training style", "domain familiarity", "topic interest", "prior exposure",
        "age group", "gender", "ethnicity", "lifestyle notes", "tone hint",
    }
    try:
        for key, meta in (ALL_TOPIC_PERSONA_FIELD_DEFS or {}).items():
            labels.add(str(key or '').replace('_', ' ').replace('-', ' ').strip().lower())
            labels.add(str(meta.get('label') or '').replace('_', ' ').replace('-', ' ').strip().lower())
    except Exception:
        pass
    return {x for x in labels if x}


def _strip_internal_persona_audit_lines(persona_lines: list[str]) -> list[str]:
    """Remove audit/schema rows from AGENT_PERSONA_CARD before it enters prompts.

    The audit data remains available as CSV columns. This function only keeps the
    prompt-facing persona text natural and compact.
    """
    cleaned: list[str] = []
    audit_labels = _audit_field_labels_for_stripping()

    for raw in persona_lines or []:
        line = str(raw or '').rstrip()
        if not line.strip():
            continue

        stripped = line.strip()
        low = stripped.lower()

        if any(low.startswith(prefix) for prefix in INTERNAL_PERSONA_AUDIT_LINE_PREFIXES):
            continue

        # Older exporter versions rendered raw importance bullets like "- [high] Trait: value".
        # Natural prompt bullets such as "- Finds ..." are preserved.
        if re.match(r"^[-*]\s*\[(?:high|medium|low)\]\s+", low):
            continue

        # Remove direct raw field rows such as:
        #   - Computational worldview: medium
        #   - Testability preference: high
        #   Institutional trust: Low
        # These are duplicate audit fields; the clean prompt already converts them
        # to natural reasoning tendencies.
        no_bullet = re.sub(r"^[-*]\s+", "", stripped).strip()
        label_part = no_bullet.split(":", 1)[0].strip().lower().replace("_", " ").replace("-", " ")
        label_part = re.sub(r"\s+", " ", label_part)
        if ":" in no_bullet and label_part in audit_labels:
            continue

        cleaned.append(line)
    return cleaned


def _augment_generated_persona_csv(out_csv_path: str, profile_state: dict | None) -> None:
    """Post-process exported persona CSVs with audit columns only.

    Important: enabled UI/profile fields are preserved as CSV columns for
    analysis, but they are no longer appended verbatim into AGENT_PERSONA_CARD.
    The prompt-facing persona should stay natural and compact; internal fields
    such as persona_layers_enabled, topic_fieldset, topic_version, and raw
    topic-causal trait labels belong in CSV columns, not in the LLM prompt.
    """
    p = Path(str(out_csv_path or '').strip())
    if not p.exists():
        return

    label_rows = _persona_state_label_lines(profile_state)

    try:
        with p.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
    except Exception:
        return

    if not rows:
        return

    # Keep audit fields as columns. Do not inject them into the persona text.
    for column, _label, _value in label_rows:
        if column not in fieldnames:
            fieldnames.append(column)
    if 'persona' not in fieldnames:
        fieldnames.append('persona')

    for row in rows:
        existing_persona = str(row.get('persona', '') or '').strip()
        if existing_persona:
            persona_lines = [ln.rstrip() for ln in existing_persona.splitlines() if str(ln).strip()]
        else:
            persona_lines = ['AGENT_PERSONA_CARD']
            name = str(row.get('agent_name', '') or '').strip()
            if name:
                persona_lines.append(f'Name: {name}')

        # Remove any internal/audit rows that may have been added by older UI versions.
        persona_lines = _strip_internal_persona_audit_lines(persona_lines)

        # Populate audit columns only. These remain available for metrics/plots.
        for column, _label, value in label_rows:
            if not str(row.get(column, '') or '').strip():
                row[column] = value

        row['persona'] = '\n'.join(persona_lines).strip()

    try:
        with p.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        return


# =============================
# CONFIG
# =============================

DEFAULT_SCRIPTS = [
    # Only scripts that actually exist and run. The former list referenced
    # several mains (v2, v5_network, v6_*, main_plot*, ...) whose files are
    # not in the repo, so selecting them made the launcher fail.
    "opinion_dynamics_test_network_qwen.py",  # main simulation
    "main_plot_network.py",                   # network plotting / animation
]

DEFAULT_MODELS = [
    # Current recommended local/Ollama options for this project.
    # Keep qwen3:8b first because it is the stable baseline for comparisons.
    "qwen3:8b",
    "none",
]


# Model thinking capability (keep in sync with MODEL_PROFILES in the simulation script).
# Only EXPLICITLY-known no-thinking models get their Think mode disabled; unknown models
# stay enabled so a new/capable model is never wrongly restricted.
_THINKING_MODELS = ("qwen3", "qwq", "deepseek-r1")
_NO_THINKING_MODELS = ()  # e.g. ("llama3", "qwen2.5") to disable Think mode for those


def _model_thinking_capability(model):
    """True (known-thinking), False (known-no-thinking), or None (unknown -> unrestricted)."""
    m = str(model or "").strip().lower()
    if any(k in m for k in _NO_THINKING_MODELS):
        return False
    if any(k in m for k in _THINKING_MODELS):
        return True
    return None

# Prompt folders follow: prompts/opinion_dynamics/Flache_2017/vXX/vXX_<mode>/
DEFAULT_PROMPT_VERSIONS = ["v37", "v52", "v113","v119", "v130"]
DEFAULT_PROMPT_MODES = [
    "default",
    "default_reverse",
    "confirmation_bias",
    "confirmation_bias_reverse",
    "strong_confirmation_bias",
    "strong_confirmation_bias_reverse",
    "control",
    "control_reverse",
]

def _find_project_root(start: Path) -> Path:
    """Best-effort project root discovery without breaking existing folder layouts."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for base in candidates:
        if (base / "prompts" / "opinion_dynamics" / "Flache_2017").exists():
            return base
    for base in candidates:
        if (base / "scripts").exists() and (base / "prompts").exists():
            return base
    return start


LAUNCHER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(LAUNCHER_DIR)
PROMPT_ROOT = PROJECT_ROOT / "prompts" / "opinion_dynamics" / "Flache_2017"


def _detect_prompt_versions():
    """Prompt version folders (vNN) actually present under PROMPT_ROOT, sorted
    numerically. Falls back to DEFAULT_PROMPT_VERSIONS if the scan finds nothing."""
    try:
        found = [p.name for p in PROMPT_ROOT.iterdir()
                 if p.is_dir() and re.fullmatch("v[0-9]+", p.name)]
        found.sort(key=lambda v: int(v[1:]))
        return found or list(DEFAULT_PROMPT_VERSIONS)
    except Exception:
        return list(DEFAULT_PROMPT_VERSIONS)


PROMPT_VERSIONS = _detect_prompt_versions()
CONFIG_DIR = PROJECT_ROOT / "scripts" / "config"
PERSONA_SCHEMA_PATH = CONFIG_DIR / "persona_schema.json"
PERSONA_PROFILES_PATH = CONFIG_DIR / "persona_profiles.json"
DEFAULT_FACT_PACK_MODES = ["off", "core", "context", "criticisms", "balanced", "full"]

# Persist UI defaults across runs
def _resolve_settings_file() -> Path:
    """Keep using the user's existing settings file if one already exists."""
    candidates = []
    seen = set()
    for c in [LAUNCHER_DIR / ".launcher_settings.json", Path.cwd() / ".launcher_settings.json"]:
        try:
            rc = c.resolve()
        except Exception:
            rc = c
        key = str(rc)
        if key not in seen:
            seen.add(key)
            candidates.append(rc)

    existing = [c for c in candidates if c.exists()]
    if existing:
        try:
            return max(existing, key=lambda x: x.stat().st_mtime)
        except Exception:
            return existing[0]
    return candidates[0]


SETTINGS_FILE = _resolve_settings_file()

DEFAULT_SETTINGS = {
    "script": DEFAULT_SCRIPTS[0] if DEFAULT_SCRIPTS else "",
    "seed": "50",
    "agents": "25",
    "steps": "150",
    "out": "seed50",
    "prompt_version": PROMPT_VERSIONS[0] if PROMPT_VERSIONS else "v52",
    "prompt_mode": DEFAULT_PROMPT_MODES[0] if DEFAULT_PROMPT_MODES else "default",
    "fact_pack_mode": "off",
    "world": "closed",
    "memory": "enabled",
    "light_memory_threshold": "3",
    "trace": "auto",        # auto|minimal|full|off
    "llm_history": "off",   # off|auto|on (history injected into prompts)
    "debug_native_thinking_on_fail": "off",  # off|on (save native shadow thinking on hard failures, if script supports it)

    # Retrieval / RAG (used only when world=open_rag)
    "rag_backend": "off",
    "rag_corpus_path": "",
    "rag_top_k": "4",
    "rag_query_mode": "claim",
    "rag_query_mode_step2": "auto",
    "rag_query_mode_step3": "claim_plus_tweet",
    "rag_content_mode": "full",
    "rag_topic_override": "",
    "rag_max_chars": "2200",

    # True-open live tools
    "web_backend": "brave",
    "web_top_k": "3",
    "web_max_chars": "1400",
    "step2_web_mode": "heuristic",
    "planner_mode": "off",
    "tool_mode": "web_only",
    "notes_mode": "off",
    "notes_max_items": "3",

    # Multiple runs (batch): off | same_seed | consecutive_seeds
    "multi_run_mode": "off",
    "multi_run_count": "5",

    "model": DEFAULT_MODELS[0] if DEFAULT_MODELS else "qwen3:8b",
    "think_mode": "off",

    # LLM decoding params (passed to opinion_dynamics* scripts that support them)
    "temperature": "0.7",
    "top_p": "0.9",
    "top_k": "40",
    "repeat_penalty": "1.1",
    "repeat_last_n": "64",
    "max_tokens": "300",
    "min_p": "",
    "presence_penalty": "",
    "frequency_penalty": "",
    "max_step_change": "1",
    "allowed_update_mode": "assimilation_only",
    "validation_strictness": "strict",
    "allow_silence": "off",
    "wrong_side_explanation_requery": "off",
    "deterministic": "off",
    "structured_output": "off",
    "solo_check": "off",
    "same_side_edge_unlock_hits": "2",
    "same_rating_step3_mode": "skip_tweet_local",
    # ADR-006 Component 3: p_reach pluggable policy pattern (feeds P21, P30).
    "p_reach_policy": "uniform",
    "p_reach_uniform_value": "1.0",
    "p_reach_homophily_k": "2.0",
    "p_reach_shadowban_fraction": "0.1",
    "p_reach_shadowban_value": "0.1",
    "p_reach_enforcement": "filter",
    # ADR-006 Component 4: Bian diagnostic opt-in.
    "include_bian_scores": "off",

    # Optional per-step LLM overrides (blank = inherit global)
    "temperature_step2": "",
    "top_p_step2": "",
    "top_k_step2": "",
    "repeat_penalty_step2": "",
    "repeat_last_n_step2": "",
    "max_tokens_step2": "",
    "min_p_step2": "",
    "presence_penalty_step2": "",
    "frequency_penalty_step2": "",

    "temperature_step3": "",
    "top_p_step3": "",
    "top_k_step3": "",
    "repeat_penalty_step3": "",
    "repeat_last_n_step3": "",
    "max_tokens_step3": "",
    "min_p_step3": "",
    "presence_penalty_step3": "",
    "frequency_penalty_step3": "",

    # Network / retrieval params
    "network_type": "ws",   # ws | er | ba | none
    "network_homophily": False,
    "k_neighbors": "4",
    "p_rewire": "0.1",
    "er_p_edge": "0.15",
    "ba_m_attach": "2",
    "ba_hub_strategy": "default",
    "ba_hub_assignment_mode": "early_position",
    "ba_hub_custom": "",
    "interaction_selection": "homophily",
    "interaction_homophily_mode": "full",
    "use_personas": False,
    "opinion_dist": "uniform",
    "custom_counts": "5,5,5,5,5",
    "names_source": "list_agent_descriptions.csv",

    "use_json_persona_profiles": False,
    "persona_selected_profile": "",
    "persona_profile_name": "Custom profile",
    "persona_profile_description": "",
    "persona_use_core_causal": True,
    "persona_use_topic_causal": False,
    "persona_use_topic_linked": False,
    "persona_use_flavor_only": False,
    # Expressive/debate-style layer: validator permissions only (outcome/fatalistic/combative wording); off by default so it never confounds the causal comparison.
    "persona_use_expressive_style": False,
    "persona_topic_mode": "off",
    "persona_topic_causal_profile_choice": "Manual",
    "persona_topic_profile_choice": "Manual",  # compatibility alias for topic-causal profile
    "persona_flavor_mode": "off",
    "persona_show_profile_label": True,
    "persona_render_style": "structured_card",
    "persona_epistemic_profile_label": "",
    "persona_institutional_trust": "medium",
    "persona_uncertainty_tolerance": "medium",
    "persona_evidence_style": "mixed",
    "persona_official_narrative_suspicion": "medium",
    "persona_openness_to_update": "medium",
    "persona_locus_of_control": "mixed",
    "persona_contrarianism": "medium",
    "persona_value_orientation": "balanced",
    "persona_agency_vs_fatalism": "balanced",
    "persona_conflict_style": "balanced",
    "persona_occupation": "",
    "persona_education_level": "",
    "persona_training_style": "",
    "persona_domain_familiarity": "",
    "persona_topic_interest": "",
    "persona_prior_exposure": "",
    "persona_computational_worldview": "",
    "persona_testability_preference": "",
    "persona_anthropic_reasoning_comfort": "",
    "persona_future_technology_prior": "",
    "persona_consciousness_intuition": "",
    "persona_metaphysical_speculation_tolerance": "",
    "persona_custom_topic_notes": "",
    "persona_age_group": "",
    "persona_flavor_gender": "",
    "persona_flavor_ethnicity": "",
    "persona_lifestyle_notes": "",
    "persona_tone_hint": "",
    "persona_topic_manual_override": False,
    "persona_topic_causal_manual_override": False,
    "persona_flavor_manual_override": False,

    "age_preset": "equal",
    "gender_preset": "equal",
    "educ_preset": "equal",
    "pol_preset": "equal",
    "eth_preset": "random",
    "occ_preset": "random",
    "early_life_preset": "random",
    "epistemic_profile_preset": "none",
    "persona_profile_label_choice": "Institutionally trusting pragmatist",
    "persona_custom_profile_label": "",
    "persona_profile_distribution_mode": "all",
}

OPINION_VALUES = [-2, -1, 0, 1, 2]

# Categories as used by network_models.py
POLITICAL_CATS = ["far_left", "left", "center_left", "center", "center_right", "right", "far_right"]
EDUCATION_CATS = ["high_school", "some_college", "bachelor", "master", "phd"]
GENDER_CATS = ["male", "female"]

# These are not mapped in network_models.py, but it compares string equality.
ETHNICITY_CATS = ["caucasian", "african_american", "hispanic", "asian_american", "other"]
OCCUPATION_CATS = ["teacher", "engineer", "nurse", "lawyer", "retail", "unemployed", "student", "manager"]

AGE_BINS = [
    ("18-25", 18, 25),
    ("26-35", 26, 35),
    ("36-50", 36, 50),
    ("51-65", 51, 65),
]

# =============================
# Weighted sampling helpers
# =============================

def _weighted_choice(rng: random.Random, items_with_weights: list[tuple[str, float]]) -> str:
    """Sample one item according to explicit numeric weights using the supplied deterministic RNG."""
    total = sum(w for _, w in items_with_weights)
    if total <= 0:
        raise ValueError("Weights must sum to > 0.")
    r = rng.random() * total
    acc = 0.0
    for item, w in items_with_weights:
        acc += w
        if r <= acc:
            return item
    return items_with_weights[-1][0]


def _expand_counts_to_list(value_to_count: dict[int, int]) -> list[int]:
    """Expand a mapping from opinion value to count into a concrete list of initial beliefs."""
    out: list[int] = []
    for v, c in value_to_count.items():
        out.extend([int(v)] * int(c))
    return out


def _build_beliefs(n_agents: int, opinion_strategy: str, custom_counts, rng: random.Random) -> list[int]:
    """Construct initial belief values for the persona CSV under equal, random, or custom-count strategies."""
    if opinion_strategy == "equal_bins":
        base = n_agents // len(OPINION_VALUES)
        rem = n_agents % len(OPINION_VALUES)
        beliefs: list[int] = []
        for v in OPINION_VALUES:
            beliefs.extend([v] * base)
        for i in range(rem):
            beliefs.append(OPINION_VALUES[i])
        rng.shuffle(beliefs)
        return beliefs

    if opinion_strategy == "random":
        return [rng.choice(OPINION_VALUES) for _ in range(n_agents)]

    if opinion_strategy == "custom_counts":
        if not custom_counts:
            raise ValueError("custom_counts required for custom_counts strategy.")
        beliefs = _expand_counts_to_list(custom_counts)
        if len(beliefs) != n_agents:
            raise ValueError(f"Custom counts sum to {len(beliefs)} but n_agents is {n_agents}.")
        rng.shuffle(beliefs)
        return beliefs

    raise ValueError(f"Unknown opinion_strategy: {opinion_strategy}")



def _build_opinions_v5_dist(n_agents: int, dist: str, rng: random.Random) -> list[int]:
    """
    Mirrors opinion_dynamics_v5_network.initialize_opinion_distribution() semantics for list_opinion_space=[-2,-1,0,1,2],
    but implemented without numpy so the launcher can also generate a consistent CSV.
    """
    dist = (dist or "uniform").strip().lower()
    space = OPINION_VALUES[:]  # [-2,-1,0,1,2]
    max_op = max(space)
    min_op = min(space)
    multiple = n_agents // 5

    if dist == "uniform":
        opinions = space * multiple
        # If n_agents not divisible by 5, fill remainder deterministically
        rem = n_agents - len(opinions)
        if rem > 0:
            opinions += space[:rem]
    elif dist == "skewed_positive":
        opinions = [max_op] * (n_agents - multiple) + [min_op] * multiple
    elif dist == "skewed_negative":
        opinions = [min_op] * (n_agents - multiple) + [max_op] * multiple
    elif dist == "positive":
        opinions = [max_op] * n_agents
    elif dist == "negative":
        opinions = [min_op] * n_agents
    else:
        raise ValueError(f"Unknown v5 distribution: {dist}")

    rng.shuffle(opinions)
    return opinions


# =============================
# Distribution presets (UI choices)
# =============================

def age_bin_weights(preset: str) -> list[tuple[str, float]]:
    """Return age-bin sampling weights for a selected demographic preset."""
    preset = preset.strip().lower()
    bins = [b[0] for b in AGE_BINS]
    if preset in ("equal", "random"):
        w = 1.0 / len(bins)
        return [(b, w) for b in bins]
    if preset == "younger":
        return [(bins[0], 0.40), (bins[1], 0.30), (bins[2], 0.20), (bins[3], 0.10)]
    if preset == "older":
        return [(bins[0], 0.10), (bins[1], 0.20), (bins[2], 0.30), (bins[3], 0.40)]
    if preset == "all_young":
        return [(bins[0], 1.0)]
    if preset == "all_old":
        return [(bins[-1], 1.0)]
    raise ValueError(f"Unknown age preset: {preset}")


def uniform_choice(rng: random.Random, items: list[str], preset: str) -> str:
    # currently equal/random are equivalent (uniform draw)
    """Draw uniformly from a categorical list; the preset argument is retained for API symmetry."""
    return rng.choice(items)


def age_from_bin(rng: random.Random, bin_label: str) -> int:
    """Sample a concrete age from a named age bin."""
    for name, lo, hi in AGE_BINS:
        if name == bin_label:
            return rng.randint(lo, hi)
    # fallback
    return rng.randint(18, 65)


def age_group_from_age(age: int) -> int:
    # matches network_models.py _age_group
    """Map a concrete age to the age-group index used by network homophily helpers."""
    if age < 18:
        return 0
    elif age <= 25:
        return 1
    elif age <= 35:
        return 2
    elif age <= 50:
        return 3
    else:
        return 4


# =============================
# Personas CSV generation (NEW EACH RUN, overwrite same name)
# =============================

def load_agent_names(names_source_csv: str, rng: random.Random, n_agents: int) -> list[str]:
    """
    Use names from an existing CSV (only names), but do not reuse other attributes.
    """
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
        # If not enough, extend with generated names
        out = names[:n_agents]
        if len(out) < n_agents:
            out.extend([f"agent_{i:02d}" for i in range(len(out), n_agents)])
        return out
    except Exception:
        return [f"agent_{i:02d}" for i in range(n_agents)]


def generate_list_agent_descriptions_csv(
    out_csv_path: str,
    n_agents: int,
    seed: int,
    opinion_strategy: str,
    custom_counts: str,
    names_source_csv: str,
    age_preset: str,
    gender_preset: str,
    education_preset: str,
    political_preset: str,
    ethnicity_preset: str,
    occupation_preset: str,
    early_life_preset: str,
    epistemic_profile_preset: str,
) -> None:
    """
    Create list_agent_descriptions.csv in the selected prompt folder.

    Columns:
        row_id, agent_id, agent_name, belief, opinion,
        political_leaning, age, gender, ethnicity, education, occupation, early_life,
        background, persona

    Notes:
    - If any preset is 'none', that field is left empty.
    - 'persona' is written as a compact multi-line AGENT_PERSONA_CARD (no communication-style line).
    - 'background' is a short mini-biography derived from available fields (not a field list).
    """
    rng = random.Random(seed)

    # Opinions: either v5 dist presets or explicit custom_counts
    if opinion_strategy in {"uniform", "skewed_positive", "skewed_negative", "positive", "negative"}:
        beliefs_numeric = _build_opinions_v5_dist(n_agents, opinion_strategy, rng)
    else:
        beliefs_numeric = _build_beliefs(n_agents, opinion_strategy, custom_counts, rng)

    names = load_agent_names(names_source_csv, rng, n_agents)

    # --- Category vocab in the text form the simulator's CSV parser accepts ---
    POL_TXT = [
        "Far Left",
        "Strong Democrat",
        "Lean Democrat",
        "Independent",
        "Lean Republican",
        "Strong Republican",
        "Far Right",
    ]

    EDU_TXT = [
        "High School",
        "Some College",
        "Bachelor's Degree",
        "Master's Degree",
        "PhD",
    ]

    GENDER_TXT = ["Male", "Female"]

    ETH_TXT = [
        "Caucasian",
        "Black",
        "African American",
        "Asian",
        "Asian American",
        "Latino",
        "Hispanic",
        "Middle Eastern",
        "Native American",
        "Mixed",
        "Other",
    ]

    OCC_TXT = [
        # General
        "Teacher",
        "Engineer",
        "Nurse",
        "Lawyer",
        "Retail Worker",
        "Unemployed",
        "Student",
        "Manager",
        # Opinion-relevant archetypes
        "Doctor",
        "Historian",
        "Mathematician",
        "Computer Engineer",
        "Politician",
        "Journalist",
        "Scientist",
        "Economist",
        "Social Worker",
    ]

    EARLY_LIFE_TXT = ["Difficult", "Okay", "Good"]

    EPISTEMIC_ARCHETYPES = {
        "trusting_pragmatist": {
            "label": "Institutionally trusting pragmatist",
            "institutional_trust": "High",
            "uncertainty_tolerance": "Medium",
            "evidence_style": "Concrete-first",
            "official_narrative_suspicion": "Low",
            "openness_to_update": "Medium",
            "background": "They usually trust established institutions unless a clear contradiction appears, and they are most persuaded by concrete details.",
        },
        "suspicious_skeptic": {
            "label": "Suspicious anti-institutional skeptic",
            "institutional_trust": "Low",
            "uncertainty_tolerance": "Low",
            "evidence_style": "Coherence-first",
            "official_narrative_suspicion": "High",
            "openness_to_update": "Low",
            "background": "They are quick to notice anomalies, distrust polished official stories, and do not revise their view easily.",
        },
        "open_minded_skeptic": {
            "label": "Open-minded skeptical reviser",
            "institutional_trust": "Low",
            "uncertainty_tolerance": "High",
            "evidence_style": "Coherence-first",
            "official_narrative_suspicion": "High",
            "openness_to_update": "High",
            "background": "They are suspicious of official stories, but they can still change their mind if a tweet gives a clear, coherent point.",
        },
        "authority_stabilizer": {
            "label": "Authority-leaning stabilizer",
            "institutional_trust": "High",
            "uncertainty_tolerance": "Low",
            "evidence_style": "Source-first",
            "official_narrative_suspicion": "Low",
            "openness_to_update": "Low",
            "background": "They rely heavily on established records and do not move easily unless the case feels unusually strong.",
        },
        "uncertainty_tolerant_agnostic": {
            "label": "Uncertainty-tolerant agnostic",
            "institutional_trust": "Medium",
            "uncertainty_tolerance": "High",
            "evidence_style": "Concrete-first",
            "official_narrative_suspicion": "Medium",
            "openness_to_update": "Medium",
            "background": "They are comfortable staying unsure when information is limited and prefer concrete points over broad claims.",
        },
        "practical_distruster": {
            "label": "Low-trust practical realist",
            "institutional_trust": "Low",
            "uncertainty_tolerance": "Medium",
            "evidence_style": "Concrete-first",
            "official_narrative_suspicion": "Medium",
            "openness_to_update": "Medium",
            "background": "They are wary of institutions, but concrete physical details still matter to them more than vague suspicion.",
        },
    }
    EPISTEMIC_ORDER = list(EPISTEMIC_ARCHETYPES.keys())

    def belief_label(op: int) -> str:
        """Convert a numeric belief value into the text label stored in persona CSVs."""
        if op <= -2:
            return "Strongly Negative"
        if op == -1:
            return "Negative"
        if op == 0:
            return "Neutral"
        if op == 1:
            return "Positive"
        return "Strongly Positive"

    # --- Samplers ---
    def sample_political_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (political_preset or "").strip().lower()
        if preset == "none":
            return ""
        if preset == "all_left":
            return "Far Left"
        if preset == "all_right":
            return "Far Right"

        if preset in ("most_left", "more_left"):
            weights = [0.18, 0.32, 0.20, 0.12, 0.10, 0.06, 0.02]
        elif preset in ("most_right", "more_right"):
            weights = [0.02, 0.06, 0.10, 0.12, 0.20, 0.32, 0.18]
        elif preset == "centered":
            weights = [0.06, 0.12, 0.16, 0.32, 0.16, 0.12, 0.06]
        else:
            # equal/random (and any unknown)
            weights = [1 / 7] * 7
        return _weighted_choice(rng, list(zip(POL_TXT, weights)))

    def sample_education_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (education_preset or "").strip().lower()
        if preset == "none":
            return ""
        if preset == "all_educated":
            return "PhD"
        if preset == "all_uneducated":
            return "High School"
        if preset == "more_educated":
            weights = [0.10, 0.15, 0.25, 0.30, 0.20]
        elif preset == "less_educated":
            weights = [0.30, 0.30, 0.25, 0.12, 0.03]
        else:
            weights = [0.20] * 5
        return _weighted_choice(rng, list(zip(EDU_TXT, weights)))

    def sample_gender_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (gender_preset or "").strip().lower()
        if preset == "none":
            return ""
        if preset == "all_male":
            return "Male"
        if preset == "all_female":
            return "Female"
        if preset == "more_male":
            weights = [0.70, 0.30]
        elif preset == "more_female":
            weights = [0.30, 0.70]
        else:
            weights = [0.50, 0.50]  # equal/random
        return _weighted_choice(rng, list(zip(GENDER_TXT, weights)))

    def sample_ethnicity_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (ethnicity_preset or "").strip().lower()
        if preset == "none":
            return ""

        key_to_label = {
            "caucasian": "Caucasian",
            "black": "Black",
            "african_american": "African American",
            "asian": "Asian",
            "asian_american": "Asian American",
            "latino": "Latino",
            "hispanic": "Hispanic",
            "middle_eastern": "Middle Eastern",
            "native_american": "Native American",
            "mixed": "Mixed",
            "other": "Other",
        }

        if preset.startswith("all_"):
            k = preset.replace("all_", "", 1)
            return key_to_label.get(k, rng.choice(ETH_TXT))

        if preset.startswith("more_") or preset.startswith("most_"):
            k = preset.replace("more_", "", 1).replace("most_", "", 1)
            target = key_to_label.get(k)
            if not target or target not in ETH_TXT:
                return rng.choice(ETH_TXT)
            target_w = 0.55 if preset.startswith("more_") else 0.80
            rest_w = (1.0 - target_w) / max(1, (len(ETH_TXT) - 1))
            items = [(lab, target_w if lab == target else rest_w) for lab in ETH_TXT]
            return _weighted_choice(rng, items)

        return rng.choice(ETH_TXT)  # equal/random

    def sample_occupation_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (occupation_preset or "").strip().lower()
        if preset == "none":
            return ""

        key_to_label = {
            "unemployed": "Unemployed",
            "doctors": "Doctor",
            "historians": "Historian",
            "mathematicians": "Mathematician",
            "computer_engineers": "Computer Engineer",
            "politicians": "Politician",
            "journalists": "Journalist",
            "scientists": "Scientist",
            "lawyers": "Lawyer",
            "economists": "Economist",
            "social_workers": "Social Worker",
        }

        if preset.startswith("all_"):
            k = preset.replace("all_", "", 1)
            return key_to_label.get(k, rng.choice(OCC_TXT))

        if preset.startswith("more_") or preset.startswith("most_"):
            k = preset.replace("more_", "", 1).replace("most_", "", 1)
            target = key_to_label.get(k)
            if not target or target not in OCC_TXT:
                return rng.choice(OCC_TXT)
            target_w = 0.55 if preset.startswith("more_") else 0.80
            rest_w = (1.0 - target_w) / max(1, (len(OCC_TXT) - 1))
            items = [(lab, target_w if lab == target else rest_w) for lab in OCC_TXT]
            return _weighted_choice(rng, items)

        return rng.choice(OCC_TXT)  # equal/random

    def sample_early_life_text() -> str:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (early_life_preset or "").strip().lower()
        if preset == "none":
            return ""

        if preset.startswith("all_"):
            k = preset.replace("all_", "", 1)
            return {"difficult": "Difficult", "ok": "Okay", "good": "Good"}.get(k, rng.choice(EARLY_LIFE_TXT))

        if preset.startswith("more_") or preset.startswith("most_"):
            k = preset.replace("more_", "", 1).replace("most_", "", 1)
            target = {"difficult": "Difficult", "ok": "Okay", "good": "Good"}.get(k)
            if not target:
                return rng.choice(EARLY_LIFE_TXT)
            target_w = 0.55 if preset.startswith("more_") else 0.80
            rest_w = (1.0 - target_w) / 2.0
            items = [(target, target_w)] + [(x, rest_w) for x in EARLY_LIFE_TXT if x != target]
            return _weighted_choice(rng, items)

        return rng.choice(EARLY_LIFE_TXT)  # equal/random

    def sample_epistemic_profile() -> dict:
        """Sample one persona/demographic attribute according to the selected preset."""
        preset = (epistemic_profile_preset or "").strip().lower()
        if preset == "none":
            return {}
        if preset in {"random", "equal"}:
            key = rng.choice(EPISTEMIC_ORDER)
            return dict(EPISTEMIC_ARCHETYPES[key])
        if preset.startswith("all_"):
            key = preset.replace("all_", "", 1)
            return dict(EPISTEMIC_ARCHETYPES.get(key, EPISTEMIC_ARCHETYPES[rng.choice(EPISTEMIC_ORDER)]))
        if preset.startswith("more_") or preset.startswith("most_"):
            key = preset.replace("more_", "", 1).replace("most_", "", 1)
            target_w = 0.55 if preset.startswith("more_") else 0.80
            items = []
            for k in EPISTEMIC_ORDER:
                w = target_w if k == key else (1.0 - target_w) / max(1, len(EPISTEMIC_ORDER) - 1)
                items.append((k, w))
            chosen = _weighted_choice(rng, items)
            return dict(EPISTEMIC_ARCHETYPES.get(chosen, EPISTEMIC_ARCHETYPES[rng.choice(EPISTEMIC_ORDER)]))
        return dict(EPISTEMIC_ARCHETYPES[rng.choice(EPISTEMIC_ORDER)])

    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Age weights are only needed if age is enabled.
    age_enabled = (age_preset or "").strip().lower() != "none"
    a_w = age_bin_weights(age_preset) if age_enabled else []

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "row_id",
            "agent_id",
            "agent_name",
            "belief",
            "opinion",
            "political_leaning",
            "age",
            "gender",
            "ethnicity",
            "education",
            "occupation",
            "early_life",
            "epistemic_profile",
            "institutional_trust",
            "uncertainty_tolerance",
            "evidence_style",
            "official_narrative_suspicion",
            "openness_to_update",
            "background",
            "persona",
        ])

        for i in range(n_agents):
            agent_id = i + 1
            row_id = i + 1

            opinion = int(beliefs_numeric[i])
            belief = belief_label(opinion)

            # Age
            if age_enabled:
                age_bin = _weighted_choice(rng, a_w)
                age = age_from_bin(rng, age_bin)
            else:
                age = ""

            gender = sample_gender_text()
            ethnicity = sample_ethnicity_text()
            education = sample_education_text()
            occupation = sample_occupation_text()
            early_life = sample_early_life_text()
            political_leaning = sample_political_text()
            epistemic = sample_epistemic_profile()
            epistemic_profile = epistemic.get("label", "")
            institutional_trust = epistemic.get("institutional_trust", "")
            uncertainty_tolerance = epistemic.get("uncertainty_tolerance", "")
            evidence_style = epistemic.get("evidence_style", "")
            official_narrative_suspicion = epistemic.get("official_narrative_suspicion", "")
            openness_to_update = epistemic.get("openness_to_update", "")

            # -----------------------------
            # Background: short, realistic mini-biography (not a field list)
            # -----------------------------
            age_num = None
            try:
                age_num = int(age) if age else None
            except Exception:
                age_num = None

            if age_num is None:
                stage = "They have a steady routine and a grounded perspective."
            elif age_num <= 25:
                stage = "They are early in adult life and still forming many views."
            elif age_num <= 40:
                stage = "They are in a busy, building phase of life and pay attention to practical outcomes."
            elif age_num <= 60:
                stage = "They have accumulated experience and tend to weigh trade-offs carefully."
            else:
                stage = "They draw on long experience and value stability and consistency."

            job_clause = f"works as a {occupation}" if occupation else "has a day-to-day routine that keeps them engaged"
            edu_clause = f" with {education} education" if education else ""
            pol_clause = (
                "They follow public issues closely and sometimes frame arguments through their political leanings."
                if political_leaning else
                "They follow public issues when they affect everyday life."
            )

            s1_templates = [
                f"{names[i]} {job_clause}{edu_clause}. {stage}",
                f"{names[i]} {job_clause}{edu_clause}. {pol_clause}",
            ]
            s2_templates = [
                "They try to evaluate controversial claims by checking coherence, assumptions, and the cost of being wrong.",
                "They are wary of sweeping assertions and look for internally consistent reasoning before shifting their view.",
                "They tend to focus on consequences and trade-offs when deciding whether an argument is persuasive.",
            ]
            background = rng.choice(s1_templates) + " " + rng.choice(s2_templates)
            if epistemic.get("background"):
                background = background + " " + epistemic["background"]
            if early_life:
                background = background + f" They describe their early years as {early_life.lower()}."

            # -----------------------------
            # Persona: CARD format (omit empty fields; no communication-style line)
            # -----------------------------
            persona_lines: list[str] = ["AGENT_PERSONA_CARD"]
            persona_lines.append(f"Name: {names[i]}")
            persona_lines.append(f"Belief: {belief}")
            persona_lines.append(f"Initial rating: {opinion}")
            if age:
                persona_lines.append(f"Age: {age}")
            if gender:
                persona_lines.append(f"Gender: {gender}")
            if ethnicity:
                persona_lines.append(f"Ethnicity: {ethnicity}")
            if education:
                persona_lines.append(f"Education: {education}")
            if occupation:
                persona_lines.append(f"Occupation: {occupation}")
            if political_leaning:
                persona_lines.append(f"Political leaning: {political_leaning}")
            if early_life:
                persona_lines.append(f"Early life: {early_life}")
            if epistemic_profile:
                persona_lines.append(f"Epistemic profile: {epistemic_profile}")
            if institutional_trust:
                persona_lines.append(f"Institutional trust: {institutional_trust}")
            if uncertainty_tolerance:
                persona_lines.append(f"Uncertainty tolerance: {uncertainty_tolerance}")
            if evidence_style:
                persona_lines.append(f"Evidence style: {evidence_style}")
            if official_narrative_suspicion:
                persona_lines.append(f"Official-narrative suspicion: {official_narrative_suspicion}")
            if openness_to_update:
                persona_lines.append(f"Openness to update: {openness_to_update}")

            persona = "\n".join(persona_lines)

            w.writerow([
                row_id,
                agent_id,
                names[i],
                belief,
                opinion,
                political_leaning,
                age,
                gender,
                ethnicity,
                education,
                occupation,
                early_life,
                epistemic_profile,
                institutional_trust,
                uncertainty_tolerance,
                evidence_style,
                official_narrative_suspicion,
                openness_to_update,
                background,
                persona,
            ])
def load_settings() -> dict:
    """Load persisted UI settings while preserving the user's existing file."""
    paths = []
    seen = set()
    for p in [SETTINGS_FILE, LAUNCHER_DIR / ".launcher_settings.json", Path.cwd() / ".launcher_settings.json"]:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp)
        if key not in seen:
            seen.add(key)
            paths.append(rp)

    for path in paths:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    merged = dict(DEFAULT_SETTINGS)
                    for k in DEFAULT_SETTINGS:
                        if k in data:
                            merged[k] = data[k]
                    globals()["SETTINGS_FILE"] = path
                    return merged
        except Exception:
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(data: dict) -> None:
    """Persist UI settings back to the same settings file the launcher is using."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _resolve_existing_path(raw_path: str, *, prefer_project_root: bool = False) -> Path:
    """Resolve relative paths robustly without forcing users to redo saved settings."""
    raw = str(raw_path or "").strip()
    if not raw:
        return Path(raw)
    p = Path(raw)
    if p.is_absolute():
        return p

    ordered = []
    if prefer_project_root:
        ordered.extend([PROJECT_ROOT / p, LAUNCHER_DIR / p, Path.cwd() / p])
    else:
        ordered.extend([LAUNCHER_DIR / p, PROJECT_ROOT / p, Path.cwd() / p])

    seen = set()
    uniq = []
    for cand in ordered:
        try:
            rc = cand.resolve()
        except Exception:
            rc = cand
        key = str(rc)
        if key not in seen:
            seen.add(key)
            uniq.append(rc)

    for cand in uniq:
        if cand.exists():
            return cand
    return uniq[0]

# =============================
# Subprocess runner with streaming logs
# =============================

class ProcessRunner:
    """Run a simulation or plotting command in a background thread and stream output into the UI."""
    def __init__(self, on_output, on_done):
        """Initialize instance state, widgets, callbacks, and default values."""
        self._proc = None
        self._thread = None
        self._on_output = on_output
        self._on_done = on_done

    def is_running(self) -> bool:
        """Return whether the managed subprocess is currently active."""
        return self._proc is not None and self._proc.poll() is None

    def start(self, cmd_list: list[str], cwd: str | None = None):
        """Start the background process or UI workflow managed by this object."""
        if self.is_running():
            raise RuntimeError("Process already running.")

        def run():
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            try:
                self._on_output(f"\n[CMD] {' '.join(cmd_list)}\n\n")
                self._proc = subprocess.Popen(
                    cmd_list,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self._on_output(line)
            except Exception as e:
                self._on_output(f"\n[ERROR] {e}\n")
            finally:
                code = None
                if self._proc is not None:
                    code = self._proc.poll()
                self._on_done(code)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the active background process or UI workflow if one is running."""
        if self.is_running():
            self._proc.terminate()


# =============================
# Tkinter UI
# =============================



class AutoScrollbar(ttk.Scrollbar):
    """Scrollbar that hides itself when not needed."""
    def set(self, lo, hi):
        """Small local helper used to update the surrounding function state."""
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            lo_f, hi_f = 0.0, 1.0
        if lo_f <= 0.0 and hi_f >= 1.0:
            try:
                self.grid_remove()
            except Exception:
                try:
                    self.pack_forget()
                except Exception:
                    pass
        else:
            try:
                self.grid()
            except Exception:
                try:
                    self.pack()
                except Exception:
                    pass
        super().set(lo, hi)

class LauncherUI(tk.Tk):
    """Tkinter application that exposes experiment, persona, RAG, network, and plotting controls."""
    def __init__(self):
        """Initialize instance state, widgets, callbacks, and default values."""
        super().__init__()
        self.title("Opinion Dynamics Launcher")
        self.geometry("1080x820")

        # --- Dark "futuristic" styling (ttk) ---
        try:
            style = ttk.Style(self)
            if "clam" in style.theme_names():
                style.theme_use("clam")

            # Palette (higher contrast)
            bg = "#070b17"        # near-black navy
            panel = "#0c1328"     # card background
            panel2 = "#101a36"    # raised elements
            fg = "#f2f4ff"        # brighter text
            muted = "#b6bdd6"     # brighter secondary text
            accent = "#22d3ee"    # cyan
            accent2 = "#8b5cf6"   # violet
            danger = "#fb7185"    # red/pink

            self.configure(bg=bg)

            # Base
            style.configure(".", background=bg, foreground=fg, fieldbackground=panel, bordercolor=panel2, lightcolor=panel2, darkcolor=panel2)

            # Containers
            style.configure("TFrame", background=bg)
            style.configure("Card.TFrame", background=panel)

            style.configure("TNotebook", background=bg, borderwidth=0)
            style.configure("TNotebook.Tab", background=panel2, foreground=muted, padding=(16, 10), borderwidth=0)
            style.map("TNotebook.Tab",
                      background=[("selected", panel)],
                      foreground=[("selected", fg), ("active", fg)])

            style.configure("TLabelframe", background=bg, foreground=fg, bordercolor=panel2)
            style.configure("TLabelframe.Label", background=bg, foreground=accent, font=("Segoe UI", 10, "bold"))

            style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 9))
            style.configure("Muted.TLabel", background=bg, foreground=muted, font=("Segoe UI", 9))

            style.configure("TSeparator", background=panel2)

            # Inputs
            style.configure("TEntry", fieldbackground=panel, foreground=fg, insertcolor=fg, padding=7, relief="flat")
            style.configure("TCombobox", fieldbackground=panel, foreground=fg, padding=6)
            style.map("TCombobox",
                      fieldbackground=[("readonly", panel)],
                      foreground=[("readonly", fg)],
                      selectbackground=[("readonly", panel)],
                      selectforeground=[("readonly", fg)])

            style.configure("TCheckbutton", background=bg, foreground=fg)
            style.map("TCheckbutton", foreground=[("disabled", muted)])

            # Buttons (ensure readable text)
            style.configure("TButton", background=panel2, foreground=fg, padding=(14, 9), borderwidth=0, relief="flat")
            style.map("TButton",
                      background=[("active", panel), ("pressed", panel)],
                      foreground=[("disabled", muted), ("active", fg)])

            style.configure("Accent.TButton", background=accent2, foreground="#ffffff", padding=(14, 9), borderwidth=0, relief="flat")
            style.map("Accent.TButton",
                      background=[("active", "#7c3aed"), ("pressed", "#6d28d9")],
                      foreground=[("active", "#ffffff"), ("pressed", "#ffffff")])

            style.configure("Danger.TButton", background=danger, foreground="#111827", padding=(14, 9), borderwidth=0, relief="flat")
            style.map("Danger.TButton",
                      background=[("active", "#fb3a63"), ("pressed", "#e11d48")],
                      foreground=[("active", "#111827"), ("pressed", "#111827")])


            # Improve Combobox affordance / visibility
            style.configure(
                "TCombobox",
                fieldbackground="#111a38",
                background="#111a38",
                foreground="#ffffff",
                padding=(8, 6),
                borderwidth=1,
                relief="flat"
            )
            style.map(
                "TCombobox",
                fieldbackground=[
                    ("readonly", "#111a38"),
                    ("focus", "#16204a"),
                    ("active", "#16204a"),
                ],
                foreground=[
                    ("readonly", "#ffffff"),
                    ("focus", "#ffffff"),
                ],
                bordercolor=[
                    ("focus", "#22d3ee"),
                    ("active", "#22d3ee"),
                ],
            )

        except Exception:
            pass



        self.runner = ProcessRunner(self.append_log_threadsafe, self.on_process_done)

        # Batch/multi-run state
        self._batch_active = False
        self._batch_queue = []  # list[dict(seed=int, out=str)]
        self._batch_total = 0
        self._batch_index = 0

        self._settings = load_settings()
        self._last_generated_prompts = None

        # =============================
        # Notebook layout
        # =============================
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)

        # Status bar
        status_bar = ttk.Frame(root)
        status_bar.pack(fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_bar, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")

        self._tab_run_outer, tab_run = self._create_scrollable_notebook_tab("Run")
        self._tab_personas_outer, tab_personas = self._create_scrollable_notebook_tab("Personas")
        self._tab_network_outer, tab_network = self._create_scrollable_notebook_tab("Network")

        self._tab_live_outer = ttk.Frame(self.nb)
        self.nb.add(self._tab_live_outer, text="Live")
        tab_live = self._tab_live_outer

        self._tab_logs_outer = ttk.Frame(self.nb)
        self.nb.add(self._tab_logs_outer, text="Logs")
        tab_logs = self._tab_logs_outer

        # Build Live tab UI (optional; attaches automatically when the script prints the CSV path)
        try:
            self._build_live_tab(tab_live)
        except Exception:
            pass


        try:
            self.nb.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed, add="+")
        except Exception:
            pass
        try:
            self._bind_global_notebook_mousewheel()
        except Exception:
            pass

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # =============================
        # RUN TAB
        # =============================
        header = ttk.Frame(tab_run)
        header.pack(fill="x", pady=(0, 10))
        title = ttk.Label(header, text="Opinion Dynamics Launcher", style="TLabel", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(header, text="Configure runs, generate personas, and launch simulations reproducibly.", style="Muted.TLabel")
        subtitle.pack(anchor="w")

        # -----------------------------
        # Run Configuration (script + basic run params)
        # -----------------------------
        cfg_run = ttk.LabelFrame(tab_run, text="Run", padding=12)
        cfg_run.pack(fill="x")

        # Script
        ttk.Label(cfg_run, text="Script:").grid(row=0, column=0, sticky="w")
        self.script_var = tk.StringVar(value=self._settings["script"])
        ttk.Combobox(cfg_run, textvariable=self.script_var, values=DEFAULT_SCRIPTS, width=70, state="readonly")             .grid(row=0, column=1, sticky="we", padx=6, columnspan=5)

        # Seed / Agents / Steps / Out
        ttk.Label(cfg_run, text="Seed:").grid(row=1, column=0, sticky="w", pady=6)
        self.seed_var = tk.StringVar(value=self._settings["seed"])
        ttk.Entry(cfg_run, textvariable=self.seed_var, width=12).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(cfg_run, text="Agents:").grid(row=1, column=2, sticky="w")
        self.agents_var = tk.StringVar(value=self._settings["agents"])
        ttk.Entry(cfg_run, textvariable=self.agents_var, width=10).grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(cfg_run, text="Steps:").grid(row=1, column=4, sticky="w")
        self.steps_var = tk.StringVar(value=self._settings["steps"])
        self._steps_entry = ttk.Entry(cfg_run, textvariable=self.steps_var, width=10)
        self._steps_entry.grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(cfg_run, text="Out:").grid(row=2, column=0, sticky="w", pady=6)
        self.out_var = tk.StringVar(value=self._settings["out"])
        ttk.Entry(cfg_run, textvariable=self.out_var, width=24).grid(row=2, column=1, sticky="we", padx=6, ipady=4)

        # Prompt selection
        ttk.Label(cfg_run, text="Prompt version:").grid(row=2, column=2, sticky="w")
        self.prompt_version_var = tk.StringVar(value=self._settings["prompt_version"])
        ttk.Combobox(cfg_run, textvariable=self.prompt_version_var,
                     values=PROMPT_VERSIONS, width=12, state="readonly").grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(cfg_run, text="Prompt mode:").grid(row=2, column=4, sticky="w")
        self.prompt_mode_var = tk.StringVar(value=self._settings["prompt_mode"])
        ttk.Combobox(cfg_run, textvariable=self.prompt_mode_var,
                     values=DEFAULT_PROMPT_MODES, width=30, state="readonly").grid(row=2, column=5, sticky="w", padx=6)

        # World mode
        ttk.Label(cfg_run, text="World:").grid(row=3, column=0, sticky="w", pady=6)
        _world = (self._settings.get("world", "closed") or "closed").strip().lower()
        if _world == "open":
            _world = "open_no_rag"  # compatibility migration
        if _world not in ["closed", "closed_strict", "closed_strict_rag", "open_no_rag", "open_rag", "true_open"]:
            _world = "closed"
        self.world_var = tk.StringVar(value=_world)
        self.world_combo = ttk.Combobox(
            cfg_run,
            textvariable=self.world_var,
            values=["closed", "closed_strict", "closed_strict_rag", "open_no_rag", "open_rag", "true_open"],
            width=18,
            state="readonly",
        )
        self.world_combo.grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(
            cfg_run,
            text="closed* = agents may use ONLY what the prompt shows (keeps social dynamics measurable); closed_strict_rag adds retrieved snippets; open* = the model may use its own knowledge (factual claims then converge on their own). Enforcement is SOFT: leaks are logged, never retried.",
            style="Muted.TLabel", wraplength=760, justify="left",
        ).grid(row=3, column=2, columnspan=4, sticky="w")

        ttk.Label(cfg_run, text="Fact pack mode:").grid(row=4, column=0, sticky="w", pady=6)
        self.fact_pack_mode_var = tk.StringVar(value=self._settings.get("fact_pack_mode", "off"))
        ttk.Combobox(
            cfg_run,
            textvariable=self.fact_pack_mode_var,
            values=DEFAULT_FACT_PACK_MODES,
            width=14,
            state="readonly",
        ).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(cfg_run, text="Curated per-topic evidence injected at runtime (separate from RAG; from fact_packs.py). Mode picks the direction: off / full / supportive / criticisms / contextual. In closed worlds shown material carries outsized weight - treat as an experiment lever.",
                  style="Muted.TLabel", wraplength=760, justify="left").grid(row=4, column=2, columnspan=4, sticky="w")

        # Memory mode
        ttk.Label(cfg_run, text="Memory:").grid(row=5, column=0, sticky="w", pady=6)
        self.memory_var = tk.StringVar(value=self._settings.get("memory", "enabled"))
        self.memory_combo = ttk.Combobox(
            cfg_run,
            textvariable=self.memory_var,
            values=["enabled", "light", "off"],
            width=12,
            state="readonly"
        )
        self.memory_combo.grid(row=5, column=1, sticky="w", padx=6)

        self.light_memory_threshold_var = tk.StringVar(value=str(self._settings.get("light_memory_threshold", "3") or "3"))
        self.light_memory_threshold_label = ttk.Label(cfg_run, text="Light threshold:")
        self.light_memory_threshold_label.grid(row=5, column=2, sticky="w", padx=(18, 0))
        self.light_memory_threshold_combo = ttk.Combobox(
            cfg_run,
            textvariable=self.light_memory_threshold_var,
            values=["1", "2", "3", "4", "5"],
            width=6,
            state="readonly"
        )
        self.light_memory_threshold_combo.grid(row=5, column=3, sticky="w", padx=6)
        ttk.Label(cfg_run, text="enabled = agent remembers its own past tweets/responses; light = truncated history + belief path (Light threshold = how many recent items are kept); off = memoryless agent.",
                  style="Muted.TLabel", wraplength=520, justify="left").grid(row=5, column=4, columnspan=2, sticky="w")
        try:
            self.memory_var.trace_add("write", lambda *_: self._update_light_memory_threshold_state())
        except Exception:
            pass
        self._update_light_memory_threshold_state()

        # Trace / terminal verbosity (controls what the script prints while running)
        ttk.Label(cfg_run, text="Trace:").grid(row=6, column=0, sticky="w", pady=6)
        self.trace_var = tk.StringVar(value=self._settings.get("trace", "auto"))
        ttk.Combobox(
            cfg_run,
            textvariable=self.trace_var,
            values=["auto", "minimal", "full", "off"],
            width=12,
            state="readonly"
        ).grid(row=6, column=1, sticky="w", padx=6)
        ttk.Label(cfg_run, text="Console verbosity only (what the Logs tab prints) - zero effect on results. auto: minimal when Memory=off, full when enabled.",
                  style="Muted.TLabel", wraplength=760, justify="left").grid(row=6, column=2, columnspan=4, sticky="w")

        # Whether to inject history into the LLM prompt (off keeps LLM Markovian)
        ttk.Label(cfg_run, text="LLM history:").grid(row=7, column=0, sticky="w", pady=6)
        self.llm_history_var = tk.StringVar(value=self._settings.get("llm_history", "off"))
        ttk.Combobox(
            cfg_run,
            textvariable=self.llm_history_var,
            values=["off", "auto", "on"],
            width=12,
            state="readonly"
        ).grid(row=7, column=1, sticky="w", padx=6)
        ttk.Label(cfg_run, text="Whether earlier turns are injected into the LLM context. off = Markovian (model sees only the current prompt - our default); on = past turns included; auto follows Memory. CHANGES BEHAVIOUR, not just logging.",
                  style="Muted.TLabel", wraplength=760, justify="left").grid(row=7, column=2, columnspan=4, sticky="w")

        # Save native shadow-thinking payloads on hard failures (if the script supports it)
        ttk.Label(cfg_run, text="Debug native thinking:").grid(row=8, column=0, sticky="w", pady=6)
        self.debug_native_thinking_on_fail_var = tk.StringVar(value=self._settings.get("debug_native_thinking_on_fail", "off"))
        ttk.Combobox(
            cfg_run,
            textvariable=self.debug_native_thinking_on_fail_var,
            values=["off", "on"],
            width=12,
            state="readonly"
        ).grid(row=8, column=1, sticky="w", padx=6)
        ttk.Label(cfg_run, text="on = when a step hard-fails, the model's raw thinking payload is saved for debugging (supported scripts only). No effect on results; leave off unless investigating failures.",
                  style="Muted.TLabel", wraplength=760, justify="left").grid(row=8, column=2, columnspan=4, sticky="w")



        ttk.Label(cfg_run, text="Solo check:").grid(row=9, column=0, sticky="w", pady=6)
        self.solo_check_var = tk.StringVar(value=self._settings.get("solo_check", "off"))
        ttk.Combobox(
            cfg_run,
            textvariable=self.solo_check_var,
            values=["off", "on"],
            width=12,
            state="readonly"
        ).grid(row=9, column=1, sticky="w", padx=6)
        ttk.Label(cfg_run, text="on = ask the model ALONE about the claim (no personas, no network, no interactions): each 'agent' is an independent sample answering step1_report.md. Measures the model's own prior on the topic - the baseline for cross-model comparisons. Uses the SAME native inference path and decoding options as normal runs, so numbers are comparable. Network/RAG/persona settings are ignored while on. (Replaces the old terminal-only v3_check script.) While on, the sections that do not apply (network, personas, RAG, world/bias, step overrides, Steps) are greyed out automatically; Steps is ignored - the number of independent samples = Agents.",
                  style="Muted.TLabel", wraplength=760, justify="left").grid(row=9, column=2, columnspan=4, sticky="w")

        ttk.Label(cfg_run, text="Script: which main runs (only test_network_qwen works from here; the old v3_check is now the 'Solo check' option above).  Seed: RNG for network build + interaction order - same seed = repeatable run.  Agents / Steps: population size and number of interactions (1 speaker->listener per step).  Out: the run-folder name under results/.  Prompt version: template set = topic + bias level (v52 moon, v119 9/11, v130 simulation; *_confirmation_bias = biased agents, strong_ = heavily; *_reverse = claim worded from the conspiracy side).  Prompt mode: which template flow variant is used.",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=10, column=0, columnspan=6, sticky="w", pady=(8, 0))

        # grid weights so script combobox stretches
        try:
            cfg_run.grid_columnconfigure(1, weight=1)
        except Exception:
            pass

        # -----------------------------
        # Global constraints (these are NOT step-specific)
        # -----------------------------
        cfg_global = ttk.LabelFrame(tab_run, text="Global constraints", padding=12)
        self._register_solo_irrelevant_frame(cfg_global)
        cfg_global.pack(fill="x", pady=(10, 0))

        ttk.Label(cfg_global, text="Model:").grid(row=0, column=0, sticky="w", pady=6)
        self.model_var = tk.StringVar(value=self._settings["model"])
        # Editable so newly pulled Ollama model tags can be used immediately without editing the launcher.
        ttk.Combobox(cfg_global, textvariable=self.model_var, values=DEFAULT_MODELS, width=18, state="normal").grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(cfg_global, text="Think mode:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.think_mode_var = tk.StringVar(value=self._settings.get("think_mode", "off"))
        self.think_mode_combo = ttk.Combobox(
            cfg_global,
            textvariable=self.think_mode_var,
            values=["off", "on", "step3_only", "step2_only"],
            width=12,
            state="readonly",
        )
        self.think_mode_combo.grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(cfg_global, text="Max step change:").grid(row=1, column=0, sticky="w", pady=6)
        self.max_step_change_var = tk.StringVar(value=self._settings.get("max_step_change", "1"))
        ttk.Combobox(cfg_global, textvariable=self.max_step_change_var, values=["1", "2"], width=8, state="readonly").grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Chain-of-thought before answering (per step). SPECIAL CASE: step3 keeps thinking ALWAYS by design - on length failures num_predict is boosted instead of dropping thinking. Greys out for models with no thinking support (per-model registry).",
                  style="Muted.TLabel", wraplength=520, justify="left").grid(row=0, column=4, columnspan=2, sticky="w")

        ttk.Label(cfg_global, text="Allowed update mode:").grid(row=1, column=2, sticky="w", padx=(18, 0))
        self.allowed_update_mode_var = tk.StringVar(value=self._settings.get("allowed_update_mode", "assimilation_only"))
        self.allowed_update_mode_combo = ttk.Combobox(
            cfg_global,
            textvariable=self.allowed_update_mode_var,
            values=["assimilation_only", "free_bounded"],
            width=20,
            state="readonly",
        )
        self.allowed_update_mode_combo.grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(cfg_global, text="assimilation_only = listener may only move TOWARD the speaker (Flache assimilation); free_bounded = any direction within Max step change (our default). This choice also decides whether same-side unlock does anything - see below.",
                  style="Muted.TLabel", wraplength=520, justify="left").grid(row=1, column=4, columnspan=2, sticky="w")
        ttk.Label(cfg_global, text="Validation strictness:").grid(row=5, column=0, sticky="w", pady=6)
        self.validation_strictness_var = tk.StringVar(value=self._settings.get("validation_strictness", "strict"))
        self.validation_strictness_combo = ttk.Combobox(
            cfg_global,
            textvariable=self.validation_strictness_var,
            values=["strict", "warn_only", "format_only"],
            width=20,
            state="readonly",
        )
        self.validation_strictness_combo.grid(row=5, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="strict = content/style checks may reject and re-ask (retries shape the text!); warn_only = everything logged, nothing re-asked; format_only = only FINAL_RATING/TWEET parsing enforced. Closed-world checks stay ON as warnings in every mode. Strictness measurably changes outcomes - report it as an experimental condition, and prefer warn_only when comparing models.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=5, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="Deterministic:").grid(row=6, column=0, sticky="w", pady=6)
        self.deterministic_var = tk.StringVar(value=self._settings.get("deterministic", "off"))
        ttk.Combobox(cfg_global, textvariable=self.deterministic_var, values=["off", "on"], width=8, state="readonly").grid(row=6, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="on = temperature 0 / greedy decoding for both steps: identical outputs on identical seed. Kills variety - use for debugging or exact replication, not for natural-sounding runs.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=6, column=2, columnspan=3, sticky="w")
        # ADR-006 Component 2: silence-as-choice global flag.
        ttk.Label(cfg_global, text="Allow silence in step2:").grid(row=7, column=0, sticky="w", pady=6)
        self.allow_silence_var = tk.StringVar(value=self._settings.get("allow_silence", "off"))
        ttk.Combobox(cfg_global, textvariable=self.allow_silence_var, values=["off", "on"], width=8, state="readonly").grid(row=7, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="on = ο agent μπορεί να επιστρέψει `<silent>` και να μη γράψει tweet (mechanism-agnostic silence option, feeds P12). off (default) = byte-identical με προ-ADR-006, το step2 template renderάρεται χωρίς silence block. Πλήρη Noels 4-way parser dispatch + rich silence log θα προστεθούν σε Task 2.3b.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=7, column=2, columnspan=3, sticky="w")
        # ADR-006 Component 3: p_reach pluggable policy pattern (feeds P21, P30).
        ttk.Label(cfg_global, text="Reach policy (p_reach):").grid(row=8, column=0, sticky="w", pady=6)
        self.p_reach_policy_var = tk.StringVar(value=self._settings.get("p_reach_policy", "uniform"))
        ttk.Combobox(cfg_global, textvariable=self.p_reach_policy_var, values=["uniform", "homophilic", "shadowban"], width=12, state="readonly").grid(row=8, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="uniform (default) = κάθε ακμή παίρνει το ίδιο p_reach, 1.0 -> byte-identical baseline. homophilic = sigmoid(k*(1-2*norm_dist)): όμοιες γνώμες βλέπονται περισσότερο (engagement amplification, feeds P30 Cinelli). shadowban = τυχαίο υποσύνολο agents παίρνει χαμηλό outgoing reach (content moderation, feeds P21). Οι τρεις sub-παράμετροι πιο κάτω ισχύουν η καθεμία ΜΟΝΟ για την policy της.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=8, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="p_reach uniform value:").grid(row=9, column=0, sticky="w", pady=6)
        self.p_reach_uniform_value_var = tk.StringVar(value=self._settings.get("p_reach_uniform_value", "1.0"))
        ttk.Entry(cfg_global, textvariable=self.p_reach_uniform_value_var, width=8).grid(row=9, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Μόνο για policy=uniform. p_reach κάθε ακμής στο [0,1]. 1.0 (default) = baseline. Dose-response sweep {1.0, 0.75, 0.5, 0.25, 0.1} απομονώνει την επίδραση της αραίωσης εμβέλειας. Κενό = ο sim βάζει το default.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=9, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="p_reach homophily k:").grid(row=10, column=0, sticky="w", pady=6)
        self.p_reach_homophily_k_var = tk.StringVar(value=self._settings.get("p_reach_homophily_k", "2.0"))
        ttk.Entry(cfg_global, textvariable=self.p_reach_homophily_k_var, width=8).grid(row=10, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Μόνο για policy=homophilic. Sigmoid sharpness: μεγαλύτερο k -> πιο απότομη αντίθεση similar/dissimilar. Default 2.0. Κενό = default.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=10, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="p_reach shadowban fraction:").grid(row=11, column=0, sticky="w", pady=6)
        self.p_reach_shadowban_fraction_var = tk.StringVar(value=self._settings.get("p_reach_shadowban_fraction", "0.1"))
        ttk.Entry(cfg_global, textvariable=self.p_reach_shadowban_fraction_var, width=8).grid(row=11, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Μόνο για policy=shadowban. Ποσοστό agents που τιμωρούνται (οι εξερχόμενες ακμές τους παίρνουν το shadowban value). Στο [0,1], default 0.1. Κενό = default.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=11, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="p_reach shadowban value:").grid(row=12, column=0, sticky="w", pady=6)
        self.p_reach_shadowban_value_var = tk.StringVar(value=self._settings.get("p_reach_shadowban_value", "0.1"))
        ttk.Entry(cfg_global, textvariable=self.p_reach_shadowban_value_var, width=8).grid(row=12, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Μόνο για policy=shadowban. p_reach των εξερχόμενων ακμών των τιμωρημένων agents. Στο [0,1], default 0.1 (οι υπόλοιποι κρατούν 1.0). Κενό = default.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=12, column=2, columnspan=3, sticky="w")
        # ADR-006 roads_not_taken 3.11 (fix gamma): reach enforcement mode.
        ttk.Label(cfg_global, text="p_reach enforcement:").grid(row=13, column=0, sticky="w", pady=6)
        self.p_reach_enforcement_var = tk.StringVar(value=self._settings.get("p_reach_enforcement", "filter"))
        ttk.Combobox(cfg_global, textvariable=self.p_reach_enforcement_var, values=["filter", "suppress"], width=10, state="readonly").grid(row=13, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Μόνο όταν p_reach < 1.0 (homophilic/shadowban). ΠΩΣ επιβάλλεται η μειωμένη εμβέλεια. filter (default) = ο throttled speaker αφαιρείται από τους υποψήφιους speakers (Bernoulli στο ΚΟΙΝΟ rng) -> σπάνια επιλέγεται· μοντελοποιεί «reduced surfacing», ΑΛΛΑ τα draws του κάνουν τις single-seed uniform-vs-treatment συγκρίσεις να αποκλίνουν (RNG-path confound). suppress = ο speaker επιλέγεται κανονικά (ίδια interaction sequence με matched baseline)· αν είναι throttled, Bernoulli σε ΑΦΙΕΡΩΜΕΝΟ rng αποφασίζει αν φτάνει το tweet, και σε αποτυχία ΔΕΝ εφαρμόζεται το belief-update· μοντελοποιεί «de-ranked/ignored delivery» και δίνει ΚΑΘΑΡΗ single-seed αιτιακή σύγκριση. roads_not_taken 3.11 fix γ.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=13, column=2, columnspan=3, sticky="w")
        # ADR-006 Component 4: Bian 5-dim diagnostic opt-in (feeds P28).
        ttk.Label(cfg_global, text="Include Bian bias scores:").grid(row=14, column=0, sticky="w", pady=6)
        self.include_bian_scores_var = tk.StringVar(value=self._settings.get("include_bian_scores", "off"))
        ttk.Combobox(cfg_global, textvariable=self.include_bian_scores_var, values=["off", "on"], width=8, state="readonly").grid(row=14, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="Πριν το run, τρέχει το tools/bian_diagnostic.py για το επιλεγμένο Ollama μοντέλο και ενσωματώνει τα 5 scores (social role KL, inter/intra agent similarity, keyword persistence, positivity) στο config self-doc του run. Cached ανά μοντέλο. off (default) = no-op. Bian 2025 (arXiv:2510.21180) validity protocol, feeds P28.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=14, column=2, columnspan=3, sticky="w")
        ttk.Label(cfg_global, text="Wrong-side explanation re-query:").grid(row=4, column=0, sticky="w", pady=6)
        self.wrong_side_requery_var = tk.StringVar(value=self._settings.get("wrong_side_explanation_requery", "off"))
        ttk.Combobox(cfg_global, textvariable=self.wrong_side_requery_var, values=["off", "on"], width=8, state="readonly").grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="If the explanation contradicts the chosen rating (wrong side / fragment / leak), ONE extra call rewrites the explanation only - THE RATING NEVER CHANGES. Cleans the transcript without touching the dynamics; adds ~1 call per bad explanation.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=4, column=2, columnspan=3, sticky="w")

        ttk.Label(cfg_global, text="Same-side edge unlock hits:").grid(row=2, column=0, sticky="w", pady=6)
        self.same_side_edge_unlock_hits_var = tk.StringVar(value=self._settings.get("same_side_edge_unlock_hits", "2"))
        self.same_side_edge_unlock_hits_combo = ttk.Combobox(
            cfg_global,
            textvariable=self.same_side_edge_unlock_hits_var,
            values=["0", "1", "2", "3", "4", "5"],
            width=8,
            state="readonly",
        )
        self.same_side_edge_unlock_hits_combo.grid(row=2, column=1, sticky="w", padx=6)
        self.same_side_edge_unlock_hits_help_label = ttk.Label(cfg_global, text="Counts consecutive same-side mild (±1) pushes; after N hits the ±2 edge unlocks from ±1. ONLY meaningful under assimilation_only: free_bounded already allows moving to the edge, so the code ignores this there (no-op) - keep 0 and it greys out. Why: without it, assimilation agents could never reach ±2 from repeated mild agreement.",
                  style="Muted.TLabel", wraplength=620, justify="left")
        self.same_side_edge_unlock_hits_help_label.grid(row=2, column=2, columnspan=4, sticky="w")
        try:
            self._last_enabled_same_side_edge_unlock_hits = self.same_side_edge_unlock_hits_var.get().strip() or "2"
            self.allowed_update_mode_var.trace_add("write", lambda *_: self._update_allowed_update_controls_state())
        except Exception:
            pass
        self._update_allowed_update_controls_state()
        try:
            self.model_var.trace_add("write", lambda *_: self._update_model_dependent_controls_state())
        except Exception:
            pass
        self._update_model_dependent_controls_state()

        ttk.Label(cfg_global, text="Same-rating Step-3:").grid(row=3, column=0, sticky="w", pady=6)
        self.same_rating_step3_mode_var = tk.StringVar(value=self._settings.get("same_rating_step3_mode", "skip_tweet_local"))
        ttk.Combobox(
            cfg_global,
            textvariable=self.same_rating_step3_mode_var,
            values=["skip_tweet_local", "skip_generic", "llm"],
            width=18,
            state="readonly",
        ).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(cfg_global, text="What happens when only ONE rating is allowed anyway (nothing to decide): skip_tweet_local = skip the LLM call, log a local note (fast, saves tokens); skip_generic = skip with a generic line; llm = still call the LLM just for the explanation text. Applies only when the allowed set has length 1 after unlock/expansion.",
                  style="Muted.TLabel", wraplength=620, justify="left").grid(row=3, column=2, columnspan=4, sticky="w")

        ttk.Label(cfg_global, text="Model: global Ollama tag used by BOTH steps unless a per-step override is set (editable so freshly pulled tags work - but beware typos).  Max step change: how far one interaction can move an opinion (1 = one notch; the prompt's ALLOWED_FINAL_RATING_SET is built from this).",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=14, column=0, columnspan=6, sticky="w", pady=(8, 0))

        # -----------------------------
        # LLM defaults (apply unless overridden per-step)
        # -----------------------------
        llm_global = ttk.LabelFrame(tab_run, text="LLM defaults (used unless overridden)", padding=12)
        llm_global.pack(fill="x", pady=(10, 0))

        ttk.Label(llm_global, text="Temperature:").grid(row=0, column=0, sticky="w")
        self.temp_var = tk.StringVar(value=self._settings.get("temperature", "0.7"))
        ttk.Entry(llm_global, textvariable=self.temp_var, width=10).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(llm_global, text="Top-p:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.top_p_var = tk.StringVar(value=self._settings.get("top_p", "0.9"))
        ttk.Entry(llm_global, textvariable=self.top_p_var, width=10).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(llm_global, text="Top-k:").grid(row=0, column=4, sticky="w", padx=(18, 0))
        self.top_k_var = tk.StringVar(value=self._settings.get("top_k", "40"))
        ttk.Entry(llm_global, textvariable=self.top_k_var, width=10).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(llm_global, text="Repeat penalty:").grid(row=1, column=0, sticky="w", pady=6)
        self.repeat_penalty_var = tk.StringVar(value=self._settings.get("repeat_penalty", "1.1"))
        ttk.Entry(llm_global, textvariable=self.repeat_penalty_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(llm_global, text="Repeat last N:").grid(row=1, column=2, sticky="w", pady=6, padx=(18, 0))
        self.repeat_last_n_var = tk.StringVar(value=self._settings.get("repeat_last_n", "64"))
        ttk.Entry(llm_global, textvariable=self.repeat_last_n_var, width=10).grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(llm_global, text="Max tokens:").grid(row=1, column=4, sticky="w", pady=6, padx=(18, 0))
        self.max_tokens_var = tk.StringVar(value=self._settings.get("max_tokens", "300"))
        ttk.Entry(llm_global, textvariable=self.max_tokens_var, width=10).grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(llm_global, text="Min-p (opt):").grid(row=2, column=0, sticky="w", pady=6)
        self.min_p_var = tk.StringVar(value=self._settings.get("min_p", ""))
        ttk.Entry(llm_global, textvariable=self.min_p_var, width=10).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(llm_global, text="(blank = don't pass)", style="Muted.TLabel").grid(row=2, column=2, columnspan=4, sticky="w")

        ttk.Label(llm_global, text="Presence penalty:").grid(row=3, column=0, sticky="w", pady=6)
        self.presence_penalty_var = tk.StringVar(value=self._settings.get("presence_penalty", ""))
        ttk.Entry(llm_global, textvariable=self.presence_penalty_var, width=10).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(llm_global, text="Frequency penalty:").grid(row=3, column=2, sticky="w", pady=6, padx=(18, 0))
        self.frequency_penalty_var = tk.StringVar(value=self._settings.get("frequency_penalty", ""))
        ttk.Entry(llm_global, textvariable=self.frequency_penalty_var, width=10).grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(llm_global, text="(0 = off, blank = don't pass)", style="Muted.TLabel").grid(row=3, column=4, columnspan=2, sticky="w")

        ttk.Label(llm_global, text="Temperature: randomness (0 = deterministic, ~0.7 = normal variety).  Top-p / Top-k / Min-p: how wide the sampling pool is (nucleus cutoff / top-K tokens / probability floor; blank Min-p = not passed).  Repeat penalty + Repeat last N: discourage repeating the last N tokens (>1 = stronger).  Max tokens: output budget - THINKING MODELS SPEND THEIR REASONING FROM THIS TOO; too small = cut/empty outputs (step3 auto-boosts on length failures).  Presence/Frequency penalty: extra anti-repetition (blank = not passed).",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=4, column=0, columnspan=6, sticky="w", pady=(8, 0))

        # -----------------------------
        # Step-2 overrides (tweet generation)
        # -----------------------------
        step2 = ttk.LabelFrame(tab_run, text="Step-2 LLM overrides (tweet) — blank = inherit global", padding=12)
        self._register_solo_irrelevant_frame(step2)
        step2.pack(fill="x", pady=(10, 0))

        ttk.Label(step2, text="Temp:").grid(row=0, column=0, sticky="w")
        self.temp_step2_var = tk.StringVar(value=self._settings.get("temperature_step2", ""))
        ttk.Entry(step2, textvariable=self.temp_step2_var, width=10).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(step2, text="Top-p:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.top_p_step2_var = tk.StringVar(value=self._settings.get("top_p_step2", ""))
        ttk.Entry(step2, textvariable=self.top_p_step2_var, width=10).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(step2, text="Top-k:").grid(row=0, column=4, sticky="w", padx=(18, 0))
        self.top_k_step2_var = tk.StringVar(value=self._settings.get("top_k_step2", ""))
        ttk.Entry(step2, textvariable=self.top_k_step2_var, width=10).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(step2, text="Repeat penalty:").grid(row=1, column=0, sticky="w", pady=6)
        self.repeat_penalty_step2_var = tk.StringVar(value=self._settings.get("repeat_penalty_step2", ""))
        ttk.Entry(step2, textvariable=self.repeat_penalty_step2_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(step2, text="Repeat last N:").grid(row=1, column=2, sticky="w", pady=6, padx=(18, 0))
        self.repeat_last_n_step2_var = tk.StringVar(value=self._settings.get("repeat_last_n_step2", ""))
        ttk.Entry(step2, textvariable=self.repeat_last_n_step2_var, width=10).grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(step2, text="Max tokens:").grid(row=1, column=4, sticky="w", pady=6, padx=(18, 0))
        self.max_tokens_step2_var = tk.StringVar(value=self._settings.get("max_tokens_step2", ""))
        ttk.Entry(step2, textvariable=self.max_tokens_step2_var, width=10).grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(step2, text="Min-p:").grid(row=2, column=0, sticky="w", pady=6)
        self.min_p_step2_var = tk.StringVar(value=self._settings.get("min_p_step2", ""))
        ttk.Entry(step2, textvariable=self.min_p_step2_var, width=10).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(step2, text="Presence pen:").grid(row=3, column=0, sticky="w", pady=6)
        self.presence_penalty_step2_var = tk.StringVar(value=self._settings.get("presence_penalty_step2", ""))
        ttk.Entry(step2, textvariable=self.presence_penalty_step2_var, width=10).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(step2, text="Frequency pen:").grid(row=3, column=2, sticky="w", pady=6, padx=(18, 0))
        self.frequency_penalty_step2_var = tk.StringVar(value=self._settings.get("frequency_penalty_step2", ""))
        ttk.Entry(step2, textvariable=self.frequency_penalty_step2_var, width=10).grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(step2, text="Step-2 writes the TWEETS. Blank fields inherit the global defaults above; set here to diverge only for tweet generation (e.g. slightly higher Temp for more varied posts).",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=4, column=0, columnspan=6, sticky="w", pady=(8, 0))

        # -----------------------------
        # Step-3 overrides (belief update)
        # -----------------------------
        step3 = ttk.LabelFrame(tab_run, text="Step-3 LLM overrides (update) — blank = inherit global", padding=12)
        self._register_solo_irrelevant_frame(step3)
        step3.pack(fill="x", pady=(10, 0))

        ttk.Label(step3, text="Temp:").grid(row=0, column=0, sticky="w")
        self.temp_step3_var = tk.StringVar(value=self._settings.get("temperature_step3", ""))
        ttk.Entry(step3, textvariable=self.temp_step3_var, width=10).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(step3, text="Top-p:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.top_p_step3_var = tk.StringVar(value=self._settings.get("top_p_step3", ""))
        ttk.Entry(step3, textvariable=self.top_p_step3_var, width=10).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(step3, text="Top-k:").grid(row=0, column=4, sticky="w", padx=(18, 0))
        self.top_k_step3_var = tk.StringVar(value=self._settings.get("top_k_step3", ""))
        ttk.Entry(step3, textvariable=self.top_k_step3_var, width=10).grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(step3, text="Repeat penalty:").grid(row=1, column=0, sticky="w", pady=6)
        self.repeat_penalty_step3_var = tk.StringVar(value=self._settings.get("repeat_penalty_step3", ""))
        ttk.Entry(step3, textvariable=self.repeat_penalty_step3_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(step3, text="Repeat last N:").grid(row=1, column=2, sticky="w", pady=6, padx=(18, 0))
        self.repeat_last_n_step3_var = tk.StringVar(value=self._settings.get("repeat_last_n_step3", ""))
        ttk.Entry(step3, textvariable=self.repeat_last_n_step3_var, width=10).grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(step3, text="Max tokens:").grid(row=1, column=4, sticky="w", pady=6, padx=(18, 0))
        self.max_tokens_step3_var = tk.StringVar(value=self._settings.get("max_tokens_step3", ""))
        ttk.Entry(step3, textvariable=self.max_tokens_step3_var, width=10).grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(step3, text="Min-p:").grid(row=2, column=0, sticky="w", pady=6)
        self.min_p_step3_var = tk.StringVar(value=self._settings.get("min_p_step3", ""))
        ttk.Entry(step3, textvariable=self.min_p_step3_var, width=10).grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(step3, text="Presence pen:").grid(row=3, column=0, sticky="w", pady=6)
        self.presence_penalty_step3_var = tk.StringVar(value=self._settings.get("presence_penalty_step3", ""))
        ttk.Entry(step3, textvariable=self.presence_penalty_step3_var, width=10).grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(step3, text="Frequency pen:").grid(row=3, column=2, sticky="w", pady=6, padx=(18, 0))
        self.frequency_penalty_step3_var = tk.StringVar(value=self._settings.get("frequency_penalty_step3", ""))
        ttk.Entry(step3, textvariable=self.frequency_penalty_step3_var, width=10).grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(step3, text="Step-3 decides the BELIEF UPDATE - the measurement core. Keep it conservative (low/inherited Temp). Step-3 always keeps thinking: a small Max tokens is auto-boosted on length failures instead of dropping the reasoning.",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=4, column=0, columnspan=6, sticky="w", pady=(8, 0))

        # -----------------------------
        # RAG / Retrieval (run-level, only active when world=open_rag or closed_strict_rag)
        # -----------------------------
        rag = ttk.LabelFrame(tab_run, text="RAG / Retrieval (active only when World = open_rag or closed_strict_rag)", padding=12)
        self._register_solo_irrelevant_frame(rag)
        rag.pack(fill="x", pady=(10, 0))

        ttk.Label(rag, text="Backend:").grid(row=0, column=0, sticky="w")
        self.rag_backend_var = tk.StringVar(value=self._settings.get("rag_backend", "off"))
        self.rag_backend_combo = ttk.Combobox(rag, textvariable=self.rag_backend_var, values=["off", "simple", "dense", "graph"], width=12, state="readonly")
        self.rag_backend_combo.grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(rag, text="(off = no retrieval. simple = word-overlap, transparent baseline. dense = embedding similarity via Ollama (pull nomic-embed-text first). graph = entity-chain retrieval for multi-hop corpora - needs <corpus>.graph.json next to the corpus. dense/graph = experimental retrieval-architecture conditions: pure top-k, content mode quotas do not apply.)", style="Muted.TLabel", wraplength=760, justify="left").grid(row=0, column=2, columnspan=4, sticky="w")

        ttk.Label(rag, text="Corpus path:").grid(row=1, column=0, sticky="w", pady=6)
        self.rag_corpus_path_var = tk.StringVar(value=self._settings.get("rag_corpus_path", ""))
        self.rag_corpus_entry = ttk.Entry(rag, textvariable=self.rag_corpus_path_var, width=72)
        self.rag_corpus_entry.grid(row=1, column=1, columnspan=4, sticky="we", padx=6)
        self.rag_browse_btn = ttk.Button(rag, text="Browse...", command=self._browse_rag_corpus)
        self.rag_browse_btn.grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(rag, text="Topic override:").grid(row=2, column=0, sticky="w", pady=6)
        self.rag_topic_override_var = tk.StringVar(value=self._settings.get("rag_topic_override", ""))
        self.rag_topic_override_entry = ttk.Entry(rag, textvariable=self.rag_topic_override_var, width=24)
        self.rag_topic_override_entry.grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(rag, text="(blank = auto from prompt version root, e.g. v52_default -> v52)", style="Muted.TLabel").grid(row=2, column=2, columnspan=4, sticky="w")

        ttk.Label(rag, text="Top-k:").grid(row=3, column=0, sticky="w", pady=6)
        self.rag_top_k_var = tk.StringVar(value=self._settings.get("rag_top_k", "4"))
        self.rag_top_k_entry = ttk.Entry(rag, textvariable=self.rag_top_k_var, width=10)
        self.rag_top_k_entry.grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(rag, text="Step2 query:").grid(row=3, column=2, sticky="w", padx=(18,0))
        self.rag_query_mode_step2_var = tk.StringVar(value=self._settings.get("rag_query_mode_step2", "auto"))
        self.rag_query_mode_step2_combo = ttk.Combobox(rag, textvariable=self.rag_query_mode_step2_var, values=["auto", "claim", "tweet", "claim_plus_tweet"], width=18, state="readonly")
        self.rag_query_mode_step2_combo.grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(rag, text="Step3 query:").grid(row=4, column=0, sticky="w", pady=6)
        self.rag_query_mode_step3_var = tk.StringVar(value=self._settings.get("rag_query_mode_step3", "claim_plus_tweet"))
        self.rag_query_mode_step3_combo = ttk.Combobox(rag, textvariable=self.rag_query_mode_step3_var, values=["auto", "claim", "tweet", "claim_plus_tweet"], width=18, state="readonly")
        self.rag_query_mode_step3_combo.grid(row=4, column=1, sticky="w", padx=6)

        ttk.Label(rag, text="Content mode:").grid(row=4, column=2, sticky="w", padx=(18,0))
        self.rag_content_mode_var = tk.StringVar(value=self._settings.get("rag_content_mode", "full"))
        self.rag_content_mode_combo = ttk.Combobox(rag, textvariable=self.rag_content_mode_var, values=["full", "balanced", "supportive_only", "criticism_only", "context_only"], width=18, state="readonly")
        self.rag_content_mode_combo.grid(row=4, column=3, sticky="w", padx=6)

        ttk.Label(rag, text="Max chars:").grid(row=3, column=4, sticky="w", padx=(18,0))
        self.rag_max_chars_var = tk.StringVar(value=self._settings.get("rag_max_chars", "2200"))
        self.rag_max_chars_entry = ttk.Entry(rag, textvariable=self.rag_max_chars_var, width=10)
        self.rag_max_chars_entry.grid(row=3, column=5, sticky="w", padx=6)

        ttk.Label(rag, text="Retrieval is not an LLM decoding stage - these control what text is fetched and injected into prompts.  Backend: off = no retrieval even in *_rag worlds; simple = lexical word-overlap scorer (deterministic + transparent; deliberately NOT a neural retriever, so evidence DIRECTION stays isolated from retriever quality).  Corpus path: folder of per-topic snippet files.  Top-k: how many snippets are shown (keep small, ~4, to not swamp the listener).  Max chars: hard cap on injected text.  Step2/Step3 query: what the lookup searches with (the claim, the tweet, or both; auto picks per step).  Content mode: which snippet direction passes the filter - full / balanced / supportive_only / criticism_only / context_only. THIS is the evidence-direction lever of the experiments; in closed worlds it can even flip hub influence.",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=5, column=0, columnspan=6, sticky="w")
        try:
            rag.grid_columnconfigure(1, weight=1)
            rag.grid_columnconfigure(3, weight=1)
        except Exception:
            pass
        try:
            self.world_combo.bind("<<ComboboxSelected>>", lambda e: (self._update_rag_controls_state(), self._update_true_open_controls_state()))
        except Exception:
            pass
        self._update_rag_controls_state()

        # -----------------------------
        # True open tools (active only when World = true_open)
        # -----------------------------
        true_open = ttk.LabelFrame(tab_run, text="True-open live tools (active only when World = true_open)", padding=12)
        self._register_solo_irrelevant_frame(true_open)
        true_open.pack(fill="x", pady=(10, 0))

        ttk.Label(true_open, text="Web backend:").grid(row=0, column=0, sticky="w")
        self.web_backend_var = tk.StringVar(value=self._settings.get("web_backend", "off"))
        self.web_backend_combo = ttk.Combobox(true_open, textvariable=self.web_backend_var, values=["off", "brave", "duckduckgo"], width=14, state="readonly")
        self.web_backend_combo.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(true_open, text="Planner:").grid(row=0, column=2, sticky="w", padx=(18,0))
        self.planner_mode_var = tk.StringVar(value=self._settings.get("planner_mode", "off"))
        self.planner_mode_combo = ttk.Combobox(true_open, textvariable=self.planner_mode_var, values=["off", "heuristic"], width=14, state="readonly")
        self.planner_mode_combo.grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(true_open, text="Tool mode:").grid(row=0, column=4, sticky="w", padx=(18,0))
        self.tool_mode_var = tk.StringVar(value=self._settings.get("tool_mode", "off"))
        self.tool_mode_combo = ttk.Combobox(true_open, textvariable=self.tool_mode_var, values=["off", "web_only", "multi"], width=14, state="readonly")
        self.tool_mode_combo.grid(row=0, column=5, sticky="w", padx=6)

        ttk.Label(true_open, text="Web top-k:").grid(row=1, column=0, sticky="w", pady=6)
        self.web_top_k_var = tk.StringVar(value=self._settings.get("web_top_k", "3"))
        self.web_top_k_entry = ttk.Entry(true_open, textvariable=self.web_top_k_var, width=10)
        self.web_top_k_entry.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(true_open, text="Web max chars:").grid(row=1, column=2, sticky="w", padx=(18,0))
        self.web_max_chars_var = tk.StringVar(value=self._settings.get("web_max_chars", "1400"))
        self.web_max_chars_entry = ttk.Entry(true_open, textvariable=self.web_max_chars_var, width=10)
        self.web_max_chars_entry.grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(true_open, text="Step2 web:").grid(row=2, column=2, sticky="w", padx=(18,0))
        self.step2_web_mode_var = tk.StringVar(value=self._settings.get("step2_web_mode", "heuristic"))
        self.step2_web_mode_combo = ttk.Combobox(true_open, textvariable=self.step2_web_mode_var, values=["off", "heuristic", "always"], width=14, state="readonly")
        self.step2_web_mode_combo.grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(true_open, text="Notes mode:").grid(row=1, column=4, sticky="w", padx=(18,0))
        self.notes_mode_var = tk.StringVar(value=self._settings.get("notes_mode", "off"))
        self.notes_mode_combo = ttk.Combobox(true_open, textvariable=self.notes_mode_var, values=["off", "heuristic"], width=14, state="readonly")
        self.notes_mode_combo.grid(row=1, column=5, sticky="w", padx=6)

        ttk.Label(true_open, text="Notes max items:").grid(row=2, column=0, sticky="w", pady=6)
        self.notes_max_items_var = tk.StringVar(value=self._settings.get("notes_max_items", "3"))
        self.notes_max_items_entry = ttk.Entry(true_open, textvariable=self.notes_max_items_var, width=10)
        self.notes_max_items_entry.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(true_open, text="Only active in the true_open world - agents get LIVE internet tools, so runs are NOT reproducible.  Web backend: search provider (off disables).  Planner: heuristic decides WHEN an agent searches.  Tool mode: web_only = search only; multi = search + notes.  Web top-k / max chars: how many results and how much text get injected.  Step2 web: whether tweet-writing may also search (off / heuristic / always; queries use claim + current belief only).  Notes: lightweight external memory of past searches, capped by Notes max items (separate from ConversationChain memory).",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=3, column=0, columnspan=6, sticky="w")
        try:
            true_open.grid_columnconfigure(1, weight=1)
            true_open.grid_columnconfigure(3, weight=1)
            true_open.grid_columnconfigure(5, weight=1)
        except Exception:
            pass
        try:
            self.tool_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._update_true_open_controls_state())
        except Exception:
            pass
        self._update_true_open_controls_state()
        # -----------------------------
        # Multiple runs (batch)
        # -----------------------------
        multi = ttk.LabelFrame(tab_run, text="Multiple runs", padding=12)
        multi.pack(fill="x", pady=(10, 0))

        ttk.Label(multi, text="Mode:").grid(row=0, column=0, sticky="w", pady=6)
        # UI labels are friendly; internal logic normalizes them.
        _mr_mode = (self._settings.get("multi_run_mode", "off") or "off").strip().lower()
        if _mr_mode not in ["off", "same_seed", "consecutive_seeds"]:
            _mr_mode = "off"
        # Display mapping
        _mr_display = {"off": "off", "same_seed": "same seed", "consecutive_seeds": "consecutive seeds"}[_mr_mode]
        self.multi_run_mode_var = tk.StringVar(value=_mr_display)
        self.multi_run_mode_combo = ttk.Combobox(
            multi,
            textvariable=self.multi_run_mode_var,
            values=["off", "same seed", "consecutive seeds"],
            width=18,
            state="readonly",
        )
        self.multi_run_mode_combo.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(multi, text="Runs:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.multi_run_count_var = tk.StringVar(value=str(self._settings.get("multi_run_count", "5")))
        self.multi_run_count_entry = ttk.Entry(multi, textvariable=self.multi_run_count_var, width=10)
        self.multi_run_count_entry.grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(multi, text="off = single run.  same seed = N repeats with the SAME seed (folders get _1,_2,... - measures LLM randomness only).  consecutive seeds = seed, seed+1, ... and the out-name's seedXX is rewritten per run (measures seed-to-seed variance - what the eval harness needs for CIs).  Runs: how many. Covers replication of ONE configuration; grids over conditions are a separate planned feature (ROADMAP).",
                  style="Muted.TLabel", wraplength=1000, justify="left").grid(row=1, column=0, columnspan=4, sticky="w")

        try:
            multi.grid_columnconfigure(3, weight=1)
        except Exception:
            pass

        self.multi_run_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._update_multi_run_controls_state())
        self._update_multi_run_controls_state()

        # Controls
        controls = ttk.Frame(tab_run, padding=(2, 12))
        controls.pack(fill="x")

        self.run_btn = ttk.Button(controls, text="Run", command=self.on_run, style="Accent.TButton")
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.on_stop, state="disabled", style="Danger.TButton")
        self.stop_btn.pack(side="left", padx=8)

        ttk.Button(controls, text="Open Log File", command=self.open_log_viewer).pack(side="left", padx=8)
        ttk.Button(controls, text="Clear Log", command=self.clear_log).pack(side="left", padx=8)

        ttk.Separator(tab_run, orient="horizontal").pack(fill="x", pady=(6, 0))

        hint = ttk.Label(
            tab_run,
            text="Tip: Configure personas on the Personas tab. Output will appear on the Logs tab.",
        )
        hint.pack(anchor="w", pady=(8, 0))

        # =============================
        # PERSONAS TAB
        # =============================

        persona = ttk.LabelFrame(tab_personas, text="Personas (creates NEW list_agent_descriptions.csv each run)", padding=12)
        self._register_solo_irrelevant_frame(persona)
        persona.pack(fill="x")

        self.use_personas_var = tk.BooleanVar(value=bool(self._settings["use_personas"]))
        ttk.Checkbutton(
            persona,
            text="Generate list_agent_descriptions.csv in selected prompt folder",
            variable=self.use_personas_var
        ).grid(row=0, column=0, sticky="w", columnspan=4)

        ttk.Label(persona, text="Opinion dist (v5):").grid(row=1, column=0, sticky="w", pady=6)
        self.opinion_strategy_var = tk.StringVar(value=self._settings["opinion_dist"])
        self.opinion_dist_combo = ttk.Combobox(
            persona,
            textvariable=self.opinion_strategy_var,
            values=["uniform", "skewed_positive", "skewed_negative", "positive", "negative", "custom_counts"],
            width=18,
            state="readonly",
        )
        self.opinion_dist_combo.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(persona, text="Custom counts (-2,-1,0,1,2):").grid(row=1, column=2, sticky="w")
        self.custom_counts_var = tk.StringVar(value=self._settings["custom_counts"])
        self.custom_counts_entry = ttk.Entry(persona, textvariable=self.custom_counts_var, width=18)
        self.custom_counts_entry.grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(persona, text="Names source CSV:").grid(row=2, column=0, sticky="w", pady=6)
        self.names_source_var = tk.StringVar(value=self._settings["names_source"])
        ttk.Entry(persona, textvariable=self.names_source_var, width=40).grid(row=2, column=1, sticky="w", padx=6, columnspan=3)

        # Classic preset controls (hidden when JSON persona mode is active)
        self.classic_persona_frame = ttk.LabelFrame(persona, text="Classic preset controls (the active path while JSON persona profiles are OFF)", padding=8)
        self.classic_persona_frame.grid(row=3, column=0, columnspan=4, sticky="we", pady=(8, 0))

        ttk.Label(self.classic_persona_frame, text="Age preset ▼").grid(row=0, column=0, sticky="w", pady=6)
        self.age_preset_var = tk.StringVar(value=self._settings["age_preset"])
        self.age_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.age_preset_var,
                     values=["none", "equal", "random", "younger", "older", "all_young", "all_old"], width=18, state="readonly")
        self.age_preset_combo.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Gender preset ▼").grid(row=0, column=2, sticky="w")
        self.gender_preset_var = tk.StringVar(value=self._settings["gender_preset"])
        self.gender_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.gender_preset_var,
                     values=["none", "equal", "random", "more_male", "more_female", "all_male", "all_female"], width=18, state="readonly")
        self.gender_preset_combo.grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Education preset ▼").grid(row=1, column=0, sticky="w", pady=6)
        self.educ_preset_var = tk.StringVar(value=self._settings["educ_preset"])
        self.educ_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.educ_preset_var,
                     values=["none", "equal", "random", "more_educated", "less_educated", "all_educated", "all_uneducated"], width=18, state="readonly")
        self.educ_preset_combo.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Political preset ▼").grid(row=1, column=2, sticky="w")
        self.pol_preset_var = tk.StringVar(value=self._settings["pol_preset"])
        self.pol_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.pol_preset_var,
                     values=["none", "equal", "random", "centered", "more_right", "most_right", "more_left", "most_left", "all_left", "all_right"], width=18, state="readonly")
        self.pol_preset_combo.grid(row=1, column=3, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Ethnicity preset ▼").grid(row=2, column=0, sticky="w", pady=6)
        self.eth_preset_var = tk.StringVar(value=self._settings["eth_preset"])
        self.eth_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.eth_preset_var,
                     values=["none","equal","random","all_caucasian","all_black","all_african_american","all_asian","all_asian_american","all_latino","all_hispanic","all_middle_eastern","all_native_american","all_mixed","all_other","more_caucasian","more_black","more_african_american","more_asian","more_asian_american","more_latino","more_hispanic","most_caucasian","most_black","most_african_american","most_asian","most_asian_american","most_latino","most_hispanic"], width=18, state="readonly")
        self.eth_preset_combo.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Occupation preset ▼").grid(row=2, column=2, sticky="w")
        self.occ_preset_var = tk.StringVar(value=self._settings["occ_preset"])
        self.occ_preset_combo = ttk.Combobox(self.classic_persona_frame, textvariable=self.occ_preset_var,
                     values=["none","equal","random","all_unemployed","all_doctors","all_historians","all_mathematicians","all_computer_engineers","all_politicians","more_doctors","more_historians","more_mathematicians","more_computer_engineers","more_politicians","most_doctors","most_historians","most_mathematicians","most_computer_engineers","most_politicians","more_journalists","more_scientists","more_lawyers","more_economists","more_social_workers"], width=18, state="readonly")
        self.occ_preset_combo.grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Early life preset ▼").grid(row=3, column=0, sticky="w", pady=6)
        self.early_life_preset_var = tk.StringVar(value=self._settings.get("early_life_preset", "random"))
        self.early_life_preset_combo = ttk.Combobox(
            self.classic_persona_frame,
            textvariable=self.early_life_preset_var,
            values=["none","random","equal","all_difficult","all_ok","all_good","more_difficult","more_ok","more_good","most_difficult","most_ok","most_good"],
            width=18,
            state="readonly"
        )
        self.early_life_preset_combo.grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame, text="Epistemic profile preset ▼").grid(row=3, column=2, sticky="w")
        self.epistemic_profile_preset_var = tk.StringVar(value=self._settings.get("epistemic_profile_preset", "none"))
        self.epistemic_profile_preset_combo = ttk.Combobox(
            self.classic_persona_frame,
            textvariable=self.epistemic_profile_preset_var,
            values=[
                "none", "random", "equal",
                "all_trusting_pragmatist", "all_suspicious_skeptic", "all_open_minded_skeptic",
                "all_authority_stabilizer", "all_uncertainty_tolerant_agnostic", "all_practical_distruster",
                "more_trusting_pragmatist", "more_suspicious_skeptic", "more_open_minded_skeptic",
                "more_authority_stabilizer", "more_uncertainty_tolerant_agnostic", "more_practical_distruster",
                "most_trusting_pragmatist", "most_suspicious_skeptic", "most_open_minded_skeptic",
                "most_authority_stabilizer", "most_uncertainty_tolerant_agnostic", "most_practical_distruster"
            ],
            width=26,
            state="readonly"
        )
        self.epistemic_profile_preset_combo.grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(self.classic_persona_frame,
                  text="Legacy demographic sampling (hidden when 'Use JSON persona profiles' is ON below). Value pattern everywhere: none = leave blank, equal = even split, random = uniform draw, more_/most_X = tilt toward X, all_X = everyone is X.  Age/Gender/Education/Political/Ethnicity/Occupation/Early life shape wording and background flavor only - they do NOT steer belief updates.  Epistemic profile preset is the exception: it assigns the 6 core archetypes (trusting pragmatist, suspicious skeptic, open-minded skeptic, authority stabilizer, uncertainty-tolerant agnostic, practical distruster) and DOES steer belief behaviour.",
                  style="Muted.TLabel", wraplength=1100, justify="left").grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        try:
            self.classic_persona_frame.grid_columnconfigure(1, weight=1)
            self.classic_persona_frame.grid_columnconfigure(3, weight=1)
        except Exception:
            pass

        ttk.Label(persona, text="CSV path (auto):").grid(row=4, column=0, sticky="w", pady=6)
        self.csv_path_var = tk.StringVar(value="")
        ttk.Entry(persona, textvariable=self.csv_path_var, width=82, state="readonly")             .grid(row=4, column=1, columnspan=3, sticky="w", padx=6)

        ttk.Label(persona, text="Generate checkbox: each run writes a FRESH list_agent_descriptions.csv into the selected prompt folder (personas regenerate per run; off = the existing CSV is used as-is).  Opinion dist: HOW initial -2..+2 ratings are assigned.  uniform = equal fifths across -2,-1,0,+1,+2 (if Agents is not divisible by 5, the remainder fills starting from -2).  skewed_positive = 4/5 of agents at +2 and 1/5 at -2, NOTHING in between.  skewed_negative = the mirror: 4/5 at -2, 1/5 at +2.  positive / negative = EVERY agent starts at +2 / -2 (consensus start).  custom_counts = you type the five counts yourself.  Positions are then shuffled by Seed, so WHO gets each value is seed-dependent but the counts are exact.  Custom counts: five comma-separated numbers (one per -2..+2 bucket, must sum to Agents); active only when dist = custom_counts.  Names source CSV: where agent names are drawn from.  CSV path (auto): the computed output target, read-only.",
                  style="Muted.TLabel", wraplength=1150, justify="left").grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))

        # Disable custom counts until selected
        self._set_custom_counts_enabled()
        try:
            self.opinion_dist_combo.bind("<<ComboboxSelected>>", lambda e: self._set_custom_counts_enabled())
        except Exception:
            pass

        # Slightly nicer column behavior
        try:
            persona.grid_columnconfigure(1, weight=1)
            persona.grid_columnconfigure(3, weight=1)
        except Exception:
            pass

        self._build_json_persona_section(tab_personas)

        # =============================
        # NETWORK TAB
        # =============================

        network = ttk.LabelFrame(
            tab_network,
            text="Network (applies to network-capable scripts)",
            padding=12
        )
        network.pack(fill="x")

        # Network type
        self._register_solo_irrelevant_frame(network)
        ttk.Label(network, text="Network type:").grid(row=0, column=0, sticky="w", pady=6)
        self.network_type_var = tk.StringVar(value=self._settings.get("network_type", "ws"))
        self.network_type_combo = ttk.Combobox(
            network,
            textvariable=self.network_type_var,
            values=["ws", "er", "ba", "none"],
            width=18,
            state="readonly",
        )
        self.network_type_combo.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(
            network,
            text="ws = small-world, er = Erdős–Rényi, ba = scale-free hubs, none = fully-mixed.",
            style="Muted.TLabel",
        ).grid(row=0, column=2, columnspan=3, sticky="w")

        # Homophily at formation (only matters when a network is used)
        self.network_homophily_var = tk.BooleanVar(value=bool(self._settings.get("network_homophily", False)))
        self.network_homophily_check = ttk.Checkbutton(
            network,
            text="Homophily at network formation (similar agents become neighbors)",
            variable=self.network_homophily_var,
        )
        self.network_homophily_check.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 2))

        # WS params
        ttk.Label(network, text="k_neighbors (even):").grid(row=2, column=0, sticky="w", pady=6)
        self.k_neighbors_var = tk.StringVar(value=self._settings.get("k_neighbors", "4"))
        self.k_neighbors_entry = ttk.Entry(network, textvariable=self.k_neighbors_var, width=10)
        self.k_neighbors_entry.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(network, text="p_rewire (0..1):").grid(row=2, column=2, sticky="w", pady=6)
        self.p_rewire_var = tk.StringVar(value=self._settings.get("p_rewire", "0.1"))
        self.p_rewire_entry = ttk.Entry(network, textvariable=self.p_rewire_var, width=10)
        self.p_rewire_entry.grid(row=2, column=3, sticky="w", padx=6)

        ttk.Label(network, text="er_p_edge (0..1):").grid(row=3, column=0, sticky="w", pady=6)
        self.er_p_edge_var = tk.StringVar(value=self._settings.get("er_p_edge", "0.15"))
        self.er_p_edge_entry = ttk.Entry(network, textvariable=self.er_p_edge_var, width=10)
        self.er_p_edge_entry.grid(row=3, column=1, sticky="w", padx=6)

        ttk.Label(network, text="ba_m_attach:").grid(row=3, column=2, sticky="w", pady=6)
        self.ba_m_attach_var = tk.StringVar(value=self._settings.get("ba_m_attach", "2"))
        self.ba_m_attach_entry = ttk.Entry(network, textvariable=self.ba_m_attach_var, width=10)
        self.ba_m_attach_entry.grid(row=3, column=3, sticky="w", padx=6)

        ttk.Label(network, text="BA hub priority:").grid(row=4, column=0, sticky="w", pady=6)
        self.ba_hub_strategy_var = tk.StringVar(value=self._settings.get("ba_hub_strategy", "default"))
        self.ba_hub_strategy_combo = ttk.Combobox(
            network,
            textvariable=self.ba_hub_strategy_var,
            values=["default", "random", "positive", "negative", "neutral", "extreme", "opposite_majority", "custom"],
            width=18,
            state="readonly",
        )
        self.ba_hub_strategy_combo.grid(row=4, column=1, sticky="w", padx=6)

        ttk.Label(network, text="Custom hubs:").grid(row=4, column=2, sticky="w", pady=6)
        self.ba_hub_custom_var = tk.StringVar(value=self._settings.get("ba_hub_custom", ""))
        self.ba_hub_custom_entry = ttk.Entry(network, textvariable=self.ba_hub_custom_var, width=28)
        self.ba_hub_custom_entry.grid(row=4, column=3, sticky="we", padx=6)

        ttk.Label(network, text="BA hub assignment mode:").grid(row=5, column=0, sticky="w", pady=6)
        self.ba_hub_assignment_mode_var = tk.StringVar(value=self._settings.get("ba_hub_assignment_mode", "early_position"))
        self.ba_hub_assignment_mode_combo = ttk.Combobox(
            network,
            textvariable=self.ba_hub_assignment_mode_var,
            values=["early_position", "actual_hubs", "early_and_actual"],
            width=18,
            state="disabled",
        )
        self.ba_hub_assignment_mode_combo.grid(row=5, column=1, sticky="w", padx=6)
        ttk.Label(
            network,
            text="early_position = current BA advantage; actual_hubs = selected group gets realized top-degree hubs; early_and_actual = both.",
            style="Muted.TLabel",
        ).grid(row=5, column=2, columnspan=3, sticky="w")

        ttk.Label(
            network,
            text="k_neighbors (ws): even number of neighbours per node before rewiring.  p_rewire (ws): fraction of edges rewired randomly (0 = ring lattice, 1 = fully random).  er_p_edge (er): probability each pair gets an edge.  ba_m_attach (ba): edges each new node brings (2 = sparse net with clear hubs).  BA hub priority: which stance gets the high-degree hubs (positive/negative = seed hubs with that belief; extreme / opposite_majority / custom variants).  Custom hubs: idx:0, agent_id:12 or exact names - BA/custom only.  Homophily at formation: similar agents become neighbours when the net is built.  Fields grey out based on Network type.",
            style="Muted.TLabel", wraplength=1100, justify="left",
        ).grid(row=6, column=0, columnspan=5, sticky="w", pady=(0, 4))

        # Interaction mode (who the listener talks to)
        interaction = ttk.LabelFrame(network, text="Interaction mode", padding=10)
        interaction.grid(row=7, column=0, columnspan=5, sticky="we", pady=(10, 0))

        self.interaction_label = ttk.Label(interaction, text="Neighbor selection:")
        self.interaction_label.grid(row=0, column=0, sticky="w")

        self.interaction_selection_var = tk.StringVar(value=self._settings.get("interaction_selection", "homophily"))
        self.interaction_selection_combo = ttk.Combobox(
            interaction,
            textvariable=self.interaction_selection_var,
            values=["random", "homophily"],
            state="readonly",
            width=12,
        )
        self.interaction_selection_combo.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(interaction, text="Homophily scoring:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.interaction_homophily_mode_var = tk.StringVar(value=self._settings.get("interaction_homophily_mode", "full"))
        self.interaction_homophily_mode_combo = ttk.Combobox(
            interaction,
            textvariable=self.interaction_homophily_mode_var,
            values=["full", "opinion_only"],
            state="readonly",
            width=14,
        )
        self.interaction_homophily_mode_combo.grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(
            interaction,
            text="Who the listener hears each step. random = uniform pick among neighbours; homophily = similarity-weighted pick with epsilon exploration.  Homophily scoring: full = traits + opinion distance, opinion_only = opinion distance alone. Affects WHO talks to WHOM - never the update rule itself.",
            style="Muted.TLabel", wraplength=1100, justify="left",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))

        # Keep scoring-mode combobox in sync when selection changes
        try:
            self.interaction_selection_combo.bind("<<ComboboxSelected>>", lambda e: self._update_network_controls_state())
        except Exception:
            pass

        # Grey-out logic when network is disabled
        self._update_network_controls_state()
        try:
            self.network_type_combo.bind("<<ComboboxSelected>>", lambda e: self._update_network_controls_state())
            self.ba_hub_strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._update_network_controls_state())
        except Exception:
            pass
        try:
            self.network_type_var.trace_add("write", lambda *_: self._update_live_network_toggle_visibility())
        except Exception:
            pass

        # nicer column behavior (match Persona tab layout: avoid stretching the label column)
        try:
            network.grid_columnconfigure(1, weight=1)
            network.grid_columnconfigure(3, weight=1)
            network.grid_columnconfigure(2, weight=0)
        except Exception:
            pass

        # =============================
        # LOGS TAB
        # =============================
        log_frame = ttk.LabelFrame(tab_logs, text="Output", padding=10)
        log_frame.pack(fill="both", expand=True)
        ttk.Label(log_frame, text="Left: live stdout/stderr of the run (read-only). Right: step waypoints - click one to jump the log there. Verbosity is set by Trace on the Run tab; Clear Log only clears this view, never files.",
                  style="Muted.TLabel", wraplength=1100, justify="left").pack(anchor="w", pady=(0, 6))

        # Draggable split: output text (left) and step waypoints (right)
        log_paned = ttk.PanedWindow(log_frame, orient="horizontal")
        log_paned.pack(fill="both", expand=True)

        out_frame = ttk.Frame(log_paned)
        wp_frame = ttk.Frame(log_paned)

        log_paned.add(out_frame, weight=4)
        log_paned.add(wp_frame, weight=1)

        # Output text
        self.log = tk.Text(
            out_frame,
            wrap="none",
            font=("Consolas", 9),
            bg="#0f1730",
            fg="#e6e8ef",
            insertbackground="#e6e8ef",
            relief="flat",
        )
        self.log.grid(row=0, column=0, sticky="nsew")

        scroll = AutoScrollbar(out_frame, orient="vertical", command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        hscroll = AutoScrollbar(out_frame, orient="horizontal", command=self.log.xview)
        hscroll.grid(row=1, column=0, sticky="ew")
        self.log.configure(yscrollcommand=scroll.set, xscrollcommand=hscroll.set)

        # Make log read-only (prevents accidental typing that looks like stdin input)
        try:
            self.log.configure(state="disabled")
            self.log.bind("<Key>", lambda e: "break")
        except Exception:
            pass

        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)

        # Log styling tags: differentiate agent output vs prompt/system/meta
        try:
            # Agent/LLM output: warm light color (no background) to reduce glare but keep separation
            self.log.tag_configure("agent", foreground="#f3e8d2", font=("Consolas", 9, "bold"))
            # Prompt/template text: green
            self.log.tag_configure("prompt", foreground="#22c55e")
            # System text: violet
            self.log.tag_configure("system", foreground="#8b5cf6")
            # Meta text: muted gray
            self.log.tag_configure("meta", foreground="#b6bdd6")
            # Errors/tracebacks: red/pink
            self.log.tag_configure("error", foreground="#fb7185")
            # Section separator
            self.log.tag_configure("sep", foreground="#44507a")
        except Exception:
            pass

        self._in_prompt_block = False

        # Waypoints panel (bookmarks)
        ttk.Label(wp_frame, text="Waypoints", style="Muted.TLabel").pack(anchor="w", pady=(2, 6))

        self.waypoints_list = tk.Listbox(
            wp_frame,
            height=28,
            bg="#0f1730",
            fg="#cbd5e1",
            selectbackground="#1f2a55",
            selectforeground="#ffffff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#233054",
        )
        self.waypoints_list.pack(fill="both", expand=True)

        wp_scroll = AutoScrollbar(wp_frame, orient="vertical", command=self.waypoints_list.yview)
        wp_scroll.pack(side="right", fill="y")
        self.waypoints_list.configure(yscrollcommand=wp_scroll.set)

        try:
            self._waypoint_seed_font = tkfont.Font(self.waypoints_list, self.waypoints_list.cget("font"))
            self._waypoint_seed_font.configure(underline=1, weight="bold")
        except Exception:
            self._waypoint_seed_font = None

        self.waypoints_list.bind("<<ListboxSelect>>", self._jump_to_waypoint)



    # ---- UI helpers ----

    def _canvas_can_scroll_y(self, canvas: tk.Canvas) -> bool:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            first, last = canvas.yview()
            return (float(first) > 0.0) or (float(last) < 1.0)
        except Exception:
            return False

    def _canvas_can_scroll_x(self, canvas: tk.Canvas) -> bool:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            first, last = canvas.xview()
            return (float(first) > 0.0) or (float(last) < 1.0)
        except Exception:
            return False

    def _bind_global_notebook_mousewheel(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        def _on_mousewheel(event):
            """Scroll the active tab's canvas - unless the event comes from an
            open combobox dropdown (its floating popdown must scroll itself,
            not drag the page underneath it around)."""
            try:
                if "popdown" in str(getattr(event, "widget", "") or "").lower():
                    return
                current_tab = self.nb.select()
                canvas = getattr(self, "_tab_scroll_canvases", {}).get(str(current_tab))
                if canvas is None or not self._canvas_can_scroll_y(canvas):
                    return
                if getattr(event, "delta", 0):
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif getattr(event, "num", None) == 4:
                    canvas.yview_scroll(-3, "units")
                elif getattr(event, "num", None) == 5:
                    canvas.yview_scroll(3, "units")
            except Exception:
                pass

        def _on_shift_mousewheel(event):
            """Horizontal counterpart of _on_mousewheel, with the same popdown guard."""
            try:
                if "popdown" in str(getattr(event, "widget", "") or "").lower():
                    return
                current_tab = self.nb.select()
                canvas = getattr(self, "_tab_scroll_canvases", {}).get(str(current_tab))
                if canvas is None or not self._canvas_can_scroll_x(canvas):
                    return
                if getattr(event, "delta", 0):
                    canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
                elif getattr(event, "num", None) == 4:
                    canvas.xview_scroll(-3, "units")
                elif getattr(event, "num", None) == 5:
                    canvas.xview_scroll(3, "units")
            except Exception:
                pass

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.bind_all(seq, _on_mousewheel, add="+")
            except Exception:
                pass
        try:
            self.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel, add="+")
        except Exception:
            pass

        def _combo_wheel(event):
            """Wheel over a CLOSED combobox scrolls the page; the default ttk
            behavior (cycling the selected value) is suppressed - accidental
            value changes while scrolling were a real hazard."""
            _on_mousewheel(event)
            return "break"

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            try:
                self.bind_class("TCombobox", seq, _combo_wheel)
            except Exception:
                pass

        def _close_floating_popdown(_event=None):
            """A dropdown left open during a tab switch keeps floating over the
            new tab; send it Escape so it closes with the switch."""
            try:
                f = self.focus_get()
            except Exception:
                f = None
            try:
                if f is not None and "popdown" in str(f).lower():
                    f.event_generate("<Escape>")
            except Exception:
                pass

        try:
            self.nb.bind("<<NotebookTabChanged>>", _close_floating_popdown, add="+")
        except Exception:
            pass

    def _create_scrollable_notebook_tab(self, title: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        outer = ttk.Frame(self.nb)
        self.nb.add(outer, text=title)

        container = ttk.Frame(outer)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg="#070b17", highlightthickness=0, bd=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        vbar = AutoScrollbar(container, orient="vertical", command=canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = AutoScrollbar(container, orient="horizontal", command=canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        inner = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _update_scrollregion(event=None):
            """Synchronize UI state or derived values after a setting changes."""
            try:
                bbox = canvas.bbox("all")
                if bbox:
                    canvas.configure(scrollregion=bbox)
            except Exception:
                pass

        def _fit_width(event=None):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            try:
                canvas.itemconfigure(window_id, width=max(1, canvas.winfo_width()))
            except Exception:
                pass
            _update_scrollregion()

        inner.bind("<Configure>", _update_scrollregion)
        canvas.bind("<Configure>", _fit_width)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        if not hasattr(self, "_tab_scroll_canvases"):
            self._tab_scroll_canvases = {}
        self._tab_scroll_canvases[str(outer)] = canvas

        self.after_idle(lambda: (canvas.yview_moveto(0), canvas.xview_moveto(0)))

        return outer, inner

    def _on_notebook_tab_changed(self, event=None):
        """Handle a Tkinter/UI event and update dependent controls."""
        try:
            current_tab = self.nb.select()
            canvas = getattr(self, "_tab_scroll_canvases", {}).get(str(current_tab))
            if canvas is not None:
                canvas.yview_moveto(0)
                canvas.xview_moveto(0)
        except Exception:
            pass

    def _append_seed_waypoint_header(self, seed_label: str, text_index: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "waypoints_list"):
            return
        try:
            list_idx = self.waypoints_list.size()
            self.waypoints_list.insert("end", seed_label)
            try:
                self.waypoints_list.itemconfig(list_idx, fg="#22d3ee")
                if getattr(self, "_waypoint_seed_font", None) is not None:
                    self.waypoints_list.itemconfig(list_idx, font=self._waypoint_seed_font)
            except Exception:
                pass
            self._waypoint_entry_map[list_idx] = text_index
        except Exception:
            pass

    def _append_step_waypoint(self, label: str, text_index: str, changed: bool = False):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "waypoints_list"):
            return
        try:
            if not hasattr(self, "_step_waypoint_list_idx"):
                self._step_waypoint_list_idx = {}
            if not hasattr(self, "_step_waypoint_changed"):
                self._step_waypoint_changed = {}
            if not hasattr(self, "_step_waypoint_label"):
                self._step_waypoint_label = {}
            if not hasattr(self, "_step_waypoint_markers"):
                self._step_waypoint_markers = {}

            step_key = str(label).strip().lower()
            # Keep markers that may have been discovered earlier from the live CSV.
            existing_markers = list(self._step_waypoint_markers.get(step_key, []))
            self._step_waypoint_label[step_key] = str(label)
            list_idx = self.waypoints_list.size()
            self.waypoints_list.insert("end", self._render_step_waypoint_label(step_key))
            self._waypoint_entry_map[list_idx] = text_index
            self._step_waypoint_list_idx[step_key] = list_idx
            self._step_waypoint_changed[step_key] = bool(existing_markers or changed)
            self._step_waypoint_markers[step_key] = existing_markers
            try:
                self._apply_step_waypoint_color(step_key, list_idx)
            except Exception:
                pass
        except Exception:
            pass

    def _waypoint_marker_for_rating(self, rating: int | None) -> str:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            r = int(rating)
        except Exception:
            return "•"
        # Keep markers compact so they read as small badges next to each step.
        # Color semantics requested by the user:
        # -2 red, -1 orange, 0 white, +1 yellow, +2 green.
        mapping = {
            -2: "🔴",
            -1: "🟠",
             0: "⚪",
             1: "🟡",
             2: "🟢",
        }
        return mapping.get(r, "•")

    def _render_step_waypoint_label(self, step_key: str, step_num: str | int | None = None) -> str:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        base_label = getattr(self, "_step_waypoint_label", {}).get(step_key)
        if not base_label:
            base_label = f"Step {step_num}" if step_num is not None else "Step"
        markers = list(getattr(self, "_step_waypoint_markers", {}).get(step_key, []))
        if not markers:
            return f"  {base_label}"
        marker_text = " ".join(self._waypoint_marker_for_rating(m) for m in markers)
        return f"  {base_label}  {marker_text}"

    def _extract_rating_change_destination(self, line_text: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        s = str(line_text or "").strip()
        if not s:
            return None
        low = s.lower()
        if "allowed_final_rating_set" in low:
            return None

        # First, trust the explicit protocol fields whenever they are present.
        m_pre = re.search(r"\bpre[_ -]?belief\b\s*[:=]\s*([+-]?2|[+-]?1|0)\b", low, flags=re.IGNORECASE)
        if m_pre:
            try:
                self._pending_pre_belief = int(m_pre.group(1))
            except Exception:
                self._pending_pre_belief = None
            self._pending_change_emitted = False
            return None

        m_post = re.search(r"\bpost[_ -]?belief\b\s*[:=]\s*([+-]?2|[+-]?1|0)\b", low, flags=re.IGNORECASE)
        if m_post:
            try:
                post_v = int(m_post.group(1))
            except Exception:
                post_v = None
            self._pending_post_belief = post_v
            pre_v = getattr(self, "_pending_pre_belief", None)
            if post_v is not None and pre_v is not None and post_v != pre_v:
                if getattr(self, "_pending_change_emitted", False):
                    return None
                self._pending_change_emitted = True
                return post_v
            return None

        m_delta = re.search(r"\bdelta[_ -]?belief\b\s*[:=]\s*([+-]?2|[+-]?1|0)\b", low, flags=re.IGNORECASE)
        if m_delta:
            try:
                delta_v = int(m_delta.group(1))
            except Exception:
                delta_v = 0
            if delta_v == 0:
                return None

            pre_v = getattr(self, "_pending_pre_belief", None)
            post_v = getattr(self, "_pending_post_belief", None)
            if getattr(self, "_pending_change_emitted", False):
                return None
            if post_v is not None and pre_v is not None and post_v != pre_v:
                self._pending_change_emitted = True
                return post_v
            if pre_v is not None and post_v is None:
                try:
                    inferred = int(pre_v) + int(delta_v)
                except Exception:
                    inferred = None
                if inferred in {-2, -1, 0, 1, 2} and inferred != pre_v:
                    self._pending_change_emitted = True
                    return inferred
            return None

        # Fallback: some runs expose only compact textual change summaries in the main log.
        # Use these as a backup, but keep the parser conservative so same-opinion interactions
        # do not get marked as changes.
        if not re.search(r"\b(?:opinion|belief|rating|change|changed|moved|move|shifted|shift|updated|update|delta)\b", low):
            return None

        arrow_patterns = [
            r"\b(?:opinion|belief|rating|delta)[^\n]{0,40}?([+-]?2|[+-]?1|0)\s*[-=]?>\s*([+-]?2|[+-]?1|0)\b",
            r"\b(?:moved|move|changed|change|shifted|shift|updated|update|went|goes|go|from)\b.{0,40}?\b([+-]?2|[+-]?1|0)\b.{0,20}?\b(?:to|into|toward|towards|->|=>)\b.{0,20}?\b([+-]?2|[+-]?1|0)\b",
            r"\b(?:rating|belief|opinion)\b.{0,20}?\b([+-]?2|[+-]?1|0)\b.{0,20}?\b(?:to|->|=>)\b.{0,20}?\b([+-]?2|[+-]?1|0)\b",
        ]
        for pat in arrow_patterns:
            m = re.search(pat, low, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                a = int(m.group(1))
                b = int(m.group(2))
            except Exception:
                continue
            if a != b:
                return b

        return None

    def _waypoint_color_for_rating(self, rating: int | None) -> str:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            r = int(rating)
        except Exception:
            return "#cbd5e1"
        # Destination color mapping:
        # -2 red, -1 orange, 0 white/gray, +1 yellow, +2 green.
        return {
            -2: "#ef4444",
            -1: "#f97316",
             0: "#d1d5db",
             1: "#facc15",
             2: "#22c55e",
        }.get(r, "#cbd5e1")

    def _apply_step_waypoint_color(self, step_key: str, list_idx: int | None = None):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            if list_idx is None:
                list_idx = getattr(self, "_step_waypoint_list_idx", {}).get(step_key)
            if list_idx is None or not hasattr(self, "waypoints_list"):
                return
            markers = list(getattr(self, "_step_waypoint_markers", {}).get(step_key, []))
            if markers:
                self.waypoints_list.itemconfig(list_idx, fg=self._waypoint_color_for_rating(markers[-1]))
            else:
                self.waypoints_list.itemconfig(list_idx, fg="#cbd5e1")
        except Exception:
            pass

    def _set_step_waypoint_markers(self, step_num: str | int | None, markers):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if step_num is None:
            return
        step_key = f"step {str(step_num).strip()}".lower()
        try:
            if not hasattr(self, "_step_waypoint_markers"):
                self._step_waypoint_markers = {}
            if not hasattr(self, "_step_waypoint_changed"):
                self._step_waypoint_changed = {}
            if not hasattr(self, "_step_waypoint_list_idx"):
                self._step_waypoint_list_idx = {}
            if not hasattr(self, "_step_waypoint_label"):
                self._step_waypoint_label = {}
            cleaned = []
            for m in list(markers or []):
                try:
                    mi = int(m)
                except Exception:
                    continue
                if mi in {-2, -1, 0, 1, 2}:
                    cleaned.append(mi)
            self._step_waypoint_markers[step_key] = cleaned
            self._step_waypoint_changed[step_key] = bool(cleaned)

            list_idx = getattr(self, "_step_waypoint_list_idx", {}).get(step_key)
            if list_idx is None or not hasattr(self, "waypoints_list"):
                return

            label = self._render_step_waypoint_label(step_key, step_num)
            self.waypoints_list.delete(list_idx)
            self.waypoints_list.insert(list_idx, label)
            self._apply_step_waypoint_color(step_key, list_idx)
        except Exception:
            pass

    def _mark_step_waypoint_changed(self, step_num: str | int | None, destination_rating: int | None = None):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if step_num is None:
            return
        step_key = f"step {str(step_num).strip()}".lower()
        try:
            markers = list(getattr(self, "_step_waypoint_markers", {}).get(step_key, []))
            if destination_rating is not None:
                try:
                    dest_i = int(destination_rating)
                except Exception:
                    return
                if dest_i not in {-2, -1, 0, 1, 2}:
                    return
                # Avoid duplicate PRE/POST/DELTA detections for the same interaction.
                if not markers or markers[-1] != dest_i:
                    markers.append(dest_i)
            self._set_step_waypoint_markers(step_num, markers)
        except Exception:
            pass

    def _sync_waypoints_from_live_data(self):
        """Backfill waypoint markers from the live opinion CSV.

        This fixes missed log-parser detections by comparing consecutive opinion vectors.
        The marker color/emoji represents the destination rating of the changed agent.
        """
        try:
            steps = list(getattr(self, "_live_steps", []) or [])
            vectors = list(getattr(self, "_live_vectors", []) or [])
        except Exception:
            return
        if len(steps) < 2 or len(vectors) < 2:
            return

        for idx in range(1, min(len(steps), len(vectors))):
            try:
                step_num = steps[idx]
                prev_vec = list(vectors[idx - 1])
                cur_vec = list(vectors[idx])
            except Exception:
                continue
            limit = min(len(prev_vec), len(cur_vec))
            markers = []
            for j in range(limit):
                try:
                    a = int(prev_vec[j])
                    b = int(cur_vec[j])
                except Exception:
                    continue
                if a != b and b in {-2, -1, 0, 1, 2}:
                    markers.append(b)
            self._set_step_waypoint_markers(step_num, markers)


    def append_log_threadsafe(self, text: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self.after(0, lambda: self._append_log(text))

    def _jump_to_waypoint(self, event=None):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "waypoints_list"):
            return
        sel = self.waypoints_list.curselection()
        if not sel:
            return
        list_idx = int(sel[0])
        idx = getattr(self, "_waypoint_entry_map", {}).get(list_idx)
        if not idx:
            return

        self.log.see(idx)
        try:
            self.log.mark_set("insert", idx)
        except Exception:
            pass

        # Brief highlight for the step line
        try:
            self.log.tag_remove("jump_hl", "1.0", "end")
            self.log.tag_configure("jump_hl", background="#1f2a55")
            self.log.tag_add("jump_hl", idx, f"{idx} lineend")
            self.after(1200, lambda: self.log.tag_remove("jump_hl", "1.0", "end"))
        except Exception:
            pass


    def _append_log(self, text: str):
        """
        Append text to the log widget with basic classification:
        - agent: model/agent natural language output
        - prompt: prompt/template text
        - system: system instructions
        - meta: runner messages, configuration, progress
        - error: tracebacks / errors

        IMPORTANT: We strip ANSI escape codes (e.g., \x1b[32;1m) so classification works and the log stays readable.
        """
        if not text:
            return

        # Temporarily enable insertions; log is read-only otherwise
        try:
            self.log.configure(state="normal")
        except Exception:
            pass

        if not hasattr(self, "_last_log_tag"):
            self._last_log_tag = None

        if not hasattr(self, "_waypoint_entry_map"):
            self._waypoint_entry_map = {}
        if not hasattr(self, "_current_run_seen_waypoints"):
            self._current_run_seen_waypoints = set()
        if not hasattr(self, "_current_waypoint_seed_key"):
            self._current_waypoint_seed_key = None
        if not hasattr(self, "_step_waypoint_list_idx"):
            self._step_waypoint_list_idx = {}
        if not hasattr(self, "_step_waypoint_changed"):
            self._step_waypoint_changed = {}
        if not hasattr(self, "_step_waypoint_label"):
            self._step_waypoint_label = {}
        if not hasattr(self, "_current_log_step_num"):
            self._current_log_step_num = None
        if not hasattr(self, "_step_waypoint_markers"):
            self._step_waypoint_markers = {}
        if not hasattr(self, "_pending_pre_belief"):
            self._pending_pre_belief = None
        if not hasattr(self, "_pending_post_belief"):
            self._pending_post_belief = None

        step_re = re.compile(r"^\s*(?:STEP\s*[:#]?\s*(\d+)|\[Step\s*(\d+)\]|Step\s+(\d+))\b", re.IGNORECASE)
        run_header_re = re.compile(r"^\s*\[(?:BATCH|RUN)\].*?seed\s*=\s*(\d+)\b(?:.*?out\s*=\s*([^|\n]+))?", re.IGNORECASE)

        raw_lines = text.splitlines(True)  # keepends
        for raw_line in raw_lines:
            line = strip_ansi(raw_line)
            s = line.strip()

            if not s:
                self.log.insert("end", line, "meta")
                continue

            # Run/seed header detection for waypoint grouping
            try:
                hm = run_header_re.match(s)
            except Exception:
                hm = None
            if hm:
                seed_txt = hm.group(1)
                out_txt = (hm.group(2) or "").strip()
                seed_label = f"Seed {seed_txt}" if not out_txt else f"Seed {seed_txt} ({out_txt})"
                seed_key = str(seed_txt).strip()
                if seed_key != getattr(self, "_current_waypoint_seed_key", None):
                    anchor_idx = self.log.index("end-1c")
                    self._append_seed_waypoint_header(seed_label, anchor_idx)
                    self._current_waypoint_seed_key = seed_key
                    self._current_run_seen_waypoints = set()
                    self._current_log_step_num = None

            try:
                sm = step_re.match(s)
            except Exception:
                sm = None
            if sm:
                step_num = next((g for g in sm.groups() if g), None)
                if step_num is not None:
                    self._current_log_step_num = str(step_num)
                    self._pending_pre_belief = None
                    self._pending_post_belief = None
                    self._pending_change_emitted = False
                if step_num is not None and step_num not in self._current_run_seen_waypoints:
                    anchor_idx = self.log.index("end-1c")
                    self._append_step_waypoint(f"Step {step_num}", anchor_idx)
                    try:
                        self._sync_waypoints_from_live_data()
                    except Exception:
                        pass
                    self._current_run_seen_waypoints.add(step_num)

            try:
                dest_rating = self._extract_rating_change_destination(s)
                if dest_rating is not None:
                    self._mark_step_waypoint_changed(getattr(self, "_current_log_step_num", None), dest_rating)
            except Exception:
                pass

            # Live-tab auto attach (printed by opinion_dynamics_v5_network.py)
            try:
                if "Opinion-change CSV:" in s:
                    self._live_maybe_attach_from_log_line(s)
            except Exception:
                pass

            tag = "agent"
            low = s.lower()

            # Explicit prompt delimiters (optional; if your sim prints these)
            if "PROMPT_START" in s or "----- PROMPT START" in s:
                self._in_prompt_block = True
                tag = "prompt"
            elif "PROMPT_END" in s or "----- PROMPT END" in s:
                tag = "prompt"
                self._in_prompt_block = False
            # Heuristic prompt block: many LangChain runs print "Prompt after formatting:" then dump the prompt
            elif low.startswith("prompt after formatting:"):
                self._in_prompt_block = True
                tag = "prompt"
            # End-of-prompt / chain markers
            elif low.startswith("> finished chain") or low.startswith("finished chain."):
                tag = "meta"
                self._in_prompt_block = False
            elif "entering new conversationchain" in low:
                tag = "meta"
                self._in_prompt_block = False
            elif self._in_prompt_block:
                tag = "prompt"
            else:
                # Errors / tracebacks
                if low.startswith("traceback (most recent call last)") or low.startswith("error:") or "unicodeencodeerror" in low or "keyerror" in low or "valueerror" in low or "exception" in low:
                    tag = "error"
                # Launcher/meta lines
                elif low.startswith("[cmd]") or low.startswith("[done]") or low.startswith("running with seed") or low.startswith("opinions generated") or low.startswith("[step"):
                    tag = "meta"
                # System/prompt-ish lines
                elif low.startswith("system:") or "system prompt" in low or "system_message" in low:
                    tag = "system"
                # Additional prompt-ish markers (outside prompt blocks)
                elif low.startswith("do the following:") or low.startswith("use exactly this format:") or low.startswith("do not add"):
                    tag = "prompt"
                # Agent output markers (as they appear in run logs)
                elif low.startswith("tweet from") or low.startswith("tweet:") or low.startswith("explanation:") or low.startswith("final_rating:"):
                    tag = "agent"
                elif low.startswith("persona passed") or low.startswith("initializing agent") or low.startswith("agent_id=") or low.startswith("agent_name=") or low.startswith("step "):
                    tag = "meta"
                elif low.startswith("agent:") or low.startswith("assistant:") or low.startswith("model:") or low.startswith("reply:"):
                    tag = "agent"

            # Insert a visual separator whenever the classification changes materially
            if self._last_log_tag is None:
                self._last_log_tag = tag
            elif tag != self._last_log_tag:
                self.log.insert("end", "\n" + ("─" * 80) + "\n", "sep")
                self._last_log_tag = tag

            self.log.insert("end", line, tag)

        self.log.see("end")

        # Restore read-only state
        try:
            self.log.configure(state="disabled")
        except Exception:
            pass


    def clear_log(self):

        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            self.log.configure(state="normal")
        except Exception:
            pass

        self.log.delete("1.0", "end")
        try:
            if hasattr(self, "waypoints_list"):
                self.waypoints_list.delete(0, "end")
            if hasattr(self, "_waypoint_entry_map"):
                self._waypoint_entry_map = {}
            if hasattr(self, "_current_run_seen_waypoints"):
                self._current_run_seen_waypoints = set()
            if hasattr(self, "_current_waypoint_seed_key"):
                self._current_waypoint_seed_key = None
            if hasattr(self, "_step_waypoint_list_idx"):
                self._step_waypoint_list_idx = {}
            if hasattr(self, "_step_waypoint_changed"):
                self._step_waypoint_changed = {}
            if hasattr(self, "_step_waypoint_label"):
                self._step_waypoint_label = {}
            if hasattr(self, "_current_log_step_num"):
                self._current_log_step_num = None
            if hasattr(self, "_step_waypoint_markers"):
                self._step_waypoint_markers = {}
            if hasattr(self, "_pending_pre_belief"):
                self._pending_pre_belief = None
            if hasattr(self, "_pending_post_belief"):
                self._pending_post_belief = None
        except Exception:
            pass

    def _update_network_controls_state(self):
        """
        Enable only the parameters relevant to the selected network type.
        """
        try:
            net_type = (self.network_type_var.get() or "").strip().lower()
            is_none = net_type in {"none", "no_network", "no network"}
            is_ws = net_type == "ws"
            is_er = net_type == "er"
            is_ba = net_type == "ba"

            general_state = "disabled" if is_none else "normal"
            ws_state = "normal" if is_ws else "disabled"
            er_state = "normal" if is_er else "disabled"
            ba_state = "normal" if is_ba else "disabled"

            try:
                self.network_homophily_check.configure(state=general_state)
            except Exception:
                pass
            try:
                self.k_neighbors_entry.configure(state=ws_state)
            except Exception:
                pass
            try:
                self.p_rewire_entry.configure(state=ws_state)
            except Exception:
                pass
            try:
                self.er_p_edge_entry.configure(state=er_state)
            except Exception:
                pass
            try:
                self.ba_m_attach_entry.configure(state=ba_state)
            except Exception:
                pass
            try:
                self.ba_hub_strategy_combo.configure(state="readonly" if is_ba else "disabled")
            except Exception:
                pass
            try:
                hub_strategy = (self.ba_hub_strategy_var.get() or "default").strip().lower()
                self.ba_hub_custom_entry.configure(state=("normal" if (is_ba and hub_strategy == "custom") else "disabled"))
                assignment_active = is_ba and hub_strategy in {"positive", "negative", "neutral", "extreme", "opposite_majority", "custom"}
                self.ba_hub_assignment_mode_combo.configure(state=("readonly" if assignment_active else "disabled"))
                if not assignment_active and hub_strategy in {"default", "random"}:
                    self.ba_hub_assignment_mode_var.set("early_position")
            except Exception:
                pass

            try:
                self.interaction_label.configure(text="Partner selection:" if is_none else "Neighbor selection:")
            except Exception:
                pass

            try:
                sel = (self.interaction_selection_var.get() or "").strip().lower()
                mode_state = "readonly" if sel == "homophily" else "disabled"
                self.interaction_homophily_mode_combo.configure(state=mode_state)
            except Exception:
                pass

            if is_none:
                try:
                    self.network_homophily_var.set(False)
                except Exception:
                    pass
            try:
                self._update_live_network_toggle_visibility()
            except Exception:
                pass
        except Exception:
            pass

    def _update_rag_controls_state(self):
        """Enable RAG widgets when world uses retrieval."""
        try:
            world = (getattr(self, "world_var", tk.StringVar(value="closed")).get() or "closed").strip().lower()
            active = world in {"open_rag", "closed_strict_rag"}
            state_entry = "normal" if active else "disabled"
            state_combo = "readonly" if active else "disabled"
            for widget_name, state in [
                ("rag_backend_combo", state_combo),
                ("rag_corpus_entry", state_entry),
                ("rag_browse_btn", state_entry),
                ("rag_top_k_entry", state_entry),
                ("rag_topic_override_entry", state_entry),
                ("rag_query_mode_step2_combo", state_combo),
                ("rag_query_mode_step3_combo", state_combo),
                ("rag_content_mode_combo", state_combo),
                ("rag_max_chars_entry", state_entry),
            ]:
                try:
                    getattr(self, widget_name).configure(state=state)
                except Exception:
                    pass
            if not active:
                try:
                    self.rag_backend_var.set("off")
                except Exception:
                    pass
        except Exception:
            pass

    def _update_true_open_controls_state(self):
        """Enable live-tool widgets only when world=true_open; notes require multi-tool mode."""
        try:
            world = (getattr(self, "world_var", tk.StringVar(value="closed")).get() or "closed").strip().lower()
            active = world == "true_open"
            tool_mode = (getattr(self, "tool_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if active:
                try:
                    web_backend = (getattr(self, "web_backend_var", tk.StringVar(value="off")).get() or "off").strip().lower()
                except Exception:
                    web_backend = "off"
                auto_key = "_true_open_auto_default_applied"
                try:
                    if not getattr(self, auto_key, False) and web_backend == "off" and tool_mode == "off":
                        self.web_backend_var.set("brave")
                        self.tool_mode_var.set("web_only")
                        self.planner_mode_var.set("heuristic")
                        tool_mode = "web_only"
                        setattr(self, auto_key, True)
                except Exception:
                    pass
            notes_active = active and tool_mode == "multi"
            state_entry = "normal" if active else "disabled"
            state_combo = "readonly" if active else "disabled"
            for widget_name, state in [
                ("web_backend_combo", state_combo),
                ("planner_mode_combo", state_combo),
                ("tool_mode_combo", state_combo),
                ("web_top_k_entry", state_entry),
                ("web_max_chars_entry", state_entry),
                ("step2_web_mode_combo", state_combo),
            ]:
                try:
                    getattr(self, widget_name).configure(state=state)
                except Exception:
                    pass
            note_combo_state = "readonly" if notes_active else "disabled"
            note_entry_state = "normal" if notes_active else "disabled"
            for widget_name, state in [
                ("notes_mode_combo", note_combo_state),
                ("notes_max_items_entry", note_entry_state),
            ]:
                try:
                    getattr(self, widget_name).configure(state=state)
                except Exception:
                    pass
            if not active:
                try:
                    self.web_backend_var.set("off")
                    self.planner_mode_var.set("off")
                    self.tool_mode_var.set("off")
                    self.step2_web_mode_var.set("off")
                    self.notes_mode_var.set("off")
                except Exception:
                    pass
            elif tool_mode != "multi":
                try:
                    self.notes_mode_var.set("off")
                except Exception:
                    pass
        except Exception:
            pass

    def _browse_rag_corpus(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        path = filedialog.askopenfilename(
            title="Select retrieval corpus",
            filetypes=[("Data files", "*.json *.jsonl *.csv *.txt *.md"), ("All files", "*.*")],
        )
        if path:
            self.rag_corpus_path_var.set(path)

    def _update_multi_run_controls_state(self):
        """
        When Multiple runs mode is off, disable the Runs entry.
        """
        try:
            mode = (getattr(self, "multi_run_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if mode == "off":
                self.multi_run_count_entry.configure(state="disabled")
            else:
                self.multi_run_count_entry.configure(state="normal")
        except Exception:
            pass


    def _set_custom_counts_enabled(self):
        """
        Enable the custom-counts textbox only when Opinion dist (v5) == custom_counts.
        """
        try:
            is_custom = (self.opinion_strategy_var.get().strip() == "custom_counts")
            state = "normal" if is_custom else "disabled"
            self.custom_counts_entry.configure(state=state)
        except Exception:
            pass

    def on_open_log(self):
        """Handle a Tkinter/UI event and update dependent controls."""
        self.open_log_viewer()

        path = filedialog.askopenfilename(
            title="Open v5 agent log (.txt)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            self.append_log_threadsafe(f"[ERROR] Could not read log file: {e}\n")
            return

        # Reset log state so separators/tags start fresh
        try:
            try:
                self.log.configure(state="normal")
            except Exception:
                pass
            self.log.delete("1.0", "end")
        except Exception:
            pass
        self._in_prompt_block = False
        if hasattr(self, "_last_log_tag"):
            self._last_log_tag = None

        # Show which file was opened (meta)
        try:
            self._append_log(f"[META] Opened log file: {path}\n")
        except Exception:
            pass

        # Render content with tags
        self._append_log(content)

        # Switch to Logs tab for convenience
        try:
            # last tab is Logs in current design
            self.nb.select(3)
        except Exception:
            pass


    

    # ---- Separate Log Viewer Window (Treeview) ----

    def open_log_viewer(self):
        """
        Dedicated window for browsing and viewing saved v5 log files.
        Changes ONLY the log UI (file list + preview). Everything else remains unchanged.
        """
        # Reuse window if already open
        if hasattr(self, "_log_viewer_win") and self._log_viewer_win is not None:
            try:
                if self._log_viewer_win.winfo_exists():
                    self._log_viewer_win.lift()
                    self._log_viewer_win.focus_force()
                    self._refresh_log_file_list()
                    return
            except Exception:
                pass

        win = tk.Toplevel(self)
        self._log_viewer_win = win
        win.title("Network Log Viewer")
        win.geometry("1480x860")
        win.configure(bg="#0b1220")

        # Local styles for this window (do not disturb the rest of the launcher UI)
        style = ttk.Style(win)
        try:
            style.theme_use("clam")  # best support for custom dark colors on Windows
        except Exception:
            pass

        style.configure(
            "Log.Treeview",
            background="#0f1730",
            fieldbackground="#0f1730",
            foreground="#dbe2f3",
            rowheight=22,
            borderwidth=0,
            relief="flat",
        )
        style.configure(
            "Log.Treeview.Heading",
            background="#121a33",
            foreground="#e6e8ef",
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Log.Treeview",
            background=[("selected", "#1f2a55"), ("!disabled", "#0f1730")],
            foreground=[("selected", "#ffffff"), ("!disabled", "#dbe2f3")],
        )

        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Log folder:", style="Muted.TLabel").pack(side="left")
        self._log_dir_var = tk.StringVar(value=str(self._network_log_dir()))
        ttk.Entry(top, textvariable=self._log_dir_var, width=105).pack(side="left", padx=8)

        ttk.Button(top, text="Refresh", command=self._refresh_log_file_list).pack(side="left", padx=4)
        ttk.Button(top, text="Browse...", command=self._browse_log_dir).pack(side="left", padx=4)

        # Body layout: left (tree) + right (preview)
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        # ---- Left: file tree ----
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Files", style="Muted.TLabel").pack(anchor="w")

        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill="y", expand=True)

        self._log_tree = ttk.Treeview(
            tree_wrap,
            columns=("name", "modified", "size"),
            show="headings",
            style="Log.Treeview",
            selectmode="browse",
            height=32,
        )

        self._log_tree.heading("name", text="Name", command=lambda: self._sort_logs("name"))
        self._log_tree.heading("modified", text="Modified", command=lambda: self._sort_logs("modified"))
        self._log_tree.heading("size", text="Size", command=lambda: self._sort_logs("size"))

        # Make filename readable and horizontal scrolling meaningful.
        self._log_tree.column("name", width=760, anchor="w", stretch=False)
        self._log_tree.column("modified", width=180, anchor="w", stretch=False)
        self._log_tree.column("size", width=90, anchor="e", stretch=False)

        self._log_tree.grid(row=0, column=0, sticky="nsew")

        sb_y = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._log_tree.yview)
        sb_y.grid(row=0, column=1, sticky="ns")

        sb_x = ttk.Scrollbar(tree_wrap, orient="horizontal", command=self._log_tree.xview)
        sb_x.grid(row=1, column=0, sticky="ew")

        self._log_tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        tree_wrap.grid_rowconfigure(0, weight=1)
        tree_wrap.grid_columnconfigure(0, weight=1)

        # Keep selection highlight even after clicking preview
        self._log_tree.bind("<ButtonRelease-1>", lambda e: self._log_tree.focus_set())
        self._log_tree.bind("<Double-Button-1>", self._open_selected_log_file)

        # ---- Right: preview ----
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        ttk.Label(right, text="Preview", style="Muted.TLabel").pack(anchor="w")

        preview_wrap = ttk.Frame(right)
        preview_wrap.pack(fill="both", expand=True)

        self._viewer_text = tk.Text(
            preview_wrap,
            wrap="none",
            font=("Consolas", 10),
            bg="#0f1730",
            fg="#dbe2f3",
            insertbackground="#dbe2f3",
            relief="flat",
            padx=10,
            pady=8,
        )
        self._viewer_text.grid(row=0, column=0, sticky="nsew")

        try:
            self._viewer_text.configure(state="disabled")
            self._viewer_text.bind("<Key>", lambda e: "break")
        except Exception:
            pass

        sb2 = ttk.Scrollbar(preview_wrap, orient="vertical", command=self._viewer_text.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self._viewer_text.configure(yscrollcommand=sb2.set)

        preview_wrap.grid_rowconfigure(0, weight=1)
        preview_wrap.grid_columnconfigure(0, weight=1)

        # Tag colors for saved v5 logs
        # Prompt text = green, LLM text = warm off-white, meta = muted gray.
        self._viewer_text.tag_configure("prompt", foreground="#22c55e")
        self._viewer_text.tag_configure("agent", foreground="#f3e8d2")
        self._viewer_text.tag_configure("meta", foreground="#9aa3c7")
        self._viewer_text.tag_configure("error", foreground="#fb7185")

        self._refresh_log_file_list()

    def _browse_log_dir(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        d = filedialog.askdirectory(title="Select network_log_conversation folder")
        if d:
            self._log_dir_var.set(d)
            self._refresh_log_file_list()

    def _refresh_log_file_list(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "_log_tree"):
            return

        self._log_tree.delete(*self._log_tree.get_children())

        log_dir = Path(self._log_dir_var.get().strip()) if hasattr(self, "_log_dir_var") else self._network_log_dir()
        if not log_dir.exists():
            return

        out_prefix = self.out_var.get().strip()
        pattern = f"{out_prefix}*.txt" if out_prefix else "*.txt"
        files = list(log_dir.glob(pattern))

        # Default sort: newest first (most useful for browsing runs)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for p in files:
            st = p.stat()
            self._log_tree.insert(
                "",
                "end",
                values=(
                    p.name,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    f"{st.st_size // 1024} KB",
                ),
            )

    def _sort_logs(self, col: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "_log_tree"):
            return
        items = [(self._log_tree.set(k, col), k) for k in self._log_tree.get_children("")]
        items.sort()
        for i, (_, k) in enumerate(items):
            self._log_tree.move(k, "", i)

    def _open_selected_log_file(self, event=None):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not hasattr(self, "_log_tree"):
            return
        sel = self._log_tree.selection()
        if not sel:
            return

        name = self._log_tree.item(sel[0])["values"][0]
        path = Path(self._log_dir_var.get().strip()) / name

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            if hasattr(self, "_viewer_text"):
                try:
                    self._viewer_text.configure(state="normal")
                except Exception:
                    pass

                self._viewer_text.delete("1.0", "end")
                self._viewer_text.insert("end", f"[ERROR] {e}\n", "error")

                try:
                    self._viewer_text.configure(state="disabled")
                except Exception:
                    pass
            return

        self._viewer_text.delete("1.0", "end")

        # Terminal-like coloring for saved v5 logs:
        #   - prompt/scaffold/instructions: green
        #   - agent output (FINAL_RATING/TWEET/EXPLANATION): warm
        #   - meta: muted
        in_prompt_block = False

        def is_sep(s: str) -> bool:
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            ss = s.strip()
            return bool(ss) and set(ss) <= set("-_=•*")

        def is_output_marker(low: str) -> bool:
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return (
                low.startswith("final_rating")
                or low.startswith("tweet:")
                or low.startswith("explanation:")
                or low.startswith("tweet from ")
            )

        def is_prompt_header(low: str) -> bool:
            # Headings / scaffold markers that should flip prompt mode ON
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return (
                low.startswith("critical clarification")
                or low.startswith("general behavior rules")
                or low.startswith("mandatory output contract")
                or low.startswith("do the following")
                or low.startswith("use exactly this format")
                or low.startswith("important interpretation rule")
                or low.startswith("claim (base proposition")
                or low.startswith("claim (")
                or low.startswith("claim:")
                or low.startswith("you are discussing")
                or low.startswith("you are not rating")
                or low.startswith("you are rating")
                or low.startswith("prompt after formatting")
                or low.startswith("system:")
                or low.startswith("below is your fixed persona")
                or low.startswith("earlier, you wrote")
                or low.startswith("earlier, you read")
                or low.startswith("you first saw")
                or low.startswith("you previously")
                or low.startswith("you saw a tweet")
                or low.startswith("the new tweet you just read")
                or low.startswith("after reading this tweet")
                or low.startswith("when asked to respond")
                or low.startswith("in particular")
                or low.startswith("any deviation")
            )

        def is_prompt_line(low: str) -> bool:
            # Lines that are typically part of instruction blocks
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return (
                low.startswith("do not ")
                or low.startswith("you must ")
                or low.startswith("final_rating must ")
                or low.startswith("do not add ")
                or low.startswith("do not produce ")
                or low.startswith("treat the claim ")
                or low.startswith("you do not know ")
                or low.startswith("- ")
                or low.startswith("• ")
            )

        for raw_line in content.splitlines(True):
            line = strip_ansi(raw_line)
            stripped = line.strip()
            low = stripped.lower()

            if is_sep(line):
                self._viewer_text.insert("end", line, "sep")
                continue

            # Output markers always win and switch prompt mode off
            if is_output_marker(low):
                in_prompt_block = False
                self._viewer_text.insert("end", line, "agent")
                continue

            # Prompt headers flip prompt mode on
            if is_prompt_header(low):
                in_prompt_block = True
                self._viewer_text.insert("end", line, "prompt")
                continue

            # CLAIM lines are prompt scaffold (even if prompt mode wasn't already on)
            if low.startswith("claim") and not is_output_marker(low):
                in_prompt_block = True
                self._viewer_text.insert("end", line, "prompt")
                continue

            # Inside prompt blocks, keep instructions green (including blank lines)
            if in_prompt_block and (stripped == "" or is_prompt_line(low) or True):
                # We keep prompt block green until an output marker appears or a new section changes it.
                self._viewer_text.insert("end", line, "prompt" if stripped else "prompt")
                continue

            # Default: meta
            self._viewer_text.insert("end", line, "meta")

        self._viewer_text.see("end")

        try:
            self._viewer_text.configure(state="disabled")
        except Exception:
            pass

    def _sort_logs(self, col: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        pass
    
    
    def _update_light_memory_threshold_state(self):
        """Synchronize UI state or derived values after a setting changes."""
        mem = (getattr(self, "memory_var", tk.StringVar(value="enabled")).get() or "enabled").strip().lower()
        enabled = (mem == "light")
        try:
            self.light_memory_threshold_combo.configure(state="readonly" if enabled else "disabled")
        except Exception:
            pass
        try:
            self.light_memory_threshold_label.configure(style=("TLabel" if enabled else "Muted.TLabel"))
        except Exception:
            pass

    def _normalize_allowed_update_mode_ui(self) -> str:
        """Normalize user-provided or file-derived text into the form expected downstream."""
        mode = (getattr(self, "allowed_update_mode_var", tk.StringVar(value="assimilation_only")).get() or "assimilation_only").strip().lower()
        if mode == "free_brounded":
            mode = "free_bounded"
        if mode not in {"assimilation_only", "free_bounded"}:
            mode = "assimilation_only"
        return mode

    def _update_model_dependent_controls_state(self):
        """Gray out controls that make no sense for the selected model. Currently: Think
        mode is disabled and locked to off for models with no thinking capability (e.g.
        llama), the same way Same-side unlock is disabled under free_bounded."""
        try:
            cap = _model_thinking_capability(self.model_var.get())
        except Exception:
            cap = None
        try:
            if cap is not False:
                self.think_mode_combo.configure(state="readonly")
                last = getattr(self, "_last_enabled_think_mode", None)
                if last and self.think_mode_var.get().strip() == "off":
                    self.think_mode_var.set(last)
            else:
                cur = self.think_mode_var.get().strip()
                if cur and cur != "off":
                    self._last_enabled_think_mode = cur
                self.think_mode_var.set("off")
                self.think_mode_combo.configure(state="disabled")
        except Exception:
            pass

    def _update_allowed_update_controls_state(self):
        """Synchronize UI state or derived values after a setting changes."""
        mode = self._normalize_allowed_update_mode_ui()
        free_mode = (mode == "free_bounded")
        try:
            if getattr(self, "allowed_update_mode_var", None) is not None and self.allowed_update_mode_var.get() != mode:
                self.allowed_update_mode_var.set(mode)
        except Exception:
            pass

        try:
            current = self.same_side_edge_unlock_hits_var.get().strip()
        except Exception:
            current = ""

        if free_mode:
            if current and current != "0":
                self._last_enabled_same_side_edge_unlock_hits = current
            try:
                self.same_side_edge_unlock_hits_var.set("0")
            except Exception:
                pass
            try:
                self.same_side_edge_unlock_hits_combo.configure(state="disabled")
            except Exception:
                pass
            try:
                self.same_side_edge_unlock_hits_help_label.configure(
                    text="(forced off when allowed_update_mode = free_bounded)",
                    style="Muted.TLabel",
                )
            except Exception:
                pass
        else:
            try:
                self.same_side_edge_unlock_hits_combo.configure(state="readonly")
            except Exception:
                pass
            try:
                if current == "0" and getattr(self, "_last_enabled_same_side_edge_unlock_hits", "") not in {"", "0"}:
                    self.same_side_edge_unlock_hits_var.set(str(self._last_enabled_same_side_edge_unlock_hits))
            except Exception:
                pass
            try:
                self.same_side_edge_unlock_hits_help_label.configure(
                    text="(0 = off; repeated same-side mild hits unlock ±2 from ±1)",
                    style="Muted.TLabel",
                )
            except Exception:
                pass

    def _collect_settings(self) -> dict:
        """Collect related UI values or data rows into a structured object."""
        return {
            "script": self.script_var.get().strip(),
            "seed": self.seed_var.get().strip(),
            "agents": self.agents_var.get().strip(),
            "steps": self.steps_var.get().strip(),
            "out": self.out_var.get().strip(),
            "prompt_version": self.prompt_version_var.get().strip(),
            "prompt_mode": self.prompt_mode_var.get().strip(),
            "fact_pack_mode": getattr(self, "fact_pack_mode_var", tk.StringVar(value="off")).get().strip(),
            "world": getattr(self, "world_var", tk.StringVar(value="closed")).get().strip(),

            "memory": getattr(self, "memory_var", tk.StringVar(value="enabled")).get().strip(),
            "light_memory_threshold": getattr(self, "light_memory_threshold_var", tk.StringVar(value="3")).get().strip(),
            "trace": getattr(self, "trace_var", tk.StringVar(value="auto")).get().strip(),
            "llm_history": getattr(self, "llm_history_var", tk.StringVar(value="off")).get().strip(),
            "debug_native_thinking_on_fail": getattr(self, "debug_native_thinking_on_fail_var", tk.StringVar(value="off")).get().strip(),

            "multi_run_mode": self._normalize_multi_run_mode(),
            "multi_run_count": getattr(self, "multi_run_count_var", tk.StringVar(value="5")).get().strip(),
            # Global constraints
            "model": self.model_var.get().strip(),
            "think_mode": getattr(self, "think_mode_var", tk.StringVar(value="off")).get().strip(),
            "max_step_change": getattr(self, "max_step_change_var", tk.StringVar(value="1")).get().strip(),
            "allowed_update_mode": self._normalize_allowed_update_mode_ui(),
            "validation_strictness": (getattr(self, "validation_strictness_var", tk.StringVar(value="strict")).get() or "strict"),
            "allow_silence": (getattr(self, "allow_silence_var", tk.StringVar(value="off")).get() or "off"),
            "wrong_side_explanation_requery": (getattr(self, "wrong_side_requery_var", tk.StringVar(value="off")).get() or "off"),
            "deterministic": (getattr(self, "deterministic_var", tk.StringVar(value="off")).get() or "off"),
            "structured_output": (getattr(self, "structured_output_var", tk.StringVar(value="off")).get() or "off"),
            "solo_check": (getattr(self, "solo_check_var", tk.StringVar(value="off")).get() or "off"),
            "same_side_edge_unlock_hits": "0" if self._normalize_allowed_update_mode_ui() == "free_bounded" else getattr(self, "same_side_edge_unlock_hits_var", tk.StringVar(value="2")).get().strip(),
            "same_rating_step3_mode": getattr(self, "same_rating_step3_mode_var", tk.StringVar(value="skip_tweet_local")).get().strip(),
            # ADR-006 Component 3: p_reach pluggable policy pattern (feeds P21, P30).
            "p_reach_policy": (getattr(self, "p_reach_policy_var", tk.StringVar(value="uniform")).get() or "uniform").strip(),
            "p_reach_uniform_value": getattr(self, "p_reach_uniform_value_var", tk.StringVar(value="1.0")).get().strip(),
            "p_reach_homophily_k": getattr(self, "p_reach_homophily_k_var", tk.StringVar(value="2.0")).get().strip(),
            "p_reach_shadowban_fraction": getattr(self, "p_reach_shadowban_fraction_var", tk.StringVar(value="0.1")).get().strip(),
            "p_reach_shadowban_value": getattr(self, "p_reach_shadowban_value_var", tk.StringVar(value="0.1")).get().strip(),
            "p_reach_enforcement": (getattr(self, "p_reach_enforcement_var", tk.StringVar(value="filter")).get() or "filter").strip(),
            "include_bian_scores": (getattr(self, "include_bian_scores_var", tk.StringVar(value="off")).get() or "off"),

            # Global LLM defaults
            "temperature": self.temp_var.get().strip(),
            "top_p": self.top_p_var.get().strip(),
            "top_k": self.top_k_var.get().strip(),
            "repeat_penalty": self.repeat_penalty_var.get().strip(),
            "repeat_last_n": self.repeat_last_n_var.get().strip(),
            "max_tokens": self.max_tokens_var.get().strip(),
            "min_p": getattr(self, "min_p_var", tk.StringVar(value="")).get().strip(),
            "presence_penalty": getattr(self, "presence_penalty_var", tk.StringVar(value="")).get().strip(),
            "frequency_penalty": getattr(self, "frequency_penalty_var", tk.StringVar(value="")).get().strip(),

            # Per-step overrides (blank = inherit global)
            "temperature_step2": getattr(self, "temp_step2_var", tk.StringVar(value="")).get().strip(),
            "top_p_step2": getattr(self, "top_p_step2_var", tk.StringVar(value="")).get().strip(),
            "top_k_step2": getattr(self, "top_k_step2_var", tk.StringVar(value="")).get().strip(),
            "repeat_penalty_step2": getattr(self, "repeat_penalty_step2_var", tk.StringVar(value="")).get().strip(),
            "repeat_last_n_step2": getattr(self, "repeat_last_n_step2_var", tk.StringVar(value="")).get().strip(),
            "max_tokens_step2": getattr(self, "max_tokens_step2_var", tk.StringVar(value="")).get().strip(),
            "min_p_step2": getattr(self, "min_p_step2_var", tk.StringVar(value="")).get().strip(),
            "presence_penalty_step2": getattr(self, "presence_penalty_step2_var", tk.StringVar(value="")).get().strip(),
            "frequency_penalty_step2": getattr(self, "frequency_penalty_step2_var", tk.StringVar(value="")).get().strip(),


            "temperature_step3": getattr(self, "temp_step3_var", tk.StringVar(value="")).get().strip(),
            "top_p_step3": getattr(self, "top_p_step3_var", tk.StringVar(value="")).get().strip(),
            "top_k_step3": getattr(self, "top_k_step3_var", tk.StringVar(value="")).get().strip(),
            "repeat_penalty_step3": getattr(self, "repeat_penalty_step3_var", tk.StringVar(value="")).get().strip(),
            "repeat_last_n_step3": getattr(self, "repeat_last_n_step3_var", tk.StringVar(value="")).get().strip(),
            "max_tokens_step3": getattr(self, "max_tokens_step3_var", tk.StringVar(value="")).get().strip(),
            "min_p_step3": getattr(self, "min_p_step3_var", tk.StringVar(value="")).get().strip(),
            "presence_penalty_step3": getattr(self, "presence_penalty_step3_var", tk.StringVar(value="")).get().strip(),
            "frequency_penalty_step3": getattr(self, "frequency_penalty_step3_var", tk.StringVar(value="")).get().strip(),

            # Network
            "network_type": getattr(self, "network_type_var", tk.StringVar(value="ws")).get().strip(),
            "network_homophily": bool(getattr(self, "network_homophily_var", tk.BooleanVar(value=False)).get()),
            "k_neighbors": getattr(self, "k_neighbors_var", tk.StringVar(value="4")).get().strip(),
            "p_rewire": getattr(self, "p_rewire_var", tk.StringVar(value="0.1")).get().strip(),
            "er_p_edge": getattr(self, "er_p_edge_var", tk.StringVar(value="0.15")).get().strip(),
            "ba_m_attach": getattr(self, "ba_m_attach_var", tk.StringVar(value="2")).get().strip(),
            "ba_hub_strategy": getattr(self, "ba_hub_strategy_var", tk.StringVar(value="default")).get().strip(),
            "ba_hub_assignment_mode": getattr(self, "ba_hub_assignment_mode_var", tk.StringVar(value="early_position")).get().strip(),
            "ba_hub_custom": getattr(self, "ba_hub_custom_var", tk.StringVar(value="")).get().strip(),

            # Retrieval / RAG
            "rag_backend": getattr(self, "rag_backend_var", tk.StringVar(value="off")).get().strip(),
            "rag_corpus_path": getattr(self, "rag_corpus_path_var", tk.StringVar(value="")).get().strip(),
            "rag_top_k": getattr(self, "rag_top_k_var", tk.StringVar(value="4")).get().strip(),
            "rag_query_mode": getattr(self, "rag_query_mode_var", tk.StringVar(value="claim")).get().strip(),
            "rag_query_mode_step2": getattr(self, "rag_query_mode_step2_var", tk.StringVar(value="auto")).get().strip(),
            "rag_query_mode_step3": getattr(self, "rag_query_mode_step3_var", tk.StringVar(value="claim_plus_tweet")).get().strip(),
            "rag_content_mode": getattr(self, "rag_content_mode_var", tk.StringVar(value="full")).get().strip(),
            "rag_topic_override": getattr(self, "rag_topic_override_var", tk.StringVar(value="")).get().strip(),
            "rag_max_chars": getattr(self, "rag_max_chars_var", tk.StringVar(value="2200")).get().strip(),
            "web_backend": getattr(self, "web_backend_var", tk.StringVar(value="off")).get().strip(),
            "web_top_k": getattr(self, "web_top_k_var", tk.StringVar(value="3")).get().strip(),
            "web_max_chars": getattr(self, "web_max_chars_var", tk.StringVar(value="1400")).get().strip(),
            "step2_web_mode": getattr(self, "step2_web_mode_var", tk.StringVar(value="heuristic")).get().strip(),
            "planner_mode": getattr(self, "planner_mode_var", tk.StringVar(value="off")).get().strip(),
            "tool_mode": getattr(self, "tool_mode_var", tk.StringVar(value="off")).get().strip(),
            "notes_mode": getattr(self, "notes_mode_var", tk.StringVar(value="off")).get().strip(),
            "notes_max_items": getattr(self, "notes_max_items_var", tk.StringVar(value="3")).get().strip(),

            # Interaction mode persistence
            "interaction_selection": getattr(self, "interaction_selection_var", tk.StringVar(value="homophily")).get().strip(),
            "interaction_homophily_mode": getattr(self, "interaction_homophily_mode_var", tk.StringVar(value="full")).get().strip(),

            "use_personas": bool(self.use_personas_var.get()),
            "opinion_dist": self.opinion_strategy_var.get().strip(),
            "custom_counts": self.custom_counts_var.get().strip(),
            "names_source": self.names_source_var.get().strip(),

            "age_preset": self.age_preset_var.get().strip(),
            "gender_preset": self.gender_preset_var.get().strip(),
            "educ_preset": self.educ_preset_var.get().strip(),
            "pol_preset": self.pol_preset_var.get().strip(),
            "eth_preset": self.eth_preset_var.get().strip(),
            "occ_preset": self.occ_preset_var.get().strip(),
            "early_life_preset": self.early_life_preset_var.get().strip(),
            "epistemic_profile_preset": getattr(self, "epistemic_profile_preset_var", tk.StringVar(value="none")).get().strip(),

            "use_json_persona_profiles": bool(getattr(self, "use_json_persona_profiles_var", tk.BooleanVar(value=False)).get()),
            "persona_selected_profile": getattr(self, "persona_selected_profile_var", tk.StringVar(value="")).get().strip(),
            "persona_profile_name": getattr(self, "persona_profile_name_var", tk.StringVar(value="Custom profile")).get().strip(),
            "persona_profile_description": getattr(self, "persona_profile_description_var", tk.StringVar(value="")).get().strip(),
            "persona_profile_label_choice": getattr(self, "persona_profile_label_choice_var", tk.StringVar(value="Institutionally trusting pragmatist")).get().strip(),
            "persona_custom_profile_label": getattr(self, "persona_custom_profile_label_var", tk.StringVar(value="")).get().strip(),
            "persona_profile_distribution_mode": getattr(self, "persona_profile_distribution_mode_var", tk.StringVar(value="all")).get().strip(),
            "persona_use_core_causal": bool(getattr(self, "persona_use_core_causal_var", tk.BooleanVar(value=True)).get()),
            "persona_use_topic_causal": bool(getattr(self, "persona_use_topic_causal_var", tk.BooleanVar(value=False)).get()),
            "persona_use_topic_linked": bool(getattr(self, "persona_use_topic_linked_var", tk.BooleanVar(value=False)).get()),
            "persona_use_flavor_only": bool(getattr(self, "persona_use_flavor_only_var", tk.BooleanVar(value=False)).get()),
            "persona_use_expressive_style": bool(getattr(self, "persona_use_expressive_style_var", tk.BooleanVar(value=False)).get()),
            "persona_topic_mode": getattr(self, "persona_topic_mode_var", tk.StringVar(value="off")).get().strip(),
            "persona_topic_causal_profile_choice": getattr(self, "persona_topic_causal_profile_choice_var", tk.StringVar(value="Manual")).get().strip(),
            "persona_topic_profile_choice": getattr(self, "persona_topic_causal_profile_choice_var", tk.StringVar(value="Manual")).get().strip(),
            "persona_flavor_mode": getattr(self, "persona_flavor_mode_var", tk.StringVar(value="off")).get().strip(),
            "persona_show_profile_label": bool(getattr(self, "persona_show_profile_label_var", tk.BooleanVar(value=True)).get()),
            "persona_render_style": getattr(self, "persona_render_style_var", tk.StringVar(value="structured_card")).get().strip(),
            "persona_epistemic_profile_label": getattr(self, "persona_epistemic_profile_label_var", tk.StringVar(value="")).get().strip(),
            "persona_institutional_trust": getattr(self, "persona_institutional_trust_var", tk.StringVar(value="medium")).get().strip(),
            "persona_uncertainty_tolerance": getattr(self, "persona_uncertainty_tolerance_var", tk.StringVar(value="medium")).get().strip(),
            "persona_evidence_style": getattr(self, "persona_evidence_style_var", tk.StringVar(value="mixed")).get().strip(),
            "persona_official_narrative_suspicion": getattr(self, "persona_official_narrative_suspicion_var", tk.StringVar(value="medium")).get().strip(),
            "persona_openness_to_update": getattr(self, "persona_openness_to_update_var", tk.StringVar(value="medium")).get().strip(),
            "persona_locus_of_control": getattr(self, "persona_locus_of_control_var", tk.StringVar(value="mixed")).get().strip(),
            "persona_contrarianism": getattr(self, "persona_contrarianism_var", tk.StringVar(value="medium")).get().strip(),
            "persona_value_orientation": getattr(self, "persona_value_orientation_var", tk.StringVar(value="balanced")).get().strip(),
            "persona_agency_vs_fatalism": getattr(self, "persona_agency_vs_fatalism_var", tk.StringVar(value="balanced")).get().strip(),
            "persona_conflict_style": getattr(self, "persona_conflict_style_var", tk.StringVar(value="balanced")).get().strip(),
            "persona_occupation": getattr(self, "persona_occupation_var", tk.StringVar(value="")).get().strip(),
            "persona_education_level": getattr(self, "persona_education_level_var", tk.StringVar(value="")).get().strip(),
            "persona_training_style": getattr(self, "persona_training_style_var", tk.StringVar(value="")).get().strip(),
            "persona_domain_familiarity": getattr(self, "persona_domain_familiarity_var", tk.StringVar(value="")).get().strip(),
            "persona_topic_interest": getattr(self, "persona_topic_interest_var", tk.StringVar(value="")).get().strip(),
            "persona_prior_exposure": getattr(self, "persona_prior_exposure_var", tk.StringVar(value="")).get().strip(),
            "persona_computational_worldview": getattr(self, "persona_computational_worldview_var", tk.StringVar(value="")).get().strip(),
            "persona_testability_preference": getattr(self, "persona_testability_preference_var", tk.StringVar(value="")).get().strip(),
            "persona_anthropic_reasoning_comfort": getattr(self, "persona_anthropic_reasoning_comfort_var", tk.StringVar(value="")).get().strip(),
            "persona_future_technology_prior": getattr(self, "persona_future_technology_prior_var", tk.StringVar(value="")).get().strip(),
            "persona_consciousness_intuition": getattr(self, "persona_consciousness_intuition_var", tk.StringVar(value="")).get().strip(),
            "persona_metaphysical_speculation_tolerance": getattr(self, "persona_metaphysical_speculation_tolerance_var", tk.StringVar(value="")).get().strip(),
            "persona_custom_topic_notes": getattr(self, "persona_custom_topic_notes_var", tk.StringVar(value="")).get().strip(),
            "persona_age_group": getattr(self, "persona_age_group_var", tk.StringVar(value="")).get().strip(),
            "persona_flavor_gender": getattr(self, "persona_flavor_gender_var", tk.StringVar(value="")).get().strip(),
            "persona_flavor_ethnicity": getattr(self, "persona_flavor_ethnicity_var", tk.StringVar(value="")).get().strip(),
            "persona_lifestyle_notes": getattr(self, "persona_lifestyle_notes_var", tk.StringVar(value="")).get().strip(),
            "persona_tone_hint": getattr(self, "persona_tone_hint_var", tk.StringVar(value="")).get().strip(),
            "persona_topic_manual_override": bool(getattr(self, "persona_topic_manual_override_var", tk.BooleanVar(value=False)).get()),
            "persona_topic_causal_manual_override": bool(getattr(self, "persona_topic_causal_manual_override_var", tk.BooleanVar(value=False)).get()),
            "persona_flavor_manual_override": bool(getattr(self, "persona_flavor_manual_override_var", tk.BooleanVar(value=False)).get()),
        }

    def _persona_config_dir(self) -> Path:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        return CONFIG_DIR

    def _ensure_persona_json_files(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        schema_path, profiles_path = ensure_config_files(self._persona_config_dir())
        self.persona_schema_path_var.set(str(schema_path))
        self.persona_profiles_path_var.set(str(profiles_path))
        return schema_path, profiles_path

    def _load_persona_json_runtime(self):
        """Load persisted data from disk and return a safe in-memory representation."""
        self._ensure_persona_json_files()
        self._persona_schema_payload = load_schema(self._persona_config_dir())
        self._persona_profiles_payload = load_profiles(self._persona_config_dir())
        return self._persona_schema_payload, self._persona_profiles_payload

    def _refresh_profile_choices(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        _schema, profiles = self._load_persona_json_runtime()
        choices = list_profile_choices(profiles)
        try:
            self.persona_profile_combo.configure(values=choices)
        except Exception:
            pass
        cur = (self.persona_selected_profile_var.get() or "").strip()
        if not cur and choices:
            self.persona_selected_profile_var.set(choices[0])
        elif cur and cur not in choices and choices:
            self.persona_selected_profile_var.set(choices[0])

    def _values_with_off(self, values):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        out = []
        for v in ["off", *list(values or [])]:
            if v not in out:
                out.append(v)
        return out

    def _active_topic_root(self) -> str:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            return topic_version_root(self.prompt_version_var.get())
        except Exception:
            return "generic"

    def _active_topic_fields(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            return get_topic_fields(self.prompt_version_var.get())
        except Exception:
            return get_topic_fields("generic")

    def _active_topic_slot_mapping(self) -> dict:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        fields = self._active_topic_fields()
        return {TOPIC_SLOT_KEYS[i]: fields[i] for i in range(min(len(TOPIC_SLOT_KEYS), len(fields)))}

    def _active_topic_specific_state_from_slots(self) -> tuple[dict, dict, dict]:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        fields = self._active_topic_fields()
        values, labels, importance = {}, {}, {}
        for slot_key, meta in zip(TOPIC_SLOT_KEYS, fields):
            var = getattr(self, f"persona_{slot_key}_var", None)
            val = var.get().strip() if var is not None else ""
            key = str(meta.get("key", slot_key))
            values[key] = val
            labels[key] = str(meta.get("label", key.replace("_", " ").title()))
            importance[key] = str(meta.get("importance", "medium"))
        return values, labels, importance

    def _refresh_topic_profile_choices(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        root = self._active_topic_root()
        choices = topic_profile_names(root)
        try:
            self.persona_topic_causal_profile_combo.configure(values=choices)
        except Exception:
            pass
        cur = self.persona_topic_causal_profile_choice_var.get().strip() or TOPIC_PROFILE_MANUAL
        if cur not in choices:
            self.persona_topic_causal_profile_choice_var.set(TOPIC_PROFILE_MANUAL)

    def _apply_topic_causal_profile_choice(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        root = self._active_topic_root()
        choice = self.persona_topic_causal_profile_choice_var.get().strip() or TOPIC_PROFILE_MANUAL
        values = get_topic_profile(root, choice)
        fields = self._active_topic_fields()
        self._persona_auto_apply_in_progress = True
        try:
            if values:
                for slot_key, meta in zip(TOPIC_SLOT_KEYS, fields):
                    key = str(meta.get("key", slot_key))
                    var = getattr(self, f"persona_{slot_key}_var", None)
                    if var is not None:
                        var.set(str(values.get(key, "")))
                self.persona_topic_causal_manual_override_var.set(False)
            else:
                # Manual means the visible topic-causal fields are directly edited by the user.
                self.persona_topic_causal_manual_override_var.set(True)
        finally:
            self._persona_auto_apply_in_progress = False
        self._update_json_persona_preview()

    # Backward-compatible alias used by older traces/settings.
    def _apply_topic_profile_choice(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        return self._apply_topic_causal_profile_choice(*_args)

    def _refresh_topic_persona_fields(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        fields = self._active_topic_fields()
        fieldset = get_topic_fieldset(self.prompt_version_var.get())
        try:
            self.topic_specific_header_label.configure(text=f"Topic-specific causal traits for {fieldset.get('label', 'Generic topic')}:")
        except Exception:
            pass
        label_widgets = [getattr(self, f"topic_trait_label_{i}", None) for i in range(1, 7)]
        combo_widgets = [
            getattr(self, "persona_computational_worldview_combo", None),
            getattr(self, "persona_testability_preference_combo", None),
            getattr(self, "persona_anthropic_reasoning_comfort_combo", None),
            getattr(self, "persona_future_technology_prior_combo", None),
            getattr(self, "persona_consciousness_intuition_combo", None),
            getattr(self, "persona_metaphysical_speculation_tolerance_combo", None),
        ]
        # Show only as many topic-trait slots as the active topic actually defines
        # (topics now have 3-4 traits after the A/B cleanup); hide the unused slots.
        for idx, (lbl, combo) in enumerate(zip(label_widgets, combo_widgets)):
            show = idx < len(fields)
            meta = fields[idx] if show else {}
            if lbl is not None:
                try:
                    if show:
                        lbl.configure(text=f"{meta.get('label', f'Topic trait {idx+1}')} ({meta.get('importance', 'medium')}):")
                        lbl.grid()
                    else:
                        lbl.grid_remove()
                except Exception:
                    pass
            if combo is not None:
                try:
                    if show:
                        combo.configure(values=list(meta.get("values", PERSONA_LEVEL_VALUES)))
                        combo.grid()
                    else:
                        combo.grid_remove()
                except Exception:
                    pass
        self._refresh_topic_profile_choices()
        self._update_json_persona_preview()

    def _local_persona_preset_catalog(self):
        """The 6 standard core-causal archetypes offered by the 'Profile label' picker.
        Core-causal only: expressive/debate-style traits live in their own opt-in layer,
        so a core profile no longer touches them. evidence_style uses schema vocabulary
        (concrete/source/coherence/intuition/mixed)."""
        return {
            "Institutionally trusting pragmatist": {
                "institutional_trust": "high", "uncertainty_tolerance": "medium", "evidence_style": "concrete_first",
                "official_narrative_suspicion": "low", "openness_to_update": "medium",
            },
            "Suspicious anti-institutional skeptic": {
                "institutional_trust": "low", "uncertainty_tolerance": "low", "evidence_style": "coherence_first",
                "official_narrative_suspicion": "high", "openness_to_update": "low",
            },
            "Open-minded skeptical reviser": {
                "institutional_trust": "low", "uncertainty_tolerance": "high", "evidence_style": "coherence_first",
                "official_narrative_suspicion": "high", "openness_to_update": "high",
            },
            "Authority-leaning stabilizer": {
                "institutional_trust": "high", "uncertainty_tolerance": "low", "evidence_style": "source_first",
                "official_narrative_suspicion": "low", "openness_to_update": "low",
            },
            "Uncertainty-tolerant agnostic": {
                "institutional_trust": "medium", "uncertainty_tolerance": "high", "evidence_style": "mixed",
                "official_narrative_suspicion": "medium", "openness_to_update": "medium",
            },
            "Low-trust practical realist": {
                "institutional_trust": "low", "uncertainty_tolerance": "medium", "evidence_style": "concrete_first",
                "official_narrative_suspicion": "medium", "openness_to_update": "medium",
            },
        }

    def _preset_profile_choice_values(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        names = list(self._local_persona_preset_catalog().keys())
        for n in preset_profile_names():
            if n not in names:
                names.append(n)
        if "Custom" not in names:
            names.append("Custom")
        return names

    def _set_widget_state_safe(self, widget, state):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            widget.configure(state=state)
        except Exception:
            pass

    def _update_classic_persona_controls_state(self):
        """Show classic preset controls only when JSON persona-profile generation is disabled."""
        try:
            use_json = bool(getattr(self, "use_json_persona_profiles_var", tk.BooleanVar(value=False)).get())
            if not hasattr(self, "classic_persona_frame"):
                return
            if use_json:
                self.classic_persona_frame.grid_remove()
            else:
                self.classic_persona_frame.grid()
        except Exception:
            pass

    def _apply_core_preset_choice(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        choice = (getattr(self, "persona_profile_label_choice_var", tk.StringVar(value="")).get() or "").strip()
        if not choice:
            return
        if choice == "Custom":
            try:
                self.persona_custom_profile_label_entry.configure(state="normal")
            except Exception:
                pass
            if not self.persona_custom_profile_label_var.get().strip():
                self.persona_custom_profile_label_var.set(self.persona_profile_name_var.get().strip() or "Custom profile")
        else:
            local = dict(self._local_persona_preset_catalog().get(choice, {}))
            if local:
                local.setdefault('epistemic_profile_label', choice)
                local.setdefault('profile_name', choice)
                local.setdefault('profile_description', '')
                self._fill_json_profile_widgets(local, preserve_choice=True)
            else:
                preset = get_preset_profile_by_name(choice)
                if preset:
                    ui_state = profile_to_ui_state(preset)
                    for extra_key in ['value_orientation', 'agency_vs_fatalism', 'conflict_style']:
                        ui_state.setdefault(extra_key, 'off')
                    self._fill_json_profile_widgets(ui_state, preserve_choice=True)
            self.persona_custom_profile_label_var.set(choice)
            try:
                self.persona_custom_profile_label_entry.configure(state="disabled")
            except Exception:
                pass
        # Map saved topic-specific fields back into the six visible topic slots for the active version.
        try:
            topic_specific = dict(ui_state.get('topic_specific_fields', {}) or {})
            if topic_specific:
                for slot_key, meta in self._active_topic_slot_mapping().items():
                    key = str(meta.get('key', slot_key))
                    var = getattr(self, f'persona_{slot_key}_var', None)
                    if var is not None and key in topic_specific:
                        var.set(topic_specific.get(key, ''))
        except Exception:
            pass
        self._update_json_profile_controls_state()
        self._update_json_persona_preview()

    def _update_json_profile_controls_state(self, *_args):
        """Synchronize UI state or derived values after a setting changes."""
        mode = (getattr(self, "persona_profile_distribution_mode_var", tk.StringVar(value="all")).get() or "all").strip().lower()
        locked = mode in {"random", "equal"}
        try:
            self.persona_profile_label_combo.configure(state="disabled" if locked else "readonly")
        except Exception:
            pass
        try:
            if (getattr(self, "persona_profile_label_choice_var", tk.StringVar(value="")).get() or "").strip() == "Custom" and not locked:
                self.persona_custom_profile_label_entry.configure(state="normal")
            else:
                self.persona_custom_profile_label_entry.configure(state="disabled")
        except Exception:
            pass
        widgets = getattr(self, "_json_profile_edit_widgets", [])
        for w in widgets:
            state = "disabled" if locked else getattr(w, "_enabled_state", "normal")
            self._set_widget_state_safe(w, state)
        self._update_classic_persona_controls_state()

    def _fill_json_profile_widgets(self, ui_state: dict, preserve_choice: bool = False):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        label = str(ui_state.get('epistemic_profile_label', '') or '')
        preset_names = set(preset_profile_names())
        if not preserve_choice:
            if label in preset_names:
                self.persona_profile_label_choice_var.set(label)
                self.persona_custom_profile_label_var.set(label)
            else:
                self.persona_profile_label_choice_var.set('Custom')
                self.persona_custom_profile_label_var.set(label)
        mapping = {
            'profile_name': self.persona_profile_name_var,
            'profile_description': self.persona_profile_description_var,
            'institutional_trust': self.persona_institutional_trust_var,
            'uncertainty_tolerance': self.persona_uncertainty_tolerance_var,
            'evidence_style': self.persona_evidence_style_var,
            'official_narrative_suspicion': self.persona_official_narrative_suspicion_var,
            'openness_to_update': self.persona_openness_to_update_var,
            'locus_of_control': self.persona_locus_of_control_var,
            'contrarianism': self.persona_contrarianism_var,
            'value_orientation': self.persona_value_orientation_var,
            'agency_vs_fatalism': self.persona_agency_vs_fatalism_var,
            'conflict_style': self.persona_conflict_style_var,
            'occupation': self.persona_occupation_var,
            'education_level': self.persona_education_level_var,
            'training_style': self.persona_training_style_var,
            'domain_familiarity': self.persona_domain_familiarity_var,
            'topic_interest': self.persona_topic_interest_var,
            'prior_exposure': self.persona_prior_exposure_var,
            'computational_worldview': self.persona_computational_worldview_var,
            'testability_preference': self.persona_testability_preference_var,
            'anthropic_reasoning_comfort': self.persona_anthropic_reasoning_comfort_var,
            'future_technology_prior': self.persona_future_technology_prior_var,
            'consciousness_intuition': self.persona_consciousness_intuition_var,
            'metaphysical_speculation_tolerance': self.persona_metaphysical_speculation_tolerance_var,
            'custom_topic_notes': self.persona_custom_topic_notes_var,
            'age_group': self.persona_age_group_var,
            'flavor_gender': self.persona_flavor_gender_var,
            'flavor_ethnicity': self.persona_flavor_ethnicity_var,
            'lifestyle_notes': self.persona_lifestyle_notes_var,
            'tone_hint': self.persona_tone_hint_var,
            'topic_mode': self.persona_topic_mode_var,
            'topic_causal_profile_choice': self.persona_topic_causal_profile_choice_var,
            'topic_profile_choice': self.persona_topic_causal_profile_choice_var,
            'flavor_mode': self.persona_flavor_mode_var,
            'render_style': self.persona_render_style_var,
            'profile_distribution_mode': self.persona_profile_distribution_mode_var,
        }
        for k, var in mapping.items():
            try:
                if k in ui_state:
                    var.set(ui_state.get(k, ""))
            except Exception:
                pass
        for k, var in {
            'use_core_causal': self.persona_use_core_causal_var,
            'use_topic_causal': self.persona_use_topic_causal_var,
            'use_topic_linked': self.persona_use_topic_linked_var,
            'use_flavor_only': self.persona_use_flavor_only_var,
            'show_profile_label': self.persona_show_profile_label_var,
        }.items():
            try:
                var.set(bool(ui_state.get(k, False if 'use_' in k else True)))
            except Exception:
                pass
        self._update_json_profile_controls_state()
        self._update_json_persona_preview()

    def _apply_selected_profile(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self._load_persona_json_runtime()
        pid = extract_profile_id(self.persona_selected_profile_var.get())
        profile = get_profile_by_id(self._persona_profiles_payload, pid)
        if not profile:
            return
        self._fill_json_profile_widgets(profile_to_ui_state(profile))

    def _collect_json_profile_state(self) -> dict:
        """Collect related UI values or data rows into a structured object."""
        profile_choice = self.persona_profile_label_choice_var.get().strip()
        custom_label = self.persona_custom_profile_label_var.get().strip()
        effective_label = custom_label if profile_choice == 'Custom' else profile_choice
        topic_values, topic_labels, topic_importance = self._active_topic_specific_state_from_slots()
        topic_version = self._active_topic_root()
        topic_fieldset_label = get_topic_label(self.prompt_version_var.get())
        return {
            'profile_id': extract_profile_id(self.persona_selected_profile_var.get() or effective_label),
            'profile_name': self.persona_profile_name_var.get().strip() or effective_label or 'Custom profile',
            'profile_description': self.persona_profile_description_var.get().strip(),
            'profile_source': 'custom' if profile_choice == 'Custom' else 'preset',
            'profile_label_choice': profile_choice,
            'custom_profile_label': custom_label,
            'epistemic_profile_label': effective_label,
            'institutional_trust': ('' if self.persona_institutional_trust_var.get().strip() == 'off' else self.persona_institutional_trust_var.get().strip()),
            'uncertainty_tolerance': ('' if self.persona_uncertainty_tolerance_var.get().strip() == 'off' else self.persona_uncertainty_tolerance_var.get().strip()),
            'evidence_style': ('' if self.persona_evidence_style_var.get().strip() == 'off' else self.persona_evidence_style_var.get().strip()),
            'official_narrative_suspicion': ('' if self.persona_official_narrative_suspicion_var.get().strip() == 'off' else self.persona_official_narrative_suspicion_var.get().strip()),
            'openness_to_update': ('' if self.persona_openness_to_update_var.get().strip() == 'off' else self.persona_openness_to_update_var.get().strip()),
            'locus_of_control': ('' if self.persona_locus_of_control_var.get().strip() == 'off' else self.persona_locus_of_control_var.get().strip()),
            'contrarianism': ('' if self.persona_contrarianism_var.get().strip() == 'off' else self.persona_contrarianism_var.get().strip()),
            'value_orientation': ('' if self.persona_value_orientation_var.get().strip() == 'off' else self.persona_value_orientation_var.get().strip()),
            'agency_vs_fatalism': ('' if self.persona_agency_vs_fatalism_var.get().strip() == 'off' else self.persona_agency_vs_fatalism_var.get().strip()),
            'conflict_style': ('' if self.persona_conflict_style_var.get().strip() == 'off' else self.persona_conflict_style_var.get().strip()),
            'occupation': self.persona_occupation_var.get().strip(),
            'education_level': self.persona_education_level_var.get().strip(),
            'training_style': self.persona_training_style_var.get().strip(),
            'domain_familiarity': self.persona_domain_familiarity_var.get().strip(),
            'topic_interest': self.persona_topic_interest_var.get().strip(),
            'prior_exposure': self.persona_prior_exposure_var.get().strip(),
            'computational_worldview': self.persona_computational_worldview_var.get().strip(),
            'testability_preference': self.persona_testability_preference_var.get().strip(),
            'anthropic_reasoning_comfort': self.persona_anthropic_reasoning_comfort_var.get().strip(),
            'future_technology_prior': self.persona_future_technology_prior_var.get().strip(),
            'consciousness_intuition': self.persona_consciousness_intuition_var.get().strip(),
            'metaphysical_speculation_tolerance': self.persona_metaphysical_speculation_tolerance_var.get().strip(),
            'custom_topic_notes': self.persona_custom_topic_notes_var.get().strip(),
            'topic_version': topic_version,
            'topic_fieldset_label': topic_fieldset_label,
            'topic_causal_profile_choice': self.persona_topic_causal_profile_choice_var.get().strip(),
            'topic_profile_choice': self.persona_topic_causal_profile_choice_var.get().strip(),
            'topic_specific_fields': topic_values,
            'topic_specific_labels': topic_labels,
            'topic_specific_importance': topic_importance,
            'age_group': self.persona_age_group_var.get().strip(),
            'flavor_gender': self.persona_flavor_gender_var.get().strip(),
            'flavor_ethnicity': self.persona_flavor_ethnicity_var.get().strip(),
            'lifestyle_notes': self.persona_lifestyle_notes_var.get().strip(),
            'tone_hint': self.persona_tone_hint_var.get().strip(),
            'use_core_causal': bool(self.persona_use_core_causal_var.get()),
            'use_topic_causal': bool(self.persona_use_topic_causal_var.get()),
            'use_topic_linked': bool(self.persona_use_topic_linked_var.get()),
            'use_flavor_only': bool(self.persona_use_flavor_only_var.get()),
            'topic_mode': self.persona_topic_mode_var.get().strip(),
            'topic_profile_choice': self.persona_topic_profile_choice_var.get().strip(),
            'flavor_mode': self.persona_flavor_mode_var.get().strip(),
            'manual_topic_override': bool(self.persona_topic_manual_override_var.get()) or self.persona_topic_mode_var.get().strip() == 'manual',
            'manual_topic_causal_override': bool(self.persona_topic_causal_manual_override_var.get()),
            'manual_flavor_override': bool(self.persona_flavor_manual_override_var.get()) or self.persona_flavor_mode_var.get().strip() == 'manual',
            'show_profile_label': bool(self.persona_show_profile_label_var.get()),
            'render_style': self.persona_render_style_var.get().strip(),
            'profile_distribution_mode': self.persona_profile_distribution_mode_var.get().strip(),
        }

    def _collect_json_profile_prompt_state(self) -> dict:
        """Collect related UI values or data rows into a structured object."""
        state = dict(self._collect_json_profile_state())
        # Keep topic notes visible/editable in the UI, but do not inject them into prompts.
        state['custom_topic_notes'] = ''
        return state

    def _save_current_json_profile(self):
        """Save a generated table, setting, or figure to disk with project naming conventions."""
        try:
            self._load_persona_json_runtime()
            profile = build_profile_from_ui(self._collect_json_profile_state())
            payload = upsert_profile(self._persona_profiles_payload, profile)
            save_profiles(self._persona_config_dir(), payload)
            self._persona_profiles_payload = payload
            self._refresh_profile_choices()
            choice = f"{profile['profile_meta']['profile_name']} [{profile['profile_meta']['profile_id']}]"
            self.persona_selected_profile_var.set(choice)
            self.status_var.set(f"Saved persona profile: {profile['profile_meta']['profile_id']}")
        except Exception as e:
            messagebox.showerror("Persona profiles", str(e))

    def _update_json_persona_preview(self):
        """Synchronize UI state or derived values after a setting changes."""
        try:
            profile = build_profile_from_ui(self._collect_json_profile_state())
            preview = render_profile_preview(profile)
            try:
                topic_gen = self.persona_topic_generated_text.get('1.0', 'end').strip()
            except Exception:
                topic_gen = ''
            try:
                flavor_gen = self.persona_flavor_generated_text.get('1.0', 'end').strip()
            except Exception:
                flavor_gen = ''
            effective_layers = []
            if bool(self.persona_use_core_causal_var.get()):
                effective_layers.append('core causal')
            if bool(self.persona_use_topic_linked_var.get()):
                effective_layers.append('topic-linked background')
            if bool(self.persona_use_flavor_only_var.get()):
                effective_layers.append('flavor-only background')
            extra = "\n\n=== EFFECTIVE PROMPT USAGE ===\n" + (", ".join(effective_layers) if effective_layers else 'none')
            if topic_gen:
                extra += "\n\n=== LIVE GENERATED TOPIC-LINKED SUGGESTION (topic notes are UI-only and not injected) ===\n" + topic_gen
            if flavor_gen:
                extra += "\n\n=== LIVE GENERATED FLAVOR SUGGESTION ===\n" + flavor_gen
            self.persona_preview_text.configure(state='normal')
            self.persona_preview_text.delete('1.0', 'end')
            self.persona_preview_text.insert('1.0', preview + extra)
            self.persona_preview_text.configure(state='disabled')
        except Exception as e:
            try:
                self.persona_preview_text.configure(state='normal')
                self.persona_preview_text.delete('1.0', 'end')
                self.persona_preview_text.insert('1.0', f'[ERROR] {e}')
                self.persona_preview_text.configure(state='disabled')
            except Exception:
                pass

    def _build_json_persona_section(self, parent):
        """Build a derived object, command, path, or UI component used by the local pipeline."""
        box = ttk.LabelFrame(parent, text="JSON persona profiles", padding=12)
        self._register_solo_irrelevant_frame(box)
        box.pack(fill='x', pady=(10,0))

        self.persona_schema_path_var = tk.StringVar(value=str(PERSONA_SCHEMA_PATH))
        self.persona_profiles_path_var = tk.StringVar(value=str(PERSONA_PROFILES_PATH))
        self.use_json_persona_profiles_var = tk.BooleanVar(value=bool(self._settings.get('use_json_persona_profiles', False)))
        self.persona_selected_profile_var = tk.StringVar(value=self._settings.get('persona_selected_profile', ''))
        self.persona_profile_name_var = tk.StringVar(value=self._settings.get('persona_profile_name', 'Custom profile'))
        self.persona_profile_description_var = tk.StringVar(value=self._settings.get('persona_profile_description', ''))
        self.persona_profile_label_choice_var = tk.StringVar(value=self._settings.get('persona_profile_label_choice', 'Institutionally trusting pragmatist'))
        self.persona_custom_profile_label_var = tk.StringVar(value=self._settings.get('persona_custom_profile_label', ''))
        self.persona_profile_distribution_mode_var = tk.StringVar(value=self._settings.get('persona_profile_distribution_mode', 'all'))

        self.persona_use_core_causal_var = tk.BooleanVar(value=bool(self._settings.get('persona_use_core_causal', True)))
        self.persona_use_topic_causal_var = tk.BooleanVar(value=bool(self._settings.get('persona_use_topic_causal', False)))
        self.persona_use_topic_linked_var = tk.BooleanVar(value=bool(self._settings.get('persona_use_topic_linked', False)))
        self.persona_use_flavor_only_var = tk.BooleanVar(value=bool(self._settings.get('persona_use_flavor_only', False)))
        self.persona_use_expressive_style_var = tk.BooleanVar(value=bool(self._settings.get('persona_use_expressive_style', False)))
        self.persona_topic_mode_var = tk.StringVar(value=self._settings.get('persona_topic_mode', 'off'))
        self.persona_topic_causal_profile_choice_var = tk.StringVar(value=self._settings.get('persona_topic_causal_profile_choice', self._settings.get('persona_topic_profile_choice', TOPIC_PROFILE_MANUAL)))
        self.persona_topic_profile_choice_var = self.persona_topic_causal_profile_choice_var  # compatibility alias
        self.persona_flavor_mode_var = tk.StringVar(value=self._settings.get('persona_flavor_mode', 'off'))
        self.persona_show_profile_label_var = tk.BooleanVar(value=bool(self._settings.get('persona_show_profile_label', True)))
        self.persona_render_style_var = tk.StringVar(value=self._settings.get('persona_render_style', 'structured_card'))

        self.persona_epistemic_profile_label_var = tk.StringVar(value=self._settings.get('persona_epistemic_profile_label', ''))
        self.persona_institutional_trust_var = tk.StringVar(value=self._settings.get('persona_institutional_trust', 'medium'))
        self.persona_uncertainty_tolerance_var = tk.StringVar(value=self._settings.get('persona_uncertainty_tolerance', 'medium'))
        self.persona_evidence_style_var = tk.StringVar(value=self._settings.get('persona_evidence_style', 'mixed'))
        self.persona_official_narrative_suspicion_var = tk.StringVar(value=self._settings.get('persona_official_narrative_suspicion', 'medium'))
        self.persona_openness_to_update_var = tk.StringVar(value=self._settings.get('persona_openness_to_update', 'medium'))
        # ADR-006 Component 1: locus_of_control (Rotter 1966) and contrarianism (Flache 2017)
        # as core_causal traits. Defaults are neutral -> byte-identical baseline.
        self.persona_locus_of_control_var = tk.StringVar(value=self._settings.get('persona_locus_of_control', 'mixed'))
        self.persona_contrarianism_var = tk.StringVar(value=self._settings.get('persona_contrarianism', 'medium'))
        self.persona_value_orientation_var = tk.StringVar(value=self._settings.get('persona_value_orientation', 'balanced'))
        self.persona_agency_vs_fatalism_var = tk.StringVar(value=self._settings.get('persona_agency_vs_fatalism', 'balanced'))
        self.persona_conflict_style_var = tk.StringVar(value=self._settings.get('persona_conflict_style', 'balanced'))

        self.persona_occupation_var = tk.StringVar(value=self._settings.get('persona_occupation', ''))
        self.persona_education_level_var = tk.StringVar(value=self._settings.get('persona_education_level', ''))
        self.persona_training_style_var = tk.StringVar(value=self._settings.get('persona_training_style', ''))
        self.persona_domain_familiarity_var = tk.StringVar(value=self._settings.get('persona_domain_familiarity', ''))
        self.persona_topic_interest_var = tk.StringVar(value=self._settings.get('persona_topic_interest', ''))
        self.persona_prior_exposure_var = tk.StringVar(value=self._settings.get('persona_prior_exposure', ''))
        self.persona_computational_worldview_var = tk.StringVar(value=self._settings.get('persona_computational_worldview', ''))
        self.persona_testability_preference_var = tk.StringVar(value=self._settings.get('persona_testability_preference', ''))
        self.persona_anthropic_reasoning_comfort_var = tk.StringVar(value=self._settings.get('persona_anthropic_reasoning_comfort', ''))
        self.persona_future_technology_prior_var = tk.StringVar(value=self._settings.get('persona_future_technology_prior', ''))
        self.persona_consciousness_intuition_var = tk.StringVar(value=self._settings.get('persona_consciousness_intuition', ''))
        self.persona_metaphysical_speculation_tolerance_var = tk.StringVar(value=self._settings.get('persona_metaphysical_speculation_tolerance', ''))
        self.persona_custom_topic_notes_var = tk.StringVar(value=self._settings.get('persona_custom_topic_notes', ''))

        self.persona_age_group_var = tk.StringVar(value=self._settings.get('persona_age_group', ''))
        self.persona_flavor_gender_var = tk.StringVar(value=self._settings.get('persona_flavor_gender', ''))
        self.persona_flavor_ethnicity_var = tk.StringVar(value=self._settings.get('persona_flavor_ethnicity', ''))
        self.persona_lifestyle_notes_var = tk.StringVar(value=self._settings.get('persona_lifestyle_notes', ''))
        self.persona_tone_hint_var = tk.StringVar(value=self._settings.get('persona_tone_hint', ''))
        self.persona_topic_manual_override_var = tk.BooleanVar(value=bool(self._settings.get('persona_topic_manual_override', False)))
        self.persona_topic_causal_manual_override_var = tk.BooleanVar(value=bool(self._settings.get('persona_topic_causal_manual_override', False)))
        self.persona_flavor_manual_override_var = tk.BooleanVar(value=bool(self._settings.get('persona_flavor_manual_override', False)))
        self._persona_auto_apply_in_progress = False

        ttk.Checkbutton(box, text='Use JSON persona profiles for CSV generation', variable=self.use_json_persona_profiles_var, command=self._update_classic_persona_controls_state).grid(row=0, column=0, columnspan=4, sticky='w')
        ttk.Label(box, text='Schema path:').grid(row=1, column=0, sticky='w', pady=4)
        ttk.Entry(box, textvariable=self.persona_schema_path_var, width=72, state='readonly').grid(row=1, column=1, columnspan=3, sticky='we', padx=6)
        ttk.Label(box, text='Profiles path:').grid(row=2, column=0, sticky='w', pady=4)
        ttk.Entry(box, textvariable=self.persona_profiles_path_var, width=72, state='readonly').grid(row=2, column=1, columnspan=3, sticky='we', padx=6)

        ttk.Label(box, text='Saved JSON profile:').grid(row=3, column=0, sticky='w', pady=4)
        self.persona_profile_combo = ttk.Combobox(box, textvariable=self.persona_selected_profile_var, values=[], width=42, state='readonly')
        self.persona_profile_combo.grid(row=3, column=1, sticky='w', padx=6)
        ttk.Button(box, text='Reload JSON', command=self._refresh_profile_choices).grid(row=3, column=2, sticky='w', padx=4)
        ttk.Button(box, text='Apply saved profile', command=self._apply_selected_profile).grid(row=3, column=3, sticky='w', padx=4)

        ttk.Label(box, text='Profile name:').grid(row=4, column=0, sticky='w', pady=4)
        ttk.Entry(box, textvariable=self.persona_profile_name_var, width=28).grid(row=4, column=1, sticky='w', padx=6)
        ttk.Label(box, text='Description:').grid(row=4, column=2, sticky='w', pady=4)
        ttk.Entry(box, textvariable=self.persona_profile_description_var, width=28).grid(row=4, column=3, sticky='w', padx=6)

        core = ttk.LabelFrame(box, text='Core causal variables', padding=8)
        core.grid(row=5, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Checkbutton(core, text='Include core causal variables', variable=self.persona_use_core_causal_var).grid(row=0, column=0, sticky='w')
        ttk.Label(core, text='Mode:').grid(row=0, column=1, sticky='e')
        self.persona_profile_distribution_mode_combo = ttk.Combobox(core, textvariable=self.persona_profile_distribution_mode_var, values=DEFAULT_SCHEMA['allowed_values']['profile_distribution_mode'], width=10, state='readonly')
        self.persona_profile_distribution_mode_combo.grid(row=0, column=2, sticky='w', padx=6)
        ttk.Label(core, text='Mode: all = EVERY agent gets this profile exactly as configured here.  equal = even split of the preset library across agents (remainder round-robin).  random = each agent draws a preset uniformly.  more / most = this profile for 55% / 80% of agents, the others share the rest.  Note: random/equal draw from the PRESET library, so your direct edits to the fields below do not enter the pool - editing is disabled there.', style='Muted.TLabel', wraplength=760, justify='left').grid(row=0, column=3, sticky='w')

        ttk.Label(core, text='Profile label:').grid(row=1, column=0, sticky='w', pady=4)
        self.persona_profile_label_combo = ttk.Combobox(core, textvariable=self.persona_profile_label_choice_var, values=self._preset_profile_choice_values(), width=30, state='readonly')
        self.persona_profile_label_combo.grid(row=1, column=1, sticky='w', padx=6)
        ttk.Label(core, text='Custom name:').grid(row=1, column=2, sticky='w', pady=4)
        self.persona_custom_profile_label_entry = ttk.Entry(core, textvariable=self.persona_custom_profile_label_var, width=26)
        self.persona_custom_profile_label_entry.grid(row=1, column=3, sticky='w', padx=6)
        ttk.Label(core, text='Select one of the 6 standard profiles or Custom.', style='Muted.TLabel').grid(row=2, column=0, columnspan=4, sticky='w')

        ttk.Label(core, text='Institutional trust:').grid(row=3, column=0, sticky='w', pady=4)
        self.persona_institutional_trust_combo = ttk.Combobox(core, textvariable=self.persona_institutional_trust_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['institutional_trust']), width=14, state='readonly')
        self.persona_institutional_trust_combo.grid(row=3, column=1, sticky='w', padx=6)
        ttk.Label(core, text='How much official institutions are trusted.', style='Muted.TLabel').grid(row=3, column=2, columnspan=2, sticky='w')

        ttk.Label(core, text='Uncertainty tolerance:').grid(row=4, column=0, sticky='w', pady=4)
        self.persona_uncertainty_tolerance_combo = ttk.Combobox(core, textvariable=self.persona_uncertainty_tolerance_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['uncertainty_tolerance']), width=14, state='readonly')
        self.persona_uncertainty_tolerance_combo.grid(row=4, column=1, sticky='w', padx=6)
        ttk.Label(core, text='How comfortable the agent is with staying unsure.', style='Muted.TLabel').grid(row=4, column=2, columnspan=2, sticky='w')

        ttk.Label(core, text='Evidence style:').grid(row=5, column=0, sticky='w', pady=4)
        self.persona_evidence_style_combo = ttk.Combobox(core, textvariable=self.persona_evidence_style_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['evidence_style']), width=16, state='readonly')
        self.persona_evidence_style_combo.grid(row=5, column=1, sticky='w', padx=6)
        ttk.Label(core, text='Preferred type of argument or support.', style='Muted.TLabel').grid(row=5, column=2, columnspan=2, sticky='w')

        ttk.Label(core, text='Official-narrative suspicion:').grid(row=6, column=0, sticky='w', pady=4)
        self.persona_official_narrative_suspicion_combo = ttk.Combobox(core, textvariable=self.persona_official_narrative_suspicion_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['official_narrative_suspicion']), width=14, state='readonly')
        self.persona_official_narrative_suspicion_combo.grid(row=6, column=1, sticky='w', padx=6)
        ttk.Label(core, text='How suspicious the agent is of official narratives.', style='Muted.TLabel').grid(row=6, column=2, columnspan=2, sticky='w')

        ttk.Label(core, text='Openness to update:').grid(row=7, column=0, sticky='w', pady=4)
        self.persona_openness_to_update_combo = ttk.Combobox(core, textvariable=self.persona_openness_to_update_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['openness_to_update']), width=14, state='readonly')
        self.persona_openness_to_update_combo.grid(row=7, column=1, sticky='w', padx=6)
        ttk.Label(core, text='How easily a strong tweet can move the agent.', style='Muted.TLabel').grid(row=7, column=2, columnspan=2, sticky='w')

        # ADR-006 Component 1: locus of control (Rotter 1966).
        ttk.Label(core, text='Locus of control:').grid(row=8, column=0, sticky='w', pady=4)
        self.persona_locus_of_control_combo = ttk.Combobox(core, textvariable=self.persona_locus_of_control_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['locus_of_control']), width=14, state='readonly')
        self.persona_locus_of_control_combo.grid(row=8, column=1, sticky='w', padx=6)
        ttk.Label(core, text='Πώς αλλάζει γνώμη ο agent: internal = από επιχειρήματα/στοιχεία, external = από την πλειοψηφία γύρω του, mixed = ενδιάμεσο (default). Rotter 1966.', style='Muted.TLabel').grid(row=8, column=2, columnspan=2, sticky='w')

        # ADR-006 Component 1: contrarianism (Flache 2017, repulsive influence).
        ttk.Label(core, text='Contrarianism:').grid(row=9, column=0, sticky='w', pady=4)
        self.persona_contrarianism_combo = ttk.Combobox(core, textvariable=self.persona_contrarianism_var, values=self._values_with_off(DEFAULT_SCHEMA['allowed_values']['contrarianism']), width=14, state='readonly')
        self.persona_contrarianism_combo.grid(row=9, column=1, sticky='w', padx=6)
        ttk.Label(core, text='Τάση να αποκλίνει από την πληθυσμιακή πλειοψηφία (very_high = συστηματικός αντιλογισμός, very_low = συμμορφωτικός, medium = ουδέτερο default). Flache 2017.', style='Muted.TLabel').grid(row=9, column=2, columnspan=2, sticky='w')

        # --- Expressive / debate-style layer (separate category) --------------------
        # These three are NOT causal belief-update drivers. They only relax the output
        # validators so persona-consistent wording is not flagged (outcome framing /
        # fatalistic framing / combative tone). Kept in their own layer, OFF by default,
        # so they never confound the causal comparison. social_conformity was retired
        # (it produced no validator permission).
        expressive = ttk.LabelFrame(box, text='Expressive / debate style - validator permissions (off by default; does NOT change belief update)', padding=8)
        expressive.grid(row=6, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Checkbutton(expressive, text='Include expressive/debate style', variable=self.persona_use_expressive_style_var).grid(row=0, column=0, columnspan=4, sticky='w')
        ttk.Label(expressive, text='Only relaxes output validators for in-character wording. Leave off unless you deliberately want expressive variety or a persona-adherence probe.', style='Muted.TLabel').grid(row=1, column=0, columnspan=4, sticky='w')

        ttk.Label(expressive, text='Value orientation:').grid(row=2, column=0, sticky='w', pady=4)
        self.persona_value_orientation_combo = ttk.Combobox(expressive, textvariable=self.persona_value_orientation_var, values=self._values_with_off(['balanced', 'procedural', 'outcome_focused']), width=16, state='readonly')
        self.persona_value_orientation_combo.grid(row=2, column=1, sticky='w', padx=6)
        ttk.Label(expressive, text='outcome_focused: validator allows consequence-based framing.', style='Muted.TLabel').grid(row=2, column=2, columnspan=2, sticky='w')

        ttk.Label(expressive, text='Agency vs fatalism:').grid(row=3, column=0, sticky='w', pady=4)
        self.persona_agency_vs_fatalism_combo = ttk.Combobox(expressive, textvariable=self.persona_agency_vs_fatalism_var, values=self._values_with_off(['balanced', 'high_agency', 'fatalistic']), width=14, state='readonly')
        self.persona_agency_vs_fatalism_combo.grid(row=3, column=1, sticky='w', padx=6)
        ttk.Label(expressive, text='fatalistic: validator allows structural / "forces bigger than us" framing.', style='Muted.TLabel').grid(row=3, column=2, columnspan=2, sticky='w')

        ttk.Label(expressive, text='Conflict style:').grid(row=4, column=0, sticky='w', pady=4)
        self.persona_conflict_style_combo = ttk.Combobox(expressive, textvariable=self.persona_conflict_style_var, values=self._values_with_off(['balanced', 'consensus_seeking', 'combative']), width=16, state='readonly')
        self.persona_conflict_style_combo.grid(row=4, column=1, sticky='w', padx=6)
        ttk.Label(expressive, text='combative: validator allows sharper / confrontational tone.', style='Muted.TLabel').grid(row=4, column=2, columnspan=2, sticky='w')

        topic_causal = ttk.LabelFrame(box, text='Topic causal traits (high causal weight; changes with prompt version)', padding=8)
        topic_causal.grid(row=7, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Checkbutton(topic_causal, text='Include topic causal traits', variable=self.persona_use_topic_causal_var).grid(row=0, column=0, sticky='w')
        ttk.Label(topic_causal, text='Topic profile:').grid(row=0, column=1, sticky='e', padx=(12,0))
        self.persona_topic_causal_profile_combo = ttk.Combobox(topic_causal, textvariable=self.persona_topic_causal_profile_choice_var, values=[TOPIC_PROFILE_MANUAL], width=32, state='readonly')
        self.persona_topic_causal_profile_combo.grid(row=0, column=2, sticky='w', padx=6)
        ttk.Checkbutton(topic_causal, text='Manual topic-causal edits', variable=self.persona_topic_causal_manual_override_var).grid(row=0, column=3, sticky='w')
        ttk.Label(topic_causal, text='These are not generated probabilistically. Pick a topic profile, or choose Manual and set the six fields directly.', style='Muted.TLabel').grid(row=1, column=0, columnspan=4, sticky='w')

        self.topic_specific_header_label = ttk.Label(topic_causal, text='Topic-specific causal traits:', style='Muted.TLabel')
        self.topic_specific_header_label.grid(row=2, column=0, columnspan=4, sticky='w', pady=(6,0))
        self.topic_trait_label_1 = ttk.Label(topic_causal, text='Topic trait 1:')
        self.topic_trait_label_1.grid(row=3, column=0, sticky='w', pady=4)
        self.persona_computational_worldview_combo = ttk.Combobox(topic_causal, textvariable=self.persona_computational_worldview_var, values=PERSONA_LEVEL_VALUES, width=12, state='readonly')
        self.persona_computational_worldview_combo.grid(row=3, column=1, sticky='w', padx=6)
        self.topic_trait_label_2 = ttk.Label(topic_causal, text='Topic trait 2:')
        self.topic_trait_label_2.grid(row=3, column=2, sticky='w', pady=4)
        self.persona_testability_preference_combo = ttk.Combobox(topic_causal, textvariable=self.persona_testability_preference_var, values=PERSONA_LEVEL_VALUES, width=12, state='readonly')
        self.persona_testability_preference_combo.grid(row=3, column=3, sticky='w', padx=6)

        self.topic_trait_label_3 = ttk.Label(topic_causal, text='Topic trait 3:')
        self.topic_trait_label_3.grid(row=4, column=0, sticky='w', pady=4)
        self.persona_anthropic_reasoning_comfort_combo = ttk.Combobox(topic_causal, textvariable=self.persona_anthropic_reasoning_comfort_var, values=PERSONA_LEVEL_VALUES, width=12, state='readonly')
        self.persona_anthropic_reasoning_comfort_combo.grid(row=4, column=1, sticky='w', padx=6)
        self.topic_trait_label_4 = ttk.Label(topic_causal, text='Topic trait 4:')
        self.topic_trait_label_4.grid(row=4, column=2, sticky='w', pady=4)
        self.persona_future_technology_prior_combo = ttk.Combobox(topic_causal, textvariable=self.persona_future_technology_prior_var, values=PERSONA_LEVEL_VALUES, width=12, state='readonly')
        self.persona_future_technology_prior_combo.grid(row=4, column=3, sticky='w', padx=6)

        self.topic_trait_label_5 = ttk.Label(topic_causal, text='Topic trait 5:')
        self.topic_trait_label_5.grid(row=5, column=0, sticky='w', pady=4)
        self.persona_consciousness_intuition_combo = ttk.Combobox(topic_causal, textvariable=self.persona_consciousness_intuition_var, values=PERSONA_CONSCIOUSNESS_VALUES, width=14, state='readonly')
        self.persona_consciousness_intuition_combo.grid(row=5, column=1, sticky='w', padx=6)
        self.topic_trait_label_6 = ttk.Label(topic_causal, text='Topic trait 6:')
        self.topic_trait_label_6.grid(row=5, column=2, sticky='w', pady=4)
        self.persona_metaphysical_speculation_tolerance_combo = ttk.Combobox(topic_causal, textvariable=self.persona_metaphysical_speculation_tolerance_var, values=PERSONA_LEVEL_VALUES, width=12, state='readonly')
        self.persona_metaphysical_speculation_tolerance_combo.grid(row=5, column=3, sticky='w', padx=6)
        ttk.Button(topic_causal, text='Reset topic-causal fields to selected profile', command=self._reset_topic_causal_to_profile).grid(row=6, column=0, sticky='w', pady=(6,0))

        derived = ttk.LabelFrame(box, text='Live inferred view (updates from core variables in real time)', padding=8)
        derived.grid(row=8, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Label(derived, text='These are generated continuously from the core variables. They do not enter prompts unless their layer is enabled below.', style='Muted.TLabel').grid(row=0, column=0, columnspan=4, sticky='w')
        ttk.Label(derived, text='Run behavior: personas are generated before the run starts and then stay fixed for the whole run unless you later build an explicit dynamic-persona condition.', style='Muted.TLabel').grid(row=1, column=0, columnspan=4, sticky='w', pady=(2,0))
        ttk.Label(derived, text='Inferred distributions:').grid(row=2, column=0, sticky='nw', pady=(6,0))
        self.persona_distribution_text = tk.Text(derived, height=5, wrap='word', bg='#0c1328', fg='#f2f4ff', insertbackground='#f2f4ff')
        self.persona_distribution_text.grid(row=2, column=1, columnspan=3, sticky='we', padx=6, pady=(6,0))
        self.persona_distribution_text.configure(state='disabled')

        ttk.Label(derived, text='Generated topic-background suggestion:').grid(row=3, column=0, sticky='nw', pady=(6,0))
        self.persona_topic_generated_text = tk.Text(derived, height=5, wrap='word', bg='#0c1328', fg='#f2f4ff', insertbackground='#f2f4ff')
        self.persona_topic_generated_text.grid(row=3, column=1, columnspan=3, sticky='we', padx=6, pady=(6,0))
        self.persona_topic_generated_text.configure(state='disabled')

        ttk.Label(derived, text='Generated flavor suggestion:').grid(row=4, column=0, sticky='nw', pady=(6,0))
        self.persona_flavor_generated_text = tk.Text(derived, height=5, wrap='word', bg='#0c1328', fg='#f2f4ff', insertbackground='#f2f4ff')
        self.persona_flavor_generated_text.grid(row=4, column=1, columnspan=3, sticky='we', padx=6, pady=(6,0))
        self.persona_flavor_generated_text.configure(state='disabled')

        ttk.Button(derived, text='Regenerate suggestions', command=self._apply_live_persona_generation).grid(row=5, column=0, sticky='w', pady=(8,0))
        ttk.Label(derived, text='Topic-background and flavor suggestions are probabilistic. Topic-causal traits use the selected topic profile or direct manual values.', style='Muted.TLabel').grid(row=5, column=1, columnspan=3, sticky='w', padx=6)

        try:
            derived.grid_columnconfigure(1, weight=1)
            derived.grid_columnconfigure(2, weight=1)
            derived.grid_columnconfigure(3, weight=1)
        except Exception:
            pass

        topic = ttk.LabelFrame(box, text='Topic-linked background: role/expertise + familiarity (medium causal weight)', padding=8)
        topic.grid(row=9, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Checkbutton(topic, text='Include topic-linked background', variable=self.persona_use_topic_linked_var).grid(row=0, column=0, sticky='w')
        ttk.Label(topic, text='Role/expertise and familiarity are separate from topic-causal traits; they matter only when relevant to the claim.', style='Muted.TLabel').grid(row=0, column=4, sticky='w')
        ttk.Label(topic, text='Mode:').grid(row=0, column=1, sticky='e')
        self.persona_topic_mode_combo = ttk.Combobox(topic, textvariable=self.persona_topic_mode_var, values=DEFAULT_SCHEMA['allowed_values']['topic_mode'], width=10, state='readonly')
        self.persona_topic_mode_combo.grid(row=0, column=2, sticky='w', padx=6)
        ttk.Checkbutton(topic, text='Manual override generated topic-background fields', variable=self.persona_topic_manual_override_var).grid(row=0, column=3, sticky='w')
        self.persona_occupation_entry = ttk.Entry(topic, textvariable=self.persona_occupation_var, width=24)
        ttk.Label(topic, text='Occupation:').grid(row=1, column=0, sticky='w', pady=4)
        self.persona_occupation_entry.grid(row=1, column=1, sticky='w', padx=6)
        ttk.Label(topic, text='Education:').grid(row=2, column=0, sticky='w', pady=4)
        self.persona_education_level_combo = ttk.Combobox(topic, textvariable=self.persona_education_level_var, values=DEFAULT_SCHEMA['allowed_values']['education_level'], width=14, state='readonly')
        self.persona_education_level_combo.grid(row=2, column=1, sticky='w', padx=6)
        ttk.Label(topic, text='Training style:').grid(row=2, column=2, sticky='w', pady=4)
        self.persona_training_style_combo = ttk.Combobox(topic, textvariable=self.persona_training_style_var, values=DEFAULT_SCHEMA['allowed_values']['training_style'], width=14, state='readonly')
        self.persona_training_style_combo.grid(row=2, column=3, sticky='w', padx=6)
        ttk.Label(topic, text='Domain familiarity:').grid(row=3, column=0, sticky='w', pady=4)
        self.persona_domain_familiarity_combo = ttk.Combobox(topic, textvariable=self.persona_domain_familiarity_var, values=DEFAULT_SCHEMA['allowed_values']['domain_familiarity'], width=12, state='readonly')
        self.persona_domain_familiarity_combo.grid(row=3, column=1, sticky='w', padx=6)
        ttk.Label(topic, text='Topic interest:').grid(row=3, column=2, sticky='w', pady=4)
        self.persona_topic_interest_entry = ttk.Entry(topic, textvariable=self.persona_topic_interest_var, width=24)
        self.persona_topic_interest_entry.grid(row=3, column=3, sticky='w', padx=6)
        ttk.Label(topic, text='Prior exposure:').grid(row=4, column=0, sticky='w', pady=4)
        self.persona_prior_exposure_entry = ttk.Entry(topic, textvariable=self.persona_prior_exposure_var, width=62)
        self.persona_prior_exposure_entry.grid(row=4, column=1, columnspan=3, sticky='we', padx=6)
        ttk.Label(topic, text='Topic notes (UI-only):').grid(row=5, column=0, sticky='w', pady=4)
        self.persona_custom_topic_notes_entry = ttk.Entry(topic, textvariable=self.persona_custom_topic_notes_var, width=62)
        self.persona_custom_topic_notes_entry.grid(row=5, column=1, columnspan=3, sticky='we', padx=6)
        ttk.Button(topic, text='Reset topic-background fields to live suggestions', command=self._reset_topic_to_generated).grid(row=6, column=0, sticky='w', pady=(6,0))
        ttk.Label(topic, text='Mode: off = layer not rendered; auto = fields filled from the live suggestions; manual (+ override tick) = your values win.  Occupation / Education / Training style: who the agent is professionally - matters only when relevant to the claim (an engineer weighs technical points more).  Domain familiarity: none..high, how well they know THIS topic.  Topic interest: how much they care.  Prior exposure: what they have already heard about the claim.  Topic notes: free text, never auto-filled by the live suggestions. CAREFUL: if filled while this layer is ON, it IS rendered into the persona card as "Topic note: ..." - keep it empty unless you want the agents to read it.',
                  style='Muted.TLabel', wraplength=1050, justify='left').grid(row=7, column=0, columnspan=4, sticky='w', pady=(6,0))

        flavor = ttk.LabelFrame(box, text='Flavor-only descriptors (low causal weight)', padding=8)
        flavor.grid(row=10, column=0, columnspan=4, sticky='we', pady=(8,0))
        ttk.Checkbutton(flavor, text='Include flavor-only descriptors', variable=self.persona_use_flavor_only_var).grid(row=0, column=0, sticky='w')
        ttk.Label(flavor, text='Mode:').grid(row=0, column=1, sticky='e')
        self.persona_flavor_mode_combo = ttk.Combobox(flavor, textvariable=self.persona_flavor_mode_var, values=DEFAULT_SCHEMA['allowed_values']['flavor_mode'], width=10, state='readonly')
        self.persona_flavor_mode_combo.grid(row=0, column=2, sticky='w', padx=6)
        ttk.Checkbutton(flavor, text='Show profile label', variable=self.persona_show_profile_label_var).grid(row=0, column=3, sticky='w')
        ttk.Checkbutton(flavor, text='Manual override generated flavor fields', variable=self.persona_flavor_manual_override_var).grid(row=0, column=4, sticky='w')
        ttk.Label(flavor, text='Age group:').grid(row=1, column=0, sticky='w', pady=4)
        self.persona_age_group_combo = ttk.Combobox(flavor, textvariable=self.persona_age_group_var, values=DEFAULT_SCHEMA['allowed_values']['age_group'], width=10, state='readonly')
        self.persona_age_group_combo.grid(row=1, column=1, sticky='w', padx=6)
        ttk.Label(flavor, text='Gender:').grid(row=1, column=2, sticky='w', pady=4)
        self.persona_flavor_gender_entry = ttk.Entry(flavor, textvariable=self.persona_flavor_gender_var, width=18)
        self.persona_flavor_gender_entry.grid(row=1, column=3, sticky='w', padx=6)
        ttk.Label(flavor, text='Ethnicity:').grid(row=2, column=0, sticky='w', pady=4)
        self.persona_flavor_ethnicity_entry = ttk.Entry(flavor, textvariable=self.persona_flavor_ethnicity_var, width=18)
        self.persona_flavor_ethnicity_entry.grid(row=2, column=1, sticky='w', padx=6)
        ttk.Label(flavor, text='Tone hint:').grid(row=2, column=2, sticky='w', pady=4)
        self.persona_tone_hint_entry = ttk.Entry(flavor, textvariable=self.persona_tone_hint_var, width=24)
        self.persona_tone_hint_entry.grid(row=2, column=3, sticky='w', padx=6)
        ttk.Label(flavor, text='Lifestyle notes:').grid(row=3, column=0, sticky='w', pady=4)
        self.persona_lifestyle_notes_entry = ttk.Entry(flavor, textvariable=self.persona_lifestyle_notes_var, width=62)
        self.persona_lifestyle_notes_entry.grid(row=3, column=1, columnspan=3, sticky='we', padx=6)
        ttk.Button(flavor, text='Reset flavor fields to live suggestions', command=self._reset_flavor_to_generated).grid(row=4, column=0, sticky='w', pady=(6,0))
        ttk.Label(flavor, text='Lowest-weight layer: shapes WORDING and voice, must not steer belief direction.  Mode: off / auto (from live suggestions) / manual (+ override tick).  Age group / Gender / Ethnicity / Lifestyle notes: demographic colour for the persona card.  Tone hint: explicit voice instruction (e.g. dry, ironic).  Show profile label: whether the archetype name (e.g. suspicious skeptic) is written INSIDE the persona card the agent sees.',
                  style='Muted.TLabel', wraplength=1050, justify='left').grid(row=5, column=0, columnspan=5, sticky='w', pady=(6,0))

        try:
            box.grid_columnconfigure(1, weight=1)
            box.grid_columnconfigure(3, weight=1)
            topic.grid_columnconfigure(1, weight=1)
            topic.grid_columnconfigure(3, weight=1)
            flavor.grid_columnconfigure(1, weight=1)
            flavor.grid_columnconfigure(3, weight=1)
        except Exception:
            pass

        self._json_profile_edit_widgets = [
            self.persona_institutional_trust_combo,
            self.persona_uncertainty_tolerance_combo,
            self.persona_evidence_style_combo,
            self.persona_official_narrative_suspicion_combo,
            self.persona_openness_to_update_combo,
            self.persona_value_orientation_combo,
            self.persona_agency_vs_fatalism_combo,
            self.persona_conflict_style_combo,
            self.persona_topic_causal_profile_combo,
            self.persona_topic_mode_combo,
            self.persona_occupation_entry,
            self.persona_education_level_combo,
            self.persona_training_style_combo,
            self.persona_domain_familiarity_combo,
            self.persona_topic_interest_entry,
            self.persona_prior_exposure_entry,
            self.persona_computational_worldview_combo,
            self.persona_testability_preference_combo,
            self.persona_anthropic_reasoning_comfort_combo,
            self.persona_future_technology_prior_combo,
            self.persona_consciousness_intuition_combo,
            self.persona_metaphysical_speculation_tolerance_combo,
            self.persona_custom_topic_notes_entry,
            self.persona_flavor_mode_combo,
            self.persona_age_group_combo,
            self.persona_flavor_gender_entry,
            self.persona_flavor_ethnicity_entry,
            self.persona_tone_hint_entry,
            self.persona_lifestyle_notes_entry,
        ]
        for w in self._json_profile_edit_widgets:
            try:
                if isinstance(w, ttk.Combobox):
                    w._enabled_state = 'readonly'
                else:
                    w._enabled_state = 'normal'
            except Exception:
                pass

        preview_bar = ttk.Frame(box)
        preview_bar.grid(row=12, column=0, columnspan=4, sticky='we', pady=(12,0))
        ttk.Button(preview_bar, text='Refresh preview', command=self._update_json_persona_preview).pack(side='left')
        ttk.Button(preview_bar, text='Save current as custom profile', command=self._save_current_json_profile).pack(side='left', padx=6)
        ttk.Label(preview_bar, text='Render style:').pack(side='left', padx=(12,4))
        ttk.Combobox(preview_bar, textvariable=self.persona_render_style_var, values=DEFAULT_SCHEMA['allowed_values']['render_style'], width=16, state='readonly').pack(side='left')

        self.persona_preview_text = tk.Text(box, height=14, wrap='word', bg='#0c1328', fg='#f2f4ff', insertbackground='#f2f4ff')
        self.persona_preview_text.grid(row=13, column=0, columnspan=4, sticky='we', pady=(6,0))
        _preview_scroll = ttk.Scrollbar(box, orient='vertical', command=self.persona_preview_text.yview)
        _preview_scroll.grid(row=13, column=4, sticky='ns', pady=(6,0))
        self.persona_preview_text.configure(state='disabled', yscrollcommand=_preview_scroll.set)

        def _preview_wheel(event):
            """Scroll the preview text itself and stop the event there, so the
            page underneath does not move at the same time."""
            try:
                if getattr(event, 'delta', 0):
                    self.persona_preview_text.yview_scroll(int(-1 * (event.delta / 120)), 'units')
                elif getattr(event, 'num', None) == 4:
                    self.persona_preview_text.yview_scroll(-3, 'units')
                elif getattr(event, 'num', None) == 5:
                    self.persona_preview_text.yview_scroll(3, 'units')
            except Exception:
                pass
            return 'break'

        for _seq in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            try:
                self.persona_preview_text.bind(_seq, _preview_wheel)
            except Exception:
                pass

        try:
            self.persona_profile_combo.bind('<<ComboboxSelected>>', self._apply_selected_profile)
            self.persona_profile_label_combo.bind('<<ComboboxSelected>>', self._apply_core_preset_choice)
            self.persona_profile_distribution_mode_combo.bind('<<ComboboxSelected>>', self._update_json_profile_controls_state)
        except Exception:
            pass

        for _var in [
            self.persona_institutional_trust_var,
            self.persona_uncertainty_tolerance_var,
            self.persona_evidence_style_var,
            self.persona_official_narrative_suspicion_var,
            self.persona_openness_to_update_var,
            self.persona_locus_of_control_var,
            self.persona_contrarianism_var,
            self.persona_value_orientation_var,
            self.persona_agency_vs_fatalism_var,
            self.persona_conflict_style_var,
        ]:
            try:
                _var.trace_add('write', self._apply_live_persona_generation)
            except Exception:
                pass
        for _var in [
            self.persona_occupation_var,
            self.persona_education_level_var,
            self.persona_training_style_var,
            self.persona_domain_familiarity_var,
            self.persona_topic_interest_var,
            self.persona_prior_exposure_var,
            self.persona_custom_topic_notes_var,
        ]:
            try:
                _var.trace_add('write', self._mark_topic_manual_override)
            except Exception:
                pass
        for _var in [
            self.persona_computational_worldview_var,
            self.persona_testability_preference_var,
            self.persona_anthropic_reasoning_comfort_var,
            self.persona_future_technology_prior_var,
            self.persona_consciousness_intuition_var,
            self.persona_metaphysical_speculation_tolerance_var,
        ]:
            try:
                _var.trace_add('write', self._mark_topic_causal_manual_override)
            except Exception:
                pass
        for _var in [
            self.persona_age_group_var,
            self.persona_flavor_gender_var,
            self.persona_flavor_ethnicity_var,
            self.persona_lifestyle_notes_var,
            self.persona_tone_hint_var,
        ]:
            try:
                _var.trace_add('write', self._mark_flavor_manual_override)
            except Exception:
                pass

        try:
            self.prompt_version_var.trace_add('write', self._refresh_topic_persona_fields)
            self.persona_topic_causal_profile_choice_var.trace_add('write', self._apply_topic_causal_profile_choice)
        except Exception:
            pass
        self._refresh_topic_persona_fields()
        self._refresh_profile_choices()
        if self.persona_selected_profile_var.get().strip():
            self._apply_selected_profile()
        else:
            self._apply_core_preset_choice()
        self._update_json_profile_controls_state()
        self._update_classic_persona_controls_state()
        try:
            self.solo_check_var.trace_add('write', self._update_solo_check_state)
        except Exception:
            pass
        self._update_solo_check_state()
        self.after(10, self._apply_live_persona_generation)



    def _safe_get_setting(self, key: str, default=""):
        """Return a sanitized value that is safe for paths, logs, or UI display."""
        try:
            return self._settings.get(key, default)
        except Exception:
            return default

    def _derive_persona_distributions(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        trust = (self.persona_institutional_trust_var.get() or "medium").strip().lower()
        update = (self.persona_openness_to_update_var.get() or "medium").strip().lower()
        suspicion = (self.persona_official_narrative_suspicion_var.get() or "medium").strip().lower()
        uncertainty = (self.persona_uncertainty_tolerance_var.get() or "medium").strip().lower()
        evidence = (self.persona_evidence_style_var.get() or "mixed").strip().lower()
        value_orientation = (getattr(self, "persona_value_orientation_var", tk.StringVar(value="balanced")).get() or "balanced").strip().lower()
        agency = (getattr(self, "persona_agency_vs_fatalism_var", tk.StringVar(value="balanced")).get() or "balanced").strip().lower()
        conflict = (getattr(self, "persona_conflict_style_var", tk.StringVar(value="balanced")).get() or "balanced").strip().lower()

        age = {"18-25": 0.24, "26-35": 0.28, "36-50": 0.28, "51-65": 0.20}
        politics = {"far_left": 0.07, "left": 0.15, "center_left": 0.18, "center": 0.20, "center_right": 0.18, "right": 0.15, "far_right": 0.07}
        # Flavor ethnicity is probabilistic too. Keep the shifts weak: ethnicity is
        # useful for population heterogeneity, but it must not become a direct
        # belief rule. Baseline roughly follows the existing US-like categories
        # used by the launcher while preserving a sizable unspecified bucket.
        ethnicity = {"unspecified": 0.22, "caucasian": 0.38, "hispanic": 0.17, "black": 0.12, "asian": 0.07, "mixed": 0.03, "other": 0.01}
        education = {"high_school": 0.18, "some_college": 0.22, "bachelor": 0.26, "master": 0.22, "phd": 0.12}
        # Wider than the classic launcher list because topic-causal profiles can
        # make specialized role lenses more plausible. These are suggestions for
        # generated persona CSVs only; they do not constrain the simulation.
        occupation = {
            "student": 0.085, "teacher": 0.085, "engineer": 0.085, "nurse": 0.075,
            "journalist": 0.075, "scientist": 0.070, "economist": 0.055,
            "social_worker": 0.065, "manager": 0.080, "retail": 0.085,
            "mathematician": 0.035, "computer_engineer": 0.040, "historian": 0.035,
            "lawyer": 0.040, "doctor": 0.035, "policy_analyst": 0.030,
            "construction_engineer": 0.025, "pilot": 0.020, "military_veteran": 0.025,
            "philosopher": 0.025, "media_worker": 0.020,
        }
        training_style = {"technical": 0.22, "humanities": 0.18, "social_science": 0.18, "mixed": 0.32, "none": 0.10}
        domain_familiarity = {"low": 0.36, "medium": 0.44, "high": 0.20}
        topic_interest = {"low": 0.30, "medium": 0.45, "high": 0.25}
        prior_exposure = {"low": 0.40, "medium": 0.40, "high": 0.20}

        def adjust(weights, key, delta):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            if key in weights:
                weights[key] = max(0.01, weights[key] + delta)

        def adjust_many(weights, pairs):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            for key, delta in pairs:
                adjust(weights, key, delta)

        def active(v):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return v not in {"", "off", None}

        if active(update):
            if update == "low":
                adjust(age, "51-65", 0.08); adjust(age, "36-50", 0.04); adjust(age, "18-25", -0.06); adjust(age, "26-35", -0.04)
            elif update == "high":
                adjust(age, "18-25", 0.06); adjust(age, "26-35", 0.04); adjust(age, "51-65", -0.06); adjust(age, "36-50", -0.04)

        if active(trust):
            if trust == "high":
                adjust(education, "bachelor", 0.03); adjust(education, "master", 0.04); adjust(education, "phd", 0.03); adjust(education, "high_school", -0.05)
                adjust(occupation, "teacher", 0.03); adjust(occupation, "scientist", 0.03); adjust(occupation, "manager", 0.02)
            elif trust == "low":
                adjust(occupation, "journalist", 0.03); adjust(occupation, "retail", 0.02); adjust(occupation, "student", 0.02)
                adjust(education, "high_school", 0.04); adjust(education, "phd", -0.02)

        if active(suspicion):
            if suspicion == "high":
                adjust(politics, "far_left", 0.04); adjust(politics, "far_right", 0.04); adjust(politics, "center", -0.06)
                adjust(occupation, "journalist", 0.03)
            elif suspicion == "low":
                adjust(politics, "center_left", 0.03); adjust(politics, "center", 0.03); adjust(politics, "center_right", 0.03)
                adjust(politics, "far_left", -0.04); adjust(politics, "far_right", -0.04)

        if active(uncertainty):
            if uncertainty == "low":
                adjust(politics, "far_left", 0.02); adjust(politics, "far_right", 0.02); adjust(politics, "center", -0.04)
            elif uncertainty == "high":
                adjust(politics, "center", 0.05); adjust(politics, "center_left", 0.02); adjust(politics, "center_right", 0.02)
                adjust(politics, "far_left", -0.03); adjust(politics, "far_right", -0.03)

        if active(evidence):
            if evidence in {"authority_first", "authority-first"}:
                adjust(occupation, "teacher", 0.03); adjust(occupation, "scientist", 0.04); adjust(education, "master", 0.03)
                adjust_many(ethnicity, [("unspecified", -0.006), ("caucasian", 0.003), ("asian", 0.003)])
            elif evidence in {"suspicion_first", "suspicion-first"}:
                adjust(occupation, "journalist", 0.04); adjust(occupation, "student", 0.02)
                adjust_many(ethnicity, [("unspecified", -0.006), ("mixed", 0.003), ("hispanic", 0.002), ("black", 0.001)])
            elif evidence in {"concrete_first", "concrete-first"}:
                adjust(occupation, "engineer", 0.03); adjust(occupation, "nurse", 0.03)
                adjust_many(ethnicity, [("unspecified", -0.006), ("asian", 0.003), ("hispanic", 0.002), ("mixed", 0.001)])
            elif evidence in {"coherence_first", "coherence-first"}:
                adjust(occupation, "teacher", 0.02); adjust(occupation, "economist", 0.02)
                adjust_many(ethnicity, [("unspecified", -0.005), ("mixed", 0.002), ("asian", 0.002), ("other", 0.001)])

        if active(value_orientation):
            if value_orientation == "procedural":
                adjust(occupation, "teacher", 0.03); adjust(politics, "center_left", 0.02); adjust(politics, "center_right", 0.02)
            elif value_orientation == "outcome_focused":
                adjust(occupation, "manager", 0.03); adjust(occupation, "economist", 0.03)

        if active(agency):
            if agency == "high_agency":
                adjust(occupation, "teacher", 0.02); adjust(occupation, "social_worker", 0.03); adjust(occupation, "student", 0.02)
            elif agency == "fatalistic":
                adjust(occupation, "retail", 0.02); adjust(occupation, "manager", 0.01)
                adjust(politics, "center", -0.02)

        if active(conflict):
            if conflict == "consensus_seeking":
                adjust(politics, "center", 0.04); adjust(politics, "far_left", -0.02); adjust(politics, "far_right", -0.02)
            elif conflict == "combative":
                adjust(politics, "far_left", 0.03); adjust(politics, "far_right", 0.03); adjust(occupation, "journalist", 0.02)

        # Topic-causal traits now shape the generated background/flavor probability
        # layer. They do not get sampled probabilistically themselves; the selected
        # topic profile or manual topic-causal values bias the probabilistic
        # suggestions for role, education, familiarity, prior exposure, age, ethnicity, and tone.
        topic_root = self._active_topic_root() if hasattr(self, '_active_topic_root') else topic_version_root(self.prompt_version_var.get())
        topic_label = get_topic_label(self.prompt_version_var.get())
        topic_profile_choice = ''
        try:
            topic_profile_choice = self.persona_topic_causal_profile_choice_var.get().strip()
        except Exception:
            topic_profile_choice = ''
        use_topic_causal = False
        try:
            use_topic_causal = bool(self.persona_use_topic_causal_var.get())
        except Exception:
            use_topic_causal = False

        topic_causal_values = {}
        topic_causal_labels = {}
        if use_topic_causal:
            try:
                topic_causal_values, topic_causal_labels, _topic_importance = self._active_topic_specific_state_from_slots()
            except Exception:
                topic_causal_values, topic_causal_labels = {}, {}

        def tval(key, default=''):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return str(topic_causal_values.get(key, default) or '').strip().lower().replace('-', '_')

        def is_high(key):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return tval(key) in {'high', 'very_high', 'computational'}

        def is_low(key):
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            return tval(key) in {'low', 'very_low', 'physicalist'}

        # v130 simulation hypothesis: computational/anthropic/testability traits
        # make technical, mathematical, philosophical, and science-heavy backgrounds
        # more plausible without forcing a belief direction.
        if topic_root == 'v130' and topic_causal_values:
            if is_high('computational_worldview'):
                adjust_many(occupation, [('computer_engineer', 0.075), ('engineer', 0.045), ('scientist', 0.035), ('mathematician', 0.030)])
                adjust_many(education, [('bachelor', 0.025), ('master', 0.035), ('phd', 0.025), ('high_school', -0.035)])
                adjust_many(training_style, [('technical', 0.090), ('mixed', 0.020), ('none', -0.040)])
                adjust_many(domain_familiarity, [('high', 0.080), ('medium', 0.030), ('low', -0.060)])
                adjust_many(age, [('18-25', 0.030), ('26-35', 0.030), ('51-65', -0.035)])
                adjust_many(ethnicity, [('asian', 0.020), ('mixed', 0.006), ('unspecified', -0.018)])
            elif is_low('computational_worldview'):
                adjust_many(occupation, [('teacher', 0.025), ('retail', 0.025), ('social_worker', 0.020), ('computer_engineer', -0.035)])
                adjust_many(training_style, [('humanities', 0.030), ('mixed', 0.020), ('technical', -0.050)])
                adjust_many(ethnicity, [('unspecified', 0.010), ('caucasian', 0.004), ('asian', -0.008)])

            if is_high('anthropic_reasoning_comfort'):
                adjust_many(occupation, [('philosopher', 0.060), ('mathematician', 0.040), ('economist', 0.025), ('scientist', 0.020)])
                adjust_many(ethnicity, [('mixed', 0.006), ('asian', 0.006), ('other', 0.002), ('unspecified', -0.010)])
                adjust_many(education, [('master', 0.030), ('phd', 0.035), ('high_school', -0.030)])
                adjust_many(training_style, [('analytical', 0.050) if 'analytical' in training_style else ('mixed', 0.030), ('mixed', 0.025)])
            elif is_low('anthropic_reasoning_comfort'):
                adjust_many(occupation, [('engineer', 0.020), ('manager', 0.020), ('retail', 0.015)])

            if is_high('future_technology_prior'):
                adjust_many(occupation, [('computer_engineer', 0.055), ('scientist', 0.045), ('engineer', 0.030), ('student', 0.020)])
                adjust_many(ethnicity, [('asian', 0.018), ('mixed', 0.004), ('unspecified', -0.014)])
                adjust_many(age, [('18-25', 0.030), ('26-35', 0.035), ('51-65', -0.030)])
                adjust_many(topic_interest, [('high', 0.060), ('low', -0.040)])
                adjust_many(prior_exposure, [('high', 0.045), ('medium', 0.020), ('low', -0.045)])
            elif is_low('future_technology_prior'):
                adjust_many(age, [('51-65', 0.030), ('36-50', 0.020), ('18-25', -0.025)])
                adjust_many(occupation, [('manager', 0.020), ('teacher', 0.020), ('retail', 0.020)])

            if tval('consciousness_intuition') == 'computational':
                adjust_many(occupation, [('computer_engineer', 0.040), ('philosopher', 0.030), ('scientist', 0.020)])
                adjust_many(training_style, [('technical', 0.035), ('mixed', 0.020)])
            elif tval('consciousness_intuition') == 'physicalist':
                adjust_many(occupation, [('doctor', 0.030), ('scientist', 0.030), ('engineer', 0.020)])
                adjust_many(training_style, [('technical', 0.025), ('formal', 0.015) if 'formal' in training_style else ('technical', 0.0)])

        # v119 Twin Towers / 9-11: security-state, motive, source, anomaly, and
        # coordination traits bias toward relevant investigative, legal, historical,
        # engineering, and policy backgrounds.
        elif topic_root == 'v119' and topic_causal_values:
            if is_high('geopolitical_motive_sensitivity'):
                adjust_many(occupation, [('historian', 0.060), ('journalist', 0.040), ('policy_analyst', 0.040), ('social_worker', 0.015)])
                adjust_many(ethnicity, [('black', 0.008), ('hispanic', 0.008), ('mixed', 0.004), ('unspecified', -0.012)])
                adjust_many(training_style, [('humanities', 0.050), ('social_science', 0.050), ('technical', -0.020)])
                adjust_many(topic_interest, [('high', 0.050), ('low', -0.030)])
            elif is_low('geopolitical_motive_sensitivity'):
                adjust_many(occupation, [('engineer', 0.025), ('scientist', 0.020), ('manager', 0.020)])

            if is_high('conspiracy_coordination_prior'):
                adjust_many(occupation, [('journalist', 0.030), ('historian', 0.020), ('policy_analyst', 0.020)])
                adjust_many(prior_exposure, [('high', 0.035), ('low', -0.025)])
            elif is_low('conspiracy_coordination_prior'):
                adjust_many(occupation, [('manager', 0.030), ('lawyer', 0.025), ('scientist', 0.020)])
                adjust_many(education, [('master', 0.020), ('phd', 0.015), ('high_school', -0.015)])

            if is_high('anomaly_sensitivity'):
                adjust_many(occupation, [('construction_engineer', 0.060), ('engineer', 0.035), ('journalist', 0.035), ('scientist', 0.020)])
                adjust_many(ethnicity, [('hispanic', 0.008), ('asian', 0.008), ('mixed', 0.004), ('unspecified', -0.012)])
                adjust_many(training_style, [('technical', 0.045), ('analytical', 0.025) if 'analytical' in training_style else ('mixed', 0.015)])
                adjust_many(domain_familiarity, [('high', 0.055), ('low', -0.035)])
            elif is_low('anomaly_sensitivity'):
                adjust_many(occupation, [('lawyer', 0.020), ('manager', 0.020), ('teacher', 0.020)])

        # v52 Moon landing: engineering/science trust and anomaly/media/motive
        # traits bias toward appropriate STEM, historical, media, or institutional
        # backgrounds. Engineering-first should visibly raise engineering odds.
        elif topic_root == 'v52' and topic_causal_values:
            if is_high('cold_war_motive_sensitivity'):
                adjust_many(occupation, [('historian', 0.060), ('journalist', 0.040), ('policy_analyst', 0.030), ('teacher', 0.020)])
                adjust_many(training_style, [('humanities', 0.055), ('social_science', 0.035), ('technical', -0.020)])
                adjust_many(topic_interest, [('high', 0.050), ('low', -0.035)])
                adjust_many(prior_exposure, [('high', 0.040), ('low', -0.030)])
            elif is_low('cold_war_motive_sensitivity'):
                adjust_many(occupation, [('scientist', 0.025), ('engineer', 0.025), ('manager', 0.015)])

            if is_high('engineering_evidence_weight'):
                adjust_many(occupation, [('engineer', 0.085), ('construction_engineer', 0.040), ('scientist', 0.035), ('mathematician', 0.020)])
                adjust_many(ethnicity, [('asian', 0.018), ('hispanic', 0.006), ('mixed', 0.004), ('unspecified', -0.016)])
                adjust_many(education, [('bachelor', 0.030), ('master', 0.035), ('phd', 0.025), ('high_school', -0.035), ('some_college', -0.020)])
                adjust_many(training_style, [('technical', 0.090), ('mixed', 0.020), ('none', -0.035)])
                adjust_many(domain_familiarity, [('high', 0.070), ('medium', 0.030), ('low', -0.060)])
                adjust_many(topic_interest, [('high', 0.045), ('low', -0.035)])
            elif is_low('engineering_evidence_weight'):
                adjust_many(occupation, [('teacher', 0.025), ('journalist', 0.020), ('retail', 0.020)])
                adjust_many(training_style, [('humanities', 0.030), ('mixed', 0.020), ('technical', -0.045)])

            if is_high('visual_anomaly_sensitivity'):
                adjust_many(occupation, [('journalist', 0.040), ('media_worker', 0.040), ('engineer', 0.020), ('student', 0.020)])
                adjust_many(ethnicity, [('mixed', 0.006), ('hispanic', 0.006), ('black', 0.004), ('unspecified', -0.010)])
                adjust_many(training_style, [('humanities', 0.025), ('mixed', 0.025), ('technical', 0.015)])
                adjust_many(topic_interest, [('high', 0.040), ('low', -0.025)])
            elif is_low('visual_anomaly_sensitivity'):
                adjust_many(occupation, [('engineer', 0.025), ('scientist', 0.020), ('manager', 0.015)])

        def normalize(weights):
            """Normalize user-provided or file-derived text into the form expected downstream."""
            total = sum(max(v, 0.01) for v in weights.values())
            if total <= 0:
                n = len(weights)
                return {k: round(100.0 / n, 1) for k in weights}
            out = {k: round((max(v, 0.01) / total) * 100.0, 1) for k, v in weights.items()}
            # fix rounding drift
            drift = round(100.0 - sum(out.values()), 1)
            if out:
                first = next(iter(out))
                out[first] = round(out[first] + drift, 1)
            return out

        return {
            "age": normalize(age),
            "politics": normalize(politics),
            "ethnicity": normalize(ethnicity),
            "education": normalize(education),
            "occupation": normalize(occupation),
            "training_style": normalize(training_style),
            "domain_familiarity": normalize(domain_familiarity),
            "topic_interest": normalize(topic_interest),
            "prior_exposure": normalize(prior_exposure),
            "topic_causal_snapshot": dict(topic_causal_values),
            "topic_causal_labels": dict(topic_causal_labels),
            "topic_causal_profile_choice": topic_profile_choice,
            "topic_root": topic_root,
            "topic_label": topic_label,
            "use_topic_causal": int(bool(use_topic_causal)),
            "core_snapshot": {
                "trust": trust, "update": update, "suspicion": suspicion, "uncertainty": uncertainty,
                "evidence": evidence, "value_orientation": value_orientation,
                "agency": agency, "conflict": conflict,
            }
        }

    def _top_distribution_labels(self, weights: dict, top_n: int = 3):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        items = sorted((weights or {}).items(), key=lambda kv: (-float(kv[1]), kv[0]))[:top_n]
        return ", ".join(f"{k} {v:.0f}%" for k, v in items) if items else ""

    def _persona_live_rng(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        rng = getattr(self, '_persona_rng', None)
        if rng is None:
            rng = random.Random(time.time_ns())
            self._persona_rng = rng
        return rng

    def _sample_percent_label(self, weights: dict):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        items = [(str(k), max(0.0, float(v))) for k, v in (weights or {}).items()]
        total = sum(w for _, w in items)
        if total <= 0 or not items:
            return ''
        rng = self._persona_live_rng()
        r = rng.random() * total
        acc = 0.0
        for label, weight in items:
            acc += weight
            if r <= acc:
                return label
        return items[-1][0]

    def _sample_weighted_option(self, options):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        items = [(str(k), max(0.0, float(v))) for k, v in (options or [])]
        total = sum(w for _, w in items)
        if total <= 0 or not items:
            return ''
        rng = self._persona_live_rng()
        r = rng.random() * total
        acc = 0.0
        for label, weight in items:
            acc += weight
            if r <= acc:
                return label
        return items[-1][0]

    def _topic_generated_fields(self, d, topic_text):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        snap = d.get('core_snapshot', {})
        evidence = snap.get('evidence', 'mixed')
        trust = snap.get('trust', 'medium')
        update = snap.get('update', 'medium')
        suspicion = snap.get('suspicion', 'medium')
        uncertainty = snap.get('uncertainty', 'medium')
        age_pick = self._sample_percent_label(d.get('age', {})) or (max(d.get('age', {}).items(), key=lambda kv: kv[1])[0] if d.get('age') else '')
        occ_pick = self._sample_percent_label(d.get('occupation', {})) or (max(d.get('occupation', {}).items(), key=lambda kv: kv[1])[0] if d.get('occupation') else '')
        edu_pick = self._sample_percent_label(d.get('education', {})) or (max(d.get('education', {}).items(), key=lambda kv: kv[1])[0] if d.get('education') else '')

        # These are now drawn from the same combined core + topic-causal
        # probability model used in the live inferred distribution display.
        training = self._sample_percent_label(d.get('training_style', {})) or 'mixed'
        familiarity = self._sample_percent_label(d.get('domain_familiarity', {})) or 'medium'
        interest = self._sample_percent_label(d.get('topic_interest', {})) or 'medium'
        exposure = self._sample_percent_label(d.get('prior_exposure', {})) or 'medium'

        out = {
            'occupation': occ_pick.replace('_', ' ').title() if occ_pick else '',
            'education_level': edu_pick.replace('_', ' ').title() if edu_pick else '',
            'training_style': training,
            'domain_familiarity': familiarity,
            'topic_interest': interest,
            'prior_exposure': exposure,
            # custom_topic_notes is deliberately NOT auto-filled: it is the
            # user's field. If filled while the topic-background layer is ON,
            # the renderer prints it in the card as 'Topic note: ...'.
        }
        return out

    def _flavor_generated_fields(self, d, flavor_text):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        age_pick = self._sample_percent_label(d.get('age', {})) or (max(d.get('age', {}).items(), key=lambda kv: kv[1])[0] if d.get('age') else '')
        pol_pick = self._sample_percent_label(d.get('politics', {})) or (max(d.get('politics', {}).items(), key=lambda kv: kv[1])[0] if d.get('politics') else '')
        lines = [ln.strip() for ln in flavor_text.splitlines() if ln.strip()]
        tone = ''
        for ln in lines:
            if ln.lower().startswith('tone suggestion:'):
                tone = ln.split(':', 1)[1].strip()
                break
        lifestyle = f"Likely tilt: {pol_pick.replace('_', ' ')}; {lines[-1] if lines else ''}".strip()
        gender = self._sample_weighted_option([('male', 0.34), ('female', 0.34), ('unspecified', 0.32)])
        ethnicity = self._sample_percent_label(d.get('ethnicity', {})) or self._sample_weighted_option([('unspecified', 0.22), ('caucasian', 0.38), ('hispanic', 0.17), ('black', 0.12), ('asian', 0.07), ('mixed', 0.03), ('other', 0.01)])
        return {
            'age_group': age_pick,
            'flavor_gender': gender,
            'flavor_ethnicity': ethnicity,
            'tone_hint': tone,
            'lifestyle_notes': lifestyle,
        }

    def _suggest_topic_and_flavor(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        d = self._derive_persona_distributions()
        snap = d.get("core_snapshot", {})
        trust = snap.get("trust", "medium")
        update = snap.get("update", "medium")
        suspicion = snap.get("suspicion", "medium")
        uncertainty = snap.get("uncertainty", "medium")
        evidence = snap.get("evidence", "mixed")
        value_orientation = snap.get("value_orientation", "balanced")
        agency = snap.get("agency", "balanced")
        conflict = snap.get("conflict", "balanced")
        politics_hint = self._top_distribution_labels(d.get("politics", {}), top_n=2)
        ethnicity_hint = self._top_distribution_labels(d.get("ethnicity", {}), top_n=3)
        age_hint = self._top_distribution_labels(d.get("age", {}), top_n=2)
        occ_hint = self._top_distribution_labels(d.get("occupation", {}), top_n=2)
        edu_hint = self._top_distribution_labels(d.get("education", {}), top_n=2)
        training_hint = self._top_distribution_labels(d.get("training_style", {}), top_n=2)
        familiarity_hint = self._top_distribution_labels(d.get("domain_familiarity", {}), top_n=2)
        interest_hint = self._top_distribution_labels(d.get("topic_interest", {}), top_n=2)
        exposure_hint = self._top_distribution_labels(d.get("prior_exposure", {}), top_n=2)
        topic_causal_values = d.get("topic_causal_snapshot", {}) or {}
        topic_causal_labels = d.get("topic_causal_labels", {}) or {}
        topic_profile_choice = d.get("topic_causal_profile_choice", "") or ""
        use_topic_causal = bool(d.get("use_topic_causal", 0))

        causal_pairs = []
        if use_topic_causal:
            for key, val in topic_causal_values.items():
                if str(val or '').strip():
                    causal_pairs.append(f"{topic_causal_labels.get(key, key.replace('_', ' ').title())}: {val}")
        causal_hint = "; ".join(causal_pairs[:6])

        topic_lines = []
        shown_core = []
        for label, val in [("institutional trust", trust), ("official-narrative suspicion", suspicion), ("openness to update", update), ("uncertainty tolerance", uncertainty)]:
            if val not in {'', 'off'}:
                shown_core.append(f"{val} {label}")
        if shown_core:
            topic_lines.append("Likely baseline lens: " + ", ".join(shown_core) + ".")
        if use_topic_causal and causal_hint:
            if topic_profile_choice and topic_profile_choice != TOPIC_PROFILE_MANUAL:
                topic_lines.append(f"Topic-causal profile: {topic_profile_choice} for {d.get('topic_label', get_topic_label(self.prompt_version_var.get()))}.")
            topic_lines.append("Topic-causal traits shaping generated probabilities: " + causal_hint + ".")
        topic_lines.append(f"Probable demographic/background tilt: age {age_hint}; politics {politics_hint}; ethnicity {ethnicity_hint}; education {edu_hint}; occupation {occ_hint}.")
        topic_lines.append(f"Probable topic-background tilt: training {training_hint}; familiarity {familiarity_hint}; interest {interest_hint}; prior exposure {exposure_hint}.")
        if evidence not in {'', 'off', 'mixed'}:
            topic_lines.append(f"They tend to evaluate arguments in a {evidence.replace('_', '-')} way and are likely to foreground that lens on topic-specific evidence.")
        if value_orientation == "procedural":
            topic_lines.append("They are more likely to treat process legitimacy and whether citizens truly had a choice as central to the judgment.")
        elif value_orientation == "outcome_focused":
            topic_lines.append("They are more likely to judge the topic by downstream consequences rather than by whether the formal process looked democratic.")
        if agency == "high_agency":
            topic_lines.append("They are more likely to see direct public participation as meaningful even when outcomes are messy.")
        elif agency == "fatalistic":
            topic_lines.append("They are more likely to assume structural pressures overpower individual or voter agency.")
        topic_lines.append(f"For {get_topic_label(self.prompt_version_var.get())} runs, the high-weight topic-causal layer changes with the selected prompt version.")
        topic_lines.append("Role/expertise is a medium/high lens only when relevant; flavor fields should mainly alter wording, not belief direction.")

        flavor_lines = []
        flavor_lines.append(f"Suggested age mix: {age_hint}.")
        flavor_lines.append(f"Suggested social/political mix: {politics_hint}.")
        flavor_lines.append(f"Suggested ethnicity mix: {ethnicity_hint}.")
        if trust == "high" and suspicion == "low":
            flavor_lines.append("Tone suggestion: measured, institutionally anchored, less drawn to anomaly-hunting.")
        elif suspicion == "high" and trust == "low":
            flavor_lines.append("Tone suggestion: sharper, more oppositional, quick to notice manipulation or hidden constraints.")
        else:
            flavor_lines.append("Tone suggestion: balanced but willing to note caveats and unresolved details.")
        if use_topic_causal and topic_causal_values:
            root = str(d.get('topic_root', '') or '')
            if root == 'v52' and str(topic_causal_values.get('engineering_evidence_weight', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: technical, engineering-first, focused on mechanism and feasibility.")
            elif root == 'v52' and str(topic_causal_values.get('visual_anomaly_sensitivity', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: visually attentive, quick to notice image/video inconsistencies.")
            elif root == 'v119' and str(topic_causal_values.get('anomaly_sensitivity', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: investigative, anomaly-attentive, detail-checking.")
            elif root == 'v119' and str(topic_causal_values.get('source_skepticism', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: source-critical, careful about both official and alternative claims.")
            elif root == 'v130' and str(topic_causal_values.get('computational_worldview', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: comfortable using computational analogies and abstract technical language.")
            elif root == 'v130' and str(topic_causal_values.get('testability_preference', '')).lower() == 'high':
                flavor_lines.append("Topic-causal flavor: empirical, test-focused, cautious about unfalsifiable claims.")
        if update == "low":
            flavor_lines.append("Likely interpersonal flavor: settled, somewhat stubborn, less eager to concede after one tweet.")
        elif update == "high":
            flavor_lines.append("Likely interpersonal flavor: flexible, willing to shift if one concrete point lands clearly.")
        else:
            flavor_lines.append("Likely interpersonal flavor: measured, not rigid, but not eager to overreact either.")
        if conflict == 'combative':
            flavor_lines.append("Debate stance: comfortable with sharper disagreement and more direct pushback.")
        elif conflict == 'consensus_seeking':
            flavor_lines.append("Debate stance: prefers accommodation and less abrasive disagreement.")
        return "\n".join(topic_lines), "\n".join(flavor_lines), d

    def _apply_live_persona_generation(self, *_args, only=None):
        """Regenerate live suggestions and apply them to the auto-managed fields.

        only=None refreshes both groups (trace callbacks / Regenerate button);
        only='topic' / only='flavor' limits BOTH the suggestion boxes and the
        field application to that group, so resetting one group can no longer
        re-roll the other one's values."""
        if not hasattr(self, 'persona_preview_text'):
            return
        topic_text, flavor_text, d = self._suggest_topic_and_flavor()
        try:
            self.persona_distribution_text.configure(state='normal')
            self.persona_distribution_text.delete('1.0', 'end')
            self.persona_distribution_text.insert(
                '1.0',
                'Age: ' + self._top_distribution_labels(d.get('age', {}), 4) + '\n' +
                'Politics: ' + self._top_distribution_labels(d.get('politics', {}), 4) + '\n' +
                'Ethnicity: ' + self._top_distribution_labels(d.get('ethnicity', {}), 4) + '\n' +
                'Education: ' + self._top_distribution_labels(d.get('education', {}), 4) + '\n' +
                'Occupation: ' + self._top_distribution_labels(d.get('occupation', {}), 4) + '\n' +
                'Training: ' + self._top_distribution_labels(d.get('training_style', {}), 4) + '\n' +
                'Familiarity: ' + self._top_distribution_labels(d.get('domain_familiarity', {}), 3) + '\n' +
                'Topic interest: ' + self._top_distribution_labels(d.get('topic_interest', {}), 3) + '\n' +
                'Prior exposure: ' + self._top_distribution_labels(d.get('prior_exposure', {}), 3)
            )
            self.persona_distribution_text.configure(state='disabled')
        except Exception:
            pass
        if only in (None, 'topic'):
            try:
                self.persona_topic_generated_text.configure(state='normal')
                self.persona_topic_generated_text.delete('1.0', 'end')
                self.persona_topic_generated_text.insert('1.0', topic_text)
                self.persona_topic_generated_text.configure(state='disabled')
            except Exception:
                pass
        if only in (None, 'flavor'):
            try:
                self.persona_flavor_generated_text.configure(state='normal')
                self.persona_flavor_generated_text.delete('1.0', 'end')
                self.persona_flavor_generated_text.insert('1.0', flavor_text)
                self.persona_flavor_generated_text.configure(state='disabled')
            except Exception:
                pass

        topic_override = bool(getattr(self, 'persona_topic_manual_override_var', tk.BooleanVar(value=False)).get())
        flavor_override = bool(getattr(self, 'persona_flavor_manual_override_var', tk.BooleanVar(value=False)).get())
        self._persona_auto_apply_in_progress = True
        try:
            if only in (None, 'topic') and not topic_override:
                for key, value in self._topic_generated_fields(d, topic_text).items():
                    var = getattr(self, f'persona_{key}_var', None)
                    if var is not None:
                        var.set(value)
            if only in (None, 'flavor') and not flavor_override:
                for key, value in self._flavor_generated_fields(d, flavor_text).items():
                    var = getattr(self, f'persona_{key}_var', None)
                    if var is not None:
                        var.set(value)
        finally:
            self._persona_auto_apply_in_progress = False
        self._update_json_persona_preview()

    def _mark_topic_manual_override(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if getattr(self, '_persona_auto_apply_in_progress', False):
            return
        if hasattr(self, 'persona_topic_manual_override_var'):
            self.persona_topic_manual_override_var.set(True)

    def _mark_topic_causal_manual_override(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if getattr(self, '_persona_auto_apply_in_progress', False):
            return
        if hasattr(self, 'persona_topic_causal_manual_override_var'):
            self.persona_topic_causal_manual_override_var.set(True)
            if hasattr(self, 'persona_topic_causal_profile_choice_var') and self.persona_topic_causal_profile_choice_var.get().strip() != TOPIC_PROFILE_MANUAL:
                # The profile remains visible as the starting point, but manual edits now take precedence.
                pass

    def _mark_flavor_manual_override(self, *_args):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if getattr(self, '_persona_auto_apply_in_progress', False):
            return
        if hasattr(self, 'persona_flavor_manual_override_var'):
            self.persona_flavor_manual_override_var.set(True)

    def _reset_topic_causal_to_profile(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self.persona_topic_causal_manual_override_var.set(False)
        self._apply_topic_causal_profile_choice()

    def _reset_topic_to_generated(self):
        """Clear the topic-background fields and refill them from fresh suggestions.
        Scoped: flavor fields and the flavor suggestion box are left untouched."""
        self.persona_topic_manual_override_var.set(False)
        self._persona_auto_apply_in_progress = True
        try:
            for key in ['occupation', 'education_level', 'training_style', 'domain_familiarity', 'topic_interest', 'prior_exposure', 'custom_topic_notes']:
                var = getattr(self, f'persona_{key}_var', None)
                if var is not None:
                    var.set('')
        finally:
            self._persona_auto_apply_in_progress = False
        self._apply_live_persona_generation(only='topic')

    def _reset_flavor_to_generated(self):
        """Clear the flavor fields and refill them from fresh suggestions.
        Scoped: topic fields and the topic suggestion box are left untouched."""
        self.persona_flavor_manual_override_var.set(False)
        self._persona_auto_apply_in_progress = True
        try:
            for key in ['age_group', 'flavor_gender', 'flavor_ethnicity', 'lifestyle_notes', 'tone_hint']:
                var = getattr(self, f'persona_{key}_var', None)
                if var is not None:
                    var.set('')
        finally:
            self._persona_auto_apply_in_progress = False
        self._apply_live_persona_generation(only='flavor')


    def _walk_widgets(self, root):
        """Collect a widget and all its descendants (best effort)."""
        out = [root]
        try:
            for ch in root.winfo_children():
                out.extend(self._walk_widgets(ch))
        except Exception:
            pass
        return out

    def _register_solo_irrelevant_frame(self, frame):
        """Frames registered here are greyed out while Solo check is ON."""
        try:
            self._solo_irrelevant_frames = getattr(self, '_solo_irrelevant_frames', [])
            self._solo_irrelevant_frames.append(frame)
        except Exception:
            pass

    def _update_solo_check_state(self, *_args):
        """Solo check uses only model / prompt version / seed / agents / decoding.
        While ON, every registered irrelevant section (network, personas, RAG,
        world/bias/global constraints, step overrides) plus the Steps entry is
        disabled, with each widget's previous state saved; turning it OFF
        restores those states and re-runs the conditional-greying updaters so
        rules like 'custom counts only when dist=custom_counts' win again."""
        on = (getattr(self, 'solo_check_var', tk.StringVar(value='off')).get() or 'off').strip().lower() == 'on'
        targets = []
        for fr in list(getattr(self, '_solo_irrelevant_frames', [])):
            targets.extend(self._walk_widgets(fr))
        steps_entry = getattr(self, '_steps_entry', None)
        if steps_entry is not None:
            targets.append(steps_entry)
        saved = getattr(self, '_solo_saved_states', None)
        if on:
            if saved is None:
                saved = {}
                for w in targets:
                    try:
                        cur = str(w.cget('state'))
                    except Exception:
                        continue
                    saved[w] = cur
                    try:
                        w.configure(state='disabled')
                    except Exception:
                        pass
                self._solo_saved_states = saved
        else:
            if saved:
                for w, st in saved.items():
                    try:
                        w.configure(state=st)
                    except Exception:
                        pass
            self._solo_saved_states = None
            for fn_name in ('_update_network_controls_state', '_update_rag_controls_state',
                            '_update_true_open_controls_state', '_update_classic_persona_controls_state',
                            '_update_json_profile_controls_state', '_update_allowed_update_controls_state',
                            '_update_light_memory_threshold_state', '_update_model_dependent_controls_state',
                            '_update_multi_run_controls_state'):
                try:
                    getattr(self, fn_name)()
                except Exception:
                    pass

    def _on_close(self):
        """Handle a Tkinter/UI event and update dependent controls."""
        save_settings(self._collect_settings())
        self.destroy()



    # ---- Command building ----

    def build_command(self, seed_override: int | None = None, out_override: str | None = None) -> list[str]:
        """Build the command-line invocation for the selected simulation or plotting script from UI state."""
        script_raw = self.script_var.get().strip()
        if not script_raw:
            raise ValueError("No script selected.")
        script_path = _resolve_existing_path(script_raw, prefer_project_root=False)
        if not script_path.exists():
            raise ValueError(f"Script not found: {script_path}")
        script = str(script_path)

        seed = int(seed_override) if seed_override is not None else int(self.seed_var.get())
        agents = int(self.agents_var.get())
        steps = int(self.steps_var.get())
        out = (out_override if out_override is not None else self.out_var.get().strip())

        model = self.model_var.get().strip()

        prompt_version = self.prompt_version_var.get().strip()
        prompt_mode = self.prompt_mode_var.get().strip()
        fact_pack_mode = (getattr(self, "fact_pack_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if fact_pack_mode not in DEFAULT_FACT_PACK_MODES:
            raise ValueError(f"Fact pack mode must be one of: {', ' .join(DEFAULT_FACT_PACK_MODES)}")
        prompt_folder = PROMPT_ROOT / prompt_version / f"{prompt_version}_{prompt_mode}"
        if not prompt_folder.exists():
            alt_root = (LAUNCHER_DIR / "prompts" / "opinion_dynamics" / "Flache_2017")
            alt_folder = alt_root / prompt_version / f"{prompt_version}_{prompt_mode}"
            if alt_folder.exists():
                prompt_folder = alt_folder
            else:
                raise ValueError(
                    f"Prompt folder does not exist:\n{prompt_folder}\n\n"
                    f"Launcher dir: {LAUNCHER_DIR}\n"
                    f"Project root: {PROJECT_ROOT}\n\n"
                    f"Generate it first from your prompt templates for {prompt_version}."
                )

        cmd = [sys.executable, script]
        cmd += ["-agents", str(agents)]
        cmd += ["-steps", str(steps)]
        cmd += ["-seed", str(seed)]
        if out:
            cmd += ["--out", out]

        if model and model.lower() != "none":
            cmd += ["-m", model]


                # Decoding params: only pass flags that the selected script actually supports.
        # (This keeps plot scripts safe, and avoids crashing older simulations.)
        script_base = os.path.basename(script).lower()
        try:
            script_text = Path(script).read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            script_text = ""

        def _supports(flag: str) -> bool:
            """Feature-detect a CLI flag by scanning the target script's source text.
            Forgiving across script versions; can false-positive if a flag only
            appears in comments, so keep flag names out of dead text in the mains."""
            return flag.lower() in script_text

        think_mode = (getattr(self, "think_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if think_mode not in {"off", "on", "step3_only", "step2_only"}:
            raise ValueError("Think mode must be one of: off, on, step3_only, step2_only.")
        if _supports("--think_mode"):
            cmd += ["--think_mode", think_mode]

        # Global decoding defaults
        if _supports("--temperature") or _supports("-t"):
            temp = float(self.temp_var.get().strip() or "0.7")
            cmd += ["-t", str(temp)]

        top_p = self.top_p_var.get().strip()
        if top_p != "" and _supports("--top_p"):
            cmd += ["--top_p", str(float(top_p))]

        top_k = self.top_k_var.get().strip()
        if top_k != "" and _supports("--top_k"):
            cmd += ["--top_k", str(int(float(top_k)))]

        rep_pen = self.repeat_penalty_var.get().strip()
        if rep_pen != "" and _supports("--repeat_penalty"):
            cmd += ["--repeat_penalty", str(float(rep_pen))]

        rep_last_n = self.repeat_last_n_var.get().strip()
        if rep_last_n != "" and _supports("--repeat_last_n"):
            cmd += ["--repeat_last_n", str(int(float(rep_last_n)))]

        max_toks = self.max_tokens_var.get().strip()
        if max_toks != "" and _supports("--max_tokens"):
            cmd += ["--max_tokens", str(int(float(max_toks)))]

        min_p = getattr(self, "min_p_var", tk.StringVar(value="")).get().strip()
        if min_p != "" and _supports("--min_p"):
            cmd += ["--min_p", str(float(min_p))]

        pres_pen = getattr(self, "presence_penalty_var", tk.StringVar(value="")).get().strip()
        if pres_pen != "" and _supports("--presence_penalty"):
            cmd += ["--presence_penalty", str(float(pres_pen))]

        freq_pen = getattr(self, "frequency_penalty_var", tk.StringVar(value="")).get().strip()
        if freq_pen != "" and _supports("--frequency_penalty"):
            cmd += ["--frequency_penalty", str(float(freq_pen))]

        msc = getattr(self, "max_step_change_var", tk.StringVar(value="")).get().strip()
        if msc != "" and _supports("--max_step_change"):
            cmd += ["--max_step_change", str(int(float(msc)))]

        allowed_update_mode = self._normalize_allowed_update_mode_ui()
        if _supports("--allowed_update_mode"):
            cmd += ["--allowed_update_mode", allowed_update_mode]
        validation_strictness = (getattr(self, "validation_strictness_var", tk.StringVar(value="strict")).get() or "strict").strip()
        if _supports("--validation_strictness"):
            cmd += ["--validation_strictness", validation_strictness]
        # ADR-006 Component 2: pass --allow_silence through if the sim exposes it.
        allow_silence = (getattr(self, "allow_silence_var", tk.StringVar(value="off")).get() or "off").strip()
        if _supports("--allow_silence"):
            cmd += ["--allow_silence", allow_silence]
        # ADR-006 Component 3: p_reach policy + sub-parameters. policy is always
        # passed (uniform is the byte-identical default); each float sub-parameter
        # is passed only when non-blank, matching the min_p convention above.
        p_reach_policy = (getattr(self, "p_reach_policy_var", tk.StringVar(value="uniform")).get() or "uniform").strip()
        if _supports("--p_reach_policy"):
            cmd += ["--p_reach_policy", p_reach_policy]
        p_reach_uniform_value = getattr(self, "p_reach_uniform_value_var", tk.StringVar(value="")).get().strip()
        if p_reach_uniform_value != "" and _supports("--p_reach_uniform_value"):
            cmd += ["--p_reach_uniform_value", str(float(p_reach_uniform_value))]
        p_reach_homophily_k = getattr(self, "p_reach_homophily_k_var", tk.StringVar(value="")).get().strip()
        if p_reach_homophily_k != "" and _supports("--p_reach_homophily_k"):
            cmd += ["--p_reach_homophily_k", str(float(p_reach_homophily_k))]
        p_reach_shadowban_fraction = getattr(self, "p_reach_shadowban_fraction_var", tk.StringVar(value="")).get().strip()
        if p_reach_shadowban_fraction != "" and _supports("--p_reach_shadowban_fraction"):
            cmd += ["--p_reach_shadowban_fraction", str(float(p_reach_shadowban_fraction))]
        p_reach_shadowban_value = getattr(self, "p_reach_shadowban_value_var", tk.StringVar(value="")).get().strip()
        if p_reach_shadowban_value != "" and _supports("--p_reach_shadowban_value"):
            cmd += ["--p_reach_shadowban_value", str(float(p_reach_shadowban_value))]
        p_reach_enforcement = (getattr(self, "p_reach_enforcement_var", tk.StringVar(value="filter")).get() or "filter").strip()
        if p_reach_enforcement != "" and _supports("--p_reach_enforcement"):
            cmd += ["--p_reach_enforcement", p_reach_enforcement]
        # ADR-006 Component 4: pass --include_bian_scores through if the sim exposes it.
        include_bian_scores = (getattr(self, "include_bian_scores_var", tk.StringVar(value="off")).get() or "off").strip()
        if _supports("--include_bian_scores"):
            cmd += ["--include_bian_scores", include_bian_scores]
        wrong_side_requery = (getattr(self, "wrong_side_requery_var", tk.StringVar(value="off")).get() or "off").strip()
        if _supports("--wrong_side_explanation_requery"):
            cmd += ["--wrong_side_explanation_requery", wrong_side_requery]
        deterministic = (getattr(self, "deterministic_var", tk.StringVar(value="off")).get() or "off").strip()
        if _supports("--deterministic"):
            cmd += ["--deterministic", deterministic]
        structured_output = (getattr(self, "structured_output_var", tk.StringVar(value="off")).get() or "off").strip()
        if _supports("--structured_output"):
            cmd += ["--structured_output", structured_output]
        solo_check = (getattr(self, "solo_check_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if _supports("--solo_check"):
            cmd += ["--solo_check", solo_check]

        ss_unlock = getattr(self, "same_side_edge_unlock_hits_var", tk.StringVar(value="")).get().strip()
        if allowed_update_mode == "free_bounded":
            ss_unlock = "0"
        if ss_unlock != "" and _supports("--same_side_edge_unlock_hits"):
            cmd += ["--same_side_edge_unlock_hits", str(int(float(ss_unlock)))]

        same_rating_step3_mode = (getattr(self, "same_rating_step3_mode_var", tk.StringVar(value="skip_tweet_local")).get() or "skip_tweet_local").strip().lower()
        if same_rating_step3_mode not in {"skip_tweet_local", "skip_generic", "llm"}:
            same_rating_step3_mode = "skip_tweet_local"
        if _supports("--same_rating_step3_mode"):
            cmd += ["--same_rating_step3_mode", same_rating_step3_mode]

        # Per-step overrides (only if the script supports them)
        step_overrides = [

            ("--temperature_step2", getattr(self, "temp_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--temperature_step3", getattr(self, "temp_step3_var", tk.StringVar(value="")).get().strip(), float),

            ("--top_p_step2", getattr(self, "top_p_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--top_p_step3", getattr(self, "top_p_step3_var", tk.StringVar(value="")).get().strip(), float),

            ("--top_k_step2", getattr(self, "top_k_step2_var", tk.StringVar(value="")).get().strip(), int),
            ("--top_k_step3", getattr(self, "top_k_step3_var", tk.StringVar(value="")).get().strip(), int),

            ("--repeat_penalty_step2", getattr(self, "repeat_penalty_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--repeat_penalty_step3", getattr(self, "repeat_penalty_step3_var", tk.StringVar(value="")).get().strip(), float),

            ("--repeat_last_n_step2", getattr(self, "repeat_last_n_step2_var", tk.StringVar(value="")).get().strip(), int),
            ("--repeat_last_n_step3", getattr(self, "repeat_last_n_step3_var", tk.StringVar(value="")).get().strip(), int),

            ("--max_tokens_step2", getattr(self, "max_tokens_step2_var", tk.StringVar(value="")).get().strip(), int),
            ("--max_tokens_step3", getattr(self, "max_tokens_step3_var", tk.StringVar(value="")).get().strip(), int),

            ("--min_p_step2", getattr(self, "min_p_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--min_p_step3", getattr(self, "min_p_step3_var", tk.StringVar(value="")).get().strip(), float),
            ("--presence_penalty_step2", getattr(self, "presence_penalty_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--presence_penalty_step3", getattr(self, "presence_penalty_step3_var", tk.StringVar(value="")).get().strip(), float),

            ("--frequency_penalty_step2", getattr(self, "frequency_penalty_step2_var", tk.StringVar(value="")).get().strip(), float),
            ("--frequency_penalty_step3", getattr(self, "frequency_penalty_step3_var", tk.StringVar(value="")).get().strip(), float),
        ]

        for flag, raw, caster in step_overrides:
            if raw == "":
                continue
            if not _supports(flag):
                continue
            try:
                val = caster(raw)
            except Exception:
                raise ValueError(f"Invalid value for {flag}: {raw}")
            cmd += [flag, str(val)]

# Use script-native version flag (keeps opinion_dynamics_v5_network.py unchanged)
        version_set = f"{prompt_version}_{prompt_mode}"
        cmd += ["-version", version_set]
        if _supports("--fact_pack_mode"):
            cmd += ["--fact_pack_mode", fact_pack_mode]
# World mode (if supported by the script)
        world = (getattr(self, "world_var", tk.StringVar(value="closed")).get() or "closed").strip().lower()
        if world == "open":
            world = "open_no_rag"  # compatibility alias
        allowed_worlds = {"closed", "closed_strict", "closed_strict_rag", "open_no_rag", "open_rag", "true_open"}
        if world not in allowed_worlds:
            raise ValueError("World must be one of: closed, closed_strict, closed_strict_rag, open_no_rag, open_rag, true_open.")
        if _supports("--world") or _supports("--world_mode"):
            cmd += ["--world", world]

        # Retrieval / RAG (used when the selected world uses retrieval)
        if world in {"open_rag", "closed_strict_rag"}:
            rag_backend = (getattr(self, "rag_backend_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if rag_backend not in {"off", "simple", "dense", "graph"}:
                raise ValueError("RAG backend must be off/simple/dense/graph.")
            if _supports("--rag_backend"):
                cmd += ["--rag_backend", rag_backend]
            if rag_backend != "off":
                rag_corpus = (getattr(self, "rag_corpus_path_var", tk.StringVar(value="")).get() or "").strip()
                if not rag_corpus:
                    raise ValueError("RAG corpus path is required when a RAG world mode is selected and RAG backend is enabled.")
                if _supports("--rag_corpus_path"):
                    cmd += ["--rag_corpus_path", rag_corpus]
                rag_top_k = (getattr(self, "rag_top_k_var", tk.StringVar(value="4")).get() or "4").strip()
                if _supports("--rag_top_k"):
                    cmd += ["--rag_top_k", str(int(float(rag_top_k)))]
                rag_query_mode_step2 = (getattr(self, "rag_query_mode_step2_var", tk.StringVar(value="auto")).get() or "auto").strip().lower()
                if rag_query_mode_step2 not in {"auto", "claim", "tweet", "claim_plus_tweet"}:
                    raise ValueError("Step2 RAG query mode must be auto, claim, tweet, or claim_plus_tweet.")
                if _supports("--rag_query_mode_step2"):
                    cmd += ["--rag_query_mode_step2", rag_query_mode_step2]
                rag_query_mode_step3 = (getattr(self, "rag_query_mode_step3_var", tk.StringVar(value="claim_plus_tweet")).get() or "claim_plus_tweet").strip().lower()
                if rag_query_mode_step3 not in {"auto", "claim", "tweet", "claim_plus_tweet"}:
                    raise ValueError("Step3 RAG query mode must be auto, claim, tweet, or claim_plus_tweet.")
                if _supports("--rag_query_mode_step3"):
                    cmd += ["--rag_query_mode_step3", rag_query_mode_step3]
                rag_content_mode = (getattr(self, "rag_content_mode_var", tk.StringVar(value="full")).get() or "full").strip().lower()
                if rag_content_mode not in {"full", "balanced", "supportive_only", "criticism_only", "context_only"}:
                    raise ValueError("RAG content mode must be full, balanced, supportive_only, criticism_only, or context_only.")
                if _supports("--rag_content_mode"):
                    cmd += ["--rag_content_mode", rag_content_mode]
                rag_topic_override = (getattr(self, "rag_topic_override_var", tk.StringVar(value="")).get() or "").strip()
                if rag_topic_override and _supports("--rag_topic_override"):
                    cmd += ["--rag_topic_override", rag_topic_override]
                rag_max_chars = (getattr(self, "rag_max_chars_var", tk.StringVar(value="2200")).get() or "2200").strip()
                if _supports("--rag_max_chars"):
                    cmd += ["--rag_max_chars", str(int(float(rag_max_chars)))]

        if world == "true_open":
            web_backend = (getattr(self, "web_backend_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if web_backend not in {"off", "brave", "duckduckgo"}:
                raise ValueError("Web backend must be off, brave, or duckduckgo.")
            if _supports("--web_backend"):
                cmd += ["--web_backend", web_backend]

            planner_mode = (getattr(self, "planner_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if planner_mode not in {"off", "heuristic"}:
                raise ValueError("Planner mode must be off or heuristic.")
            if _supports("--planner_mode"):
                cmd += ["--planner_mode", planner_mode]

            tool_mode = (getattr(self, "tool_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if tool_mode not in {"off", "web_only", "multi"}:
                raise ValueError("Tool mode must be off, web_only, or multi.")
            if _supports("--tool_mode"):
                cmd += ["--tool_mode", tool_mode]

            web_top_k = (getattr(self, "web_top_k_var", tk.StringVar(value="3")).get() or "3").strip()
            if _supports("--web_top_k"):
                cmd += ["--web_top_k", str(int(float(web_top_k)))]

            web_max_chars = (getattr(self, "web_max_chars_var", tk.StringVar(value="1400")).get() or "1400").strip()
            if _supports("--web_max_chars"):
                cmd += ["--web_max_chars", str(int(float(web_max_chars)))]

            step2_web_mode = (getattr(self, "step2_web_mode_var", tk.StringVar(value="heuristic")).get() or "heuristic").strip().lower()
            if step2_web_mode not in {"off", "heuristic", "always"}:
                raise ValueError("Step2 web mode must be off, heuristic, or always.")
            if _supports("--step2_web_mode"):
                cmd += ["--step2_web_mode", step2_web_mode]

            notes_mode = (getattr(self, "notes_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
            if tool_mode != "multi":
                notes_mode = "off"
            if notes_mode not in {"off", "heuristic"}:
                raise ValueError("Notes mode must be off or heuristic.")
            if _supports("--notes_mode"):
                cmd += ["--notes_mode", notes_mode]

            notes_max_items = (getattr(self, "notes_max_items_var", tk.StringVar(value="3")).get() or "3").strip()
            if _supports("--notes_max_items"):
                cmd += ["--notes_max_items", str(int(float(notes_max_items)))]
        # Memory mode (if supported by the script): enabled|light|off
        mem = (getattr(self, "memory_var", tk.StringVar(value="enabled")).get() or "enabled").strip().lower()
        if mem not in {"enabled", "light", "off"}:
            raise ValueError("Memory must be 'enabled', 'light', or 'off'.")
        if _supports("--memory"):
            cmd += ["--memory", mem]
        light_threshold = (getattr(self, "light_memory_threshold_var", tk.StringVar(value="3")).get() or "3").strip()
        try:
            light_threshold_i = int(float(light_threshold))
        except Exception:
            light_threshold_i = 3
        if light_threshold_i < 1 or light_threshold_i > 5:
            raise ValueError("Light memory threshold must be between 1 and 5.")
        if mem == "light" and _supports("--light_memory_threshold"):
            cmd += ["--light_memory_threshold", str(light_threshold_i)]

        # Trace / logging verbosity (scripts that expose these flags)
        trace_raw = (getattr(self, "trace_var", tk.StringVar(value="auto")).get() or "auto").strip().lower()
        if trace_raw not in {"auto", "minimal", "full", "off"}:
            trace_raw = "auto"
        trace_eff = trace_raw
        if trace_raw == "auto":
            trace_eff = "minimal" if mem == "off" else "full"
        if _supports("--trace"):
            cmd += ["--trace", trace_eff]

        # LLM history injection into the prompt (scripts that expose these flags)
        llm_hist_raw = (getattr(self, "llm_history_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if llm_hist_raw not in {"off", "auto", "on"}:
            llm_hist_raw = "off"
        # If memory is off, force Markov + no history injection.
        llm_hist_eff = "off" if mem == "off" else ("on" if llm_hist_raw == "auto" else llm_hist_raw)
        if _supports("--llm_history"):
            cmd += ["--llm_history", llm_hist_eff]

        debug_native_raw = (getattr(self, "debug_native_thinking_on_fail_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if debug_native_raw not in {"off", "on"}:
            debug_native_raw = "off"
        if _supports("--debug_native_thinking_on_fail"):
            cmd += ["--debug_native_thinking_on_fail", debug_native_raw]

        # Network + interaction params
        # Use capability detection instead of hard-coded script name checks,
        # so opinion_dynamics_test_network.py and future network-capable scripts
        # receive the same UI-controlled flags when they support them.
        supports_any_network_flag = any(
            _supports(flag)
            for flag in (
                "--network_type",
                "--interaction_selection",
                "--interaction_homophily_mode",
                "--k_neighbors",
                "--p_rewire",
                "--er_p_edge",
                "--ba_m_attach",
                "--ba_hub_strategy",
                "--ba_hub_assignment_mode",
                "--ba_hub_custom",
                "--network_homophily",
        )
        )

        if supports_any_network_flag:
            net_type = (getattr(self, "network_type_var", tk.StringVar(value="ws")).get() or "ws").strip().lower()
            if net_type not in {"ws", "er", "ba", "none"}:
                raise ValueError("Network type must be one of: ws, er, ba, none.")

            if _supports("--network_type"):
                cmd += ["--network_type", net_type]

            # Interaction selection (applies both with and without a network)
            sel = (getattr(self, "interaction_selection_var", tk.StringVar(value="homophily")).get() or "homophily").strip().lower()
            if sel not in {"random", "homophily"}:
                raise ValueError("Interaction selection must be 'random' or 'homophily'.")

            if _supports("--interaction_selection"):
                cmd += ["--interaction_selection", sel]

            if sel == "homophily" and _supports("--interaction_homophily_mode"):
                mode = (getattr(self, "interaction_homophily_mode_var", tk.StringVar(value="full")).get() or "full").strip().lower()
                if mode not in {"full", "opinion_only"}:
                    raise ValueError("Homophily scoring must be 'full' or 'opinion_only'.")
                cmd += ["--interaction_homophily_mode", mode]

            if net_type != "none":
                if net_type == "ws":
                    if _supports("--k_neighbors"):
                        k_raw = (getattr(self, "k_neighbors_var", tk.StringVar(value="4")).get() or "4").strip()
                        k_val = int(float(k_raw))
                        if k_val <= 0 or (k_val % 2) != 0:
                            raise ValueError("k_neighbors must be a positive EVEN integer (e.g., 4, 6, 8).")
                        if k_val >= agents:
                            raise ValueError("k_neighbors must be < number of agents.")
                        cmd += ["--k_neighbors", str(k_val)]

                    if _supports("--p_rewire"):
                        p_raw = (getattr(self, "p_rewire_var", tk.StringVar(value="0.1")).get() or "0.1").strip()
                        p_val = float(p_raw)
                        if not (0.0 <= p_val <= 1.0):
                            raise ValueError("p_rewire must be between 0 and 1.")
                        cmd += ["--p_rewire", str(p_val)]
                elif net_type == "er":
                    if _supports("--er_p_edge"):
                        er_raw = (getattr(self, "er_p_edge_var", tk.StringVar(value="0.15")).get() or "0.15").strip()
                        er_val = float(er_raw)
                        if not (0.0 <= er_val <= 1.0):
                            raise ValueError("er_p_edge must be between 0 and 1.")
                        cmd += ["--er_p_edge", str(er_val)]
                elif net_type == "ba":
                    if _supports("--ba_m_attach"):
                        ba_raw = (getattr(self, "ba_m_attach_var", tk.StringVar(value="2")).get() or "2").strip()
                        ba_val = int(float(ba_raw))
                        if ba_val <= 0 or ba_val >= agents:
                            raise ValueError("ba_m_attach must be a positive integer < number of agents.")
                        cmd += ["--ba_m_attach", str(ba_val)]
                    hub_strategy = (getattr(self, "ba_hub_strategy_var", tk.StringVar(value="default")).get() or "default").strip().lower()
                    valid_hub_strategies = {"default", "random", "positive", "negative", "neutral", "extreme", "opposite_majority", "custom"}
                    if hub_strategy not in valid_hub_strategies:
                        raise ValueError("BA hub priority must be one of: default, random, positive, negative, neutral, extreme, opposite_majority, custom.")
                    if _supports("--ba_hub_strategy"):
                        cmd += ["--ba_hub_strategy", hub_strategy]
                    assignment_active = hub_strategy in {"positive", "negative", "neutral", "extreme", "opposite_majority", "custom"}
                    if assignment_active:
                        hub_assignment_mode = (getattr(self, "ba_hub_assignment_mode_var", tk.StringVar(value="early_position")).get() or "early_position").strip().lower()
                        valid_assignment_modes = {"early_position", "actual_hubs", "early_and_actual"}
                        if hub_assignment_mode not in valid_assignment_modes:
                            raise ValueError("BA hub assignment mode must be one of: early_position, actual_hubs, early_and_actual.")
                        if _supports("--ba_hub_assignment_mode"):
                            cmd += ["--ba_hub_assignment_mode", hub_assignment_mode]
                    if hub_strategy == "custom":
                        custom_hubs = (getattr(self, "ba_hub_custom_var", tk.StringVar(value="")).get() or "").strip()
                        if not custom_hubs:
                            raise ValueError("BA hub priority is custom, but no custom hubs were provided.")
                        if _supports("--ba_hub_custom"):
                            cmd += ["--ba_hub_custom", custom_hubs]

                hom = bool(getattr(self, "network_homophily_var", tk.BooleanVar(value=False)).get())
                if hom and _supports("--network_homophily"):
                    cmd += ["--network_homophily"]


        if self.use_personas_var.get():
            # Fixed filename, overwritten each run (as requested)
            csv_path = prompt_folder / "list_agent_descriptions.csv"
            self.csv_path_var.set(str(csv_path))

            opinion_strategy = self.opinion_strategy_var.get().strip()

            # v5 hybrid opinion control:
            # - If opinion_strategy is one of the v5 "-dist" presets, we pass "-dist <preset>"
            # - If opinion_strategy == "custom_counts", we pass "--custom_counts a,b,c,d,e"
            v5_dist_presets = {"uniform","skewed_positive","skewed_negative","positive","negative"}
            custom_counts = None
            custom_counts_str = None
            if opinion_strategy == "custom_counts":
                custom_counts_str = self.custom_counts_var.get().strip()
                parts = [p.strip() for p in custom_counts_str.split(",")]
                if len(parts) != 5:
                    raise ValueError("Custom counts must have 5 comma-separated integers (for -2,-1,0,1,2).")
                counts = list(map(int, parts))
                if sum(counts) != agents:
                    raise ValueError(f"Custom counts sum to {sum(counts)} but agents is {agents}.")
                custom_counts = {-2: counts[0], -1: counts[1], 0: counts[2], 1: counts[3], 2: counts[4]}
                # Pass through to the simulator's --custom_counts
                cmd += ["--custom_counts", custom_counts_str]
            else:
                if opinion_strategy not in v5_dist_presets:
                    raise ValueError(f"Unsupported v5 opinion distribution: {opinion_strategy}")
                cmd += ["-dist", opinion_strategy]

            use_json_persona_generation = bool(getattr(self, "use_json_persona_profiles_var", tk.BooleanVar(value=False)).get())
            use_json_persona_generation = use_json_persona_generation or bool(getattr(self, "persona_use_core_causal_var", tk.BooleanVar(value=False)).get()) or bool(getattr(self, "persona_use_topic_causal_var", tk.BooleanVar(value=False)).get()) or bool(getattr(self, "persona_use_topic_linked_var", tk.BooleanVar(value=False)).get()) or bool(getattr(self, "persona_use_flavor_only_var", tk.BooleanVar(value=False)).get())
            if use_json_persona_generation:
                profile = build_profile_from_ui(self._collect_json_profile_prompt_state())
                _SUPPORT_GENERATE_CSV_FROM_PROFILE(
                    out_csv_path=str(csv_path),
                    n_agents=agents,
                    opinion_strategy=opinion_strategy,
                    custom_counts=custom_counts,
                    seed=seed,
                    names_source_csv=self.names_source_var.get().strip(),
                    profile=profile,
                    profile_distribution_mode=getattr(self, "persona_profile_distribution_mode_var", tk.StringVar(value="all")).get().strip(),
                )
                _augment_generated_persona_csv(str(csv_path), self._collect_json_profile_prompt_state())
            else:
                generate_list_agent_descriptions_csv(
                    out_csv_path=str(csv_path),
                    n_agents=agents,
                    opinion_strategy=opinion_strategy,
                    custom_counts=custom_counts,
                    seed=seed,
                    names_source_csv=self.names_source_var.get().strip(),
                    age_preset=self.age_preset_var.get().strip(),
                    gender_preset=self.gender_preset_var.get().strip(),
                    education_preset=self.educ_preset_var.get().strip(),
                    political_preset=self.pol_preset_var.get().strip(),
                    ethnicity_preset=self.eth_preset_var.get().strip(),
                    occupation_preset=self.occ_preset_var.get().strip(),
                    early_life_preset=self.early_life_preset_var.get().strip(),
                    epistemic_profile_preset=getattr(self, "epistemic_profile_preset_var", tk.StringVar(value="none")).get().strip(),
                )

        return cmd

    # ---- Run/Stop ----

    def _results_base_dir(self) -> Path:
        """
        Base results directory for this experiment/model/version set.
        Mirrors how v5 stores CSVs/artifacts (no hardcoded absolute paths).
        """
        experiment_id = "Flache_2017"
        model_name = self.model_var.get().strip() or "llama3"
        version_set = f"{self.prompt_version_var.get().strip()}_{self.prompt_mode_var.get().strip()}".strip("_")
        base1 = Path("results") / "opinion_dynamics" / experiment_id / model_name / version_set
        base2 = Path("results") / "opinion_dynamics" / experiment_id / model_name
        return base1 if base1.exists() else base2

    def _run_results_dir(self) -> Path:
        """
        Directory where this run's CSVs are stored (based on out).
        """
        out_name = (self.out_var.get().strip() or "").strip()
        if out_name:
            return self._results_base_dir() / out_name
        return self._results_base_dir()

    def _network_log_dir(self) -> Path:
        """
        v5 log folder
        """
        experiment_id = "Flache_2017"
        model_name = self.model_var.get().strip() or "llama3"
        
        return Path("results") / "opinion_dynamics" / experiment_id / model_name/ "network_log_conversation"


    # ---- Multiple runs helpers ----

    def _normalize_multi_run_mode(self) -> str:
        """Normalize user-provided or file-derived text into the form expected downstream."""
        raw = (getattr(self, "multi_run_mode_var", tk.StringVar(value="off")).get() or "off").strip().lower()
        if raw in {"same seed", "same_seed", "same"}:
            return "same_seed"
        if raw in {"consecutive seeds", "consecutive_seeds", "consecutive"}:
            return "consecutive_seeds"
        return "off"

    def _derive_out_for_seed(self, out_template: str, seed_int: int) -> str:
        """
        Replace first occurrence of seed<digits> with the target seed. If missing, prefix seed<seed_int>_.
        """
        out_template = (out_template or "").strip()
        if not out_template:
            return f"seed{seed_int}"
        new_out, n = re.subn(r"seed\d+", f"seed{seed_int}", out_template, count=1, flags=re.IGNORECASE)
        if n == 0:
            return f"seed{seed_int}_{out_template}"
        return new_out

    def _start_next_batch_run(self):
        """
        Pop next (seed,out) from queue and run it.
        """
        if not self._batch_active:
            return
        if not self._batch_queue:
            # finished
            self._batch_active = False
            self.run_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self._append_log("\n[BATCH] All runs completed.\n")
            return

        nxt = self._batch_queue.pop(0)
        self._batch_index += 1
        seed_i = int(nxt["seed"])
        out_i = str(nxt["out"])
        try:
            cmd = self.build_command(seed_override=seed_i, out_override=out_i)
        except Exception as e:
            self._append_log(f"\n[BATCH][ERROR] Failed to build command for run {self._batch_index}/{self._batch_total}: {e}\n")
            # stop the batch
            self._batch_queue = []
            self._batch_active = False
            self.run_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            return

        self.status_var.set(f"Running batch {self._batch_index}/{self._batch_total} (seed={seed_i})…")
        self._append_log(f"\n[BATCH] Run {self._batch_index}/{self._batch_total} | seed={seed_i} | out={out_i}\n")
        self.runner.start(cmd_list=cmd, cwd=str(PROJECT_ROOT))

    def on_run(self):
        """Handle a Tkinter/UI event and update dependent controls."""
        if self.runner.is_running():
            messagebox.showinfo("Running", "A process is already running.")
            return

        mode = self._normalize_multi_run_mode()
        try:
            n_runs = int(float((getattr(self, "multi_run_count_var", tk.StringVar(value="1")).get() or "1").strip()))
        except Exception:
            n_runs = 1

        # Single run path (default)
        if mode == "off" or n_runs <= 1:
            try:
                cmd = self.build_command()
                save_settings(self._collect_settings())
            except Exception as e:
                messagebox.showerror("Invalid configuration", str(e))
                return

            self._batch_active = False
            self._batch_queue = []
            self.run_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.status_var.set("Running…")
            try:
                self._append_log(f"\n[RUN] seed={self.seed_var.get().strip()} | out={self.out_var.get().strip()}\n")
            except Exception:
                pass
            self.runner.start(cmd_list=cmd, cwd=str(PROJECT_ROOT))
            return

        # Batch run path
        try:
            base_seed = int(self.seed_var.get().strip())
            base_out = (self.out_var.get().strip() or "")
            if n_runs < 1:
                raise ValueError("Runs must be >= 1.")
            queue: list[dict] = []
            if mode == "same_seed":
                for i in range(n_runs):
                    out_i = base_out if i == 0 else (f"{base_out}_{i}" if base_out else f"seed{base_seed}_{i}")
                    queue.append({"seed": base_seed, "out": out_i})
            else:  # consecutive_seeds
                for i in range(n_runs):
                    seed_i = base_seed + i
                    out_i = self._derive_out_for_seed(base_out, seed_i)
                    queue.append({"seed": seed_i, "out": out_i})

            # Save settings once (base settings, not per-run overrides)
            save_settings(self._collect_settings())
        except Exception as e:
            messagebox.showerror("Invalid configuration", str(e))
            return

        self._batch_active = True
        self._batch_queue = queue
        self._batch_total = len(queue)
        self._batch_index = 0

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._append_log(f"\n[BATCH] Starting {self._batch_total} runs (mode={mode}, base_seed={base_seed}).\n")
        self._start_next_batch_run()


    # =============================
    # LIVE TAB (real-time distribution)
    # =============================

    def _build_live_tab(self, parent):
        """
        Live distribution viewer.

        It attaches automatically if the running script prints:
          [INFO] Opinion-change CSV: <path>

        Or you can attach manually via the Browse button.
        """
        # --- state ---
        self._live_import_ok = False
        self._live_csv_path = None
        self._live_fh = None
        self._live_poll_job = None

        self._live_steps = []
        self._live_counts = []   # list of 5-tuples: counts for [-2,-1,0,1,2]
        self._live_bdp = []      # list of (B,D,P)
        self._live_vectors = []  # full opinion vectors for network view
        self._live_view_mode = tk.StringVar(value="distribution")
        self._live_network_graph = None
        self._live_network_pos = None
        self._live_network_sig = None
        self._live_nx = None
        self._live_build_small_world = None
        self._live_build_erdos_renyi = None
        self._live_build_barabasi_albert = None

        self._live_follow_var = tk.BooleanVar(value=True)
        self._live_pause_var = tk.BooleanVar(value=False)
        self._live_status_var = tk.StringVar(value="Not attached.")
        self._live_csv_var = tk.StringVar(value="")
        self._live_idx_var = tk.IntVar(value=0)

        # --- UI ---
        header = ttk.Frame(parent)
        header.pack(fill="x", pady=(0, 10))

        ttk.Label(
            header,
            text="Live opinion distribution",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="Updates as each step completes. Use the slider to go back in time.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.LabelFrame(parent, text="Controls", padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Opinion-change CSV:").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(controls, textvariable=self._live_csv_var, width=95, state="readonly")
        entry.grid(row=0, column=1, sticky="we", padx=6)

        ttk.Button(controls, text="Browse…", command=self._live_browse_attach).grid(row=0, column=2, sticky="e", padx=4)
        ttk.Button(controls, text="Detach", command=self._live_detach).grid(row=0, column=3, sticky="e", padx=4)

        self._live_view_btn = ttk.Button(controls, text="Show network", command=self._live_toggle_view)
        self._live_view_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))

        ttk.Checkbutton(controls, text="Follow latest", variable=self._live_follow_var, command=self._live_on_follow_toggle).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(controls, text="Pause auto-refresh", variable=self._live_pause_var).grid(row=1, column=2, sticky="w", pady=(6, 0), padx=(10, 0))

        ttk.Label(controls, textvariable=self._live_status_var, style="Muted.TLabel").grid(row=1, column=3, sticky="e", pady=(6, 0))
        ttk.Label(controls, text="Opinion-change CSV: the opinion series being watched - attaches automatically to the current run, or Browse any finished run to replay it (Detach stops watching).  Show network: toggle line-chart vs network view.  Follow latest: auto-jump to the newest step as it lands.  Pause auto-refresh: freeze polling while you inspect.  Step slider below: scrub through history.",
                  style="Muted.TLabel", wraplength=1100, justify="left").grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        try:
            controls.grid_columnconfigure(1, weight=1)
        except Exception:
            pass

        # Slider (index over stored history)
        slider_wrap = ttk.Frame(parent)
        slider_wrap.pack(fill="x", pady=(10, 6))

        ttk.Label(slider_wrap, text="Step:").pack(side="left")
        self._live_step_label_var = tk.StringVar(value="—")
        ttk.Label(slider_wrap, textvariable=self._live_step_label_var, style="Muted.TLabel").pack(side="left", padx=(6, 0))

        ttk.Button(slider_wrap, text="◀", width=3, command=lambda: self._live_step_by(-1)).pack(side="left", padx=(10, 0))

        self._live_scale = tk.Scale(
            slider_wrap,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            length=820,
            resolution=1,
            command=lambda _v: self._live_on_slider(),
        )
        self._live_scale.pack(side="left", fill="x", expand=True, padx=(8, 0))

        ttk.Button(slider_wrap, text="▶", width=3, command=lambda: self._live_step_by(1)).pack(side="left", padx=(8, 0))
        ttk.Button(slider_wrap, text="Go live", command=self._live_go_live).pack(side="left", padx=(10, 0))

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)

        # Lazy matplotlib import so the launcher still runs if matplotlib isn't installed.
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            import networkx as nx
            from network_models import build_small_world, build_erdos_renyi, build_barabasi_albert
            self._mpl_Figure = Figure
            self._mpl_Canvas = FigureCanvasTkAgg
            self._mpl_Toolbar = NavigationToolbar2Tk
            self._live_nx = nx
            self._live_build_small_world = build_small_world
            self._live_build_erdos_renyi = build_erdos_renyi
            self._live_build_barabasi_albert = build_barabasi_albert
            self._live_import_ok = True
        except Exception as e:
            self._live_import_ok = False
            ttk.Label(
                plot_frame,
                text=f"matplotlib is not available, so the Live tab can't render plots.\n{e}",
                style="Muted.TLabel",
                justify="left",
            ).pack(anchor="w", pady=10)
            return

        self._live_fig = self._mpl_Figure(figsize=(7.2, 4.8), dpi=100)
        self._live_ax = self._live_fig.add_subplot(111)

        self._live_canvas = self._mpl_Canvas(self._live_fig, master=plot_frame)
        self._live_canvas.draw()
        self._live_canvas.get_tk_widget().pack(fill="both", expand=True)
        try:
            w = self._live_canvas.get_tk_widget()
            w.bind("<Left>", lambda _e: self._live_step_by(-1))
            w.bind("<Down>", lambda _e: self._live_step_by(-1))
            w.bind("<Right>", lambda _e: self._live_step_by(1))
            w.bind("<Up>", lambda _e: self._live_step_by(1))
            w.bind("<Button-1>", lambda _e: w.focus_set())
            self.bind_all("<Left>", lambda e: self._live_keyboard_step(-1, e), add="+")
            self.bind_all("<Down>", lambda e: self._live_keyboard_step(-1, e), add="+")
            self.bind_all("<Right>", lambda e: self._live_keyboard_step(1, e), add="+")
            self.bind_all("<Up>", lambda e: self._live_keyboard_step(1, e), add="+")
        except Exception:
            pass

        try:
            toolbar = self._mpl_Toolbar(self._live_canvas, plot_frame)
            toolbar.update()
        except Exception:
            pass

        self._live_init_plot()
        self._update_live_network_toggle_visibility()
        self._live_schedule_poll()

    def _live_opinion_color(self, rating: int | None) -> str:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            r = int(rating)
        except Exception:
            r = 0
        mapping = {
            -2: "#ef4444",  # red
            -1: "#f97316",  # orange
             0: "#d1d5db",  # light gray
             1: "#facc15",  # yellow
             2: "#22c55e",  # green
        }
        return mapping.get(r, "#d1d5db")

    def _live_network_selected(self) -> bool:
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            net_type = (self.network_type_var.get() or "").strip().lower()
        except Exception:
            return False
        return net_type not in {"", "none", "no_network", "no network"}

    def _update_live_network_toggle_visibility(self):
        """Synchronize UI state or derived values after a setting changes."""
        btn = getattr(self, "_live_view_btn", None)
        if btn is None:
            return
        visible = bool(self._live_network_selected())
        try:
            if visible:
                btn.grid()
            else:
                btn.grid_remove()
        except Exception:
            pass
        if not visible and str(getattr(self, "_live_view_mode", tk.StringVar(value="distribution")).get()) == "network":
            try:
                self._live_view_mode.set("distribution")
            except Exception:
                pass
        self._update_live_view_button_text()
        try:
            if getattr(self, "_live_import_ok", False):
                if getattr(self, "_live_counts", None):
                    self._live_update_plot(int(float(self._live_scale.get())))
                else:
                    self._live_init_plot()
        except Exception:
            pass

    def _update_live_view_button_text(self):
        """Synchronize UI state or derived values after a setting changes."""
        btn = getattr(self, "_live_view_btn", None)
        if btn is None:
            return
        try:
            mode = str(self._live_view_mode.get())
        except Exception:
            mode = "distribution"
        btn.configure(text="Show distribution" if mode == "network" else "Show network")

    def _live_toggle_view(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not self._live_network_selected():
            return
        try:
            mode = str(self._live_view_mode.get())
        except Exception:
            mode = "distribution"
        new_mode = "distribution" if mode == "network" else "network"
        try:
            self._live_view_mode.set(new_mode)
        except Exception:
            pass
        self._update_live_view_button_text()

        # Recreate the target plot from scratch. This prevents the old bug where
        # switching back from network mode left stale/removed bar artists behind.
        try:
            if new_mode == "distribution":
                self._live_draw_distribution_placeholder()
            else:
                self._live_draw_network_placeholder()
        except Exception:
            pass

        if self._live_counts:
            try:
                idx = int(float(self._live_scale.get()))
            except Exception:
                idx = len(self._live_counts) - 1
            self._live_update_plot(idx)
        else:
            self._live_init_plot()

    def _live_current_network_signature(self, agent_count: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            net_type = (self.network_type_var.get() or "ws").strip().lower()
            hom = bool(self.network_homophily_var.get())
            k = int(float(self.k_neighbors_var.get()))
            p = float(self.p_rewire_var.get())
            er = float(self.er_p_edge_var.get())
            ba = int(float(self.ba_m_attach_var.get()))
            seed = int(float(self.seed_var.get()))
            hub_strategy = (getattr(self, "ba_hub_strategy_var", tk.StringVar(value="default")).get() or "default").strip().lower()
            hub_custom = (getattr(self, "ba_hub_custom_var", tk.StringVar(value="")).get() or "").strip()
            hub_assignment_mode = (getattr(self, "ba_hub_assignment_mode_var", tk.StringVar(value="early_position")).get() or "early_position").strip().lower()
            if hub_assignment_mode not in {"early_position", "actual_hubs", "early_and_actual"}:
                hub_assignment_mode = "early_position"
            if hub_strategy in {"default", "random"}:
                hub_assignment_mode = "early_position"
        except Exception:
            return None
        return (net_type, hom, k, p, er, ba, seed, int(agent_count), hub_strategy, hub_custom, hub_assignment_mode)


    def _live_ba_hub_priority_nodes(self, strategy: str, custom_text: str, opinions, agent_count: int, seed: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        strategy = str(strategy or "default").strip().lower()
        if strategy == "default":
            return []
        rng = random.Random(seed)
        idxs = list(range(int(agent_count)))
        ops = [0 for _ in idxs]
        try:
            if opinions and len(opinions) == int(agent_count):
                ops = [int(x) for x in opinions]
        except Exception:
            ops = [0 for _ in idxs]

        if strategy == "random":
            selected = idxs[:]
            rng.shuffle(selected)
            return selected
        if strategy == "positive":
            selected = [i for i in idxs if ops[i] > 0]
        elif strategy == "negative":
            selected = [i for i in idxs if ops[i] < 0]
        elif strategy == "neutral":
            selected = [i for i in idxs if ops[i] == 0]
        elif strategy == "extreme":
            selected = [i for i in idxs if abs(ops[i]) == 2]
        elif strategy == "opposite_majority":
            pos = [i for i in idxs if ops[i] > 0]
            neg = [i for i in idxs if ops[i] < 0]
            if len(pos) > len(neg):
                selected = neg
            elif len(neg) > len(pos):
                selected = pos
            else:
                selected = []
        elif strategy == "custom":
            selected = []
            agent_names = []
            try:
                hdr = list(getattr(self, "_live_header", []) or [])
                # opinion-change CSV header is: time_step,<agent1>,...,<agentN>
                agent_names = [str(x).strip() for x in hdr[1:1 + int(agent_count)]]
            except Exception:
                agent_names = []
            name_to_idx = {str(name).strip().lower(): i for i, name in enumerate(agent_names) if str(name).strip()}
            for raw_tok in re.split(r"[,;\n]+", str(custom_text or "")):
                raw_tok = raw_tok.strip()
                tok = raw_tok.lower()
                if not tok:
                    continue
                m = re.match(r"^(?:idx|index|local)\s*[:=]\s*(-?\d+)\s*$", tok)
                if m:
                    val = int(m.group(1))
                else:
                    m = re.match(r"^(?:agent_id|agent|id)\s*[:=]\s*(-?\d+)\s*$", tok)
                    if m:
                        # agent_id is shown as 1-based in the UI labels.
                        val = int(m.group(1)) - 1
                    elif re.fullmatch(r"-?\d+", tok):
                        val = int(tok)
                    elif tok in name_to_idx:
                        val = int(name_to_idx[tok])
                    else:
                        val = None
                if val is not None and 0 <= int(val) < int(agent_count) and int(val) not in selected:
                    selected.append(int(val))
        else:
            selected = []
        rng.shuffle(selected)
        return selected

    def _live_build_network_if_needed(self, agent_count: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not getattr(self, "_live_import_ok", False):
            return False
        if not self._live_network_selected():
            return False
        sig = self._live_current_network_signature(agent_count)
        if sig is None:
            return False
        if self._live_network_graph is not None and self._live_network_sig == sig:
            return True
        nx = getattr(self, "_live_nx", None)
        if nx is None:
            return False
        try:
            seed = int(sig[6])
            net_type = sig[0]
            hom = bool(sig[1])
            init_opinions = None
            if self._live_vectors and len(self._live_vectors[0]) == int(agent_count):
                init_opinions = [int(x) for x in self._live_vectors[0]]
            if net_type == "ws":
                neighbors = self._live_build_small_world(
                    num_agents=int(agent_count),
                    k=int(sig[2]),
                    p_rewire=float(sig[3]),
                    seed=seed,
                    homophily=hom,
                    opinions=init_opinions,
                )
            elif net_type == "er":
                neighbors = self._live_build_erdos_renyi(
                    num_agents=int(agent_count),
                    p_edge=float(sig[4]),
                    seed=seed,
                    homophily=hom,
                    opinions=init_opinions,
                )
            elif net_type == "ba":
                hub_strategy = str(sig[8] if len(sig) > 8 else "default").strip().lower()
                hub_custom = str(sig[9] if len(sig) > 9 else "").strip()
                hub_assignment_mode = str(sig[10] if len(sig) > 10 else "early_position").strip().lower()
                if hub_assignment_mode not in {"early_position", "actual_hubs", "early_and_actual"}:
                    hub_assignment_mode = "early_position"
                if hub_strategy in {"default", "random"}:
                    hub_assignment_mode = "early_position"
                hub_priority_nodes = self._live_ba_hub_priority_nodes(
                    hub_strategy,
                    hub_custom,
                    init_opinions,
                    int(agent_count),
                    seed,
                )
                self._live_ba_hub_priority_set = set(int(x) for x in (hub_priority_nodes or []))
                self._live_ba_hub_assignment_mode = hub_assignment_mode
                try:
                    import inspect as _inspect
                    params = set(_inspect.signature(self._live_build_barabasi_albert).parameters.keys())
                except Exception:
                    params = set()
                try:
                    kwargs = dict(
                        num_agents=int(agent_count),
                        m_attach=int(sig[5]),
                        seed=seed,
                        homophily=hom,
                        opinions=init_opinions,
                    )
                    if "hub_assignment_mode" in params:
                        kwargs["hub_assignment_mode"] = hub_assignment_mode
                    if "hub_strategy" in params:
                        kwargs["hub_strategy"] = hub_strategy
                    if "hub_agent_ids" in params and hub_strategy == "custom":
                        kwargs["hub_agent_ids"] = hub_priority_nodes
                    elif "hub_priority_nodes" in params:
                        kwargs["hub_priority_nodes"] = hub_priority_nodes
                    neighbors = self._live_build_barabasi_albert(**kwargs)
                except TypeError:
                    # Compatibility with an older network_models.py. The graph still draws,
                    # but BA target-group assignment requires network_models support for target-group assignment.
                    neighbors = self._live_build_barabasi_albert(
                        num_agents=int(agent_count),
                        m_attach=int(sig[5]),
                        seed=seed,
                        homophily=hom,
                        opinions=init_opinions,
                    )
            else:
                self._live_ba_hub_priority_set = set()
                return False
            if net_type != "ba":
                self._live_ba_hub_priority_set = set()
                self._live_ba_hub_assignment_mode = "early_position"
            G = nx.Graph()
            for node in range(int(agent_count)):
                G.add_node(int(node))
            for src, nbrs in (neighbors or {}).items():
                for dst in (nbrs or set()):
                    G.add_edge(int(src), int(dst))
            if net_type == "ws":
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G, seed=seed)
            self._live_network_graph = G
            self._live_network_pos = pos
            self._live_network_sig = sig
            return True
        except Exception:
            self._live_network_graph = None
            self._live_network_pos = None
            self._live_network_sig = None
            self._live_ba_hub_priority_set = set()
            return False

    def _live_draw_distribution_placeholder(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self._live_ax.clear()
        xs = [-2, -1, 0, 1, 2]
        # Keep the distribution view plain and neutral.
        self._live_bars = self._live_ax.bar(xs, [0, 0, 0, 0, 0], edgecolor="#334155")
        self._live_ax.set_xticks(xs)
        self._live_ax.set_xticklabels([str(x) for x in xs])
        self._live_ax.set_xlabel("Opinion")
        self._live_ax.set_ylabel("Count")
        self._live_ax.set_title("Opinion distribution (no data yet)")
        self._live_vline = self._live_ax.axvline(0, linestyle="--", color="#94a3b8", linewidth=1.2)
        self._live_text = self._live_ax.text(
            0.02, 0.98, "",
            transform=self._live_ax.transAxes,
            va="top",
            ha="left",
        )

    def _live_draw_network_placeholder(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self._live_ax.clear()
        self._live_ax.set_title("Network view (no data yet)")
        self._live_ax.text(
            0.5, 0.5,
            "Attach a live CSV and run a networked experiment to view the network.",
            transform=self._live_ax.transAxes,
            ha="center",
            va="center",
            color="#cbd5e1",
        )
        self._live_ax.set_axis_off()

    def _live_init_plot(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not getattr(self, "_live_import_ok", False):
            return
        mode = str(getattr(self, "_live_view_mode", tk.StringVar(value="distribution")).get())
        if mode == "network" and self._live_network_selected():
            self._live_draw_network_placeholder()
        else:
            self._live_draw_distribution_placeholder()
        try:
            self._live_canvas.draw_idle()
        except Exception:
            pass

    def _live_browse_attach(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        path = filedialog.askopenfilename(
            title="Attach opinion-change CSV (network_opinion_change_*.csv)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self._live_attach(path)

    def _live_detach(self):
        # keep history, but stop reading from file
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            if self._live_poll_job is not None:
                self.after_cancel(self._live_poll_job)
        except Exception:
            pass
        self._live_poll_job = None

        try:
            if self._live_fh is not None:
                self._live_fh.close()
        except Exception:
            pass

        self._live_fh = None
        self._live_csv_path = None
        try:
            self._live_csv_var.set("")
            self._live_status_var.set("Detached.")
        except Exception:
            pass

    def _live_on_follow_toggle(self):
        # If user turns follow on, jump to latest immediately.
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            if bool(self._live_follow_var.get()):
                self._live_go_live()
        except Exception:
            pass

    def _live_go_live(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            self._live_follow_var.set(True)
        except Exception:
            pass

        if self._live_counts:
            last_idx = len(self._live_counts) - 1
            try:
                self._live_scale.configure(to=last_idx)
                self._live_scale.set(last_idx)
            except Exception:
                pass
            self._live_update_plot(last_idx)

    def _live_schedule_poll(self):
        # Polling is light: reads only newly appended lines.
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            if self._live_poll_job is not None:
                return
            self._live_poll_job = self.after(60, self._live_poll)
        except Exception:
            pass

    def _live_poll(self):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        self._live_poll_job = None

        # Reschedule early so UI stays responsive even if parsing is slow.
        try:
            self._live_poll_job = self.after(60, self._live_poll)
        except Exception:
            pass

        if not getattr(self, "_live_import_ok", False):
            return
        if not self._live_fh:
            return

        appended = 0
        try:
            while True:
                pos = self._live_fh.tell()
                line = self._live_fh.readline()
                if not line:
                    self._live_fh.seek(pos)
                    break
                line = line.strip()
                if not line:
                    continue
                # Skip header-like lines that might appear from overwrite.
                if line.lower().startswith("time_step"):
                    continue
                if self._live_ingest_csv_line(line):
                    appended += 1
                    try:
                        self._sync_waypoints_from_live_data()
                    except Exception:
                        pass
        except Exception:
            return

        if appended <= 0:
            return

        # Update slider range and plot depending on follow/pause
        last_idx = len(self._live_counts) - 1
        try:
            self._live_scale.configure(to=last_idx)
        except Exception:
            pass

        if bool(self._live_pause_var.get()):
            return

        if bool(self._live_follow_var.get()):
            try:
                self._live_scale.set(last_idx)
            except Exception:
                pass
            self._live_update_plot(last_idx)

    def _live_attach(self, csv_path: str):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not csv_path:
            return

        # Close existing handle but keep history only if same file; otherwise reset.
        if self._live_csv_path != csv_path:
            self._live_steps = []
            self._live_counts = []
            self._live_bdp = []
            self._live_vectors = []
            self._live_network_graph = None
            self._live_network_pos = None
            self._live_network_sig = None
            try:
                self._live_scale.configure(from_=0, to=0)
                self._live_scale.set(0)
            except Exception:
                pass

        # Close previous handle
        try:
            if self._live_fh is not None:
                self._live_fh.close()
        except Exception:
            pass

        self._live_csv_path = csv_path
        try:
            self._live_csv_var.set(csv_path)
        except Exception:
            pass

        try:
            fh = open(csv_path, "r", encoding="utf-8", errors="replace")
        except Exception as e:
            try:
                self._live_status_var.set(f"Could not open CSV: {e}")
            except Exception:
                pass
            return

        self._live_fh = fh

        # Read header and all existing rows.
        try:
            header = fh.readline()
            # If file starts with BOM or weirdness, normalize
            if header.startswith("\ufeff"):
                header = header.lstrip("\ufeff")
            if header:
                self._live_header = [h.strip() for h in header.strip().split(",") if h.strip()]
        except Exception:
            self._live_header = None

        # ingest any existing rows
        try:
            for raw in fh:
                line = raw.strip()
                if not line or line.lower().startswith("time_step"):
                    continue
                self._live_ingest_csv_line(line)
                try:
                    self._sync_waypoints_from_live_data()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self._live_status_var.set("Attached. Waiting for new steps…")
        except Exception:
            pass

        try:
            self._sync_waypoints_from_live_data()
        except Exception:
            pass

        # Update UI immediately
        if self._live_counts:
            last_idx = len(self._live_counts) - 1
            try:
                self._live_scale.configure(to=last_idx)
                if bool(self._live_follow_var.get()):
                    self._live_scale.set(last_idx)
                    self._live_update_plot(last_idx)
                else:
                    self._live_update_plot(int(self._live_scale.get()))
            except Exception:
                self._live_update_plot(last_idx)
        else:
            self._live_init_plot()

        self._live_schedule_poll()

    def _live_ingest_csv_line(self, line: str) -> bool:
        """
        Parse one CSV row: time_step,<agent1>,...,<agentN>
        We compute distribution counts + BDP without storing the full opinion vector.
        """
        try:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                return False
            step = int(float(parts[0]))
            # agent opinions (ints in [-2..2])
            vals = []
            for p in parts[1:]:
                if p == "":
                    continue
                vals.append(int(float(p)))
            if not vals:
                return False

            counts = [0, 0, 0, 0, 0]  # -2..2
            for v in vals:
                if v < -2:
                    v = -2
                elif v > 2:
                    v = 2
                counts[v + 2] += 1

            B, D, P = self._live_counts_to_bdp(counts)

            self._live_steps.append(step)
            self._live_counts.append(tuple(counts))
            self._live_bdp.append((B, D, P))
            self._live_vectors.append(tuple(vals))
            return True
        except Exception:
            return False

    def _live_counts_to_bdp(self, counts):
        # counts: [-2,-1,0,1,2]
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        N = float(sum(counts)) if counts else 0.0
        if N <= 0:
            return 0.0, 0.0, 0.0
        ex = (-2.0 * counts[0] + -1.0 * counts[1] + 0.0 * counts[2] + 1.0 * counts[3] + 2.0 * counts[4]) / N
        ex2 = (4.0 * counts[0] + 1.0 * counts[1] + 0.0 * counts[2] + 1.0 * counts[3] + 4.0 * counts[4]) / N
        var = max(0.0, ex2 - ex * ex)
        D = var ** 0.5
        denom = var + ex * ex
        P = 0.0 if denom == 0.0 else (var - ex * ex) / denom
        return ex, D, P

    def _live_step_by(self, delta: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not getattr(self, "_live_counts", None):
            return
        try:
            cur = int(float(self._live_scale.get()))
        except Exception:
            cur = len(self._live_counts) - 1
        new_idx = max(0, min(cur + int(delta), len(self._live_counts) - 1))
        try:
            self._live_follow_var.set(False)
            self._live_scale.set(new_idx)
        except Exception:
            pass
        self._live_update_plot(new_idx)

    def _live_keyboard_step(self, delta: int, event=None):
        # Avoid hijacking arrows while typing in text/entry widgets.
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            w = self.focus_get()
            cls = str(w.winfo_class()).lower() if w is not None else ""
            if any(tok in cls for tok in ("entry", "text", "spinbox", "combobox")):
                return
        except Exception:
            pass
        self._live_step_by(delta)

    def _live_on_slider(self):
        # User interaction => stop following unless they explicitly Go live.
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        try:
            self._live_follow_var.set(False)
        except Exception:
            pass
        idx = 0
        try:
            idx = int(float(self._live_scale.get()))
        except Exception:
            idx = 0
        self._live_update_plot(idx)

    def _live_update_plot(self, idx: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not getattr(self, "_live_import_ok", False):
            return
        if not self._live_counts:
            return

        idx = max(0, min(int(idx), len(self._live_counts) - 1))
        mode = str(getattr(self, "_live_view_mode", tk.StringVar(value="distribution")).get())
        if mode == "network" and self._live_network_selected():
            self._live_update_network_plot(idx)
            return

        counts = self._live_counts[idx]
        step = self._live_steps[idx]
        B, D, P = self._live_bdp[idx]

        try:
            # Always redraw the distribution view from scratch. This is more robust than
            # reusing bar artists after the axes have been cleared by the network view.
            self._live_ax.clear()
            xs = [-2, -1, 0, 1, 2]
            self._live_bars = self._live_ax.bar(xs, counts, edgecolor="#334155")
            self._live_vline = self._live_ax.axvline(B, linestyle="--", color="#94a3b8", linewidth=1.2)
            ymax = max(counts) if counts else 1
            ytop = max(1.0, float(ymax) * 1.15)
            self._live_ax.set_ylim(0, ytop)
            self._live_ax.set_xticks(xs)
            # Bottom axis shows only opinion values; left axis shows agent counts.
            self._live_ax.set_xticklabels([str(x) for x in xs])
            self._live_ax.set_xlabel("Opinion")
            self._live_ax.set_ylabel("Count")
            self._live_ax.set_title(f"Opinion distribution (step {step})")
            self._live_text = self._live_ax.text(
                0.02, 0.98, f"B={B:.3f}   D={D:.3f}   P={P:.3f}",
                transform=self._live_ax.transAxes,
                va="top",
                ha="left",
            )
        except Exception:
            pass

        try:
            self._live_step_label_var.set(str(step))
        except Exception:
            pass

        try:
            self._live_canvas.draw_idle()
        except Exception:
            pass

    def _live_update_network_plot(self, idx: int):
        """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
        if not self._live_vectors:
            self._live_draw_network_placeholder()
            try:
                self._live_canvas.draw_idle()
            except Exception:
                pass
            return

        idx = max(0, min(int(idx), len(self._live_vectors) - 1))
        opinions = [int(x) for x in self._live_vectors[idx]]
        step = self._live_steps[idx]
        counts = self._live_counts[idx]
        B, D, P = self._live_bdp[idx]
        agent_count = len(opinions)
        if not self._live_build_network_if_needed(agent_count):
            self._live_draw_network_placeholder()
            try:
                self._live_step_label_var.set(str(step))
                self._live_canvas.draw_idle()
            except Exception:
                pass
            return

        G = self._live_network_graph
        pos = self._live_network_pos
        nx = self._live_nx
        self._live_ax.clear()

        try:
            nx.draw_networkx_edges(G, pos, ax=self._live_ax, edge_color="#64748b", width=1.2, alpha=0.55)
        except Exception:
            pass

        node_colors = [self._live_opinion_color(opinions[node] if node < len(opinions) else 0) for node in G.nodes()]
        deg = dict(G.degree())
        deg_vals = [int(deg.get(node, 0)) for node in G.nodes()]
        dmin = min(deg_vals) if deg_vals else 0
        dmax = max(deg_vals) if deg_vals else 0
        net_type = ""
        try:
            net_type = str((self.network_type_var.get() or "")).strip().lower()
        except Exception:
            net_type = ""
        targeted_nodes = set(getattr(self, "_live_ba_hub_priority_set", set()) or set())
        hub_assignment_mode = str(getattr(self, "_live_ba_hub_assignment_mode", "early_position") or "early_position")

        def _live_scale_node_size(d: int) -> float:
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            if dmax <= dmin:
                return 720.0 if net_type == "ba" else 560.0
            if net_type == "ba":
                min_size, max_size = 420.0, 1700.0
            else:
                min_size, max_size = 460.0, 980.0
            return float(min_size + (float(d) - float(dmin)) * (max_size - min_size) / float(dmax - dmin))

        node_sizes = [_live_scale_node_size(int(deg.get(node, 0))) for node in G.nodes()]
        hub_cut = None
        if deg_vals:
            top_hub_idx = max(0, min(len(deg_vals) - 1, max(0, int(len(deg_vals) * 0.2) - 1)))
            hub_cut = sorted(deg_vals, reverse=True)[top_hub_idx]

        node_edgecolors = []
        node_linewidths = []
        for node in G.nodes():
            is_top_hub = hub_cut is not None and int(deg.get(node, 0)) >= int(hub_cut)
            is_targeted = int(node) in targeted_nodes
            if is_top_hub and is_targeted:
                node_edgecolors.append("#a855f7")  # purple = targeted and became a top hub
                node_linewidths.append(3.0)
            elif is_targeted:
                node_edgecolors.append("#22d3ee")  # cyan = selected for BA hub priority
                node_linewidths.append(2.5)
            elif is_top_hub:
                node_edgecolors.append("#facc15")  # gold = actual top-degree hub
                node_linewidths.append(2.2)
            else:
                node_edgecolors.append("#f8fafc")
                node_linewidths.append(1.0)

        try:
            nx.draw_networkx_nodes(
                G, pos, ax=self._live_ax,
                node_color=node_colors,
                edgecolors=node_edgecolors,
                linewidths=node_linewidths,
                node_size=node_sizes,
            )
        except Exception:
            pass

        if agent_count <= 14:
            try:
                labels = {int(node): str(int(node) + 1) for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, ax=self._live_ax, font_size=8, font_color="#0f172a")
            except Exception:
                pass

        title_suffix = ""
        if net_type == "ba":
            title_suffix = " • hub size ∝ degree"
        self._live_ax.set_title(f"Network view (step {step}){title_suffix}")
        legend = "-2 red   -1 orange   0 gray   +1 yellow   +2 green"
        degree_note = ""
        if net_type == "ba":
            max_deg = max(deg_vals) if deg_vals else 0
            target_n = len(targeted_nodes) if targeted_nodes else 0
            degree_note = f"\nNode size ∝ degree   gold=actual top hubs   cyan=BA target group   purple=both   mode={hub_assignment_mode}   max degree={max_deg}   targets={target_n}"
        self._live_ax.text(
            0.02,
            0.98,
            f"B={B:.3f}   D={D:.3f}   P={P:.3f}\n{legend}{degree_note}",
            transform=self._live_ax.transAxes,
            va="top",
            ha="left",
            color="#e5e7eb",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.75, edgecolor="#334155"),
        )
        try:
            self._live_step_label_var.set(str(step))
        except Exception:
            pass
        try:
            self._live_ax.set_axis_off()
        except Exception:
            pass
        try:
            self._live_canvas.draw_idle()
        except Exception:
            pass

    def _live_maybe_attach_from_log_line(self, line: str):
        """
        Auto-attach when the running script prints the opinion-change CSV path.
        Expected line:
          [INFO] Opinion-change CSV: <path>
        """
        try:
            m = re.search(r"Opinion-change CSV:\s*(.+)\s*$", line)
            if not m:
                return
            p = m.group(1).strip()
            if not p:
                return
            if self._live_csv_path == p:
                return
            self._live_attach(p)
        except Exception:
            return

    def on_stop(self):
        # Stop current run and cancel any pending batch queue.
        """Handle a Tkinter/UI event and update dependent controls."""
        self._batch_queue = []
        self._batch_active = False
        self._batch_total = 0
        self._batch_index = 0

        self.runner.stop()
        self.status_var.set("Termination requested.")
        self.append_log_threadsafe("\n[INFO] Termination requested.\n")


    def on_process_done(self, return_code):
        """Handle a Tkinter/UI event and update dependent controls."""
        def finish():
            """Helper used by this script to keep the experiment UI and analysis pipeline organized."""
            self._append_log(f"\n[DONE] Return code: {return_code}\n")

            # Continue batch if active and there are queued runs.
            if self._batch_active and self._batch_queue:
                # keep buttons as-is; start next run immediately
                self._append_log(f"[BATCH] Continuing… ({self._batch_index}/{self._batch_total} completed)\n")
                self._start_next_batch_run()
                return

            # Batch finished (or not in batch mode)
            self._batch_active = False
            self._batch_queue = [] 
            self._batch_total = 0
            self._batch_index = 0
 
            self.status_var.set("Ready.")
            self.run_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

        self.after(0, finish)



if __name__ == "__main__":
    LauncherUI().mainloop()
