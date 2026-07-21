import os
from os.path import join
from collections import defaultdict, Counter
import pandas as pd
import re
import csv
import time
import hashlib
import urllib.parse
import urllib.request
import html as html_lib
import argparse
from numpy.random import choice
import random
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prompts.opinion_dynamics.Flache_2017.content.fact_packs import FACT_PACKS
from prompts.opinion_dynamics.Flache_2017.content.world_rules import WORLD_RULES as EXTERNAL_WORLD_RULES, WORLD_LABELS
from network_models import build_small_world, build_erdos_renyi, build_barabasi_albert, choose_partner_scoring
from network_models import DirectedNetwork, evolve_once  # coevolving network, opt-in via --network_evolves
from network_models import assign_p_reach  # ADR-006 Component 3 (pluggable reach policy)
from step2_io import render_step2_template  # ADR-006 Component 2 (silence-as-choice) template renderer
import event_injection  # timed population-level shock, opt-in via --event_step
import run_naming  # shared mode-abbreviation + run-stem naming
import opinion_metrics  # shared B/D/P definitions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# ============================
# RETRY / DEBUG CONFIG
# ============================
# When strict format enforcement triggers retries, you can enable verbose visibility
# into the raw model outputs that failed validation. This does NOT affect agent decisions;
# it only improves observability and reproducibility of protocol compliance.
DEBUG_RETRY_OUTPUT = True         # fast mode default: keep console quieter during large runs
DEBUG_IO =True               # print receiver IO debug
DEBUG_RETRY_MAX_CHARS = 1200       # max chars printed per failed attempt
DEBUG_RETRY_SAVE_FILES = True     # default: disable heavy retry-debug file dumps for faster local runs
RETRY_DEBUG_DIR = True            # set at runtime (inside main) to the current run's csv_folder/_retry_debug
DEBUG_DUMP_NATIVE_OLLAMA_JSON = True   # default: skip raw /api/chat payload dumps for faster local runs
DEBUG_DISABLE_PROTOCOL_FALLBACKS = False  # default: use real protocol fallbacks instead of passing through invalid raw/sanitized output
NATIVE_OLLAMA_DEBUG_CONTEXT = {'step': '', 'agent_name': '', 'attempt': None, 'label': ''}
NATIVE_FAILURE_SHADOW_DEBUG_SEEN = set()


# ============================
# AGENT ERROR CSV (deprecated / disabled)
# ============================
# We no longer create the per-run *_network_agent_errors_*.csv file.
CURRENT_INTERACTION_STEP = None   # set each loop iteration to (t+1) in main()

# Run-level metrics used for quick diagnostics.
RUN_METRICS = Counter()
RUN_OUTPUT_DIR = None  # set in main(); used by the atexit metrics dump



def _metric_inc(key: str, n: int = 1):
    """
    Increment a run-level metric counter in RUN_METRICS.
    
        The simulation records many protocol, repair, RAG, and interaction diagnostics as sparse
        counters. This helper centralizes the "create if missing, then increment" pattern so new
        diagnostics can be added without pre-declaring every possible metric key.
    """
    try:
        RUN_METRICS[str(key)] += int(n)
    except Exception:
        pass


def _load_bian_scores_if_requested():
    """ADR-006 Component 4: if --include_bian_scores on, run (or reuse a
    cached) Bian 5-dim diagnostic for this run's model and return its scores
    dict, else None. Cached per model under ~/.claude_bian_cache/<model>.json;
    any failure is non-fatal (returns None with a printed note)."""
    if str(getattr(args, "include_bian_scores", "off")).strip().lower() != "on":
        return None
    try:
        import subprocess
        model = str(getattr(args, "model_name", "") or "")
        if not model:
            return None
        cache_dir = os.path.join(os.path.expanduser("~"), ".claude_bian_cache")
        os.makedirs(cache_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        cache_path = os.path.join(cache_dir, safe + ".json")
        if not os.path.exists(cache_path):
            tool = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools", "bian_diagnostic.py")
            print(f"[bian] running 5-dim diagnostic for {model} (one-time, cached)...")
            subprocess.run([sys.executable, tool, "--model", model, "--out", cache_path],
                           check=True, timeout=1800)
        with open(cache_path, encoding="utf-8") as _fh:
            return json.load(_fh).get("scores")
    except Exception as _e:
        print(f"[bian] diagnostic skipped ({_e})")
        return None


def _dump_run_metrics_json():
    """Write run-level RUN_METRICS counters to <output_dir>/metrics_<run>.json.

    Registered via atexit so it fires once at process end. Writes a NEW file
    only; it never touches existing result CSVs and never raises.
    """
    try:
        out_dir = globals().get('RUN_OUTPUT_DIR')
        if not out_dir:
            return
        run_id = globals().get('RUN_EXPORT_ID') or 'run'
        import os as _os, json as _json
        path = _os.path.join(out_dir, f"metrics_{run_id}.json")
        # forced holds = interactions where the listener did not genuinely evaluate
        # (pipeline artifact / skipped step3). Counted in the main loop as holds_total.
        _holds = int(RUN_METRICS.get("holds_total", 0))
        _levents = int(RUN_METRICS.get("listener_events_total", 0))
        # Self-document the methodology knobs so a result is always attributable to its
        # regime (validation strictness, update mode, ...) and never an invisible confound.
        try:
            _cfg = {
                "validation_strictness": globals().get("VALIDATION_STRICTNESS"),
                "allowed_update_mode": globals().get("ALLOWED_UPDATE_MODE"),
                "wrong_side_explanation_requery": globals().get("WRONG_SIDE_EXPL_REQUERY"),
                "deterministic": globals().get("DETERMINISTIC"),
                "model_name": str(getattr(args, "model_name", "")),
                "version_set": str(getattr(args, "version_set", "")),
                "world": str(getattr(args, "world", "")),
                "network_type": str(getattr(args, "network_type", "")),
                "num_agents": int(getattr(args, "num_agents", 0) or 0),
                "num_steps": int(getattr(args, "num_steps", 0) or 0),
                "max_step_change": getattr(args, "max_step_change", None),
                "distribution": str(getattr(args, "distribution", "")),
                "seed": getattr(args, "seed", None),
                # decoding regime (temperature etc. are experimental conditions too)
                "temperature": getattr(args, "temperature", None),
                "top_p": getattr(args, "top_p", None),
                "top_k": getattr(args, "top_k", None),
                "structured_output": globals().get("STRUCTURED_OUTPUT"),
                "rag_backend": str(getattr(args, "rag_backend", "")),
                "rag_content_mode": str(getattr(args, "rag_content_mode", "")),
                "rag_top_k": getattr(args, "rag_top_k", None),
                "fact_pack_mode": str(getattr(args, "fact_pack_mode", "")),
                "think_mode": str(getattr(args, "think_mode", "")),
                # ADR-006 Component 3: reach policy self-doc (feeds P21, P30).
                "p_reach_policy": str(getattr(args, "p_reach_policy", "uniform")),
                "p_reach_uniform_value": getattr(args, "p_reach_uniform_value", None),
                "p_reach_homophily_k": getattr(args, "p_reach_homophily_k", None),
                "p_reach_shadowban_fraction": getattr(args, "p_reach_shadowban_fraction", None),
                "p_reach_shadowban_value": getattr(args, "p_reach_shadowban_value", None),
                "p_reach_enforcement": str(getattr(args, "p_reach_enforcement", "filter")),
            }
        except Exception:
            _cfg = {}
        data = {"run": run_id,
                "config": _cfg,
                "summary": {"forced_holds": _holds, "listener_events": _levents,
                            "held_fraction": round(_holds / _levents, 4) if _levents else 0.0,
                            "closed_world_leak_step2": int(RUN_METRICS.get("closed_world_leak_step2", 0)),
                            "closed_world_leak_step3": int(RUN_METRICS.get("closed_world_leak_step3", 0)),
                            "closed_world_leak_fraction": round((int(RUN_METRICS.get("closed_world_leak_step2", 0)) + int(RUN_METRICS.get("closed_world_leak_step3", 0))) / (2 * _levents), 4) if _levents else 0.0},
                "metrics": dict(sorted(RUN_METRICS.items()))}
        _bian = globals().get("RUN_BIAN_SCORES")
        if _bian:
            data["bian_scores"] = _bian
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


try:
    import atexit as _atexit
    _atexit.register(_dump_run_metrics_json)
except Exception:
    pass


def _metric_inc_transition(pre_value: int, post_value: int, prefix: str = "transition"):
    """
    Increment the transition counter for a listener belief movement.
    
        Transition counters are stored as metric keys such as trans_-1_to_0. They are later exported
        with the run metrics and are used to reconstruct the 5x5 pre-belief by post-belief transition
        matrix without rereading the full step summary.
    """
    try:
        pre_i = int(pre_value)
        post_i = int(post_value)
    except Exception:
        return
    _metric_inc(f"{prefix}::{pre_i}->{post_i}")


def _metric_allowed_key(allowed_ratings) -> str:
    """
    Convert an allowed-rating set into a stable metric suffix.
    
        Step-3 validators often need to group outcomes by the legal rating set shown to the model.
        Sorting and joining the set here ensures that {0, 1} and {1, 0} produce the same exported
        key and can be compared across runs.
    """
    try:
        vals = [int(x) for x in (allowed_ratings or [])]
    except Exception:
        vals = []
    return "[" + ",".join(str(x) for x in vals) + "]"


def _speaker_stance_label(value: int | None) -> str:
    """
    Map a numeric speaker belief to the exposure-side label used in summaries.
    
        The exposure/persuasion CSV collapses individual ratings into support, oppose, and neutral
        speaker groups. This allows the analysis to ask whether support-side or oppose-side speakers
        were more frequent and more persuasive, independently of exact rating intensity.
    """
    try:
        v = int(value)
    except Exception:
        return "unknown"
    return "support" if v > 0 else ("oppose" if v < 0 else "uncertain")


def _listener_prev_label(value: str) -> str:
    """
    Map a listener's pre-interaction belief to a coarse diagnostic side.
    
        The label is used in artifact and movement summaries where exact ratings are too granular but
        the listener's prior side still matters, for example support-to-oppose versus same-side updates.
    """
    v = str(value or "none").strip().lower()
    return v if v in {"none", "read", "write"} else "other"


def _step_bucket_label(step_num: int) -> str:
    """
    Bucket an interaction index into early/middle/late windows for diagnostics.
    
        Some repair and movement rates are easier to interpret over coarse run phases than per-step.
        The bucket labels support time-local summaries without assuming a specific total run length.
    """
    try:
        t = int(step_num)
    except Exception:
        return "unknown"
    if t <= 10:
        return "01_10"
    if t <= 20:
        return "11_20"
    if t <= 30:
        return "21_30"
    if t <= 40:
        return "31_40"
    return "41_plus"


def _same_bin_softening_tag(pre_value: int, final_value: int, explanation_text: str) -> bool:
    """Detect only *bridge-worthy* softening inside the same mild bin.

    Mild hesitation alone should NOT force 1 -> 0 or -1 -> 0.
    This helper fires only when the explanation suggests that the previous mild
    lean no longer fits as the best label, i.e. a genuine move toward uncertainty.
    """
    try:
        pre_i = int(pre_value)
        final_i = int(final_value)
    except Exception:
        return False
    if pre_i != final_i or abs(pre_i) != 1:
        return False
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not s:
        return False
    bridge_worthy_patterns = [
        r"\b0 fits? better\b",
        r"\brating 0 fits? better\b",
        r"\bneutral fits? better\b",
        r"\bmove(?:d|s)? me toward 0\b",
        r"\bmove(?:d|s)? my rating toward 0\b",
        r"\bmove(?:d|s)? me to 0\b",
        r"\bmoved? to neutral\b",
        r"\bgenuinely unsure\b",
        r"\btruly unsure\b",
        r"\bboth sides seem (?:about )?equally plausible\b",
        r"\bequally plausible now\b",
        r"\bno longer clearly lean\b",
        r"\bno longer really lean\b",
        r"\bcan no longer say i lean\b",
        r"\bcan'?t really maintain my previous mild (?:support|agreement|opposition|disagreement)\b",
        r"\bmy previous mild (?:support|agreement|opposition|disagreement) no longer fits\b",
        r"\bmildly for(?: the claim)? no longer fits? best\b",
        r"\bmildly against(?: the claim)? no longer fits? best\b",
        r"\bmildly for(?: the claim)? no longer seems like the best fit\b",
        r"\bmildly against(?: the claim)? no longer seems like the best fit\b",
        r"\bthe current mild (?:support|opposition) no longer fits\b",
        r"\bthe current rating no longer fits as well\b",
        r"\bthe previous mild lean no longer fits\b",
        r"\bi'?m now genuinely unsure which side fits better\b",
        r"\bi'?m no longer clearly on that side\b",
        r"\bno longer clearly for the claim\b",
        r"\bno longer clearly against the claim\b",
    ]
    return any(re.search(p, s, flags=re.I) for p in bridge_worthy_patterns)

def _ensure_retry_debug_dir():
    """
    Create the retry-debug directory only when a debug artifact must be written.
    
        Most runs do not need per-attempt debug files. Delaying directory creation keeps normal output
        folders cleaner while still allowing failed or suspicious LLM turns to be inspected when debug
        saving is enabled.
    """
    global RETRY_DEBUG_DIR
    if not DEBUG_RETRY_SAVE_FILES:
        return None
    if not RETRY_DEBUG_DIR:
        return None
    os.makedirs(RETRY_DEBUG_DIR, exist_ok=True)
    return RETRY_DEBUG_DIR


def _set_native_ollama_debug_context(step: str = '', agent_name: str = '', attempt=None, label: str = ''):
    """
    Store contextual labels for the next native Ollama call.
    
        Native debug dumps need the current step, agent, stage, and file prefix in order to produce
        useful filenames. This helper sets that transient context before the low-level LLM call.
    """
    try:
        NATIVE_OLLAMA_DEBUG_CONTEXT['step'] = str(step or '')
        NATIVE_OLLAMA_DEBUG_CONTEXT['agent_name'] = str(agent_name or '')
        NATIVE_OLLAMA_DEBUG_CONTEXT['attempt'] = attempt
        NATIVE_OLLAMA_DEBUG_CONTEXT['label'] = str(label or '')
    except Exception:
        pass


def _clear_native_ollama_debug_context():
    """
    Clear the transient native-Ollama debug context after a call.
    
        The context is global because the low-level request function is shared by Step-2, Step-3, and
        true-open calls. Clearing it prevents a later unrelated call from inheriting stale debug labels.
    """
    _set_native_ollama_debug_context('', '', None, '')


def _debug_disable_protocol_fallbacks() -> bool:
    """
    Return whether deterministic protocol fallbacks are intentionally disabled.
    
        This is a debugging switch, not a normal experimental mode. It lets a developer observe raw
        model failures instead of immediately replacing them with safe fallback outputs.
    """
    try:
        return bool(DEBUG_DISABLE_PROTOCOL_FALLBACKS and DEBUG_RETRY_SAVE_FILES)
    except Exception:
        return False


def _maybe_skip_protocol_fallback(step: str, fallback_text: str, raw_text: str = '', sanitized_text: str = '') -> str:
    """
    Gate a deterministic fallback when raw-failure debugging is enabled.
    
        Normal experiments should keep fallbacks on so CSVs remain well-formed. When debugging model
        behavior, this helper can bypass the fallback path and return the original invalid output for
        inspection.
    """
    if not _debug_disable_protocol_fallbacks():
        return str(fallback_text or '')
    passthrough = str(sanitized_text or raw_text or '').strip()
    try:
        print(f"[debug][{step}] protocol fallback disabled; returning passthrough invalid output instead of fabricated fallback")
    except Exception:
        pass
    return passthrough


def _render_effective_request_messages(request_payload) -> str:
    """
    Render the exact chat messages that will be sent to the LLM.
    
        The function is used only for diagnostics. It serializes system, history, and current prompt
        messages so prompt duplication, hidden history leakage, or missing qwen control tokens can be
        inspected from debug files.
    """
    try:
        msgs = list((request_payload or {}).get('messages') or [])
    except Exception:
        msgs = []
    if not msgs:
        return ''
    rendered = []
    for idx, msg in enumerate(msgs, start=1):
        try:
            role = str((msg or {}).get('role') or f'message_{idx}').strip() or f'message_{idx}'
            content = _clean_debug_prompt_text((msg or {}).get('content') or '')
        except Exception:
            role = f'message_{idx}'
            content = ''
        rendered.append(f"----- {role.upper()} MESSAGE {idx} -----")
        rendered.append(content or '<EMPTY>')
    return "\n\n".join(rendered).strip()


def _dump_native_ollama_payload_debug(prompt_text: str, request_payload, response_payload, extracted_text: str = '', extracted_thinking: str = '', note: str = ''):
    """
    Write native Ollama request/response payloads for failed or inspected calls.
    
        The native path exposes information that the LangChain wrapper may hide, including qwen
        thinking fields and done reasons. This dump is useful when qwen returns empty content, length
        truncation, or protocol-breaking output.
    """
    d = _ensure_retry_debug_dir()
    if not d or not DEBUG_DUMP_NATIVE_OLLAMA_JSON:
        return None
    ctx = dict(NATIVE_OLLAMA_DEBUG_CONTEXT or {})
    step = _safe_file_token(ctx.get('step') or 'native')
    agent = _safe_file_token(ctx.get('agent_name') or 'unknown_agent')
    label = _safe_file_token(ctx.get('label') or 'payload')
    attempt = ctx.get('attempt')
    ts = int(time.time() * 1000)
    run_prefix = _safe_file_token(REPAIR_LOG_RUN_LABEL) if REPAIR_LOG_RUN_LABEL else 'run'
    fname = f"{run_prefix}_{agent}_{step}_native_{label}_attempt{attempt if attempt is not None else 'na'}_{ts}.txt"
    fpath = os.path.join(d, fname)
    try:
        import json as _json
        effective_prompt = _render_effective_request_messages(request_payload)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(f"STEP: {ctx.get('step','')}\n")
            f.write(f"AGENT: {ctx.get('agent_name','')}\n")
            f.write(f"ATTEMPT: {'' if attempt is None else attempt}\n")
            f.write(f"LABEL: {ctx.get('label','')}\n")
            if note:
                f.write(f"NOTE: {note}\n")
            f.write('\n===== USER PROMPT TEXT (debug arg) =====\n')
            f.write(_clean_debug_prompt_text(prompt_text))
            f.write('\n\n===== EFFECTIVE REQUEST MESSAGES =====\n')
            f.write(effective_prompt or '<UNAVAILABLE>')
            f.write('\n\n===== REQUEST PAYLOAD =====\n')
            f.write(_json.dumps(request_payload, ensure_ascii=False, indent=2, default=str))
            f.write('\n\n===== RESPONSE PAYLOAD =====\n')
            f.write(_json.dumps(response_payload, ensure_ascii=False, indent=2, default=str))
            f.write('\n\n===== EXTRACTED FINAL TEXT =====\n')
            f.write(str(extracted_text or ''))
            f.write('\n\n===== EXTRACTED THINKING =====\n')
            f.write(str(extracted_thinking or ''))
        return fpath
    except Exception as e:
        try:
            print(f"[warn] could not save native Ollama payload debug: {e}")
        except Exception:
            pass
        return None

def _clean_debug_prompt_text(prompt_text: str) -> str:
    """Remove bulky prompt fragments before saving compact debug text."""
    s = str(prompt_text or "")
    if not s:
        return s
    try:
        s = _compact_qwen_prompt_text(s)
    except Exception:
        pass
    # Remove repeated blank lines and duplicated scaffolding that only adds noise in debug dumps.
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

HARD_FAILURE_TOKENS = [
    'empty_final_text', 'empty_only_retry', 'empty_retry_failed', 'step3_empty_retry_failed',
    'empty/broken tweet body', 'broken tweet body', 'empty explanation', 'missing explanation',
    'missing_final_rating', 'missing final rating', 'refusal/meta output', 'not_in_allowed_set',
    'invalid_input_tweet_skip_step3', 'no usable tweet content',
]


def _reason_blob(reasons) -> str:
    """Join validation reasons into a single searchable diagnostic string."""
    if isinstance(reasons, (list, tuple)):
        return " | ".join(str(r or "") for r in reasons).lower()
    return str(reasons or "").lower()


def _is_hard_failure(reasons) -> bool:
    """Classify validation reasons that should be treated as hard protocol failures."""
    low = _reason_blob(reasons)
    return any(tok in low for tok in HARD_FAILURE_TOKENS)


def _should_emit_error_debug(event_type: str, reasons) -> bool:
    """Control when retry-debug artifacts are emitted.

    Policy:
    - debug_native_thinking_on_fail = on  -> emit all debug artifacts for protocol/error/style events
    - debug_native_thinking_on_fail = off -> emit only hard/structural failures
    """
    et = str(event_type or '').strip().lower()
    debug_event_types = {
        'warning', 'retry', 'fallback', 'validation', 'native_rescue',
        'rewrite', 'repair', 'soft_cleanup'
    }
    if et not in debug_event_types:
        return False
    if DEBUG_NATIVE_THINKING_ON_FAIL:
        return True
    return _is_hard_failure(reasons)



def _should_capture_native_thinking_on_fail(event_type: str, reasons) -> bool:
    """Control expensive native shadow-debug calls.

    Policy:
    - debug_native_thinking_on_fail = on  -> capture native thinking for all emitted debug/error events
    - debug_native_thinking_on_fail = off -> capture none
    """
    if not DEBUG_NATIVE_THINKING_ON_FAIL:
        return False
    et = str(event_type or '').strip().lower()
    return et in {"warning", "retry", "fallback", "validation", "native_rescue", "rewrite", "repair", "soft_cleanup"}

def _infer_shadow_include_history(attempt) -> bool:
    """Infer whether a shadow debug call should include conversation history."""
    a = str(attempt or '').strip().lower()
    return a not in {'1b', '1c', '2', '2b', '2c'}


def _capture_native_thinking_shadow_debug(conversation, *, step: str, event_type: str, agent_name: str = None, attempt=None, prompt_text: str = '', include_history: bool = True):
    """Capture qwen thinking text for failed native calls without changing the simulation state."""
    if conversation is None or not prompt_text:
        return None
    key_blob = f"{step}|{event_type}|{agent_name or ''}|{attempt}|{hashlib.sha1(str(prompt_text).encode('utf-8', errors='replace')).hexdigest()}"
    if key_blob in NATIVE_FAILURE_SHADOW_DEBUG_SEEN:
        return None
    NATIVE_FAILURE_SHADOW_DEBUG_SEEN.add(key_blob)
    try:
        import requests
        model, options, think = _collect_native_ollama_options(conversation)
        base_url = _get_native_ollama_endpoint(conversation)
        url = base_url.rstrip('/') + '/api/chat'
        prompt_to_send = str(prompt_text or '')
        step_ctx = str(step or '').strip().lower()
        if _is_qwen_family_model(model):
            prompt_to_send = _compact_qwen_prompt_text(prompt_to_send, step=step_ctx)
            think = _apply_prompt_control_think_override(prompt_to_send, model, think)
        messages = _collect_native_ollama_messages(conversation, prompt_to_send, include_history=include_history)
        payload = {'model': model, 'messages': messages, 'stream': False}
        if think is not None:
            payload['think'] = think
        if options:
            payload['options'] = dict(options)
        prev_ctx = dict(NATIVE_OLLAMA_DEBUG_CONTEXT or {})
        _set_native_ollama_debug_context(step=step, agent_name=agent_name, attempt=attempt, label=f'shadow_{event_type}')
        try:
            resp = requests.post(url, json=payload, headers=_native_ollama_request_headers(base_url), timeout=(20, 180))
            resp.raise_for_status()
            data = resp.json()
            text = _extract_native_ollama_text(data)
            thinking = _extract_native_ollama_thinking(data)
            note = f'shadow_debug_{event_type}'
            _dump_native_ollama_payload_debug(
                prompt_text=prompt_to_send,
                request_payload=payload,
                response_payload=data,
                extracted_text=text,
                extracted_thinking=thinking,
                note=note,
            )
        finally:
            _set_native_ollama_debug_context(
                step=prev_ctx.get('step', ''),
                agent_name=prev_ctx.get('agent_name', ''),
                attempt=prev_ctx.get('attempt'),
                label=prev_ctx.get('label', ''),
            )
    except Exception as e:
        try:
            d = _ensure_retry_debug_dir()
            if d:
                ts = int(time.time() * 1000)
                fname = f"{_safe_file_token(agent_name or 'unknown_agent')}_{_safe_file_token(step)}_shadow_{_safe_file_token(event_type)}_{ts}.txt"
                with open(os.path.join(d, fname), 'w', encoding='utf-8') as f:
                    f.write(f"STEP: {step}\nEVENT_TYPE: {event_type}\nAGENT: {agent_name or ''}\nATTEMPT: {attempt if attempt is not None else ''}\n")
                    f.write(f"NOTE: native shadow debug failed: {e}\n\n===== USER PROMPT TEXT =====\n{_clean_debug_prompt_text(prompt_text)}\n")
        except Exception:
            pass
        return None


def _dump_retry_debug_event(step: str, event_type: str, agent_name: str = None, attempt: int = None, reasons=None, prompt_text: str = '', raw_text: str = '', sanitized_text: str = '', final_text: str = '', conversation=None, include_history: bool | None = None):
    """
    Save one retry, warning, repair, or fallback event into the debug folder.
    
        These files are not part of the primary dataset. They are developer-facing artifacts used to
        inspect why validation rejected an output, what prompt was retried, and what final text entered
        the CSV pipeline.
    """
    if not _should_emit_error_debug(event_type, reasons):
        return None
    d = _ensure_retry_debug_dir()
    if not d:
        return None
    name = agent_name or 'unknown_agent'
    reason_str = ", ".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons or '')
    ts = int(time.time() * 1000)
    run_prefix = _safe_file_token(REPAIR_LOG_RUN_LABEL) if REPAIR_LOG_RUN_LABEL else 'run'
    agent_prefix = _safe_file_token(name)
    fname = f"{run_prefix}_{agent_prefix}_{step}_{event_type}_attempt{attempt if attempt is not None else 'na'}_{ts}.txt"
    fpath = os.path.join(d, fname)
    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(f"STEP: {step}\n")
            f.write(f"EVENT_TYPE: {event_type}\n")
            f.write(f"AGENT: {name}\n")
            f.write(f"ATTEMPT: {'' if attempt is None else attempt}\n")
            f.write(f"REASONS: {reason_str}\n")
            f.write('\n===== USER PROMPT TEXT =====\n')
            f.write(_clean_debug_prompt_text(prompt_text))
            f.write('\n\n===== RAW OUTPUT =====\n')
            f.write(raw_text or '')
            f.write('\n\n===== SANITIZED OUTPUT =====\n')
            f.write(sanitized_text or '')
            f.write('\n\n===== FINAL OUTPUT =====\n')
            f.write(final_text or '')
        if conversation is not None and _should_capture_native_thinking_on_fail(event_type, reasons):
            if include_history is None:
                include_history = _infer_shadow_include_history(attempt)
            _capture_native_thinking_shadow_debug(conversation, step=step, event_type=event_type, agent_name=name, attempt=attempt, prompt_text=prompt_text, include_history=bool(include_history))
        return fpath
    except Exception as e:
        try:
            print(f"[warn] could not save retry debug event: {e}")
        except Exception:
            pass
        return None

REPAIR_LOG_CSV_PATH = None  # set at runtime to csv_folder/_retry_debug/repair_events.csv
REPAIR_LOG_RUN_LABEL = ''  # set at runtime for unique repair filenames
RAG_RETRIEVAL_LOG_CSV_PATH = None  # created only for RAG runs
RUN_EXPORT_ID = ''  # set at runtime for stable per-row/per-log run id
WEB_EVENTS_LOG_CSV_PATH = None  # created only when live web is enabled for the run
NATIVE_EVENTS_LOG_CSV_PATH = None  # always created under _retry_debug for native/Ollama diagnostics


def _rag_logging_enabled() -> bool:
    """
    Return whether local RAG retrieval decisions should be logged.
    
        Retrieval logs are important for auditing which snippets were shown to the model. They can be
        disabled for speed or cleaner output when retrieval provenance is not being inspected.
    """
    return _rag_enabled_for_world(getattr(args, "world", "closed")) and str(getattr(args, "rag_backend", "off")).strip().lower() != "off"


def _ensure_rag_retrieval_log_file():
    """
    Create the RAG retrieval-event CSV with a stable header.
    
        The retrieval log records query text, selected snippet IDs, snippet directions, and balance
        shortfalls. It is the main audit trail for confirming that supportive, criticism, or balanced
        RAG modes injected the expected evidence.
    """
    global RAG_RETRIEVAL_LOG_CSV_PATH
    if not _rag_logging_enabled():
        return None
    d = _ensure_retry_debug_dir()
    if not d:
        return None
    if not RAG_RETRIEVAL_LOG_CSV_PATH:
        run_tok = _safe_file_token(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or 'run')
        RAG_RETRIEVAL_LOG_CSV_PATH = os.path.join(d, f'rag_retrieval_{run_tok}.csv')
    try:
        if not os.path.exists(RAG_RETRIEVAL_LOG_CSV_PATH):
            with open(RAG_RETRIEVAL_LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'run_id',
                    'time_step','prompt_step','prompt_variant','agent_name','version_set','topic_key',
                    'query_mode','content_mode','effective_top_k','query_text',
                    'retrieved_ids','retrieved_directions','rendered_sections','retrieved_texts','retriever'
                ])
        return RAG_RETRIEVAL_LOG_CSV_PATH
    except Exception:
        return None


def _rag_render_section_labels(rows: list[dict], content_mode: str = 'full') -> list[str]:
    """
    Choose prompt section labels for retrieved snippets by evidence direction.
    
        Clear labels help the model distinguish claim-supporting, claim-challenging, and contextual
        material. Keeping this mapping centralized prevents Step-2 and Step-3 from using inconsistent
        labels for the same content mode.
    """
    cleaned_rows = [r for r in (rows or []) if str(r.get('text', '')).strip()]
    if not cleaned_rows:
        return []
    mode = str(content_mode or 'full').strip().lower()
    if mode == 'criticism_only':
        return ['claim-challenging']
    if mode == 'supportive_only':
        return ['claim-supporting']
    if mode == 'context_only':
        return ['contextual']

    labels = []
    directions = [_rag_row_direction(r) for r in cleaned_rows]
    if 'supportive' in directions:
        labels.append('claim-supporting')
    if 'criticism' in directions:
        labels.append('claim-challenging')
    if 'context' in directions:
        labels.append('contextual')
    if 'unknown' in directions:
        labels.append('mixed-direction')
    return labels


def _log_rag_retrieval(step_kind: str, prompt_variant: str, agent_name: str, query_mode: str, query_text: str, rows: list[dict], effective_top_k: int):
    """
    Append one local-RAG retrieval event to the provenance CSV.
    
        The row captures the query mode, content mode, selected chunks, and any balanced-mode shortfall.
        These records are used to verify that a run actually received the intended evidence environment.
    """
    csv_path = _ensure_rag_retrieval_log_file()
    if not csv_path:
        return
    try:
        content_mode = str(getattr(args, 'rag_content_mode', 'full') or 'full')
        directions = '|'.join(_rag_row_direction(r) for r in (rows or []))
        sections = '|'.join(_rag_render_section_labels(rows, content_mode=content_mode))
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                str(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or ''),
                CURRENT_INTERACTION_STEP if CURRENT_INTERACTION_STEP is not None else '',
                str(step_kind or ''),
                str(prompt_variant or ''),
                str(agent_name or ''),
                str(getattr(args, 'version_set', '') or ''),
                str(_active_rag_topic_key() or ''),
                str(query_mode or ''),
                content_mode,
                int(effective_top_k),
                str(query_text or ''),
                '|'.join(str(r.get('id','')) for r in (rows or [])),
                directions,
                sections,
                ' || '.join(str(r.get('text','')).replace('\n',' ').strip() for r in (rows or [])),
                str(getattr(args, 'rag_backend', '') or ''),
            ])
    except Exception:
        pass


def _safe_file_token(s: str) -> str:
    """Sanitize arbitrary text so it can be safely used inside filenames."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
    s = s.strip("._")
    return s or "run"

SOFT_REPAIR_TAGS = {
    'step3_hidden_state_explanation_rewrite',
    'step3_neutral_balance_soft_cleanup',
}

NON_REPAIR_TAGS = {
    'singleton_fastpath',
    'same_rating_skip_tweet_local',
    'same_rating_skip_generic',
    'singleton_skip_generic',
    # Pipeline/input artifact: no Step-3 LLM judgment happened. Keep it visible in
    # interaction-quality/native diagnostics, but do not count it as a semantic repair.
    'invalid_input_tweet_skip_step3',
}
NON_REPAIR_TAG_PREFIXES = {
    'step3_mode:',
    'step3_skip_reason:',
    'step3_explanation_source:',
}


def _split_repair_tags(tags):
    """
    Normalize a repair-tag field into individual tag tokens.
    
        Repair tags may arrive as comma/pipe/semicolon-separated strings depending on the code path.
        This helper gives metrics and summaries one consistent representation.
    """
    tags = [str(t) for t in (tags or []) if str(t).strip()]
    soft, hard, neutral = [], [], []
    for t in tags:
        if t in SOFT_REPAIR_TAGS:
            soft.append(t)
        elif t in NON_REPAIR_TAGS or any(t.startswith(pref) for pref in NON_REPAIR_TAG_PREFIXES):
            neutral.append(t)
        else:
            hard.append(t)
    return {'soft': soft, 'hard': hard, 'neutral': neutral}




def _ensure_repair_log_file():
    """
    Create the Step-3 repair-event CSV before the first repair is written.
    
        Step-3 repairs are potential interpretive confounds. A dedicated log preserves pre/post beliefs,
        raw model output, final output, allowed ratings, and repair tags for later manual review.
    """
    global REPAIR_LOG_CSV_PATH
    d = _ensure_retry_debug_dir()
    if not d:
        return None
    if not REPAIR_LOG_CSV_PATH:
        REPAIR_LOG_CSV_PATH = os.path.join(d, 'repair_events.csv')
    try:
        if not os.path.exists(REPAIR_LOG_CSV_PATH):
            with open(REPAIR_LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'Run ID',
                    'Time Step',
                    'Prompt Step',
                    'Agent Name',
                    'Repair Tags',
                    'Soft Cleanup Tags',
                    'Pre Belief',
                    'Speaker Belief',
                    'Allowed Ratings',
                    'Final Rating',
                    'Raw Output Attempt 1',
                    'Raw Output Attempt 2',
                    'Final Output',
                ])
        return REPAIR_LOG_CSV_PATH
    except Exception as e:
        try:
            print(f"[warn] could not initialize repair log CSV: {e}")
        except Exception:
            pass
        return None

def _dump_repair_event(step: str, agent_name: str, repair_tags, raw_text_attempt_1: str, raw_text_attempt_2: str, final_text: str, pre_belief=None, speaker_belief=None, allowed_ratings=None):
    """
    Record one Step-3 repair or fallback event with full local context.
    
        The repair log is separate from the step summary because it keeps more diagnostic detail than a
        normal interaction row should carry. It is used to determine whether qwen/protocol artifacts are
        rare noise or structured by condition, persona, stance, or allowed set.
    """
    tags = list(repair_tags or [])
    if not tags:
        return

    tag_groups = _split_repair_tags(tags)
    hard_tags = list(tag_groups.get('hard', []))
    soft_tags = list(tag_groups.get('soft', []))

    if soft_tags:
        _metric_inc('soft_cleanup_events_total')
        _metric_inc(f"{str(step).lower()}_soft_cleanup_events")
        for tag in soft_tags:
            _metric_inc(f"soft_cleanup_tag::{str(tag)}")

    emit_soft_only_debug = bool(soft_tags) and bool(DEBUG_NATIVE_THINKING_ON_FAIL)
    if not hard_tags and not emit_soft_only_debug:
        return

    if hard_tags:
        _metric_inc('repair_events_total')
        _metric_inc(f"{str(step).lower()}_repair_events")
        for tag in hard_tags:
            _metric_inc(f"repair_tag::{str(tag)}")
    ts_step = CURRENT_INTERACTION_STEP if CURRENT_INTERACTION_STEP is not None else -1
    name = agent_name or 'unknown_agent'
    tags_str = '|'.join(str(t) for t in hard_tags)
    soft_tags_str = '|'.join(str(t) for t in soft_tags)
    debug_event_type = 'repair' if hard_tags else 'soft_cleanup'
    final_rating = None
    try:
        final_rating = extract_belief(final_text)
    except Exception:
        final_rating = None

    csv_path = _ensure_repair_log_file()
    if csv_path:
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    str(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or ''),
                    ts_step,
                    str(step),
                    str(name),
                    tags_str,
                    soft_tags_str,
                    '' if pre_belief is None else int(pre_belief),
                    '' if speaker_belief is None else int(speaker_belief),
                    '' if allowed_ratings is None else str(list(allowed_ratings)),
                    '' if final_rating is None else int(final_rating),
                    str(raw_text_attempt_1 or ''),
                    str(raw_text_attempt_2 or ''),
                    str(final_text or ''),
                ])
        except Exception as e:
            try:
                print(f"[warn] could not append repair event: {e}")
            except Exception:
                pass

    d = _ensure_retry_debug_dir()
    if d:
        ts = int(time.time() * 1000)
        run_prefix = _safe_file_token(REPAIR_LOG_RUN_LABEL) if REPAIR_LOG_RUN_LABEL else "run"
        agent_prefix = _safe_file_token(name)
        fname = f"{run_prefix}_{agent_prefix}_{step}_{debug_event_type}_{ts}.txt"
        fpath = os.path.join(d, fname)
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(f"TIME_STEP: {ts_step}\n")
                f.write(f"STEP: {step}\n")
                f.write(f"EVENT_TYPE: {debug_event_type}\n")
                f.write(f"AGENT: {name}\n")
                f.write(f"REPAIR_TAGS: {tags_str}\n")
                if soft_tags_str:
                    f.write(f"SOFT_CLEANUP_TAGS: {soft_tags_str}\n")
                if pre_belief is not None:
                    f.write(f"PRE_BELIEF: {int(pre_belief)}\n")
                if speaker_belief is not None:
                    f.write(f"SPEAKER_BELIEF: {int(speaker_belief)}\n")
                if allowed_ratings is not None:
                    f.write(f"ALLOWED_RATINGS: {list(allowed_ratings)}\n")
                if final_rating is not None:
                    f.write(f"FINAL_RATING: {int(final_rating)}\n")
                f.write("\n===== RAW OUTPUT ATTEMPT 1 =====\n")
                f.write(str(raw_text_attempt_1 or ''))
                f.write("\n\n===== RAW OUTPUT ATTEMPT 2 =====\n")
                f.write(str(raw_text_attempt_2 or ''))
                f.write("\n\n===== FINAL OUTPUT =====\n")
                f.write(str(final_text or ''))
        except Exception as e:
            try:
                print(f"[warn] could not save repair event file: {e}")
            except Exception:
                pass

def _compute_bdp_from_beliefs(beliefs):
    """
    Compute the population-level B, D, and P metrics from current beliefs.
    
        B is the mean belief, D is the population standard deviation, and P is the bipolarization
        share 4*pos*neg. The definitions live in scripts/opinion_metrics.py so that the simulator,
        the plots and the eval tools cannot drift apart.
    """
    return opinion_metrics.bdp(beliefs)


def _unanimous_belief(list_agents) -> int | None:
    """Return the unanimous belief value if all agents agree, otherwise return None."""
    try:
        vals = [int(getattr(a, 'current_belief')) for a in (list_agents or [])]
    except Exception:
        return None
    if not vals:
        return None
    first = int(vals[0])
    return first if all(int(v) == first for v in vals) else None


def _write_step_summary_csv(step_summary_path: str, rows):
    """
    Write the per-interaction CSV that serves as the main analysis table.
    
        Each row corresponds to one listener update attempt and includes pre/post beliefs, deltas,
        allowed ratings, speaker/listener identity, persona fields, network metadata, and repair tags.
        Most downstream analyses begin from this file.
    """
    if not step_summary_path:
        return
    try:
        os.makedirs(os.path.dirname(step_summary_path), exist_ok=True)
        df = pd.DataFrame(list(rows or []))
        if not df.empty and 'time_step' in df.columns:
            df = df.sort_values('time_step').reset_index(drop=True)
        df.to_csv(step_summary_path, index=False)
        try:
            print(f"[INFO] Step-summary CSV: {step_summary_path}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[warn] could not write step summary CSV: {e}")
        except Exception:
            pass


def _write_step2_event_summary_csv(step2_event_path: str, rows):
    """
    Write Step-2 tweet-generation diagnostics to CSV.
    
        This file records whether the speaker output was clean, repaired, warned, or salvaged before
        handoff. It helps separate tweet-production artifacts from listener-update artifacts.
    """
    if not step2_event_path:
        return
    try:
        os.makedirs(os.path.dirname(step2_event_path), exist_ok=True)
        df = pd.DataFrame(list(rows or []))
        if not df.empty and 'time_step' in df.columns:
            df = df.sort_values(['time_step', 'speaker_agent_id', 'listener_agent_id'], na_position='last').reset_index(drop=True)
        df.to_csv(step2_event_path, index=False)
        try:
            print(f"[INFO] Step-2 event CSV: {step2_event_path}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[warn] could not write step-2 event CSV: {e}")
        except Exception:
            pass


def _reason_tag_token(reason: str) -> str:
    """
    Normalize a validation or warning reason into a compact metric tag.
    
        Free-form validator messages are hard to aggregate directly. This helper converts them to a
        predictable token suitable for metric keys and CSV tag fields.
    """
    s = re.sub(r'[^a-z0-9]+', '_', str(reason or '').strip().lower()).strip('_')
    return s[:120] or 'unknown'


def _set_step2_call_info(conversation, **kwargs):
    """
    Store the current Step-2 call context for nested validators and loggers.
    
        Some Step-2 helpers are called without explicit step/agent arguments. This shared context lets
        them attribute warnings and debug events to the correct speaker turn.
    """
    try:
        setattr(conversation, "_last_step2_call_info", dict(kwargs))
    except Exception:
        pass



def _write_agent_summary_csv(agent_summary_path: str, rows):
    """
    Write one aggregate row per agent after the run finishes.
    
        The exported fields combine final belief, total movement, move counts, speaking/listening counts,
        persona fields, and network metrics. This table is used to connect individual trajectories to
        persona and topology.
    """
    if not agent_summary_path:
        return
    try:
        os.makedirs(os.path.dirname(agent_summary_path), exist_ok=True)
        df = pd.DataFrame(list(rows or []))
        if not df.empty and 'agent_id' in df.columns:
            df = df.sort_values('agent_id').reset_index(drop=True)
        df.to_csv(agent_summary_path, index=False)
        try:
            print(f"[INFO] Agent-summary CSV: {agent_summary_path}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[warn] could not write agent summary CSV: {e}")
        except Exception:
            pass


def _csv_clean_value(v):
    """Normalize Python values before writing them into CSV cells."""
    if v is None:
        return ""
    try:
        import pandas as _pd
        if _pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def _agent_persona_fields(agent, prefix: str = "") -> dict:
    """
    Collect persona attributes from an Agent for CSV export.
    
        The simulator keeps persona data both as prompt text and structured fields. This helper exports
        the structured version so belief movement can be grouped by epistemic profile, trust, openness,
        demographic fields, or topic-specific traits.
    """
    out = {}
    fields = [
        'epistemic_profile', 'institutional_trust', 'uncertainty_tolerance',
        'evidence_style', 'official_narrative_suspicion', 'openness_to_update',
        'value_orientation', 'agency_vs_fatalism', 'conflict_style',  # social_conformity retired (was inert: never produced a validator permission)
        'age', 'gender', 'ethnicity', 'education', 'occupation',
        'political_leaning', 'early_life',
        # New structured persona-profile fields from the launcher. Keep these
        # separate from the legacy aliases so plots can analyze the exact UI
        # variable that varied in the run.
        'education_level', 'training_style', 'domain_familiarity',
        'topic_interest', 'prior_exposure', 'age_group', 'flavor_gender',
        'flavor_ethnicity', 'lifestyle_notes', 'tone_hint'
    ]
    for f in fields:
        out[f"{prefix}{f}"] = _csv_clean_value(getattr(agent, f, ""))
    return out





def _belief_side_label(value) -> str:
    """Map a numeric belief to negative, neutral, or positive side labels."""
    try:
        v = int(value)
    except Exception:
        return "unknown"
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def _normalize_ba_hub_assignment_mode(mode: str = "early_position") -> str:
    """
    Normalize the selected BA hub-assignment mode across UI, CLI, and network code.
    
        Several historical names refer to the same semantics. This function keeps the runtime tolerant
        of older launcher presets while ensuring metrics and output filenames use one canonical value.
    """
    s = str(mode or "early_position").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "early_position",
        "current": "early_position",
        "early": "early_position",
        "early_positions": "early_position",
        "early_position_priority": "early_position",
        "probabilistic": "early_position",
        "odds": "early_position",
        "actual": "actual_hubs",
        "actual_hub": "actual_hubs",
        "actual_hubs": "actual_hubs",
        "top_hub": "actual_hubs",
        "top_hubs": "actual_hubs",
        "highest_degree": "actual_hubs",
        "realized_hubs": "actual_hubs",
        "both": "early_and_actual",
        "early_actual": "early_and_actual",
        "early_and_actual": "early_and_actual",
        "early_plus_actual": "early_and_actual",
    }
    return aliases.get(s, "early_position")


def _parse_ba_hub_custom_indices(custom_text: str, list_agents) -> list[int]:
    """
    Parse a custom BA hub-priority list supplied through CLI or the launcher.
    
        The accepted formats support explicit indices and agent names. The result is a list of local
        agent indices used to force selected agents into high-priority BA positions when requested.
    """
    text = str(custom_text or "").strip()
    if not text:
        return []
    raw_tokens = [t.strip() for t in re.split(r"[,;\n]+", text) if str(t).strip()]
    by_name = {str(getattr(a, 'agent_name', '') or '').strip().lower(): i for i, a in enumerate(list_agents or [])}
    by_agent_id = {}
    for i, a in enumerate(list_agents or []):
        try:
            by_agent_id[int(getattr(a, 'agent_id'))] = i
        except Exception:
            pass

    out = []
    seen = set()
    n = len(list_agents or [])
    for tok in raw_tokens:
        low = tok.lower().strip()
        local_idx = None
        m_idx = re.match(r"^(?:idx|index|local)\s*[:=]\s*(-?\d+)\s*$", low)
        m_agent = re.match(r"^(?:agent_id|agentid|id)\s*[:=]\s*(-?\d+)\s*$", low)
        if m_idx:
            local_idx = int(m_idx.group(1))
        elif m_agent:
            local_idx = by_agent_id.get(int(m_agent.group(1)))
        elif re.fullmatch(r"-?\d+", low):
            # Plain numbers are treated as local indices to avoid ambiguity.
            local_idx = int(low)
        else:
            local_idx = by_name.get(low)

        if local_idx is None:
            continue
        if 0 <= int(local_idx) < n and int(local_idx) not in seen:
            seen.add(int(local_idx))
            out.append(int(local_idx))
    return out


def _build_ba_hub_priority_indices(strategy: str, custom_text: str, list_agents, opinions_by_idx, seed: int | None = None) -> list[int]:
    """
    Build the local-agent priority order used for controlled BA hub assignment.
    
        Depending on the selected strategy, this may target positive agents, negative agents, actual hubs,
        or a custom list. The priority order is passed to the network builder so hub identity is controlled
        after graph construction rather than accidentally tied to input order.
    """
    strategy = str(strategy or "default").strip().lower()
    if strategy == "minority":
        strategy = "opposite_majority"
    n = len(list_agents or [])
    if n <= 0 or strategy == "default":
        return []

    rng = random.Random(seed)
    idxs = list(range(n))

    def op(i: int) -> int:
        """
        Move selected agents to the front or back of the BA priority list.
        
            This local helper keeps the priority-list transformations readable inside
            _build_ba_hub_priority_indices while preserving the selected relative order.
        """
        try:
            return int(opinions_by_idx.get(i, getattr(list_agents[i], 'init_belief', 0)))
        except Exception:
            try:
                return int(getattr(list_agents[i], 'init_belief', 0))
            except Exception:
                return 0

    if strategy == "custom":
        selected = _parse_ba_hub_custom_indices(custom_text, list_agents)
    elif strategy == "random":
        selected = idxs[:]
        rng.shuffle(selected)
    elif strategy == "positive":
        selected = [i for i in idxs if op(i) > 0]
    elif strategy == "negative":
        selected = [i for i in idxs if op(i) < 0]
    elif strategy == "neutral":
        selected = [i for i in idxs if op(i) == 0]
    elif strategy == "extreme":
        selected = [i for i in idxs if abs(op(i)) == 2]
    elif strategy == "opposite_majority":
        pos = [i for i in idxs if op(i) > 0]
        neg = [i for i in idxs if op(i) < 0]
        if len(pos) > len(neg):
            selected = neg
        elif len(neg) > len(pos):
            selected = pos
        else:
            selected = []
    else:
        selected = []

    # Deterministic shuffle within the selected side so the same seed is reproducible
    # but we do not always privilege the lowest local index within a side.
    selected = list(selected)
    rng.shuffle(selected)
    return selected


def _compute_network_metrics_by_idx(num_agents: int, neighbors: dict, *, network_type: str, ba_hub_strategy: str = "default", ba_hub_assignment_mode: str = "early_position", ba_hub_priority_indices=None, opinions_by_idx=None) -> dict[int, dict]:
    """
    Compute degree and centrality diagnostics for each local agent index.
    
        These metrics are exported with agent summaries and hub diagnostics. They verify which agents
        actually became high-degree nodes and support degree-weighted opinion analyses.
    """
    priority_order = [int(i) for i in (ba_hub_priority_indices or []) if 0 <= int(i) < int(num_agents)]
    priority = set(priority_order)
    assignment_mode = _normalize_ba_hub_assignment_mode(ba_hub_assignment_mode)
    degrees = {i: int(len((neighbors or {}).get(i, set()))) for i in range(int(num_agents))}
    total_degree = float(sum(degrees.values()) or 0)
    sorted_nodes = sorted(range(int(num_agents)), key=lambda i: (-degrees.get(i, 0), i))
    ranks = {node: rank + 1 for rank, node in enumerate(sorted_nodes)}
    max_degree = max(degrees.values()) if degrees else 0
    top_cutoff = max(1, min(int(num_agents), len(priority) if priority else max(1, int(round(0.10 * int(num_agents))))))
    top_nodes = set(sorted_nodes[:top_cutoff])

    out = {}
    for i in range(int(num_agents)):
        try:
            init_belief = int((opinions_by_idx or {}).get(i, 0))
        except Exception:
            init_belief = 0
        targeted = int(i in priority)
        is_top = int(i in top_nodes)
        out[i] = {
            'network_type': str(network_type or ''),
            'network_degree': int(degrees.get(i, 0)),
            'network_degree_rank': int(ranks.get(i, 0)),
            'network_degree_share': 0.0 if total_degree <= 0 else float(degrees.get(i, 0) / total_degree),
            'network_top_hub_flag': is_top,
            'network_max_degree_hub_flag': int(degrees.get(i, 0) == max_degree and max_degree > 0),
            'network_initial_side': _belief_side_label(init_belief),
            'ba_hub_strategy': str(ba_hub_strategy or 'default'),
            'ba_hub_assignment_mode': str(assignment_mode),
            'ba_hub_targeted_flag': targeted,
            'ba_hub_priority_order': (priority_order.index(i) + 1) if i in priority else '',
            # Separates the two meanings that were previously conflated:
            # early_position = who receives the early preferential-attachment advantage;
            # actual_hubs = who is assigned to the realized highest-degree nodes.
            'ba_early_priority_flag': int(targeted and assignment_mode in {'early_position', 'early_and_actual'}),
            'ba_actual_hub_assigned_flag': int(targeted and is_top and assignment_mode in {'actual_hubs', 'early_and_actual'}),
        }
    return out

def _network_metric_fields_for_idx(network_metrics_by_idx: dict, idx: int, prefix: str = "") -> dict:
    """Return the exported network-metric field names for one agent index."""
    vals = dict((network_metrics_by_idx or {}).get(int(idx), {}) or {})
    return {f"{prefix}{k}": v for k, v in vals.items()}


def _write_network_hub_metrics_csv(path: str, rows):
    """
    Write the network-level hub/centrality diagnostic CSV.
    
        The file is an audit artifact for BA/WS/ER experiments. It records node degree, centrality,
        initial/final beliefs, and whether a node was selected as a targeted hub.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(list(rows or []))
        if not df.empty and 'network_degree_rank' in df.columns:
            df = df.sort_values(['network_degree_rank', 'agent_id'], na_position='last').reset_index(drop=True)
        df.to_csv(path, index=False)
        try:
            print(f"[INFO] Network hub metrics CSV: {path}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[warn] could not write network hub metrics CSV: {e}")
        except Exception:
            pass


def _relabel_neighbor_dict(neighbors: dict, order: list[int], num_agents: int) -> dict[int, set[int]]:
    """Relabel a neighbor dict from BA generation positions to local agent indices."""
    try:
        n = int(num_agents)
    except Exception:
        n = len(order or [])
    cleaned = []
    seen = set()
    for raw in list(order or []):
        try:
            i = int(raw)
        except Exception:
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            cleaned.append(i)
    cleaned.extend(i for i in range(n) if i not in seen)
    mapping = {pos: cleaned[pos] for pos in range(min(n, len(cleaned)))}
    out = {i: set() for i in range(n)}
    for src, nbrs in (neighbors or {}).items():
        try:
            s2 = int(mapping.get(int(src), int(src)))
        except Exception:
            continue
        out.setdefault(s2, set())
        for dst in (nbrs or set()):
            try:
                d2 = int(mapping.get(int(dst), int(dst)))
            except Exception:
                continue
            if 0 <= s2 < n and 0 <= d2 < n and s2 != d2:
                out.setdefault(s2, set()).add(d2)
                out.setdefault(d2, set()).add(s2)
    return out


def _relabel_neighbors_by_degree_to_priority(neighbors: dict, priority_order: list[int], num_agents: int) -> dict[int, set[int]]:
    """Compatibility fallback: assign current highest-degree nodes to priority agents."""
    n = int(num_agents)
    priority_clean = []
    seen = set()
    for raw in list(priority_order or []):
        try:
            i = int(raw)
        except Exception:
            continue
        if 0 <= i < n and i not in seen:
            seen.add(i)
            priority_clean.append(i)
    if not priority_clean:
        return {int(i): set(int(j) for j in (neighbors or {}).get(i, set())) for i in range(n)}

    base = {i: set(int(j) for j in (neighbors or {}).get(i, set()) if 0 <= int(j) < n and int(j) != i) for i in range(n)}
    # Ensure symmetry before relabeling.
    for i, nbrs in list(base.items()):
        for j in list(nbrs):
            base.setdefault(j, set()).add(i)

    degrees = {i: len(base.get(i, set())) for i in range(n)}
    current_by_degree = sorted(range(n), key=lambda i: (-degrees.get(i, 0), i))
    desired_order = list(priority_clean) + [i for i in range(n) if i not in set(priority_clean)]
    old_to_new = {old: desired_order[pos] for pos, old in enumerate(current_by_degree[:len(desired_order)])}
    for i in range(n):
        old_to_new.setdefault(i, i)

    relabeled = {i: set() for i in range(n)}
    for old_i, nbrs in base.items():
        new_i = int(old_to_new.get(old_i, old_i))
        for old_j in nbrs:
            new_j = int(old_to_new.get(old_j, old_j))
            if new_i == new_j:
                continue
            relabeled[new_i].add(new_j)
            relabeled[new_j].add(new_i)
    return relabeled


def _build_barabasi_albert_compat(
    *,
    num_agents: int,
    m_attach: int,
    seed: int | None = None,
    homophily: bool = False,
    opinions=None,
    attributes=None,
    ba_hub_strategy: str = "default",
    ba_hub_assignment_mode: str = "early_position",
    ba_hub_priority_indices=None,
):
    """Build BA with hub priority across old/new network_models APIs.

    Current project copies have used different keyword names:
      - hub_priority_nodes  (older qwen runtime patch)
      - hub_strategy / hub_agent_ids (network_models1.py)
      - hub_assignment_mode (new: early_position / actual_hubs / early_and_actual)
      - no hub args at all (older baseline)

    This wrapper keeps the simulator additive/backward-compatible instead of
    forcing every local network_models.py copy to have the same signature.
    """
    strategy = str(ba_hub_strategy or "default").strip().lower()
    assignment_mode = _normalize_ba_hub_assignment_mode(ba_hub_assignment_mode)
    priority = [int(i) for i in (ba_hub_priority_indices or []) if str(i).strip() != ""]

    try:
        import inspect as _inspect
        params = set(_inspect.signature(build_barabasi_albert).parameters.keys())
    except Exception:
        params = set()

    common = dict(
        num_agents=int(num_agents),
        m_attach=int(m_attach),
        seed=seed,
        homophily=bool(homophily),
        opinions=opinions,
        attributes=attributes,
    )

    def maybe_force_actual(neigh):
        """
        Force actual-hub reassignment when a legacy mode requires it.
        
            Older launcher options sometimes implied post-construction actual-hub assignment without naming
            that mode explicitly. This local helper preserves compatibility while normalizing the final mode.
        """
        if assignment_mode in {"actual_hubs", "early_and_actual"} and priority:
            return _relabel_neighbors_by_degree_to_priority(neigh, priority, int(num_agents))
        return neigh

    if "hub_priority_nodes" in params:
        kw = dict(common)
        kw["hub_priority_nodes"] = priority
        if "hub_assignment_mode" in params:
            kw["hub_assignment_mode"] = assignment_mode
            return build_barabasi_albert(**kw)
        return maybe_force_actual(build_barabasi_albert(**kw))

    if "hub_strategy" in params or "hub_agent_ids" in params:
        kw = dict(common)
        if "hub_strategy" in params:
            kw["hub_strategy"] = strategy
        if "hub_agent_ids" in params:
            kw["hub_agent_ids"] = priority if strategy == "custom" else None
        if "hub_assignment_mode" in params:
            kw["hub_assignment_mode"] = assignment_mode
            return build_barabasi_albert(**kw)
        return maybe_force_actual(build_barabasi_albert(**kw))

    # Last-resort compatibility for older BA builders: generate the base BA graph
    # and relabel early BA positions to the selected priority nodes.
    base = build_barabasi_albert(**common)
    if not priority or strategy in {"default", "none", "networkx_default"}:
        return base
    order = list(priority) + [i for i in range(int(num_agents)) if i not in set(priority)]
    early = _relabel_neighbor_dict(base, order, int(num_agents))
    if assignment_mode in {"actual_hubs", "early_and_actual"}:
        return _relabel_neighbors_by_degree_to_priority(early, priority, int(num_agents))
    return early

def _persona_value_norm(value) -> str:
    """Normalize persona-field values before policy matching and validation checks."""
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _compile_persona_validation_policy(*, epistemic_profile="", institutional_trust="", uncertainty_tolerance="", evidence_style="", official_narrative_suspicion="", openness_to_update="", value_orientation="", agency_vs_fatalism="", conflict_style="") -> dict:
    """
    Translate persona traits into permissions used by output validators.
    
        Validators should not flag all distrust or authority language equally. This policy records when
        a persona makes certain explanation styles expected, for example suspicion-first wording for
        low-trust agents or source-oriented wording for authority-leaning agents.
    """
    profile = _persona_value_norm(epistemic_profile)
    trust = _persona_value_norm(institutional_trust)
    uncertainty = _persona_value_norm(uncertainty_tolerance)
    evidence = _persona_value_norm(evidence_style)
    suspicion = _persona_value_norm(official_narrative_suspicion)
    openness = _persona_value_norm(openness_to_update)
    value_orient = _persona_value_norm(value_orientation)
    agency = _persona_value_norm(agency_vs_fatalism)
    conflict = _persona_value_norm(conflict_style)

    active = any([profile, trust, uncertainty, evidence, suspicion, openness, value_orient, agency, conflict])
    low_trust = trust in {"low", "very low"}
    high_trust = trust in {"high", "very high"}
    high_suspicion = suspicion in {"high", "very high"}
    high_uncertainty = uncertainty in {"high", "very high"}
    low_openness = openness in {"low", "very low"}
    high_openness = openness in {"high", "very high"}
    suspicion_first = evidence in {"suspicion first", "suspicion-first", "suspicion_first"}
    authority_first = evidence in {"authority first", "authority-first", "authority_first"}
    coherence_first = evidence in {"coherence first", "coherence-first", "coherence_first"}

    allow_distrust = bool(active and (low_trust or high_suspicion or suspicion_first or "skeptic" in profile or "distrust" in profile))
    allow_authority = bool(active and (high_trust or authority_first or "authority" in profile or "trusting" in profile or "stabilizer" in profile))
    allow_uncertainty = bool(active and (high_uncertainty or high_openness or "agnostic" in profile or "open-minded" in profile or "open minded" in profile))
    allow_coherence = bool(active and (coherence_first or "coherence" in evidence or "coherent" in profile))

    return {
        "active": active,
        "profile": profile,
        "institutional_trust": trust,
        "uncertainty_tolerance": uncertainty,
        "evidence_style": evidence,
        "official_narrative_suspicion": suspicion,
        "openness_to_update": openness,
        "value_orientation": value_orient,
        "agency_vs_fatalism": agency,
        "conflict_style": conflict,
        "allow_distrust_language": allow_distrust,
        "allow_authority_language": allow_authority,
        "allow_uncertainty_language": allow_uncertainty,
        "allow_coherence_language": allow_coherence,
        "allow_update_language": bool(active and high_openness),
        "allow_combative_tone": bool(active and conflict == "combative"),
        "allow_outcome_framing": bool(active and value_orient in {"outcome_focused", "outcome-focused", "outcome focused"}),
        "allow_fatalistic_process_framing": bool(active and agency == "fatalistic"),
        "closed_world_attitude_language_ok": bool(active and (allow_distrust or allow_authority or allow_uncertainty or allow_coherence)),
        "allow_persona_trait_self_reference": bool(active and (allow_distrust or allow_authority or allow_uncertainty or high_openness or low_openness)),
    }


CURRENT_PERSONA_VALIDATION_POLICY: dict = {}


def _set_current_persona_validation_policy(policy: dict | None):
    """Set the active persona validation policy for the current agent call."""
    global CURRENT_PERSONA_VALIDATION_POLICY
    CURRENT_PERSONA_VALIDATION_POLICY = dict(policy or {})


def _current_persona_validation_policy() -> dict:
    """Return the currently active persona validation policy, if any."""
    return dict(CURRENT_PERSONA_VALIDATION_POLICY or {})


CURRENT_VALIDATION_CONTEXT: dict = {"prompt_text": "", "grounding_text": "", "material_profile": ""}


def _derive_validation_material_profile(prompt_text: str = "", grounding_text: str = "") -> str:
    """Classify shown material so validators know whether evidence is grounded, institutional, or distrust-related."""
    prompt_low = str(prompt_text or "").lower()
    grounding_low = str(grounding_text or "").lower()

    has_rag = ("retrieved snippets" in prompt_low) or ("retrieved snippets" in grounding_low)
    has_fact = ("fact pack" in prompt_low) or ("fact pack" in grounding_low)

    rag_mode = str(getattr(args, "rag_content_mode", "full") or "full").strip().lower() if 'args' in globals() else 'full'
    fact_mode = str(getattr(args, "fact_pack_mode", "off") or "off").strip().lower() if 'args' in globals() else 'off'

    profiles = []
    if has_rag:
        if rag_mode in {"criticism_only", "supportive_only", "context_only", "balanced"}:
            profiles.append(rag_mode)
        elif rag_mode == "full":
            profiles.append("balanced")
    if has_fact:
        if fact_mode in {"criticisms", "criticism_only"}:
            profiles.append("criticism_only")
        elif fact_mode in {"core", "supportive_only"}:
            profiles.append("supportive_only")
        elif fact_mode == "context":
            profiles.append("context_only")
        elif fact_mode in {"balanced", "full"}:
            profiles.append("balanced")

    if not profiles:
        return ""
    uniq = {p for p in profiles if p}
    if len(uniq) == 1:
        return next(iter(uniq))
    if "balanced" in uniq:
        return "balanced"
    if "criticism_only" in uniq and "supportive_only" in uniq:
        return "balanced"
    return next(iter(uniq))


def _set_current_validation_context(prompt_text: str = "", grounding_text: str = "", material_profile: str = ""):
    """
    Store the local prompt, tweet, grounding, and material profile for validators.
    
        Many validators need to know whether a phrase is grounded in shown RAG/fact-pack/web material.
        This function sets that per-call context before Step-2 or Step-3 validation begins.
    """
    global CURRENT_VALIDATION_CONTEXT
    prompt_str = str(prompt_text or "")
    grounding_str = str(grounding_text or _extract_external_grounding_text(prompt_str) or "")
    profile = str(material_profile or _derive_validation_material_profile(prompt_str, grounding_str) or "")
    CURRENT_VALIDATION_CONTEXT = {
        "prompt_text": prompt_str,
        "grounding_text": grounding_str,
        "material_profile": profile,
    }


def _current_validation_context() -> dict:
    """Return the active validation context dictionary."""
    return dict(CURRENT_VALIDATION_CONTEXT or {})


def _current_validation_grounding_text() -> str:
    """Return the currently shown grounding material used for strict closed-world checks."""
    return str((_current_validation_context() or {}).get("grounding_text") or "")


def _current_validation_material_profile() -> str:
    """Return the current material-profile flags derived from the prompt and tweet."""
    return str((_current_validation_context() or {}).get("material_profile") or "")


def _grounding_has_distrust_family(text: str) -> bool:
    """Detect whether shown material contains distrust or anti-institutional framing."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:independent verification|verification gap|official narratives?|official story|cover-up|coverup|curated|too polished|too perfect|skeptic(?:al|ism)?|doubt|unresolved|anomaly|missing corroboration)\b", low, flags=re.I))


def _grounding_has_authority_family(text: str) -> bool:
    """Detect whether shown material contains authority or institutional-source framing."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:official records?|historical records?|documented|widely accepted|consensus|authorities?|experts?|scientific community|credible sources?|multiple independent sources?|corroborating data|official confirmation|verified historical data|historical data|evidence left behind|retroreflectors?|lunar laser ranging)\b", low, flags=re.I))


def _grounding_has_support_artifact_family(text: str) -> bool:
    """Detect whether shown material contains claim-supporting concrete artifacts."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:retroreflectors?|lunar laser ranging|laser ranging|telemetry|tracking|mission records?|apollo missions?|lunar samples?|moon rocks?|tracking data|mission data|brought back lunar samples)\b", low, flags=re.I))


def _grounding_has_challenge_artifact_family(text: str) -> bool:
    """Detect whether shown material contains claim-challenging artifact claims."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:missing (?:original )?materials?|missing tapes?|sstv|unresolved anomalies?|lighting|shadows?|radiation exposure|van allen|incomplete (?:visuals?|footage|documentation)|harder to check directly|questions about (?:shadows|lighting|object behavior))\b", low, flags=re.I))


def _grounding_supports_external_family(family: str, grounding_text: str, matched_text: str = "") -> bool:
    """Check whether external-source wording is licensed by shown grounding material."""
    grounding = str(grounding_text or "")
    fam = str(family or "").strip().lower()
    if not grounding or not fam:
        return False
    grounded = _grounded_term_families_in_text(grounding)
    if fam in grounded:
        return True
    if matched_text and _matched_text_is_explicitly_grounded(matched_text, grounding):
        return True
    if fam in {"verification", "records", "documents", "sources", "reports"}:
        if _grounding_has_authority_family(grounding) or _grounding_has_support_artifact_family(grounding):
            return True
    if fam in {"evidence", "verification"} and (_grounding_has_support_artifact_family(grounding) or _grounding_has_challenge_artifact_family(grounding)):
        return True
    return False


def _validator_allows_generic_reason(reason: str | None, text: str = "", policy: dict | None = None) -> bool:
    """
    Decide whether a generic-looking reason is acceptable in the current context.
    
        Some explanations are normally too vague, but may be defensible when a persona or shown material
        explicitly supports broad distrust, authority, or uncertainty language. This function prevents
        validators from over-repairing such persona-consistent outputs.
    """
    if _persona_allows_generic_reason(reason, text, policy):
        return True

    r = str(reason or "").strip().lower()
    body = str(text or "")
    profile = _current_validation_material_profile()
    grounding = _current_validation_grounding_text()

    distrust_reasons = {
        "generic_existing_skepticism", "stock_skeptical_phrase", "verification_on_site_formula",
        "circumstantial_unverifiable_formula", "generic_common_skeptical_claim",
        "generic_remain_skeptical", "generic_remain_unconvinced", "generic_still_uncertain",
        "generic_both_sides", "generic_not_enough_evidence", "generic_need_more_evidence",
        "generic_no_direct_proof", "generic_lacks_direct_proof", "generic_not_fully_proven",
        "generic_not_definitively_proven", "generic_no_concrete_evidence", "generic_no_credible_evidence",
        "debate_summary_formula",
    }
    authority_reasons = {
        "generic_strong_evidence", "generic_overwhelming_evidence", "generic_claim_credibility",
        "generic_verified_historical_data", "generic_historical_data_reference",
        "generic_documented_moon_landings", "generic_confirms_claim", "generic_documented_claim",
        "overwhelming_evidence_formula", "undeniable_evidence_formula", "back_that_up_formula",
        "stock_evidence_phrase",
    }

    if profile == "criticism_only" and r in distrust_reasons:
        if _persona_has_distrust_markers(body) or _grounding_has_distrust_family(grounding):
            return True
    if profile == "supportive_only" and r in authority_reasons:
        if _persona_has_authority_markers(body) or _grounding_has_authority_family(grounding):
            return True
    return False


def _validator_allows_strict_closed_phrase(reason: str | None, text: str = "", prompt_text: str = "", policy: dict | None = None) -> bool:
    """Decide whether strict-closed wording is acceptable because it is grounded or persona-consistent."""
    if _persona_allows_strict_closed_phrase(reason, text, policy):
        return True
    grounding = str(_extract_external_grounding_text(prompt_text or _current_validation_context().get("prompt_text", "")) or _current_validation_grounding_text() or "")
    if not grounding:
        return False
    body = str(text or "")
    reason_low = str(reason or "").strip().lower()
    if _matched_text_is_explicitly_grounded(body, grounding):
        return True
    if _grounding_supports_external_family("verification", grounding, matched_text=body):
        if any(tok in reason_low for tok in ["verifiable", "direct_evidence", "documented", "historical", "official_confirmation", "corroborating", "credible_sources", "independent_sources", "public_evidence"]):
            return True
    if _grounding_has_authority_family(grounding) and _persona_has_authority_markers(body):
        return True
    if _grounding_has_support_artifact_family(grounding) and re.search(r"\b(?:retroreflectors?|telemetry|tracking|mission records?|lunar samples?|moon rocks?|laser ranging|supports?|confirm|confirmed|validate|validated)\b", body, flags=re.I):
        return True
    if _grounding_has_distrust_family(grounding) and _persona_has_distrust_markers(body):
        return True
    return False




def _persona_has_authority_markers(text: str) -> bool:
    """Return whether persona text contains authority-trusting markers."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:official records?|historical records?|documented|widely accepted|consensus|authorities?|experts?|scientific community|credible sources?|multiple independent sources?|corroborating data|official confirmation|verified historical data|historical data)\b", low, flags=re.I))


def _persona_has_distrust_markers(text: str) -> bool:
    """Return whether persona text contains distrust or suspicion markers."""
    low = _persona_value_norm(text)
    return bool(re.search(r"\b(?:institutional trust|official narratives?|official story|official stories|institutional credibility|verification gap|independent verification|cover-up|coverup|too polished|too perfect|curated|existing skepticism|remaining doubt|skepticism|doubt)\b", low, flags=re.I))


def _persona_allows_hidden_state_phrase(text: str, policy: dict | None = None) -> bool:
    """
    Allow selected internal-state phrases only when persona traits justify them.
    
        The model is generally discouraged from saying things like "my existing skepticism." However,
        suspicion-first or distrust personas may naturally refer to skepticism. This gate keeps that
        exception explicit instead of allowing every internal-state phrase unconditionally.
    """
    p = dict(policy or _current_persona_validation_policy() or {})
    if not p.get("active"):
        return False
    low = _persona_value_norm(text)
    if not low:
        return False
    if re.search(r"\bmy institutional trust\b|\baligns? with my institutional trust\b", low, flags=re.I):
        return bool(p.get("allow_persona_trait_self_reference"))
    if re.search(r"\bmy openness to update\b|\blow openness to update\b", low, flags=re.I):
        return bool(p.get("allow_persona_trait_self_reference") or p.get("allow_update_language"))
    if re.search(r"\bmy persona\b", low, flags=re.I):
        return bool(p.get("allow_persona_trait_self_reference"))
    if re.search(r"\bmy (?:prior|existing) skepticism\b|\bexisting skepticism\b|\bremaining doubt\b", low, flags=re.I):
        return bool(p.get("allow_distrust_language"))
    return False


def _persona_allows_generic_reason(reason: str | None, text: str = "", policy: dict | None = None) -> bool:
    """Allow otherwise-generic reasons when the active persona/context justifies them."""
    p = dict(policy or _current_persona_validation_policy() or {})
    if not p.get("active"):
        return False
    r = str(reason or "").strip().lower()
    low = _persona_value_norm(text)
    if not r:
        return False
    if r in {"generic_existing_skepticism", "stock_skeptical_phrase", "verification_on_site_formula", "circumstantial_unverifiable_formula", "generic_common_skeptical_claim"}:
        return bool(p.get("allow_distrust_language"))
    if r in {"generic_strong_evidence", "generic_overwhelming_evidence", "generic_claim_credibility", "generic_verified_historical_data", "generic_historical_data_reference", "generic_documented_moon_landings", "generic_confirms_claim", "generic_documented_claim", "overwhelming_evidence_formula", "undeniable_evidence_formula", "back_that_up_formula", "stock_evidence_phrase"}:
        return bool(p.get("allow_authority_language") and _persona_has_authority_markers(low or r))
    if r in {"debate_summary_formula"}:
        return bool(p.get("allow_uncertainty_language"))
    return False


def _persona_allows_strict_closed_phrase(reason: str | None, text: str = "", policy: dict | None = None) -> bool:
    """Allow strict closed-world phrases when the active persona/context justifies them."""
    p = dict(policy or _current_persona_validation_policy() or {})
    if not p.get("active"):
        return False
    r = str(reason or "").strip().lower()
    low = _persona_value_norm(text)
    authority_reasons = {"strict_closed_overwhelming_evidence", "strict_closed_strong_evidence", "strict_closed_verifiable_evidence", "strict_closed_historical_evidence", "strict_closed_direct_evidence", "strict_closed_public_evidence", "strict_closed_official_confirmation", "strict_closed_documented_evidence", "strict_closed_credible_sources", "strict_closed_multiple_independent_sources", "strict_closed_corroborating_data", "strict_closed_global_consensus", "strict_closed_public_knowledge"}
    if r in authority_reasons:
        return bool(p.get("allow_authority_language") and _persona_has_authority_markers(low or r))
    return False



def _ensure_native_events_log_file():
    """Create the native-events CSV even before any native failure occurs.

    This is intentionally separate from repair_events.csv: repairs are semantic/
    protocol interventions, while native_events.csv records backend-level emission
    diagnostics such as empty final text, length stops, thinking length, and local
    Qwen recovery. The rows are cheap and do not change agent decisions.
    """
    global NATIVE_EVENTS_LOG_CSV_PATH
    d = _ensure_retry_debug_dir()
    if not d:
        return None
    if not NATIVE_EVENTS_LOG_CSV_PATH:
        run_tok = _safe_file_token(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or 'run')
        NATIVE_EVENTS_LOG_CSV_PATH = os.path.join(d, f'native_events_{run_tok}.csv')
    try:
        os.makedirs(os.path.dirname(NATIVE_EVENTS_LOG_CSV_PATH), exist_ok=True)
        if not os.path.exists(NATIVE_EVENTS_LOG_CSV_PATH):
            with open(NATIVE_EVENTS_LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'run_id', 'time_step', 'step_kind', 'agent_name', 'label', 'attempt',
                    'event_type', 'model', 'done_reason', 'had_final_text', 'final_text_len',
                    'thinking_len', 'num_predict', 'think', 'note'
                ])
        return NATIVE_EVENTS_LOG_CSV_PATH
    except Exception:
        return None


def _append_native_event_row(*, step_kind: str = '', agent_name: str = '', label: str = '', attempt: str = '',
                             event_type: str = '', model: str = '', done_reason: str = '', text: str = '',
                             thinking: str = '', payload=None, note: str = ''):
    """Append one native-Ollama execution event to the diagnostic CSV."""
    csv_path = _ensure_native_events_log_file()
    if not csv_path:
        return
    payload = dict(payload or {})
    opts = dict(payload.get('options') or {})
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                str(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or ''),
                CURRENT_INTERACTION_STEP if CURRENT_INTERACTION_STEP is not None else '',
                str(step_kind or ''),
                str(agent_name or ''),
                str(label or ''),
                str(attempt or ''),
                str(event_type or ''),
                str(model or payload.get('model', '') or ''),
                str(done_reason or ''),
                int(bool(str(text or '').strip())),
                len(str(text or '')),
                len(str(thinking or '')),
                opts.get('num_predict', ''),
                payload.get('think', ''),
                str(note or ''),
            ])
    except Exception:
        return


def _web_logging_enabled() -> bool:
    """Return whether true-open web-search events should be logged."""
    try:
        return bool(TRUE_OPEN_WORLD_ENABLED and WEB_BACKEND in {"duckduckgo", "brave"} and TOOL_MODE in {"web_only", "multi"})
    except Exception:
        return False


def _ensure_web_events_log_file():
    """Create the true-open web-event CSV and header when needed."""
    global WEB_EVENTS_LOG_CSV_PATH
    if not _web_logging_enabled():
        return None
    if not WEB_EVENTS_LOG_CSV_PATH:
        d = _ensure_retry_debug_dir() or os.getcwd()
        run_tok = _safe_file_token(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or 'run')
        WEB_EVENTS_LOG_CSV_PATH = os.path.join(d, f"web_events_{run_tok}.csv")
    try:
        os.makedirs(os.path.dirname(WEB_EVENTS_LOG_CSV_PATH), exist_ok=True)
        if not os.path.exists(WEB_EVENTS_LOG_CSV_PATH):
            with open(WEB_EVENTS_LOG_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([
                    'run_id', 'time_step', 'agent_name', 'step_kind', 'role', 'previous_interaction_type',
                    'current_belief', 'backend', 'planner_mode', 'tool_mode', 'notes_mode',
                    'query', 'raw_results_count', 'kept_results_count', 'used_web',
                    'observable_primary_source', 'observable_secondary_source',
                    'raw_result_titles', 'kept_result_titles', 'tweet_text'
                ])
        return WEB_EVENTS_LOG_CSV_PATH
    except Exception:
        return None


def _append_web_event_row(*, agent_name: str = '', step_kind: str = 'step3', previous_interaction_type: str = '',
                          current_belief=None, query: str = '', raw_results=None, kept_results=None, meta: dict | None = None,
                          tweet_text: str = ''):
    """Append one true-open search or retrieval event to the diagnostic CSV."""
    csv_path = _ensure_web_events_log_file()
    if not csv_path:
        return
    raw_results = list(raw_results or [])
    kept_results = list(kept_results or [])
    meta = dict(meta or {})
    try:
        role = 'listener' if str(step_kind or 'step3').strip().lower() == 'step3' else 'speaker'
        raw_titles = '|'.join(str((r or {}).get('title', '') or '').strip()[:160] for r in raw_results if str((r or {}).get('title', '') or '').strip())
        kept_titles = '|'.join(str((r or {}).get('title', '') or '').strip()[:160] for r in kept_results if str((r or {}).get('title', '') or '').strip())
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                str(RUN_EXPORT_ID or REPAIR_LOG_RUN_LABEL or ''),
                CURRENT_INTERACTION_STEP if CURRENT_INTERACTION_STEP is not None else '',
                str(agent_name or ''),
                str(step_kind or ''),
                role,
                str(previous_interaction_type or ''),
                '' if current_belief is None else int(current_belief),
                str(WEB_BACKEND or ''),
                str(meta.get('planner_mode', PLANNER_MODE) or ''),
                str(meta.get('tool_mode', TOOL_MODE) or ''),
                str(meta.get('notes_mode', NOTES_MODE) or ''),
                str(query or ''),
                int(len(raw_results)),
                int(len(kept_results)),
                int(meta.get('used_web', 0) or 0),
                str(meta.get('observable_primary_source', '') or ''),
                str(meta.get('observable_secondary_source', '') or ''),
                raw_titles,
                kept_titles,
                str(tweet_text or '').replace('\n', ' ').strip(),
            ])
    except Exception:
        return


def _write_run_metrics_csv(metrics_path: str):
    """
    Write aggregate run counters and transition metrics after simulation completion.
    
        This compact file records total moves, transition counts, repair/artifact counters, RAG events,
        early stop state, and other run-level diagnostics used to compare experimental conditions.
    """
    if not metrics_path:
        return
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Metric', 'Count'])
            for key, value in sorted(RUN_METRICS.items()):
                w.writerow([str(key), int(value)])
    except Exception as e:
        try:
            print(f"[warn] could not write run metrics CSV: {e}")
        except Exception:
            pass

# --- UTF-8 safe stdout/stderr on Windows (prevents UnicodeEncodeError) ---
import sys
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain_core.prompts  import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage

USE_FAKE_LLM = (os.environ.get("FAKE_LLM", "").strip() == "1")  # ADR-006 Task 5.1: offline regression toggle (default off)

def build_chat_ollama(model: str, temperature: float, **opts):
    """
    Create a ChatOllama client while tolerating langchain_ollama API differences.
    
        The installed wrapper version may accept decoding parameters directly or through an `options`
        dictionary. This helper inspects the constructor and routes model options accordingly, so the
        rest of the simulator can request temperature/top_p/top_k/etc. in one uniform way.
    """
    try:
        import inspect
        sig = inspect.signature(ChatOllama)
        allowed = set(sig.parameters.keys())
    except Exception:
        allowed = {"model", "temperature"}

    kwargs = {}
    if "model" in allowed:
        kwargs["model"] = model
    if "temperature" in allowed:
        kwargs["temperature"] = temperature

    accepted = set()
    extra_opts = {}

    for k, v in (opts or {}).items():
        if v is None:
            continue
        if k in allowed:
            kwargs[k] = v
            accepted.add(k)
        else:
            extra_opts[k] = v

    # Many Ollama backends accept extra decoding params via a generic 'options' dict.
    # If the current ChatOllama supports 'options', forward any unknown opts there (without
    # interfering with the accepted explicit kwargs).
    if extra_opts and ("options" in allowed):
        existing = kwargs.get("options")
        if isinstance(existing, dict):
            existing.update(extra_opts)
            kwargs["options"] = existing
        else:
            kwargs["options"] = dict(extra_opts)

    # Fallback: if NONE of the extra opts were accepted and no options were attached above,
    # try passing all opts via 'options' (older wrapper variants).
    if opts and (not accepted) and ("options" in allowed) and ("options" not in kwargs):
        kwargs["options"] = {k: v for k, v in (opts or {}).items() if v is not None}

    return ChatOllama(**kwargs)


def safe_format(template: str, **kwargs) -> str:
    """
    Format prompt templates without breaking literal braces used in rating notation.
    
        Prompt files often contain text such as {-2, -1, 0, 1, 2}. Plain str.format would treat those
        braces as placeholders. This helper temporarily protects the intended placeholders, escapes all
        other braces, and then formats the prompt safely.
    """
    if template is None:
        return ""
    s = str(template)

    # Protect known placeholders
    for k in kwargs.keys():
        s = s.replace("{" + k + "}", f"@@@{k}@@@")

    # Escape any remaining braces
    s = s.replace("{", "{{").replace("}", "}}")

    # Restore placeholders
    for k in kwargs.keys():
        s = s.replace(f"@@@{k}@@@", "{" + k + "}")

    try:
        return s.format(**kwargs)
    except Exception:
        # Fall back to escaped template instead of crashing the simulation.
        return s



def _clean_optional_prompt_text(value: str) -> str:
    """Normalize optional prompt text and treat off/none/null values as empty."""
    s = str(value or "").strip()
    if s.lower() in {"off", "none", "null", "false"}:
        return ""
    return s


def _qwen_prompt_control_prefix(model_name: str, think_flag) -> str:
    """Return the /think or /no_think control prefix required by qwen-family models."""
    model = str(model_name or "").strip().lower()
    if "qwen3" not in model:
        return ""
    if think_flag is True:
        return "/think\n"
    if think_flag is False:
        return "/no_think\n"
    return ""


def _prepend_model_prompt_control(prompt_text: str, model_name: str, think_flag):
    """Add qwen prompt-control tokens to a prompt without duplicating existing controls."""
    s = str(prompt_text or "")
    prefix = _qwen_prompt_control_prefix(model_name, think_flag)
    if not prefix:
        return s
    stripped = s.lstrip()
    if stripped.startswith("/think") or stripped.startswith("/no_think"):
        return s
    return prefix + s

def _prompt_control_think_override(prompt_text: str, model_name: str = ""):
    """
    Read `/think` or `/no_think` from the prompt and convert it to an Ollama option.
    
        Qwen models may obey textual control tokens, but the native Ollama API is more reliable when
        the same instruction is mirrored in the JSON payload as top-level `think=True/False`. This
        function extracts that explicit override.
    """
    if model_name and not _is_qwen_family_model(model_name):
        return None
    stripped = str(prompt_text or "").lstrip()
    if re.match(r"^/no_think(?:\s|$)", stripped, flags=re.I):
        return False
    if re.match(r"^/think(?:\s|$)", stripped, flags=re.I):
        return True
    return None


def _apply_prompt_control_think_override(prompt_text: str, model_name: str, current_think):
    """Resolve prompt-level /think or /no_think instructions into the native think option."""
    override = _prompt_control_think_override(prompt_text, model_name)
    return current_think if override is None else override


def _set_wrapper_prompt_control_think_override(conversation, prompt_text: str):
    """Best-effort temporary `think` override for ChatOllama wrapper calls.

    Some langchain_ollama versions expose `think` as an attribute, others route
    it through `options`. We patch both when possible, then restore them after
    the call. This does not affect non-Qwen models or prompts without an explicit
    /think or /no_think prefix.
    """
    try:
        llm = getattr(conversation, 'llm', None)
        model_name = str(getattr(llm, 'model', '') or '')
        override = _prompt_control_think_override(prompt_text, model_name)
        if llm is None or override is None:
            return lambda: None

        had_attr = hasattr(llm, 'think')
        old_attr = getattr(llm, 'think', None) if had_attr else None
        old_options = None

        try:
            opt = getattr(llm, 'options', None)
            if isinstance(opt, dict):
                old_options = dict(opt)
                new_opt = dict(opt)
                new_opt['think'] = override
                try:
                    setattr(llm, 'options', new_opt)
                except Exception:
                    try:
                        opt['think'] = override
                    except Exception:
                        pass
        except Exception:
            old_options = None

        try:
            setattr(llm, 'think', override)
        except Exception:
            pass

        def _restore():
            """Restore the wrapper LLM think-setting after a temporary prompt-control override."""
            try:
                if old_options is not None:
                    setattr(llm, 'options', old_options)
            except Exception:
                pass
            try:
                if had_attr:
                    setattr(llm, 'think', old_attr)
            except Exception:
                pass
        return _restore
    except Exception:
        return lambda: None


# Per-model capability registry. The simulation engine is model-agnostic; only this thin
# layer differs per model. Capability is PER-MODEL, not per-family (qwen3 thinks, qwen2.5
# does not; a future reasoning-llama would think). Onboard a model by adding an entry whose
# KEY is a substring of its tag. Unknown models return has_thinking=None (unrestricted), so
# expansion never wrongly blocks a capable model.
MODEL_PROFILES = {
    "qwen3": {"has_thinking": True, "chat_path": "native"},
    "qwq": {"has_thinking": True, "chat_path": "native"},
    "deepseek-r1": {"has_thinking": True, "chat_path": "native"},
    # Add explicit no-thinking models you use so their Think mode is correctly disabled, e.g.:
    # "qwen2.5": {"has_thinking": False, "chat_path": "wrapper"},
    # "llama3":  {"has_thinking": False, "chat_path": "wrapper"},
}
_UNKNOWN_PROFILE = {"has_thinking": None, "chat_path": "wrapper"}


def _model_profile(model_name: str) -> dict:
    """Look up a model tag in MODEL_PROFILES (longest matching key wins). Unknown models get
    a permissive profile (has_thinking=None) and are never restricted."""
    m = str(model_name or "").strip().lower()
    best = None
    for key, prof in MODEL_PROFILES.items():
        if key in m and (best is None or len(key) > len(best[0])):
            best = (key, prof)
    return best[1] if best else _UNKNOWN_PROFILE


def _model_has_thinking(model_name: str):
    """True (known-thinking), False (known-no-thinking), or None (unknown -> unrestricted)."""
    return _model_profile(model_name).get("has_thinking")


def _is_qwen_family_model(model_name: str) -> bool:
    """Return whether a model name belongs to the qwen model family."""
    model = str(model_name or "").strip().lower()
    return "qwen" in model


def _compact_qwen_prompt_text(prompt_text: str, step: str = "") -> str:
    """
    Remove duplicate prompt scaffolding before sending long prompts to qwen models.
    
        The function is conservative: it collapses repeated blocks and blank lines but preserves world
        rules, persona information, fact packs, RAG context, and required output format. The goal is to
        reduce length-related failures without changing experimental semantics.
    """
    s = str(prompt_text or "")
    if not s:
        return s

    prefix = ""
    body = s
    stripped = s.lstrip()
    if stripped.startswith("/think"):
        prefix = "/think\n"
        body = stripped[len("/think"):].lstrip("\n")
    elif stripped.startswith("/no_think"):
        prefix = "/no_think\n"
        body = stripped[len("/no_think"):].lstrip("\n")

    blocks = [blk.strip() for blk in re.split(r"\n{2,}", body) if blk.strip()]
    if not blocks:
        return (prefix.strip() if prefix else s)

    kept = []
    seen = set()
    for blk in blocks:
        norm = re.sub(r"\s+", " ", blk).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(blk)

    compacted = "\n\n".join(kept).strip()
    compacted = re.sub(r"\n{3,}", "\n\n", compacted)
    if prefix:
        compacted = prefix + compacted.lstrip()
    return compacted or s

def _boost_qwen_num_predict_on_length(options: dict, step: str = "") -> dict:
    """
    Increase qwen generation budget after a native response ends because of length.
    
        Length-truncated qwen outputs often contain thinking but not the final protocol lines. This
        helper raises num_predict within step/world-specific bounds for a retry, avoiding both repeated
        truncation and unnecessarily large generations.
    """
    opts = dict(options or {})
    current = opts.get("num_predict")
    try:
        current_i = int(current) if current is not None else 0
    except Exception:
        current_i = 0

    step_norm = str(step or "step2").strip().lower()
    world_norm = _normalize_world_mode(getattr(args, "world", "closed"))
    floor = _qwen_world_num_predict_floor(step_norm, world_norm, False)
    if step_norm == "step2":
        ceiling = max(floor, floor + 128)
        bump = 96
        boosted = floor if current_i <= 0 else min(max(current_i + bump, floor), ceiling)
    elif step_norm == "step3":
        # step3 keeps thinking on every call; a length-truncation empty means the reasoning used
        # the whole budget before the final lines. A tiny +64 bump cannot help, so give step3 real
        # room (~8k) so the thinking plus FINAL_RATING/EXPLANATION both fit, without disabling think.
        boosted = max(int(floor), 8192)
    else:
        ceiling = max(floor, floor + 128)
        bump = 96
        boosted = floor if current_i <= 0 else min(max(current_i + bump, floor), ceiling)

    opts["num_predict"] = boosted
    return opts





def _heuristic_freeform_protocol_body(raw_text: str, field: str = "tweet") -> str:
    """Best-effort salvage of natural text when the model misses protocol headers."""
    s = _strip_model_think_and_fence_artifacts(raw_text)
    if not s:
        return ""

    kept = []
    for ln in str(s).splitlines():
        raw = ln.strip()
        if not raw:
            continue
        low = raw.lower()
        if re.match(r"^/?(?:think|no_think)\b", low):
            continue
        if re.match(r"^final(?:_| )rating\s*:", low):
            continue
        if re.match(r"^(assistant|system|user)\s*:", low):
            continue
        if re.match(r"^(note|notes|warning|warnings|debug)\s*:", low):
            continue
        if re.match(r"^(tweet|explanation|reason|reasoning|anchor)\s*:", low):
            raw = re.sub(r"(?i)^(tweet|explanation|reason|reasoning)\s*:\s*", "", raw).strip()
            if not raw:
                continue
            low = raw.lower()
        if re.match(r"^(claim|current_rating|current belief|allowed_final_rating_set|allowed_set|task|rating scale|output format|use exactly|core rules|movement rule|consistency rule|explanation rule|world rules?)\b", low):
            continue
        if "{" in raw and "}" in raw and len(raw) < 120:
            continue
        if re.match(r"^<[^>]+>$", raw):
            continue
        if any(tok in low for tok in ["output exactly", "nothing else", "as instructed", "exactly 2 lines", "follow the format", "empty_final_text", "length_retry"]):
            continue
        kept.append(raw)

    out = " ".join(kept).strip()
    out = re.sub(r"\s+", " ", out).strip()
    if field == "tweet":
        out = _strip_leaked_prompt_headers(out)
        out = re.sub(r"(?i)^here(?:'s| is) (?:my )?(?:tweet|post)[:\-]\s*", "", out).strip()
    return out

# ============================
# SIMPLE LOCAL RAG
# ============================

def _rag_enabled_for_world(world: str) -> bool:
    """Return whether the selected world mode permits retrieval-augmented context."""
    w = _normalize_world_mode(world)
    return w in {"open_rag", "closed_strict_rag"}

def _normalize_rag_text(s: str) -> str:
    """Normalize text for simple lexical RAG scoring."""
    s = str(s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _rag_tokens(s: str) -> set[str]:
    """Tokenize normalized text for overlap-based retrieval."""
    toks = [t for t in _normalize_rag_text(s).split() if len(t) >= 3]
    return set(toks)

def _extract_claim_from_prompt_text(prompt_text: str) -> str:
    """Extract the CLAIM field from a prompt when no explicit claim argument is available."""
    s = str(prompt_text or "")
    if not s:
        return ""
    m = re.search(r"(?im)^CLAIM:\s*(.+?)\s*$", s)
    if not m:
        return ""
    claim = str(m.group(1) or "").strip()
    if not claim or '{' in claim or '}' in claim:
        return ""
    return claim

def _resolve_version_metadata_path(raw_path: str) -> Path:
    """Resolve the version-metadata file path across absolute, project, and working-directory locations."""
    raw = str(raw_path or "").strip()
    if not raw:
        return ROOT / "prompts" / "opinion_dynamics" / "Flache_2017" / "content" / "version_metadata" / "version_metadata.json"
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [ROOT / p, Path(__file__).resolve().parent / p, Path.cwd() / p]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _load_version_metadata(raw_path: str) -> dict:
    """Load topic/version metadata used for claim text and RAG topic filtering."""
    p = _resolve_version_metadata_path(raw_path)
    try:
        import json as _json
        if not p.exists():
            return {}
        obj = _json.loads(p.read_text(encoding='utf-8'))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _version_metadata_entry(version_set: str) -> dict:
    """Return the metadata entry for the active version root."""
    version_root = _version_prefix(version_set)
    try:
        entry = VERSION_METADATA.get(version_root, {})
        return entry if isinstance(entry, dict) else {}
    except Exception:
        return {}


def _version_metadata_topic_key(version_set: str) -> str:
    """Return the corpus topic key associated with a version set."""
    entry = _version_metadata_entry(version_set)
    val = str(entry.get("topic_key", "") or "").strip()
    return val or _version_prefix(version_set)


def _version_metadata_theory_statement(version_set: str) -> str:
    """Return the claim statement associated with a version set."""
    entry = _version_metadata_entry(version_set)
    return str(entry.get("theory_statement", "") or "").strip()


def _current_claim_text(prompt_text: str = "") -> str:
    """Resolve the active claim from globals, metadata, or prompt text."""
    claim = str(THEORY_STATEMENT or "").strip()
    if claim:
        return claim
    claim = _version_metadata_theory_statement(getattr(args, "version_set", ""))
    if claim:
        return claim
    return _extract_claim_from_prompt_text(prompt_text)

def _load_simple_rag_corpus(path: str) -> list[dict]:
    """
    Load the local retrieval corpus from text, JSON, JSONL, or CSV sources.
    
        The loader normalizes heterogeneous corpus formats into dictionaries with text plus metadata
        such as topic and direction. Keeping this normalization here lets retrieval code treat all corpus
        sources uniformly.
    """
    p = Path(str(path or "").strip())
    if not p.exists():
        return []
    rows: list[dict] = []
    suffix = p.suffix.lower()
    def _append_text(chunk_text: str, idx: int):
        """Append one corpus chunk while preserving available metadata fields."""
        txt = str(chunk_text or "").strip()
        if txt:
            rows.append({"id": f"{p.stem}_{idx}", "text": txt})
    try:
        if suffix in {".txt", ".md"}:
            raw = p.read_text(encoding="utf-8", errors="replace")
            parts = [x.strip("-• \n\t") for x in re.split(r"\n\s*\n", raw) if x.strip()]
            for i, part in enumerate(parts, start=1):
                _append_text(part, i)
        elif suffix == ".jsonl":
            import json
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    txt = obj.get("text") or obj.get("chunk") or obj.get("content") or obj.get("snippet") or ""
                    row = {"id": obj.get("id") or f"{p.stem}_{i}", "text": str(txt or "").strip()}
                    for meta_key in ["label", "direction", "polarity", "type", "category", "tags", "topic", "topic_key", "topic_id", "topic_name", "version", "version_key", "entities"]:
                        if meta_key in obj:
                            row[meta_key] = obj.get(meta_key)
                    if row["text"]:
                        rows.append(row)
        elif suffix == ".json":
            import json
            obj = json.loads(p.read_text(encoding="utf-8"))
            seq = obj.get("chunks", obj) if isinstance(obj, dict) else obj
            if isinstance(seq, list):
                for i, item in enumerate(seq, start=1):
                    if isinstance(item, dict):
                        txt = item.get("text") or item.get("chunk") or item.get("content") or item.get("snippet") or ""
                        row = {"id": item.get("id") or f"{p.stem}_{i}", "text": str(txt or "").strip()}
                        for meta_key in ["label", "direction", "polarity", "type", "category", "tags", "topic", "topic_key", "topic_id", "topic_name", "version", "version_key", "entities"]:
                            if meta_key in item:
                                row[meta_key] = item.get(meta_key)
                        if row["text"]:
                            rows.append(row)
                    else:
                        txt = str(item)
                        _append_text(txt, i)
        elif suffix == ".csv":
            import csv as _csv
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = _csv.DictReader(f)
                preferred = None
                for cand in ["text", "chunk", "content", "snippet", "body"]:
                    if cand in (reader.fieldnames or []):
                        preferred = cand
                        break
                if preferred is None:
                    return []
                for i, row in enumerate(reader, start=1):
                    txt = row.get(preferred, "")
                    out_row = {"id": row.get("id") or f"{p.stem}_{i}", "text": str(txt or "").strip()}
                    for meta_key in ["label", "direction", "polarity", "type", "category", "tags", "topic", "topic_key", "topic_id", "topic_name", "version", "version_key", "entities"]:
                        if meta_key in row:
                            out_row[meta_key] = row.get(meta_key)
                    if out_row["text"]:
                        rows.append(out_row)
    except Exception:
        return []
    return rows

def _build_rag_query(claim_txt: str, tweet_txt: str, mode: str) -> str:
    """
    Build the local-RAG query from the claim, the current tweet, or both.
    
        Step-2 and Step-3 can retrieve against different context. Claim-only retrieval keeps evidence
        topic-centered, tweet-only retrieval reacts to the current message, and claim-plus-tweet combines
        both signals.
    """
    mode = str(mode or "claim").strip().lower()
    claim_txt = str(claim_txt or "").strip()
    tweet_txt = str(tweet_txt or "").strip()
    if mode == "tweet":
        return tweet_txt or claim_txt
    if mode == "claim_plus_tweet":
        return f"{claim_txt} {tweet_txt}".strip() or claim_txt or tweet_txt
    return claim_txt or tweet_txt

def _rag_topic_key_from_version_set(version_set: str) -> str:
    """Return the RAG topic key implied by a version_set value."""
    return _version_metadata_topic_key(version_set)


def _active_rag_topic_key() -> str:
    """Return the currently active RAG topic key, including manual overrides."""
    override = str(getattr(args, "rag_topic_override", "") or "").strip()
    if override:
        return override
    return _rag_topic_key_from_version_set(getattr(args, "version_set", ""))


def _row_matches_rag_topic(row: dict, topic_key: str) -> bool:
    """
    Return whether a corpus row belongs to the active topic filter.
    
        Topic filtering prevents snippets from other claims from leaking into a run. Rows without topic
        metadata are allowed as a recall-oriented fallback, so mixed corpora should preferably include
        explicit topic keys.
    """
    target = str(topic_key or "").strip().lower()
    if not target:
        return True
    vals = []
    for key in ["topic_key", "topic", "topic_id", "version", "version_key"]:
        v = row.get(key)
        if v is not None:
            vals.append(str(v).strip().lower())
    if not vals:
        return True
    for v in vals:
        if v == target:
            return True
        if v.startswith(target + "__") or v.startswith(target + "-") or v.startswith(target + ":"):
            return True
    return False


def _filter_rag_corpus_by_topic(corpus: list[dict], topic_key: str) -> list[dict]:
    """Filter the RAG corpus to chunks relevant to the active topic."""
    if not corpus:
        return []
    filtered = [row for row in corpus if _row_matches_rag_topic(row, topic_key)]
    return filtered


def _rag_row_direction(row: dict) -> str:
    """Classify one RAG row as supportive, criticism, context, or unknown."""
    vals = []
    for key in ["direction", "polarity", "label", "type", "category", "tags"]:
        v = row.get(key)
        if isinstance(v, (list, tuple, set)):
            vals.extend(str(x).lower() for x in v)
        elif v is not None:
            vals.append(str(v).lower())
    blob = " ".join(vals)
    if any(tok in blob for tok in ["criticism", "critical", "challenging", "challenge", "negative", "against", "anti"]):
        return "criticism"
    if any(tok in blob for tok in ["context", "contextual"]):
        return "context"
    if any(tok in blob for tok in ["supportive", "support", "positive", "for", "pro", "core"]):
        return "supportive"
    return "unknown"


def _rag_query_mode_for_step(step_kind: str, tweet_txt: str = "") -> str:
    """
    Resolve the effective query mode separately for Step-2 and Step-3.
    
        Step-specific overrides take precedence over the global RAG query mode. In auto mode, Step-2 can
        use claim-plus-tweet when tweet text exists, while Step-3 defaults toward the received tweet so
        retrieval is aligned with the listener's immediate evidence.
    """
    step_kind = str(step_kind or "step3").strip().lower()
    if step_kind == "step2":
        mode = str(getattr(args, "rag_query_mode_step2", "") or "").strip().lower()
    else:
        mode = str(getattr(args, "rag_query_mode_step3", "") or "").strip().lower()
    if not mode:
        mode = str(getattr(args, "rag_query_mode", "claim")).strip().lower()
    if mode == "auto":
        return ("claim_plus_tweet" if step_kind == "step2" and str(tweet_txt or "").strip() else ("tweet" if str(tweet_txt or "").strip() else "claim"))
    return mode if mode in {"claim", "tweet", "claim_plus_tweet"} else "claim"



def _rag_stable_seed(*parts) -> int:
    """Build a deterministic seed for reproducible stochastic RAG chunk selection."""
    blob = "||".join(str(p or "") for p in parts)
    digest = hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16)


def _rag_seeded_pick_rows(scored_rows: list[dict], quota: int, *, query: str = "", content_mode: str = "", group_key: str = "", topic_key: str = "", step_kind: str = "") -> list[dict]:
    """
    Select retrieved chunks reproducibly while preserving limited variety.
    
        After lexical scoring, the function samples from a high-scoring candidate pool using a stable
        seed derived from run/topic/query information. This avoids always showing the same row while
        keeping matched-seed comparisons reproducible.
    """
    quota = max(0, int(quota))
    if quota <= 0 or not scored_rows:
        return []

    pool_size = min(len(scored_rows), max(quota * 3, quota + 2))
    pool = list(scored_rows[:pool_size])

    top_score = max(int(it.get("score", 0)) for it in pool)
    near_best = [it for it in pool if int(it.get("score", 0)) >= max(1, top_score - 1)]
    candidates = near_best if len(near_best) >= quota else pool

    rng = random.Random(
        _rag_stable_seed(
            getattr(args, "seed", 0),
            topic_key,
            content_mode,
            group_key,
            step_kind,
            query,
        )
    )

    chosen = []
    available = list(candidates)
    while available and len(chosen) < quota:
        weights = []
        for it in available:
            score = max(1, int(it.get("score", 0)))
            weights.append(float(score * score))
        total = sum(weights)
        r = rng.random() * total if total > 0 else 0.0
        acc = 0.0
        idx = len(available) - 1
        for i, w in enumerate(weights):
            acc += w
            if acc >= r:
                idx = i
                break
        chosen.append(available.pop(idx))

    if len(chosen) < quota:
        chosen_ids = {str(it.get("id", "")) for it in chosen}
        for it in pool:
            if len(chosen) >= quota:
                break
            iid = str(it.get("id", ""))
            if iid in chosen_ids:
                continue
            chosen.append(it)
            chosen_ids.add(iid)

    chosen.sort(key=lambda x: (-int(x.get("score", 0)), int(x.get("length", 0)), str(x.get("id", ""))))
    return [it.get("row") for it in chosen[:quota]]


def _rag_score_rows(query: str, corpus: list[dict]) -> list[dict]:
    """Score corpus rows by lexical token overlap with the retrieval query."""
    q = _rag_tokens(query)
    if not q:
        return []
    scored_rows = []
    for row in corpus:
        chunk_text = str(row.get("text", "") or "").strip()
        if not chunk_text:
            continue
        overlap = q & _rag_tokens(chunk_text)
        score = len(overlap)
        if score <= 0:
            continue
        scored_rows.append({
            "score": score,
            "row": row,
            "direction": _rag_row_direction(row),
            "length": len(chunk_text),
            "id": str(row.get("id", "") or "").strip(),
        })
    scored_rows.sort(key=lambda x: (-int(x.get("score", 0)), int(x.get("length", 0)), str(x.get("id", ""))))
    return scored_rows


def _retrieve_simple_chunks(query: str, corpus: list[dict], top_k: int = 4, content_mode: str = "full", *, agent_name: str = "", step_kind: str = "", topic_key: str = "") -> list[dict]:
    """
    Retrieve local-corpus chunks for the active RAG content mode.
    
        Supportive, criticism, context, full, and balanced modes are implemented here. Balanced mode
        explicitly tries to draw from both supportive and critical rows and records shortfalls so the
        evidence environment can be audited later.
    """
    scored = _rag_score_rows(query, corpus)
    if not scored:
        return []

    mode = str(content_mode or "full").strip().lower()
    limit = max(1, int(top_k))

    if mode == "supportive_only":
        supportive_scored = [item for item in scored if item.get("direction") == "supportive"]
        return _rag_seeded_pick_rows(
            supportive_scored,
            limit,
            query=query,
            content_mode=mode,
            group_key="supportive",
            topic_key=topic_key,
            step_kind=step_kind,
        )

    if mode == "criticism_only":
        criticism_scored = [item for item in scored if item.get("direction") == "criticism"]
        return _rag_seeded_pick_rows(
            criticism_scored,
            limit,
            query=query,
            content_mode=mode,
            group_key="criticism",
            topic_key=topic_key,
            step_kind=step_kind,
        )

    if mode == "context_only":
        context_scored = [item for item in scored if item.get("direction") == "context"]
        return _rag_seeded_pick_rows(
            context_scored,
            limit,
            query=query,
            content_mode=mode,
            group_key="context",
            topic_key=topic_key,
            step_kind=step_kind,
        )

    if mode == "balanced":
        if limit < 2:
            limit = 2
        if limit % 2 != 0:
            limit -= 1
        per_side = max(1, limit // 2)

        supportive_scored = [item for item in scored if item.get("direction") == "supportive"]
        criticism_scored = [item for item in scored if item.get("direction") == "criticism"]
        context_scored = [item for item in scored if item.get("direction") == "context"]
        unknown_scored = [item for item in scored if item.get("direction") == "unknown"]

        selected_supportive = _rag_seeded_pick_rows(
            supportive_scored,
            per_side,
            query=query,
            content_mode=mode,
            group_key="supportive",
            topic_key=topic_key,
            step_kind=step_kind,
        )
        selected_criticism = _rag_seeded_pick_rows(
            criticism_scored,
            per_side,
            query=query,
            content_mode=mode,
            group_key="criticism",
            topic_key=topic_key,
            step_kind=step_kind,
        )

        out = []
        for i in range(per_side):
            if i < len(selected_criticism):
                out.append(selected_criticism[i])
            if i < len(selected_supportive):
                out.append(selected_supportive[i])

        if len(selected_supportive) < per_side or len(selected_criticism) < per_side:
            _metric_inc("rag_balance_shortfall_total")
            _metric_inc(f"rag_balance_shortfall::{len(selected_supportive)}support_{len(selected_criticism)}criticism")
            selected_ids = {str(r.get("id", "")) for r in out}
            extras = [item.get("row") for item in (criticism_scored + supportive_scored + context_scored + unknown_scored)]
            for row in extras:
                if len(out) >= limit:
                    break
                rid = str(row.get("id", "") or "")
                if rid and rid in selected_ids:
                    continue
                out.append(row)
                if rid:
                    selected_ids.add(rid)
        return out[:limit]

    return _rag_seeded_pick_rows(
        scored,
        limit,
        query=query,
        content_mode=mode,
        group_key="all",
        topic_key=topic_key,
        step_kind=step_kind,
    )


def _rag_header_for_direction(direction: str) -> str:
    """Return the prompt header used for a group of retrieved snippets."""
    d = str(direction or "").strip().lower()
    if d == "criticism":
        return "Retrieved snippets (claim-challenging):"
    if d == "supportive":
        return "Retrieved snippets (claim-supporting):"
    if d == "context":
        return "Retrieved snippets (contextual):"
    if d == "mixed":
        return "Retrieved snippets (mixed-direction):"
    return "Retrieved snippets:"


def _effective_rag_top_k(world: str, configured_top_k: int, content_mode: str = "full") -> int:
    """Normalize and cap top-k retrieval counts for balanced and strict closed-world modes."""
    try:
        top_k = int(configured_top_k)
    except Exception:
        top_k = 4
    top_k = max(1, top_k)
    mode = str(content_mode or "full").strip().lower()
    if mode == "balanced":
        if top_k < 2:
            top_k = 2
        if top_k % 2 != 0:
            top_k -= 1
        return max(2, top_k)
    if _normalize_world_mode(world) == 'closed_strict_rag':
        return min(top_k, 3)
    return top_k


def _rag_compact_snippet_text(text: str, max_chars: int = 480) -> str:
    """Compact a single RAG row without turning fragments into fake sentences.

    Earlier runs used a short per-snippet limit and appended a period after
    truncation. That produced broken prompt evidence such as
    "simulated observers could vastly." For controlled RAG experiments, the
    retrieved row should remain a complete snippet whenever possible. If a row
    is genuinely too long, truncate at a word boundary and mark it with an
    ellipsis so the model can see that the text was shortened.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    try:
        max_chars = max(160, int(max_chars or 480))
    except Exception:
        max_chars = 480
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars + 1]
    boundary_candidates = [cut.rfind(". "), cut.rfind("; "), cut.rfind(" — "), cut.rfind(" - "), cut.rfind(" ")]
    punct_idx = max(boundary_candidates)
    if punct_idx >= max(80, int(max_chars * 0.60)):
        s = cut[:punct_idx].strip(" ,;:-—")
    else:
        s = cut[:max_chars].rsplit(" ", 1)[0].strip(" ,;:-—")
    return (s or cut[:max_chars].strip(" ,;:-—")) + "…"


def _rag_truncate_rendered_context(text: str, max_chars: int) -> str:
    """Truncate the whole rendered RAG block only at safe boundaries."""
    s = str(text or "").strip()
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 0
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    cut = s[:max_chars + 1]
    idx = max(cut.rfind("\n- "), cut.rfind("\n\n"), cut.rfind(". "), cut.rfind(" "))
    if idx >= max(120, int(max_chars * 0.65)):
        return cut[:idx].rstrip(" ,;:-—") + "…"
    return cut[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-—") + "…"

def _render_rag_context(rows: list[dict], max_chars: int, content_mode: str = "full") -> str:
    """
    Render selected RAG rows into the prompt block shown to the LLM.
    
        Rows are grouped by direction and printed under stable headers. The rendered block is truncated
        safely so snippets remain readable and do not create misleading broken sentences when shortened.
    """
    cleaned_rows = [r for r in (rows or []) if str(r.get("text", "")).strip()]
    if not cleaned_rows:
        return ""

    mode = str(content_mode or "full").strip().lower()

    if mode == "criticism_only":
        s = _rag_header_for_direction("criticism") + "\n" + "\n".join(
            f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in cleaned_rows
        )
        return _rag_truncate_rendered_context(s, max_chars)
    if mode == "supportive_only":
        s = _rag_header_for_direction("supportive") + "\n" + "\n".join(
            f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in cleaned_rows
        )
        return _rag_truncate_rendered_context(s, max_chars)
    if mode == "context_only":
        s = _rag_header_for_direction("context") + "\n" + "\n".join(
            f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in cleaned_rows
        )
        return _rag_truncate_rendered_context(s, max_chars)

    supportive = [r for r in cleaned_rows if _rag_row_direction(r) == "supportive"]
    criticism = [r for r in cleaned_rows if _rag_row_direction(r) == "criticism"]
    context = [r for r in cleaned_rows if _rag_row_direction(r) == "context"]
    unknown = [r for r in cleaned_rows if _rag_row_direction(r) == "unknown"]

    sections = []
    if supportive:
        sections.append(
            _rag_header_for_direction("supportive") + "\n" + "\n".join(
                f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in supportive
            )
        )
    if criticism:
        sections.append(
            _rag_header_for_direction("criticism") + "\n" + "\n".join(
                f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in criticism
            )
        )
    if context:
        sections.append(
            _rag_header_for_direction("context") + "\n" + "\n".join(
                f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in context
            )
        )
    if unknown:
        sections.append(
            _rag_header_for_direction("mixed") + "\n" + "\n".join(
                f"- {_rag_compact_snippet_text(str(r.get('text', '')).strip())}" for r in unknown
            )
        )

    s = "\n\n".join(sec for sec in sections if sec.strip())
    return _rag_truncate_rendered_context(s, max_chars)

def _maybe_init_rag_corpus() -> list[dict]:
    """
    Load the RAG corpus lazily only for worlds/backends that actually use retrieval.
    
        Non-RAG runs should not pay corpus-loading cost or accidentally access retrieval material. This
        helper is called at startup and leaves the global corpus empty unless the selected configuration
        requires local RAG.
    """
    if not _rag_enabled_for_world(getattr(args, "world", "closed")):
        return []
    if str(getattr(args, "rag_backend", "off")).strip().lower() == "off":
        return []
    path = str(getattr(args, "rag_corpus_path", "") or "").strip()
    if not path:
        return []
    return _load_simple_rag_corpus(path)

_RAG_ARCH_STATE = {"retrievers": None, "alias_map": None, "checked": False}


def _rag_architecture_backend() -> str:
    """Return 'dense'/'graph' when an architecture retrieval backend is selected, else ''."""
    b = str(getattr(args, "rag_backend", "off")).strip().lower()
    return b if b in ("dense", "graph") else ""


def _init_rag_architecture():
    """
    Fail-fast setup for the dense/graph retrieval backends.

        Imports tools/retrievers, loads the entity alias map (graph) and pre-embeds the corpus
        (dense) at startup, so a missing sidecar or a down Ollama aborts the run before any agent
        steps instead of failing mid-simulation. No-op for off/simple.
    """
    backend = _rag_architecture_backend()
    if not backend or _RAG_ARCH_STATE["checked"]:
        return
    _RAG_ARCH_STATE["checked"] = True
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import retrievers as _retr
    except Exception as e:
        raise SystemExit(f"[config] rag_backend={backend} needs tools/retrievers.py: {e}")
    _RAG_ARCH_STATE["retrievers"] = _retr
    corpus_path = str(getattr(args, "rag_corpus_path", "") or "").strip()
    stem = os.path.splitext(corpus_path)[0]
    if not RAG_CORPUS:
        raise SystemExit(f"[config] rag_backend={backend}: empty/missing corpus at {corpus_path!r}")
    if backend == "graph":
        import json as _json
        gpath = stem + ".graph.json"
        if not os.path.exists(gpath):
            raise SystemExit(f"[config] rag_backend=graph needs the alias map next to the corpus: {gpath}")
        gdata = _json.load(open(gpath, encoding="utf-8"))
        _RAG_ARCH_STATE["alias_map"] = {e: [a.lower() for a in v.get("aliases", [])]
                                        for e, v in gdata.get("entities", {}).items()}
        if not any(r.get("entities") for r in RAG_CORPUS):
            raise SystemExit("[config] rag_backend=graph: corpus rows carry no 'entities' tags")
    if backend == "dense":
        try:
            _retr.dense_rank("warmup query", RAG_CORPUS, k=1,
                             model=str(getattr(args, "rag_embed_model", "nomic-embed-text")),
                             cache_path=stem + ".embcache.json")
        except Exception as e:
            raise SystemExit(f"[config] rag_backend=dense could not embed the corpus (is Ollama up? "
                             f"pulled {getattr(args, 'rag_embed_model', 'nomic-embed-text')!r}?): {e}")
    print(f"[config] rag_backend={backend} ready (architecture retrieval, corpus={os.path.basename(corpus_path)})")


def _retrieve_architecture_chunks(query: str, corpus: list[dict], top_k: int, backend: str) -> list[dict]:
    """
    Top-k retrieval via the dense or graph architecture, returning raw corpus rows.

        Order is preserved as ranked (graph: path order, seed outward), so the prompt renders a
        coherent chain. Pure top-k by design: content-mode quotas do not apply to the architecture
        conditions - direction balance is a property of the corpus (multihop spec rule 6).
    """
    _init_rag_architecture()
    _retr = _RAG_ARCH_STATE["retrievers"]
    stem = os.path.splitext(str(getattr(args, "rag_corpus_path", "") or "").strip())[0]
    if backend == "graph":
        ids = _retr.graph_rank(query, corpus, _RAG_ARCH_STATE["alias_map"] or {}, k=int(top_k))
    else:
        ids = _retr.dense_rank(query, corpus, k=int(top_k),
                               model=str(getattr(args, "rag_embed_model", "nomic-embed-text")),
                               cache_path=stem + ".embcache.json")
    by_id = {str(r.get("id", "")): r for r in corpus}
    return [by_id[i] for i in ids if i in by_id]


def _rag_context_for_interaction(tweet_text: str = "", step_kind: str = "step3", agent_name: str = "", prompt_variant: str = "", prompt_text: str = "") -> str:
    """
    Build and log the retrieved context for one Step-2 or Step-3 interaction.
    
        The function resolves topic filtering, query text, top-k, content mode, selection, logging, and
        rendering in one place. Both prompt-building paths call this wrapper instead of manipulating the
        corpus directly.
    """
    if not _rag_enabled_for_world(getattr(args, "world", "closed")):
        return ""
    _rag_active_backend = str(getattr(args, "rag_backend", "off")).strip().lower()
    if _rag_active_backend not in ("simple", "dense", "graph"):
        return ""
    if not RAG_CORPUS:
        return ""
    topic_key = _active_rag_topic_key()
    filtered_corpus = _filter_rag_corpus_by_topic(RAG_CORPUS, topic_key)
    if not filtered_corpus:
        return ""
    query_mode = _rag_query_mode_for_step(step_kind=step_kind, tweet_txt=str(tweet_text or ""))
    query = _build_rag_query(
        claim_txt=_current_claim_text(prompt_text=prompt_text),
        tweet_txt=str(tweet_text or ""),
        mode=query_mode,
    )
    rag_content_mode = str(getattr(args, "rag_content_mode", "full") or "full")
    effective_top_k = _effective_rag_top_k(
        getattr(args, 'world', 'closed'),
        getattr(args, 'rag_top_k', 4),
        content_mode=rag_content_mode,
    )
    if _rag_active_backend in ("dense", "graph"):
        rows = _retrieve_architecture_chunks(
            query=query,
            corpus=filtered_corpus,
            top_k=effective_top_k,
            backend=_rag_active_backend,
        )
    else:
        rows = _retrieve_simple_chunks(
            query=query,
            corpus=filtered_corpus,
            top_k=effective_top_k,
            content_mode=rag_content_mode,
            agent_name=agent_name,
            step_kind=step_kind,
            topic_key=topic_key,
        )
    _log_rag_retrieval(
        step_kind=step_kind,
        prompt_variant=prompt_variant,
        agent_name=agent_name,
        query_mode=query_mode,
        query_text=query,
        rows=rows,
        effective_top_k=effective_top_k,
    )
    return _render_rag_context(
        rows,
        max_chars=int(getattr(args, "rag_max_chars", 2200)),
        content_mode=str(getattr(args, "rag_content_mode", "full") or "full"),
    )


def _rag_context_value(tweet_text: str = "", step_kind: str = "step3", agent_name: str = "", prompt_variant: str = "", prompt_text: str = "") -> str:
    """Only inject retrieved context in actual RAG worlds; otherwise keep prompts blank."""
    if not _rag_enabled_for_world(getattr(args, "world", "closed")):
        return ""
    return _rag_context_for_interaction(tweet_text=tweet_text, step_kind=step_kind, agent_name=agent_name, prompt_variant=prompt_variant, prompt_text=prompt_text)

#######################
# Build argument parser
#######################
parser = argparse.ArgumentParser(description="Argument Parser for Opinion Dynamics Script")
parser.add_argument(
    "-m",
    "--model_name",
    default="qwen3:8b",
    type=str,
    help="Name of the LLM to use as agents",
)
parser.add_argument(
    "-t",
    "--temperature",
    default=0.7,
    type=float,
    help="Parameter that influences the randomness of the model's responses",
)

parser.add_argument(
    "--top_p",
    default=0.9,
    type=float,
    help="Nucleus sampling parameter (0-1). Lower = more conservative, higher = more diverse.",
)
parser.add_argument(
    "--top_k",
    default=40,
    type=int,
    help="Top-k sampling parameter. Lower = more conservative, higher = more diverse.",
)
parser.add_argument(
    "--repeat_penalty",
    default=1.1,
    type=float,
    help="Penalty for repeating tokens. 1.0 disables; >1 discourages repetition.",
)
parser.add_argument(
    "--repeat_last_n",
    default=64,
    type=int,
    help="How many recent tokens to consider for repetition penalty (Ollama option).",
)

parser.add_argument(
    "--frequency_penalty",
    default=0.0,
    type=float,
    help="Frequency penalty for repeated tokens (Ollama option). 0.0 disables; higher discourages repetition proportional to frequency.",
)
parser.add_argument(
    "--presence_penalty",
    default=0.0,
    type=float,
    help="Presence penalty for repeated tokens (Ollama option). 0.0 disables; higher discourages reusing tokens that already appeared.",
)

parser.add_argument(
    "--max_tokens",
    default=300,
    type=int,
    help="Maximum tokens to generate per LLM call (mapped to Ollama num_predict when available).",
)

# ------------------------------------------
# Optional decoding parameter: min_p (Ollama)
# ------------------------------------------
parser.add_argument(
    "--min_p",
    default=None,
    type=float,
    help="Minimum probability sampling (0-1). If supported by your backend/wrapper, can improve diversity control alongside top_p.",
)

# ---------------------------------------------------------
# Per-step LLM overrides (Step-2 = tweet, Step-3 = update)
# If not provided, each step inherits the global settings.
# ---------------------------------------------------------
parser.add_argument(
    "--model_name_step2",
    default=None,
    type=str,
    help="Override model for Step-2 only (tweet generation). Default: --model_name",
)
parser.add_argument(
    "--model_name_step3",
    default=None,
    type=str,
    help="Override model for Step-3 only (belief update). Default: --model_name",
)

parser.add_argument(
    "--temperature_step2",
    default=None,
    type=float,
    help="Override temperature for Step-2 only. Default: --temperature",
)
parser.add_argument(
    "--temperature_step3",
    default=None,
    type=float,
    help="Override temperature for Step-3 only. Default: --temperature",
)

parser.add_argument(
    "--top_p_step2",
    default=None,
    type=float,
    help="Override top_p for Step-2 only. Default: --top_p",
)
parser.add_argument(
    "--top_p_step3",
    default=None,
    type=float,
    help="Override top_p for Step-3 only. Default: --top_p",
)

parser.add_argument(
    "--top_k_step2",
    default=None,
    type=int,
    help="Override top_k for Step-2 only. Default: --top_k",
)
parser.add_argument(
    "--top_k_step3",
    default=None,
    type=int,
    help="Override top_k for Step-3 only. Default: --top_k",
)

parser.add_argument(
    "--repeat_penalty_step2",
    default=None,
    type=float,
    help="Override repeat_penalty for Step-2 only. Default: --repeat_penalty",
)
parser.add_argument(
    "--repeat_penalty_step3",
    default=None,
    type=float,
    help="Override repeat_penalty for Step-3 only. Default: --repeat_penalty",
)

parser.add_argument(
    "--repeat_last_n_step2",
    default=None,
    type=int,
    help="Override repeat_last_n for Step-2 only. Default: --repeat_last_n",
)
parser.add_argument(
    "--repeat_last_n_step3",
    default=None,
    type=int,
    help="Override repeat_last_n for Step-3 only. Default: --repeat_last_n",
)


parser.add_argument(
    "--frequency_penalty_step2",
    default=None,
    type=float,
    help="Override frequency_penalty for Step-2 only. Default: --frequency_penalty",
)
parser.add_argument(
    "--frequency_penalty_step3",
    default=None,
    type=float,
    help="Override frequency_penalty for Step-3 only. Default: --frequency_penalty",
)
parser.add_argument(
    "--presence_penalty_step2",
    default=None,
    type=float,
    help="Override presence_penalty for Step-2 only. Default: --presence_penalty",
)
parser.add_argument(
    "--presence_penalty_step3",
    default=None,
    type=float,
    help="Override presence_penalty for Step-3 only. Default: --presence_penalty",
)

parser.add_argument(
    "--min_p_step2",
    default=None,
    type=float,
    help="Override min_p for Step-2 only. Default: --min_p",
)
parser.add_argument(
    "--min_p_step3",
    default=None,
    type=float,
    help="Override min_p for Step-3 only. Default: --min_p",
)

parser.add_argument(
    "--max_tokens_step2",
    default=None,
    type=int,
    help="Override max_tokens for Step-2 only. Default: --max_tokens",
)
parser.add_argument(
    "--max_tokens_step3",
    default=None,
    type=int,
    help="Override max_tokens for Step-3 only. Default: --max_tokens",
)

parser.add_argument(
    "--think_mode",
    default="off",
    choices=["off", "on", "step3_only", "step2_only"],
    type=str,
    help="Thinking mode for Qwen/Ollama-compatible models: off, on, step3_only, or step2_only.",
)
parser.add_argument(
    "--ollama_chat_path",
    default="auto",
    choices=["wrapper", "native", "auto"],
    type=str,
    help=(
        "Execution path for Ollama chat calls. wrapper = always use ChatOllama/ConversationChain; "
        "native = force Ollama /api/chat; auto = use native only for Qwen-family models unless overridden by env."
    ),
)

parser.add_argument(
    "--debug_native_thinking_on_fail",
    default="on",
    choices=["off", "on"],
    type=str,
    help="When on, make a diagnostic-only native /api/chat call on retry/fallback/warning failures and save extracted thinking in debug files. This never affects the run output.",
)

parser.add_argument(
    "--theory_statement",
    default=os.environ.get("OD_THEORY_STATEMENT", ""),
    type=str,
    help="Optional claim text used to fill {THEORY_STATEMENT} in prompt templates.",
)
parser.add_argument(
    "--bias_text",
    default=os.environ.get("OD_BIAS_TEXT", ""),
    type=str,
    help="Optional bias text used to fill {BIAS} in prompt templates.",
)

parser.add_argument(
    "--llm_seed_step2",
    default=None,
    type=int,
    help="Override decoding seed for Step-2 only. Default: --llm_seed (or -seed if omitted).",
)
parser.add_argument(
    "--llm_seed_step3",
    default=None,
    type=int,
    help="Override decoding seed for Step-3 only. Default: --llm_seed (or -seed if omitted).",
)
parser.add_argument(
    "--llm_seed",
    default=None,
    type=int,
    help="Optional decoding seed for the LLM. If omitted, uses -seed value.",
)
parser.add_argument(
    "-agents",
    "--num_agents",
    default=5,
    type=int,
    help="Number of agents participating in the study",
)
parser.add_argument(
    "-steps",
    "--num_steps",
    default=5,
    type=int,
    help="Number of steps or pair samples in the experiment",
)
parser.add_argument(
    "-dist",
    "--distribution",
    default="uniform",
    choices=["uniform", "skewed_positive", "skewed_negative", "positive", "negative"],
    type=str,
    help="Type of initial opinion distribution",
)
parser.add_argument(
    "--custom_counts",
    type=str,
    default=None,
    help="Custom opinion counts for [-2,-1,0,1,2], e.g. '8,3,3,3,8'. Overrides -dist."
)

parser.add_argument(
    "-version",
    "--version_set",
    default="v61_default",
    type=str,
    help="Prompt version directory to use",
)

parser.add_argument(
    "--fact_pack_mode",
    default="off",
    choices=["off", "core", "context", "criticisms", "balanced", "full", "on"],
    type=str,
    help="Runtime fact-pack mode injected into prompts via {FACT_PACK}.",
)

parser.add_argument(
    "--world",
    "--world_mode",
    dest="world",
    default="closed",
    choices=["closed", "closed_strict", "closed_strict_rag", "open_no_rag", "open_rag", "true_open", "open"],
    type=str,
    help=(
        "World mode injected into prompts via {WORLD}/{WORLD_RULES}. "
        "closed = no outside access; "
        "closed_strict = no outside access + do not use any extra knowledge beyond current prompt/tweet; "
        "closed_strict_rag = no outside access + use retrieved snippets from a bounded local corpus only; "
        "open_no_rag = Internet/online access (no retrieval context provided); "
        "open_rag = use retrieval snippets provided by the system; "
        "true_open = Internet + you may leave home and ask people. "
        "(Legacy alias: open -> open_no_rag.)"
    ),
)

parser.add_argument(
    "--step3_open_world_rules",
    default="on",
    choices=["on", "off"],
    type=str,
    help=(
        "Controls {STEP3_OPEN_WORLD_RULES} injection in Step-3 prompts. "
        "on = inject the Step-3 open-world output rule for open-world modes; "
        "off = pass blank."
    ),
)

parser.add_argument(
    "--rag_backend",
    default="off",
    choices=["off", "simple", "dense", "graph"],
    type=str,
    help=("Retrieval backend. 'simple' = lexical overlap; 'dense' = embedding cosine "
          "(Ollama, cached); 'graph' = entity-graph chain retrieval (needs <corpus>.graph.json). "
          "dense/graph are experimental retrieval-architecture conditions (tools/retrievers.py); they use pure "
          "top-k - content-mode quotas do not apply."),
)
parser.add_argument(
    "--rag_embed_model",
    default="nomic-embed-text",
    type=str,
    help="Embedding model for rag_backend=dense (Ollama). Vectors cached next to the corpus.",
)
parser.add_argument(
    "--solo_check",
    default="off",
    choices=["off", "on"],
    type=str,
    help=("Solo model check (formerly opinion_dynamics_v3_check.py): NO personas, NO network, NO "
          "interactions - each 'agent' is an independent sample of the model answering "
          "step1_report.md about the claim. Measures the model's own prior on the topic - the "
          "baseline for cross-model comparisons. Uses the SAME native inference path and decoding options as normal runs, so "
          "solo numbers are directly comparable to run numbers. Ignores network/RAG settings."),
)
parser.add_argument(
    "--rag_corpus_path",
    default="",
    type=str,
    help="Path to local retrieval corpus (.txt, .md, .csv, .json, .jsonl).",
)
parser.add_argument(
    "--rag_top_k",
    default=4,
    type=int,
    help="Number of retrieved chunks/snippets to inject. In balanced mode, an even top_k is enforced so both sides can be represented equally.",
)
parser.add_argument(
    "--rag_query_mode",
    default="claim",
    choices=["claim", "tweet", "claim_plus_tweet"],
    type=str,
    help="Legacy global retrieval query mode. Used as fallback when step-specific modes are not set.",
)
parser.add_argument(
    "--rag_query_mode_step2",
    default="auto",
    choices=["auto", "claim", "tweet", "claim_plus_tweet"],
    type=str,
    help="How to build the retrieval query for Step-2.",
)
parser.add_argument(
    "--rag_query_mode_step3",
    default="tweet",
    choices=["auto", "claim", "tweet", "claim_plus_tweet"],
    type=str,
    help="How to build the retrieval query for Step-3.",
)
parser.add_argument(
    "--rag_content_mode",
    default="full",
    choices=["full", "balanced", "supportive_only", "criticism_only", "context_only"],
    type=str,
    help="Which subset of the retrieval corpus to use for injection.",
)
parser.add_argument(
    "--rag_topic_override",
    default="",
    type=str,
    help="Optional manual override for the RAG topic key. Blank = derive automatically from version metadata (or fall back to the prompt version root).",
)
parser.add_argument(
    "--version_metadata_path",
    default="prompts/opinion_dynamics/Flache_2017/content/version_metadata/version_metadata.json",
    type=str,
    help="Path to the master version metadata JSON. Each version root maps to one topic_key and one theory_statement.",
)
parser.add_argument(
    "--rag_max_chars",
    default=2200,
    type=int,
    help="Maximum characters of rendered retrieved context injected into prompts.",
)

parser.add_argument(
    "--web_backend",
    default="off",
    choices=["off", "duckduckgo", "brave"],
    type=str,
    help="Live web search backend for true_open world. HTML search backends: 'brave' or 'duckduckgo'.",
)
parser.add_argument(
    "--web_top_k",
    default=3,
    type=int,
    help="Maximum live web results to inject in true_open Step-3.",
)
parser.add_argument(
    "--web_max_chars",
    default=1400,
    type=int,
    help="Maximum characters of rendered live-web context injected into true_open prompts.",
)
parser.add_argument(
    "--step2_web_mode",
    default="off",
    choices=["off", "heuristic", "always"],
    type=str,
    help="Live web search policy for true_open Step-2. Queries are always built from the claim plus the agent's current belief / stance hint, not from previous tweets.",
)
parser.add_argument(
    "--planner_mode",
    default="off",
    choices=["off", "heuristic"],
    type=str,
    help="Observable tool planner mode. heuristic = rule-based Step-3 tool gating in true_open.",
)
parser.add_argument(
    "--tool_mode",
    default="off",
    choices=["off", "web_only", "multi"],
    type=str,
    help="Tool mode for true_open. web_only = live search only; multi = live search plus notes tool.",
)
parser.add_argument(
    "--notes_mode",
    default="off",
    choices=["off", "heuristic"],
    type=str,
    help="External notes tool mode for true_open multi-tool runs. Notes are separate from ConversationChain memory.",
)
parser.add_argument(
    "--notes_max_items",
    default=3,
    type=int,
    help="Maximum recent external notes to inject into true_open Step-3 prompts.",
)

parser.add_argument(
    "--memory",
    default="enabled",
    choices=["enabled", "light", "off"],
    type=str,
    help="Agent memory mode: enabled keeps the full running memory of seen/written tweets and received responses; light keeps a compact recent memory with a belief path plus a truncated recent history; off clears/pops memory so each decision uses only the current tweet + prompt.",
)

parser.add_argument(
    "--persona_mode",
    default="full",
    choices=["full", "none"],
    type=str,
    help="Persona detail level. full uses the whole persona card (occupation, "
         "politics, background, behavioural layers). none blanks every persona "
         "attribute except the agent name (initial belief still comes from the "
         "opinion distribution), making agents homogeneous - a control for how "
         "much of the dynamics is driven by persona heterogeneity.",
)

parser.add_argument(
    "--allow_silence",
    default="off",
    choices=["off", "on"],
    type=str,
    help="ADR-006 Component 2 (silence-as-choice, feeds P12). off (default) = "
         "byte-identical with pre-ADR-006 behavior: the step2 template renders "
         "without any silence-option instructions. on = a mechanism-agnostic "
         "block is injected that lets the agent return the exact token "
         "<silent> in place of a tweet. The Task 2.3b parser dispatch and "
         "silence-log wiring (deferred to a follow-up task) is what actually "
         "records the event; with just the flag on, a <silent> response from "
         "the model would currently fall through to the existing parse-fail "
         "path.",
)

# --- ADR-006 Component 3: p_reach pluggable policy pattern (feeds P21, P30) ---
parser.add_argument(
    "--p_reach_policy",
    default="uniform",
    choices=["uniform", "homophilic", "shadowban"],
    type=str,
    help="ADR-006 Component 3. Policy for assigning reach probability to "
         "directed edges (src -> dst): whether the tweet from src actually "
         "reaches dst. uniform (default 1.0) = byte-identical baseline. "
         "homophilic = sigmoid(k * (1 - 2 * normalized_distance)); mimics "
         "engagement-based amplification (P30 Cinelli). shadowban = a random "
         "subset of agents get low outgoing reach; mimics content moderation "
         "(P21).",
)
parser.add_argument(
    "--p_reach_uniform_value",
    default=1.0,
    type=float,
    help="ADR-006 Component 3, uniform policy. Every edge gets this p_reach. "
         "Default 1.0 preserves byte-identical baseline; dose-response sweep "
         "{1.0, 0.75, 0.5, 0.25, 0.1} isolates the effect of reach dilution.",
)
parser.add_argument(
    "--p_reach_homophily_k",
    default=2.0,
    type=float,
    help="ADR-006 Component 3, homophilic policy. Sigmoid sharpness. Higher "
         "k -> steeper similar/dissimilar contrast. Default 2.0.",
)
parser.add_argument(
    "--p_reach_shadowban_fraction",
    default=0.1,
    type=float,
    help="ADR-006 Component 3, shadowban policy. Fraction of agents to "
         "penalise (their outgoing edges get shadowban_value). Default 0.1.",
)
parser.add_argument(
    "--p_reach_shadowban_value",
    default=0.1,
    type=float,
    help="ADR-006 Component 3, shadowban policy. Reach probability for the "
         "penalised agents' outgoing edges. Default 0.1.",
)
parser.add_argument(
    "--p_reach_enforcement",
    default="filter",
    choices=["filter", "suppress"],
    type=str,
    help="ADR-006 Component 3, roads_not_taken 3.11 (fix gamma). HOW a sub-1.0 "
         "p_reach is enforced. filter (default, byte-identical): a candidate "
         "speaker is dropped from the listener's candidate set with prob "
         "1 - p_reach (Bernoulli on the MAIN rng) -> a throttled speaker is "
         "rarely CHOSEN (models reduced feed surfacing, but its rng draws make "
         "single-seed uniform-vs-treatment comparisons diverge). suppress "
         "(gamma, skip-preserving): candidates are NOT filtered, so the speaker "
         "is chosen exactly as in a uniform baseline (identical interaction "
         "sequence); if the chosen speaker is throttled, a DEDICATED-rng "
         "Bernoulli decides whether the tweet lands, and on failure the "
         "listener's belief update is not applied (models de-ranked/ignored "
         "delivery). suppress yields a clean single-seed causal contrast because "
         "the main rng stream is untouched.",
)

# --- ADR-006 Component 4: Bian 5-dim diagnostic opt-in (feeds P28) ---
parser.add_argument(
    "--include_bian_scores",
    default="off",
    choices=["off", "on"],
    type=str,
    help="ADR-006 Component 4. When on, run tools/bian_diagnostic.py for the "
         "model of this run (cached per model under ~/.claude_bian_cache) and "
         "copy the 5 validity scores into run_metrics.json. off (default) = "
         "no-op, byte-identical baseline.",
)

parser.add_argument(
    "--trace",
    default="auto",
    choices=["auto", "off", "minimal", "full"],
    type=str,
    help="Terminal tracing: auto=minimal when --memory off, full when --memory enabled. "
         "minimal prints only the current prompt + output; full also prints stored history.",
)

parser.add_argument(
    "--llm_history",
    default="off",
    choices=["off", "on", "auto"],
    type=str,
    help="Whether past conversation history is injected into the LLM prompt. "
         "off enforces Markovian decisions (state = current belief + current tweet), even if --memory is enabled. "
         "auto=on when --memory enabled.",
)

parser.add_argument(
    "-seed",
    "--seed",
    default=1,
    type=int,
    help="Random seed for agent selection and interactions.",
)

parser.add_argument(
    "--epsilon_uniform",
    default=0.02,
    type=float,
    help="Exploration rate for neighbor selection (probability of choosing a random neighbor regardless of score).",
)
parser.add_argument(
    "--interaction_selection",
    default="homophily",
    choices=["random", "homophily"],
    type=str,
    help="How the listener chooses a partner: random (uniform) or homophily (score-weighted). Works for both network and fully-mixed (no network).",
)
parser.add_argument(
    "--interaction_homophily_mode",
    default="full",
    choices=["full", "opinion_only"],
    type=str,
    help="If interaction_selection=homophily: 'full' uses opinion+persona scoring; 'opinion_only' uses only opinion similarity (same mapping as the full score's opinion component).",
)


# --- Event injection. All default-off: with --event_step unset (<=0)
#     no event fires and the run is byte-identical. ---
parser.add_argument(
    "--event_step", default=-1, type=int,
    help="1-indexed step at which a one-off event is broadcast to every agent. "
         "<=0 disables it. Fire it AFTER opinions have settled, or it perturbs a "
         "still-moving trajectory and measures nothing.",
)
parser.add_argument(
    "--event_text", default="", type=str,
    help="The event every agent reacts to, e.g. 'NASA announced the Moon landings "
         "were staged'. Empty disables the event even if --event_step is set.",
)
parser.add_argument(
    "--event_persist", default="reaction", choices=["reaction", "headline", "both"], type=str,
    help="What stays in memory after the event. reaction=impulse (ages out, "
         "measures recovery/hysteresis); headline=step change (pinned, never ages "
         "out, measures a new equilibrium); both=event+reaction pinned.",
)

# --- Coevolving network. All default-off: with --network_evolves
#     absent the run is byte-identical to a static-network run. ---
parser.add_argument(
    "--network_evolves",
    action="store_true",
    help="Let the network rewire during the run (add one edge, cut one edge per "
         "eligible interaction, so the edge count is conserved). Off by default; "
         "when off the topology is static exactly as before.",
)
parser.add_argument(
    "--evolve_directed",
    action="store_true",
    help="With --network_evolves: treat follows as one-way (Twitter-style, "
         "in-degree != out-degree). Default is reciprocal (Facebook-style), "
         "which reproduces the symmetric static behaviour.",
)
parser.add_argument(
    "--evolve_burnin", default=50, type=int,
    help="Steps before rewiring starts, so it acts on formed opinions not the seed.",
)
parser.add_argument(
    "--evolve_p_add", default=0.07, type=float,
    help="Per-eligible-interaction probability of attempting a friend-of-friend add.",
)
parser.add_argument(
    "--evolve_add_threshold", default=100.0, type=float,
    help="Min tie score to add. On the opinion-only scale (0..100) 100 means "
         "'agree exactly'. Lower to let near-agreement form ties.",
)
parser.add_argument(
    "--evolve_cut_threshold", default=25.0, type=float,
    help="Max tie score to cut. On the opinion-only scale a tie at opinion "
         "distance >=2 scores <=25, so 25 cuts those.",
)
parser.add_argument(
    "--evolve_soft_cut_distance", default=2, type=int,
    help="Min opinion distance for a tie to be cuttable.",
)
parser.add_argument(
    "--evolve_min_out_degree", default=2, type=int,
    help="Never cut an agent's last-but-one out-edge; keeps everyone able to hear "
         "someone. Note: in Barabasi-Albert with m=2 many nodes start at degree 2, "
         "so a value of 2 blocks most cuts - lower it or raise m to see rewiring.",
)

parser.add_argument(
    "--max_step_change",
    default=1,
    type=int,
    help="Maximum absolute belief change allowed per interaction (applied after Step-3 rating is parsed).",
)

parser.add_argument(
    "--allowed_update_mode",
    default="assimilation_only",
    choices=["assimilation_only", "free_bounded", "free_brounded"],
    type=str,
    help="Belief-update constraint. assimilation_only keeps the old non-away-from-speaker allowed set. "
         "free_bounded allows one bounded step in either direction within max_step_change; "
         "same-rating interactions can also change. free_brounded is accepted as a typo alias.",
)
parser.add_argument(
    "--validation_strictness",
    default="strict",
    choices=["strict", "warn_only", "format_only"],
    type=str,
    help="How hard the content validators enforce. strict = current behaviour (content retries in Step-2 and "
         "content rejections in Step-3). warn_only / format_only turn OFF content enforcement so the model's own "
         "wording stands; FINAL_RATING format and the allowed-ratings / max-step-change rules are always enforced. "
         "Use non-strict to compare different models without the filters homogenising their outputs.",
)
parser.add_argument(
    "--wrong_side_explanation_requery",
    default="off",
    choices=["off", "on"],
    type=str,
    help="When on, a Step-3 explanation whose side contradicts the (already valid) FINAL_RATING triggers ONE "
         "explanation-only re-query with the rating held fixed (no rating guard runs). Closed-world guarded; if the "
         "re-query is still wrong-side, a fragment, or adds outside evidence, the deterministic rewrite is used. "
         "Explanations are transcript-only, so this does not change B/D/P; it only makes wrong-side transcripts genuine.",
)
parser.add_argument(
    "--structured_output",
    choices=["off", "on"],
    default="off",
    help=(
        "Experimental guardrail: when on, step2/step3 native calls request grammar-constrained "
        "JSON from Ollama (format=json) and the reply is converted back to the canonical "
        "FINAL_RATING/TWEET|EXPLANATION text before any parsing. The model can then only emit "
        "valid structure, so post-hoc format repairs (and their selection bias) shrink. "
        "Default off; treat as an experimental condition. May reduce thinking-token usage on "
        "thinking models because format constraints apply to the whole output."
    ),
)
parser.add_argument(
    "--deterministic",
    default="off",
    choices=["off", "on"],
    type=str,
    help="When on, forces greedy decoding (temperature 0, top_k 1) for all native LLM calls, overriding the "
         "per-step temperatures. With a fixed seed this makes runs reproducible, so an A/B difference reflects "
         "the variable under test rather than sampling noise.",
)

parser.add_argument(
    "--same_side_edge_unlock_hits",
    default=2,
    type=int,
    help="If a +1 (or -1) agent is hit by the same mild side and stays unchanged this many times, unlock +2 (or -2) on future same mild-side hits. Set <=0 to disable.",
)

parser.add_argument(
    "--same_rating_step3_mode",
    default="skip_tweet_local",
    choices=["skip_tweet_local", "skip_generic", "llm"],
    type=str,
    help="How Step-3 handles singleton same-rating interactions after all unlock/expansion logic has run: skip_tweet_local (default), skip_generic, or llm.",
)

# -----------------------------
# NETWORK (structure)
# -----------------------------
parser.add_argument(
    "--network_type",
    default="none",
    choices=["ws", "er", "ba", "none"],
    type=str,
    help="Network structure: ws = Watts–Strogatz small-world, er = Erdős–Rényi, ba = Barabási–Albert, none = fully-mixed.",
)
parser.add_argument(
    "--network_homophily",
    action="store_true",
    help="If set (and network_type != none), relabel the initial network so similar agents are closer in the starting structure.",
)
parser.add_argument(
    "--k_neighbors",
    default=4,
    type=int,
    help="WS parameter k (must be even and < num_agents). Each node connects to k neighbors in the initial ring lattice.",
)
parser.add_argument(
    "--p_rewire",
    default=0.1,
    type=float,
    help="WS rewiring probability p (0..1).",
)
parser.add_argument(
    "--er_p_edge",
    default=0.15,
    type=float,
    help="ER edge probability p (0..1).",
)
parser.add_argument(
    "--ba_m_attach",
    default=2,
    type=int,
    help="BA parameter m: edges attached by each new node (must be > 0 and < num_agents).",
)
parser.add_argument(
    "--ba_hub_strategy",
    default="default",
    choices=["default", "random", "positive", "negative", "neutral", "extreme", "opposite_majority", "minority", "custom"],
    type=str,
    help=(
        "BA hub-priority strategy. In BA networks, early attachment positions usually become hubs. "
        "This option maps selected agents onto those early positions so a chosen side can receive structural visibility."
    ),
)
parser.add_argument(
    "--ba_hub_assignment_mode",
    default="early_position",
    choices=["early_position", "actual_hubs", "early_and_actual"],
    type=str,
    help=(
        "BA hub assignment mode. early_position gives selected agents the original early-node preferential-attachment advantage; "
        "actual_hubs assigns selected agents to the realized highest-degree BA nodes; "
        "early_and_actual starts with early priority and then also forces selected agents onto realized top hubs."
    ),
)
parser.add_argument(
    "--ba_hub_custom",
    default="",
    type=str,
    help=(
        "Custom BA hub-priority agents when --ba_hub_strategy=custom. "
        "Accepts local indices like idx:0, agent ids like agent_id:12, or agent names, separated by commas."
    ),
)

parser.add_argument("-test", "--test_run", action="store_true", help="Set flag if test run")
parser.add_argument("--no_rating", action="store_true", help="Set flag if prompt is no rating")
parser.add_argument(
    "-out", "--out", "--output_file", dest="output_file", type=str, default="test1.csv", help="Name of the output file"
)
args = parser.parse_args()
DEFAULT_MAX_STEP_CHANGE = args.max_step_change
ALLOWED_UPDATE_MODE = str(getattr(args, "allowed_update_mode", "assimilation_only") or "assimilation_only").strip().lower()
VALIDATION_STRICTNESS = str(getattr(args, "validation_strictness", "strict") or "strict").strip().lower()
if VALIDATION_STRICTNESS not in {"strict", "warn_only", "format_only"}:
    VALIDATION_STRICTNESS = "strict"


def _content_enforcement_enabled() -> bool:
    """True only in 'strict' mode. When False (warn_only / format_only), content
    validators return a clean result, so they neither retry Step-2 nor reject
    Step-3. Format checks and the allowed-ratings / max-step-change rules are
    always enforced regardless of this setting."""
    return str(globals().get("VALIDATION_STRICTNESS", "strict") or "strict").strip().lower() == "strict"

WRONG_SIDE_EXPL_REQUERY = str(getattr(args, "wrong_side_explanation_requery", "off") or "off").strip().lower()
if WRONG_SIDE_EXPL_REQUERY not in {"off", "on"}:
    WRONG_SIDE_EXPL_REQUERY = "off"


def _wrong_side_explanation_requery_enabled() -> bool:
    """True when the optional 1-shot explanation-only re-query is enabled. It never
    changes the rating (rating stays fixed); it only replaces a wrong-side explanation
    with a genuine model explanation, closed-world guarded."""
    return str(globals().get("WRONG_SIDE_EXPL_REQUERY", "off") or "off").strip().lower() == "on"

DETERMINISTIC = str(getattr(args, "deterministic", "off") or "off").strip().lower()
if DETERMINISTIC not in {"off", "on"}:
    DETERMINISTIC = "off"
STRUCTURED_OUTPUT = str(getattr(args, "structured_output", "off") or "off").strip().lower()
if STRUCTURED_OUTPUT not in {"off", "on"}:
    STRUCTURED_OUTPUT = "off"


def _structured_output_enabled() -> bool:
    """Return whether step2/step3 native calls should use JSON-constrained output."""
    return STRUCTURED_OUTPUT == "on"


def _structured_json_to_canonical(raw_text: str, step: str):
    """Convert a structured JSON reply back to the canonical text format.

    step2 expects {"final_rating": int, "tweet": str}       -> FINAL_RATING/TWEET text
    step3 expects {"final_rating": int, "explanation": str} -> FINAL_RATING/EXPLANATION text
    Returns None (caller keeps the raw text) when parsing fails, and counts the failure
    so structured-output reliability is measurable per run.
    """
    import json as _json
    import re as _re
    s = str(raw_text or "").strip()
    if not s:
        return None
    s = _re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()   # tolerate code fences
    try:
        obj = _json.loads(s)
    except Exception:
        try:
            m = _re.search(r"\{.*\}", s, _re.S)
            obj = _json.loads(m.group(0)) if m else None
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        try:
            RUN_METRICS["structured_output_parse_fail_total"] += 1
        except Exception:
            pass
        return None
    rating = obj.get("final_rating", obj.get("rating"))
    body_key = "tweet" if str(step).strip().lower() == "step2" else "explanation"
    body = obj.get(body_key) or obj.get("text") or ""
    try:
        rating = int(rating)
    except Exception:
        try:
            RUN_METRICS["structured_output_parse_fail_total"] += 1
        except Exception:
            pass
        return None
    label = "TWEET" if body_key == "tweet" else "EXPLANATION"
    try:
        RUN_METRICS["structured_output_converted_total"] += 1
    except Exception:
        pass
    return "FINAL_RATING: " + str(rating) + "\n" + label + ": " + str(body).strip()


def _deterministic_mode_enabled() -> bool:
    """True when greedy/deterministic decoding is forced (temperature 0, top_k 1)."""
    return str(globals().get("DETERMINISTIC", "off") or "off").strip().lower() == "on"
if ALLOWED_UPDATE_MODE == "free_brounded":
    ALLOWED_UPDATE_MODE = "free_bounded"
if ALLOWED_UPDATE_MODE not in {"assimilation_only", "free_bounded"}:
    ALLOWED_UPDATE_MODE = "assimilation_only"
SAME_SIDE_EDGE_UNLOCK_HITS = getattr(args, "same_side_edge_unlock_hits", 0)
if ALLOWED_UPDATE_MODE == "free_bounded":
    # In free-bounded mode the ordinary allowed set already includes edge moves
    # when max_step_change permits them, so the same-side unlock mechanism would
    # double-count a constraint that is intentionally disabled.
    SAME_SIDE_EDGE_UNLOCK_HITS = 0
SAME_RATING_STEP3_MODE = str(getattr(args, "same_rating_step3_mode", "skip_tweet_local") or "skip_tweet_local").strip().lower()
OLLAMA_CHAT_PATH_MODE = str(getattr(args, "ollama_chat_path", "wrapper") or "wrapper").strip().lower()
DEBUG_NATIVE_THINKING_ON_FAIL = str(getattr(args, "debug_native_thinking_on_fail", "off") or "off").strip().lower() == "on"

# ============================
# AGENT MEMORY (FULL vs LIGHT vs OFF)
# ============================
MEMORY_MODE = str(getattr(args, "memory", "enabled") or "enabled").strip().lower()
if MEMORY_MODE not in {"enabled", "light", "off"}:
    MEMORY_MODE = "enabled"
MEMORY_ENABLED = MEMORY_MODE != "off"
MEMORY_LIGHT_ENABLED = MEMORY_MODE == "light"
PERSONA_MODE = str(getattr(args, "persona_mode", "full") or "full").strip().lower()
if PERSONA_MODE not in {"full", "none"}:
    PERSONA_MODE = "full"
# ADR-006 Component 2: silence-as-choice flag. Default "off" preserves
# byte-identical baseline. The step2 template loader (below) always calls
# render_step2_template(..., allow_silence=ALLOW_SILENCE), and the renderer's
# off-branch strips the ALLOW_SILENCE_BLOCK marker with its surrounding blank
# lines - i.e. the rendered text equals what the raw template produced before
# Task 2.1 inserted the marker.
ALLOW_SILENCE = str(getattr(args, "allow_silence", "off") or "off").strip().lower() == "on"

# These rules are injected only when stored memory is exposed to the LLM.
# They are prepended at memory-render time, not saved into memory, so they do not
# accumulate inside the agent's own remembered history.
STEP2_MEMORY_USE_RULES = """
Memory use rule:
- Earlier tweets are context, not text to copy.
- Do not repeat the same wording just because it appears in memory.
- Use memory to maintain continuity with your present view, but write a fresh tweet around one concrete point.
- Your tweet must still reflect your current PRESENT VIEW.
""".strip()

STEP3_MEMORY_USE_RULES = """
Memory use rule:
- The recent belief path and earlier exchanges are context, not new evidence.
- Do not keep a rating merely because the same rating appears repeatedly in memory.
- Use memory to remember prior concrete points, but judge the current tweet on its own concrete point.
- If the current tweet gives a clearer reason to shift, update normally within the allowed rating set.
""".strip()

def _memory_use_rules_for_step(step_kind: str) -> str:
    """
    Return memory-use instructions only when memory is visible to the model.
    
        The simulator can store memory for logs while still running Markovian LLM calls. This helper
        prevents memory instructions from appearing when `llm_history` is disabled.
    """
    try:
        if not bool(MEMORY_ENABLED and LLM_HISTORY_ENABLED):
            return ""
    except Exception:
        return ""
    step = str(step_kind or "").strip().lower()
    if step == "step2":
        return STEP2_MEMORY_USE_RULES
    if step == "step3":
        return STEP3_MEMORY_USE_RULES
    return ""

# Light-memory compression keeps a compact trajectory + a short recent episodic window.
LIGHT_MEMORY_BELIEF_PATH_LEN = 5
LIGHT_MEMORY_MAX_EVENTS = 3
LIGHT_MEMORY_EXTRA_OWN_TWEETS = 2

try:
    print(f"[config] memory={MEMORY_MODE}")
except Exception:
    pass


# ============================
# TERMINAL TRACE / LLM HISTORY
# ============================
_TRACE_RAW = str(getattr(args, "trace", "auto")).strip().lower()
if _TRACE_RAW == "auto":
    TRACE_MODE = "full" if MEMORY_ENABLED else "minimal"
else:
    TRACE_MODE = _TRACE_RAW  # off|minimal|full

_LLMH_RAW = str(getattr(args, "llm_history", "off")).strip().lower()
if _LLMH_RAW == "auto":
    LLM_HISTORY_ENABLED = bool(MEMORY_ENABLED)
else:
    LLM_HISTORY_ENABLED = (_LLMH_RAW == "on")

try:
    print(f"[config] trace={TRACE_MODE} llm_history={'on' if LLM_HISTORY_ENABLED else 'off'} trace_raw={_TRACE_RAW} llm_history_raw={_LLMH_RAW} memory={MEMORY_MODE}")
except Exception:
    pass

try:
    print(f"[config] ollama_chat_path={OLLAMA_CHAT_PATH_MODE}")
except Exception:
    pass

try:
    print(f"[config] debug_native_thinking_on_fail={'on' if DEBUG_NATIVE_THINKING_ON_FAIL else 'off'}")
except Exception:
    pass

_WORLD_RAW_FOR_TOOLS = str(getattr(args, "world", "closed") or "closed").strip().lower()
if _WORLD_RAW_FOR_TOOLS == "open":
    _WORLD_RAW_FOR_TOOLS = "open_no_rag"
TRUE_OPEN_WORLD_ENABLED = _WORLD_RAW_FOR_TOOLS == "true_open"
WEB_BACKEND = str(getattr(args, "web_backend", "off") or "off").strip().lower()
PLANNER_MODE = str(getattr(args, "planner_mode", "off") or "off").strip().lower()
TOOL_MODE = str(getattr(args, "tool_mode", "off") or "off").strip().lower()
NOTES_MODE = str(getattr(args, "notes_mode", "off") or "off").strip().lower()
WEB_TOP_K = max(1, int(getattr(args, "web_top_k", 3) or 3))
WEB_MAX_CHARS = max(200, int(getattr(args, "web_max_chars", 1400) or 1400))
STEP2_WEB_MODE = str(getattr(args, "step2_web_mode", "off") or "off").strip().lower()
NOTES_MAX_ITEMS = max(1, int(getattr(args, "notes_max_items", 3) or 3))
TRUE_OPEN_WEB_ACTIVE = bool(TRUE_OPEN_WORLD_ENABLED and WEB_BACKEND != "off" and TOOL_MODE in {"web_only", "multi"})
TRUE_OPEN_NOTES_ACTIVE = bool(TRUE_OPEN_WORLD_ENABLED and TOOL_MODE == "multi" and NOTES_MODE != "off")

try:
    print(f"[config] true_open_tools web_backend={WEB_BACKEND} planner={PLANNER_MODE} tool_mode={TOOL_MODE} notes_mode={NOTES_MODE} step2_web_mode={STEP2_WEB_MODE}")
except Exception:
    pass



# ============================
# WORLD MODE / FACT PACK (runtime content)
# ============================

STEP2_OPEN_WORLD_STYLE_RULES_GENERIC = """
Open-world Step-2 style addition:
- Sound like a person making one pointed observation, not like a debate summary.
- Prefer one concrete point over a vague summary when possible.
- If you are mixed, make the two sides of the uncertainty feel specific rather than template-like.
""".strip()


STEP2_OPEN_WORLD_ZERO_STYLE_RULES_GENERIC = """
Open-world Step-2 addition for PRESENT VIEW 0:
- For 0, keep both clauses moderate; do not make one side sound settled and the other side tentative.
- Avoid strongly conclusive words on either side.
- Prefer one concrete point pulling toward the claim and one concrete unresolved point pulling away from it.
- Avoid generic unresolved wording when you can name the exact unresolved point.
""".strip()


STEP2_OPEN_WORLD_MILD_STYLE_RULES_GENERIC = """
Open-world Step-2 addition for PRESENT VIEW -1 or 1:
- For -1 or 1, sound mildly on that side, not like a full -2 or +2 declaration.
- Prefer one concrete point over a generic fallback line when you can name one sharper point.
- Avoid vague mild formulations when you can name the precise unresolved detail or specific support.
- Hedge naturally when it fits a mild stance, with wording like "I think", "I lean", or "it seems", but keep the tweet claim-facing.
""".strip()


STEP2_OPEN_WORLD_STRONG_STYLE_RULES_GENERIC = """
Open-world Step-2 addition for PRESENT VIEW -2 or 2:
- For -2 or 2, sound firm, but still prefer one concrete point over a broad slogan.
- Avoid all-purpose summary lines when you can name one sharper reason.
""".strip()

STEP3_OPEN_WORLD_RULES_GENERIC = """
Open-world addition:
- You may use broader recall or shown material only when it directly clarifies the specific point raised in the tweet.
- Do not introduce a separate outside rebuttal when the tweet does not raise that point.
- Do not replace the tweet's point with a generic verdict about the whole claim.
- If the tweet gives one concrete point, answer that point; do not switch to a different shown result.
- Keep EXPLANATION tied to one concrete tweet-local or shown point.
- Generic evidence language (for example "global consensus," "overwhelming evidence," or "documented fact") is allowed, but if the tweet raises one concrete anomaly or gap, EXPLANATION must answer that point directly or explain why that point is too weak; do not use generic support language as a substitute for the local hinge.
""".strip()

STEP2_FACT_PACK_RULES_GENERIC = """
Fact-pack addition:
- Supportive points count in favor of the claim; criticism points count against; contextual points provide context only.
- You may use one concrete point from the fact pack if it fits your present view; do not summarize the whole pack.
- Stay close to the wording of the shown point you use.
- Do not mention the fact pack itself in the tweet.
""".strip()

STEP3_FACT_PACK_RULES_GENERIC = """
Fact-pack evaluation addition:
- Supportive points count in favor of the claim; criticism points count against; contextual points are context only.
- Judge the tweet's point first; use fact-pack material only to clarify or limit that point.
- Do not let fact-pack material decide the final rating by itself.
- If you stay against a claim-challenging point, name the specific weakness in that point itself.
- Keep EXPLANATION tied to one concrete point from the tweet or shown material.
""".strip()

STEP2_CRITICISM_ONLY_RULES_GENERIC = """
Criticism-only addition:
- When only criticism material is shown, treat it as relevant shown material that counts against the claim in direction.
- Use criticism-only material only as claim-challenging material for the tweet, not as support for the claim itself.
- Do not flip a criticism into positive proof or into a reason that the claim is true.
- If the shown criticism points against your PRESENT VIEW, you may still keep your current side, but do not describe the criticism itself as supportive evidence for your side.
- If you resist the shown criticism, name the specific limitation, gap, or unresolved weakness in that criticism itself.
- If your PRESENT VIEW is 0, use the criticism as the side pulling against the claim rather than as a neutral discussion point.
""".strip()

STEP3_CRITICISM_ONLY_RULES_GENERIC = """
Criticism-only evaluation addition:
- When only criticism material is shown, treat it as relevant shown material that weighs against the claim in direction.
- Judge how strongly the tweet challenges the claim, not whether the criticism can be turned into support for the claim.
- Do not answer a criticism by reinterpreting that same criticism as positive proof that the claim is true.
- A criticism-only point may be weak, moderate, or strong, but its direction is against the claim.
- If you do not move, explain why the criticism is not strong enough.
- If you resist the criticism, explain the specific weakness, gap, or limitation in that criticism itself.
""".strip()

STEP2_RAG_RULES_GENERIC = """
Retrieved-context addition:
- Supportive points count in favor of the claim; criticism points count against; contextual points provide context only.
- You may use one concrete point from the retrieved material if it fits your present view; do not summarize the whole pack.
- Stay close to the wording of the shown point you use.
- Do not mention retrieval, snippets, sources, or the prompt itself in the tweet.
""".strip()

STEP3_RAG_RULES_GENERIC = """
Retrieved-context evaluation addition:
- Supportive points count in favor of the claim; criticism points count against; contextual points are context only.
- Judge the tweet's point first; use retrieved material only to clarify or limit that point.
- Do not let retrieved material decide the final rating by itself.
- If you stay against a claim-challenging point, name the specific weakness in that point itself.
- Keep EXPLANATION tied to one concrete point from the tweet or shown material.
- Do not mention retrieval, snippets, sources, or the prompt itself in the tweet.
""".strip()

STEP2_MISMATCHED_SHOWN_MATERIAL_RULES_GENERIC = """
Mismatched-shown-material rule:
- If the shown material points in a different direction from your PRESENT VIEW, you may still keep your current side by naming the specific limitation in that shown point.
- Do not leave the tweet empty just because the shown point leans the other way.
- Do not turn a criticism into positive proof, and do not turn supportive material into negative proof.
- State the claim-facing point directly, not your update process.
""".strip()

STEP2_NO_FACT_PACK_CLOSED_RULES_GENERIC = """
Closed-world, no extra material:
- Use only the claim wording, any shown tweet, and your persona.
- Persona may shape what feels plausible but does not count as unseen evidence.
- Treat broad public validation (records, experts, consensus, "the science") as unsupported on both sides unless explicitly shown.
- Keep the tweet specific and local; avoid empty boilerplate like "no evidence" or "unclear" unless you name the concrete gap.
""".strip()

STEP2_STRICT_CLOSED_CASE_RULES_GENERIC = """
Strict closed-world Step-2 addition:
- Stay within the claim, any shown tweet, your persona, and any material explicitly shown in this prompt.
- Do not use outside facts, public knowledge, or broad validation unless that material is explicitly shown.
- If shown material is present, use only one concrete shown point and do not treat it as automatically decisive.
- Keep the tweet specific and local; avoid empty boilerplate like "no evidence" or "unclear" unless you name the concrete gap.
""".strip()

STEP3_NO_FACT_PACK_CLOSED_RULES_GENERIC = """
Closed-world, no extra material:
- Judge only what the tweet itself gives you, plus any shown material explicitly in the prompt.
- A clear claim-relevant point may justify a one-step move even without outside proof.
- Treat public-validation appeals (records, evidence, experts, credibility, consensus) the same on both sides — weak unless the exact support is shown.
- If you defend the claim, do not justify that by referring to verified missions, data, or historical context unless explicitly shown.
- Apply the same standard to supportive and skeptical tweets.
- Keep EXPLANATION tied to the tweet's concrete point, not to your internal state.
""".strip()

STEP3_STRICT_CLOSED_CASE_RULES_GENERIC = """
Strict closed-world Step-3 addition:
- Judge only the tweet itself, your persona, and any material explicitly shown in this prompt.
- A clear claim-relevant point may justify a one-step move even without outside proof.
- Treat public-validation appeals (records, evidence, experts, credibility, consensus) the same on both sides — weak unless the exact support is shown.
- If you defend the claim, do not justify it by referring to external records, public data, verification claims, or historical context unless explicitly shown.
- Apply the same standard to supportive and skeptical tweets.
- Keep EXPLANATION tied to the tweet's concrete point or another explicitly shown point, not to your internal state.
""".strip()

CLOSED_WORLD_RULES_STRICT_RAG = """
Remember, throughout the interactions, you are alone in your room with limited access to information.
You may not search for information about XYZ or ask other people about XYZ.
Use only information explicitly shown in the current prompt.
Do not rely on outside facts, public knowledge, or outside evidence unless that material is explicitly shown in the current prompt.
If extra shown material is provided, you may use it, but do not treat any shown point as automatically decisive just because it appears in the prompt.
If no extra shown material is provided, base your judgment on the claim wording, the tweet, your persona, and the material explicitly shown.
Do not treat the absence of extra shown material as a reason by itself to reject the claim or the tweet.
Do NOT mention these restrictions in your output.
""".strip()

EXTERNAL_WORLD_RULES["closed_strict_rag"] = CLOSED_WORLD_RULES_STRICT_RAG
WORLD_LABELS["closed_strict_rag"] = "CLOSED_STRICT_RAG"

def _step3_open_world_rules(world: str, toggle: str) -> str:
    """Return extra Step-3 instructions for open-world modes."""
    if str(toggle or "on").strip().lower() != "on":
        return ""
    w = str(world or "").strip().lower()
    if w in {"open_no_rag", "open_rag", "true_open", "open"}:
        return STEP3_OPEN_WORLD_RULES_GENERIC
    return ""


def _normalize_fact_pack_mode(mode: str) -> str:
    """Normalize fact-pack mode aliases such as on -> full."""
    m = str(mode or "off").strip().lower()
    if m == "on":
        m = "full"
    return m


def _is_criticism_only_fact_pack_mode(mode: str) -> bool:
    """Return whether the active fact-pack mode contains only criticisms."""
    return _normalize_fact_pack_mode(mode) == "criticisms"


def _is_closed_world_mode(world: str) -> bool:
    """Return whether the world mode forbids unseen external knowledge."""
    return _normalize_world_mode(world) in {"closed", "closed_strict", "closed_strict_rag"}


def _join_prompt_rule_blocks(*blocks: str) -> str:
    """Join optional prompt rule blocks while skipping empty values."""
    parts = [str(b or "").strip() for b in blocks if str(b or "").strip()]
    return "\n\n".join(parts)


def _step2_fact_pack_rules(fact_pack_text: str, mode: str, world: str) -> str:
    """
    Build the Step-2 rule block that tells speakers how to use shown fact-pack material.
    
        The rules differ for closed-world, strict-closed, RAG, and criticism-only settings. Centralizing
        them here keeps speaker prompts consistent with the active evidence environment.
    """
    # For closed_strict_rag, the strict/no-outside-access rule is already supplied by
    # CLOSED_WORLD_RULES_STRICT_RAG, and RAG is extra shown material. Do not append the
    # no-extra-material closed-case block on RAG prompts.
    world_norm = _normalize_world_mode(world)
    strict_extra = STEP2_STRICT_CLOSED_CASE_RULES_GENERIC if world_norm == "closed_strict" else ""
    rag_active = _rag_enabled_for_world(world)
    parts = []
    if str(fact_pack_text or "").strip():
        parts.append(STEP2_FACT_PACK_RULES_GENERIC)
        if _is_criticism_only_fact_pack_mode(mode):
            parts.append(STEP2_CRITICISM_ONLY_RULES_GENERIC)
    elif _is_closed_world_mode(world) and not rag_active:
        parts.append(STEP2_NO_FACT_PACK_CLOSED_RULES_GENERIC)
    # RAG rules are injected only through the explicit {STEP2_RAG_RULES} placeholder.
    if strict_extra:
        parts.append(strict_extra)
    return _join_prompt_rule_blocks(*parts)


def _step3_fact_pack_rules(fact_pack_text: str, mode: str, world: str) -> str:
    """
    Build the Step-3 rule block that tells listeners how to evaluate shown fact-pack material.
    
        Listener prompts need stronger safeguards than speaker prompts because final ratings are updated
        here. The generated block clarifies whether shown material is allowed and whether outside evidence
        is forbidden.
    """
    rag_active = _rag_enabled_for_world(world)
    # For closed_strict_rag, RAG snippets are extra shown material, so avoid appending
    # closed-world/no-extra-material case rules. CLOSED_WORLD_RULES_STRICT_RAG plus
    # STEP3_RAG_RULES_GENERIC already provide the correct constraints.
    world_norm = _normalize_world_mode(world)
    strict_extra = STEP3_STRICT_CLOSED_CASE_RULES_GENERIC if world_norm == "closed_strict" else ""
    parts = []
    if str(fact_pack_text or "").strip():
        parts.append(STEP3_FACT_PACK_RULES_GENERIC)
        if _is_criticism_only_fact_pack_mode(mode):
            parts.append(STEP3_CRITICISM_ONLY_RULES_GENERIC)
    elif _is_closed_world_mode(world) and not rag_active:
        parts.append(STEP3_NO_FACT_PACK_CLOSED_RULES_GENERIC)
    if strict_extra:
        parts.append(strict_extra)
    # RAG rules are injected only through the explicit {STEP3_RAG_RULES} placeholder.
    return _join_prompt_rule_blocks(*parts)

def _normalize_world_mode(world: str) -> str:
    """Normalize world-mode aliases and fall back to closed mode when invalid."""
    w = str(world or "closed").strip().lower()
    if w == "open":
        w = "open_no_rag"
    if w not in EXTERNAL_WORLD_RULES:
        w = "closed"
    return w


def _is_open_world_mode(world: str) -> bool:
    """Return whether a world mode allows model background knowledge or live search."""
    return _normalize_world_mode(world) in {"open_no_rag", "open_rag", "true_open"}




def _qwen_world_num_predict_floor(step: str, world: str, think_flag) -> int:
    """
    Choose a minimum qwen generation budget for the current step and world mode.
    
        RAG and open-world prompts are longer than closed prompts, and Step-2/Step-3 have different output
        lengths. This helper prevents the native path from using generation budgets that are too small for
        protocol completion.
    """
    w = _normalize_world_mode(world)
    step = str(step or "step2").strip().lower()
    base = {
        "step2": {
            "closed": 768,
            "closed_strict": 768,
            "closed_strict_rag": 896,
            "open_no_rag": 896,
            "open_rag": 960,
            "true_open": 1152,
        },
        "step3": {
            "closed": 896,
            "closed_strict": 896,
            "closed_strict_rag": 960,
            "open_no_rag": 960,
            "open_rag": 1024,
            "true_open": 1280,
        },
    }.get(step, {}).get(w, 896 if step == "step3" else 768)
    if think_flag is True:
        if step == "step2":
            return max(base + 256, 1280)
        return max(base + 512, 1792)
    return base




def _build_world_aware_step2_compact_retry_prompt(agent_name: str, claim_txt: str, current_belief: int) -> str:
    """
    Build a compact Step-2 retry prompt after a tweet-generation failure.
    
        The retry keeps only the essential claim, present view, world constraints, and output format. It is
        used when the original prompt produced malformed or off-protocol text and a shorter instruction is
        more likely to succeed.
    """
    belief_i = int(current_belief)
    claim_clean = str(claim_txt or '').strip()
    common = (
        "Output exactly 2 lines and nothing else.\n"
        f"FINAL_RATING: {belief_i}\n"
        "TWEET:\n\n"
        "Keep it natural and person-like, not slogan-like.\n"
        "Prefer one concrete point over broad, stock, or summary-style language.\n"
    )
    if belief_i == 0:
        stance_block = (
            "For 0, keep both sides moderate.\n"
            "Use one concrete point pulling one way and one concrete unresolved point pulling the other way when possible.\n"
            "Do not let one side sound settled while the other is only a weak caveat.\n"
        )
    elif belief_i == 1:
        stance_block = (
            "For 1, sound mildly for the claim, not fully certain.\n"
            "Use one concrete supporting point and natural mild hedging when it fits.\n"
        )
    elif belief_i == -1:
        stance_block = (
            "For -1, sound mildly against the claim, not fully certain.\n"
            "Use one concrete unresolved detail or gap and natural mild hedging when it fits.\n"
        )
    elif belief_i > 1:
        stance_block = "For 2, use one concrete supporting point over a broad slogan.\n"
    else:
        stance_block = "For -2, use one concrete gap, mismatch, or unresolved detail over a broad anti-claim slogan.\n"

    if _is_open_world_mode(WORLD):
        return (
            "/no_think\n"
            f"Now, {agent_name or 'you'}, rewrite the tweet in a compact form that matches the present view exactly.\n\n"
            f"CLAIM: {claim_clean}\n"
            f"PRESENT VIEW: {belief_i}\n\n"
            + common
            + stance_block
            + "Open-world modes may use public knowledge if genuinely relevant.\n"
            + "1 or 2 overall reads FOR the claim; -1 or -2 overall reads AGAINST; 0 may be genuinely mixed.\n"
            + "Start your response with FINAL_RATING:"
        )
    return (
        "/no_think\n"
        f"Now, {agent_name or 'you'}, rewrite the tweet in a compact form that matches the present view exactly.\n\n"
        f"CLAIM: {claim_clean}\n"
        f"PRESENT VIEW: {belief_i}\n\n"
        + common
        + stance_block
        + "No outside records, institutions, experts, archives, or public evidence sources unless shown in this compact prompt.\n"
        + "1 or 2 overall reads FOR the claim; -1 or -2 overall reads AGAINST; 0 may be genuinely mixed.\n"
        + "Start your response with FINAL_RATING:"
    )




def _normalize_step2_request_header_text(prompt_text: str) -> str:
    """Fix duplicated Step-2 request phrasing without flattening the prompt layout."""
    s = str(prompt_text or "")
    if not s:
        return s

    patterns = [
        (r"(?i)\b(write another tweet)\s*write another tweet\b", r"\1"),
        (r"(?i)\b(write a tweet)\s*write another tweet\b", r"\1"),
        (r"(?i)\b(write another tweet)\s*write a tweet\b", r"\1"),
        (r"(?i)\b(write a tweet)\s*write a tweet\b", r"\1"),
        (r"(?i)\banother tweetabout\b", "another tweet about"),
        (r"(?i)\btweetabout\b", "tweet about"),
        (r"(?i)\babout this claim\s*:\s*about this claim\s*:", "about this claim:"),
    ]

    out_lines = []
    for line in s.splitlines():
        fixed = str(line or "")
        for pat, repl in patterns:
            fixed = re.sub(pat, repl, fixed)
        fixed = re.sub(r"(?i)(please write (?:another )?tweet)\s*,\s*", r"\1 ", fixed)
        fixed = re.sub(r" {2,}", " ", fixed)
        out_lines.append(fixed.rstrip())

    s = "\n".join(out_lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _fix_quoted_protocol_label_breaks(prompt_text: str) -> str:
    """Repair protocol labels accidentally broken inside quoted instructional prose.

    Example bad form:
      after "
      TWEET:"
    Correct form:
      after "TWEET:"
    """
    s = str(prompt_text or "")
    if not s:
        return s

    # Re-join labels that were split inside quotes by later line-boundary restoration.
    s = re.sub(r"([\"'])\s*\n\s*(TWEET:|FINAL_RATING:|EXPLANATION:)\s*([\"'])", r"\1\2\3", s)
    # Common explicit instructional forms in Step-2/Step-3 templates.
    s = re.sub(r"(?i)(after\s+[\"'])\s*\n\s*(TWEET:|FINAL_RATING:|EXPLANATION:)\s*([\"'])", r"\1\2\3", s)
    s = re.sub(r"(?i)(line\s+[12]\s+must\s+contain\s+real\s+(?:tweet\s+text|text)\s+after\s+[\"'])\s*\n\s*(TWEET:|FINAL_RATING:|EXPLANATION:)\s*([\"'])", r"\1\2\3", s)

    return s


def _restore_step2_prompt_block_boundaries(prompt_text: str) -> str:
    """Restore readable Step-2 block layout after formatting/normalization.

    This is cosmetic/readability-oriented and should not change prompt meaning.
    It only reinstates missing line/block boundaries when sections were glued together.
    """
    s = str(prompt_text or "")
    if not s:
        return s

    major_markers = [
        r"CLAIM:\s*",
        r"PRESENT VIEW:\s*",
        r"AGENT_PERSONA_CARD",
        r"Retrieved snippets \(claim-supporting\):",
        r"Retrieved snippets \(claim-challenging\):",
        r"Retrieved-context addition:",
        r"Closed-world, no extra material:",
        r"Strict closed-world Step-2 addition:",
        r"Task\b",
        r"Direction and strength\b",
        r"Format\b",
        r"Output:",
    ]
    for pat in major_markers:
        s = re.sub(rf"(?<!\n\n)\s*({pat})", r"\n\n\1", s)

    basic_line_markers = [
        r"Name:\s*",
        r"CURRENT_BELIEF:\s*",
        r"CURRENT VIEW:\s*",
    ]
    for pat in basic_line_markers:
        s = re.sub(rf"(?<!\n)\s*({pat})", r"\n\1", s)

    quoted_sensitive_markers = [
        r"TWEET:\s*",
        r"FINAL_RATING:\s*",
        r"EXPLANATION:\s*",
    ]
    for pat in quoted_sensitive_markers:
        # Avoid breaking instructional prose like: after "TWEET:"
        s = re.sub(rf"(?<!\n)(?<![\"'])\s*({pat})", r"\n\1", s)

    prose_starts = [
        r"Remember, throughout the interactions",
        r"Remember, you are role playing",
        r"You may not search",
        r"Use only information explicitly shown",
        r"Do not rely on outside facts",
        r"If extra shown material is provided",
        r"If no extra shown material is provided",
        r"Do not treat the absence of extra shown material",
        r"Do NOT mention these restrictions",
        r"Write one short tweet",
        r"Write it as a public tweet",
    ]
    for pat in prose_starts:
        s = re.sub(rf"(?<!\n)\s*({pat})", r"\n\1", s)

    fixed_lines = []
    for line in s.splitlines():
        ln = re.sub(r"[ \t]{2,}", " ", str(line or "")).rstrip()
        fixed_lines.append(ln)
    s = "\n".join(fixed_lines)

    s = re.sub(r"\s+-\s+", r"\n- ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = _fix_quoted_protocol_label_breaks(s)
    return s.strip()
def _extract_step2_shown_tweet_from_prompt(prompt_text: str) -> str:
    """Extract shown tweet text from a Step-2 prompt for rescue prompting."""
    s = str(prompt_text or "")
    if not s:
        return ""
    m = re.search(r"(?is)\nTWEET:\s*\n(.*?)(?:\n\s*\n(?:Remember,|No-extra-material closed-world addition:|Strict closed-world Step-2 addition:|Task|Rating scale meaning|Core rules|Reason rule|Strength rule|Polarity control|Writing rule|Output rules|Use EXACTLY this format:)|\Z)", s)
    if not m:
        return ""
    return re.sub(r"\s+", " ", str(m.group(1) or "")).strip()


def _build_strict_closed_step2_native_rescue_prompt(agent_name: str, claim_txt: str, current_belief: int, shown_tweet: str = "", shown_material: str = "") -> str:
    """Build a native Step-2 rescue prompt that forbids unsupported outside knowledge."""
    belief_i = int(current_belief)
    claim_clean = str(claim_txt or '').strip()
    shown = re.sub(r"\s+", " ", str(shown_tweet or "")).strip()
    shown_mat = _compact_rescue_grounding_text(shown_material, max_chars=420)
    lines = [
        "/no_think",
        f"CLAIM: {claim_clean}",
        f"PRESENT VIEW: {belief_i}",
    ]
    if shown:
        lines.extend(["", "SHOWN TWEET:", shown])
    if shown_mat:
        lines.extend(["", "SHOWN MATERIAL:", shown_mat])

    stance_rule = {
        -2: "Write one short public tweet that is clearly AGAINST the claim. Use one local gap, mismatch, or unresolved detail.",
        -1: "Write one short public tweet that is mildly AGAINST the claim. Use one local gap, mismatch, or unresolved detail. Do not sound fully settled.",
         0: "Write one short public tweet with exactly two short clauses: one point each way. Keep both sides moderate.",
         1: "Write one short public tweet that is mildly FOR the claim. Use one local supporting point. Do not sound fully settled.",
         2: "Write one short public tweet that is clearly FOR the claim. Use one local supporting point.",
    }.get(belief_i, "Write one short public tweet that matches PRESENT VIEW exactly.")

    lines.extend([
        "",
        stance_rule,
        "Use only the claim wording and any shown text above.",
        "Use one short local point only.",
        "No outside-source backing or broad public validation unless the exact support is shown above.",
        "Output exactly 2 lines and nothing else.",
        "Line 1 must start exactly with FINAL_RATING:",
        "Line 2 must start exactly with TWEET:",
        f"FINAL_RATING: {belief_i}",
        "TWEET:",
        "Start your response with FINAL_RATING:",
    ])
    return "\n".join(str(x) for x in lines if x is not None).strip()


def _build_strict_closed_step2_ultra_compact_rescue_prompt(agent_name: str, claim_txt: str, current_belief: int, shown_tweet: str = "", shown_material: str = "") -> str:
    """Build the smallest strict-closed Step-2 rescue prompt for stubborn protocol failures."""
    belief_i = int(current_belief)
    claim_clean = str(claim_txt or '').strip()
    shown = re.sub(r"\s+", " ", str(shown_tweet or "")).strip()
    shown_mat = _compact_rescue_grounding_text(shown_material, max_chars=420)
    stance_rule = {
        -2: "Write one firm AGAINST tweet with one short local gap or mismatch.",
        -1: "Write one mild AGAINST tweet with one short unresolved gap.",
         0: "Write one mixed tweet with two short clauses: one point each way.",
         1: "Write one mild FOR tweet with one short local supporting point. Use mild wording and do not sound settled or definitive.",
         2: "Write one firm FOR tweet with one short local supporting point.",
    }.get(belief_i, "Write one short tweet that matches the present view exactly.")
    lines = [
        "/no_think",
        f"CLAIM: {claim_clean}",
        f"PRESENT VIEW: {belief_i}",
    ]
    if shown:
        lines.extend(["", "SHOWN TWEET:", shown])
    if shown_mat:
        lines.extend(["", "SHOWN MATERIAL:", shown_mat])
    lines.extend([
        "",
        stance_rule,
        "Use only the claim wording, any shown tweet, and any shown material.",
        "No outside-source backing or broad public validation unless the exact support is shown above.",
        "If a broad agreement, evidence, source, history, or verification appeal is not shown here, do not use it as support.",
        "Keep it natural and short. No planning or explanation.",
        "Output exactly 2 lines and nothing else.",
        "Line 1 must start exactly with FINAL_RATING:",
        "Line 2 must start exactly with TWEET:",
        f"FINAL_RATING: {belief_i}",
        "TWEET:",
        "Start your response with FINAL_RATING:",
    ])
    return "\n".join(str(x) for x in lines if x is not None).strip()


def _rescue_grounding_item_is_useful(line: str) -> bool:
    """Filter prompt lines so rescue prompts keep only real grounding material."""
    low = re.sub(r"\s+", " ", str(line or "")).strip().lower()
    if not low:
        return False
    if low.startswith((
        "task", "rating scale", "core rules", "reason rule", "strength rule",
        "polarity control", "writing rule", "output rules", "evaluation rule",
        "decision rule", "explanation rule", "output discipline", "output format",
    )):
        return False
    if "persona may shape plausibility" in low:
        return False
    if "do not turn it into unseen evidence" in low:
        return False
    if low.startswith("remember,") or low.startswith("do not "):
        return False
    return True


def _extract_external_grounding_text(prompt_text: str) -> str:
    """
    Extract shown grounding material from prompt sections used by validators and rescues.
    
        The function looks for fact-pack, RAG, web-search, and other explicit material blocks while
        ignoring task instructions. Its output is the basis for deciding whether source/evidence language
        is grounded in the current prompt.
    """
    s = str(prompt_text or "")
    if not s:
        return ""
    items = []
    seen = set()
    patterns = [
        r"(?is)\nFACT PACK(?: ABOUT .*?)?\n(.*?)(?:\n\s*\n(?:Task|Rating scale meaning|Core rules|Reason rule|Strength rule|Polarity control|Writing rule|Output rules|Use EXACTLY this format:|Evaluation rule|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
        r"(?is)\nRetrieved snippets(?: \(shown material\))?\n(.*?)(?:\n\s*\n(?:Task|Rating scale meaning|Core rules|Reason rule|Strength rule|Polarity control|Writing rule|Output rules|Use EXACTLY this format:|Evaluation rule|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
        r"(?is)\nLIVE WEB SEARCH RESULTS \(shown material\)\n(.*?)(?:\n\s*\n(?:Task|Rating scale meaning|Core rules|Reason rule|Strength rule|Polarity control|Writing rule|Output rules|Use EXACTLY this format:|Evaluation rule|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, s):
            block = str(m.group(1) or "")
            for ln in block.splitlines():
                line = re.sub(r"\s+", " ", ln).strip()
                if not line:
                    continue
                if line.startswith(("- ", "• ")):
                    line = re.sub(r"^[•-]\s*", "", line).strip()
                if not _rescue_grounding_item_is_useful(line):
                    continue
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(line)

    heading_re = re.compile(r"(?i)^(?:FACT PACK(?: ABOUT .*?)?|Retrieved snippets(?:\s*\([^)]*\))?|LIVE WEB SEARCH RESULTS \(shown material\))\s*:?\s*$")
    stop_re = re.compile(r"(?i)^(?:Retrieved-context(?: evaluation)? addition:|Closed-world, no extra material:|Task\b|Rating scale meaning\b|Core rules\b|Reason rule\b|Strength rule\b|Polarity control\b|Writing rule\b|Output rules\b|Use EXACTLY this format:|Evaluation rule\b|Decision rule\b|How to think about the rating\b|Explanation rule\b|Output discipline\b|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:|Bias rule\b)\s*$")
    collecting = False
    for ln in s.splitlines():
        raw = str(ln or "").strip()
        if not raw:
            continue
        if heading_re.match(raw):
            collecting = True
            continue
        if collecting and stop_re.match(raw):
            collecting = False
            continue
        if collecting:
            if raw.startswith(("- ", "• ")):
                line = re.sub(r"^[•-]\s*", "", raw).strip()
                if not _rescue_grounding_item_is_useful(line):
                    continue
                key = line.lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(line)

    persona_lines = []
    for line in s.splitlines():
        raw = line.strip()
        low = raw.lower()
        if not low:
            continue
        if "persona may shape plausibility" in low:
            continue
        if "do not turn it into unseen evidence" in low:
            continue
        explicit_persona_grounding = any(tok in low for tok in [
            "institutional trust:",
            "trusts institutions",
            "trust in institutions",
            "trusts official",
            "trust official",
            "trusts authorities",
            "trust authorities",
            "trusts experts",
            "trust experts",
            "government source",
            "official source",
        ])
        if explicit_persona_grounding and not low.startswith("remember,") and not low.startswith("do not "):
            persona_lines.append(raw)
    for line in persona_lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            items.append(line)

    return "\n".join(f"- {x}" for x in items[:6]).strip()


def _compact_rescue_grounding_text(text: str, max_chars: int = 700) -> str:
    """Shorten grounding material for compact rescue prompts without changing its role."""
    raw_lines = [
        re.sub(r"\s+", " ", ln).strip()
        for ln in str(text or "").splitlines()
        if re.sub(r"\s+", " ", ln).strip()
    ]
    cleaned = []
    seen = set()
    for ln in raw_lines:
        line = re.sub(r"^[•-]\s*", "", ln).strip()
        if not _rescue_grounding_item_is_useful(line):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f"- {line}")
    s = "\n".join(cleaned).strip()
    if not s:
        return ""
    max_chars = max(120, int(max_chars or 700))
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars + 1]
    punct_idx = max(cut.rfind("\n- "), cut.rfind('. '), cut.rfind('; '), cut.rfind(', '), cut.rfind(' - '), cut.rfind(' '))
    if punct_idx >= max(80, int(max_chars * 0.6)):
        s = cut[:punct_idx].strip(" ,;:-")
    else:
        s = cut[:max_chars].strip(" ,;:-")
    return s + " ..."

def _persona_trust_allows_external_family(prompt_text: str, family: str) -> bool:
    """
    Return whether persona trust traits permit source-oriented explanation language.
    
        Authority-leaning agents and distrust-oriented agents can legitimately use different source
        frames. This check gives validators a controlled way to allow that variation without opening the
        door to unsupported outside-knowledge claims.
    """
    s = str(prompt_text or "").lower()
    if not s:
        return False
    trustish = any(tok in s for tok in ["institutional trust", "trusts institutions", "trust in institutions", "trusts official", "trust official", "trusts authorities", "trust authorities", "trusts experts", "trust experts"])
    if not trustish:
        return False
    return family in {"verification", "records", "documents", "official_confirmation", "experts", "institutions"}
def _true_open_stopwords() -> set[str]:
    """Return stopwords used by true-open query and result scoring."""
    return {
        "the","a","an","and","or","but","if","then","that","this","those","these","there","here",
        "have","has","had","not","never","on","in","to","of","for","with","from","by","as",
        "is","are","was","were","be","been","being","it","its","they","them","their","you","your",
        "we","our","us","believe","claim","claims","tweet","says","say","because","since","according",
        "exists","exist","confirm","confirmed","evidence","proof"
    }


def _true_open_tokens(s: str) -> list[str]:
    """Tokenize true-open query/result text for scoring."""
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9'_-]+", str(s or ""))]
    return toks


_TRUE_OPEN_GENERIC_TOPIC_TOKENS = {
    "moon", "landed", "landing", "landings", "astronaut", "astronauts", "apollo",
    "claim", "claims", "nasa", "america", "american", "united", "states", "space",
    "mission", "missions", "humans", "human"
}

_TRUE_OPEN_GENERAL_BAD_TITLE_PATTERNS = [
    r"^did .+\?$",
    r"\bconspirac",
    r"\bclick here\b",
    r"\bwhat you need to know\b",
    r"\ball you need to know\b",
    r"\btracker\b",
    r"\blive updates?\b",
    r"\binterview\b",
]

_TRUE_OPEN_GENERAL_LOW_QUALITY_PATTERNS = [
    r"\b(complete overview|complete guide|full guide|mission overview|overview of|full list|list of|discover the full list)\b",
    r"\b(photos?|images?|picture|pictures|gallery|wallpaper)\b",
    r"\b(what's next|what is next|upcoming mission|next mission)\b",
    r"\b(how many .*missions|how many .*people|who they are|who flew|which missions?)\b",
    r"\b(tracker|live blog|live updates?|breaking news|exclusive|watch live)\b",
    r"\b(interview|clip|viral|quote of the day|what .* said)\b",
    r"\b(repost|aggregator|roundup|explainer)\b",
]

_TRUE_OPEN_MOON_EVIDENCE_BOOST_PATTERNS = [
    r"\b(retroreflector|retroreflectors|laser ranging|lunar samples?|moon rocks?|landing sites?|orbiter images?|independent(?:ly)? tracked|telemetry|ground stations?)\b",
    r"\b(lro|lunar reconnaissance orbiter|surveyor 3|footprints?|descent stage|sample catalog|regolith)\b",
    r"\b(flag waving without wind|horizontal rod|camera exposure|no stars|vacuum explanation|moon rocks|lunar rocks)\b",
]

_TRUE_OPEN_MOON_DISTRACTOR_PATTERNS = [
    r"\b(artemis|why haven't astronauts returned|returned to the moon|50-year delay|50 year delay|haven't landed on the moon since apollo 17|why haven't we gone back)\b",
    r"\b(list of people who flew to the moon|twenty-eight people have traveled to the vicinity of the moon)\b",
    r"\b(launches four people on artemis ii|10-day journey around the moon|artemis ii astronauts|around the moon to test systems)\b",
    r"\b(buzz aldrin|kim kardashian|myth 1: the waving flag|what countries have landed on the moon)\b",
    r"\b(tracker|live update|interview|returned to the moon since|why haven't .* returned|crewed lunar flyby|historic moon mission|splashes down|late 2028)\b",
]

_TRUE_OPEN_MOON_VERIFICATION_PATTERNS = [
    r"\b(independent verification|other nations|ground stations|tracking stations|telemetry|lro|orbiter images?|landing sites?)\b",
    r"\b(records?|archives?|transcripts?|recordings?|samples?|moon rocks?|lunar rocks?)\b",
    r"\b(flag waving|no stars|camera exposure|vacuum|retroreflectors?|laser ranging)\b",
]


def _true_open_hinge_tokens(tweet_txt: str, claim_txt: str = "") -> set[str]:
    """Extract focused tokens from the claim and tweet for live-search queries."""
    tweet_toks = _true_open_tokens(tweet_txt)
    claim_toks = set(_true_open_tokens(claim_txt))
    generic = _true_open_stopwords() | _TRUE_OPEN_GENERIC_TOPIC_TOKENS | claim_toks
    return {t for t in tweet_toks if len(t) >= 4 and t not in generic}


def _true_open_focus_flags(tweet_txt: str) -> dict:
    """Detect which evidence families the true-open search should emphasize."""
    low = str(tweet_txt or '').lower()
    return {
        "evidence_focus": bool(re.search(r"\b(evidence|proof|proven|undeniable|irrefutable|photo|photos|video|videos|footage|sample|samples|rock|rocks|record|records|archive|archives|verify|verification|verifiable|authentic|authenticity|hoax|fake|flag|radiation|belt|belts|retroreflector|unverifiable|circumstantial)\b", low)),
        "return_focus": bool(re.search(r"\b(return|returned|returning|back to the moon|since apollo 17|50-year|50 year|why haven't|why havent|gone back|go back)\b", low)),
        "people_focus": bool(re.search(r"\b(12|twelve|who flew|who walked|walked on the moon|people who flew)\b", low)),
        "debate_focus": bool(re.search(r"\b(debate|debates|question|questions|doubt|doubts|persist|authenticity|hoax|fake|conspiracy|myth)\b", low)),
        "footage_focus": bool(re.search(r"\b(footage|video|videos|flag|waving|wind|camera|film|broadcast|stars?)\b", low)),
        "artifact_focus": bool(re.search(r"\b(sample|samples|rock|rocks|artifact|artifacts|retroreflector|footprints|descent stage|landing site|lunar sample|moon rock)\b", low)),
        "verification_focus": bool(re.search(r"\b(verification|verify|confirmed|confirm|tracking|other nations|ground stations|independent|telemetry|landing site images|orbiter images)\b", low)),
        "records_focus": bool(re.search(r"\b(records?|archives?|transcripts?|recordings?|transparency)\b", low)),
    }

def _true_open_url_path_penalty(url: str) -> int:
    """Penalize low-value URL paths in true-open result scoring."""
    try:
        parsed = urllib.parse.urlparse(str(url or ""))
        path = f"{parsed.netloc} {parsed.path}".lower()
    except Exception:
        path = ""
    if not path:
        return 0
    penalty = 0
    if re.search(r"\b(tracker|live|updates?|interview|video|gallery|photos?|images?|slideshow|timeline|list)\b", path):
        penalty -= 4
    if re.search(r"\b(artemis|return(?:ed|ing)?-?to-?the-?moon|why-havent|why-haven't|since-apollo-17|complete-overview|again|around-the-moon|moon-again|launch(?:es|ed|ing)?|future-mission)\b", path):
        penalty -= 6
    if re.search(r"\b(what-do-you-think|how-many|which-missions|full-list|people-have-been-to-the-moon)\b", path):
        penalty -= 5
    return penalty

def _true_open_source_specificity_bonus(blob_low: str, focus_flags: dict) -> int:
    """Reward search results that mention concrete sources, titles, or claim-specific details."""
    bonus = 0
    if (focus_flags.get("evidence_focus") or focus_flags.get("artifact_focus") or focus_flags.get("footage_focus")) and re.search(r"\b(photo|photos|video|videos|footage|sample|samples|rock|rocks|record|records|archive|archives|retroreflector|flag|radiation|belt|belts|footprints?|descent stage|landing site|orbiter images?)\b", blob_low):
        bonus += 5
    if (focus_flags.get("verification_focus") or focus_flags.get("records_focus")) and re.search(r"\b(independent verification|other nations|ground stations|tracking stations|telemetry|transcripts?|recordings?|archives?|landing sites?|lro|orbiter images?)\b", blob_low):
        bonus += 4
    if focus_flags.get("debate_focus") and re.search(r"\b(hoax|fake|debunk|authentic|authenticity|question|questions|myth|conspiracy)\b", blob_low):
        bonus += 2
    if focus_flags.get("people_focus") and re.search(r"\b(12|twelve|walked on the moon|flew to the moon|moonwalkers)\b", blob_low):
        bonus += 2
    return bonus

def _true_open_domain_quality(url: str) -> int:
    """Score a result domain for source quality in true-open mode."""
    try:
        host = (urllib.parse.urlparse(str(url or "")).netloc or "").lower()
    except Exception:
        host = ""
    if not host:
        return 0
    if host.startswith("www."):
        host = host[4:]
    high = (".gov", ".edu", "nasa.gov", "esa.int", "wikipedia.org", "britannica.com", "apnews.com", "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com")
    med = ("theguardian.com", "nationalgeographic.com", "smithsonianmag.com", "history.com", "scientificamerican.com", "space.com", "cnn.com", "abcnews.go.com", "bigthink.com", "iop.org")
    low = ("msn.com", "yahoo.com", "news.yahoo.com", "ithy.com", "quora.com", "factually.co", "beautifulpublicdata.com")
    very_low = ("youtube.com", "tiktok.com", "reddit.com", "blogspot.", "wordpress.", "pinterest.", "instagram.com", "facebook.com")
    if any(x in host for x in high):
        return 4
    if any(x in host for x in med):
        return 2
    if any(x in host for x in low):
        return -2
    if any(x in host for x in very_low):
        return -4
    return 0


def _true_open_topic_profile(claim_txt: str, tweet_txt: str = "") -> str:
    """Return topic-specific scoring hints for true-open retrieval."""
    blob = f"{claim_txt or ''} {tweet_txt or ''}".lower()
    if re.search(r"\b(apollo|moon landing|moon landings|lunar|astronauts? .*moon|moon .*astronauts?)\b", blob):
        return "moon_landing"
    return "generic"

def _score_true_open_result_general(row: dict, claim_txt: str, tweet_txt: str) -> tuple[int, dict]:
    """Compute the general quality/relevance score for one live-search result."""
    title = re.sub(r"\s+", " ", str((row or {}).get("title", "") or "")).strip()
    snippet = re.sub(r"\s+", " ", str((row or {}).get("snippet", "") or "")).strip()
    url = str((row or {}).get("url", "") or "").strip()
    blob = f"{title} {snippet}".strip()
    if not blob:
        return (-999, {})
    claim_toks = {t for t in _true_open_tokens(claim_txt) if t not in _true_open_stopwords()}
    tweet_toks = {t for t in _true_open_tokens(tweet_txt) if t not in _true_open_stopwords()}
    hinge_toks = _true_open_hinge_tokens(tweet_txt, claim_txt)
    blob_toks = set(_true_open_tokens(blob))
    claim_overlap = len(claim_toks & blob_toks)
    tweet_overlap = len(tweet_toks & blob_toks)
    hinge_overlap = len(hinge_toks & blob_toks)
    focus_flags = _true_open_focus_flags(tweet_txt)
    score = 4 * claim_overlap + 5 * tweet_overlap + 7 * hinge_overlap
    score += _true_open_domain_quality(url) * 3
    score += _true_open_url_path_penalty(url)
    if len(snippet) >= 80:
        score += 2
    elif len(snippet) >= 35:
        score += 1
    low_title = title.lower()
    low_blob = blob.lower()
    if any(re.search(p, low_title) for p in _TRUE_OPEN_GENERAL_BAD_TITLE_PATTERNS):
        score -= 4
    if '?' in title and tweet_overlap == 0 and hinge_overlap == 0:
        score -= 2
    if hinge_toks and hinge_overlap == 0:
        score -= 6
    score += _true_open_source_specificity_bonus(low_blob, focus_flags)
    if any(re.search(p, low_blob) for p in _TRUE_OPEN_GENERAL_LOW_QUALITY_PATTERNS):
        score -= 4
    score += _true_open_contextual_page_type_penalty(title, snippet, url, tweet_txt, focus_flags, hinge_overlap)
    clean = {"title": title[:160], "snippet": snippet[:320], "url": url[:500], "topic_profile": _true_open_topic_profile(claim_txt, tweet_txt), "hinge_overlap": hinge_overlap}
    return score, clean


def _score_true_open_result_topic_layer(clean: dict, claim_txt: str, tweet_txt: str) -> int:
    """Add topic-specific relevance adjustments to a live-search result score."""
    topic_profile = str((clean or {}).get("topic_profile", "generic") or "generic")
    if topic_profile == "generic":
        return 0
    low_blob = f"{clean.get('title','')} {clean.get('snippet','')}".lower()
    focus_flags = _true_open_focus_flags(tweet_txt)
    hinge_overlap = int((clean or {}).get("hinge_overlap", 0) or 0)
    score = 0
    if topic_profile == "moon_landing":
        if (focus_flags.get("evidence_focus") or focus_flags.get("artifact_focus") or focus_flags.get("footage_focus") or focus_flags.get("verification_focus") or focus_flags.get("records_focus")) and any(re.search(p, low_blob) for p in _TRUE_OPEN_MOON_EVIDENCE_BOOST_PATTERNS):
            score += 7
        if (focus_flags.get("verification_focus") or focus_flags.get("records_focus")) and any(re.search(p, low_blob) for p in _TRUE_OPEN_MOON_VERIFICATION_PATTERNS):
            score += 5
        if not focus_flags.get("return_focus") and any(re.search(p, low_blob) for p in _TRUE_OPEN_MOON_DISTRACTOR_PATTERNS):
            score -= 6
        if (focus_flags.get("evidence_focus") or focus_flags.get("artifact_focus") or focus_flags.get("footage_focus") or focus_flags.get("verification_focus") or focus_flags.get("records_focus")) and re.search(r"\b(how many .*missions landed on the moon|what's next in moon exploration|which apollo missions)\b", low_blob):
            score -= 4
        if (focus_flags.get("evidence_focus") or focus_flags.get("artifact_focus") or focus_flags.get("footage_focus") or focus_flags.get("verification_focus") or focus_flags.get("records_focus")) and hinge_overlap == 0 and re.search(r"\b(artemis|returned to the moon|why haven't|tracker|interview|launch|crewed lunar flyby|historic moon mission)\b", low_blob):
            score -= 5
    return score


def _score_true_open_result(row: dict, claim_txt: str, tweet_txt: str) -> tuple[int, dict]:
    """Combine general and topic-specific scoring for one live-search result."""
    score, clean = _score_true_open_result_general(row, claim_txt, tweet_txt)
    if score <= -999 or not clean:
        return score, clean
    score += _score_true_open_result_topic_layer(clean, claim_txt, tweet_txt)
    return score, clean


def _rerank_true_open_results(results: list[dict], claim_txt: str, tweet_txt: str, show_k: int = 3, step_kind: str = "step3") -> list[dict]:
    """Sort live-search results by the combined true-open relevance score."""
    scored = []
    seen = set()
    hinge_toks = _true_open_hinge_tokens(tweet_txt, claim_txt)
    focus_flags = _true_open_focus_flags(tweet_txt)
    topic_profile = _true_open_topic_profile(claim_txt, tweet_txt)
    evidence_focus = topic_profile == "moon_landing" and any([
        focus_flags.get('evidence_focus'), focus_flags.get('artifact_focus'), focus_flags.get('footage_focus'),
        focus_flags.get('verification_focus'), focus_flags.get('records_focus')
    ])
    generic_topic = topic_profile == 'generic'
    for row in results or []:
        score, clean = _score_true_open_result(row, claim_txt, tweet_txt)
        min_accept = 1 if generic_topic else 2
        if score <= min_accept or not clean:
            continue
        blob_low = f"{clean.get('title','')} {clean.get('snippet','')}".lower()
        blob_toks = set(_true_open_tokens(f"{clean.get('title','')} {clean.get('snippet','')}"))
        hinge_match = len(hinge_toks & blob_toks)
        verification_match = any(re.search(p, blob_low) for p in (_TRUE_OPEN_MOON_EVIDENCE_BOOST_PATTERNS + _TRUE_OPEN_MOON_VERIFICATION_PATTERNS))
        distractor = any(re.search(p, blob_low) for p in _TRUE_OPEN_MOON_DISTRACTOR_PATTERNS)
        if hinge_toks and hinge_match == 0:
            hinge_gate = 8 if generic_topic else 12
            if score < hinge_gate:
                continue
        if topic_profile == 'moon_landing' and not focus_flags.get('return_focus') and distractor and hinge_match == 0 and not verification_match:
            continue
        key = (clean.get("title", "").strip().lower(), (urllib.parse.urlparse(clean.get("url", "")).netloc or "").lower())
        if key in seen:
            continue
        seen.add(key)
        scored.append((score, clean))
    scored.sort(key=lambda x: (-x[0], len(x[1].get("snippet", "")), x[1].get("title", "")))
    if not scored:
        return []
    best = int(scored[0][0])
    if evidence_focus:
        dynamic_cutoff = max(12, best - 3)
    elif generic_topic:
        dynamic_cutoff = max(4, best - 6)
    else:
        dynamic_cutoff = max(8, best - 4)
    kept = [(score, row) for score, row in scored if score >= dynamic_cutoff]
    if not kept:
        kept = [scored[0]]
    if generic_topic and len(kept) < max(1, int(show_k or 3)):
        near_top = [(score, row) for score, row in scored if score >= max(3, best - 7)]
        if near_top:
            kept = near_top
    if topic_profile == 'moon_landing' and not focus_flags.get('return_focus'):
        direct_rows = []
        high_quality_direct_rows = []
        for score, row in kept:
            blob_low = f"{row.get('title','')} {row.get('snippet','')}".lower()
            has_direct = int(row.get('hinge_overlap', 0) or 0) > 0 or any(re.search(p, blob_low) for p in (_TRUE_OPEN_MOON_EVIDENCE_BOOST_PATTERNS + _TRUE_OPEN_MOON_VERIFICATION_PATTERNS))
            is_distractor = any(re.search(p, blob_low) for p in _TRUE_OPEN_MOON_DISTRACTOR_PATTERNS)
            domain_quality = _true_open_domain_quality(str(row.get('url', '') or ''))
            generic_overview = any(re.search(p, blob_low) for p in _TRUE_OPEN_GENERAL_LOW_QUALITY_PATTERNS)
            if has_direct and not (is_distractor and int(row.get('hinge_overlap', 0) or 0) == 0):
                direct_rows.append((score, row))
                if domain_quality >= 0 and not generic_overview:
                    high_quality_direct_rows.append((score, row))
        if high_quality_direct_rows:
            kept = high_quality_direct_rows
        elif direct_rows:
            kept = direct_rows
        elif evidence_focus:
            return []
    max_keep = 3 if evidence_focus else max(1, int(show_k or 3))
    return [row for _, row in kept[:max_keep]]

def _render_live_web_context(results: list[dict], max_chars: int = 1400, *, no_strong_results: bool = False) -> str:
    """Render selected live-search results into prompt context for true-open runs."""
    rows = []
    total = 0
    if no_strong_results:
        return ""
    for idx, row in enumerate(results or [], start=1):
        title = re.sub(r"\s+", " ", str((row or {}).get("title", ""))).strip()
        snippet = _true_open_compact_result_snippet((row or {}).get("snippet", ""), max_chars=180)
        if not (title or snippet):
            continue
        piece = f"- Relevant web result {idx}: {title}"
        if snippet:
            piece += f" — {snippet}"
        if total + len(piece) > max_chars and rows:
            break
        rows.append(piece)
        total += len(piece)
    if not rows:
        return ""
    return "LIVE WEB SEARCH RESULTS (shown material)\n" + "\n".join(rows)

def _world_vars(world: str):
    """Return world-mode prompt variables used by safe_format."""
    w = _normalize_world_mode(world)
    return WORLD_LABELS.get(w, "CLOSED"), EXTERNAL_WORLD_RULES.get(w, EXTERNAL_WORLD_RULES["closed"])


def _version_prefix(version_set: str) -> str:
    """Extract the root version identifier from a version_set string."""
    return str(version_set or "").strip().split("_")[0]


def _join_bullets(lines):
    """Render a list of strings as prompt bullet points."""
    items = [str(x).strip() for x in (lines or []) if str(x).strip()]
    return "\n".join(f"- {x}" for x in items)


def build_fact_pack_text(version_set: str, mode: str) -> str:
    """
    Build the topic fact-pack block injected into prompts for the selected mode.
    
        Fact packs are curated evidence sets separate from local RAG. The mode selects supportive,
        critical, contextual, balanced, full, or off content so experiments can isolate evidence direction.
    """
    m = str(mode or "off").strip().lower()
    if m in {"", "off", "none"}:
        return ""
    if m == "on":
        m = "full"

    version_key = _version_prefix(version_set)
    pack = FACT_PACKS.get(version_key)
    if not pack:
        return ""

    title = str(pack.get("title", "")).strip()
    core = list(pack.get("core", []))
    context = list(pack.get("context", []))
    criticisms = list(pack.get("criticisms", []))
    notes = list(pack.get("notes", []))

    sections = []

    def add_section(header: str, lines):
        """Append one fact-pack section if it has content for the selected mode."""
        block = _join_bullets(lines)
        if block:
            sections.append(f"{header}:\n{block}")

    if m == "core":
        add_section("Supportive points sometimes cited", core)
    elif m == "context":
        add_section("Contextual points sometimes cited", context)
    elif m == "criticisms":
        add_section("Claim-challenging points", criticisms)
    elif m == "balanced":
        add_section("Supportive points sometimes cited", core)
        add_section("Contextual points sometimes cited", context)
        add_section("Claim-challenging points", criticisms)
    elif m == "full":
        add_section("Supportive points sometimes cited", core)
        add_section("Contextual points sometimes cited", context)
        add_section("Claim-challenging points", criticisms)
        add_section("Notes", notes)
    else:
        return ""

    body = "\n\n".join(s for s in sections if s.strip())
    if not body:
        return ""
    return f"{title}\n\n{body}" if title else body


WORLD, WORLD_RULES = _world_vars(getattr(args, "world", "closed"))
def _shown_material_direction(world: str, fact_pack_mode: str, rag_content_mode: str) -> str:
    """Classify shown material as supportive, critical, mixed, context, or unknown."""
    if _rag_enabled_for_world(world):
        mode = str(rag_content_mode or "full").strip().lower()
        if mode == "criticism_only":
            return "criticism_only"
        if mode == "supportive_only":
            return "supportive_only"
        if mode == "context_only":
            return "context_only"
        return "mixed"
    if _is_criticism_only_fact_pack_mode(fact_pack_mode):
        return "criticism_only"
    if _normalize_fact_pack_mode(fact_pack_mode) in {"core", "context"}:
        return "supportive_only"
    if _normalize_fact_pack_mode(fact_pack_mode) in {"balanced", "full"}:
        return "mixed"
    return "none"


def _step2_mismatched_shown_material_rules(current_belief: int, world: str, fact_pack_mode: str, rag_content_mode: str) -> str:
    """
    Generate extra Step-2 safeguards when shown evidence conflicts with speaker stance.
    
        For example, a positive speaker seeing criticism-only material should not turn that criticism into
        support without qualification. These rules reduce direction inversions in generated tweets.
    """
    direction = _shown_material_direction(world, fact_pack_mode, rag_content_mode)
    try:
        b = int(current_belief)
    except Exception:
        return ""
    if direction == "criticism_only" and b > 0:
        return STEP2_MISMATCHED_SHOWN_MATERIAL_RULES_GENERIC
    if direction == "supportive_only" and b < 0:
        return STEP2_MISMATCHED_SHOWN_MATERIAL_RULES_GENERIC
    return ""


STEP3_OPEN_WORLD_RULES = _step3_open_world_rules(
    getattr(args, "world", "closed"),
    getattr(args, "step3_open_world_rules", "on"),
)
FACT_PACK_TEXT = build_fact_pack_text(
    getattr(args, "version_set", ""),
    getattr(args, "fact_pack_mode", "off"),
)

# For clean comparison, closed_strict_rag uses retrieval-only, not static fact packs.
if _normalize_world_mode(getattr(args, "world", "closed")) == "closed_strict_rag":
    FACT_PACK_TEXT = ""

STEP2_FACT_PACK_RULES = _step2_fact_pack_rules(
    FACT_PACK_TEXT,
    getattr(args, "fact_pack_mode", "off"),
    getattr(args, "world", "closed"),
)
STEP3_FACT_PACK_RULES = _step3_fact_pack_rules(
    FACT_PACK_TEXT,
    getattr(args, "fact_pack_mode", "off"),
    getattr(args, "world", "closed"),
)
STEP2_RAG_RULES = STEP2_RAG_RULES_GENERIC if _rag_enabled_for_world(getattr(args, "world", "closed")) else ""
STEP3_RAG_RULES = STEP3_RAG_RULES_GENERIC if _rag_enabled_for_world(getattr(args, "world", "closed")) else ""
THEORY_STATEMENT = _clean_optional_prompt_text(getattr(args, "theory_statement", ""))
BIAS_TEXT = _clean_optional_prompt_text(getattr(args, "bias_text", ""))
VERSION_METADATA = _load_version_metadata(getattr(args, "version_metadata_path", ""))
RAG_CORPUS = _maybe_init_rag_corpus()
_init_rag_architecture()


def _bias_mode_from_text(bias_text: str) -> str:
    """Infer none/weak/strong confirmation-bias mode from prompt text."""
    s = re.sub(r"\s+", " ", str(bias_text or "")).strip().lower()
    if not s:
        return "default"
    if "strong confirmation bias" in s:
        return "strong"
    if "weak confirmation bias" in s:
        return "weak"
    if "no confirmation bias" in s:
        return "default"
    if "confirmation bias" in s and "strong" not in s:
        return "weak"
    return "default"


def _is_strong_confirmation_bias_active(bias_text: str | None = None) -> bool:
    """Return whether strong confirmation bias is active for the current run."""
    return _bias_mode_from_text(BIAS_TEXT if bias_text is None else bias_text) == "strong"


def _is_free_bounded_update_mode() -> bool:
    """Return whether the active update mode allows movement in either direction."""
    return str(globals().get("ALLOWED_UPDATE_MODE", "assimilation_only") or "assimilation_only").strip().lower() == "free_bounded"


def _step3_update_mode_note(allowed_str: str) -> str:
    """Build the Step-3 prompt note explaining the allowed-rating set."""
    if not _is_free_bounded_update_mode():
        return ""
    return (
        "\nUpdate mode: free_bounded.\n"
        f"- {allowed_str} is the only movement boundary.\n"
        "- You may stay or move one bounded step in either direction, including after a same-rating interaction.\n"
        "- A final rating may move toward the speaker, away from the speaker, or stay unchanged if the current tweet and shown material justify it.\n"
        "- The explanation must still match FINAL_RATING and cite one concrete point from the tweet or shown material."
    )



def _step3_free_bounded_decision_rule_block(allowed_str: str) -> str:
    """Return Step-3 decision rules specific to free_bounded updates."""
    return (
        "Decision rule\n"
        "1. Read the current tweet and identify its strongest concrete point.\n"
        "2. Compare that point with any shown material and your recent memory, but do not treat memory as new evidence.\n"
        f"3. Choose the allowed final rating from {allowed_str} that best matches your updated belief.\n"
        "4. You may move one bounded step toward the tweet, away from the tweet, or stay unchanged if that is what the tweet and shown material justify.\n"
        "5. Same-rating interactions can still change your opinion; the speaker's rating is not a movement constraint in this mode.\n"
        "6. Base the explanation on one concrete point from the tweet or shown material, not on your internal state."
    )


def _replace_step3_decision_rule_for_update_mode(prompt_text: str, allowed_str: str) -> str:
    """
    Replace legacy Step-3 movement rules with rules matching the active update mode.
    
        The same prompt templates can be reused for assimilation_only and free_bounded experiments. This
        function ensures the final decision instructions match the allowed-rating set computed by the code.
    """
    s = str(prompt_text or "")
    if not _is_free_bounded_update_mode():
        return s
    block = _step3_free_bounded_decision_rule_block(allowed_str)
    pat = r"(?is)Decision rule\s*\n.*?(?=\n\s*(?:Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)\b)"
    new_s, n = re.subn(pat, block + "\n", s, count=1)
    if n:
        return re.sub(r"\n{3,}", "\n\n", new_s).strip()
    new_s, n = re.subn(r"(?is)(\n\s*Explanation rule\b)", "\n" + block + "\n\\1", s, count=1)
    if n:
        return re.sub(r"\n{3,}", "\n\n", new_s).strip()
    return (s.rstrip() + "\n\n" + block).strip()


def _strong_bias_extra_note(pre_belief: int, tweet_stance: str | None) -> str:
    """Return an additional Step-3 caution note under strong confirmation bias."""
    if not _is_strong_confirmation_bias_active():
        return ""
    try:
        pre = int(pre_belief)
    except Exception:
        return ""
    if pre == 0 or tweet_stance not in {"support", "oppose"}:
        return ""

    incompatible = (pre < 0 and tweet_stance == "support") or (pre > 0 and tweet_stance == "oppose")
    if not incompatible:
        return ""

    if abs(pre) >= 2:
        return (
            "\nStrong-bias reminder:\n"
            "- Because your current view is already strong in the opposite direction, move away from it only if this tweet gives a very strong, specific, claim-relevant reason.\n"
            "- A single ordinary point is usually not enough to shift a strongly held opposite view, even by one step."
        )

    return (
        "\nStrong-bias reminder:\n"
        "- Because you currently lean the other way, do not change direction unless this tweet gives a clearly stronger, specific, claim-relevant reason than an ordinary opposing post."
    )



def _strip_legacy_step3_stance_hints(prompt_text: str) -> str:
    """Remove older Step-3 stance-hint lines if a generated prompt still contains them."""
    s = str(prompt_text or "")
    s = re.sub(r"^\s*Note:\s*the tweet appears to argue\s+(?:FOR|AGAINST)\s+the claim\.\s*$", "", s, flags=re.I | re.M)
    s = re.sub(r"^\s*Note:\s*the tweet appears to be\s+UNCERTAIN/MIXED\.\s*$", "", s, flags=re.I | re.M)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _active_fact_pack_polarity_points(version_set: str, mode: str) -> dict:
    """Map current fact-pack sections into generic positive/negative point pools.

    Convention:
      - core/context => positive/supportive points
      - criticisms   => negative/skeptical points
    This lets future topics reuse the same section names without code changes.
    """
    version_key = _version_prefix(version_set)
    pack = FACT_PACKS.get(version_key) or {}
    m = str(mode or "off").strip().lower()
    if m in {"", "off", "none"}:
        return {"positive": [], "negative": []}
    if m == "on":
        m = "full"

    positive = []
    negative = []

    if m in {"core", "balanced", "full"}:
        positive.extend(list(pack.get("core", []) or []))
    if m in {"context", "balanced", "full"}:
        positive.extend(list(pack.get("context", []) or []))
    if m in {"criticisms", "balanced", "full"}:
        negative.extend(list(pack.get("criticisms", []) or []))

    def _uniq(seq):
        """Deduplicate items while preserving their first-seen order."""
        out = []
        seen = set()
        for x in seq:
            s = str(x or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return {"positive": _uniq(positive), "negative": _uniq(negative)}


_ACTIVE_FACT_PACK_POLARITY_POINTS = _active_fact_pack_polarity_points(
    getattr(args, "version_set", ""),
    getattr(args, "fact_pack_mode", "off"),
)

_FACT_MATCH_STOPWORDS = {
    "the","a","an","and","or","but","if","then","than","that","this","these","those","to","of","for","from","in","on","at","by","with","as","is","are","was","were","be","been","being","it","its","it's","their","there","here","about","into","after","before","during","over","under","between","through","across","because","while","when","where","which","who","whom","whose","can","could","would","should","may","might","must","do","does","did","done","have","has","had","having","not","no","yes","more","most","less","least","many","much","some","any","each","every","other","another","same","only","just","also","still","very","than","too","few","all","both","either","neither","one","two"
}


def _fact_match_tokens(text: str) -> list[str]:
    """Tokenize fact-pack text for evidence-matching heuristics."""
    raw = re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
    out = []
    for tok in raw:
        if tok.isdigit():
            out.append(tok)
            continue
        if len(tok) < 4:
            continue
        if tok in _FACT_MATCH_STOPWORDS:
            continue
        out.append(tok)
    return out


def _tweet_matches_fact_pack_polarity(tweet_text: str, polarity: str) -> bool:
    """Check whether a tweet overlaps with supportive or critical fact-pack material."""
    pools = _ACTIVE_FACT_PACK_POLARITY_POINTS or {}
    points = list(pools.get(str(polarity or "").strip().lower(), []) or [])
    if not points:
        return False

    tweet_tokens = set(_fact_match_tokens(tweet_text))
    if not tweet_tokens:
        return False

    for point in points:
        point_tokens = set(_fact_match_tokens(point))
        if not point_tokens:
            continue
        overlap = tweet_tokens & point_tokens
        if len(overlap) >= 2:
            return True
        if any(tok.isdigit() for tok in overlap) and len(overlap) >= 1:
            # Numeric anchor plus at least one shared long token often captures fact-pack points cleanly.
            return True
    return False



_FACT_SPECIFICITY_GENERIC_TOKENS = {
    "evidence", "data", "documentation", "documented", "documents", "records", "record",
    "sources", "source", "proof", "support", "supports", "supportive", "claim", "claims",
    "program", "success", "successful", "returned", "return", "achievement", "achievements",
    "details", "detail", "information", "context", "result", "results", "reported", "report",
    "facts", "fact", "study", "studies", "research", "history", "historical",
    "finding", "findings", "material", "materials", "object", "objects", "item", "items",
    "event", "events", "process", "processes", "system", "systems", "mission", "missions"
}


def _specific_fact_point_profiles(polarity: str) -> list[dict]:
    """Build topic-agnostic specificity profiles from the active fact pack.

    The goal is generic behavior across topics and fact packs, not hardcoded entity lists.
    A point is treated as sufficiently specific when it has either:
      - a numeric anchor, or
      - at least two non-generic content tokens, or
      - one non-generic content token plus enough total content to indicate a concrete point.
    """
    pol = str(polarity or '').strip().lower()
    pools = _ACTIVE_FACT_PACK_POLARITY_POINTS or {}
    points = list(pools.get(pol, []) or [])
    profiles: list[dict] = []
    for point in points:
        point_tokens = set(_fact_match_tokens(point))
        if not point_tokens:
            continue
        digits = {tok for tok in point_tokens if tok.isdigit()}
        non_generic = {tok for tok in point_tokens if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
        generic = {tok for tok in point_tokens if tok in _FACT_SPECIFICITY_GENERIC_TOKENS}
        # Keep only points that look concretely grounded, but avoid requiring extremely narrow overlap.
        if not (digits or len(non_generic) >= 2 or (len(non_generic) >= 1 and len(point_tokens) >= 4)):
            continue
        profiles.append({
            "all_tokens": point_tokens,
            "digits": digits,
            "non_generic": non_generic,
            "generic": generic,
        })
    return profiles


def _tweet_matches_specific_fact_pack_polarity(tweet_text: str, polarity: str) -> bool:
    """Generic specificity matcher for center-projection repairs.

    This remains topic-agnostic by deriving specificity from the *active fact pack*.
    It is intentionally a bit looser than an entity-exact matcher, because different topics
    and fact-pack styles may express concrete points with varying wording.
    """
    pol = str(polarity or '').strip().lower()
    if not _tweet_matches_fact_pack_polarity(tweet_text, pol):
        return False

    tweet_tokens = set(_fact_match_tokens(tweet_text))
    if not tweet_tokens:
        return False

    tweet_digits = {tok for tok in tweet_tokens if tok.isdigit()}
    tweet_non_generic = {tok for tok in tweet_tokens if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
    tweet_all_content = {tok for tok in tweet_tokens if tok not in _FACT_MATCH_STOPWORDS}

    profiles = _specific_fact_point_profiles(pol)
    if not profiles:
        return False

    for prof in profiles:
        sig_digits = set(prof.get("digits", set()))
        sig_terms = set(prof.get("non_generic", set()))
        sig_all = set(prof.get("all_tokens", set()))

        # Strongest signal: shared numeric anchor.
        if tweet_digits & sig_digits:
            return True

        overlap_terms = tweet_non_generic & sig_terms
        if len(overlap_terms) >= 2:
            return True

        # One distinctive shared term is enough when the fact-pack point itself is concrete.
        if len(overlap_terms) == 1 and (sig_digits or len(sig_terms) >= 3 or len(sig_all) >= 5):
            return True

        # Fallback: two shared content tokens across the full point, as long as at least
        # one of them is not merely a generic evidence word.
        overlap_all = tweet_all_content & sig_all
        if len(overlap_all) >= 2 and (overlap_terms or sig_digits):
            return True
    return False



def _short_fact_point_label(point_text: str) -> str:
    """Create a compact human-readable label for a matched fact-pack point."""
    toks = [t for t in _fact_match_tokens(point_text) if t not in _FACT_SPECIFICITY_GENERIC_TOKENS and not t.isdigit()]
    if not toks:
        toks = [t for t in _fact_match_tokens(point_text) if not t.isdigit()]
    label = "_".join(toks[:2]).strip("_")
    return label or "generic_point"


def _best_matching_fact_pack_point_label(tweet_text: str) -> str:
    """Return the best fact-pack point label matched by a tweet."""
    tweet_tokens = set(_fact_match_tokens(tweet_text))
    if not tweet_tokens:
        return "none"
    best_label = "none"
    best_score = 0
    pools = _ACTIVE_FACT_PACK_POLARITY_POINTS or {}
    for pol in ("positive", "negative"):
        for point in list(pools.get(pol, []) or []):
            point_tokens = set(_fact_match_tokens(point))
            if not point_tokens:
                continue
            overlap = tweet_tokens & point_tokens
            score = len(overlap)
            if score > best_score:
                best_score = score
                best_label = f"{pol}::{_short_fact_point_label(point)}"
    return best_label if best_score >= 2 else "none"


def _log_step2_cue_metric(tweet_text: str):
    """Increment a Step-2 cue diagnostic metric."""
    label = _best_matching_fact_pack_point_label(tweet_text)
    _metric_inc(f"step2_cue::{label}")

def _same_rating_generic_fact_dismissal_tag(pre_value: int, final_value: int, tweet_stance_local: str | None, tweet_text: str, explanation_text: str) -> str | None:
    """Flag same-rating generic dismissal of a specific opposite-side fact-based point."""
    try:
        pre_i = int(pre_value)
        final_i = int(final_value)
    except Exception:
        return None

    if final_i != pre_i:
        return None

    stance = str(tweet_stance_local or "").strip().lower()
    opposite_fact = False
    if pre_i < 0 and stance == "support" and _tweet_matches_specific_fact_pack_polarity(tweet_text, "positive"):
        opposite_fact = True
    elif pre_i > 0 and stance == "oppose" and _tweet_matches_fact_pack_polarity(tweet_text, "negative"):
        opposite_fact = True
    elif pre_i == 0 and stance == "support" and _tweet_matches_specific_fact_pack_polarity(tweet_text, "positive"):
        opposite_fact = True
    elif pre_i == 0 and stance == "oppose" and _tweet_matches_fact_pack_polarity(tweet_text, "negative"):
        opposite_fact = True
    else:
        return None

    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not expl:
        return None

    generic_dismissal = bool(re.search(
        r"\b("
        r"not enough(?: by itself| from this alone| on its own)?|"
        r"not enough to change|"
        r"does(?: not|n't) change my (?:current )?(?:stance|mind|rating)|"
        r"does(?: not|n't) (?:really )?address my concerns|"
        r"does(?: not|n't) (?:fully|directly|really) address (?:the )?(?:issue|concern|problem)|"
        r"does(?: not|n't) settle (?:it|the issue|the claim)|"
        r"does(?: not|n't) prove (?:it|the claim)|"
        r"does(?: not|n't) show (?:it|the claim)|"
        r"does(?: not|n't) establish (?:it|the claim)|"
        r"from (?:this|that) alone|"
        r"by itself|"
        r"on its own|"
        r"in this tweet|"
        r"still have doubts|"
        r"still doubt|"
        r"still unconvinced|"
        r"remain unconvinced|"
        r"remain skeptical|"
        r"remain doubtful|"
        r"still concerns me|"
        r"still raises concerns|"
        r"still leaves me doubtful|"
        r"still leaves me skeptical|"
        r"still leaves room for doubt|"
        r"still not enough|"
        r"still (?:do not|don't) see (?:any )?concrete evidence|"
        r"do(?:es)? not provide (?:any )?concrete evidence|"
        r"not enough concrete evidence|"
        r"still unsure about the validity|"
        r"still unsure about the authenticity|"
        r"still unsure about the reliability|"
        r"still have concerns about the validity|"
        r"still have concerns about the authenticity|"
        r"still have concerns about the reliability|"
        r"still skeptical about the overall claim|"
        r"still skeptical about the mission|"
        r"still skeptical about the landing|"
        r"still need stronger evidence|"
        r"still need more convincing evidence|"
        r"still need more concrete evidence|"
        r"not enough (?:detail|details|information|context)|"
        r"does(?: not|n't) give me enough (?:detail|details|information|context)|"
        r"does(?: not|n't) tell me enough|"
        r"does(?: not|n't) explain (?:how|why)|"
        r"i need more than (?:that|this)|"
        r"does(?: not|n't) outweigh my (?:doubts|skepticism|concerns)|"
        r"does(?: not|n't) overcome my (?:doubts|skepticism|concerns)|"
        r"remain(?:s|ing)? unconvinced by this (?:alone|point)|"
        r"remain(?:s|ing)? skeptical despite this (?:point|evidence)|"
        r")\b",
        expl,
        flags=re.I,
    ))

    # Strong-edge holds (-2/+2) need a more specific counterreason than broad
    # meta-skepticism when the tweet contains a concrete opposite-side fact-pack point.
    strong_edge_vague_hold = False
    if abs(pre_i) == 2:
        strong_edge_vague_hold = bool(re.search(
            r"\b("
            r"still (?:do not|don't) see (?:any )?concrete evidence|"
            r"still unsure about the (?:overall )?(?:validity|authenticity|reliability|legitimacy)|"
            r"still have concerns about the (?:overall )?(?:validity|authenticity|reliability|legitimacy)|"
            r"still skeptical about the (?:overall )?(?:claim|mission|landing)|"
            r"still unconvinced about the (?:overall )?(?:claim|mission|landing)|"
            r"not enough concrete evidence"
            r")\b",
            expl,
            flags=re.I,
        ))

    if not generic_dismissal and not strong_edge_vague_hold:
        return None

    best_clause = _best_reason_clause_from_explanation(expl)
    if best_clause is None:
        return "specific_fact_generic_dismissal"

    # If the only "best" clause is itself generic no-change language, still flag it.
    if re.search(
        r"\b(not enough|still have doubts|does(?: not|n't) change|does(?: not|n't) (?:really )?address|does(?: not|n't) settle|does(?: not|n't) prove|does(?: not|n't) show|does(?: not|n't) establish|remain(?:s|ing)? skeptical|remain(?:s|ing)? unconvinced|valid point but|specific point but|i need more than (?:that|this)|from (?:this|that) alone|by itself|on its own|in this tweet)\b",
        best_clause,
        flags=re.I,
    ):
        return "specific_fact_generic_dismissal"

    return None

def _fact_based_opposite_side_inversion_tag(pre_value: int, tweet_stance_local: str | None, tweet_text: str, explanation_text: str) -> str | None:
    """
    Detect when a tweet uses fact-pack material in the opposite direction from its polarity.
    
        The tag helps identify cases where a supportive fact is framed as criticism or vice versa. These
        events are logged as Step-2 warnings because they can contaminate the intended evidence condition.
    """
    try:
        pre_i = int(pre_value)
    except Exception:
        return None

    stance = str(tweet_stance_local or "").strip().lower()
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not expl:
        return None

    if pre_i < 0 and stance == "support" and _tweet_matches_specific_fact_pack_polarity(tweet_text, "positive"):
        # Pro fact interpreted as reinforcing negativity / skepticism.
        if re.search(
            r"\b("
            r"even more skeptical|more skeptical|"
            r"more doubtful|still more doubtful|"
            r"more against the claim|"
            r"reinforces? my (?:negative|skeptical|doubtful) stance|"
            r"strengthens? my (?:negative|skeptical|doubtful) stance|"
            r"makes? me (?:even )?less likely to believe|"
            r"push(?:es|ed)? me (?:even )?more negative"
            r")\b",
            expl,
            flags=re.I,
        ):
            return "specific_fact_polarity_inversion"

    if pre_i > 0 and stance == "oppose" and _tweet_matches_fact_pack_polarity(tweet_text, "negative"):
        # Anti fact interpreted as reinforcing positivity / belief.
        if re.search(
            r"\b("
            r"even more convinced|more convinced|"
            r"more likely to believe|"
            r"more in favor of the claim|"
            r"reinforces? my (?:positive|supportive|belief) stance|"
            r"strengthens? my (?:positive|supportive|belief) stance|"
            r"makes? me (?:even )?more likely to believe|"
            r"push(?:es|ed)? me (?:even )?more positive"
            r")\b",
            expl,
            flags=re.I,
        ):
            return "specific_fact_polarity_inversion"

    return None


try:
    print(f"[config] world={WORLD} (mode={getattr(args, 'world', 'closed')})")
except Exception:
    pass

try:
    print(
        f"[config] step3_open_world_rules={getattr(args, 'step3_open_world_rules', 'on')} "
        f"active={'yes' if bool(str(STEP3_OPEN_WORLD_RULES).strip()) else 'no'}"
    )
except Exception:
    pass

try:
    print(
        f"[config] fact_pack_mode={getattr(args, 'fact_pack_mode', 'off')} "
        f"active={'yes' if bool(str(FACT_PACK_TEXT).strip()) else 'no'}"
    )
except Exception:
    pass

try:
    print(f"[config] step2_fact_pack_rules_active={'yes' if bool(str(STEP2_FACT_PACK_RULES).strip()) else 'no'}")
    print(f"[config] step2_fact_pack_rules_len={len(str(STEP2_FACT_PACK_RULES or ''))}")
except Exception:
    pass

try:
    print(f"[config] step3_fact_pack_rules_active={'yes' if bool(str(STEP3_FACT_PACK_RULES).strip()) else 'no'}")
except Exception:
    pass


def _ensure_step3_open_world_placeholder(prompt_template: str) -> str:
    """Ensure Step-3 templates have a safe insertion point for {STEP3_OPEN_WORLD_RULES}."""
    s = str(prompt_template or "")
    if "{STEP3_OPEN_WORLD_RULES}" in s:
        return s
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{STEP3_OPEN_WORLD_RULES}", 1)
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{STEP3_OPEN_WORLD_RULES}\n\n{FACT_PACK}", 1)
    return s.rstrip() + "\n\n{STEP3_OPEN_WORLD_RULES}"


def _ensure_step2_fact_pack_rules_placeholder(prompt_template: str) -> str:
    """Ensure Step-2 templates have a safe insertion point for {STEP2_FACT_PACK_RULES}."""
    s = str(prompt_template or "")
    if "{STEP2_FACT_PACK_RULES}" in s:
        return s
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{STEP2_FACT_PACK_RULES}", 1)
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{STEP2_FACT_PACK_RULES}", 1)
    return s.rstrip() + "\n\n{STEP2_FACT_PACK_RULES}"


def _ensure_step3_fact_pack_rules_placeholder(prompt_template: str) -> str:
    """Ensure Step-3 templates have a safe insertion point for {STEP3_FACT_PACK_RULES}."""
    s = str(prompt_template or "")
    if "{STEP3_FACT_PACK_RULES}" in s:
        return s
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{STEP3_FACT_PACK_RULES}", 1)
    if "{STEP3_OPEN_WORLD_RULES}" in s:
        return s.replace("{STEP3_OPEN_WORLD_RULES}", "{STEP3_OPEN_WORLD_RULES}\n\n{STEP3_FACT_PACK_RULES}", 1)
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{STEP3_FACT_PACK_RULES}", 1)
    return s.rstrip() + "\n\n{STEP3_FACT_PACK_RULES}"


def _ensure_step2_rag_rules_placeholder(prompt_template: str) -> str:
    """Ensure Step-2 templates have a safe insertion point for {STEP2_RAG_RULES}."""
    s = str(prompt_template or "")
    if "{STEP2_RAG_RULES}" in s:
        return s
    if "{STEP2_FACT_PACK_RULES}" in s:
        return s.replace("{STEP2_FACT_PACK_RULES}", "{STEP2_FACT_PACK_RULES}\n\n{STEP2_RAG_RULES}", 1)
    if "{RAG_CONTEXT}" in s:
        return s.replace("{RAG_CONTEXT}", "{RAG_CONTEXT}\n\n{STEP2_RAG_RULES}", 1)
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{STEP2_RAG_RULES}", 1)
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{STEP2_RAG_RULES}", 1)
    return s.rstrip() + "\n\n{STEP2_RAG_RULES}"


def _ensure_step2_mismatched_rules_placeholder(prompt_template: str) -> str:
    """Ensure Step-2 templates have a safe insertion point for {STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}."""
    s = str(prompt_template or "")
    if "{STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}" in s:
        return s
    if "{STEP2_RAG_RULES}" in s:
        return s.replace("{STEP2_RAG_RULES}", "{STEP2_RAG_RULES}\n\n{STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}", 1)
    if "{STEP2_FACT_PACK_RULES}" in s:
        return s.replace("{STEP2_FACT_PACK_RULES}", "{STEP2_FACT_PACK_RULES}\n\n{STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}", 1)
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}", 1)
    return s.rstrip() + "\n\n{STEP2_MISMATCHED_SHOWN_MATERIAL_RULES}"


def _ensure_step3_rag_rules_placeholder(prompt_template: str) -> str:
    """Ensure Step-3 templates have a safe insertion point for {STEP3_RAG_RULES}."""
    s = str(prompt_template or "")
    if "{STEP3_RAG_RULES}" in s:
        return s
    if "{STEP3_FACT_PACK_RULES}" in s:
        return s.replace("{STEP3_FACT_PACK_RULES}", "{STEP3_FACT_PACK_RULES}\n\n{STEP3_RAG_RULES}", 1)
    if "{RAG_CONTEXT}" in s:
        return s.replace("{RAG_CONTEXT}", "{RAG_CONTEXT}\n\n{STEP3_RAG_RULES}", 1)
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{STEP3_RAG_RULES}", 1)
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{STEP3_RAG_RULES}", 1)
    return s.rstrip() + "\n\n{STEP3_RAG_RULES}"


def _ensure_rag_context_placeholder(prompt_template: str) -> str:
    """Ensure templates have a safe insertion point for {RAG_CONTEXT} only in RAG worlds."""
    s = str(prompt_template or "")
    # Do not inject a RAG placeholder at all for non-RAG worlds.
    if not _rag_enabled_for_world(getattr(args, "world", "closed")):
        return s
    if "{RAG_CONTEXT}" in s:
        return s
    if "{FACT_PACK}" in s:
        return s.replace("{FACT_PACK}", "{FACT_PACK}\n\n{RAG_CONTEXT}", 1)
    if "{WORLD_RULES}" in s:
        return s.replace("{WORLD_RULES}", "{WORLD_RULES}\n{RAG_CONTEXT}", 1)
    return s.rstrip() + "\n\n{RAG_CONTEXT}"

STEP3_OPEN_WORLD_RULES = _step3_open_world_rules(
    getattr(args, "world", "closed"),
    getattr(args, "step3_open_world_rules", "on"),
)

try:
    print(f"[config] world={WORLD} (mode={getattr(args, 'world', 'closed')})")
except Exception:
    pass

try:
    print(
        f"[config] step3_open_world_rules={getattr(args, 'step3_open_world_rules', 'on')} "
        f"active={'yes' if bool(str(STEP3_OPEN_WORLD_RULES).strip()) else 'no'}"
    )
except Exception:
    pass

try:
    print(f"[config] max_step_change={DEFAULT_MAX_STEP_CHANGE}")
    print(f"[config] allowed_update_mode={ALLOWED_UPDATE_MODE}")
    print(f"[config] validation_strictness={VALIDATION_STRICTNESS}")
    print(f"[config] wrong_side_explanation_requery={WRONG_SIDE_EXPL_REQUERY}")
    print(f"[config] deterministic={DETERMINISTIC}")
    print(f"[config] same_side_edge_unlock_hits={SAME_SIDE_EDGE_UNLOCK_HITS}")
    print(f"[config] same_rating_step3_mode={SAME_RATING_STEP3_MODE}")
    print(f"[config] think_mode={getattr(args, 'think_mode', 'off')}")
    print(f"[config] theory_statement={'set' if bool(THEORY_STATEMENT) else 'empty'}")
    print(f"[config] version_metadata_entries={len(VERSION_METADATA) if isinstance(VERSION_METADATA, dict) else 0}")
    print(f"[config] bias_text={'set' if bool(BIAS_TEXT) else 'empty'}")
except Exception:
    pass
def _ensure_persona_card_contains_row_fields(persona_text: str, row_meta, agent_name: str = "") -> str:
    """Backfill AGENT_PERSONA_CARD from CSV columns when the persona field is stale/name-only."""
    try:
        raw = str(persona_text or "").strip()
    except Exception:
        raw = ""
    lines = [ln.rstrip() for ln in raw.splitlines() if str(ln).strip()]
    if not lines:
        lines = ["AGENT_PERSONA_CARD"]
    if not any(str(ln).strip().upper() == "AGENT_PERSONA_CARD" for ln in lines[:2]):
        lines.insert(0, "AGENT_PERSONA_CARD")

    def _get(col: str) -> str:
        """Read one optional row/key value without failing on missing fields."""
        try:
            return _csv_clean_value(row_meta.get(col, ""))
        except Exception:
            return ""

    normalized = {ln.split(':', 1)[0].strip().lower(): idx for idx, ln in enumerate(lines) if ':' in ln}

    def _add(label: str, value: str):
        """Append one formatted persona-card line when the value is present."""
        value = str(value or "").strip()
        if not value:
            return
        key = label.strip().lower()
        line = f"{label}: {value}"
        if key in normalized:
            lines[normalized[key]] = line
        else:
            normalized[key] = len(lines)
            lines.append(line)

    _add("Name", agent_name or _get("agent_name"))
    _add("Epistemic profile", _get("epistemic_profile"))
    _add("Institutional trust", _get("institutional_trust"))
    _add("Uncertainty tolerance", _get("uncertainty_tolerance"))
    _add("Evidence style", _get("evidence_style"))
    _add("Official-narrative suspicion", _get("official_narrative_suspicion"))
    _add("Openness to update", _get("openness_to_update"))
    _add("Value orientation", _get("value_orientation"))
    _add("Agency vs fatalism", _get("agency_vs_fatalism"))
    _add("Conflict style", _get("conflict_style"))
    _add("Occupation", _get("occupation"))
    _add("Education", _get("education") or _get("education_level"))
    _add("Education level", _get("education_level"))
    _add("Training style", _get("training_style"))
    _add("Domain familiarity", _get("domain_familiarity"))
    _add("Topic interest", _get("topic_interest"))
    _add("Prior exposure", _get("prior_exposure"))
    _add("Age", _get("age") or _get("age_group"))
    _add("Age group", _get("age_group"))
    _add("Gender", _get("gender") or _get("flavor_gender"))
    _add("Flavor gender", _get("flavor_gender"))
    _add("Ethnicity", _get("ethnicity") or _get("flavor_ethnicity"))
    _add("Flavor ethnicity", _get("flavor_ethnicity"))
    _add("Lifestyle notes", _get("lifestyle_notes"))
    _add("Tone hint", _get("tone_hint"))
    _add("Political leaning", _get("political_leaning"))
    _add("Early life", _get("early_life"))
    return "\n".join(lines).strip()


def make_public_persona(persona_text: str) -> str:
    """
    Build the persona text that is safe to show to the agent during decision steps.
    
        The public card removes stale or state-leaking fields such as initial rating and current belief,
        while preserving stable causal/persona attributes. This prevents persona text from duplicating or
        contradicting the numeric belief state maintained by the simulator.
    """
    if not persona_text:
        return persona_text

    lines = str(persona_text).splitlines()
    filtered = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("Initial rating:"):
            continue
        if s.startswith("Belief:"):
            continue
        filtered.append(ln)
    return "\n".join(filtered).strip()



# ============================
# HISTORY GATING + TERMINAL TRACE
# ============================

# ============================
# HISTORY GATING + TERMINAL TRACE
# ============================

# LangChain v0.2+ validates ConversationChain.memory via Pydantic and requires it to be a BaseMemory.
# We wrap a real BaseMemory (ConversationBufferMemory) but optionally *hide* history from the prompt
# (Markovian mode) while still *storing* it for debugging/logging.
# We need ConversationChain(memory=...) to receive a valid Memory object.
# We use a real ConversationBufferMemory but optionally *hide* history from the LLM prompt
# (Markovian mode) while still *storing* it for debugging/logging.
class GatedConversationBufferMemory(ConversationBufferMemory):
    """
    Conversation memory wrapper that can store history while hiding it from the prompt.
    
        This is what makes memory-off/Markovian runs possible without removing logging infrastructure.
        When the history gate is closed, the LLM receives only the current prompt even though memory events
        may still be recorded for debugging or later analysis.
    """

    expose_history: bool = True  # pydantic field (works for both v1/v2)
    memory_step_kind: str = ""  # step2/step3; controls the temporary memory-use rule

    def _empty_payload(self):
        """Return an empty memory payload when conversation history must be hidden."""
        keys = getattr(self, "memory_variables", None) or [getattr(self, "memory_key", "history")]
        if bool(getattr(self, "return_messages", False)):
            return {k: [] for k in keys}
        return {k: "" for k in keys}

    def _prepend_memory_use_rule(self, payload):
        """Prepend step-specific memory-use instructions to visible history."""
        rule = _memory_use_rules_for_step(getattr(self, "memory_step_kind", ""))
        if not rule:
            return payload
        try:
            keys = getattr(self, "memory_variables", None) or [getattr(self, "memory_key", "history")]
            key = keys[0]
            if bool(getattr(self, "return_messages", False)):
                hist = list(payload.get(key) or [])
                if hist:
                    payload[key] = [HumanMessage(content=rule)] + hist
            else:
                hist = str(payload.get(key) or "")
                if hist.strip():
                    payload[key] = rule + "\n\n" + hist
        except Exception:
            return payload
        return payload

    def load_memory_variables(self, inputs):
        """
        Return visible memory to LangChain only when the history gate is open.
        
            The method overrides ConversationBufferMemory behavior so a run can distinguish between storing
            interaction history and actually conditioning future LLM calls on that history.
        """
        if not bool(getattr(self, "expose_history", True)):
            return self._empty_payload()
        payload = super().load_memory_variables(inputs)
        return self._prepend_memory_use_rule(payload)

    async def aload_memory_variables(self, inputs):
        """Memory-gate method for aload memory variables."""
        if not bool(getattr(self, "expose_history", True)):
            return self._empty_payload()
        parent = super()
        if hasattr(parent, "aload_memory_variables"):
            payload = await parent.aload_memory_variables(inputs)
        else:
            payload = parent.load_memory_variables(inputs)
        return self._prepend_memory_use_rule(payload)


def apply_history_gate_to_memory(memory_obj, expose_history: bool):
    """Enable/disable whether history is exposed to the LLM prompt.

    We do NOT monkeypatch methods on pydantic models (that fails on some LangChain builds).
    Instead, we rely on GatedConversationBufferMemory.expose_history.
    """
    if hasattr(memory_obj, "expose_history"):
        try:
            memory_obj.expose_history = bool(expose_history)
        except Exception:
            pass
    return memory_obj


def _set_memory_step_kind(conversation_or_memory, step_kind: str):
    """Select the temporary memory-use rule for the next LLM call.

    The rule is injected when history is loaded, not when history is saved.
    This keeps stored memory clean while giving Step-2 and Step-3 different
    anti-repetition / anti-anchor guidance.
    """
    try:
        mem = getattr(conversation_or_memory, "memory", None) or conversation_or_memory
        if mem is not None and hasattr(mem, "memory_step_kind"):
            mem.memory_step_kind = str(step_kind or "")
    except Exception:
        pass


def _conversation_has_visible_history(conversation_or_memory) -> int:
    """Return 1 only when memory is actually enabled, exposed, and non-empty.

    This is used for CSV/source metadata. It intentionally checks the rendered
    LLM-memory object, not only the global MEMORY_MODE flag, so light memory with
    llm_history=auto is logged correctly while memory=off/hidden stays 0.
    """
    try:
        if not bool(MEMORY_ENABLED and LLM_HISTORY_ENABLED):
            return 0
    except Exception:
        return 0
    try:
        mem = getattr(conversation_or_memory, "memory", None) or conversation_or_memory
        if mem is None:
            return 0
        if hasattr(mem, "expose_history") and not bool(getattr(mem, "expose_history", True)):
            return 0
        chat_mem = getattr(mem, "chat_memory", None)
        if chat_mem is None:
            return 0
        msgs = getattr(chat_mem, "messages", None)
        return int(bool(msgs))
    except Exception:
        return 0


# ============================
# TERMINAL TRACE HELPERS
# ============================

def _msg_role(msg) -> str:
    """
    Return a LangChain message role in native-Ollama form.
    
        The native API expects roles such as system, user, and assistant. This helper hides version-specific
        message-object differences from the message collection code.
    """
    t = getattr(msg, "type", "") or msg.__class__.__name__
    tl = str(t).lower()
    if "human" in tl or "user" in tl:
        return "Human"
    if "ai" in tl or "assistant" in tl:
        return "AI"
    if "system" in tl:
        return "System"
    return str(t)

def _msg_text(msg) -> str:
    """
    Return text content from a LangChain message-like object.
    
        Different LangChain versions expose message content through slightly different attributes. This
        helper gives tracing and native message conversion one stable access point.
    """
    c = getattr(msg, "content", "")
    # Some message types store content as list of parts; stringify safely.
    if isinstance(c, list):
        try:
            return "\n".join(str(x) for x in c)
        except Exception:
            return str(c)
    return str(c)

def _get_history_messages(conversation):
    """
    Return the current conversation memory messages in a wrapper-compatible form.
    
        LangChain memory internals vary across versions. This helper isolates the access pattern so trace
        logging and native Ollama message construction can use the same history retrieval logic.
    """
    try:
        mem = getattr(conversation, "memory", None)
        if mem is None:
            return []
        chat_mem = getattr(mem, "chat_memory", None)
        if chat_mem is None:
            return []
        msgs = getattr(chat_mem, "messages", None)
        return list(msgs) if msgs else []
    except Exception:
        return []


def _history_messages_with_memory_use_rule(conversation, include_history: bool = True):
    """Return stored history plus the temporary Step-specific memory-use rule.

    Native Ollama calls build request messages manually from chat_memory, so they do
    not pass through ConversationBufferMemory.load_memory_variables(). Without this
    helper, light/full memory is shown to Qwen without the anti-copy/anti-anchor rule.
    The rule is prepended only at render time and is never saved into memory.
    """
    if not include_history:
        return []
    hist = _get_history_messages(conversation)
    if not hist:
        return []
    try:
        mem = getattr(conversation, "memory", None)
        if mem is not None and hasattr(mem, "expose_history") and not bool(getattr(mem, "expose_history", True)):
            return []
        step_kind = getattr(mem, "memory_step_kind", "") if mem is not None else ""
        rule = _memory_use_rules_for_step(step_kind)
    except Exception:
        rule = ""
    if not str(rule or "").strip():
        return hist
    try:
        if hist and str(getattr(hist[0], "content", "") or "").strip() == str(rule).strip():
            return hist
    except Exception:
        pass
    try:
        return [HumanMessage(content=str(rule).strip())] + hist
    except Exception:
        from langchain_core.messages import HumanMessage as _HM
        return [_HM(content=str(rule).strip())] + hist


def _trace_prompt(conversation, prompt_text: str, tag: str = "LLM") -> None:
    """Controlled tracing that preserves the *old* launcher colors.

    The launcher marks everything after the line "Prompt after formatting:" as a green prompt block
    until it sees a "Finished chain." marker. We exploit that intentionally.

    TRACE_MODE:
      - minimal: show only the current input as seen by the agent (no past history)
      - full:    additionally show stored chat history for debugging
      - off:     no tracing
    """
    mode = (TRACE_MODE or "off").strip().lower()
    if mode == "off":
        return

    show_history = (mode == "full")
    hist_msgs = _get_history_messages(conversation) if show_history else []

    # Start a prompt block recognizable by the launcher (green)
    print("Prompt after formatting:")

    mem_s = "enabled" if MEMORY_ENABLED else "off"
    llmh_s = "on" if LLM_HISTORY_ENABLED else "off"
    # This stays inside the prompt block, so it will also appear green.
    print(f"[trace={mode} memory={mem_s} llm_history={llmh_s}]")

    if show_history:
        print("----- HISTORY START -----")
        if hist_msgs:
            for msg in hist_msgs:
                role = _msg_role(msg)
                label = {"human": "Human", "ai": "AI", "system": "System"}.get(role, role.capitalize())
                content = _msg_text(msg).rstrip("\n")
                lines = content.splitlines() or [""]
                print(f"{label}: {lines[0]}")
                for ln in lines[1:]:
                    print(ln)
        print("----- HISTORY END -----")

        if not LLM_HISTORY_ENABLED:
            print("[NOTE] History above is printed for debugging but NOT injected into the model (Markovian).")

    # Current input as the agent sees it (Markovian if LLM_HISTORY_ENABLED is off)
    print(str(prompt_text).rstrip("\n"))


def _trace_output(output_text: str, tag: str = "LLM") -> None:
    """Print model output and close the green prompt block in the launcher."""
    mode = (TRACE_MODE or "off").strip().lower()
    if mode == "off":
        return

    out = str(output_text).rstrip("\n")
    lines = out.splitlines() or [""]

    # Keep the output inside the prompt block (green) like LangChain verbose did.
    print(f"AI: {lines[0]}")
    for ln in lines[1:]:
        print(ln)

    # Close the prompt block for the launcher classifier.
    print("> Finished chain.")



class Agent:
    """
    Simulation agent backed by two LLM chains, one for speaking and one for listening.
    
        Each Agent owns a stable persona, a mutable belief state, optional/light memory buffers, counters,
        true-open notes, and transcript logs. The simulation loop calls produce_tweet for Step-2 and
        receive_tweet for Step-3.
    """
    def __init__(
        self, agent_id, persona, model_name, temperature, max_tokens, prompt_template_root,
        top_p=None, top_k=None, repeat_penalty=None, repeat_last_n=None, llm_seed=None,
    ):
        """
        Initialize one LLM-backed agent and its runtime state.
        
            The constructor stores identity, initial/current belief, persona fields, memory buffers, logging
            containers, true-open notes, and per-step LLM settings. It also creates separate Step-2 and Step-3
            ChatOllama/ConversationChain objects so speaking and listening can use different prompts and
            decoding parameters.
        """
        # Initialize LLM agent, its identity, name, and persona by reading as step #1
        self.agent_id = agent_id
        self.prompt_template_root = prompt_template_root

        self.count_tweet_written = 0
        self.count_tweet_seen = 0

        if args.distribution == "uniform":
            df_agents = pd.read_csv(
                join(self.prompt_template_root, "list_agent_descriptions.csv")
            )
        else:
            df_agents = pd.read_csv(
                join(self.prompt_template_root, "list_agent_descriptions.csv")
            )

        self.agent_name = str(df_agents.loc[self.agent_id - 1, "agent_name"])
        self.init_belief = df_agents.loc[self.agent_id - 1, "opinion"]
        self.current_belief = self.init_belief
        row_meta = df_agents.loc[self.agent_id - 1]
        self.persona = _ensure_persona_card_contains_row_fields(persona, row_meta, self.agent_name)
        background = str(row_meta.get("background", "") or "").strip()
        self.background = background
        self.persona_public = make_public_persona(self.persona)
        self.epistemic_profile = _csv_clean_value(row_meta.get("epistemic_profile", ""))
        self.institutional_trust = _csv_clean_value(row_meta.get("institutional_trust", ""))
        self.uncertainty_tolerance = _csv_clean_value(row_meta.get("uncertainty_tolerance", ""))
        self.evidence_style = _csv_clean_value(row_meta.get("evidence_style", ""))
        self.official_narrative_suspicion = _csv_clean_value(row_meta.get("official_narrative_suspicion", ""))
        self.openness_to_update = _csv_clean_value(row_meta.get("openness_to_update", ""))
        self.age_group = _csv_clean_value(row_meta.get("age_group", ""))
        self.flavor_gender = _csv_clean_value(row_meta.get("flavor_gender", ""))
        self.flavor_ethnicity = _csv_clean_value(row_meta.get("flavor_ethnicity", ""))
        self.education_level = _csv_clean_value(row_meta.get("education_level", ""))
        self.training_style = _csv_clean_value(row_meta.get("training_style", ""))
        self.domain_familiarity = _csv_clean_value(row_meta.get("domain_familiarity", ""))
        self.topic_interest = _csv_clean_value(row_meta.get("topic_interest", ""))
        self.prior_exposure = _csv_clean_value(row_meta.get("prior_exposure", ""))
        self.lifestyle_notes = _csv_clean_value(row_meta.get("lifestyle_notes", ""))
        self.tone_hint = _csv_clean_value(row_meta.get("tone_hint", ""))

        # Legacy aliases remain populated for old CSVs, while newer UI columns
        # fall back into them where appropriate.
        self.age = _csv_clean_value(row_meta.get("age", "")) or self.age_group
        self.gender = _csv_clean_value(row_meta.get("gender", "")) or self.flavor_gender
        self.ethnicity = _csv_clean_value(row_meta.get("ethnicity", "")) or self.flavor_ethnicity
        self.education = _csv_clean_value(row_meta.get("education", "")) or self.education_level
        self.occupation = _csv_clean_value(row_meta.get("occupation", ""))
        self.political_leaning = _csv_clean_value(row_meta.get("political_leaning", ""))
        self.early_life = _csv_clean_value(row_meta.get("early_life", ""))
        self.value_orientation = _csv_clean_value(row_meta.get("value_orientation", ""))
        self.agency_vs_fatalism = _csv_clean_value(row_meta.get("agency_vs_fatalism", ""))
        self.conflict_style = _csv_clean_value(row_meta.get("conflict_style", ""))
        self.persona_policy = _compile_persona_validation_policy(
            epistemic_profile=self.epistemic_profile,
            institutional_trust=self.institutional_trust,
            uncertainty_tolerance=self.uncertainty_tolerance,
            evidence_style=self.evidence_style,
            official_narrative_suspicion=self.official_narrative_suspicion,
            openness_to_update=self.openness_to_update,
            value_orientation=self.value_orientation,
            agency_vs_fatalism=self.agency_vs_fatalism,
            conflict_style=self.conflict_style,
        )
        if PERSONA_MODE == "none":
            # Persona-off control: strip every persona attribute to isolate how
            # much of the collective dynamics is driven by agent heterogeneity.
            # Only the name survives; the initial belief still comes from the
            # opinion distribution. Every downstream consumer (prompt card,
            # homophily scoring, validation policy) then sees an identical,
            # empty persona.
            _name_only = "AGENT_PERSONA_CARD\nName: " + str(self.agent_name)
            self.persona = _name_only
            self.persona_public = _name_only
            self.background = ""
            for _attr in (
                "epistemic_profile", "institutional_trust", "uncertainty_tolerance",
                "evidence_style", "official_narrative_suspicion", "openness_to_update",
                "age_group", "flavor_gender", "flavor_ethnicity", "education_level",
                "training_style", "domain_familiarity", "topic_interest",
                "prior_exposure", "lifestyle_notes", "tone_hint", "age", "gender",
                "ethnicity", "education", "occupation", "political_leaning",
                "early_life", "value_orientation", "agency_vs_fatalism", "conflict_style",
            ):
                setattr(self, _attr, "")
            self.persona_policy = _compile_persona_validation_policy()
        self.count_tweet_written, self.count_tweet_seen = 0, 0
        self.previous_interaction_type = "none"
        self.same_side_pos_edge_hits = 0
        self.same_side_neg_edge_hits = 0
        self.last_seen_tweet = ""
        self.last_written_tweet = ""
        # Durable speaker-side tweet handoff cache.  Step-3 must never depend only
        # on the immediate sanitized return string, because some Step-2 parser/
        # sanitizer near-misses can otherwise erase a valid raw TWEET line before
        # the listener receives it.
        self.last_public_tweet = ""
        self.last_public_tweet_raw_output = ""
        self.last_public_tweet_final_output = ""
        self.last_public_tweet_step = ""
        self.memory_events = []
        self.memory_event_counter = 0

        print(f"Initializing agent_id={self.agent_id}, agent_name={self.agent_name}")
        print("Persona passed to Agent:")
        print(self.persona[:200], "...\n")

        # Keep persona conditioning in the system prompt + step prompts only.
        # Qwen is more likely than Llama to over-literalize duplicated persona blocks,
        # so we avoid injecting the full persona as an extra Human message.

        # OLLAMA LLM
        # We allow separate decoding settings for Step-2 (tweet) and Step-3 (belief update).
        # Both LLM chains share the SAME LangChain memory object, so history stays consistent.
        def _pick(step_val, global_val):
            """
            Choose a per-step LLM setting override, falling back to the global setting.
            
                Step-2 and Step-3 can use different decoding parameters. This helper keeps that override logic
                compact and consistently typed inside Agent initialization.
            """
            return global_val if step_val is None else step_val

        # Model overrides (optional)
        step2_model = args.model_name_step2 or model_name
        step3_model = args.model_name_step3 or model_name

        # Temperature overrides (optional)
        step2_temperature = _pick(getattr(args, "temperature_step2", None), temperature)
        step3_temperature = _pick(getattr(args, "temperature_step3", None), temperature)
        if _is_qwen_family_model(step2_model) and _is_open_world_mode(getattr(args, "world", "closed")) and getattr(args, "temperature_step2", None) is None:
            # In open worlds we want slightly more natural variety, but not the stronger
            # overshoot that starts making +1 tweets sound like +2. Use a fixed mild bump.
            try:
                step2_temperature = 0.74
            except Exception:
                step2_temperature = 0.74

        # Length overrides (optional)
        step2_max_tokens = _pick(getattr(args, "max_tokens_step2", None), max_tokens)
        step3_max_tokens = _pick(getattr(args, "max_tokens_step3", None), max_tokens)

        # Sampling overrides (optional)
        step2_top_p = _pick(getattr(args, "top_p_step2", None), top_p)
        step3_top_p = _pick(getattr(args, "top_p_step3", None), top_p)

        step2_top_k = _pick(getattr(args, "top_k_step2", None), top_k)
        step3_top_k = _pick(getattr(args, "top_k_step3", None), top_k)

        step2_repeat_penalty = _pick(getattr(args, "repeat_penalty_step2", None), repeat_penalty)
        step3_repeat_penalty = _pick(getattr(args, "repeat_penalty_step3", None), repeat_penalty)

        step2_repeat_last_n = _pick(getattr(args, "repeat_last_n_step2", None), repeat_last_n)
        step3_repeat_last_n = _pick(getattr(args, "repeat_last_n_step3", None), repeat_last_n)

        # Safer Step-3 defaults for format-sensitive judgment prompts when no explicit override was provided.
        if _is_qwen_family_model(step3_model):
            if getattr(args, "temperature_step3", None) is None:
                step3_temperature = 0.25
            if getattr(args, "top_p_step3", None) is None:
                step3_top_p = 0.90
            if getattr(args, "repeat_penalty_step3", None) is None:
                step3_repeat_penalty = 1.10
            if getattr(args, "repeat_last_n_step3", None) is None:
                step3_repeat_last_n = 128

        # Presence/Frequency penalties (optional; Ollama option)
        base_frequency_penalty = getattr(args, "frequency_penalty", 0.0)
        base_presence_penalty = getattr(args, "presence_penalty", 0.0)

        step2_frequency_penalty = _pick(getattr(args, "frequency_penalty_step2", None), base_frequency_penalty)
        step3_frequency_penalty = _pick(getattr(args, "frequency_penalty_step3", None), base_frequency_penalty)

        step2_presence_penalty = _pick(getattr(args, "presence_penalty_step2", None), base_presence_penalty)
        step3_presence_penalty = _pick(getattr(args, "presence_penalty_step3", None), base_presence_penalty)

        if _is_qwen_family_model(step2_model) and _normalize_world_mode(getattr(args, "world", "closed")) in {"closed_strict", "closed_strict_rag"}:
            if getattr(args, "temperature_step2", None) is None:
                step2_temperature = 0.50
            if getattr(args, "top_p_step2", None) is None:
                step2_top_p = 0.80
            if getattr(args, "repeat_penalty_step2", None) is None:
                step2_repeat_penalty = 1.05
            if getattr(args, "repeat_last_n_step2", None) is None:
                step2_repeat_last_n = 64
            if getattr(args, "presence_penalty_step2", None) is None:
                step2_presence_penalty = 0.0
            if getattr(args, "frequency_penalty_step2", None) is None:
                step2_frequency_penalty = 0.0

        # min_p (optional; backend-dependent)
        base_min_p = getattr(args, "min_p", None)
        step2_min_p = _pick(getattr(args, "min_p_step2", None), base_min_p)
        step3_min_p = _pick(getattr(args, "min_p_step3", None), base_min_p)

        # Seeds (optional per-step override)
        step2_seed = _pick(getattr(args, "llm_seed_step2", None), llm_seed)
        step3_seed = _pick(getattr(args, "llm_seed_step3", None), llm_seed)

        think_mode = str(getattr(args, "think_mode", "off") or "off").strip().lower()
        step2_think = None
        step3_think = None
        if think_mode == "on":
            step2_think = True
            step3_think = True
        elif think_mode == "off":
            step2_think = False
            step3_think = False
        elif think_mode == "step3_only":
            step2_think = False
            step3_think = True
        elif think_mode == "step2_only":
            step2_think = True
            step3_think = False

        # A model KNOWN to lack thinking never runs hidden reasoning, regardless of
        # think_mode. Unknown models are left as-is (respect think_mode).
        if _model_has_thinking(step2_model) is False:
            step2_think = False
        if _model_has_thinking(step3_model) is False:
            step3_think = False

        # True-open Qwen runs become much less stable when hidden thinking is enabled:
        # long internal traces crowd out the required two-line protocol output.
        if _normalize_world_mode(getattr(args, "world", "closed")) == "true_open":
            if _is_qwen_family_model(step2_model):
                step2_think = False
            if _is_qwen_family_model(step3_model):
                step3_think = False

        # Qwen-family models often need a larger generation budget because they may emit
        # internal planning before the final protocol answer. Apply world-specific floors.
        if _is_qwen_family_model(step2_model):
            try:
                s2_floor = _qwen_world_num_predict_floor("step2", getattr(args, "world", "closed"), step2_think)
                step2_max_tokens = max(int(step2_max_tokens), int(s2_floor))
            except Exception:
                step2_max_tokens = 896
        if _is_qwen_family_model(step3_model):
            try:
                s3_floor = _qwen_world_num_predict_floor("step3", getattr(args, "world", "closed"), step3_think)
                step3_max_tokens = max(int(step3_max_tokens), int(s3_floor))
            except Exception:
                step3_max_tokens = 1024

        llm_opts_step2 = {
            "top_p": step2_top_p,
            "top_k": step2_top_k,
            "repeat_penalty": step2_repeat_penalty,
            "repeat_last_n": step2_repeat_last_n,
            "frequency_penalty": step2_frequency_penalty,
            "presence_penalty": step2_presence_penalty,
            "min_p": step2_min_p,
            "num_predict": step2_max_tokens,
            "max_tokens": step2_max_tokens,
            "seed": step2_seed,
            "think": step2_think,
        }
        llm_opts_step3 = {
            "top_p": step3_top_p,
            "top_k": step3_top_k,
            "repeat_penalty": step3_repeat_penalty,
            "repeat_last_n": step3_repeat_last_n,
            "frequency_penalty": step3_frequency_penalty,
            "presence_penalty": step3_presence_penalty,
            "min_p": step3_min_p,
            "num_predict": step3_max_tokens,
            "max_tokens": step3_max_tokens,
            "seed": step3_seed,
            "think": step3_think,
        }

        self.step2_model = step2_model
        self.step3_model = step3_model
        self.step2_think = step2_think
        self.step3_think = step3_think
        self.last_step2_raw_response_attempt_1 = ''
        self.last_step2_raw_response_attempt_2 = ''
        self.last_step2_warning_tags = []
        self.last_step2_quality = 'ok'
        self.last_step2_fallback_used = 0
        self.last_step2_final_output = ''
        self.last_step2_valid_output = 1
        self.last_step2_pipeline_artifact = 0
        # Same-step Step-2 -> Step-3 handoff diagnostics. These are not LLM
        # decisions; they let us distinguish parser loss from true empty output.
        self.last_step2_handoff_tweet = ''
        self.last_step2_handoff_source = ''
        self.last_step2_handoff_diag = {}
        self.last_step2_parsed_tweet_len = 0
        self.last_step3_valid_interaction = 1
        self.last_step3_pipeline_artifact = 0
        self.last_step3_real_evaluation = 1
        self.last_step3_quality = 'ok'

        llm_step2 = build_chat_ollama(model=step2_model, temperature=step2_temperature, **llm_opts_step2)
        llm_step3 = build_chat_ollama(model=step3_model, temperature=step3_temperature, **llm_opts_step3)

        # Create placeholders for back-and-forth history with the LLM
        history_message_holder = MessagesPlaceholder(variable_name="history")
        question_placeholder = HumanMessagePromptTemplate.from_template("{input}")

        with open(
            join(self.prompt_template_root, "step1_persona.md"), "r", encoding="utf-8"
        ) as f:
            sys_prompt = f.read()

        sys_prompt = safe_format(sys_prompt.split("\n---------------------------\n")[0], 
            AGENT_PERSONA=self.persona_public, AGENT_NAME=self.agent_name,
            AGENT_BACKGROUND=self.background,
            WORLD=WORLD,
            WORLD_RULES=WORLD_RULES,
            FACT_PACK=FACT_PACK_TEXT,
            STEP2_FACT_PACK_RULES=STEP2_FACT_PACK_RULES,
            RAG_CONTEXT="",
            STEP2_RAG_RULES="",
            STEP3_RAG_RULES="",
            STEP2_MISMATCHED_SHOWN_MATERIAL_RULES="",
            THEORY_STATEMENT=THEORY_STATEMENT,
            BIAS=BIAS_TEXT,
        )
        systems_prompt = SystemMessagePromptTemplate.from_template(sys_prompt)

        # Initialize the LLM agent with the language chain and its memory
        chat_prompt = ChatPromptTemplate.from_messages(
            [systems_prompt, history_message_holder, question_placeholder]
        )

        base_memory = GatedConversationBufferMemory(
            return_messages=True,
            ai_prefix=self.agent_name,
            human_prefix="Game Master",
            expose_history=LLM_HISTORY_ENABLED,
        )

        memory = apply_history_gate_to_memory(base_memory, expose_history=LLM_HISTORY_ENABLED)

        agent_conversation_step2 = ConversationChain(
            llm=llm_step2,
            memory=memory,
            prompt=chat_prompt,
            verbose=False,
        )
        agent_conversation_step3 = ConversationChain(
            llm=llm_step3,
            memory=memory,
            prompt=chat_prompt,
            verbose=False,
        )

        self.memory_step2 = agent_conversation_step2
        self.memory_step3 = agent_conversation_step3

        # Backward-compat: default chain points to Step-3.
        self.memory = agent_conversation_step3

        # Separate per-agent transcript buffer for network_log_conversation.
        # This is debug/output only and does NOT affect what the LLM sees.
        self.network_log_entries = []
        self.true_open_notes = []
        self.last_step2_tool_meta = {}
        self.last_step3_tool_meta = {}

    def append_network_log(self, text):
        """
        Append one human-readable entry to the agent transcript log.
        
            These transcript logs are for qualitative inspection and do not affect model conditioning unless
            memory/history is explicitly enabled through the run configuration.
        """
        try:
            s = str(text or '').rstrip()
        except Exception:
            s = ''
        if s:
            self.network_log_entries.append(s)
    def receive_tweet(self, tweet, previous_interaction_type, tweet_written_count, add_to_memory, allowed_ratings=None, allowed_set_str=None, speaker_belief=None):
        """
        Run Step-3 for this listener: read a tweet and update the agent's belief.
        
            The method builds the listener prompt, injects persona/world/RAG/fact-pack context, applies the
            precomputed allowed-rating set, calls the Step-3 LLM pipeline, updates current_belief from the
            finalized output, and logs the interaction transcript.
        """
        assert previous_interaction_type in ["write", "read", "none"]
        _set_current_persona_validation_policy(getattr(self, "persona_policy", {}))
        _set_current_validation_context("", "", "")

        # Leak-free allowed set: passed in from the simulation loop (may include the no-away constraint).
        # We expose ONLY the set of allowed ratings to the model (no speaker rating).
        if allowed_set_str is None:
            allowed_set_str = _format_allowed_ratings(int(self.current_belief), allowed_ratings)

        tweet_input_norm = re.sub(r"\s+", " ", str(tweet or "")).strip()
        self.last_seen_tweet = tweet_input_norm

        if (not tweet_input_norm) or (not _tweet_text_has_usable_step3_content(tweet_input_norm)):
            _metric_inc('step3_invalid_input_from_missing_tweet')
            _metric_inc('step3_invalid_interaction_events_total')
            _metric_inc('pipeline_input_failure::invalid_input_tweet_skip_step3')
            try:
                _append_native_event_row(
                    step_kind='step3',
                    agent_name=self.agent_name,
                    label='pre_step3_input_guard',
                    attempt='0',
                    event_type='pipeline_input_failure',
                    text=str(tweet or ''),
                    note='invalid_input_tweet_skip_step3: no usable tweet reached Step-3 after handoff recovery',
                )
            except Exception:
                pass
            fallback = f"FINAL_RATING: {int(self.current_belief)}\nEXPLANATION: No usable tweet content was available to evaluate, so I keep my rating unchanged."
            _dump_retry_debug_event(
                'step3',
                'warning',
                agent_name=self.agent_name,
                attempt=0,
                reasons='invalid_input_tweet_skip_step3',
                prompt_text='',
                raw_text=str(tweet or ''),
                sanitized_text='',
                final_text=fallback,
            )
            _dump_retry_debug_event(
                'step3',
                'fallback',
                agent_name=self.agent_name,
                attempt=0,
                reasons='invalid_input_tweet_skip_step3',
                prompt_text='',
                raw_text=str(tweet or ''),
                sanitized_text='',
                final_text=fallback,
            )
            self.last_step3_raw_response = ''
            self.last_step3_raw_response_attempt_1 = ''
            self.last_step3_raw_response_attempt_2 = ''
            self.last_step3_repair_tags = ['invalid_input_tweet_skip_step3']
            self.last_step3_tool_meta = _mark_prompt_source_meta({"tool_sequence": [], "observable_primary_source": "tweet", "observable_secondary_source": "", "used_web": 0, "used_notes": 0, "used_rag": 0, "used_fact_pack": int(bool(str(FACT_PACK_TEXT or '').strip())), "used_memory_history": 0, "used_tweet_only": 1, "web_query": "", "web_results_count": 0}, agent=self, conversation=self.memory_step3, step_kind="step3", rag_context_text="")
            self.last_step3_valid_interaction = 0
            self.last_step3_pipeline_artifact = 1
            self.last_step3_real_evaluation = 0
            self.last_step3_quality = 'invalid_interaction'
            return fallback

        if previous_interaction_type == "write":
            with open(
                join(
                    self.prompt_template_root, "step3_receive_tweet_prev_tweet.md"
                ),
                "r",encoding="utf-8"
            ) as f:
                prompt_instructions = f.read()

            prompt_template = _ensure_step3_open_world_placeholder(
                prompt_instructions.split("\n---------------------------\n")[0]
            )
            prompt_template = _ensure_step3_fact_pack_rules_placeholder(prompt_template)
            prompt_template = _ensure_step3_rag_rules_placeholder(prompt_template)
            prompt_template = _ensure_rag_context_placeholder(prompt_template)
            rag_context_text = _rag_context_value(tweet_text=tweet, step_kind="step3", agent_name=self.agent_name, prompt_variant=previous_interaction_type, prompt_text=prompt_template)
            true_open_context_text, true_open_meta = _build_true_open_context(self, tweet, previous_interaction_type=previous_interaction_type)
            combined_context_text = _combine_shown_material_blocks(rag_context_text, true_open_context_text)
            true_open_meta = _mark_prompt_source_meta(true_open_meta, agent=self, conversation=self.memory_step3, step_kind="step3", rag_context_text=rag_context_text)
            self.last_step3_tool_meta = dict(true_open_meta)
            prompt = safe_format(prompt_template, 
                AGENT_NAME=self.agent_name,
                AGENT_PERSONA=self.persona_public,
                CURRENT_BELIEF=int(self.current_belief),
                PREVIOUS_RATING=int(self.current_belief),
                TWEET_WRITTEN_COUNT=tweet_written_count,
                SUPERSCRIPT=get_superscript(tweet_written_count),
                TWEET=tweet,
                ALLOWED_SET=allowed_set_str,
                ALLOWED_RATING=allowed_set_str,
                WORLD=WORLD,
                WORLD_RULES=WORLD_RULES,
                STEP3_OPEN_WORLD_RULES=STEP3_OPEN_WORLD_RULES,
                FACT_PACK=FACT_PACK_TEXT,
                STEP3_FACT_PACK_RULES=STEP3_FACT_PACK_RULES,
                RAG_CONTEXT=combined_context_text,
                STEP3_RAG_RULES=STEP3_RAG_RULES,
                THEORY_STATEMENT=THEORY_STATEMENT,
                BIAS=BIAS_TEXT,
            )
            prompt = _prepend_model_prompt_control(prompt, self.step3_model, self.step3_think)
            response, raw_step3_attempt_1, raw_step3_attempt_2, repair_tags = get_step3_llm_response(self.memory_step3, prompt, pre_belief=int(self.current_belief), add_to_memory=add_to_memory, agent_name=self.agent_name, allowed_ratings=allowed_ratings, speaker_belief=speaker_belief)
            _update_true_open_notes(self, getattr(self, "last_step3_tool_meta", {}))

        elif previous_interaction_type == "read":
            with open(
                join(
                    self.prompt_template_root, "step3_receive_tweet_prev_read.md"
                ),
                "r",encoding="utf-8"
            ) as f:
                prompt_instructions = f.read()

            prompt_template = _ensure_step3_open_world_placeholder(
                prompt_instructions.split("\n---------------------------\n")[0]
            )
            prompt_template = _ensure_step3_fact_pack_rules_placeholder(prompt_template)
            prompt_template = _ensure_step3_rag_rules_placeholder(prompt_template)
            prompt_template = _ensure_rag_context_placeholder(prompt_template)
            rag_context_text = _rag_context_value(tweet_text=tweet, step_kind="step3", agent_name=self.agent_name, prompt_variant=previous_interaction_type, prompt_text=prompt_template)
            true_open_context_text, true_open_meta = _build_true_open_context(self, tweet, previous_interaction_type=previous_interaction_type)
            combined_context_text = _combine_shown_material_blocks(rag_context_text, true_open_context_text)
            true_open_meta = _mark_prompt_source_meta(true_open_meta, agent=self, conversation=self.memory_step3, step_kind="step3", rag_context_text=rag_context_text)
            self.last_step3_tool_meta = dict(true_open_meta)
            prompt = safe_format(prompt_template, 
                AGENT_NAME=self.agent_name,
                AGENT_PERSONA=self.persona_public,
                CURRENT_BELIEF=int(self.current_belief),
                PREVIOUS_RATING=int(self.current_belief),
                TWEET=tweet,
                ALLOWED_SET=allowed_set_str,
                ALLOWED_RATING=allowed_set_str,
                WORLD=WORLD,
                WORLD_RULES=WORLD_RULES,
                STEP3_OPEN_WORLD_RULES=STEP3_OPEN_WORLD_RULES,
                FACT_PACK=FACT_PACK_TEXT,
                STEP3_FACT_PACK_RULES=STEP3_FACT_PACK_RULES,
                RAG_CONTEXT=combined_context_text,
                STEP3_RAG_RULES=STEP3_RAG_RULES,
                THEORY_STATEMENT=THEORY_STATEMENT,
                BIAS=BIAS_TEXT,
            )
            prompt = _prepend_model_prompt_control(prompt, self.step3_model, self.step3_think)
            response, raw_step3_attempt_1, raw_step3_attempt_2, repair_tags = get_step3_llm_response(self.memory_step3, prompt, pre_belief=int(self.current_belief), add_to_memory=add_to_memory, agent_name=self.agent_name, allowed_ratings=allowed_ratings, speaker_belief=speaker_belief)
            _update_true_open_notes(self, getattr(self, "last_step3_tool_meta", {}))

        elif previous_interaction_type == "none":
            with open(
                join(
                    self.prompt_template_root, "step3_receive_tweet_prev_none.md"
                ),
                "r",encoding="utf-8"
            ) as f:
                prompt_instructions = f.read()

            prompt_template = _ensure_step3_open_world_placeholder(
                prompt_instructions.split("\n---------------------------\n")[0]
            )
            prompt_template = _ensure_step3_fact_pack_rules_placeholder(prompt_template)
            prompt_template = _ensure_step3_rag_rules_placeholder(prompt_template)
            prompt_template = _ensure_rag_context_placeholder(prompt_template)
            rag_context_text = _rag_context_value(tweet_text=tweet, step_kind="step3", agent_name=self.agent_name, prompt_variant=previous_interaction_type, prompt_text=prompt_template)
            true_open_context_text, true_open_meta = _build_true_open_context(self, tweet, previous_interaction_type=previous_interaction_type)
            combined_context_text = _combine_shown_material_blocks(rag_context_text, true_open_context_text)
            true_open_meta = _mark_prompt_source_meta(true_open_meta, agent=self, conversation=self.memory_step3, step_kind="step3", rag_context_text=rag_context_text)
            self.last_step3_tool_meta = dict(true_open_meta)
            prompt = safe_format(prompt_template, 
                AGENT_NAME=self.agent_name,
                AGENT_PERSONA=self.persona_public,
                CURRENT_BELIEF=int(self.current_belief),
                PREVIOUS_RATING=int(self.current_belief),
                TWEET=tweet,
                ALLOWED_SET=allowed_set_str,
                ALLOWED_RATING=allowed_set_str,
                WORLD=WORLD,
                WORLD_RULES=WORLD_RULES,
                STEP3_OPEN_WORLD_RULES=STEP3_OPEN_WORLD_RULES,
                FACT_PACK=FACT_PACK_TEXT,
                STEP3_FACT_PACK_RULES=STEP3_FACT_PACK_RULES,
                RAG_CONTEXT=combined_context_text,
                STEP3_RAG_RULES=STEP3_RAG_RULES,
                THEORY_STATEMENT=THEORY_STATEMENT,
                BIAS=BIAS_TEXT,
            )
            prompt = _prepend_model_prompt_control(prompt, self.step3_model, self.step3_think)
            response, raw_step3_attempt_1, raw_step3_attempt_2, repair_tags = get_step3_llm_response(self.memory_step3, prompt, pre_belief=int(self.current_belief), add_to_memory=add_to_memory, agent_name=self.agent_name, allowed_ratings=allowed_ratings, speaker_belief=speaker_belief)
            _update_true_open_notes(self, getattr(self, "last_step3_tool_meta", {}))

        # Expose raw/final Step-3 audit data to the main loop without changing the public log format.
        self.last_step3_raw_response = raw_step3_attempt_1
        self.last_step3_raw_response_attempt_1 = raw_step3_attempt_1
        self.last_step3_raw_response_attempt_2 = raw_step3_attempt_2
        self.last_step3_repair_tags = list(repair_tags or [])
        _artifact_tags = {'invalid_input_tweet_skip_step3', 'step3_empty_retry_failed', 'main_missing_final_rating_fallback', 'native_rescue_failed'}
        _has_artifact = any(str(tag or '').split(':',1)[0] in _artifact_tags for tag in (repair_tags or []))
        self.last_step3_valid_interaction = int(not _has_artifact)
        self.last_step3_pipeline_artifact = int(_has_artifact)
        self.last_step3_real_evaluation = int(not _has_artifact)
        self.last_step3_quality = 'pipeline_artifact' if _has_artifact else 'ok'
        if _has_artifact:
            _metric_inc('step3_pipeline_artifact_events_total')
        else:
            _metric_inc('step3_real_evaluation_events_total')

        if not add_to_memory:
            # REMOVE last (user, ai) pair so Step-3 does not accumulate in memory.
            # (add_to_memory=False keeps this step stateless by design.)
            try:
                msgs = self.memory_step3.memory.chat_memory.messages
            except Exception:
                try:
                    msgs = self.memory_step3.chat_memory.messages
                except Exception:
                    msgs = None
            if msgs is not None:
                if len(msgs) >= 2:
                    msgs.pop()
                    msgs.pop()
                else:
                    msgs.clear()

        return response

    def produce_tweet(self, previous_interaction_type, tweet_written_count, add_to_memory):
        """
        Run Step-2 for this speaker: generate a public tweet from the current belief.
        
            The method builds the speaker prompt, injects persona/world/RAG/fact-pack context, calls the
            Step-2 LLM pipeline, sanitizes the output, and caches the public tweet so the listener handoff can
            recover from minor formatting artifacts.
        """
        assert previous_interaction_type in ["write", "read", "none"]
        _set_current_persona_validation_policy(getattr(self, "persona_policy", {}))
        _set_current_validation_context("", "", "")

        if previous_interaction_type == "write":
            filename = "step2_produce_tweet_prev_tweet.md"
        elif previous_interaction_type == "read":
            filename = "step2_produce_tweet_prev_read.md"
        else:
            filename = "step2_produce_tweet_prev_none.md"

        with open(join(self.prompt_template_root, filename), "r", encoding="utf-8") as f:
            prompt_instructions = f.read()

        # ADR-006 Component 2: always route the template through the silence
        # renderer. With ALLOW_SILENCE=False (default) the marker + surrounding
        # blank lines are stripped so the text is byte-identical with the
        # pre-Task-2.1 template. With True the mechanism-agnostic silence
        # option block is substituted in.
        prompt_instructions = render_step2_template(prompt_instructions, allow_silence=ALLOW_SILENCE)

        prompt_template = prompt_instructions.split("\n---------------------------\n")[0]
        prompt_template = _ensure_step2_fact_pack_rules_placeholder(prompt_template)
        prompt_template = _ensure_step2_rag_rules_placeholder(prompt_template)
        prompt_template = _ensure_step2_mismatched_rules_placeholder(prompt_template)
        prompt_template = _ensure_rag_context_placeholder(prompt_template)

        step2_rag_tweet = ""
        if previous_interaction_type == "read":
            step2_rag_tweet = str(getattr(self, "last_seen_tweet", "") or "")
        elif previous_interaction_type == "write":
            step2_rag_tweet = str(getattr(self, "last_written_tweet", "") or "")

        rag_context_text = _rag_context_value(
            tweet_text=step2_rag_tweet,
            step_kind="step2",
            agent_name=self.agent_name,
            prompt_variant=previous_interaction_type,
            prompt_text=prompt_template,
        )
        step2_true_open_seed_text = _build_true_open_step2_seed_text(self, previous_interaction_type=previous_interaction_type)  # claim + current belief only
        true_open_context_text, true_open_meta = _build_true_open_context(
            self,
            step2_true_open_seed_text,
            previous_interaction_type=previous_interaction_type,
            step_kind="step2",
            current_belief=int(self.current_belief),
        )
        combined_context_text = _combine_shown_material_blocks(rag_context_text, true_open_context_text)
        true_open_meta = _mark_prompt_source_meta(true_open_meta, agent=self, conversation=self.memory_step2, step_kind="step2", rag_context_text=rag_context_text)
        self.last_step2_tool_meta = dict(true_open_meta or {})

        fmt_kwargs = dict(
            AGENT_NAME=self.agent_name,
            AGENT_PERSONA=self.persona_public,
            CURRENT_BELIEF=int(self.current_belief),
            TWEET_WRITTEN_COUNT=tweet_written_count,
            SUPERSCRIPT=get_superscript(tweet_written_count),
            WORLD=WORLD,
            WORLD_RULES=WORLD_RULES,
            FACT_PACK=FACT_PACK_TEXT,
            STEP2_FACT_PACK_RULES=STEP2_FACT_PACK_RULES,
            RAG_CONTEXT=combined_context_text,
            STEP2_RAG_RULES=STEP2_RAG_RULES,
            STEP2_MISMATCHED_SHOWN_MATERIAL_RULES=_step2_mismatched_shown_material_rules(
                int(self.current_belief),
                getattr(args, "world", "closed"),
                getattr(args, "fact_pack_mode", "off"),
                getattr(args, "rag_content_mode", "full"),
            ),
            THEORY_STATEMENT=THEORY_STATEMENT,
            BIAS=BIAS_TEXT,
            # Back-compat if a prompt file still contains it.
            LAST_R_TAG="NONE",
        )
        prompt = safe_format(prompt_template, **fmt_kwargs)
        if "{STEP2_FACT_PACK_RULES}" in prompt:
            prompt = prompt.replace("{STEP2_FACT_PACK_RULES}", STEP2_FACT_PACK_RULES or "")
        if _is_open_world_mode(WORLD):
            prompt = prompt.rstrip() + "\n\n" + STEP2_OPEN_WORLD_STYLE_RULES_GENERIC.strip()
            current_belief_i = int(self.current_belief)
            if current_belief_i == 0:
                prompt = prompt.rstrip() + "\n\n" + STEP2_OPEN_WORLD_ZERO_STYLE_RULES_GENERIC.strip()
            elif abs(current_belief_i) == 1:
                prompt = prompt.rstrip() + "\n\n" + STEP2_OPEN_WORLD_MILD_STYLE_RULES_GENERIC.strip()
            elif abs(current_belief_i) == 2:
                prompt = prompt.rstrip() + "\n\n" + STEP2_OPEN_WORLD_STRONG_STYLE_RULES_GENERIC.strip()

        # If a legacy template omitted CURRENT_BELIEF, prefix it (do not add extra rules).
        if "CURRENT_BELIEF" not in prompt_template:
            prompt = f"CURRENT_BELIEF: {int(self.current_belief)}\n\n" + prompt

        prompt = _restore_step2_prompt_block_boundaries(_normalize_step2_request_header_text(prompt))
        prompt = _fix_quoted_protocol_label_breaks(prompt)
        prompt = _prepend_model_prompt_control(prompt, self.step2_model, self.step2_think)

        tweet = get_step2_llm_response(
            self.memory_step2,
            prompt,
            expected_current_belief=int(self.current_belief),
            add_to_memory=add_to_memory,
            agent_name=self.agent_name,
        )

        step2_info = getattr(self.memory_step2, "_last_step2_call_info", {}) or {}
        self.last_step2_raw_response_attempt_1 = str(step2_info.get("raw_attempt_1", "") or "")
        self.last_step2_raw_response_attempt_2 = str(step2_info.get("raw_attempt_2", "") or "")
        self.last_step2_warning_tags = list(step2_info.get("warning_tags", []) or [])
        self.last_step2_quality = str(step2_info.get("quality", "ok") or "ok")
        self.last_step2_fallback_used = int(bool(step2_info.get("fallback_used", False)))
        self.last_step2_final_output = str(step2_info.get("final_output", tweet) or tweet)
        # Use the same-step Step-2 artifacts immediately. Do not wait for the
        # later speaker/listener handoff path to rediscover the tweet from a cache.
        # This is the authoritative recovery point: raw native output, final output,
        # and sanitized output are all still local to this Step-2 call.
        try:
            recovered_public_tweet, recovered_source, recovered_diag = _recover_same_step_step2_tweet_for_handoff(
                speaker_agent=self,
                step2_return=tweet,
                sanitized_text=sanitize_tweet_for_listener(self.last_step2_final_output),
            )
        except Exception:
            recovered_public_tweet, recovered_source, recovered_diag = '', '', {}
        self.last_step2_handoff_tweet = str(recovered_public_tweet or '')
        self.last_step2_handoff_source = str(recovered_source or '')
        self.last_step2_handoff_diag = dict(recovered_diag or {})
        try:
            self.last_step2_parsed_tweet_len = int(len(str(recovered_public_tweet or '')))
        except Exception:
            self.last_step2_parsed_tweet_len = 0
        if recovered_public_tweet:
            _remember_agent_public_tweet(
                self,
                recovered_public_tweet,
                final_output=self.last_step2_final_output,
                raw_output=(self.last_step2_raw_response_attempt_2 or self.last_step2_raw_response_attempt_1 or self.last_step2_final_output or tweet),
                source=f'produce_tweet_same_step::{recovered_source or "unknown"}',
            )
        self.last_step2_valid_output = int(bool(recovered_public_tweet))
        self.last_step2_pipeline_artifact = int((not self.last_step2_valid_output) or self.last_step2_quality in {'fallback_hard', 'fallback_soft', 'invalid'})
        if self.last_step2_pipeline_artifact:
            _metric_inc('step2_pipeline_artifact_events_total')
        else:
            _metric_inc('step2_real_output_events_total')

        if not add_to_memory:
            # REMOVE last (user, ai) pair so Step-2 does not accumulate in memory.
            # (add_to_memory=False keeps this step stateless by design.)
            try:
                msgs = self.memory_step2.memory.chat_memory.messages
            except Exception:
                try:
                    msgs = self.memory_step2.chat_memory.messages
                except Exception:
                    msgs = None
            if msgs is not None:
                if len(msgs) >= 2:
                    msgs.pop()
                    msgs.pop()
                else:
                    msgs.clear()


        return tweet

    def add_to_memory(
        self,
        previos_interaction_type,
        current_interaction_type,
        tweet_written_count,
        tweet_written=None,
        tweet_seen=None,
        response=None,
    ):
        """
        Store a compact interaction note in this agent's conversation memory.
        
            Memory storage and memory visibility are separate controls. This method may record history for
            future memory-enabled runs or logs even when current Markovian runs hide that history from the LLM.
        """
        if current_interaction_type == "write":
            assert tweet_written is not None
        elif current_interaction_type == "read":
            assert tweet_seen is not None and response is not None
        else:
            raise ValueError(
                f"current_interaction_type must be either 'write' or 'read'. Got {current_interaction_type}"
            )
        assert previos_interaction_type in ["write", "read", "none"]
        assert current_interaction_type in ["write", "read"]

        if previos_interaction_type == "write":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_tweet_cur_tweet.md",
                    ),
                    "r",encoding="utf-8"    
                ) as f:
                    prompt_instructions = f.read()

                prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                    TWEET_WRITTEN_COUNT_LAST=tweet_written_count - 1,
                    SUPERSCRIPT_LAST=get_superscript(tweet_written_count - 1),
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_tweet_cur_read.md",
                    ),
                    "r",encoding="utf-8"    
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                        BELIEF_RATING=belief,
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_WRITTEN_COUNT=tweet_written_count,
                        SUPERSCRIPT=get_superscript(tweet_written_count),
                        TWEET_SEEN=tweet_seen,
                        REASONING=reasoning,
                    )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

        elif previos_interaction_type == "read":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_read_cur_tweet.md",
                    ),
                    "r", encoding="utf-8"   
                ) as f:
                    prompt_instructions = f.read()

                prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_read_cur_read.md",
                    ),
                    "r",encoding="utf-8"
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

        elif previos_interaction_type == "none":
            if current_interaction_type == "write":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_none_cur_tweet.md",
                    ),
                    "r",encoding="utf-8"
                ) as f:
                    prompt_instructions = f.read()

                prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                    TWEET_WRITTEN_COUNT=tweet_written_count,
                    SUPERSCRIPT=get_superscript(tweet_written_count),
                    TWEET_WRITTEN=tweet_written,
                )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

            elif current_interaction_type == "read":
                with open(
                    join(
                        self.prompt_template_root,
                        "step2b_add_to_memory_prev_none_cur_read.md",
                    ),
                    "r", encoding="utf-8"   
                ) as f:
                    prompt_instructions = f.read()

                if not rating_flag:
                    belief = extract_belief(response)
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_SEEN=tweet_seen, REASONING=reasoning, BELIEF_RATING=belief
                    )
                else:
                    reasoning = extract_reasoning(response)
                    prompt = safe_format(prompt_instructions.split("\n---------------------------\n")[0], 
                        TWEET_SEEN=tweet_seen, REASONING=reasoning
                    )
                self._append_memory_event(prompt, current_interaction_type=current_interaction_type, rating_snapshot=self.current_belief)

    def _append_memory_event(self, prompt_text: str, *, current_interaction_type: str = '', rating_snapshot=None, pinned: bool = False):
        """
        Append one structured event to the light-memory buffer.
        
            Light memory stores recent interactions as compact records instead of full transcripts. This keeps
            future prompts shorter while preserving enough belief-path context for memory-enabled conditions.
        """
        prompt = str(prompt_text or '').strip()
        if not prompt:
            return
        if not MEMORY_LIGHT_ENABLED:
            try:
                self.memory.memory.chat_memory.add_user_message(prompt)
            except Exception:
                pass
            return

        try:
            rating_value = int(self.current_belief if rating_snapshot is None else rating_snapshot)
        except Exception:
            try:
                rating_value = int(self.current_belief)
            except Exception:
                rating_value = None

        self.memory_event_counter += 1
        self.memory_events.append({
            'seq': int(self.memory_event_counter),
            'prompt': prompt,
            'interaction_type': str(current_interaction_type or '').strip().lower(),
            'rating': rating_value,
            # Pinned events (a persisting event headline) are never aged
            # out by the light-memory selector, so they colour every later prompt.
            'pinned': bool(pinned),
        })
        self._rebuild_light_memory_history()

    def _belief_path_message(self) -> str:
        """
        Summarize the agent's belief trajectory for light-memory prompts.
        
            The summary gives memory-enabled agents a compact view of how their own rating has changed over
            time without replaying every previous interaction.
        """
        if not MEMORY_LIGHT_ENABLED:
            return ''
        recent = self.memory_events[-LIGHT_MEMORY_BELIEF_PATH_LEN:]
        ratings = []
        for evt in recent:
            try:
                ratings.append(str(int(evt.get('rating'))))
            except Exception:
                continue
        if not ratings:
            return ''
        return 'Recent belief path (oldest -> newest): ' + ' -> '.join(ratings)

    def _selected_light_memory_events(self):
        """
        Select the subset of compact memory events that should remain visible.
        
            Light-memory mode intentionally keeps only a recent or representative subset so prompts do not grow
            unbounded during long simulations.
        """
        if not MEMORY_LIGHT_ENABLED:
            return list(self.memory_events)
        events = list(self.memory_events)
        if not events:
            return []
        recent_events = events[-LIGHT_MEMORY_MAX_EVENTS:] if LIGHT_MEMORY_MAX_EVENTS > 0 else []
        selected_ids = {int(evt.get('seq')) for evt in recent_events}
        extra_own = []
        if LIGHT_MEMORY_EXTRA_OWN_TWEETS > 0:
            for evt in reversed(events[:-len(recent_events)] if recent_events else events):
                if str(evt.get('interaction_type') or '').strip().lower() != 'write':
                    continue
                seq = int(evt.get('seq'))
                if seq in selected_ids:
                    continue
                extra_own.append(evt)
                selected_ids.add(seq)
                if len(extra_own) >= LIGHT_MEMORY_EXTRA_OWN_TWEETS:
                    break
        # Force-keep pinned events regardless of the recency window; when
        # nothing is pinned this dict is exactly recent_events + extra_own, so a run
        # with no event is unaffected.
        chosen = {int(e.get('seq')): e for e in (recent_events + extra_own)}
        for evt in events:
            if evt.get('pinned'):
                chosen[int(evt.get('seq'))] = evt
        return sorted(chosen.values(), key=lambda e: int(e.get('seq', 0)))

    def _rebuild_light_memory_history(self):
        """
        Reconstruct the visible memory buffer from selected compact memory events.
        
            This is used after adding or pruning light-memory records so the LangChain memory object reflects
            the compressed history that the run configuration allows the LLM to see.
        """
        if not MEMORY_LIGHT_ENABLED:
            return
        try:
            chat_mem = self.memory.memory.chat_memory
        except Exception:
            return
        selected = self._selected_light_memory_events()
        rebuilt = []
        header = self._belief_path_message()
        if header:
            rebuilt.append(HumanMessage(content=header))
        for evt in selected:
            prompt = str(evt.get('prompt') or '').strip()
            if prompt:
                rebuilt.append(HumanMessage(content=prompt))
        try:
            chat_mem.messages = rebuilt
        except Exception:
            try:
                msgs = getattr(chat_mem, 'messages', None)
                if msgs is not None:
                    msgs.clear()
                    msgs.extend(rebuilt)
            except Exception:
                pass

    def get_count_tweet_written(self):
        """Agent method for get count tweet written within the simulation runtime."""
        return self.count_tweet_written

    def increase_count_tweet_written(self):
        """Agent method for increase count tweet written within the simulation runtime."""
        self.count_tweet_written += 1

    def get_count_tweet_seen(self):
        """Agent method for get count tweet seen within the simulation runtime."""
        return self.count_tweet_seen

    def increase_count_tweet_seen(self):
        """Agent method for increase count tweet seen within the simulation runtime."""
        self.count_tweet_seen += 1

    def outdate_persona_memory(self):
        """Agent method for outdate persona memory within the simulation runtime."""
        self.persona = convert_text_from_present_to_past(self.persona)


def get_superscript(count):
    """Return the English ordinal suffix for any positive integer.

    The old implementation only covered 1-23, so later prompts could say
    things like "35None tweet" or "53None tweet". Keep this tiny and
    deterministic because it is only display/prompt wording.
    """
    try:
        n = abs(int(count))
    except Exception:
        return "th"
    if 10 <= (n % 100) <= 20:
        return "th"
    last = n % 10
    if last == 1:
        return "st"
    if last == 2:
        return "nd"
    if last == 3:
        return "rd"
    return "th"


def _should_use_native_ollama_chat(conversation) -> bool:
    """Decide whether to use Ollama's native /api/chat path.

    wrapper = always use ChatOllama/ConversationChain
    native  = always use Ollama /api/chat
    auto    = use native only for Qwen-family models
    """
    try:
        llm = getattr(conversation, 'llm', None)
        model = str(getattr(llm, 'model', '') or '').strip().lower()
    except Exception:
        model = ''

    env_force_native = os.environ.get('OD_FORCE_NATIVE_OLLAMA', '').strip().lower()
    if env_force_native in {'1', 'true', 'yes', 'on'}:
        return True

    env_path = os.environ.get('OD_OLLAMA_CHAT_PATH', '').strip().lower()
    mode = env_path or str(globals().get('OLLAMA_CHAT_PATH_MODE', 'wrapper') or 'wrapper').strip().lower()

    if mode == 'native':
        return True
    if mode == 'wrapper':
        return False
    return ('qwen' in model)


def _lc_message_to_ollama_dict(msg) -> dict:
    """Convert a LangChain message object into the role/content format expected by Ollama."""
    role = _msg_role(msg).lower()
    if role == 'human':
        role = 'user'
    elif role == 'ai':
        role = 'assistant'
    elif role not in {'system','user','assistant','tool'}:
        role = 'user'
    content = _msg_text(msg)
    if isinstance(content, list):
        content = '\\n'.join(str(x) for x in content)
    return {'role': role, 'content': str(content or '')}


def _split_prompt_blocks_preserve_prefix(text: str) -> tuple[str, list[str]]:
    """Split prompt text while preserving qwen control prefixes."""
    s = str(text or '')
    prefix = ''
    body = s
    stripped = s.lstrip()
    if stripped.startswith('/think'):
        prefix = '/think\n'
        body = stripped[len('/think'):].lstrip('\n')
    elif stripped.startswith('/no_think'):
        prefix = '/no_think\n'
        body = stripped[len('/no_think'):].lstrip('\n')
    blocks = [blk.strip() for blk in re.split(r"\n{2,}", body) if blk and blk.strip()]
    return prefix, blocks


def _normalize_prompt_block(block: str) -> str:
    """Normalize one prompt block before native Ollama message construction."""
    return re.sub(r"\s+", " ", str(block or '')).strip().lower()


def _dedupe_user_message_against_system(system_content: str, user_content: str) -> str:
    """Remove exact duplicated user blocks already present in the system message."""
    user_prefix, user_blocks = _split_prompt_blocks_preserve_prefix(user_content)
    if not user_blocks:
        return str(user_content or '')

    _, system_blocks = _split_prompt_blocks_preserve_prefix(system_content)
    system_norms = {_normalize_prompt_block(b) for b in system_blocks if _normalize_prompt_block(b)}

    kept = []
    seen_user = set()
    for blk in user_blocks:
        norm = _normalize_prompt_block(blk)
        if not norm:
            continue
        if norm in seen_user:
            continue
        seen_user.add(norm)
        if norm in system_norms:
            continue
        kept.append(blk)

    compacted = '\n\n'.join(kept).strip()
    if user_prefix:
        compacted = user_prefix + compacted.lstrip()
    return compacted or (user_prefix.rstrip() if user_prefix else str(user_content or '').strip())


def _collect_native_ollama_messages(conversation, prompt_text: str, include_history: bool = True):
    """
    Assemble the native Ollama chat message list for one LLM call.
    
        The function combines system prompt, optionally visible memory/history, and current user prompt,
        while avoiding duplicated prompt blocks. It is the native-path equivalent of what LangChain would
        send through ConversationChain.
    """
    history = _history_messages_with_memory_use_rule(conversation, include_history=include_history)
    try:
        formatted = conversation.prompt.format_prompt(history=history, input=prompt_text)
        msgs = formatted.to_messages()
    except Exception:
        msgs = []
        # Best-effort fallback: reconstruct from memory plus current prompt.
        llm_hist = history or []
        msgs.extend(llm_hist)
        from langchain_core.messages import HumanMessage
        msgs.append(HumanMessage(content=prompt_text))
    out = []
    for m in msgs:
        d = _lc_message_to_ollama_dict(m)
        if d.get('content') is None:
            d['content'] = ''
        out.append(d)

    # Reduce cross-message duplication without stripping semantic user-only blocks.
    try:
        system_contents = [str(m.get('content') or '') for m in out if str(m.get('role') or '') == 'system']
        if system_contents:
            system_joined = '\n\n'.join(system_contents)
            for m in reversed(out):
                if str(m.get('role') or '') == 'user':
                    m['content'] = _dedupe_user_message_against_system(system_joined, str(m.get('content') or ''))
                    break
    except Exception:
        pass
    return out

def _extract_native_ollama_text(payload: dict) -> str:
    """Extract final assistant content from a native Ollama response object."""
    if not isinstance(payload, dict):
        return ''
    msg = payload.get('message') or {}
    content = msg.get('content')
    if isinstance(content, list):
        try:
            content = '\n'.join(str(x) for x in content)
        except Exception:
            content = str(content)
    text = str(content or '').strip()
    if text:
        return text
    # Fallbacks for API variations / wrappers, but NEVER fall back to `thinking`.
    # Ollama documents `message.thinking` / `thinking` as a separate reasoning trace,
    # while `message.content` / `response` is the final answer. Returning reasoning here
    # pollutes Step-2 with planning text and breaks protocol parsing.
    for key in ('response', 'content', 'output_text', 'text'):
        val = payload.get(key)
        if isinstance(val, list):
            try:
                val = '\n'.join(str(x) for x in val)
            except Exception:
                val = str(val)
        if val:
            return str(val).strip()
    return ''



def _extract_native_ollama_thinking(payload: dict) -> str:
    """Extract qwen thinking content from a native Ollama response object."""
    if not isinstance(payload, dict):
        return ''
    msg = payload.get('message') or {}
    thinking = msg.get('thinking') or payload.get('thinking') or ''
    if isinstance(thinking, list):
        try:
            thinking = '\n'.join(str(x) for x in thinking)
        except Exception:
            thinking = str(thinking)
    return str(thinking or '').strip()

def _get_native_ollama_endpoint(conversation) -> str:
    """Resolve the native Ollama endpoint URL from the configured LLM object."""
    llm = getattr(conversation, 'llm', None)
    for attr in ('base_url', 'client_kwargs'):
        try:
            val = getattr(llm, attr, None)
            if isinstance(val, str) and val.strip():
                return val.rstrip('/')
            if isinstance(val, dict):
                base = val.get('base_url') or val.get('host')
                if isinstance(base, str) and base.strip():
                    return base.rstrip('/')
        except Exception:
            pass
    env_host = os.environ.get('OLLAMA_HOST') or os.environ.get('OLLAMA_BASE_URL')
    if env_host and str(env_host).strip():
        return str(env_host).rstrip('/')
    return 'http://localhost:11434'


def _native_ollama_request_headers(base_url: str = '') -> dict:
    """Headers for direct Ollama HTTP calls.

    Free ngrok endpoints may return a 403/browser-warning interstitial for API
    clients unless the request includes ngrok-skip-browser-warning. Adding the
    header is harmless for localhost/Ollama and avoids falling through to the
    slower ChatOllama path. The env var lets you disable/override it if needed.
    """
    headers = {
        'User-Agent': os.environ.get('OLLAMA_HTTP_USER_AGENT', 'llm-sim-ollama-client/1.0'),
    }
    url = str(base_url or '').strip().lower()
    skip_val = os.environ.get('NGROK_SKIP_BROWSER_WARNING', '')
    if skip_val or 'ngrok' in url:
        headers['ngrok-skip-browser-warning'] = skip_val or 'true'
    extra = os.environ.get('OLLAMA_HTTP_EXTRA_HEADERS', '').strip()
    if extra:
        for part in re.split(r'[;\n]+', extra):
            if ':' not in part:
                continue
            k, v = part.split(':', 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                headers[k] = v
    return headers


def _collect_native_ollama_options(conversation) -> tuple[str, dict, object]:
    """
    Translate wrapper-level LLM settings into native Ollama API options.
    
        The native API uses names such as num_predict, repeat_penalty, and top_p. This helper extracts the
        values from the ChatOllama object, applies prompt-level think overrides, and returns a clean options
        dictionary for /api/chat.
    """
    llm = getattr(conversation, 'llm', None)
    model = str(getattr(llm, 'model', '') or '').strip()
    options = {}
    # Collect explicit attrs.
    for name in (
        'temperature', 'top_p', 'top_k', 'repeat_penalty', 'repeat_last_n',
        'frequency_penalty', 'presence_penalty', 'min_p', 'num_predict',
        'max_tokens', 'seed', 'num_ctx', 'stop', 'keep_alive'
    ):
        try:
            val = getattr(llm, name, None)
        except Exception:
            val = None
        if val is not None:
            options[name] = val
    # Merge options dict if present.
    try:
        llm_opts = getattr(llm, 'options', None)
        if isinstance(llm_opts, dict):
            for k, v in llm_opts.items():
                if v is not None:
                    options[k] = v
    except Exception:
        pass
    think = None
    try:
        if hasattr(llm, 'think'):
            think = getattr(llm, 'think')
    except Exception:
        think = None
    if think is None:
        think = options.pop('think', None)
    if 'max_tokens' in options and 'num_predict' not in options:
        options['num_predict'] = options['max_tokens']
    options.pop('max_tokens', None)
    if _deterministic_mode_enabled():
        options['temperature'] = 0.0
        options['top_k'] = 1
        options['top_p'] = 1.0
    try:
        ctx = dict(NATIVE_OLLAMA_DEBUG_CONTEXT or {})
        step_ctx = str(ctx.get('step') or '').strip().lower()
        attempt_ctx = str(ctx.get('attempt') or '').strip().lower()
        label_ctx = str(ctx.get('label') or '').strip().lower()

        # Compact retry prompts stay deliberately tiny.
        if 'num_predict' in options:
            npv = int(options.get('num_predict') or 0)
            if step_ctx == 'step3' and attempt_ctx in {'1b', '1c', '2'}:
                options['num_predict'] = max(160, min(npv or 224, 224))
            elif step_ctx == 'step2' and attempt_ctx in {'2'}:
                options['num_predict'] = max(160, min(npv or 224, 224))

        # Native rescue should be deterministic and must not burn budget in hidden reasoning.
        if label_ctx.startswith('native_rescue'):
            think = False
            try:
                npv = int(options.get('num_predict') or 0)
            except Exception:
                npv = 0
            if step_ctx == 'step3':
                floor = 1280 if 'compact' in label_ctx else 1152
                options['num_predict'] = max(npv, floor) if npv > 0 else floor
            elif step_ctx == 'step2':
                if 'ultra' in label_ctx:
                    floor = 640
                elif 'compact' in label_ctx:
                    floor = 768
                else:
                    floor = 1024
                options['num_predict'] = max(npv, floor) if npv > 0 else floor
    except Exception:
        pass
    return model, options, think


def _save_native_ollama_turn(conversation, prompt_text: str, reply_text: str) -> None:
    """Manually store a native Ollama user/assistant turn in LangChain memory."""
    try:
        mem = getattr(conversation, 'memory', None)
        chat_mem = getattr(mem, 'chat_memory', None)
        if chat_mem is not None:
            chat_mem.add_user_message(prompt_text)
            chat_mem.add_ai_message(reply_text)
            return
    except Exception:
        pass
    try:
        conversation.memory.save_context({'input': prompt_text}, {'response': reply_text})
    except Exception:
        pass


def _extract_prompt_int_field(prompt_text: str, field_name: str, default: int | None = None) -> int | None:
    """Extract one integer field from prompt text by label."""
    try:
        m = re.search(rf"(?im)^\s*{re.escape(str(field_name))}\s*:\s*([+-]?\d+)\s*$", str(prompt_text or ""))
        return int(m.group(1)) if m else default
    except Exception:
        return default


def _extract_prompt_allowed_set(prompt_text: str) -> list[int]:
    """Extract the Step-3 allowed-rating set from prompt text."""
    s = str(prompt_text or "")
    m = re.search(r"(?im)^\s*(?:ALLOWED_FINAL_RATING_SET|ALLOWED_SET)\s*:\s*(\[[^\n]+\])\s*$", s)
    if not m:
        return []
    try:
        return [int(x) for x in re.findall(r"[+-]?\d+", m.group(1) or "")]
    except Exception:
        return []


def _extract_prompt_tweet_text(prompt_text: str) -> str:
    """Extract the tweet shown to the listener from prompt text."""
    s = str(prompt_text or "")
    m = re.search(
        r"(?is)\nTWEET:\s*\n(.*?)\n\s*\n(?:Remember,|Open-world addition:|Evaluation rule|Task|Rating scale meaning|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)",
        s,
    )
    if not m:
        return ""
    return re.sub(r"\s+", " ", str(m.group(1) or "")).strip()


def _extract_compact_step3_shown_material(prompt_text: str, *, max_items: int = 2, max_chars_per_item: int = 240) -> str:
    """Extract compact shown-material context from a Step-3 prompt."""
    s = str(prompt_text or "")
    if not s:
        return ""
    patterns = [
        r"(?is)\nLIVE WEB SEARCH RESULTS \(shown material\)\n(.*?)(?:\n\s*\n(?:Evaluation rule|Task|Rating scale meaning|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
        r"(?is)\nRetrieved snippets(?: \(shown material\))?\n(.*?)(?:\n\s*\n(?:Evaluation rule|Task|Rating scale meaning|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
        r"(?is)\nFACT PACK(?: ABOUT .*?)?\n(.*?)(?:\n\s*\n(?:Evaluation rule|Task|Rating scale meaning|Decision rule|How to think about the rating|Explanation rule|Output discipline|Output format:|ALLOWED_FINAL_RATING_SET:|ALLOWED_SET:)|\Z)",
    ]
    items = []
    for pat in patterns:
        m = re.search(pat, s)
        if not m:
            continue
        block = str(m.group(1) or "")
        for ln in block.splitlines():
            line = re.sub(r"\s+", " ", ln).strip()
            if not line:
                continue
            if not (line.startswith('- ') or line.startswith('• ')):
                continue
            line = re.sub(r"^[•-]\s*", "", line).strip()
            if not line:
                continue
            if len(line) > max_chars_per_item:
                line = line[:max_chars_per_item].rstrip(' ,;:-') + '…'
            items.append(f"- {line}")
            if len(items) >= max_items:
                break
        if items:
            break
    if not items:
        return ""
    return "SHOWN MATERIAL:\n" + "\n".join(items)


def _extract_reasoning_quoted_candidate(raw_text: str, field: str = "tweet") -> str:
    """Recover a likely protocol answer from qwen thinking text."""
    s = str(raw_text or "")
    if not s:
        return ""
    quoted = re.findall(r'["“](.{20,280}?)["”]', s, flags=re.S)
    usable = []
    for q in quoted:
        cand = re.sub(r"\s+", " ", str(q or "")).strip()
        low = cand.lower()
        if len(cand) < 20:
            continue
        if any(tok in low for tok in ["present view", "current view", "final_rating", "tweet:", "explanation:", "role-play", "allowed rating", "allowed set", "rating means"]):
            continue
        if field == "tweet":
            cand = _strip_leaked_prompt_headers(cand)
            cand = re.sub(r"(?i)^here(?:'s| is) (?:my )?(?:tweet|post)[:\-]\s*", "", cand).strip()
        usable.append(cand)
    return usable[-1] if usable else ""


def _classify_tweet_stance(tweet_text: str) -> str | None:
    """Best-effort generic stance label for fallback-only Step-3 recovery.

    Returns one of: support, challenge, mixed, or None when unusable.
    This is intentionally lightweight and topic-agnostic.
    """
    s = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    if not s:
        return None

    support_cues = [
        'i think', 'i believe', 'is real', 'did happen', 'supports', 'confirms', 'strong evidence',
        'well documented', 'documented', 'credible', 'convincing', 'shows that', 'indicates that',
        'makes me think', 'lean toward', 'leans toward', 'more likely'
    ]
    challenge_cues = [
        'i doubt', 'i strongly doubt', 'not real', 'did not happen', 'hoax', 'fake', 'staged',
        'unresolved', 'does not hold together', "doesn't hold together", 'keeps me uncertain',
        'not enough evidence', 'gap', 'mismatch', 'inconsistent', 'unlikely', 'skeptical'
    ]
    mixed_cues = [
        'but', 'however', 'mixed', 'uncertain', 'unsure', 'on the one hand', 'on the other hand',
        'some force', 'still unresolved', 'questions remain'
    ]

    has_support = any(c in s for c in support_cues)
    has_challenge = any(c in s for c in challenge_cues)
    has_mixed = any(c in s for c in mixed_cues)

    if (has_support and has_challenge) or has_mixed:
        return 'mixed'
    if has_support:
        return 'support'
    if has_challenge:
        return 'challenge'
    return None


def _recover_native_qwen_empty_content(prompt_text: str, thinking_text: str, step_ctx: str) -> str:
    """
    Recover a protocol answer when qwen returns thinking but empty final content.
    
        Some qwen reasoning runs place useful final-rating information inside the thinking field while the
        visible message content is empty. This recovery path attempts a conservative reconstruction using
        the prompt, allowed set, and thinking text before falling back to normal repair logic.
    """
    step_norm = str(step_ctx or "").strip().lower()
    claim_txt = _current_claim_text(prompt_text)
    if step_norm.startswith("step2"):
        current = _extract_prompt_int_field(prompt_text, "PRESENT VIEW", 0)
        candidate = _extract_reasoning_quoted_candidate(thinking_text, field="tweet")
        if candidate:
            candidate = _limit_to_n_sentences(candidate, max_sentences=2)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                return f"FINAL_RATING: {int(current)}\nTWEET: {candidate}"
        return _fallback_step2(expected_current_belief=int(current), claim_txt=claim_txt)

    if step_norm.startswith("step3"):
        pre_belief = _extract_prompt_int_field(prompt_text, "PRESENT VIEW", 0)
        allowed = _extract_prompt_allowed_set(prompt_text)
        tweet_txt = _extract_prompt_tweet_text(prompt_text)
        tweet_stance = None
        try:
            tweet_stance = _classify_tweet_stance(tweet_txt)
        except Exception:
            tweet_stance = None
        candidate = _extract_reasoning_quoted_candidate(thinking_text, field="explanation")
        if candidate:
            rating = extract_belief(thinking_text)
            if rating is None:
                rating = int(pre_belief)
            try:
                rating = int(rating)
            except Exception:
                rating = int(pre_belief)
            if allowed and rating not in allowed:
                rating = int(pre_belief) if int(pre_belief) in allowed else int(allowed[0])
            return _rewrite_step3_output_with_explanation(int(rating), candidate)
        return _fallback_step3(pre_belief=int(pre_belief), claim_txt=claim_txt, tweet_stance=tweet_stance, tweet_txt=tweet_txt)

    return ""


def _native_ollama_chat(conversation, prompt_text: str, *, include_history: bool = True, save_turn: bool = True, seed_override: int | None = None) -> str:
    """
    Execute one direct Ollama /api/chat call with qwen-aware retries and logging.
    
        The native path is preferred for qwen because it exposes thinking, done reasons, and low-level
        options more reliably than the wrapper. The function handles length retries, empty-content recovery,
        debug dumps, memory saving, and native-events logging.
    """
    import requests

    model, options, think = _collect_native_ollama_options(conversation)
    base_url = _get_native_ollama_endpoint(conversation)
    url = base_url.rstrip('/') + '/api/chat'
    ctx = dict(NATIVE_OLLAMA_DEBUG_CONTEXT or {})
    step_ctx = str(ctx.get('step') or '').strip().lower()
    label_ctx = str(ctx.get('label') or '').strip().lower()
    prompt_to_send = str(prompt_text or '')
    if _is_qwen_family_model(model):
        prompt_to_send = _compact_qwen_prompt_text(prompt_to_send, step=step_ctx)
        think = _apply_prompt_control_think_override(prompt_to_send, model, think)
    structured_step = step_ctx if (_structured_output_enabled() and step_ctx in {"step2", "step3"}) else None
    if structured_step:
        _skey = "tweet" if structured_step == "step2" else "explanation"
        prompt_to_send = (prompt_to_send.rstrip() +
                          "\n\nOUTPUT FORMAT OVERRIDE: respond with ONLY a JSON object, exactly "
                          '{"final_rating": <integer>, "' + _skey + '": "<text>"} - no other keys, no extra text.')
    messages = _collect_native_ollama_messages(conversation, prompt_to_send, include_history=include_history)

    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
    }
    if structured_step:
        payload['format'] = 'json'   # grammar-constrained decoding: the model cannot emit non-JSON
    if think is not None:
        payload['think'] = think
    if options:
        opts = dict(options)
        if seed_override is not None:
            opts['seed'] = int(seed_override)
        payload['options'] = opts

    final_payload = payload
    resp = requests.post(url, json=payload, headers=_native_ollama_request_headers(base_url), timeout=(30, 600))
    resp.raise_for_status()
    data = resp.json()
    text = _extract_native_ollama_text(data)
    thinking = _extract_native_ollama_thinking(data)
    # Thinking/token accounting: reasoning budget usage is a per-model property worth
    # reporting (cost + behaviour); eval_count/prompt_eval_count come from Ollama itself.
    try:
        _sk = step_ctx or 'other'
        RUN_METRICS[f'native_calls_total::{_sk}'] += 1
        RUN_METRICS[f'thinking_chars_total::{_sk}'] += len(thinking or '')
        if thinking:
            RUN_METRICS[f'thinking_calls_total::{_sk}'] += 1
        _ec = data.get('eval_count'); _pc = data.get('prompt_eval_count')
        if isinstance(_ec, int):
            RUN_METRICS[f'output_tokens_total::{_sk}'] += _ec
        if isinstance(_pc, int):
            RUN_METRICS[f'prompt_tokens_total::{_sk}'] += _pc
    except Exception:
        pass
    note = ('empty_final_text' if not text else '')
    _dump_native_ollama_payload_debug(
        prompt_text=prompt_to_send,
        request_payload=payload,
        response_payload=data,
        extracted_text=text,
        extracted_thinking=thinking,
        note=note,
    )

    done_reason = str(data.get('done_reason') or '').strip().lower()
    _append_native_event_row(
        step_kind=step_ctx,
        agent_name=str(ctx.get('agent_name') or ''),
        label=label_ctx,
        attempt='1',
        event_type=('empty_final_text' if not text else ('length_stop' if done_reason == 'length' else 'native_response')),
        model=model,
        done_reason=done_reason,
        text=text,
        thinking=thinking,
        payload=payload,
    )
    should_retry_empty_qwen = (not text) and _is_qwen_family_model(model)
    if should_retry_empty_qwen:
        retry_payload = dict(payload)
        retry_payload['options'] = _boost_qwen_num_predict_on_length(dict(payload.get('options') or {}), step=step_ctx)
        retry_prompt = prompt_to_send
        if step_ctx in {"step2", "step3"}:
            retry_prompt = _compact_qwen_prompt_text(prompt_to_send, step=step_ctx)
        retry_payload['messages'] = _collect_native_ollama_messages(conversation, retry_prompt, include_history=include_history)
        try:
            print(f"[info][ollama][native] empty final text (done_reason={done_reason or 'unknown'}); retrying once with boosted num_predict={retry_payload['options'].get('num_predict')}")
        except Exception:
            pass
        resp2 = requests.post(url, json=retry_payload, headers=_native_ollama_request_headers(base_url), timeout=(30, 900))
        resp2.raise_for_status()
        data2 = resp2.json()
        text2 = _extract_native_ollama_text(data2)
        thinking2 = _extract_native_ollama_thinking(data2)
        done_reason2 = str(data2.get('done_reason') or '').strip().lower()
        _append_native_event_row(
            step_kind=step_ctx,
            agent_name=str(ctx.get('agent_name') or ''),
            label=label_ctx,
            attempt='2',
            event_type=('retry_empty_final_text' if not text2 else ('retry_length_stop' if done_reason2 == 'length' else 'retry_native_response')),
            model=model,
            done_reason=done_reason2,
            text=text2,
            thinking=thinking2,
            payload=retry_payload,
            note=('length_retry' if done_reason == 'length' else 'empty_content_retry'),
        )
        _dump_native_ollama_payload_debug(
            prompt_text=retry_prompt,
            request_payload=retry_payload,
            response_payload=data2,
            extracted_text=text2,
            extracted_thinking=thinking2,
            note=(("length_retry" if done_reason == 'length' else 'empty_content_retry') + ('_empty_final_text' if not text2 else '')),
        )
        final_payload = retry_payload
        if text2:
            text = text2
            thinking = thinking2
            data = data2
        else:
            thinking = thinking2 or thinking
            data = data2

    if not text and _is_qwen_family_model(model):
        recovered = _recover_native_qwen_empty_content(prompt_to_send, thinking, step_ctx)
        if recovered:
            text = recovered
            _append_native_event_row(
                step_kind=step_ctx,
                agent_name=str(ctx.get('agent_name') or ''),
                label=label_ctx,
                attempt='local_recovery',
                event_type='qwen_empty_content_local_recovery',
                model=model,
                done_reason=str(data.get('done_reason') or '').strip().lower(),
                text=text,
                thinking=thinking,
                payload=final_payload,
                note='recovered_from_message_thinking_or_prompt_context',
            )
            try:
                print(f"[info][ollama][native] recovered empty Qwen content with local {step_ctx or 'protocol'} fallback")
            except Exception:
                pass

    if not text and thinking:
        try:
            print('[warn][ollama][native] received reasoning in message.thinking but empty final message.content; reasoning omitted from protocol output')
        except Exception:
            pass
    try:
        setattr(
            conversation,
            '_last_native_ollama_meta',
            {
                'step': step_ctx,
                'label': label_ctx,
                'done_reason': str(data.get('done_reason') or '').strip().lower(),
                'had_text': bool(str(text or '').strip()),
                'num_predict': dict(final_payload.get('options') or {}).get('num_predict'),
                'think': final_payload.get('think', None),
            },
        )
    except Exception:
        pass
    if structured_step and text:
        converted = _structured_json_to_canonical(text, structured_step)
        if converted:
            text = converted
    if save_turn:
        _save_native_ollama_turn(conversation, prompt_to_send, text)
    return text


def _extract_chatollama_invoke_text(resp) -> str:
    """Extract assistant text from a LangChain ChatOllama invoke response."""
    if resp is None:
        return ''
    if isinstance(resp, str):
        return resp

    try:
        content = getattr(resp, 'content', None)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get('text') or part.get('content') or part.get('value') or ''
                    if txt:
                        parts.append(str(txt))
                elif part is not None:
                    parts.append(str(part))
            content = "\n".join(p for p in parts if str(p).strip()).strip()
        if content is not None and str(content).strip():
            return str(content)
    except Exception:
        pass

    for attr in ('text', 'output_text', 'response'):
        try:
            val = getattr(resp, attr, None)
            if val is not None and str(val).strip():
                return str(val)
        except Exception:
            pass

    for attr in ('additional_kwargs', 'response_metadata'):
        try:
            data = getattr(resp, attr, None)
            if isinstance(data, dict):
                for key in ('content', 'text', 'output_text', 'response'):
                    val = data.get(key)
                    if val is not None and str(val).strip():
                        return str(val)
        except Exception:
            pass

    try:
        if hasattr(resp, 'model_dump'):
            dumped = resp.model_dump()
        elif hasattr(resp, 'dict'):
            dumped = resp.dict()
        else:
            dumped = None
        if isinstance(dumped, dict):
            stack = [dumped]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for key in ('content', 'text', 'output_text', 'response'):
                        val = cur.get(key)
                        if val is not None and str(val).strip():
                            return str(val)
                    stack.extend(cur.values())
                elif isinstance(cur, (list, tuple)):
                    stack.extend(cur)
        
    except Exception:
        pass

    return ''


def _extract_chatollama_invoke_thinking(resp) -> str:
    """Extract qwen thinking metadata from a ChatOllama response when available."""
    if resp is None:
        return ''
    if isinstance(resp, str):
        return ''

    candidate_keys = ('thinking', 'reasoning', 'reasoning_content', 'thought', 'thoughts')
    try:
        for attr in candidate_keys:
            val = getattr(resp, attr, None)
            if val is not None and str(val).strip():
                return str(val)
    except Exception:
        pass

    for attr in ('additional_kwargs', 'response_metadata'):
        try:
            data = getattr(resp, attr, None)
            if isinstance(data, dict):
                stack = [data]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for key in candidate_keys:
                            val = cur.get(key)
                            if val is not None and str(val).strip():
                                return str(val)
                        stack.extend(cur.values())
                    elif isinstance(cur, (list, tuple)):
                        stack.extend(cur)
        except Exception:
            pass

    try:
        if hasattr(resp, 'model_dump'):
            dumped = resp.model_dump()
        elif hasattr(resp, 'dict'):
            dumped = resp.dict()
        else:
            dumped = None
        if isinstance(dumped, dict):
            stack = [dumped]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for key in candidate_keys:
                        val = cur.get(key)
                        if val is not None and str(val).strip():
                            return str(val)
                    stack.extend(cur.values())
                elif isinstance(cur, (list, tuple)):
                    stack.extend(cur)
    except Exception:
        pass

    return ''


def _recover_wrapper_empty_output(conversation, prompt, *, resp=None, use_history: bool = True, save_turn: bool = True):
    """Recover a wrapper-path output when ChatOllama returns empty content."""
    recovered = ''
    recovery_source = ''
    try:
        step_ctx = str((NATIVE_OLLAMA_DEBUG_CONTEXT or {}).get('step') or '').strip().lower()
    except Exception:
        step_ctx = ''

    try:
        model_name, _opts, _think = _collect_native_ollama_options(conversation)
    except Exception:
        model_name = ''

    thinking_text = _extract_chatollama_invoke_thinking(resp)
    if thinking_text and _is_qwen_family_model(model_name):
        try:
            recovered = _recover_native_qwen_empty_content(prompt, thinking_text, step_ctx)
            recovered = '' if recovered is None else str(recovered)
            if recovered.strip():
                recovery_source = 'wrapper_reasoning_recovery'
        except Exception:
            recovered = ''
            recovery_source = ''

    if not str(recovered or '').strip():
        try:
            recovered = _native_ollama_chat(conversation, prompt, include_history=use_history, save_turn=False)
            recovered = '' if recovered is None else str(recovered)
            if recovered.strip():
                recovery_source = 'wrapper_native_recovery'
        except Exception:
            recovered = ''
            recovery_source = ''

    if recovered.strip() and save_turn and use_history:
        try:
            _save_native_ollama_turn(conversation, prompt, recovered)
        except Exception:
            pass

    return recovered, recovery_source


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5), reraise=True)
def _invoke_llm_with_retry(llm, msgs):
    """Call the LLM with a bounded retry on transient errors (e.g. Ollama hiccups).

    Wraps only the network call, before any conversation turn is saved, so a failed
    attempt cannot duplicate history. After the attempts are exhausted the original
    exception is re-raised so the existing predict() fallback still runs.
    """
    return llm.invoke(msgs)
def _invoke_chatollama_path(conversation, prompt, *, use_history: bool = True, save_turn: bool = True):
    """
    Execute one LLM call through the LangChain ChatOllama wrapper.
    
        This wrapper path is used when native mode is disabled or as a fallback after a native failure. It
        applies temporary qwen think overrides, extracts text from several possible response formats, and
        restores wrapper state afterward.
    """
    out = ''
    resp = None
    _restore_prompt_control_think = _set_wrapper_prompt_control_think_override(conversation, prompt)
    try:
        formatted = conversation.prompt.format_prompt(
            history=[] if not use_history else _get_history_messages(conversation),
            input=prompt,
        )
        msgs = formatted.to_messages()
        resp = _invoke_llm_with_retry(conversation.llm, msgs)
        out = _extract_chatollama_invoke_text(resp)
        recovery_source = ''
        if not str(out or '').strip():
            out, recovery_source = _recover_wrapper_empty_output(
                conversation,
                prompt,
                resp=resp,
                use_history=use_history,
                save_turn=save_turn,
            )
        try:
            setattr(conversation, '_last_wrapper_llm_response', resp)
            setattr(
                conversation,
                '_last_wrapper_llm_meta',
                {
                    'used_invoke': True,
                    'had_text': bool(str(out or '').strip()),
                    'use_history': bool(use_history),
                    'save_turn': bool(save_turn),
                    'empty_recovery_source': recovery_source,
                },
            )
        except Exception:
            pass
        if save_turn and use_history and str(out or '').strip() and not recovery_source:
            _save_native_ollama_turn(conversation, prompt, str(out))
        try:
            _restore_prompt_control_think()
        except Exception:
            pass
        return '' if out is None else str(out)
    except Exception:
        used_predict = True
        out = conversation.predict(input=prompt)
        out = '' if out is None else str(out)
        recovery_source = ''
        if used_predict and not out.strip():
            try:
                pop_last_turn_from_conversation(conversation)
            except Exception:
                pass
            out, recovery_source = _recover_wrapper_empty_output(
                conversation,
                prompt,
                resp=None,
                use_history=use_history,
                save_turn=save_turn,
            )
        try:
            setattr(
                conversation,
                '_last_wrapper_llm_meta',
                {
                    'used_invoke': False,
                    'used_predict_fallback': True,
                    'had_text': bool(str(out or '').strip()),
                    'use_history': bool(use_history),
                    'save_turn': bool(save_turn),
                    'empty_recovery_source': recovery_source,
                },
            )
        except Exception:
            pass
        try:
            _restore_prompt_control_think()
        except Exception:
            pass
        return out


def get_llm_response(conversation, prompt, *, use_history: bool = True, save_turn: bool = True):
    """
    Route a Step-2 or Step-3 prompt through the configured LLM execution path.
    
        This is the single high-level LLM entry point. It traces prompts/outputs when requested, chooses
        native versus wrapper execution, falls back when needed, and returns plain text to the Step-specific
        validation pipeline.
    """
    if USE_FAKE_LLM:
        m = re.search(r"name:\s*([^,]+)", prompt)
        name = m.group(1).strip() if m else "Agent"
        _trace_prompt(conversation, prompt, tag="FAKE_LLM")
        out = f"{name} tweets: I currently express my opinion about the topic. (dummy tweet)"
        _trace_output(out, tag="FAKE_LLM")
        return out
    else:
        _trace_prompt(conversation, prompt, tag="LLM")
        out = None
        native_err = None
        used_native = False
        if _should_use_native_ollama_chat(conversation):
            used_native = True
            try:
                out = _native_ollama_chat(conversation, prompt, include_history=use_history, save_turn=save_turn)
            except Exception as e:
                native_err = e
                out = None
                try:
                    print(f"[warn][llm] native Ollama chat path failed; falling back to ChatOllama/ConversationChain: {e}")
                except Exception:
                    pass
            if out is not None and not str(out).strip():
                out = None
                try:
                    print("[warn][llm] native Ollama chat returned empty final text after recovery; falling back to ChatOllama/ConversationChain")
                except Exception:
                    pass

        if out is None:
            out = _invoke_chatollama_path(conversation, prompt, use_history=use_history, save_turn=save_turn)
            if used_native and native_err is None:
                try:
                    print("[info][llm] ChatOllama/ConversationChain fallback used after empty native final output")
                except Exception:
                    pass
        out = '' if out is None else str(out)
        _trace_output(out, tag="LLM")
        return out




def pop_last_turn_from_conversation(conversation) -> None:
    """Remove the last (user, ai) message pair from LangChain memory.

    We use this to prevent invalid retry attempts from polluting the agent's memory.
    """
    try:
        msgs = conversation.memory.chat_memory.messages
    except Exception:
        try:
            msgs = conversation.memory.chat_memory.messages
        except Exception:
            return
    if not msgs:
        return
    if len(msgs) >= 2:
        msgs.pop()
        msgs.pop()
    else:
        msgs.clear()







def _strip_model_think_and_fence_artifacts(text: str) -> str:
    """Remove hidden-reasoning wrappers while PRESERVING the final protocol payload.

    Qwen-family models often wrap the usable answer in one of these containers:
      - <think>...</think> / <thinking>...</thinking>
      - ```...``` fenced blocks
      - <answer>...</answer> / <response>...</response>

    The old implementation deleted entire fenced blocks, which could erase the only valid
    FINAL_RATING/TWEET or FINAL_RATING/EXPLANATION payload. Here we unwrap containers but keep
    their inner content whenever that inner content may be the real answer.
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""

    # Remove hidden reasoning blocks completely.
    s = re.sub(r"(?is)<think\b[^>]*>.*?</think>", " ", s)
    s = re.sub(r"(?is)<thinking\b[^>]*>.*?</thinking>", " ", s)

    # If the whole payload is wrapped in a common XML tag, unwrap it.
    for _ in range(3):
        m = re.match(
            r"(?is)^\s*<(response|answer|output|assistant|final|final_answer)\b[^>]*>\s*(.*?)\s*</\1>\s*$",
            s,
        )
        if not m:
            break
        s = (m.group(2) or "").strip()

    # Unwrap fenced code blocks while preserving their contents.
    def _unwrap_fence(m):
        """Remove one layer of markdown code-fence wrapping from model output."""
        inner = (m.group(1) or "").strip()
        return ("\n" + inner + "\n") if inner else "\n"

    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(?is)```(?:[A-Za-z0-9_+\-.:]+)?\s*\n(.*?)\n```", _unwrap_fence, s)
        s = re.sub(r"(?is)~~~(?:[A-Za-z0-9_+\-.:]+)?\s*\n(.*?)\n~~~", _unwrap_fence, s)

    # Remove any leftover fence marker lines only.
    s = re.sub(r"(?m)^\s*```(?:[A-Za-z0-9_+\-.:]+)?\s*$", " ", s)
    s = re.sub(r"(?m)^\s*~~~(?:[A-Za-z0-9_+\-.:]+)?\s*$", " ", s)

    # Remove lone XML wrapper lines that sometimes leak around the payload.
    s = re.sub(r"(?im)^\s*</?(?:response|answer|output|analysis|assistant|final|final_answer)\s*>\s*$", " ", s)

    return s.strip()


def _normalize_protocol_headers(text: str) -> str:
    """Normalize common markdown-decorated protocol headers.

    Handles variants such as:
      - **FINAL_RATING:** 1
      - - TWEET: ...
      - > EXPLANATION: ...
      - `FINAL RATING:` -1
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""

    # Remove simple quote / bullet prefixes that sometimes precede protocol headers.
    s = re.sub(r"(?im)^\s*[>•\-*]+\s*(?=(?:\*\*|__|`)?\s*(?:FINAL(?:_| )RATING|TWEET|EXPLANATION|ANCHOR|REASONING|REASON)\b)", "", s)

    def _repl(m):
        """Normalize one matched protocol header to the canonical spelling."""
        header = (m.group(1) or "").strip().upper().replace(" ", "_")
        return f"{header}: "

    s = re.sub(
        r"(?im)^\s*(?:\*\*|__|`)?\s*(FINAL(?:_| )RATING|TWEET|EXPLANATION|ANCHOR|REASONING|REASON)\s*:(?:\*\*|__|`)?\s*",
        _repl,
        s,
    )
    return s

def _step2_tweet_has_valid_public_body(tweet_text: str) -> bool:
    """Accept only tweet bodies with enough real content to be worth keeping."""
    s = _compact_step2_tweet_text(tweet_text, max_sentences=2)
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    if not s:
        return False
    if _is_obviously_non_tweet_payload(s):
        return False
    if not _tweet_text_has_usable_step3_content(s):
        return False
    words = re.findall(r"[A-Za-z']+", s)
    if len(words) < 4:
        return False
    if len(words) == 4 and len(s) < 22 and not re.search(r"[.!?]$", s):
        return False
    return True


def _normalize_step2_missing_tweet_label(raw_text: str) -> str:
    """Normalize a common rescue-format miss without changing tweet content.

    Accept only the narrow case:
      FINAL_RATING: <allowed rating>
      <plain tweet text>

    This fixes native-rescue outputs that omit the literal ``TWEET:`` label.
    It does not invent content from prompts or from protocol/instruction lines.
    """
    s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(raw_text or ""))
    if not s:
        return ""
    if re.search(r"(?im)^\s*TWEET\s*:", s):
        return ""
    m = re.search(r"(?im)^\s*FINAL(?:_| )RATING\s*:\s*([+-]?\d)\s*$", s)
    if not m:
        return ""
    try:
        rating = int(m.group(1))
    except Exception:
        return ""
    if rating not in {-2, -1, 0, 1, 2}:
        return ""

    tail = s[m.end():]
    candidate_lines = []
    stop_header = re.compile(r"(?i)^\s*(?:FINAL(?:_| )RATING|TWEET|EXPLANATION|REASONING|ANCHOR|NOTES?|CLAIM|PRESENT VIEW|CURRENT VIEW|SHOWN MATERIAL|SHOWN TWEET|TASK|OUTPUT|FORMAT|START YOUR RESPONSE|AGENT_PERSONA_CARD|RATING SCALE)\b")
    bad_instruction = re.compile(r"(?i)\b(?:output exactly|nothing else|start your response|line 2 must|use only|write one short|final_rating:|tweet:)\b")
    for ln in str(tail or "").splitlines():
        raw = str(ln or "").strip().strip('`').strip()
        if not raw:
            continue
        if stop_header.match(raw):
            break
        if bad_instruction.search(raw):
            break
        if re.match(r"^/?(?:think|no_think)\b", raw, flags=re.I):
            continue
        if re.match(r"^[>•*\-]+\s*$", raw):
            continue
        candidate_lines.append(raw)
        # A real Step-2 tweet is capped to 1-2 sentences; more lines are more
        # likely to be protocol spillover than missing-label content.
        if len(candidate_lines) >= 2:
            break

    if not candidate_lines:
        return ""
    body = " ".join(candidate_lines).strip()
    body = _strip_leaked_prompt_headers(body)
    body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?)\s*:", "", body).strip()
    body = re.sub(r"\[[^\]]+\]", "", body).strip()
    body = re.sub(r"\s+", " ", body).strip()
    body = _compact_step2_tweet_text(body, max_sentences=2)
    if not _step2_tweet_has_valid_public_body(body):
        return ""
    return f"FINAL_RATING: {rating}\nTWEET: {body}".strip()

def _extract_step2_multiline_tweet_body(text: str) -> str:
    """Extract a TWEET body even when the header is followed by later non-empty lines."""
    s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(text or ""))
    if not s:
        return ""

    lines = str(s).splitlines()
    collecting = False
    body_lines = []
    header_re = re.compile(r"(?im)^\s*(FINAL(?:_| )RATING|TWEET|EXPLANATION|REASONING|REASON|ANCHOR|NOTES?)\s*:\s*(.*)$")

    for ln in lines:
        m = header_re.match(str(ln or ""))
        if not collecting:
            if not m:
                continue
            field = str(m.group(1) or "").strip().upper().replace(" ", "_")
            if field != "TWEET":
                continue
            tail = str(m.group(2) or "").strip()
            if tail:
                body_lines.append(tail)
            collecting = True
            continue

        if m:
            break
        raw = str(ln or "").strip()
        if not raw:
            continue
        body_lines.append(raw)

    body = " ".join(body_lines).strip()
    body = _strip_leaked_prompt_headers(body)
    body = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", body).strip()
    body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?)\s*:", "", body).strip()
    body = re.sub(r"\[[^\]]+\]", "", body).strip()
    body = re.sub(r"\s+", " ", body).strip()

    if not _step2_tweet_has_valid_public_body(body):
        return ""
    return body

def strip_step2_meta_lines(tweet_text: str) -> str:
    """Remove meta/compliance/scaffolding text sometimes emitted by the model in Step-2.

    Goal: enforce a clean 'social channel' tweet body with no protocol contamination.

    Removes, for example:
      - Note / Reminder / Please note / P.S. meta lines
      - Anything referencing prompt/rules/format/tag selection
      - Lines containing CURRENT_BELIEF / FINAL_RATING / LAST_R_TAG / EXPLANATION / TRUTH_DECISION
      - Sentence-by-sentence scaffolding (e.g., 'Sentence 1:', 'Anchor:', 'Assumption:', 'Reasoning:')
      - Markdown/code-fence junk

    This function is conservative: it removes protocol/meta text but does not rewrite stance/content.
    """
    if tweet_text is None:
        return ""
    s = _strip_model_think_and_fence_artifacts(tweet_text)
    if not s:
        return ""

    # Drop code fences that sometimes appear in outputs
    s = re.sub(r"(?s)```.*?```", " ", s)

    lines = s.splitlines()
    kept: list[str] = []

    # Whole-line prefixes that are almost always protocol/meta
    meta_prefixes = (
        "note:",
        "please note",
        "reminder:",
        "p.s.",
        "ps:",
        "here is my response",
        "here is the response",
        "output format",
        "format error",
        "format:",
        "response format",
        "tweet micro-structure",
        "tweet structure",
        "sentence",
        "anchor",
        "assumption",
        "reasoning:",
        "rules:",
        "requirements:",
    )

    # Substrings that indicate contamination (drop the entire line)
    meta_substrings = (
        "current_belief",
        "current belief",
        "final_rating",
        "final rating",
        "last_r_tag",
        "last r tag",
        "truth_decision",
        "truth decision",
        "explanation:",
        "reason:",
        "as per the instructions",
        "as per instructions",
        "as instructed",
        "as required",
        "required output",
        "exact format",
        "output must",
        "do not include",
        "do not add",
        "must not",
        "prompt",
        "instructions",
        "game master",
        "tag hygiene",
        "r-tag",
        "r tag",
        "i chose [r",
        "i choose [r",
        "i selected [r",
        "the tag i chose",
        "the r tag",
    )

    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue

        low = raw.lower()

        # Remove obvious fenced markers
        if low in {"```", "---"}:
            continue

        # Drop whole-line scaffolding/meta
        if any(low.startswith(p) for p in meta_prefixes):
            continue

        # Remove explicit 'Sentence N:' scaffolding
        if re.match(r"^sentence\s*\d+\s*:", low):
            continue

        # Drop if the line mentions protocol terms
        if any(sub in low for sub in meta_substrings):
            continue

        # Inline meta: if 'note:' appears mid-line, keep only the pre-note portion
        if re.search(r"\bnote\s*:", raw, flags=re.IGNORECASE):
            raw = re.split(r"\bnote\s*:", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if not raw:
                continue
            low = raw.lower()

        # If line still contains strong protocol keywords, drop it.
        if re.search(r"(output\s*format|exact\s*format|instructions?|prompt|required\s*output)", low):
            continue

        kept.append(raw)

    out = " ".join(kept).strip()

    # Strip any leading reasoning tag like '[R3]' or 'R3'
    out = re.sub(r"^\s*\[R[1-8]\]\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"^\s*R([1-8])\b\s*", "", out, flags=re.IGNORECASE)

    # Strip any lingering headers the model may echo
    out = re.sub(r"(?im)^\s*TWEET\s*:\s*", "", out).strip()
    out = re.sub(r"(?im)^\s*FINAL_RATING\s*:\s*[+-]?\d+\s*", "", out).strip()

    # Remove any remaining bracket tags anywhere (e.g., [R3], [A1])
    out = re.sub(r"\[[^\]]+\]", "", out).strip()

    # Normalize whitespace
    out = " ".join(out.split())
    return out

def _limit_to_n_sentences(text: str, max_sentences: int = 2) -> str:
    """Trim free text to at most N simple sentence-like chunks.

    Conservative helper for Step-2 salvage/fallback so we keep tweet text short
    without depending on a heavier sentence tokenizer.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s or max_sentences <= 0:
        return ""

    # Split on sentence-final punctuation followed by whitespace / end-of-string.
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p and p.strip()]

    if not parts:
        return s

    out = " ".join(parts[:max_sentences]).strip()

    # If the original had no terminal punctuation and the split did nothing,
    # keep the original text rather than truncating by token count here.
    return out or s


def _compact_step2_tweet_text(tweet_text: str, max_sentences: int = 2) -> str:
    """Compact a Step-2 tweet so the public handoff remains short and readable."""
    s = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not s:
        return ""
    s = _limit_to_n_sentences(s, max_sentences=max_sentences)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _polish_compact_step2_retry_tweet(tweet_text: str, rating_value: int, claim_txt: str = "") -> str:
    """Light cleanup for compact Step-2 rewrite outputs without phrase-level steering."""
    s = _compact_step2_tweet_text(tweet_text, max_sentences=2)
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if _generic_canned_reason_reason(s, field="tweet"):
        return _extract_protocol_body(_fallback_step2(expected_current_belief=int(rating_value), claim_txt=claim_txt), "TWEET") or s
    return _compact_step2_tweet_text(s, max_sentences=2)

def _first_natural_sentence_from_text(text: str) -> str:
    """Extract one short tweet-like sentence from malformed Step-2 output."""
    s = _strip_model_think_and_fence_artifacts(text)
    if not s:
        return ""
    s = _normalize_protocol_headers(s)
    s = _strip_leaked_prompt_headers(s)
    # Remove obvious protocol/meta labels inline.
    s = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", " ", s)
    s = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?|OUTPUT FORMAT|ALLOWED(?:[_ ]FINAL)?[_ ]RATING(?:[_ ]SET)?|CURRENT[_ ]RATING)\b\s*:?", " ", s)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""

    parts = re.split(r'(?<=[.!?])\s+', s)
    for part in parts:
        cand = str(part or '').strip(' -–—:;,.\t\n\r')
        if not cand:
            continue
        low = cand.lower()
        if _is_obviously_non_tweet_payload(cand):
            continue
        if any(tok in low for tok in ['output format', 'required output', 'instructions', 'prompt', 'game master', 'allowed final rating', 'current rating']):
            continue
        if len(re.findall(r'[A-Za-z]', cand)) < 12:
            continue
        cand = _limit_to_n_sentences(cand, max_sentences=1)
        cand = re.sub(r'\s+', ' ', cand).strip()
        if cand and not _is_obviously_non_tweet_payload(cand):
            return cand

    lines = [ln.strip(' -–—:;,.\t\n\r') for ln in re.split(r'[\r\n]+', s) if ln and ln.strip()]
    for cand in lines:
        low = cand.lower()
        if _is_obviously_non_tweet_payload(cand):
            continue
        if any(tok in low for tok in ['output format', 'required output', 'instructions', 'prompt', 'game master', 'allowed final rating', 'current rating']):
            continue
        if len(re.findall(r'[A-Za-z]', cand)) < 18:
            continue
        if len(cand.split()) < 5:
            continue
        cand = _limit_to_n_sentences(cand, max_sentences=1)
        cand = re.sub(r'\s+', ' ', cand).strip()
        if cand and not _is_obviously_non_tweet_payload(cand):
            return cand
    return ""


def _salvage_step2_tweet_like_text(raw_text: str, sanitized_text: str = "") -> str:
    """Best-effort recovery for near-miss Step-2 outputs before falling back."""
    candidates = []
    for src in (sanitized_text, raw_text):
        if not src:
            continue
        try:
            body = extract_tweet_text(src) or ""
        except Exception:
            body = ""
        if body:
            candidates.append(body)
        try:
            body2 = _heuristic_freeform_protocol_body(src, field="tweet") or ""
        except Exception:
            body2 = ""
        if body2:
            candidates.append(body2)
        stripped = strip_step2_meta_lines(src) if src else ""
        if stripped:
            candidates.append(stripped)
        natural = _first_natural_sentence_from_text(src) if src else ""
        if natural:
            candidates.append(natural)

    for cand in candidates:
        cand = _strip_leaked_prompt_headers(cand)
        cand = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", cand).strip()
        cand = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?)\s*:", "", cand).strip()
        cand = re.sub(r"\[[^\]]+\]", "", cand)
        cand = _compact_step2_tweet_text(cand, max_sentences=2)
        if cand and not _is_obviously_non_tweet_payload(cand) and _tweet_text_has_usable_step3_content(cand):
            return cand
    return ""

def sanitize_tweet_for_listener(step2_output: str) -> str:
    """Produce the only string that may cross the speaker->listener boundary.

    Fail closed: if we cannot confidently recover a natural tweet body, return "".
    The receiver already has an empty-input fallback in Step-3, which is safer than leaking
    protocol labels like FINAL_RATING into the listener prompt.
    """
    if step2_output is None:
        return ""
    s = str(step2_output).strip()
    if not s:
        return ""

    s_norm = sanitize_step2_output(s)
    tweet_body = extract_tweet_text(s_norm) if s_norm else extract_tweet_text(s)
    tweet_body = _strip_leaked_prompt_headers(tweet_body)

    tweet_body = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", tweet_body).strip()
    tweet_body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET)\s*:", "", tweet_body).strip()
    tweet_body = re.sub(r"\[[^\]]+\]", "", tweet_body)
    tweet_body = re.sub(r"\s*\n\s*", " ", tweet_body).strip()
    tweet_body = re.sub(r"\s+", " ", tweet_body).strip()
    tweet_body = _compact_step2_tweet_text(tweet_body, max_sentences=2)

    if (not tweet_body) or _is_obviously_non_tweet_payload(tweet_body) or (not _tweet_text_has_usable_step3_content(tweet_body)):
        tweet_body = _salvage_step2_tweet_like_text(raw_text=s, sanitized_text=s_norm)
    if _is_obviously_non_tweet_payload(tweet_body) or (not _tweet_text_has_usable_step3_content(tweet_body)):
        return ""
    return tweet_body


def _recover_step2_tweet_for_listener(speaker_agent=None, step2_output: str = "", sanitized_text: str = "") -> str:
    """
    Recover the public tweet body that should be handed from speaker to listener.
    
        The handoff is stricter than display parsing: it must avoid passing labels, prompt fragments, or
        empty strings to Step-3. This function tries sanitized output, raw output, cached same-step output,
        and lenient extraction in that order.
    """
    candidates = []

    def _add_candidate(value):
        """Add a possible tweet-handoff candidate if it is non-empty and new."""
        try:
            v = str(value or "").strip()
        except Exception:
            v = ""
        if v:
            candidates.append(v)

    # Current interaction artifacts first. Only then use the durable previous-cache
    # fallback. This avoids incorrectly preferring an older tweet when the latest
    # Step-2 output/raw artifacts contain a valid TWEET line.
    _add_candidate(sanitized_text)
    _add_candidate(step2_output)
    if speaker_agent is not None:
        _add_candidate(getattr(speaker_agent, "last_step2_final_output", ""))
        # Prefer the second raw attempt before the first because it is commonly the
        # native/compact rescue attempt and may contain the final usable TWEET line.
        _add_candidate(getattr(speaker_agent, "last_step2_raw_response_attempt_2", ""))
        _add_candidate(getattr(speaker_agent, "last_step2_raw_response_attempt_1", ""))
        # Last resort: previous durable speaker-side cache.
        _add_candidate(getattr(speaker_agent, "last_public_tweet", ""))
        _add_candidate(getattr(speaker_agent, "last_written_tweet", ""))
        _add_candidate(getattr(speaker_agent, "last_public_tweet_final_output", ""))
        _add_candidate(getattr(speaker_agent, "last_public_tweet_raw_output", ""))

    seen = set()
    for cand in candidates:
        key = re.sub(r"\s+", " ", cand).strip()
        if not key or key in seen:
            continue
        seen.add(key)

        # First treat the candidate as a possibly-protocol Step-2 output.
        try:
            public = sanitize_tweet_for_listener(cand)
        except Exception:
            public = ""
        if public and (not _is_obviously_non_tweet_payload(public)) and _tweet_text_has_usable_step3_content(public):
            return public

        # Then use the same-step handoff extractor. It accepts minor Step-2
        # protocol noise but still never invents new content.
        try:
            public = _extract_step2_tweet_for_handoff_lenient(cand)
        except Exception:
            public = ""
        if public and (not _is_obviously_non_tweet_payload(public)) and _tweet_text_has_usable_step3_content(public):
            return public

        # Then treat it as raw output and salvage a TWEET-like body.
        try:
            public = _salvage_step2_tweet_like_text(raw_text=cand, sanitized_text="")
        except Exception:
            public = ""
        if public and (not _is_obviously_non_tweet_payload(public)) and _tweet_text_has_usable_step3_content(public):
            return public

        # Last, treat it as already-natural tweet text.
        public = _strip_leaked_prompt_headers(cand)
        public = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", public).strip()
        public = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?)\s*:", "", public).strip()
        public = re.sub(r"\[[^\]]+\]", "", public)
        public = re.sub(r"\s+", " ", public).strip()
        public = _compact_step2_tweet_text(public, max_sentences=2)
        if public and (not _is_obviously_non_tweet_payload(public)) and _tweet_text_has_usable_step3_content(public):
            return public

    return ""


def _remember_agent_public_tweet(agent, public_tweet: str, *, final_output: str = "", raw_output: str = "", source: str = "") -> str:
    """Persist the last valid public tweet on the speaker agent for later handoff recovery."""
    try:
        # When caching immediately after Step-2, recover only from the current
        # Step-2 artifacts. Do not let an older cached tweet make the new Step-2
        # output look valid. The pre-Step-3 handoff guard may use the old cache as
        # a final safety net later.
        current_artifacts = "\n".join([str(final_output or ""), str(raw_output or "")])
        public = _recover_step2_tweet_for_listener(None, step2_output=current_artifacts, sanitized_text=public_tweet)
    except Exception:
        public = str(public_tweet or "").strip()
    if public and (not _is_obviously_non_tweet_payload(public)) and _tweet_text_has_usable_step3_content(public):
        try:
            agent.last_public_tweet = public
            agent.last_written_tweet = public
            agent.last_public_tweet_final_output = str(final_output or "")
            agent.last_public_tweet_raw_output = str(raw_output or final_output or public_tweet or "")
            agent.last_public_tweet_step = CURRENT_INTERACTION_STEP if CURRENT_INTERACTION_STEP is not None else ""
        except Exception:
            pass
        if source:
            try:
                _metric_inc(f"step2_public_tweet_cached::{source}")
            except Exception:
                pass
    return public if public else ""


def _clean_public_tweet_for_handoff(text: str) -> str:
    """Normalize a candidate tweet body for the speaker->listener boundary.

    This is intentionally less strict than Step-2 validation: by the time we call
    it, the text has already been produced by Step-2. The only question is whether
    it is safe and usable as listener-visible tweet text.
    """
    s = str(text or "").strip()
    if not s:
        return ""
    try:
        s = _strip_model_think_and_fence_artifacts(s)
    except Exception:
        pass
    try:
        s = _strip_leaked_prompt_headers(s)
    except Exception:
        pass
    s = re.sub(r"(?im)^\s*FINAL(?:_| )RATING\s*:?\s*[+-]?\d+\s*$", " ", s)
    s = re.sub(r"(?im)^\s*(?:TWEET|EXPLANATION|REASONING|ANCHOR|NOTES?)\s*:\s*", " ", s)
    s = re.sub(r"\[[^\]]+\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    try:
        s = _compact_step2_tweet_text(s, max_sentences=2)
    except Exception:
        pass
    s = re.sub(r"\s+", " ", s).strip()
    if (not s) or _is_obviously_non_tweet_payload(s) or (not _tweet_text_has_usable_step3_content(s)):
        return ""
    return s


def _extract_step2_tweet_for_handoff_lenient(text: str) -> str:
    """Extract a listener-safe tweet from raw same-step Step-2 output.

    The normal Step-2 validator can be intentionally conservative. For handoff,
    we should not lose a valid tweet merely because the model had minor protocol
    noise. This function never invents new content; it only extracts text already
    present in the current Step-2 output.
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(raw))
    except Exception:
        s = raw

    candidates = []

    # Standard protocol field, including multi-line body after TWEET:.
    for m in re.finditer(
        r"(?ims)^\s*TWEET\s*:\s*(.*?)(?=\n\s*(?:FINAL(?:_| )RATING|EXPLANATION|REASONING|ANCHOR|NOTES?)\s*:|\Z)",
        s,
    ):
        body = (m.group(1) or "").strip()
        if body:
            candidates.append(body)

    # Common near-miss: FINAL_RATING line followed by an unlabeled tweet sentence.
    m2 = re.search(r"(?ims)^\s*FINAL(?:_| )RATING\s*:\s*[+-]?\d+\s*\n+(.+?)\s*$", s)
    if m2:
        body = (m2.group(1) or "").strip()
        if body and not re.match(r"(?is)^\s*(?:TWEET|EXPLANATION|REASONING|ANCHOR|NOTES?)\s*:\s*$", body):
            candidates.append(body)

    # Existing project-specific fallback helpers.
    try:
        body = extract_tweet_text(s)
        if body:
            candidates.append(body)
    except Exception:
        pass
    try:
        body = _heuristic_freeform_protocol_body(s, field="tweet")
        if body:
            candidates.append(body)
    except Exception:
        pass
    try:
        body = _first_natural_sentence_from_text(s)
        if body:
            candidates.append(body)
    except Exception:
        pass

    seen = set()
    for cand in candidates:
        public = _clean_public_tweet_for_handoff(cand)
        key = re.sub(r"\s+", " ", public).strip().lower()
        if public and key not in seen:
            return public
        if key:
            seen.add(key)
    return ""


def _step2_handoff_artifact_diag(speaker_agent=None, *, step2_return: str = "", sanitized_text: str = "") -> dict:
    """
    Summarize where Step-2 handoff content disappeared.
    
        The returned lengths and flags help distinguish an LLM that produced no tweet from a parser or
        sanitizer that lost the tweet before Step-3.
    """
    out = {
        'sanitized_len': len(str(sanitized_text or '')),
        'step2_return_len': len(str(step2_return or '')),
        'step2_final_output_len': 0,
        'raw_attempt_1_len': 0,
        'raw_attempt_2_len': 0,
        'speaker_cache_len': 0,
    }
    if speaker_agent is not None:
        out['step2_final_output_len'] = len(str(getattr(speaker_agent, 'last_step2_final_output', '') or ''))
        out['raw_attempt_1_len'] = len(str(getattr(speaker_agent, 'last_step2_raw_response_attempt_1', '') or ''))
        out['raw_attempt_2_len'] = len(str(getattr(speaker_agent, 'last_step2_raw_response_attempt_2', '') or ''))
        out['speaker_cache_len'] = len(str(getattr(speaker_agent, 'last_public_tweet', '') or ''))
    return out


def _recover_same_step_step2_tweet_for_handoff(speaker_agent=None, *, step2_return: str = "", sanitized_text: str = "") -> tuple[str, str, dict]:
    """Authoritative same-step Step-2 -> Step-3 recovery.

    Returns (tweet, source, diagnostics). The source label tells whether the
    listener-visible tweet came from sanitized text, final Step-2 output, raw
    native attempt 1/2, or the durable speaker cache. Same-step raw/final outputs
    are checked before any older cache.
    """
    labeled = []

    def add(label, value):
        """
        Add one recovered handoff candidate if it is usable and not a duplicate.
        
            This local helper keeps the candidate recovery order explicit while avoiding repeated tweet bodies
            in the fallback search list.
        """
        v = str(value or "").strip()
        if v:
            labeled.append((label, v))

    add('sanitized_current', sanitized_text)
    add('step2_return', step2_return)
    if speaker_agent is not None:
        add('step2_handoff_tweet_attr', getattr(speaker_agent, 'last_step2_handoff_tweet', ''))
        add('step2_final_output', getattr(speaker_agent, 'last_step2_final_output', ''))
        # Prefer raw attempt 2 first; it may be a compact/native rescue.
        add('step2_raw_attempt_2', getattr(speaker_agent, 'last_step2_raw_response_attempt_2', ''))
        add('step2_raw_attempt_1', getattr(speaker_agent, 'last_step2_raw_response_attempt_1', ''))
        add('speaker_public_cache', getattr(speaker_agent, 'last_public_tweet', ''))
        add('speaker_last_written_tweet', getattr(speaker_agent, 'last_written_tweet', ''))
        add('speaker_public_final_output_cache', getattr(speaker_agent, 'last_public_tweet_final_output', ''))
        add('speaker_public_raw_output_cache', getattr(speaker_agent, 'last_public_tweet_raw_output', ''))

    diag = _step2_handoff_artifact_diag(speaker_agent, step2_return=step2_return, sanitized_text=sanitized_text)
    seen = set()
    for label, cand in labeled:
        key = re.sub(r"\s+", " ", cand).strip()
        if not key or key in seen:
            continue
        seen.add(key)

        public = ""
        try:
            public = sanitize_tweet_for_listener(cand)
        except Exception:
            public = ""
        if not public:
            try:
                public = _extract_step2_tweet_for_handoff_lenient(cand)
            except Exception:
                public = ""
        if not public:
            try:
                public = _salvage_step2_tweet_like_text(raw_text=cand, sanitized_text="")
            except Exception:
                public = ""
        public = _clean_public_tweet_for_handoff(public)
        if public:
            diag['recovery_source'] = label
            diag['handoff_tweet_len'] = len(public)
            try:
                _metric_inc(f'step3_handoff_recovery_source::{label}')
            except Exception:
                pass
            return public, label, diag

    diag['recovery_source'] = ''
    diag['handoff_tweet_len'] = 0
    return "", "", diag


def _format_step2_handoff_diag_note(diag: dict, *, speaker_name: str = "", listener_name: str = "", source: str = "") -> str:
    """Format Step-2 handoff diagnostics for logs and summaries."""
    parts = []
    if speaker_name:
        parts.append(f"speaker={speaker_name}")
    if listener_name:
        parts.append(f"listener={listener_name}")
    if source:
        parts.append(f"source={source}")
    for k in ['sanitized_len', 'step2_return_len', 'step2_final_output_len', 'raw_attempt_1_len', 'raw_attempt_2_len', 'speaker_cache_len', 'handoff_tweet_len']:
        if k in (diag or {}):
            parts.append(f"{k}={(diag or {}).get(k)}")
    return '; '.join(parts)


def sanitize_step2_output(raw: str) -> str:
    """
    Normalize Step-2 output while failing closed on missing or unsafe tweet text.
    
        The sanitizer accepts minor header variants but does not invent public tweet content from protocol
        residue. Its job is to produce a safe two-line FINAL_RATING/TWEET payload or leave enough evidence
        for the validator to trigger repair.
    """
    if raw is None:
        return ""
    s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(raw))
    if not s:
        return ""

    # Keep only the protocol tail if a rating line exists.
    m = re.search(r"(?im)^\s*FINAL(?:_| )RATING\s*:\s*([+-]?\d)\b", s)
    if not m:
        return ""

    rating = m.group(1)
    s = s[m.start():].lstrip()
    s = re.sub(r"(?im)^\s*FINAL\s+RATING\s*:", "FINAL_RATING:", s)

    missing_tweet_label = _normalize_step2_missing_tweet_label(s)
    if missing_tweet_label:
        return missing_tweet_label

    # Extract optional legacy ANCHOR and remove it from the body.
    m_anchor = re.search(r"(?im)^\s*ANCHOR\s*:\s*(.+?)\s*$", s)
    anchor_line = (m_anchor.group(1).strip() if m_anchor else "")
    if anchor_line:
        s = re.sub(r"(?im)^\s*ANCHOR\s*:\s*.+?$", "", s).strip()

    # Only trust an explicit TWEET field, otherwise use a heuristic salvage.
    m_tweet = re.search(
        r"(?ims)^\s*TWEET\s*:\s*(.+?)(?:\n\s*(?:ANCHOR|EXPLANATION|REASONING|NOTES?|FINAL(?:_| )RATING)\s*:|\Z)",
        s,
    )
    if m_tweet:
        tweet_body = (m_tweet.group(1) or "").strip()
    else:
        tweet_body = _extract_step2_multiline_tweet_body(s) or _heuristic_freeform_protocol_body(s, field="tweet").strip()

    tweet_body = _strip_leaked_prompt_headers(tweet_body)
    tweet_body = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", tweet_body).strip()
    tweet_body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET)\s*:", "", tweet_body).strip()
    tweet_body = re.sub(r"\[[^\]]+\]", "", tweet_body).strip()
    tweet_body = re.sub(r"\s+", " ", tweet_body).strip()

    if not _step2_tweet_has_valid_public_body(tweet_body):
        return ""

    # Optional legacy anchor folding, but only into a valid tweet body.
    if anchor_line:
        def _starts_with_anchor(x: str) -> bool:
            """Return whether text begins with an old-style rating anchor sentence."""
            return bool(re.match(r"(?is)^I\s+(?:accept|reject)\s+the\s+claim\s+as\s+written\s*:", x)) or bool(
                re.match(r"(?is)^I\s+am\s+unsure\s+about\s+the\s+claim\s+as\s+written\s*:", x)
            )
        if tweet_body and not _starts_with_anchor(tweet_body):
            a = anchor_line.strip()
            if a and not re.search(r"[.!?]\s*$", a):
                a += "."
            tweet_body = re.sub(r"\s+", " ", f"{a} {tweet_body}").strip()

    return f"FINAL_RATING: {rating}\nTWEET: {tweet_body}".strip()



_GROUNDED_TERM_FAMILIES = {
    "evidence": [r"\bevidence\b", r"\bproof\b"],
    "photos": [r"\bphoto\b", r"\bphotos\b", r"\bphotograph\b", r"\bphotographs\b", r"\bimage\b", r"\bimages\b"],
    "videos": [r"\bvideo\b", r"\bvideos\b", r"\bfootage\b"],
    "witnesses": [r"\bwitness\b", r"\bwitnesses\b"],
    "experts": [r"\bexpert\b", r"\bexperts\b", r"\bscientist\b", r"\bscientists\b", r"\bhistorian\b", r"\bhistorians\b"],
    "studies": [r"\bstudy\b", r"\bstudies\b", r"\bresearch\b"],
    "articles": [r"\barticle\b", r"\barticles\b"],
    "websites": [r"\bwebsite\b", r"\bwebsites\b", r"\bsite\b", r"\bsites\b", r"\bonline\b", r"\binternet\b"],
    "documentaries": [r"\bdocumentary\b", r"\bdocumentaries\b"],
    "records": [r"\brecord\b", r"\brecords\b"],
    "documents": [r"\bdocument\b", r"\bdocuments\b", r"\bfile\b", r"\bfiles\b", r"\bleak\b", r"\bleaks\b", r"\btranscript\b", r"\btranscripts\b"],
    "archives": [r"\barchive\b", r"\barchives\b"],
    "sources": [r"\bsource\b", r"\bsources\b"],
    "reports": [r"\breport\b", r"\breports\b"],
    "verification": [r"\bverification\b", r"\bverified\b", r"\bverify\b", r"\bconfirm\b", r"\bconfirmed\b", r"\bconfirmation\b"],
    "announcements": [r"\bannouncement\b", r"\bannouncements\b"],
}

_FIRST_PERSON_LOOKUP_PATTERNS = [
    r"\bi checked\b",
    r"\bi looked up\b",
    r"\bi searched\b",
    r"\bi read\b",
    r"\bi watched\b",
    r"\bi saw\b",
    r"\bi verified\b",
    r"\bi reviewed\b",
    r"\bi found online\b",
    r"\bi found on the internet\b",
]

_SOURCE_ATTRIBUTION_PATTERNS = {
    "experts": [
        r"\bexperts say\b",
        r"\bexperts believe\b",
        r"\bexperts agree\b",
        r"\bscientists say\b",
        r"\bhistorians say\b",
    ],
    "studies": [
        r"\bstudies show\b",
        r"\bresearch shows\b",
        r"\bresearch proves\b",
        r"\bstudies prove\b",
    ],
    "sources": [
        r"\bmultiple sources\b",
        r"\bsources say\b",
        r"\bsources confirm\b",
        r"\bofficial sources\b",
        r"\breliable sources\b",
    ],
    "documents": [
        r"\bdocuments show\b",
        r"\brecords show\b",
        r"\barchives show\b",
        r"\bthe documents prove\b",
        r"\bleaked files\b",
    ],
    "reports": [
        r"\breports show\b",
        r"\bconfirmed reports\b",
        r"\bofficial reports\b",
    ],
    "verification": [
        r"\bverified by\b",
        r"\bhas been verified\b",
        r"\bwas verified\b",
        r"\bconfirmed by\b",
        r"\bproven by\b",
    ],
}

_STRICT_CLOSED_UNGROUNDED_TERM_PATTERNS = {
    "experts": [r"\bexpert\b", r"\bexperts\b", r"\bscientist\b", r"\bscientists\b", r"\bhistorian\b", r"\bhistorians\b"],
    "studies": [r"\bstudy\b", r"\bstudies\b", r"\bresearch\b"],
    "sources": [r"\bsource\b", r"\bsources\b"],
    "documents": [r"\bdocument\b", r"\bdocuments\b", r"\brecord\b", r"\brecords\b", r"\barchive\b", r"\barchives\b"],
    "verification": [r"\bverification\b", r"\bverified\b", r"\bverify\b", r"\bconfirmed\b", r"\bconfirm\b"],
    "announcements": [r"\bannouncement\b", r"\bannouncements\b"],
}

_STRICT_CLOSED_LOCAL_MISSING_PROOF_PATTERNS = [
    r"\bwithout (?:direct|solid|clear|concrete|tangible|verifiable) proof\b",
    r"\bwithout (?:direct|solid|clear|concrete|tangible|verifiable) evidence\b",
    r"\bno (?:direct|solid|clear|concrete|tangible|verifiable) proof\b",
    r"\bno (?:direct|solid|clear|concrete|tangible|verifiable) evidence\b",
    r"\bnot enough (?:direct|solid|clear|concrete|tangible|verifiable) proof\b",
    r"\bnot enough (?:direct|solid|clear|concrete|tangible|verifiable) evidence\b",
    r"\black of (?:direct|solid|clear|concrete|tangible|verifiable) proof\b",
    r"\black of (?:direct|solid|clear|concrete|tangible|verifiable) evidence\b",
    r"\bmissing (?:direct|solid|clear|concrete|tangible|verifiable) proof\b",
    r"\bmissing (?:direct|solid|clear|concrete|tangible|verifiable) evidence\b",
    r"\bhard to (?:verify|confirm)\b",
    r"\bdifficult to (?:verify|confirm)\b",
    r"\bverifiable details?\b",
    r"\bconcrete details?\b",
    r"\btangible proof\b",
]




_STRICT_CLOSED_OUTSIDE_STYLE_PATTERNS = [
    r"\bwhat (?:we|people) know about\b",
    r"\bwhat (?:we(?:'re| are)?|people are) capable of\b",
    r"\bcurrent (?:technical|technological) (?:capabilities|limitations)\b",
    r"\bexperts? (?:say|believe|agree|claim)\b",
    r"\bstudies? (?:show|suggest|prove)\b",
    r"\bresearch (?:shows|suggests|proves)\b",
    r"\b(?:credible|official|reliable) sources?\b",
    r"\bmultiple sources?\b",
    r"\bmedia coverage\b",
    r"\ball over the news\b",
    r"\bhistory shows\b",
    r"\bscience says\b",
    r"\bpublic (?:knowledge|record)\b",
    r"\barchives?\b",
    r"\brecords?\b",
    r"\bdocuments?\b",
    r"\bpublic reports?\b",
    r"\bverification\b",
    r"\bverified\b",
    r"\bcross[- ]checks?\b",
    r"\bverifiable\b",
    r"\bconsensus\b",
    r"\bwidely known\b",
    r"\bknown information\b",
    r"\bbackground knowledge\b",
    r"\bshared publicly\b",
    r"\bpublicly shared\b",
    r"\bsecret from (?:the )?public\b",
    r"\bsecret from us\b",
    r"\bhard to sustain without contradictions\b",
    r"\bcoordinated explanation\b",
    r"\bmutually consistent\b",
    r"\bmutually consistent traces?\b",
    r"\bcompeting narratives?\b",
    r"\bdecisive discriminator\b",
    r"\blarge coordinated deception\b",
    r"\bpublicly available\b",
    r"\bpublicly available information\b",
    r"\blarge,? complex deceptions? are hard to sustain\b",
    r"\bleave(?:s|ing)? contradictions\b",
    r"\bwithout leaving contradictions\b",
    r"\bwhat humans can achieve\b",
    r"\bhumans? can achieve\b",
    r"\bwith (?:today'?s|modern|current) technology\b",
    r"\busing (?:today'?s|modern|current) technology\b",
    r"\btechnology (?:we|humans) have today\b",
    r"\badvanced technology\b",
    r"\btechnological advancements?\b",
    r"\badvancements? and resources\b",
    r"\bspace programs?\b",
    r"\binternational attention\b",
    r"\bhumans? exploring space\b",
    r"\bexploring (?:outer )?space\b",
    r"\bwithout being detected\b",
    r"\bwithout detection\b",
    r"\bstrongest verifiable traces?\b",
    r"\bofficial confirmation\b",
    r"\bofficially confirmed\b",
    r"\bconfirmed by (?:officials?|authorities|the government)\b",
    r"\btechnological feat\b",
    r"\bwhat (?:our|a|the) (?:country|nation|government) can achieve\b",
    r"\bcapable of amazing things\b",
]


def _grounded_term_families_in_text(text: str) -> set[str]:
    """Return the grounded evidence/source families explicitly present in text."""
    out = set()
    s = str(text or "")
    if not s:
        return out
    for family, patterns in _GROUNDED_TERM_FAMILIES.items():
        for pat in patterns:
            if re.search(pat, s, flags=re.I):
                out.add(family)
                break
    return out


def _normalized_grounding_text(text: str) -> str:
    """Return normalized grounding text for lexical groundedness checks."""
    s = str(text or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _matched_text_is_explicitly_grounded(match_text: str, grounding_text: str) -> bool:
    """Check whether a generated phrase is supported by shown grounding text."""
    mt = _normalized_grounding_text(match_text)
    gt = _normalized_grounding_text(grounding_text)
    return bool(mt and gt and mt in gt)


def _step2_possible_unsupported_v130_topic_claim(tweet_text: str, claim_text: str = "", prompt_text: str = "", extra_grounding_text: str = "") -> list[str]:
    """Warning-only diagnostic for strict-closed v130 Step-2 tweets.

    It flags strong technical/philosophical topic claims only when the same
    concept family is absent from the shown tweet/material. It must never force a
    retry or fallback by itself.
    """
    tweet = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not tweet:
        return []
    prompt = str(prompt_text or "")
    grounding = "\n".join([
        str(extra_grounding_text or ""),
        str(_extract_external_grounding_text(prompt) or ""),
        str(_extract_step2_shown_tweet_from_prompt(prompt) or ""),
    ])
    if not _v130_context_is_active(tweet, claim_text, prompt, grounding):
        return []

    # Only flag the narrower families that are easy to invent in strict closed
    # world. Common, already-corpus-grounded families such as lack of testability
    # are handled by the normal unsupported-reference checks.
    watch_families = {"consciousness_code", "ordinary_physics_math", "origin_question"}
    tweet_families = _v130_anti_simulation_families(tweet) & watch_families
    if not tweet_families:
        return []
    grounded_families = _v130_anti_simulation_families(grounding)
    return sorted(tweet_families - grounded_families)


_PM1_STRONG_WORDS = re.compile(r"\b(?:strongly|definitely|absolutely|certainly|undeniabl[ey]|unquestionabl[ey]|impossible|proven|guaranteed|no doubt|without doubt|beyond doubt|completely|totally|100%|clearly (?:true|false))\b", re.I)


def _step2_pm1_overstrong_wording(tweet_text: str, rating) -> list:
    if not _content_enforcement_enabled():
        return []
    """Warning-only: a mild (+/-1) tweet that uses strong/absolute wording.

    Returns the strong tokens found. It never triggers a retry or fallback; the
    caller only logs it, so it does not alter the tweet or the dynamics.
    """
    try:
        if abs(int(rating)) != 1:
            return []
    except Exception:
        return []
    found = _PM1_STRONG_WORDS.findall(str(tweet_text or ""))
    return sorted(set(str(x).strip().lower() for x in found)) if found else []


def _unsupported_external_reference_flags(tweet_text: str, world_mode: str, fact_pack_text: str = "", extra_grounding_text: str = "") -> tuple[bool, list[str]]:
    """Detect unsupported external-reference language in Step-2 tweets.

    Soft-policy version:
      - In closed-world modes, still ban first-person lookup claims and explicit source-attribution claims.
      - Tolerate unavoidable topic/domain language when it is not part of an authority-style appeal.
      - Treat external-language hits as warnings/contamination only; do not turn them into a special rewrite path.
      - In open-world modes, stay permissive.
    """
    s = str(tweet_text or "")
    if not s.strip():
        return False, []

    world_norm = _normalize_world_mode(world_mode)
    grounding_text = (str(fact_pack_text or '') + "\n" + str(extra_grounding_text or '')).strip()
    grounded = _grounded_term_families_in_text(grounding_text)
    is_open_world = world_norm in {"open", "open_no_rag", "open_rag", "true_open"}
    is_closed_strict = world_norm in {"closed_strict", "closed_strict_rag"}
    flags = []

    if is_open_world:
        return False, []

    for pat in _FIRST_PERSON_LOOKUP_PATTERNS:
        if re.search(pat, s, flags=re.I):
            flags.append(f"lookup:{pat}")

    for family, patterns in _SOURCE_ATTRIBUTION_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, s, flags=re.I)
            if m:
                matched_text = m.group(0)
                if (family not in grounded
                        and not _matched_text_is_explicitly_grounded(matched_text, grounding_text)
                        and not _grounding_supports_external_family(family, grounding_text, matched_text=matched_text)
                        and not _persona_trust_allows_external_family(extra_grounding_text, family)):
                    flags.append(f"attribution:{family}")
                break

    if is_closed_strict:
        strong_outside_style_patterns = [
            (r"\bexperts? (?:say|believe|agree|claim)\b", "experts"),
            (r"\bstudies? (?:show|suggest|prove)\b", "studies"),
            (r"\bresearch (?:shows|suggests|proves)\b", "studies"),
            (r"\b(?:credible|official|reliable) sources?\b", "sources"),
            (r"\bmultiple sources?\b", "sources"),
            (r"\bmedia coverage\b", None),
            (r"\ball over the news\b", None),
            (r"\bhistory shows\b", None),
            (r"\bscience says\b", None),
            (r"\bconsensus\b", None),
            (r"\bwidely known\b", None),
            (r"\bknown information\b", None),
            (r"\bbackground knowledge\b", None),
            (r"\bshared publicly\b", None),
            (r"\bpublicly shared\b", None),
            (r"\bpublicly available(?: information)?\b", None),
            (r"\bofficial confirmation\b", "verification"),
            (r"\bofficially confirmed\b", "verification"),
            (r"\bconfirmed by (?:officials?|authorities|the government)\b", "verification"),
        ]
        for pat, family in strong_outside_style_patterns:
            m = re.search(pat, s, flags=re.I)
            if m:
                matched_text = m.group(0)
                if family and family in grounded:
                    continue
                if _matched_text_is_explicitly_grounded(matched_text, grounding_text):
                    continue
                if family and _grounding_supports_external_family(family, grounding_text, matched_text=matched_text):
                    continue
                flags.append(f"outside_style:{pat}")

    deduped = []
    seen = set()
    for flg in flags:
        if flg not in seen:
            deduped.append(flg)
            seen.add(flg)
    return bool(deduped), deduped


def _record_step2_success_metrics(source: str, contaminated: bool = False):
    """Increment Step-2 success metrics after a valid tweet is accepted."""
    src = str(source or "main").strip().lower()
    src = "rescue" if src.startswith("rescue") else "main"
    key = f"step2_{src}_success_{'contaminated' if contaminated else 'clean'}"
    _metric_inc(key)


def _step3_has_strict_closed_contamination(warning_list) -> bool:
    """Detect Step-3 explanations that introduce forbidden outside knowledge."""
    for w in (warning_list or []):
        ws = str(w or "").strip().lower()
        if ws.startswith("strict_closed_"):
            return True
    return False


def _record_step3_success_metrics(source: str = "main", contaminated: bool = False):
    """Increment Step-3 success metrics after a valid belief update is accepted."""
    src = str(source or "main").strip().lower()
    if src not in {"main", "rescue"}:
        src = "main"
    _metric_inc(f"step3_{src}_success_{'contaminated' if contaminated else 'clean'}")



def _extract_explanation_text(step3_text: str) -> str:
    """Extract the explanation block from Step-3 output.

    Accepts:
      - EXPLANATION: <text>
      - EXPLANATION : <text>
      - Multi-line continuation after the EXPLANATION header (best-effort)
    """
    if step3_text is None:
        return ""
    s = str(step3_text).strip()
    if not s:
        return ""

    lines = s.splitlines()
    out_parts = []
    started = False

    for ln in lines:
        if not started:
            m = re.match(r"(?im)^\s*EXPLANATION\s*:\s*(.*)\s*$", ln)
            if m:
                started = True
                tail = (m.group(1) or "").strip()
                if tail:
                    out_parts.append(tail)
                continue
        else:
            # Stop if we see another protocol header that indicates contamination.
            if re.match(r"(?im)^\s*FINAL(?:_| )RATING\s*:", ln):
                break
            if re.match(r"(?im)^\s*(TWEET|ANCHOR|TRUTH[ _]?DECISION|REASON)\s*:", ln):
                break
            if ln.strip():
                out_parts.append(ln.strip())

    return " ".join(out_parts).strip()





def _extract_tweet_from_step3_prompt(prompt_text: str) -> str:
    """Best-effort extraction of the Step-2 tweet embedded inside a Step-3 prompt."""
    if not prompt_text:
        return ""
    s = str(prompt_text)
    m = re.search(r"(?ims)^\s*TWEET\s*:\s*(.+?)(?:\n\s*[A-Z][A-Z0-9_ ()-]*\s*:|\Z)", s)
    if not m:
        return ""
    tweet = re.sub(r"\s+", " ", (m.group(1) or "")).strip()
    tweet = _strip_leaked_prompt_headers(tweet)
    return tweet


def _split_simple_sentences(text: str):
    """Split text into simple sentence-like units for explanation checks."""
    if not text:
        return []
    s = re.sub(r"\s+", " ", str(text)).strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", s) if p.strip()]
    return parts


def _strip_tweet_reason_noise(text: str) -> str:
    """Remove filler and protocol noise before comparing tweet reasons."""
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#\w+", " ", s)
    s = re.sub(r"[\'\"“”‘’]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,!?:;-")


def _tweet_reason_text(tweet_text: str) -> str:
    """Extract the short reason portion of a Step-2 tweet, if any."""
    if not tweet_text:
        return ""
    parts = _split_simple_sentences(tweet_text)
    if len(parts) >= 2:
        return parts[1].strip()

    # Single-sentence fallback: remove the standard opener if present.
    s = parts[0] if parts else str(tweet_text).strip()
    opener_patterns = [
        r"(?i)^I strongly disagree with the claim as written\.\s*",
        r"(?i)^I somewhat disagree with the claim as written\.\s*",
        r"(?i)^I'm unsure about the claim as written\.\s*",
        r"(?i)^I am unsure about the claim as written\.\s*",
        r"(?i)^I somewhat agree with the claim as written\.\s*",
        r"(?i)^I strongly agree with the claim as written\.\s*",
    ]
    out = s
    for pat in opener_patterns:
        out = re.sub(pat, "", out).strip()
    return out


def detected_stance_from_tweet(tweet_text: str) -> str | None:
    """Deterministically infer tweet stance, but conservatively.

    Return support/oppose only when the signal is clear enough.
    If the wording is mixed or ambiguous, return None so the tweet guard
    does not create a false negative.
    """
    if not tweet_text:
        return None

    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return None

    opener_map = [
        (r"^i strongly disagree with the claim as written\.", "oppose"),
        (r"^i somewhat disagree with the claim as written\.", "oppose"),
        (r"^i strongly agree with the claim as written\.", "support"),
        (r"^i somewhat agree with the claim as written\.", "support"),
        (r"^i(?: am|[' ]m)? unsure about the claim as written\.", "uncertain"),
    ]
    for pat, label in opener_map:
        if re.search(pat, s, flags=re.I):
            return label

    if re.search(r"\b(mixed|unsure|uncertain|on the fence|not sure|hard to tell|i'm not sure|i am not sure)\b", s, flags=re.I):
        return "uncertain"

    rebuttal_support_patterns = [
        r"\bnot as dangerous as claimed\b",
        r"\bweren(?:'|’)t as dangerous as claimed\b",
        r"\bwasn(?:'|’)t as dangerous as claimed\b",
        r"\bdoes(?: not|n't) prove\b",
        r"\bdoes(?: not|n't) show\b",
        r"\bdoes(?: not|n't) mean\b",
        r"\bcan be explained by\b",
        r"\bis explained by\b",
        r"\bexplained by\b",
        r"\bweakens? the objection\b",
        r"\bdoes(?: not|n't) rule out\b",
        r"\bstill (?:looks|seems) feasible\b",
        r"\b(?:so|therefore|thus) .*feasib(?:le|ility)\b",
    ]

    strong_oppose_patterns = [
        r"\bi strongly disagree\b",
        r"\bi somewhat disagree\b",
        r"\bi (?:do not|don't) believe\b",
        r"\bi (?:do not|don't) buy\b",
        r"\bthere(?: is|'s)? no way\b",
        r"\bno way (?:that|this|it)\b",
        r"\bcan't be true\b",
        r"\bcannot be true\b",
        r"\bit(?:'s| is) impossible\b",
        r"\bdid(?: not|n't) happen\b",
        r"\btoo good to be true\b",
        r"\bdoes(?: not|n't) add up\b",
        r"\bno credible evidence\b",
        r"\bno solid evidence\b",
        r"\bfake\b",
        r"\bhoax\b",
    ]

    soft_oppose_patterns = [
        r"\bi doubt\b",
        r"\bi highly doubt\b",
        r"\bi(?: am|'m)? skeptical\b",
        r"\bi(?: am|'m)? suspicious\b",
        r"\bnot believable\b",
        r"\bnot convincing\b",
        r"\bimplausible\b",
        r"\bunconvincing\b",
    ]

    strong_support_patterns = [
        r"\bi strongly agree\b",
        r"\bi somewhat agree\b",
        r"\bi believe (?:the claim|it)\b",
        r"\bi think (?:the claim|it) is true\b",
        r"\bi think it happened\b",
        r"\bit happened\b",
        r"\bdid happen\b",
        r"\bi(?: am|'m)? convinced\b",
        r"\bsupport(?:s)? the claim\b",
        r"\blikely happened\b",
        r"\b(?:therefore|thus|so) the (?:claim|landing|mission) (?:was )?feasib(?:le|ility)\b",
        r"\bshows? the (?:claim|landing|mission) was feasib(?:le|ility)\b",
    ]

    soft_support_patterns = [
        r"\bthe claim is plausible\b",
        r"\bthe claim seems plausible\b",
        r"\bthe tweet makes a convincing case\b",
    ]

    strong_oppose = sum(bool(re.search(p, s, flags=re.I)) for p in strong_oppose_patterns)
    soft_oppose = sum(bool(re.search(p, s, flags=re.I)) for p in soft_oppose_patterns)
    strong_support = sum(bool(re.search(p, s, flags=re.I)) for p in strong_support_patterns)
    soft_support = sum(bool(re.search(p, s, flags=re.I)) for p in soft_support_patterns)

    rebuttal_support = sum(bool(re.search(p, s, flags=re.I)) for p in rebuttal_support_patterns)

    if rebuttal_support > 0 and strong_oppose == 0:
        return "support"
    if strong_oppose > 0 and strong_support == 0 and rebuttal_support == 0:
        return "oppose"
    if strong_support > 0 and strong_oppose == 0:
        return "support"

    if soft_oppose > 0 and strong_support == 0 and soft_support == 0:
        return "oppose"
    if soft_support > 0 and strong_oppose == 0 and soft_oppose == 0:
        return "support"

    return None


def no_move_if_weak_reason(tweet_text: str) -> bool:
    """Return True only for truly empty / opener-only / ultra-thin tweets.

    This should catch obvious non-reasons, but it should NOT freeze every mildly weak tweet.
    """
    if not tweet_text:
        return True

    reason = _tweet_reason_text(tweet_text)
    if not reason:
        return True

    reason_clean = _strip_tweet_reason_noise(reason)
    if not reason_clean:
        return True

    raw_reason = str(reason).strip()
    if re.fullmatch(r"(?:#\w+\s*)+", raw_reason):
        return True

    if not re.search(r"[A-Za-z]", reason_clean):
        return True

    words = re.findall(r"[A-Za-z']+", reason_clean)
    word_count = len(words)
    if word_count <= 2:
        return True

    low = reason_clean.lower().strip(" .,!?:;#")

    # Ultra-generic affect / slogan fragments with no real reason content.
    very_weak_exact = {
        "just because",
        "who knows",
        "maybe maybe not",
        "it feels off",
        "it sounds off",
        "it seems off",
        "not sure",
        "hard to tell",
        "just vibes",
        "good point",
        "bad point",
    }
    if low in very_weak_exact:
        return True

    # Generic one-sided complaints or slogan-like objections without a concrete mechanism
    # should not be strong enough on their own to move a neutral listener.
    very_weak_patterns = [
        r"\bno(?:t)? (?:real|clear|concrete) evidence\b",
        r"\bno proof\b",
        r"\bnot enough (?:evidence|proof)\b",
        r"\bneed more (?:evidence|proof)\b",
        r"\bstill (?:do not|don't) see (?:any )?(?:real |clear |concrete )?(?:evidence|proof)\b",
        r"\black of (?:real |clear |concrete )?(?:evidence|proof)\b",
        r"\bmissing (?:real |clear |concrete )?(?:evidence|proof)\b",
        r"\b(?:no|not enough|lack of|missing) (?:real |clear |concrete )?(?:evidence|proof) (?:in|from) this tweet\b",
        r"\b(?:from|in) this tweet (?:alone )?(?:there is|there's) no(?:t)? (?:real |clear |concrete )?(?:evidence|proof)\b",
        r"\bnot enough (?:from|in) (?:this|that) (?:alone|tweet|post)\b",
        r"\b(?:from|on) (?:this|that) (?:alone|tweet|post)\b",
        r"\b(?:by itself|on its own|from this alone|from that alone)\b",
        r"\bneed more than (?:this|that)\b",
        r"\btoo good to be true\b",
        r"\bdoes(?: not|n't) add up\b",
        r"\bwe(?:'d| would) have more evidence by now\b",
        r"\bgrainy (?:photos|footage|images)\b",
        r"\bjust do not buy it\b",
        r"\bjust don't buy it\b",
    ]
    if any(re.search(p, low, flags=re.I) for p in very_weak_patterns):
        return True

    # Short bare noun phrases are usually too thin unless they contain a real clause marker.
    has_clause_marker = bool(re.search(r"\b(because|since|so|but|which|that|if|while|although|though)\b", low, flags=re.I))
    has_claim_eval = bool(re.search(r"\b(plausible|implausible|convincing|unconvincing|credible|incredible|consistent|inconsistent|possible|impossible)\b", low, flags=re.I))
    if word_count <= 3 and not has_clause_marker and not has_claim_eval:
        return True

    return False


# --- Step-3 topic/semantic helpers used by the top-level guard ---
_STEP3_TOPIC_HELPERS_PATCH = True

def _step3_explanation_on_topic(expl_text: str, tweet_text: str = "") -> bool:
    """Check whether a Step-3 explanation is grounded in the tweet, claim, or shown material."""
    expl = re.sub(r"\s+", " ", str(expl_text or "")).strip()
    tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not expl:
        return False
    expl_toks = set(_fact_match_tokens(expl))
    tw_toks = set(_fact_match_tokens(tw))
    shared_digits = {tok for tok in (expl_toks & tw_toks) if tok.isdigit()}
    shared_non_generic = {tok for tok in (expl_toks & tw_toks) if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
    if shared_digits or len(shared_non_generic) >= 1:
        return True
    motifs = [
        "retroreflector", "retroreflectors", "lunar", "samples", "stars", "flag",
        "shadows", "shadow", "tapes", "sstv", "tracks", "imagery", "telemetry",
        "surveyor", "orbiter", "van", "allen"
    ]
    expl_low = expl.lower()
    tw_low = tw.lower()
    return any(m in expl_low and m in tw_low for m in motifs)

def _tweet_contains_hedge_language(tweet_text: str, explanation_text: str = "") -> bool:
    """Detect hedge / softening cues that make a same-side tweet weaker than an extreme prior."""
    tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not tw and not expl:
        return False

    tweet_patterns = [
        r"\bi lean toward\b",
        r"\bi lean against\b",
        r"\bhard to dismiss\b",
        r"\bhard to rule out\b",
        r"\bworth exploring\b",
        r"\bpossibilit(?:y|ies)\b",
        r"\bmaybe\b",
        r"\bmight\b",
        r"\bperhaps\b",
        r"\bprobably\b",
        r"\bit seems\b",
        r"\bseems\b",
        r"\bappears\b",
        r"\bthough\b",
        r"\bhowever\b",
        r"\bbut\b",
        r"\byet\b",
        r"\bstill\b",
        r"\bspeculative\b",
        r"\bremains speculative\b",
        r"\bwithout (?:more |clear |concrete |stronger )?evidence\b",
        r"\bnot entirely sure\b",
        r"\bnot fully sure\b",
        r"\bnot certain\b",
    ]
    expl_patterns = [
        r"\bslightly reduc(?:e|ing) skepticism\b",
        r"\bslightly less skeptical\b",
        r"\bsoften(?:s|ed|ing)?\b",
        r"\bmore cautious now\b",
        r"\bless certain now\b",
        r"\btempers? my confidence\b",
        r"\bslightly reduc(?:e|ing) confidence\b",
    ]
    return any(re.search(p, tw, flags=re.I) for p in tweet_patterns) or any(re.search(p, expl, flags=re.I) for p in expl_patterns)


def _same_side_weakening_soften_allowed(pre_belief: int, proposed_rating: int, tweet_stance: str | None, tweet_text: str = "", explanation_text: str = "") -> bool:
    """Allow one-step center-ward softening for same-side tweets when the move remains semantically aligned.

    This is intentionally broader than hedge-language only. A listener can reasonably soften from
    an extreme rating to the milder same-side rating when the tweet still points in the same
    direction and the public explanation stays aligned with that direction.
    """
    try:
        pre = int(pre_belief)
        prop = int(proposed_rating)
    except Exception:
        return False

    same_side_soften = False
    if pre < 0 and tweet_stance == "oppose" and prop > pre and prop <= 0 and abs(prop) < abs(pre):
        same_side_soften = True
    elif pre > 0 and tweet_stance == "support" and prop < pre and prop >= 0 and abs(prop) < abs(pre):
        same_side_soften = True

    if not same_side_soften:
        return False

    # Keep this limited to local one-step moderation rather than larger jumps.
    if abs(prop - pre) != 1:
        return False

    # Hedged / mixed same-side wording should always be allowed to soften.
    if _tweet_contains_hedge_language(tweet_text, explanation_text):
        return True

    # Also allow clean same-side softening when the explanation still clearly matches the tweet's
    # direction. This prevents false positives like -2 -> -1 on an oppose tweet whose explanation
    # is plainly oppose-aligned.
    label = _step3_expl_semantic_label(explanation_text)
    if tweet_stance in {"support", "oppose"} and label == tweet_stance:
        return True

    return False


def _tweet_has_explicit_balancing_clause(tweet_text: str = "") -> bool:
    """Return whether a tweet contains an explicit contrast or balancing clause."""
    tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    if not tw:
        return False
    has_contrast = bool(re.search(r"\b(?:but|however|though|yet|while|although)\b", tw))
    has_uncertainty = bool(re.search(r"\b(?:uncertain|unsure|not sure|speculative|still speculative|remains speculative|proof uncertain|evidence remains speculative|unfalsifiab(?:le|ility)|unresolved|mixed|both sides|either way|not enough to decide|not sure which way to lean)\b", tw))
    return has_contrast and has_uncertainty


def _step3_explanation_has_contrastive_weakening(explanation_text: str = "") -> bool:
    """Detect explanations that accept part of a point while weakening its conclusion."""
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not expl:
        return False
    return bool(re.search(r"\b(?:but|however|though|yet|still|remains?|not enough|does(?: not|n't)|lacks?|lack of|missing|without|too speculative|speculative|unclear|unresolved|limited|gap|limitation)\b", expl))


def _is_hard_step3_polarity_contradiction(final_rating: int, explanation_text: str, tweet_text: str = "", tweet_stance_local: str | None = None) -> bool:
    """Return whether rating direction and explanation polarity clearly contradict each other."""
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not expl:
        return False
    try:
        final_i = int(final_rating)
    except Exception:
        return False
    if final_i == 0:
        return False
    label = _step3_expl_semantic_label(expl)
    if label not in {"support", "oppose"}:
        return False
    if _step3_explanation_has_contrastive_weakening(expl):
        return False
    if _tweet_contains_hedge_language(tweet_text, expl):
        return False
    if final_i > 0:
        return label == "oppose"
    return label == "support"


def _apply_tweet_based_step3_guard(pre_belief: int, proposed_rating: int, tweet_stance: str | None, weak_reason: bool, allowed_ratings=None, explanation_text: str = "", tweet_text: str = "", bias_text: str | None = None):
    """Protocol-level Step-3 guard using tweet stance plus optional extra friction in strong-bias mode.

    The agent still decides the rating. This guard only blocks structurally inconsistent moves and,
    in strong-confirmation-bias mode, adds extra friction for incompatible moves away from the
    current side when the justification is weak, generic, or not clearly anchored to the tweet.

    In free_bounded update mode, away-from-speaker / away-from-tweet movement is an intentional
    experimental possibility. The allowed set already handles the numeric boundary, so this guard
    becomes observation-only and does not project the rating.
    """
    try:
        pre = int(pre_belief)
        prop = int(proposed_rating)
    except Exception:
        return int(pre_belief), "tweet_guard_parse_failure"

    if _is_free_bounded_update_mode():
        return prop, None

    if tweet_stance == "support" and prop < pre:
        if _same_side_weakening_soften_allowed(pre, prop, tweet_stance, tweet_text=tweet_text, explanation_text=explanation_text):
            return prop, None
        return pre, "detected_support_no_negative_move"
    if tweet_stance == "oppose" and prop > pre:
        if _same_side_weakening_soften_allowed(pre, prop, tweet_stance, tweet_text=tweet_text, explanation_text=explanation_text):
            return prop, None
        return pre, "detected_oppose_no_positive_move"
    if tweet_stance == "uncertain" and abs(prop) > abs(pre):
        label = _step3_expl_semantic_label(explanation_text)
        explicit_balance = _tweet_has_explicit_balancing_clause(tweet_text)
        if explicit_balance and label in {"support", "oppose"} and not _step3_explanation_has_contrastive_weakening(explanation_text):
            return pre, "detected_uncertain_no_polarization"

    strong_mode = _is_strong_confirmation_bias_active(bias_text)
    if not strong_mode or pre == 0 or tweet_stance not in {"support", "oppose"}:
        return prop, None

    incompatible = (pre < 0 and tweet_stance == "support") or (pre > 0 and tweet_stance == "oppose")
    if not incompatible or prop == pre:
        return prop, None

    moving_away_from_current_side = (pre < 0 and prop > pre) or (pre > 0 and prop < pre)
    if not moving_away_from_current_side:
        return prop, None

    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    on_topic = _step3_explanation_on_topic(expl, tweet_text) if expl else False
    semantic = _step3_expl_semantic_label(expl) if expl else None
    generic_gap = _step3_generic_gap_reason(expl, tweet_text=tweet_text, pre_value=pre, final_value=prop, tweet_stance_local=tweet_stance) if expl else None
    visible_meta = _visible_planning_meta_reason(expl, field="explanation") if expl else None
    canned_meta = _generic_canned_reason_reason(expl, field="explanation") if expl else None
    weak_or_generic = bool(weak_reason or generic_gap or visible_meta or canned_meta or not on_topic)

    if abs(pre) >= 2:
        if weak_or_generic or semantic not in {tweet_stance, "uncertain"}:
            return pre, "strong_bias_extreme_incompatible_hold"
    elif abs(pre) == 1:
        if weak_or_generic:
            return pre, "strong_bias_incompatible_hold"

    return prop, None


def _canonical_step3_explanation_from_tweet(tweet_stance: str | None, pre_belief: int, final_rating: int, weak_reason: bool = False) -> str:
    """Deterministic short anchor that stays consistent with tweet-local strict-closed wording."""
    try:
        pre = int(pre_belief)
    except Exception:
        pre = 0
    try:
        final = int(final_rating)
    except Exception:
        final = pre

    if tweet_stance == "support":
        if final > pre:
            return "The tweet gives a concrete supportive point that is strong enough for a small move toward the claim."
        if final == pre:
            if final < 0:
                return "The tweet raises a possibility, but it still rests on a speculative jump."
            return "The tweet adds a supportive point, but the support still stays too speculative to move further."
        return "The tweet raises a possibility, but the support still stays too speculative to overturn the current doubt."

    if tweet_stance == "oppose":
        if final < pre:
            return "The tweet gives a concrete challenge that is strong enough for a small move against the claim."
        if final == pre:
            if final > 0:
                return "The tweet raises a challenge, but the challenge still stays too limited to overturn the claim."
            return "The tweet raises a concrete gap, and that gap still supports a skeptical reading."
        return "The tweet raises a concrete gap, but it is not enough to overcome the remaining support for the claim."

    if tweet_stance == "uncertain":
        if abs(final) < abs(pre):
            return "The tweet is mixed enough to justify a small move toward a more uncertain rating."
        if final == pre:
            return "The tweet stays mixed, but it still does not clearly beat the best reason for a different rating."
        return "The tweet stays mixed, but it is not strong enough to support a clearer rating."

    return _step3_generic_anchor_for_rating(final, tweet_stance)


def _canonical_step3_explanation(tag: str, pre_belief: int, final_rating: int, explanation_text: str = "") -> str:
    """Build a short anchor guaranteed to agree with the projected/final rating."""
    try:
        final_i = int(final_rating)
    except Exception:
        final_i = int(pre_belief)
    if tag in {"semantic_stay_projection", "semantic_no_move_negative_projection", "semantic_no_move_positive_projection", "policy_stay", "weak_reason_no_move"}:
        return "The tweet has a point, but it is still not strong or specific enough to change the rating."
    if tag == "detected_support_no_negative_move":
        if final_i >= 0:
            return "The tweet raises a possibility for the claim, but that support is still not enough to overturn the current doubt."
        return "The tweet raises a possibility for the claim, but it is not enough to displace the current skeptical reading."
    if tag == "detected_oppose_no_positive_move":
        if final_i <= 0:
            return "The tweet raises a concrete gap, and that gap still supports a skeptical reading."
        return "The tweet raises a concrete gap, but that challenge still stays too limited to overturn the current support."
    if tag == "detected_uncertain_no_polarization":
        return "The tweet stays mixed, but it is not strong enough to support a clearer rating."
    if tag in {"strong_bias_incompatible_hold", "strong_bias_extreme_incompatible_hold"}:
        return "The tweet is not specific and strong enough to justify changing direction here."
    if tag == "semantic_support_projection":
        return "The tweet raises a possibility for the claim, but that support is still not enough to overturn the current doubt."
    if tag == "semantic_oppose_projection":
        return "The tweet raises a concrete gap, but that challenge still stays too limited to overturn the current support."
    if tag == "semantic_uncertain_projection":
        return "The tweet stays mixed, but it is not strong enough to support a clearer rating."
    if tag == "semantic_confidence_only_projection":
        return "Confidence alone is not enough to justify a bigger move."
    if tag == "singleton_forced_rating":
        return "Only one final rating is permitted here."
    if tag == "policy_project":
        return "The proposed rating was outside the allowed move, so the closest permitted rating fits best."
    if tag == "boundary_reasoning_contradiction":
        return "The tweet's direction does not support a stronger move here."
    if tag == "nochange_support_move_pressure":
        return "The tweet adds a supportive point, but that support is still not enough to justify a move."
    if tag == "nochange_oppose_move_pressure":
        return "The tweet raises a concrete challenge, but it is still not strong enough to justify a move."
    if tag == "nochange_uncertain_move_pressure":
        return "The tweet stays mixed, but it is still not strong enough to justify a move."
    if tag == "nochange_directional_move_pressure":
        return "The tweet has some directional pull, but it is still not strong enough to justify a move."
    if tag == "move_stay_wording_contradiction":
        return _step3_generic_anchor_for_rating(final_i, None)
    if tag == "nochange_directional_move_pressure":
        return "The tweet has some directional pull, but it does not clearly outweigh the strongest reason for the current rating."
    if tag == "move_stay_wording_contradiction":
        return _step3_generic_anchor_for_rating(final_i, None)

def _step3_force_rating_aligned_explanation(explanation_text: str, final_rating: int) -> str:
    """Rewrite a Step-3 explanation so it is consistent with the chosen final rating."""
    try:
        r = int(final_rating)
    except Exception:
        r = 0
    expl = _step3_anchor_clean_text(explanation_text, final_rating=r)
    if not expl:
        return _step3_generic_anchor_for_rating(r, None)
    label = _step3_expl_semantic_label(expl)
    mixed_markers = bool(re.search(r"(?i)\b(but|however|though|although|yet|still|unresolved|uncertain|uncertainty|mixed|not enough|not strong enough|question|questions|doubt|gap|limitation|despite)\b", expl))
    negative_softening_markers = bool(re.search(r"(?i)\b(mild skepticism|mildly skeptical|slightly doubtful|still doubtful|still skeptical|less confident|softens? my view|shift(?:s|ed)? .*mild skepticism|raises? some doubt|introduces? doubt)\b", expl))
    positive_softening_markers = bool(re.search(r"(?i)\b(still some support|still some evidence|some support remains|some evidence remains|less doubtful|softens? my skepticism|weakens? my previous disagreement|nudges? me less negative)\b", expl))
    contradicts = False
    if r > 0 and label == "oppose":
        contradicts = not (mixed_markers or positive_softening_markers)
    elif r < 0 and label == "support":
        contradicts = not (mixed_markers or negative_softening_markers)
    elif r == 0 and label in {"support", "oppose"}:
        contradicts = not mixed_markers
    if contradicts:
        return _step3_generic_anchor_for_rating(r, None)
    return expl

def _rewrite_step3_output_with_explanation(final_rating: int, explanation: str) -> str:
    """Replace only the explanation part of a Step-3 protocol output."""
    explanation = _step3_force_rating_aligned_explanation(explanation, int(final_rating))
    explanation = re.sub(r"\s+", " ", str(explanation or "")).strip()
    return f"FINAL_RATING: {int(final_rating)}\nEXPLANATION: {explanation}"


def _format_step3_output_preserve_explanation(final_rating: int, explanation: str) -> str:
    """Format a valid Step-3 explanation without semantic/state-language rewriting.

    Use this when the model already returned a parseable FINAL_RATING + EXPLANATION and
    we only want whitespace normalization / canonical 2-line formatting.
    """
    explanation = re.sub(r"\s+", " ", str(explanation or "")).strip()
    if not explanation:
        return _rewrite_step3_output_with_explanation(final_rating=int(final_rating), explanation="")
    return f"FINAL_RATING: {int(final_rating)}\nEXPLANATION: {explanation}"


def _step3_text_has_oppose_markers(text: str) -> bool:
    """Detect anti-claim markers in Step-3 explanation text."""
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not s:
        return False
    # "Challenges our assumptions about reality" is usually pro-simulation in
    # this topic, not an anti-claim challenge. Mask it before generic negative
    # marker detection so valid +1 explanations are not rewritten.
    s_for_oppose = re.sub(r"\bchalleng(?:e|es|ed|ing)\s+(?:our\s+)?assumptions?(?:\s+about\s+reality)?\b", " ", s, flags=re.I)
    generic_negative = bool(re.search(
        r"\b(?:lack|lacks|absence|loss|lost|missing|incomplete|unavailable|unresolved|unclear|uncertain|uncertainty|unreliable|reliability|question|questions|doubt|doubts|concern|concerns|risk|risks|inconsisten|not enough|insufficient|no direct|no independent|no credible|no clear|no agreed|no concrete|without (?:clear |concrete |direct )?(?:evidence|proof|test|framework)|circumstantial|skeptic|skeptical|skepticism|challenge|challenges|challenging|challenged|weakens?|undermines?|erodes?|casts? doubt|gap|gaps|harder to check|hard to check|harder to verify|hard to verify|harder to trust|hard to trust|harder to fully trust|hard to fully trust|does(?: not|n't) prove|do(?:es)? not confirm|alone (?:do(?:es)? not|doesn't)|fails? to show|fails? to establish)\b",
        s_for_oppose,
        flags=re.I,
    ))
    simulation_negative = bool(re.search(
        r"\b(?:speculative|speculation|untestable|not testable|testable framework|clear test|direct test|agreed direct test|unfalsifiab(?:le|ility)|unverifiable|verifiability|without evidence|without concrete evidence|no evidence|no direct evidence|not evidence[- ]based|ordinary physics|unverified assumptions?|uncertain assumptions?|future tech(?:nology)?|civilization survival|possibility alone|coherent without being likely|philosophical(?:ly)?|thought experiment)\b",
        s_for_oppose,
        flags=re.I,
    ))
    topic_negative = bool(re.search(
        r"\b(?:fake|hoax|staged|radiation|van allen|sstv|shadow|shadows|lighting)\b",
        s_for_oppose,
        flags=re.I,
    ))
    return bool(generic_negative or simulation_negative or topic_negative)


def _step3_text_has_support_markers(text: str) -> bool:
    """Detect pro-claim markers in Step-3 explanation text."""
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not s:
        return False
    # Avoid treating "lack of evidence" / "missing proof" as supportive merely
    # because the word evidence/proof appears.
    if re.search(r"\b(?:lack|lacks|absence|missing|no|without|insufficient|not enough|weak|limited)\b.{0,45}\b(?:evidence|proof|support|confirmation|test|framework)\b", s, flags=re.I):
        generic_positive = False
    else:
        generic_positive = bool(re.search(
            r"\b(?:supports?|supported|supporting|back(?:s|ed)? up|favors?|favours?|points? toward|counts? in favor|counts? in favour|reinforces?|strengthens?|credible support|concrete support|evidence for|proof of|confirm(?:s|ed|ing)?|validat(?:e|es|ed|ing)|corroborat(?:e|es|ed|ing)|verif(?:y|ies|ied)|document(?:s|ed|ation) supports?)\b",
            s,
            flags=re.I,
        ))
    # Topic-specific artifacts remain as optional examples, not the general logic.
    topic_positive = bool(re.search(
        r"\b(?:lunar samples?|moon rocks?|retroreflectors?|laser ranging|tracked|tracking|mission records?|apollo missions?|physical presence|surface tracks|carried astronauts|lunar surface)\b",
        s,
        flags=re.I,
    ))
    simulation_positive = bool(re.search(
        r"\b(?:advanced civilizations? (?:could|can|might|may|eventually) run|ancestor simulations?|detailed simulations?|simulated observers?|simulated minds? could be real observers?|real observers rather than merely fictional|simulation argument|many simulations?|countless simulations?|randomly located observer|substrate[- ]independent|computational process|mathematical rules|stable laws|generated reality|simulated world|lack of direct outside access|no obvious glitches|hide(?:s)? (?:its )?(?:nature|artificial structure)|plausible scenario|could be generated|compatible with the idea)\b",
        s,
        flags=re.I,
    ))
    return bool(generic_positive or topic_positive or simulation_positive)


def _step3_tweet_clause_direction(clause_text: str) -> str:
    """Classify a tweet clause as support, opposition, mixed, or unknown."""
    s = re.sub(r"\s+", " ", str(clause_text or "")).strip().lower()
    if not s:
        return "unknown"
    has_oppose = _step3_text_has_oppose_markers(s)
    has_support = _step3_text_has_support_markers(s)
    if has_oppose and not has_support:
        return "oppose"
    if has_support and not has_oppose:
        return "support"
    if has_oppose and has_support:
        if re.search(r"\b(despite|but|however|though|yet|challenge|undermine|cast(?:s)? doubt|risk|radiation|missing|lost|gap|unresolved)\b", s):
            return "oppose"
        return "support"
    return "unknown"


def _step3_trim_tweet_clause(clause_text: str, max_words: int = 18) -> str:
    """Trim a tweet clause to the most useful local evidence phrase."""
    s = re.sub(r"\s+", " ", str(clause_text or "")).strip()
    s = re.sub(r"(?i)^(i think|i lean(?: toward| towards)?|not sure if|i'm not sure if|i am not sure if|strongly disagree[—:-]?|strongly agree[—:-]?|i'm unsure if|i am unsure if)\s*", "", s).strip(" ,;:-")
    s = re.sub(r"(?i)^(the moon landing|moon landing|the moon landings|moon landings|us astronauts landed on the moon|us astronauts have landed on the moon)\s+(?:was|were|is|are)\s+", "", s).strip(" ,;:-")
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).strip(" ,;:-")
        dangling = {"the", "a", "an", "and", "or", "but", "if", "that", "because", "about", "of", "to", "in", "on", "for", "with", "from", "this", "these", "those", "into", "onto", "than", "via", "lunar"}
        parts = s.split()
        while parts and parts[-1].strip(" ,.;:-").lower() in dangling:
            parts = parts[:-1]
        s = " ".join(parts).strip(" ,;:-") or s
    return s


def _step3_extract_tweet_local_clauses(tweet_text: str) -> tuple[str, str]:
    """Extract tweet-local clauses used to build grounded Step-3 explanations."""
    s = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not s:
        return "", ""
    parts = re.split(r"(?i)\b(?:but|however|though|although|yet|despite)\b", s, maxsplit=1)
    if len(parts) == 2:
        return _step3_trim_tweet_clause(parts[0]), _step3_trim_tweet_clause(parts[1])
    return _step3_trim_tweet_clause(s), ""
def _step3_point_fragment(clause: str) -> str:
    """Return a compact phrase describing the tweet point being evaluated."""
    s = re.sub(r"\s+", " ", str(clause or "")).strip(" ,.;:-")
    if not s:
        return "one concrete point"

    # Deterministic repair explanations often splice tweet clauses into a
    # template. Strip trailing meta-clauses like "which strongly supports..."
    # so fallback text does not become ungrammatical ("which supports, but...").
    s = re.sub(r"\s+#\w+\b", "", s).strip(" ,.;:-")
    s = re.sub(
        r"(?i),?\s+(?:which|that)\s+(?:strongly\s+)?"
        r"(?:supports?|supporting|confirms?|confirming|proves?|proving|suggests?|suggesting|"
        r"points?\s+(?:to|toward|towards)|backs?\s+up|counts?\s+in\s+favou?r|"
        r"makes?\s+it\s+hard\s+to\s+doubt)\b.*$",
        "",
        s,
    ).strip(" ,.;:-")
    s = re.sub(
        r"(?i),?\s+(?:which|that)\s+(?:raises?|raising|casts?|casting|keeps?|keeping|"
        r"creates?|creating|introduces?|introducing|leaves?|leaving|makes?)\b.*$",
        "",
        s,
    ).strip(" ,.;:-")
    s = re.sub(r"(?i)\bso\s+it'?s\s+not\s+completely\s+impossible\b.*$", "", s).strip(" ,.;:-")

    if not s:
        return "one concrete point"
    return s[:1].lower() + s[1:] if len(s) > 1 else s.lower()


def _step3_point_sentence(clause: str) -> str:
    """Return a sentence-form description of the tweet point being evaluated."""
    return f"The tweet highlights {_step3_point_fragment(clause)}"


def _step3_support_point_sentence(clause: str) -> str:
    """Compact support-side phrase for deterministic Step-3 fallbacks."""
    frag = _step3_point_fragment(clause)
    low = frag.lower()
    if re.search(r"\bretroreflectors?\b", low, flags=re.I):
        return "Retroreflectors provide support"
    if re.search(r"\blaser[- ]ranging\b", low, flags=re.I):
        return "Laser-ranging evidence provides support"
    if re.search(r"\b(?:tracking|telemetry|mission records?)\b", low, flags=re.I):
        return "The tracking-record point provides support"
    if re.search(r"\b(?:human presence|lunar presence)\b", low, flags=re.I):
        return "The human-presence point provides support"
    if re.search(r"\b(?:advanced civilizations?|ancestor simulations?|detailed simulations?|countless simulations?|simulated observers?)\b", low, flags=re.I):
        return "The advanced-simulation possibility provides support"
    if re.search(r"\b(?:mathematical rules|stable laws|generated reality|no obvious glitches|hide its nature|outside access)\b", low, flags=re.I):
        return "The simulation-compatibility point provides support"
    return "That point provides support"


def _step3_challenge_point_sentence(clause: str) -> str:
    """Compact challenge-side phrase for deterministic Step-3 fallbacks."""
    frag = _step3_point_fragment(clause)
    low = frag.lower()
    if re.search(r"\b(?:missing|lost|loss)\b.{0,45}\b(?:tapes?|records?|materials?|documentation|footage)\b", low, flags=re.I) or re.search(r"\b(?:tapes?|records?|materials?|documentation|footage)\b.{0,45}\b(?:missing|lost|loss)\b", low, flags=re.I):
        return "Missing records raise a question"
    if re.search(r"\b(?:lighting|shadow|shadows|footage|visuals?|staged)\b", low, flags=re.I):
        return "The footage concern raises a question"
    if re.search(r"\b(?:radiation|van allen)\b", low, flags=re.I):
        return "The radiation concern raises a question"
    if re.search(r"\b(?:clear test|direct test|testable|untestable|unfalsifiab|speculative|speculation)\b", low, flags=re.I):
        return "The lack of a clear test keeps the claim speculative"
    if re.search(r"\b(?:future tech|future technology|civilization survival|unverified assumptions?|uncertain assumptions?|choose not to create|conscious simulated)\b", low, flags=re.I):
        return "Unverified assumptions about advanced civilizations weaken the claim"
    if re.search(r"\b(?:ordinary physics|mathematical laws|possibility alone|coherent without being likely|not evidence[- ]based|without (?:clear |concrete |direct )?evidence|no (?:clear |concrete |direct )?evidence)\b", low, flags=re.I):
        return "The evidence gap keeps the simulation claim weak"
    if re.search(r"\b(?:gap|gaps|unresolved|hard(?:er)? to verify|hard(?:er)? to trust|doubt|questions?)\b", low, flags=re.I):
        return "That challenge raises a question"
    return "That challenge raises a question"


def _step3_tweet_local_explanation(tweet_text: str, final_rating: int, pre_belief: int, tweet_stance: str | None = None) -> str:
    """Build a Step-3 fallback explanation anchored directly in the tweet."""
    try:
        final_i = int(final_rating)
    except Exception:
        final_i = 0
    try:
        pre_i = int(pre_belief)
    except Exception:
        pre_i = final_i
    left, right = _step3_extract_tweet_local_clauses(tweet_text)
    left_dir = _step3_tweet_clause_direction(left)
    right_dir = _step3_tweet_clause_direction(right)
    support_clause = left if left_dir == 'support' else (right if right_dir == 'support' else '')
    oppose_clause = left if left_dir == 'oppose' else (right if right_dir == 'oppose' else '')
    if final_i == 0:
        if support_clause and oppose_clause:
            return f"{_step3_point_sentence(support_clause)}, but {_step3_point_fragment(oppose_clause)} still keeps both sides in play."
        if support_clause:
            return f"{_step3_support_point_sentence(support_clause)}, but that support is still not enough to settle the claim."
        if oppose_clause:
            return f"{_step3_point_sentence(oppose_clause)}, but there is still enough pull the other way to stay mixed."
        return "The tweet has one real point, but an unresolved gap still keeps both sides in play."
    if final_i > pre_i:
        if support_clause:
            return f"{_step3_point_sentence(support_clause)}, which is enough for a small move toward the claim."
        return "The tweet gives a concrete supportive point that is enough for a small move toward the claim."
    if final_i < pre_i:
        if oppose_clause:
            return f"{_step3_point_sentence(oppose_clause)}, which is enough for a small move against the claim."
        return "The tweet gives a concrete challenge that is enough for a small move against the claim."
    # For same-rating or no-change cases on mixed tweets, anchor the explanation
    # on the side that actually matches the chosen FINAL_RATING. Previously this
    # could produce negative ratings justified only by a supportive retroreflector
    # clause, e.g. FINAL_RATING:-1 with EXPLANATION anchored only on confirmation.
    if final_i == pre_i == -2 and oppose_clause:
        return f"{_step3_point_sentence(oppose_clause)}, and that still supports a strongly skeptical reading."
    if final_i == pre_i == 2 and support_clause:
        return f"{_step3_point_sentence(support_clause)}, and that still supports a strongly favorable reading."
    if final_i < 0 and oppose_clause:
        return f"{_step3_point_sentence(oppose_clause)}, and that still supports a skeptical reading."
    if final_i > 0 and support_clause:
        return f"{_step3_point_sentence(support_clause)}, and that still supports the claim."
    if final_i > 0 and oppose_clause:
        return f"{_step3_point_sentence(oppose_clause)}, but the challenge still stays too limited to overturn the claim."
    if final_i < 0 and support_clause:
        return f"{_step3_support_point_sentence(support_clause)}, but that support is not enough to justify a larger shift here."
    return _step3_generic_anchor_for_rating(final_i, tweet_stance)


def _step3_illegal_rating_fallback_explanation(tweet_text: str, fallback_rating: int, attempted_rating: int | None, pre_belief: int | None = None, tweet_stance: str | None = None) -> str:
    """Tweet-local explanation for impossible Step-3 rating jumps.

    When the model proposes a FINAL_RATING outside ALLOWED_FINAL_RATING_SET, we
    keep the legal fallback rating but avoid a mismatched explanation that only
    repeats the attempted direction. The wording stays topic-general and uses
    only explicit clauses from the current tweet.
    """
    try:
        fb = int(fallback_rating)
    except Exception:
        fb = int(pre_belief) if pre_belief is not None else 0
    try:
        attempted = int(attempted_rating) if attempted_rating is not None else None
    except Exception:
        attempted = None

    left, right = _step3_extract_tweet_local_clauses(tweet_text)
    left_dir = _step3_tweet_clause_direction(left)
    right_dir = _step3_tweet_clause_direction(right)
    support_clause = left if left_dir == "support" else (right if right_dir == "support" else "")
    oppose_clause = left if left_dir == "oppose" else (right if right_dir == "oppose" else "")

    attempted_more_supportive = attempted is not None and attempted > fb
    attempted_more_skeptical = attempted is not None and attempted < fb

    if attempted_more_supportive:
        if support_clause:
            if fb < 0:
                return f"{_step3_support_point_sentence(support_clause)}, but that support is not enough to justify a larger shift here."
            if fb == 0:
                if oppose_clause:
                    return f"{_step3_support_point_sentence(support_clause)}, but {_step3_challenge_point_sentence(oppose_clause).lower()} still keeps the issue unsettled."
                return f"{_step3_support_point_sentence(support_clause)}, but that support is still not enough to settle the claim."
            return f"{_step3_support_point_sentence(support_clause)}, but that point is not enough to justify a stronger rating."
        return "The tweet gives some support, but it is not enough to justify a larger shift here."

    if attempted_more_skeptical:
        if oppose_clause:
            if fb > 0:
                return f"{_step3_challenge_point_sentence(oppose_clause)}, but that challenge is not enough to justify a larger shift here."
            if fb == 0:
                if support_clause:
                    return f"{_step3_challenge_point_sentence(oppose_clause)}, but {_step3_support_point_sentence(support_clause).lower()} still keeps the issue unsettled."
                return f"{_step3_challenge_point_sentence(oppose_clause)}, but that challenge is still not enough to settle the claim."
            return f"{_step3_challenge_point_sentence(oppose_clause)}, but that point is not enough to justify a stronger negative rating."
        return "The tweet raises some doubt, but it is not enough to justify a larger shift here."

    return _step3_tweet_local_explanation(tweet_text, fb, int(pre_belief) if pre_belief is not None else fb, tweet_stance)




def _step3_rating_direction_label(rating: int | None) -> str:
    """Map a Step-3 rating to a prose direction label."""
    try:
        r = int(rating)
    except Exception:
        return "unknown"
    if r > 0:
        return "support"
    if r < 0:
        return "oppose"
    return "balanced"


def _step3_movement_direction_label(pre_belief: int | None, final_rating: int | None) -> str:
    """Describe whether the listener moved up, down, or stayed unchanged."""
    try:
        p = int(pre_belief)
        r = int(final_rating)
    except Exception:
        return "unknown"
    if r > p:
        return "support"
    if r < p:
        return "oppose"
    return "stay"


def _step3_explanation_direction_labels(explanation_text: str) -> set[str]:
    """Classify an explanation into coarse support/oppose/balanced labels."""
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not expl:
        return set()
    has_support = _step3_text_has_support_markers(expl)
    has_oppose = _step3_text_has_oppose_markers(expl)
    has_balance = _step3_explanation_has_contrastive_weakening(expl) or bool(re.search(
        r"\b(?:mixed|balanced|both sides|on one hand|on the other hand|uncertain|unsure|not settled|not enough to decide|keeps? both sides|cuts? both ways)\b",
        expl,
        flags=re.I,
    ))
    if has_support and has_oppose:
        return {"balanced", "support", "oppose"}
    if has_balance and (has_support or has_oppose):
        labels = {"balanced"}
        # Keep the directional label too, but the balanced label prevents false
        # repairs on bounded partial updates and memory runs.
        if has_support:
            labels.add("support")
        if has_oppose:
            labels.add("oppose")
        return labels
    if has_balance:
        return {"balanced"}
    if has_support:
        return {"support"}
    if has_oppose:
        return {"oppose"}
    return set()


def _step3_tweet_direction_labels(tweet_text: str = "") -> set[str]:
    """Topic-general direction labels for the received tweet."""
    tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not tw:
        return set()
    left, right = _step3_extract_tweet_local_clauses(tw)
    pieces = [x for x in [left, right] if str(x or "").strip()]
    if not pieces:
        pieces = [tw]
    labels = set()
    for piece in pieces:
        if _step3_text_has_support_markers(piece):
            labels.add("support")
        if _step3_text_has_oppose_markers(piece):
            labels.add("oppose")
    if "support" in labels and "oppose" in labels:
        labels.add("balanced")
    elif _tweet_has_explicit_balancing_clause(tw):
        labels.add("balanced")
    return labels


def _step3_mask_negated_certainty_phrases(text: str) -> str:
    """Remove negated certainty phrases before overshoot checks.

    Without this, neutral explanations like "does not definitively prove X"
    were incorrectly treated as overconfident proof language because the narrow
    overshoot regex saw "definitively" in isolation.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not s:
        return ""
    negator = r"(?:not|no|without|lack(?:s|ing)?(?:\s+of)?|absence\s+of|missing|does(?:\s+not|n't)|do(?:\s+not|n't)|did(?:\s+not|n't)|cannot|can't|is(?:\s+not|n't)|are(?:\s+not|n't)|was(?:\s+not|n't)|were(?:\s+not|n't))"
    certainty = r"(?:undeniable|irrefutable|conclusive|conclusively|definitive|definitively|certain|certainly|clearly|fully|completely|prove|proves|proven|disprove|disproves|disproven|confirm|confirms|confirmed|settle|settles|settled|validate|validates|validated)"
    # Allow a short plain-word span between the negator and the certainty word:
    # e.g. "does not definitively prove", "not enough to settle".
    # Keep this regex deliberately simple to avoid pathological backtracking.
    return re.sub(rf"(?:^|\s)(?:{negator})(?:\s+\w+){{0,4}}\s+(?:{certainty})\b", " ", s, flags=re.I)


def _step3_has_strong_support_overshoot(text: str) -> bool:
    """Very strong pro-claim wording, topic-general."""
    s = _step3_mask_negated_certainty_phrases(text)
    if not s:
        return False
    return bool(re.search(
        r"\b(?:undeniable|irrefutable|conclusive|conclusively|definitive|definitively|beyond doubt|settles? (?:it|this|the matter)|fully settles?|completely settles?|proves? (?:it|this|the claim)?\s*(?:true|right|correct|valid|real)?|confirms? (?:it|this|the claim)?\s*(?:completely|fully|beyond doubt|definitively|conclusively))\b",
        s,
        flags=re.I,
    ))


def _step3_has_strong_oppose_overshoot(text: str) -> bool:
    """Very strong anti-claim wording, topic-general."""
    s = _step3_mask_negated_certainty_phrases(text)
    if not s:
        return False
    return bool(re.search(
        r"\b(?:proves? (?:it|this|the claim)?\s*(?:false|wrong|fake|invalid|impossible|a hoax|staged)|definitively disproves?|conclusively disproves?|completely undermines?|settles? (?:against|the case against)|definitely false|clearly false|certainly false|undeniably false|irrefutably false|fake|hoax|staged)\b",
        s,
        flags=re.I,
    ))


def _step3_explanation_too_extreme_for_final_rating(explanation_text: str, final_rating: int) -> bool:
    """Narrow guard for explanations that overshoot the final rating's strength.

    This is intentionally topic-general. It flags only very strong certainty or
    disproof language. Normal phrases like "supports", "casts doubt", "raises a
    concern", or "leaves a gap" should pass.
    """
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not s:
        return False
    try:
        r = int(final_rating)
    except Exception:
        return False

    strong_support = _step3_has_strong_support_overshoot(s)
    strong_oppose = _step3_has_strong_oppose_overshoot(s)

    if r == 2:
        # Strong support is allowed at +2; strong disproof is not.
        return strong_oppose
    if r == -2:
        # Strong opposition is allowed at -2; conclusive support is not.
        return strong_support
    if r == 1:
        # +1 is supportive but not maximal certainty, and it should not sound anti-claim.
        return strong_support or strong_oppose
    if r == -1:
        # -1 is skeptical but not maximal rejection, and it should not sound fully pro-claim.
        return strong_support or strong_oppose
    if r == 0:
        # 0 should not contain conclusive one-sided language.
        return strong_support or strong_oppose
    return False


def _step3_explanation_limits_opposite_side(explanation_text: str) -> bool:
    """True when an explanation mentions one side only to limit it.

    Examples:
    - "X supports the claim, but not enough to settle it."
    - "Y raises doubt, but not enough to overturn the support."
    """
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not s:
        return False
    return bool(re.search(
        r"\b(?:but|however|though|although|yet|despite|still|remains?|not enough|not sufficient|too limited|too weak|does(?: not|n't)|cannot|can't|lacks?|lack of|without|unresolved|unclear|mixed|balanced|keeps? .* in play|doesn'?t erase|does not erase|doesn'?t overturn|does not overturn|doesn'?t negate|does not negate)\b",
        s,
        flags=re.I,
    ))




def _v130_context_is_active(*texts) -> bool:
    """True only when the local text actually concerns the simulation topic."""
    blob = re.sub(r"\s+", " ", " ".join(str(t or "") for t in texts)).strip().lower()
    return bool(re.search(
        r"\b(?:simulation hypothesis|computer simulation|simulated reality|simulated observers?|ancestor simulations?|advanced civilizations?|simulated world|simulation theory)\b",
        blob,
        flags=re.I,
    ))


def _v130_anti_simulation_families(text: str) -> set[str]:
    """Narrow v130 anti-claim concept families.

    These are used only for Step-3 grounding checks and Step-2 warning-only
    diagnostics. A family is accepted only when the same family appears in the
    tweet or shown material, so this is not a broad whitelist.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not s:
        return set()
    checks = [
        ("testability_evidence", r"\b(?:lack(?:s)? (?:of )?(?:a )?(?:clear |direct |agreed )?test|no (?:clear |direct |agreed )?test|without (?:a )?(?:clear |direct )?test|testable framework|not testable|untestable|unfalsifiab(?:le|ility)|speculative|speculation|without (?:clear |concrete |direct )?evidence|no (?:clear |concrete |direct )?evidence|without (?:clear |concrete |direct )?proof|no (?:clear |concrete |direct )?proof|not evidence[- ]based|thought experiment)\b"),
        ("future_assumptions", r"\b(?:unverified assumptions?|uncertain assumptions?|future tech(?:nology)?|advanced civilizations? (?:aren't|are not|not) proven|civilization survival|capabilities and motives|future tech and motives|motives (?:remain|are) uncertain|too uncertain to support)\b"),
        ("possibility_not_probability", r"\b(?:possibility alone|coherent without being likely|logically possible (?:but|without|is not)|possible without being likely|not enough to believe|does(?: not|n't) make it likely)\b"),
        ("ordinary_physics_math", r"\b(?:ordinary physics|natural mathematical rules?|math(?:ematics)? (?:in|of) (?:our )?universe|nature,? not code|without requiring a simulation|does(?: not|n't) require a simulation|non[- ]simulated physical theories)\b"),
        ("consciousness_code", r"\b(?:consciousness|conscious beings?|known code|does(?: not|n't) follow (?:any )?(?:known )?code|predefined rules?|requires? predefined rules?|simulations? require (?:predefined )?rules?)\b"),
        ("origin_question", r"\b(?:does(?: not|n't) explain why reality exists|pushes? the question back|origin of reality|why reality exists)\b"),
    ]
    return {label for label, pat in checks if re.search(pat, s, flags=re.I)}


def _v130_family_grounded_in_tweet_or_material(explanation_text: str, tweet_text: str = "", grounding_text: str = "") -> bool:
    """Check whether a v130 objection family is grounded in the tweet or shown material."""
    expl_families = _v130_anti_simulation_families(explanation_text)
    if not expl_families:
        return False
    evidence_text = "\n".join([str(tweet_text or ""), str(grounding_text or "")])
    evidence_families = _v130_anti_simulation_families(evidence_text)
    return bool(expl_families & evidence_families)


def _step3_valid_v130_negative_explanation(explanation_text: str, final_rating: int, tweet_text: str = "", grounding_text: str = "") -> bool:
    """Allow only tweet/material-grounded anti-simulation explanations for negative Step-3 ratings.

    This prevents false wrong-side rewrites like replacing a valid local reason
    with generic fallback text, while avoiding a broad whitelist that would accept
    invented Step-3 explanations.
    """
    try:
        r = int(final_rating)
    except Exception:
        return False
    if r >= 0:
        return False
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    grounding = str(grounding_text or _current_validation_grounding_text() or "")
    if not s or not _v130_context_is_active(s, tweet_text, grounding):
        return False
    return _v130_family_grounded_in_tweet_or_material(s, tweet_text=tweet_text, grounding_text=grounding)



def _step3_explanation_obvious_fragment_reason(expl_text: str) -> str | None:
    """Module-level fragment detector used by Step-3 guards.

    A later Step-3 routine also defines a local helper with this name for its own
    audit path. The wrong-side rewrite guard runs at module scope, so it needs a
    module-level definition too. Keep this conservative to avoid rewriting valid
    short explanations.
    """
    expl = re.sub(r"\s+", " ", str(expl_text or "")).strip()
    if not expl:
        return None
    words = re.findall(r"[A-Za-z']+", expl)
    low = expl.lower().strip()
    has_terminal = bool(re.search(r"[.!?]$", expl))

    predicateish = bool(re.search(
        r"\b(?:is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|may|might|must|should|would|will|supports?|weakens?|undermines?|raises?|casts?|leaves?|keeps?|makes?|shows?|suggests?|confirms?|validates?|challenges?|complicates?|erodes?|provides?|points?|counts?|fits?|settles?|proves?|fails?)\b",
        low,
        flags=re.I,
    ))
    if len(words) < 5 and not predicateish:
        return 'obvious_fragment_too_short'

    last = words[-1].lower() if words else ''
    dangling_tail = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'that', 'because', 'about',
        'of', 'to', 'in', 'on', 'for', 'with', 'from', 'this', 'these', 'those',
        'into', 'onto', 'than', 'via', 'through', 'during', 'while', 'despite', 'although'
    }
    if last in dangling_tail:
        return 'obvious_fragment_dangling_tail'

    if (not has_terminal) and re.search(
        r"(?i)\b(?:used|shown|seen|found|confirmed|validated|tested|measured|raised|left|placed|reported|cited)\s+(?:in|on|for|with|by|through|during)\s+(?:laser|lunar|public|official|historical|physical|direct|independent|original|later|scientific|technical|visual|experimental)$",
        expl,
    ):
        return 'obvious_fragment_dangling_phrase'

    if re.search(r"(?i)\b(?:resemblance|similarity|comparison|sign|idea|claim|possibility|reason|point|argument|reference)\s+to$", expl):
        return 'obvious_fragment_dangling_phrase'
    if re.search(r"(?i)\b(?:because|without|with|for|from|about|that|through|during|despite|although)\s*$", expl):
        return 'obvious_fragment_dangling_tail'
    if len(words) <= 8 and not has_terminal:
        return 'obvious_fragment_no_terminal_punctuation'
    return None

def _step3_should_keep_original_explanation_before_wrong_side_rewrite(explanation_text: str, final_rating: int, tweet_text: str = "", pre_belief: int | None = None) -> bool:
    """Guard against unnecessary Step-3 wrong-side fallback rewrites.

    The fallback rewrite is only for clear semantic mismatches. If the original
    explanation already contains a same-side, bounded, non-fragment local theme,
    keep the model's wording instead of replacing it with a generic fallback.
    """
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not s:
        return False
    if _step3_explanation_obvious_fragment_reason(s):
        return False
    try:
        r = int(final_rating)
    except Exception:
        return False
    if _step3_explanation_too_extreme_for_final_rating(s, r):
        return False

    labels = _step3_explanation_direction_labels(s)
    final_dir = _step3_rating_direction_label(r)
    if final_dir in labels:
        return True
    if "balanced" in labels and abs(r) <= 1:
        return True

    # Topic-local v130 support/oppose patterns. These are narrow by design and
    # require the theme to appear in the explanation itself, not just in the prompt.
    low = s.lower()
    if r > 0 and re.search(r"\b(?:simulated observers?|simulated minds? could be real observers?|real observers rather than merely fictional|outnumber|observer[- ]count|advanced civilizations?|ancestor simulations?|many simulations?|stable laws|limited observable horizons|information[- ]like constraints?|no obvious glitches|hide(?:s)? its artificial structure|lack of direct outside access|computational process|generated reality)\b", low, flags=re.I):
        return True
    if r < 0 and re.search(r"\b(?:lack(?:s)? (?:of )?(?:a )?(?:clear |direct |definitive |agreed )?test|no (?:clear |direct |agreed )?test|speculative|without (?:clear |concrete |direct )?evidence|no (?:clear |concrete |direct )?evidence|uncertain assumptions?|unverified assumptions?|future tech(?:nology)?|civilization survival|possibility alone|coherent without being likely|ordinary physics|unfalsifiab(?:le|ility))\b", low, flags=re.I):
        return True
    return False


def _step3_explanation_wrong_side_anchor(explanation_text: str, final_rating: int, tweet_text: str = "", pre_belief: int | None = None) -> bool:
    """Detect clear Step-3 rating/explanation mismatches with bounded dual anchors.

    Policy:
    - Accept explanations compatible with the final rating.
    - Accept explanations compatible with the movement direction.
    - For memory safety, do NOT accept explanations that overshoot the final
      rating's strength, e.g. FINAL_RATING:-1 with "this proves the claim false"
      or FINAL_RATING:0 with conclusive one-sided wording.
    - Moving into 0 should be neutral/balanced, especially when the movement goes
      away from the received tweet. Pure one-sided explanations for 0 are flagged
      unless they explicitly contain limiting/balancing language.
    """
    try:
        r = int(final_rating)
    except Exception:
        return False
    try:
        p = int(pre_belief)
    except Exception:
        p = r

    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not expl:
        return False

    if _step3_valid_v130_negative_explanation(expl, r, tweet_text=tweet_text, grounding_text=_current_validation_grounding_text()):
        return False

    labels = _step3_explanation_direction_labels(expl)
    if not labels:
        return False

    # First, enforce bounded strength. This protects memory from contradictory
    # rating/explanation pairs even when the side label technically matches.
    if _step3_explanation_too_extreme_for_final_rating(expl, r):
        return True

    final_dir = _step3_rating_direction_label(r)
    move_dir = _step3_movement_direction_label(p, r)
    tweet_dirs = _step3_tweet_direction_labels(tweet_text)
    limiting = _step3_explanation_limits_opposite_side(expl)

    # One-step softening inside the same side is allowed to cite the opposing
    # point directly. Example: pre=+2 -> final=+1 may say that unsupported
    # assumptions weaken likelihood. That is the reason for softening, not a
    # wrong-side anchor. The previous detector rewrote these valid cases.
    same_side_softening = False
    try:
        same_side_softening = bool(p != 0 and r != 0 and ((p > 0 and r > 0) or (p < 0 and r < 0)) and abs(r) < abs(p))
    except Exception:
        same_side_softening = False
    if same_side_softening and move_dir in labels:
        return False

    # Neutral final ratings are the most memory-sensitive. A 0 explanation should
    # be balanced or explicitly limited; pure one-sided text is too easy for later
    # memory to misread as +1/-1 logic.
    if r == 0:
        if "balanced" in labels:
            return False
        # If it moved into 0 and gives a one-sided movement reason, require an
        # explicit limit/balance phrase. Otherwise it sounds like it should have
        # moved past 0.
        return True

    # Same-rating cases should usually match the final rating, unless the text is
    # explicitly limiting the opposite side.
    if move_dir == "stay":
        if final_dir in labels:
            return False
        if "balanced" in labels and abs(r) <= 1:
            return False
        if limiting:
            return False
        return True

    # Final-rating-compatible explanations are valid, provided they are not too
    # extreme. This covers -2 -> -1 with a mild skeptical explanation, or +2 -> +1
    # with a mild supportive explanation.
    if final_dir in labels:
        return False

    # Movement-compatible explanations are valid only when bounded. If the agent
    # moves with the tweet's direction, pure movement-side explanation can pass.
    # If the agent moves away from the tweet, require balanced/limiting wording so
    # we do not save a memory pair that looks stronger than the final rating.
    if move_dir in {"support", "oppose"} and move_dir in labels:
        if "balanced" in labels or limiting:
            return False
        if not tweet_dirs or move_dir in tweet_dirs or "balanced" in tweet_dirs:
            return False
        return True

    # Balanced text is acceptable for mild non-zero final ratings; it means the
    # agent did not fully flip despite some pull.
    if "balanced" in labels and abs(r) <= 1:
        return False

    return True


def _step3_generic_simulation_or_bounded_fallback(tweet_text: str, final_rating: int, pre_belief: int, direction: str = "") -> str:
    """Topic-grounded fallback for cases where clause extraction failed."""
    tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    direction = str(direction or "").strip().lower()
    try:
        r = int(final_rating)
    except Exception:
        r = 0
    if re.search(r"\b(?:clear test|direct test|testable framework|not testable|untestable|unfalsifiab|speculative|speculation)\b", tw, flags=re.I):
        if r < 0:
            return "The lack of a clear test keeps the simulation claim speculative."
        if r == 0:
            return "The simulation idea is plausible, but the lack of a clear test keeps it unsettled."
        return "The simulation argument still gives the claim some support, though the lack of a clear test limits certainty."
    if re.search(r"\b(?:future tech|future technology|civilization survival|unverified assumptions?|uncertain assumptions?)\b", tw, flags=re.I):
        if r < 0:
            return "Unverified assumptions about advanced civilizations weaken the simulation claim."
        if r == 0:
            return "The future-technology assumptions keep the simulation claim unsettled."
        return "The simulation argument still gives the claim some support, though future-technology assumptions limit certainty."
    if re.search(r"\b(?:advanced civilizations?|ancestor simulations?|countless simulations?|detailed simulations?|simulated observers?)\b", tw, flags=re.I):
        if r > 0:
            return "The observer-count argument gives the claim some support."
        if r == 0:
            return "The advanced-simulation possibility gives support, but uncertainty keeps the claim unsettled."
        return "The advanced-simulation possibility gives some support, but it is not enough to remove the doubt here."
    if direction == "oppose":
        return "The tweet gives a concrete challenge that weighs against the claim here."
    if direction == "support":
        return "The tweet gives a concrete point that supports the claim here."
    return "The tweet gives one concrete point, but the update remains bounded."


def _step3_v130_support_limit_phrase(text: str) -> str:
    """Return a compact support-side phrase for v130 positive fallback text.

    Used only after the rating is already fixed. It does not expand validator
    acceptance; it only prevents deterministic fallback from producing generic
    strings like "That point provides support, which supports the claim."
    """
    low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if re.search(r"\b(?:simulated observers?|outnumber|observer[- ]count|odds|randomly located observer)\b", low, flags=re.I):
        return "The observer-count argument gives the claim some support"
    if re.search(r"\b(?:advanced civilizations?|ancestor simulations?|many simulations?|countless simulations?|detailed simulations?)\b", low, flags=re.I):
        return "The advanced-simulation argument gives the claim some support"
    if re.search(r"\b(?:no obvious glitches|hide(?:s)? its artificial structure|hide(?:s)? its nature|outside access|simulated world)\b", low, flags=re.I):
        return "The hidden-simulation point gives the claim some support"
    if re.search(r"\b(?:stable laws|limited observable horizons|information[- ]like constraints?|mathematical rules|generated reality)\b", low, flags=re.I):
        return "The simulation-compatibility point gives the claim some support"
    return "The simulation argument still gives the claim some support"


def _step3_v130_challenge_limit_phrase(text: str) -> str:
    """Return a compact limiting phrase for v130 positive fallback text."""
    low = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if re.search(r"\b(?:future tech(?:nology)?|civilization survival|survive long enough|motives?|methods?|why and how|number of simulations?|uncertain assumptions?|unverified assumptions?)\b", low, flags=re.I):
        return "uncertainty about future civilizations limits certainty"
    if re.search(r"\b(?:conscious simulations?|conscious simulated beings?|ethical|practical|cultural reasons|choose not to create)\b", low, flags=re.I):
        return "uncertainty about conscious simulations limits certainty"
    if re.search(r"\b(?:clear test|direct test|testable framework|not testable|untestable|unfalsifiab|speculative|speculation)\b", low, flags=re.I):
        return "the lack of a clear test limits certainty"
    if re.search(r"\b(?:concrete evidence|direct evidence|stronger evidence|proof|evidence gap|no evidence|without evidence)\b", low, flags=re.I):
        return "the evidence gap limits certainty"
    if re.search(r"\b(?:ordinary physics|reference class|mathematical laws|nature,? not code|without requiring a simulation)\b", low, flags=re.I):
        return "alternative explanations limit certainty"
    return "the unresolved challenge limits certainty"


def _step3_v130_positive_limited_fallback(tweet_text: str, support_clause: str = "", oppose_clause: str = "") -> str:
    """Balanced-but-positive fallback for FINAL_RATING > 0.

    For v130, use topic-grounded wording. For other topics, fall back to the
    existing generic local-point wording so simulation-specific text does not
    leak into unrelated experiments.
    """
    blob = " ".join(str(x or "") for x in [support_clause, oppose_clause, tweet_text, _current_validation_grounding_text()]).strip()
    if not _v130_context_is_active(blob):
        if support_clause and oppose_clause:
            return f"{_step3_support_point_sentence(support_clause)}, though that challenge limits certainty."
        if support_clause:
            return f"{_step3_support_point_sentence(support_clause)}, which supports the claim."
        if oppose_clause:
            return f"{_step3_challenge_point_sentence(oppose_clause)}, but that challenge is not enough to overturn the support here."
        return "The claim still has some support, but the update remains bounded."

    support = _step3_v130_support_limit_phrase(blob)
    challenge = _step3_v130_challenge_limit_phrase(blob)
    return f"{support}, though {challenge}."


def _step3_safe_fallback_explanation_for_rating(tweet_text: str, final_rating: int, pre_belief: int, tweet_stance: str | None = None) -> str:
    """Conservative local rewrite for rating/explanation mismatches after native rescue fails.

    This is movement-aware. It should not force explanations to match the final
    rating sign when the actual update moved in the opposite direction, e.g.
    -2 -> -1 or +2 -> +1.
    """
    try:
        r = int(final_rating)
    except Exception:
        r = 0
    try:
        p = int(pre_belief)
    except Exception:
        p = r
    left, right = _step3_extract_tweet_local_clauses(tweet_text)
    left_dir = _step3_tweet_clause_direction(left)
    right_dir = _step3_tweet_clause_direction(right)
    support_clause = left if left_dir == 'support' else (right if right_dir == 'support' else '')
    oppose_clause = left if left_dir == 'oppose' else (right if right_dir == 'oppose' else '')
    support_frag = _step3_point_fragment(support_clause) if support_clause else 'the supportive detail'
    oppose_frag = _step3_point_fragment(oppose_clause) if oppose_clause else 'the unresolved challenge'

    # Bounded transition-aware fallback. Do not force the explanation to match
    # only the final sign, but also do not let it overshoot the final rating.
    if r == 0:
        if support_clause and oppose_clause:
            return f"{_step3_support_point_sentence(support_clause)}, but {_step3_challenge_point_sentence(oppose_clause).lower()} keeps the claim unresolved."
        if support_clause:
            return f"{_step3_support_point_sentence(support_clause)}, but that support is still not enough to settle the claim."
        if oppose_clause:
            return f"{_step3_challenge_point_sentence(oppose_clause)}, but that challenge is still not enough to settle the claim."
        return "The tweet gives one concrete point, but it is still not enough to settle the claim."

    if r > p:
        if support_clause:
            if r < 0:
                return f"{_step3_support_point_sentence(support_clause)}, which softens the doubt but does not fully settle the claim."
            if r == 0:
                return f"{_step3_support_point_sentence(support_clause)}, but the claim still remains mixed."
            return f"{_step3_support_point_sentence(support_clause)}, which supports a move toward the claim."
        if oppose_clause:
            if r <= 0:
                return f"{_step3_challenge_point_sentence(oppose_clause)}, but that challenge is not enough to keep the stronger negative rating."
            return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)
        return _step3_generic_simulation_or_bounded_fallback(tweet_text, r, p, direction="support")

    if r < p:
        if oppose_clause:
            if r > 0:
                return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)
            if r == 0:
                return f"{_step3_challenge_point_sentence(oppose_clause)}, but the claim still remains mixed."
            return f"{_step3_challenge_point_sentence(oppose_clause)}, which supports a move against the claim."
        if support_clause:
            if r >= 0:
                return f"{_step3_support_point_sentence(support_clause)}, but that support is not enough to keep the stronger positive rating."
            return f"{_step3_support_point_sentence(support_clause)}, but that support is not enough to remove the remaining doubt."
        return _step3_generic_simulation_or_bounded_fallback(tweet_text, r, p, direction="oppose")


    if r > 0:
        if support_clause:
            if oppose_clause:
                return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)
            return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)
        if oppose_clause:
            return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)
        return _step3_v130_positive_limited_fallback(tweet_text, support_clause=support_clause, oppose_clause=oppose_clause)

    if r < 0:
        if oppose_clause:
            if support_clause:
                return f"{_step3_challenge_point_sentence(oppose_clause)}, so {_step3_support_point_sentence(support_clause).lower()} is not enough to remove the doubt here."
            return f"{_step3_challenge_point_sentence(oppose_clause)}, which leaves a concrete reason for doubt."
        if support_clause:
            return f"{_step3_support_point_sentence(support_clause)}, but that support is still not enough to remove the remaining doubt here."
        return _step3_generic_anchor_for_rating(r, tweet_stance)

    return _step3_generic_anchor_for_rating(r, tweet_stance)


def _step3_explanation_echoes_tweet_evidence_complaint(explanation_text: str, tweet_text: str = "") -> bool:
    """Detect when a Step-3 explanation correctly echoes a tweet evidence/testability complaint."""
    expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    tweet = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    if not expl or not tweet:
        return False

    evidenceish = r"\b(?:evidence|confirm|confirmation|test|testable|verify|verification|speculative|speculation)\b"
    if not re.search(evidenceish, expl, flags=re.I):
        return False
    if not re.search(evidenceish, tweet, flags=re.I):
        return False

    shared_local_patterns = [
        r"\bdirect way to (?:confirm|test|verify)\b",
        r"\black of direct evidence\b",
        r"\black of verifiable (?:evidence|proof)\b",
        r"\bverifiable (?:evidence|proof)\b",
        r"\bwithout (?:concrete|clear|direct|verifiable) (?:evidence|proof)\b",
        r"\bremains speculative\b",
        r"\bjust speculation\b",
        r"\btestable boundary\b",
    ]
    if any(re.search(p, expl, flags=re.I) and re.search(p, tweet, flags=re.I) for p in shared_local_patterns):
        return True

    if re.search(r"\bdirect evidence\b", expl, flags=re.I) and re.search(r"\b(?:direct way to confirm|direct way to test|concrete evidence|clear evidence|speculative)\b", tweet, flags=re.I):
        return True
    if re.search(r"\bverifiable (?:evidence|proof)\b", expl, flags=re.I) and re.search(r"\bverifiable (?:evidence|proof)\b", tweet, flags=re.I):
        return True
    if re.search(r"\black of verifiable (?:evidence|proof)\b", expl, flags=re.I) and re.search(r"\bwithout verifiable (?:evidence|proof)\b", tweet, flags=re.I):
        return True

    return False


def _step3_strict_closed_external_reason_reason(explanation_text: str, tweet_text: str = "", prompt_text: str = "") -> str | None:
    """Flag strict-closed Step-3 explanations that smuggle in unseen evidence/verification language.

    Important: do NOT flag tweet-local echoing. If the explanation is simply naming the same
    evidence/verification complaint already present in the received tweet, that is not an unseen
    outside-source appeal in strict closed world.
    """
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not s:
        return None
    low = s.lower()
    tweet_low = re.sub(r"\s+", " ", str(tweet_text or "")).strip().lower()
    if _step3_explanation_echoes_tweet_evidence_complaint(low, tweet_low):
        return None
    patterns = [
        (r"\boverwhelming evidence\b", "strict_closed_overwhelming_evidence"),
        (r"\bstrong evidence\b", "strict_closed_strong_evidence"),
        (r"\bdirect verifiable evidence\b", "strict_closed_verifiable_evidence"),
        (r"\bverifiable evidence\b", "strict_closed_verifiable_evidence"),
        (r"\bhistorical evidence\b", "strict_closed_historical_evidence"),
        (r"\bdirect evidence\b", "strict_closed_direct_evidence"),
        (r"\bpublic evidence\b", "strict_closed_public_evidence"),
        (r"\bofficial confirmation\b", "strict_closed_official_confirmation"),
        (r"\bwell-documented\b", "strict_closed_documented_evidence"),
        (r"\bcredible sources?\b", "strict_closed_credible_sources"),
        (r"\bmultiple independent sources?\b", "strict_closed_multiple_independent_sources"),
        (r"\bcorroborating data\b", "strict_closed_corroborating_data"),
        (r"\bglobal consensus\b", "strict_closed_global_consensus"),
        (r"\bwhat everyone knows\b", "strict_closed_public_knowledge"),
    ]
    for pat, reason in patterns:
        if re.search(pat, low, flags=re.I):
            if _validator_allows_strict_closed_phrase(reason, s, prompt_text=prompt_text, policy=_current_persona_validation_policy()):
                continue
            if tweet_low and re.search(pat, tweet_low, flags=re.I):
                continue
            if _step3_explanation_echoes_tweet_evidence_complaint(low, tweet_low):
                continue
            return reason
    return None



def _interaction_quality_label(tags) -> str:
    """Classify an interaction as move, hold, invalid, or artifact for reporting."""
    tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
    if any(
        ('repair_attempt_failed' in t)
        or ('fallback' in t)
        or ('missing_final_rating' in t)
        or ('empty_input_tweet_skip_step3' in t)
        for t in tag_list
    ):
        return "fatal_fallback"
    if tag_list:
        return "soft_violation"
    return "ok"

def _sentence_case_cleanup(text: str) -> str:
    """Clean capitalization and spacing after deterministic explanation rewriting."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    parts = re.split(r"([.!?]\s+)", s)
    out = []
    capitalize_next = True
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[.!?]\s+", part):
            out.append(part)
            capitalize_next = True
            continue
        seg = part.lstrip()
        if capitalize_next and seg:
            seg = seg[:1].upper() + seg[1:]
        out.append(seg if part == seg else (part[:len(part)-len(seg)] + seg))
        capitalize_next = False
    return ''.join(out).strip()

def _step3_expl_semantic_label(expl: str) -> str | None:
    """Very small semantic classifier for Step-3 explanations.

    Returns one of: "stay", "support", "oppose", "uncertain", or None.
    """
    if not expl:
        return None

    s = re.sub(r"\s+", " ", str(expl)).strip().lower()
    if not s:
        return None

    stay_patterns = [
        r"\bi(?: am|[' ]m)? staying at (?:my|the) current rating\b",
        r"\bi prefer to stay at (?:my|the) current rating\b",
        r"\bi(?: am|[' ]m)? sticking with (?:my|the) current rating\b",
        r"\bi did(?: not|n't) change my rating\b",
        r"\bnot strong enough to justify a rating change\b",
        r"\bnot strongly enough to justify a rating change\b",
        r"\bnot persuasive enough to justify a rating change\b",
        r"\bnot (?:strong|strongly|persuasive|convincing) enough to justify a change\b",
        r"\bnot (?:strong|strongly|persuasive|convincing) enough to change my (?:stance|mind|rating)\b",
        r"\bnot enough evidence to change my (?:stance|mind|rating)\b",
        r"\bnot enough to change my (?:stance|mind|rating)\b",
        r"\bdoes(?: not|n't) provide enough .* to justify a rating change\b",
        r"\bdoes(?: not|n't) provide enough .* to change my (?:stance|mind|rating)\b",
        r"\bdoes(?: not|n't) change my (?:stance|mind|rating)\b",
        r"\bnot strong enough to justify moving more negative\b",
        r"\bnot persuasive enough to justify moving more negative\b",
        r"\bnot convincing enough to justify moving more negative\b",
        r"\bnot strong enough to justify moving more positive\b",
        r"\bnot persuasive enough to justify moving more positive\b",
        r"\bnot convincing enough to justify moving more positive\b",
        r"\bnot strong enough to move more negative\b",
        r"\bnot persuasive enough to move more negative\b",
        r"\bnot convincing enough to move more negative\b",
        r"\bnot strong enough to move more positive\b",
        r"\bnot persuasive enough to move more positive\b",
        r"\bnot convincing enough to move more positive\b",
        r"\bkeep(?:ing)? (?:my|the) current rating\b",
        r"\bi remain at (?:my|the) current rating\b",
        r"\bi remain\b.*\bcurrent rating\b",
        r"\bi remain strongly negative\b",
        r"\bi remain somewhat negative\b",
        r"\bi remain somewhat disagree(?:ing)?\b",
        r"\bi remain neutral\b",
        r"\bi remain unsure\b",
        r"\bi remain somewhat positive\b",
        r"\bi remain somewhat agree(?:ing)?\b",
        r"\bi remain strongly positive\b",
        r"\bi(?: am|[' ]m)? still strongly negative\b",
        r"\bi(?: am|[' ]m)? still somewhat negative\b",
        r"\bi(?: am|[' ]m)? still neutral\b",
        r"\bi(?: am|[' ]m)? still unsure\b",
        r"\bi(?: am|[' ]m)? still somewhat positive\b",
        r"\bi(?: am|[' ]m)? still strongly positive\b",
        r"\bi(?: am|[' ]m)? staying at strongly negative\b",
        r"\bi(?: am|[' ]m)? staying at somewhat disagree\b",
        r"\bi(?: am|[' ]m)? staying at neutral\b",
        r"\bi(?: am|[' ]m)? staying at somewhat agree\b",
        r"\bi(?: am|[' ]m)? staying at strongly positive\b",
        r"\bi stay at a strong agreement\b",
        r"\bi stay at a strong disagreement\b",
    ]
    for pat in stay_patterns:
        if re.search(pat, s, flags=re.I):
            return "stay"

    uncertain_patterns = [
        r"\bonly expresses uncertainty\b",
        r"\bexpresses uncertainty\b",
        r"\bthe tweet is uncertain\b",
        r"\bthe tweet is unsure\b",
        r"\bno information\b",
        r"\bnot enough information\b",
        r"\bneed more information\b",
        r"\bmove toward(?:s)? uncertainty\b",
        r"\bmove(?:d|s)? (?:my )?rating toward 0\b",
        r"\bmove(?:d|s)? me toward 0\b",
        r"\bmove(?:d|s)? (?:my )?rating to 0\b",
        r"\bmove(?:d|s)? me to 0\b",
        r"\bmade me move to 0\b",
        r"\bmakes me move to 0\b",
        r"\bbecame more uncertain\b",
        r"\bmade me more uncertain\b",
        r"\bpushed me toward uncertainty\b",
        r"\braised some uncertainty\b",
        r"\bintroduces? uncertainty\b",
        r"\bcreates? uncertainty\b",
        r"\bleaves? uncertainty\b",
        r"\bnot sure which to trust\b",
        r"\bthis weakens my current view without fully reversing it\b",
    ]
    for pat in uncertain_patterns:
        if re.search(pat, s, flags=re.I):
            return "uncertain"

    support_patterns = [
        r"\bargues? for the claim\b",
        r"\bthe tweet argues? for the claim\b",
        r"\bsupports the claim\b",
        r"\bstrongly supports the claim\b",
        r"\bevidence for the claim\b",
        r"\breason to believe the claim\b",
        r"\bstrong reason to believe the claim\b",
        r"\bstrong enough for a small positive move\b",
        r"\bstrong enough to justify a small positive move\b",
        r"\bjustif(?:y|ies) a small positive move\b",
        r"\bpush(?:es)? me (?:slightly )?more positive\b",
        r"\bthe fact pack .* supports the claim\b",
        r"\blean(?:ing)? slightly in favor\b",
        r"\blean(?:ing)? toward believing\b",
        r"\bgives? me some reason to believe\b",
        r"\bmakes? me less skeptical\b",
        r"\bsupport(?:ing)? the claim\b",
        r"\bstrong foundation of evidence supporting the claim\b",
        r"\bincreases? my confidence in the claim\b",
        r"\bmakes? me mildly convinced\b",
        r"\blean(?:ing)? more positively toward(?:s)? the claim\b",
        r"\blean(?:ing)? more positively\b",
        r"\bweakens? my previous mild disagreement\b",
        r"\bweakens? my previous mild opposition\b",
        r"\bmildly for fits? better\b",
        r"\bmildly for the claim fits? better\b",
    ]
    for pat in support_patterns:
        if re.search(pat, s, flags=re.I):
            return "support"

    oppose_patterns = [
        r"\bargues? against the claim\b",
        r"\bthe tweet argues? against the claim\b",
        r"\bopposes the claim\b",
        r"\bstrongly opposes the claim\b",
        r"\bevidence against the claim\b",
        r"\breason to doubt the claim\b",
        r"\bstrong enough for a small negative move\b",
        r"\bstrong enough to justify a small negative move\b",
        r"\bjustif(?:y|ies) a small negative move\b",
        r"\bpush(?:es)? me (?:slightly )?more negative\b",
        r"\bundermines the claim\b",
        r"\bcontradicts the claim\b",
        r"\blean(?:ing)? slightly against\b",
        r"\blean(?:ing)? against the claim\b",
        r"\bgives? me some reason to doubt\b",
        r"\bmakes? me less confident in the claim\b",
        r"\bundercuts? my confidence in the claim\b",
        r"\bmakes? me mildly doubtful\b",
        r"\blean(?:ing)? more negatively toward(?:s)? the claim\b",
        r"\blean(?:ing)? more negatively\b",
        r"\bweakens? my previous mild agreement\b",
        r"\bweakens? my previous view of being mildly for\b",
        r"\bmildly against fits? better\b",
        r"\bmildly against the claim fits? better\b",
        r"\bloss of .* introduces? uncertainty\b",
        r"\bmissing .* introduces? uncertainty\b",
        r"\blost .* introduces? uncertainty\b",
        r"\bmissing .* makes? .* hard(?:er)? to (?:verify|trust|check)\b",
        r"\blost .* makes? .* hard(?:er)? to (?:verify|trust|check)\b",
        r"\bhard(?:er)? to (?:verify|trust|check)\b",
    ]
    for pat in oppose_patterns:
        if re.search(pat, s, flags=re.I):
            return "oppose"

    return None


def validate_step3_output(text: str, pre_belief: int, claim: str = "", max_step_change: int = None, allowed_ratings=None):
    """
    Validate a listener Step-3 response against protocol and movement constraints.
    
        The validator checks for a legal FINAL_RATING, an explanation, membership in the allowed-rating set,
        and obvious semantic/protocol contradictions. It returns structured reasons so the caller can decide
        whether to accept, retry, repair, or fall back.
    """
    warnings = []

    if not text or not str(text).strip():
        return False, "empty", warnings

    s = str(text).strip()

    if _is_refusal_or_meta(s):
        return False, "refusal_or_meta", warnings

    if not re.search(r"(?im)^\s*FINAL(?:_| )RATING\s*:\s*[+-]?\d\b", s):
        return False, "missing FINAL_RATING header", warnings

    post = extract_belief(s)
    if post is None:
        return False, "missing/invalid FINAL_RATING value", warnings

    try:
        post_i = int(post)
    except Exception:
        return False, "missing/invalid FINAL_RATING value", warnings

    if post_i not in {-2, -1, 0, 1, 2}:
        return False, "FINAL_RATING out of range", warnings

    if allowed_ratings is not None:
        try:
            allowed_set = {int(x) for x in allowed_ratings}
        except Exception:
            allowed_set = None
        if allowed_set is not None and int(post_i) not in allowed_set:
            return False, f"not_in_allowed_set:{int(pre_belief)}->{int(post_i)}", warnings

    if max_step_change is not None:
        try:
            msc = int(max_step_change)
        except Exception:
            msc = None
        if msc is not None and msc >= 0:
            try:
                if abs(int(post_i) - int(pre_belief)) > msc:
                    return False, f"max_step_change_exceeded:{int(pre_belief)}->{int(post_i)}", warnings
            except Exception:
                pass

    if not re.search(r"(?im)^\s*EXPLANATION\s*:", s):
        return False, "missing EXPLANATION header", warnings

    expl = _extract_explanation_text(s)
    if not expl:
        return False, "empty EXPLANATION", warnings

    visible_meta_reason = _visible_planning_meta_reason(expl, field="explanation")
    if visible_meta_reason:
        return False, f"explanation_visible_meta:{visible_meta_reason}", warnings

    generic_reason = _generic_canned_reason_reason(expl, field="explanation")
    if generic_reason:
        return False, f"explanation_generic_canned:{generic_reason}", warnings

    sent_count = len([p for p in re.split(r"(?<=[.!?])\s+", str(expl).strip()) if p.strip()])
    if sent_count > 1:
        warnings.append("explanation_not_anchor_short")
    if len(str(expl).strip()) > 220:
        warnings.append("explanation_longer_than_expected")

    return True, "ok", warnings

def sanitize_step3_output(raw: str) -> str:
    """
    Normalize Step-3 output into the canonical FINAL_RATING/EXPLANATION shape.
    
        The sanitizer removes qwen/fence artifacts and accepts minor header spelling variants while keeping
        the final rating and explanation separable for downstream validation and CSV export.
    """
    if raw is None:
        return ""
    s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(raw)).strip()
    if not s:
        return ""

    # Bare 2-line rescue: first non-empty line is just the rating, second is the explanation.
    lines_raw = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines_raw:
        first = lines_raw[0]
        if re.fullmatch(r"[+-]?\d", first):
            try:
                r0 = int(first)
            except Exception:
                r0 = None
            if r0 in {-2, -1, 0, 1, 2}:
                tail = " ".join(lines_raw[1:]).strip()
                tail = re.sub(r"^(?:EXPLANATION\s*:\s*)", "", tail, flags=re.I).strip()
                if tail:
                    tail = re.sub(r"\s+", " ", tail).strip()
                    return f"FINAL_RATING: {r0}\nEXPLANATION: {tail}"

    # Drop anything before the first FINAL_RATING / FINAL RATING line.
    m = re.search(r"(?im)^\s*FINAL(?:_| )RATING\s*:\s*[+-]?\d\b", s)
    if m:
        s = s[m.start():].lstrip()

    # Normalize rating header token to FINAL_RATING:
    s = re.sub(r"(?im)^\s*FINAL\s+RATING\s*:", "FINAL_RATING:", s)

    # If EXPLANATION header is missing but there is text after FINAL_RATING, insert it.
    if not re.search(r"(?im)^\s*EXPLANATION\s*:", s):
        lines = [ln.rstrip() for ln in s.splitlines()]
        idx_fr = None
        for i, ln in enumerate(lines):
            if re.match(r"(?im)^\s*FINAL_RATING\s*:\s*[+-]?\d\b", ln):
                idx_fr = i
                break
        if idx_fr is not None:
            j = idx_fr + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                lines[j] = "EXPLANATION: " + lines[j].lstrip()
                s = "\n".join(lines)

    # Collapse multi-line explanation into one line for logs/parsing.
    expl = _extract_explanation_text(s)
    r = extract_belief(s)
    if r is not None and expl:
        expl_one = re.sub(r"\s+", " ", expl).strip()
        return f"FINAL_RATING: {int(r)}\nEXPLANATION: {expl_one}".strip()

    return s.strip()


def _resolve_invalid_step3_rating(proposed_rating, pre_belief: int, allowed_ratings) -> int:
    """Resolve an invalid Step-3 FINAL_RATING by clipping to the allowed boundary.

    Behavior:
      - If the proposed rating is already allowed, keep it.
      - If the proposal points upward relative to PREVIOUS_RATING, project to the
        highest allowed rating above PREVIOUS_RATING.
      - If the proposal points downward relative to PREVIOUS_RATING, project to the
        lowest allowed rating below PREVIOUS_RATING.
      - If there is no allowed move in that direction, keep PREVIOUS_RATING.

    Intuition:
      repeated illegal answers still carry directional information. If the model keeps
      trying to move farther than allowed, clip that move to the boundary of the allowed
      set instead of collapsing all the way back to stay.
    """
    try:
        pre = int(pre_belief)
    except Exception:
        pre = 0

    try:
        allowed = [int(x) for x in (allowed_ratings or [])]
    except Exception:
        allowed = []

    seen = set()
    allowed = [x for x in allowed if not (x in seen or seen.add(x))]

    if not allowed:
        return pre

    allowed_set = set(allowed)

    try:
        prop = int(proposed_rating)
    except Exception:
        return pre if pre in allowed_set else int(allowed[0])

    if prop in allowed_set:
        return prop

    if prop > pre:
        upward_allowed = [a for a in allowed if a > pre]
        if upward_allowed:
            return int(max(upward_allowed))
    elif prop < pre:
        downward_allowed = [a for a in allowed if a < pre]
        if downward_allowed:
            return int(min(downward_allowed))

    return pre if pre in allowed_set else int(allowed[0])



def _same_rating_cross_bin_wording_tag(pre_value: int, final_value: int, explanation_text: str) -> str | None:
    """Detect same-rating explanations that explicitly describe a different category.

    Narrow by design: only flag explicit cross-bin wording when the rating stayed
    the same. Mild within-bin weakening such as 'less confident' or 'slightly less
    sure' should pass.
    """
    try:
        pre_i = int(pre_value)
        final_i = int(final_value)
    except Exception:
        return None

    if final_i != pre_i:
        return None

    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not s:
        return None

    neutralish_patterns = [
        r"\bmoving to a neutral stance\b",
        r"\bmove(?:d|s|ing)? to a neutral stance\b",
        r"\bmove(?:d|s|ing)? toward(?:s)? neutral\b",
        r"\btoward(?:s)? a neutral stance\b",
        r"\bmore neutral now\b",
        r"\bnow neutral\b",
        r"\bnow unsure\b",
        r"\bbecoming unsure\b",
        r"\bmore unsure now\b",
        r"\bleaning more toward being unsure\b",
        r"\bleaning toward being unsure\b",
        r"\broughly balanced now\b",
        r"\bmixed now\b",
        r"\bfeel unsure overall\b",
    ]
    if final_i != 0:
        for pat in neutralish_patterns:
            if re.search(pat, s, flags=re.I):
                return "same_rating_cross_bin_wording"
    else:
        for pat in [r"\bshould move one step more positive\b", r"\bshould move one step more negative\b", r"\blean(?:ing)? more positive\b", r"\blean(?:ing)? more negative\b"]:
            if re.search(pat, s, flags=re.I):
                return "same_rating_cross_bin_wording"

    if final_i > 0:
        opposite_patterns = [
            r"\blean(?:ing)? slightly against\b",
            r"\blean(?:ing)? against the claim\b",
            r"\bnow against the claim\b",
            r"\bmore skeptical now\b",
            r"\bsomewhat reject\b",
            r"\bdoubt the claim\b",
            r"\bshould move one step more negative\b",
            r"\bshould move one step toward 0\b",
            r"\bshould move toward a more negative stance\b",
            r"\bthis pushes me toward 0\b",
            r"\bthis moves me toward 0\b",
        ]
        for pat in opposite_patterns:
            if re.search(pat, s, flags=re.I):
                return "same_rating_cross_bin_wording"
    elif final_i < 0:
        opposite_patterns = [
            r"\blean(?:ing)? slightly in favor\b",
            r"\blean(?:ing)? toward believing\b",
            r"\bmore inclined to believe\b",
            r"\bnow supportive\b",
            r"\bsomewhat accept\b",
            r"\bbelieve the claim\b",
            r"\bshould move one step more positive\b",
            r"\bshould move one step toward 0\b",
            r"\bshould move toward a more positive stance\b",
            r"\bthis pushes me toward 0\b",
            r"\bthis moves me toward 0\b",
        ]
        for pat in opposite_patterns:
            if re.search(pat, s, flags=re.I):
                return "same_rating_cross_bin_wording"

    return None




def _moved_rating_stale_category_wording_tag(pre_value: int, final_value: int, explanation_text: str) -> str | None:
    """Detect moved-rating explanations that still describe the old or wrong category.

    Example: FINAL_RATING=1 but the explanation still says 'strong belief'.
    """
    try:
        pre_i = int(pre_value)
        final_i = int(final_value)
    except Exception:
        return None

    if final_i == pre_i:
        return None

    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip().lower()
    if not s:
        return None

    # Ignore wording that is clearly describing the previous state or the transition itself.
    s = _strip_transition_context_for_state_checks(s).lower()
    if not s:
        return None

    stale_patterns_by_final = {
        2: [
            r"\bunsure\b", r"\bneutral\b", r"\bmixed\b", r"\blean(?:ing)? against\b",
            r"\bsomewhat reject\b", r"\bstrongly reject\b",
        ],
        1: [
            r"\bstrong belief\b", r"\bstrongly believe\b", r"\bfirmly believe\b",
            r"\bcertain it happened\b", r"\bstrongly accept\b", r"\bmoving to a neutral stance\b",
            r"\bnow unsure\b", r"\bneutral\b", r"\bmixed\b",
            r"\blean(?:ing)? against\b", r"\bstrongly reject\b",
        ],
        0: [
            r"\bstrong belief\b", r"\bstrongly believe\b", r"\bfirmly believe\b",
            r"\bstill lean(?:ing)? toward\b", r"\bsomewhat believe\b",
            r"\blean(?:ing)? against\b", r"\bsomewhat reject\b", r"\bstrongly reject\b",
        ],
        -1: [
            r"\bstrong disbelief\b", r"\bstrongly reject\b", r"\bfirmly reject\b",
            r"\bstrongly against\b", r"\bneutral\b", r"\bnow unsure\b",
            r"\bmixed\b", r"\blean(?:ing)? in favor\b", r"\bsomewhat believe\b",
            r"\bstrongly believe\b",
        ],
        -2: [
            r"\bunsure\b", r"\bneutral\b", r"\bmixed\b", r"\blean(?:ing)? in favor\b",
            r"\bsomewhat believe\b", r"\bstrongly believe\b",
        ],
    }
    for pat in stale_patterns_by_final.get(final_i, []):
        if re.search(pat, s, flags=re.I):
            return "moved_rating_stale_category_wording"
    return None


def _split_step3_explanation_clauses(expl: str) -> list[str]:
    """Split a Step-3 explanation into candidate reason clauses."""
    s = re.sub(r"\s+", " ", str(expl or "")).strip()
    if not s:
        return []
    parts = re.split(r"\s*(?:,\s+but\s+|\s+but\s+|\s+though\s+|\s+although\s+|\s+however,?\s+|;\s+|,\s+so\s+|\s+so\s+)", s, flags=re.I)
    return [p.strip(" ,.;:-") for p in parts if p and p.strip(" ,.;:-")]


def _step3_clause_is_category_like(clause: str) -> bool:
    """Detect clauses that are only category labels rather than actual reasons."""
    s = re.sub(r"\s+", " ", str(clause or "")).strip().lower()
    if not s:
        return False
    category_patterns = [
        r"\bstrong belief\b", r"\bstrongly believe\b", r"\bfirmly believe\b",
        r"\bstrong disbelief\b", r"\bstrongly reject\b", r"\bfirmly reject\b",
        r"\bnot enough to change my\b", r"\bchange my current rating\b",
        r"\bcurrent rating\b", r"\bmoving to a neutral stance\b", r"\bneutral stance\b",
        r"\bnow neutral\b", r"\bnow unsure\b", r"\bmore neutral\b", r"\bmixed\b",
        r"\blean(?:ing)? toward\b", r"\blean(?:ing)? against\b", r"\bsomewhat believe\b",
        r"\bsomewhat reject\b", r"\bstrongly accept\b", r"\bstrongly against\b",
        r"\bless confident\b", r"\bless sure\b", r"\bmore confident\b", r"\bweaker\b",
    ]
    return any(re.search(p, s, flags=re.I) for p in category_patterns)


def _step3_clause_reason_score(clause: str) -> int:
    """Score an explanation clause for usefulness as a preserved reason."""
    s = re.sub(r"\s+", " ", str(clause or "")).strip()
    low = s.lower()
    if not s:
        return -999
    score = 0
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", s)
    score += min(len(words), 12)
    if _step3_clause_is_category_like(s):
        score -= 5
    reason_patterns = [
        r"\bbecause\b", r"\bconcern\b", r"\bdetails?\b", r"\bevidence\b",
        r"\black of\b", r"\bdoes(?: not|n't) explain\b", r"\binconsistent\b",
        r"\bcontradict\b", r"\bvalid concern\b", r"\braises? doubt\b",
        r"\bsupports?\b", r"\bexplains?\b", r"\bcredible\b", r"\breported\b",
        r"\blogic\b", r"\bcoherent\b", r"\bplausible\b", r"\bimplausible\b",
        r"\bphotos?\b", r"\bimages?\b", r"\brecords?\b", r"\btimeline\b",
    ]
    for pat in reason_patterns:
        if re.search(pat, low, flags=re.I):
            score += 4
    if len(words) >= 5:
        score += 2
    return score


def _best_reason_clause_from_explanation(expl: str) -> str | None:
    """Select the best tweet-grounded reason clause from an explanation."""
    clauses = _split_step3_explanation_clauses(expl)
    if not clauses:
        return None
    scored = sorted((( _step3_clause_reason_score(c), c) for c in clauses), reverse=True)
    best_score, best_clause = scored[0]
    if best_score < 4:
        return None
    return best_clause.strip(' ,.;:-')


def _strip_transition_context_for_state_checks(expl: str) -> str:
    """Remove transition scaffolding before checking semantic state-language markers."""
    s = re.sub(r"\s+", " ", str(expl or "")).strip()
    if not s:
        return ""
    patterns = [
        r"\bfrom my previous [^,.;:]*",
        r"\bfrom my old [^,.;:]*",
        r"\bfrom my prior [^,.;:]*",
        r"\bfrom my current [^,.;:]*",
        r"\bfrom a previous [^,.;:]*",
        r"\bfrom an? earlier [^,.;:]*",
        r"\bmy previous [^,.;:]*",
        r"\bmy old [^,.;:]*",
        r"\bmy prior [^,.;:]*",
        r"\brather than (?:staying|remaining|being) [^,.;:]*",
        r"\binstead of (?:staying|remaining|being) [^,.;:]*",
        r"\bno longer strongly believe\b",
        r"\bno longer firmly believe\b",
        r"\bno longer strongly reject\b",
        r"\bno longer firmly reject\b",
        r"\bthan my previous [^,.;:]*",
    ]
    for pat in patterns:
        s = re.sub(pat, " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()





def get_step3_llm_response(conversation, prompt: str, pre_belief: int, add_to_memory: bool, agent_name: str = None, allowed_ratings=None, speaker_belief: int | None = None):
    """
    Run the full Step-3 listener update pipeline.
    
        The function calls the LLM, sanitizes and validates the response, applies one or more repair/rescue
        paths when protocol or semantic checks fail, enforces the allowed-rating set, logs repair tags, and
        returns the finalized listener update used to change the agent's belief.
    """
    _set_memory_step_kind(conversation, "step3")
    """Step-3 call — agent decision is authoritative.

    Returns:
        (final_text, raw_attempt_1, raw_attempt_2, repair_tags)

    Policy:
        - FINAL_RATING chosen by the LLM is accepted as-is if it is inside allowed_set.
        - Soft explanation/style issues are warning-only when the rating is legal.
        - If attempt 1 fails to produce a legal rating, fall back deterministically instead of running a long repair cascade.
        - Detection functions run mainly for logging/tagging only; hard protocol issues may still trigger a fallback.
        - No projection or clarification calls; keep cheap deterministic cleanup only when needed for rating/explanation alignment.
    """
    repair_tags = []

    # ------------------------------------------------------------------ helpers

    def _extract_claim_from_prompt(p: str) -> str:
        """
        Extract the active claim from the Step-3 prompt text.
        
            The local Step-3 repair code uses the claim when building compact rescue prompts or deterministic
            explanations without depending on outer-scope globals.
        """
        if not p:
            return ""
        m = re.search(r"(?im)^\s*CLAIM\s*\(.*?\)\s*:\s*(.+?)\s*$", str(p))
        if not m:
            m = re.search(r"(?im)^\s*CLAIM\s*:\s*(.+?)\s*$", str(p))
        if not m:
            return ""
        return (m.group(1) or "").strip()

    def _allowed_set(pre: int):
        """Fallback allowed set for the local validator when the caller passed no
        allowed_ratings: plain +-max_step_change neighborhood, deliberately
        mode-agnostic (no not-away filter). The real pipeline always passes the
        set from _compute_allowed_ratings, which enforces the update mode."""
        pre = int(pre)
        try:
            msc = int(DEFAULT_MAX_STEP_CHANGE) if DEFAULT_MAX_STEP_CHANGE is not None else 1
        except Exception:
            msc = 1
        msc = max(0, msc)
        allowed = set()
        for d in range(-msc, msc + 1):
            allowed.add(max(-2, min(2, pre + d)))
        return sorted(allowed)

    def _parse_rating(step3_text: str):
        """Parse a FINAL_RATING value from local Step-3 output text."""
        try:
            r = extract_belief(step3_text)
            return int(r) if r is not None else None
        except Exception:
            return None

    def _ensure_has_explanation(cleaned_text: str, rating_hint):
        """Ensure local Step-3 text contains an EXPLANATION field before validation."""
        if cleaned_text:
            r = _parse_rating(cleaned_text)
            expl = _extract_explanation_text(cleaned_text)
            if r is not None and str(expl or "").strip():
                return _format_step3_output_preserve_explanation(final_rating=int(r), explanation=(expl or ""))
            if r is not None:
                return _rewrite_step3_output_with_explanation(final_rating=int(r), explanation="")
        if rating_hint is not None:
            return _rewrite_step3_output_with_explanation(final_rating=int(rating_hint), explanation="")
        return str(cleaned_text or "").strip()

    _STEP3_SOFT_WARNING_ONLY_TAGS = {
        "generic_strong_evidence",
        "generic_overwhelming_evidence",
        "generic_existing_skepticism",
        "generic_fact_pack_reference",
        "generic_common_skeptical_claim",
        "generic_valid_evidence_but",
        "generic_claim_credibility",
        "generic_does_not_outweigh",
        "generic_not_addressed_by_fact_pack",
        "generic_tweet_point_formula",
        "generic_known_facts_reference",
        "generic_overall_evidence",
        "generic_broader_context",
        "generic_shown_above_reference",
        "generic_whole_case_reference",
        "generic_entire_case_reference",
        "generic_web_alignment",
        "generic_verified_historical_data",
        "generic_historical_data_reference",
        "generic_documented_moon_landings",
        "generic_confirms_claim",
        "generic_documented_claim",
        "tweet_explanation_polarity_contradiction",
        "moved_rating_stale_category_wording",
        "same_rating_cross_bin_wording",
        "explanation_meta_visible_planning",
        "explanation_canned_formula",
        "mentions_hidden_state",
        "mentions_current_rating",
    }

    def _is_step3_soft_warning_only_meta(meta_text: str | None) -> bool:
        """Return whether local warnings are soft enough to accept the Step-3 output."""
        m = str(meta_text or "").strip().lower()
        return m in _STEP3_SOFT_WARNING_ONLY_TAGS

    def _step3_explanation_on_topic(expl_text: str, tweet_text: str = "") -> bool:
        """Check whether a Step-3 explanation is grounded in the tweet, claim, or shown material."""
        expl = re.sub(r"\s+", " ", str(expl_text or "")).strip()
        tw = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
        if not expl:
            return False
        expl_toks = set(_fact_match_tokens(expl))
        tw_toks = set(_fact_match_tokens(tw))
        shared_digits = {tok for tok in (expl_toks & tw_toks) if tok.isdigit()}
        shared_non_generic = {tok for tok in (expl_toks & tw_toks) if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
        if shared_digits or len(shared_non_generic) >= 1:
            return True
        # Fallback: if both texts mention the same clearly named Apollo evidence motif, treat as on-topic.
        motifs = [
            "retroreflector", "retroreflectors", "lunar", "samples", "sample", "rocks", "moon rocks",
            "stars", "flag", "shadows", "shadow", "tapes", "sstv", "tracks", "imagery",
            "telemetry", "communications", "ground stations", "ground station", "experiments",
            "surveyor", "orbiter", "van", "allen"
        ]
        expl_low = expl.lower(); tw_low = tw.lower()
        return any(m in expl_low and m in tw_low for m in motifs)

    def _has_hard_step3_issue(reason_text, warning_list, meta_text) -> bool:
        """Return whether local validation tags include a hard Step-3 issue."""
        joined = " | ".join([str(reason_text or "")] + [str(w or "") for w in (warning_list or [])] + [str(meta_text or "")]).lower()
        hard_markers = [
            "not_in_allowed_set",
            "missing_final_rating",
            "missing_explanation",
            "empty_explanation",
            "direction",
            "projection",
            "boundary",
            "cross_bin",
            "true_open_state_based_explanation",
            "true_open_generic_claim_verdict",
        ]
        return any(tok in joined for tok in hard_markers)

    def _has_only_hidden_state_family(reason_text, warning_list, meta_text) -> bool:
        """Check whether local warnings are limited to hidden-state wording."""
        items = [str(reason_text or "")] + [str(w or "") for w in (warning_list or [])] + [str(meta_text or "")]
        tokens = set()
        for item in items:
            s = str(item or "").strip().lower()
            if not s or s in {"ok", "none"}:
                continue
            if "mentions_hidden_state" in s:
                tokens.add("mentions_hidden_state")
            else:
                tokens.add(s)
        return bool(tokens) and tokens == {"mentions_hidden_state"}

    def _soft_accept_hidden_state_family(rating_value, cleaned_text):
        """Soft-accept outputs whose only defect is acceptable hidden-state phrasing."""
        return _soft_accept_attempt1_text(rating_value, cleaned_text, 'mentions_hidden_state')

    def _soft_accept_attempt1_text(rating_value, cleaned_text, meta_text):
        """Accept a first-attempt Step-3 output when only benign warning tags remain."""
        try:
            rating_i = int(rating_value)
        except Exception:
            return ""
        expl = _extract_explanation_text(cleaned_text) or ""
        if not expl.strip():
            return ""
        meta_s = str(meta_text or "").strip()
        soft_meta_tags = {'mentions_current_rating', 'mentions_hidden_state'}
        if meta_s and meta_s not in soft_meta_tags and not _is_step3_soft_warning_only_meta(meta_s):
            return ""
        cleaned_expl = re.sub(r"\s+", " ", str(expl or "")).strip(" ,.;:-")
        if not cleaned_expl.strip():
            return ""
        return _format_step3_output_preserve_explanation(final_rating=rating_i, explanation=cleaned_expl)

    def _clean_true_open_local_rewrite_explanation(expl_text: str) -> str:
        """Clean a true-open local rewrite explanation before insertion."""
        s = re.sub(r"\s+", " ", str(expl_text or "")).strip()
        if not s:
            return ""
        repls = [
            (r"(?i)\bkeeps? the rating mixed\b", "still leaves things mixed"),
            (r"(?i)\bkeep(?:ing)? the same rating\b", "leave things where they are"),
            (r"(?i)\bjustify a bigger move\b", "justify a stronger shift here"),
            (r"(?i)\bchange my rating\b", "change things here"),
            (r"(?i)\bcurrent balance after this tweet\b", "specific point in this tweet"),
            (r"(?i)\bremaining doubt after this tweet\b", "remaining doubt here"),
        ]
        for pat, rep in repls:
            s = re.sub(pat, rep, s)
        s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
        s = _sentence_case_cleanup(s)
        return s

    def _rewrite_true_open_meta_to_tweet_local(rating_value, meta_text=None, reason_text=None, warning_list=None):
        """Rewrite true-open meta explanations into tweet-local explanations."""
        if _normalize_world_mode(WORLD) != 'true_open':
            return ""
        joined = " | ".join(
            [str(meta_text or ""), str(reason_text or "")] + [str(w or "") for w in (warning_list or [])]
        ).lower()
        trigger_tokens = {
            'true_open_state_based_explanation',
            'true_open_generic_claim_verdict',
            'mentions_current_rating',
        }
        if not any(tok in joined for tok in trigger_tokens):
            return ""
        try:
            rating_i = int(rating_value)
        except Exception:
            return ""
        local_expl = _step3_tweet_local_explanation(tweet_txt, rating_i, int(pre_belief), tweet_stance)
        local_expl = _clean_true_open_local_rewrite_explanation(local_expl)
        if not local_expl:
            return ""
        return _rewrite_step3_output_with_explanation(final_rating=rating_i, explanation=local_expl)

    def _rewrite_step3_explanation_only_for_polarity(rating_value, explanation_text: str = "") -> str:
        """Rewrite only the explanation when polarity, not rating, needs repair."""
        try:
            rating_i = int(rating_value)
        except Exception:
            return ""
        local_expl = _step3_tweet_local_explanation(tweet_txt, rating_i, int(pre_belief), tweet_stance)
        if not local_expl:
            local_expl = _canonical_step3_explanation_from_tweet(tweet_stance, int(pre_belief), rating_i)
        if _normalize_world_mode(WORLD) == 'true_open':
            local_expl = _clean_true_open_local_rewrite_explanation(local_expl)
        else:
            local_expl = _sentence_case_cleanup(re.sub(r"\s+", " ", str(local_expl or "")).strip(" ,.;:-"))
        if not local_expl:
            return ""
        rewritten = _rewrite_step3_output_with_explanation(final_rating=rating_i, explanation=local_expl)
        if _would_semantically_invert(rating_i, explanation_text, rewritten, tweet_stance):
            return ""
        return rewritten

    def _explanation_claim_direction(expl_text: str) -> str:
        """Classify a local explanation as claim-supporting or claim-opposing."""
        s = re.sub(r"\s+", " ", str(expl_text or "")).strip().lower()
        if not s:
            return "unknown"
        pro_patterns = [
            r"\bsupports? the claim\b",
            r"\bstrengthens? the case for the claim\b",
            r"\bstrengthens? the claim\b",
            r"\bgives? (?:some )?support for the claim\b",
            r"\breinforces? the claim\b",
            r"\brefutes? (?:the )?(?:challenge|objection|uncertainty|doubt)\b",
            r"\bcounters? (?:the )?(?:challenge|objection|uncertainty|doubt)\b",
            r"\bcontradicted by\b",
            r"\bexplained by\b",
            r"\bevidence (?:for|supporting) (?:the )?(?:claim|landings?|missions?)\b",
            r"\bphysical evidence supporting\b",
            r"\bthird-party evidence supporting\b",
            r"\bpush(?:es|ed)? me (?:slightly )?(?:more )?positive\b",
            r"\bnudg(?:es|ed)? me (?:slightly )?(?:more )?positive\b",
            r"\bjustif(?:y|ies) moving (?:slightly )?(?:more )?positive\b",
            r"\bmakes? the claim more plausible\b",
            r"\bmore plausible overall\b",
        ]
        anti_patterns = [
            r"\bargues? against the claim\b",
            r"\bchallenges? the claim\b",
            r"\bweakens? the claim\b",
            r"\bundermines? the claim\b",
            r"\braises? doubts? about the claim\b",
            r"\bcasts? doubt on the claim\b",
            r"\bkeeps? the claim uncertain\b",
            r"\bleaves? the claim uncertain\b",
            r"\bquestions? the claim(?:'s)? validity\b",
            r"\bpush(?:es|ed)? me (?:slightly )?(?:more )?negative\b",
            r"\bnudg(?:es|ed)? me (?:slightly )?(?:more )?negative\b",
            r"\bjustif(?:y|ies) moving (?:slightly )?(?:more )?negative\b",
            r"\bmakes? the claim less plausible\b",
            r"\bless plausible overall\b",
            r"\bdoes(?: not|n't) justify moving (?:slightly )?(?:more )?positive\b",
        ]
        if any(re.search(p, s, flags=re.I) for p in pro_patterns):
            return "support"
        if any(re.search(p, s, flags=re.I) for p in anti_patterns):
            return "oppose"
        return "unknown"

    def _rating_direction_label(rating_value: int | None) -> str:
        """Map a local rating to a support/oppose/neutral label."""
        try:
            r = int(rating_value)
        except Exception:
            return "unknown"
        if r > 0:
            return "support"
        if r < 0:
            return "oppose"
        return "uncertain"

    def _would_semantically_invert(raw_rating: int, raw_expl: str, candidate_text: str, tweet_stance_local: str | None = None) -> bool:
        """Detect whether a repair would invert the intended semantic direction."""
        raw_dir = _explanation_claim_direction(raw_expl)
        candidate_expl = _extract_explanation_text(candidate_text) or ""
        candidate_dir = _explanation_claim_direction(candidate_expl)
        rating_dir = _rating_direction_label(raw_rating)
        if raw_dir in {"support", "oppose"} and candidate_dir in {"support", "oppose"} and raw_dir != candidate_dir:
            return True
        if candidate_dir in {"support", "oppose"} and rating_dir in {"support", "oppose"} and candidate_dir != rating_dir:
            return True
        if raw_dir == "unknown" and tweet_stance_local in {"support", "oppose"} and candidate_dir in {"support", "oppose"}:
            if candidate_dir != tweet_stance_local and tweet_stance_local == rating_dir:
                return True
        return False

    def _attempt(prompt_to_use: str, attempt_id):
        """Run one local LLM attempt and return sanitized text plus validation details."""
        _set_current_validation_context(prompt_to_use)
        _set_native_ollama_debug_context(step='step3', agent_name=agent_name, attempt=attempt_id, label='attempt')
        stateless_retry = str(attempt_id) == '1b'
        try:
            raw_text = get_llm_response(
                conversation,
                prompt_to_use,
                use_history=not stateless_retry,
                save_turn=not stateless_retry,
            )
        finally:
            _clear_native_ollama_debug_context()
        cleaned = sanitize_step3_output(raw_text)
        r = _parse_rating(cleaned)
        expl = _extract_explanation_text(cleaned)
        if r is not None:
            if str(expl or "").strip():
                cleaned = _format_step3_output_preserve_explanation(final_rating=int(r), explanation=(expl or ""))
            else:
                expl = _heuristic_freeform_protocol_body(raw_text, field="explanation")
                if str(expl or "").strip():
                    cleaned = _format_step3_output_preserve_explanation(final_rating=int(r), explanation=(expl or ""))
        cleaned = _ensure_has_explanation(cleaned, r)
        expl_final = _extract_explanation_text(cleaned) or ""
        if r is not None:
            only_meta_force = _visible_planning_meta_reason(expl_final, field="explanation")
            if only_meta_force:
                local_expl = _step3_tweet_local_explanation(tweet_txt, int(r), int(pre_belief), tweet_stance)
                if local_expl:
                    cleaned = _rewrite_step3_output_with_explanation(final_rating=int(r), explanation=local_expl)
                    expl_final = _extract_explanation_text(cleaned) or local_expl
        raw_rating_before_guard = int(r) if r is not None else None
        raw_expl_before_guard = str(expl_final or "")
        weak_reason_flag = no_move_if_weak_reason(tweet_txt)
        generic_gap_reason = _step3_generic_gap_reason(expl_final, tweet_text=tweet_txt, pre_value=int(pre_belief), final_value=(int(r) if r is not None else None), tweet_stance_local=tweet_stance)
        open_world_generic_reason = _step3_open_world_generic_reference_reason(expl_final, tweet_text=tweet_txt) if _is_open_world_mode(WORLD) else None
        true_open_point_mismatch_reason = _step3_true_open_point_mismatch_reason(expl_final, tweet_text=tweet_txt, prompt_text=prompt) if _normalize_world_mode(WORLD) == "true_open" else None
        strict_closed_external_reason = _step3_strict_closed_external_reason_reason(expl_final, tweet_txt, prompt_text=prompt_to_use) if _normalize_world_mode(WORLD) in {"closed_strict", "closed_strict_rag"} else None
        meta_reason = (_visible_planning_meta_reason(expl_final, field="explanation") or _generic_canned_reason_reason(expl_final, field="explanation") or true_open_point_mismatch_reason or open_world_generic_reason or generic_gap_reason or (_same_rating_generic_fact_dismissal_tag(int(pre_belief), int(r), tweet_stance, tweet_txt, expl_final) if r is not None else None))
        if r is not None:
            guarded_r, guard_tag = _apply_tweet_based_step3_guard(
                int(pre_belief),
                int(r),
                tweet_stance,
                weak_reason_flag,
                allowed_ratings=allowed_list,
                explanation_text=expl_final,
                tweet_text=tweet_txt,
                bias_text=BIAS_TEXT,
            )
            if guard_tag:
                repaired_expl = _canonical_step3_explanation(guard_tag, int(pre_belief), int(guarded_r), expl_final)
                candidate_cleaned = _rewrite_step3_output_with_explanation(final_rating=int(guarded_r), explanation=repaired_expl)
                if raw_rating_before_guard is not None and _would_semantically_invert(raw_rating_before_guard, raw_expl_before_guard, candidate_cleaned, tweet_stance):
                    repair_tags.append("step3_guard_blocked_semantic_inversion")
                else:
                    cleaned = candidate_cleaned
                    r = int(guarded_r)
                    expl_final = _extract_explanation_text(cleaned) or repaired_expl
                    if not meta_reason:
                        meta_reason = guard_tag
        ok, val_reason, val_warnings = validate_step3_output(
            cleaned,
            pre_belief=int(pre_belief),
            claim=claim_txt,
            max_step_change=DEFAULT_MAX_STEP_CHANGE,
            allowed_ratings=allowed_list,
        )
        if strict_closed_external_reason:
            _metric_inc('closed_world_leak_step3')
            val_warnings = list(val_warnings or []) + [strict_closed_external_reason]
        return r, cleaned, raw_text, meta_reason, ok, val_reason, val_warnings

    def _compact_step3_rescue_prompt(*, compact: bool = False, fixed_rating: int | None = None, original_explanation: str = "") -> str:
        """Build the compact local rescue prompt for a failed Step-3 attempt."""
        local_claim = str(claim_txt or "").strip()
        local_tweet = _strip_leaked_prompt_headers(str(tweet_txt or "").strip())
        shown_material = _extract_compact_step3_shown_material(prompt, max_items=(1 if compact else 2), max_chars_per_item=(180 if compact else 240))
        left_clause, right_clause = _step3_extract_tweet_local_clauses(local_tweet)
        support_clause = left_clause if _step3_tweet_clause_direction(left_clause) == 'support' else (right_clause if _step3_tweet_clause_direction(right_clause) == 'support' else '')
        oppose_clause = left_clause if _step3_tweet_clause_direction(left_clause) == 'oppose' else (right_clause if _step3_tweet_clause_direction(right_clause) == 'oppose' else '')
        mixed_local_tweet = bool(support_clause and oppose_clause)
        lines = [
            "/no_think",
            f"CLAIM: {local_claim}",
            f"PRESENT VIEW: {int(pre_belief)}",
            f"ALLOWED_FINAL_RATING_SET: {allowed_str}",
            "",
            "TWEET:",
            local_tweet,
        ]
        if shown_material:
            lines.extend(["", shown_material])
        strict_closed_now = _normalize_world_mode(WORLD) in {"closed_strict", "closed_strict_rag"}
        if compact:
            try:
                fixed_i = int(fixed_rating) if fixed_rating is not None else None
            except Exception:
                fixed_i = None
            if fixed_i is not None:
                lines.extend([
                    "",
                    f"Keep FINAL_RATING fixed at {fixed_i}.",
                    "Only repair EXPLANATION; do not choose a different rating.",
                    "Use one concrete point from the tweet or shown material only.",
                    "The explanation must fit the fixed rating and any bounded one-step movement from PRESENT VIEW.",
                    "Do not introduce alternative explanations, credibility doubts, or outside-source issues unless those words appear above.",
                ])
                if str(original_explanation or "").strip():
                    lines.extend(["Original explanation to improve:", str(original_explanation or "").strip()])
                if mixed_local_tweet:
                    lines.extend([
                        f"Supportive point in this tweet: {support_clause}",
                        f"Unresolved gap in this tweet: {oppose_clause}",
                    ])
                if fixed_i == 0:
                    lines.append("Because FINAL_RATING is 0, mention both a support point and an unresolved point only if both are present in the tweet or shown material.")
                    lines.append("If only one side is present, do not invent the other side; say the point is not enough to settle the claim.")
                lines.extend([
                    "Output exactly 2 lines and nothing else.",
                    f"FINAL_RATING: {fixed_i}",
                    "EXPLANATION: <one short sentence>",
                ])
            else:
                lines.extend([
                    "",
                    "Choose one allowed final rating and explain it in one sentence.",
                    "Use only a concrete point from the tweet or shown material.",
                    "If FINAL_RATING is 0, mention both a support point and an unresolved point when both are present.",
                    "Do not introduce alternative explanations, credibility doubts, or outside-source issues unless those words appear above.",
                ])
                if mixed_local_tweet:
                    lines.extend([
                        f"Supportive point in this tweet: {support_clause}",
                        f"Unresolved gap in this tweet: {oppose_clause}",
                    ])
                lines.extend([
                    "Output exactly 2 lines and nothing else.",
                    f"FINAL_RATING: <one of {allowed_str}>",
                    "EXPLANATION: <one short sentence>",
                ])
        else:
            lines.extend([
                "",
                "Task: give your current honest belief about the claim after seeing this tweet.",
                "Base the decision mainly on this tweet, not on a fresh overall evaluation of the whole claim.",
                "Apply the same standard of credibility, specificity, clarity, and claim-relevance to both sides.",
                "Use shown material only when it directly clarifies this tweet's point.",
                "First compare the new tweet with the best reason for your current rating.",
                ("The allowed set is the only movement boundary; a change may move either direction if justified." if _is_free_bounded_update_mode() else "A change should follow the tweet's direction unless there is a clear reason to stay."),
                "If you keep the same rating, EXPLANATION must name the exact weakness or limitation that stopped a move.",
                "If FINAL_RATING is 0, EXPLANATION must include one point with some force and one unresolved point in one short sentence.",
            ])
            if mixed_local_tweet:
                lines.extend([
                    f"Supportive point in this tweet: {support_clause}",
                    f"Unresolved gap in this tweet: {oppose_clause}",
                    "Do not let the supportive wording erase the unresolved gap, and do not answer the gap with a different outside-source claim.",
                ])
            if strict_closed_now:
                lines.extend([
                    "In strict closed world, keep the explanation tweet-local.",
                    "Treat broad appeals to agreement, evidence, sources, or credibility as unsupported unless the exact support is shown above.",
                    "If the tweet uses broad evidence, proof, consensus, or source language without one specific local point, explain that weakness directly.",
                    "Do not answer one outside-source appeal with a different outside-source appeal.",
                ])
            lines.extend([
                "Do not write analysis, bullets, or extra text.",
                "Output exactly 2 lines and nothing else.",
                f"FINAL_RATING: <one of {allowed_str}>",
                "EXPLANATION: <one short sentence tied to one concrete point>",
            ])
        return "\n".join(str(x) for x in lines if x is not None).strip()

    def _native_rescue_attempt(prompt_to_use: str, attempt_id, *, include_history: bool = True, compact: bool = False):
        """Run a native-Ollama rescue attempt for the current Step-3 failure."""
        _set_current_validation_context(prompt_to_use)
        label_name = 'native_rescue_compact' if compact else 'native_rescue'
        _set_native_ollama_debug_context(step='step3', agent_name=agent_name, attempt=attempt_id, label=label_name)
        try:
            raw_text = _native_ollama_chat(
                conversation,
                prompt_to_use,
                include_history=include_history,
                save_turn=False,
            )
        finally:
            _clear_native_ollama_debug_context()
        meta = dict(getattr(conversation, '_last_native_ollama_meta', {}) or {})
        cleaned = sanitize_step3_output(raw_text)
        r = _parse_rating(cleaned)
        expl = _extract_explanation_text(cleaned)
        if r is not None:
            if str(expl or "").strip():
                cleaned = _format_step3_output_preserve_explanation(final_rating=int(r), explanation=(expl or ""))
            else:
                expl = _heuristic_freeform_protocol_body(raw_text, field="explanation")
                if str(expl or "").strip():
                    cleaned = _format_step3_output_preserve_explanation(final_rating=int(r), explanation=(expl or ""))
        cleaned = _ensure_has_explanation(cleaned, r)
        expl_final = _extract_explanation_text(cleaned) or ""
        if r is not None:
            only_meta_force = _visible_planning_meta_reason(expl_final, field="explanation")
            if only_meta_force:
                local_expl = _step3_tweet_local_explanation(tweet_txt, int(r), int(pre_belief), tweet_stance)
                if local_expl:
                    cleaned = _rewrite_step3_output_with_explanation(final_rating=int(r), explanation=local_expl)
                    expl_final = _extract_explanation_text(cleaned) or local_expl
        raw_rating_before_guard = int(r) if r is not None else None
        raw_expl_before_guard = str(expl_final or "")
        weak_reason_flag = no_move_if_weak_reason(tweet_txt)
        generic_gap_reason = _step3_generic_gap_reason(expl_final, tweet_text=tweet_txt, pre_value=int(pre_belief), final_value=(int(r) if r is not None else None), tweet_stance_local=tweet_stance)
        open_world_generic_reason = _step3_open_world_generic_reference_reason(expl_final, tweet_text=tweet_txt) if _is_open_world_mode(WORLD) else None
        true_open_point_mismatch_reason = _step3_true_open_point_mismatch_reason(expl_final, tweet_text=tweet_txt, prompt_text=prompt) if _normalize_world_mode(WORLD) == "true_open" else None
        strict_closed_external_reason = _step3_strict_closed_external_reason_reason(expl_final, tweet_txt, prompt_text=prompt_to_use) if _normalize_world_mode(WORLD) in {"closed_strict", "closed_strict_rag"} else None
        meta_reason = (_visible_planning_meta_reason(expl_final, field="explanation") or _generic_canned_reason_reason(expl_final, field="explanation") or true_open_point_mismatch_reason or open_world_generic_reason or generic_gap_reason or (_same_rating_generic_fact_dismissal_tag(int(pre_belief), int(r), tweet_stance, tweet_txt, expl_final) if r is not None else None))
        if r is not None:
            guarded_r, guard_tag = _apply_tweet_based_step3_guard(
                int(pre_belief),
                int(r),
                tweet_stance,
                weak_reason_flag,
                allowed_ratings=allowed_list,
                explanation_text=expl_final,
                tweet_text=tweet_txt,
                bias_text=BIAS_TEXT,
            )
            if guard_tag:
                repaired_expl = _canonical_step3_explanation(guard_tag, int(pre_belief), int(guarded_r), expl_final)
                candidate_cleaned = _rewrite_step3_output_with_explanation(final_rating=int(guarded_r), explanation=repaired_expl)
                if raw_rating_before_guard is not None and _would_semantically_invert(raw_rating_before_guard, raw_expl_before_guard, candidate_cleaned, tweet_stance):
                    repair_tags.append("step3_guard_blocked_semantic_inversion")
                else:
                    cleaned = candidate_cleaned
                    r = int(guarded_r)
                    expl_final = _extract_explanation_text(cleaned) or repaired_expl
                    if not meta_reason:
                        meta_reason = guard_tag
        ok, val_reason, val_warnings = validate_step3_output(
            cleaned,
            pre_belief=int(pre_belief),
            claim=claim_txt,
            max_step_change=DEFAULT_MAX_STEP_CHANGE,
            allowed_ratings=allowed_list,
        )
        if strict_closed_external_reason:
            _metric_inc('closed_world_leak_step3')
            val_warnings = list(val_warnings or []) + [strict_closed_external_reason]
        return r, cleaned, raw_text, meta_reason, ok, val_reason, val_warnings, meta



    def _log_detection_tags(final_rating, expl):
        """Record local detection/repair tags for Step-3 diagnostics."""
        # Runs all detectors for analysis only. Never modifies rating or explanation.
        detectors = [
            lambda: _step3_explanation_contradiction_tag_local(int(pre_belief), final_rating, expl, tweet_stance),
            lambda: _same_rating_generic_fact_dismissal_tag(int(pre_belief), final_rating, tweet_stance, tweet_txt, expl),
            lambda: _fact_based_opposite_side_inversion_tag(int(pre_belief), tweet_stance, tweet_txt, expl),
            lambda: _same_rating_cross_bin_wording_tag(int(pre_belief), final_rating, expl),
            lambda: _moved_rating_stale_category_wording_tag(int(pre_belief), final_rating, expl),
        ]
        for fn in detectors:
            try:
                tag = fn()
                if tag:
                    repair_tags.append(tag)
            except Exception:
                pass

    def _step3_explanation_has_specific_tweet_anchor(expl_text: str, tweet_text_local: str = "") -> bool:
        """Check whether local explanation cites a specific tweet point."""
        expl = re.sub(r"\s+", " ", str(expl_text or "")).strip()
        tw = re.sub(r"\s+", " ", str(tweet_text_local or "")).strip()
        if not expl or not tw:
            return False
        expl_toks = set(_fact_match_tokens(expl))
        tw_toks = set(_fact_match_tokens(tw))
        shared_non_generic = {
            tok for tok in (expl_toks & tw_toks)
            if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()
        }
        if len(shared_non_generic) >= 1:
            return True
        tw_terms = [t for t in _fact_match_tokens(tw) if t not in _FACT_SPECIFICITY_GENERIC_TOKENS and not t.isdigit()]
        tw_terms = sorted(set(tw_terms), key=len, reverse=True)
        for term in tw_terms:
            if len(term) < 6:
                continue
            if re.search(rf"(?i)\b{re.escape(term)}\b", expl):
                return True
        return False

    def _step3_explanation_attacks_tweet_weakness(expl_text: str, tweet_text_local: str = "", tweet_stance_local: str | None = None, final_rating_local=None) -> bool:
        """Check whether local explanation critiques a weakness in the tweet."""
        expl = re.sub(r"\s+", " ", str(expl_text or "")).strip().lower()
        if not expl:
            return False
        weakness_markers = bool(re.search(r"\b(lacks?|lack of|missing|without|unsupported|speculative|too speculative|too vague|unclear|gap|limitation|not testable|hard to test|no clear framework|no concrete point|does(?: not|n't) give)\b", expl))
        explicit_support_claim = bool(re.search(r"\b(supports? the claim|strengthens? the claim|reinforces? the claim|more plausible overall|gives? (?:some )?support for the claim)\b", expl))
        explicit_oppose_claim = bool(re.search(r"\b(challenges? the claim|argues? against the claim|weakens? the claim|undermines? the claim|less plausible overall)\b", expl))
        if explicit_support_claim or explicit_oppose_claim:
            return False
        try:
            fr = int(final_rating_local) if final_rating_local is not None else None
        except Exception:
            fr = None
        if tweet_stance_local == 'support' and fr is not None and fr <= 0 and weakness_markers:
            return True
        if tweet_stance_local == 'oppose' and fr is not None and fr >= 0 and weakness_markers:
            return True
        return False

    def _step3_explanation_obvious_fragment_reason(expl_text: str) -> str | None:
        """Detect broken fragment explanations in local Step-3 output."""
        expl = re.sub(r"\s+", " ", str(expl_text or "")).strip()
        if not expl:
            return None
        words = re.findall(r"[A-Za-z']+", expl)
        low = expl.lower().strip()
        has_terminal = bool(re.search(r"[.!?]$", expl))

        # Short can be valid (e.g. "Lost tapes weaken the record."). Flag only
        # short text that lacks an obvious predicate or trails off syntactically.
        predicateish = bool(re.search(
            r"\b(?:is|are|was|were|be|being|been|has|have|had|do|does|did|can|could|may|might|must|should|would|will|supports?|weakens?|undermines?|raises?|casts?|leaves?|keeps?|makes?|shows?|suggests?|confirms?|validates?|challenges?|complicates?|erodes?|provides?|points?|counts?|fits?|settles?|proves?|fails?)\b",
            low,
            flags=re.I,
        ))
        if len(words) < 5 and not predicateish:
            return 'obvious_fragment_too_short'

        last = words[-1].lower() if words else ''
        dangling_tail = {
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'that', 'because', 'about',
            'of', 'to', 'in', 'on', 'for', 'with', 'from', 'this', 'these', 'those',
            'into', 'onto', 'than', 'via', 'through', 'during', 'while', 'despite', 'although'
        }
        if last in dangling_tail:
            return 'obvious_fragment_dangling_tail'

        # Incomplete modifier endings like "used in laser" are fragments even
        # though they contain a participle. Keep this generic: only fire without
        # terminal punctuation and when the phrase ends with a common modifier
        # word that usually requires a following noun.
        if (not has_terminal) and re.search(
            r"(?i)\b(?:used|shown|seen|found|confirmed|validated|tested|measured|raised|left|placed|reported|cited)\s+(?:in|on|for|with|by|through|during)\s+(?:laser|lunar|public|official|historical|physical|direct|independent|original|later|scientific|technical|visual|experimental)$",
            expl,
        ):
            return 'obvious_fragment_dangling_phrase'

        if re.search(r"(?i)\b(?:resemblance|similarity|comparison|sign|idea|claim|possibility|reason|point|argument|reference)\s+to$", expl):
            return 'obvious_fragment_dangling_phrase'
        if re.search(r"(?i)\b(?:because|without|with|for|from|about|that|through|during|despite|although)\s*$", expl):
            return 'obvious_fragment_dangling_tail'
        if len(words) <= 8 and not has_terminal:
            return 'obvious_fragment_no_terminal_punctuation'
        return None

    def _step3_debug_audit_tags(final_rating, explanation_text: str, final_text: str = "") -> list[str]:
        """Build local Step-3 audit tags for accepted or repaired outputs."""
        expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
        final_blob = re.sub(r"\s+", " ", str(final_text or "")).strip()
        tags = []
        if not expl:
            return ['accepted_missing_explanation']
        if re.search(r"(?i)\b(?:my current stance|my current view|my current belief|my existing skepticism|remaining doubt after this tweet|reinforcing existing skepticism|aligns with my current stance|current balance after this tweet|my institutional trust|my openness to update|my persona)\b", expl):
            if not _persona_allows_hidden_state_phrase(expl, _current_persona_validation_policy()):
                tags.append('accepted_hidden_state_language')
        if re.search(r"(?i)^the tweet cites\b", expl) or re.search(r"(?i)\bthe tweet cites\b", final_blob):
            tags.append('accepted_clause_corruption_tweet_cites')
        fragment_reason = _step3_explanation_obvious_fragment_reason(expl)
        if fragment_reason:
            tags.append(f'accepted_{fragment_reason}')
        anchored_to_tweet = _step3_explanation_has_specific_tweet_anchor(expl, tweet_txt)
        attacks_tweet_weakness = _step3_explanation_attacks_tweet_weakness(expl, tweet_txt, tweet_stance, final_rating)
        generic_formula_hit = bool(re.search(r"(?i)\b(?:claim unresolved|keeps? the rating at 0|keeps? the rating|lacks specific evidence|not enough to change|beyond general skepticism)\b", expl))
        if generic_formula_hit and not anchored_to_tweet and not attacks_tweet_weakness:
            tags.append('accepted_generic_formula')
        try:
            if final_rating is not None:
                contradiction = _step3_explanation_contradiction_tag_local(int(pre_belief), int(final_rating), expl, tweet_stance)
                if contradiction:
                    suppress_contradiction = (
                        contradiction == 'tweet_explanation_polarity_contradiction'
                        and (
                            attacks_tweet_weakness
                            or _step3_explanation_has_contrastive_weakening(expl)
                            or (final_rating is not None and not _is_hard_step3_polarity_contradiction(int(final_rating), expl, tweet_txt, tweet_stance))
                        )
                    )
                    if not suppress_contradiction:
                        tags.append(f'accepted_{contradiction}')
        except Exception:
            pass
        try:
            generic_gap = _step3_generic_gap_reason(expl, tweet_text=tweet_txt, pre_value=int(pre_belief), final_value=(int(final_rating) if final_rating is not None else None), tweet_stance_local=tweet_stance)
            if generic_gap and not (anchored_to_tweet or attacks_tweet_weakness):
                tags.append(f'accepted_{generic_gap}')
        except Exception:
            pass
        try:
            stale = _moved_rating_stale_category_wording_tag(int(pre_belief), int(final_rating), expl) if final_rating is not None else None
            if stale:
                tags.append(f'accepted_{stale}')
        except Exception:
            pass
        try:
            cross_bin = _same_rating_cross_bin_wording_tag(int(pre_belief), int(final_rating), expl) if final_rating is not None else None
            if cross_bin:
                tags.append(f'accepted_{cross_bin}')
        except Exception:
            pass
        try:
            fact_dismiss = _same_rating_generic_fact_dismissal_tag(int(pre_belief), int(final_rating), tweet_stance, tweet_txt, expl) if final_rating is not None else None
            if fact_dismiss and not anchored_to_tweet:
                tags.append(f'accepted_{fact_dismiss}')
        except Exception:
            pass
        if tweet_stance == 'oppose' and re.search(r"(?i)\bdoes(?: not|n't) provide counterarguments?\b", expl):
            tags.append('accepted_counterargument_mismatch')
        if _is_open_world_mode(WORLD):
            try:
                ow = _step3_open_world_generic_reference_reason(expl, tweet_text=tweet_txt)
                if ow:
                    tags.append(f'accepted_{ow}')
            except Exception:
                pass
        if _normalize_world_mode(WORLD) == 'true_open':
            try:
                mismatch = _step3_true_open_point_mismatch_reason(expl, tweet_text=tweet_txt, prompt_text=prompt)
                if mismatch:
                    tags.append(f'accepted_{mismatch}')
            except Exception:
                pass
        if _normalize_world_mode(WORLD) in {'closed_strict', 'closed_strict_rag'}:
            try:
                strict_external = _step3_strict_closed_external_reason_reason(expl, tweet_txt, prompt_text=prompt)
                if strict_external:
                    tags.append(f'accepted_{strict_external}')
            except Exception:
                pass
        seen_tags = set()
        out = []
        for tag in tags:
            tok = str(tag or '').strip()
            if not tok or tok in seen_tags:
                continue
            seen_tags.add(tok)
            out.append(tok)
        return out

    def _emit_step3_accepted_audit(final_text: str, raw_first: str = '', raw_second: str = '', *, source: str = 'main'):
        """Emit local audit metrics after a Step-3 output is accepted."""
        if not DEBUG_NATIVE_THINKING_ON_FAIL:
            return
        rating = _parse_rating(final_text)
        expl = _extract_explanation_text(final_text) or ''
        audit_tags = _step3_debug_audit_tags(rating, expl, final_text)
        if not audit_tags:
            return
        _dump_retry_debug_event(
            'step3',
            'warning',
            agent_name=agent_name,
            attempt=f'accepted_{source}',
            reasons=[f'accepted_warning:{t}' for t in audit_tags],
            prompt_text=prompt + stance_note,
            raw_text=str(raw_second or raw_first or ''),
            sanitized_text=str(final_text or ''),
            final_text=str(final_text or ''),
            conversation=conversation,
            include_history=True,
        )

    def _step3_explanation_contradiction_tag_local(pre_value, final_value, explanation_text, tweet_stance_local=None):
        """Return a local tag when rating and explanation contradict each other."""
        # In free_bounded mode, movement can go toward or away from the speaker/tweet.
        # Broad polarity-rewrite logic caused many false positives. Clear opposite-side-only
        # explanation mismatches are handled later by _step3_explanation_wrong_side_anchor().
        if _is_free_bounded_update_mode():
            return None
        expl = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
        if not expl:
            return "empty_explanation"
        try:
            pre_i = int(pre_value)
            final_i = int(final_value)
        except Exception:
            return None
    
        directional_move_claim = bool(re.search(
            r"\b(push(?:es|ed)? me (?:slightly )?(?:more )?positive"
            r"|push(?:es|ed)? me (?:slightly )?(?:more )?negative"
            r"|nudg(?:es|ed)? me (?:slightly )?(?:more )?positive"
            r"|nudg(?:es|ed)? me (?:slightly )?(?:more )?negative"
            r"|made me move (?:slightly )?(?:more )?positive"
            r"|made me move (?:slightly )?(?:more )?negative"
            r"|makes me move (?:slightly )?(?:more )?positive"
            r"|makes me move (?:slightly )?(?:more )?negative)\b",
            expl, flags=re.I,
        ))
        toward_zero_claim = bool(re.search(
            r"\b(move(?:d|s)? my rating toward 0|move(?:d|s)? me toward 0|became more uncertain|pushed me toward uncertainty)\b",
            expl, flags=re.I,
        ))
        try:
            lo = min(int(x) for x in allowed_list)
            hi = max(int(x) for x in allowed_list)
        except Exception:
            lo = hi = final_i
        legal_boundary = (
            final_i == pre_i
            and (
                (tweet_stance_local == "support" and final_i == hi)
                or (tweet_stance_local == "oppose" and final_i == lo)
            )
        )
        if final_i == pre_i and (directional_move_claim or toward_zero_claim) and not legal_boundary:
            if tweet_stance_local == "support":
                return "nochange_support_move_pressure"
            if tweet_stance_local == "oppose":
                return "nochange_oppose_move_pressure"
            if tweet_stance_local == "uncertain":
                return "nochange_uncertain_move_pressure"
            return "nochange_directional_move_pressure"
    
        moving_toward_zero = ((pre_i < 0 <= final_i and final_i > pre_i) or (pre_i > 0 >= final_i and final_i < pre_i) or (pre_i < 0 and final_i < 0 and final_i > pre_i) or (pre_i > 0 and final_i > 0 and final_i < pre_i))
        explicit_support_claim = bool(re.search(
            r"\b(supports? the claim|strengthens? the claim|strengthens? the case for the claim|reinforces? the claim|gives? (?:some )?support for the claim|makes? the claim more plausible|more plausible overall|more likely to believe|push(?:es|ed)? me (?:slightly )?(?:more )?positive|nudg(?:es|ed)? me (?:slightly )?(?:more )?positive|refutes? (?:the )?(?:challenge|objection|uncertainty|doubt)|counters? (?:the )?(?:challenge|objection|uncertainty|doubt)|contradicted by|explained by|evidence (?:for|supporting) (?:the )?(?:claim|landings?|missions?)|physical evidence supporting|third-party evidence supporting)\b",
            expl, flags=re.I,
        ))
        explicit_oppose_claim = bool(re.search(
            r"\b(argues? against the claim|challenges? the claim|weakens? the claim|undermines? the claim|makes? the claim less plausible|less plausible overall|less likely to believe|push(?:es|ed)? me (?:slightly )?(?:more )?negative|nudg(?:es|ed)? me (?:slightly )?(?:more )?negative|raises? doubts? about the claim|casts? doubt on the claim|keeps? the claim uncertain|leaves? the claim uncertain|questions? the claim(?:'s)? validity)\b",
            expl, flags=re.I,
        ))
        support_then_weakness = bool(re.search(
            r"\b(?:supports? the claim|strengthens? the claim|strengthens? the case for the claim|reinforces? the claim|gives? (?:some )?support for the claim|makes? the claim more plausible|more plausible overall)\b.*\b(?:but|however|though|yet)\b.*\b(lacks?|lack of|missing|without|does(?: not|n't)|not enough|too vague|too generic|insufficient|unsupported|unclear|unresolved|weak(?:ness)?|gap|limitation)\b",
            expl, flags=re.I,
        ))
        oppose_then_weakness = bool(re.search(
            r"\b(?:argues? against the claim|challenges? the claim|weakens? the claim|undermines? the claim|makes? the claim less plausible|less plausible overall)\b.*\b(?:but|however|though|yet)\b.*\b(lacks?|lack of|missing|without|does(?: not|n't)|not enough|too vague|too generic|insufficient|unsupported|unclear|unresolved|weak(?:ness)?|gap|limitation)\b",
            expl, flags=re.I,
        ))
    
        if final_i > 0 and explicit_oppose_claim and not moving_toward_zero and not oppose_then_weakness:
            return "tweet_explanation_polarity_contradiction"
        if final_i < 0 and explicit_support_claim and not moving_toward_zero and not support_then_weakness:
            return "tweet_explanation_polarity_contradiction"
        if final_i == 0 and (explicit_support_claim or explicit_oppose_claim) and directional_move_claim and not toward_zero_claim:
            return "tweet_explanation_polarity_contradiction"
        return None
    
    # ------------------------------------------------------------------ setup

    prompt = _strip_legacy_step3_stance_hints(prompt)
    claim_txt = _extract_claim_from_prompt(prompt)
    tweet_txt = _extract_tweet_from_step3_prompt(prompt)
    tweet_stance = detected_stance_from_tweet(tweet_txt)
    if tweet_stance is None and speaker_belief is not None:
        try:
            sb = int(speaker_belief)
            tweet_stance = "support" if sb > 0 else ("oppose" if sb < 0 else "uncertain")
        except Exception:
            pass

    if allowed_ratings is not None:
        try:
            allowed_list = [int(x) for x in allowed_ratings]
        except Exception:
            allowed_list = None
    else:
        allowed_list = None

    if not allowed_list:
        allowed_list = _allowed_set(int(pre_belief))

    seen = set()
    allowed_list = [int(x) for x in allowed_list if not (int(x) in seen or seen.add(int(x)))]
    allowed_set = set(int(x) for x in allowed_list)
    allowed_str = _format_allowed_ratings(int(pre_belief), allowed_list)
    prompt = _replace_step3_decision_rule_for_update_mode(prompt, allowed_str)

    # Keep only the movement boundary plus an optional strong-bias reminder.
    # Do not pre-label the tweet's stance inside the prompt.
    stance_note = f"\n\nALLOWED_FINAL_RATING_SET: {allowed_str}"
    stance_note += _step3_update_mode_note(allowed_str)
    strong_bias_note = _strong_bias_extra_note(int(pre_belief), tweet_stance)
    if strong_bias_note:
        stance_note += "\n" + strong_bias_note.strip()

    def _finalize_step3_success(out_text, raw_first, raw_second, tags, *, warning_list=None, source="main"):
        """Finalize a local Step-3 success by formatting output and logging repairs."""
        final_out = str(out_text or '')
        final_raw_second = raw_second
        final_tags = list(tags or [])
        final_warnings = list(warning_list or [])

        parsed_rating = _parse_rating(final_out)
        current_expl = _extract_explanation_text(final_out) or ''
        if parsed_rating is not None and _step3_explanation_wrong_side_anchor(current_expl, int(parsed_rating), tweet_txt, pre_belief=int(pre_belief)):
            if _step3_should_keep_original_explanation_before_wrong_side_rewrite(current_expl, int(parsed_rating), tweet_txt, pre_belief=int(pre_belief)):
                final_out = _format_step3_output_preserve_explanation(final_rating=int(parsed_rating), explanation=current_expl)
                current_expl = _extract_explanation_text(final_out) or current_expl
            else:
                # Wrong-side explanations are semantic mismatches. Use deterministic
                # tweet-local/shown-material cleanup instead of native rescue: native
                # rescue repeatedly introduced broad outside-style support such as
                # "multiple independent evidence sources" in strict-closed/RAG runs.
                rescue_used = False
                original_rating_for_rescue = int(parsed_rating)
                if original_rating_for_rescue == 0:
                    local_expl0 = _step3_safe_fallback_explanation_for_rating(tweet_txt, original_rating_for_rescue, int(pre_belief), tweet_stance)
                    local_bad0 = bool(local_expl0 and _step3_explanation_wrong_side_anchor(local_expl0, original_rating_for_rescue, tweet_txt, pre_belief=int(pre_belief)))
                    local_frag0 = _step3_explanation_obvious_fragment_reason(local_expl0) if local_expl0 else 'missing_local_neutral_cleanup'
                    if local_expl0 and not local_bad0 and not local_frag0:
                        final_out = _format_step3_output_preserve_explanation(final_rating=original_rating_for_rescue, explanation=local_expl0)
                        final_tags.append('step3_neutral_balance_soft_cleanup')
                        parsed_rating = original_rating_for_rescue
                        current_expl = _extract_explanation_text(final_out) or current_expl
                        source = 'neutral_local_cleanup'
                        rescue_used = True
                # Deliberately skip native rescue for wrong-side preserve-rating cases.
                # The deterministic fallback below keeps the fixed legal rating and
                # builds a bounded explanation from the current tweet instead of
                # allowing a second LLM call to add unsupported broad-evidence claims.
                if not rescue_used and _wrong_side_explanation_requery_enabled():
                    try:
                        _rq_prompt = _compact_step3_rescue_prompt(compact=True, fixed_rating=int(parsed_rating), original_explanation=current_expl)
                        _rq_raw = _native_ollama_chat(conversation, _rq_prompt, include_history=False, save_turn=False)
                        _rq_expl = _extract_explanation_text(sanitize_step3_output(_rq_raw)) or ''
                        _rq_ok = bool(_rq_expl and _rq_expl.strip())
                        if _rq_ok and _step3_explanation_wrong_side_anchor(_rq_expl, int(parsed_rating), tweet_txt, pre_belief=int(pre_belief)):
                            _rq_ok = False
                        if _rq_ok and _step3_explanation_obvious_fragment_reason(_rq_expl):
                            _rq_ok = False
                        if _rq_ok and _normalize_world_mode(WORLD) in {"closed_strict", "closed_strict_rag"} and _step3_strict_closed_external_reason_reason(_rq_expl, tweet_txt, prompt_text=prompt):
                            _rq_ok = False
                        if _rq_ok:
                            # Rating stays FIXED at parsed_rating; only the explanation is replaced.
                            final_out = _format_step3_output_preserve_explanation(final_rating=int(parsed_rating), explanation=_rq_expl)
                            final_tags.append('step3_wrong_side_explanation_requery')
                            current_expl = _extract_explanation_text(final_out) or current_expl
                            rescue_used = True
                    except Exception:
                        pass
                if not rescue_used:
                    local_expl = _step3_safe_fallback_explanation_for_rating(tweet_txt, int(parsed_rating), int(pre_belief), tweet_stance)
                    if local_expl:
                        final_out = _format_step3_output_preserve_explanation(final_rating=int(parsed_rating), explanation=local_expl)
                        final_tags.append('step3_wrong_side_explanation_fallback_rewrite')
                        current_expl = _extract_explanation_text(final_out) or current_expl
        fragment_reason = _step3_explanation_obvious_fragment_reason(current_expl)
        if fragment_reason:
            _dump_retry_debug_event(
                'step3',
                'warning',
                agent_name=agent_name,
                attempt=f'accepted_{source}',
                reasons=[f'accepted_warning:accepted_{fragment_reason}'],
                prompt_text=prompt + stance_note,
                raw_text=str(final_raw_second or raw_first or ''),
                sanitized_text=str(final_out or ''),
                final_text=str(final_out or ''),
                conversation=conversation,
                include_history=True,
            )
            # Minor short/no-punctuation fragments stay warning-only. Hard dangling
            # fragments are the only ones rewritten/rescued because the previous
            # broad rewrite path produced clipped, worse explanations.
            hard_fragment = str(fragment_reason or '') in {'obvious_fragment_too_short', 'obvious_fragment_dangling_tail', 'obvious_fragment_dangling_phrase'}
            if (not hard_fragment) and str(fragment_reason or '') == 'obvious_fragment_no_terminal_punctuation':
                if current_expl and not re.search(r"[.!?]$", current_expl.strip()) and parsed_rating is not None:
                    final_out = _rewrite_step3_output_with_explanation(final_rating=int(parsed_rating), explanation=current_expl.strip() + '.')
                    current_expl = _extract_explanation_text(final_out) or current_expl
            elif hard_fragment and parsed_rating is not None:
                local_expl = _step3_tweet_local_explanation(tweet_txt, int(parsed_rating), int(pre_belief), tweet_stance)
                if local_expl and not _step3_explanation_obvious_fragment_reason(local_expl):
                    final_out = _format_step3_output_preserve_explanation(final_rating=int(parsed_rating), explanation=local_expl)
                    final_tags.append('step3_fragment_explanation_rewrite')
                elif str(source or '') not in {'fragment_rescue'}:
                    rescue_prompt_fragment = _compact_step3_rescue_prompt(compact=True)
                    r_frag, txt_frag, raw_frag, meta_frag, ok_frag, reason_frag, warn_frag, native_meta_frag = _native_rescue_attempt(
                        rescue_prompt_fragment,
                        '1nf',
                        include_history=False,
                        compact=True,
                    )
                    if r_frag is not None:
                        txt_frag = _ensure_has_explanation(txt_frag, r_frag)
                    frag_expl = _extract_explanation_text(txt_frag) or ''
                    if r_frag is not None and int(r_frag) in allowed_set and ok_frag and not _step3_explanation_obvious_fragment_reason(frag_expl):
                        final_out = txt_frag
                        final_raw_second = raw_frag or final_raw_second
                        final_warnings.extend(list(warn_frag or []))
                        final_tags.append('step3_fragment_native_rescue')
                        source = 'fragment_rescue'


        _record_step3_success_metrics(source=source, contaminated=_step3_has_strict_closed_contamination(final_warnings))
        _emit_step3_accepted_audit(final_out, raw_first=raw_first, raw_second=final_raw_second, source=source)
        return final_out, raw_first, final_raw_second, final_tags

    # ------------------------------------------------------------------ singleton

    singleton_result = _build_singleton_step3_output(
        pre_belief=int(pre_belief),
        speaker_belief=speaker_belief,
        allowed_ratings=allowed_list,
        tweet_txt=tweet_txt,
        tweet_stance=tweet_stance,
        same_rating_mode=SAME_RATING_STEP3_MODE,
    )
    if singleton_result is not None:
        final_text, singleton_tags = singleton_result
        repair_tags.extend(list(singleton_tags or []))
        return final_text, "", "", repair_tags

    # ------------------------------------------------------------------ attempt 1

    r1, txt1, raw1, meta1, ok1, reason1, warn1 = _attempt(prompt + stance_note, 1)
    raw2 = ""
    attempt1_candidate_text = ""

    if (not str(raw1 or "").strip()) or (not str(txt1 or "").strip()):
        _metric_inc('step3_native_rescue_attempts')
        rescue_prompt = _compact_step3_rescue_prompt(compact=False)
        r2, txt2, raw2, meta2, ok2, reason2, warn2, native_meta2 = _native_rescue_attempt(rescue_prompt, "1n", include_history=True, compact=False)
        if r2 is not None:
            txt2 = _ensure_has_explanation(txt2, r2)
        rescue_reasons_1n = []
        if (not str(raw2 or '').strip()) and (not str(txt2 or '').strip()):
            rescue_reasons_1n.append('empty_attempt_native_rescue')
        elif r2 is None or int(r2) not in allowed_set or not ok2:
            rescue_reasons_1n.append('invalid_attempt_native_rescue')
        if rescue_reasons_1n:
            _dump_retry_debug_event('step3', 'native_rescue', agent_name=agent_name, attempt='1n', reasons=rescue_reasons_1n, prompt_text=rescue_prompt, raw_text=raw2, sanitized_text=txt2, final_text=txt2, conversation=conversation, include_history=True)
        if r2 is not None and int(r2) in allowed_set and ok2:
            _metric_inc('step3_native_rescue_success')
            expl2 = _extract_explanation_text(txt2) or ""
            _log_detection_tags(final_rating=int(r2), expl=expl2)
            return _finalize_step3_success(txt2, raw1, raw2, repair_tags, warning_list=warn2, source="rescue")

        native_done2 = str((native_meta2 or {}).get('done_reason') or '').strip().lower()
        native_empty2 = not bool(str(raw2 or '').strip()) and not bool(str(txt2 or '').strip())
        if native_done2 == 'length':
            _metric_inc('step3_native_rescue_length_stop')
        if native_empty2:
            _metric_inc('step3_native_rescue_empty_final')

        if native_empty2 or native_done2 == 'length':
            _metric_inc('step3_native_rescue_compact_attempts')
            rescue_prompt_compact = _compact_step3_rescue_prompt(compact=True)
            r3, txt3, raw3, meta3, ok3, reason3, warn3, native_meta3 = _native_rescue_attempt(rescue_prompt_compact, "1n2", include_history=False, compact=True)
            if r3 is not None:
                txt3 = _ensure_has_explanation(txt3, r3)
            rescue_reasons_1n2 = []
            if (not str(raw3 or '').strip()) and (not str(txt3 or '').strip()):
                rescue_reasons_1n2.append('empty_attempt_native_rescue_compact')
            elif r3 is None or int(r3) not in allowed_set or not ok3:
                rescue_reasons_1n2.append('invalid_attempt_native_rescue_compact')
            if rescue_reasons_1n2:
                _dump_retry_debug_event('step3', 'native_rescue', agent_name=agent_name, attempt='1n2', reasons=rescue_reasons_1n2, prompt_text=rescue_prompt_compact, raw_text=raw3, sanitized_text=txt3, final_text=txt3, conversation=conversation, include_history=False)
            if r3 is not None and int(r3) in allowed_set and ok3:
                _metric_inc('step3_native_rescue_compact_success')
                expl3 = _extract_explanation_text(txt3) or ""
                _log_detection_tags(final_rating=int(r3), expl=expl3)
                return _finalize_step3_success(txt3, raw1, raw3, repair_tags, warning_list=warn3, source="rescue")
            native_done3 = str((native_meta3 or {}).get('done_reason') or '').strip().lower()
            if native_done3 == 'length':
                _metric_inc('step3_native_rescue_compact_length_stop')
            if (not str(raw3 or '').strip()) and (not str(txt3 or '').strip()):
                _metric_inc('step3_native_rescue_compact_empty_final')
            raw2 = raw3
            txt2 = txt3
            meta2 = meta3
            ok2 = ok3
            reason2 = reason3
            warn2 = warn3

        if (not str(raw2 or '').strip()) and (not str(txt2 or '').strip()):
            _metric_inc('step3_fallback_after_native_rescue_failure')
            repair_tags.append('step3_empty_retry_failed')
            fallback = _fallback_step3(pre_belief=int(pre_belief), claim_txt=claim_txt, tweet_stance=tweet_stance, tweet_txt=tweet_txt)
            _dump_retry_debug_event('step3', 'fallback', agent_name=agent_name, attempt='1n', reasons=repair_tags + ['native_rescue_failed'], prompt_text=rescue_prompt, raw_text=raw2, sanitized_text=txt2, final_text=fallback, conversation=conversation, include_history=True)
            return fallback, raw1, raw2, repair_tags
        if str(raw2 or "").strip() or str(txt2 or "").strip():
            r1, txt1, meta1, ok1, reason1, warn1 = r2, txt2, meta2, ok2, reason2, warn2

    if r1 is not None and int(r1) in allowed_set and ok1:
        expl1 = _extract_explanation_text(txt1) or ""
        hidden_only_open = _is_open_world_mode(WORLD) and _has_only_hidden_state_family(reason1, warn1, meta1)
        if hidden_only_open:
            meta1 = ""
            reason1 = ""
            warn1 = [w for w in (warn1 or []) if "mentions_hidden_state" not in str(w).lower()]
        if meta1 and not _is_step3_soft_warning_only_meta(meta1):
            repair_tags.append(f"step3_warning_attempt_1:{meta1}")
            _dump_retry_debug_event('step3', 'warning', agent_name=agent_name, attempt=1, reasons=[meta1], prompt_text=prompt + stance_note, raw_text=raw1, sanitized_text=txt1, final_text=txt1, conversation=conversation, include_history=True)
        hard_issue1_main = _has_hard_step3_issue(reason1, warn1, meta1)
        true_open_rewritten = _rewrite_true_open_meta_to_tweet_local(int(r1), meta1, reason1, warn1)
        if true_open_rewritten and not _would_semantically_invert(int(r1), expl1, true_open_rewritten, tweet_stance):
            repair_tags.append('step3_true_open_local_rewrite')
            _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(true_open_rewritten) or expl1)
            return _finalize_step3_success(true_open_rewritten, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        contradiction_local = _step3_explanation_contradiction_tag_local(int(pre_belief), int(r1), expl1, tweet_stance)
        if contradiction_local == "tweet_explanation_polarity_contradiction" and not _is_free_bounded_update_mode():
            aligned_final = _rewrite_step3_explanation_only_for_polarity(int(r1), expl1)
            if aligned_final:
                repair_tags.append("step3_polarity_alignment_rewrite")
                _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(aligned_final) or expl1)
                return _finalize_step3_success(aligned_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        _log_detection_tags(final_rating=int(r1), expl=expl1)
        return _finalize_step3_success(txt1, raw1, raw2, repair_tags, warning_list=warn1, source="main")

    if r1 is not None and int(r1) in allowed_set:
        expl1_current = _extract_explanation_text(txt1) or ""
        on_topic1 = _step3_explanation_on_topic(expl1_current, tweet_txt)
        if _is_open_world_mode(WORLD) and _has_only_hidden_state_family(reason1, warn1, meta1):
            meta1 = ""
            reason1 = ""
            warn1 = [w for w in (warn1 or []) if "mentions_hidden_state" not in str(w).lower()]
        soft_meta1 = _is_step3_soft_warning_only_meta(meta1)
        hard_issue1 = _has_hard_step3_issue(reason1, warn1, meta1)
        current_final = _preserve_step3_output(int(r1), txt1, claim_txt)
        polarity_meta_tags = {
            "detected_oppose_no_positive_move",
            "detected_support_no_negative_move",
            "tweet_explanation_polarity_contradiction",
        }
        meta_tokens1 = {tok.strip() for tok in str(meta1 or "").split("|") if tok.strip()}
        if meta1:
            _dump_retry_debug_event('step3', 'warning', agent_name=agent_name, attempt=1, reasons=[meta1], prompt_text=prompt + stance_note, raw_text=raw1, sanitized_text=txt1, final_text=current_final, conversation=conversation, include_history=True)
            repair_tags.append(f"step3_warning_attempt_1:{meta1}")
        if warn1:
            _dump_retry_debug_event('step3', 'warning', agent_name=agent_name, attempt=1, reasons=[f"warning:{w}" for w in warn1], prompt_text=prompt + stance_note, raw_text=raw1, sanitized_text=txt1, final_text=current_final, conversation=conversation, include_history=True)
        if meta_tokens1 & polarity_meta_tags:
            hard_contradiction = (
                'tweet_explanation_polarity_contradiction' in meta_tokens1
                and _is_hard_step3_polarity_contradiction(int(r1), expl1_current, tweet_txt, tweet_stance)
            )
            if hard_contradiction and not _is_free_bounded_update_mode():
                aligned_expl = _step3_safe_fallback_explanation_for_rating(tweet_txt, int(r1), int(pre_belief), tweet_stance)
                aligned_final = _rewrite_step3_output_with_explanation(final_rating=int(r1), explanation=aligned_expl)
                repair_tags.append("step3_polarity_alignment_rewrite")
                _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(aligned_final) or aligned_expl)
                return _finalize_step3_success(aligned_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        true_open_rewritten = _rewrite_true_open_meta_to_tweet_local(int(r1), meta1, reason1, warn1)
        if true_open_rewritten and not _would_semantically_invert(int(r1), expl1_current, true_open_rewritten, tweet_stance):
            repair_tags.append('step3_true_open_local_rewrite')
            _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(true_open_rewritten) or expl1_current)
            return _finalize_step3_success(true_open_rewritten, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        contradiction_local = _step3_explanation_contradiction_tag_local(int(pre_belief), int(r1), expl1_current, tweet_stance)
        if contradiction_local == "tweet_explanation_polarity_contradiction":
            aligned_final = _rewrite_step3_explanation_only_for_polarity(int(r1), expl1_current)
            if aligned_final:
                repair_tags.append("step3_polarity_alignment_rewrite")
                _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(aligned_final) or expl1_current)
                return _finalize_step3_success(aligned_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        # Hidden-state wording should be cleaned when possible, but it remains a soft issue:
        # persona can influence the rating internally while the public explanation stays tweet-facing.
        hidden_stateish = False
        joined_issue_text = " | ".join(
            [str(meta1 or ""), str(reason1 or "")] + [str(w or "") for w in (warn1 or [])]
        ).lower()
        if "mentions_hidden_state" in joined_issue_text:
            hidden_stateish = True
        only_hidden_stateish = _has_only_hidden_state_family(reason1, warn1, meta1)
        open_world_like = _is_open_world_mode(WORLD)
        if open_world_like and only_hidden_stateish:
            _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(current_final) or expl1_current)
            return _finalize_step3_success(current_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        if hidden_stateish and on_topic1:
            rewritten_hidden = _soft_accept_hidden_state_family(int(r1), txt1)
            if rewritten_hidden and not _would_semantically_invert(int(r1), expl1_current, rewritten_hidden, tweet_stance):
                repair_tags.append("step3_hidden_state_explanation_rewrite")
                _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(rewritten_hidden) or expl1_current)
                return _finalize_step3_success(rewritten_hidden, raw1, raw2, repair_tags, warning_list=warn1, source="main")
            if only_hidden_stateish:
                repair_tags.append("step3_hidden_state_soft_accept")
                _log_detection_tags(final_rating=int(r1), expl=expl1_current)
                return _finalize_step3_success(current_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")
        if (soft_meta1 or on_topic1) and (not hard_issue1 or only_hidden_stateish):
            _log_detection_tags(final_rating=int(r1), expl=_extract_explanation_text(current_final) or expl1_current)
            return _finalize_step3_success(current_final, raw1, raw2, repair_tags, warning_list=warn1, source="main")

    reasons1 = []
    if meta1:
        if not _is_step3_soft_warning_only_meta(meta1):
            repair_tags.append(f"step3_invalid_explanation_attempt_1:{meta1}")
        reasons1.append(meta1 if str(meta1).startswith('generic_') else f"visible_meta:{meta1}")
    if reason1 and reason1 != 'ok':
        reasons1.append(reason1)
    if warn1:
        reasons1.extend([f"warning:{w}" for w in warn1])
    if r1 is None:
        reasons1.append('missing_final_rating')
    elif int(r1) not in allowed_set:
        reasons1.append(f"not_in_allowed_set:{int(r1)}")
    _dump_retry_debug_event('step3', 'warning', agent_name=agent_name, attempt=1, reasons=reasons1 or 'attempt_1_invalid', prompt_text=prompt + stance_note, raw_text=raw1, sanitized_text=txt1, final_text='', conversation=conversation, include_history=True)

    # ------------------------------------------------------------------ fallback: stay at pre_belief

    repair_tags.append("repair_attempt_failed")
    invalid_rating = int(r1) if (r1 is not None and int(r1) not in allowed_set) else None
    fallback = _fallback_step3(pre_belief=int(pre_belief), claim_txt=claim_txt, tweet_stance=tweet_stance, tweet_txt=tweet_txt, invalid_rating=invalid_rating)
    fallback_final = _maybe_skip_protocol_fallback('step3', fallback, raw_text=raw1, sanitized_text=txt1)
    _dump_retry_debug_event('step3', 'fallback', agent_name=agent_name, attempt=1, reasons=repair_tags + reasons1, prompt_text=prompt + stance_note, raw_text=raw1, sanitized_text=txt1, final_text=fallback_final, conversation=conversation, include_history=True)
    return fallback_final, raw1, raw2, repair_tags



def get_step2_llm_response(conversation, prompt, expected_current_belief, add_to_memory, agent_name: str = None):
    """
    Run the full Step-2 speaker tweet-generation pipeline.
    
        The function calls the LLM, sanitizes the tweet output, checks rating/stance consistency, performs
        compact retry or native rescue when needed, records warning tags, caches handoff text, and returns a
        finalized public tweet payload for the listener.
    """
    _set_memory_step_kind(conversation, "step2")
    """Step-2 call with at most one cheap fatal-only retry.

    Policy:
      - Make one normal LLM call.
      - Allow one extra attempt only for genuinely fatal format failures.
      - Keep FINAL_RATING locked to expected_current_belief (protocol intent).
      - Use deterministic fallbacks or lexical softening instead of extra rewrite calls for soft/style issues.

    Returns canonical 2-line Step-2 output:
        FINAL_RATING: X
        TWEET: <tweet text>
    """
    world_norm = _normalize_world_mode(WORLD)
    is_closed_strict = world_norm in {"closed_strict", "closed_strict_rag"}
    max_attempts = 2  # allow one extra pass only for fatal format failures

    def _extract_claim_from_prompt(p: str) -> str:
        """
        Extract the active claim from the Step-2 prompt text.
        
            The local Step-2 repair code uses this when rebuilding compact retry prompts after malformed tweet
            generation.
        """
        if not p:
            return ""
        m = re.search(r"(?im)^\s*CLAIM\s*\(.*?\)\s*:\s*(.+?)\s*$", str(p))
        if not m:
            m = re.search(r"(?im)^\s*CLAIM\s*:\s*(.+?)\s*$", str(p))
        if not m:
            return ""
        return (m.group(1) or "").strip()

    def _log_step2_event(event_type: str, attempt: int, reason, raw_text: str, sanitized_text: str, final_text: str = ""):
        """Record one local Step-2 attempt, warning, or repair event."""
        _dump_retry_debug_event(
            step='step2',
            event_type=event_type,
            agent_name=agent_name,
            attempt=attempt,
            reasons=reason,
            prompt_text=prompt_to_use,
            raw_text=raw_text,
            sanitized_text=sanitized_text,
            final_text=final_text,
            conversation=conversation,
            include_history=(int(attempt) <= 1 if str(attempt).isdigit() else _infer_shadow_include_history(attempt)),
        )

    claim_txt = _extract_claim_from_prompt(prompt)
    raw_attempt_1 = ""
    raw_attempt_2 = ""
    warning_tags = []
    step2_quality = "ok"
    fallback_used = False

    def _remember_attempt(attempt_num: int, raw_text_value: str):
        """Store local Step-2 attempt text for later diagnostics or fallback."""
        nonlocal raw_attempt_1, raw_attempt_2
        if int(attempt_num) <= 1:
            raw_attempt_1 = str(raw_text_value or "")
        else:
            raw_attempt_2 = str(raw_text_value or "")

    def _add_warning_tag(attempt_num: int, reason_text: str):
        """Add one Step-2 warning tag if it is not already present."""
        warning_tags.append(f"attempt_{int(attempt_num)}:{_reason_tag_token(reason_text)}")

    def _step2_debug_audit_tags(tweet_text_value: str, current_rating: int) -> list[str]:
        """Build local Step-2 audit tags for accepted or repaired tweet outputs."""
        tweet_local = re.sub(r"\s+", " ", str(tweet_text_value or "")).strip()
        if not tweet_local:
            return ['accepted_empty_tweet']
        tags = []
        try:
            plan_meta = _visible_planning_meta_reason(tweet_local, field='tweet')
            if plan_meta:
                tags.append(f'accepted_{plan_meta}')
        except Exception:
            pass
        try:
            canned = _generic_canned_reason_reason(tweet_local, field='tweet')
            if canned:
                tags.append(f'accepted_generic_canned_tweet:{canned}')
        except Exception:
            pass
        try:
            persona_canned = _step2_persona_canned_reason(tweet_local, current_belief=current_rating)
            if persona_canned:
                tags.append(f'accepted_persona_canned:{persona_canned}')
        except Exception:
            pass
        try:
            wrong_dir, wrong_dir_reason = _clear_step2_wrong_direction(tweet_local, current_rating)
            if wrong_dir:
                tags.append(f'accepted_wrong_direction:{wrong_dir_reason}')
        except Exception:
            pass
        try:
            self_undermining, self_undermining_reason = _step2_self_undermining_concession(tweet_local, current_rating)
            if self_undermining:
                tags.append(f'accepted_self_undermining:{self_undermining_reason}')
        except Exception:
            pass
        try:
            intensity_mismatch, intensity_reason = _step2_speaker_intensity_mismatch(tweet_local, current_rating)
            if intensity_mismatch:
                tags.append(f'accepted_intensity_mismatch:{intensity_reason}')
        except Exception:
            pass
        try:
            grounding_text = _extract_external_grounding_text(prompt_to_use)
            unsupported, unsupported_flags = _unsupported_external_reference_flags(
                tweet_text=tweet_local,
                world_mode=WORLD,
                fact_pack_text=FACT_PACK_TEXT,
                extra_grounding_text=grounding_text,
            )
            if unsupported:
                tags.append(f'accepted_unsupported_external_reference:{unsupported_flags}')
        except Exception:
            pass
        out = []
        seen = set()
        for tag in tags:
            tok = str(tag or '').strip()
            if not tok or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
        return out

    def _emit_step2_accepted_audit(final_output: str):
        """Emit Step-2 audit metrics after a tweet output is accepted."""
        if not DEBUG_NATIVE_THINKING_ON_FAIL:
            return final_output
        tweet_local = _strip_leaked_prompt_headers(extract_tweet_text(final_output) or '')
        audit_tags = _step2_debug_audit_tags(tweet_local, int(expected_current_belief))
        if audit_tags:
            raw_for_debug = str(raw_attempt_2 or raw_attempt_1 or '')
            sanitized_for_debug = sanitize_step2_output(raw_for_debug) if raw_for_debug else ''
            _log_step2_event('warning', 'accepted', [f'accepted_warning:{t}' for t in audit_tags], raw_for_debug, sanitized_for_debug, final_output)
        return final_output

    def _finalize_step2_success(final_output: str, *, contaminated: bool = False, source: str = "main"):
        """Finalize a valid Step-2 tweet output and update handoff state."""
        _record_step2_success_metrics(source=source, contaminated=contaminated)
        audited_output = _emit_step2_accepted_audit(final_output)
        _log_step2_cue_metric(audited_output)
        _set_step2_call_info(
            conversation,
            raw_attempt_1=raw_attempt_1,
            raw_attempt_2=raw_attempt_2,
            warning_tags=warning_tags,
            quality=step2_quality,
            fallback_used=fallback_used,
            final_output=audited_output,
        )
        return audited_output

    def _native_rescue_step2_attempt(prompt_to_use: str, attempt_id, *, include_history: bool = True):
        """Run a native-Ollama rescue attempt for the current Step-2 failure."""
        _set_current_validation_context(prompt_to_use)
        _set_native_ollama_debug_context(step='step2', agent_name=agent_name, attempt=attempt_id, label='native_rescue')
        # Use seed+7 so rescue calls don't replay the same degenerate empty-output
        # path that already failed at the original seed.  The offset is small and
        # deterministic so runs remain reproducible.
        try:
            _base_seed = (
                getattr(args, 'llm_seed_step2', None)
                or getattr(args, 'llm_seed', None)
                or getattr(args, 'seed', None)
            )
            _rescue_seed = (int(_base_seed) + 7) if _base_seed is not None else None
        except Exception:
            _rescue_seed = None
        try:
            raw_text = _native_ollama_chat(
                conversation,
                prompt_to_use,
                include_history=include_history,
                save_turn=False,
                seed_override=_rescue_seed,
            )
        finally:
            _clear_native_ollama_debug_context()
        meta = dict(getattr(conversation, '_last_native_ollama_meta', {}) or {})
        raw_text_str = "" if raw_text is None else str(raw_text)
        text = sanitize_step2_output(raw_text_str)
        tweet_one_line = extract_tweet_text(text) or extract_tweet_text(raw_text_str) or ""
        tweet_one_line = _strip_leaked_prompt_headers(tweet_one_line)
        tweet_one_line = re.sub(r"\s+", " ", tweet_one_line).strip()
        if not tweet_one_line:
            tweet_one_line = strip_step2_meta_lines(text or raw_text_str)
            tweet_one_line = re.sub(r"\s+", " ", tweet_one_line).strip()
        if not tweet_one_line:
            tweet_one_line = _heuristic_freeform_protocol_body(raw_text_str, field="tweet")
            tweet_one_line = re.sub(r"\s+", " ", tweet_one_line).strip()
            if tweet_one_line and not _step2_tweet_has_valid_public_body(tweet_one_line):
                tweet_one_line = ""
        if not tweet_one_line:
            tweet_one_line = _salvage_step2_tweet_like_text(raw_text=raw_text_str, sanitized_text=text)
            if tweet_one_line and not _step2_tweet_has_valid_public_body(tweet_one_line):
                tweet_one_line = ""
        if not tweet_one_line:
            tweet_one_line = _first_natural_sentence_from_text(raw_text_str) or _first_natural_sentence_from_text(text)
            tweet_one_line = _compact_step2_tweet_text(tweet_one_line, max_sentences=2) if tweet_one_line else ""
            if tweet_one_line and not _step2_tweet_has_valid_public_body(tweet_one_line):
                tweet_one_line = ""
        if tweet_one_line:
            tweet_one_line = _compact_step2_tweet_text(tweet_one_line, max_sentences=2)
        return raw_text_str, text, tweet_one_line, meta

    for attempt in range(1, max_attempts + 1):
        stateless_retry = attempt > 1
        if attempt == 1:
            prompt_to_use = _fix_quoted_protocol_label_breaks(_restore_step2_prompt_block_boundaries(_normalize_step2_request_header_text(prompt)))
        else:
            prompt_to_use = _fix_quoted_protocol_label_breaks(_restore_step2_prompt_block_boundaries(_normalize_step2_request_header_text(_build_world_aware_step2_compact_retry_prompt(
                agent_name=agent_name,
                claim_txt=claim_txt,
                current_belief=int(expected_current_belief),
            ))))
        _set_current_validation_context(prompt_to_use)
        _set_native_ollama_debug_context(step='step2', agent_name=agent_name, attempt=attempt, label='attempt')
        try:
            raw_text = get_llm_response(conversation, prompt_to_use, use_history=not stateless_retry, save_turn=not stateless_retry)
        finally:
            _clear_native_ollama_debug_context()
        raw_text_str = "" if raw_text is None else str(raw_text)
        _remember_attempt(attempt, raw_text_str)

        if _is_refusal_or_meta(raw_text_str):
            reason = 'refusal/meta output'
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            sanitized = sanitize_step2_output(raw_text_str)
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, sanitized, '')
            if attempt < max_attempts:
                if step2_quality == 'ok':
                    step2_quality = 'retry_success'
                _log_step2_event('retry', attempt, reason, raw_text_str, sanitized, '')
                continue
            step2_quality = 'fallback_hard'
            fallback_used = True
            final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
            final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=raw_text_str, sanitized_text=sanitized)
            _log_step2_event('fallback', attempt, reason, raw_text_str, sanitized, final_out)
            _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
            return final_out

        text = sanitize_step2_output(raw_text_str)
        r_i = int(expected_current_belief)

        tweet_body = extract_tweet_text(text) or extract_tweet_text(raw_text_str) or ""
        tweet_body = _strip_leaked_prompt_headers(tweet_body)
        tweet_one_line = re.sub(r"\s+", " ", tweet_body).strip()
        if tweet_one_line and not _step2_tweet_has_valid_public_body(tweet_one_line):
            tweet_one_line = ""

        if not tweet_one_line:
            tweet_one_line = strip_step2_meta_lines(text or raw_text_str)
            tweet_one_line = re.sub(r"\s+", " ", tweet_one_line).strip()

        if not tweet_one_line:
            tweet_one_line = _heuristic_freeform_protocol_body(raw_text_str, field="tweet")
            tweet_one_line = re.sub(r"\s+", " ", tweet_one_line).strip()

        if not tweet_one_line:
            tweet_one_line = _salvage_step2_tweet_like_text(raw_text=raw_text_str, sanitized_text=text)

        if not tweet_one_line:
            tweet_one_line = _first_natural_sentence_from_text(raw_text_str) or _first_natural_sentence_from_text(text)
            tweet_one_line = _compact_step2_tweet_text(tweet_one_line, max_sentences=2) if tweet_one_line else ""
        if tweet_one_line and _is_obviously_non_tweet_payload(tweet_one_line):
            tweet_one_line = ""

        if not tweet_one_line:
            reason = 'empty/broken tweet body'
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            _metric_inc('step2_empty_main')
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, '')
            if attempt < max_attempts:
                shown_tweet_ctx = _extract_step2_shown_tweet_from_prompt(prompt)
                shown_material_ctx = _extract_external_grounding_text(prompt_to_use)
                if is_closed_strict:
                    rescue_prompt = _build_strict_closed_step2_native_rescue_prompt(agent_name or 'you', claim_txt, int(expected_current_belief), shown_tweet=shown_tweet_ctx, shown_material=shown_material_ctx)
                    _metric_inc('step2_compact_rescue_attempts')
                    rescue_raw, rescue_text, rescue_tweet, rescue_meta = _native_rescue_step2_attempt(rescue_prompt, '1n_sc', include_history=False)
                    _remember_attempt(2, rescue_raw)
                    _log_step2_event('native_rescue', '1n_sc', 'step2_compact_native_rescue', rescue_raw, rescue_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {rescue_tweet}".strip() if rescue_tweet else ''))
                    rescue_has_bad_external, rescue_external_flags = _unsupported_external_reference_flags(
                        tweet_text=rescue_tweet,
                        world_mode=WORLD,
                        fact_pack_text=FACT_PACK_TEXT,
                        extra_grounding_text=shown_material_ctx,
                    ) if rescue_tweet else (False, [])
                    rescue_intensity_mismatch, rescue_intensity_reason = _step2_speaker_intensity_mismatch(rescue_tweet, r_i) if rescue_tweet else (False, "")
                    if rescue_tweet and not rescue_intensity_mismatch:
                        if rescue_has_bad_external:
                            _metric_inc('step2_compact_rescue_unsupported_external')
                            _log_step2_event('warning', '1n_sc', f"unsupported external-reference language: {rescue_external_flags}", rescue_raw, rescue_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {rescue_tweet}".strip() if rescue_tweet else ''))
                        _metric_inc('step2_compact_rescue_success')
                        raw_text_str = rescue_raw
                        text = rescue_text
                        tweet_one_line = rescue_tweet
                        step2_contaminated = bool(rescue_has_bad_external)
                        if step2_quality == 'ok':
                            step2_quality = 'retry_success'
                    else:
                        if rescue_tweet and rescue_intensity_mismatch:
                            _metric_inc('step2_compact_rescue_intensity_mismatch')
                            _log_step2_event('warning', '1n_sc', f"speaker-side intensity mismatch: {rescue_intensity_reason}", rescue_raw, rescue_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {rescue_tweet}".strip() if rescue_tweet else ''))
                        if str((rescue_meta or {}).get('done_reason') or '').strip().lower() == 'length':
                            _metric_inc('step2_compact_rescue_length_stop')
                        _metric_inc('step2_compact_rescue_fail')

                        ultra_prompt = _build_strict_closed_step2_ultra_compact_rescue_prompt(agent_name or 'you', claim_txt, int(expected_current_belief), shown_tweet=shown_tweet_ctx, shown_material=shown_material_ctx)
                        _metric_inc('step2_ultra_compact_rescue_attempts')
                        ultra_raw, ultra_text, ultra_tweet, ultra_meta = _native_rescue_step2_attempt(ultra_prompt, '1n_uc', include_history=False)
                        _remember_attempt(2, ultra_raw)
                        _log_step2_event('native_rescue', '1n_uc', 'step2_ultra_compact_native_rescue', ultra_raw, ultra_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {ultra_tweet}".strip() if ultra_tweet else ''))
                        ultra_has_bad_external, ultra_external_flags = _unsupported_external_reference_flags(
                            tweet_text=ultra_tweet,
                            world_mode=WORLD,
                            fact_pack_text=FACT_PACK_TEXT,
                            extra_grounding_text=shown_material_ctx,
                        ) if ultra_tweet else (False, [])
                        ultra_intensity_mismatch, ultra_intensity_reason = _step2_speaker_intensity_mismatch(ultra_tweet, r_i) if ultra_tweet else (False, "")
                        if ultra_tweet and not ultra_intensity_mismatch:
                            if ultra_has_bad_external:
                                _metric_inc('step2_ultra_compact_rescue_unsupported_external')
                                _log_step2_event('warning', '1n_uc', f"unsupported external-reference language: {ultra_external_flags}", ultra_raw, ultra_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {ultra_tweet}".strip() if ultra_tweet else ''))
                            _metric_inc('step2_ultra_compact_rescue_success')
                            raw_text_str = ultra_raw
                            text = ultra_text
                            tweet_one_line = ultra_tweet
                            step2_contaminated = bool(ultra_has_bad_external)
                            if step2_quality == 'ok':
                                step2_quality = 'retry_success'
                        else:
                            if ultra_tweet and ultra_intensity_mismatch:
                                _metric_inc('step2_ultra_compact_rescue_intensity_mismatch')
                                _log_step2_event('warning', '1n_uc', f"speaker-side intensity mismatch: {ultra_intensity_reason}", ultra_raw, ultra_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {ultra_tweet}".strip() if ultra_tweet else ''))
                            if str((ultra_meta or {}).get('done_reason') or '').strip().lower() == 'length':
                                _metric_inc('step2_ultra_compact_rescue_length_stop')
                            _metric_inc('step2_ultra_compact_rescue_fail')
                            step2_quality = 'fallback_hard'
                            fallback_used = True
                            final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
                            final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=ultra_raw or rescue_raw or raw_text_str, sanitized_text=ultra_text or rescue_text or text)
                            _log_step2_event('fallback', '1n_uc', reason, ultra_raw or rescue_raw, ultra_text or rescue_text, final_out)
                            _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
                            return final_out
                else:
                    _metric_inc('step2_native_rescue_attempts')
                    rescue_raw, rescue_text, rescue_tweet, rescue_meta = _native_rescue_step2_attempt(prompt, '1n', include_history=True)
                    _remember_attempt(2, rescue_raw)
                    _log_step2_event('native_rescue', '1n', 'step2_empty_native_rescue', rescue_raw, rescue_text, (f"FINAL_RATING: {int(r_i)}\nTWEET: {rescue_tweet}".strip() if rescue_tweet else ''))
                    if rescue_tweet:
                        _metric_inc('step2_native_rescue_success')
                        raw_text_str = rescue_raw
                        text = rescue_text
                        tweet_one_line = rescue_tweet
                        if step2_quality == 'ok':
                            step2_quality = 'retry_success'
                    else:
                        if str((rescue_meta or {}).get('done_reason') or '').strip().lower() == 'length':
                            _metric_inc('step2_native_rescue_length_stop')
                        _metric_inc('step2_native_rescue_empty_final')
                        step2_quality = 'retry_success'
                        _log_step2_event('retry', attempt, reason, raw_text_str, text, '')
                        continue
            else:
                step2_quality = 'fallback_hard'
                fallback_used = True
                final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
                final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=raw_text_str, sanitized_text=text)
                _log_step2_event('fallback', attempt, reason, raw_text_str, text, final_out)
                _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
                return final_out

        tweet_one_line = _compact_step2_tweet_text(tweet_one_line, max_sentences=2)
        if stateless_retry:
            tweet_one_line = _polish_compact_step2_retry_tweet(tweet_one_line, r_i, claim_txt=claim_txt)

        step2_contaminated = False
        grounding_text = _extract_external_grounding_text(prompt_to_use)
        has_unsupported_external_ref, external_ref_flags = _unsupported_external_reference_flags(
            tweet_text=tweet_one_line,
            world_mode=WORLD,
            fact_pack_text=FACT_PACK_TEXT,
            extra_grounding_text=grounding_text,
        )
        if has_unsupported_external_ref:
            _metric_inc('closed_world_leak_step2')
            reason = f"unsupported external-reference language: {external_ref_flags}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            if step2_quality == 'ok':
                step2_quality = 'warning_soft'
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            step2_contaminated = True

        topic_claim_flags = _step2_possible_unsupported_v130_topic_claim(
            tweet_text=tweet_one_line,
            claim_text=claim_txt,
            prompt_text=prompt_to_use,
            extra_grounding_text=grounding_text,
        )
        if topic_claim_flags:
            reason = f"step2_possible_unsupported_topic_claim:{topic_claim_flags}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            _metric_inc('step2_possible_unsupported_topic_claim')
            for flag in topic_claim_flags:
                _metric_inc(f'step2_possible_unsupported_topic_claim::{flag}')
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            if step2_quality == 'ok':
                step2_quality = 'warning_soft'
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            # Warning-only: do not mark contaminated, retry, or fallback here.
        pm1_strong_flags = _step2_pm1_overstrong_wording(tweet_one_line, r_i)
        if pm1_strong_flags:
            reason = f"step2_pm1_overstrong_wording:{pm1_strong_flags}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            _metric_inc('step2_pm1_overstrong_wording')
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            # Warning-only: mild +/-1 tweet used strong wording; do not retry or fallback.


        wrong_dir, wrong_dir_reason = _clear_step2_wrong_direction(tweet_one_line, r_i)
        if wrong_dir:
            reason = f"clear wrong-direction tweet opener: {wrong_dir_reason}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            if attempt < max_attempts:
                if step2_quality == 'ok':
                    step2_quality = 'retry_success'
                _log_step2_event('retry', attempt, reason, raw_text_str, text, '')
                continue
            step2_quality = 'fallback_hard'
            fallback_used = True
            final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
            final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=raw_text_str, sanitized_text=text)
            _log_step2_event('fallback', attempt, reason, raw_text_str, text, final_out)
            _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
            return final_out

        self_undermining, self_undermining_reason = _step2_self_undermining_concession(tweet_one_line, r_i)
        if self_undermining:
            reason = f"self-undermining mixed concession: {self_undermining_reason}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            if attempt < max_attempts:
                if step2_quality == 'ok':
                    step2_quality = 'retry_success'
                _log_step2_event('retry', attempt, reason, raw_text_str, text, '')
                continue
            if _step2_self_undermining_soft_only(tweet_one_line, r_i, self_undermining_reason):
                if step2_quality == 'ok':
                    step2_quality = 'warning_soft'
                return _finalize_step2_success(current_final, contaminated=step2_contaminated, source='main')
            step2_quality = 'fallback_hard'
            fallback_used = True
            final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
            final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=raw_text_str, sanitized_text=text)
            _log_step2_event('fallback', attempt, reason, raw_text_str, text, final_out)
            _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
            return final_out

        canned_reason = _generic_canned_reason_reason(tweet_one_line, field="tweet")
        if canned_reason:
            reason = f"generic canned tweet reasoning: {canned_reason}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            if attempt < max_attempts and r_i in {0, 1}:
                if step2_quality == 'ok':
                    step2_quality = 'retry_success'
                _log_step2_event('retry', attempt, reason, raw_text_str, text, '')
                continue
            if step2_quality == 'ok':
                step2_quality = 'warning_soft'
            return _finalize_step2_success(current_final, contaminated=step2_contaminated, source='main')

        persona_canned_reason = _step2_persona_canned_reason(tweet_one_line, current_belief=r_i)
        if persona_canned_reason:
            reason = f"step2_too_impersonal_or_canned: {persona_canned_reason}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            # Do not spend a full retry/rewrite cycle on the common neutral canned pattern.
            if persona_canned_reason == 'overwhelming_evidence_formula':
                if step2_quality == 'ok':
                    step2_quality = 'warning_soft'
                return _finalize_step2_success(current_final, contaminated=step2_contaminated, source='main')
            if attempt < max_attempts and r_i in {0, 1}:
                if step2_quality == 'ok':
                    step2_quality = 'retry_success'
                _log_step2_event('retry', attempt, reason, raw_text_str, text, '')
                continue
            if step2_quality == 'ok':
                step2_quality = 'warning_soft'
            return _finalize_step2_success(current_final, contaminated=step2_contaminated, source='main')

        intensity_mismatch, intensity_reason = _step2_speaker_intensity_mismatch(tweet_one_line, r_i)
        if intensity_mismatch:
            reason = f"speaker-side intensity mismatch: {intensity_reason}"
            try:
                print(f"[warn][step2][{agent_name or 'UNKNOWN'}] {reason} on attempt {attempt}/{max_attempts}")
            except Exception:
                pass
            current_final = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
            _add_warning_tag(attempt, reason)
            _log_step2_event('warning', attempt, reason, raw_text_str, text, current_final)
            if _step2_intensity_mismatch_soft_only(tweet_one_line, r_i, intensity_reason):
                if step2_quality == 'ok':
                    step2_quality = 'warning_soft'
                return _finalize_step2_success(current_final, contaminated=step2_contaminated, source='main')
            step2_quality = 'fallback_hard'
            fallback_used = True
            final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
            final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text=raw_text_str, sanitized_text=text)
            _log_step2_event('fallback', attempt, reason, raw_text_str, text, final_out)
            _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
            return final_out

        final_out = f"FINAL_RATING: {int(r_i)}\nTWEET: {tweet_one_line}".strip()
        if attempt > 1:
            if step2_quality == 'ok':
                step2_quality = 'retry_success'
            _log_step2_event('rewrite_success', attempt, 'step2_retry_succeeded', raw_text_str, text, final_out)
        return _finalize_step2_success(final_out, contaminated=step2_contaminated, source=('rescue' if attempt > 1 or raw_attempt_2 else 'main'))

    final_fb = _fallback_step2(expected_current_belief=int(expected_current_belief), claim_txt=claim_txt)
    final_out = _maybe_skip_protocol_fallback('step2', final_fb, raw_text='', sanitized_text='')
    warning_tags.append('loop_exhausted')
    step2_quality = 'fallback_hard'
    fallback_used = True
    _dump_retry_debug_event('step2', 'fallback', agent_name=agent_name, attempt=max_attempts, reasons='loop_exhausted', prompt_text=prompt, raw_text='', sanitized_text='', final_text=final_out, conversation=conversation, include_history=False)
    _set_step2_call_info(conversation, raw_attempt_1=raw_attempt_1, raw_attempt_2=raw_attempt_2, warning_tags=warning_tags, quality=step2_quality, fallback_used=fallback_used, final_output=final_out)
    return final_out

def _strip_leaked_prompt_headers(tweet_body: str) -> str:
    """Remove leaked meta headers (e.g., 'Assumption Stress-Test:') from within the tweet body.

    Keeps the substantive content after the colon.
    This is protocol hygiene only (does not change stance).
    """
    if not tweet_body:
        return tweet_body
    s = str(tweet_body)

    # Remove variations like: "Assumption Stress-Test: <text>"
    s = re.sub(r"(?i)\bassumption\s*stress\s*[-–]?\s*test\s*:\s*", "", s)

    # Also remove standalone header occurrences (rare)
    s = re.sub(r"(?im)^\s*assumption\s*stress\s*[-–]?\s*test\s*$", "", s).strip()

    # Remove leaked template placeholder tokens that should never surface in tweet text.
    s = re.sub(r"(?i)\b(?:AGENT_PERSONA_CARD|FACT_PACK|BIAS|WORLD_RULES|STEP3_OPEN_WORLD_RULES|STEP3_FACT_PACK_RULES|STEP3_RAG_RULES|STEP2_RAG_RULES)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    return s





# Global protocol knobs (set in main after argparse)
# DEFAULT_MAX_STEP_CHANGE is set from CLI (args.max_step_change). Do not override it here.
# DEFAULT_MAX_STEP_CHANGE: if set, Step-3 FINAL_RATING must not change more than this amount per interaction.

# Policy when Step-3 FINAL_RATING is invalid (out of ALLOWED_SET / violates step-change).
# - 'project': smart projection: clamp simple same-direction overshoots to the nearest allowed
#              boundary, but fall back to PREVIOUS_RATING when the invalid move points in the
#              wrong direction or is otherwise inconsistent.
# - 'stay': keep PREVIOUS_RATING (if allowed) instead of moving.
INVALID_STEP3_POLICY = "project"

def _is_refusal_or_meta(text: str) -> bool:
    """Heuristic detector for model refusals / meta-chat that break protocol."""
    if not text:
        return False
    s = str(text).strip().lower()
    # Common refusal / policy boilerplate patterns (covers many local model guardrails too).
    needles = [
        "i cannot", "i can't", "cannot comply", "can't comply", "i'm sorry", "i am sorry",
        "as an ai", "as a language model", "i'm an ai", "i am an ai",
        "i can't help with", "i cannot help with", "i won't", "i will not",
        "i'm unable", "i am unable", "cannot assist", "can't assist",
    ]
    return any(n in s for n in needles)

def _visible_planning_meta_reason(text: str, field: str = "generic") -> str | None:
    if not _content_enforcement_enabled():
        return None
    """Detect visible planning / prompt-echo text that should not appear in final protocol fields."""
    if not text:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip().lower()
    if not s:
        return None

    generic_patterns = [
        (r"\bthe user wants me to\b", "mentions_user_request"),
        (r"\bthe prompt (?:says|asks|wants)\b", "mentions_prompt"),
        (r"\bthe instruction(?:s)? (?:say|says|ask|asks|want|wants)\b", "mentions_instructions"),
        (r"\boutput exactly\b", "mentions_output_protocol"),
        (r"\bfinal[_ ]rating\b", "mentions_final_rating_label"),
        (r"\ballowed(?:[_ ]final)?[_ ]rating(?:[_ ]set)?\b", "mentions_allowed_set"),
        (r"\bprevious[_ ]rating\b", "mentions_previous_rating"),
        (r"\bcurrent[_ ]rating\b", "mentions_current_rating"),
        (r"\b(?:remains?|staying) at (?:my |the |a )?(?:current |same )?(?:rating|stance|position)\b", "mentions_current_rating"),
        (r"\brole-?play as\b", "mentions_roleplay"),
        (r"\bfirst,? i should\b", "visible_planning_first_i_should"),
        (r"\bi should (?:write|respond|choose|make sure|keep|avoid|output)\b", "visible_planning_i_should"),
        (r"\bi need to (?:write|respond|choose|make sure|keep|avoid|output)\b", "visible_planning_i_need_to"),
        (r"\blet'?s see\b", "visible_planning_lets_see"),
        (r"\bthe task is to\b", "mentions_task"),
        (r"\bi have to (?:write|respond|choose|decide|explain)\b", "visible_planning_i_have_to"),
        (r"\bi'?m being asked to\b", "visible_planning_being_asked"),
        (r"\bas the (?:listener|speaker|agent)\b", "mentions_role_context"),
    ]

    tweet_patterns = [
        (r"\bwrite a tweet\b", "mentions_write_tweet"),
        (r"\bthis tweet should\b", "mentions_tweet_planning"),
        (r"\bthe tweet should\b", "mentions_tweet_planning"),
        (r"\bmy tweet\b", "mentions_my_tweet"),
    ]

    explanation_patterns = [
        (r"\bchoose a rating\b", "mentions_choose_rating"),
        (r"\bi will keep the same rating\b", "visible_rating_planning"),
        (r"\bi should keep the same rating\b", "visible_rating_planning"),
        (r"\bi need to explain\b", "mentions_explanation_planning"),
        (r"\bmy explanation\b", "mentions_my_explanation"),
        (r"\bmy (?:current|existing|prior) (?:view|belief|stance)\b", "mentions_hidden_state"),
        (r"\bmy (?:prior|existing) skepticism\b", "mentions_hidden_state"),
        (r"\bmy openness to update\b", "mentions_hidden_state"),
        (r"\bmy institutional trust\b", "mentions_hidden_state"),
        (r"\baligns? with my institutional trust\b", "mentions_hidden_state"),
        (r"\bmy existing strong agreement\b", "mentions_hidden_state"),
        (r"\blow openness to update\b", "mentions_hidden_state"),
        (r"\bmy persona\b", "mentions_hidden_state"),
    ]

    pats = list(generic_patterns)
    if field == "tweet":
        pats.extend(tweet_patterns)
    elif field == "explanation":
        pats.extend(explanation_patterns)

    for pat, reason in pats:
        m = re.search(pat, s, flags=re.I)
        if not m:
            continue
        if field == "explanation" and reason == "mentions_hidden_state":
            matched_text = m.group(0)
            if _persona_allows_hidden_state_phrase(matched_text, _current_persona_validation_policy()):
                continue
        return reason
    return None




def _generic_canned_reason_reason(text: str, field: str = "generic") -> str | None:
    if not _content_enforcement_enabled():
        return None
    """Detect broad canned reasoning that ignores concrete shown material.

    This fires only when some fact-pack / RAG / shown-material grounding is actually present in
    the current prompt context. It intentionally checks the material shown in this prompt, not
    the whole source file on disk.
    """
    if not text or not str(text).strip():
        return None
    grounding_text = _current_validation_grounding_text()
    if not str(grounding_text or '').strip():
        return None

    s = re.sub(r"\s+", " ", str(text)).strip()
    low = s.lower()

    patterns = [
        (r"\bno credible evidence\b", "generic_no_credible_evidence"),
        (r"\bno (?:real|solid|clear|concrete) evidence\b", "generic_no_concrete_evidence"),
        (r"\bnot definitively proven\b", "generic_not_definitively_proven"),
        (r"\b(?:can't|cannot) be definitively proven\b", "generic_not_definitively_proven"),
        (r"\bnot fully proven\b", "generic_not_fully_proven"),
        (r"\bdoes(?: not|n't) fully prove\b", "generic_not_fully_proven"),
        (r"\bwithout (?:direct|solid|clear|concrete) proof\b", "generic_no_direct_proof"),
        (r"\blacks? (?:direct|solid|clear|concrete) proof\b", "generic_lacks_direct_proof"),
        (r"\bfrom this (?:tweet|message) alone\b", "generic_from_this_alone"),
        (r"\bby itself\b", "generic_from_this_alone"),
        (r"\bon its own\b", "generic_from_this_alone"),
        (r"\bnot enough (?:evidence|proof|detail|details|information|context)\b", "generic_not_enough_evidence"),
        (r"\bstill need (?:more|stronger) (?:evidence|proof)\b", "generic_need_more_evidence"),
        (r"\bremain(?:s|ing)? skeptical\b", "generic_remain_skeptical"),
        (r"\bremain(?:s|ing)? unconvinced\b", "generic_remain_unconvinced"),
        (r"\bstill (?:uncertain|unsure|skeptical|unconvinced)\b", "generic_still_uncertain"),
        (r"\bboth sides (?:seem|sound) (?:about )?(?:equally )?(?:plausible|possible|uncertain)\b", "generic_both_sides"),
    ]
    matched_reason = None
    for pat, reason in patterns:
        if re.search(pat, low, flags=re.I):
            matched_reason = reason
            break
    if not matched_reason:
        return None
    if _validator_allows_generic_reason(matched_reason, s, _current_persona_validation_policy()):
        return None

    # If the text clearly mentions a concrete shown-material point, do not flag it as canned.
    if _tweet_matches_specific_fact_pack_polarity(s, 'positive') or _tweet_matches_specific_fact_pack_polarity(s, 'negative'):
        return None

    toks = set(_fact_match_tokens(s))
    digits = {tok for tok in toks if tok.isdigit()}
    non_generic = {tok for tok in toks if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
    if digits or len(non_generic) >= 2:
        return None

    return matched_reason


def _step3_open_world_generic_reference_reason(text: str, tweet_text: str = "") -> str | None:
    """Flag open-world explanations that lean on generic web/background confirmation instead of one tweet-local point."""
    if not text:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip()
    low = s.lower()
    patterns = [
        (r"\baligns? with (?:the )?(?:web results|shown results|live web results)\b", "generic_web_alignment"),
        (r"\b(?:web|shown|live web) results? (?:confirm|support|show)\b", "generic_web_alignment"),
        (r"\bverified historical data\b", "generic_verified_historical_data"),
        (r"\bhistorical data\b", "generic_historical_data_reference"),
        (r"\bdocumented moon landings?\b", "generic_documented_moon_landings"),
        (r"\bconfirms? (?:the claim|it was real)\b", "generic_confirms_claim"),
        (r"\bthe claim is well documented\b", "generic_documented_claim"),
    ]
    for pat, reason in patterns:
        if re.search(pat, low, flags=re.I):
            if _validator_allows_generic_reason(reason, s, _current_persona_validation_policy()):
                return None
            if reason == 'stock_evidence_phrase' and _step2_stock_evidence_has_concrete_local_point(s):
                return None
            return reason
    return None


def _step3_true_open_point_mismatch_reason(text: str, tweet_text: str = "", prompt_text: str = "") -> str | None:
    """Flag true-open explanations that do not answer the tweet's actual point.

    This is intentionally shape-based, not a banned-word list:
    - allow normal claim language in true open
    - flag explanations that rely on prior-state inertia as the reason
    - flag whole-claim verdicts that are not grounded in the tweet or shown material
    """
    if not text:
        return None
    expl = re.sub(r"\s+", " ", str(text or "")).strip()
    if not expl:
        return None
    low = expl.lower()

    shown = _extract_compact_step3_shown_material(prompt_text or "", max_items=3, max_chars_per_item=280)
    ref_text = " ".join(x for x in [str(tweet_text or ""), str(shown or "")] if str(x).strip())
    ref_low = ref_text.lower()

    expl_toks = set(_fact_match_tokens(expl))
    ref_toks = set(_fact_match_tokens(ref_text))
    shared_digits = {tok for tok in (expl_toks & ref_toks) if tok.isdigit()}
    shared_non_generic = {tok for tok in (expl_toks & ref_toks) if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}
    grounded = bool(shared_digits or len(shared_non_generic) >= 1)

    motifs = [
        "apollo", "nasa", "lunar", "samples", "sample", "footage", "video", "videos",
        "moon", "rocks", "rock", "evidence", "telemetry", "retroreflector", "surveyor",
        "sstv", "tapes", "records", "data", "shadows", "flag", "radiation", "van allen"
    ]
    if not grounded:
        grounded = any(m in low and m in ref_low for m in motifs)

    state_inertia = bool(re.search(
        r"\b(?:current (?:view|rating|balance)|existing (?:skepticism|doubt|view|stance)|prior stance|prior view|prior rating|already at\s*-?[012]|no change is needed|no change needed|already unchanged|existing skepticism|remaining doubt)\b",
        low,
        flags=re.I,
    ))
    if state_inertia:
        if not _persona_allows_hidden_state_phrase(expl, _current_persona_validation_policy()):
            return "true_open_state_based_explanation"

    whole_claim_verdict = bool(re.search(
        r"\b(?:the claim is|claim is|it is)\s+(?:factually correct|factually incorrect|true|false|proven|disproven|verified|settled)\b|\bfactually (?:correct|incorrect)\b",
        low,
        flags=re.I,
    ))
    weakness_language = bool(re.search(
        r"\b(?:lacks?|lack of|missing|without|does(?: not|n't)|not enough|too vague|too generic|insufficient|unsupported|unclear|unresolved|weak(?:ness)?|gap|limitation|fails? to|does(?: not|n't) address|does(?: not|n't) explain)\b",
        low,
        flags=re.I,
    ))
    if whole_claim_verdict and not grounded and not weakness_language:
        return "true_open_generic_claim_verdict"
    return None


def _step2_stock_evidence_has_concrete_local_point(text: str) -> bool:
    """Detect whether stock evidence language is grounded in a concrete local point."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return False
    low = s.lower()
    toks = set(_fact_match_tokens(s))
    non_generic = {tok for tok in toks if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}

    local_gap_patterns = [
        r"\bdirect way to (?:test|confirm|verify)\b",
        r"\b(?:does(?: not|n't)|cannot|can't) offer a direct way to (?:test|confirm|verify)\b",
        r"\btestable boundary\b",
        r"\bwithout a testable boundary\b",
        r"\bdetect the code\b",
        r"\bif we(?:'re| are) in a simulation,? why\b",
        r"\bwhy can(?:not|'t) we detect\b",
        r"\blaw[- ]governed\b",
        r"\binformation[- ]like universe\b",
        r"\badvanced civilizations? could simulate us\b",
    ]
    if any(re.search(p, low, flags=re.I) for p in local_gap_patterns):
        return True
    if '?' in s and len(non_generic) >= 1:
        return True
    if re.search(r"\b(?:why|how|what|where|which)\b", low, flags=re.I) and len(non_generic) >= 2:
        return True
    if len(non_generic) >= 4 and re.search(r"\b(?:code|rules?|boundary|program|simulat(?:e|ed|ion|or)|detail|details|gap|mismatch|assumption|unresolved|test|confirm|verify)\b", low, flags=re.I):
        return True
    if len(non_generic) >= 5 and re.search(r"[,:;]|\bbut\b|\bbecause\b", low, flags=re.I):
        return True
    return False


def _step2_persona_canned_reason(text: str, current_belief: int | None = None) -> str | None:
    """Conservative detector for Step-2 tweets that read like impersonal canned summaries, not a person's tweet."""
    if not text:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip()
    low = s.lower()
    if not low:
        return None

    # If the tweet already sounds person-owned / conversational, do not flag it.
    if re.search(r"\b(i|i'm|i am|i think|i still|i just|i don't|i do not|i can't|i cannot|for me|to me|honestly|personally|from what i can tell)\b", low, flags=re.I):
        return None

    open_world = _is_open_world_mode(WORLD)
    patterns = [
        (r"^(?:the|this)\s+(?:evidence|claim|tweet|fact pack|argument|case)\b", "impersonal_summary_opening"),
        (r"\bhistoric event,? but debates? (?:about .*?)?persist\b", "debate_summary_formula"),
        (r"\b(?:records?|photos?|videos?|samples?) back that up\b", "back_that_up_formula"),
        (r"\btoo circumstantial and unverifiable\b", "circumstantial_unverifiable_formula"),
        (r"\bindependent verification on-site\b", "verification_on_site_formula"),
        (r"\bphysical evidence is undeniable\b", "undeniable_evidence_formula"),
        (r"\bevidence is overwhelming\b", "overwhelming_evidence_formula"),
        (r"\bcommon skeptical claim\b", "stock_skeptical_phrase"),
        (r"\bdoes(?: not|n't) outweigh\b", "weighing_formula"),
        (r"\bthe fact pack\b", "mentions_fact_pack"),
        (r"\bthe tweet(?:'s)? point\b", "impersonal_tweet_summary"),
    ]
    if not open_world:
        # For belief=0, evidence-absence language ("without clear evidence",
        # "no valid evidence") IS the correct 0-stance expression, not a canned
        # phrase.  Only apply the stock_evidence_phrase check for non-zero beliefs.
        try:
            _skip_stock = (current_belief is not None and int(current_belief) == 0)
        except Exception:
            _skip_stock = False
        if not _skip_stock:
            patterns.insert(1, (r"\b(?:overwhelming|strong|clear|credible|valid) evidence\b", "stock_evidence_phrase"))
    for pat, reason in patterns:
        if re.search(pat, low, flags=re.I):
            if _validator_allows_generic_reason(reason, s, _current_persona_validation_policy()):
                return None
            if reason == 'stock_evidence_phrase' and _step2_stock_evidence_has_concrete_local_point(s):
                return None
            return reason
    return None


def _step3_generic_gap_reason(text: str, tweet_text: str = "", pre_value: int | None = None, final_value: int | None = None, tweet_stance_local: str | None = None) -> str | None:
    if not _content_enforcement_enabled():
        return None
    """Detect Step-3 explanations that rely on generic weighing language instead of a concrete tweet-specific gap."""
    if not text:
        return None
    s = re.sub(r"\s+", " ", str(text)).strip()
    low = s.lower()
    if not low:
        return None

    matched_reason = None
    patterns = [
        (r"\bstrong evidence\b", "generic_strong_evidence"),
        (r"\boverwhelming evidence\b", "generic_overwhelming_evidence"),
        (r"\bexisting skepticism\b", "generic_existing_skepticism"),
        (r"\bclaim(?:'s)? credibility\b", "generic_claim_credibility"),
        (r"\bdoes(?: not|n't) outweigh\b", "generic_does_not_outweigh"),
        (r"\bcommon skeptical claim\b", "generic_common_skeptical_claim"),
        (r"\bvalid evidence,? but\b", "generic_valid_evidence_but"),
        (r"\bthe fact pack\b", "generic_fact_pack_reference"),
        (r"\bknown facts?\b", "generic_known_facts_reference"),
        (r"\boverall evidence\b", "generic_overall_evidence"),
        (r"\bbroader context\b", "generic_broader_context"),
        (r"\bshown above\b", "generic_shown_above_reference"),
        (r"\bwhole case\b", "generic_whole_case_reference"),
        (r"\bentire case\b", "generic_entire_case_reference"),
        (r"\bnot addressed by the fact pack\b", "generic_not_addressed_by_fact_pack"),
        (r"\bthe tweet(?:'s)? point does(?: not|n't)\b", "generic_tweet_point_formula"),
    ]
    for pat, reason in patterns:
        if re.search(pat, low, flags=re.I):
            matched_reason = reason
            break
    if not matched_reason:
        return None
    if _validator_allows_generic_reason(matched_reason, s, _current_persona_validation_policy()):
        return None

    tweet_toks = set(_fact_match_tokens(tweet_text or ""))
    expl_toks = set(_fact_match_tokens(s))
    shared_digits = {tok for tok in (tweet_toks & expl_toks) if tok.isdigit()}
    shared_non_generic = {tok for tok in (tweet_toks & expl_toks) if tok not in _FACT_SPECIFICITY_GENERIC_TOKENS and not tok.isdigit()}

    has_specific_gap_language = bool(re.search(
        r"\b(?:because|since|whereas|while|but not|only addresses?|focus(?:es|ed)? on|does(?: not|n't) (?:address|explain|resolve|answer|undercut|rule out|account for)|fails? to (?:address|explain|resolve|answer|undercut|rule out|account for)|leaves? open|still leaves?)\b",
        low,
        flags=re.I,
    ))

    # If the explanation both anchors itself to the tweet's concrete point and states a specific limitation, allow it.
    if has_specific_gap_language and (shared_digits or len(shared_non_generic) >= 2):
        return None

    # Fact-pack/global-weighing formulas are too generic unless they are clearly tied to a specific tweet gap.
    severe = {
        "generic_overwhelming_evidence",
        "generic_strong_evidence",
        "generic_existing_skepticism",
        "generic_claim_credibility",
        "generic_does_not_outweigh",
        "generic_fact_pack_reference",
        "generic_not_addressed_by_fact_pack",
        "generic_tweet_point_formula",
    }
    if matched_reason in severe:
        return matched_reason

    # For milder stock phrases, tolerate only when there is strong tweet-specific grounding.
    if shared_digits or len(shared_non_generic) >= 2:
        return None
    return matched_reason


def _clear_step2_wrong_direction(tweet_text: str, expected_current_belief: int) -> tuple[bool, str]:
    if not _content_enforcement_enabled():
        return False, ""
    """Very conservative detector for obviously wrong-direction Step-2 tweets.

    This is intentionally light-touch: it only flags clear polarity failures, mainly from the
    opening sentence, so Step-2 retry does not become overly strict.
    """
    if not tweet_text:
        return False, ""

    try:
        r_i = int(expected_current_belief)
    except Exception:
        return False, ""

    if r_i == 0:
        return False, ""

    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return False, ""

    opener = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip()
    if not opener:
        opener = s

    # Words like "plausible", "coherent", "consistent" read as positive only when
    # they are NOT negated in the same clause.  "lacks a plausible mechanism" or
    # "not consistent with reality" should not register as FOR signals.
    _NEG_BEFORE = re.compile(
        r"\b(?:lack[s]?|without|no\b|not\b|doesn?'?t?|isn?'?t?|can'?t|cannot|"
        r"fail[s]?|missing|absent|neither|nor)\s*(?:\w+\s+){0,3}$",
        re.I,
    )

    def _negated_context(opener_text: str, match_start: int) -> bool:
        """True when a negating word appears within ~4 words before the match."""
        window = opener_text[max(0, match_start - 60): match_start]
        return bool(_NEG_BEFORE.search(window))

    # Patterns that unambiguously read FOR the claim (no negation risk).
    unambiguous_pos_patterns = [
        r"\bi believe\b",
        r"\bi buy\b",
        r"\bi think\b.*\b(?:real|likely|plausible|happened|landed)\b",
        r"\bseems believable\b",
        r"\bseems plausible\b",
        r"\bmakes sense\b",
        r"\bsounds convincing\b",
        r"\bwell explained\b",
        r"\bi[' ]?m convinced\b",
        r"\bi am convinced\b",
        r"\bi lean toward believing\b",
        r"\bi lean toward accepting\b",
        r"\bi lean toward support(?:ing)?\b",
        r"\bi lean toward the claim\b",
        r"\bi(?: am|[' ]m)? somewhat in favor\b",
        r"\bi(?: somewhat)? accept\b",
        r"\blikely real\b",
        r"\bprobably happened\b",
    ]
    # Patterns that look positive but can appear in negated contexts — check
    # that they are not preceded by a negating word before counting them.
    ambiguous_pos_patterns = [
        r"\bplausible\b",
        r"\bcoherent\b",
        r"\bconsistent\b",
    ]

    neg_patterns = [
        r"\bi doubt\b",
        r"\bi don't buy\b",
        r"\bi do not buy\b",
        r"\bi think\b.*\b(?:fake|hoax|staged|unlikely|did not happen)\b",
        r"\bnot convinced\b",
        r"\bi(?: am|[' ]m)? skeptical\b",
        r"\bi(?: am|[' ]m)? unconvinced\b",
        r"\bi(?: still )?(?:do not|don't) see (?:any )?concrete evidence\b",
        r"\bi(?: still )?(?:do not|don't) see proof\b",
        r"\bi(?: am|[' ]m)? doubtful\b",
        r"\bi lean against\b",
        r"\bi lean toward doubting\b",
        r"\bi lean toward doubt\b",
        r"\bi lean toward skepticism\b",
        r"\bi lean toward rejecting\b",
        r"\bi lean toward disbelieving\b",
        r"\btoo vague\b",
        r"\bdoesn't add up\b",
        r"\bdoes not add up\b",
        r"\bhard to believe\b",
        r"\bnot enough (?:information|evidence|detail)\b",
        r"\binconsistent\b",
        r"\bunexplained\b",
    ]

    # Scope: judge only the opening sentence for strong ratings (+/-2, strict),
    # but judge the whole tweet for mild ratings (+/-1) so a tentative +/-1 may
    # open with a concession to the other side as long as the tweet does not,
    # overall, lean clearly to the wrong sign.
    scope = s if abs(r_i) == 1 else opener
    scope_label = "tweet" if abs(r_i) == 1 else "opener"
    # Strip common intensifiers so "I strongly doubt" / "I really believe" still match the
    # base direction patterns (e.g. \bi doubt\b). Intensifiers do not change polarity.
    scope = re.sub(r"\b(?:strongly|really|very|seriously|quite|absolutely|truly|totally|completely|fully|deeply|honestly|genuinely)\s+", "", scope)

    has_pos = any(re.search(p, scope, flags=re.I) for p in unambiguous_pos_patterns)
    if not has_pos:
        for p in ambiguous_pos_patterns:
            m = re.search(p, scope, re.I)
            if m and not _negated_context(scope, m.start()):
                has_pos = True
                break

    has_neg = any(re.search(p, scope, flags=re.I) for p in neg_patterns)

    if r_i > 0 and has_pos:
        return False, ""
    if r_i < 0 and has_neg:
        return False, ""

    if r_i > 0 and has_neg and not has_pos:
        return True, f"clear {scope_label} reads AGAINST for a positive current rating"
    if r_i < 0 and has_pos and not has_neg:
        return True, f"clear {scope_label} reads FOR for a negative current rating"
    return False, ""





def _step2_self_undermining_concession(tweet_text: str, expected_current_belief: int) -> tuple[bool, str]:
    if not _content_enforcement_enabled():
        return False, ""
    """Light-touch detector for mixed / self-undermining Step-2 tweets at nonzero ratings.

    Goal: catch tweets like 'I somewhat agree ... but I still need more evidence' or the
    negative-side mirror, without over-policing natural mild wording.
    """
    if not tweet_text:
        return False, ""

    try:
        r_i = int(expected_current_belief)
    except Exception:
        return False, ""

    if r_i == 0:
        return False, ""

    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return False, ""

    mixed_connector = bool(re.search(r"\b(?:but|however|though|although|still)\b", s, flags=re.I))
    if not mixed_connector:
        return False, ""

    negative_caveat_patterns = [
        r"\bstill need more evidence\b",
        r"\bstill need stronger evidence\b",
        r"\bstill unsure\b",
        r"\bi(?: am|[' ]m)? still unsure\b",
        r"\bnot enough (?:information|evidence|detail)\b",
        r"\black of (?:evidence|details?)\b",
        r"\bdoes(?: not|n't) add up\b",
        r"\bhard to believe\b",
        r"\braises? doubt\b",
        r"\bmakes? me hesitant\b",
    ]
    positive_caveat_patterns = [
        r"\bseems plausible\b",
        r"\bmakes sense\b",
        r"\bsounds convincing\b",
        r"\bcoherent\b",
        r"\bconsistent\b",
        r"\bevidence for the claim\b",
        r"\bsupports the claim\b",
        r"\bgives? me some reason to believe\b",
        r"\bmakes? me less skeptical\b",
        r"\bmore open to the claim\b",
    ]

    if r_i > 0 and any(re.search(p, s, flags=re.I) for p in negative_caveat_patterns):
        return True, "positive tweet contains an opposite-direction or uncertainty concession"
    if r_i < 0 and any(re.search(p, s, flags=re.I) for p in positive_caveat_patterns):
        return True, "negative tweet contains an opposite-direction or uncertainty concession"
    return False, ""



def _step2_self_undermining_soft_only(tweet_text: str, expected_current_belief: int, reason_text: str = "") -> bool:
    """Keep borderline mixed Step-2 tweets as warnings instead of hard failures.

    Step-2 should be stricter than Step-3, but not absolutist. If the tweet still clearly lands
    on the expected side and the concession is mild / local, keep it as a warning-only case.
    """
    if not tweet_text:
        return False
    try:
        r_i = int(expected_current_belief)
    except Exception:
        return False
    if r_i == 0:
        return False
    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return False
    if not re.search(r"\b(?:but|however|though|although|still|yet)\b", s, flags=re.I):
        return False

    opener = re.split(r"\b(?:but|however|though|although|still|yet)\b", s, maxsplit=1, flags=re.I)[0].strip()
    tail = re.split(r"\b(?:but|however|though|although|still|yet)\b", s, maxsplit=1, flags=re.I)
    tail = tail[1].strip() if len(tail) > 1 else ""

    mild_tail = bool(re.search(r"\b(?:some questions remain|details remain unresolved|not fully settled|still leaves room|still raises questions|one gap|one unresolved point|not everything is explained)\b", tail, flags=re.I))
    if r_i > 0:
        strong_main = bool(re.search(r"\b(?:i think|i believe|i lean|probably|likely|makes sense|sounds plausible|evidence|records?|samples?|retroreflectors?|documented)\b", opener, flags=re.I))
        return bool(strong_main and mild_tail)
    if r_i < 0:
        strong_main = bool(re.search(r"\b(?:i doubt|i don't buy|i do not buy|skeptical|unconvinced|doesn't add up|does not add up|verification gap|independent verification|cover-up|hard to believe)\b", opener, flags=re.I))
        return bool(strong_main and mild_tail)
    return False


def _step2_speaker_intensity_mismatch(tweet_text: str, expected_current_belief: int) -> tuple[bool, str]:
    """Detect obvious speaker-side strength mismatches in Step-2 tweets.

    Goal:
      - CURRENT_RATING = 0 must not read like a strong one-sided endorsement/rejection.
      - CURRENT_RATING = 1 should not read like 2.
      - CURRENT_RATING = -1 should not read like -2.

    This is intentionally lightweight and only targets clear intensity failures.
    """
    if not tweet_text:
        return False, ""

    try:
        r_i = int(expected_current_belief)
    except Exception:
        return False, ""

    if r_i not in {-1, 0, 1}:
        return False, ""

    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return False, ""

    opener = re.split(r"(?<=[.!?])\s+", s, maxsplit=1)[0].strip() or s

    strong_pos_patterns = [
        r"\bi(?: am|[' ]m)? (?:absolutely|completely|totally|fully) convinced\b",
        r"\bno doubt\b",
        r"\bwithout (?:any )?doubt\b",
        r"\bdefinitely\b",
        r"\bcertain(?:ly)?\b",
        r"\bundeniabl(?:e|y)\b",
        r"\bthis proves?\b",
        r"\bit(?:'s| is) proof\b",
        r"\bproof that\b",
        r"\bclearly proves?\b",
        r"\bsettled fact\b",
        r"\boverwhelming(?:ly)?\b",
    ]
    strong_neg_patterns = [
        r"\bimpossible\b",
        r"\bcomplete hoax\b",
        r"\bdefinitely false\b",
        r"\bclearly fake\b",
        r"\bproves? (?:it(?:'s| is)? fake|the claim is false)\b",
        r"\bno chance\b",
        r"\bundeniabl(?:e|y) fake\b",
        r"\btotal fraud\b",
        r"\btotal fabrication\b",
        r"\babsolute(?:ly)? nonsense\b",
    ]

    strong_pos_hits = sum(1 for p in strong_pos_patterns if re.search(p, opener, flags=re.I))
    strong_neg_hits = sum(1 for p in strong_neg_patterns if re.search(p, opener, flags=re.I))
    has_strong_pos = strong_pos_hits > 0
    has_strong_neg = strong_neg_hits > 0
    mild_hedge = bool(re.search(r"\b(i think|i lean|it seems|probably|maybe|from what i can tell|i'm leaning|i am leaning)\b", s, flags=re.I))

    if r_i == 0 and (has_strong_pos or has_strong_neg):
        has_balance_connector = bool(re.search(r"\b(?:but|however|though|although|yet|still)\b", s, flags=re.I))
        counter_pull_patterns = [
            r"\bquestions persist\b",
            r"\bstill questions\b",
            r"\bstill doubts?\b",
            r"\blingering doubts?\b",
            r"\bdoubts? about\b",
            r"\bnot fully settled\b",
            r"\bnot entirely settled\b",
            r"\bnot everyone agrees\b",
            r"\bunclear\b",
            r"\bnot conclusive\b",
            r"\bopen questions?\b",
            r"\bunresolved\b",
            r"\bauthenticit\w*\b",
            r"\borigin\b",
            r"\bfootage\b",
            r"\black of independent verification\b",
            r"\bindependent verification\b",
            r"\bsamples?' origin\b",
        ] if has_strong_pos else [
            r"\bsome evidence\b",
            r"\bthere is evidence\b",
            r"\bthere are records\b",
            r"\bdocumented\b",
            r"\bmission footage\b",
            r"\blunar rocks\b",
            r"\bstill some support\b",
            r"\bphotos?\b",
            r"\bvideos?\b",
            r"\bsamples?\b",
        ]
        has_counter_pull = any(re.search(p, s, flags=re.I) for p in counter_pull_patterns)
        strong_hits = strong_pos_hits if has_strong_pos else strong_neg_hits
        if has_balance_connector and has_counter_pull and strong_hits <= 1:
            return False, ""
        if not (has_balance_connector and has_counter_pull):
            return True, "neutral tweet uses strong one-sided certainty language"
    if r_i == 0:
        dominant_pos = bool(re.search(r"(?i)\b(overwhelming(?:ly)?|undeniabl(?:e|y)|proves?|settled fact|clearly proves?)\b", opener))
        dominant_neg = bool(re.search(r"(?i)\b(complete hoax|definitely false|clearly fake|total fraud|total fabrication)\b", opener))
        weak_other_side = bool(re.search(r"(?i)\b(some questions remain|questions remain|details remain unresolved|not fully settled|can't confirm)\b", s))
        concrete_counter = bool(re.search(r"(?i)\b(doubts? about|lingering doubts?|authenticit\w*|origin|footage|independent verification|lack of independent verification|unresolved details?|one exact detail)\b", s))
        if (dominant_pos or dominant_neg) and weak_other_side and not concrete_counter:
            return True, "neutral tweet lets one side dominate with strong certainty language"
    if r_i == 1 and has_strong_pos and (strong_pos_hits >= 2 or not mild_hedge):
        return True, "mild positive tweet sounds as strong as +2"
    if r_i == -1 and has_strong_neg and (strong_neg_hits >= 2 or not mild_hedge):
        return True, "mild negative tweet sounds as strong as -2"
    return False, ""

def _step2_intensity_mismatch_soft_only(tweet_text: str, expected_current_belief: int, reason_text: str = "") -> bool:
    """Keep some intensity mismatches as warnings instead of hard fallback."""
    if not tweet_text:
        return False
    try:
        r_i = int(expected_current_belief)
    except Exception:
        return False
    s = re.sub(r"\s+", " ", str(tweet_text)).strip().lower()
    if not s:
        return False
    if r_i == 0:
        has_balance_connector = bool(re.search(r"\b(?:but|however|though|although|yet|still)\b", s, flags=re.I))
        explicit_uncertainty = bool(re.search(r"\b(uncertain|unsure|mixed|questions remain|still raise questions|keeps? me uncertain|keeps? me unsure|not fully settled|open questions?|torn)\b", s, flags=re.I))
        strong_hits = sum(bool(re.search(p, s, flags=re.I)) for p in [
            r"\bundeniabl(?:e|y)\b",
            r"\boverwhelming(?:ly)?\b",
            r"\bproves?\b",
            r"\bsettled\b",
            r"\bclearly\b",
            r"\bobvious(?:ly)?\b",
        ])
        concrete_counter = bool(re.search(r"(?i)\b(doubts? about|lingering doubts?|authenticit\w*|origin|footage|independent verification|lack of independent verification|unresolved details?|one exact detail|samples?' origin)\b", s))
        return bool(has_balance_connector and explicit_uncertainty and strong_hits <= 1 and concrete_counter)
    if r_i in {1, -1}:
        mild_hedge = bool(re.search(r"\b(i think|i lean|it seems|probably|maybe|from what i can tell|i'm leaning|i am leaning)\b", s, flags=re.I))
        strong_hits = sum(bool(re.search(p, s, flags=re.I)) for p in [
            r"\bundeniabl(?:e|y)\b",
            r"\boverwhelming(?:ly)?\b",
            r"\bdefinitely\b",
            r"\bwithout (?:any )?doubt\b",
            r"\bno doubt\b",
        ])
        return bool(mild_hedge and strong_hits <= 1)
    return False

def _extract_claim_from_anchor(text: str) -> str:
    """Best-effort extraction of the claim string from an anchor sentence.

    Supports both:
      - I ... claim as written: '{CLAIM}'.
      - I ... claim as written: {CLAIM}.
    """
    if not text:
        return ""
    s = str(text).strip()

    # Quoted form
    m = re.search(r"(?i)claim as written\s*:\s*['\"]([^'\"]+)['\"]", s)
    if m:
        return (m.group(1) or "").strip()

    # Unquoted form: capture after colon up to sentence-ending punctuation
    m = re.search(r"(?i)claim as written\s*:\s*(.+?)\s*[.!?]\s*$", s)
    if m:
        return (m.group(1) or "").strip()

    return ""





def _true_open_should_search(tweet_text: str) -> bool:
    """
    Decide whether Step-3 should perform a live search in true_open mode.
    
        The decision balances the active world mode, configured search budget, tweet content, and whether
        the interaction contains enough claim-relevant material to justify a web lookup.
    """
    if not TRUE_OPEN_WEB_ACTIVE:
        return False
    s = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if len(s) < 4:
        return False
    if PLANNER_MODE != "heuristic":
        return True
    s_low = s.lower()
    topic_profile = _true_open_topic_profile(_current_claim_text(), s)
    # In true_open Step-3, default to searching for almost any non-trivial claim-facing tweet.
    # Skip only obviously contentless / conversational fragments.
    if topic_profile == "moon_landing":
        if re.fullmatch(r"(?i)(yes|no|maybe|idk|i don't know|not sure|hmm|ok|okay|sure|nah)[.!?]*", s_low):
            return False
        return True
    factual_cues = (
        "http", "www.", "study", "report", "archive", "records", "data", "evidence", "proof",
        "document", "documents", "history", "historical", "because", "since", "according", "sample",
        "verification", "verify", "photo", "video", "tracked", "mission", "missions"
    )
    words = re.findall(r"[A-Za-z']+", s)
    return len(words) >= 3 or any(tok in s_low for tok in factual_cues)


def _true_open_query_hint_phrases(tweet_txt: str, claim_txt: str = "") -> list[str]:
    """Return search-query hints based on claim and tweet focus."""
    low = re.sub(r"\s+", " ", str(tweet_txt or "")).strip().lower()
    hints = []
    patterns = [
        (r"moon rocks?|lunar samples?|apollo samples?", "apollo moon rocks lunar samples evidence"),
        (r"photos?|images?|footage|video", "apollo moon landing photos footage evidence"),
        (r"retroreflectors?|laser", "apollo retroreflectors laser ranging evidence"),
        (r"on-site verification|direct verification|independent verification|verify the evidence", "apollo landing site images lunar reconnaissance orbiter independent tracking evidence"),
        (r"other nations|ground stations|tracking stations|telemetry", "apollo independent tracking other nations ground stations telemetry"),
        (r"records?|archives?|transcripts?|recordings?|transparency", "apollo mission records transcripts archives evidence"),
        (r"no credible evidence|unprovable|no real evidence|hoax|fake", "moon landing hoax claim evidence debunk"),
        (r"returned to verify|why no one returned|why haven'?t .*returned", "apollo evidence return to the moon verification"),
        (r"witness|global witness|transcripts?|audio recordings?", "apollo mission transcripts recordings witness evidence"),
        (r"flag|waving|wind", "moon landing flag waving without wind explanation"),
        (r"no stars?|stars? .*not visible", "moon landing photos no stars camera exposure explanation"),
        (r"radiation|van allen", "apollo van allen belts moon mission explanation"),
    ]
    for pat, hint in patterns:
        if re.search(pat, low):
            hints.append(hint)
    if not hints and _true_open_topic_profile(claim_txt, tweet_txt) == "moon_landing":
        if re.search(r"evidence|proof|sample|rock|photo|image|footage|verify|verification|flag|radiation|records|archives|tracking|transparency", low):
            hints.append("apollo moon landing evidence")
    out = []
    seen = set()
    for h in hints:
        h_norm = h.strip().lower()
        if h_norm and h_norm not in seen:
            out.append(h.strip())
            seen.add(h_norm)
    return out[:3]


def _true_open_exact_query_focus(tweet_txt: str) -> str:
    """Build exact query focus text for true-open retrieval."""
    low = re.sub(r"\s+", " ", str(tweet_txt or "")).strip().lower()
    exact_patterns = [
        (r"independent on-site verification|independent onsite verification", "independent on-site verification"),
        (r"direct on-site verification|direct onsite verification", "direct on-site verification"),
        (r"independent verification from other nations|other nations", "independent verification from other nations"),
        (r"moon rocks?|lunar samples?", "moon rocks lunar samples"),
        (r"retroreflectors?|laser ranging", "retroreflectors laser ranging"),
        (r"flag waving|flag .*wind|waving without wind", "flag waving without wind"),
        (r"no stars?|stars? .*not visible", "no stars in moon photos"),
        (r"records?|archives?|transcripts?|recordings?|transparency", "NASA records transparency"),
        (r"footage|video", "moon landing footage"),
        (r"radiation|van allen", "van allen belts"),
        (r"no credible evidence|no real evidence", "no credible evidence"),
        (r"hoax|fake", "moon landing hoax"),
    ]
    for pat, phrase in exact_patterns:
        if re.search(pat, low):
            return phrase
    return ""


def _build_true_open_query(claim_txt: str, tweet_txt: str) -> str:
    """Build the Step-3 live-search query for true_open mode."""
    claim = re.sub(r"\s+", " ", str(claim_txt or "")).strip()
    tweet = re.sub(r"\s+", " ", str(tweet_txt or "")).strip()
    tweet = re.sub(r"(?i)^twee?t:\s*", "", tweet).strip()
    topic_profile = _true_open_topic_profile(claim, tweet)
    hinge_terms = list(_true_open_hinge_tokens(tweet, claim))[:8]
    hint_phrases = _true_open_query_hint_phrases(tweet, claim)
    exact_focus = _true_open_exact_query_focus(tweet)
    tweet_words = re.findall(r"[A-Za-z0-9'_-]+", tweet)

    pieces = []
    if claim:
        pieces.append(claim)
    if exact_focus:
        pieces.append(f'"{exact_focus}"')
    if hint_phrases:
        pieces.extend(hint_phrases[:2])

    if topic_profile == 'generic':
        # Preserve named entities and concrete tweet phrasing for non-moon topics.
        namedish = []
        for tok in tweet_words:
            clean = str(tok or '').strip("'_- ")
            if not clean:
                continue
            low = clean.lower()
            if low in _true_open_stopwords():
                continue
            if len(clean) >= 4 or any(ch.isdigit() for ch in clean):
                namedish.append(clean)
        if namedish:
            pieces.append(" ".join(namedish[:10]))
        compact_tweet = tweet[:150].strip()
        if compact_tweet:
            pieces.append(compact_tweet)
        if not namedish and hinge_terms:
            pieces.append(" ".join(hinge_terms[:6]))
        if not claim or len(claim) < 18:
            pieces.append("history political context consequences evidence")
    else:
        if hinge_terms:
            pieces.append(" ".join(hinge_terms[:4]))
        elif tweet_words:
            pieces.append(" ".join(tweet_words[:8]))

    q = " ".join(p for p in pieces if p).strip()
    q = re.sub(r"\s+", " ", q).strip()
    return q[:320]


def _true_open_should_search_step2(seed_text: str, current_belief: int | None = None) -> bool:
    """Decide whether Step-2 should perform live search in true_open mode."""
    if not TRUE_OPEN_WEB_ACTIVE:
        return False
    mode = str(STEP2_WEB_MODE or "off").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True
    s = re.sub(r"\s+", " ", str(seed_text or "")).strip()
    if len(s) >= 12:
        return True
    try:
        return abs(int(current_belief)) >= 1
    except Exception:
        return False


def _build_true_open_step2_seed_text(agent, previous_interaction_type: str = "") -> str:
    """Build the Step-2 true-open search seed from the claim plus current belief only.

    We intentionally do not seed Step-2 web search from the agent's previous seen/written
    tweets. That keeps Step-2 retrieval aligned to the current stance rather than inheriting
    whichever earlier tweet happened to appear most recently.
    """
    claim = _current_claim_text()
    try:
        belief = int(getattr(agent, "current_belief", 0))
    except Exception:
        belief = 0
    if belief >= 2:
        stance = "strong supportive evidence physical traces independent confirmation"
    elif belief == 1:
        stance = "supportive evidence concrete supporting details"
    elif belief == -1:
        stance = "skeptical objections unresolved details hoax questions"
    elif belief <= -2:
        stance = "strong skeptical objections hoax questions contradictions"
    else:
        stance = "mixed evidence unresolved questions both sides"
    return f"{claim} {stance}".strip()


def _build_true_open_query_step2(claim_txt: str, seed_text: str, current_belief: int | None = None) -> str:
    """Build the Step-2 live-search query for true_open mode."""
    try:
        belief = int(current_belief)
    except Exception:
        belief = 0
    if belief > 0:
        stance_hint = "supportive evidence"
    elif belief < 0:
        stance_hint = "skeptical objections hoax questions"
    else:
        stance_hint = "mixed evidence unresolved questions"
    base = re.sub(r"\s+", " ", str(seed_text or "")).strip()
    shaped = f"{base} {stance_hint}".strip()
    return _build_true_open_query(claim_txt, shaped)


def _strip_search_html_text(value: str) -> str:
    """Strip tags and normalize text extracted from search HTML."""
    s = re.sub(r"<[^>]+>", " ", str(value or ""))
    s = html_lib.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_search_result_url(href: str, *, host_hint: str = "") -> str:
    """Normalize search-result URLs before deduplication and logging."""
    href = re.sub(r"\s+", "", str(href or "")).strip()
    if not href:
        return ""
    href = html_lib.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/"):
        base = host_hint or ""
        href = (base.rstrip("/") + href) if base else href
    try:
        parsed = urllib.parse.urlparse(href)
        q = urllib.parse.parse_qs(parsed.query or "")
        for key in ("uddg", "rut", "u", "url", "target"):
            vals = q.get(key) or []
            if vals:
                cand = html_lib.unescape(str(vals[0] or "")).strip()
                if cand.startswith("http://") or cand.startswith("https://"):
                    return cand
    except Exception:
        pass
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return ""


def _dedupe_search_items(items: list[dict], fetch_k: int) -> list[dict]:
    """Remove duplicate search results while preserving order."""
    out = []
    seen = set()
    for item in items:
        title = re.sub(r"\s+", " ", str(item.get("title", "") or "")).strip()
        snippet = re.sub(r"\s+", " ", str(item.get("snippet", "") or "")).strip()
        href = _normalize_search_result_url(item.get("url", ""))
        if not title or not href:
            continue
        try:
            parsed = urllib.parse.urlparse(href)
            host = (parsed.netloc or "").lower()
        except Exception:
            host = ""
        if not host or host.endswith("brave.com") or host.endswith("duckduckgo.com"):
            continue
        key = (title.lower(), href)
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": title[:160], "snippet": snippet[:320], "url": href[:500]})
        if len(out) >= fetch_k:
            break
    return out


def _live_web_search_duckduckgo(query: str, top_k: int = 3) -> list[dict]:
    """Run a DuckDuckGo HTML search for true_open mode."""
    q = str(query or "").strip()
    if not q:
        return []
    fetch_k = max(10, int(top_k or 3) * 5)
    try:
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    raw = html_lib.unescape(raw)
    items = []
    blocks = re.findall(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*class="result__a"|$)', raw, flags=re.I|re.S)
    for href, title_html, tail in blocks:
        title = _strip_search_html_text(title_html)
        snippet_match = re.search(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>', tail, flags=re.I|re.S)
        snippet_html = (snippet_match.group(1) or snippet_match.group(2) or "") if snippet_match else ""
        snippet = _strip_search_html_text(snippet_html)
        items.append({"title": title, "snippet": snippet, "url": _normalize_search_result_url(href, host_hint="https://html.duckduckgo.com")})
        if len(items) >= fetch_k:
            break
    return _dedupe_search_items(items, fetch_k)


def _live_web_search_brave(query: str, top_k: int = 3) -> list[dict]:
    """Run a Brave search for true_open mode when configured."""
    q = str(query or "").strip()
    if not q:
        return []
    fetch_k = max(10, int(top_k or 3) * 5)
    try:
        url = "https://search.brave.com/search?q=" + urllib.parse.quote(q) + "&source=web"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    raw = html_lib.unescape(raw)
    items = []
    patterns = [
        r'<a[^>]*href="([^"]+)"[^>]*class="[^"]*(?:snippet-title|heading|result-header|result-title|title)[^"]*"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*href=|$)',
        r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>(.*?)(?=<h2[^>]*>|$)',
    ]
    for pat in patterns:
        for href, title_html, tail in re.findall(pat, raw, flags=re.I | re.S):
            title = _strip_search_html_text(title_html)
            if not title:
                continue
            snippet_match = re.search(r'<(?:div|p)[^>]*class="[^"]*(?:snippet|description|excerpt)[^"]*"[^>]*>(.*?)</(?:div|p)>', tail, flags=re.I | re.S)
            snippet_html = (snippet_match.group(1) or "") if snippet_match else ""
            snippet = _strip_search_html_text(snippet_html)
            items.append({"title": title, "snippet": snippet, "url": _normalize_search_result_url(href, host_hint="https://search.brave.com")})
            if len(items) >= fetch_k:
                break
        if items:
            break

    if not items:
        for href, title_html, tail in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]*href=|$)', raw, flags=re.I | re.S):
            href_norm = _normalize_search_result_url(href, host_hint="https://search.brave.com")
            if not href_norm:
                continue
            title = _strip_search_html_text(title_html)
            if len(title) < 15:
                continue
            snippet_match = re.search(r'<(?:div|p)[^>]*class="[^"]*(?:snippet|description|excerpt)[^"]*"[^>]*>(.*?)</(?:div|p)>', tail[:1800], flags=re.I | re.S)
            snippet = _strip_search_html_text((snippet_match.group(1) if snippet_match else tail[:500]))
            items.append({"title": title, "snippet": snippet, "url": href_norm})
            if len(items) >= fetch_k:
                break

    return _dedupe_search_items(items, fetch_k)


def _true_open_compact_result_snippet(text: str, max_chars: int = 180) -> str:
    """Shorten a live-search result snippet for prompt insertion."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[: max_chars + 1]
    punct_idx = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "), cut.rfind(" - "), cut.rfind(" "))
    if punct_idx >= max(60, int(max_chars * 0.6)):
        s = cut[:punct_idx].strip(" ,;:-")
    else:
        s = cut[:max_chars].strip(" ,;:-")
    return s + "."


def _true_open_contextual_page_type_penalty(title: str, snippet: str, url: str, tweet_txt: str, focus_flags: dict, hinge_overlap: int) -> int:
    """Penalize live-search results whose page type is likely unhelpful."""
    title_low = str(title or "").lower()
    snippet_low = str(snippet or "").lower()
    blob_low = f"{title_low} {snippet_low}".strip()
    try:
        parsed = urllib.parse.urlparse(str(url or ""))
        host_low = (parsed.netloc or "").lower()
        path_low = (parsed.path or "").lower()
    except Exception:
        host_low, path_low = "", ""

    penalty = 0

    if re.search(r"\b(list|timeline|all[- ]time list|gallery|slideshow|photos?)\b", title_low):
        penalty -= 4
    if re.search(r"\b(interview|exclusive|what to know|here'?s what to know|tracker|live update|live updates|what do you think|complete overview|how many people have been to the moon|how many .*missions)\b", blob_low):
        penalty -= 6
    if re.search(r"\b(memorab(?:ilia|ilia)|souvenir|collectibles?)\b", blob_low):
        penalty -= 3
    if host_low.endswith('wikipedia.org'):
        penalty -= 1

    future_mission_pat = r"\b(artemis|future mission|launch|crewed lunar flyby|historic moon mission|splashes down|late 2028|returned to the moon|what's next|around the moon again|10-day journey around the moon|again for the first time in 50 years)\b"
    if re.search(future_mission_pat, blob_low):
        penalty -= 5
        if (focus_flags.get('evidence_focus') or focus_flags.get('artifact_focus') or focus_flags.get('footage_focus') or focus_flags.get('verification_focus') or focus_flags.get('records_focus')) and not focus_flags.get('return_focus'):
            penalty -= 6
    if hinge_overlap == 0:
        if focus_flags.get('footage_focus') and re.search(r"\b(gallery|slideshow|all[- ]time list|history of)\b", blob_low):
            penalty -= 3
        if (focus_flags.get('evidence_focus') or focus_flags.get('artifact_focus') or focus_flags.get('verification_focus') or focus_flags.get('records_focus')) and re.search(future_mission_pat, blob_low):
            penalty -= 7
        if re.search(r"/(gallery|photos|slideshow|list|timeline|live|updates?)(?:/|$)", path_low):
            penalty -= 3
    return penalty

def _render_notes_context(agent, max_items: int = 3, max_chars: int = 700) -> str:
    """Render accumulated true-open notes back into prompt context."""
    notes = list(getattr(agent, "true_open_notes", []) or [])[-max(1, int(max_items or 3)):]
    if not notes:
        return ""
    rows = []
    total = 0
    for idx, note in enumerate(notes, start=1):
        s = re.sub(r"\s+", " ", str(note or "")).strip()
        if not s:
            continue
        piece = f"- Note {idx}: {s}"
        if total + len(piece) > max_chars and rows:
            break
        rows.append(piece)
        total += len(piece)
    if not rows:
        return ""
    return "EXTERNAL NOTES (shown material)\n" + "\n".join(rows)


def _combine_shown_material_blocks(*blocks: str) -> str:
    """Combine RAG, fact-pack, web, and other shown material into one block."""
    parts = [str(b or "").strip() for b in blocks if str(b or "").strip()]
    return "\n\n".join(parts)


def _finalize_observable_source_meta(meta: dict) -> dict:
    """Finalize source metadata fields for a true-open interaction."""
    meta = dict(meta or {})
    used_web = int(bool(meta.get("used_web", 0)))
    used_notes = int(bool(meta.get("used_notes", 0)))
    used_rag = int(bool(meta.get("used_rag", 0)))
    used_fact_pack = int(bool(meta.get("used_fact_pack", 0)))
    used_memory_history = int(bool(meta.get("used_memory_history", 0)))

    meta["used_web"] = used_web
    meta["used_notes"] = used_notes
    meta["used_rag"] = used_rag
    meta["used_fact_pack"] = used_fact_pack
    meta["used_memory_history"] = used_memory_history
    meta["used_tweet_only"] = int(not any([used_web, used_notes, used_rag, used_fact_pack, used_memory_history]))

    primary = "tweet"
    secondary = ""
    if used_web:
        primary = "web"
        if used_notes:
            secondary = "notes"
        elif used_rag:
            secondary = "rag"
        elif used_fact_pack:
            secondary = "fact_pack"
        elif used_memory_history:
            secondary = "memory_history"
    elif used_notes:
        primary = "notes"
        if used_rag:
            secondary = "rag"
        elif used_fact_pack:
            secondary = "fact_pack"
        elif used_memory_history:
            secondary = "memory_history"
    elif used_rag:
        primary = "rag"
        if used_fact_pack:
            secondary = "fact_pack"
        elif used_memory_history:
            secondary = "memory_history"
    elif used_fact_pack:
        primary = "fact_pack"
        if used_memory_history:
            secondary = "memory_history"
    elif used_memory_history:
        primary = "memory_history"

    meta["observable_primary_source"] = primary
    meta["observable_secondary_source"] = secondary
    return meta


def _mark_prompt_source_meta(meta: dict, *, agent=None, conversation=None, step_kind: str = "step3", rag_context_text: str = "") -> dict:
    """Finalize source-use metadata for any prompt path, not only true-open tools.

    This fixes light-memory runs where trace/llm_history=auto exposed memory to the
    LLM but the step-summary CSV still logged used_memory_history=0 because the run
    was closed/RAG rather than true-open.
    """
    meta = dict(meta or {})
    if str(rag_context_text or "").strip():
        meta["used_rag"] = 1
    else:
        meta["used_rag"] = int(bool(meta.get("used_rag", 0)))
    meta["used_fact_pack"] = int(bool(str(FACT_PACK_TEXT or "").strip()))
    conv = conversation
    if conv is None and agent is not None:
        step_norm = str(step_kind or "step3").strip().lower()
        conv = getattr(agent, "memory_step2", None) if step_norm == "step2" else getattr(agent, "memory_step3", None)
    meta["used_memory_history"] = _conversation_has_visible_history(conv)
    return _finalize_observable_source_meta(meta)


def _build_true_open_context(agent, tweet_text: str, previous_interaction_type: str = "", step_kind: str = "step3", current_belief: int | None = None):
    """
    Build the complete live-web context block shown in true_open runs.
    
        This combines query construction, search execution, result reranking, snippet rendering, source
        metadata, and per-agent note updates. It is deliberately isolated from local RAG so closed/RAG runs
        remain reproducible without web access.
    """
    meta = {
        "tool_sequence": [],
        "planner_mode": PLANNER_MODE,
        "tool_mode": TOOL_MODE,
        "notes_mode": NOTES_MODE,
        "step_kind": str(step_kind or "step3"),
        "web_query": "",
        "web_results_count": 0,
        "web_result_titles": [],
        "web_context": "",
        "notes_context": "",
        "used_web": 0,
        "used_notes": 0,
        "used_rag": 0,
        "used_fact_pack": 0,
        "used_memory_history": 0,
        "used_tweet_only": 1,
        "observable_primary_source": "tweet",
        "observable_secondary_source": "",
    }
    step_norm = str(step_kind or "step3").strip().lower()
    if not TRUE_OPEN_WORLD_ENABLED:
        return "", _mark_prompt_source_meta(meta, agent=agent, step_kind=step_norm, rag_context_text="")


    if TOOL_MODE == "multi" and NOTES_MODE != "off" and step_norm == "step3":
        notes_context = _render_notes_context(agent, max_items=NOTES_MAX_ITEMS)
        if notes_context:
            meta["tool_sequence"].append("notes_read")
            meta["notes_context"] = notes_context
            meta["used_notes"] = 1

    should_search = _true_open_should_search(tweet_text) if step_norm == "step3" else _true_open_should_search_step2(tweet_text, current_belief=current_belief)
    if should_search:
        if step_norm == "step2":
            query = _build_true_open_query_step2(_current_claim_text(), tweet_text, current_belief=current_belief)
        else:
            query = _build_true_open_query(_current_claim_text(), tweet_text)
        meta["web_query"] = query
        if query:
            if WEB_BACKEND == "brave":
                raw_results = _live_web_search_brave(query, top_k=WEB_TOP_K)
            elif WEB_BACKEND == "duckduckgo":
                raw_results = _live_web_search_duckduckgo(query, top_k=WEB_TOP_K)
            else:
                raw_results = []
            show_k = max(1, min(3, int(WEB_TOP_K or 3))) if step_norm == "step3" else max(1, int(WEB_TOP_K or 3))
            results = _rerank_true_open_results(raw_results, _current_claim_text(), tweet_text, show_k=show_k, step_kind=step_norm) if raw_results else []
            if raw_results and not results:
                # Fallback: keep a small cleaned subset instead of collapsing to zero retained web context.
                cleaned_fallback = []
                seen_fb = set()
                for row in raw_results:
                    score, clean = _score_true_open_result(row, _current_claim_text(), tweet_text)
                    if not clean:
                        continue
                    key = (clean.get("title", "").strip().lower(), clean.get("url", "").strip().lower())
                    if key in seen_fb:
                        continue
                    seen_fb.add(key)
                    cleaned_fallback.append((score, clean))
                cleaned_fallback.sort(key=lambda x: (-int(x[0]), len(x[1].get("snippet", "")), x[1].get("title", "")))
                results = [row for _, row in cleaned_fallback[:show_k]]
        else:
            raw_results = []
            results = []
        meta["tool_sequence"].append("web_search")
        if results:
            meta["web_results_count"] = len(results)
            meta["web_result_titles"] = [str(r.get("title", "")) for r in results]
            max_chars = min(WEB_MAX_CHARS, 850 if step_norm == "step2" else 1300)
            meta["web_context"] = _render_live_web_context(results, max_chars=max_chars)
            meta["used_web"] = 1
        else:
            meta["web_results_count"] = 0
            max_chars = min(WEB_MAX_CHARS, 850 if step_norm == "step2" else 1300)
            meta["web_context"] = _render_live_web_context([], max_chars=max_chars, no_strong_results=True)
            meta["used_web"] = 0
        _append_web_event_row(
            agent_name=str(getattr(agent, 'agent_name', '') or ''),
            step_kind=step_norm,
            previous_interaction_type=previous_interaction_type,
            current_belief=(current_belief if current_belief is not None else getattr(agent, 'current_belief', None)),
            query=query,
            raw_results=raw_results,
            kept_results=results,
            meta=meta,
            tweet_text=tweet_text,
        )
    combined = _combine_shown_material_blocks(meta.get("notes_context", ""), meta.get("web_context", ""))

    meta = _mark_prompt_source_meta(meta, agent=agent, step_kind=step_norm, rag_context_text="")
    return combined, meta


def _update_true_open_notes(agent, meta: dict):
    """Update per-agent notes with observed true-open search results."""
    if agent is None or not TRUE_OPEN_NOTES_ACTIVE or TOOL_MODE != "multi":
        return
    notes = list(getattr(agent, "true_open_notes", []) or [])
    titles = list((meta or {}).get("web_result_titles", []) or [])
    query = str((meta or {}).get("web_query", "") or "").strip()
    if titles:
        note = f"Search on '{query}' surfaced: {titles[0]}" if query else f"Recent web result: {titles[0]}"
        if note not in notes:
            notes.append(note[:260])
    max_keep = max(3, int(NOTES_MAX_ITEMS or 3) * 4)
    agent.true_open_notes = notes[-max_keep:]


def _tweet_text_has_usable_step3_content(tweet_text: str) -> bool:
    """
    Return whether the listener received real tweet content rather than protocol residue.
    
        Missing or malformed Step-2 handoff is treated as a pipeline artifact. Step-3 should then produce a
        no-change fallback instead of asking the listener to evaluate an empty or meta prompt.
    """
    s = re.sub(r"\s+", " ", str(tweet_text or "")).strip()
    if not s:
        return False
    words = re.findall(r"[A-Za-z']+", s)
    if len(words) < 3:
        return False
    last = words[-1].lower() if words else ""
    dangling_tail = {"the", "a", "an", "and", "or", "but", "if", "that", "because", "about", "of", "to", "in", "on", "for", "with", "from", "this", "these", "those"}
    if last in dangling_tail and len(words) <= 8:
        return False
    if len(s) < 24 and not re.search(r"[.!?]$", s) and last not in {"fake", "real", "true", "false", "hoax"}:
        return False
    return True


def _compute_allowed_ratings(pre_belief: int, speaker_belief: int, max_step_change: int | None, update_mode: str | None = None):
    """
    Compute the legal final ratings for a listener before Step-3 is called.
    
        This is the main enforcement point for max_step_change, assimilation_only, and free_bounded update
        mechanics. Step-3 is instructed to choose only from this set, and validators later reject outputs
        outside it.
    """
    pre = int(pre_belief)
    spk = int(speaker_belief)
    mode = str(update_mode or globals().get("ALLOWED_UPDATE_MODE", "assimilation_only") or "assimilation_only").strip().lower()
    if mode == "free_brounded":
        mode = "free_bounded"
    try:
        k = int(max_step_change) if max_step_change is not None else 1
    except Exception:
        k = 1
    k = max(0, k)

    # neighborhood around pre
    neigh = []
    for d in range(-k, k + 1):
        c = max(-2, min(2, pre + d))
        if c not in neigh:
            neigh.append(c)

    if mode == "free_bounded":
        allowed = list(neigh)
    else:
        # not-away filter (non-increasing distance to speaker)
        allowed = [c for c in neigh if abs(spk - c) <= abs(spk - pre)]

    # Ensure pre is first (helps "default stay" under high temperature).
    if pre in allowed:
        allowed = [pre] + [x for x in allowed if x != pre]

    # Order the rest toward moderation (lower |x| first), then numeric.
    tail = allowed[1:]
    tail_sorted = sorted(dict.fromkeys(tail), key=lambda x: (abs(int(x)), int(x)))
    return [allowed[0]] + tail_sorted if allowed else [pre]

def _apply_same_side_edge_unlock(pre_belief: int, speaker_belief: int, allowed_ratings, agent=None, unlock_hits: int | None = None, tweet_text: str = ""):
    """
    Optionally expand the allowed set after repeated same-side mild exposures.
    
        This experimental mechanism can allow +1 to reach +2 or -1 to reach -2 after enough same-side hits.
        It is intentionally separate from the normal allowed-set computation so it can be enabled, disabled,
        or audited independently.
    """
    try:
        pre = int(pre_belief)
        spk = int(speaker_belief)
        vals = [int(x) for x in (allowed_ratings or [])]
    except Exception:
        return list(allowed_ratings or [])

    if _is_free_bounded_update_mode():
        return vals

    try:
        needed = int(SAME_SIDE_EDGE_UNLOCK_HITS if unlock_hits is None else unlock_hits)
    except Exception:
        needed = 0
    if needed <= 0 or agent is None:
        return vals

    if pre == 1 and spk == 1:
        hits = int(getattr(agent, 'same_side_pos_edge_hits', 0) or 0)
        if hits >= needed and 2 not in vals:
            vals = list(vals) + [2]
            _metric_inc('same_side_edge_unlock_positive_available')
    elif pre == -1 and spk == -1:
        hits = int(getattr(agent, 'same_side_neg_edge_hits', 0) or 0)
        if hits >= needed and -2 not in vals:
            vals = list(vals) + [-2]
            _metric_inc('same_side_edge_unlock_negative_available')

    # keep pre first, then moderation, then numeric for the tail
    if pre in vals:
        vals = [pre] + [x for x in vals if x != pre]
    tail = vals[1:]
    tail_sorted = sorted(dict.fromkeys(tail), key=lambda x: (abs(int(x)), int(x)))
    return [vals[0]] + tail_sorted if vals else [pre]


def _update_same_side_edge_unlock_counters(agent, pre_belief: int, post_belief: int, speaker_belief: int):
    """
    Update the counters used by the optional same-side edge-unlock mechanism.
    
        Counters track repeated same-side interactions that did not already move the agent. If this feature
        is enabled, these counters determine when the allowed set may reopen the corresponding edge rating.
    """
    if _is_free_bounded_update_mode():
        return
    if agent is None:
        return
    try:
        pre = int(pre_belief)
        post = int(post_belief)
        spk = int(speaker_belief)
    except Exception:
        return

    if post != pre:
        agent.same_side_pos_edge_hits = 0
        agent.same_side_neg_edge_hits = 0
        return

    if pre == 1 and spk == 1:
        agent.same_side_pos_edge_hits = int(getattr(agent, 'same_side_pos_edge_hits', 0) or 0) + 1
        _metric_inc('same_side_edge_unlock_positive_counter_inc')
    elif pre == -1 and spk == -1:
        agent.same_side_neg_edge_hits = int(getattr(agent, 'same_side_neg_edge_hits', 0) or 0) + 1
        _metric_inc('same_side_edge_unlock_negative_counter_inc')


def _choose_step3_singleton_mode(pre_belief: int, speaker_belief: int | None, allowed_ratings, same_rating_mode: str | None = None) -> str:
    """
    Decide how to handle Step-3 when only one final rating is legally allowed.
    
        Singleton allowed sets can be skipped, summarized with a deterministic explanation, or still sent to
        the LLM depending on the run configuration. The choice affects speed, artifact rate, and explanation
        richness but not the legal final rating.
    """
    try:
        allowed_list = sorted({int(x) for x in (allowed_ratings or [])})
    except Exception:
        allowed_list = []
    if len(allowed_list) != 1:
        return "llm"
    mode = str(same_rating_mode or SAME_RATING_STEP3_MODE or "skip_tweet_local").strip().lower()
    try:
        same_rating = speaker_belief is not None and int(pre_belief) == int(speaker_belief)
    except Exception:
        same_rating = False
    if same_rating:
        if mode == "llm":
            return "llm"
        if mode == "skip_generic":
            return "same_rating_skip_generic"
        return "same_rating_skip_tweet_local"
    return "singleton_skip_generic"


def _singleton_step3_explanation(pre_belief: int, final_rating: int, tweet_txt: str, tweet_stance: str | None, mode: str) -> tuple[str, str]:
    """Build a deterministic explanation when the allowed-rating set is a singleton."""
    mode = str(mode or "singleton_skip_generic").strip().lower()
    if mode == "same_rating_skip_tweet_local" and _tweet_text_has_usable_step3_content(tweet_txt):
        expl = _step3_tweet_local_explanation(tweet_txt, int(final_rating), int(pre_belief), tweet_stance)
        expl = _step3_force_rating_aligned_explanation(expl, int(final_rating))
        if expl:
            return expl, "local_tweet"
    return "This tweet does not give me a concrete reason to change my view.", "local_generic"


def _build_singleton_step3_output(pre_belief: int, speaker_belief: int | None, allowed_ratings, tweet_txt: str, tweet_stance: str | None = None, same_rating_mode: str | None = None):
    """
    Build a protocol-safe Step-3 response when the allowed set has one rating.
    
        This avoids unnecessary model calls in cases where the movement rule already fixes the listener's
        final rating, while still producing an explanation suitable for logs and downstream parsing.
    """
    try:
        allowed_list = sorted({int(x) for x in (allowed_ratings or [])})
    except Exception:
        allowed_list = []
    mode = _choose_step3_singleton_mode(pre_belief, speaker_belief, allowed_list, same_rating_mode=same_rating_mode)
    if mode == "llm" or len(allowed_list) != 1:
        return None
    forced = int(allowed_list[0])
    expl, source = _singleton_step3_explanation(int(pre_belief), forced, tweet_txt, tweet_stance, mode)
    final_text = _rewrite_step3_output_with_explanation(final_rating=forced, explanation=expl)
    tags = [
        mode,
        f"step3_mode:{mode}",
        "step3_skip_reason:singleton_after_unlock_processing",
        f"step3_explanation_source:{source}",
    ]
    return final_text, tags


def _format_allowed_ratings(pre_belief: int, allowed_ratings) -> str:
    """Format allowed ratings for Llama-style robustness.

    Recommended format: a single-line Python-like list of ints, e.g. [1, 0] or [0, -1, 1].
    """
    if allowed_ratings is None:
        return "[]"
    try:
        vals = [int(x) for x in allowed_ratings]
    except Exception:
        return "[]"
    return "[" + ", ".join(str(v) for v in vals) + "]"





def _step3_anchor_sentences(text: str) -> list[str]:
    """Return canonical anchor sentences for each final rating."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return []
    parts = [p.strip(" \n\t") for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    return parts

def _step3_anchor_clean_text(expl_text: str, final_rating: int | None = None) -> str:
    """Clean an anchor sentence before composing fallback explanations."""
    s = re.sub(r"\s+", " ", str(expl_text or "")).strip()
    if not s:
        return ""
    repls = [
        (r"(?i)\bwhile my (?:current|existing|prior) (?:view|belief|stance) remains [^,]+,\s*", ""),
        (r"(?i)\bmy (?:current|existing|prior) (?:view|belief|stance) remains unchanged because\b", "The tweet is not strong enough to justify a bigger move because"),
        (r"(?i)\bmy (?:current|existing|prior) (?:view|belief|stance) remains unchanged\b", "The tweet is not strong enough to justify a bigger move"),
        (r"(?i)\bmy (?:prior|existing) skepticism remains unchanged because\b", "The tweet is not strong enough to justify a bigger move because"),
        (r"(?i)\bmy (?:prior|existing) skepticism remains unchanged\b", "The tweet is not strong enough to justify a bigger move"),
        (r"(?i)\bmy (?:current|existing|prior) skepticism\b", "the remaining doubt after this tweet"),
        (r"(?i)\bmy institutional trust\b", "the tweet's pull"),
        (r"(?i)\bmy openness to update\b", "the tweet's force"),
        (r"(?i)\bmy persona\b", "the tweet's pull"),
        (r"(?i)\bthe rating stays the same because\b", "The tweet is not strong enough to justify a bigger move because"),
        (r"(?i)\bthe rating stays the same\b", "The tweet is not strong enough to justify a bigger move"),
        (r"(?i)\bI remain unchanged because\b", "The tweet is not strong enough to justify a bigger move because"),
        (r"(?i)\bI remain unchanged\b", "The tweet is not strong enough to justify a bigger move"),
        (r"(?i)\bmy current rating\b", "the current balance"),
        (r"(?i)\bcurrent rating\b", "the current balance"),
        (r"(?i)\bmy current view\b", "the current balance"),
        (r"(?i)\bcurrent view\b", "the current balance"),
    ]
    for pat, rep in repls:
        if _persona_allows_hidden_state_phrase(s, _current_persona_validation_policy()) and any(tok in pat for tok in ["skepticism", "institutional trust", "openness to update", "my persona"]):
            continue
        s = re.sub(pat, rep, s)
    s = re.sub(r"(?i)^while\s+", "", s)
    s = _sentence_case_cleanup(s).strip(" ,.;:-")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    parts = _step3_anchor_sentences(s)
    if not parts:
        return ""
    if int(final_rating or 999) == 0 and len(parts) >= 2:
        first = parts[0].rstrip('.!?')
        second = parts[1].rstrip('.!?')
        out = f"{first}, but {second}."
    else:
        out = parts[0]
        if out and out[-1] not in ".!?":
            out += "."
    out = re.sub(r"\s+", " ", out).strip()
    return out

def _step3_generic_anchor_for_rating(final_rating: int, tweet_stance: str | None = None) -> str:
    """Return a generic rating-aligned anchor when no better local reason is available."""
    try:
        r = int(final_rating)
    except Exception:
        r = 0
    stance = str(tweet_stance or "").strip().lower()
    if r == 0:
        if stance == "support":
            return "The tweet gives one supportive point, but it is still not enough to settle the claim."
        if stance == "oppose":
            return "The tweet raises one concrete challenge, but it is still not enough to settle the claim."
        return "The tweet gives some force in one direction, but there is still not enough here to settle the claim."
    if r > 0:
        if stance == "oppose":
            return "The tweet raises a concern, but it is not specific or strong enough to outweigh the support for the claim here."
        return "The tweet gives a concrete supportive point that fits the claim."
    if r < 0:
        if stance == "support":
            return "The tweet gives a supportive point, but it is not specific or strong enough to remove the remaining doubt here."
        return "The tweet gives a concrete challenge that weighs against the claim here."
    return "The tweet is not specific enough to settle the claim."


def _fallback_step2(expected_current_belief: int, claim_txt: str) -> str:
    """
    Create a deterministic Step-2 tweet when the model cannot produce a valid one.
    
        The fallback is intentionally simple and stance-aligned. It preserves run continuity and makes the
        artifact visible through warning/repair metrics rather than letting malformed text enter Step-3.
    """
    r = int(expected_current_belief)
    claim_short = re.sub(r"\s+", " ", str(claim_txt or "the claim")).strip()
    claim_short = re.sub(r"(?i)^theory xyz that claims that\s*", "", claim_short).strip()
    claim_short = claim_short[:90].rstrip(' ,.;:') if claim_short else "the claim"

    if r >= 2:
        tweet = f"I strongly accept {claim_short}. One concrete part of the case still outweighs the challenge here."
    elif r == 1:
        tweet = f"I lean toward {claim_short}. The push against it here still feels too thin to overturn that."
    elif r == 0:
        tweet = f"I am mixed on {claim_short}. One point supports it, but one exact detail still feels unresolved."
    elif r == -1:
        tweet = f"I lean against {claim_short}. One key part still feels too unresolved for me to buy it."
    else:
        tweet = f"I strongly doubt {claim_short}. One central part of the story still does not hold together for me."

    tweet = re.sub(r"\s+", " ", str(tweet or "")).strip()
    tweet = _limit_to_n_sentences(tweet, max_sentences=2)
    return f"FINAL_RATING: {r}\nTWEET: {tweet}"

def _render_display_step3_explanation(explanation_text: str, final_rating: int | None = None) -> str:
    """Human-friendly rendering for logs/UI only."""
    s = re.sub(r"\s+", " ", str(explanation_text or "")).strip()
    if not s:
        return ""
    replacements = [
        (r"(?i)^the tweet gives a supportive point, but it does not clearly outweigh the strongest reason on the other side here\.?$", "It does add a supportive point, but it still does not outweigh the strongest reason on the other side here."),
        (r"(?i)^the tweet gives a challenging point, but it does not clearly outweigh the strongest reason on the other side here\.?$", "It raises a real challenge, but it still does not outweigh the strongest reason on the other side here."),
        (r"(?i)^the tweet is mixed or uncertain, but it does not clearly outweigh the strongest reason for the current rating\.?$", "It adds some uncertainty, but it still does not clearly outweigh the strongest reason for my current rating."),
        (r"(?i)^the tweet has a point, but it does not clearly outweigh the strongest reason for the current rating\.?$", "It has a point, but it still does not clearly outweigh the strongest reason for my current rating."),
        (r"(?i)^the tweet supports the claim, but its point does not clearly outweigh the strongest reason against it here\.?$", "It supports the claim, but it still does not outweigh the strongest reason against it here."),
        (r"(?i)^the tweet challenges the claim, but its point does not clearly outweigh the strongest reason for it here\.?$", "It challenges the claim, but it still does not outweigh the strongest reason for it here."),
        (r"(?i)^the tweet is not strong enough to justify a change because (.+)$", r"It still does not clearly outweigh the strongest reason for my current rating because \1"),
    ]
    for pat, repl in replacements:
        s = re.sub(pat, repl, s)
    if final_rating == 0 and not re.search(r"(?i)^i\b", s):
        if s.lower().startswith("it "):
            s = "I'm still mixed because " + s[3:].rstrip('.') + "."
    s = _sentence_case_cleanup(re.sub(r"\s+", " ", s).strip())
    if s and not re.search(r"[.!?]$", s):
        s += "."
    return s


def _render_display_step3_output(step3_text: str) -> str:
    """Render final Step-3 output for logs after display-level cleanup."""
    r = extract_belief(step3_text)
    expl = _extract_explanation_text(step3_text)
    if r is None or not expl:
        return ""
    disp = _render_display_step3_explanation(expl, int(r))
    if not disp:
        return ""
    return f"FINAL_RATING: {int(r)}\nEXPLANATION: {disp}"


def _fallback_step3(pre_belief: int, claim_txt: str, tweet_stance: str | None = None, tweet_txt: str = "", invalid_rating: int | None = None) -> str:
    """
    Create a deterministic no-crash Step-3 output when validation/rescue cannot recover.
    
        The fallback respects the allowed-rating set and produces a conservative explanation. It is logged
        as an artifact so analysis can judge whether the run remains interpretable.
    """
    r = int(pre_belief)
    if _tweet_text_has_usable_step3_content(tweet_txt):
        expl_one_line = ""
        if invalid_rating is not None:
            expl_one_line = _step3_illegal_rating_fallback_explanation(tweet_txt, r, invalid_rating, pre_belief=r, tweet_stance=tweet_stance)
        if not expl_one_line:
            expl_one_line = _step3_safe_fallback_explanation_for_rating(tweet_txt, r, r, tweet_stance)
        if not expl_one_line or _step3_explanation_wrong_side_anchor(expl_one_line, r, tweet_txt, pre_belief=r):
            expl_one_line = _step3_tweet_local_explanation(tweet_txt, r, r, tweet_stance)
        expl_one_line = _step3_force_rating_aligned_explanation(expl_one_line, r)
    else:
        expl_one_line = "No usable tweet content was available to evaluate, so I keep my rating unchanged."
    if not expl_one_line:
        expl_one_line = _step3_generic_anchor_for_rating(r, tweet_stance)
    return f"FINAL_RATING: {r}\nEXPLANATION: {expl_one_line}"

def _preserve_step3_output(final_rating: int, base_text: str, claim_txt: str) -> str:
    """Normalize Step-3 output to EXACTLY two lines while preserving a short anchor-like explanation."""
    r = int(final_rating)
    expl = _extract_explanation_text(base_text) or ""
    expl = _step3_force_rating_aligned_explanation(expl, r)
    if not expl:
        return _fallback_step3(pre_belief=r, claim_txt=claim_txt)
    return f"FINAL_RATING: {r}\nEXPLANATION: {expl}".strip()


def _rewrite_step3_output(final_rating: int, base_text: str, claim_txt: str) -> str:
    """Canonicalize Step-3 output to EXACTLY two lines while keeping the chosen rating."""
    r = int(final_rating)
    expl = _extract_explanation_text(base_text) or ""
    expl = _step3_force_rating_aligned_explanation(expl, r)
    if not expl:
        expl = _step3_generic_anchor_for_rating(r, None)
    return f"FINAL_RATING: {r}\nEXPLANATION: {expl}".strip()




# <<< CHANGED: added random_seed parameter and use it for shuffling
def initialize_opinion_distribution(
    num_agents: int,
    list_opinion_space: list,
    distribution_type: str = "uniform",
    random_seed: int | None = None,
):
    """
    Create the initial belief vector for the selected distribution mode.
    
        Uniform mode allocates agents evenly across the five belief bins when possible. Other modes support
        skewed, all-positive/all-negative, and custom count distributions used in hub and baseline tests.
    """
    max_opinion = max(list_opinion_space)
    min_opinion = min(list_opinion_space)
    multiple = num_agents // 5

    if distribution_type == "uniform":
        list_opinions = list_opinion_space * multiple
    elif distribution_type == "skewed_positive":
        list_opinions = [max_opinion] * (num_agents - multiple) + [min_opinion] * multiple
    elif distribution_type == "skewed_negative":
        list_opinions = [min_opinion] * (num_agents - multiple) + [max_opinion] * multiple
    elif distribution_type == "positive":
        list_opinions = [max_opinion] * num_agents
    elif distribution_type == "negative":
        list_opinions = [min_opinion] * num_agents
    else:
        raise NotImplementedError

    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(list_opinions)
    else:
        np.random.shuffle(list_opinions)

    return list_opinions
# <<< CHANGED END


def convert_text_from_present_to_past(text):
    """
    Lightweight rule-based present→past converter.
    Used because spaCy/pyinflect do not support Python 3.12.
    """
    irregular = {
        "am": "was",
        "are": "were",
        "is": "was",
        "have": "had",
        "has": "had",
        "do": "did",
        "does": "did",
        "go": "went",
        "say": "said",
        "make": "made",
        "feel": "felt",
        "think": "thought",
        "want": "wanted",
        "need": "needed",
        "love": "loved",
        "like": "liked",
        "work": "worked",
        "live": "lived",
    }

    words = text.split()
    new_words = []

    for w in words:
        lw = w.lower()

        if lw in irregular:
            replacement = irregular[lw]
            if w[0].isupper():
                replacement = replacement.capitalize()
            new_words.append(replacement)
            continue

        if lw.endswith("s") and len(lw) > 2:
            stem = lw[:-1]
            past = stem + "ed"
            if w[0].isupper():
                past = past.capitalize()
            new_words.append(past)
            continue

        if lw.endswith("e"):
            past = lw + "d"
        else:
            past = lw + "ed"

        if w[0].isupper():
            past = past.capitalize()

        new_words.append(past)

    return " ".join(new_words)


def main(
    num_agents,
    num_steps,
    experiment_id,
    model_name,
    temperature,
    max_tokens,
    opinion_space,
    prompt_template_root,
    date_version,
    path_result,
):
    """
    Run one complete opinion-dynamics experiment and export all result artifacts.
    
        The function resolves configuration, initializes prompts/RAG/fact packs/personas/network, builds
        agents, executes the speaker-listener interaction loop, updates beliefs, records metrics at each
        step, handles early stopping, and writes the CSV/log outputs consumed by analysis scripts.
    """
     # Base prompt root: prompts/opinion_dynamics/Flache_2017
    base_prompt_root = os.path.join(prompt_template_root, experiment_id)
    # Version-specific prompt directory, e.g. Flache_2017/v102/v102_confirmation_bias
    version_prefix = args.version_set.split("_")[0]
    prompt_template_root = os.path.join(base_prompt_root, version_prefix, args.version_set)

    # Initialize the opinion distribution for the agent
    if args.custom_counts:
        parts = [p.strip() for p in args.custom_counts.split(",")]
        if len(parts) != 5:
            raise ValueError("custom_counts must have 5 comma-separated integers for [-2,-1,0,1,2].")
        counts = list(map(int, parts))
        if sum(counts) != args.num_agents:
            raise ValueError(f"custom_counts sum to {sum(counts)} but agents is {args.num_agents}.")

        list_opinion = ([-2] * counts[0]) + ([-1] * counts[1]) + ([0] * counts[2]) + ([1] * counts[3]) + ([2] * counts[4])

        # shuffle deterministically like the existing code
        if args.seed is not None:
            rng = np.random.default_rng(args.seed)
            rng.shuffle(list_opinion)
        else:
            np.random.shuffle(list_opinion)

        print(f"Opinions generated (custom_counts): {list_opinion}")
    else:
        list_opinion = initialize_opinion_distribution(
            num_agents=args.num_agents,
            list_opinion_space=opinion_space,
            distribution_type=args.distribution,
            random_seed=args.seed
        )
        print(f"Opinions generated (-dist={args.distribution}): {list_opinion}")
        
    print(f"Running with seed = {args.seed}")
    print("Opinions generated:", list_opinion)

    # Reading in the list of agents and creating a dataframe of selected agents
    if args.distribution == "uniform":
        df_agents = pd.read_csv(join(prompt_template_root, "list_agent_descriptions.csv"))
    else:
        df_agents = pd.read_csv(
            join(prompt_template_root, "list_agent_descriptions.csv")
        )

    df_agents_selected = pd.DataFrame()

    # Initialize one agent per an opinion in list_opinion
    df_agents_selected = pd.DataFrame()
    used_ids = set()

    for opinion in list_opinion:
        candidates = df_agents[df_agents["opinion"] == opinion]
        if candidates.empty:
            raise ValueError(f"No agents in CSV with opinion {opinion}")

        while True:
            df_agent_candidate = candidates.sample(
                n=1, random_state=args.seed + choice(range(10_000))
            )
            agent_id = df_agent_candidate["agent_id"].iloc[0]
            if agent_id not in used_ids:
                used_ids.add(agent_id)
                df_agents_selected = pd.concat(
                    [df_agents_selected, df_agent_candidate], ignore_index=True
                )
                break

    print(df_agents_selected[["agent_id", "agent_name", "age", "opinion"]])

    list_agents = []
    list_agent_ids = df_agents_selected["agent_id"]
    if "persona" not in df_agents_selected.columns:
        raise ValueError("list_agent_descriptions.csv is missing required column: 'persona'")

    list_agent_persona = df_agents_selected["persona"]

    for persona, agent_id in zip(list_agent_persona, list_agent_ids):
        # Ensure persona is always a string and never NaN
        if pd.isna(persona):
            persona = ""
        else:
            persona = str(persona)

        agent = Agent(
            agent_id,
            persona,
            model_name,
            temperature,
            max_tokens,
            prompt_template_root,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            repeat_last_n=args.repeat_last_n,
            llm_seed=(args.llm_seed if args.llm_seed is not None else args.seed),
        )
        list_agents.append(agent)
        
    df_agents_selected = df_agents_selected.reset_index(drop=True)
    num_agents = len(list_agents)

    def to_age_group(age_val):
        """Map a raw age value to a coarse demographic group for summaries."""
        try:
            a = int(age_val)
        except Exception:
            return None
        if a < 18:
            return 0
        elif a <= 25:
            return 1
        elif a <= 35:
            return 2
        elif a <= 50:
            return 3
        else:
            return 4

    def map_political_to_score(text):
        """Map a political-leaning label to a numeric ordering for analysis."""
        if not isinstance(text, str):
            return 0
        s = text.strip().lower()
        if "far left" in s:
            return -3
        if "strong democrat" in s:
            return -2
        if "lean democrat" in s:
            return -1
        if "democrat" in s:
            return -1
        if "center-left" in s or "center left" in s:
            return -1
        if "center-right" in s or "center right" in s:
            return 1
        if "lean republican" in s:
            return 2
        if "strong republican" in s:
            return 3
        if "far right" in s:
            return 3
        if "independent" in s or "centrist" in s or "moderate" in s:
            return 0
        return 0

    def map_education_to_level(text):
        """Map an education label to a coarse ordinal level for analysis."""
        if not isinstance(text, str):
            return 2  # mid-point
        s = text.lower()
        if "high school" in s:
            return 0
        if "associate" in s or "some college" in s:
            return 1
        if "bachelor" in s:
            return 2
        if "master" in s:
            return 3
        if "phd" in s or "doctorate" in s:
            return 4
        return 2

    attributes = {}
    opinions_by_idx = {}

    for idx, row in df_agents_selected.iterrows():
        attributes[idx] = {
            "age_group": to_age_group(row.get("age")),
            "gender": str(row.get("gender", "")).strip().lower(),
            "ethnicity": str(row.get("ethnicity", "")).strip().lower(),
            "edu_level": map_education_to_level(row.get("education", "")),
            "occupation": str(row.get("occupation", "")).strip().lower(),
            "pol_score": map_political_to_score(row.get("political_leaning", "")),
        }
        opinions_by_idx[idx] = int(row["opinion"])

    # ---- NEW: build small-world network over local indices 0..num_agents-1
    # Network selection (CLI-controlled)
    net_type = str(getattr(args, "network_type", "ws")).strip().lower()
    if net_type not in {"ws", "er", "ba", "none"}:
        raise ValueError("network_type must be one of: 'ws', 'er', 'ba', 'none'.")

    ba_hub_strategy = str(getattr(args, "ba_hub_strategy", "default") or "default").strip().lower()
    ba_hub_assignment_mode = _normalize_ba_hub_assignment_mode(getattr(args, "ba_hub_assignment_mode", "early_position"))
    ba_hub_custom = str(getattr(args, "ba_hub_custom", "") or "").strip()
    ba_hub_priority_indices = []

    if net_type == "none":
        # Fully-mixed: no adjacency constraints; partner can be chosen randomly or via homophily (see --interaction_selection).
        neighbors = {}
    elif net_type == "ws":
        k_neighbors = int(getattr(args, "k_neighbors", 4))
        p_rewire = float(getattr(args, "p_rewire", 0.1))

        if (k_neighbors % 2) != 0:
            raise ValueError("k_neighbors must be even for a WS network.")
        if k_neighbors <= 0 or k_neighbors >= num_agents:
            raise ValueError("k_neighbors must be > 0 and < num_agents.")
        if not (0.0 <= p_rewire <= 1.0):
            raise ValueError("p_rewire must be between 0 and 1.")

        neighbors = build_small_world(
            num_agents=num_agents,
            k=k_neighbors,
            p_rewire=p_rewire,
            seed=args.seed,
            homophily=bool(getattr(args, "network_homophily", False)),
            opinions=opinions_by_idx,
            attributes=attributes,
        )
    elif net_type == "er":
        er_p_edge = float(getattr(args, "er_p_edge", 0.15))
        if not (0.0 <= er_p_edge <= 1.0):
            raise ValueError("er_p_edge must be between 0 and 1.")
        neighbors = build_erdos_renyi(
            num_agents=num_agents,
            p_edge=er_p_edge,
            seed=args.seed,
            homophily=bool(getattr(args, "network_homophily", False)),
            opinions=opinions_by_idx,
            attributes=attributes,
        )
    else:
        ba_m_attach = int(getattr(args, "ba_m_attach", 2))
        if ba_m_attach <= 0 or ba_m_attach >= num_agents:
            raise ValueError("ba_m_attach must be > 0 and < num_agents.")
        ba_hub_strategy = str(getattr(args, "ba_hub_strategy", "default") or "default").strip().lower()
        ba_hub_assignment_mode = _normalize_ba_hub_assignment_mode(getattr(args, "ba_hub_assignment_mode", "early_position"))
        ba_hub_custom = str(getattr(args, "ba_hub_custom", "") or "").strip()
        ba_hub_priority_indices = _build_ba_hub_priority_indices(
            ba_hub_strategy,
            ba_hub_custom,
            list_agents,
            opinions_by_idx,
            seed=args.seed,
        )
        neighbors = _build_barabasi_albert_compat(
            num_agents=num_agents,
            m_attach=ba_m_attach,
            seed=args.seed,
            homophily=bool(getattr(args, "network_homophily", False)),
            opinions=opinions_by_idx,
            attributes=attributes,
            ba_hub_strategy=ba_hub_strategy,
            ba_hub_assignment_mode=ba_hub_assignment_mode,
            ba_hub_priority_indices=ba_hub_priority_indices,
        )

    network_metrics_by_idx = _compute_network_metrics_by_idx(
        num_agents=num_agents,
        neighbors=neighbors,
        network_type=net_type,
        ba_hub_strategy=ba_hub_strategy,
        ba_hub_assignment_mode=ba_hub_assignment_mode,
        ba_hub_priority_indices=ba_hub_priority_indices,
        opinions_by_idx=opinions_by_idx,
    )
    agent_local_index_by_agent_id = {
        int(getattr(agent, 'agent_id')): int(idx)
        for idx, agent in enumerate(list_agents)
    }
    if net_type == "ba":
        _metric_inc(f"ba_hub_strategy::{ba_hub_strategy}")
        _metric_inc(f"ba_hub_assignment_mode::{ba_hub_assignment_mode}")
        _metric_inc('ba_hub_targeted_agents_total', len(ba_hub_priority_indices))
        _metric_inc('ba_hub_targeted_top_hub_count', sum(1 for i in ba_hub_priority_indices if int(network_metrics_by_idx.get(int(i), {}).get('network_top_hub_flag', 0))))
        _metric_inc('ba_hub_targeted_max_degree_count', sum(1 for i in ba_hub_priority_indices if int(network_metrics_by_idx.get(int(i), {}).get('network_max_degree_hub_flag', 0))))
    rng = random.Random(args.seed)

    dict_agent_tweet = defaultdict(list)
    dict_agent_response = defaultdict(list)
    dict_csv = dict()
    for agent in list_agents:
        dict_csv["time_step"] = [0]
        dict_csv[agent.agent_name] = [agent.init_belief]
        
    # New-scheme run folder: compact bias abbreviation, no date/distribution
    # suffix (see scripts/run_naming.py). Every artifact below derives its file
    # name from os.path.basename(csv_folder), so the folder name IS the stem.
    _run_stem = run_naming.run_stem(
        os.path.splitext(args.output_file)[0],
        args.num_agents,
        args.num_steps,
        args.version_set,
        strict=True,
    )
    csv_folder = os.path.join(path_result, _run_stem)

    out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_opinion_change.csv")
    )
    interaction_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_interactions.csv")
    )

    os.makedirs(csv_folder, exist_ok=True)
    # Retry debug dump directory for failed LLM-format attempts
    global RETRY_DEBUG_DIR
    global REPAIR_LOG_CSV_PATH
    global REPAIR_LOG_RUN_LABEL
    global RAG_RETRIEVAL_LOG_CSV_PATH
    global RUN_EXPORT_ID
    global WEB_EVENTS_LOG_CSV_PATH
    global NATIVE_EVENTS_LOG_CSV_PATH
    RETRY_DEBUG_DIR = os.path.join(csv_folder, "_retry_debug")
    # Run id for retry-debug logs, the metrics JSON, and per-row run labels:
    # the compact stem (folder name), so every artifact shares one id.
    RUN_EXPORT_ID = _safe_file_token(os.path.basename(csv_folder))
    globals()['RUN_OUTPUT_DIR'] = csv_folder
    REPAIR_LOG_RUN_LABEL = RUN_EXPORT_ID
    REPAIR_LOG_CSV_PATH = os.path.join(RETRY_DEBUG_DIR, f"repair_events_{REPAIR_LOG_RUN_LABEL}.csv")
    RAG_RETRIEVAL_LOG_CSV_PATH = None
    NATIVE_EVENTS_LOG_CSV_PATH = os.path.join(RETRY_DEBUG_DIR, f"native_events_{RUN_EXPORT_ID}.csv")
    WEB_EVENTS_LOG_CSV_PATH = os.path.join(csv_folder, f"web_events_{RUN_EXPORT_ID}.csv") if _web_logging_enabled() else None
    os.makedirs(RETRY_DEBUG_DIR, exist_ok=True)
    _ensure_repair_log_file()
    _ensure_native_events_log_file()
    metrics_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_run_metrics.csv")
    )
    step_summary_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_step_summary.csv")
    )
    step2_event_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_step2_events.csv")
    )

    agent_summary_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_agent_summary.csv")
    )
    network_hub_metrics_out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_hub_metrics.csv")
    )
    if DEBUG_RETRY_SAVE_FILES:
        os.makedirs(RETRY_DEBUG_DIR, exist_ok=True)


    os.makedirs(os.path.dirname(interaction_out_name), exist_ok=True)
    
    # Stream opinion-change CSV during the run (enables Live tab in ui_lancher.py)
    try:
        print(f"[INFO] Opinion-change CSV: {out_name}")
    except Exception:
        pass
    
    with open(interaction_out_name, "w", newline="", encoding="utf-8") as f, open(out_name, "w", newline="", encoding="utf-8") as f_op:
        writer = csv.writer(f, delimiter=",")
        op_writer = csv.writer(f_op, delimiter=",")
    
        # Header + initial snapshot (t=0)
        op_writer.writerow(["time_step"] + [a.agent_name for a in list_agents])
        op_writer.writerow([0] + [a.init_belief for a in list_agents])
        try:
            f_op.flush()
        except Exception:
            pass

        step_summary_rows = []
        step2_event_rows = []
        agent_stats = {
            int(a.agent_id): {
                'times_as_speaker': 0,
                'times_as_listener': 0,
                'num_moves_up': 0,
                'num_moves_down': 0,
                'num_stays': 0,
                'abs_total_delta': 0,
                'crossed_zero_count': 0,
                'repair_events': 0,
                'soft_cleanup_events': 0,
                'valid_listener_interactions': 0,
                'invalid_listener_interactions': 0,
            }
            for a in list_agents
        }

        interaction_header = [
            "Run ID",
            "Seed",
            "Version Set",
            "World",
            "Fact Pack Mode",
            "RAG Content Mode",
            "Model Name Step2",
            "Model Name Step3",
            "Time Step",
            "Agent_J Name",
            "Agent_J Agent ID",
            "Agent_J Belief",
            "Agent_J Tweet",
            "Agent_J Raw Step2 Attempt 1",
            "Agent_J Raw Step2 Attempt 2",
            "Agent_J Step2 Warning Tags",
            "Agent_J Step2 Fallback Used",
            "Agent_J Step2 Interaction Quality",
            "Agent_I Name",
            "Agent_I Agent ID",
            "Agent_I Pre-Belief",
            "Agent_I Allowed Ratings",
            "Agent_I Response",
            "Agent_I Post-Belief",
            "Agent_I Delta-Belief",
            "Agent_I Raw Response Attempt 1",
            "Agent_I Raw Response Attempt 2",
            "Agent_I Repair Tags",
            "Agent_I Soft Cleanup Tags",
            "Agent_I Hard Repair",
            "Agent_I Soft Cleanup",
            "Agent_I Interaction Quality",
        ]
        writer.writerow(interaction_header)

        mild_unanimity_value = None
        mild_unanimity_remaining = None
        early_stop_triggered = False
        early_stop_step = None
        early_stop_reason = ''

        # --- Coevolving network setup. dir_net stays None unless
        #     --network_evolves is set OR --p_reach_policy is non-default, so
        #     the default path is untouched. When on, neighbors is aliased to
        #     dir_net.following: rewiring the directed graph is then visible to
        #     speaker selection for free, with no other change to the loop.
        #     Fully-mixed runs (net_type == "none") do not use dir_net - there
        #     is no edge set to attach reach probabilities to. ---
        dir_net = None
        _p_reach_policy = str(getattr(args, "p_reach_policy", "uniform") or "uniform")
        _p_reach_uniform_value = float(getattr(args, "p_reach_uniform_value", 1.0))
        # ADR-006 Component 3: activate p_reach only when the user asks for it.
        # uniform + 1.0 is the byte-identical baseline; every edge draws 1.0
        # via DirectedNetwork.get_p_reach(), so no dir_net is needed.
        _p_reach_active = (_p_reach_policy != "uniform" or _p_reach_uniform_value != 1.0)
        if (getattr(args, "network_evolves", False) or _p_reach_active) and net_type != "none":
            dir_net = DirectedNetwork.from_undirected(
                neighbors, num_agents,
                reciprocal=not getattr(args, "evolve_directed", False),
            )
            neighbors = dir_net.following
            # Populate p_reach for every existing edge under the chosen policy.
            # Only runs when the user explicitly asks for a non-default policy;
            # rewires (evolve_once) do NOT re-populate here - the sim leaves the
            # rewire+policy interaction to a later ADR when we have data.
            if _p_reach_active:
                # ADR-006 roads_not_taken 3.11 (fix gamma). Enforcement mode decides
                # HOW a sub-1.0 reach acts (see argparse help). "suppress"
                # (skip-preserving) draws ALL p_reach randomness from a DEDICATED rng
                # so the main interaction stream stays byte-identical to a uniform
                # baseline (same listeners AND speakers) - the only difference is
                # whether a throttled tweet's belief update is applied. "filter"
                # (default) keeps the main rng for byte-identical pre-gamma behaviour.
                _p_reach_enforcement = str(getattr(args, "p_reach_enforcement", "filter")).strip().lower()
                _p_reach_rng = random.Random(int(getattr(args, "seed", 0) or 0) + 987654321)
                _p_reach_params = {
                    "uniform_value": _p_reach_uniform_value,
                    "homophily_k": float(getattr(args, "p_reach_homophily_k", 2.0)),
                    "shadowban_value": float(getattr(args, "p_reach_shadowban_value", 0.1)),
                }
                if _p_reach_policy == "shadowban":
                    _sb_fraction = float(getattr(args, "p_reach_shadowban_fraction", 0.1))
                    _n_banned = max(1, int(round(_sb_fraction * num_agents)))
                    # In suppress mode the shadowban draw must also avoid the main rng
                    # (otherwise the treatment's stream diverges from uniform).
                    _sb_rng = _p_reach_rng if _p_reach_enforcement == "suppress" else rng
                    _p_reach_params["shadowban_agents"] = set(
                        _sb_rng.sample(range(num_agents), min(_n_banned, num_agents))
                    )
                for _src in list(dir_net.following.keys()):
                    for _dst in dir_net.following[_src]:
                        dir_net.p_reach[(_src, _dst)] = assign_p_reach(
                            _p_reach_policy, _p_reach_params,
                            _src, _dst, opinions_by_idx,
                        )

        # --- Event injection setup. Enabled only when --event_step > 0
        #     and --event_text is non-empty; otherwise event_enabled stays False
        #     and not one line of the event path runs. ---
        _event_step = int(getattr(args, "event_step", -1))
        _event_text = str(getattr(args, "event_text", "") or "").strip()
        _event_persist = str(getattr(args, "event_persist", "reaction"))
        event_enabled = _event_step > 0 and bool(_event_text)
        event_fired = False
        event_log_rows = []
        event_template = None
        if event_enabled:
            _tmpl_root = getattr(list_agents[0], "prompt_template_root", None) if list_agents else None
            event_template = event_injection.load_event_template(_tmpl_root)
            print(f"[INFO] Event injection armed: step={_event_step}, persist={_event_persist}, "
                  f"text={_event_text[:60]!r}")

        for t in range(num_steps):
            global CURRENT_INTERACTION_STEP
            CURRENT_INTERACTION_STEP = t + 1

            # --- Fire the one-off event, once, at the chosen step, BEFORE
            #     this step's interaction. Every agent reacts in-character via the
            #     normal LLM entry point (save_turn=False so it is not logged as a
            #     step-2/3 turn); the reaction is stored as an 'event' memory record,
            #     pinned when the persist mode keeps the headline. ---
            if event_enabled and not event_fired and (t + 1) == _event_step:
                event_fired = True

                def _event_react(_agent, _prompt):
                    return get_llm_response(_agent.memory, _prompt, use_history=True, save_turn=False)

                def _event_store(_agent, _text, _pinned):
                    _agent._append_memory_event(_text, current_interaction_type="event", pinned=bool(_pinned))

                print(f"[INFO] EVENT at step {t + 1}: broadcasting to {len(list_agents)} agents.")
                event_log_rows = event_injection.broadcast_event(
                    list_agents, _event_text, t + 1,
                    _event_react, _event_store,
                    template=event_template, persist=_event_persist,
                )
            dict_csv["time_step"].append(t + 1)
            row = []
            row.extend([
                str(RUN_EXPORT_ID or ''),
                int(args.seed),
                str(args.version_set),
                str(getattr(args, 'world', 'closed')),
                str(getattr(args, 'fact_pack_mode', 'off')),
                str(getattr(args, 'rag_content_mode', 'full')),
                str(args.model_name_step2 or args.model_name),
                str(args.model_name_step3 or args.model_name),
                int(t + 1),
            ])

             # ---- NEW: pick listener index at random, speaker via homophily in small-world network
            listener_idx = rng.randrange(num_agents)      # i

            # Candidate speakers depend on network constraints:
            # - If network_type == none: fully mixed (any other agent can speak).
            # - Else: choose from neighbors; if a node has no neighbors, fall back to fully mixed.
            if net_type == "none":
                candidate_speakers = [idx for idx in range(num_agents) if idx != listener_idx]
            else:
                neigh_set = neighbors.get(listener_idx, set())
                candidate_speakers = list(neigh_set) if neigh_set else [idx for idx in range(num_agents) if idx != listener_idx]
                # ADR-006 Component 3: Bernoulli guard on candidate speakers.
                # Only fires when dir_net exists AND a non-default p_reach policy
                # is active - guaranteeing byte-identical baseline otherwise (a
                # uniform=1.0 draw always succeeds anyway, but we skip the loop
                # to keep the RNG stream identical to pre-Component-3 runs).
                if dir_net is not None and _p_reach_active and neigh_set and _p_reach_enforcement == "filter":
                    _filtered = [
                        _s for _s in candidate_speakers
                        if rng.random() < dir_net.get_p_reach(_s, listener_idx)
                    ]
                    if _filtered:
                        candidate_speakers = _filtered
                    # If everyone was filtered out, fall through to the current
                    # candidate list - matches the existing "listener has no
                    # neighbors" fallback semantics rather than silently
                    # dropping the interaction. A counter for these events is
                    # deferred to Task 2.3b-style dispatch work.
                    # (suppress mode does NOT filter candidates here; the reach
                    # check happens after speaker selection - see below.)

            sel_mode = str(getattr(args, "interaction_selection", "homophily")).strip().lower()
            hom_mode = str(getattr(args, "interaction_homophily_mode", "full")).strip().lower()

            if sel_mode == "random":
                speaker_idx = rng.choice(candidate_speakers)
            else:
                # Homophily-weighted selection (with epsilon exploration).
                speaker_idx = choose_partner_scoring(
                    agent_idx=listener_idx,
                    candidates=candidate_speakers,
                    opinions=opinions_by_idx,
                    attributes=attributes,
                    epsilon_uniform=args.epsilon_uniform,
                    rng=rng,
                    score_mode=hom_mode,
                )

            agent_i = list_agents[listener_idx]   # listener
            agent_j = list_agents[speaker_idx]    # speaker

            # ADR-006 fix gamma (suppress mode): the throttled speaker was eligible
            # and may have been chosen (candidates unfiltered -> main stream identical
            # to uniform). Draw the reach Bernoulli from the DEDICATED rng; on failure
            # the tweet does not land, and the listener's belief update is skipped at
            # the commit site below. filter mode leaves this False.
            _reach_suppressed = False
            if dir_net is not None and _p_reach_active and _p_reach_enforcement == "suppress":
                if _p_reach_rng.random() >= dir_net.get_p_reach(speaker_idx, listener_idx):
                    _reach_suppressed = True

            beliefs_before_step = [int(a.current_belief) for a in list_agents]
            B_before_step, D_before_step, P_before_step = _compute_bdp_from_beliefs(beliefs_before_step)

            print(f"STEP {t+1}: speaker={agent_j.agent_name}, listener={agent_i.agent_name}")
            assert agent_i.agent_id != agent_j.agent_id, "Bug: self-interaction detected!"
            row.append(agent_j.agent_name)
            row.append(int(agent_j.agent_id))

            if agent_i.get_count_tweet_seen() + agent_i.get_count_tweet_written() == 1:
                agent_i.outdate_persona_memory()
            if agent_j.get_count_tweet_seen() + agent_j.get_count_tweet_written() == 1:
                agent_j.outdate_persona_memory()

            row.append(agent_j.current_belief)
            agent_j_previos_interaction_type = agent_j.previous_interaction_type

    
            if agent_j_previos_interaction_type in ["none", "write", "read"]:
                tweet_j = agent_j.produce_tweet(
                    previous_interaction_type=agent_j_previos_interaction_type,
                    tweet_written_count=agent_j.get_count_tweet_written(),
                    add_to_memory=False,
                )
                print(f"TWEET from {agent_j.agent_name} (sanitized for listeners): {sanitize_tweet_for_listener(tweet_j)}")
                print(f"RAW_STEP2_OUTPUT from {agent_j.agent_name}: {tweet_j}")
                agent_j.increase_count_tweet_written()
                row.append(tweet_j)
                row.append(str(getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or ''))
                row.append(str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or ''))
                row.append('|'.join(list(getattr(agent_j, 'last_step2_warning_tags', []) or [])))
                row.append(int(bool(getattr(agent_j, 'last_step2_fallback_used', 0))))
                row.append(str(getattr(agent_j, 'last_step2_quality', 'ok') or 'ok'))
                agent_j.append_network_log(
                    "\n".join([
                        f"TIME STEP: {t+1}",
                        "ROLE: SPEAKER",
                        f"TARGET_AGENT: {agent_i.agent_name}",
                        f"PREVIOUS_INTERACTION_TYPE: {agent_j_previos_interaction_type}",
                        f"CURRENT_RATING: {int(agent_j.current_belief)}",
                        "TWEET_OUTPUT:",
                        str(tweet_j or ''),
                    ])
                )
            else:
                raise ValueError(
                    "agent_j_previos_interaction_type is not valid: {}".format(
                        agent_j_previos_interaction_type
                    )
                )

            if MEMORY_ENABLED:
                agent_j.add_to_memory(
                                tweet_written=sanitize_tweet_for_listener(tweet_j),
                                previos_interaction_type=agent_j_previos_interaction_type,
                                current_interaction_type="write",
                                tweet_written_count=agent_j.get_count_tweet_written(),
                            )

            row.append(agent_i.agent_name)
            row.append(int(agent_i.agent_id))
            agent_i_pre_belief = agent_i.current_belief
            row.append(agent_i_pre_belief)

            agent_i_previos_interaction_type = agent_i.previous_interaction_type
            if agent_i_previos_interaction_type in ["none", "write", "read"]:
                # Receiver should see only natural tweet content (no FINAL_RATING, no TWEET header, no [Rk]).
                # Same-step Step-2 output is authoritative: raw/final local artifacts are
                # checked before any older speaker cache, so a valid current TWEET line
                # cannot be lost by the later history/memory handoff.
                initial_tweet_for_listener = sanitize_tweet_for_listener(tweet_j)
                tweet_for_listener, handoff_source, handoff_diag = _recover_same_step_step2_tweet_for_handoff(
                    speaker_agent=agent_j,
                    step2_return=str(tweet_j or ''),
                    sanitized_text=str(initial_tweet_for_listener or ''),
                )
                try:
                    agent_j.last_step2_handoff_tweet = str(tweet_for_listener or '')
                    agent_j.last_step2_handoff_source = str(handoff_source or '')
                    agent_j.last_step2_handoff_diag = dict(handoff_diag or {})
                    agent_j.last_step2_parsed_tweet_len = int(len(str(tweet_for_listener or '')))
                except Exception:
                    pass
                if tweet_for_listener and ((not initial_tweet_for_listener) or str(handoff_source or '') not in {'sanitized_current', ''}):
                    _metric_inc('step3_empty_tweet_handoff_recovered')
                    try:
                        _append_native_event_row(
                            step_kind='step3',
                            agent_name=agent_i.agent_name,
                            label='pre_step3_handoff_recovery',
                            attempt='0',
                            event_type='pipeline_recovered_empty_tweet',
                            text=tweet_for_listener,
                            note=_format_step2_handoff_diag_note(handoff_diag, speaker_name=agent_j.agent_name, listener_name=agent_i.agent_name, source=handoff_source),
                        )
                    except Exception:
                        pass
                if tweet_for_listener:
                    _remember_agent_public_tweet(
                        agent_j,
                        tweet_for_listener,
                        final_output=str(getattr(agent_j, 'last_step2_final_output', '') or tweet_j or ''),
                        raw_output=str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or tweet_j or ''),
                        source=f'pre_step3_handoff::{handoff_source or "unknown"}',
                    )
                if DEBUG_IO:
                    print("\n=== DEBUG: STRING PASSED TO RECEIVER ===")
                    print(tweet_for_listener)

                residue_pat = r"(?i)\bFINAL(?:_| )RATING\b|\bEXPLANATION\b|\bTWEET\s*:"
                if tweet_for_listener and _is_obviously_non_tweet_payload(tweet_for_listener):
                    tweet_for_listener = ""
                if (not tweet_for_listener) or re.search(residue_pat, tweet_for_listener):
                    salvaged_listener_tweet, residue_source, residue_diag = _recover_same_step_step2_tweet_for_handoff(
                        speaker_agent=agent_j,
                        step2_return="\n".join([
                            str(tweet_j or ''),
                            str(getattr(agent_j, 'last_step2_final_output', '') or ''),
                            str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or ''),
                            str(getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or ''),
                        ]),
                        sanitized_text=str(tweet_for_listener or ''),
                    )
                    if salvaged_listener_tweet and (not _is_obviously_non_tweet_payload(salvaged_listener_tweet)) and not re.search(residue_pat, salvaged_listener_tweet):
                        tweet_for_listener = salvaged_listener_tweet
                        _metric_inc('step3_protocol_residue_tweet_handoff_recovered')
                        try:
                            _append_native_event_row(
                                step_kind='step3',
                                agent_name=agent_i.agent_name,
                                label='pre_step3_residue_recovery',
                                attempt='0',
                                event_type='pipeline_recovered_empty_tweet',
                                text=tweet_for_listener,
                                note=_format_step2_handoff_diag_note(residue_diag, speaker_name=agent_j.agent_name, listener_name=agent_i.agent_name, source=residue_source),
                            )
                        except Exception:
                            pass
                    else:
                        fail_blob = "\n".join([
                            str(tweet_j or ''),
                            str(getattr(agent_j, 'last_step2_final_output', '') or ''),
                            str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or ''),
                            str(getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or ''),
                        ])
                        _dump_retry_debug_event(
                            'step3',
                            'warning',
                            agent_name=agent_i.agent_name,
                            attempt=0,
                            reasons='listener_boundary_empty_or_protocol_residue',
                            prompt_text='',
                            raw_text=fail_blob,
                            sanitized_text=str(tweet_for_listener or ''),
                            final_text='',
                        )
                        try:
                            _append_native_event_row(
                                step_kind='step3',
                                agent_name=agent_i.agent_name,
                                label='pre_step3_handoff_unrecovered',
                                attempt='0',
                                event_type='pipeline_input_failure',
                                text=str(tweet_for_listener or ''),
                                note=_format_step2_handoff_diag_note(residue_diag, speaker_name=agent_j.agent_name, listener_name=agent_i.agent_name, source='unrecovered'),
                            )
                        except Exception:
                            pass
                        tweet_for_listener = ""

                # Always let the receiver agent decide whether the tweet is admissible/weak.
                # Do NOT short-circuit Step-3 with a code-level veto, as this artificially
                # increases diagonal (no-change) transitions.
                                # Leak-free allowed rating set for Step-3 (no speaker value exposed to the model):
                # Allowed rating set is mode-dependent:
                # - assimilation_only: neighborhood(listener, max_step_change) intersect non-away-from-speaker
                # - free_bounded: full bounded neighborhood around listener; same-rating interactions may move.
                try:
                    msc_local = int(DEFAULT_MAX_STEP_CHANGE) if DEFAULT_MAX_STEP_CHANGE is not None else 1
                except Exception:
                    msc_local = 1
                allowed_ratings_i = _compute_allowed_ratings(
                    pre_belief=int(agent_i_pre_belief),
                    speaker_belief=int(agent_j.current_belief),
                    max_step_change=msc_local,
                    update_mode=ALLOWED_UPDATE_MODE,
                )
                allowed_ratings_i = _apply_same_side_edge_unlock(
                    pre_belief=int(agent_i_pre_belief),
                    speaker_belief=int(agent_j.current_belief),
                    allowed_ratings=allowed_ratings_i,
                    agent=agent_i,
                    unlock_hits=SAME_SIDE_EDGE_UNLOCK_HITS,
                    tweet_text=tweet_for_listener,
                )
                allowed_set_str_i = _format_allowed_ratings(int(agent_i_pre_belief), allowed_ratings_i)
                row.append(str(list(allowed_ratings_i or [])))

                step2_event_rows.append({
                    'run_id': str(RUN_EXPORT_ID or ''),
                    'seed': int(args.seed),
                    'version_set': str(args.version_set),
                    'world': str(getattr(args, 'world', 'closed')),
                    'allowed_update_mode': str(ALLOWED_UPDATE_MODE),
                    'fact_pack_mode': str(getattr(args, 'fact_pack_mode', 'off')),
                    'rag_content_mode': str(getattr(args, 'rag_content_mode', 'full')),
                    'model_name_step2': str(getattr(agent_j, 'step2_model', args.model_name_step2 or args.model_name)),
                    'time_step': int(t + 1),
                    'speaker_name': agent_j.agent_name,
                    'speaker_agent_id': int(agent_j.agent_id),
                    'listener_name': agent_i.agent_name,
                    'listener_agent_id': int(agent_i.agent_id),
                    'speaker_previous_interaction_type': str(agent_j_previos_interaction_type),
                    'speaker_current_belief': int(agent_j.current_belief),
                    'step2_final_output': str(tweet_j or ''),
                    'step2_sanitized_tweet_for_listener': str(tweet_for_listener or ''),
                    'step2_raw_attempt_1': str(getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or ''),
                    'step2_raw_attempt_2': str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or ''),
                    'step2_warning_tags': '|'.join(list(getattr(agent_j, 'last_step2_warning_tags', []) or [])),
                    'step2_fallback_used': int(bool(getattr(agent_j, 'last_step2_fallback_used', 0))),
                    'step2_interaction_quality': str(getattr(agent_j, 'last_step2_quality', 'ok') or 'ok'),
                    'step2_valid_output_for_listener': int(getattr(agent_j, 'last_step2_valid_output', 1)),
                    'step2_pipeline_artifact': int(getattr(agent_j, 'last_step2_pipeline_artifact', 0)),
                    'step2_handoff_source': str(getattr(agent_j, 'last_step2_handoff_source', '') or ''),
                    'step2_handoff_tweet_len': len(str(tweet_for_listener or '')),
                    'step2_parsed_tweet_len': int(getattr(agent_j, 'last_step2_parsed_tweet_len', 0) or 0),
                    'step2_raw_attempt_1_len': len(str(getattr(agent_j, 'last_step2_raw_response_attempt_1', '') or '')),
                    'step2_raw_attempt_2_len': len(str(getattr(agent_j, 'last_step2_raw_response_attempt_2', '') or '')),
                    'step2_final_output_len': len(str(getattr(agent_j, 'last_step2_final_output', '') or '')),
                })

                response_i = agent_i.receive_tweet(
                    tweet_for_listener,
                    previous_interaction_type=agent_i_previos_interaction_type,
                    tweet_written_count=agent_i.get_count_tweet_written(),
                    add_to_memory=False,
                    allowed_ratings=allowed_ratings_i,
                    allowed_set_str=allowed_set_str_i,
                    speaker_belief=int(agent_j.current_belief),
                )
                print(f"FINAL_RESPONSE from {agent_i.agent_name}: {response_i}")
                agent_i.increase_count_tweet_seen()
                row.append(response_i)
                response_i_row_idx = len(row) - 1
            else:
                raise ValueError(
                    "agent_i_previos_interaction_type is not valid: {}".format(
                        agent_i_previos_interaction_type
                    )
                )
            
            assert "FINAL_RATING" not in tweet_for_listener, "Leak: receiver got FINAL_RATING"
            assert "EXPLANATION" not in tweet_for_listener, "Leak: receiver got EXPLANATION"
            if DEBUG_IO:
                print("=== DEBUG OK: no leakage ===\n")

            if MEMORY_ENABLED:
                agent_i.add_to_memory(
                                tweet_seen=tweet_for_listener,
                                response=response_i,
                                previos_interaction_type=agent_i_previos_interaction_type,
                                current_interaction_type="read",
                                tweet_written_count=agent_i.get_count_tweet_written(),
                            )
            raw_response_i = getattr(agent_i, "last_step3_raw_response", response_i)
            raw_response_i_attempt_1 = getattr(agent_i, "last_step3_raw_response_attempt_1", raw_response_i)
            raw_response_i_attempt_2 = getattr(agent_i, "last_step3_raw_response_attempt_2", "")
            repair_tags_i = list(getattr(agent_i, "last_step3_repair_tags", []) or [])
            repair_tag_groups_i = _split_repair_tags(repair_tags_i)
            hard_repair_tags_i = list(repair_tag_groups_i.get('hard', []))
            soft_cleanup_tags_i = list(repair_tag_groups_i.get('soft', []))
            had_hard_repair_i = int(bool(hard_repair_tags_i))
            had_soft_cleanup_i = int(bool(soft_cleanup_tags_i))
            print(f"RESPONSE_ATTEMPT1 from {agent_i.agent_name}: {raw_response_i_attempt_1}")
            if raw_response_i_attempt_2:
                print(f"RESPONSE_ATTEMPT2 from {agent_i.agent_name}: {raw_response_i_attempt_2}")

            raw_belief = extract_belief(response_i)
            if raw_belief is None:
                claim_txt = _extract_claim_from_anchor(tweet_for_listener)
                repair_tags_i.append("main_missing_final_rating_fallback")
                response_i = _fallback_step3(pre_belief=int(agent_i_pre_belief), claim_txt=claim_txt, tweet_stance=detected_stance_from_tweet(tweet_for_listener), tweet_txt=tweet_for_listener)
                try:
                    row[response_i_row_idx] = response_i
                except Exception:
                    pass
                new_belief = int(agent_i_pre_belief)
            else:
                new_belief = int(raw_belief)

            # Final safety check only; the primary validation/repair now lives inside get_step3_llm_response().
            if int(new_belief) not in set(int(x) for x in (allowed_ratings_i or [])):
                claim_txt = _extract_claim_from_anchor(tweet_for_listener)
                original_output = response_i
                new_belief = _resolve_invalid_step3_rating(
                    proposed_rating=int(new_belief),
                    pre_belief=int(agent_i_pre_belief),
                    allowed_ratings=allowed_ratings_i,
                )
                response_i = _rewrite_step3_output(final_rating=int(new_belief), base_text=original_output, claim_txt=claim_txt)
                repair_tags_i.append("main_final_safety_projection")
                _dump_retry_debug_event('step3', 'rewrite', agent_name=agent_i.agent_name, attempt=None, reasons=['main_final_safety_projection', f'projected_to:{int(new_belief)}'], prompt_text='', raw_text=original_output, sanitized_text=original_output, final_text=response_i)
                try:
                    row[response_i_row_idx] = response_i
                except Exception:
                    pass

            final_expl_text = _extract_explanation_text(response_i)
            display_step3_output = _render_display_step3_output(response_i)
            display_step3_expl = _render_display_step3_explanation(final_expl_text, int(new_belief))
            if _same_bin_softening_tag(int(agent_i_pre_belief), int(new_belief), final_expl_text):
                _metric_inc('same_bin_softening_count')
                _metric_inc(f'same_bin_softening::pre={int(agent_i_pre_belief)}')
            if 'same_rating_cross_bin_wording' in repair_tags_i:
                _metric_inc('same_rating_cross_bin_wording_count')
            if 'moved_rating_stale_category_wording' in repair_tags_i:
                _metric_inc('moved_rating_stale_category_wording_count')
            if 'explanation_reason_preserved' in repair_tags_i:
                _metric_inc('explanation_reason_preserved_count')
            if 'same_rating_cross_bin_wording' in repair_tags_i or 'moved_rating_stale_category_wording' in repair_tags_i:
                _metric_inc('explanation_category_repair_count')

            if _reach_suppressed:
                # gamma skip-preserving shadowban: the throttled speaker's tweet did
                # not reach this listener, so no opinion change is applied. The LLM
                # was still queried and the main rng stream is untouched (the
                # interaction body draws no main rng), giving a clean single-seed
                # causal contrast vs uniform - the sole difference is whether
                # throttled tweets land.
                new_belief = agent_i_pre_belief
                _metric_inc('p_reach_suppressed_interactions')
            agent_i.current_belief = int(new_belief)
            agent_i_post_belief = agent_i.current_belief
            opinions_by_idx[listener_idx] = agent_i.current_belief

            # --- Coevolving network step. Runs only when --network_evolves
            #     is set. Placed AFTER the belief update so rewiring sees the current
            #     opinion, not the pre-interaction one: an add joins agents who now
            #     agree, a cut drops a tie that has become discordant. Coupled so the
            #     edge count is conserved. Default scorer is opinion-only. ---
            if dir_net is not None:
                evolve_once(
                    dir_net, listener_idx, speaker_idx, opinions_by_idx, attributes,
                    step=t,
                    burnin_steps=int(getattr(args, "evolve_burnin", 50)),
                    add_score_threshold=float(getattr(args, "evolve_add_threshold", 100.0)),
                    cut_score_threshold=float(getattr(args, "evolve_cut_threshold", 25.0)),
                    soft_cut_distance=int(getattr(args, "evolve_soft_cut_distance", 2)),
                    p_add=float(getattr(args, "evolve_p_add", 0.07)),
                    min_out_degree=int(getattr(args, "evolve_min_out_degree", 2)),
                    rng=rng,
                )
            delta_belief_i = int(agent_i_post_belief) - int(agent_i_pre_belief)
            if int(agent_i_post_belief) != int(agent_i_pre_belief):
                print(
                    f"BELIEF_CHANGE: {int(agent_i_pre_belief)} -> {int(agent_i_post_belief)} "
                    f"| listener={agent_i.agent_name} | step={t+1}"
                )
            _metric_inc('listener_events_total')
            _metric_inc_transition(int(agent_i_pre_belief), int(agent_i_post_belief))
            listener_valid_interaction_flag = int(getattr(agent_i, 'last_step3_valid_interaction', 1))
            listener_pipeline_artifact_flag = int(getattr(agent_i, 'last_step3_pipeline_artifact', 0))
            if listener_pipeline_artifact_flag:
                _metric_inc('holds_total')
                _metric_inc(f'holds_by_step::{int(t) + 1}')
            if listener_valid_interaction_flag and not listener_pipeline_artifact_flag:
                _metric_inc('listener_valid_interaction_events_total')
            else:
                _metric_inc('listener_invalid_interaction_events_total')

            speaker_stance_label = _speaker_stance_label(agent_j.current_belief)
            _metric_inc(f'speaker_stance::{speaker_stance_label}::events')
            _metric_inc_transition(int(agent_i_pre_belief), int(agent_i_post_belief), prefix=f'speaker_stance::{speaker_stance_label}::transition')

            prev_label = _listener_prev_label(agent_i_previos_interaction_type)
            _metric_inc(f'listener_prev::{prev_label}::events')
            _metric_inc_transition(int(agent_i_pre_belief), int(agent_i_post_belief), prefix=f'listener_prev::{prev_label}::transition')

            allowed_key = _metric_allowed_key(allowed_ratings_i)
            _metric_inc(f'allowed_set::{allowed_key}::events')
            _metric_inc_transition(int(agent_i_pre_belief), int(agent_i_post_belief), prefix=f'allowed_set::{allowed_key}::transition')
            try:
                _allowed_vals = [int(x) for x in (allowed_ratings_i or [])]
            except Exception:
                _allowed_vals = []
            if len(_allowed_vals) == 1:
                _metric_inc(f'allowed_singleton::{_allowed_vals[0]}')
            elif len(_allowed_vals) == 2:
                _metric_inc(f'allowed_pair::{allowed_key}')

            bucket_label = _step_bucket_label(t + 1)
            _metric_inc(f'step_bucket::{bucket_label}::events')
            _metric_inc_transition(int(agent_i_pre_belief), int(agent_i_post_belief), prefix=f'step_bucket::{bucket_label}::transition')

            _metric_inc(f'pre_bin::{int(agent_i_pre_belief)}::events')
            if delta_belief_i > 0:
                _metric_inc('moves_positive')
                _metric_inc(f'speaker_stance::{speaker_stance_label}::moves')
                _metric_inc(f'listener_prev::{prev_label}::moves')
                _metric_inc(f'step_bucket::{bucket_label}::moves')
                _metric_inc(f'pre_bin::{int(agent_i_pre_belief)}::moves')
            elif delta_belief_i < 0:
                _metric_inc('moves_negative')
                _metric_inc(f'speaker_stance::{speaker_stance_label}::moves')
                _metric_inc(f'listener_prev::{prev_label}::moves')
                _metric_inc(f'step_bucket::{bucket_label}::moves')
                _metric_inc(f'pre_bin::{int(agent_i_pre_belief)}::moves')
            else:
                _metric_inc('moves_zero')
                _metric_inc(f'speaker_stance::{speaker_stance_label}::nochange')
                _metric_inc(f'listener_prev::{prev_label}::nochange')
                _metric_inc(f'step_bucket::{bucket_label}::nochange')
                _metric_inc(f'pre_bin::{int(agent_i_pre_belief)}::nochange')
                _metric_inc(f'stay_from_{int(agent_i_pre_belief)}')

            # Counter-attitudinal moves: listener moves in the direction of the opposing speaker stance.
            if speaker_stance_label == 'oppose' and int(agent_i_pre_belief) > 0 and delta_belief_i < 0:
                _metric_inc(f'counter_move::pre={int(agent_i_pre_belief)}::speaker=against')
            if speaker_stance_label == 'support' and int(agent_i_pre_belief) < 0 and delta_belief_i > 0:
                _metric_inc(f'counter_move::pre={int(agent_i_pre_belief)}::speaker=for')

            # Convenience edge-weakening metrics for the key adjacent moves.
            if int(agent_i_pre_belief) == 2 and int(agent_i_post_belief) == 1:
                _metric_inc('edge_weakening::2->1')
            if int(agent_i_pre_belief) == 1 and int(agent_i_post_belief) == 0:
                _metric_inc('edge_weakening::1->0')
            if int(agent_i_pre_belief) == -1 and int(agent_i_post_belief) == 0:
                _metric_inc('edge_weakening::-1->0')
            if int(agent_i_pre_belief) == -2 and int(agent_i_post_belief) == -1:
                _metric_inc('edge_weakening::-2->-1')

            beliefs_after_step = [int(a.current_belief) if idx != listener_idx else int(agent_i_post_belief) for idx, a in enumerate(list_agents)]
            B_after_step, D_after_step, P_after_step = _compute_bdp_from_beliefs(beliefs_after_step)
            crossed_zero_flag = int(((int(agent_i_pre_belief) < 0 and int(agent_i_post_belief) >= 0) or (int(agent_i_pre_belief) > 0 and int(agent_i_post_belief) <= 0)) and int(agent_i_pre_belief) != int(agent_i_post_belief))
            became_more_extreme_flag = int(abs(int(agent_i_post_belief)) > abs(int(agent_i_pre_belief)))
            became_less_extreme_flag = int(abs(int(agent_i_post_belief)) < abs(int(agent_i_pre_belief)))
            distance_to_speaker_before = abs(int(agent_i_pre_belief) - int(agent_j.current_belief))
            distance_to_speaker_after = abs(int(agent_i_post_belief) - int(agent_j.current_belief))
            moved_flag = int(delta_belief_i != 0)
            counter_attitudinal_move_flag = int(
                (speaker_stance_label == 'oppose' and int(agent_i_pre_belief) > 0 and delta_belief_i < 0)
                or (speaker_stance_label == 'support' and int(agent_i_pre_belief) < 0 and delta_belief_i > 0)
            )
            listener_stats = agent_stats.get(int(agent_i.agent_id), None)
            if listener_stats is not None:
                listener_stats['times_as_listener'] += 1
                listener_stats['num_moves_up'] += int(delta_belief_i > 0)
                listener_stats['num_moves_down'] += int(delta_belief_i < 0)
                listener_stats['num_stays'] += int(delta_belief_i == 0)
                listener_stats['abs_total_delta'] += int(abs(delta_belief_i))
                listener_stats['crossed_zero_count'] += int(crossed_zero_flag)
                listener_stats['repair_events'] += int(had_hard_repair_i)
                listener_stats['soft_cleanup_events'] += int(had_soft_cleanup_i)
                listener_stats['valid_listener_interactions'] += int(listener_valid_interaction_flag and not listener_pipeline_artifact_flag)
                listener_stats['invalid_listener_interactions'] += int((not listener_valid_interaction_flag) or listener_pipeline_artifact_flag)
            speaker_stats = agent_stats.get(int(agent_j.agent_id), None)
            if speaker_stats is not None:
                speaker_stats['times_as_speaker'] += 1

            step_summary_rows.append({
                'run_id': str(RUN_EXPORT_ID or ''),
                'seed': int(args.seed),
                'version_set': str(args.version_set),
                'world': str(getattr(args, 'world', 'closed')),
                'fact_pack_mode': str(getattr(args, 'fact_pack_mode', 'off')),
                'rag_content_mode': str(getattr(args, 'rag_content_mode', 'full')),
                'model_name_step2': str(getattr(agent_j, 'step2_model', args.model_name_step2 or args.model_name)),
                'model_name_step3': str(getattr(agent_i, 'step3_model', args.model_name_step3 or args.model_name)),
                'time_step': int(t + 1),
                'interactions': 1,
                'selection_mode': sel_mode,
                'homophily_mode': hom_mode,
                'speaker_name': agent_j.agent_name,
                'speaker_agent_id': int(agent_j.agent_id),
                **_network_metric_fields_for_idx(network_metrics_by_idx, speaker_idx, prefix='speaker_'),
                **_agent_persona_fields(agent_j, prefix='speaker_'),
                'speaker_belief': int(agent_j.current_belief),
                'speaker_tweet': str(tweet_for_listener or tweet_j or ''),
                'speaker_step2_warning_tags': '|'.join(list(getattr(agent_j, 'last_step2_warning_tags', []) or [])),
                'speaker_step2_fallback_used': int(bool(getattr(agent_j, 'last_step2_fallback_used', 0))),
                'speaker_step2_interaction_quality': str(getattr(agent_j, 'last_step2_quality', 'ok') or 'ok'),
                'speaker_step2_valid_output': int(getattr(agent_j, 'last_step2_valid_output', 1)),
                'speaker_step2_pipeline_artifact': int(getattr(agent_j, 'last_step2_pipeline_artifact', 0)),
                'speaker_stance_label': speaker_stance_label,
                'listener_name': agent_i.agent_name,
                'listener_agent_id': int(agent_i.agent_id),
                **_network_metric_fields_for_idx(network_metrics_by_idx, listener_idx, prefix='listener_'),
                **_agent_persona_fields(agent_i, prefix='listener_'),
                'listener_previous_interaction_type': prev_label,
                'listener_pre_belief': int(agent_i_pre_belief),
                'listener_allowed_ratings': str(list(allowed_ratings_i or [])),
                'listener_response': str(response_i or ''),
                'listener_display_response': str(display_step3_output or ''),
                'listener_valid_interaction': int(getattr(agent_i, 'last_step3_valid_interaction', 1)),
                'listener_pipeline_artifact': int(getattr(agent_i, 'last_step3_pipeline_artifact', 0)),
                'listener_real_evaluation': int(getattr(agent_i, 'last_step3_real_evaluation', 1)),
                'listener_display_explanation': str(display_step3_expl or ''),
                'listener_raw_response_attempt_1': str(raw_response_i_attempt_1 or ''),
                'listener_raw_response_attempt_2': str(raw_response_i_attempt_2 or ''),
                'listener_post_belief': int(agent_i_post_belief),
                'delta_belief': int(delta_belief_i),
                'move_direction': 'positive' if delta_belief_i > 0 else 'negative' if delta_belief_i < 0 else 'none',
                'moves_total': int(moved_flag),
                'moves_positive': int(delta_belief_i > 0),
                'moves_negative': int(delta_belief_i < 0),
                'moves_zero': int(delta_belief_i == 0),
                'abs_total_delta': int(abs(delta_belief_i)),
                'net_delta': int(delta_belief_i),
                'crossed_zero_count': int(crossed_zero_flag),
                'became_more_extreme_count': int(became_more_extreme_flag),
                'became_less_extreme_count': int(became_less_extreme_flag),
                'distance_to_speaker_before': int(distance_to_speaker_before),
                'distance_to_speaker_after': int(distance_to_speaker_after),
                'counter_attitudinal_moves': int(counter_attitudinal_move_flag),
                'repair_events': int(had_hard_repair_i),
                'soft_cleanup_events': int(had_soft_cleanup_i),
                'interaction_quality': _interaction_quality_label(repair_tags_i),
                'repair_tags': '|'.join(hard_repair_tags_i),
                'soft_cleanup_tags': '|'.join(soft_cleanup_tags_i),
                'allowed_ratings': str(list(allowed_ratings_i or [])),
                'planner_mode': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('planner_mode', 'off') or 'off'),
                'tool_mode': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('tool_mode', 'off') or 'off'),
                'notes_mode': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('notes_mode', 'off') or 'off'),
                'tool_sequence': '|'.join(list(getattr(agent_i, 'last_step3_tool_meta', {}).get('tool_sequence', []) or [])),
                'web_query': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('web_query', '') or ''),
                'web_results_count': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('web_results_count', 0) or 0),
                'observable_primary_source': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('observable_primary_source', 'tweet') or 'tweet'),
                'observable_secondary_source': str(getattr(agent_i, 'last_step3_tool_meta', {}).get('observable_secondary_source', '') or ''),
                'used_web': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_web', 0) or 0),
                'used_notes': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_notes', 0) or 0),
                'used_rag': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_rag', 0) or 0),
                'used_fact_pack': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_fact_pack', 0) or 0),
                'used_memory_history': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_memory_history', 0) or 0),
                'used_tweet_only': int(getattr(agent_i, 'last_step3_tool_meta', {}).get('used_tweet_only', 1) or 0),
                'B_before': float(B_before_step),
                'D_before': float(D_before_step),
                'P_before': float(P_before_step),
                'B_after': float(B_after_step),
                'D_after': float(D_after_step),
                'P_after': float(P_after_step),
                'simulation_active': 1,
                'is_synthetic': 0,
                'early_stop_reason': '',
            })

            row.append(agent_i_post_belief)
            row.append(delta_belief_i)
            row.append(raw_response_i_attempt_1)
            row.append(raw_response_i_attempt_2)
            row.append("|".join(hard_repair_tags_i))
            row.append("|".join(soft_cleanup_tags_i))
            interaction_quality_i = _interaction_quality_label(repair_tags_i)
            row.append(int(had_hard_repair_i))
            row.append(int(had_soft_cleanup_i))
            row.append(interaction_quality_i)
            writer.writerow(row)

            agent_i.append_network_log(
                "\n".join([
                    f"TIME STEP: {t+1}",
                    "ROLE: LISTENER",
                    f"SOURCE_AGENT: {agent_j.agent_name}",
                    f"PREVIOUS_INTERACTION_TYPE: {agent_i_previos_interaction_type}",
                    f"PRE_BELIEF: {int(agent_i_pre_belief)}",
                    f"SPEAKER_BELIEF: {int(agent_j.current_belief)}",
                    f"ALLOWED_FINAL_RATING_SET: {allowed_set_str_i}",
                    "TWEET_SEEN:",
                    str(tweet_for_listener or ''),
                    "RAW_STEP3_ATTEMPT_1:",
                    str(raw_response_i_attempt_1 or ''),
                    "RAW_STEP3_ATTEMPT_2:",
                    str(raw_response_i_attempt_2 or ''),
                    "FINAL_STEP3_OUTPUT:",
                    str(response_i or ''),
                    "DISPLAY_STEP3_OUTPUT:",
                    str(display_step3_output or ''),
                    f"POST_BELIEF: {int(agent_i_post_belief)}",
                    f"DELTA_BELIEF: {delta_belief_i}",
                    f"REPAIR_TAGS: {'|'.join(hard_repair_tags_i)}",
                    f"SOFT_CLEANUP_TAGS: {'|'.join(soft_cleanup_tags_i)}",
                ])
            )
            if hard_repair_tags_i or soft_cleanup_tags_i:
                _dump_repair_event(
                    step='step3',
                    agent_name=agent_i.agent_name,
                    repair_tags=repair_tags_i,
                    raw_text_attempt_1=raw_response_i_attempt_1,
                    raw_text_attempt_2=raw_response_i_attempt_2,
                    final_text=response_i,
                    pre_belief=int(agent_i_pre_belief),
                    speaker_belief=int(agent_j.current_belief),
                    allowed_ratings=allowed_ratings_i,
                )

            agent_i.previous_interaction_type = "read"
            agent_j.previous_interaction_type = "write"

            for agent in list_agents:
                dict_csv[agent.agent_name].append(agent.current_belief)

            # Live snapshot for UI (one row per step)
            try:
                op_writer.writerow([t + 1] + [a.current_belief for a in list_agents])
                f_op.flush()
            except Exception:
                pass

            dict_agent_tweet[agent_j].append((t + 1, agent_j.current_belief, tweet_j))
            dict_agent_response[agent_i].append((t + 1, response_i, display_step3_output))
            
            # ---- EXPORT NETWORK EDGES EVERY 5 STEPS ----
            if t % 5 == 0:
                if dir_net is not None:
                    # Directed export: one row per directed edge, plus a mutual flag.
                    # Never dedupe with i<j here - under direction that drops every
                    # high-to-low edge.
                    edge_rows_step = [
                        (s, d, list_agents[s].agent_name, list_agents[d].agent_name, int(mutual), dir_net.get_p_reach(s, d))
                        for s, d, mutual in dir_net.edges()
                    ]
                    df_edges_step = pd.DataFrame(
                        edge_rows_step,
                        columns=["src_idx", "dst_idx", "src_name", "dst_name", "mutual", "p_reach"],
                    )
                else:
                    edge_rows_step = []
                    for i, neighs in neighbors.items():
                        for j in neighs:
                            if i < j:
                                edge_rows_step.append((i, j, list_agents[i].agent_name, list_agents[j].agent_name))

                    df_edges_step = pd.DataFrame(
                        edge_rows_step,
                        columns=["i_idx", "j_idx", "i_name", "j_name"]
                    )

                step_edge_path = (
                    os.path.join(csv_folder, os.path.basename(csv_folder) + f"_step_{t:03d}_edges.csv")
                )

                df_edges_step.to_csv(step_edge_path, index=False)
                print(f"[Step {t}] Saved evolving network edges -> {step_edge_path}")

            # Periodic checkpoint of the main analysis table. It is otherwise
            # written only after the loop, so a kill mid-run (e.g. on a rented
            # instance) would lose every derived row. _write_step_summary_csv
            # rewrites the whole growing list, so this is idempotent; the final
            # write still happens at the end.
            if step_summary_out_name and (t % 5 == 0):
                try:
                    _write_step_summary_csv(step_summary_out_name, step_summary_rows)
                except Exception as _e:
                    print(f"[warn] periodic step_summary checkpoint failed: {_e}")

            unanimous_belief = _unanimous_belief(list_agents)
            if unanimous_belief is None:
                mild_unanimity_value = None
                mild_unanimity_remaining = None
            elif int(unanimous_belief) in (-2, 0, 2):
                early_stop_triggered = True
                early_stop_step = int(t + 1)
                early_stop_reason = f"unanimous_{int(unanimous_belief)}"
                _metric_inc('early_stop_triggered_total')
                _metric_inc(f'early_stop_reason::{early_stop_reason}')
                break
            elif int(unanimous_belief) in (-1, 1):
                if mild_unanimity_value != int(unanimous_belief):
                    mild_unanimity_value = int(unanimous_belief)
                    mild_unanimity_remaining = int(max(0, num_agents))
                else:
                    mild_unanimity_remaining = max(0, int(mild_unanimity_remaining or 0) - 1)
                    if mild_unanimity_remaining <= 0:
                        early_stop_triggered = True
                        early_stop_step = int(t + 1)
                        early_stop_reason = f"unanimous_{int(unanimous_belief)}_buffer_exhausted"
                        _metric_inc('early_stop_triggered_total')
                        _metric_inc(f'early_stop_reason::{early_stop_reason}')
                        break
            else:
                mild_unanimity_value = None
                mild_unanimity_remaining = None

        # --- Event injection: dump the per-agent reactions. relevance is
        #     left blank here and computed offline from event_text vs the claim, so
        #     the hot loop stays free of embedding calls. ---
        if event_log_rows:
            try:
                _ev_path = os.path.join(csv_folder, os.path.basename(csv_folder) + "_event_log.csv")
                pd.DataFrame(event_log_rows, columns=event_injection.EVENT_LOG_COLUMNS).to_csv(_ev_path, index=False)
                print(f"[INFO] Event log -> {_ev_path}")
            except Exception as _e:
                print(f"[warn] could not write event log: {_e}")

        # --- Coevolving network: dump the edge-change log. One row per
        #     directed add/cut with step, endpoints and reason, so the per-step
        #     edge snapshots can be explained, not just observed. ---
        if dir_net is not None and dir_net.changes:
            try:
                _chg_path = os.path.join(csv_folder, os.path.basename(csv_folder) + "_edge_changes.csv")
                pd.DataFrame(
                    dir_net.changes,
                    columns=["step", "src_idx", "dst_idx", "action", "reason"],
                ).to_csv(_chg_path, index=False)
                print(f"[INFO] Edge-change log -> {_chg_path}")
            except Exception as _e:
                print(f"[warn] could not write edge-change log: {_e}")

        if early_stop_triggered and early_stop_step is not None and int(early_stop_step) < int(num_steps):
            current_beliefs_fill = [int(a.current_belief) for a in list_agents]
            B_fill, D_fill, P_fill = _compute_bdp_from_beliefs(current_beliefs_fill)
            if dir_net is not None:
                final_edge_rows = [
                    (s, d, list_agents[s].agent_name, list_agents[d].agent_name, int(mutual), dir_net.get_p_reach(s, d))
                    for s, d, mutual in dir_net.edges()
                ]
                final_edge_cols = ["src_idx", "dst_idx", "src_name", "dst_name", "mutual", "p_reach"]
            else:
                final_edge_rows = []
                for i, neighs in neighbors.items():
                    for j in neighs:
                        if i < j:
                            final_edge_rows.append((i, j, list_agents[i].agent_name, list_agents[j].agent_name))
                final_edge_cols = ["i_idx", "j_idx", "i_name", "j_name"]
            try:
                print(f"[INFO] Early stop at step {int(early_stop_step)} ({early_stop_reason}); padding synthetic plateau rows through step {int(num_steps)}.")
            except Exception:
                pass

            for synthetic_step in range(int(early_stop_step) + 1, int(num_steps) + 1):
                CURRENT_INTERACTION_STEP = synthetic_step
                dict_csv["time_step"].append(synthetic_step)
                for idx_agent, agent in enumerate(list_agents):
                    dict_csv[agent.agent_name].append(current_beliefs_fill[idx_agent])

                synthetic_row = ["" for _ in range(len(interaction_header))]
                synthetic_row[0] = str(RUN_EXPORT_ID or '')
                synthetic_row[1] = int(args.seed)
                synthetic_row[2] = str(args.version_set)
                synthetic_row[3] = str(getattr(args, 'world', 'closed'))
                synthetic_row[4] = str(getattr(args, 'fact_pack_mode', 'off'))
                synthetic_row[5] = str(getattr(args, 'rag_content_mode', 'full'))
                synthetic_row[6] = str(args.model_name_step2 or args.model_name)
                synthetic_row[7] = str(args.model_name_step3 or args.model_name)
                synthetic_row[8] = int(synthetic_step)
                synthetic_row[17] = 'synthetic_plateau_fill'
                synthetic_row[31] = 'synthetic_plateau_fill'
                writer.writerow(synthetic_row)

                step_summary_rows.append({
                    'run_id': str(RUN_EXPORT_ID or ''),
                    'seed': int(args.seed),
                    'version_set': str(args.version_set),
                    'world': str(getattr(args, 'world', 'closed')),
                    'fact_pack_mode': str(getattr(args, 'fact_pack_mode', 'off')),
                    'rag_content_mode': str(getattr(args, 'rag_content_mode', 'full')),
                    'model_name_step2': str(args.model_name_step2 or args.model_name),
                    'model_name_step3': str(args.model_name_step3 or args.model_name),
                    'time_step': int(synthetic_step),
                    'interactions': 0,
                    'selection_mode': 'plateau_fill',
                    'homophily_mode': '',
                    'speaker_name': '',
                    'speaker_agent_id': '',
                    'speaker_belief': '',
                    'speaker_tweet': '',
                    'speaker_step2_warning_tags': '',
                    'speaker_step2_fallback_used': 0,
                    'speaker_step2_interaction_quality': 'synthetic_plateau_fill',
                    'speaker_step2_valid_output': 0,
                    'speaker_step2_pipeline_artifact': 0,
                    'speaker_stance_label': '',
                    'listener_name': '',
                    'listener_agent_id': '',
                    'listener_previous_interaction_type': 'none',
                    'listener_pre_belief': '',
                    'listener_allowed_ratings': '',
                    'listener_response': '',
                    'listener_display_response': '',
                    'listener_valid_interaction': 0,
                    'listener_pipeline_artifact': 0,
                    'listener_real_evaluation': 0,
                    'listener_display_explanation': '',
                    'listener_raw_response_attempt_1': '',
                    'listener_raw_response_attempt_2': '',
                    'listener_post_belief': '',
                    'delta_belief': 0,
                    'move_direction': 'none',
                    'moves_total': 0,
                    'moves_positive': 0,
                    'moves_negative': 0,
                    'moves_zero': 0,
                    'abs_total_delta': 0,
                    'net_delta': 0,
                    'crossed_zero_count': 0,
                    'became_more_extreme_count': 0,
                    'became_less_extreme_count': 0,
                    'distance_to_speaker_before': '',
                    'distance_to_speaker_after': '',
                    'counter_attitudinal_moves': 0,
                    'repair_events': 0,
                    'soft_cleanup_events': 0,
                    'interaction_quality': 'synthetic_plateau_fill',
                    'repair_tags': '',
                    'soft_cleanup_tags': '',
                    'allowed_ratings': '',
                    'planner_mode': 'off',
                    'tool_mode': 'off',
                    'notes_mode': 'off',
                    'tool_sequence': '',
                    'web_query': '',
                    'web_results_count': 0,
                    'observable_primary_source': 'plateau_fill',
                    'observable_secondary_source': '',
                    'used_web': 0,
                    'used_notes': 0,
                    'used_rag': 0,
                    'used_fact_pack': 0,
                    'used_memory_history': 0,
                    'used_tweet_only': 0,
                    'B_before': float(B_fill),
                    'D_before': float(D_fill),
                    'P_before': float(P_fill),
                    'B_after': float(B_fill),
                    'D_after': float(D_fill),
                    'P_after': float(P_fill),
                    'simulation_active': 0,
                    'is_synthetic': 1,
                    'early_stop_reason': str(early_stop_reason or 'plateau_fill'),
                })

                try:
                    op_writer.writerow([int(synthetic_step)] + list(current_beliefs_fill))
                    f_op.flush()
                except Exception:
                    pass

                if (int(synthetic_step) - 1) % 5 == 0:
                    df_edges_step = pd.DataFrame(
                        final_edge_rows,
                        columns=final_edge_cols
                    )
                    step_edge_path = (
                        os.path.join(csv_folder, os.path.basename(csv_folder) + f"_step_{int(synthetic_step)-1:03d}_edges.csv")
                    )
                    df_edges_step.to_csv(step_edge_path, index=False)

        # (opinion-change CSV is streamed to out_name during the run for the Live tab)
        # pd.DataFrame.from_dict(dict_csv).to_csv(out_name)
        # ---- NEW: summarize final neighbor opinions (echo-chamber diagnostics)
    neighbor_summary_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_neighbor_summary.csv")
    )

    os.makedirs(os.path.dirname(neighbor_summary_name), exist_ok=True)
    with open(neighbor_summary_name, "w", newline="", encoding="utf-8") as f_sum:
        w_sum = csv.writer(f_sum)
        w_sum.writerow(
            [
                "Agent_Name",
                "Own_Final_Opinion",
                "Mean_Neighbor_Opinion",
                "Num_Neighbors",
                "Frac_Same_Opinion",
                "Frac_Close_Opinion_(|Δ|<=1)",
                "Frac_Opposite_Opinion_(|Δ|>=3)",
                "Neighbors_-2",
                "Neighbors_-1",
                "Neighbors_0",
                "Neighbors_1",
                "Neighbors_2",
            ]
        )
        

        for idx, agent in enumerate(list_agents):
            own = agent.current_belief
            neigh_indices = neighbors.get(idx, set())
            if not neigh_indices:
                continue

            neigh_beliefs = [opinions_by_idx[j] for j in neigh_indices]
            num_n = len(neigh_beliefs)
            mean_nb = sum(neigh_beliefs) / num_n

            diffs = [abs(own - b) for b in neigh_beliefs]
            same_frac = sum(1 for d in diffs if d == 0) / num_n
            close_frac = sum(1 for d in diffs if d <= 1) / num_n
            opp_frac = sum(1 for d in diffs if d >= 3) / num_n

            counts = Counter(neigh_beliefs)

            w_sum.writerow(
                [
                    agent.agent_name,
                    own,
                    mean_nb,
                    num_n,
                    same_frac,
                    close_frac,
                    opp_frac,
                    counts.get(-2, 0),
                    counts.get(-1, 0),
                    counts.get(0, 0),
                    counts.get(1, 0),
                    counts.get(2, 0),
                ]
            )
    try:
        final_beliefs = [int(a.current_belief) for a in list_agents]
        init_beliefs = [int(a.init_belief) for a in list_agents]
        _metric_inc('agents_total', len(list_agents))
        _metric_inc('final_more_positive_than_init', sum(1 for i, f in zip(init_beliefs, final_beliefs) if f > i))
        _metric_inc('final_more_negative_than_init', sum(1 for i, f in zip(init_beliefs, final_beliefs) if f < i))
        _metric_inc('final_same_as_init', sum(1 for i, f in zip(init_beliefs, final_beliefs) if f == i))
        if RUN_METRICS.get('listener_valid_interaction_events_total', 0):
            _metric_inc('sensitivity_valid_interactions_available', 1)
    except Exception:
        pass
    agent_summary_rows = []
    for agent in list_agents:
        stats = dict(agent_stats.get(int(agent.agent_id), {}) or {})
        agent_summary_rows.append({
            'run_id': str(RUN_EXPORT_ID or ''),
            'seed': int(args.seed),
            'version_set': str(args.version_set),
            'world': str(getattr(args, 'world', 'closed')),
            'fact_pack_mode': str(getattr(args, 'fact_pack_mode', 'off')),
            'rag_content_mode': str(getattr(args, 'rag_content_mode', 'full')),
            'model_name_step2': str(getattr(agent, 'step2_model', args.model_name_step2 or args.model_name)),
            'model_name_step3': str(getattr(agent, 'step3_model', args.model_name_step3 or args.model_name)),
            'agent_id': int(agent.agent_id),
            'agent_name': str(agent.agent_name),
            'init_belief': int(agent.init_belief),
            'final_belief': int(agent.current_belief),
            'delta_from_init': int(int(agent.current_belief) - int(agent.init_belief)),
            **_network_metric_fields_for_idx(network_metrics_by_idx, agent_local_index_by_agent_id.get(int(agent.agent_id), int(agent.agent_id) - 1), prefix=''),
            **_agent_persona_fields(agent, prefix=''),
            **stats,
        })

    network_hub_metric_rows = []
    for agent in list_agents:
        local_idx = agent_local_index_by_agent_id.get(int(agent.agent_id), int(agent.agent_id) - 1)
        network_hub_metric_rows.append({
            'run_id': str(RUN_EXPORT_ID or ''),
            'seed': int(args.seed),
            'version_set': str(args.version_set),
            'world': str(getattr(args, 'world', 'closed')),
            'fact_pack_mode': str(getattr(args, 'fact_pack_mode', 'off')),
            'rag_content_mode': str(getattr(args, 'rag_content_mode', 'full')),
            'agent_id': int(agent.agent_id),
            'agent_name': str(agent.agent_name),
            'init_belief': int(agent.init_belief),
            'final_belief': int(agent.current_belief),
            'delta_from_init': int(int(agent.current_belief) - int(agent.init_belief)),
            **_network_metric_fields_for_idx(network_metrics_by_idx, local_idx, prefix=''),
        })

    _write_run_metrics_csv(metrics_out_name)
    _write_network_hub_metrics_csv(network_hub_metrics_out_name, network_hub_metric_rows)
    _write_step_summary_csv(step_summary_out_name, step_summary_rows)
    _write_step2_event_summary_csv(step2_event_out_name, step2_event_rows)
    _write_agent_summary_csv(agent_summary_out_name, agent_summary_rows)

    return (
        post_process_memory(list_agents, path_result, date_version),
        post_process_tweet(dict_agent_tweet, path_result, date_version, csv_folder),
        post_process_response(dict_agent_response, path_result, date_version, csv_folder),
    )


def post_process_memory(list_agents, path_result, date_version):
    """Clean memory text before storing or displaying it."""
    for agent in list_agents:
        out_name = os.path.join(
            path_result,
            "network_log_conversation",
            args.output_file.split(".cs")[0]
            + "_"
            + str(args.num_agents)
            + "_"
            + str(args.num_steps)
            + "_"
            + args.version_set
            + "_"
            + date_version
            + "_network_agent_"
            + str(agent.agent_id)
            + "_"
            + args.distribution
            + ".txt",
        )
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(agent.memory.prompt.messages[0].prompt.template)
            f.write("\n------------------------------\n")
            for entry in getattr(agent, 'network_log_entries', []) or []:
                f.write(str(entry))
                f.write("\n------------------------------\n")
    return



def post_process_tweet(dict_agent_tweet, path_result, date_version, csv_folder):
    """
    Clean generated tweet text for public handoff and transcript display.
    
        The cleanup removes protocol labels, fences, and obvious meta text while preserving the speaker's
        substantive stance. It is used after Step-2 validation, not as a substitute for validation.
    """
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_agent_tweet_history.csv")
    )
    with open(out_name, "w", newline="", encoding="utf-8") as g:
        writer = csv.writer(g, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Tweet Time Step",
                "Belief When Tweeting",
                "Tweet Chain",
            ]
        )

        for agent in dict_agent_tweet.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_tweet[agent]]
            belief_when_tweeting = [time_step[1] for time_step in dict_agent_tweet[agent]]
            tweet_chain = [time_step[2] for time_step in dict_agent_tweet[agent]]

            row.append(list(time_step_changes))
            row.append(list(belief_when_tweeting))
            row.append(list(tweet_chain))

            writer.writerow(row)
    return


def post_process_response(dict_agent_response, path_result, date_version,csv_folder):
    """
    Clean listener response text before parsing or display.
    
        This normalizes Step-3 formatting artifacts and strips non-content wrappers, while leaving final
        rating extraction to the dedicated parser so repair/validation decisions remain explicit.
    """
    if not os.path.exists(path_result):
        os.makedirs(path_result)
        print("Created a fresh directory!")

    out_name = (
        os.path.join(csv_folder, os.path.basename(csv_folder) + "_agent_response_history.csv")
    )
    with open(out_name, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(
            [
                "Agent Name",
                "Original Belief",
                "Belief Changes Time Step",
                "Belief Change Chain",
                "Response chain",
                "Display response chain",
            ]
        )

        for agent in dict_agent_response.keys():
            row = []
            row.append(agent.agent_name)
            row.append(agent.init_belief)

            time_step_changes = [time_step[0] for time_step in dict_agent_response[agent]]
            belief_changes = []
            response_chain = []
            display_response_chain = []
            for item in dict_agent_response[agent]:
                ts = item[0] if len(item) >= 1 else ""
                resp = item[1] if len(item) >= 2 else ""
                disp = item[2] if len(item) >= 3 else _render_display_step3_output(resp)
                b = extract_belief(resp)
                belief_changes.append("" if b is None else b)
                response_chain.append(resp)
                display_response_chain.append(disp)

            row.append(list(time_step_changes))
            row.append(list(belief_changes))
            row.append(list(response_chain))
            row.append(list(display_response_chain))

            writer.writerow(row)
    return

from typing import Optional


def _is_obviously_non_tweet_payload(text: str) -> bool:
    """Detect outputs that are clearly not public tweet content."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return True
    low = s.lower()

    # Ultra-short fragments should never cross the speaker->listener boundary.
    words = re.findall(r"[A-Za-z']+", s)
    if len(words) < 2 or len(re.findall(r"[A-Za-z]", s)) < 6:
        return True

    if low in {
        "final_rating", "final rating", "tweet", "explanation", "reason", "reasoning",
        "anchor", "allowed_set", "allowed final rating set"
    }:
        return True

    if re.fullmatch(r"(?is)final(?:_| )rating\s*:?[-+\s]?\d*", s):
        return True
    if re.fullmatch(r"(?is)(tweet|explanation|reason|reasoning|anchor)\s*: ?", s):
        return True

    strong_meta = [
        r"\bfinal(?:_| )rating\b",
        # Do not reject ordinary tweet text just because it contains the word
        # "explanation" (e.g. "not a reliable explanation for reality").
        # Header/protocol cases are already caught by the exact/fullmatch checks
        # above and by the explicit EXPLANATION: stripping in extractors.
        r"\boutput format\b",
        r"\bexact format\b",
        r"\brequired output\b",
        r"\binstructions?\b",
        r"\bprompt\b",
        r"\ballowed(?:[_ ]final)?[_ ]rating(?:[_ ]set)?\b",
        r"\bcurrent[_ ]rating\b",
        r"\bprevious[_ ]rating\b",
    ]
    if any(re.search(p, low, flags=re.I) for p in strong_meta):
        return True

    return False

def _extract_protocol_body(text: str, field: str = "TWEET") -> str:
    """Extract a single protocol field body from model output.

    This is a small utility for local fallback/rewrite paths. It normalizes common
    protocol header variants and returns only the requested field body without
    leaking other protocol lines.
    """
    s = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(text or ""))
    if not s:
        return ""

    fld = str(field or "").strip().upper().replace(" ", "_")
    if fld == "TWEET":
        return extract_tweet_text(s)

    m = re.search(
        rf"(?ims)^\s*{re.escape(fld)}\s*:\s*(.+?)(?:\n\s*(?:FINAL(?:_| )RATING|TWEET|EXPLANATION|REASONING|ANCHOR|NOTES?)\s*:|\Z)",
        s,
    )
    body = (m.group(1) or "").strip() if m else _heuristic_freeform_protocol_body(s, field=(fld.lower() if fld else "tweet")).strip()
    body = _strip_leaked_prompt_headers(body)
    body = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", body).strip()
    body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET|NOTES?)\s*:", "", body).strip()
    body = re.sub(r"\[[^\]]+\]", "", body).strip()
    body = re.sub(r"\s+", " ", body).strip()
    return body


def extract_tweet_text(step2_output: str) -> str:
    """
    Extract only the public TWEET body from a Step-2 protocol response.
    
        The function fails closed: if the candidate still looks like labels, instructions, or other protocol
        residue, it returns an empty string so the listener path can log a missing-handoff artifact instead
        of evaluating junk.
    """
    if step2_output is None:
        return ""
    text = str(step2_output).strip()
    if not text:
        return ""

    text = _normalize_protocol_headers(_strip_model_think_and_fence_artifacts(text))

    m = re.search(
        r"(?ims)^\s*TWEET\s*:\s*(.+?)(?:\n\s*(?:EXPLANATION|REASONING|NOTES?|ANCHOR|FINAL(?:_| )RATING)\s*:|\Z)",
        text,
    )
    if m:
        body = (m.group(1) or "").strip()
    else:
        missing_tweet_label = _normalize_step2_missing_tweet_label(text)
        if missing_tweet_label:
            m2 = re.search(
                r"(?ims)^\s*TWEET\s*:\s*(.+?)(?:\n\s*(?:EXPLANATION|REASONING|NOTES?|ANCHOR|FINAL(?:_| )RATING)\s*:|\Z)",
                missing_tweet_label,
            )
            body = (m2.group(1) or "").strip() if m2 else ""
        else:
            body = _extract_step2_multiline_tweet_body(text) or _heuristic_freeform_protocol_body(text, field="tweet").strip()

    body = _strip_leaked_prompt_headers(body)
    body = re.sub(r"(?im)\bFINAL(?:_| )RATING\b\s*:?\s*[+-]?\d*", "", body).strip()
    body = re.sub(r"(?im)\b(?:EXPLANATION|REASONING|ANCHOR|TWEET)\s*:", "", body).strip()
    body = re.sub(r"\[[^\]]+\]", "", body).strip()
    body = re.sub(r"\s+", " ", body).strip()

    if not _step2_tweet_has_valid_public_body(body):
        return ""
    return body

def extract_belief(reply: str) -> Optional[int]:
    """
    Extract the integer FINAL_RATING value from a protocol response.
    
        Both Step-2 and Step-3 rely on this parser after sanitation. Invalid or missing values return None
        so callers can trigger retry, projection, or fallback rather than silently accepting bad output.
    """
    if not reply:
        return None

    lines = [ln.strip() for ln in str(reply).splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = [lines[0]] + lines[1:5]

    for line in candidates:
        up = line.upper()
        if up.startswith("FINAL_RATING:") or up.startswith("FINAL RATING:"):
            val_str = line.split(":", 1)[1].strip()
            if re.fullmatch(r"[+-]?\d+", val_str):
                r = int(val_str)
                if r in (-2, -1, 0, 1, 2):
                    return r
            return None

    return None

def extract_reasoning(tweet):
    """
    Extract the EXPLANATION body from a Step-3 protocol response.
    
        This parser intentionally reads only the explanation field. Rating extraction and semantic checks
        are handled separately so each validation failure can be tagged precisely.
    """
    if not rating_flag:
        reasoning = tweet.split("\nFinal Answer")[0]
    else:
        reasoning = "Reasoning:" + tweet.split("\nReasoning")[-1]
    return reasoning


def _run_solo_check():
    """
    Solo model check (merged from opinion_dynamics_v3_check.py, 2026-07-17).

        N independent samples of the model answer the version's step1_report.md (fallback: the
        shared llm_check_template) about the claim - no personas, no network, no interactions.
        This measures the model's own prior on the topic and serves as the per-model baseline
        for cross-model comparisons. Runs on the SAME native Ollama path and decoding options
        as normal runs (single inference stack -> comparable numbers), keeps the v3 update rule
        (delta clamped to +/-1 per step from a neutral start) and the v3 output layout
        (*_opinion_change_* trajectories CSV + raw responses CSV), and adds a metrics JSON with
        config self-documentation and call counters.
    """
    import json as _json

    random.seed(args.seed)
    np.random.seed(args.seed)

    experiment_id = "Flache_2017"
    model_name = str(args.model_name)
    model_name_for_path = re.sub(r'[<>:"/\\|?*]', "_", model_name).strip().rstrip(".")
    num_agents = int(args.num_agents)
    num_steps = int(args.num_steps)

    base_root = os.path.join("prompts", "opinion_dynamics", experiment_id)
    vset = str(getattr(args, "version_set", "") or "").strip()
    candidates = []
    if vset:
        candidates.append(os.path.join(base_root, _version_prefix(vset), vset, "step1_report.md"))
    candidates.append(os.path.join(base_root, "llm_check_template", "step1_report.md"))
    report_path = next((c for c in candidates if os.path.exists(c)), None)
    if not report_path:
        raise SystemExit(f"[solo_check] no step1_report.md found (tried: {candidates})")

    claim = _current_claim_text() or ""
    base_prompt = open(report_path, encoding="utf-8").read()
    base_prompt = base_prompt.replace("{THEORY_STATEMENT}", claim).replace("{CLAIM}", claim)

    llm = build_chat_ollama(model_name, float(args.temperature))
    conv = type("SoloConversation", (), {"llm": llm})()

    path_result = os.path.join("results", "opinion_dynamics", experiment_id, model_name_for_path)
    if getattr(args, "test_run", False):
        path_result = os.path.join(path_result, "test_runs")
    os.makedirs(path_result, exist_ok=True)
    stem = os.path.splitext(str(getattr(args, "output_file", "") or "llm_check"))[0] or "llm_check"
    tag = f"{stem}_{num_agents}_{num_steps}_{vset or 'llm_check'}"
    out_name = os.path.join(path_result, f"{tag}_opinion_change_20231212_uniform.csv")
    raw_name = os.path.join(path_result, f"{tag}_llm_check_raw_responses_20231212.csv")
    metrics_name = os.path.join(path_result, f"{tag}_solo_check_metrics.json")

    print(f"[solo_check] template: {report_path}")
    print(f"[solo_check] claim: {claim[:120]}")
    print(f"[solo_check] model={model_name} agents={num_agents} steps={num_steps} "
          f"temperature={args.temperature}")

    beliefs = [0 for _ in range(num_agents)]
    data = {"time_step": [0]}
    for k in range(num_agents):
        data[f"Agent_{k+1}"] = [0]
    calls = parse_fail = reprompts = 0

    with open(raw_name, "w", newline="", encoding="utf-8") as raw_f:
        raw_writer = csv.writer(raw_f)
        raw_writer.writerow(["time_step", "agent_index", "response_text", "extracted_rating"])
        for t in range(num_steps):
            data["time_step"].append(t + 1)
            for k in range(num_agents):
                _set_native_ollama_debug_context(step="solo_check", agent_name=f"Agent_{k+1}",
                                                 attempt=None, label="solo")
                response = ""
                for _attempt in range(3):
                    response = _native_ollama_chat(conv, base_prompt,
                                                   include_history=False, save_turn=False)
                    calls += 1
                    if re.search(r"[+-]?\d*\.\d+", response or ""):
                        reprompts += 1
                        continue
                    ok = False
                    for m in re.findall(r"[-+]?\d+", response or ""):
                        try:
                            if -2 <= int(m) <= 2:
                                ok = True
                                break
                        except ValueError:
                            continue
                    if ok:
                        break
                    reprompts += 1
                rating = extract_belief(response)
                if rating is None:
                    parse_fail += 1
                    new_belief = beliefs[k]
                else:
                    delta = max(-1, min(1, int(rating) - beliefs[k]))
                    new_belief = max(-2, min(2, beliefs[k] + delta))
                beliefs[k] = new_belief
                data[f"Agent_{k+1}"].append(new_belief)
                raw_writer.writerow([t + 1, k + 1, response, rating])
            print(f"[solo_check] step {t+1}/{num_steps} mean belief: "
                  f"{sum(beliefs)/max(1, num_agents):+.2f}")

    pd.DataFrame.from_dict(data).to_csv(out_name, index=False, encoding="utf-8")
    dist = {str(v): beliefs.count(v) for v in (-2, -1, 0, 1, 2)}
    metrics = {
        "mode": "solo_check",
        "config": {
            "model": model_name,
            "temperature": float(args.temperature),
            "top_p": getattr(args, "top_p", None),
            "top_k": getattr(args, "top_k", None),
            "seed": int(args.seed),
            "version_set": vset,
            "template": report_path,
            "claim": claim,
            "num_agents": num_agents,
            "num_steps": num_steps,
        },
        "counters": {"native_calls": calls, "parse_failures": parse_fail,
                     "integer_reprompts": reprompts},
        "final": {"mean_belief": sum(beliefs) / max(1, num_agents), "distribution": dist},
    }
    with open(metrics_name, "w", encoding="utf-8") as mf:
        _json.dump(metrics, mf, indent=1)
    print(f"[solo_check] trajectories -> {out_name}")
    print(f"[solo_check] raw responses -> {raw_name}")
    print(f"[solo_check] metrics -> {metrics_name}")
    print(f"[solo_check] final mean belief {metrics['final']['mean_belief']:+.2f}, "
          f"distribution {dist}, parse failures {parse_fail}/{calls}")


if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)

    if str(getattr(args, "solo_check", "off")).strip().lower() == "on":
        _run_solo_check()
        sys.exit(0)

    experiment_id = "Flache_2017"
    model_name = args.model_name
    # ADR-006 Component 4: opt-in Bian diagnostic (no-op unless --include_bian_scores on).
    globals()["RUN_BIAN_SCORES"] = _load_bian_scores_if_requested()
    model_name_for_path = re.sub(r'[<>:"/\\|?*]', "_", model_name).strip().rstrip(".")
    temperature = args.temperature
    max_tokens = args.max_tokens
    num_agents = args.num_agents
    assert num_agents % 5 == 0, "Number of agents must be a multiple of 5!"
    num_steps = args.num_steps
    prompt_template_root = "prompts/opinion_dynamics"
    path_result = "results/opinion_dynamics/{}/{}".format(experiment_id, model_name_for_path)
    if args.test_run:
        path_result = "results/opinion_dynamics/{}/{}/test_runs".format(experiment_id, model_name_for_path)
    date_version = "20231212"

    global rating_flag
    # rating_flag is used throughout the Agent class to switch parsing/format rules.
    # It is TRUE when running a no-rating prompt, and FALSE when ratings are expected.
    rating_flag = bool(args.no_rating)

    LIST_OPINION_SPACE = [-2, -1, 0, 1, 2]

    main(
        num_agents=num_agents,
        num_steps=num_steps,
        experiment_id=experiment_id, 
        model_name=model_name, 
        temperature=temperature,  
        max_tokens=max_tokens, 
        opinion_space=LIST_OPINION_SPACE,
        prompt_template_root=prompt_template_root,
        date_version=date_version,
        path_result=path_result,
    )
 
