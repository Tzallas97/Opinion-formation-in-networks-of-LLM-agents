#!/usr/bin/env python3
"""Event injection: a population-level shock delivered at one step.

This module is the PURE part - no LLM, no simulator objects - so it can be
unit-tested on its own. The simulator supplies two callables:

  react_fn(agent, prompt_text) -> str   run the model for one agent
  store_fn(agent, reaction_text, pinned) store the reaction in the agent's memory

Keeping those injected means the shape of the mechanism (who is asked, what is
logged, what persists) is testable without a running model, exactly as the
network rewiring was made testable apart from the simulator.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

__all__ = [
    "DEFAULT_EVENT_TEMPLATE", "load_event_template", "build_event_prompt",
    "event_relevance", "should_pin", "broadcast_event", "EVENT_LOG_COLUMNS",
]

# A minimal, bias-agnostic reaction prompt. It asks for a view on the EVENT and
# explicitly does not ask the agent to re-rate the main claim - that separation
# is the whole point. One shared default; a file can override it.
DEFAULT_EVENT_TEMPLATE = """\
Something just happened that everyone is talking about.

EVENT: {EVENT_TEXT}

You are {AGENT_NAME}. React in your own voice: what do you make of this event, and
how does it sit with you? Speak only about the event. Do not give it a rating and
do not restate your position on anything else.
"""

EVENT_LOG_COLUMNS = ["step", "agent_idx", "agent_name", "reaction", "relevance", "persist"]


def load_event_template(template_root: Optional[str]) -> str:
    """Return a shared event_react.md if present, else the built-in default.

    A run never fails for a missing template - the default is always valid.
    """
    if template_root:
        path = os.path.join(template_root, "event_react.md")
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                return text
        except (OSError, UnicodeDecodeError):
            pass
    return DEFAULT_EVENT_TEMPLATE


def build_event_prompt(template: str, agent_name: str, event_text: str,
                       current_belief: Any = None) -> str:
    """Fill the event template. Unknown placeholders are left intact rather than
    raising, so a hand-edited template cannot crash a paid run."""
    fields = {
        "EVENT_TEXT": str(event_text or "").strip(),
        "AGENT_NAME": str(agent_name or "").strip(),
        "CURRENT_BELIEF": "" if current_belief is None else str(current_belief),
    }

    class _Safe(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    try:
        return str(template).format_map(_Safe(fields))
    except (ValueError, IndexError):
        # a stray brace in the template - fall back to plain replacement
        out = str(template)
        for key, value in fields.items():
            out = out.replace("{" + key + "}", value)
        return out


def event_relevance(event_text: str, claim_text: str,
                    embed_fn: Optional[Callable[[list], list]] = None) -> Optional[float]:
    """Cosine between the event and the claim in embedding space, in [-1, 1].

    Relevance is a MEASURED continuous variable, not a relevant/irrelevant switch:
    it turns "seemingly unrelated" into a point on an axis and yields a dose-
    response curve instead of two bars. `embed_fn` takes a list of texts
    and returns a list of vectors (e.g. semantic_analysis.ollama_embed bound to a
    model+cache). Returns None when no embedder is supplied, so a run without one
    still works - it just cannot report relevance.
    """
    ev = str(event_text or "").strip()
    cl = str(claim_text or "").strip()
    if not ev or not cl or embed_fn is None:
        return None
    try:
        vectors = embed_fn([ev, cl])
        a, b = vectors[0], vectors[1]
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return None
        return float(dot / (na * nb))
    except Exception:
        return None


def should_pin(persist: str) -> bool:
    """Whether the event stays in memory permanently (step-change) rather than
    ageing out (impulse). 'headline' and 'both' pin; 'reaction' does not."""
    return str(persist or "reaction").strip().lower() in ("headline", "both")


def broadcast_event(
    agents: Iterable[Any],
    event_text: str,
    step: int,
    react_fn: Callable[[Any, str], str],
    store_fn: Optional[Callable[[Any, str, bool], None]] = None,
    *,
    template: Optional[str] = None,
    persist: str = "reaction",
    relevance: Optional[float] = None,
    name_of: Callable[[Any], str] = lambda a: getattr(a, "agent_name", ""),
    belief_of: Callable[[Any], Any] = lambda a: getattr(a, "current_belief", None),
) -> list[dict]:
    """Ask every agent, once, for its reaction to the event; store each reaction.

    Broadcast (every agent asked directly) is deliberate: if the event spread only
    through normal interaction its reach would be capped by the interaction
    topology, which is itself under study. Returns one event-log row per agent.
    """
    tmpl = template if template is not None else DEFAULT_EVENT_TEMPLATE
    pin = should_pin(persist)
    rows = []
    for idx, agent in enumerate(agents):
        name = name_of(agent)
        prompt = build_event_prompt(tmpl, name, event_text, belief_of(agent))
        try:
            reaction = str(react_fn(agent, prompt) or "").strip()
        except Exception as exc:  # one agent's failure must not abort the shock
            reaction = ""
            print(f"[warn] event reaction failed for {name}: {exc}")
        if store_fn is not None and reaction:
            stored = reaction if persist.lower() != "headline" else str(event_text).strip()
            if persist.lower() == "both":
                stored = f"{str(event_text).strip()} - reaction: {reaction}"
            try:
                store_fn(agent, stored, pin)
            except Exception as exc:
                print(f"[warn] could not store event memory for {name}: {exc}")
        rows.append({
            "step": int(step),
            "agent_idx": int(idx),
            "agent_name": name,
            "reaction": reaction,
            "relevance": "" if relevance is None else round(float(relevance), 4),
            "persist": str(persist).strip().lower(),
        })
    return rows
