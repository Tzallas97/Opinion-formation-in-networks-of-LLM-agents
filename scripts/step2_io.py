"""Step2 input/output helpers for ADR-006 §Component 2 (silence-as-choice).

Two responsibilities live here so the main simulator can call one small module:

1) ``render_step2_template(raw_text, allow_silence)`` — substitutes the
   ``<!--ALLOW_SILENCE_BLOCK-->`` marker in a step2 template with a
   mechanism-agnostic silence-option block (when ``allow_silence=True``) or
   strips the marker cleanly (when ``allow_silence=False``, the byte-identical
   baseline). Placed here rather than in ``create_prompts.py`` because that
   script is a build-time prompt-folder generator, not a runtime module.

2) ``parse_step2_output(raw)`` — classifies raw LLM output into one of six
   kinds: ``silent`` (explicit ``<silent>`` token, the agent's choice),
   ``refusal_error`` / ``refusal_canned`` / ``refusal_generated`` (the three
   Noels 2025 hard-censorship subtypes, arXiv:2504.03803 §Fig. 1), ``tweet``
   (well-formed FINAL_RATING + TWEET output), or ``parse_fail`` (empty or
   unrecognised). The distinction between ``silent`` and the three refusal
   kinds is what lets the P12 paper separate communicative silence
   (agent-chosen) from RLHF-driven refusal, the confound the owner flagged.

Both functions are pure and testable in isolation. The main simulator wires
them into the step2 pipeline in Task 2.3.
"""
from __future__ import annotations

import re

# The marker that Task 2.1 inserted into the three step2 templates. Kept as a
# module-level constant so tests can import it and never hard-code the string
# in more than one place.
SILENCE_BLOCK_MARKER = "<!--ALLOW_SILENCE_BLOCK-->"

# The silence-option instruction that gets substituted when --allow_silence is
# on. Intentionally MECHANISM-AGNOSTIC: it must not steer the agent toward any
# specific reason to be silent (extremity, topic, majority, etc.), because that
# would pre-decide which of the four candidate mechanisms produces silence and
# defeat the disentangling analysis. Task 2.4's tools/silence_mechanism.py
# attributes silence to a mechanism post-hoc; the prompt must stay neutral.
SILENCE_BLOCK_TEXT = (
    "You may also choose to remain quiet this round instead of writing a tweet. "
    "If that is what you want to do, return exactly `<silent>` as your entire "
    "response, on its own line, with nothing else. This is a normal option, not "
    "a refusal; do not add any explanation."
)


def render_step2_template(raw_text: str, allow_silence: bool) -> str:
    """Substitute the silence marker in a step2 template.

    - ``allow_silence=False`` (default in the sim) collapses the marker and its
      surrounding blank lines so the rendered text is byte-identical with the
      pre-ADR-006 template.
    - ``allow_silence=True`` replaces the marker with the mechanism-agnostic
      instruction block above.

    The two-phase strip on the off-branch is defensive: the preferred pattern
    (``\\n\\n<marker>\\n\\n``) matches the exact insertion Task 2.1 used; the
    fallback strips a bare marker in case a future template puts the marker
    elsewhere.
    """
    if allow_silence:
        return raw_text.replace(SILENCE_BLOCK_MARKER, SILENCE_BLOCK_TEXT)
    # off: strip the marker together with one pair of surrounding blank lines
    # so the pre-ADR-006 spacing is preserved verbatim.
    out = raw_text.replace(f"\n\n{SILENCE_BLOCK_MARKER}\n\n", "\n\n")
    # Fallback: bare marker with different surrounding whitespace.
    out = out.replace(SILENCE_BLOCK_MARKER, "")
    return out


# --------------------------------------------------------------------------- #
# Parser: Noels 2025 arXiv:2504.03803 §Fig. 1 (hard censorship taxonomy) +    #
# explicit <silent> token + normal tweet + parse_fail fallback.               #
# --------------------------------------------------------------------------- #

# Structured error tokens the LLM runtime may inject when a safety filter
# fires. Case-insensitive match anywhere in the output.
_REFUSAL_ERROR_RE = re.compile(
    r"\[\s*(safety[\s_-]*filter|content[\s_-]*filter|"
    r"safety[\s_-]*violation|blocked|error)",
    flags=re.IGNORECASE,
)

# Self-referential prefixes the model itself generates when declining to answer.
# Anchored at start-of-string (after strip) so a normal tweet that mentions
# "as an AI" mid-sentence does NOT trip this branch. Priority order in
# ``parse_step2_output`` ensures FINAL_RATING/TWEET marker output wins anyway.
_REFUSAL_GENERATED_RE = re.compile(
    r"^\s*("
    r"as an ai|as a language model|as a large language model|"
    r"i(?:'m| am) (?:not able|unable|sorry)|"
    r"i (?:cannot|can't|do not|don't) "
    r"(?:express|share|provide|give|comment|help|discuss)"
    # NB: "have" was intentionally dropped from the verb list.
    # "I don't have information on X" is a canned deflection (missing-info
    # phrasing), not a self-referential AI-capability refusal - see the
    # _REFUSAL_CANNED_RES below.
    r")",
    flags=re.IGNORECASE,
)

# A small library of canned deflection phrases. Extended as the pilot runs of
# Task 2.5+ surface new patterns; the whole list is regex, one per family.
_REFUSAL_CANNED_RES = [
    re.compile(r"many sites on the internet", re.IGNORECASE),
    re.compile(r"i (?:don't|do not) have (?:information|details|data) (?:on|about)",
               re.IGNORECASE),
    re.compile(r"consult (?:a|an) (?:professional|expert|authority)", re.IGNORECASE),
    re.compile(r"is a sensitive topic", re.IGNORECASE),
]


def parse_step2_output(raw: str) -> tuple[str, str]:
    """Classify a step2 LLM output into one of six kinds.

    Returns ``(kind, payload)`` where ``payload`` is the extracted tweet text
    when ``kind == "tweet"`` and the empty string in every other case. The
    dispatcher in the main simulator (wired in Task 2.3) uses ``kind`` to
    decide whether to continue the interaction (tweet) or to record the event
    in the appropriate counter (silent / refusal_* / parse_fail).

    Priority order matters:
      1. empty/whitespace-only -> parse_fail (nothing to inspect)
      2. exact ``<silent>`` (case-insensitive, trimmed) -> silent (agent choice)
      3. contains BOTH ``FINAL_RATING`` and ``TWEET`` markers -> tweet
         (real content wins over refusal prefixes that appear mid-sentence)
      4. structured error token anywhere -> refusal_error
      5. self-referential prefix at start -> refusal_generated
      6. matches a canned-deflection pattern -> refusal_canned
      7. otherwise -> parse_fail
    """
    text = (raw or "").strip()
    if not text:
        return ("parse_fail", "")

    if text.lower() == "<silent>":
        return ("silent", "")

    if "FINAL_RATING" in text.upper() and "TWEET" in text.upper():
        m = re.search(r"TWEET\s*:\s*(.+)$", text, flags=re.DOTALL | re.IGNORECASE)
        payload = m.group(1).strip() if m else ""
        return ("tweet", payload)

    if _REFUSAL_ERROR_RE.search(text):
        return ("refusal_error", "")

    if _REFUSAL_GENERATED_RE.match(text):
        return ("refusal_generated", "")

    for pattern in _REFUSAL_CANNED_RES:
        if pattern.search(text):
            return ("refusal_canned", "")

    return ("parse_fail", "")
