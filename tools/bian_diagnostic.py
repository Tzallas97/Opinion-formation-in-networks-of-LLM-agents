#!/usr/bin/env python3
"""ADR-006 Component 4: Bian 5-dim diagnostic probe.

Runs five short probes against any Ollama model to score the systematic biases
that Bian et al. 2025 identified in LLM social simulations (social-role over-
representation, response homogeneity, primacy, positivity/"utopian" skew). The
tool is standalone: it writes ``bian_scores.json`` and is reused as config
self-doc inside ``run_metrics.json`` when a run opts in (Task 4.2).

The five probes (higher = more biased on every axis):
  1. social_role_kl      -- KL(model_roles || ILO-2023 baseline) over the ten
                            ISCO-08 major groups. High = the model over-represents
                            a narrow slice of occupations (typically professionals).
  2. inter_agent_sim     -- mean pairwise TF-IDF cosine across different agents'
                            utterances in a shared discussion. High = homogeneous
                            population (agents sound alike).
  3. intra_agent_sim     -- mean cosine of one agent's successive utterances.
                            High = repetitive (an agent keeps saying the same thing).
  4. keyword_persistence -- fraction of later utterances still carrying the first
                            utterance's salient keyword. Proxy for primacy.
  5. positivity          -- (positive - negative) / (positive + negative) sentiment
                            words over free generations, in [-1, 1]. High = utopian.

Similarity uses the same transparent TF-IDF + cosine as
``tools/semantic_analysis.py`` (idf = log((1+n)/(1+df)) + 1, so identical texts
score cosine 1.0). Kept inline so the diagnostic stays self-contained and
offline-testable; swappable for nomic-embed later as a documented condition.

Reference: Bian, N. et al. (2025). Social Simulations with Large Language Model
Risk Utopian Illusion. arXiv:2510.21180 (``papers/Social Simulations with Large
Language Model Risk Utopian Illusion - 2510.21180v1.pdf``).

CLI:  python tools/bian_diagnostic.py --model qwen3:8b --out bian_scores_qwen3.json
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")

# --------------------------------------------------------------------------- #
# Transparent TF-IDF + cosine (mirrors tools/semantic_analysis.py).           #
# idf carries a +1 floor, so a term shared by every compared text keeps a      #
# non-zero weight and identical texts score cosine 1.0 -- the property the     #
# homogeneity probes rely on.                                                  #
# --------------------------------------------------------------------------- #

def _tokens(text):
    return [w.lower() for w in _WORD.findall(text or "")]


def build_tfidf(all_texts):
    """Fit IDF over the union of the compared texts so they share one space."""
    df = Counter()
    toks = [_tokens(t) for t in all_texts]
    for tk in toks:
        df.update(set(tk))
    n = max(1, len(all_texts))
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    def vec(text):
        tf = Counter(_tokens(text))
        total = sum(tf.values()) or 1
        return {w: (c / total) * idf.get(w, 0.0) for w, c in tf.items()}
    return vec


def cosine(a, b):
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _mean_pairwise_cosine(texts):
    texts = [t for t in texts if (t or "").strip()]
    if len(texts) < 2:
        return 0.0
    vec = build_tfidf(texts)
    vs = [vec(t) for t in texts]
    total, cnt = 0.0, 0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            total += cosine(vs[i], vs[j])
            cnt += 1
    return total / cnt if cnt else 0.0


def _mean_successive_cosine(texts):
    texts = [t for t in texts if (t or "").strip()]
    if len(texts) < 2:
        return 0.0
    vec = build_tfidf(texts)
    vs = [vec(t) for t in texts]
    sims = [cosine(vs[i], vs[i + 1]) for i in range(len(vs) - 1)]
    return sum(sims) / len(sims) if sims else 0.0


# --------------------------------------------------------------------------- #
# ISCO-08 major groups + approximate ILO global employment baseline.          #
# Shares are rounded ILO ILOSTAT-style figures, used only as a KL reference    #
# (the diagnostic is about the *shape* of the divergence, not exact rates).    #
# --------------------------------------------------------------------------- #

ISCO08_LABELS = [
    "armed_forces", "managers", "professionals", "technicians", "clerical",
    "service_sales", "skilled_agricultural", "craft_trades", "plant_machine",
    "elementary",
]

ILO_BASELINE = {
    "armed_forces": 0.005, "managers": 0.06, "professionals": 0.11,
    "technicians": 0.09, "clerical": 0.08, "service_sales": 0.18,
    "skilled_agricultural": 0.16, "craft_trades": 0.12, "plant_machine": 0.075,
    "elementary": 0.12,
}

# Keyword -> ISCO major group. Checked in this order; first substring hit wins
# (more specific / higher-skill groups first to resolve overlaps sensibly).
_ROLE_KEYWORDS = [
    ("managers", ["manager", "director", "ceo", "executive", "supervisor",
                  "administrator", "president", "founder", "entrepreneur",
                  "head of"]),
    ("professionals", ["engineer", "scientist", "doctor", "physician", "lawyer",
                       "attorney", "professor", "teacher", "nurse", "architect",
                       "accountant", "developer", "programmer", "analyst",
                       "researcher", "pharmacist", "psychologist", "economist",
                       "journalist", "designer", "dentist", "consultant"]),
    ("technicians", ["technician", "paramedic", "draftsman",
                     "associate professional", "it support"]),
    ("clerical", ["clerk", "secretary", "receptionist", "typist", "bookkeeper",
                  "administrative assistant", "data entry"]),
    ("service_sales", ["salesperson", "sales", "cashier", "waiter", "waitress",
                       "chef", "cook", "barista", "retail", "shop assistant",
                       "hairdresser", "police officer", "security guard",
                       "flight attendant", "caregiver", "customer service"]),
    ("skilled_agricultural", ["farmer", "fisher", "rancher", "forester",
                              "agricultural worker"]),
    ("craft_trades", ["electrician", "plumber", "carpenter", "mechanic",
                      "welder", "mason", "blacksmith", "tailor", "machinist"]),
    ("plant_machine", ["driver", "operator", "assembler", "factory worker",
                       "machine operator"]),
    ("elementary", ["cleaner", "labourer", "laborer", "janitor", "porter",
                    "domestic worker", "garbage collector"]),
    ("armed_forces", ["soldier", "military", "armed forces", "marine"]),
]

_POSITIVE = {
    "good", "great", "excellent", "happy", "love", "wonderful", "positive",
    "hope", "hopeful", "benefit", "beneficial", "trust", "safe", "safety",
    "success", "successful", "improve", "better", "best", "agree", "support",
    "optimistic", "confident", "strong", "healthy", "progress", "opportunity",
    "fair", "kind", "helpful", "effective", "reliable", "peaceful", "bright",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "hate", "fear", "worry", "danger", "dangerous",
    "risk", "risky", "harm", "harmful", "negative", "distrust", "fail",
    "failure", "worse", "worst", "disagree", "oppose", "pessimistic", "weak",
    "sick", "threat", "problem", "crisis", "concern", "doubt", "unfair",
    "corrupt", "violence", "unsafe", "useless", "broken", "bleak",
}
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "was",
    "for", "with", "as", "at", "by", "it", "this", "that", "you", "your",
    "their", "am", "be", "been", "has", "have", "had", "do", "does", "not",
    "no", "but", "so", "if", "who", "what", "one", "sentence", "opinion",
}

# Distinct persona seeds; a well-behaved (diverse) model diverges across them,
# a homogeneous one converges -> the similarity probes read that difference.
_PERSONA_SEEDS = [
    "a retired schoolteacher", "a young climate activist", "a small-business owner",
    "a factory worker", "a university scientist", "a stay-at-home parent",
    "a military veteran", "a hospital nurse", "a family farmer",
    "a software developer", "a religious community leader", "a professional athlete",
]

_ROLE_PROMPT = ("Think of one ordinary working adult chosen at random from the "
                "real-world population. Reply with ONLY their real occupation, in "
                "one to three words. No name, no sentence, no fictional or fantasy "
                "jobs.")


# --------------------------------------------------------------------------- #
# LLM caller (real path). Tests inject a fake chat_fn and never reach this.    #
# --------------------------------------------------------------------------- #

def _ollama_generate(prompt, model, url=None, timeout=120, think=None):
    payload = {"model": model, "prompt": prompt, "stream": False}
    if think is not None:
        payload["think"] = bool(think)  # reasoning models (qwen3) honour this
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request((url or OLLAMA_URL) + "/api/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #

def _classify_role(text):
    """Map a free-text occupation to an ISCO-08 major group label, or None."""
    t = (text or "").lower()
    for label, kws in _ROLE_KEYWORDS:
        for kw in kws:
            if kw in t:
                return label
    return None


def _kl_divergence(p, q):
    """KL(p || q) over shared keys; q is assumed strictly positive (ILO baseline)."""
    kl = 0.0
    for k, pk in p.items():
        if pk > 0:
            qk = q.get(k, 1e-9) or 1e-9
            kl += pk * math.log(pk / qk)
    return kl


def _salient_keyword(text):
    """Most frequent content token, tie-broken by earliest appearance."""
    toks = [t for t in _tokens(text) if t not in _STOP and len(t) > 2]
    if not toks:
        return None
    freq = Counter(toks)
    return max(toks, key=lambda w: (freq[w], -toks.index(w)))


def _agent_sequence(chat_fn, seed_idx, n_turns, model):
    persona = _PERSONA_SEEDS[seed_idx % len(_PERSONA_SEEDS)]
    out = []
    for t in range(n_turns):
        if t == 0:
            p = (f"You are {persona}. In one sentence, state your opinion on a "
                 f"current public issue.")
        else:
            p = (f"You are {persona}. Continue the discussion with one more "
                 f"sentence adding a new point.")
        out.append(str(chat_fn(p, model=model) or "").strip())
    return out


# --------------------------------------------------------------------------- #
# The five probes                                                            #
# --------------------------------------------------------------------------- #

def probe_social_roles(chat_fn, n=100, model=None):
    """KL divergence of the model's generated occupation mix from the ILO baseline."""
    counts = Counter()
    for _ in range(n):
        label = _classify_role(str(chat_fn(_ROLE_PROMPT, model=model) or ""))
        if label is not None:
            counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    alpha = 0.5  # Laplace smoothing so every group has positive mass for KL
    denom = total + alpha * len(ISCO08_LABELS)
    p = {label: (counts[label] + alpha) / denom for label in ISCO08_LABELS}
    return _kl_divergence(p, ILO_BASELINE)


def probe_inter_agent_sim(chat_fn, n_dialogues=20, n_agents=6, model=None):
    """Mean pairwise cosine across different agents' utterances (homogeneity)."""
    vals = []
    for d in range(n_dialogues):
        utts = []
        for a in range(n_agents):
            persona = _PERSONA_SEEDS[(d + a) % len(_PERSONA_SEEDS)]
            p = (f"You are {persona}. In one sentence, share your opinion on a "
                 f"current public issue.")
            utts.append(str(chat_fn(p, model=model) or "").strip())
        vals.append(_mean_pairwise_cosine(utts))
    return sum(vals) / len(vals) if vals else 0.0


def probe_intra_agent_sim(chat_fn, n_dialogues=20, n_turns=5, model=None):
    """Mean cosine of one agent's successive utterances (repetitiveness)."""
    vals = []
    for d in range(n_dialogues):
        seq = _agent_sequence(chat_fn, d, n_turns, model)
        vals.append(_mean_successive_cosine(seq))
    return sum(vals) / len(vals) if vals else 0.0


def probe_keyword_persistence(chat_fn, n_dialogues=20, n_turns=5, model=None):
    """Fraction of later utterances still carrying the first utterance's keyword."""
    rates = []
    for d in range(n_dialogues):
        seq = _agent_sequence(chat_fn, d, n_turns, model)
        if len(seq) < 2:
            continue
        kw = _salient_keyword(seq[0])
        if not kw:
            continue
        later = seq[1:]
        hits = sum(1 for u in later if kw in _tokens(u))
        rates.append(hits / len(later))
    return sum(rates) / len(rates) if rates else 0.0


def probe_positivity(chat_fn, n=100, model=None):
    """(#positive - #negative) / (#positive + #negative) sentiment words in [-1, 1]."""
    pos = neg = 0
    for i in range(n):
        seed = _PERSONA_SEEDS[i % len(_PERSONA_SEEDS)]
        p = (f"You are {seed}. In one or two sentences, describe how you feel "
             f"about the future of society.")
        toks = _tokens(chat_fn(p, model=model))
        pos += sum(1 for t in toks if t in _POSITIVE)
        neg += sum(1 for t in toks if t in _NEGATIVE)
    total = pos + neg
    return (pos - neg) / total if total else 0.0


# --------------------------------------------------------------------------- #
# Orchestration                                                              #
# --------------------------------------------------------------------------- #

def run_all_probes(model, chat_fn=None, out_path=None, n_samples=100, url=None, think=None):
    """Run the five probes and (optionally) write bian_scores.json.

    ``chat_fn(prompt, model=None) -> str`` is injected by tests; when omitted the
    real Ollama ``/api/generate`` path is used. ``n_samples`` scales every probe
    (the ADR defaults are n=100 role prompts, 20 dialogues x 6 agents).
    """
    if chat_fn is None:
        def chat_fn(prompt, model=model):
            return _ollama_generate(prompt, model, url=url, think=think)

    n = max(1, int(n_samples))
    n_dlg = max(2, min(n, 20))
    n_ag = min(6, max(2, n))
    n_turns = min(6, max(2, n))

    scores = {
        "social_role_kl": float(probe_social_roles(chat_fn, n=n, model=model)),
        "inter_agent_sim": float(probe_inter_agent_sim(chat_fn, n_dialogues=n_dlg, n_agents=n_ag, model=model)),
        "intra_agent_sim": float(probe_intra_agent_sim(chat_fn, n_dialogues=n_dlg, n_turns=n_turns, model=model)),
        "keyword_persistence": float(probe_keyword_persistence(chat_fn, n_dialogues=n_dlg, n_turns=n_turns, model=model)),
        "positivity": float(probe_positivity(chat_fn, n=n, model=model)),
    }
    data = {
        "model": model,
        "think": think,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_samples": n,
        "scores": scores,
        "reference": "Bian et al. 2025, arXiv:2510.21180",
    }
    if out_path:
        parent = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Bian 5-dim diagnostic probe (ADR-006 Component 4).")
    ap.add_argument("--model", required=True, help="Ollama model tag, e.g. qwen3:8b")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: bian_scores_<model>.json)")
    ap.add_argument("--n_samples", type=int, default=100,
                    help="samples per probe (default 100; ADR spec)")
    ap.add_argument("--url", default=None,
                    help="Ollama base URL (default $OLLAMA_URL or localhost:11434)")
    ap.add_argument("--think", choices=["default", "on", "off"], default="default",
                    help="reasoning mode for models that support it (qwen3): default "
                         "= model default, on/off = force. Running BOTH on and off is "
                         "itself a diagnostic -- does chain-of-thought change the bias?")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if --out already exists (cache override)")
    args = ap.parse_args(argv)
    think = {"default": None, "on": True, "off": False}[args.think]

    out = args.out or ("bian_scores_"
                       + re.sub(r"[^A-Za-z0-9_.-]", "_", args.model) + ".json")
    if os.path.exists(out) and not args.force:
        print(f"[bian] cached scores at {out} (use --force to recompute)")
        data = json.load(open(out, encoding="utf-8"))
    else:
        data = run_all_probes(args.model, out_path=out,
                              n_samples=args.n_samples, url=args.url, think=think)
    print(json.dumps(data["scores"], indent=2))
    print(f"[bian] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
