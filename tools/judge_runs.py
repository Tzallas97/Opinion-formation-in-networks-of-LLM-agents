#!/usr/bin/env python3
"""LLM-as-judge (read-only) for finished opinion-dynamics runs.

Scores the TEXT quality of a run's outputs - things the CSV numbers cannot see:
  tweets    : stance_match (does wording intensity fit the rating +-1 vs +-2),
              closed_world_leak (invokes outside facts/sources not shown),
              quality (coherent, on-topic, self-contained)
  responses : direction_support (does the explanation support the move/hold),
              closed_world_leak, quality

Design principles:
  - NO chain-of-thought: the judge answers with a bare JSON verdict. Asking a
    judge to reason aloud before scoring is known to inflate false acceptances
    on hard cases, so verdicts are collected without any reasoning step.
  - cross-family judge: default judge model is llama3.1:8b so qwen runs are not
    graded by their own family (self-preference bias).
  - bounded cost: a seeded random sample of items per run (--sample).
  - calibration: --export-calibration writes a small CSV for human labels.

Read-only: never touches run data; writes judge_scores.csv / judge_summary.json
into the report folder (or --out).

Context inheritance: with --context light/full (default light) the judge is
given the run's OWN ground truth, auto-extracted from the files:
  claim            from the version's step3 template ("CLAIM: ..." line)
  config line      world / RAG mode / bias version / model (interactions CSV)
  author persona   from the version's list_agent_descriptions.csv (card + traits)
  world rules      the EXACT closed-world wording (EXTERNAL_WORLD_RULES in the
                   main script, matched by the World column)      [full only]
  bias rule        the exact "Bias rule" block from the template  [full only]
No new simulation is run - judging happens on the files, with their own context.

Usage (Ollama must be running):
  python tools/judge_runs.py RUN_DIR [RUN_DIR ...] [--model llama3.1:8b]
         [--sample 120] [--context light|full|off] [--out DIR] [--dry-run]
         [--export-calibration N]
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, random, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_runs  # reuse spec handling / file discovery

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

TWEET_RUBRIC = """You are grading ONE short social-media post from a simulation.
The author holds an opinion rating {rating} on the claim: "{claim}".
Scale: -2 strongly against, -1 mildly against, 0 mixed/unsure, +1 mildly for, +2 strongly for.
{context}Closed-world leak definition: citing outside facts, statistics, named studies, links,
or real-world events NOT shown to the author counts as a leak.

POST: "{text}"

Answer ONLY with compact JSON, no other text, no reasoning steps:
{{"stance_match": <1-5, does the wording intensity and direction fit rating {rating}>,
"closed_world_leak": <0 or 1>,{persona_dim}
"quality": <1-5, coherent, on-topic, reads like a real post>}}"""

RESPONSE_RUBRIC = """You are grading ONE listener reaction from a simulation.
The listener read a post and moved their opinion from {pre} to {post}
(scale -2 strongly against ... +2 strongly for) on the claim: "{claim}".
{context}Closed-world leak definition: citing outside facts, statistics, named studies,
or real-world events NOT shown to the listener counts as a leak.

POST THEY READ: "{tweet}"
THEIR EXPLANATION: "{text}"

Answer ONLY with compact JSON, no other text, no reasoning steps:
{{"direction_support": <1-5, does the explanation actually support moving {pre} -> {post} (or holding)>,
"closed_world_leak": <0 or 1>,{persona_dim}
"quality": <1-5, coherent, complete sentence(s), addresses the post>}}"""

DIMS = ["stance_match", "direction_support", "closed_world_leak", "quality", "persona_consistency"]

RATIONALE_RUBRIC = """A previous evaluation FLAGGED the text below as a closed-world leak:
it invokes outside facts, sources, statistics, or real-world events that were never shown
to the author. That verdict is FINAL - do not re-score it, do not argue with it.
{context}TEXT: "{text}"

In one or two short sentences, name EXACTLY which outside material is invoked
(the specific fact, source, statistic, or named event). Plain text only, no JSON."""


def _clean(t):
    import re
    t = str(t or "").strip()
    t = re.sub(r'^FINAL_RATING:\s*-?\d+\s*(?:\n|\\n|\s)*(?:TWEET:|EXPLANATION:)\s*', '', t, flags=re.I)
    return re.sub(r'\s+', ' ', t).strip()


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _extract_world_rules(world, script_path=None):
    """Pull the EXACT world-rules wording the agents saw, straight from the main
    script source: EXTERNAL_WORLD_RULES["<world>"] = CONST / dict literal."""
    import re
    script_path = script_path or os.path.join(REPO_ROOT, "scripts", "opinion_dynamics_test_network_qwen.py")
    try:
        src = open(script_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    w = str(world or "").strip().lower()
    if not w:
        return ""
    m = re.search(r"EXTERNAL_WORLD_RULES\[[\"']%s[\"']\]\s*=\s*([A-Z_][A-Z0-9_]*)" % re.escape(w), src)
    if m:
        c = re.search(r'%s\s*=\s*"""(.*?)"""' % m.group(1), src, re.S)
        if c: return c.group(1).strip()
    m = re.search(r"[\"']%s[\"']\s*:\s*\"\"\"(.*?)\"\"\"" % re.escape(w), src, re.S)  # dict literal form
    return m.group(1).strip() if m else ""

def build_run_context(run, prompts_root=None):
    """Auto-extract the run's own ground truth (claim, exact wording, personas)."""
    prompts_root = prompts_root or os.path.join(REPO_ROOT, "prompts")
    meta = run.get("meta", {})
    version = str(meta.get("version", "") or "")
    ctx = {"claim": "", "bias_rule": "", "world_rules": "", "personas": {}, "config_line": "", "version_dir": ""}
    ctx["config_line"] = ", ".join(f"{k}={meta.get(k)}" for k in ("world", "rag", "factpack", "version", "model") if meta.get(k))
    if version:
        hits = glob.glob(os.path.join(prompts_root, "**", version), recursive=True)
        vdir = next((h for h in sorted(hits) if os.path.isdir(h)), None)
        if vdir:
            ctx["version_dir"] = vdir
            tpl = None
            for pat in ("step3_receive_tweet_prev_read.md", "step3*.md", "step2*.md"):
                c = sorted(glob.glob(os.path.join(vdir, pat))) or sorted(glob.glob(os.path.join(vdir, "template", pat)))
                if c: tpl = c[0]; break
            if tpl:
                txt = open(tpl, encoding="utf-8", errors="replace").read()
                for line in txt.splitlines():
                    if line.strip().upper().startswith("CLAIM:"):
                        ctx["claim"] = line.split(":", 1)[1].strip(); break
                lines = txt.splitlines(); block = []
                for i, line in enumerate(lines):
                    if line.strip().lower() == "bias rule":
                        for l2 in lines[i:]:
                            if block and not l2.strip(): break
                            block.append(l2)
                        break
                ctx["bias_rule"] = "\n".join(block).strip()
            pcsv = os.path.join(vdir, "list_agent_descriptions.csv")
            if os.path.exists(pcsv):
                try:
                    trait_cols = ["epistemic_profile", "institutional_trust", "uncertainty_tolerance",
                                  "evidence_style", "official_narrative_suspicion", "openness_to_update"]
                    for row in csv.DictReader(open(pcsv, encoding="utf-8")):
                        name = (row.get("agent_name") or "").strip()
                        if not name: continue
                        traits = {k: row[k].strip() for k in trait_cols if row.get(k, "").strip()}
                        card = (row.get("persona") or "").strip()[:500]
                        ctx["personas"][name] = {"card": card, "traits": traits}
                except Exception:
                    pass
    ctx["world_rules"] = _extract_world_rules(meta.get("world"))
    return ctx

def make_context_block(ctx, item, mode):
    """Render the CONTEXT section injected into the rubric for one item."""
    if mode == "off" or not ctx:
        return "", ""
    lines = []
    if ctx.get("config_line"):
        lines.append("Run configuration: " + ctx["config_line"] + ".")
    p = ctx.get("personas", {}).get((item.get("agent") or "").strip())
    persona_dim = ""
    if p:
        bits = []
        if p["traits"]: bits.append("traits: " + ", ".join(f"{k}={v}" for k, v in p["traits"].items()))
        if p["card"]: bits.append(p["card"].replace("\n", " | "))
        lines.append("AUTHOR PERSONA (exactly as given to the agent): " + " ; ".join(bits))
        persona_dim = '\n"persona_consistency": <1-5, does the voice fit this persona>,'
    if mode == "full":
        if ctx.get("world_rules"):
            lines.append("WORLD RULES the agent was given (verbatim):\n" + ctx["world_rules"][:900])
        if ctx.get("bias_rule"):
            lines.append("BIAS RULE the agent was given (verbatim):\n" + ctx["bias_rule"][:700])
    if item.get("shown"):
        lines.append("MATERIAL EXPLICITLY SHOWN to them in THIS interaction (retrieved snippets - referring to THIS material is NOT a leak):\n" + str(item["shown"])[:900])
    return ("\n".join(lines) + "\n\n") if lines else "", persona_dim

def load_rag_shown(spec):
    """Map (time_step, prompt_step, agent_name) -> the EXACT retrieved snippets shown
    in that interaction, from the run's rag_retrieval_*.csv provenance log
    (written by the main script into <run>/_retry_debug/). Empty dict if absent."""
    run_dir = (os.path.dirname(spec) or ".") if os.path.isfile(spec) else spec
    hits = glob.glob(os.path.join(run_dir, "_retry_debug", "rag_retrieval_*.csv"))
    shown = {}
    if not hits:
        return shown
    try:
        for row in csv.DictReader(open(sorted(hits)[0], encoding="utf-8")):
            key = (str(row.get("time_step") or "").strip(),
                   str(row.get("prompt_step") or "").strip().lower(),
                   str(row.get("agent_name") or "").strip())
            txt = str(row.get("retrieved_texts") or "").strip()
            if txt:
                shown[key] = txt
    except Exception:
        return {}
    return shown


def collect_items(spec, claim):
    """Pull judgeable items out of one run's interactions CSV."""
    run = eval_runs.load_run(spec)
    ic = None
    # reuse the same file the loader used, via its prefix logic
    is_file = os.path.isfile(spec)
    run_dir = (os.path.dirname(spec) or ".") if is_file else spec
    prefix = None
    if is_file:
        b = os.path.basename(spec); cut = b.lower().find("opinion_change")
        prefix = b[:cut].rstrip("_-") if cut > 0 else None
    pats = ["*network_interactions*.csv", "*interactions*.csv"]
    for p in pats:
        hit = eval_runs._one(run_dir, (prefix + "*" + p.lstrip("*")) if prefix else p)
        if hit: ic = hit; break
    items = []
    if not ic:
        return run, items
    shown = load_rag_shown(spec)
    for i, row in enumerate(csv.DictReader(open(ic, encoding="utf-8"))):
        step = row.get("Time Step")
        tw = _clean(row.get("Agent_J Tweet"))
        if tw:
            items.append({"kind": "tweet", "idx": i, "step": step,
                          "agent": row.get("Agent_J Name"), "rating": row.get("Agent_J Belief"),
                          "text": tw, "claim": claim,
                          "shown": shown.get((step, "step2", (row.get("Agent_J Name") or "").strip()), "")})
        resp = _clean(row.get("Agent_I Response"))
        if resp:
            items.append({"kind": "response", "idx": i, "step": step,
                          "agent": row.get("Agent_I Name"),
                          "pre": row.get("Agent_I Pre-Belief"), "post": row.get("Agent_I Post-Belief"),
                          "tweet": tw, "text": resp, "claim": claim,
                          "shown": shown.get((step, "step3", (row.get("Agent_I Name") or "").strip()), "")})
    return run, items


def build_prompt(item, ctx=None, mode="light"):
    context, persona_dim = make_context_block(ctx, item, mode)
    if item["kind"] == "tweet":
        return TWEET_RUBRIC.format(rating=item["rating"], claim=item["claim"], text=item["text"],
                                   context=context, persona_dim=persona_dim)
    return RESPONSE_RUBRIC.format(pre=item["pre"], post=item["post"], claim=item["claim"],
                                  tweet=item["tweet"], text=item["text"],
                                  context=context, persona_dim=persona_dim)


def list_ollama_models(url=None, timeout=5):
    """Return locally installed Ollama model names (GET /api/tags). [] if unreachable."""
    try:
        req = urllib.request.Request((url or OLLAMA_URL) + "/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        return []


def _ollama_chat(model, prompt, url=None, timeout=120, json_format=True, num_predict=220):
    """Single non-streaming chat call, deterministic, no thinking.
    json_format=True adds grammar-constrained JSON output (the scoring pass);
    the rationale pass uses plain text (json_format=False, shorter budget)."""
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "num_predict": int(num_predict), "num_ctx": 8192}}
    if json_format:
        body["format"] = "json"
    req = urllib.request.Request((url or OLLAMA_URL) + "/api/chat",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message", {}) or {}).get("content", "")


def _flagged(verdict):
    """True when the locked verdict marks a closed-world leak (the flagged set)."""
    try:
        return int(verdict.get("closed_world_leak", 0)) == 1 and "parse_error" not in verdict
    except Exception:
        return False


def build_rationale_prompt(item, ctx=None, mode="light"):
    context, _ = make_context_block(ctx, item, mode)
    return RATIONALE_RUBRIC.format(context=context, text=item["text"])


def judge_items(items, model, cache_path, url=None, log=print, ctx=None, mode="light",
                explain_flagged=False):
    """Score items (with an on-disk cache so re-runs are free).

    Two-pass design: pass 1 scores everything no-CoT; pass 2
    runs ONLY for items already flagged as leaks, asking for a plain-text rationale.
    The score is locked before any rationale is generated, so the explanation cannot
    contaminate the verdict (the CoT-inflates-false-positives failure mode)."""
    cache = {}
    if os.path.exists(cache_path):
        try: cache = json.load(open(cache_path, encoding="utf-8"))
        except Exception: cache = {}
    out, called = [], 0
    for it in items:
        prompt = build_prompt(it, ctx=ctx, mode=mode)
        key = hashlib.sha256((model + "||" + prompt).encode("utf-8")).hexdigest()
        if key in cache:
            verdict = cache[key]
        else:
            raw = _ollama_chat(model, prompt, url=url)
            try:
                verdict = json.loads(raw)
            except Exception:
                verdict = {"parse_error": 1, "raw": raw[:200]}
            cache[key] = verdict
            called += 1
            if called % 20 == 0:
                log(f"  ... {called} new judge calls")
                json.dump(cache, open(cache_path, "w", encoding="utf-8"))
        rec = dict(it); rec.pop("claim", None)
        rec["judge"] = verdict
        out.append(rec)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"))

    if explain_flagged:
        flagged = [rec for rec in out if _flagged(rec.get("judge", {}))]
        if flagged:
            log(f"  pass 2: rationales for {len(flagged)} flagged item(s) (scores are locked)")
        for rec in flagged:
            rprompt = build_rationale_prompt(rec, ctx=ctx, mode=mode)
            rkey = hashlib.sha256((model + "||RATIONALE||" + rprompt).encode("utf-8")).hexdigest()
            if rkey in cache:
                rec["rationale"] = str(cache[rkey].get("_rationale", "")).strip()
                continue
            raw = _ollama_chat(model, rprompt, url=url, json_format=False, num_predict=120)
            rec["rationale"] = str(raw or "").strip()
            cache[rkey] = {"_rationale": rec["rationale"]}
            called += 1
        json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    return out, called


def summarize(scored):
    s = {"n_items": len(scored), "n_tweets": 0, "n_responses": 0, "parse_errors": 0}
    acc = {d: [] for d in DIMS}
    for rec in scored:
        v = rec.get("judge", {})
        if "parse_error" in v:
            s["parse_errors"] += 1; continue
        s["n_tweets" if rec["kind"] == "tweet" else "n_responses"] += 1
        for d in DIMS:
            if d in v:
                try: acc[d].append(float(v[d]))
                except Exception: pass
    for d, vals in acc.items():
        if vals:
            s[d + "_mean"] = round(sum(vals) / len(vals), 3)
            s[d + "_n"] = len(vals)
    s["rationales"] = sum(1 for rec in scored if str(rec.get("rationale", "")).strip())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="run folders (or loose opinion_change CSVs)")
    ap.add_argument("--model", default="llama3.1:8b",
                    help="judge model (use a DIFFERENT family than the graded runs)")
    ap.add_argument("--url", default=None, help="Ollama URL (default env OLLAMA_URL or localhost)")
    ap.add_argument("--claim", default="", help="claim text (default: auto-extracted from the version's template)")
    ap.add_argument("--context", choices=["off", "light", "full"], default="light",
                    help="how much run context the judge sees (light: config+persona; full: +world rules & bias verbatim)")
    ap.add_argument("--prompts-root", default=None, help="prompts folder (default: <repo>/prompts)")
    ap.add_argument("--sample", type=int, default=120, help="max items judged per run (seeded)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="output folder (default: judge_report next to first run)")
    ap.add_argument("--labels", default=None, help="comma-separated labels")
    ap.add_argument("--dry-run", action="store_true", help="build prompts and outputs, no LLM calls")
    ap.add_argument("--export-calibration", type=int, default=0, metavar="N",
                    help="also export N random items to calibration.csv for human labels")
    ap.add_argument("--explain-flagged", choices=["on", "off"], default="on",
                    help="two-pass rationale: AFTER scores are locked, ask the judge to explain "
                         "each closed-world-leak item in plain text (cheap - flagged items only; "
                         "cannot contaminate the scores)")
    a = ap.parse_args()

    specs = []
    for p in a.paths:
        p = p.rstrip("/\\")
        specs += [p] if (os.path.isfile(p) or eval_runs.is_run_dir(p)) else eval_runs.find_run_specs(p)
    if not specs:
        sys.exit("no runs found")
    labels = [x.strip() for x in a.labels.split(",")] if a.labels else [eval_runs.spec_name(s) for s in specs]
    claim = a.claim or "the claim under discussion in this simulation"
    out_dir = a.out or os.path.join(os.path.dirname(specs[0]) if os.path.isfile(specs[0]) else os.path.dirname(specs[0].rstrip("/\\")) or ".", "judge_report")
    os.makedirs(out_dir, exist_ok=True)

    rng = random.Random(a.seed)
    all_summaries = {}
    for spec, label in zip(specs, labels):
        run = eval_runs.load_run(spec)
        ctx = build_run_context(run, prompts_root=a.prompts_root) if a.context != "off" else {}
        run_claim = a.claim or (ctx.get("claim") if ctx else "") or claim
        if ctx:
            got = [k for k in ("claim", "world_rules", "bias_rule") if ctx.get(k)] + (["personas(%d)" % len(ctx["personas"])] if ctx.get("personas") else [])
            print(f"[{label}] context: {', '.join(got) or 'none found'}")
        run, items = collect_items(spec, run_claim)
        if not items:
            print(f"[{label}] no judgeable items, skipping"); continue
        if len(items) > a.sample:
            items = rng.sample(items, a.sample)
            items.sort(key=lambda x: (x["idx"], x["kind"]))
        print(f"[{label}] {len(items)} items to judge (model {a.model}{', DRY RUN' if a.dry_run else ''})")

        if a.export_calibration:
            calib = rng.sample(items, min(a.export_calibration, len(items)))
            cpath = os.path.join(out_dir, f"calibration_{label}.csv")
            with open(cpath, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["kind", "step", "agent", "rating_or_move", "text", "human_stance_or_support(1-5)", "human_leak(0/1)", "human_quality(1-5)"])
                for it in calib:
                    mv = it.get("rating") if it["kind"] == "tweet" else f"{it.get('pre')}->{it.get('post')}"
                    w.writerow([it["kind"], it["step"], it["agent"], mv, it["text"], "", "", ""])
            print(f"  calibration sheet -> {cpath}")

        if a.dry_run:
            scored = [dict(it, judge={"dry_run": 1}) for it in items]
            called = 0
            # save one full example prompt so you can inspect exactly what the judge will see
            if items:
                open(os.path.join(out_dir, f"example_prompt_{label}.txt"), "w", encoding="utf-8").write(
                    build_prompt(items[0], ctx=ctx, mode=a.context))
                if a.explain_flagged == "on":
                    open(os.path.join(out_dir, f"example_rationale_prompt_{label}.txt"), "w", encoding="utf-8").write(
                        build_rationale_prompt(items[0], ctx=ctx, mode=a.context))
        else:
            cache_path = os.path.join(out_dir, f"judge_cache_{label}.json")
            scored, called = judge_items(items, a.model, cache_path, url=a.url, ctx=ctx, mode=a.context,
                                         explain_flagged=(a.explain_flagged == "on"))

        spath = os.path.join(out_dir, f"judge_scores_{label}.csv")
        with open(spath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kind", "idx", "step", "agent", "rating", "pre", "post", "text",
                        "stance_match", "direction_support", "closed_world_leak", "quality",
                        "persona_consistency", "parse_error", "rationale"])
            for rec in scored:
                v = rec.get("judge", {})
                w.writerow([rec["kind"], rec["idx"], rec["step"], rec.get("agent", ""),
                            rec.get("rating", ""), rec.get("pre", ""), rec.get("post", ""), rec["text"],
                            v.get("stance_match", ""), v.get("direction_support", ""),
                            v.get("closed_world_leak", ""), v.get("quality", ""),
                            v.get("persona_consistency", ""), v.get("parse_error", ""),
                            rec.get("rationale", "")])
        summ = summarize(scored)
        summ.update({"label": label, "run": run["name"], "judge_model": a.model,
                     "sampled": len(items), "new_calls": called, "dry_run": bool(a.dry_run),
                     "context_mode": a.context, "claim_used": run_claim[:120]})
        all_summaries[label] = summ
        json.dump(summ, open(os.path.join(out_dir, f"judge_summary_{label}.json"), "w", encoding="utf-8"), indent=1)
        print(f"  scores -> {spath}")

    # cross-run comparison table
    if all_summaries:
        cmp_path = os.path.join(out_dir, "judge_comparison.md")
        L = ["# Judge comparison\n",
             f"Judge model: `{a.model}` (cross-family, no chain-of-thought, temperature 0, JSON-constrained).\n",
             "| condition | items | stance_match | direction_support | closed-world leak rate | quality | persona_consistency | parse errors |",
             "|---|---|---|---|---|---|---|---|"]
        for label, s in all_summaries.items():
            L.append(f"| **{label}** | {s.get('n_items')} | {s.get('stance_match_mean','-')} | {s.get('direction_support_mean','-')} | {s.get('closed_world_leak_mean','-')} | {s.get('quality_mean','-')} | {s.get('persona_consistency_mean','-')} | {s.get('parse_errors',0)} |")
        L.append("\nScores are 1-5 means (leak is a 0/1 rate). Anchor with the calibration sheets before quoting in the thesis.")
        L.append("Flagged (leak=1) items carry a plain-text rationale in judge_scores_<label>.csv - written AFTER the score locked, so it cannot bias the verdict.")
        open(cmp_path, "w", encoding="utf-8").write("\n".join(L))
        print("comparison ->", cmp_path)


if __name__ == "__main__":
    main()
