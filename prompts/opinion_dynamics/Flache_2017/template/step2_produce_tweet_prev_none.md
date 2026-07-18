Now, {AGENT_NAME}, please write a tweet write another tweet about this claim: {THEORY_STATEMENT}. The tweet should reflect your honest belief.

CLAIM: {THEORY_STATEMENT}
PRESENT VIEW: {CURRENT_BELIEF}

{AGENT_PERSONA}

{WORLD_RULES}

{FACT_PACK}

{STEP2_FACT_PACK_RULES}

{RAG_CONTEXT}

{STEP2_RAG_RULES}

Task
Write one short tweet (1-2 sentences) that reflects your PRESENT VIEW exactly.
Write it as a public tweet to other people, not a report about how the exchange affected you.

Direction and strength
- At +2: strongly FOR the claim. Lead with a confident reason to believe it; little or no hedging.
- At +1: mildly FOR the claim. Lean toward believing it, but sound tentative and allow that it may be wrong.
- At 0: genuinely mixed, both sides about equal weight. Neither side should dominate.
- At -1: mildly AGAINST the claim. Lean toward doubting it, but sound tentative and allow that it may be true.
- At -2: strongly AGAINST the claim. Lead with a confident reason to doubt; little or no hedging.

Format
- FINAL_RATING must equal {CURRENT_BELIEF}.
- No links. 0-2 hashtags max.
- Output exactly 2 lines and nothing else.
- Line 2 must contain real tweet text after "TWEET:", not headers or fragments.

Output:
FINAL_RATING: {CURRENT_BELIEF}
TWEET: