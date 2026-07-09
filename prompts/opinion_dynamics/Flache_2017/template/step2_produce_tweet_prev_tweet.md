Now, {AGENT_NAME}, after you wrote your {TWEET_WRITTEN_COUNT}{SUPERSCRIPT} tweet, please write another tweet write another tweet about this claim: {THEORY_STATEMENT}.

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
- At +2 or +1: tweet must LEAN FOR the claim. Start with a reason TO believe it. Any caveat must be shorter than the main point.
- At -2 or -1: tweet must LEAN AGAINST. Start with a reason to doubt. Any caveat must be shorter than the main point.
- At 0: genuinely mixed, both sides about equal weight. Neither side should dominate.

Format
- FINAL_RATING must equal {CURRENT_BELIEF}.
- No links. 0-2 hashtags max.
- Output exactly 2 lines and nothing else.
- Line 2 must contain real tweet text after "TWEET:", not headers or fragments.

Output:
FINAL_RATING: {CURRENT_BELIEF}
TWEET: