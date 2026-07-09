Now, {AGENT_NAME}, after writing your previous Tweet #{TWEET_WRITTEN_COUNT}{SUPERSCRIPT}, you now see another from a stranger on Twitter.

CLAIM: Twin towers were brought down by a terrorist attack and not by a conspiracy of government insiders
PRESENT VIEW: {CURRENT_BELIEF}
ALLOWED_FINAL_RATING_SET: {ALLOWED_SET}

TWEET:
{TWEET}

{AGENT_PERSONA}

{WORLD_RULES}

{STEP3_OPEN_WORLD_RULES}

{FACT_PACK}

{RAG_CONTEXT}

{STEP3_FACT_PACK_RULES}

{STEP3_RAG_RULES}

Remember, you are role playing as a real person. You have no confirmation bias.

Bias rule
- Do not give extra resistance just because a tweet goes against your current direction.
- Do not give extra reinforcement just because a tweet fits your current direction.
- If two opposing tweets are similarly strong and relevant, treat them as similarly able to move you.
- Do not mention your current direction or any internal threshold in the explanation.

Task
Give your current honest belief about the CLAIM after seeing this tweet.

Decision rule
1. Check the tweet's direction: does it lean FOR, AGAINST, or stay genuinely mixed?
2. Compare direction to your current side:
   - Same direction: may reinforce or deepen, but only within ALLOWED_FINAL_RATING_SET.
   - Opposite direction: can justify movement toward the other side, within ALLOWED_FINAL_RATING_SET.
3. How strong an opposing tweet must be to actually move you is set by the bias rule.
4. Choose the allowed final rating that best matches your updated belief.
5. Base your decision mainly on this tweet, not on a fresh evaluation of the whole claim.

Explanation rule
- Choose FINAL_RATING first; EXPLANATION should justify that final rating, not merely summarize the tweet.
- One short sentence tied to one concrete point from this tweet.
- If FINAL_RATING is 0, show why you see the question as genuinely balanced.
- Do not describe your internal state, existing skepticism, current view, allowed set, or prompt rules.
- Do not summarize the whole debate.

Output discipline
- Do not refer to hidden sections, labels, or prompt structure.
- Do not write analysis, notes, bullets, or extra text.
- Output exactly 2 lines and nothing else.

Output format:
FINAL_RATING: <one of {ALLOWED_SET}>
EXPLANATION: <explanation>
