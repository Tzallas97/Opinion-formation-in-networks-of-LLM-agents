# prompts/opinion_dynamics/Flache_2017/content/world_rules.py

# prompts/opinion_dynamics/Flache_2017/content/world_rules.py

CLOSED_WORLD_RULES = """
Remember, throughout the interactions, you are alone in your room with limited access to the Internet.
You may not search for information about XYZ or ask other people about XYZ.
Base your belief only on your current rating, your persona, the claim wording, the tweet, and any material explicitly shown in the current prompt.
Do not rely on outside facts, public knowledge, or unseen evidence that is not explicitly shown in the current prompt.
If retrieved snippets, fact-pack entries, notes, or live-web results are shown in the prompt, they count as grounded shown material for this interaction.
Do NOT mention these restrictions in your output.
""".strip()

CLOSED_WORLD_RULES_STRICT = """
Remember, throughout the interactions, you are alone in your room with limited access to information.
You may not search for information about XYZ or ask other people about XYZ.
Use only information explicitly shown in the current prompt.
Do not answer a tweet with an unstated rebuttal or outside explanation unless that exact rebuttal is explicitly shown in the current prompt.
Do not rely on outside facts, public knowledge, source credibility, detectability, motive, feasibility, background capabilities, or prior achievements unless that material is explicitly shown in the current prompt.
If retrieved snippets, fact-pack entries, notes, or live-web results are shown in the prompt, they count as grounded shown material for this interaction.
Do NOT mention these restrictions in your output.
""".strip()


CLOSED_WORLD_RULES_STRICT_RAG = """
Remember, throughout the interactions, you are alone in your room with limited access to information.
You may not search for information about XYZ or ask other people about XYZ.
Use only information explicitly shown in the current prompt.
Do not answer a tweet with an unstated rebuttal or outside explanation unless that exact rebuttal is explicitly shown in the current prompt.
Do not rely on outside facts, public knowledge, source credibility, detectability, motive, feasibility, background capabilities, or prior achievements unless that material is explicitly shown in the current prompt.
If retrieved snippets, fact-pack entries, notes, or live-web results are shown in the prompt, they count as grounded shown material for this interaction.
Do NOT mention these restrictions in your output.
""".strip()


OPEN_WORLD_RULES_NO_RAG = """
Remember, throughout the interactions, you are alone in your room with full access to the Internet.
You may use broader recall, public knowledge, or online information about XYZ when it is genuinely relevant to the claim or to the tweet you are evaluating.
Do NOT mention artificial limitations like "I can't browse".
""".strip()

OPEN_WORLD_RULES_RAG = """
Remember, throughout the interactions, you are alone in your room with access to information through a retrieval system.
You may use retrieved information about XYZ only when it is explicitly provided in the current prompt.
Do not claim to retrieve or know evidence that is not explicitly present in the current prompt.
""".strip()

TRUE_OPEN_WORLD_RULES = """
Remember, throughout the interactions, you are not confined to your room and you have access to the Internet and the outside world.
You may use online information, public knowledge, broader real-world recall, and information you could reasonably get from the outside world when it is genuinely relevant to the claim or to the tweet you are evaluating.
Do NOT mention artificial limitations like "I can't browse".
""".strip()

WORLD_RULES = {
    "closed": CLOSED_WORLD_RULES,
    "closed_strict": CLOSED_WORLD_RULES_STRICT,
    "closed_strict_rag": CLOSED_WORLD_RULES_STRICT_RAG,
    "open_no_rag": OPEN_WORLD_RULES_NO_RAG,
    "open_rag": OPEN_WORLD_RULES_RAG,
    "true_open": TRUE_OPEN_WORLD_RULES,
    "open": OPEN_WORLD_RULES_NO_RAG,
}

WORLD_LABELS = {
    "closed": "CLOSED",
    "closed_strict": "CLOSED_STRICT",
    "closed_strict_rag": "CLOSED_STRICT_RAG",
    "open_no_rag": "OPEN_NO_RAG",
    "open_rag": "OPEN_RAG",
    "true_open": "TRUE_OPEN",
    "open": "OPEN_NO_RAG",
}
