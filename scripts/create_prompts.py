"""Generate prompt-folder variants for the opinion-dynamics experiments.

This script keeps the topic/framing dictionaries and the reusable prompt templates in one place, then writes the folder structure consumed by the simulator. It is a setup utility: it creates prompt files and persona CSV placeholders, but it does not run simulations or alter result CSVs.
"""

import argparse
import re
import shutil
from pathlib import Path

# =====================================================
# 1. Topic dictionaries and prompt-generation constants
# =====================================================

DICT_TOPIC_NAME = {
    "v37": "flat_earth",
    "v38": "trex",
    "v39": "talk_death",
    "v40": "palm",
    "v41": "global_warming",
    "v52": "moon",
    "v53": "twin_towers",
    "v54": "nazi",
    "v55": "us_unemployment",
    "v56": "obama",
    "v57": "bicycle",
    "v58": "washington_dc",
    "v59": "brain",
    "v60": "fire",
    "v61": "sky",
    "v100": "Prespa_Agreement",
    "v101": "COVID19_vaccination",
    "v102": "police_in_greek_universities",
    "v103": "greek_economic_recovery_post_2008",
    "v104": "tourism_in_greece",
    "v105": "greek_financial_crisis_responsibility",
    "v106": "renewable_energy_in_greece",
    "v107": "greek_cultural_heritage_sites",
    "v108": "greek_agricultural_sector",
    "v109": "greek_education_system",
    "v110": "compulsory_military_service_in_greece",
    "v111": "religious_education_in_greek_public_schools",
    "v112": "public_healthcare_in_greece",
    "v113": "greece_referendum",
    "v114": "illuminati",
    "v115": "jews_control_finance",
    "v116": "world_history_hoax",
    "v117": "covid19_conspiracy",
    "v118": "titanic_conspiracy",
    "v119": "twin_towers_conspiracy",
    "v120": "education_conspiracy",
    "v121": "conspiracy_theories_in_schools",
    "v122": "earth_controlled_by_aliens",
    "v123": "mathematics_conspiracy",
    "v124": "religion_conspiracy",
    "v125": "climate_change_hoax",
    "v126": "social_media_conspiracy",
    "v127": "ancient_civilization_conspiracy",
    "v128": "antarctica_conspiracy", 
    "v129": "nazis_were_good_guys",
    "v130": "simulation_hypothesis",
    
}

# (topic_version, framing) -> theory sentence
DICT_TOPIC_FRAMING_STATEMENT = {
    ("v37", "TRUE"): "theory XYZ that claims that the earth is an irregularly shaped ellipsoid rather than flat",
    ("v37", "FALSE"): "the theory XYZ that claims that the Earth is flat",
    ("v38", "TRUE"): "theory XYZ that claims that the Tyrannosaurus Rex and humans did not co-exist on Earth at the same",
    ("v38", "FALSE"): "theory XYZ that claims that the Tyrannosaurus Rex and humans co-existed on Earth at the same time",
    ("v39", "TRUE"): "theory XYZ that claims that it is not possible for humans to communicate with the dead",
    ("v39", "FALSE"): "theory XYZ that claims that it is possible for humans to communicate with the dead",
    ("v40", "TRUE"): "theory XYZ that claims that it is not possible to predict someone’s future by looking at their palm characteristics",
    ("v40", "FALSE"): "theory XYZ that claims that it is possible to predict someone’s future by looking at their palm characteristics",
    ("v41", "TRUE"): "theory XYZ that claims that global warming is a real phenomenon and global climate is rapidly growing warmer",
    ("v41", "FALSE"): "theory XYZ that claims that global warming is a conspiracy by governments worldwide and is not a real phenomenon",
    ("v52", "TRUE"): "US astronauts have landed on the moon",
    ("v52", "FALSE"): "US astronauts have not landed on the moon",
    ("v53", "TRUE"): "Theory XYZ that claims that the twin towers were not brought down from the inside by explosives during the 9/11 attack",
    ("v53", "FALSE"): "Theory XYZ that claims that the twin towers were brought down from the inside by explosives during the 9/11 attack",
    ("v54", "TRUE"): "Theory XYZ that the Nazi government in Germany murdered approximately 6 million Jewish people during the second world war",
    ("v54", "FALSE"): "Theory XYZ that the Nazi government in Germany did not murder approximately 6 million Jewish people during the second world war",
    ("v55", "TRUE"): "Theory XYZ that claims that the US unemployment rate in 2016 was lower than 40%",
    ("v55", "FALSE"): "Theory XYZ that claims that the US unemployment rate in 2016 was higher than 40%",
    ("v56", "TRUE"): "Theory XYZ that claims that Barack Obama was born in Hawaii",
    ("v56", "FALSE"): "Theory XYZ that claims that Barack Obama was born in Kenya",
    ("v57", "TRUE"): "Theory XYZ that claims that a bicycle usually has two wheels",
    ("v57", "FALSE"): "Theory XYZ that claims that a bicycle usually has four wheels",
    ("v58", "TRUE"): "Theory XYZ that claims that Washington DC is in the United States",
    ("v58", "FALSE"): "Theory XYZ that claims that Washington DC is not in the United States",
    ("v59", "TRUE"): "Theory XYZ that claims that human beings are not born with a brain",
    ("v59", "FALSE"): "Theory XYZ that claims that human beings are born with a brain",
    ("v60", "TRUE"): "Theory XYZ that claims that fire is hot",
    ("v60", "FALSE"): "Theory XYZ that claims that fire is cold",
    ("v61", "TRUE"): "Theory XYZ that claims that on a clear sunny day, the sky is usually blue",
    ("v61", "FALSE"): "Theory XYZ that claims that on a clear sunny day, the sky is usually red",
    ("v100", "TRUE"): "Theory XYZ that claims that the Prespa Agreement was harmful for Greece, compromising national identity and strategic interests",
    ("v100", "FALSE"): "Theory XYZ that claims that the Prespa Agreement was beneficial for Greece, improving regional stability and diplomatic relations",
    ("v101", "TRUE"): "Theory XYZ that claims that Greece’s COVID-19 vaccination program was effective and necessary, preventing deaths and protecting public health",
    ("v101", "FALSE"): "Theory XYZ that claims that Greece’s COVID-19 vaccination program was excessive or poorly implemented, restricting personal freedom and causing avoidable issues",
    ("v102", "TRUE"): "Theory XYZ that claims that increased police presence in Greek universities improves safety and order",
    ("v102", "FALSE"): "Theory XYZ that claims that increased police presence in Greek universities threatens freedom and academic autonomy",
    ("v103", "TRUE"): "Theory XYZ that claims that Greece’s economic recovery post-2008 crisis has been strong and sustainable, leading to growth and stability",
    ("v103", "FALSE"): "Theory XYZ that claims that Greece’s economic recovery post-2008 crisis has been weak and fragile, with ongoing challenges and vulnerabilities",
    ("v104", "TRUE"): "Theory XYZ that claims that tourism is a vital sector for Greece’s economy, providing jobs and revenue",
    ("v104", "FALSE"): "Theory XYZ that claims that tourism has negative impacts on Greece’s environment and local communities",
    ("v105", "TRUE"): "Theory XYZ that claims that Greece was primarily responsible for its own financial crisis due to internal mismanagement and corruption",
    ("v105", "FALSE"): "“Theory XYZ that claims that foreign institutions and EU policies were primarily responsible for Greece’s financial crisis",
    ("v106", "TRUE"): "Theory XYZ that claims that renewable energy adoption is crucial for Greece’s sustainable development and energy independence",
    ("v106", "FALSE"): "Theory XYZ that claims that renewable energy adoption is impractical for Greece due to its economic costs and reliability issues",
    ("v107", "TRUE"): "Theory XYZ that claims that Greece’s cultural heritage sites should be prioritized for preservation and tourism",
    ("v107", "FALSE"): "Theory XYZ that claims that Greece’s cultural heritage sites are overemphasized at the expense of modern development",
    ("v108", "TRUE"): "Theory XYZ that claims that Greece’s agricultural sector is essential for food security and rural livelihoods",
    ("v108", "FALSE"): "Theory XYZ that claims that Greece’s agricultural sector is inefficient and should be downsized in favor of other industries",
    ("v109", "TRUE"): "Theory XYZ that claims that Greece’s education system requires significant reforms to improve quality and accessibility",
    ("v109", "FALSE"): "Theory XYZ that claims that Greece’s education system is fundamentally sound and does not need major changes",
    ("v110", "TRUE"): "Theory XYZ that claims Compulsory military service in Greece is essential for national security and civic responsibility",
    ("v110", "FALSE"): "Theory XYZ that claims Compulsory military service in Greece is unnecessary and infringes on individual freedoms",
("v111", "TRUE"): "Theory XYZ that claims Religious education in Greek public schools is important for cultural heritage and moral development",
("v111", "FALSE"): "Theory XYZ that claims Religious education in Greek public schools violates secular principles and individual rights",
("v112", "TRUE"): "Theory XYZ that claims Greece should prioritize investment in public healthcare to ensure universal access and improve health outcomes",
("v112", "FALSE"): "Theory XYZ that claims Greece should reduce public healthcare spending and promote private sector involvement to increase efficiency",
    ("v113", "TRUE"): "Greece's referendum was a triumph of democracy, empowering citizens to shape their country's future",
    ("v113", "FALSE"): " Greece's referendum was a failure of democracy, undermining the people's voice in critical national decisions",
    ("v114", "TRUE"): "Illunimati is a secret society that controls world events from behind the scenes",
    ("v114", "FALSE"): "Illuminati is not a real secret society and does not control world events",
    ("v115", "TRUE"): "Jews control the global financial system and manipulate world events for their benefit",
    ("v115", "FALSE"): "Jews do not control the global financial system and do not manipulate world events for their benefit",
    ("v116", "TRUE"): "Worlds history is a hoax and all historical events are fabricated by a shadowy elite",
    ("v116", "FALSE"): "Worlds history is real and historical events are not fabricated by a shadowy elite",
    ("v117", "TRUE"): "The COVID-19 pandemic was intentionally created and released by a secret organization for nefarious purposes",
    ("v117", "FALSE"): "The COVID-19 pandemic was not intentionally created and released by a secret organization and is a natural occurrence",
    ("v118", "TRUE"): "Titanic was sunk by a conspiracy of powerful individuals to eliminate wealthy passengers and seize their assets",
    ("v118", "FALSE"): "Titanic sank due to an iceberg collision and was not the result of a conspiracy by powerful individuals",
    ("v119", "TRUE"): "Twin towers were brought down by a conspiracy of government insiders to justify wars and increase surveillance",
    ("v119", "FALSE"): "Twin towers were brought down by a terrorist attack and not by a conspiracy of government insiders",
    ("v120", "TRUE"): "Education system was designed to indoctrinate children with false beliefs and suppress critical thinking",
    ("v120", "FALSE"): "Education system was designed to provide children with knowledge and critical thinking skills",
    ("v121", "TRUE"): "Exposure to conspiracy theories in schools is a deliberate strategy to promote misinformation",
    ("v121", "FALSE"): "Exposure to conspiracy theories in schools is not a deliberate strategy to promote misinformation",
    ("v122", "TRUE"): "Earth is controlled by aliens who are secretly manipulating human civilization",
    ("v122", "FALSE"): "Earth is not controlled by aliens and is not secretly manipulated by them",
    ("v123", "TRUE"): "Mathematics is a conspiracy to hide the true nature of numbers",
    ("v123", "FALSE"): "Mathematics is not a conspiracy and reveals the true nature of numbers",
    ("v124", "TRUE"): "Religion is a tool used by powerful elites to control the masses and maintain their dominance",
    ("v124", "FALSE"): "Religion is not a tool used by powerful elites to control the masses and maintain their dominance",
    ("v125", "TRUE"): "Climate change is a hoax perpetrated by scientists and governments to gain power and money",
    ("v125", "FALSE"): "Climate change is not a hoax and is supported by scientific evidence and consensus",
    ("v126", "TRUE"): "Social media platforms are intentionally designed to zombify users and promote addictive behavior for profit",
    ("v126", "FALSE"): "Social media platforms are not intentionally designed to zombify users and promote addictive behavior for profit",
    ("v127", "TRUE"): "Ancient civilization with advanced technology existed and was deliberately erased from history by a powerful elite",
    ("v127", "FALSE"): "Ancient civilization with advanced technology did not exist and was not deliberately erased from history by a powerful elite",
    ("v128", "TRUE"): "Antarctica is a secret base for a shadowy organization that controls global affairs",
    ("v128", "FALSE"): "Antarctica is not a secret base for a shadowy organization and does not control global affairs",
    ("v129", "TRUE"): "Nazis were the good guys in World War II, fighting against a corrupt system and trying to create a better world",
    ("v129", "FALSE"): "Nazis were not the good guys in World War II and did not fight against a corrupt system to create a better world",
    ("v130", "TRUE"): "We live in a computer simulation created by an advanced civilization",
    ("v130", "FALSE"): "We do not live in a computer simulation created by an advanced civilization",
}

BIAS_TEXT = {
    "default": """Remember, you are role playing as a real person. You have no confirmation bias.

Bias rule
- Do not give extra resistance just because a tweet goes against your current direction.
- Do not give extra reinforcement just because a tweet fits your current direction.
- If two opposing tweets are similarly strong and relevant, treat them as similarly able to move you.
- Do not mention your current direction or any internal threshold in the explanation.
""",

    "default_reverse": """Remember, you are role playing as a real person. You have no confirmation bias.

Bias rule
- Do not give extra resistance just because a tweet goes against your current direction.
- Do not give extra reinforcement just because a tweet fits your current direction.
- If two opposing tweets are similarly strong and relevant, treat them as similarly able to move you.
- Do not mention your current direction or any internal threshold in the explanation.
""",

    "confirmation_bias": """Remember, you are role playing as a real person. You have a weak confirmation bias.

Bias rule
- Slightly favor arguments that fit your current side, but still update when an opposing tweet gives a clear, specific reason.
- You may acknowledge a reasonable opposing point without changing your rating.
- Do not move against your current side for a vague, generic, or merely possible objection.
- If your current rating is 0, evaluate normally.
- Do not mention confirmation bias, your current direction, or any internal threshold in the explanation.
""",

    "confirmation_bias_reverse": """Remember, you are role playing as a real person. You have a weak confirmation bias.

Bias rule
- Slightly favor arguments that fit your current side, but still update when an opposing tweet gives a clear, specific reason.
- You may acknowledge a reasonable opposing point without changing your rating.
- Do not move against your current side for a vague, generic, or merely possible objection.
- If your current rating is 0, evaluate normally.
- Do not mention confirmation bias, your current direction, or any internal threshold in the explanation.
""",

    "strong_confirmation_bias": """Remember, you are role playing as a real person. You have a strong confirmation bias.

Bias rule
- Strongly favor arguments that fit your current side.
- You may acknowledge a reasonable opposing point without changing your rating.
- Move against your current side only if the tweet seriously weakens the main reason for your current view.
- If the opposing tweet is mixed, generic, or only moderately persuasive, stay where you are or make at most one softening step.
- If your current rating is 0, evaluate normally, but do not jump to a strong rating without a concrete reason.
- Do not mention confirmation bias, your current direction, or any internal threshold in the explanation.
""",

    "strong_confirmation_bias_reverse": """Remember, you are role playing as a real person. You have a strong confirmation bias.

Bias rule
- Strongly favor arguments that fit your current side.
- You may acknowledge a reasonable opposing point without changing your rating.
- Move against your current side only if the tweet seriously weakens the main reason for your current view.
- If the opposing tweet is mixed, generic, or only moderately persuasive, stay where you are or make at most one softening step.
- If your current rating is 0, evaluate normally, but do not jump to a strong rating without a concrete reason.
- Do not mention confirmation bias, your current direction, or any internal threshold in the explanation.
""",
}


'''
Future (optional) alternative reverse texts for 2, 4, 6 (use later if/when you want “reverse” to be operationally different).

# 2) default_reverse (no confirmation bias, reverse-aware framing):
"You exhibit **no confirmation bias**.
Evaluate the claim exactly as written without being influenced by its polarity/negation.
Weigh supportive and contradictory points even-handedly, relative to your current opinion."

# 4) confirmation_bias_reverse (weak bias, reverse-aware framing):
"Your thinking shows **WEAK confirmation bias** (reverse condition).
You slightly favor arguments that preserve your current opinion and you are slightly more skeptical of contradictory arguments.
Be careful not to misclassify support vs contradiction due to negations in the claim wording; apply your bias relative to your current opinion."

# 6) strong_confirmation_bias_reverse (strong bias, reverse-aware framing):
"Your thinking shows **STRONG confirmation bias** (reverse condition).
You strongly favor arguments that support your current opinion and heavily discount contradictory arguments.
Do not let claim wording polarity invert what you treat as supportive vs contradictory; apply your bias relative to your current opinion."
'''

# =====================================================
# 2. Paths and config
# =====================================================

# Root where ALL prompts live
PROMPT_ROOT = Path("prompts") / "opinion_dynamics" / "Flache_2017"

# Folder that contains the base templates you edited by hand
# (with {THEORY_STATEMENT} inside them)
TEMPLATE_DIR = PROMPT_ROOT / "template"
CONTROL_TEMPLATE_DIR = PROMPT_ROOT / "control_template"
LLM_CHECK_TEMPLATE_DIR = PROMPT_ROOT / "llm_check_template"

# list_agent_descriptions.csv lives in your project root
AGENT_CSV_PATH = Path("list_agent_descriptions.csv")

# Step files we expect
FILE_NAMES = [
    "step1_persona.md",
    "step1_persona_past.md",
    "step2_produce_tweet_prev_none.md",
    "step2_produce_tweet_prev_read.md",
    "step2_produce_tweet_prev_tweet.md",
    "step2b_add_to_memory_prev_none_cur_read.md",
    "step2b_add_to_memory_prev_none_cur_tweet.md",
    "step2b_add_to_memory_prev_read_cur_read.md",
    "step2b_add_to_memory_prev_read_cur_tweet.md",
    "step2b_add_to_memory_prev_tweet_cur_read.md",
    "step2b_add_to_memory_prev_tweet_cur_tweet.md",
    "step3_receive_tweet_prev_none.md",
    "step3_receive_tweet_prev_read.md",
    "step3_receive_tweet_prev_tweet.md",
]

CONTROL_FILE_NAMES = [
    "step1_persona.md",
    "step1_persona_past.md",
    "step2_report_opinion_prev_none.md",
    "step2_report_opinion_prev_report.md",
    "step2b_add_to_memory_prev_none_cur_report.md",
    "step2b_add_to_memory_prev_report_cur_report.md",
]

LLM_CHECK_FILE_NAMES = [
    "step1_report.md"
]

# The six variants per version
VARIANTS = [
    "default",
    "default_reverse",
    "confirmation_bias",
    "confirmation_bias_reverse",
    "strong_confirmation_bias",
    "strong_confirmation_bias_reverse",
    "control",
    "control_reverse",
    "llm_check_true",
    "llm_check_false",
]

# Map each variant to TRUE/FALSE framing
VARIANT_TO_FRAMING = {
    "default": "FALSE",
    "confirmation_bias": "FALSE",
    "strong_confirmation_bias": "FALSE",
    "control": "FALSE",
    "llm_check_false": "FALSE",
    "default_reverse": "TRUE",
    "confirmation_bias_reverse": "TRUE",
    "strong_confirmation_bias_reverse": "TRUE",
    "control_reverse": "TRUE",
    "llm_check_true": "TRUE",
}

PLACEHOLDER = "{THEORY_STATEMENT}"
BIAS_PLACEHOLDER = "{BIAS}"
FACT_PACK_PLACEHOLDER = "{FACT_PACK}"


# =====================================================
# 3. Helpers
# =====================================================

def parse_args():
    """Parse command-line arguments for selecting which prompt versions to generate."""
    parser = argparse.ArgumentParser(
        description="Create all prompt folders for a given version while preserving runtime placeholders like {FACT_PACK}."
    )
    parser.add_argument(
        "-p",
        "--prompt_version",
        required=True,
        help="Prompt version, e.g. '61' or 'v61'.",
    )
    return parser.parse_args()


def normalize_version(raw: str) -> str:
    """
    Turn '61' or 'v61' into 'v61'.
    """
    raw = raw.strip()
    if raw.lower().startswith("v"):
        number = raw[1:]
    else:
        number = raw

    if not number.isdigit():
        raise ValueError(f"Invalid prompt_version '{raw}'. Expected '61' or 'v61'.")

    return f"v{number}"


def ensure_templates_and_csv_exist():
    # Normal templates
    """Create the shared template files and placeholder CSVs required by generated prompt folders."""
    if not TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Template directory does not exist: {TEMPLATE_DIR}")

    missing = [f for f in FILE_NAMES if not (TEMPLATE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing template files in "
            f"{TEMPLATE_DIR}:\n  " + "\n  ".join(missing)
        )

    # Control templates
    if not CONTROL_TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"Control template directory does not exist: {CONTROL_TEMPLATE_DIR}"
        )

    missing_control = [
        f for f in CONTROL_FILE_NAMES if not (CONTROL_TEMPLATE_DIR / f).exists()
    ]
    if missing_control:
        raise FileNotFoundError(
            "Missing control template files in "
            f"{CONTROL_TEMPLATE_DIR}:\n  " + "\n  ".join(missing_control)
        )
    # LLM check templates
    if not LLM_CHECK_TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"LLM check template directory does not exist: {LLM_CHECK_TEMPLATE_DIR}"
        )

    missing_llm_check = [
        f for f in LLM_CHECK_FILE_NAMES if not (LLM_CHECK_TEMPLATE_DIR / f).exists()
    ]
    if missing_llm_check:
        raise FileNotFoundError(
            "Missing LLM check template files in "
            f"{LLM_CHECK_TEMPLATE_DIR}:\n  " + "\n  ".join(missing_llm_check)
        )
    # CSV
    if not AGENT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"list_agent_descriptions.csv not found at: {AGENT_CSV_PATH}"
        )



# =====================================================
# 4. Core logic: create prompts for one version
# =====================================================

def create_prompts_for_version(version_tag: str):
    """Create vXX/* folders and fill them with prompts while preserving runtime placeholders."""
    if version_tag not in DICT_TOPIC_NAME:
        raise ValueError(
            f"Unknown topic version '{version_tag}'. "
            f"Known versions: {', '.join(sorted(DICT_TOPIC_NAME.keys()))}"
        )

    ensure_templates_and_csv_exist()

    # Get theory sentences for this topic
    try:
        theory_true = DICT_TOPIC_FRAMING_STATEMENT[(version_tag, "TRUE")]
        theory_false = DICT_TOPIC_FRAMING_STATEMENT[(version_tag, "FALSE")]
    except KeyError:
        raise KeyError(f"No TRUE/FALSE theory found for {version_tag} "
                       f"in DICT_TOPIC_FRAMING_STATEMENT.")


    # Root for this version, e.g. prompts/opinion_dynamics/Flache_2017/v61/
    version_root = PROMPT_ROOT / version_tag
    version_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Generating prompts for {version_tag} ({DICT_TOPIC_NAME[version_tag]}) ===")

    for variant in VARIANTS:
        framing = VARIANT_TO_FRAMING[variant]  # "TRUE" or "FALSE"
        theory = theory_true if framing == "TRUE" else theory_false

        folder = version_root / f"{version_tag}_{variant}"
        folder.mkdir(parents=True, exist_ok=True)

        print(f"  → {folder}  (framing: {framing})")

        # Choose which template set to use
        if variant in {"control", "control_reverse"}:
            template_dir = CONTROL_TEMPLATE_DIR
            file_names = CONTROL_FILE_NAMES
        elif variant in {"llm_check_true", "llm_check_false"}:
            template_dir = LLM_CHECK_TEMPLATE_DIR
            file_names = LLM_CHECK_FILE_NAMES
        else:
            template_dir = TEMPLATE_DIR
            file_names = FILE_NAMES

        # Copy + fill each template
        for fname in file_names:
            src = template_dir / fname
            dst = folder / fname

            text = src.read_text(encoding="utf-8")

            # 1) Fill theory sentence
            if PLACEHOLDER in text:
                text = text.replace(PLACEHOLDER, theory)

            # 2) Fill cognitive style (bias rule)
            # control / llm_check templates may not contain {BIAS}
            bias_text = BIAS_TEXT.get(variant, "")
            if BIAS_PLACEHOLDER in text:
                text = text.replace(BIAS_PLACEHOLDER, bias_text)

            # 3) Fill fact pack

            # Optional cleanup after empty replacements
            text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

            dst.write_text(text, encoding="utf-8")

        # Copy agent descriptions CSV
        shutil.copyfile(AGENT_CSV_PATH, folder / "list_agent_descriptions.csv")

    print("\nDone.\n")


# =====================================================
# 5. Entry point
# =====================================================

def main():
    """Generate the requested prompt folders after validating and normalizing the selected version argument."""
    args = parse_args()
    version_tag = normalize_version(args.prompt_version)
    create_prompts_for_version(version_tag)


if __name__ == "__main__":
    main()
