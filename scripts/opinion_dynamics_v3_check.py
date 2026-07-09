import os
from os.path import join
import argparse
import csv
from datetime import date
import random
import re

import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from langchain_ollama import ChatOllama

USE_FAKE_LLM = False


def sanitize_model_name_for_path(model_name: str) -> str:
    """Return a Windows-safe results-folder name for Ollama model tags.

    Ollama model names such as qwen3:8b are valid model identifiers but invalid
    Windows folder names because of the colon. This mirrors the qwen network
    scripts/plotters: use the original model name for ChatOllama, but a sanitized
    name such as qwen3_8b for result paths.
    """
    s = str(model_name or "").strip()
    if not s:
        return "model"
    s = re.sub(r'[<>:"/\\|?*]+', "_", s)
    s = re.sub(r"_+", "_", s).strip(" ._")
    return s or "model"


def _output_stem(filename: str) -> str:
    """Return filename without its extension, preserving old -out behavior."""
    stem = os.path.splitext(str(filename or "llm_check.csv"))[0]
    return stem or "llm_check"


#######################
# Argument parser
#######################
parser = argparse.ArgumentParser(
    description=(
        "Opinion Dynamics v3 (LLM check): "
        "no personas, no interactions. "
        "Each 'agent' is just an independent sample of the LLM, "
        "reporting its belief using step1_report.md."
    )
)

parser.add_argument(
    "-m",
    "--model_name",
    default="qwen3:8b",
    type=str,
    help="Name of the LLM to use.",
)
parser.add_argument(
    "-t",
    "--temperature",
    default=0.7,
    type=float,
    help="Sampling temperature.",
)
parser.add_argument(
    "-agents",
    "--num_agents",
    default=10,
    type=int,
    help="Number of independent LLM samples (virtual agents).",
)
parser.add_argument(
    "-steps",
    "--num_steps",
    default=5,
    type=int,
    help="Number of reports per agent.",
)
parser.add_argument(
    "-version",
    "--version_set",
    default="v102_llm_check_true",
    type=str,
    help=(
        "Prompt version directory to use, e.g. 'v102_llm_check_true'. "
        "Final path will be prompts/opinion_dynamics/Flache_2017/"
        "<vXX>/<version_set>/step1_report.md"
    ),
)
parser.add_argument(
    "-seed",
    "--seed",
    default=1,
    type=int,
    help="Random seed (mainly for reproducible shuffling if needed).",
)
parser.add_argument(
    "-test",
    "--test_run",
    action="store_true",
    help="If set, place results under .../test_runs/.",
)
parser.add_argument(
    "-out",
    "--output_file",
    type=str,
    default="llm_check.csv",
    help="Base name of the output file (without path).",
)
parser.add_argument(
    "--theory_statement",
    type=str,
    default="We live in a computer simulation created by an advanced civilization",
    help="Statement inserted into {THEORY_STATEMENT} in step1_report.md.",
)

args = parser.parse_args()

#########################################
# LLM helpers (similar style to v2)
#########################################

class SimpleConversation:
    """
    Small wrapper so we can reuse get_integer_llm_response from v2 style:
    it just calls llm.invoke(prompt).content
    """

    def __init__(self, llm):
        self.llm = llm

    def predict(self, input: str) -> str:
        resp = self.llm.invoke(input)
        # ChatOllama returns a ChatMessage; we want the text content
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(50))
def get_llm_response(conversation: SimpleConversation, prompt: str) -> str:
    if USE_FAKE_LLM:
        return "FINAL_RATING: 0\nBELIEF:\n(dummy)\n\nEXPLANATION:\n(dummy explanation)"
    else:
        return conversation.predict(input=prompt)


def get_integer_llm_response(conversation: SimpleConversation, prompt: str) -> str:
    """
    Call the LLM but insist that the reply actually contains
    an *integer rating* in [-2, 2]. If not, we re-prompt a few times.
    This is identical in spirit to v2.
    """
    max_attempts = 3
    last_response = ""

    for attempt in range(max_attempts):
        response = get_llm_response(conversation, prompt)
        last_response = response

        # 1) reject floats like "1.5"
        if re.search(r"[+-]?\d*\.\d+", response):
            print("Bad (non-integral) value found, re-prompting...")
            continue

        # 2) extract integers and keep only those in [-2, 2]
        candidates = []
        for m in re.findall(r"[-+]?\d+", response):
            try:
                val = int(m)
            except ValueError:
                continue
            if -2 <= val <= 2:
                candidates.append(val)

        if candidates:
            # success: we have at least one valid rating
            return response

        print("No valid integer rating in reply, re-prompting...")

    print(
        "WARNING: Failed to get a valid integer rating after "
        f"{max_attempts} attempts. Using last response anyway."
    )
    return last_response


#########################################
# Belief extraction (same as your v2)
#########################################

def extract_belief(reply: str):
    """
    Same as in v2: look only at the FIRST line,
    expect 'FINAL_RATING: X' with X in {-2,-1,0,1,2}.
    """
    if not reply:
        return None

    first_line = reply.strip().splitlines()[0]

    if first_line.startswith("FINAL_RATING:"):
        try:
            r = int(first_line.split(":")[1].strip())
            if r in [-2, -1, 0, 1, 2]:
                return r
        except Exception:
            return None

    return None


#########################################
# Main check experiment
#########################################

def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    experiment_id = "Flache_2017"
    model_name = args.model_name
    temperature = args.temperature
    num_agents = args.num_agents
    num_steps = args.num_steps

    # Base prompt root: prompts/opinion_dynamics/Flache_2017
    prompt_template_root_base = os.path.join("prompts", "opinion_dynamics")
    base_prompt_root = os.path.join(prompt_template_root_base, experiment_id)

    # Version-specific prompt directory, e.g. Flache_2017/v102/v102_llm_check_true
    version_prefix = args.version_set.split("_")[0]  # "v102"
    prompt_template_root = os.path.join(base_prompt_root, version_prefix, args.version_set)

    # Read the pure LLM-prior report prompt.
    # This check script is not a persona/interactions run; it should use step1_report.md.
    report_path = join(prompt_template_root, "step1_report.md")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Could not find step1_report.md at: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()

    base_prompt = base_prompt.replace("{THEORY_STATEMENT}", str(args.theory_statement))

    print(f"Using prompt from: {report_path}\n")

    # Setup LLM + simple conversation wrapper
    llm = ChatOllama(model=model_name, temperature=temperature)
    conversation = SimpleConversation(llm)

    # We'll track belief trajectories for each virtual agent
    # Start each at 0 (neutral)
    current_beliefs = [0 for _ in range(num_agents)]

    # Prepare CSV dict: time_step + one column per Agent_k
    dict_csv = {"time_step": [0]}
    for k in range(num_agents):
        dict_csv[f"Agent_{k+1}"] = [current_beliefs[k]]

    # Prepare output path. Use the original model tag for Ollama, but a Windows-safe
    # folder name for results, matching the qwen network scripts.
    model_name_for_path = sanitize_model_name_for_path(model_name)
    path_result = os.path.join("results", "opinion_dynamics", experiment_id, model_name_for_path)
    if args.test_run:
        path_result = os.path.join(path_result, "test_runs")
    print(f"Saving results under model folder: {model_name_for_path}")

    today = date.today().strftime("%Y%m%d")
    date_version = "20231212"

    out_name = (
        os.path.join(path_result, _output_stem(args.output_file))
        + "_"
        + str(num_agents)
        + "_"
        + str(num_steps)
        + "_"
        + args.version_set
        + "_opinion_change_"
        + date_version
        +"_uniform"
        + ".csv"
    )

    os.makedirs(os.path.dirname(out_name), exist_ok=True)

    # Also optional raw-response log
    raw_out_name = (
        os.path.join(path_result, _output_stem(args.output_file))
        + "_"
        + str(num_agents)
        + "_"
        + str(num_steps)
        + "_"
        + args.version_set
        + "_llm_check_raw_responses_"
        + date_version
        + ".csv"
    )

    with open(raw_out_name, "w", newline="", encoding="utf-8") as raw_f:
        raw_writer = csv.writer(raw_f)
        raw_writer.writerow(["time_step", "agent_index", "response_text", "extracted_rating"])

        # Time loop
        for t in range(num_steps):
            print(f"\n===== STEP {t+1} =====")
            dict_csv["time_step"].append(t + 1)

            for k in range(num_agents):
                # Call LLM with the single prompt
                response = get_integer_llm_response(conversation, base_prompt)
                print(f"Agent_{k+1} RESPONSE: {response[:200]}...")

                rating = extract_belief(response)

                # Apply the SAME delta rule as in v2
                if rating is None:
                    # keep old belief
                    new_belief = current_beliefs[k]
                else:
                    prev = current_beliefs[k]
                    delta = rating - prev
                    if delta > 1:
                        delta = 1
                    elif delta < -1:
                        delta = -1
                    new_belief = prev + delta
                    new_belief = max(-2, min(2, new_belief))

                current_beliefs[k] = new_belief
                dict_csv[f"Agent_{k+1}"].append(new_belief)

                raw_writer.writerow([t + 1, k + 1, response, rating])

    # Save opinion trajectories
    import pandas as pd

    pd.DataFrame.from_dict(dict_csv).to_csv(out_name, index=False, encoding="utf-8")
    print(f"\nSaved LLM-check opinion trajectories to: {out_name}")
    print(f"Saved raw responses to: {raw_out_name}")


if __name__ == "__main__":
    main()
