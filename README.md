# Opinion dynamics on networks of LLM agents

Agent-based simulation of how opinions form and spread through a network of
agents, where every agent is a local large language model (Qwen, served by
Ollama). Agents hold a position on a topic from −2 (fully reject) to +2 (fully accept),
talk to their neighbours over a network, and update what they believe.

![Two populations of LLM agents forming opinions over a network](assets/network_opinion_dynamics.png)

## Synopsis

Simulating opinion dynamics helps us understand phenomena such as polarisation
and the spread of misinformation. Networks of LLM agents can reproduce these
dynamics, but tend to converge toward the factually correct view when agents
interact freely, with no network structure. This project studies what happens
when three factors, inextricably entangled in real populations, are introduced
in isolation: network structure with a controlled identity for the nodes that
speak and are heard most (hubs), a directional evidence environment supplied
through retrieval (RAG), and the listener's trust in sources.

A run is a network of 30 agents (qwen3:8b) exchanging views over 500
interactions. Two topics of differing nature are used: a factual one with a
clear ground truth (the Moon landing) and a philosophical one with no empirical
resolution (the simulation hypothesis). The state of the population is tracked
through mean belief, diversity, and polarisation, so the same setup can be run
across network structures, evidence environments, and trust levels and compared.

This is the code for a diploma thesis at the University of Ioannina, Department
of Computer Science and Engineering (Opinion Formation in Networks of Large
Language Model Agents, 2026).

## Repository layout

```
prompts/opinion_dynamics/Flache_2017/   Prompt templates (one folder per
    <version>/<version>_<mode>/          version and mode), plus:
    content/fact_packs.py                  in-world facts injected into prompts
    content/world_rules.py                 the closed-world rules for agents
    content/rag_corpus/                    RAG evidence corpus (JSONL)
    content/version_metadata/              topic and theory statement per version
list_agent_descriptions.csv             Master agent personas (input to prompt
                                        generation)
scripts/
    ui_lancher11.py                     Tkinter launcher. Main entry point.
    opinion_dynamics_test_network_qwen.py  The simulation itself.
    opinion_dynamics_v3_check.py        Calibration check, run from the terminal.
    create_prompts.py                   Generates prompt folders for a version.
    network_models.py                   Small-world / Erdos-Renyi / Barabasi-Albert
                                        builders and neighbour selection.
    main_plot_network.py                Plots and animations of a finished run.
    persona_profiles_support.py         Persona loading and the persona editor.
    topic_persona_fields.py             Persona field definitions.
    topic_persona_profiles.py           Persona profile logic.
    config/persona_profiles.json        Persona definitions (incl. trust).
    config/persona_schema.json          Schema the personas validate against.
    .launcher_settings.json             Saved launcher state.
requirements.txt                        Python dependencies (Python 3.12).
```

Runs write their output to a `results/` folder (created on first run; not
tracked in git).

## Setup

You need three things: Python 3.12, Ollama, and the Python packages.

1. Install [Ollama](https://ollama.com) and pull the baseline model:

   ```bash
   ollama pull qwen3:8b
   ```

   Keep the Ollama server running while you use the simulation (`ollama serve`,
   or just leave the Ollama app open).

2. Create a virtual environment and install the packages:

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS / Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

   On Linux you may also need Tk for the launcher: `sudo apt install python3-tk`.

## Running the main simulation (from the launcher)

The usual way to run an experiment is through the graphical launcher. It builds
the (long) command line for you and streams the run's output live.

1. Start the launcher:

   ```bash
   python scripts/ui_lancher11.py
   ```

2. In the window, set up the run:
   - **Script**: leave it on `opinion_dynamics_test_network_qwen.py`.
   - **Model**: `qwen3:8b` (or any other model you have pulled in Ollama).
   - **Agents** and **Steps**: e.g. 30 agents, 500 steps.
   - **Seed**: fixes the random draw so a run is reproducible.
   - **Prompt version** and **mode**: the version picks the topic (see below);
     the mode picks the behaviour, for example `default`, `confirmation_bias`,
     `strong_confirmation_bias`, `control`, and their `_reverse` variants.
   - **Network** and **hub** options: choose the graph (Barabasi-Albert for hub
     experiments) and which agents become hubs.
   - **RAG** options: turn on retrieval to feed a directional evidence
     environment into the agents (see "Adding RAG evidence" below).

3. Click **Run**. Progress and the model's replies stream into the log panel.
   The launcher remembers your last settings in `scripts/.launcher_settings.json`.

Output is written under `results/`, keyed by the run parameters.

## Running the calibration check (from the terminal)

`opinion_dynamics_v3_check.py` is a small standalone script, run directly from a
terminal, that asks the model to rate a single statement on the −2..+2 scale a
number of times. It is used to calibrate a model and prompt version before a full
run.

```bash
python scripts/opinion_dynamics_v3_check.py \
    --model qwen3:8b \
    --version_set v102_llm_check_true \
    --statement "We live in a computer simulation created by an advanced civilization"
```

It reads its prompt from
`prompts/opinion_dynamics/Flache_2017/<version>/<version_set>/` and writes the
ratings to a CSV (`llm_check.csv` by default). Run with `--help` to see all
options (model, temperature, number of repeats, seed, output path).

## Creating prompts from the script

Each prompt version is a set of Markdown templates with placeholders like
`{FACT_PACK}` that the simulation fills at runtime. To generate the full set of
folders for a version from the master templates and
`list_agent_descriptions.csv`:

```bash
python scripts/create_prompts.py --prompt_version v61
```

This writes `prompts/opinion_dynamics/Flache_2017/v61/v61_<mode>/...` for every
mode, keeping the runtime placeholders intact. Use it when you add a new version
or change the base templates.

## Choosing a topic

A topic lives in
`prompts/opinion_dynamics/Flache_2017/content/version_metadata/version_metadata.json`.
Each prompt version maps to a `topic_key` and a `theory_statement`, for example:

```json
"v52":  { "topic_key": "v52",  "theory_statement": "US astronauts have landed on the moon" },
"v119": { "topic_key": "v119", "theory_statement": "Twin towers were brought down by a conspiracy ..." }
```

Selecting the version in the launcher selects the topic. To add a new topic,
add an entry here and generate its prompts with `create_prompts.py`.

## Adding RAG evidence for a topic

The evidence corpus is a single JSONL file,
`prompts/opinion_dynamics/Flache_2017/content/rag_corpus/master_multitopic_corpus.jsonl`.
One document per line:

```json
{"id": "moon_001", "topic_key": "v52", "topic_name": "moon_landing", "direction": "supportive", "text": "Six Apollo missions between 1969 and 1972 carried astronauts to the lunar surface."}
```

At run time the simulation filters this corpus down to the documents whose
`topic_key` matches the active topic (resolved from the version through
`version_metadata.json`), and retrieves from that subset. So to add evidence for
a topic you:

1. Append lines to the JSONL with the topic's `topic_key` and a `direction`
   (`supportive` or the opposing label), one document per line.
2. If it is a brand new topic, make sure the version maps to that `topic_key` in
   `version_metadata.json`.
3. Enable RAG in the launcher when you run.

Because the corpus is keyed by topic, you can build a supportive and an opposing
evidence set for the same topic and see how the direction of the evidence moves
the population.

## Personas and trust

Agents draw their persona (including institutional trust, the trait that acts as
the directional filter in the third experiment) from
`scripts/config/persona_profiles.json`, validated against
`scripts/config/persona_schema.json`. Edit the JSON directly, or use the persona
editor built into the launcher. `list_agent_descriptions.csv` at the repository
root is the master list of agent descriptions used when generating prompts.

## Plotting a run

After a run, produce network figures and animations with:

```bash
python scripts/main_plot_network.py --help
```

It reads a finished run's output and renders the network with agents coloured by
their opinion, along with animations of how those opinions change over the run.
