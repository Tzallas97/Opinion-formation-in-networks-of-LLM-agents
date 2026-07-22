# Opinion dynamics on networks of LLM agents

[![tests](https://github.com/Tzallas97/Opinion-formation-in-networks-of-LLM-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/Tzallas97/Opinion-formation-in-networks-of-LLM-agents/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Agent-based simulation of how opinions form and spread through a network of
agents, where every agent is a local large language model (Qwen, served by
Ollama). Agents hold a position on a topic from −2 (fully reject) to +2 (fully accept),
talk to their neighbours over a network, and update what they believe.

![Two populations of LLM agents forming opinions over a network](assets/network_opinion_dynamics.png)

> **At a glance** — a fully **offline-reproducible** Python research system (~2 MB, **82 passing tests**, no GPU required): 30 local LLM agents (Qwen via Ollama) deliberate across a social network while I isolate three forces — network structure, retrieval-based evidence (RAG), and trust — and measure how opinions form, polarise, and spread. Ships a Tkinter launcher, a cross-family **LLM-as-judge** evaluation layer, semantic-similarity metrics, and interactive HTML viewers.
>
> **Tech:** Python 3.12 · LLM orchestration (Ollama / Qwen, LangChain) · Retrieval-Augmented Generation · multi-agent systems · NetworkX · pandas / NumPy · matplotlib / Plotly · pytest · GitHub Actions CI.

The core thesis pipeline is the simulation, the launcher, and the plotting script. The repo also
ships a larger offline analysis and evaluation toolkit (`tools/`) that goes beyond the thesis; it
is documented at the end of this README under [Extended toolkit](#extended-toolkit--programme-expansion).

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

## Quickstart (offline — no GPU, Ollama, or model download)

A full experiment talks to a local LLM, but the entire pipeline also runs in a
deterministic **offline mode** (`FAKE_LLM=1`) that swaps the model for a stub.
So you can clone the repo and reproduce the full test suite in a couple of
minutes, with nothing installed but the Python packages:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest
FAKE_LLM=1 PYTHONHASHSEED=0 python -m pytest tests/ -q   # 82 tests, ~2-3 min
```

The same `FAKE_LLM=1` toggle powers an end-to-end offline regression test of the
simulation itself (`tests/test_adr006_baseline_regression.py`), so the pipeline
is exercised without any model. For real runs with live inference, see
[Setup](#setup).

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
    opinion_dynamics_test_network_qwen.py  The simulation itself (also the solo
                                        model check, via --solo_check on).
    create_prompts.py                   Generates prompt folders for a version.
    network_models.py                   Small-world / Erdos-Renyi / Barabasi-Albert
                                        builders and neighbour selection.
    run_naming.py                       Shared run-folder / file naming + the mode
                                        abbreviations (read by every tool).
    main_plot_network.py                Plots and animations of a finished run.
    plot_launcher.py                    Tkinter GUI that drives main_plot_network.
    compare_runs.py                     Overlay several runs on one figure (+ GUI).
    persona_profiles_support.py         Persona loading and the persona editor.
    topic_persona_fields.py             Persona field definitions.
    topic_persona_profiles.py           Persona profile logic.
    config/persona_profiles.json        Persona definitions (incl. trust).
    config/persona_schema.json          Schema the personas validate against.
tools/                                  Extended analysis/evaluation toolkit
                                        (see "Extended toolkit" below).
tests/                                  Headless GUI smoke test + tkinter stub.
docs/                                   User-facing guides for each GUI and tool.
requirements.txt                        Python dependencies (Python 3.12).
```

The launcher saves its state to `scripts/.launcher_settings.json` at runtime;
that file is not tracked (the launcher falls back to built-in defaults if it is
absent).

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

## Running the solo model check

The solo check asks the model about a claim on its own — no personas, no network,
no interactions — so each "agent" is an independent sample of the model's own
prior on the topic. It uses the same native inference path and decoding options
as a full run, so the solo numbers are directly comparable to in-network
behaviour. This replaces the old standalone calibration script.

From the launcher, set **Solo check: on** (the network, persona and RAG panels
grey out, since they do not apply). Or from the terminal:

```bash
python scripts/opinion_dynamics_test_network_qwen.py --solo_check on \
    -m qwen3:8b -agents 10 -steps 5 -v v130_llm_check_true
```

Each of the `-agents` samples answers the version's `step1_report.md`; the run
writes an `*_opinion_change_*.csv` (same shape the eval tools read) plus a
`*_solo_check_metrics.json` with the full config.

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

The easiest way is the plotting GUI. Point it at a results folder and it finds
every finished run inside, reads back each run's parameters from its file names
(agents, steps, seed, version, date, distribution, model — nothing typed by
hand), and plots the ones you tick:

```bash
python scripts/plot_launcher.py
```

Or run the plotting script directly on one run:

```bash
python scripts/main_plot_network.py --help
```

It reads a finished run's output and produces the full figure set: opinion
distributions (initial/final, an opinion×time heatmap, a fragmentation curve,
a rating-share streamgraph, and a distribution ridgeline), belief trajectories,
B/D/P time series, an empirical Markov transition matrix, network diagnostics
(degree, assortativity over time, hub assignment, ego networks), mechanism plots
(influence matrix, argument-theme effectiveness, RAG evidence direction), and
animations of the network and the distribution. All figures share one modern
style, and each is skipped with a note if its source CSV is absent, so the run
never crashes. Full guide: `docs/main_plot_guide.md`.

To compare several runs on one figure (mean opinion, diversity, fragmentation,
and final distribution overlaid, one colour per run):

```bash
python scripts/compare_runs.py            # GUI: pick 2-6 run folders
python scripts/compare_runs.py A B C --out cmp.png   # or from the CLI
```

Run outputs use a compact naming scheme (`<out>_<agents>_<steps>_v<ver>_<mode>`,
e.g. `..._v119_strong_r`, with the short mode abbreviations in
`scripts/run_naming.py`). Every tool also reads the older long naming, so runs
made before the change keep working.

## Extended toolkit — programme expansion

Everything below lives in `tools/` and is **not part of the diploma thesis**. The thesis uses
the simulation, the launcher, and the plotting above. This toolkit is an ongoing expansion of the
programme — an offline ecosystem for evaluating and probing runs, and for the retrieval-architecture
experiments — kept in the repo so the work can grow past the thesis into follow-up studies. It is
tested (unit / mock / functional on real run files), but it did not produce any thesis result.
Items marked \* also need a running Ollama for their full mode; without it they fall back to a
dependency-free lexical mode or are skipped.

Each tool runs standalone from the command line, and most are also reachable from one GUI
(`tools/eval_launcher.py`). Deep-dive docs live in `docs/` (`eval_judge_guide.md`,
`ui_launcher_guide.md`, `main_plot_guide.md`, `multihop_corpus_spec.md`).

### Evaluation and reporting

`tools/eval_runs.py` — the analysis harness. Give it run folders (or any parent folder; it scans
recursively and finds every run inside by its `*opinion_change*.csv`) and it computes per-run
B/D/P trajectories, convergence, holds/repairs/cleanups/fallbacks/leaks, agent-level bootstrap
confidence intervals, pairwise deltas and figures, writing `eval_report.md` + `aggregate.csv`.

```bash
python tools/eval_runs.py results/opinion_dynamics/Flache_2017/qwen3_8b --out eval_out
```

`tools/master_report.py` — aggregates a master CSV (accumulated by `eval_runs --append-csv`) into
a per-condition summary: mean ± std across runs, i.e. the real seed-to-seed dispersion that
within-run CIs cannot show. One row per condition instead of one per run.

```bash
python tools/master_report.py master.csv --out master_out
```

`tools/eval_launcher.py` — one Tkinter GUI over the whole toolkit: pick runs, then run the
comparison, the judge, the semantic analysis, the inference-emergence and scale metrics, and the
HTML viewers, each with an explanation next to every option.

```bash
python tools/eval_launcher.py
```

### LLM-as-judge and text-space metrics

`tools/judge_runs.py` \* — a read-only LLM judge that scores tweets (stance match, closed-world
leak, quality) and responses (direction support, leak, quality). It inherits context from the run
files themselves (claim, config, personas, verbatim world rules and bias wording, and the exact
RAG snippets each agent was shown), runs cross-family by default (a Llama judge for Qwen runs),
uses no chain-of-thought, greedy decoding and a SHA-256 cache so re-runs are free, and can export
a calibration sheet for human anchoring. Always `--dry-run` first to inspect the prompt.

```bash
python tools/judge_runs.py <run_folder> --dry-run --sample 120 --context full
```

`tools/semantic_analysis.py` \* — measures what the −2..+2 ratings cannot: homogenisation (are
agents starting to say the same thing?), text convergence over time, stance separation (do FOR and
AGAINST posts actually read differently?), and self-repetition. `tfidf` backend runs anywhere;
`ollama` backend uses real embeddings.

```bash
python tools/semantic_analysis.py <run_folder> --backend tfidf
```

`tools/scale_infoloss.py` — quantifies how much of the text the integer scale compresses away:
variance explained (R²), leave-one-out text→rating recovery (are the five levels textually real?),
flat-rating drift, and distinct "voices" per rating.

```bash
python tools/scale_infoloss.py <run_folder> [--backend ollama]
```

### Retrieval-architecture experiments

A multi-hop evidence corpus lets retrieval itself become the experimental variable. Snippets are
individually thin but connect into reasoning chains through shared entities; the ground truth
(chains, implied conclusions, null probes) is kept in a separate evaluation-only file.

`tools/retrievers.py` — three retrieval architectures over the same corpus: `lexical` (token
overlap), `dense`\* (embedding cosine), and `graph` (entity-graph chain recovery). The difference
between them *is* the variable.

`tools/corpus_lint.py` — validity linter for the corpus: field/id integrity, tag↔text
consistency, chain reachability, shared-vocabulary and balance warnings. Run it after editing the
corpus.

```bash
python tools/corpus_lint.py prompts/opinion_dynamics/Flache_2017/content/rag_corpus/multihop_v119.jsonl
```

`tools/chain_benchmark.py` — the offline gate: before any simulation spends compute, it checks
whether `graph` actually beats `lexical`/`dense` at recovering full chains from the corpus's ground
truth. Writes `chain_benchmark.md` with a PASS/REWORK verdict.

```bash
python tools/chain_benchmark.py <corpus>.jsonl --backends lexical,graph,dense
```

`tools/inference_emergence.py` \* — do agents articulate conclusions written *nowhere* in the
corpus? It scores every tweet against each chain's implied conclusion, with a threshold
auto-calibrated from null probes (plausible-sounding claims the corpus does not support). Runs that
never saw the corpus act as a negative control.

```bash
python tools/inference_emergence.py <run_folder> --groundtruth <corpus>.groundtruth.json
```

### Interactive viewers

`tools/build_viewer.py` / `tools/build_ab_viewer.py` — self-contained offline HTML viewers: a
single-run viewer (network + time scrubber + per-agent tweets/replies panel) and an A/B viewer
(two runs side by side on one synchronized scrubber). No dependencies at view time.

```bash
python tools/build_viewer.py <run_folder>          # writes <run>_viewer.html
python tools/build_ab_viewer.py <run_a> <run_b>    # writes an A/B viewer
```

### Solo model check

Not a tool but a mode of the main script: it asks the model about a claim on its own — no
personas, no network, no interactions — so each "agent" is an independent sample of the model's
own prior. Set **Solo check: on** in the launcher, or pass `--solo_check on` to
`opinion_dynamics_test_network_qwen.py`. It uses the same native inference path and decoding
options as a full run, so the solo numbers are directly comparable to in-network behaviour.
