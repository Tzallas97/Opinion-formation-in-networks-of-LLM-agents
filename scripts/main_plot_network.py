import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import glob
import re
import warnings
import networkx as nx
import glob
import plotly.graph_objects as go


def sanitize_model_name_for_path(model_name: str) -> str:
    s = str(model_name or "").strip()
    if not s:
        return "qwen3_8b"
    return re.sub(r'[<>:"/\|?*]', "_", s).strip().rstrip(".") or "qwen3_8b"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot opinion trajectories + initial/final distributions "
            "(bias, diversity, polarization) from opinion_change CSV."
        )
    )

    parser.add_argument(
        "-agents", "--num_agents",
        type=int,
        required=True,
        help="Number of agents (must match the simulation).",
    )
    parser.add_argument(
        "-steps", "--num_steps",
        type=int,
        required=True,
        help="Number of steps (must match the simulation).",
    )
    parser.add_argument(
        "-seed", "--seed",
        type=int,
        default=1,
        help="Seed used in the simulation (used only in default output_file name).",
    )
    parser.add_argument(
        "-v", "--version_set",
        type=str,
        required=True,
        help="Prompt version used (e.g., v61_confirmation_bias).",
    )
    parser.add_argument(
        "-dist", "--distribution",
        type=str,
        default="uniform",
        help="Opinion distribution used in the simulation (e.g., uniform).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20231212",
        help="Date string used inside opinion_dynamics_v2.py (default: 20231212).",
    )
    parser.add_argument(
        "-out", "--output_file",
        type=str,
        default="",
        help=(
            "Base output name used in the simulation (e.g., seed7_1_temp_0_7). "
            "If empty, we assume 'seed{seed}'."
        ),
    )
    parser.add_argument(
        "--no_annotation",
        action="store_true",
        help="If set, removes axis labels and title (clean figure).",
    )
    parser.add_argument(
        "--figure_file_type",
        type=str,
        choices=["png", "pdf"],
        default="png",
        help="Figure file type (png or pdf).",
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="qwen3:8b",
        help="Model name used for the simulation results path (e.g., qwen3:8b, qwen3.5:4b).",
    )

    return parser.parse_args()

def _plot_safe_file_token(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
    s = s.strip("._")
    return s or "run"


def _output_base_token(args) -> str:
    raw = args.output_file if getattr(args, "output_file", "") else f"seed{getattr(args, 'seed', '')}"
    raw = os.path.splitext(os.path.basename(str(raw or "").strip()))[0]
    return _plot_safe_file_token(raw or f"seed{getattr(args, 'seed', '')}")


def _version_root_token(version_set: str) -> str:
    s = str(version_set or "").strip()
    m = re.match(r"^(v\d+)", s, flags=re.I)
    return _plot_safe_file_token((m.group(1).lower() if m else s) or "v")


def _bias_short_token(version_set: str) -> str:
    """Match the simulator's compact bias naming.

    default -> no, default_reverse -> no_r,
    confirmation_bias -> weak, confirmation_bias_reverse -> weak_r,
    strong_confirmation_bias -> strong, strong_confirmation_bias_reverse -> strong_r.
    """
    s = str(version_set or "").strip().lower()
    root = _version_root_token(s).lower()
    mode = s[len(root):].lstrip("_") if s.startswith(root) else s
    is_reverse = mode.endswith("_reverse") or mode == "reverse" or "_reverse_" in mode
    if "strong_confirmation_bias" in mode:
        return "strong_r" if is_reverse else "strong"
    if "confirmation_bias" in mode:
        return "weak_r" if is_reverse else "weak"
    if mode in {"", "default"}:
        return "no"
    if mode == "default_reverse":
        return "no_r"
    if "control" in mode:
        return "control_r" if is_reverse else "control"
    if "llm_check_true" in mode:
        return "checktrue_r" if is_reverse else "checktrue"
    if "llm_check_false" in mode:
        return "checkfalse_r" if is_reverse else "checkfalse"
    return _plot_safe_file_token(mode or "no")


def _short_run_base_from_args(args) -> str:
    """Compact run folder/file stem used by qwen13+.

    Format: seed/out + agents + steps + version_root + short_bias.
    Example: seed68_tscw31ragbaposhubs_30_501_v130_strong_r
    """
    out_base = _output_base_token(args)
    try:
        seed_token = f"seed{int(getattr(args, 'seed'))}"
    except Exception:
        seed_token = "seed"
    if out_base.lower().startswith(seed_token.lower()):
        seed_out = out_base
    else:
        seed_out = f"{seed_token}_{out_base}" if out_base else seed_token
    parts = [
        seed_out,
        str(int(getattr(args, "num_agents", 0))),
        str(int(getattr(args, "num_steps", 0))),
        _version_root_token(getattr(args, "version_set", "")),
        _bias_short_token(getattr(args, "version_set", "")),
    ]
    return _plot_safe_file_token("_".join(parts))


def _old_run_base_from_args(args) -> str:
    base_name = args.output_file if getattr(args, "output_file", "") else f"seed{args.seed}"
    return f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}_{args.date}_{args.distribution}"


def _model_results_root(args) -> str:
    model_name = sanitize_model_name_for_path(args.model_name)
    return os.path.join("results", "opinion_dynamics", "Flache_2017", model_name)


def _experiment_folder_candidates(args) -> list[str]:
    """Return new + old possible run folders without deleting either.

    qwen13+ writes the compact folder. Older runs write the date/distribution folder.
    Plotting should read either, preferring an existing compact folder when present.
    """
    root = _model_results_root(args)
    short_dir = os.path.join(root, _short_run_base_from_args(args))
    old_dir = os.path.join(root, _old_run_base_from_args(args))
    # Extra permissive fallbacks for partially renamed folders.
    short = _short_run_base_from_args(args)
    base = _output_base_token(args)
    version_root = _version_root_token(args.version_set)
    glob_patterns = [
        os.path.join(root, short + "*"),
        os.path.join(root, f"*{base}*{args.num_agents}*{args.num_steps}*{version_root}*{_bias_short_token(args.version_set)}*"),
        os.path.join(root, f"*{base}*{args.num_agents}*{args.num_steps}*{args.version_set}*{args.distribution}*"),
    ]
    out = []
    seen = set()
    for cand in [short_dir, old_dir]:
        key = os.path.normpath(cand)
        if key not in seen:
            out.append(cand); seen.add(key)
    for pat in glob_patterns:
        for cand in sorted(glob.glob(pat)):
            if os.path.isdir(cand):
                key = os.path.normpath(cand)
                if key not in seen:
                    out.append(cand); seen.add(key)
    return out


def build_experiment_folder(args) -> str:
    for cand in _experiment_folder_candidates(args):
        if os.path.isdir(cand):
            return cand
    # Prefer the compact qwen13+ folder for future runs when nothing exists yet.
    return _experiment_folder_candidates(args)[0]


def _score_candidate_path(path: str, args, *, token: str = "") -> tuple[int, int, str]:
    b = os.path.basename(path)
    folder = os.path.basename(os.path.dirname(path))
    short = _short_run_base_from_args(args)
    base = _output_base_token(args)
    score = 0
    if short in b or short in folder:
        score += 20
    if base in b or base in folder:
        score += 6
    if str(args.version_set) in b or str(args.version_set) in folder:
        score += 5
    if _version_root_token(args.version_set) in b or _version_root_token(args.version_set) in folder:
        score += 3
    if str(args.distribution) in b or str(args.distribution) in folder:
        score += 2
    if str(args.num_agents) in b:
        score += 1
    if str(args.num_steps) in b:
        score += 1
    if token and token in b:
        score += 8
    # Prefer shorter compact names when the semantic score is tied.
    return (score, -len(b), path)


def _find_existing_csv(args, *, description: str, candidate_names: list[str], glob_tokens: list[str]) -> str:
    tried = []
    matches = []
    for results_dir in _experiment_folder_candidates(args):
        for name in candidate_names:
            p = os.path.join(results_dir, name)
            tried.append(p)
            if os.path.exists(p):
                return p
        for token in glob_tokens:
            for pat in [
                os.path.join(results_dir, f"*{token}*.csv"),
                os.path.join(results_dir, "_retry_debug", f"*{token}*.csv"),
            ]:
                tried.append(pat)
                matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    if matches:
        primary_token = glob_tokens[0] if glob_tokens else ""
        return sorted((_score_candidate_path(m, args, token=primary_token) for m in matches))[-1][2]
    raise FileNotFoundError(
        f"Could not find {description}. Tried compact and legacy names:\n" + "\n".join(tried[:80])
    )

def extract_version_root(version_set: str) -> str:
    """
    From version_set like 'v102_default', 'v102_confirmation_bias', etc.
    extract just the root version prefix: 'v102'.
    """
    return version_set.split("_")[0]

    

def build_input_csv_path(args) -> str:
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_opinion_change.csv",
        f"{old_stem}_network_opinion_change_{args.date}_{args.distribution}.csv",
        f"{old_stem}_{args.date}_network_opinion_change_{args.distribution}.csv",
    ]
    return _find_existing_csv(
        args,
        description="opinion-change CSV",
        candidate_names=candidate_names,
        glob_tokens=["opinion_change", "network_opinion_change"],
    )


def build_input_step_summary_csv_path(args) -> str:
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_step_summary.csv",
        f"{old_stem}_network_step_summary_{args.date}_{args.distribution}.csv",
        f"{old_stem}_{args.date}_network_step_summary_{args.distribution}.csv",
    ]
    return _find_existing_csv(args, description="step summary CSV", candidate_names=candidate_names, glob_tokens=["step_summary"])


def build_input_agent_summary_csv_path(args) -> str:
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_agent_summary.csv",
        f"{old_stem}_network_agent_summary_{args.date}_{args.distribution}.csv",
        f"{old_stem}_{args.date}_network_agent_summary_{args.distribution}.csv",
    ]
    return _find_existing_csv(args, description="agent summary CSV", candidate_names=candidate_names, glob_tokens=["agent_summary"])


def build_input_run_metrics_csv_path(args) -> str:
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_run_metrics.csv",
        f"{old_stem}_network_run_metrics_{args.date}_{args.distribution}.csv",
        f"{old_stem}_{args.date}_network_run_metrics_{args.distribution}.csv",
    ]
    return _find_existing_csv(args, description="run metrics CSV", candidate_names=candidate_names, glob_tokens=["run_metrics"])


def build_input_repair_events_csv_path(args) -> str:
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_repair_events.csv",
        "repair_events.csv",
        f"repair_events_{base_name}_seed{args.seed}.csv",
        f"repair_events_{base_name}.csv",
    ]
    return _find_existing_csv(args, description="repair events CSV", candidate_names=candidate_names, glob_tokens=["repair_events"])


def build_input_rag_retrieval_csv_path(args) -> str:
    """Find the RAG retrieval CSV under either compact or legacy result folders."""
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"rag_retrieval_{short}.csv",
        f"rag_retrieval_{base_name}_seed{args.seed}.csv",
        f"rag_retrieval_{base_name}.csv",
    ]
    return _find_existing_csv(args, description="RAG retrieval CSV", candidate_names=candidate_names, glob_tokens=["rag_retrieval"])


def build_input_interactions_csv_path(args) -> str:
    """Find interactions CSV from compact qwen13+ or legacy runtime names."""
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    short = _short_run_base_from_args(args)
    candidate_names = [
        f"{short}_interactions.csv",
        f"{old_stem}_network_interactions_{args.date}_{args.distribution}.csv",
        f"{old_stem}_{args.date}_network_interactions_{args.distribution}.csv",
    ]
    path = _find_existing_csv(args, description="interactions CSV", candidate_names=candidate_names, glob_tokens=["interactions", "network_interactions"])
    print(f"[markov] Using interactions CSV: {path}")
    return path

def _build_plot_subdir(csv_path: str, args) -> str:
    model_name = sanitize_model_name_for_path(args.model_name)
    experiment_id = "Flache_2017"
    version_root = extract_version_root(args.version_set)


    csv_basename = os.path.basename(csv_path)
    csv_stem, _ = os.path.splitext(csv_basename)
    csv_stem = csv_stem.replace("_opinion_change", "")

    out_dir = os.path.join(
        "results",
        "opinion_dynamics",
        experiment_id,
        model_name,
        "plots",
        version_root,
        csv_stem,
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir



def _get_run_stem(csv_path: str) -> str:
    csv_basename = os.path.basename(csv_path)
    csv_stem, _ = os.path.splitext(csv_basename)
    # Legacy runtime used *_network_opinion_change; qwen13+ uses *_opinion_change.
    csv_stem = csv_stem.replace("_network_opinion_change", "")
    csv_stem = csv_stem.replace("_opinion_change", "")
    return csv_stem

def build_output_timeseries_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    out_file = f"{run_stem}_timeseries.{args.figure_file_type}"
    return os.path.join(out_dir, out_file)


def build_output_histogram_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    out_file = f"{run_stem}_final_distribution.{args.figure_file_type}"
    return os.path.join(out_dir, out_file)


def build_output_initial_histogram_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    out_file = f"{run_stem}_initial_distribution.{args.figure_file_type}"
    return os.path.join(out_dir, out_file)

def build_output_step50_distribution_path(args, csv_path: str) -> str:
    """
    Step-50 distribution snapshot saved alongside initial/final distribution plots.
    Always saved as PNG (per your request), independent of --figure_file_type.
    """
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_step50_distribution.png")


def build_output_bdp_timeseries_path(args, csv_path: str) -> str:
    """
    Build path for CSV storing B, D, P at every time step.
    """
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    out_file = f"{run_stem}_BDP_timeseries.csv"
    return os.path.join(out_dir, out_file)

def build_output_bdp_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    out_file = f"{run_stem}_BDP_timeseries.{args.figure_file_type}"
    return os.path.join(out_dir, out_file)

def build_output_distribution_frames_dir(args, csv_path: str) -> str:
    """
    Directory for histogram frames used in the distribution GIF.
    """
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    frames_dir = os.path.join(out_dir, f"{run_stem}_distribution_frames")
    os.makedirs(frames_dir, exist_ok=True)
    return frames_dir


def build_output_distribution_gif_path(args, csv_path: str) -> str:
    """
    GIF showing opinion distribution evolution.
    """
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_opinion_distribution_evolution.gif")

def build_output_markov_counts_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_counts.csv")


def build_output_markov_probs_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_probs.csv")


def build_output_markov_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_matrix.{args.figure_file_type}")

def build_output_markov_counts_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(
        out_dir,
        f"{run_stem}_markov_transition_counts.{args.figure_file_type}"
    )


def build_output_step_movement_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_step_movement_intensity.{args.figure_file_type}")


def build_output_cumulative_drift_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_cumulative_drift.{args.figure_file_type}")


def build_output_profile_trajectory_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_trajectory_by_profile.{args.figure_file_type}")


def build_output_final_distribution_by_profile_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_final_distribution_by_profile.{args.figure_file_type}")


def build_output_movement_by_profile_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_movement_by_profile.{args.figure_file_type}")


def build_output_agent_small_multiples_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_agent_small_multiples.{args.figure_file_type}")


def build_output_markov_window_counts_path(args, csv_path: str, label: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_counts_{label}.csv")


def build_output_markov_window_probs_path(args, csv_path: str, label: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_probs_{label}.csv")


def build_output_markov_window_plot_path(args, csv_path: str, label: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_matrix_{label}.{args.figure_file_type}")


def build_output_markov_window_counts_plot_path(args, csv_path: str, label: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_markov_transition_counts_{label}.{args.figure_file_type}")


def build_output_cue_metrics_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_step2_cue_concentration.csv")


def build_output_cue_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_step2_cue_concentration.{args.figure_file_type}")


def build_output_repair_timeseries_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_repair_events_over_time.{args.figure_file_type}")


def build_output_repair_tags_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_repair_tag_counts.{args.figure_file_type}")


def build_output_repair_tags_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_repair_tag_counts.csv")

def build_output_source_provenance_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_source_provenance.{args.figure_file_type}")


def build_output_source_provenance_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_source_provenance.csv")


# Backward-compatible aliases for earlier patch names / IDE stale references
def build_output_source_provenance_path(args, csv_path: str) -> str:
    return build_output_source_provenance_plot_path(args, csv_path)

def _build_output_source_provenance_plot_path(args, csv_path: str) -> str:
    return build_output_source_provenance_plot_path(args, csv_path)

def _build_output_source_provenance_csv_path(args, csv_path: str) -> str:
    return build_output_source_provenance_csv_path(args, csv_path)


def build_output_individual_trajectories_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_individual_belief_trajectories.{args.figure_file_type}")


def build_output_persona_trait_delta_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_persona_trait_delta.csv")


def build_output_persona_trait_delta_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_persona_trait_delta.{args.figure_file_type}")


def build_output_rag_direction_breakdown_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_rag_direction_breakdown.csv")


def build_output_rag_direction_breakdown_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_rag_direction_breakdown.{args.figure_file_type}")


def build_output_ego_network_timeline_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_ego_network_belief_timelines.{args.figure_file_type}")


def build_output_edge_belief_diff_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_edge_belief_differential_heatmap.csv")


def build_output_edge_belief_diff_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_edge_belief_differential_heatmap.{args.figure_file_type}")


def build_output_run_summary_card_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_summary_card.csv")


def build_output_run_summary_card_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_summary_card.{args.figure_file_type}")


def build_output_agent_neighborhood_alignment_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_agent_neighborhood_alignment.csv")


def build_output_agent_neighborhood_alignment_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_agent_neighborhood_alignment.{args.figure_file_type}")



def build_output_paired_distribution_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_initial_final_distribution_pair.{args.figure_file_type}")


def build_output_exposure_ecology_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_exposure_ecology.csv")


def build_output_exposure_ecology_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_exposure_ecology.{args.figure_file_type}")


def build_output_final_belief_strip_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_final_belief_strip_by_trait.{args.figure_file_type}")

def plot_opinion_trajectories(
    csv_path: str,
    out_path: str,
    no_annotation: bool,
    fig_type: str,
    num_agents: int,
):
    df = pd.read_csv(csv_path)

    if "time_step" not in df.columns:
        raise ValueError(f"'time_step' column not found in {csv_path}")

    time_steps = df["time_step"].values
    agent_cols = list(df.columns)[-num_agents:]
    opinions = df[agent_cols].to_numpy()

    # ---- compute B, D, P time series ----
    B_series = opinions.mean(axis=1)
    D_series = opinions.std(axis=1, ddof=0)
    var = D_series ** 2
    denom = var + B_series ** 2
    P_series = np.where(denom == 0, 0.0, (var - B_series ** 2) / denom)

    # ---- colors by initial opinion ----
    opinion_colors = {
        2: "#0066FF",
        1: "#66C2FF",
        0: "#999999",
        -1: "#FF8080",
        -2: "#E60000",
    }

    plt.figure(figsize=(5, 4) if fig_type == "png" else (4, 3))
    displacement = 0.03

    # ---- agent trajectories ----
    for i in range(num_agents):
        initial_opinion = opinions[0, i]
        color = opinion_colors.get(initial_opinion, "#000000")

        y = []
        current_val = initial_opinion
        for t in range(len(time_steps)):
            if t > 0 and opinions[t, i] != opinions[t - 1, i]:
                current_val = opinions[t, i]
            y.append(current_val + i * displacement)

        plt.plot(time_steps, y, linewidth=2, color=color, alpha=0.85)

    # ---- mean opinion line ----
    plt.plot(
        time_steps,
        B_series,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Mean opinion (B)",
    )

    plt.yticks([-2, -1, 0, 1, 2])
    plt.ylim(-2.5, 2.5)
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)

    step = max(1, len(time_steps) // 10)
    plt.xticks(time_steps[::step])

    if not no_annotation:
        plt.xlabel("Time step")
        plt.ylabel("Opinion")
        plt.title("Evolution of Agent Opinions")
        plt.legend(loc="lower right")

        # ---- BDP annotation (final values) ----
        text = (
            f"B = {B_series[-1]:.2f}\n"
            f"D = {D_series[-1]:.2f}\n"
            f"P = {P_series[-1]:.2f}"
        )
        plt.gca().text(
            0.02,
            0.98,
            text,
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()



def compute_B_D_P(df, agent_cols, row_index):
    opinions = df.iloc[row_index][agent_cols].values.astype(float)
    B = opinions.mean()
    D = opinions.std(ddof=0)
    var = D ** 2
    denom = var + B ** 2
    P = 0.0 if denom == 0 else (var - B ** 2) / denom
    return opinions, B, D, P

def compute_markov_transition_matrix_from_interactions(interactions_csv_path: str, step_range=None):
    """
    Build one 5x5 Markov transition matrix from a single interactions CSV.

    Uses ONLY listener updates:
      Agent_I Pre-Belief -> Agent_I Post-Belief
    """
    df = pd.read_csv(interactions_csv_path)

    pre_col = "Agent_I Pre-Belief"
    post_col = "Agent_I Post-Belief"

    if pre_col not in df.columns or post_col not in df.columns:
        raise ValueError(
            f"Missing required columns in {interactions_csv_path}.\n"
            f"Need: '{pre_col}' and '{post_col}'.\n"
            f"Found: {list(df.columns)}"
        )
    
    

    states = [-2, -1, 0, 1, 2]
    idx = {s: i for i, s in enumerate(states)}

    # Clean + filter
    step_col = None
    for cand in ["Time Step", "time_step", "TimeStep"]:
        if cand in df.columns:
            step_col = cand
            break
    if step_range is not None and step_col is not None:
        lo, hi = step_range
        df = df[(df[step_col] >= lo) & (df[step_col] <= hi)].copy()

    sub = df[[pre_col, post_col]].dropna().copy()
    sub[pre_col] = sub[pre_col].astype(int)
    sub[post_col] = sub[post_col].astype(int)
    sub = sub[sub[pre_col].isin(states) & sub[post_col].isin(states)]

    counts = np.zeros((5, 5), dtype=int)
    for a, b in zip(sub[pre_col].to_numpy(), sub[post_col].to_numpy()):
        counts[idx[a], idx[b]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums != 0)

    counts_df = pd.DataFrame(counts, index=states, columns=states)
    probs_df = pd.DataFrame(probs, index=states, columns=states)
    return counts_df, probs_df

def plot_markov_counts_matrix(counts_df: pd.DataFrame, out_path: str, fig_type: str, title: str = "Empirical Markov Transition Counts"):
    """
    Heatmap of RAW transition counts.
    """
    fig, ax = plt.subplots(figsize=(5, 4) if fig_type == "png" else (4, 3), dpi=150)
    im = ax.imshow(counts_df.values)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(counts_df.columns.tolist())
    ax.set_yticklabels(counts_df.index.tolist())
    ax.set_xlabel("Next rating (t+1)")
    ax.set_ylabel("Current rating (t)")
    ax.set_title(title)

    # annotate cells with integers
    for i in range(5):
        for j in range(5):
            ax.text(
                j, i,
                f"{int(counts_df.values[i, j])}",
                ha="center", va="center", fontsize=8
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_with_BDP(opinions, B, D, P, out_path, fig_type, title):
    """Histogram with B/D/P panel rendered inside the plot.

    Notes (important for GIF creation):
    - Fixed canvas size (figsize + dpi) to keep all frames identical pixel dimensions.
    - Do NOT use bbox_inches='tight' when saving, because it can change the output size
      depending on title length and tick label extents.
    """
    bins = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    centers = np.array([-2, -1, 0, 1, 2])

    # Fixed canvas => stable frame sizes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.20, top=0.86)
    
    ax.hist(
        opinions,
        bins=bins,
        rwidth=0.8,
        align="mid",
        color="lightgray",
        edgecolor="black",
    )

    ax.set_xlabel("Opinion (o)")
    ax.set_ylabel("# Agents")
    ax.set_xticks(centers)
    ax.set_title(title)

    # Mean (bias)
    ax.axvline(B, color="red", linestyle="--", linewidth=1)

    # B/D/P panel INSIDE axes
    text_str = f"B = {B: .2f}\nD = {D: .2f}\nP = {P: .2f}"
    ax.text(
        0.98, 0.98,
        text_str,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        zorder=10,
    )

    fig.tight_layout()
    fig.savefig(out_path)  # no bbox_inches='tight'
    plt.close(fig)


def plot_initial_distribution(csv_path: str, out_path: str, num_agents: int, fig_type: str):
    df = pd.read_csv(csv_path)
    all_cols = list(df.columns)
    agent_cols = all_cols[-num_agents:]
    opinions, B, D, P = compute_B_D_P(df, agent_cols, 0)
    plot_distribution_with_BDP(opinions, B, D, P, out_path, fig_type, "Initial Opinion Distribution")


def plot_final_distribution(csv_path: str, out_path: str, num_agents: int, fig_type: str):
    df = pd.read_csv(csv_path)
    all_cols = list(df.columns)
    agent_cols = all_cols[-num_agents:]
    opinions, B, D, P = compute_B_D_P(df, agent_cols, len(df) - 1)
    plot_distribution_with_BDP(opinions, B, D, P, out_path, fig_type, "Final Opinion Distribution")
    
    
def save_BDP_timeseries(csv_path: str, out_path: str, num_agents: int) -> None:
    df = pd.read_csv(csv_path)
    all_cols = list(df.columns)
    agent_cols = all_cols[-num_agents:]

    records = []
    for row_index in range(len(df)):
        _, B, D, P = compute_B_D_P(df, agent_cols, row_index)
        time_step = int(df.iloc[row_index]["time_step"])
        records.append({"time_step": time_step, "B": B, "D": D, "P": P})

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    print(f"Saved B,D,P time series to: {out_path}")
    
    
def plot_BDP_timeseries_from_csv(bdp_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(bdp_csv_path)

    plt.figure(figsize=(4, 4) if fig_type == "png" else (3.2, 3.2))

    t = df["time_step"]

    plt.subplot(3, 1, 1)
    plt.plot(t, df["B"], linewidth=1.5)
    plt.ylabel("B (mean)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(t, df["D"], linewidth=1.5)
    plt.ylabel("D (diversity)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(t, df["P"], linewidth=1.5)
    plt.ylabel("P (polarization)")
    plt.xlabel("Time step")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved B,D,P timeseries plot to: {out_path}")
    
def plot_step_movement_from_summary(step_summary_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(step_summary_csv_path)
    if df.empty:
        return

    t = df["time_step"]
    fig, axes = plt.subplots(3, 1, figsize=(6, 6) if fig_type == "png" else (4.8, 4.8), sharex=True, dpi=150)

    axes[0].plot(t, df["moves_total"], label="moves_total", linewidth=1.6)
    if "repair_events" in df.columns:
        axes[0].plot(t, df["repair_events"], label="repair_events", linewidth=1.2)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-step movement intensity")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t, df["net_delta"], label="net_delta", linewidth=1.6)
    axes[1].plot(t, df["abs_total_delta"], label="abs_total_delta", linewidth=1.6)
    axes[1].set_ylabel("Delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    if {"B_before", "B_after"}.issubset(df.columns):
        axes[2].plot(t, df["B_before"], label="B_before", linewidth=1.2)
        axes[2].plot(t, df["B_after"], label="B_after", linewidth=1.6)
    axes[2].set_ylabel("B")
    axes[2].set_xlabel("Time step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved step movement plot to: {out_path}")


def plot_cumulative_drift_from_summary(step_summary_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(step_summary_csv_path).sort_values("time_step").reset_index(drop=True)
    if df.empty:
        return

    pos = df["net_delta"].clip(lower=0).cumsum()
    neg = (-df["net_delta"].clip(upper=0)).cumsum()
    net = df["net_delta"].cumsum()

    fig, ax = plt.subplots(figsize=(5.5, 4) if fig_type == "png" else (4.4, 3.2), dpi=150)
    ax.plot(df["time_step"], pos, label="cumulative positive moves", linewidth=1.8)
    ax.plot(df["time_step"], neg, label="cumulative negative moves", linewidth=1.8)
    ax.plot(df["time_step"], net, label="cumulative net delta", linewidth=2.0)
    ax.set_title("Cumulative drift")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative change")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cumulative drift plot to: {out_path}")


def save_source_provenance_counts(step_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(step_summary_csv_path)
    if "observable_primary_source" not in df.columns:
        raise ValueError("Step summary CSV does not contain observable_primary_source.")
    counts = df["observable_primary_source"].fillna("tweet").astype(str).str.strip()
    counts = counts.replace({"": "tweet"}).value_counts().rename_axis("observable_primary_source").reset_index(name="count")
    total = int(counts["count"].sum()) if not counts.empty else 0
    counts["share"] = counts["count"] / total if total else 0.0
    counts.to_csv(out_csv_path, index=False)
    return counts


def plot_source_provenance_from_summary(step_summary_csv_path: str, out_path: str, out_csv_path: str, fig_type: str) -> None:
    counts = save_source_provenance_counts(step_summary_csv_path, out_csv_path)
    if counts.empty:
        raise ValueError("No provenance rows available.")
    plt.figure(figsize=(8, 6))
    plt.pie(counts["count"], labels=counts["observable_primary_source"], autopct="%1.1f%%")
    plt.title("Observable primary source provenance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _profile_series_map(agent_summary_df: pd.DataFrame) -> dict:
    if agent_summary_df.empty or "agent_name" not in agent_summary_df.columns:
        return {}
    profile_col = "epistemic_profile" if "epistemic_profile" in agent_summary_df.columns else None
    out = {}
    for _, row in agent_summary_df.iterrows():
        name = str(row.get("agent_name", "")).strip()
        profile = str(row.get(profile_col, "") if profile_col else "").strip() or "unknown"
        out[name] = profile
    return out


def plot_trajectory_by_profile(csv_path: str, agent_summary_csv_path: str, out_path: str, fig_type: str, num_agents: int) -> None:
    df = pd.read_csv(csv_path)
    agents = pd.read_csv(agent_summary_csv_path)
    if df.empty or agents.empty:
        return
    time_steps = df["time_step"].values
    agent_cols = list(df.columns)[-num_agents:]
    profile_map = _profile_series_map(agents)
    profiles = {}
    for col in agent_cols:
        profiles.setdefault(profile_map.get(col, "unknown"), []).append(col)
    fig, ax = plt.subplots(figsize=(6, 4) if fig_type == "png" else (4.8, 3.2), dpi=150)
    for profile, cols in sorted(profiles.items()):
        arr = df[cols].to_numpy(dtype=float)
        mean_series = arr.mean(axis=1)
        ax.plot(time_steps, mean_series, linewidth=2.0, label=f"{profile} (n={len(cols)})")
    ax.set_title("Mean trajectory by epistemic profile")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean opinion")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory-by-profile plot to: {out_path}")


def plot_final_distribution_by_profile(agent_summary_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(agent_summary_csv_path)
    if df.empty or "final_rating" not in df.columns:
        return
    profile_col = "epistemic_profile" if "epistemic_profile" in df.columns else None
    if not profile_col:
        return
    ratings = [-2, -1, 0, 1, 2]
    pivot = (
        df.assign(epistemic_profile=df[profile_col].fillna("unknown").replace("", "unknown"))
          .groupby(["epistemic_profile", "final_rating"]).size().unstack(fill_value=0)
          .reindex(columns=ratings, fill_value=0)
    )
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(7, 4) if fig_type == "png" else (5.6, 3.2), width=0.85)
    ax.set_title("Final distribution by epistemic profile")
    ax.set_xlabel("Epistemic profile")
    ax.set_ylabel("Count")
    ax.legend(title="Final rating", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved final-distribution-by-profile plot to: {out_path}")


def plot_movement_by_profile(step_summary_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(step_summary_csv_path)
    if df.empty or "listener_epistemic_profile" not in df.columns or "move_direction" not in df.columns:
        return
    pivot = (
        df.assign(listener_epistemic_profile=df["listener_epistemic_profile"].fillna("unknown").replace("", "unknown"))
          .groupby(["listener_epistemic_profile", "move_direction"]).size().unstack(fill_value=0)
          .reindex(columns=["positive", "negative", "none"], fill_value=0)
    )
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", stacked=True, figsize=(7, 4) if fig_type == "png" else (5.6, 3.2), width=0.85)
    ax.set_title("Movement by listener epistemic profile")
    ax.set_xlabel("Epistemic profile")
    ax.set_ylabel("Count")
    ax.legend(title="Move direction", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved movement-by-profile plot to: {out_path}")


def plot_agent_small_multiples(csv_path: str, agent_summary_csv_path: str, out_path: str, fig_type: str, num_agents: int) -> None:
    df = pd.read_csv(csv_path)
    agents = pd.read_csv(agent_summary_csv_path)
    if df.empty or agents.empty:
        return
    agent_cols = list(df.columns)[-num_agents:]
    time_steps = df["time_step"].values
    n = len(agent_cols)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    meta = agents.set_index("agent_name") if "agent_name" in agents.columns else pd.DataFrame()
    for ax, name in zip(axes, agent_cols):
        series = df[name].to_numpy(dtype=float)
        ax.plot(time_steps, series, linewidth=1.6)
        profile = "unknown"
        if not meta.empty and name in meta.index and "epistemic_profile" in meta.columns:
            profile = str(meta.loc[name, "epistemic_profile"] or "unknown")
        initial = series[0] if len(series) else np.nan
        final = series[-1] if len(series) else np.nan
        ax.set_title(f"{name}\n{profile}\n{initial:.0f}→{final:.0f}", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-2.2, 2.2)
        ax.set_yticks([-2, -1, 0, 1, 2])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Agent trajectories by epistemic profile", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved agent small-multiples plot to: {out_path}")


def markov_windows_from_max_step(max_step: int):
    max_step = int(max(1, max_step))
    cut1 = int(np.ceil(max_step / 3.0))
    cut2 = int(np.ceil((2.0 * max_step) / 3.0))
    windows = [("early", 1, cut1)]
    if cut1 + 1 <= cut2:
        windows.append(("mid", cut1 + 1, cut2))
    if cut2 + 1 <= max_step:
        windows.append(("late", cut2 + 1, max_step))
    return windows


def plot_markov_matrix(probs_df: pd.DataFrame, out_path: str, fig_type: str, title: str = "Empirical Markov Transition Matrix"):
    """
    Simple heatmap (default matplotlib colormap).
    """
    fig, ax = plt.subplots(figsize=(5, 4) if fig_type == "png" else (4, 3), dpi=150)
    im = ax.imshow(probs_df.values)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(probs_df.columns.tolist())
    ax.set_yticklabels(probs_df.index.tolist())
    ax.set_xlabel("Next rating (t+1)")
    ax.set_ylabel("Current rating (t)")
    ax.set_title(title)

    # annotate cells
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{probs_df.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    
def make_distribution_frames(
    df: pd.DataFrame,
    agent_cols,
    frames_dir: str,
    step50_out_path: str,
    step_interval: int = 5,
):
    """
    Generate histogram frames for the opinion distribution at multiple steps,
    using the SAME styling as plot_initial_distribution / plot_final_distribution.

    Also saves a single snapshot: step50_distribution.png next to initial/final plots
    (i.e., NOT inside the frames directory), if time_step == 50 exists.
    """
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []
    saved_step50 = False

    for row_index in range(0, len(df), step_interval):
        opinions, B, D, P = compute_B_D_P(df, agent_cols, row_index)
        time_step = int(df.iloc[row_index]["time_step"])

        # Frame (saved into frames_dir for the GIF)
        frame_path = os.path.join(frames_dir, f"dist_step_{time_step:03d}.png")
        plot_distribution_with_BDP(
            opinions=opinions,
            B=B,
            D=D,
            P=P,
            out_path=frame_path,
            fig_type="png",  # frames are always PNG
            title=f"Distribution at step {time_step}",
        )
        frame_paths.append(frame_path)

        # Step-50 snapshot saved alongside initial/final plots
        if (not saved_step50) and (time_step == 50):
            plot_distribution_with_BDP(
                opinions=opinions,
                B=B,
                D=D,
                P=P,
                out_path=step50_out_path,
                fig_type="png",
                title="Distribution at step 50",
            )
            saved_step50 = True

    return frame_paths




def make_distribution_gif(frame_paths, out_path: str, fps: int = 2):
    """Convert a list of PNG frames into a GIF.

    If any frames end up with slightly different pixel sizes (e.g., from older cached
    frames or environment differences), we pad them to a common (H, W) before writing.
    """
    if not frame_paths:
        print("No frames to build GIF.")
        return

    images = [imageio.imread(fp) for fp in frame_paths]
    if not images:
        print("No frames to build GIF.")
        return

    # Normalize shapes by padding to max height/width
    import numpy as _np

    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    max_h, max_w = max(heights), max(widths)

    padded = []
    for im in images:
        h, w = im.shape[0], im.shape[1]
        pad_h = max_h - h
        pad_w = max_w - w

        if pad_h == 0 and pad_w == 0:
            padded.append(im)
            continue

        # Create a white canvas
        if im.ndim == 2:
            canvas = _np.full((max_h, max_w), 255, dtype=im.dtype)
            canvas[:h, :w] = im
        else:
            c = im.shape[2]
            canvas = _np.full((max_h, max_w, c), 255, dtype=im.dtype)
            canvas[:h, :w, :] = im

        padded.append(canvas)

    imageio.mimsave(out_path, padded, fps=fps)
    print(f"Saved GIF to: {out_path}")


def save_step2_cue_metrics_from_run_metrics(run_metrics_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(run_metrics_csv_path)
    if df.empty or "Metric" not in df.columns or "Count" not in df.columns:
        out = pd.DataFrame(columns=["polarity", "cue_label", "count"])
        out.to_csv(out_csv_path, index=False)
        return out

    sub = df[df["Metric"].astype(str).str.startswith("step2_cue::")].copy()
    rows = []
    for _, r in sub.iterrows():
        metric = str(r["Metric"])
        count = int(r["Count"])
        label = metric.split("step2_cue::", 1)[1]
        if label == "none":
            polarity = "none"
            cue_label = "none"
        elif "::" in label:
            polarity, cue_label = label.split("::", 1)
        else:
            polarity, cue_label = "unknown", label
        rows.append({"polarity": polarity, "cue_label": cue_label, "count": count})

    out = pd.DataFrame(rows).sort_values(["polarity", "count", "cue_label"], ascending=[True, False, True])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved step2 cue concentration CSV to: {out_csv_path}")
    return out


def plot_step2_cue_concentration(cue_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if cue_df is None or cue_df.empty:
        return
    cue_df = cue_df.copy()
    cue_df["label"] = cue_df["polarity"].astype(str) + "::" + cue_df["cue_label"].astype(str)
    cue_df = cue_df.sort_values("count", ascending=True)

    fig_h = max(3.5, 0.35 * len(cue_df))
    fig, ax = plt.subplots(figsize=(7, fig_h) if fig_type == "png" else (5.6, min(fig_h, 8)), dpi=150)
    ax.barh(cue_df["label"], cue_df["count"])
    ax.set_title("Step-2 cue concentration")
    ax.set_xlabel("Count")
    ax.set_ylabel("Cue")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved step2 cue concentration plot to: {out_path}")


def save_repair_tag_counts(repair_events_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(repair_events_csv_path)
    if df.empty or "Repair Tags" not in df.columns:
        out = pd.DataFrame(columns=["repair_tag", "count"])
        out.to_csv(out_csv_path, index=False)
        return out

    counts = {}
    for raw in df["Repair Tags"].fillna(""):
        for tag in [t.strip() for t in str(raw).split("|") if t.strip()]:
            counts[tag] = counts.get(tag, 0) + 1
    out = pd.DataFrame(sorted(counts.items(), key=lambda x: (-x[1], x[0])), columns=["repair_tag", "count"])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved repair tag counts CSV to: {out_csv_path}")
    return out


def plot_repair_events_over_time(repair_events_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(repair_events_csv_path)
    if df.empty or "Time Step" not in df.columns:
        return
    counts = df.groupby(["Time Step", "Prompt Step"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(6, 4) if fig_type == "png" else (4.8, 3.2), dpi=150)
    for col in counts.columns:
        ax.plot(counts.index, counts[col], label=str(col), linewidth=1.5)
    ax.set_title("Repair events over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Repair count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved repair-events-over-time plot to: {out_path}")


def plot_repair_tag_counts(repair_tag_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if repair_tag_df is None or repair_tag_df.empty:
        return
    top = repair_tag_df.head(20).iloc[::-1]
    fig_h = max(3.5, 0.28 * len(top))
    fig, ax = plt.subplots(figsize=(7, fig_h) if fig_type == "png" else (5.6, min(fig_h, 8)), dpi=150)
    ax.barh(top["repair_tag"], top["count"])
    ax.set_title("Top repair tags")
    ax.set_xlabel("Count")
    ax.set_ylabel("Repair tag")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved repair tag count plot to: {out_path}")


# -----------------------------
# Network frames + GIF (all-in-one; replaces make_network_slideshow.py)
# -----------------------------


def _collect_edge_files(results_dir: str, out_prefix: str, version_set: str) -> list[str]:
    """Collect edge CSVs for compact qwen13+ and legacy naming schemes."""
    patterns = []
    if version_set:
        patterns.extend([
            os.path.join(results_dir, f"{out_prefix}_step_*_{version_set}_edges.csv"),
            os.path.join(results_dir, f"{out_prefix}_step_*_edges.csv"),
            # qwen13+ compact: <short_run_base>_step_000_edges.csv
            os.path.join(results_dir, "*step_*_edges.csv"),
        ])
    else:
        patterns.extend([
            os.path.join(results_dir, f"{out_prefix}_step_*_edges.csv"),
            os.path.join(results_dir, "*step_*_edges.csv"),
        ])
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    def _step_key(p: str) -> tuple[int, str]:
        base = os.path.basename(p)
        m = re.search(r"_step_(\d+)", base)
        return (int(m.group(1)) if m else 10**9, base)

    return sorted(set(files), key=_step_key)

def _opinion_to_color(o) -> str:
    """Color mapping consistent with your histogram scheme."""
    try:
        o = int(o)
    except Exception:
        return "black"

    if o <= -2:
        return "#d73027"   # strong negative
    if o == -1:
        return "#fc8d59"   # mild negative
    if o == 0:
        return "#cccccc"   # neutral
    if o == 1:
        return "#91bfdb"   # mild positive
    if o >= 2:
        return "#4575b4"   # strong positive


def _pad_images_to_same_shape(images: list[np.ndarray]) -> list[np.ndarray]:
    """Pad images to a common (max_h, max_w) with white background."""
    if not images:
        return images
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        h, w = im.shape[:2]
        if h == max_h and w == max_w:
            padded.append(im)
            continue
        if im.ndim == 2:
            canvas = np.full((max_h, max_w), 255, dtype=im.dtype)
            canvas[:h, :w] = im
        else:
            c = im.shape[2]
            canvas = np.full((max_h, max_w, c), 255, dtype=im.dtype)
            canvas[:h, :w, :] = im
        padded.append(canvas)
    return padded


def generate_network_frames_and_gif(
    csv_path: str,
    plot_dir: str,
    out_prefix: str,
    version_set: str,
    num_agents: int,
    frame_duration: float = 0.5,
    num_fades: int = 2,
) -> tuple[str | None, str | None]:
    """
    Generate:
      - network_frames/frame_step_###.png
      - <out_prefix>_network.gif
    in the SAME folder as your other plots (plot_dir).

    IMPORTANT:
    - out_prefix should be args.output_file (i.e., what you pass with --out)
    - edge CSVs are expected next to the opinion_change CSV (same results dir)
    """
    results_dir = os.path.dirname(csv_path)

    # 1) Load opinions
    df_op = pd.read_csv(csv_path)
    if "time_step" in df_op.columns:
        df_op = df_op.set_index("time_step")
    df_op = df_op.loc[:, ~df_op.columns.str.startswith("Unnamed")]
    agent_names = list(df_op.columns)[-num_agents:]

    # 2) Collect edge files for this run
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    if not edge_files:
        print("[network] No edge files found; skipping network GIF.")
        return None, None

    # 3) Fixed layout from first graph
    first_edges = pd.read_csv(edge_files[0])
    G0 = nx.Graph()
    for _, row in first_edges.iterrows():
        G0.add_edge(row["i_name"], row["j_name"])
    for name in agent_names:
        if name not in G0:
            G0.add_node(name)
    pos = nx.kamada_kawai_layout(G0)

    # 4) Frames dir INSIDE plot_dir (next to other plots)
    run_stem = _get_run_stem(csv_path)
    frames_dir = os.path.join(plot_dir, f"{run_stem}_network_frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths: list[str] = []

    for edge_path in edge_files:
        base = os.path.basename(edge_path)
        m = re.search(r"_step_(\d+)", base)
        if not m:
            continue
        step_t = int(m.group(1))          # 0-based in filename
        time_step = step_t + 1            # your opinion CSV is 1-based

        if time_step not in df_op.index:
            # Skip silently; your sim sometimes doesn't log every step in edges
            continue

        row_op = df_op.loc[time_step]

        df_edges = pd.read_csv(edge_path)
        G = nx.Graph()
        for _, r in df_edges.iterrows():
            G.add_edge(r["i_name"], r["j_name"])
        for name in agent_names:
            if name not in G:
                G.add_node(name)

        node_colors = []
        for n in G.nodes():
            if n in row_op.index:
                node_colors.append(_opinion_to_color(row_op[n]))
            else:
                node_colors.append("black")

        deg = dict(G.degree())
        deg_values = [int(deg.get(n, 0)) for n in G.nodes()]
        dmin = min(deg_values) if deg_values else 0
        dmax = max(deg_values) if deg_values else 0

        def _scale_node_size(d: int) -> float:
            if dmax <= dmin:
                return 380.0
            min_size, max_size = 220.0, 1050.0
            return float(min_size + (float(d) - float(dmin)) * (max_size - min_size) / float(dmax - dmin))

        node_sizes = [_scale_node_size(int(deg.get(n, 0))) for n in G.nodes()]
        hub_cut = None
        if deg_values:
            top_hub_idx = max(0, min(len(deg_values) - 1, max(0, int(len(deg_values) * 0.2) - 1)))
            hub_cut = sorted(deg_values, reverse=True)[top_hub_idx]
        edgecolors = ["#e5e7eb" if hub_cut is None or int(deg.get(n, 0)) < int(hub_cut) else "#facc15" for n in G.nodes()]
        linewidths = [0.8 if hub_cut is None or int(deg.get(n, 0)) < int(hub_cut) else 1.8 for n in G.nodes()]

        # Fixed canvas (consistent frame size)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        fig.subplots_adjust(left=0.22, right=0.97, bottom=0.20, top=0.86)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors=edgecolors, linewidths=linewidths, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

        ax.set_title(f"Network at step {time_step} (node size ∝ degree)")
        ax.text(
            0.02,
            0.98,
            "node size ∝ degree; gold outline = actual top-degree hubs",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="#cbd5e1"),
        )
        ax.axis("off")

        # Avoid bbox_inches='tight' to keep identical image sizes
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

        frame_file = os.path.join(frames_dir, f"frame_step_{step_t:03d}.png")
        fig.savefig(frame_file)
        plt.close(fig)
        frame_paths.append(frame_file)

    if not frame_paths:
        print("[network] No frames generated; skipping network GIF.")
        return frames_dir, None

    # 5) Build GIF with optional fade transitions
    run_stem = _get_run_stem(csv_path)
    gif_path = os.path.join(plot_dir, f"{run_stem}_network.gif")

    base_images = [imageio.imread(fp).astype("float32") for fp in frame_paths]
    images = []
    for i in range(len(base_images)):
        images.append(base_images[i].astype("uint8"))
        if i < len(base_images) - 1 and num_fades > 0:
            for k in range(1, num_fades + 1):
                alpha = k / (num_fades + 1)
                blended = (1 - alpha) * base_images[i] + alpha * base_images[i + 1]
                images.append(blended.astype("uint8"))

    # Safety padding (should be unnecessary with fixed canvas, but keeps it bulletproof)
    images = _pad_images_to_same_shape(images)
    imageio.mimsave(gif_path, images, duration=frame_duration)
    print(f"[network] Saved frames in: {frames_dir}")
    print(f"[network] Saved GIF: {gif_path}")

    return frames_dir, gif_path

def generate_network_3d_html(
    csv_path: str,
    plot_dir: str,
    out_prefix: str,
    version_set: str,
    num_agents: int,
    step_stride: int = 1,
    layout_seed: int = 42,
) -> str | None:
    """
    Build ONE interactive Plotly 3D HTML with a slider over time steps.

    Inputs match your existing network GIF function:
    - csv_path: the *_network_opinion_change_*.csv
    - plot_dir: the same plot folder where you save other figures
    - out_prefix: args.output_file (or seed{seed})
    - version_set: args.version_set
    - num_agents: args.num_agents

    Output:
      <plot_dir>/<out_prefix>_network3d.html
    """
    results_dir = os.path.dirname(csv_path)

    # 1) Load opinions (same as your GIF function)
    df_op = pd.read_csv(csv_path)
    if "time_step" in df_op.columns:
        df_op = df_op.set_index("time_step")
    df_op = df_op.loc[:, ~df_op.columns.str.startswith("Unnamed")]
    agent_names = list(df_op.columns)[-num_agents:]

    # 2) Collect edge files (reuse your existing collector)
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    if not edge_files:
        print("[network3d] No edge files found; skipping 3D HTML.")
        return None

    # Optionally downsample for performance
    if step_stride > 1:
        edge_files = edge_files[::step_stride]

    # 3) Build a base graph for a FIXED 3D layout
    #    Use union of nodes + edges from first edge file (consistent with your 2D approach),
    #    then spring_layout(dim=3) for 3D coordinates.
    first_edges = pd.read_csv(edge_files[0])
    G0 = nx.Graph()
    for _, r in first_edges.iterrows():
        G0.add_edge(r["i_name"], r["j_name"])
    for name in agent_names:
        if name not in G0:
            G0.add_node(name)

    pos3d = nx.spring_layout(G0, dim=3, seed=layout_seed)

    # Utility: build Plotly trace arrays for a given graph + opinions row
    def _edge_xyz_lines(G: nx.Graph):
        xe, ye, ze = [], [], []
        for u, v in G.edges():
            x0, y0, z0 = pos3d[u]
            x1, y1, z1 = pos3d[v]
            xe += [x0, x1, None]
            ye += [y0, y1, None]
            ze += [z0, z1, None]
        return xe, ye, ze

    def _node_xyz_and_colors_and_sizes(G: nx.Graph, row_op: pd.Series):
        # Degree per node at this step
        deg = dict(G.degree())

        # Sizes: map degree range -> marker size range
        # Tune these two numbers if you want bigger/smaller nodes overall.
        min_size, max_size = 6, 20

        deg_values = [deg.get(n, 0) for n in agent_names]
        dmin, dmax = min(deg_values), max(deg_values)

        def _scale(d: int) -> float:
            if dmax == dmin:
                return float((min_size + max_size) / 2)
            return min_size + (d - dmin) * (max_size - min_size) / (dmax - dmin)

        xs, ys, zs, cs, labels, sizes = [], [], [], [], [], []
        for n in agent_names:
            x, y, z = pos3d.get(n, (0.0, 0.0, 0.0))
            xs.append(x); ys.append(y); zs.append(z)
            labels.append(str(n))

            if n in row_op.index:
                cs.append(_opinion_to_color(row_op[n]))
            else:
                cs.append("black")

            sizes.append(_scale(deg.get(n, 0)))

        return xs, ys, zs, cs, labels, sizes


    # 4) Create initial frame from the first edge file
    # Parse step from filename (same pattern you already use)
    def _time_step_from_edge_file(p: str) -> int | None:
        base = os.path.basename(p)
        m = re.search(r"_step_(\d+)", base)
        if not m:
            return None
        step_t = int(m.group(1))      # 0-based in filename
        return step_t + 1             # your opinion CSV is 1-based (same as your GIF)
    
    t0 = _time_step_from_edge_file(edge_files[0])
    if t0 is None or t0 not in df_op.index:
        # Find first usable
        t0 = None
        for p in edge_files:
            t = _time_step_from_edge_file(p)
            if t is not None and t in df_op.index:
                t0 = t
                break
        if t0 is None:
            print("[network3d] No edge files match opinion time steps; skipping 3D HTML.")
            return None

    # Build graph for initial frame
    def _graph_from_edge_file(edge_path: str) -> nx.Graph:
        df_edges = pd.read_csv(edge_path)
        G = nx.Graph()
        for _, r in df_edges.iterrows():
            G.add_edge(r["i_name"], r["j_name"])
        for name in agent_names:
            if name not in G:
                G.add_node(name)
        return G

    G_init = _graph_from_edge_file(edge_files[0])
    row0 = df_op.loc[t0]

    xe0, ye0, ze0 = _edge_xyz_lines(G_init)
    xn0, yn0, zn0, c0, labels0, s0 = _node_xyz_and_colors_and_sizes(G_init, row0)

    edge_trace0 = go.Scatter3d(
        x=xe0, y=ye0, z=ze0,
        mode="lines",
        line=dict(width=2),
        opacity=0.35,
        hoverinfo="skip",
        name="edges",
    )

    node_trace0 = go.Scatter3d(
    x=xn0, y=yn0, z=zn0,
    mode="markers+text",
    text=labels0,
    textposition="top center",
    marker=dict(size=s0, color=c0),
    hovertemplate="<b>%{text}</b><extra></extra>",
    name="nodes",
)

    # 5) Build frames (slider)
    frames = []
    slider_steps = []

    for edge_path in edge_files:
        t = _time_step_from_edge_file(edge_path)
        if t is None or t not in df_op.index:
            continue

        Gt = _graph_from_edge_file(edge_path)
        row = df_op.loc[t]

        xe, ye, ze = _edge_xyz_lines(Gt)
        xn, yn, zn, cc, labels, ss = _node_xyz_and_colors_and_sizes(Gt, row)

        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Scatter3d(x=xe, y=ye, z=ze, mode="lines",
                                 line=dict(width=2), opacity=0.35, hoverinfo="skip"),
                    go.Scatter3d(
                        x=xn, y=yn, z=zn, mode="markers+text",
                        text=labels, textposition="top center",
                        marker=dict(size=ss, color=cc),
                        hovertemplate="<b>%{text}</b><extra></extra>",
                    ),
                ],
                traces=[0, 1],
            )
        )

        slider_steps.append(
            dict(
                method="animate",
                args=[[str(t)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                label=str(t),
            )
        )

    if not frames:
        print("[network3d] No frames generated; skipping 3D HTML.")
        return None

    fig = go.Figure(data=[edge_trace0, node_trace0], frames=frames)

    fig.update_layout(
        title=f"3D Network (slider) — {out_prefix} — {version_set}",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"fromcurrent": True, "frame": {"duration": 350, "redraw": True}, "transition": {"duration": 0}}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    ),
                ],
                x=0.02, y=0.98, xanchor="left", yanchor="top",
            )
        ],
        sliders=[
            dict(
                active=0,
                pad={"t": 30, "b": 10},
                currentvalue={"prefix": "time_step: "},
                steps=slider_steps,
            )
        ],
    )

    run_stem = _get_run_stem(csv_path)
    out_html = os.path.join(plot_dir, f"{run_stem}_network3d.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[network3d] Saved interactive 3D HTML: {out_html}")
    return out_html

# -----------------------------
# Presentation-ready run reports, influence matrices, theme effectiveness, and network metrics
# -----------------------------

def build_output_run_report_summary_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_run_report_summary.csv")


def build_output_run_report_dashboard_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_run_report_dashboard.{args.figure_file_type}")


def build_output_influence_matrix_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_influence_matrix.csv")


def build_output_influence_matrix_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_influence_matrix.{args.figure_file_type}")


def build_output_theme_effectiveness_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_theme_effectiveness.csv")


def build_output_theme_effectiveness_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_theme_effectiveness.{args.figure_file_type}")


def build_output_network_assortativity_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_network_assortativity.csv")


def build_output_network_assortativity_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_network_assortativity.{args.figure_file_type}")


def build_output_ba_hub_assignment_csv_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_ba_hub_assignment_metrics.csv")


def build_output_ba_hub_assignment_plot_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_ba_hub_assignment_metrics.{args.figure_file_type}")


def _first_existing_path(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _try_builder(builder, args):
    try:
        return builder(args)
    except Exception:
        return None


def _numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([default] * (0 if df is None else len(df)), index=(None if df is None else df.index), dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _valid_step_rows(step_summary_df: pd.DataFrame) -> pd.DataFrame:
    if step_summary_df is None or step_summary_df.empty:
        return pd.DataFrame()
    df = step_summary_df.copy()
    if "is_synthetic" in df.columns:
        df = df[pd.to_numeric(df["is_synthetic"], errors="coerce").fillna(0).astype(int) == 0].copy()
    if "simulation_active" in df.columns:
        # Keep active rows and legacy rows where the column is blank.
        active = pd.to_numeric(df["simulation_active"], errors="coerce")
        df = df[(active.fillna(1).astype(int) == 1)].copy()
    return df.reset_index(drop=True)


def _rating_distribution_counts(values) -> dict:
    states = [-2, -1, 0, 1, 2]
    vals = pd.Series(values).dropna().astype(int).tolist()
    counts = {s: int(vals.count(s)) for s in states}
    return counts


def _rating_entropy(counts: dict) -> float:
    arr = np.asarray([counts.get(s, 0) for s in [-2, -1, 0, 1, 2]], dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _dist_compact(counts: dict) -> str:
    return ";".join(f"{s}:{int(counts.get(s, 0))}" for s in [-2, -1, 0, 1, 2])


def _belief_stats_from_values(values) -> dict:
    vals = np.asarray(pd.Series(values).dropna().astype(float).tolist(), dtype=float)
    if vals.size == 0:
        return {"B": 0.0, "D": 0.0, "P": 0.0, "entropy": 0.0, "positive_share": 0.0, "negative_share": 0.0, "zero_share": 0.0, "edge_share": 0.0}
    B = float(vals.mean())
    D = float(vals.std(ddof=0))
    var = D ** 2
    denom = var + B ** 2
    P = 0.0 if denom == 0 else float((var - B ** 2) / denom)
    counts = _rating_distribution_counts(vals)
    n = float(vals.size)
    return {
        "B": B,
        "D": D,
        "P": P,
        "entropy": _rating_entropy(counts),
        "positive_share": float((vals > 0).sum() / n),
        "negative_share": float((vals < 0).sum() / n),
        "zero_share": float((vals == 0).sum() / n),
        "edge_share": float((np.abs(vals) == 2).sum() / n),
    }


def _sum_col(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _count_bool(mask) -> int:
    try:
        return int(pd.Series(mask).fillna(False).astype(bool).sum())
    except Exception:
        return 0


def _first_nonempty(df: pd.DataFrame, col: str, default=""):
    if df is None or df.empty or col not in df.columns:
        return default
    vals = df[col].dropna().astype(str).map(str.strip)
    vals = vals[vals != ""]
    if vals.empty:
        return default
    return vals.iloc[0]


def save_run_report_summary(
    csv_path: str,
    out_csv_path: str,
    args,
    step_summary_csv_path: str | None = None,
    agent_summary_csv_path: str | None = None,
    run_metrics_csv_path: str | None = None,
    repair_events_csv_path: str | None = None,
    rag_retrieval_csv_path: str | None = None,
) -> pd.DataFrame:
    """Create one presentation-friendly row summarizing the run."""
    df_op = pd.read_csv(csv_path)
    if "time_step" not in df_op.columns:
        raise ValueError("opinion_change CSV must contain time_step")
    agent_cols = list(df_op.columns)[-int(args.num_agents):]
    initial_vals = df_op.iloc[0][agent_cols].astype(float).values
    final_vals = df_op.iloc[-1][agent_cols].astype(float).values
    initial_counts = _rating_distribution_counts(initial_vals)
    final_counts = _rating_distribution_counts(final_vals)
    initial_stats = _belief_stats_from_values(initial_vals)
    final_stats = _belief_stats_from_values(final_vals)

    # Last actual belief change from trajectory table.
    last_change_step = int(df_op.iloc[0]["time_step"])
    for i in range(1, len(df_op)):
        prev = df_op.iloc[i - 1][agent_cols].astype(float).values
        cur = df_op.iloc[i][agent_cols].astype(float).values
        if np.any(prev != cur):
            last_change_step = int(df_op.iloc[i]["time_step"])
    final_time_step = int(df_op.iloc[-1]["time_step"])
    final_plateau_length = max(0, final_time_step - last_change_step)

    step_df = pd.DataFrame()
    if step_summary_csv_path and os.path.exists(step_summary_csv_path):
        try:
            step_df = _valid_step_rows(pd.read_csv(step_summary_csv_path))
        except Exception:
            step_df = pd.DataFrame()

    pre = _numeric_series(step_df, "listener_pre_belief")
    post = _numeric_series(step_df, "listener_post_belief")
    delta = _numeric_series(step_df, "delta_belief")
    speaker = _numeric_series(step_df, "speaker_belief")
    dist_before = _numeric_series(step_df, "distance_to_speaker_before")
    dist_after = _numeric_series(step_df, "distance_to_speaker_after")

    bridge_to_zero = _count_bool((pre != post) & (((pre < 0) & (post == 0)) | ((pre > 0) & (post == 0))))
    bridge_across_zero = _count_bool((pre < 0) & (post > 0) | ((pre > 0) & (post < 0)))
    toward_zero = _count_bool((pre != post) & (post.abs() < pre.abs()))
    away_from_zero = _count_bool((pre != post) & (post.abs() > pre.abs()))
    edge_escape = _count_bool(((pre == 2) & (post < 2)) | ((pre == -2) & (post > -2)))
    edge_absorption = _count_bool(((pre < 2) & (post == 2)) | ((pre > -2) & (post == -2)))
    negative_attractor_entry = _count_bool((pre >= 0) & (post < 0))
    positive_attractor_entry = _count_bool((pre <= 0) & (post > 0))
    moved_toward_speaker = _count_bool((pre != post) & (dist_after < dist_before))
    moved_away_from_speaker = _count_bool((pre != post) & (dist_after > dist_before))
    moved_without_distance_change = _count_bool((pre != post) & (dist_after == dist_before))

    repair_rows = pd.DataFrame()
    if repair_events_csv_path and os.path.exists(repair_events_csv_path):
        try:
            repair_rows = pd.read_csv(repair_events_csv_path)
        except Exception:
            repair_rows = pd.DataFrame()

    run_metrics_df = pd.DataFrame()
    if run_metrics_csv_path and os.path.exists(run_metrics_csv_path):
        try:
            run_metrics_df = pd.read_csv(run_metrics_csv_path)
        except Exception:
            run_metrics_df = pd.DataFrame()

    def _metric_count(metric_name: str) -> int:
        if run_metrics_df.empty or "Metric" not in run_metrics_df.columns or "Count" not in run_metrics_df.columns:
            return 0
        sub = run_metrics_df[run_metrics_df["Metric"].astype(str) == metric_name]
        if sub.empty:
            return 0
        return int(pd.to_numeric(sub["Count"], errors="coerce").fillna(0).sum())

    hard_repair_count = _sum_col(step_df, "repair_events")
    soft_cleanup_count = _sum_col(step_df, "soft_cleanup_events")
    if not repair_rows.empty and "Repair Tags" in repair_rows.columns:
        # Keep step-summary as source of truth, but repair file can fill older runs.
        hard_repair_count = max(hard_repair_count, int(repair_rows["Repair Tags"].fillna("").astype(str).str.strip().ne("").sum()))
    if not repair_rows.empty and "Soft Cleanup Tags" in repair_rows.columns:
        soft_cleanup_count = max(soft_cleanup_count, int(repair_rows["Soft Cleanup Tags"].fillna("").astype(str).str.strip().ne("").sum()))

    row = {
        "run_id": _first_nonempty(step_df, "run_id", _get_run_stem(csv_path)),
        "run_stem": _get_run_stem(csv_path),
        "seed": int(getattr(args, "seed", -1)),
        "version_set": str(getattr(args, "version_set", "")),
        "world": _first_nonempty(step_df, "world", ""),
        "fact_pack_mode": _first_nonempty(step_df, "fact_pack_mode", ""),
        "rag_content_mode": _first_nonempty(step_df, "rag_content_mode", ""),
        "selection_mode": _first_nonempty(step_df, "selection_mode", ""),
        "homophily_mode": _first_nonempty(step_df, "homophily_mode", ""),
        "model_name_step2": _first_nonempty(step_df, "model_name_step2", ""),
        "model_name_step3": _first_nonempty(step_df, "model_name_step3", ""),
        "num_agents": int(getattr(args, "num_agents", len(agent_cols))),
        "num_steps_requested": int(getattr(args, "num_steps", final_time_step)),
        "final_time_step": final_time_step,
        "last_change_step": last_change_step,
        "final_plateau_length": final_plateau_length,
        "initial_distribution": _dist_compact(initial_counts),
        "final_distribution": _dist_compact(final_counts),
    }
    for state in [-2, -1, 0, 1, 2]:
        row[f"initial_count_{state}"] = initial_counts[state]
        row[f"final_count_{state}"] = final_counts[state]
    for key, val in initial_stats.items():
        row[f"initial_{key}"] = float(val)
    for key, val in final_stats.items():
        row[f"final_{key}"] = float(val)

    n_interactions = int(len(step_df))
    row.update({
        "interactions": n_interactions,
        "valid_listener_interactions": int((pd.to_numeric(step_df.get("listener_valid_interaction", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int) == 1).sum()) if not step_df.empty else 0,
        "listener_pipeline_artifacts": _sum_col(step_df, "listener_pipeline_artifact"),
        "step2_fallbacks": _sum_col(step_df, "speaker_step2_fallback_used"),
        "step2_invalid_outputs": int((pd.to_numeric(step_df.get("speaker_step2_valid_output", pd.Series(dtype=float)), errors="coerce").fillna(1).astype(int) == 0).sum()) if not step_df.empty else 0,
        "total_moves": _sum_col(step_df, "moves_total"),
        "positive_moves": _sum_col(step_df, "moves_positive"),
        "negative_moves": _sum_col(step_df, "moves_negative"),
        "no_change_events": _sum_col(step_df, "moves_zero"),
        "absolute_total_delta": _sum_col(step_df, "abs_total_delta"),
        "net_delta": int(pd.to_numeric(delta, errors="coerce").fillna(0).sum()) if len(delta) else 0,
        "bridge_to_zero_count": bridge_to_zero,
        "bridge_across_zero_count": bridge_across_zero,
        "toward_zero_moves": toward_zero,
        "away_from_zero_moves": away_from_zero,
        "edge_escape_count": edge_escape,
        "edge_absorption_count": edge_absorption,
        "negative_attractor_entry_count": negative_attractor_entry,
        "positive_attractor_entry_count": positive_attractor_entry,
        "moved_toward_speaker_count": moved_toward_speaker,
        "moved_away_from_speaker_count": moved_away_from_speaker,
        "moved_without_distance_change_count": moved_without_distance_change,
        "counter_attitudinal_moves": _sum_col(step_df, "counter_attitudinal_moves"),
        "hard_repairs": hard_repair_count,
        "soft_cleanups": soft_cleanup_count,
        "validation_failures_total": _metric_count("validation_failures_total"),
        "format_failures_total": _metric_count("format_failures_total"),
        "empty_final_text_events": _metric_count("native::empty_final_text") + _metric_count("empty_final_text"),
    })
    if n_interactions > 0:
        for numerator in [
            "total_moves", "positive_moves", "negative_moves", "bridge_to_zero_count", "bridge_across_zero_count",
            "moved_toward_speaker_count", "moved_away_from_speaker_count", "hard_repairs", "soft_cleanups",
            "step2_fallbacks", "listener_pipeline_artifacts",
        ]:
            row[f"{numerator}_rate"] = float(row[numerator] / n_interactions)
    else:
        for numerator in ["total_moves", "positive_moves", "negative_moves", "bridge_to_zero_count", "bridge_across_zero_count", "moved_toward_speaker_count", "moved_away_from_speaker_count", "hard_repairs", "soft_cleanups", "step2_fallbacks", "listener_pipeline_artifacts"]:
            row[f"{numerator}_rate"] = 0.0

    # RAG retrieval direction is a tracked metric whenever RAG retrieval logging exists:
    # it shows whether support/challenge/context retrieval coincided with positive, negative, or no movement.
    try:
        rag_summary = summarize_rag_retrieval_direction_metrics(
            rag_retrieval_csv_path=rag_retrieval_csv_path,
            step_summary_csv_path=step_summary_csv_path,
        )
        for key, value in rag_summary.items():
            row[key] = value
    except Exception:
        pass

    out = pd.DataFrame([row])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved run report summary CSV to: {out_csv_path}")
    return out


def plot_run_report_dashboard(report_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if report_df is None or report_df.empty:
        return
    row = report_df.iloc[0]
    states = [-2, -1, 0, 1, 2]
    final_counts = [int(row.get(f"final_count_{s}", 0)) for s in states]
    initial_counts = [int(row.get(f"initial_count_{s}", 0)) for s in states]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7) if fig_type == "png" else (8, 5.6), dpi=150)

    ax = axes[0, 0]
    x = np.arange(len(states))
    width = 0.38
    ax.bar(x - width / 2, initial_counts, width, label="initial")
    ax.bar(x + width / 2, final_counts, width, label="final")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in states])
    ax.set_xlabel("Rating")
    ax.set_ylabel("Agents")
    ax.set_title("Initial vs final distribution")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[0, 1]
    labels = ["B", "D", "P", "entropy"]
    vals = [float(row.get(f"final_{k}", 0.0)) for k in labels]
    ax.bar(labels, vals)
    ax.set_title("Final aggregate state")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    movement_labels = ["moves", "+ moves", "- moves", "bridge→0", "across 0", "toward speaker"]
    movement_vals = [
        int(row.get("total_moves", 0)),
        int(row.get("positive_moves", 0)),
        int(row.get("negative_moves", 0)),
        int(row.get("bridge_to_zero_count", 0)),
        int(row.get("bridge_across_zero_count", 0)),
        int(row.get("moved_toward_speaker_count", 0)),
    ]
    ax.barh(movement_labels[::-1], movement_vals[::-1])
    ax.set_title("Movement summary")
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1, 1]
    quality_labels = ["hard repairs", "soft cleanups", "step2 fallbacks", "pipeline artifacts"]
    quality_vals = [
        int(row.get("hard_repairs", 0)),
        int(row.get("soft_cleanups", 0)),
        int(row.get("step2_fallbacks", 0)),
        int(row.get("listener_pipeline_artifacts", 0)),
    ]
    ax.barh(quality_labels[::-1], quality_vals[::-1])
    ax.set_title("Quality / intervention counts")
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", alpha=0.3)

    run_label = str(row.get("run_stem", row.get("run_id", "run")))
    fig.suptitle(f"Run report dashboard — {run_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved run report dashboard to: {out_path}")


def save_influence_matrix(step_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = _valid_step_rows(pd.read_csv(step_summary_csv_path))
    if df.empty:
        out = pd.DataFrame(columns=["listener_pre_belief", "speaker_stance_label", "events"])
        out.to_csv(out_csv_path, index=False)
        return out

    for col in ["listener_pre_belief", "listener_post_belief", "speaker_belief", "delta_belief", "distance_to_speaker_before", "distance_to_speaker_after"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "speaker_stance_label" not in df.columns:
        df["speaker_stance_label"] = np.where(df["speaker_belief"] > 0, "support", np.where(df["speaker_belief"] < 0, "oppose", "uncertain"))

    df = df.dropna(subset=["listener_pre_belief", "listener_post_belief", "delta_belief", "speaker_belief"])
    df["moved"] = (df["delta_belief"] != 0).astype(int)
    df["positive_move"] = (df["delta_belief"] > 0).astype(int)
    df["negative_move"] = (df["delta_belief"] < 0).astype(int)
    df["toward_speaker"] = ((df["delta_belief"] != 0) & (df["distance_to_speaker_after"] < df["distance_to_speaker_before"])).astype(int)
    df["away_from_speaker"] = ((df["delta_belief"] != 0) & (df["distance_to_speaker_after"] > df["distance_to_speaker_before"])).astype(int)
    df["bridge_to_zero"] = ((df["delta_belief"] != 0) & (((df["listener_pre_belief"] < 0) & (df["listener_post_belief"] == 0)) | ((df["listener_pre_belief"] > 0) & (df["listener_post_belief"] == 0)))).astype(int)
    df["bridge_across_zero"] = (((df["listener_pre_belief"] < 0) & (df["listener_post_belief"] > 0)) | ((df["listener_pre_belief"] > 0) & (df["listener_post_belief"] < 0))).astype(int)
    df["more_extreme"] = (df["listener_post_belief"].abs() > df["listener_pre_belief"].abs()).astype(int)
    df["less_extreme"] = (df["listener_post_belief"].abs() < df["listener_pre_belief"].abs()).astype(int)

    group_cols = ["listener_pre_belief", "speaker_stance_label"]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        pre, stance = keys
        n = int(len(sub))
        rows.append({
            "listener_pre_belief": int(pre),
            "speaker_stance_label": str(stance),
            "events": n,
            "mean_delta": float(sub["delta_belief"].mean()) if n else 0.0,
            "move_rate": float(sub["moved"].mean()) if n else 0.0,
            "positive_move_rate": float(sub["positive_move"].mean()) if n else 0.0,
            "negative_move_rate": float(sub["negative_move"].mean()) if n else 0.0,
            "toward_speaker_rate": float(sub["toward_speaker"].mean()) if n else 0.0,
            "away_from_speaker_rate": float(sub["away_from_speaker"].mean()) if n else 0.0,
            "bridge_to_zero_rate": float(sub["bridge_to_zero"].mean()) if n else 0.0,
            "bridge_across_zero_rate": float(sub["bridge_across_zero"].mean()) if n else 0.0,
            "more_extreme_rate": float(sub["more_extreme"].mean()) if n else 0.0,
            "less_extreme_rate": float(sub["less_extreme"].mean()) if n else 0.0,
        })
    out = pd.DataFrame(rows).sort_values(["listener_pre_belief", "speaker_stance_label"])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved influence matrix CSV to: {out_csv_path}")
    return out


def plot_influence_matrix(influence_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if influence_df is None or influence_df.empty:
        return
    states = [-2, -1, 0, 1, 2]
    stances = [s for s in ["oppose", "uncertain", "support"] if s in set(influence_df["speaker_stance_label"].astype(str))]
    if not stances:
        stances = sorted(influence_df["speaker_stance_label"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4) if fig_type == "png" else (7.2, 3.2), dpi=150)

    for ax, value_col, title in [
        (axes[0], "mean_delta", "Mean listener delta"),
        (axes[1], "toward_speaker_rate", "Move-toward-speaker rate"),
    ]:
        pivot = (influence_df.pivot_table(index="listener_pre_belief", columns="speaker_stance_label", values=value_col, aggfunc="mean")
                 .reindex(index=states, columns=stances))
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto")
        ax.set_xticks(range(len(stances)))
        ax.set_xticklabels(stances, rotation=25, ha="right")
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels([str(s) for s in states])
        ax.set_xlabel("Speaker stance")
        ax.set_ylabel("Listener pre-belief")
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved influence matrix plot to: {out_path}")


_THEME_PATTERNS = {
    "anti:no_direct_test": r"\b(?:no|lack(?:s)?|without|absence of|absent)\s+(?:a\s+)?(?:clear\s+|direct\s+|agreed\s+)?(?:test|testability|testable framework|way to test)|\buntestable\b|\bnot testable\b",
    "anti:no_evidence_or_proof": r"\b(?:no|lack(?:s)?|without|absence of)\s+(?:clear\s+|concrete\s+|direct\s+|solid\s+)?(?:evidence|proof|empirical grounding|observable evidence)|\bevidence remains elusive\b",
    "anti:future_tech_assumptions": r"\b(?:future tech|future technology|advanced civilizations?\s+(?:would|could|might)|civilization survival|technology and motives)\b",
    "anti:motive_assumptions": r"\b(?:motives?|intent|intentions?|why and how|choose not|ethical|practical|cultural)\b",
    "anti:unfalsifiable": r"\b(?:unfalsifiable|falsifiability|no way to test|every possible observation)\b",
    "anti:natural_explanation": r"\b(?:ordinary physics|natural explanations?|real-world physics|nature,? not code|without needing a constructed framework|doesn'?t require a constructed)\b",
    "anti:reference_class": r"\b(?:reference class|observer-count arguments? are sensitive|particular person)\b",
    "pro:observer_count": r"\b(?:simulated observers?\s+(?:could|might|would)?\s*(?:vastly\s+)?outnumber|outnumber(?:ing)? real|more likely to be in a simulated layer|randomly located observer|chance we'?re in one)\b",
    "pro:many_simulations": r"\b(?:many|countless|trillions of|detailed|ancestor)\s+simulations?|\brun countless\b|\brun many\b",
    "pro:advanced_civilizations": r"\badvanced civilizations?\b",
    "pro:hidden_simulation": r"\b(?:hide its artificial structure|hide artificial structures?|perfect simulation|without us ever knowing|lack of direct outside access|no obvious glitches)\b",
    "pro:stable_laws_limits": r"\b(?:stable laws|limited horizons|information-like constraints|physical laws|mathematical precision)\b",
    "pro:logical_possibility": r"\b(?:logical(?:ly)? possible|plausible hypothesis|plausible idea|worth considering|compelling possibility|thought experiment|intriguing)\b",
}


def classify_argument_themes(text: str) -> list[str]:
    s = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not s:
        return []
    themes = []
    for label, pattern in _THEME_PATTERNS.items():
        if re.search(pattern, s, flags=re.I):
            themes.append(label)
    return themes or ["uncoded"]


def save_theme_effectiveness(step_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = _valid_step_rows(pd.read_csv(step_summary_csv_path))
    if df.empty:
        out = pd.DataFrame(columns=["theme", "events"])
        out.to_csv(out_csv_path, index=False)
        return out
    for col in ["listener_pre_belief", "listener_post_belief", "speaker_belief", "delta_belief"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    for _, r in df.iterrows():
        text_parts = [
            str(r.get("speaker_tweet", "") or ""),
            str(r.get("listener_display_explanation", "") or ""),
            str(r.get("listener_response", "") or ""),
        ]
        themes = classify_argument_themes(" ".join(text_parts))
        for theme in themes:
            rows.append({
                "theme": theme,
                "time_step": r.get("time_step", np.nan),
                "listener_pre_belief": r.get("listener_pre_belief", np.nan),
                "listener_post_belief": r.get("listener_post_belief", np.nan),
                "speaker_belief": r.get("speaker_belief", np.nan),
                "delta_belief": r.get("delta_belief", np.nan),
                "moved": int(float(r.get("delta_belief", 0) or 0) != 0),
                "positive_move": int(float(r.get("delta_belief", 0) or 0) > 0),
                "negative_move": int(float(r.get("delta_belief", 0) or 0) < 0),
                "bridge_to_zero": int((float(r.get("listener_pre_belief", 0) or 0) != float(r.get("listener_post_belief", 0) or 0)) and ((float(r.get("listener_pre_belief", 0) or 0) < 0 and float(r.get("listener_post_belief", 0) or 0) == 0) or (float(r.get("listener_pre_belief", 0) or 0) > 0 and float(r.get("listener_post_belief", 0) or 0) == 0))),
                "more_extreme": int(abs(float(r.get("listener_post_belief", 0) or 0)) > abs(float(r.get("listener_pre_belief", 0) or 0))),
                "less_extreme": int(abs(float(r.get("listener_post_belief", 0) or 0)) < abs(float(r.get("listener_pre_belief", 0) or 0))),
            })
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        out = pd.DataFrame(columns=["theme", "events"])
        out.to_csv(out_csv_path, index=False)
        return out
    agg = expanded.groupby("theme").agg(
        events=("theme", "size"),
        mean_delta=("delta_belief", "mean"),
        move_rate=("moved", "mean"),
        positive_move_rate=("positive_move", "mean"),
        negative_move_rate=("negative_move", "mean"),
        bridge_to_zero_rate=("bridge_to_zero", "mean"),
        more_extreme_rate=("more_extreme", "mean"),
        less_extreme_rate=("less_extreme", "mean"),
    ).reset_index().sort_values(["events", "theme"], ascending=[False, True])
    agg.to_csv(out_csv_path, index=False)
    print(f"Saved theme effectiveness CSV to: {out_csv_path}")
    return agg


def plot_theme_effectiveness(theme_df: pd.DataFrame, out_path: str, fig_type: str, top_n: int = 14) -> None:
    if theme_df is None or theme_df.empty:
        return
    df = theme_df.sort_values("events", ascending=False).head(top_n).copy()
    df = df.sort_values("events", ascending=True)
    labels = df["theme"].astype(str).tolist()
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.35 * len(df))) if fig_type == "png" else (8.8, max(3.2, 0.28 * len(df))), dpi=150)
    axes[0].barh(y, df["events"])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("Theme frequency")
    axes[0].set_xlabel("Events")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[1].barh(y, df["mean_delta"])
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].axvline(0, linewidth=1)
    axes[1].set_title("Mean belief delta after theme")
    axes[1].set_xlabel("Mean Δ")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved theme effectiveness plot to: {out_path}")


def _sign_label_from_rating(v) -> str:
    try:
        x = int(v)
    except Exception:
        return "unknown"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "neutral"


def _graph_from_edge_file_with_nodes(edge_path: str, agent_names: list[str]) -> nx.Graph:
    df_edges = pd.read_csv(edge_path)
    G = nx.Graph()
    for _, r in df_edges.iterrows():
        i_name = r.get("i_name", None)
        j_name = r.get("j_name", None)
        if pd.notna(i_name) and pd.notna(j_name):
            G.add_edge(str(i_name), str(j_name))
    for name in agent_names:
        if name not in G:
            G.add_node(name)
    return G




def _safe_numeric_assortativity(G: nx.Graph, attr: str) -> float:
    """Return numeric assortativity, or NaN when the metric is undefined.

    NetworkX emits RuntimeWarning for collapsed/degenerate graphs such as all
    nodes having the same rating/sign. In that case the statistic is undefined,
    not zero, so we store NaN and keep the plotting pipeline quiet.
    """
    try:
        if G is None or G.number_of_edges() == 0:
            return float("nan")
        vals = []
        for n in G.nodes():
            v = G.nodes[n].get(attr, None)
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            vals.append(v)
        if len(vals) < 2 or len(set(vals)) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = nx.numeric_assortativity_coefficient(G, attr)
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _safe_attribute_assortativity(G: nx.Graph, attr: str) -> float:
    """Return attribute assortativity, or NaN when the metric is undefined."""
    try:
        if G is None or G.number_of_edges() == 0:
            return float("nan")
        vals = []
        for n in G.nodes():
            v = G.nodes[n].get(attr, None)
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            vals.append(v)
        if len(vals) < 2 or len(set(vals)) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = nx.attribute_assortativity_coefficient(G, attr)
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def save_network_assortativity_metrics(csv_path: str, out_prefix: str, version_set: str, num_agents: int, out_csv_path: str) -> pd.DataFrame:
    results_dir = os.path.dirname(csv_path)
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    if not edge_files:
        out = pd.DataFrame(columns=["time_step"])
        out.to_csv(out_csv_path, index=False)
        print("[network-assortativity] No edge files found; wrote empty CSV.")
        return out
    df_op = pd.read_csv(csv_path)
    if "time_step" in df_op.columns:
        df_op = df_op.set_index("time_step")
    df_op = df_op.loc[:, ~df_op.columns.str.startswith("Unnamed")]
    agent_names = list(df_op.columns)[-num_agents:]

    rows = []
    for edge_path in edge_files:
        base = os.path.basename(edge_path)
        m = re.search(r"_step_(\d+)", base)
        if not m:
            continue
        time_step = int(m.group(1)) + 1
        if time_step not in df_op.index:
            continue
        G = _graph_from_edge_file_with_nodes(edge_path, agent_names)
        row_op = df_op.loc[time_step]
        ratings = {n: int(row_op[n]) for n in agent_names if n in row_op.index and pd.notna(row_op[n])}
        nx.set_node_attributes(G, ratings, "rating")
        nx.set_node_attributes(G, {n: _sign_label_from_rating(v) for n, v in ratings.items()}, "sign")

        same_sign = 0
        opposite_sign = 0
        same_rating = 0
        abs_diffs = []
        valid_edges = 0
        for u, v in G.edges():
            if u not in ratings or v not in ratings:
                continue
            valid_edges += 1
            ru, rv = ratings[u], ratings[v]
            su, sv = _sign_label_from_rating(ru), _sign_label_from_rating(rv)
            same_sign += int(su == sv)
            opposite_sign += int(su != sv)
            same_rating += int(ru == rv)
            abs_diffs.append(abs(ru - rv))
        numeric_assort = _safe_numeric_assortativity(G, "rating")
        sign_assort = _safe_attribute_assortativity(G, "sign")

        pos_nodes = [n for n, r in ratings.items() if r > 0]
        neg_nodes = [n for n, r in ratings.items() if r < 0]
        zero_nodes = [n for n, r in ratings.items() if r == 0]
        pos_components = list(nx.connected_components(G.subgraph(pos_nodes))) if pos_nodes else []
        neg_components = list(nx.connected_components(G.subgraph(neg_nodes))) if neg_nodes else []
        rows.append({
            "time_step": time_step,
            "edges": int(valid_edges),
            "same_sign_edge_share": float(same_sign / valid_edges) if valid_edges else 0.0,
            "opposite_sign_edge_share": float(opposite_sign / valid_edges) if valid_edges else 0.0,
            "same_rating_edge_share": float(same_rating / valid_edges) if valid_edges else 0.0,
            "mean_abs_edge_belief_diff": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
            "numeric_belief_assortativity": numeric_assort,
            "sign_assortativity": sign_assort,
            "positive_node_count": int(len(pos_nodes)),
            "negative_node_count": int(len(neg_nodes)),
            "neutral_node_count": int(len(zero_nodes)),
            "positive_cluster_count": int(len(pos_components)),
            "negative_cluster_count": int(len(neg_components)),
            "largest_positive_component_size": int(max((len(c) for c in pos_components), default=0)),
            "largest_negative_component_size": int(max((len(c) for c in neg_components), default=0)),
        })
    out = pd.DataFrame(rows).sort_values("time_step") if rows else pd.DataFrame(columns=["time_step"])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved network assortativity CSV to: {out_csv_path}")
    return out


def plot_network_assortativity_metrics(metrics_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if metrics_df is None or metrics_df.empty or "time_step" not in metrics_df.columns:
        return
    df = metrics_df.sort_values("time_step").copy()
    t = df["time_step"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7) if fig_type == "png" else (5.6, 5.6), sharex=True, dpi=150)
    if "same_sign_edge_share" in df.columns:
        axes[0].plot(t, df["same_sign_edge_share"], label="same-sign edge share", linewidth=1.8)
    if "opposite_sign_edge_share" in df.columns:
        axes[0].plot(t, df["opposite_sign_edge_share"], label="opposite-sign edge share", linewidth=1.8)
    axes[0].set_ylabel("Share")
    axes[0].set_title("Network belief mixing")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    if "mean_abs_edge_belief_diff" in df.columns:
        axes[1].plot(t, df["mean_abs_edge_belief_diff"], label="mean abs edge belief diff", linewidth=1.8)
    if "numeric_belief_assortativity" in df.columns:
        axes[1].plot(t, df["numeric_belief_assortativity"], label="numeric assortativity", linewidth=1.5)
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    for col, label in [
        ("largest_positive_component_size", "largest positive component"),
        ("largest_negative_component_size", "largest negative component"),
        ("neutral_node_count", "neutral nodes"),
    ]:
        if col in df.columns:
            axes[2].plot(t, df[col], label=label, linewidth=1.8)
    axes[2].set_ylabel("Agents")
    axes[2].set_xlabel("Time step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved network assortativity plot to: {out_path}")


def save_ba_hub_assignment_metrics(agent_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    """Summarize BA target-group assignment versus realized hub status.

    Reads the runtime agent summary fields produced by the BA assignment-mode patch:
      ba_hub_assignment_mode, ba_hub_targeted_flag, ba_early_priority_flag,
      ba_actual_hub_assigned_flag, network_top_hub_flag, network_max_degree_hub_flag.
    The output makes it explicit whether a target group merely had early BA
    priority, was forcibly assigned to actual top hubs, or actually became hubs.
    """
    cols = [
        "group", "count", "ba_hub_strategy", "ba_hub_assignment_mode",
        "mean_network_degree", "mean_degree_rank", "top_hub_share", "max_hub_share",
        "targeted_share", "early_priority_share", "actual_hub_assigned_share",
        "positive_initial_share", "negative_initial_share", "neutral_initial_share",
    ]
    try:
        df = pd.read_csv(agent_summary_csv_path)
    except Exception:
        out = pd.DataFrame(columns=cols)
        out.to_csv(out_csv_path, index=False)
        return out
    if df is None or df.empty or "ba_hub_strategy" not in df.columns:
        out = pd.DataFrame(columns=cols)
        out.to_csv(out_csv_path, index=False)
        return out

    work = df.copy()
    for c in [
        "network_degree", "network_degree_rank", "network_top_hub_flag", "network_max_degree_hub_flag",
        "ba_hub_targeted_flag", "ba_early_priority_flag", "ba_actual_hub_assigned_flag",
    ]:
        if c not in work.columns:
            work[c] = 0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)
    if "network_initial_side" not in work.columns:
        work["network_initial_side"] = "unknown"
    if "ba_hub_assignment_mode" not in work.columns:
        work["ba_hub_assignment_mode"] = "early_position"

    def _mode_value(col: str) -> str:
        try:
            vals = [str(x) for x in work[col].dropna().astype(str).unique().tolist() if str(x).strip()]
            return "|".join(sorted(vals)) if vals else ""
        except Exception:
            return ""

    def _summ(label: str, sub: pd.DataFrame) -> dict:
        n = int(len(sub))
        if n <= 0:
            return {k: (0 if k == "count" else "") for k in cols} | {"group": label}
        side = sub["network_initial_side"].astype(str).str.lower()
        return {
            "group": label,
            "count": n,
            "ba_hub_strategy": _mode_value("ba_hub_strategy"),
            "ba_hub_assignment_mode": _mode_value("ba_hub_assignment_mode"),
            "mean_network_degree": float(sub["network_degree"].mean()),
            "mean_degree_rank": float(sub["network_degree_rank"].mean()),
            "top_hub_share": float(sub["network_top_hub_flag"].mean()),
            "max_hub_share": float(sub["network_max_degree_hub_flag"].mean()),
            "targeted_share": float(sub["ba_hub_targeted_flag"].mean()),
            "early_priority_share": float(sub["ba_early_priority_flag"].mean()),
            "actual_hub_assigned_share": float(sub["ba_actual_hub_assigned_flag"].mean()),
            "positive_initial_share": float((side == "positive").mean()),
            "negative_initial_share": float((side == "negative").mean()),
            "neutral_initial_share": float((side == "neutral").mean()),
        }

    rows = [
        _summ("all_agents", work),
        _summ("targeted_agents", work[work["ba_hub_targeted_flag"] > 0]),
        _summ("non_targeted_agents", work[work["ba_hub_targeted_flag"] <= 0]),
        _summ("early_priority_agents", work[work["ba_early_priority_flag"] > 0]),
        _summ("actual_hub_assigned_agents", work[work["ba_actual_hub_assigned_flag"] > 0]),
        _summ("realized_top_hubs", work[work["network_top_hub_flag"] > 0]),
        _summ("max_degree_hubs", work[work["network_max_degree_hub_flag"] > 0]),
    ]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(out_csv_path, index=False)
    print(f"Saved BA hub assignment metrics CSV to: {out_csv_path}")
    return out


def plot_ba_hub_assignment_metrics(metrics_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if metrics_df is None or metrics_df.empty or "group" not in metrics_df.columns:
        return
    df = metrics_df.copy()
    groups = [g for g in ["all_agents", "targeted_agents", "realized_top_hubs", "actual_hub_assigned_agents"] if g in set(df["group"].astype(str))]
    if not groups:
        return
    sub = df[df["group"].astype(str).isin(groups)].set_index("group").loc[groups]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6) if fig_type == "png" else (6.4, 4.8), dpi=150)

    x = np.arange(len(groups))
    labels = [g.replace("_", "\n") for g in groups]
    width = 0.24
    for offset, col, label in [
        (-width, "top_hub_share", "realized top-hub share"),
        (0.0, "early_priority_share", "early-priority share"),
        (width, "actual_hub_assigned_share", "actual-hub-assigned share"),
    ]:
        if col in sub.columns:
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            axes[0].bar(x + offset, vals, width=width, label=label)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Share")
    axes[0].set_title("BA hub assignment check")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    if "mean_network_degree" in sub.columns:
        axes[1].bar(x, pd.to_numeric(sub["mean_network_degree"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Mean degree")
    axes[1].grid(True, axis="y", alpha=0.3)

    mode_text = ""
    try:
        mode = str(metrics_df.get("ba_hub_assignment_mode", pd.Series([""])).dropna().astype(str).iloc[0])
        strategy = str(metrics_df.get("ba_hub_strategy", pd.Series([""])).dropna().astype(str).iloc[0])
        if mode or strategy:
            mode_text = f"target={strategy}; assignment={mode}"
    except Exception:
        mode_text = ""
    if mode_text:
        fig.suptitle(mode_text, fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BA hub assignment metrics plot to: {out_path}")



# -----------------------------
# Added analysis/reporting graphs
# -----------------------------

def plot_individual_belief_trajectories(csv_path: str, out_path: str, fig_type: str, num_agents: int) -> None:
    """Standard graph: one overlaid belief trajectory per agent."""
    df = pd.read_csv(csv_path)
    if df.empty or "time_step" not in df.columns:
        return
    agent_cols = list(df.columns)[-num_agents:]
    t = df["time_step"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5) if fig_type == "png" else (6.4, 4.0), dpi=150)
    for name in agent_cols:
        y = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        ax.plot(t, y, linewidth=1.8, alpha=0.85, label=str(name))
        if len(t) > 0 and np.isfinite(y[-1]):
            ax.text(t[-1], y[-1], f" {name}", fontsize=6, va="center", alpha=0.8)
    ax.set_title("Individual belief trajectories")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Belief")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylim(-2.25, 2.25)
    ax.grid(True, alpha=0.3)
    if len(agent_cols) <= 12:
        ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved individual belief trajectories to: {out_path}")




def _rating_color_map():
    return {
        -2: "#b2182b",
        -1: "#ef8a62",
        0: "#d9d9d9",
        1: "#67a9cf",
        2: "#2166ac",
    }


def _final_belief_series_from_agent_summary(agent_df: pd.DataFrame) -> pd.Series:
    for col in ["final_belief", "final_rating", "current_belief"]:
        if col in agent_df.columns:
            return pd.to_numeric(agent_df[col], errors="coerce")
    return pd.Series([np.nan] * len(agent_df), index=agent_df.index)


def _choose_trait_for_final_strip(agent_df: pd.DataFrame) -> str | None:
    preferred = [
        "institutional_trust",
        "epistemic_profile",
        "topic_causal_profile_choice",
        "topic_profile_choice",
        "official_narrative_suspicion",
        "openness_to_update",
        "evidence_style",
    ]
    for col in preferred:
        if col not in agent_df.columns:
            continue
        vals = agent_df[col].fillna("").astype(str).str.strip()
        uniq = [v for v in vals.unique().tolist() if v]
        if len(uniq) >= 2:
            return col
    return None


def plot_paired_initial_final_distribution(csv_path: str, out_path: str, fig_type: str, num_agents: int) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or "time_step" not in df.columns:
        return
    agent_cols = list(df.columns)[-num_agents:]
    initial_vals = pd.to_numeric(df.iloc[0][agent_cols], errors="coerce").dropna().astype(int).tolist()
    final_vals = pd.to_numeric(df.iloc[-1][agent_cols], errors="coerce").dropna().astype(int).tolist()
    initial_counts = _rating_distribution_counts(initial_vals)
    final_counts = _rating_distribution_counts(final_vals)
    states = [-2, -1, 0, 1, 2]
    colors = [_rating_color_map().get(s, "#999999") for s in states]
    max_count = max([initial_counts.get(s, 0) for s in states] + [final_counts.get(s, 0) for s in states] + [1])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0) if fig_type == "png" else (5.8, 4.8), dpi=150, sharex=True)
    for ax, counts, title in [
        (axes[0], initial_counts, "Initial distribution"),
        (axes[1], final_counts, "Final distribution"),
    ]:
        vals = [int(counts.get(s, 0)) for s in states]
        ax.barh([str(s) for s in states], vals, color=colors)
        ax.set_xlim(0, max_count * 1.15)
        ax.set_title(title)
        ax.set_xlabel("Agents")
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v + max_count * 0.02, i, str(v), va="center", fontsize=8)
    fig.suptitle("Paired initial→final distribution comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paired initial/final distribution plot to: {out_path}")


def save_exposure_ecology(step_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = _valid_step_rows(pd.read_csv(step_summary_csv_path))
    if df.empty:
        out = pd.DataFrame(columns=["time_step", "support_events", "oppose_events", "uncertain_events", "cum_support", "cum_oppose", "cum_uncertain", "cum_total"])
        out.to_csv(out_csv_path, index=False)
        return out
    if "time_step" not in df.columns:
        if "Time Step" in df.columns:
            df["time_step"] = pd.to_numeric(df["Time Step"], errors="coerce")
        else:
            raise ValueError("step summary CSV missing time_step")
    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce")
    if "speaker_stance_label" not in df.columns:
        if "speaker_belief" in df.columns:
            sb = pd.to_numeric(df["speaker_belief"], errors="coerce")
            df["speaker_stance_label"] = np.where(sb > 0, "support", np.where(sb < 0, "oppose", "uncertain"))
        else:
            df["speaker_stance_label"] = "uncertain"
    work = df.dropna(subset=["time_step"]).copy()
    counts = (work.groupby(["time_step", "speaker_stance_label"]).size().unstack(fill_value=0).reset_index().sort_values("time_step"))
    for col in ["support", "oppose", "uncertain"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["time_step", "support", "oppose", "uncertain"]].copy()
    counts = counts.rename(columns={"support": "support_events", "oppose": "oppose_events", "uncertain": "uncertain_events"})
    counts["cum_support"] = counts["support_events"].cumsum()
    counts["cum_oppose"] = counts["oppose_events"].cumsum()
    counts["cum_uncertain"] = counts["uncertain_events"].cumsum()
    counts["cum_total"] = counts[["cum_support", "cum_oppose", "cum_uncertain"]].sum(axis=1)
    counts.to_csv(out_csv_path, index=False)
    print(f"Saved exposure ecology CSV to: {out_csv_path}")
    return counts


def plot_exposure_ecology(ecology_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if ecology_df is None or ecology_df.empty:
        return
    df = ecology_df.copy()
    x = pd.to_numeric(df["time_step"], errors="coerce").fillna(0).to_numpy(dtype=float)
    support = pd.to_numeric(df.get("cum_support", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(dtype=float)
    oppose = pd.to_numeric(df.get("cum_oppose", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(dtype=float)
    uncertain = pd.to_numeric(df.get("cum_uncertain", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.6) if fig_type == "png" else (6.0, 3.7), dpi=150)
    ax.stackplot(x, oppose, uncertain, support, labels=["oppose", "uncertain", "support"], colors=["#ef8a62", "#d9d9d9", "#67a9cf"], alpha=0.9)
    ax.set_title("Exposure ecology: cumulative speaker events")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative events")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exposure ecology plot to: {out_path}")


def plot_final_belief_strip_by_trait(agent_summary_csv_path: str, out_path: str, fig_type: str) -> None:
    df = _agent_summary_with_delta(agent_summary_csv_path)
    if df is None or df.empty:
        return
    trait_col = _choose_trait_for_final_strip(df)
    if not trait_col:
        return
    final_vals = _final_belief_series_from_agent_summary(df)
    work = df.copy()
    work["_final_belief"] = final_vals
    work[trait_col] = work[trait_col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    work = work.dropna(subset=["_final_belief"]).copy()
    if work.empty:
        return
    counts = work[trait_col].value_counts()
    categories = counts.index.tolist()
    y_map = {cat: idx for idx, cat in enumerate(categories)}
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 1.0 + 0.55 * len(categories))) if fig_type == "png" else (6.4, max(2.8, 0.8 + 0.45 * len(categories))), dpi=150)
    color_map = _rating_color_map()
    for _, row in work.iterrows():
        cat = str(row[trait_col])
        x = float(row["_final_belief"])
        y = float(y_map.get(cat, 0)) + float(rng.uniform(-0.12, 0.12))
        ax.scatter(x, y, s=38, alpha=0.85, color=color_map.get(int(round(x)), "#777777"), edgecolors="none")
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(categories)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlim(-2.35, 2.35)
    ax.axvline(0, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Final belief")
    ax.set_title(f"Agent-level final belief strip plot by {trait_col}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved final belief strip plot to: {out_path}")

def _agent_summary_with_delta(agent_summary_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(agent_summary_csv_path)
    if df.empty:
        return df

    # Current simulator builds use init_belief/final_belief/delta_from_init, while
    # older plotters expected initial_rating/final_rating/total_delta. Support both
    # so persona-trait plots work across old and new result folders.
    if "total_delta" not in df.columns:
        if "delta_from_init" in df.columns:
            df["total_delta"] = pd.to_numeric(df["delta_from_init"], errors="coerce")
        else:
            initial_candidates = ["initial_rating", "init_belief", "initial_belief"]
            final_candidates = ["final_rating", "final_belief", "current_belief"]
            initial_col = next((c for c in initial_candidates if c in df.columns), None)
            final_col = next((c for c in final_candidates if c in df.columns), None)
            if initial_col and final_col:
                df["total_delta"] = pd.to_numeric(df[final_col], errors="coerce") - pd.to_numeric(df[initial_col], errors="coerce")
            else:
                df["total_delta"] = np.nan
    return df


def _available_persona_trait_columns(agent_summary_df: pd.DataFrame) -> list[str]:
    """Return all persona/descriptive columns that vary across agents.

    This intentionally includes the full persona card surface, not only the three
    core causal traits. Constant fields are skipped so the plot stays useful.
    """
    preferred = [
        "epistemic_profile",
        "institutional_trust",
        "uncertainty_tolerance",
        "evidence_style",
        "official_narrative_suspicion",
        "openness_to_update",
        "value_orientation",
        "social_conformity",
        "agency_vs_fatalism",
        "conflict_style",
        "age",
        "gender",
        "ethnicity",
        "education",
        "occupation",
        "political_leaning",
        "early_life",
        # New structured persona-profile fields emitted by ui_lancher11.py
        "education_level",
        "training_style",
        "domain_familiarity",
        "topic_interest",
        "prior_exposure",
        "age_group",
        "flavor_gender",
        "flavor_ethnicity",
        "lifestyle_notes",
        "tone_hint",
    ]
    skip = {
        "agent_id", "agent_name", "initial_rating", "final_rating", "init_belief",
        "final_belief", "current_belief", "delta_from_init", "total_delta",
        "abs_total_delta", "moves_total", "positive_moves", "negative_moves",
        "no_moves", "first_change_step", "last_change_step", "final_plateau_length",
    }
    found = []
    for col in preferred:
        if col in agent_summary_df.columns:
            vals = agent_summary_df[col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
            if vals.nunique(dropna=True) > 1:
                found.append(col)
    # Include any extra persona-like columns that were added by newer launchers.
    for col in agent_summary_df.columns:
        if col in found or col in skip:
            continue
        low = str(col).lower()
        looks_persona = any(tok in low for tok in [
            "profile", "trust", "uncertainty", "evidence", "suspicion", "openness",
            "orientation", "conform", "agency", "fatalism", "conflict",
            "education", "occupation", "political", "early", "gender", "ethnicity", "age",
            "training", "domain", "familiarity", "topic", "interest", "exposure",
            "lifestyle", "tone", "flavor", "persona", "trait", "background",
        ])
        if not looks_persona:
            continue
        vals = agent_summary_df[col].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        if vals.nunique(dropna=True) > 1:
            found.append(col)
    return found


def save_persona_trait_delta(agent_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    df = _agent_summary_with_delta(agent_summary_csv_path)
    rows = []
    if df.empty or "total_delta" not in df.columns:
        out = pd.DataFrame(columns=["trait", "trait_value", "n_agents", "mean_total_delta", "median_total_delta", "min_total_delta", "max_total_delta"])
        out.to_csv(out_csv_path, index=False)
        return out

    df["total_delta"] = pd.to_numeric(df["total_delta"], errors="coerce")
    traits = _available_persona_trait_columns(df)

    for trait in traits:
        vals = df[trait].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        if vals.nunique(dropna=True) <= 1:
            continue
        tmp = df.assign(_trait_value=vals)
        for value, g in tmp.groupby("_trait_value", dropna=False):
            d = pd.to_numeric(g["total_delta"], errors="coerce").dropna()
            if d.empty:
                continue
            rows.append({
                "trait": trait,
                "trait_value": str(value),
                "n_agents": int(len(d)),
                "mean_total_delta": float(d.mean()),
                "median_total_delta": float(d.median()),
                "min_total_delta": float(d.min()),
                "max_total_delta": float(d.max()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["trait", "mean_total_delta", "trait_value"], ascending=[True, False, True]).reset_index(drop=True)
    out.to_csv(out_csv_path, index=False)
    print(f"Saved persona trait × total-delta CSV to: {out_csv_path}")
    return out


def plot_persona_trait_delta(persona_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if persona_df is None or persona_df.empty:
        return
    traits = persona_df["trait"].dropna().unique().tolist()
    n = len(traits)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3, 2.6 * n)) if fig_type == "png" else (6.4, max(2.4, 2.1 * n)), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, trait in zip(axes, traits):
        sub = persona_df[persona_df["trait"] == trait].copy()
        sub = sub.sort_values("mean_total_delta")
        labels = sub["trait_value"].astype(str) + " (n=" + sub["n_agents"].astype(int).astype(str) + ")"
        ax.barh(labels, sub["mean_total_delta"].astype(float))
        ax.axvline(0, linewidth=1)
        ax.set_title(f"{trait} × total belief delta")
        ax.set_xlabel("Mean total delta (final - initial)")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved persona trait × total-delta plot to: {out_path}")


def _split_pipe_values(value: str) -> list[str]:
    return [x.strip().lower() for x in str(value or "").split("|") if x and x.strip()]


def _rag_direction_label(value: str) -> str:
    dirs = set(_split_pipe_values(value))
    dirs.discard("unknown")
    if not dirs:
        return "unknown"
    has_support = any(d in {"supportive", "support", "claim-supporting"} for d in dirs)
    has_crit = any(d in {"criticism", "critical", "challenge", "challenging", "claim-challenging"} for d in dirs)
    has_context = any(d in {"context", "contextual"} for d in dirs)
    parts = []
    if has_support:
        parts.append("supportive")
    if has_crit:
        parts.append("criticism")
    if has_context:
        parts.append("context")
    return "+".join(parts) if parts else "+".join(sorted(dirs))


def _load_valid_step_summary(step_summary_csv_path: str | None) -> pd.DataFrame:
    if not step_summary_csv_path or not os.path.exists(step_summary_csv_path):
        return pd.DataFrame()
    try:
        return _valid_step_rows(pd.read_csv(step_summary_csv_path))
    except Exception:
        return pd.DataFrame()


def summarize_rag_retrieval_direction_metrics(rag_retrieval_csv_path: str | None, step_summary_csv_path: str | None = None) -> dict:
    if not rag_retrieval_csv_path or not os.path.exists(rag_retrieval_csv_path):
        return {
            "rag_retrieval_rows": 0,
            "rag_supportive_rows": 0,
            "rag_criticism_rows": 0,
            "rag_context_rows": 0,
            "rag_mixed_direction_rows": 0,
        }
    try:
        rag = pd.read_csv(rag_retrieval_csv_path)
    except Exception:
        return {"rag_retrieval_rows": 0}
    if rag.empty:
        return {"rag_retrieval_rows": 0}
    directions_col = "retrieved_directions" if "retrieved_directions" in rag.columns else None
    if directions_col is None:
        return {"rag_retrieval_rows": int(len(rag))}
    labels = rag[directions_col].fillna("").astype(str).map(_rag_direction_label)
    out = {"rag_retrieval_rows": int(len(rag))}
    out["rag_supportive_rows"] = int(labels.str.contains("supportive", regex=False).sum())
    out["rag_criticism_rows"] = int(labels.str.contains("criticism", regex=False).sum())
    out["rag_context_rows"] = int(labels.str.contains("context", regex=False).sum())
    out["rag_mixed_direction_rows"] = int(labels.str.contains("+", regex=False).sum())

    step_df = _load_valid_step_summary(step_summary_csv_path)
    if not step_df.empty and "time_step" in rag.columns and "time_step" in step_df.columns and "delta_belief" in step_df.columns:
        use = rag.copy()
        if "prompt_step" in use.columns:
            # Listener decisions are the rows directly tied to belief deltas; fall back to all if absent.
            step3 = use[use["prompt_step"].astype(str).str.lower().eq("step3")].copy()
            if not step3.empty:
                use = step3
        use["rag_direction_label"] = use[directions_col].fillna("").astype(str).map(_rag_direction_label)
        use["time_step"] = pd.to_numeric(use["time_step"], errors="coerce")
        tmp = step_df.copy()
        tmp["time_step"] = pd.to_numeric(tmp["time_step"], errors="coerce")
        tmp["delta_belief"] = pd.to_numeric(tmp["delta_belief"], errors="coerce")
        joined = use.merge(tmp[["time_step", "delta_belief"]], on="time_step", how="left")
        for label, g in joined.groupby("rag_direction_label"):
            tok = re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "unknown"
            d = pd.to_numeric(g["delta_belief"], errors="coerce").dropna()
            if d.empty:
                continue
            out[f"rag_{tok}_mean_delta"] = float(d.mean())
            out[f"rag_{tok}_positive_move_rate"] = float((d > 0).mean())
            out[f"rag_{tok}_negative_move_rate"] = float((d < 0).mean())
            out[f"rag_{tok}_no_move_rate"] = float((d == 0).mean())
    return out


def save_rag_direction_breakdown(rag_retrieval_csv_path: str, step_summary_csv_path: str, out_csv_path: str) -> pd.DataFrame:
    rag = pd.read_csv(rag_retrieval_csv_path)
    if rag.empty or "retrieved_directions" not in rag.columns:
        out = pd.DataFrame(columns=["rag_direction_label", "retrieval_rows", "mean_delta", "positive_move_rate", "negative_move_rate", "no_move_rate"])
        out.to_csv(out_csv_path, index=False)
        return out
    if "prompt_step" in rag.columns:
        step3 = rag[rag["prompt_step"].astype(str).str.lower().eq("step3")].copy()
        if not step3.empty:
            rag = step3
    rag["rag_direction_label"] = rag["retrieved_directions"].fillna("").astype(str).map(_rag_direction_label)
    rag["time_step"] = pd.to_numeric(rag.get("time_step"), errors="coerce")
    step_df = _load_valid_step_summary(step_summary_csv_path)
    if not step_df.empty and "time_step" in step_df.columns:
        step_df = step_df.copy()
        step_df["time_step"] = pd.to_numeric(step_df["time_step"], errors="coerce")
        keep_cols = [c for c in ["time_step", "delta_belief", "listener_pre_belief", "listener_post_belief", "speaker_belief"] if c in step_df.columns]
        joined = rag.merge(step_df[keep_cols], on="time_step", how="left")
    else:
        joined = rag.copy()
        joined["delta_belief"] = np.nan
    rows = []
    for label, g in joined.groupby("rag_direction_label", dropna=False):
        d = pd.to_numeric(g.get("delta_belief"), errors="coerce")
        rows.append({
            "rag_direction_label": str(label),
            "retrieval_rows": int(len(g)),
            "mean_delta": float(d.mean()) if d.notna().any() else np.nan,
            "positive_move_rate": float((d > 0).mean()) if d.notna().any() else np.nan,
            "negative_move_rate": float((d < 0).mean()) if d.notna().any() else np.nan,
            "no_move_rate": float((d == 0).mean()) if d.notna().any() else np.nan,
            "mean_effective_top_k": float(pd.to_numeric(g.get("effective_top_k", pd.Series(dtype=float)), errors="coerce").mean()) if "effective_top_k" in g.columns else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("retrieval_rows", ascending=False).reset_index(drop=True)
    out.to_csv(out_csv_path, index=False)
    print(f"Saved RAG direction breakdown CSV to: {out_csv_path}")
    return out


def plot_rag_direction_breakdown(rag_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if rag_df is None or rag_df.empty:
        return
    labels = rag_df["rag_direction_label"].astype(str)
    x = np.arange(len(rag_df))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6) if fig_type == "png" else (6.4, 4.8), dpi=150)
    axes[0].bar(x, rag_df["retrieval_rows"].astype(float))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Retrieval rows")
    axes[0].set_title("RAG retrieval direction breakdown")
    axes[0].grid(True, axis="y", alpha=0.3)
    if "mean_delta" in rag_df.columns:
        axes[1].bar(x, pd.to_numeric(rag_df["mean_delta"], errors="coerce"))
        axes[1].axhline(0, linewidth=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_ylabel("Mean listener delta")
        axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RAG direction breakdown plot to: {out_path}")


def _edge_pairs_from_files(edge_files: list[str]) -> list[tuple[str, str]]:
    pairs = set()
    for fp in edge_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if not {"i_name", "j_name"}.issubset(df.columns):
            continue
        for _, r in df.iterrows():
            a, b = str(r["i_name"]), str(r["j_name"])
            if not a or not b:
                continue
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _neighbor_map_from_edge_files(edge_files: list[str]) -> dict[str, set[str]]:
    out = {}
    for a, b in _edge_pairs_from_files(edge_files):
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def plot_ego_network_belief_timelines(csv_path: str, out_prefix: str, version_set: str, num_agents: int, out_path: str, fig_type: str) -> None:
    results_dir = os.path.dirname(csv_path)
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    if not edge_files:
        return
    neighbors = _neighbor_map_from_edge_files(edge_files)
    if not neighbors:
        return
    df = pd.read_csv(csv_path)
    if df.empty or "time_step" not in df.columns:
        return
    agent_cols = list(df.columns)[-num_agents:]
    t = df["time_step"].to_numpy()
    n = len(agent_cols)
    cols = 2 if n <= 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 2.8), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, agent in zip(axes, agent_cols):
        for nb in sorted(neighbors.get(agent, [])):
            if nb in df.columns:
                ax.plot(t, pd.to_numeric(df[nb], errors="coerce"), linewidth=0.9, alpha=0.35)
                if len(t) > 0:
                    ax.text(t[-1], float(pd.to_numeric(df[nb], errors="coerce").iloc[-1]), f" {nb}", fontsize=5, alpha=0.45, va="center")
        ax.plot(t, pd.to_numeric(df[agent], errors="coerce"), linewidth=2.6, label=agent)
        ax.set_title(f"{agent} ego timeline", fontsize=9)
        ax.set_ylim(-2.25, 2.25)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(True, alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Ego-network belief timelines: focal agent thick, neighbours thin", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ego-network belief timelines to: {out_path}")


def save_agent_neighborhood_alignment(csv_path: str, out_prefix: str, version_set: str, num_agents: int, out_csv_path: str) -> pd.DataFrame:
    """Long-form table: each agent's belief relative to its neighbourhood over time.

    Columns include the focal rating, neighbour mean/min/max rating, and the
    focal-minus-neighbour-mean difference. This is meant to show whether an
    agent is embedded in a supportive local pocket, a hostile bridge, or a
    mixed neighbourhood.
    """
    results_dir = os.path.dirname(csv_path)
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    neighbors = _neighbor_map_from_edge_files(edge_files) if edge_files else {}
    df = pd.read_csv(csv_path)
    if df.empty or "time_step" not in df.columns or not neighbors:
        out = pd.DataFrame(columns=[
            "time_step", "agent_name", "agent_belief", "neighbor_count",
            "neighbor_mean_belief", "neighbor_min_belief", "neighbor_max_belief",
            "belief_minus_neighbor_mean", "same_sign_neighbor_share",
        ])
        out.to_csv(out_csv_path, index=False)
        return out

    agent_cols = list(df.columns)[-num_agents:]
    rows = []
    for _, row in df.iterrows():
        try:
            time_step = int(row["time_step"])
        except Exception:
            time_step = row.get("time_step", "")
        for agent in agent_cols:
            if agent not in df.columns:
                continue
            val = pd.to_numeric(pd.Series([row.get(agent)]), errors="coerce").iloc[0]
            nb_vals = []
            for nb in sorted(neighbors.get(agent, [])):
                if nb in df.columns:
                    nb_val = pd.to_numeric(pd.Series([row.get(nb)]), errors="coerce").iloc[0]
                    if pd.notna(nb_val):
                        nb_vals.append(float(nb_val))
            if not nb_vals or pd.isna(val):
                rows.append({
                    "time_step": time_step, "agent_name": agent, "agent_belief": float(val) if pd.notna(val) else np.nan,
                    "neighbor_count": 0, "neighbor_mean_belief": np.nan, "neighbor_min_belief": np.nan,
                    "neighbor_max_belief": np.nan, "belief_minus_neighbor_mean": np.nan,
                    "same_sign_neighbor_share": np.nan,
                })
                continue
            nb_arr = np.asarray(nb_vals, dtype=float)
            agent_sign = 0 if float(val) == 0 else (1 if float(val) > 0 else -1)
            nb_signs = np.where(nb_arr > 0, 1, np.where(nb_arr < 0, -1, 0))
            same_share = float(np.mean(nb_signs == agent_sign)) if len(nb_signs) else np.nan
            mean_nb = float(np.mean(nb_arr))
            rows.append({
                "time_step": time_step,
                "agent_name": agent,
                "agent_belief": float(val),
                "neighbor_count": int(len(nb_vals)),
                "neighbor_mean_belief": mean_nb,
                "neighbor_min_belief": float(np.min(nb_arr)),
                "neighbor_max_belief": float(np.max(nb_arr)),
                "belief_minus_neighbor_mean": float(float(val) - mean_nb),
                "same_sign_neighbor_share": same_share,
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_csv_path, index=False)
    print(f"Saved agent-neighbourhood alignment CSV to: {out_csv_path}")
    return out


def plot_agent_neighborhood_alignment(alignment_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    """Panel plot: focal agent rating vs mean/min/max neighbourhood rating."""
    if alignment_df is None or alignment_df.empty:
        return
    required = {"time_step", "agent_name", "agent_belief", "neighbor_mean_belief"}
    if not required.issubset(alignment_df.columns):
        return
    agents = alignment_df["agent_name"].dropna().astype(str).unique().tolist()
    if not agents:
        return
    n = len(agents)
    cols = 2 if n <= 6 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.9), dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, agent in zip(axes, agents):
        sub = alignment_df[alignment_df["agent_name"].astype(str) == str(agent)].sort_values("time_step")
        t = pd.to_numeric(sub["time_step"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["agent_belief"], errors="coerce").to_numpy(dtype=float)
        nb_mean = pd.to_numeric(sub["neighbor_mean_belief"], errors="coerce").to_numpy(dtype=float)
        nb_min = pd.to_numeric(sub.get("neighbor_min_belief", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
        nb_max = pd.to_numeric(sub.get("neighbor_max_belief", pd.Series(np.nan, index=sub.index)), errors="coerce").to_numpy(dtype=float)
        ax.fill_between(t, nb_min, nb_max, alpha=0.16, label="neighbour range")
        ax.plot(t, nb_mean, linestyle="--", linewidth=1.8, alpha=0.9, label="neighbour mean")
        ax.plot(t, y, linewidth=2.6, label="agent")
        final_diff = np.nan
        if len(y) and len(nb_mean):
            final_diff = y[-1] - nb_mean[-1] if np.isfinite(y[-1]) and np.isfinite(nb_mean[-1]) else np.nan
        suffix = "" if not np.isfinite(final_diff) else f"\nfinal Δ vs neigh={final_diff:+.2f}"
        ax.set_title(f"{agent}{suffix}", fontsize=8)
        ax.set_ylim(-2.25, 2.25)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.grid(True, alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels() if len(axes) else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Agent ratings relative to neighbourhood", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved agent-neighbourhood alignment plot to: {out_path}")


def save_edge_belief_differential(csv_path: str, out_prefix: str, version_set: str, num_agents: int, out_csv_path: str, sample_every: int = 10) -> pd.DataFrame:
    results_dir = os.path.dirname(csv_path)
    edge_files = _collect_edge_files(results_dir, out_prefix, version_set)
    pairs = _edge_pairs_from_files(edge_files)
    df = pd.read_csv(csv_path)
    if df.empty or not pairs or "time_step" not in df.columns:
        out = pd.DataFrame()
        out.to_csv(out_csv_path, index=False)
        return out
    time_values = [int(x) for x in df["time_step"].dropna().astype(int).tolist()]
    sampled = [t for t in time_values if t == min(time_values) or t % sample_every == 0]
    if time_values and time_values[-1] not in sampled:
        sampled.append(time_values[-1])
    rows = []
    by_step = df.set_index("time_step")
    for a, b in pairs:
        row = {"edge": f"{a} — {b}", "agent_i": a, "agent_j": b}
        for t in sampled:
            if t in by_step.index and a in by_step.columns and b in by_step.columns:
                va = pd.to_numeric(pd.Series([by_step.loc[t, a]]), errors="coerce").iloc[0]
                vb = pd.to_numeric(pd.Series([by_step.loc[t, b]]), errors="coerce").iloc[0]
                row[f"step_{int(t)}"] = abs(float(va) - float(vb)) if pd.notna(va) and pd.notna(vb) else np.nan
            else:
                row[f"step_{int(t)}"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv_path, index=False)
    print(f"Saved edge-level belief differential CSV to: {out_csv_path}")
    return out


def plot_edge_belief_differential(edge_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if edge_df is None or edge_df.empty:
        return
    step_cols = [c for c in edge_df.columns if str(c).startswith("step_")]
    if not step_cols:
        return
    data = edge_df[step_cols].to_numpy(dtype=float)
    labels_y = edge_df["edge"].astype(str).tolist()
    labels_x = [c.replace("step_", "") for c in step_cols]
    fig_h = max(4.0, 0.35 * len(labels_y))
    fig, ax = plt.subplots(figsize=(9, fig_h) if fig_type == "png" else (7.2, min(fig_h, 9)), dpi=150)
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Edge")
    ax.set_title("Edge-level belief differential |belief_i - belief_j|")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Absolute belief difference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved edge-level belief differential heatmap to: {out_path}")


def save_run_summary_card_data(report_df: pd.DataFrame, agent_summary_csv_path: str | None, out_csv_path: str) -> pd.DataFrame:
    if report_df is None or report_df.empty:
        out = pd.DataFrame()
        out.to_csv(out_csv_path, index=False)
        return out
    row = dict(report_df.iloc[0])
    top_movers = []
    if agent_summary_csv_path and os.path.exists(agent_summary_csv_path):
        try:
            agents = _agent_summary_with_delta(agent_summary_csv_path)
            if not agents.empty and "agent_name" in agents.columns and "total_delta" in agents.columns:
                agents["abs_total_delta"] = pd.to_numeric(agents["total_delta"], errors="coerce").abs()
                use_cols = [c for c in ["agent_name", "total_delta", "openness_to_update", "institutional_trust", "official_narrative_suspicion", "epistemic_profile"] if c in agents.columns]
                for _, r in agents.sort_values("abs_total_delta", ascending=False).head(5).iterrows():
                    traits = []
                    for c in ["openness_to_update", "institutional_trust", "official_narrative_suspicion", "epistemic_profile"]:
                        if c in r.index and str(r.get(c, "")).strip():
                            traits.append(f"{c}={r.get(c)}")
                    top_movers.append(f"{r.get('agent_name')}: Δ={r.get('total_delta')}" + (" [" + "; ".join(traits) + "]" if traits else ""))
                row["top_movers_by_persona"] = " | ".join(top_movers)
        except Exception:
            row["top_movers_by_persona"] = ""
    out = pd.DataFrame([row])
    out.to_csv(out_csv_path, index=False)
    print(f"Saved run summary card CSV to: {out_csv_path}")
    return out


def plot_run_summary_card(card_df: pd.DataFrame, out_path: str, fig_type: str) -> None:
    if card_df is None or card_df.empty:
        return
    r = card_df.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 8.5) if fig_type == "png" else (8.5, 6.6), dpi=150)
    ax.axis("off")
    title = str(r.get("run_stem", r.get("run_id", "run")))
    lines = [
        "RUN COMPARISON SUMMARY CARD",
        "Purpose: cross-run stacking, not single-run review.",
        "",
        f"Run: {title}",
        f"Version / condition: {r.get('version_set', '')}",
        f"World / RAG / fact pack: {r.get('world', '')} / {r.get('rag_content_mode', '')} / {r.get('fact_pack_mode', '')}",
        f"Network / homophily: {r.get('selection_mode', '')} / {r.get('homophily_mode', '')}",
        f"Seed / agents / final step: {r.get('seed', '')} / {r.get('num_agents', '')} / {r.get('final_time_step', '')}",
        "",
        f"Endpoint B/D/P: {float(r.get('final_B', 0.0)):.2f} / {float(r.get('final_D', 0.0)):.2f} / {float(r.get('final_P', 0.0)):.2f}",
        f"Final distribution: {r.get('final_distribution', '')}",
        f"Move rate: {float(r.get('total_moves_rate', 0.0)):.2f}    +move rate: {float(r.get('positive_moves_rate', 0.0)):.2f}    -move rate: {float(r.get('negative_moves_rate', 0.0)):.2f}",
        f"Repair rate: {float(r.get('hard_repairs_rate', 0.0)):.3f}    Soft-cleanup rate: {float(r.get('soft_cleanups_rate', 0.0)):.3f}",
        f"RAG rows support/challenge/context: {int(r.get('rag_supportive_rows', 0) or 0)} / {int(r.get('rag_criticism_rows', 0) or 0)} / {int(r.get('rag_context_rows', 0) or 0)}",
        "",
        "Top movers by persona:",
    ]
    movers = str(r.get("top_movers_by_persona", "") or "")
    if movers:
        lines.extend(["- " + x.strip() for x in movers.split(" | ") if x.strip()])
    else:
        lines.append("- unavailable")
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved run summary card to: {out_path}")



# -----------------------------
# Network/influence diagnostic graphs from qwen10 metrics CSVs
# -----------------------------


def _find_metric_csv_by_token(args, token: str) -> str:
    """Find diagnostic CSVs from both compact and legacy names.

    Legacy names include network_structure_summary / network_agent_metrics /
    interaction_influence_events / agent_influence_summary / exposure_persuasion_summary.
    qwen13+ compact names use structure_summary / agent_metrics /
    influence_events / influence_summary / exposure_persuasion.
    """
    aliases = {
        "network_structure_summary": ["structure_summary", "network_structure_summary"],
        "network_agent_metrics": ["agent_metrics", "network_agent_metrics"],
        "interaction_influence_events": ["influence_events", "interaction_influence_events"],
        "agent_influence_summary": ["influence_summary", "agent_influence_summary"],
        "exposure_persuasion_summary": ["exposure_persuasion", "exposure_persuasion_summary"],
        "network_hub_metrics": ["hub_metrics", "network_hub_metrics"],
    }
    toks = aliases.get(str(token), [str(token)])
    short = _short_run_base_from_args(args)
    base_name = args.output_file if args.output_file else f"seed{args.seed}"
    old_stem = f"{base_name}_{args.num_agents}_{args.num_steps}_{args.version_set}"
    candidate_names = []
    for tok in toks:
        candidate_names.extend([
            f"{short}_{tok}.csv",
            f"{old_stem}_{tok}_{args.date}_{args.distribution}.csv",
            f"{old_stem}_{args.date}_{tok}_{args.distribution}.csv",
        ])
    return _find_existing_csv(args, description=f"{token} CSV", candidate_names=candidate_names, glob_tokens=toks)

def build_input_network_structure_summary_csv_path(args) -> str:
    return _find_metric_csv_by_token(args, "network_structure_summary")


def build_input_network_agent_metrics_csv_path(args) -> str:
    return _find_metric_csv_by_token(args, "network_agent_metrics")


def build_input_interaction_influence_events_csv_path(args) -> str:
    return _find_metric_csv_by_token(args, "interaction_influence_events")


def build_input_agent_influence_summary_csv_path(args) -> str:
    return _find_metric_csv_by_token(args, "agent_influence_summary")


def build_input_exposure_persuasion_summary_csv_path(args) -> str:
    return _find_metric_csv_by_token(args, "exposure_persuasion_summary")


def build_output_network_degree_distribution_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_network_degree_distribution.{args.figure_file_type}")


def build_output_degree_vs_total_delta_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_network_degree_vs_total_delta.{args.figure_file_type}")


def build_output_degree_vs_influence_score_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_network_degree_vs_influence_score.{args.figure_file_type}")


def build_output_bridge_vs_total_delta_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_bridge_score_vs_total_delta.{args.figure_file_type}")


def build_output_bridge_vs_influence_score_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_bridge_score_vs_influence_score.{args.figure_file_type}")


def build_output_exposure_adjusted_persuasion_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_exposure_adjusted_persuasion.{args.figure_file_type}")


def build_output_ba_hub_target_status_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_ba_hub_target_status.{args.figure_file_type}")


def build_output_stabilisation_timeline_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_stabilisation_timeline.{args.figure_file_type}")


def build_output_move_count_over_time_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_move_count_over_time.{args.figure_file_type}")


def build_output_centrality_weighted_B_path(args, csv_path: str) -> str:
    out_dir = _build_plot_subdir(csv_path, args)
    run_stem = _get_run_stem(csv_path)
    return os.path.join(out_dir, f"{run_stem}_centrality_weighted_B_over_time.{args.figure_file_type}")


def _coerce_num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series([default] * (0 if df is None else len(df)), dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _agent_display_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for _, row in df.iterrows():
        name = str(row.get("agent_name", "")).strip()
        aid = str(row.get("agent_id", row.get("agent_local_idx", ""))).strip()
        if name and aid:
            labels.append(f"{aid}: {name}")
        elif name:
            labels.append(name)
        elif aid:
            labels.append(aid)
        else:
            labels.append(str(len(labels)))
    return labels


def _merge_agent_network_and_influence(network_agent_csv_path: str, agent_influence_csv_path: str | None = None) -> pd.DataFrame:
    base = pd.read_csv(network_agent_csv_path)
    if base.empty:
        return base
    if agent_influence_csv_path and os.path.exists(agent_influence_csv_path):
        try:
            inf = pd.read_csv(agent_influence_csv_path)
            if not inf.empty and "agent_id" in base.columns and "agent_id" in inf.columns:
                keep = [c for c in inf.columns if c not in set(base.columns) or c == "agent_id"]
                base = base.merge(inf[keep], on="agent_id", how="left", suffixes=("", "_influence"))
        except Exception:
            pass
    return base


def plot_network_degree_distribution(network_agent_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(network_agent_csv_path)
    if df.empty or "network_degree" not in df.columns:
        return
    degrees = _coerce_num(df, "network_degree")
    fig, ax = plt.subplots(figsize=(6, 4) if fig_type == "png" else (4.8, 3.2), dpi=150)
    bins = np.arange(int(degrees.min()), int(degrees.max()) + 2) - 0.5 if len(degrees) else 10
    ax.hist(degrees, bins=bins, edgecolor="black")
    ax.set_title("Network degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Agent count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved degree distribution plot to: {out_path}")


def plot_scatter_with_labels(df: pd.DataFrame, x_col: str, y_col: str, out_path: str, fig_type: str, title: str, xlabel: str, ylabel: str) -> None:
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col]).copy()
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.8) if fig_type == "png" else (5.6, 3.8), dpi=150)
    marker_sizes = None
    if "network_degree" in work.columns and x_col != "network_degree":
        deg = pd.to_numeric(work["network_degree"], errors="coerce").fillna(1)
        marker_sizes = 40 + 25 * deg
    ax.scatter(work[x_col], work[y_col], s=marker_sizes if marker_sizes is not None else 80, alpha=0.8)
    for label, x, y in zip(_agent_display_labels(work), work[x_col], work[y_col]):
        ax.annotate(str(label), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6, alpha=0.8)
    ax.axhline(0, linewidth=1, linestyle="--", alpha=0.5)
    if x_col not in {"bridge_score"}:
        ax.axvline(float(work[x_col].median()), linewidth=1, linestyle=":", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {title} plot to: {out_path}")


def plot_exposure_adjusted_persuasion(exposure_csv_path: str, out_path: str, fig_type: str) -> None:
    df = pd.read_csv(exposure_csv_path)
    if df.empty:
        return
    row = df.iloc[0].to_dict()
    sides = ["support", "oppose", "neutral"]
    labels = ["support speaker", "oppose speaker", "neutral speaker"]
    move_rates = [float(row.get(f"move_rate_given_{s}_speaker", 0) or 0) for s in sides]
    toward_rates = [float(row.get(f"toward_speaker_rate_{s}", 0) or 0) for s in sides]
    net_delta = [float(row.get(f"net_delta_per_{s}_exposure", 0) or 0) for s in sides]
    exposures = [int(float(row.get(f"{s}_speaker_exposures", 0) or 0)) for s in sides]

    x = np.arange(len(sides))
    width = 0.27
    fig, ax1 = plt.subplots(figsize=(8, 4.8) if fig_type == "png" else (6.4, 3.8), dpi=150)
    ax1.bar(x - width, move_rates, width, label="move rate")
    ax1.bar(x, toward_rates, width, label="toward-speaker rate")
    ax1.bar(x + width, net_delta, width, label="net delta/exposure")
    ax1.axhline(0, linewidth=1, linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{lab}\nN={n}" for lab, n in zip(labels, exposures)])
    ax1.set_ylabel("Rate / delta per exposure")
    ax1.set_title("Exposure-adjusted persuasion")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exposure-adjusted persuasion plot to: {out_path}")


def plot_ba_hub_target_status(network_agent_csv_path: str, agent_influence_csv_path: str | None, out_path: str, fig_type: str) -> None:
    df = _merge_agent_network_and_influence(network_agent_csv_path, agent_influence_csv_path)
    if df.empty or "network_degree" not in df.columns:
        return
    df = df.copy()
    df["network_degree"] = pd.to_numeric(df["network_degree"], errors="coerce").fillna(0)
    df["initial_belief"] = pd.to_numeric(df.get("initial_belief", 0), errors="coerce").fillna(0)
    df["final_belief"] = pd.to_numeric(df.get("final_belief", 0), errors="coerce").fillna(0)
    df["mean_abs_listener_delta_caused"] = pd.to_numeric(df.get("mean_abs_listener_delta_caused", 0), errors="coerce").fillna(0)
    df["ba_hub_targeted_flag"] = pd.to_numeric(df.get("ba_hub_targeted_flag", 0), errors="coerce").fillna(0).astype(int)
    df["network_top_hub_flag"] = pd.to_numeric(df.get("network_top_hub_flag", 0), errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["network_degree", "agent_id" if "agent_id" in df.columns else "agent_name"], ascending=[False, True]).reset_index(drop=True)

    y = np.arange(len(df))
    fig_h = max(4.5, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_h) if fig_type == "png" else (7.2, min(fig_h, 9)), dpi=150)
    labels = _agent_display_labels(df)
    sizes = 70 + 140 * df["mean_abs_listener_delta_caused"].clip(lower=0)
    ax.scatter(df["network_degree"], y, s=sizes, alpha=0.75)
    for i, row in df.iterrows():
        tag_parts = []
        if int(row.get("ba_hub_targeted_flag", 0)):
            tag_parts.append("target")
        if int(row.get("network_top_hub_flag", 0)):
            tag_parts.append("top-hub")
        if int(row.get("ba_actual_hub_assigned_flag", 0) or 0):
            tag_parts.append("actual-assigned")
        tag = " | ".join(tag_parts)
        belief = f"{int(row['initial_belief'])}→{int(row['final_belief'])}"
        if tag:
            ax.annotate(f"{belief}; {tag}", (row["network_degree"], i), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
        else:
            ax.annotate(belief, (row["network_degree"], i), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center", alpha=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Degree")
    ax.set_title("BA hub target status: structure, belief change, and influence")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BA hub target-status plot to: {out_path}")


def plot_move_count_over_time(influence_events_csv_path: str, out_path: str, fig_type: str, window: int = 25) -> None:
    df = pd.read_csv(influence_events_csv_path)
    if df.empty or "time_step" not in df.columns:
        return
    df = df.copy()
    df["time_step"] = pd.to_numeric(df["time_step"], errors="coerce")
    df["moved_flag"] = pd.to_numeric(df.get("moved_flag", 0), errors="coerce").fillna(0)
    df["abs_listener_delta"] = pd.to_numeric(df.get("abs_listener_delta", df["moved_flag"]), errors="coerce").fillna(0)
    by_step = df.groupby("time_step").agg(moves=("moved_flag", "sum"), abs_delta=("abs_listener_delta", "sum")).reset_index().sort_values("time_step")
    by_step["rolling_moves"] = by_step["moves"].rolling(window=window, min_periods=1).sum()
    by_step["rolling_abs_delta"] = by_step["abs_delta"].rolling(window=window, min_periods=1).sum()
    fig, ax = plt.subplots(figsize=(7, 4.2) if fig_type == "png" else (5.6, 3.3), dpi=150)
    ax.plot(by_step["time_step"], by_step["moves"], label="moves per step", linewidth=1.0, alpha=0.65)
    ax.plot(by_step["time_step"], by_step["rolling_moves"], label=f"rolling moves ({window})", linewidth=2.0)
    ax.plot(by_step["time_step"], by_step["rolling_abs_delta"], label=f"rolling abs delta ({window})", linewidth=1.6)
    ax.set_title("Movement count over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved move-count-over-time plot to: {out_path}")


def _first_numeric_from_df(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.iloc[0])


def plot_stabilisation_timeline(bdp_csv_path: str, network_structure_csv_path: str | None, out_path: str, fig_type: str) -> None:
    bdp = pd.read_csv(bdp_csv_path)
    if bdp.empty or "time_step" not in bdp.columns or "B" not in bdp.columns:
        return
    structure = pd.DataFrame()
    if network_structure_csv_path and os.path.exists(network_structure_csv_path):
        try:
            structure = pd.read_csv(network_structure_csv_path)
        except Exception:
            structure = pd.DataFrame()
    last_move = _first_numeric_from_df(structure, "last_move_step")
    stab_B = _first_numeric_from_df(structure, "stabilisation_step_B")
    stab_dist = _first_numeric_from_df(structure, "stabilisation_step_distribution")

    fig, ax = plt.subplots(figsize=(7, 4.2) if fig_type == "png" else (5.6, 3.3), dpi=150)
    ax.plot(bdp["time_step"], bdp["B"], label="B", linewidth=2.0)
    if "D" in bdp.columns:
        ax.plot(bdp["time_step"], bdp["D"], label="D", linewidth=1.5)
    if "P" in bdp.columns:
        ax.plot(bdp["time_step"], bdp["P"], label="P", linewidth=1.5)
    for step, label in [(last_move, "last move"), (stab_B, "B stabilised"), (stab_dist, "distribution stabilised")]:
        if step is not None and np.isfinite(step):
            ax.axvline(step, linestyle="--", linewidth=1.2, alpha=0.75)
            ax.text(step, ax.get_ylim()[1], f" {label}: {int(step)}", rotation=90, va="top", fontsize=7)
    ax.set_title("Stabilisation timeline")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stabilisation timeline plot to: {out_path}")


def plot_centrality_weighted_B_over_time(csv_path: str, network_agent_csv_path: str, out_path: str, fig_type: str, num_agents: int) -> None:
    df = pd.read_csv(csv_path)
    agent_df = pd.read_csv(network_agent_csv_path)
    if df.empty or agent_df.empty or "time_step" not in df.columns or "network_degree" not in agent_df.columns:
        return
    agent_cols = list(df.columns)[-num_agents:]
    # Match degrees by agent_name when available; otherwise use CSV order.
    degree_by_name = {}
    if "agent_name" in agent_df.columns:
        for _, row in agent_df.iterrows():
            degree_by_name[str(row.get("agent_name", "")).strip()] = float(pd.to_numeric(pd.Series([row.get("network_degree", 0)]), errors="coerce").fillna(0).iloc[0])
    degrees = np.asarray([degree_by_name.get(str(col).strip(), np.nan) for col in agent_cols], dtype=float)
    if np.isnan(degrees).all():
        degrees = pd.to_numeric(agent_df["network_degree"], errors="coerce").fillna(0).to_numpy(dtype=float)[:len(agent_cols)]
    degrees = np.nan_to_num(degrees, nan=0.0)
    if degrees.sum() <= 0:
        degrees = np.ones_like(degrees)
    vals = df[agent_cols].to_numpy(dtype=float)
    weighted_B = (vals * degrees.reshape(1, -1)).sum(axis=1) / degrees.sum()
    unweighted_B = vals.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.2) if fig_type == "png" else (5.6, 3.3), dpi=150)
    ax.plot(df["time_step"], unweighted_B, label="unweighted B", linewidth=1.8)
    ax.plot(df["time_step"], weighted_B, label="degree-weighted B", linewidth=2.0)
    ax.axhline(0, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title("Centrality-weighted mean belief over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean belief")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved centrality-weighted B plot to: {out_path}")


def plot_network_influence_diagnostics(
    *,
    args,
    csv_path: str,
    bdp_csv_path: str,
    network_structure_csv_path: str | None,
    network_agent_csv_path: str,
    interaction_influence_csv_path: str | None,
    agent_influence_csv_path: str | None,
    exposure_persuasion_csv_path: str | None,
) -> None:
    """Generate the high-value network/influence diagnostic graphs.

    The function is deliberately tolerant: each graph is skipped independently if
    its required CSV or columns are absent, so old runs remain plottable.
    """
    try:
        plot_network_degree_distribution(
            network_agent_csv_path=network_agent_csv_path,
            out_path=build_output_network_degree_distribution_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[network-influence] Skipping degree distribution: {e}")

    try:
        merged = _merge_agent_network_and_influence(network_agent_csv_path, agent_influence_csv_path)
        plot_scatter_with_labels(
            merged, "network_degree", "total_delta",
            build_output_degree_vs_total_delta_path(args, csv_path),
            args.figure_file_type,
            "Network degree vs total belief delta",
            "Degree", "Final - initial belief",
        )
        influence_col = "mean_abs_listener_delta_caused" if "mean_abs_listener_delta_caused" in merged.columns else "listener_move_rate"
        plot_scatter_with_labels(
            merged, "network_degree", influence_col,
            build_output_degree_vs_influence_score_path(args, csv_path),
            args.figure_file_type,
            "Network degree vs influence score",
            "Degree", influence_col,
        )
        if "bridge_score" in merged.columns:
            plot_scatter_with_labels(
                merged, "bridge_score", "total_delta",
                build_output_bridge_vs_total_delta_path(args, csv_path),
                args.figure_file_type,
                "Bridge score vs total belief delta",
                "Bridge score", "Final - initial belief",
            )
            plot_scatter_with_labels(
                merged, "bridge_score", influence_col,
                build_output_bridge_vs_influence_score_path(args, csv_path),
                args.figure_file_type,
                "Bridge score vs influence score",
                "Bridge score", influence_col,
            )
    except Exception as e:
        print(f"[network-influence] Skipping scatter diagnostics: {e}")

    if exposure_persuasion_csv_path and os.path.exists(exposure_persuasion_csv_path):
        try:
            plot_exposure_adjusted_persuasion(
                exposure_csv_path=exposure_persuasion_csv_path,
                out_path=build_output_exposure_adjusted_persuasion_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e:
            print(f"[network-influence] Skipping exposure-adjusted persuasion: {e}")

    try:
        plot_ba_hub_target_status(
            network_agent_csv_path=network_agent_csv_path,
            agent_influence_csv_path=agent_influence_csv_path,
            out_path=build_output_ba_hub_target_status_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[network-influence] Skipping BA hub target status: {e}")

    if interaction_influence_csv_path and os.path.exists(interaction_influence_csv_path):
        try:
            plot_move_count_over_time(
                influence_events_csv_path=interaction_influence_csv_path,
                out_path=build_output_move_count_over_time_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e:
            print(f"[network-influence] Skipping move-count-over-time: {e}")

    try:
        plot_stabilisation_timeline(
            bdp_csv_path=bdp_csv_path,
            network_structure_csv_path=network_structure_csv_path,
            out_path=build_output_stabilisation_timeline_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[network-influence] Skipping stabilisation timeline: {e}")

    try:
        plot_centrality_weighted_B_over_time(
            csv_path=csv_path,
            network_agent_csv_path=network_agent_csv_path,
            out_path=build_output_centrality_weighted_B_path(args, csv_path),
            fig_type=args.figure_file_type,
            num_agents=args.num_agents,
        )
    except Exception as e:
        print(f"[network-influence] Skipping centrality-weighted B: {e}")

def main():
    args = parse_args()
    csv_path = build_input_csv_path(args)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find input CSV: {csv_path}")

    out_ts_path = build_output_timeseries_path(args, csv_path)
    plot_opinion_trajectories(
        csv_path=csv_path,
        out_path=out_ts_path,
        no_annotation=args.no_annotation,
        fig_type=args.figure_file_type,
        num_agents=args.num_agents,
    )

    # Standard graph for every experiment: one overlaid belief line per agent.
    plot_individual_belief_trajectories(
        csv_path=csv_path,
        out_path=build_output_individual_trajectories_path(args, csv_path),
        fig_type=args.figure_file_type,
        num_agents=args.num_agents,
    )

    out_init_hist_path = build_output_initial_histogram_path(args, csv_path)
    plot_initial_distribution(
        csv_path=csv_path,
        out_path=out_init_hist_path,
        num_agents=args.num_agents,
        fig_type=args.figure_file_type,
    )

    out_hist_path = build_output_histogram_path(args, csv_path)
    plot_final_distribution(
        csv_path=csv_path,
        out_path=out_hist_path,
        num_agents=args.num_agents,
        fig_type=args.figure_file_type,
    )
    plot_paired_initial_final_distribution(
        csv_path=csv_path,
        out_path=build_output_paired_distribution_plot_path(args, csv_path),
        fig_type=args.figure_file_type,
        num_agents=args.num_agents,
    )
    out_bdp_csv = build_output_bdp_timeseries_path(args, csv_path)
    save_BDP_timeseries(
        csv_path=csv_path,
        out_path=out_bdp_csv,
        num_agents=args.num_agents,
    )

    out_bdp_plot = build_output_bdp_plot_path(args, csv_path)
    plot_BDP_timeseries_from_csv(
        bdp_csv_path=out_bdp_csv,
        out_path=out_bdp_plot,
        fig_type=args.figure_file_type,
    )

    # Presentation-ready report outputs. These are deterministic post-processing layers
    # built from the existing CSVs, so they do not affect the simulation dynamics.
    step_summary_csv_for_report = _try_builder(build_input_step_summary_csv_path, args)
    agent_summary_csv_for_report = _try_builder(build_input_agent_summary_csv_path, args)
    run_metrics_csv_for_report = _try_builder(build_input_run_metrics_csv_path, args)
    repair_events_csv_for_report = _try_builder(build_input_repair_events_csv_path, args)
    rag_retrieval_csv_for_report = _try_builder(build_input_rag_retrieval_csv_path, args)

    try:
        report_df = save_run_report_summary(
            csv_path=csv_path,
            out_csv_path=build_output_run_report_summary_csv_path(args, csv_path),
            args=args,
            step_summary_csv_path=step_summary_csv_for_report,
            agent_summary_csv_path=agent_summary_csv_for_report,
            run_metrics_csv_path=run_metrics_csv_for_report,
            repair_events_csv_path=repair_events_csv_for_report,
            rag_retrieval_csv_path=rag_retrieval_csv_for_report,
        )
        card_df = save_run_summary_card_data(
            report_df=report_df,
            agent_summary_csv_path=agent_summary_csv_for_report,
            out_csv_path=build_output_run_summary_card_csv_path(args, csv_path),
        )
        plot_run_summary_card(
            card_df=card_df,
            out_path=build_output_run_summary_card_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        plot_run_report_dashboard(
            report_df=report_df,
            out_path=build_output_run_report_dashboard_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[run-report] Skipping run report outputs due to error: {e}")

    if step_summary_csv_for_report:
        try:
            influence_df = save_influence_matrix(
                step_summary_csv_path=step_summary_csv_for_report,
                out_csv_path=build_output_influence_matrix_csv_path(args, csv_path),
            )
            plot_influence_matrix(
                influence_df=influence_df,
                out_path=build_output_influence_matrix_plot_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e:
            print(f"[influence-matrix] Skipping influence matrix due to error: {e}")
        try:
            theme_df = save_theme_effectiveness(
                step_summary_csv_path=step_summary_csv_for_report,
                out_csv_path=build_output_theme_effectiveness_csv_path(args, csv_path),
            )
            plot_theme_effectiveness(
                theme_df=theme_df,
                out_path=build_output_theme_effectiveness_plot_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e:
            print(f"[theme-effectiveness] Skipping theme outputs due to error: {e}")

    if step_summary_csv_for_report and rag_retrieval_csv_for_report:
        try:
            rag_df = save_rag_direction_breakdown(
                rag_retrieval_csv_path=rag_retrieval_csv_for_report,
                step_summary_csv_path=step_summary_csv_for_report,
                out_csv_path=build_output_rag_direction_breakdown_csv_path(args, csv_path),
            )
            plot_rag_direction_breakdown(
                rag_df=rag_df,
                out_path=build_output_rag_direction_breakdown_plot_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e:
            print(f"[rag-direction] Skipping RAG direction breakdown due to error: {e}")

    try:
        out_prefix_for_network_metrics = args.output_file if args.output_file else f"seed{args.seed}"
        network_metrics_df = save_network_assortativity_metrics(
            csv_path=csv_path,
            out_prefix=out_prefix_for_network_metrics,
            version_set=args.version_set,
            num_agents=args.num_agents,
            out_csv_path=build_output_network_assortativity_csv_path(args, csv_path),
        )
        plot_network_assortativity_metrics(
            metrics_df=network_metrics_df,
            out_path=build_output_network_assortativity_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[network-assortativity] Skipping network metrics due to error: {e}")

    try:
        agent_summary_csv_for_ba = build_input_agent_summary_csv_path(args)
        ba_assignment_df = save_ba_hub_assignment_metrics(
            agent_summary_csv_path=agent_summary_csv_for_ba,
            out_csv_path=build_output_ba_hub_assignment_csv_path(args, csv_path),
        )
        plot_ba_hub_assignment_metrics(
            metrics_df=ba_assignment_df,
            out_path=build_output_ba_hub_assignment_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[ba-hub-assignment] Skipping BA hub assignment metrics due to error: {e}")

    try:
        network_structure_csv_for_diag = _try_builder(build_input_network_structure_summary_csv_path, args)
        network_agent_csv_for_diag = _try_builder(build_input_network_agent_metrics_csv_path, args)
        interaction_influence_csv_for_diag = _try_builder(build_input_interaction_influence_events_csv_path, args)
        agent_influence_csv_for_diag = _try_builder(build_input_agent_influence_summary_csv_path, args)
        exposure_persuasion_csv_for_diag = _try_builder(build_input_exposure_persuasion_summary_csv_path, args)
        if not network_agent_csv_for_diag:
            raise FileNotFoundError("network_agent_metrics CSV not found")
        plot_network_influence_diagnostics(
            args=args,
            csv_path=csv_path,
            bdp_csv_path=out_bdp_csv,
            network_structure_csv_path=network_structure_csv_for_diag,
            network_agent_csv_path=network_agent_csv_for_diag,
            interaction_influence_csv_path=interaction_influence_csv_for_diag,
            agent_influence_csv_path=agent_influence_csv_for_diag,
            exposure_persuasion_csv_path=exposure_persuasion_csv_for_diag,
        )
    except Exception as e:
        print(f"[network-influence] Skipping qwen10 network/influence diagnostic plots due to error: {e}")

    try:
        out_prefix_for_network_detail = args.output_file if args.output_file else f"seed{args.seed}"
        plot_ego_network_belief_timelines(
            csv_path=csv_path,
            out_prefix=out_prefix_for_network_detail,
            version_set=args.version_set,
            num_agents=args.num_agents,
            out_path=build_output_ego_network_timeline_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        edge_diff_df = save_edge_belief_differential(
            csv_path=csv_path,
            out_prefix=out_prefix_for_network_detail,
            version_set=args.version_set,
            num_agents=args.num_agents,
            out_csv_path=build_output_edge_belief_diff_csv_path(args, csv_path),
            sample_every=10,
        )
        plot_edge_belief_differential(
            edge_df=edge_diff_df,
            out_path=build_output_edge_belief_diff_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        neighborhood_df = save_agent_neighborhood_alignment(
            csv_path=csv_path,
            out_prefix=out_prefix_for_network_detail,
            version_set=args.version_set,
            num_agents=args.num_agents,
            out_csv_path=build_output_agent_neighborhood_alignment_csv_path(args, csv_path),
        )
        plot_agent_neighborhood_alignment(
            alignment_df=neighborhood_df,
            out_path=build_output_agent_neighborhood_alignment_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[network-detail] Skipping ego/edge detail plots due to error: {e}")

    try:
        step_summary_csv = build_input_step_summary_csv_path(args)
        plot_step_movement_from_summary(
            step_summary_csv_path=step_summary_csv,
            out_path=build_output_step_movement_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        plot_cumulative_drift_from_summary(
            step_summary_csv_path=step_summary_csv,
            out_path=build_output_cumulative_drift_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        try:
            plot_source_provenance_from_summary(
                step_summary_csv_path=step_summary_csv,
                out_path=build_output_source_provenance_plot_path(args, csv_path),
                out_csv_path=build_output_source_provenance_csv_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e2:
            print(f"[source-provenance] Skipping provenance outputs due to error: {e2}")
        try:
            ecology_df = save_exposure_ecology(
                step_summary_csv_path=step_summary_csv,
                out_csv_path=build_output_exposure_ecology_csv_path(args, csv_path),
            )
            plot_exposure_ecology(
                ecology_df=ecology_df,
                out_path=build_output_exposure_ecology_plot_path(args, csv_path),
                fig_type=args.figure_file_type,
            )
        except Exception as e2:
            print(f"[exposure-ecology] Skipping exposure ecology outputs due to error: {e2}")
    except Exception as e:
        print(f"[step-summary] Skipping step-summary plots due to error: {e}")

    try:
        step_summary_csv = build_input_step_summary_csv_path(args)
        agent_summary_csv = build_input_agent_summary_csv_path(args)
        plot_trajectory_by_profile(
            csv_path=csv_path,
            agent_summary_csv_path=agent_summary_csv,
            out_path=build_output_profile_trajectory_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
            num_agents=args.num_agents,
        )
        plot_final_distribution_by_profile(
            agent_summary_csv_path=agent_summary_csv,
            out_path=build_output_final_distribution_by_profile_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        plot_movement_by_profile(
            step_summary_csv_path=step_summary_csv,
            out_path=build_output_movement_by_profile_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        plot_agent_small_multiples(
            csv_path=csv_path,
            agent_summary_csv_path=agent_summary_csv,
            out_path=build_output_agent_small_multiples_path(args, csv_path),
            fig_type=args.figure_file_type,
            num_agents=args.num_agents,
        )
        plot_final_belief_strip_by_trait(
            agent_summary_csv_path=agent_summary_csv,
            out_path=build_output_final_belief_strip_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        persona_delta_df = save_persona_trait_delta(
            agent_summary_csv_path=agent_summary_csv,
            out_csv_path=build_output_persona_trait_delta_csv_path(args, csv_path),
        )
        plot_persona_trait_delta(
            persona_df=persona_delta_df,
            out_path=build_output_persona_trait_delta_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[persona-plots] Skipping persona-aware plots due to error: {e}")

    try:
        run_metrics_csv = build_input_run_metrics_csv_path(args)
        cue_df = save_step2_cue_metrics_from_run_metrics(
            run_metrics_csv_path=run_metrics_csv,
            out_csv_path=build_output_cue_metrics_csv_path(args, csv_path),
        )
        plot_step2_cue_concentration(
            cue_df=cue_df,
            out_path=build_output_cue_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[run-metrics] Skipping cue concentration outputs due to error: {e}")

    try:
        repair_events_csv = build_input_repair_events_csv_path(args)
        repair_tag_df = save_repair_tag_counts(
            repair_events_csv_path=repair_events_csv,
            out_csv_path=build_output_repair_tags_csv_path(args, csv_path),
        )
        plot_repair_events_over_time(
            repair_events_csv_path=repair_events_csv,
            out_path=build_output_repair_timeseries_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
        plot_repair_tag_counts(
            repair_tag_df=repair_tag_df,
            out_path=build_output_repair_tags_plot_path(args, csv_path),
            fig_type=args.figure_file_type,
        )
    except Exception as e:
        print(f"[repair-events] Skipping repair plots due to error: {e}")

    # Distribution evolution GIF
    df = pd.read_csv(csv_path)
    all_cols = list(df.columns)
    agent_cols = all_cols[-args.num_agents:]

    frames_dir = build_output_distribution_frames_dir(args, csv_path)
    frame_paths = make_distribution_frames(
        df=df,
        agent_cols=agent_cols,
        frames_dir=frames_dir,
        step50_out_path=build_output_step50_distribution_path(args, csv_path),
        step_interval=5,  # change to 1 if you want every step
    )

    gif_path = build_output_distribution_gif_path(args, csv_path)
    make_distribution_gif(frame_paths, gif_path, fps=2)

    # Network frames + GIF (saved in the SAME plot folder)
    plot_dir = _build_plot_subdir(csv_path, args)
    out_prefix = args.output_file if args.output_file else f"seed{args.seed}"
    generate_network_frames_and_gif(
        csv_path=csv_path,
        plot_dir=plot_dir,
        out_prefix=out_prefix,
        version_set=args.version_set,
        num_agents=args.num_agents,
        frame_duration=0.5,
        num_fades=2,
    )
    generate_network_3d_html(
        csv_path=csv_path,
        plot_dir=plot_dir,
        out_prefix=out_prefix,
        version_set=args.version_set,
        num_agents=args.num_agents,
        step_stride=1,     # set to 5 if you want it lighter
        layout_seed=42,
    )

    
        # -----------------------------
    # Markov transition matrix from interactions CSV (one matrix per run)
    # -----------------------------
    try:
        interactions_csv = build_input_interactions_csv_path(args)
        print(f"[markov] Using interactions CSV: {interactions_csv}")
        counts_df, probs_df = compute_markov_transition_matrix_from_interactions(interactions_csv)

        out_counts = build_output_markov_counts_path(args, csv_path)
        out_probs = build_output_markov_probs_path(args, csv_path)

        out_probs_plot = build_output_markov_plot_path(args, csv_path)
        out_counts_plot = build_output_markov_counts_plot_path(args, csv_path)

        counts_df.to_csv(out_counts)
        probs_df.to_csv(out_probs)

        plot_markov_matrix(probs_df, out_probs_plot, args.figure_file_type)
        plot_markov_counts_matrix(counts_df, out_counts_plot, args.figure_file_type)

        print(f"Saved Markov counts CSV  to: {out_counts}")
        print(f"Saved Markov probs  CSV  to: {out_probs}")
        print(f"Saved Markov probs  plot to: {out_probs_plot}")
        print(f"Saved Markov counts plot to: {out_counts_plot}")

        _markov_df = pd.read_csv(interactions_csv)
        step_col = None
        for cand in ["Time Step", "time_step", "TimeStep"]:
            if cand in _markov_df.columns:
                step_col = cand
                break
        if step_col is None:
            raise ValueError(f"Could not find step column in interactions CSV: {interactions_csv}")
        max_step = int(_markov_df[step_col].max())
        for label, lo, hi in markov_windows_from_max_step(max_step):
            counts_w, probs_w = compute_markov_transition_matrix_from_interactions(interactions_csv, step_range=(lo, hi))
            out_counts_w = build_output_markov_window_counts_path(args, csv_path, label)
            out_probs_w = build_output_markov_window_probs_path(args, csv_path, label)
            out_probs_plot_w = build_output_markov_window_plot_path(args, csv_path, label)
            out_counts_plot_w = build_output_markov_window_counts_plot_path(args, csv_path, label)
            counts_w.to_csv(out_counts_w)
            probs_w.to_csv(out_probs_w)
            title_suffix = f" ({label}: steps {lo}-{hi})"
            plot_markov_matrix(probs_w, out_probs_plot_w, args.figure_file_type, title="Empirical Markov Transition Matrix" + title_suffix)
            plot_markov_counts_matrix(counts_w, out_counts_plot_w, args.figure_file_type, title="Empirical Markov Transition Counts" + title_suffix)
            print(f"Saved Markov window '{label}' counts CSV  to: {out_counts_w}")
            print(f"Saved Markov window '{label}' probs  CSV  to: {out_probs_w}")
            print(f"Saved Markov window '{label}' probs  plot to: {out_probs_plot_w}")
            print(f"Saved Markov window '{label}' counts plot to: {out_counts_plot_w}")

    except Exception as e:
        print(f"[markov] Skipping Markov matrix due to error: {e}")

if __name__ == "__main__":
    main()
