"""Network construction and partner-selection utilities for opinion-dynamics runs.

The simulator imports this module to build Watts-Strogatz, Erdos-Renyi, and Barabasi-Albert neighbor dictionaries, apply optional structural homophily, and choose interaction partners using opinion/persona similarity scores. Functions return plain ``dict[int, set[int]]`` structures so the runtime can update edges without depending on NetworkX objects.
"""

import random
from typing import Dict, Set, Sequence, Mapping, Union, Any, Callable
import networkx as nx

OpinionContainer = Union[Sequence[int], Mapping[int, int]]


def _safe_int(x: Any, default: int | None = None) -> int | None:
    """Convert a value to int while preserving a caller-provided default on invalid input."""
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _homophily_order_indices(
    num_agents: int,
    seed: int | None,
    opinions: OpinionContainer | None,
    attributes: Mapping[int, Mapping[str, Any]] | None,
) -> list[int]:
    """
    Return a permutation of agent indices that clusters agents by *initial opinion only*.

    Earlier versions used persona / demographic attributes as extra structural-homophily
    dimensions. For the current experimental design, starting-network homophily should be
    interpretable as opinion homophily only. The attributes argument is intentionally ignored
    here; it is kept only for API compatibility with the existing builder calls.

    Ties within the same opinion are shuffled deterministically by seed, then preserved by
    Python's stable sort.
    """
    idxs = list(range(num_agents))
    rng = random.Random(seed)
    rng.shuffle(idxs)  # deterministic tie-breaking within each opinion bin

    def get_op(i: int) -> int:
        """Return the integer initial opinion for an agent, using zero when it is unavailable."""
        if opinions is None:
            return 0
        try:
            return int(opinions[i])  # type: ignore[index]
        except Exception:
            try:
                return int(opinions.get(i, 0))  # type: ignore[union-attr]
            except Exception:
                return 0

    idxs.sort(key=lambda i: get_op(i))
    return idxs




def build_small_world(
    num_agents: int,
    k: int,
    p_rewire: float,
    seed: int | None = None,
    *,
    homophily: bool = False,
    opinions: OpinionContainer | None = None,
    attributes: Mapping[int, Mapping[str, Any]] | None = None,
) -> Dict[int, Set[int]]:
    """
    Build a Watts–Strogatz small-world network using networkx.

    Returns:
        neighbors: dict mapping node i -> set of neighbors (undirected).
    """
    if k % 2 != 0:
        raise ValueError("k must be even (k/2 neighbors on each side).")

    # Optionally make the initial ring lattice opinion-homophilic by ordering
    # same/similar initial opinions adjacently, then applying the standard WS rewiring.
    if homophily and (opinions is not None or attributes is not None):
        order = _homophily_order_indices(
            num_agents=num_agents,
            seed=seed,
            opinions=opinions,
            attributes=attributes,
        )
        G_pos = nx.watts_strogatz_graph(
            n=num_agents,
            k=k,
            p=p_rewire,
            seed=seed,
        )
        # Relabel ring positions -> agent indices (permutation)
        mapping = {pos: order[pos] for pos in range(num_agents)}
        G = nx.relabel_nodes(G_pos, mapping, copy=True)
    else:
        G = nx.watts_strogatz_graph(
            n=num_agents,
            k=k,
            p=p_rewire,
            seed=seed,
        )

    neighbors: Dict[int, Set[int]] = {
        int(node): set(int(nbr) for nbr in G.neighbors(node))
        for node in G.nodes()
    }
    return neighbors


def build_erdos_renyi(
    num_agents: int,
    p_edge: float,
    seed: int | None = None,
    *,
    homophily: bool = False,
    opinions: OpinionContainer | None = None,
    attributes: Mapping[int, Mapping[str, Any]] | None = None,
) -> Dict[int, Set[int]]:
    """Build an Erdős–Rényi G(n, p) network."""
    if not (0.0 <= p_edge <= 1.0):
        raise ValueError("p_edge must be between 0 and 1.")

    G_base = nx.erdos_renyi_graph(n=num_agents, p=p_edge, seed=seed)

    if homophily and (opinions is not None or attributes is not None):
        order = _homophily_order_indices(
            num_agents=num_agents,
            seed=seed,
            opinions=opinions,
            attributes=attributes,
        )
        mapping = {pos: order[pos] for pos in range(num_agents)}
        G = nx.relabel_nodes(G_base, mapping, copy=True)
    else:
        G = G_base

    neighbors: Dict[int, Set[int]] = {
        int(node): set(int(nbr) for nbr in G.neighbors(node))
        for node in G.nodes()
    }
    return neighbors


def _ba_hub_order_indices(
    num_agents: int,
    seed: int | None,
    opinions: OpinionContainer | None,
    attributes: Mapping[int, Mapping[str, Any]] | None,
    *,
    hub_strategy: str = "default",
    hub_agent_ids: Sequence[int] | None = None,
    homophily: bool = False,
) -> list[int]:
    """Return agent-index order used to relabel BA positions.

    In a BA graph, low-numbered generated positions are older and usually become
    the highest-degree hubs. By relabeling those early positions, we can choose
    which agents/types are most likely to become the popular hubs while keeping
    the actual preferential-attachment mechanism intact.
    """
    strategy = str(hub_strategy or "default").strip().lower()
    valid_strategies = {
        "default", "networkx_default", "none",
        "random", "positive", "negative", "extreme", "neutral",
        "opposite_majority", "minority_side", "anti_majority", "counter_majority",
        "custom",
    }
    if strategy not in valid_strategies:
        strategy = "default"

    all_ids = list(range(num_agents))
    rng = random.Random(seed)

    def get_op(i: int) -> int:
        """Return the integer initial opinion for an agent, using zero when it is unavailable."""
        if opinions is None:
            return 0
        try:
            return int(opinions[i])  # type: ignore[index]
        except Exception:
            try:
                return int(opinions.get(i, 0))  # type: ignore[union-attr]
            except Exception:
                return 0

    def stable_shuffled_ids() -> list[int]:
        """Return a seed-stable shuffled copy of all local agent ids for tie-breaking."""
        ids = list(all_ids)
        rng.shuffle(ids)
        return ids

    if strategy in {"opposite_majority", "minority_side", "anti_majority", "counter_majority"}:
        # Give the structural hub advantage to the side that is initially
        # outnumbered. This is useful for counter-driving runs where one side
        # would otherwise dominate simply because it has more speakers.
        pos_count = sum(1 for i in all_ids if get_op(i) > 0)
        neg_count = sum(1 for i in all_ids if get_op(i) < 0)
        if pos_count > neg_count:
            strategy = "negative"
        elif neg_count > pos_count:
            strategy = "positive"
        else:
            strategy = "neutral"

    # Current behavior: no relabeling unless structural homophily is requested.
    if strategy in {"default", "networkx_default", "none"} and not homophily:
        return all_ids

    if strategy == "random":
        order = stable_shuffled_ids()
    elif strategy == "positive":
        order = stable_shuffled_ids()
        order.sort(key=lambda i: (-get_op(i), i))
    elif strategy == "negative":
        order = stable_shuffled_ids()
        order.sort(key=lambda i: (get_op(i), i))
    elif strategy == "extreme":
        order = stable_shuffled_ids()
        order.sort(key=lambda i: (-abs(get_op(i)), -get_op(i), i))
    elif strategy == "neutral":
        order = stable_shuffled_ids()
        order.sort(key=lambda i: (abs(get_op(i)), i))
    elif strategy == "custom":
        seen: set[int] = set()
        priority: list[int] = []
        for raw in hub_agent_ids or []:
            j = _safe_int(raw, None)
            if j is None or not (0 <= j < num_agents) or j in seen:
                continue
            seen.add(j)
            priority.append(j)
        remainder = [i for i in stable_shuffled_ids() if i not in seen]
        if homophily and (opinions is not None or attributes is not None):
            remainder.sort(key=lambda i: get_op(i))
        order = priority + remainder
    elif homophily and (opinions is not None or attributes is not None):
        # Keep previous homophilic BA relabeling behavior.
        order = _homophily_order_indices(num_agents, seed, opinions, attributes)
    else:
        order = all_ids

    # If a non-custom strategy is combined with homophily, preserve the chosen hub
    # priority first, then lightly cluster equal-priority tails by initial opinion.
    if strategy not in {"default", "networkx_default", "none", "custom"} and homophily and opinions is not None:
        def priority_group(i: int):
            """Group agents by the selected BA hub-priority rule before final ordering."""
            op = get_op(i)
            if strategy == "positive":
                return -op
            if strategy == "negative":
                return op
            if strategy == "extreme":
                return -abs(op)
            if strategy == "neutral":
                return abs(op)
            return 0
        order.sort(key=lambda i: (priority_group(i), get_op(i), i))

    # Ensure it is a complete permutation even if bad input slipped through.
    seen = set()
    cleaned = []
    for i in order:
        if 0 <= int(i) < num_agents and int(i) not in seen:
            seen.add(int(i))
            cleaned.append(int(i))
    cleaned.extend(i for i in all_ids if i not in seen)
    return cleaned[:num_agents]




def _normalize_ba_hub_assignment_mode(mode: Any = "early_position") -> str:
    """Normalize BA hub-assignment semantics.

    early_position: selected agents are mapped onto early BA positions. This is
        the original behavior: an attachment advantage, not a hard guarantee.
    actual_hubs: selected agents are mapped onto the realized highest-degree
        BA positions after the graph is generated.
    early_and_actual: start from early_position, then swap labels so selected
        agents also occupy the realized top-degree positions.
    """
    s = str(mode or "early_position").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "early_position",
        "current": "early_position",
        "early": "early_position",
        "early_positions": "early_position",
        "early_position_priority": "early_position",
        "probabilistic": "early_position",
        "odds": "early_position",
        "actual": "actual_hubs",
        "actual_hub": "actual_hubs",
        "actual_hubs": "actual_hubs",
        "top_hub": "actual_hubs",
        "top_hubs": "actual_hubs",
        "highest_degree": "actual_hubs",
        "realized_hubs": "actual_hubs",
        "both": "early_and_actual",
        "early_actual": "early_and_actual",
        "early_and_actual": "early_and_actual",
        "early_plus_actual": "early_and_actual",
    }
    return aliases.get(s, "early_position")


def _ba_degree_sorted_positions(G: nx.Graph, num_agents: int) -> list[int]:
    """Generated BA positions sorted by realized degree, highest first."""
    return sorted(range(int(num_agents)), key=lambda i: (-int(G.degree[i]), int(i)))


def _ba_selected_priority_indices(
    num_agents: int,
    seed: int | None,
    opinions: OpinionContainer | None,
    attributes: Mapping[int, Mapping[str, Any]] | None,
    *,
    hub_strategy: str = "default",
    hub_agent_ids: Sequence[int] | None = None,
    homophily: bool = False,
) -> list[int]:
    """Return only the agents targeted by the hub strategy, in priority order."""
    strategy = str(hub_strategy or "default").strip().lower()
    if strategy == "minority":
        strategy = "opposite_majority"
    if strategy in {"minority_side", "anti_majority", "counter_majority"}:
        strategy = "opposite_majority"

    all_ids = list(range(int(num_agents)))
    rng = random.Random(seed)

    def get_op(i: int) -> int:
        """Return the integer initial opinion for an agent, using zero when it is unavailable."""
        if opinions is None:
            return 0
        try:
            return int(opinions[i])  # type: ignore[index]
        except Exception:
            try:
                return int(opinions.get(i, 0))  # type: ignore[union-attr]
            except Exception:
                return 0

    def stable_shuffled_ids() -> list[int]:
        """Return a seed-stable shuffled copy of all local agent ids for tie-breaking."""
        ids = list(all_ids)
        rng.shuffle(ids)
        return ids

    if strategy == "custom":
        seen: set[int] = set()
        selected: list[int] = []
        for raw in hub_agent_ids or []:
            j = _safe_int(raw, None)
            if j is None or not (0 <= j < int(num_agents)) or j in seen:
                continue
            seen.add(j)
            selected.append(j)
        return selected

    if strategy == "random":
        return stable_shuffled_ids()

    if strategy == "opposite_majority":
        pos = [i for i in all_ids if get_op(i) > 0]
        neg = [i for i in all_ids if get_op(i) < 0]
        if len(pos) > len(neg):
            strategy = "negative"
        elif len(neg) > len(pos):
            strategy = "positive"
        else:
            strategy = "neutral"

    ids = stable_shuffled_ids()
    if strategy == "positive":
        selected = [i for i in ids if get_op(i) > 0]
        selected.sort(key=lambda i: (-get_op(i), i))
    elif strategy == "negative":
        selected = [i for i in ids if get_op(i) < 0]
        selected.sort(key=lambda i: (get_op(i), i))
    elif strategy == "neutral":
        selected = [i for i in ids if get_op(i) == 0]
        selected.sort(key=lambda i: (abs(get_op(i)), i))
    elif strategy == "extreme":
        selected = [i for i in ids if abs(get_op(i)) == 2]
        selected.sort(key=lambda i: (-abs(get_op(i)), -get_op(i), i))
    else:
        selected = []

    if homophily and opinions is not None and selected:
        selected.sort(key=lambda i: get_op(i))
    return selected


def _complete_position_agent_mapping(position_to_agent: Mapping[int, int], num_agents: int) -> dict[int, int]:
    """Return a complete one-to-one mapping from generated positions to agent ids."""
    n = int(num_agents)
    mapping: dict[int, int] = {}
    used_agents: set[int] = set()
    for raw_pos, raw_agent in (position_to_agent or {}).items():
        pos = _safe_int(raw_pos, None)
        agent = _safe_int(raw_agent, None)
        if pos is None or agent is None:
            continue
        if not (0 <= pos < n and 0 <= agent < n):
            continue
        if agent in used_agents:
            continue
        mapping[pos] = agent
        used_agents.add(agent)
    remaining_agents = [i for i in range(n) if i not in used_agents]
    for pos in range(n):
        if pos not in mapping:
            mapping[pos] = remaining_agents.pop(0)
    return mapping


def _ba_actual_hub_mapping(G_base: nx.Graph, agent_order: Sequence[int], num_agents: int) -> dict[int, int]:
    """Map realized highest-degree BA positions to the front of agent_order."""
    n = int(num_agents)
    cleaned: list[int] = []
    seen: set[int] = set()
    for raw in list(agent_order or []):
        i = _safe_int(raw, None)
        if i is not None and 0 <= i < n and i not in seen:
            seen.add(i)
            cleaned.append(i)
    cleaned.extend(i for i in range(n) if i not in seen)
    degree_positions = _ba_degree_sorted_positions(G_base, n)
    return _complete_position_agent_mapping({pos: cleaned[rank] for rank, pos in enumerate(degree_positions)}, n)


def _ba_early_and_actual_mapping(
    G_base: nx.Graph,
    early_order: Sequence[int],
    selected_agents: Sequence[int],
    num_agents: int,
) -> dict[int, int]:
    """Start with early-position mapping, then force selected agents onto actual top hubs."""
    n = int(num_agents)
    early = list(early_order or [])
    seen: set[int] = set()
    cleaned: list[int] = []
    for raw in early:
        i = _safe_int(raw, None)
        if i is not None and 0 <= i < n and i not in seen:
            seen.add(i)
            cleaned.append(i)
    cleaned.extend(i for i in range(n) if i not in seen)

    mapping = _complete_position_agent_mapping({pos: cleaned[pos] for pos in range(n)}, n)
    selected_order = []
    selected_seen: set[int] = set()
    selected_set = {int(i) for i in selected_agents if _safe_int(i, None) is not None and 0 <= int(i) < n}
    for agent in cleaned:
        if agent in selected_set and agent not in selected_seen:
            selected_seen.add(agent)
            selected_order.append(agent)

    if not selected_order:
        return mapping

    degree_positions = _ba_degree_sorted_positions(G_base, n)
    top_positions = degree_positions[:len(selected_order)]

    for pos, desired_agent in zip(top_positions, selected_order):
        current_agent = mapping.get(pos)
        if current_agent == desired_agent:
            continue
        current_pos_of_desired = None
        for p, a in mapping.items():
            if a == desired_agent:
                current_pos_of_desired = p
                break
        if current_pos_of_desired is None:
            continue
        mapping[current_pos_of_desired] = current_agent
        mapping[pos] = desired_agent
    return _complete_position_agent_mapping(mapping, n)


def build_barabasi_albert(
    num_agents: int,
    m_attach: int,
    seed: int | None = None,
    *,
    homophily: bool = False,
    opinions: OpinionContainer | None = None,
    attributes: Mapping[int, Mapping[str, Any]] | None = None,
    hub_strategy: str = "default",
    hub_agent_ids: Sequence[int] | None = None,
    hub_assignment_mode: str = "early_position",
) -> Dict[int, Set[int]]:
    """Build a Barabási–Albert preferential-attachment network.

    `hub_strategy` controls which agents are targeted for hub advantage.

    `hub_assignment_mode` controls what "hub advantage" means:
    - early_position: selected agents occupy early BA positions. This is the
      original probabilistic advantage; selected agents are more likely, but not
      guaranteed, to become the highest-degree hubs.
    - actual_hubs: the BA graph is generated first, then selected agents are
      assigned to the realized highest-degree positions.
    - early_and_actual: selected agents receive the early-position advantage and
      are also swapped onto the realized highest-degree positions.
    """
    if m_attach <= 0 or m_attach >= num_agents:
        raise ValueError("m_attach must be > 0 and < num_agents.")

    n = int(num_agents)
    G_base = nx.barabasi_albert_graph(n=n, m=m_attach, seed=seed)
    assignment_mode = _normalize_ba_hub_assignment_mode(hub_assignment_mode)
    strategy = str(hub_strategy or "default").strip().lower()

    order = _ba_hub_order_indices(
        num_agents=n,
        seed=seed,
        opinions=opinions,
        attributes=attributes,
        hub_strategy=strategy,
        hub_agent_ids=hub_agent_ids,
        homophily=homophily,
    )
    selected_agents = _ba_selected_priority_indices(
        num_agents=n,
        seed=seed,
        opinions=opinions,
        attributes=attributes,
        hub_strategy=strategy,
        hub_agent_ids=hub_agent_ids,
        homophily=homophily,
    )

    # No target group means assignment mode should not invent a manipulation.
    if not selected_agents and strategy in {"default", "networkx_default", "none"}:
        assignment_mode = "early_position"

    if assignment_mode == "actual_hubs" and selected_agents:
        mapping = _ba_actual_hub_mapping(G_base, order, n)
    elif assignment_mode == "early_and_actual" and selected_agents:
        mapping = _ba_early_and_actual_mapping(G_base, order, selected_agents, n)
    else:
        mapping = _complete_position_agent_mapping({pos: order[pos] for pos in range(n)}, n)

    # Only relabel when the mapping differs from identity. This preserves exact
    # old behavior for default non-homophilic BA runs.
    if any(int(mapping.get(pos, pos)) != int(pos) for pos in range(n)):
        G = nx.relabel_nodes(G_base, mapping, copy=True)
    else:
        G = G_base

    neighbors: Dict[int, Set[int]] = {
        int(node): set(int(nbr) for nbr in G.neighbors(node))
        for node in G.nodes()
    }
    return neighbors

# -----------------------------
# SCORING-BASED HOMOPHILY
# -----------------------------

# Opinion distance -> points: the opinion component of compute_pair_score.
OPINION_POINTS = {
    0: 100,  # same opinion
    1: 60,
    2: 25,
    3: 10,
    4: 5,    # opposite extremes
}

# Ordinal encodings for the persona attributes used by compute_pair_score.
POLITICAL_MAP = {
    "far_left": -3,
    "left": -2,
    "center_left": -1,
    "center": 0,
    "center_right": 1,
    "right": 2,
    "far_right": 3,
}

EDUCATION_ORDER = [
    "high_school",
    "some_college",
    "bachelor",
    "master",
    "phd",
]
EDU_INDEX = {name: idx for idx, name in enumerate(EDUCATION_ORDER)}


def _age_group(age: Any) -> int | None:
    """Map numeric age to coarse age group index."""
    try:
        a = int(age)
    except (TypeError, ValueError):
        return None

    if a < 18:
        return 0
    elif a <= 25:
        return 1
    elif a <= 35:
        return 2
    elif a <= 50:
        return 3
    else:
        return 4


def _political_score(pol: Any) -> int:
    """Map political leaning value to ordinal -3..3."""
    if pol is None:
        return 0
    s = str(pol).strip().lower()
    return POLITICAL_MAP.get(s, 0)


def _education_index(edu: Any) -> int | None:
    """Map education labels to an ordered numeric level for pair-score comparison."""
    if edu is None:
        return None
    s = str(edu).strip().lower()
    return EDU_INDEX.get(s)


def compute_pair_score(
    i: int,
    j: int,
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
) -> float:
    """
    Compute total homophily score S_total(i, j) based on:
    - opinion
    - political leaning
    - education
    - age
    - background/ethnicity
    - gender
    - occupation
    """
    # ---- Opinion score (dominant factor)
    op_i = opinions[i]
    op_j = opinions[j]

    if op_i is None or op_j is None:
        return 0.0

    try:
        d_op = abs(float(op_i) - float(op_j))
    except (TypeError, ValueError):
        d_op = 4.0

    d_op_int = int(min(max(d_op, 0), 4))  # clamp 0..4
    s_opinion = OPINION_POINTS.get(d_op_int, 0)  # 0–100

    # Pull attribute dicts (may be missing keys; handle gently)
    attr_i = attributes.get(i, {})
    attr_j = attributes.get(j, {})

    # ---- Political leaning (0–40) ----
    pol_i = attr_i.get("pol_score", None)
    pol_j = attr_j.get("pol_score", None)

    if pol_i is None or pol_j is None:
        # fallback to string field if numeric is not present
        pol_i = _political_score(attr_i.get("political_leaning"))
        pol_j = _political_score(attr_j.get("political_leaning"))
    else:
        pol_i = int(pol_i)
        pol_j = int(pol_j)

    d_pol = abs(pol_i - pol_j)  # 0..6

    if d_pol == 0:
        s_pol = 40.0
    elif d_pol == 1:
        s_pol = 30.0
    elif d_pol == 2:
        s_pol = 15.0
    elif d_pol == 3:
        s_pol = 8.0
    else:
        s_pol = 3.0

    # ---- Education (0–20) ----
    edu_i = attr_i.get("edu_level", None)
    edu_j = attr_j.get("edu_level", None)

    if edu_i is None or edu_j is None:
        edu_i = _education_index(attr_i.get("education"))
        edu_j = _education_index(attr_j.get("education"))

    if edu_i is not None and edu_j is not None:
        edu_i = int(edu_i)
        edu_j = int(edu_j)
        d_edu = abs(edu_i - edu_j)  # 0..4

        if d_edu == 0:
            s_edu = 20.0
        elif d_edu == 1:
            s_edu = 15.0
        elif d_edu == 2:
            s_edu = 7.0
        else:
            s_edu = 3.0
    else:
        s_edu = 0.0


    # ---- Age group (0–25) ----
    age_i = attr_i.get("age_group", None)
    age_j = attr_j.get("age_group", None)

    if age_i is None or age_j is None:
        age_i = _age_group(attr_i.get("age"))
        age_j = _age_group(attr_j.get("age"))

    if age_i is not None and age_j is not None:
        age_i = int(age_i)
        age_j = int(age_j)
        d_age = abs(age_i - age_j)  # 0..4

        if d_age == 0:
            s_age = 25.0
        elif d_age == 1:
            s_age = 15.0
        elif d_age == 2:
            s_age = 6.0
        else:
            s_age = 2.0
    else:
        s_age = 0.0


    # ---- Background / ethnicity (0–10)
    bg_i = attr_i.get("ethnicity")
    bg_j = attr_j.get("ethnicity")
    if bg_i is not None and bg_j is not None:
        s_bg = 10.0 if str(bg_i).strip().lower() == str(bg_j).strip().lower() else 0.0
    else:
        s_bg = 0.0

    # ---- Gender (0–5)
    g_i = attr_i.get("gender")
    g_j = attr_j.get("gender")
    if g_i is not None and g_j is not None:
        s_gender = 5.0 if str(g_i).strip().lower() == str(g_j).strip().lower() else 0.0
    else:
        s_gender = 0.0

    # ---- Occupation (0–20)
    occ_i = attr_i.get("occupation")
    occ_j = attr_j.get("occupation")
    if occ_i is not None and occ_j is not None:
        s_occ = 20.0 if str(occ_i).strip().lower() == str(occ_j).strip().lower() else 0.0
    else:
        s_occ = 0.0

    # Total score: sum of all dimensions
    s_total = (
        s_opinion
        + s_pol
        + s_edu
        + s_age
        + s_bg
        + s_gender
        + s_occ
    )

    return s_total


def compute_opinion_only_score(
    i: int,
    j: int,
    opinions: OpinionContainer,
) -> float:
    """Opinion-only homophily score (0..100), using the SAME mapping as the opinion component in `compute_pair_score`."""
    op_i = opinions[i]
    op_j = opinions[j]
    if op_i is None or op_j is None:
        return 0.0
    try:
        d_op = abs(float(op_i) - float(op_j))
    except (TypeError, ValueError):
        return 0.0

    d_op_int = int(min(max(d_op, 0), 4))  # clamp 0..4
    return float(OPINION_POINTS.get(d_op_int, 0))


def opinion_only_pair_score(i, j, opinions, attributes=None) -> float:
    """4-arg adapter so an opinion-only score is a drop-in for any pair scorer.

    This is the DEFAULT tie-formation scorer for evolving networks. It ignores
    `attributes` deliberately: `compute_pair_score` reads a legacy attribute
    scheme (pol_score, edu_level, age, ethnicity, gender, occupation) that the
    current persona-profile system does not populate and that no actual run has
    used, so scoring on it mixes in constant noise. A profile-aware scorer is a
    planned makeover; when it exists, pass it to evolve_once as `score_fn` -
    the evolution logic does not change.
    """
    return compute_opinion_only_score(i, j, opinions)


def choose_partner_scoring(
    agent_idx: int,
    candidates: Sequence[int],
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
    epsilon_uniform: float = 0.0,
    rng: random.Random | None = None,
    score_mode: str = "full",
) -> int:
    """
    Choose ONE interaction partner from an arbitrary candidate list using scoring-based homophily.

    - If score_mode == "full": use `compute_pair_score` (opinion + persona attributes).
    - If score_mode == "opinion_only": use ONLY the opinion similarity component.

    Probabilities are proportional to the chosen score (same conversion as `choose_neighbor_scoring`).
    With probability `epsilon_uniform`, ignore scores and pick uniformly at random (exploration).
    """
    if rng is None:
        rng = random

    cand = [c for c in candidates if c != agent_idx]
    if not cand:
        raise ValueError("No valid candidates (empty list or only self).")

    # With small probability, ignore scores (exploration)
    if epsilon_uniform > 0.0 and rng.random() < epsilon_uniform:
        return rng.choice(cand)

    mode = (score_mode or "full").strip().lower()

    scores: list[float] = []
    valid: list[int] = []

    for j in cand:
        if mode == "opinion_only":
            s = compute_opinion_only_score(agent_idx, j, opinions)
        else:
            s = compute_pair_score(agent_idx, j, opinions, attributes)

        if s > 0:
            scores.append(s)
            valid.append(j)

    # If all candidates had zero/invalid scores, fall back to uniform
    if not valid:
        return rng.choice(cand)

    total = sum(scores)
    if total <= 0:
        return rng.choice(valid)

    # Weighted random choice using cumulative sum
    r = rng.random() * total
    cumsum = 0.0
    for j, s in zip(valid, scores):
        cumsum += s
        if r <= cumsum:
            return j

    # Numerical safety fallback
    return valid[-1]



def choose_neighbor_scoring(
    agent_idx: int,
    neighbors: Dict[int, Set[int]],
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
    epsilon_uniform: float = 0.0,
    rng: random.Random | None = None,
    score_mode: str = "full",
) -> int:
    """
    Choose ONE neighbor for `agent_idx` using scoring-based homophily.
    Convenience wrapper over `choose_partner_scoring` for neighbor dicts; the
    current simulator builds candidate lists and calls that function directly.

    - For each neighbor j, compute a score.
      * score_mode="full" -> `compute_pair_score(i,j)` (opinion + persona attributes)
      * score_mode="opinion_only" -> opinion-only similarity
    - Sample j with probability proportional to the score.
    - With probability `epsilon_uniform`, ignore scores and pick a neighbor uniformly at random.

    Returns:
        neighbor_id: chosen neighbor j.
    """
    neigh_set = neighbors.get(agent_idx, set())
    if not neigh_set:
        raise ValueError(f"Agent {agent_idx} has no neighbors.")

    return choose_partner_scoring(
        agent_idx=agent_idx,
        candidates=list(neigh_set),
        opinions=opinions,
        attributes=attributes,
        epsilon_uniform=epsilon_uniform,
        rng=rng,
        score_mode=score_mode,
    )


# ==================================================================== directed
# Directed adjacency: follows are stored as two maps so in-degree and
#
# One edge is TWO dict entries, not two edges: following[a] gets b and
# followers[b] gets a. With reciprocal=True every write is mirrored, so
# following == followers identically and the structure is equivalent to the
# undirected Dict[int, Set[int]] the rest of the code has always used.
#
# Why direction is needed at all: in an undirected graph in-degree and
# out-degree are the same number, so "who has an audience" cannot be expressed.
# out-degree = whom I listen to (information diet).
# in-degree  = who listens to me (influence).


class DirectedNetwork:
    """Adjacency with orientation, and reciprocity as a policy rather than a fork.

    The two maps are always maintained together - callers cannot update one and
    forget the other, which is the failure mode this class exists to prevent.
    """

    __slots__ = ("following", "followers", "reciprocal", "n", "changes")

    def __init__(self, num_agents: int, reciprocal: bool = True):
        self.n = int(num_agents)
        self.reciprocal = bool(reciprocal)
        self.following = {i: set() for i in range(self.n)}
        self.followers = {i: set() for i in range(self.n)}
        self.changes = []          # (step, src, dst, "add"|"cut", reason)

    # ---------------------------------------------------------------- building

    @classmethod
    def from_undirected(cls, neighbors, num_agents: int, reciprocal: bool = True):
        """Build from the legacy Dict[int, Set[int]] the constructors return."""
        net = cls(num_agents, reciprocal=reciprocal)
        for src, nbrs in (neighbors or {}).items():
            for dst in nbrs:
                net.add_edge(int(src), int(dst))
        return net

    def to_undirected_map(self):
        """Legacy view: {i: set of everyone i is tied to, either direction}.

        Only equal to `following` when reciprocal. Provided so consumers that
        have not been audited for orientation keep working, never as the shape
        new code should read.
        """
        return {i: set(self.following[i]) | set(self.followers[i]) for i in range(self.n)}

    # ----------------------------------------------------------------- editing

    def has_edge(self, src: int, dst: int) -> bool:
        return int(dst) in self.following.get(int(src), ())

    def add_edge(self, src: int, dst: int, step: int = -1, reason: str = "") -> bool:
        """Add src -> dst (and the mirror when reciprocal). False if it existed."""
        src, dst = int(src), int(dst)
        if src == dst or not (0 <= src < self.n and 0 <= dst < self.n):
            return False
        if self.has_edge(src, dst):
            return False
        self.following[src].add(dst)
        self.followers[dst].add(src)
        self.changes.append((step, src, dst, "add", reason))
        if self.reciprocal and not self.has_edge(dst, src):
            self.following[dst].add(src)
            self.followers[src].add(dst)
            self.changes.append((step, dst, src, "add", reason + "|mirror"))
        return True

    def remove_edge(self, src: int, dst: int, step: int = -1, reason: str = "") -> bool:
        """Remove src -> dst (and the mirror when reciprocal). False if absent."""
        src, dst = int(src), int(dst)
        if not self.has_edge(src, dst):
            return False
        self.following[src].discard(dst)
        self.followers[dst].discard(src)
        self.changes.append((step, src, dst, "cut", reason))
        if self.reciprocal and self.has_edge(dst, src):
            self.following[dst].discard(src)
            self.followers[src].discard(dst)
            self.changes.append((step, dst, src, "cut", reason + "|mirror"))
        return True

    # ---------------------------------------------------------------- measures

    def out_degree(self, i: int) -> int:
        """How many agents i listens to. Zero means i is deaf: it can never be
        assigned a speaker, so it leaves the simulation without any error."""
        return len(self.following.get(int(i), ()))

    def in_degree(self, i: int) -> int:
        """How many agents listen to i. This is the influence measure that does
        not exist in an undirected graph."""
        return len(self.followers.get(int(i), ()))

    def edge_count(self) -> int:
        """Directed edges. A reciprocal pair counts as two."""
        return sum(len(v) for v in self.following.values())

    def edges(self):
        """(src, dst, is_mutual) for every directed edge, sorted.

        Never dedupe with `if i < j` the way undirected exports do - under
        direction that silently drops every high-to-low edge.
        """
        out = []
        for src in range(self.n):
            for dst in sorted(self.following[src]):
                out.append((src, dst, self.has_edge(dst, src)))
        return out

    def isolated(self):
        """Agents that hear nobody. Under direction these are invisible to a
        degree check that only counts ties, because they may still have
        followers."""
        return [i for i in range(self.n) if self.out_degree(i) == 0]

    def check_invariants(self):
        """Raise if the two maps disagree, or if reciprocity is violated while
        claimed. Cheap; call it in tests and after any bulk edit."""
        for src in range(self.n):
            for dst in self.following[src]:
                if src not in self.followers[dst]:
                    raise AssertionError(f"following has {src}->{dst}, followers does not")
        for dst in range(self.n):
            for src in self.followers[dst]:
                if dst not in self.following[src]:
                    raise AssertionError(f"followers has {src}->{dst}, following does not")
        if self.reciprocal:
            for src in range(self.n):
                if self.following[src] != self.followers[src]:
                    raise AssertionError(f"reciprocal claimed but agent {src} differs")
        return True


def evolve_once(
    net: "DirectedNetwork",
    listener_idx: int,
    speaker_idx: int,
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
    step: int,
    burnin_steps: int = 50,
    add_score_threshold: float = 100.0,
    cut_score_threshold: float = 25.0,
    soft_cut_distance: int = 2,
    p_add: float = 0.07,
    max_degree: int = 8,
    min_out_degree: int = 2,
    score_fn: Callable[..., float] | None = None,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Coupled one-in-one-out rewiring on a DirectedNetwork. Returns (added, cut).

    Add and cut are COUPLED so the edge count cannot drift. An independent
    add-probability and cut-probability would let density wander, and density
    drift confounds "the network rewired" with "the network got denser" - denser
    graphs converge faster for reasons unrelated to opinion.

    Coupling rule: a cut is attempted only if an add succeeded, and the add is
    ROLLED BACK when no edge meets the cut criteria. Edge count is therefore
    exactly conserved and the rule cannot deadlock. Degree *sequence* is left
    free on purpose - fixing it would keep a hub a hub by construction and rule
    out the outcome the experiment exists to observe.

    Scoring defaults to opinion only (`score_fn=opinion_only_pair_score`); pass a
    profile-aware scorer here when one exists, without touching this logic. The
    default thresholds are opinion-scaled: an exact-agreement candidate scores
    100 (so add_score_threshold=100 means "agree exactly"), and a tie at opinion
    distance >= 2 scores <= 25 (so cut_score_threshold=25 catches it). The old
    140 / 60 defaults were sized for the multi-attribute score and are
    unreachable on the 0..100 opinion-only scale.

    Direction: candidates come from `following[via]` - I discover people that
    someone I already listen to listens to - and the new edge is focal -> k.
    Under reciprocal=True the mirror is written automatically, reproducing the
    old symmetric behaviour exactly.

    `min_out_degree` guards OUT-degree, not total ties. An agent with
    out-degree 0 hears nobody and can never be assigned a speaker: it leaves the
    simulation silently while still looking connected, because it may retain
    followers. The undirected `min_degree` check cannot see this.
    """
    if rng is None:
        rng = random
    if score_fn is None:
        score_fn = opinion_only_pair_score
    if step < burnin_steps:
        return (0, 0)

    added = _evolve_try_add(net, listener_idx, speaker_idx, opinions, attributes,
                            step, add_score_threshold, p_add, max_degree, score_fn, rng)
    if not added:
        return (0, 0)

    cut = _evolve_try_cut(net, listener_idx, speaker_idx, opinions, attributes,
                          step, cut_score_threshold, soft_cut_distance,
                          min_out_degree, score_fn, rng)
    if not cut:
        net.remove_edge(added[0], added[1], step=step, reason="rollback:no_cut_available")
        return (0, 0)
    return (1, 1)


def _evolve_try_add(net, listener_idx, speaker_idx, opinions, attributes, step,
                    add_score_threshold, p_add, max_degree, score_fn, rng):
    """Friends-of-friends addition. Returns the (src, dst) added, or None."""
    for focal, via in ((listener_idx, speaker_idx), (speaker_idx, listener_idx)):
        if focal >= net.n or via >= net.n:
            continue
        candidates = [k for k in net.following[via]
                      if k != focal and k not in net.following[focal]]
        rng.shuffle(candidates)
        for k in candidates:
            if net.out_degree(focal) >= max_degree or net.in_degree(k) >= max_degree:
                continue
            op_f, op_k = opinions[focal], opinions[k]
            if op_f is None or op_k is None:
                continue
            try:
                if abs(int(op_f) - int(op_k)) != 0:      # exact agreement, as before
                    continue
            except Exception:
                continue
            if score_fn(focal, k, opinions, attributes) >= add_score_threshold \
                    and rng.random() < p_add:
                if net.add_edge(focal, k, step=step, reason="fof_add"):
                    return (focal, k)
    return None


def _evolve_try_cut(net, listener_idx, speaker_idx, opinions, attributes, step,
                    cut_score_threshold, soft_cut_distance, min_out_degree, score_fn, rng):
    """Cut the listener->speaker tie when it is weak. Returns True if cut."""
    if not net.has_edge(listener_idx, speaker_idx):
        return False
    op_l, op_s = opinions[listener_idx], opinions[speaker_idx]
    if op_l is None or op_s is None:
        return False
    try:
        d_op = abs(int(op_l) - int(op_s))
    except Exception:
        return False
    score = score_fn(listener_idx, speaker_idx, opinions, attributes)
    if not (score <= cut_score_threshold and d_op >= soft_cut_distance):
        return False
    # Guard OUT-degree on both sides: never leave an agent unable to hear anyone.
    if net.out_degree(listener_idx) <= min_out_degree:
        return False
    if net.reciprocal and net.out_degree(speaker_idx) <= min_out_degree:
        return False
    return net.remove_edge(listener_idx, speaker_idx, step=step, reason="weak_tie_cut")
