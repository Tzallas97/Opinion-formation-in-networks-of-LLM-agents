import math
import random
from typing import Dict, Set, Sequence, Mapping, Hashable, Union, Any
import networkx as nx

OpinionContainer = Union[Sequence[int], Mapping[int, int]]


def _safe_int(x: Any, default: int | None = None) -> int | None:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _norm_cat(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    return s if s else None


def _cat_map(values: list[str | None]) -> dict[str, int]:
    vals = sorted({v for v in values if v is not None})
    return {v: i for i, v in enumerate(vals)}


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

# Opinion distance -> points (your homophily curve)
OPINION_POINTS = {
    0: 100,  # same opinion
    1: 60,
    2: 25,
    3: 10,
    4: 5,    # opposite extremes
}

# Helper mappings: adjust these to match your CSV values if needed
POLITICAL_MAP = {
    # Example mapping – change to match your data
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


def maybe_add_new_friends_before_interaction(
    listener_idx: int,
    speaker_idx: int,
    neighbors: Dict[int, Set[int]],
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
    step: int,
    burnin_steps: int = 50,
    add_score_threshold: float = 140.0,
    p_add: float = 0.07,
    max_new_edges_per_step: int = 1,
    max_degree: int = 8,
    rng: random.Random | None = None,
) -> None:
    """
    Called at the START of an interaction between (listener_idx, speaker_idx).

    Models: "we go for coffee and I might bring a friend".

    - Only runs after burnin_steps.
    - Only creates edges via friends-of-friends:
        * L with neighbors of S
        * S with neighbors of L
    - New edge (focal, k) created only if:
        * same opinion (|op_focal - op_k| == 0)
        * compute_pair_score(focal, k, ...) >= add_score_threshold
        * probability p_add
        * degrees stay within [min_degree, max_degree]
        * at most max_new_edges_per_step per call (both sides combined)
    """
    if rng is None:
        rng = random

    if step < burnin_steps:
        return
    if listener_idx not in neighbors or speaker_idx not in neighbors:
        return

    degree = {i: len(neighs) for i, neighs in neighbors.items()}
    new_edges_added = 0

    def try_form_edges(focal: int, via: int) -> None:
        nonlocal new_edges_added
        if new_edges_added >= max_new_edges_per_step:
            return

        focal_neighs = neighbors[focal]
        via_neighs = neighbors[via]

        # Candidates: neighbors of 'via' that are not focal and not already neighbors of 'focal'
        candidates = [
            k for k in via_neighs
            if k != focal and k not in focal_neighs
        ]

        rng.shuffle(candidates)

        for k in candidates:
            if new_edges_added >= max_new_edges_per_step:
                break
            if degree[focal] >= max_degree or degree[k] >= max_degree:
                continue

            op_f = opinions[focal]
            op_k = opinions[k]
            if op_f is None or op_k is None:
                continue

            # EXACT same opinion required
            try:
                d_op_fk = abs(int(op_f) - int(op_k))
            except Exception:
                d_op_fk = 999

            if d_op_fk != 0:
                continue

            s_fk = compute_pair_score(focal, k, opinions, attributes)
            if s_fk >= add_score_threshold and rng.random() < p_add:
                neighbors[focal].add(k)
                neighbors[k].add(focal)
                degree[focal] += 1
                degree[k] += 1
                new_edges_added += 1

    # Listener may meet speaker's friends
    try_form_edges(listener_idx, speaker_idx)
    # Speaker may meet listener's friends
    try_form_edges(speaker_idx, listener_idx)



def maybe_cut_edge_after_interaction(
    listener_idx: int,
    speaker_idx: int,
    neighbors: Dict[int, Set[int]],
    opinions: OpinionContainer,
    attributes: Mapping[int, Mapping[str, Any]],
    step: int,
    burnin_steps: int = 50,
    cut_score_threshold: float = 60.0,
    soft_cut_distance: int = 2,
    p_cut: float = 0.3,
    min_degree: int = 2,
    rng: random.Random | None = None,
) -> None:
    """
    Called at the END of an interaction between (listener_idx, speaker_idx),
    AFTER the listener's opinion has potentially changed.

    Edge (L,S) is considered weak if:
      - S_LS <= cut_score_threshold  (low overall similarity)
      - and opinion distance >= soft_cut_distance (meaningful disagreement)

    If weak and both nodes have degree > min_degree,
    cut edge with probability p_cut.
    """
    if rng is None:
        rng = random

    if step < burnin_steps:
        return
    if listener_idx not in neighbors or speaker_idx not in neighbors:
        return
    if speaker_idx not in neighbors[listener_idx]:
        return

    degree = {i: len(neighs) for i, neighs in neighbors.items()}

    op_l = opinions[listener_idx]
    op_s = opinions[speaker_idx]
    if op_l is None or op_s is None:
        return

    try:
        d_op = abs(int(op_l) - int(op_s))
    except Exception:
        return

    s_ls = compute_pair_score(listener_idx, speaker_idx, opinions, attributes)

    weak_edge = (s_ls <= cut_score_threshold and d_op >= soft_cut_distance)

    if weak_edge and degree[listener_idx] > min_degree and degree[speaker_idx] > min_degree:
        if rng.random() < p_cut:
            neighbors[listener_idx].remove(speaker_idx)
            neighbors[speaker_idx].remove(listener_idx)



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
