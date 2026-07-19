"""Single source of truth for the population-level opinion metrics B, D and P.

This module is the single definition of B, D and P for the whole codebase, so the
simulator, the plots and the evaluation tools cannot drift apart.

Definitions
-----------
B : mean belief over the population.
D : population standard deviation (ddof=0).
P : bipolarization, ``4 * pos * neg`` where ``pos`` and ``neg`` are the shares of
    agents strictly beyond ``+POLE_THRESHOLD`` and ``-POLE_THRESHOLD``.
    Range [0, 1]. 1.0 is a perfect 50/50 split between the two camps; 0.0 means
    at least one camp is empty.

Why not (var - B^2) / (var + B^2)
---------------------------------
That expression, previously used by the simulator and the plotting script, is
algebraically ``(r^2 - 1) / (r^2 + 1)`` with ``r = D / |B|``. It is a monotone
rescaling of the spread-to-mean-displacement ratio and carries no information
about bimodality: every zero-mean population scores exactly 1.0. A population of
16 agents at 0 with one agent at +1 and one at -1 scores maximum "polarization"
under it, while this module's P scores it 0.012, which is why it is not used.

Known limitation
----------------
P is threshold-based, so it does not distinguish a +-1 split from a +-2 split
(both score 1.0). A distance-sensitive index is a candidate future ADDITION;
it must never silently redefine P.
"""

from __future__ import annotations

POLE_THRESHOLD = 0.5

__all__ = ["POLE_THRESHOLD", "polarization", "bdp", "bdp_series"]


def _as_float_list(values):
    """Coerce any iterable of numbers to a plain list of floats, dropping NaNs."""
    out = []
    for v in (values if values is not None else []):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:  # drop NaN
            out.append(f)
    return out


def polarization(values) -> float:
    """P = 4 * pos * neg. Returns 0.0 for an empty population."""
    vals = _as_float_list(values)
    n = len(vals)
    if n == 0:
        return 0.0
    pos = sum(1 for x in vals if x > POLE_THRESHOLD) / n
    neg = sum(1 for x in vals if x < -POLE_THRESHOLD) / n
    return 4.0 * pos * neg


def bdp(values):
    """Return (B, D, P) for one population snapshot. Empty input gives zeros."""
    vals = _as_float_list(values)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    B = sum(vals) / n
    D = (sum((x - B) ** 2 for x in vals) / n) ** 0.5
    return float(B), float(D), polarization(vals)


def bdp_series(matrix):
    """Vectorised (B, D, P) over a 2-D array whose rows are time steps.

    Falls back to the pure-Python path when numpy is unavailable. Returns three
    numpy arrays when numpy is present, otherwise three lists.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        rows = [bdp(row) for row in matrix]
        return ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])

    try:
        arr = np.asarray(matrix, dtype=float)
    except ValueError as exc:
        raise ValueError(
            "bdp_series expects a rectangular 2-D matrix (rows = time steps, "
            "columns = agents); got rows of unequal length"
        ) from exc
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        empty = np.zeros(arr.shape[0] if arr.ndim > 1 else 0)
        return empty, empty.copy(), empty.copy()

    B = arr.mean(axis=1)
    D = arr.std(axis=1, ddof=0)
    n = arr.shape[1]
    pos = (arr > POLE_THRESHOLD).sum(axis=1) / n
    neg = (arr < -POLE_THRESHOLD).sum(axis=1) / n
    return B, D, 4.0 * pos * neg
