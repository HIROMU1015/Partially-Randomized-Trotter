"""Artificial shared-basis merge PoC, not a fitter or a molecular entry point.

Reuse the native spin-lift/key producer and general-J validation. The returned
group count is an outer-rank upper bound, not a minimal tensor rank. Even a
contiguous merge is only a possible local PF rewrite; no sampling-law or
compiled-price equivalence is implied. Existing production SOURCE is untouched.
"""
import math

VERSION = 'exact_shared_basis_general_J_merge_poc_v1'


def merge(J, U, groups, *, ld, mode='adjacent'):
    """Sum general J in declared exact-basis groups, retaining the original U.

    ``adjacent`` forbids crossing an intervening factor; ``representation``
    permits noncontiguous groups for a Hamiltonian-only comparison. Neither
    mode crosses the deterministic/tail boundary. No gauge alignment, nearby-U
    threshold, frame sorting, PSD projection, zero pruning or new codec.
    Binary64 fsum is rounded; numerical reconstruction must still be checked.
    """
    import numpy as np
    from .c1_general_k_adapter import _symmetric
    from .c_df_basis_switch import exact_keys

    J, U = np.asarray(J), np.asarray(U)
    if (J.dtype != np.float64 or U.dtype != np.float64 or J.ndim != 3
            or J.shape != U.shape or not len(J) or J.shape[1] != J.shape[2]
            or not J.shape[1] or not np.isfinite(J).all()):
        raise ValueError('FINITE_FLOAT64_J_U_BATCH')
    for a in J:
        _symmetric(a, 'general_J')
    keys = exact_keys(U, 0)  # exact current _matrix/_spin_lift convention
    rank, n = len(J), J.shape[-1]
    if type(ld) is not int or not 0 <= ld <= rank:
        raise ValueError('ORIGINAL_PREFIX_LD')
    if mode not in ('adjacent', 'representation'):
        raise ValueError('EXPLICIT_MERGE_MODE')
    if (not isinstance(groups, tuple) or not groups
            or any(not isinstance(g, tuple) or not g for g in groups)
            or any(type(i) is not int for g in groups for i in g)):
        raise ValueError('ORDERED_PARTITION')
    flat = [i for g in groups for i in g]
    if (sorted(flat) != list(range(rank))
            or any(tuple(sorted(g)) != g for g in groups)
            or tuple(g[0] for g in groups) != tuple(sorted(g[0] for g in groups))):
        raise ValueError('ORDERED_PARTITION')
    adjacent = all(g == tuple(range(g[0], g[-1]+1)) for g in groups)
    if mode == 'adjacent' and not adjacent:
        raise ValueError('NONCONTIGUOUS_PF_REWRITE_FORBIDDEN')
    for g in groups:
        if any((i < ld) != (g[0] < ld) for i in g):
            raise ValueError('HEAD_TAIL_BOUNDARY')
        if any(keys[i] != keys[g[0]] for i in g):
            raise ValueError('EXACT_NATIVE_BASIS_REQUIRED')
    try:
        merged = np.asarray([[[math.fsum(float(J[k, i, j]) for k in g)
                               for j in range(n)] for i in range(n)]
                             for g in groups], dtype=np.float64)
    except OverflowError as exc:
        raise ValueError('UNREPRESENTABLE_J_SUM') from exc
    if not np.isfinite(merged).all():
        raise ValueError('UNREPRESENTABLE_J_SUM')
    return merged, U[[g[0] for g in groups]].copy(), dict(
        original_frames=rank, group_count=len(groups), groups=groups,
        ld=sum(g[0] < ld for g in groups), contiguous_groups=adjacent,
        tensor_family='ordinary_general_J_CDF_at_most_group_count',
        minimal_outer_rank=None, finite_PR_law_preserved=None,
        compiled_cost_change=None, effective_J_null_dimensions=(rank-len(groups))*n*(n+1)//2)
