"""Opt-in TOTAL RPE estimator Z1+T1; the original partial decoder is unchanged.

Z1: X=Y=0 at m=0 gives +0.0; at m>=1 the previous estimate is held.
T1: a branch tie (gap <= 1e-12, or exact) keeps the computed argmin with the
    lowest branch ID and is NOT an unresolved stop.
Otherwise every operation is the original one: exact count -> binary64 mean,
math.atan2, the reference wrap, and rpe_decoder_fast._nearest (scalar rule).
On histories the original decoder resolves, outputs are bit-identical. Zero
holds and tie choices are recorded as diagnostics. Not an RMSE qualification.
"""
import math

from . import rpe_decoder_fast as F

ESTIMATOR = 'consecutive_dyadic_nearest_branch_principal_energy_total_v1'
ORIGINAL_ESTIMATOR = 'consecutive_dyadic_nearest_branch_principal_energy_v1'
DECODED_TOTAL = 'DECODED_TOTAL_ZERO_HOLD_TIE_ARGMIN_NOT_RMSE_QUALIFIED'
RULES = dict(zero='Z1_HOLD_PREVIOUS_ESTIMATE_M0_PLUS_ZERO',
             tie='T1_COMPUTED_ARGMIN_LOWEST_BRANCH_ID_NO_UNRESOLVED_STOP',
             tie_diagnostic_tolerance_rad=1e-12)
TIE_TOLERANCE = 1e-12
SOURCE = dict(module='src/trottertracks/rpe_total_decoder.py', estimator=ESTIMATOR,
    original_estimator=ORIGINAL_ESTIMATOR, transition_kernel='src/trottertracks/rpe_decoder_fast.py:_nearest',
    proposal=dict(commit='104dacdc64eb01c7f1b5207a32c4a4d1c6a83a37',
                  path='docs/research/rpe_accuracy_contract_w0_20261114/REPORT.md'))


def initial(principal, zero):
    """Round 0: wrapped atan2 phase, or +0.0 for X=Y=0 (no prior is used)."""
    import numpy as np
    return np.where(zero, 0.0, principal)


def step(principal, estimate, q, zero):
    """ONE Z1+T1 transition, shared by the decoder and the exact DP.

    Returns (estimate, tie) where tie flags non-zero rows whose two smallest
    computed branch distances differ by <= 1e-12 (the original stop rule).
    """
    import numpy as np
    value, _, best, runner, _ = F._nearest(principal, estimate, q)
    tie = (np.abs(runner - best) <= TIE_TOLERANCE) & ~zero
    return np.where(zero, estimate, value), tie


def decode_coordinates_batch_total(coordinates, *, h):
    """Array API (trials, rounds, X/Y); same input checks as the original."""
    import numpy as np
    if isinstance(h, bool) or not math.isfinite(float(h)) or float(h) <= 0:
        raise ValueError('Positive finite h required')
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1] == 0 or xy.shape[2] != 2:
        raise ValueError('Coordinates must have shape (trials, nonempty rounds, 2)')
    if not np.all(np.isfinite(xy)) or np.any(np.abs(xy) > 1):
        raise ValueError('Finite binary-outcome means required')
    if (1 << (xy.shape[1]-1)) > 2**52:
        raise ValueError('Branch integer arithmetic outside exact binary64 domain')
    zeros = np.all(xy == 0, axis=-1)
    estimate = None; ties = np.zeros(zeros.shape, dtype=bool); path = []
    for m in range(xy.shape[1]):
        phase = np.fromiter(map(math.atan2, xy[:, m, 1], xy[:, m, 0]), dtype=np.float64, count=len(xy))
        principal = F._wrap(phase)
        if m:
            estimate, ties[:, m] = step(principal, estimate, 1 << m, zeros[:, m])
        else:
            estimate = initial(principal, zeros[:, 0])
        path.append(estimate)
    out = []
    for i in range(len(xy)):
        zero_rounds = [m for m in range(xy.shape[1]) if zeros[i, m]]
        tie_rounds = [m for m in range(xy.shape[1]) if ties[i, m]]
        final = float(path[-1][i])
        out.append(dict(status=DECODED_TOTAL, estimator=ESTIMATOR, energy_estimate_ha=-final/float(h),
            final_phase_estimate=final, phase_path=[float(v[i]) for v in path],
            zero_hold_rounds=zero_rounds, tie_argmin_rounds=tie_rounds,
            partial_decoder_status=('UNRESOLVED_ZERO_COORDINATES' if zero_rounds else
                                    'UNRESOLVED_BRANCH_TIE' if tie_rounds else F.DECODED),
            rules=dict(RULES), prior_alias_qualified=False, rmse_guaranteed=False))
    return out


def decode_supplied_coordinates_total(coordinates, *, h):
    """Scalar wrapper with the original accepted coordinate types."""
    xy = list(coordinates)
    if any(type(v) not in (int, float) for row in xy for v in row):
        raise ValueError('Finite binary-outcome means required')
    return decode_coordinates_batch_total([xy], h=h)[0]


def decode_counts_total(rounds, *, h, M, prior):
    """Original exact count conversion, then the total estimator.

    The prior is diagnostic only, exactly as for the original decoder.
    """
    from copy import deepcopy
    from . import rpe_decoder_statistics as S
    S._integer(M, 'M')
    if len(prior) != 2 or any(not math.isfinite(float(v)) for v in prior) or prior[0] >= prior[1]:
        raise ValueError('Ordered finite energy prior required')
    by_m = {r['m']: r for r in rounds}
    if len(by_m) != len(rounds) or sorted(by_m) != list(range(M+1)):
        raise ValueError('Complete consecutive measurement rounds required')
    coordinates = []
    for m in range(M+1):
        row = {axis: S.count_mean(**by_m[m][axis]) for axis in S.p.AXES}
        coordinates.append((row['X']['mean'], row['Y']['mean']))
    out = decode_supplied_coordinates_total(coordinates, h=float(h))
    energy = out['energy_estimate_ha']
    out.update(count_conversion=S.COUNT_RULE, prior=deepcopy(prior), prior_applied_to_decoder=False,
        estimate_in_supplied_prior=prior[0] <= energy <= prior[1], decoder_source=deepcopy(SOURCE),
        output_range='[-pi_f/h, pi_f/h] binary64 principal range; not a certified real enclosure',
        alias_qualified=False)
    return out
