"""Independent nearest-branch decoder; original scalar/vector references stay intact.

Four neighbouring lattice points suffice for the two closest circle branches.
Scalar compatibility uses argmin's lowest-index exact-tie rule. Legacy vector
compatibility retains NumPy's unstable argsort tie rule with a rare exact-tie
fallback. No physical inputs, RNG, loss, confidence interval or N calculation.
"""
import math

DECODED = 'DECODED_SUPPLIED_COORDINATES_NOT_RMSE_QUALIFIED'
SOURCE = dict(module='src/trottertracks/rpe_decoder_fast.py',
    reference='src/trotterlib/rpe_four_round_phase_validation.py:reconstruct_rpe_phase',
    B='170047ef6e7a72aef5f4c6ff9c73f942097a694e')


def _arrays(phases, q_values):
    import numpy as np
    q = tuple(int(v) for v in q_values)
    x = tuple(np.asarray(v, dtype=np.float64) for v in phases)
    if not x or len(x) != len(q) or any(v.shape != x[0].shape for v in x):
        raise ValueError('Equal nonempty round sequence and equal array shapes required')
    if q != tuple(1 << m for m in range(len(q))):
        raise ValueError('Consecutive dyadic q_values starting at one required')
    if q[-1] > 2**52:
        raise ValueError('Branch integer arithmetic outside exact binary64 domain')
    if any(not np.all(np.isfinite(v)) for v in x):
        raise ValueError('Finite phases required')
    return x, q


def _wrap(x):
    # Preserve the reference's operation order, including +/- pi convention.
    import numpy as np
    return (np.asarray(x) + math.pi) % (2.0 * math.pi) - math.pi


def _nearest(principal, estimate, q, *, tie_rule='scalar'):
    import numpy as np
    if tie_rule not in ('scalar', 'legacy_vector'):
        raise ValueError('Explicit scalar or legacy_vector tie rule required')
    k0 = np.floor((q * estimate - principal) / (2.0 * math.pi)).astype(np.int64)
    branches = np.stack([(k0 - 1 + j) % q for j in range(min(q, 4))], axis=-1)
    branches.sort(axis=-1)  # argmin then preserves lowest original branch ID.
    candidates = _wrap((principal[..., None] + 2.0 * math.pi * branches) / q)
    distances = np.abs(np.angle(np.exp(1j * (candidates - estimate[..., None]))))
    selected = np.argmin(distances, axis=-1)
    take = lambda values: np.take_along_axis(values, selected[..., None], axis=-1)[..., 0]
    best = take(distances); branch = take(branches); value = take(candidates)
    next_distance = np.partition(distances, 1, axis=-1)[..., 1]
    ties = best == next_distance
    if tie_rule == 'legacy_vector' and np.any(ties):
        # np.argsort(kind='quicksort') is not stable, so there is no specified
        # lowest-ID tie rule to substitute. Reproduce only those exact-tie rows.
        flat = np.flatnonzero(ties.ravel())
        pp = principal.ravel()[flat]; ee = estimate.ravel()[flat]
        all_candidates = _wrap((pp[:, None] + 2.0 * math.pi * np.arange(q, dtype=np.float64)) / q)
        dd = np.abs(np.angle(np.exp(1j * (all_candidates - ee[:, None]))))
        order = np.argsort(dd, axis=-1)
        branch = np.asarray(branch).copy(); value = np.asarray(value).copy()
        branch.ravel()[flat] = order[:, 0]
        value.ravel()[flat] = all_candidates[np.arange(len(flat)), order[:, 0]]
    return value, branch, best, next_distance, ties


def reconstruct_batch(measured_round_phases, q_values, *, tie_rule='scalar'):
    """Return arrays of all diagnostics, O(batch) branch work per round.

    Legacy-vector exact ties alone take O(q) time/storage, preserving the old
    unstable argsort semantics. Scalar/B mode never enumerates all branches.
    Scientific equivalence is tested for M=0..12; this is no RMSE qualification.
    """
    import numpy as np
    phases, q = _arrays(measured_round_phases, q_values)
    estimate = _wrap(phases[0]); rounds = []
    ambiguous = np.zeros(estimate.shape, dtype=bool)
    exact_ties = np.zeros(estimate.shape, dtype=bool)
    for m, (phase, qq) in enumerate(zip(phases, q)):
        principal = _wrap(phase)
        if m:
            estimate, branch, best, runner, ties = _nearest(principal, estimate, qq, tie_rule=tie_rule)
            ambiguous |= np.isclose(best, runner, rtol=0., atol=1e-12)
            exact_ties |= ties
        else:
            branch = np.zeros(estimate.shape, dtype=np.int64)
            best = np.zeros(estimate.shape); runner = None
        rounds.append(dict(q_m=qq, measured_round_phase=principal, selected_branch=branch,
            base_phase_estimate=estimate, distance_from_previous=best,
            next_smallest_branch_distance=runner))
    return dict(rounds=rounds, final_phase_estimate=estimate, ambiguous=ambiguous,
        exact_distance_tie=exact_ties, tie_rule=tie_rule)


def vectorized_reconstruction_fast(measured_round_phases, q_values):
    """Drop-in tuple API preserving the original vectorized argsort tie rule."""
    import numpy as np
    result = reconstruct_batch(measured_round_phases, q_values, tie_rule='legacy_vector')
    return (np.asarray(result['final_phase_estimate'], dtype=np.float64),
            [np.asarray(r['selected_branch'], dtype=np.int64) for r in result['rounds']],
            np.asarray(result['ambiguous'], dtype=bool))


def _decoded_trial(result, index):
    rounds = []
    for row in result['rounds']:
        record = dict(q_m=row['q_m'])
        for key in ('measured_round_phase', 'selected_branch', 'base_phase_estimate',
                    'distance_from_previous', 'next_smallest_branch_distance'):
            if row[key] is not None:
                record[key] = int(row[key][index]) if key == 'selected_branch' else float(row[key][index])
        rounds.append(record)
    return dict(rounds=rounds, final_phase_estimate=float(result['final_phase_estimate'][index]))


def reconstruct_rpe_phase_fast(measured_round_phases, q_values):
    import numpy as np
    result = reconstruct_batch([np.array([v], dtype=np.float64) for v in measured_round_phases], q_values)
    return _decoded_trial(result, 0)


def decode_coordinates_batch(coordinates, *, h):
    """Array API (trials, rounds, X/Y), canonical scalar semantics.

    Use math.atan2 exactly as A's scalar wrapper: NumPy arctan2 may differ by
    one ULP near a branch boundary. Signed zeros and zero-coordinate precedence
    are retained. Invalid values are rejected for the entire supplied batch.
    """
    import numpy as np
    if isinstance(h, bool) or not math.isfinite(float(h)) or float(h) <= 0:
        raise ValueError('Positive finite h required')
    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1] == 0 or xy.shape[2] != 2:
        raise ValueError('Coordinates must have shape (trials, nonempty rounds, 2)')
    if not np.all(np.isfinite(xy)) or np.any(np.abs(xy) > 1):
        raise ValueError('Finite binary-outcome means required')
    zero = np.any(np.all(xy == 0, axis=-1), axis=-1)
    phases = [np.fromiter(map(math.atan2, xy[:, m, 1], xy[:, m, 0]), dtype=np.float64, count=len(xy))
              for m in range(xy.shape[1])]
    decoded = reconstruct_batch(phases, [1 << m for m in range(xy.shape[1])])
    out = []
    for i in range(len(xy)):
        if zero[i]:
            out.append(dict(status='UNRESOLVED_ZERO_COORDINATES', energy_estimate_ha=None))
        elif decoded['ambiguous'][i]:
            out.append(dict(status='UNRESOLVED_BRANCH_TIE', energy_estimate_ha=None))
        else:
            record = _decoded_trial(decoded, i)
            out.append(dict(status=DECODED, energy_estimate_ha=-record['final_phase_estimate']/float(h),
                decoded=record, prior_alias_qualified=False, rmse_guaranteed=False))
    return out


def decode_supplied_coordinates_fast(coordinates, *, h):
    """Scalar A-wrapper-compatible output, without q-branch enumeration."""
    # Preserve the original accepted scalar coordinate types, including bool
    # rejection. B's count conversion supplies ordinary Python floats.
    xy = list(coordinates)
    if any(type(v) not in (int, float) for row in xy for v in row):
        raise ValueError('Finite binary-outcome means required')
    return decode_coordinates_batch([xy], h=h)[0]


def decode_sampled_counts(samples, rounds, *, M, h, prior, count_mean, decoder_source):
    """B simulate batch hook; reuse B's exact count_mean on unique counts.

    samples are B's already-generated integer arrays. No RNG, probabilities,
    shot choice, loss or confidence interval is reimplemented here. Returning
    B's existing trial dictionaries keeps summarize() unchanged.
    """
    import numpy as np
    from copy import deepcopy
    if type(M) is not int or M < 0 or [r['m'] for r in rounds] != list(range(M+1)) or len(samples) != M+1:
        raise ValueError('Complete ordered consecutive rounds required')
    if len(prior) != 2 or any(not math.isfinite(float(v)) for v in prior) or prior[0] >= prior[1]:
        raise ValueError('Ordered finite energy prior required')
    shape = np.asarray(samples[0]['X']).shape
    if len(shape) != 1:
        raise ValueError('One-dimensional sampled count arrays required')
    coordinates = np.empty((shape[0], M+1, 2), dtype=np.float64)
    for m, row in enumerate(rounds):
        for axis_index, axis in enumerate(('X', 'Y')):
            values = np.asarray(samples[m][axis]); n = row['N_'+axis]
            if type(n) is not int or n < 1 or values.shape != shape or values.dtype.kind not in 'iu':
                raise ValueError('Exact shot integers and matching integer count arrays required')
            if np.any(values < 0) or np.any(values > n):
                raise ValueError('Count outside [0,shots]')
            unique, inverse = np.unique(values, return_inverse=True)
            # Keep B's signed-integer subtraction / exact Fraction -> float
            # conversion. In particular do not substitute 2*k/N - 1.
            means = np.array([count_mean(plus=int(k), shots=n)['mean'] for k in unique])
            coordinates[:, m, axis_index] = means[inverse]
    results = decode_coordinates_batch(coordinates, h=h)
    for result in results:
        energy = result['energy_estimate_ha']
        result.update(count_conversion='EXACT_SIGNED_INTEGER_SUM_THEN_BINARY64_RATIO_V1',
            decoder_source=deepcopy(decoder_source), prior=deepcopy(prior), prior_applied_to_decoder=False,
            estimate_in_supplied_prior=None if energy is None else prior[0] <= energy <= prior[1],
            output_range='(-pi/h,pi/h] nominal principal-phase range; binary64 not a certified real enclosure',
            alias_qualified=False, unresolved_loss=None)
        result['decoder_acceleration_source'] = deepcopy(SOURCE)
    return results
