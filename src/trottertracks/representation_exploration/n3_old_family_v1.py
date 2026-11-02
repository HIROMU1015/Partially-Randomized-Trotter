"""Finite proposals for a one-variable OLD family; exact certificates only.

No production instance is evaluated on import. NumPy is lazy and is used only
for candidate eigenvectors/abscissas, never to declare PSD or infeasibility.
"""
from fractions import Fraction as F
import math
import resource
import time

from .n3_old_family_contract_v1 import (
    Arithmetic, CAPS, DYADIC_BITS, GOLDEN_STEPS, GRID, decode, encode, fingerprint,
)


def schur_PSD(matrix, a):
    """Producer's exact Schur elimination; checker uses independent LDL."""
    M = [[a.q(x) for x in row] for row in matrix]
    for i in range(len(M)):
        pivot = M[i][i]
        if pivot < 0: return False
        if pivot == 0:
            if any(M[j][i] for j in range(i + 1, len(M))): return False
            continue
        for j in range(i + 1, len(M)):
            for k in range(j, len(M)):
                M[j][k] = a.sub(M[j][k], a.div(a.mul(M[j][i], M[k][i]), pivot))
                M[k][j] = M[j][k]
    return True


def mixture_weight(p, q, a):
    """Find t in [0,1] with both t*p+(1-t)*q strictly negative."""
    lo, hi = F(0), F(1)
    for x, y in zip(p, q):
        slope = a.sub(x, y)
        if slope == 0:
            if y >= 0: return None
        elif slope > 0: hi = min(hi, a.div(-y, slope))
        else: lo = max(lo, a.div(-y, slope))
    if lo >= hi: return None
    t = a.div(a.add(lo, hi), 2)
    if not 0 <= t <= 1: return None
    traces = [a.add(a.mul(t, x), a.mul(a.sub(1, t), y)) for x, y in zip(p, q)]
    return t if all(x < 0 for x in traces) else None


def diagnose(instance):
    started, cpu = time.perf_counter(), time.process_time()
    A, v, nu, U, c, delta = decode(instance)
    a = Arithmetic(); n = len(A); D = a.add(a.sub(c, U), delta)
    beta2 = a.total(a.mul(x, x) for x in v)
    target = a.mul(delta, D); alpha = a.div(target, nu)
    B = [[a.sub(A[i][j], a.div(U, nu)) if i == j else a.q(A[i][j]) for j in range(n)] for i in range(n)]
    M0 = [[a.sub(alpha if i == j else F(0), a.mul(v[i], v[j])) for j in range(n)] for i in range(n)]
    M1 = [[a.add(M0[i][j], a.mul(D, B[i][j])) for j in range(n)] for i in range(n)]
    counts = {k: 0 for k in ('numeric_eigh_calls', 'primal_PSD_probes', 'ray_probes', 'mixture_probes')}
    events, rays, seen_rays, seen_k = [], [], set(), set()
    exhausted = []

    def finish(status, certificate=None):
        return {'schema_version': 1, 'input_id': instance['id'], 'instance_sha256': fingerprint(instance),
                'status': status, 'certificate': certificate, 'NORM_at_target': target >= beta2,
                'finite_failure_proves_infeasible': False, 'events': events,
                'resources': {'counts': counts, 'arithmetic_operations': a.operations,
                    'max_monitored_rational_bits': a.max_bits, 'exhausted_substages': exhausted,
                    'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu,
                    'peak_worker_rss_bytes_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
                'new_OCC_proposals': 0, 'new_U_c_bounds': 0, 'ground_or_gap_calls': 0,
                'central_theme': None, 'mandatory_stop': True, 'next_stage_authorized': False}

    def separator(W, origins):
        tr0 = a.total(a.mul(W[i][j], M0[j][i]) for i in range(n) for j in range(n))
        tr1 = a.total(a.mul(W[i][j], M1[j][i]) for i in range(n) for j in range(n))
        if max(tr0, tr1) >= 0: raise AssertionError('NONSTRICT_DUAL_PROPOSAL')
        status = 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION' if target < beta2 else 'OLD_FAMILY_SEPARATED_NORM_RECOVERS_TARGET'
        return finish(status, encode({'kind': 'dual', 'W': W, 'endpoint_traces': [tr0, tr1],
                                     'trace_W': a.total(W[i][i] for i in range(n)), 'origins': origins}))

    def ray(vector, origin):
        key = tuple(vector)
        if key in seen_rays or not any(vector): return None
        if counts['ray_probes'] == CAPS['ray_probes']:
            if 'ray_probes' not in exhausted: exhausted.append('ray_probes')
            return None
        seen_rays.add(key); counts['ray_probes'] += 1
        norm2 = a.total(a.mul(z, z) for z in vector)
        W = [[a.div(a.mul(x, y), norm2) for y in vector] for x in vector]
        endpoints = tuple(a.total(a.mul(W[i][j], M[j][i]) for i in range(n) for j in range(n)) for M in (M0, M1))
        events.append({'stage': 'ray', 'origin': origin, 'endpoint_traces': encode(list(endpoints))})
        if max(endpoints) < 0: return separator(W, [origin])
        # Cached endpoint traces make mixture proposals scalar work.
        for previous, traces, old_origin in rays:
            if counts['mixture_probes'] == CAPS['mixture_probes']:
                if 'mixture_probes' not in exhausted: exhausted.append('mixture_probes')
                break
            counts['mixture_probes'] += 1
            weight = mixture_weight(endpoints, traces, a)
            if weight is not None:
                mixture = [[a.add(a.mul(weight, W[i][j]), a.mul(a.sub(1, weight), previous[i][j])) for j in range(n)] for i in range(n)]
                return separator(mixture, [origin, old_origin, {'weight': encode(weight)}])
        rays.append((W, endpoints, origin))
        return None

    def primal(x, origin):
        k = a.mul(D, x)
        if k in seen_k: return None
        seen_k.add(k)
        if counts['primal_PSD_probes'] >= CAPS['primal_PSD_probes']: raise ArithmeticError('PRIMAL_PROBE_CAP')
        counts['primal_PSD_probes'] += 1
        M = [[a.add(M0[i][j], a.mul(k, B[i][j])) for j in range(n)] for i in range(n)]
        valid = schur_PSD(M, a)
        events.append({'stage': 'primal', 'x': encode(x), 'kappa': encode(k), 'origin': origin, 'exact_PSD': valid})
        if valid:
            mu = a.div(a.sub(a.mul(k, U), target), nu)
            return finish('OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH', encode({'kind': 'primal', 'kappa': k, 'mu': mu, 'origin': origin}))
        return None

    # Coefficient directions first; no numerical eigensolver if they certify dual.
    for vector, name in [(v, 'coupling')] + [([F(i == j) for i in range(n)], 'coordinate_' + str(j)) for j in range(n)]:
        answer = ray(vector, {'kind': 'analytic', 'name': name})
        if answer: return answer
    for j in range(GRID + 1):
        answer = primal(F(j, GRID), {'kind': 'fixed_grid', 'j': j})
        if answer: return answer

    import numpy as np
    F0, F1 = np.asarray(M0, float), np.asarray(M1, float)
    if not np.isfinite(F0).all() or not np.isfinite(F1).all(): raise ArithmeticError('NUMERIC_CONVERSION')
    normalization = max(1., float(np.max(np.abs(F0))), float(np.max(np.abs(F1))))
    F0, F1 = F0 / normalization, F1 / normalization
    cache = {}

    def probe(x):
        dx = F(round(x * (1 << DYADIC_BITS)), 1 << DYADIC_BITS)
        if dx in cache: return cache[dx]
        if counts['numeric_eigh_calls'] >= CAPS['numeric_eigh_calls']: raise ArithmeticError('NUMERIC_EIGH_CAP')
        counts['numeric_eigh_calls'] += 1
        values, vectors = np.linalg.eigh((1. - float(dx)) * F0 + float(dx) * F1)
        if not np.isfinite(values).all() or not np.isfinite(vectors).all(): raise ArithmeticError('NUMERIC_NONFINITE')
        events.append({'stage': 'numeric', 'x': encode(dx), 'normalized_min_eigenvalue': float(values[0]),
                       'qualification': 'candidate only; never a PSD or infeasibility verdict'})
        answer = primal(dx, {'kind': 'numeric_abscissa', 'call': counts['numeric_eigh_calls']})
        if answer is None:
            vector = [F(round(float(t) * (1 << DYADIC_BITS)), 1 << DYADIC_BITS) for t in vectors[:, 0]]
            answer = ray(vector, {'kind': 'numeric_min_eigenvector', 'call': counts['numeric_eigh_calls']})
        cache[dx] = float(values[0]), answer
        return cache[dx]

    lo, hi, ratio = 0., 1., (math.sqrt(5.) - 1.) / 2.
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    y1, answer = probe(x1)
    if answer: return answer
    y2, answer = probe(x2)
    if answer: return answer
    # Finite maximization of a concave score is proposal machinery only.
    for _ in range(GOLDEN_STEPS):
        if y1 < y2:
            lo, x1, y1 = x1, x2, y2
            x2 = lo + ratio * (hi - lo); y2, answer = probe(x2)
        else:
            hi, x2, y2 = x2, x1, y1
            x1 = hi - ratio * (hi - lo); y1, answer = probe(x1)
        if answer: return answer
    return finish('UNDETERMINED_FINITE_PROPOSALS')
