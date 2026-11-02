"""N3 OCC v1: coefficient proposals and exact sufficient witnesses.

No Fock spectra, ground states, gap, circuit library or old science runner.
The eight transfer inputs must only be evaluated after separate authorization.
"""
from __future__ import annotations
from contextlib import contextmanager
from fractions import Fraction as F
from itertools import combinations
import hashlib
import json
import math
import resource
import time
import numpy as np

INPUT_PATH = 'artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json'
INPUT_SHA = 'c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4'
METHODS = ('NORM', 'FIVE', 'OLD_DIRECT', 'OCC')
ROWS = ('c0', 'c_occ', 'c_star')
RATIONAL_BITS = 4096
CAPS = {'U': (0, 0), 'C0': (1, 11), 'COCC': (1, 12), 'CDH': (1, 12),
        'NORM': (0, 0), 'FIVE': (0, 5), 'OLD_DIRECT': (128, 10),
        'OCC': (138, 120), 'TRANSPORT': (0, 0), 'FALLBACK': (0, 3), 'SECONDARY': (16, 6)}
MAX_EIG = 850
MAX_PSD = 512


def canon(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canon(x)).hexdigest()


def enc(x):
    if isinstance(x, F):
        return [x.numerator, x.denominator]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [enc(v) for v in x]
    return x


def dec(x):
    if not isinstance(x, list) or len(x) != 2 or any(type(v) is not int for v in x) or x[1] <= 0:
        raise ValueError('RATIONAL_SCHEMA')
    return F(*x)


class Unresolved(ArithmeticError):
    pass


class Arithmetic:
    """Guard rational operands/results; Fraction internal temporary integers excluded."""
    def __init__(self, cap=RATIONAL_BITS):
        self.cap = cap
        self.max_bits = 0
        self.operations = 0

    def q(self, x):
        x = F(x)
        b = max(abs(x.numerator).bit_length(), x.denominator.bit_length())
        self.max_bits = max(self.max_bits, b)
        if b > self.cap:
            raise Unresolved('UNRESOLVED_ARITHMETIC_CAP')
        return x

    def _op(self, x, y, op):
        x, y = self.q(x), self.q(y)
        self.operations += 1
        return self.q(op(x, y))

    def add(self, x, y): return self._op(x, y, lambda a, b: a + b)
    def sub(self, x, y): return self._op(x, y, lambda a, b: a - b)
    def mul(self, x, y): return self._op(x, y, lambda a, b: a * b)
    def div(self, x, y): return self._op(x, y, lambda a, b: a / b)
    def neg(self, x): return self.sub(0, x)
    def total(self, values):
        answer = F(0)
        for value in values:
            answer = self.add(answer, value)
        return answer


def identity(n): return [[F(i == j) for j in range(n)] for i in range(n)]
def outer(v, a): return [[a.mul(x, y) for y in v] for x in v]
def scaled(M, s, a): return [[a.mul(x, s) for x in row] for row in M]
def plus(A, B, a): return [[a.add(x, y) for x, y in zip(r, s)] for r, s in zip(A, B)]
def shifted(M, z, a): return [[a.sub(x, z) if i == j else a.q(x) for j, x in enumerate(row)] for i, row in enumerate(M)]
def quadratic(M, v, a): return a.total(a.mul(a.mul(v[i], M[i][j]), v[j]) for i in range(len(v)) for j in range(len(v)))


def decode_case(case, a):
    fields = {'id', 'base', 'modes', 'particles', 'active', 'external', 'scale', 'amplitude', 'direction', 'h', 'density'}
    if set(case) != fields: raise ValueError('UNKNOWN_OR_ORACLE_CASE_FIELD')
    n, nu, active, d = case['modes'], case['particles'], case['active'], case['external']
    if type(n) is not int or not 2 <= n <= 10 or type(nu) is not int or not 1 <= nu <= n - 1: raise ValueError('MODE_PARTICLE_SCOPE')
    if not isinstance(active, list) or any(type(i) is not int for i in active) or type(d) is not int or not 0 <= d < n or active != [i for i in range(n) if i != d]: raise ValueError('ORDERED_ONE_EXTERNAL_PARTITION')
    h = [[a.q(dec(x)) for x in row] for row in case['h']]
    if len(h) != n or any(len(r) != n for r in h) or any(h[i][j] != h[j][i] for i in range(n) for j in range(n)): raise ValueError('REAL_SYMMETRIC_ONE_BODY')
    den = {}
    for row in case['density']:
        if set(row) != {'i', 'j', 'coefficient'}: raise ValueError('DENSITY_SCHEMA')
        i, j = row['i'], row['j']
        if type(i) is not int or type(j) is not int or not 0 <= i < n or not 0 <= j < n or i == j: raise ValueError('DENSITY_INDEX')
        key = tuple(sorted((i, j)))
        value = a.q(dec(row['coefficient']))
        if key in den or value < 0: raise ValueError('NONNEGATIVE_UNORDERED_DENSITY')
        den[key] = value
    scale = a.q(dec(case['scale']))
    if scale <= 0: raise ValueError('POSITIVE_SCALE')
    a.q(dec(case['amplitude']))
    if any(type(case[k]) is not str for k in ('id', 'base', 'direction')): raise ValueError('ID_SCHEMA')
    return h, den, active, d, nu, a.div(scale, 250)


class Engine:
    def __init__(self, bit_cap=RATIONAL_BITS, cached=True):
        self.a = Arithmetic(bit_cap)
        self.cached = cached
        self.proofs = {}
        self.psd_cache = {}
        self.numeric_cache = {}
        self.primitives = {}
        self.events = []
        self.event = None
        self.counts = {'eig': 0, 'psd': 0, 'eig_requests': 0, 'psd_requests': 0,
                       'cache_hits': 0, 'occ_spectral_pairs': 0, 'bound_spectral_pairs': 0}

    @contextmanager
    def stage(self, name, kind, dependencies=()):
        if self.event is not None: raise ValueError('nested cost stages')
        event = {'id': name, 'kind': kind, 'dependencies': list(dependencies), 'eig': 0, 'psd': 0,
                 'primitive_requests': [], 'started_wall': time.perf_counter(), 'started_cpu': time.process_time()}
        self.event = event
        try:
            yield event
        finally:
            event['wall_seconds'] = time.perf_counter() - event.pop('started_wall')
            event['cpu_seconds'] = time.process_time() - event.pop('started_cpu')
            event['overhead_wall_seconds'] = max(0., event['wall_seconds'] - sum(r['actual_wall_seconds'] for r in event['primitive_requests']))
            event['overhead_cpu_seconds'] = max(0., event['cpu_seconds'] - sum(r['actual_cpu_seconds'] for r in event['primitive_requests']))
            self.events.append(event)
            self.event = None

    def _primitive(self, key, category, operation):
        if self.event is None: raise ValueError('primitive outside cost stage')
        self.counts[category + '_requests'] += 1
        cached = self.cached and key in self.primitives
        if cached:
            self.counts['cache_hits'] += 1
            previous = self.primitives[key]
            result = previous['result']
            wall = cpu = 0.
        else:
            cap = CAPS[self.event['kind']][0 if category == 'eig' else 1]
            denied = 'UNRESOLVED_STAGE_' + category.upper() + '_CAP' if self.event[category] >= cap else 'UNRESOLVED_CASE_' + category.upper() + '_CAP' if self.counts[category] >= (MAX_EIG if category == 'eig' else MAX_PSD) else None
            if denied:
                self.event['primitive_requests'].append({'key': key, 'category': category, 'cache_hit': False,
                    'denied': True, 'failure': denied, 'actual_wall_seconds': 0., 'actual_cpu_seconds': 0.,
                    'reference_wall_seconds': 0., 'reference_cpu_seconds': 0., 'origin_event': self.event['id']})
                raise Unresolved(denied)
            self.event[category] += 1
            self.counts[category] += 1
            start, start_cpu = time.perf_counter(), time.process_time()
            try:
                result = operation()
            except Exception as ex:
                wall, cpu = time.perf_counter() - start, time.process_time() - start_cpu
                self.event['primitive_requests'].append({'key': key, 'category': category, 'cache_hit': False,
                    'failure': type(ex).__name__ + ':' + str(ex), 'actual_wall_seconds': wall, 'actual_cpu_seconds': cpu,
                    'reference_wall_seconds': wall, 'reference_cpu_seconds': cpu, 'origin_event': self.event['id']})
                raise
            wall, cpu = time.perf_counter() - start, time.process_time() - start_cpu
            previous = {'result': result, 'wall_seconds': wall, 'cpu_seconds': cpu,
                        'category': category, 'origin_event': self.event['id']}
            self.primitives.setdefault(key, previous)
        self.event['primitive_requests'].append({'key': key, 'category': category, 'cache_hit': cached,
            'actual_wall_seconds': wall, 'actual_cpu_seconds': cpu,
            'reference_wall_seconds': previous['wall_seconds'], 'reference_cpu_seconds': previous['cpu_seconds'],
            'origin_event': previous['origin_event']})
        return result

    def numeric(self, matrix, vectors=False):
        try: matrix = np.asarray(matrix, dtype=np.float64)
        except (OverflowError, ValueError) as ex: raise Unresolved('UNRESOLVED_NUMERIC_CONVERSION') from ex
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] > 9 or not np.isfinite(matrix).all(): raise Unresolved('UNRESOLVED_NUMERIC_INPUT')
        key = ('eigh:' if vectors else 'eigvalsh:') + hashlib.sha256(matrix.tobytes()).hexdigest()
        def calculation():
            try: answer = np.linalg.eigh(matrix) if vectors else np.linalg.eigvalsh(matrix)
            except np.linalg.LinAlgError as ex: raise Unresolved('UNRESOLVED_EIGENSOLVER') from ex
            if not np.isfinite(answer[0] if vectors else answer).all(): raise Unresolved('UNRESOLVED_NONFINITE_PROPOSAL')
            return answer
        return self._primitive(key, 'eig', calculation)

    def psd(self, M, strict=False):
        a, n = self.a, len(M)
        if not n or any(len(r) != n for r in M) or any(M[i][j] != M[j][i] for i in range(n) for j in range(n)): raise ValueError('PSD_SCHEMA')
        M = [[a.q(x) for x in r] for r in M]
        key = 'psd:' + digest({'matrix': enc(M), 'strict': strict})
        def calculation():
            residual = [r[:] for r in M]
            L = identity(n)
            pivots = []
            status = 'PSD'
            for k in range(n):
                p = residual[k][k]
                pivots.append(p)
                if p < 0: status = 'NEGATIVE_PIVOT'; break
                if p == 0:
                    if strict: status = 'ZERO_PIVOT_NOT_SPD'; break
                    if any(residual[k][j] != 0 for j in range(k + 1, n)): status = 'ZERO_PIVOT_NONZERO_ROW'; break
                    continue
                for i in range(k + 1, n): L[i][k] = a.div(residual[i][k], p)
                for i in range(k + 1, n):
                    for j in range(i, n):
                        residual[i][j] = a.sub(residual[i][j], a.mul(a.mul(L[i][k], p), L[j][k]))
                        residual[j][i] = residual[i][j]
            record = {'id': key, 'matrix_sha256': digest(enc(M)), 'strict': strict,
                      'status': status, 'accepted': status == 'PSD', 'pivots': enc(pivots)}
            if status == 'PSD': record['L'] = enc(L)
            self.proofs[key] = record
            return record
        return self._primitive(key, 'psd', calculation)

    def inverse_quad(self, proof, v):
        a = self.a
        if not proof['accepted'] or not proof['strict']: raise ValueError('inverse needs strict PSD')
        L = [[dec(x) for x in r] for r in proof['L']]
        pivots = [dec(x) for x in proof['pivots']]
        y = []
        for i in range(len(v)): y.append(a.sub(v[i], a.total(a.mul(L[i][j], y[j]) for j in range(i))))
        return a.total(a.div(a.mul(x, x), d) for x, d in zip(y, pivots))

    def spectral(self, G, k, role, precomputed=None, required=None):
        a, n = self.a, len(G)
        if k == 0: return {'k': 0, 'lower': [0, 1], 'empty': True}, []
        if k == n:
            # trace identity is exact; no eigensolver/PSD expenditure.
            return {'k': k, 'lower': enc(a.total(G[i][i] for i in range(n))), 'full_trace': True}, []
        values, vectors = precomputed if precomputed is not None else self.numeric(G, vectors=True)
        tau_float = (values[k - 1] + values[k]) / 2
        E = max([F(1)] + [abs(x) for r in G for x in r])
        attempts = []
        chosen = None
        for bits in (24, 32, 40):
            tau = a.div(a.q(round(float(tau_float) * (1 << bits))), 1 << bits)
            Z_float = (vectors * np.maximum(float(tau) - values, 0)) @ vectors.T
            for factor in (1, 4):
                counter = 'occ_spectral_pairs' if role == 'OCC' else 'bound_spectral_pairs'
                self.counts[counter] += 1
                if self.counts[counter] > (180 if role == 'OCC' else 12): raise Unresolved('UNRESOLVED_SPECTRAL_PAIR_CAP')
                inflate = a.div(a.mul(a.mul(4 * n, E), factor), 1 << bits)
                Z = [[a.div(a.q(round(float((Z_float[i, j] + Z_float[j, i]) / 2) * (1 << bits))), 1 << bits) for j in range(n)] for i in range(n)]
                for i in range(n): Z[i][i] = a.add(Z[i][i], inflate)
                p = self.psd(Z)
                R = plus(shifted(G, tau, a), Z, a)
                q = self.psd(R)
                witness = {'k': k, 'tau': enc(tau), 'Z': enc(Z), 'lower': enc(a.sub(a.mul(k, tau), a.total(Z[i][i] for i in range(n)))),
                           'Z_proof': p['id'], 'residual_proof': q['id'], 'bits': bits, 'inflate_factor': factor,
                           'valid_spectral': p['accepted'] and q['accepted']}
                attempts.append(witness)
                if witness['valid_spectral']:
                    if chosen is None or dec(witness['lower']) > dec(chosen['lower']): chosen = witness
                    if role == 'OCC' and required is not None and dec(witness['lower']) >= required:
                        return witness, attempts
        return chosen, attempts


def diagonal(h, den, occ, a):
    occupied = set(occ)
    return a.total([h[i][i] for i in occ] + [x for (i, j), x in den.items() if i in occupied and j in occupied])


def upper_bound(h, den, active, nu, e):
    a = e.a
    dets = list(combinations(active, nu))
    if len(dets) > 256: raise ValueError('DETERMINANT_CAP')
    values = [diagonal(h, den, occ, a) for occ in dets]
    trials = [{'kind': 'determinant', 'occupations': [list(occ)], 'value': enc(x), 'relative_sign': 1} for occ, x in zip(dets, values)]
    pair_count = 0
    for i, j in combinations(range(len(dets)), 2):
        if pair_count == 128: break
        left, right = set(dets[i]) - set(dets[j]), set(dets[j]) - set(dets[i])
        if len(left) == len(right) == 1:
            p, q = next(iter(left)), next(iter(right))
            element = a.mul(h[p][q], (-1) ** (dets[i].index(p) + dets[j].index(q)))
            if element:
                pair_count += 1
                value = a.sub(a.div(a.add(values[i], values[j]), 2), abs(element))
                trials.append({'kind': 'pair', 'occupations': [list(dets[i]), list(dets[j])], 'relative_sign': -1 if element > 0 else 1,
                               'matrix_element': enc(element), 'value': enc(value)})
    chosen = min(range(len(trials)), key=lambda i: (dec(trials[i]['value']), i))
    return {'U': trials[chosen]['value'], 'selected': chosen, 'trials': trials, 'determinants': len(dets), 'pairs': pair_count}


def c0_bound(A, hdd, nu, e):
    a, m = e.a, len(A)
    r = min(a.sub(A[i][i], a.total(abs(A[i][j]) for j in range(m) if i != j)) for i in range(m))
    span = a.sub(max(A[i][i] for i in range(m)), min(A[i][i] for i in range(m)))
    old = []
    for rho in (F(0), F(1, 20), F(1, 10), F(3, 20), F(1, 5)):
        lam = a.add(r, a.mul(rho, span))
        proof = e.psd(shifted(A, lam, a), strict=True)
        old.append({'lambda': enc(lam), 'rho': enc(rho), 'proof': proof['id'], 'SPD': proof['accepted']})
    ell = max([r] + [dec(x['lambda']) for x in old if x['SPD']])
    refined, failure = [], None
    try:
        proposal = float(e.numeric(A)[0])
        E = max([F(1)] + [abs(x) for row in A for x in row])
        step = a.div(E, 1 << 24)
        floor = a.mul(math.floor(a.div(a.q(F(proposal)), step)), step)
        for offset in (1, 2, 4, 8, 16, 32):
            lower = a.sub(floor, a.mul(offset, step))
            proof = e.psd(shifted(A, lower, a))
            refined.append({'lambda': enc(lower), 'offset': offset, 'proof': proof['id'], 'valid': proof['accepted']})
            if proof['accepted']: ell = max(ell, lower)
    except Unresolved as ex: failure = str(ex)
    return {'c': enc(a.add(hdd, a.mul(nu - 1, ell))), 'ell': enc(ell), 'gershgorin': enc(r),
            'old_lambda': old, 'refined': refined, 'proposal_failure': failure}


def spectral_bound(G, k, hdd, e):
    witness, attempts = e.spectral(G, k, 'BOUND')
    if witness is None: return {'c': None, 'witness': None, 'attempts': attempts, 'status': 'UNRESOLVED_SPECTRAL_PROPOSAL'}
    return {'c': enc(e.a.add(hdd, dec(witness['lower']))), 'witness': witness, 'attempts': attempts, 'status': 'CERTIFIED_LOWER'}


def density_bound(h, den, active, ext, nu, e):
    occs = list(combinations(active, nu - 1))
    if len(occs) > 256: raise ValueError('EXTERNAL_DIAGONAL_CAP')
    rows = [{'occupation': list(occ), 'energy': enc(diagonal(h, den, (*occ, ext), e.a))} for occ in occs]
    minimum = min(dec(x['energy']) for x in rows)
    O = [[h[i][j] if i != j else F(0) for j in active] for i in active]
    lower = spectral_bound(O, nu - 1, minimum, e)
    return dict(lower, diagonal_minimum=enc(minimum), diagonal_rows=rows)


def binding(case_sha, source, U, c, delta, row):
    return {'input_sha256': case_sha, 'source_fingerprint': source, 'U': enc(U), 'c': enc(c), 'delta': enc(delta), 'row': row}


def margins(w, A, v, nu, U, c, delta, e):
    a = e.a
    D = a.add(a.sub(c, U), delta)
    k = dec(w['kappa']) if w['kind'] != 'norm' else F(0)
    if w['kind'] == 'norm':
        beta = a.total(a.mul(x, x) for x in v)
        margin = a.sub(a.mul(delta, D), beta)
        psd_valid = True
    elif w['kind'] == 'old':
        mu = dec(w['mu'])
        margin = a.sub(a.mul(delta, D), a.sub(a.mul(k, U), a.mul(nu, mu)))
        psd_valid = e.proofs[w['proof']]['accepted']
    else:
        margin = a.add(a.sub(dec(w['spectral']['lower']), a.mul(k, U)), a.mul(delta, D))
        s = w['spectral']
        psd_valid = bool(s.get('empty') or s.get('full_trace') or (e.proofs[s['Z_proof']]['accepted'] and e.proofs[s['residual_proof']]['accepted']))
    return {'D': enc(D), 'affine_margin': enc(margin), 'PSD_valid': psd_valid,
            'accepted': bool(c > U and 0 <= k <= D and margin >= 0 and psd_valid)}


def old_candidate(k, mu, A, v, nu, U, c, delta, e, bind):
    a = e.a
    M = plus(shifted(scaled(A, k, a), mu, a), scaled(outer(v, a), -1, a), a)
    proof = e.psd(M)
    w = {'kind': 'old', 'kappa': enc(k), 'mu': enc(mu), 'proof': proof['id'], 'binding': bind}
    w.update(margins(w, A, v, nu, U, c, delta, e))
    return w


def search(A, v, nu, U, c, delta, kind, e):
    a = e.a
    D = a.add(a.sub(c, U), delta)
    try:
        An = np.asarray(A, float)
        vv = np.outer(np.asarray(v, float), np.asarray(v, float))
    except (OverflowError, ValueError) as ex: raise Unresolved('UNRESOLVED_NUMERIC_CONVERSION') from ex
    eye = np.eye(len(A))
    delta_D = a.mul(delta, D)
    normalization = max(1., float(np.max(np.abs(An))), float(np.max(np.abs(vv))), abs(float(delta_D)))
    old_base = (float(a.div(delta_D, nu)) * eye - vv) / normalization
    old_slope = float(D) * (An - float(a.div(U, nu)) * eye) / normalization
    occ_base, occ_slope = -vv / normalization, float(D) * An / normalization
    best, calls, values = None, 0, {}
    def objective(x):
        nonlocal best, calls
        calls += 1
        if calls > 128: raise Unresolved('UNRESOLVED_SEARCH_CALL_CAP')
        if x not in values:
            matrix = old_base + x * old_slope if kind == 'OLD_DIRECT' else occ_base + x * occ_slope
            eigs = e.numeric(matrix)
            score = float(eigs[0]) if kind == 'OLD_DIRECT' else float(np.sum(eigs[:nu]) - (x * float(D) * float(U) - float(delta_D)) / normalization)
            if not math.isfinite(score): raise Unresolved('UNRESOLVED_NONFINITE_SCORE')
            values[x] = score
        score = values[x]
        if best is None or (score, -x) > (best[1], -best[0]): best = (x, score)
        return score
    objective(0.); objective(1.)
    left, right = 0., 1.
    phi = (math.sqrt(5) - 1) / 2
    x1, x2 = right - phi * (right - left), left + phi * (right - left)
    f1, f2 = objective(x1), objective(x2)
    while right - left > 2 ** -40 and calls < 128:
        if f1 < f2:
            left = x1; x1 = x2; f1 = f2; x2 = left + phi * (right - left); f2 = objective(x2)
        else:
            right = x2; x2 = x1; f2 = f1; x1 = right - phi * (right - left); f1 = objective(x1)
    return {'x': best[0], 'score': best[1], 'calls': calls, 'bracket': [left, right], 'role': 'proposal_only'}


def dyadic_candidates(x, a):
    x = a.q(F(x))
    values = {F(0), F(1)}
    for bits in (16, 24, 32, 40):
        v = a.mul(x, 1 << bits)
        values.add(a.div(math.floor(v), 1 << bits)); values.add(a.div(math.ceil(v), 1 << bits))
    return sorted(values)


def generate_method(kind, A, v, nu, U, c, delta, c0, e, bind):
    a = e.a
    attempts, proposal = [], None
    if kind == 'NORM':
        w = {'kind': 'norm', 'binding': bind}
        w.update(margins(w, A, v, nu, U, c, delta, e)); attempts.append(w)
    elif kind == 'FIVE':
        for old in c0['old_lambda']:
            if old['SPD']:
                try:
                    k = e.inverse_quad(e.proofs[old['proof']], v)
                    attempts.append(old_candidate(k, a.mul(k, dec(old['lambda'])), A, v, nu, U, c, delta, e, bind))
                except Unresolved as ex:
                    return method_result(attempts, proposal, c, U, kind, str(ex))
    elif c <= U:
        return {'accepted': False, 'status': 'BOUND_PRECONDITION_FAILED', 'attempts': [], 'proposal': None, 'witness': None}
    else:
        proposal = search(A, v, nu, U, c, delta, kind, e)
        D = a.add(a.sub(c, U), delta)
        for x in dyadic_candidates(proposal['x'], a):
            try:
                k = a.mul(D, x)
                if kind == 'OLD_DIRECT':
                    mu = a.div(a.sub(a.mul(k, U), a.mul(delta, D)), nu)
                    attempts.append(old_candidate(k, mu, A, v, nu, U, c, delta, e, bind))
                else:
                    K = plus(scaled(A, k, a), scaled(outer(v, a), -1, a), a)
                    required = a.sub(a.mul(k, U), a.mul(delta, D))
                    spec, spectral_attempts = e.spectral(K, nu, 'OCC', required=required)
                    entry = {'kind': 'occ', 'kappa': enc(k), 'binding': bind, 'spectral': spec,
                             'spectral_attempts': spectral_attempts, 'x': enc(x)}
                    if spec is None:
                        entry.update(accepted=False, status='UNRESOLVED_SPECTRAL_PROPOSAL')
                    else: entry.update(margins(entry, A, v, nu, U, c, delta, e))
                    attempts.append(entry)
                    if entry['accepted']: break
            except Unresolved as ex:
                return method_result(attempts, proposal, c, U, kind, str(ex))
    return method_result(attempts, proposal, c, U, kind)


def method_result(attempts, proposal, c, U, kind, failure=None):
    selected = next((i for i, x in enumerate(attempts) if x['accepted']), None)
    success = selected is not None
    status = 'CERTIFIED' if success else 'BOUND_PRECONDITION_FAILED' if c <= U else 'NORM_CONDITION_FALSE' if kind == 'NORM' else 'UNRESOLVED_FINITE_SEARCH'
    if failure and not success: status = failure
    return {'accepted': success, 'status': status, 'attempts': attempts, 'proposal': proposal, 'partial_failure': failure,
            'witness': attempts[selected] if success else None}


def transport(w, target_binding, A, v, nu, U, c, delta, e, origin):
    # Same kappa, mu/tau/Z and PSD proof; no numerical reproposal.
    copy = dict(w, binding=target_binding, origin=origin)
    copy.update(margins(copy, A, v, nu, U, c, delta, e))
    return copy


def operational(rows, A, v, nu, U, delta, e, case_sha, source):
    results = []
    for row_index, row in enumerate(rows):
        if row['c'] is None:
            results.append({'row': row['row'], 'baseline_star': False, 'baseline_certificates': [],
                            'same_method_inherited': [], 'fallback': [], 'occ_operational': False,
                            'raw_occ_generated': False, 'occ_specific_gain': False})
            continue
        c = dec(row['c'])
        bind = binding(case_sha, source, U, c, delta, row['row'])
        inherited = []
        for origin in rows[:row_index + 1]:
            if origin['c'] is None or dec(origin['c']) > c: continue
            for method in origin['methods']:
                for j, w in enumerate(method['attempts']):
                    if w.get('PSD_valid') and w.get('spectral', True) is not None:
                        event_id = '.'.join(('TRANSPORT', row['row'], origin['row'], method['method'], str(j)))
                        with e.stage(event_id, 'TRANSPORT', (method['event_id'],)):
                            moved = transport(w, bind, A, v, nu, U, c, delta, e, {'row': origin['row'], 'method': method['method'], 'attempt': j})
                            moved['event_id'] = event_id
                        if moved['accepted']: inherited.append(moved)
        baseline = [w for w in inherited if w['origin']['method'] != 'OCC']
        same_occ = [w for w in inherited if w['origin']['method'] == 'OCC']
        fallback = []
        for w in baseline:
            event_id = w['event_id'].replace('TRANSPORT.', 'FALLBACK.', 1)
            with e.stage(event_id, 'FALLBACK', (w['event_id'],)):
                if w['kind'] == 'norm':
                    Z, tau, k = outer(v, e.a), F(0), F(0)
                    lower = e.a.neg(e.a.total(e.a.mul(x, x) for x in v))
                    p = e.psd(Z); q = e.psd(scaled(identity(len(A)), 0, e.a))
                else:
                    Z, tau, k = scaled(identity(len(A)), 0, e.a), dec(w['mu']), dec(w['kappa'])
                    lower = e.a.mul(nu, tau)
                    p = e.psd(Z); q = e.proofs[w['proof']]
                spectral = {'k': nu, 'tau': enc(tau), 'Z': enc(Z), 'lower': enc(lower), 'Z_proof': p['id'], 'residual_proof': q['id'], 'valid_spectral': True}
                embedded = {'kind': 'occ', 'kappa': enc(k), 'spectral': spectral, 'binding': bind}
                embedded.update(margins(embedded, A, v, nu, U, c, delta, e))
                if not embedded['accepted']: raise AssertionError('SOUNDNESS_STOP_FALLBACK')
                fallback.append({'origin': w['origin'], 'certificate': w, 'embedded_occ': embedded, 'event_id': event_id,
                                 'embedding': 'kappa0_tau0_Zvv' if w['kind'] == 'norm' else 'same_kappa_tau_mu_Zzero'})
        raw_occ = next(m['accepted'] for m in row['methods'] if m['method'] == 'OCC')
        results.append({'row': row['row'], 'baseline_star': bool(baseline), 'baseline_certificates': baseline,
                        'same_method_inherited': same_occ, 'fallback': fallback,
                        'occ_operational': bool(raw_occ or same_occ or fallback),
                        'raw_occ_generated': raw_occ, 'occ_specific_gain': bool(raw_occ and not baseline)})
    return results


def reconstructed_cost(e, requested_events):
    by_id = {x['id']: x for x in e.events}
    needed = set()
    def visit(key):
        if key in needed: return
        needed.add(key)
        for dep in by_id[key]['dependencies']: visit(dep)
    for key in requested_events: visit(key)
    primitives, wall, cpu = set(), 0., 0.
    for event in e.events:
        if event['id'] not in needed: continue
        wall += event['overhead_wall_seconds']; cpu += event['overhead_cpu_seconds']
        for r in event['primitive_requests']:
            if r['key'] not in primitives:
                primitives.add(r['key']); wall += r['reference_wall_seconds']; cpu += r['reference_cpu_seconds']
    return {'event_ids': sorted(needed), 'unique_primitives': len(primitives),
            'reconstructed_wall_seconds': wall, 'reconstructed_cpu_seconds': cpu,
            'qualification': 'path reconstruction from factor-event overhead and first observed primitive costs; not a measured cold standalone run'}


def _generate_case(case, source_fingerprint='synthetic', cached=True, bit_cap=RATIONAL_BITS, engine=None):
    start, cpu = time.perf_counter(), time.process_time()
    e = engine if engine is not None else Engine(bit_cap, cached)
    a = e.a
    h, den, active, ext, nu, delta = decode_case(case, a)
    A, v = [[h[i][j] for j in active] for i in active], [h[i][ext] for i in active]
    case_sha = digest(case)
    with e.stage('U', 'U'): upper = upper_bound(h, den, active, nu, e)
    U = dec(upper['U'])
    with e.stage('C0', 'C0'): b0 = c0_bound(A, h[ext][ext], nu, e)
    extdiag = [den.get(tuple(sorted((i, ext))), F(0)) for i in active]
    Gext = [[a.add(A[i][j], extdiag[i]) if i == j else A[i][j] for j in range(len(A))] for i in range(len(A))]
    with e.stage('COCC', 'COCC'):
        try: bo = spectral_bound(Gext, nu - 1, h[ext][ext], e)
        except Unresolved as ex: bo = {'c': None, 'witness': None, 'attempts': [], 'status': str(ex)}
    with e.stage('CDH', 'CDH'):
        try: bd = density_bound(h, den, active, ext, nu, e)
        except Unresolved as ex: bd = {'c': None, 'witness': None, 'attempts': [], 'status': str(ex)}
    star = max(dec(x['c']) for x in (b0, bo, bd) if x['c'] is not None)
    bounds = {'c0': b0, 'c_occ': bo, 'c_dh': bd, 'c_star': enc(star),
              'available_star_components': [k for k, x in zip(('c0', 'c_occ', 'c_dh'), (b0, bo, bd)) if x['c'] is not None]}
    raw_rows = []
    c_values = (b0['c'], bo['c'], enc(star))
    for row_name, cv in zip(ROWS, c_values):
        methods = []
        for method in METHODS:
            eid = row_name + '.' + method
            deps = ['U', 'C0'] + (['COCC'] if row_name != 'c0' else []) + (['CDH'] if row_name == 'c_star' else [])
            with e.stage(eid, method, deps):
                if cv is None: result = {'accepted': False, 'status': 'UNRESOLVED_LOWER_DEPENDENCY', 'attempts': [], 'witness': None, 'proposal': None}
                else:
                    c = dec(cv)
                    try: result = generate_method(method, A, v, nu, U, c, delta, b0, e, binding(case_sha, source_fingerprint, U, c, delta, row_name))
                    except Unresolved as ex: result = {'accepted': False, 'status': str(ex), 'attempts': [], 'witness': None, 'proposal': None}
            result.update(method=method, event_id=eid)
            methods.append(result)
        raw_rows.append({'row': row_name, 'c': cv, 'U': enc(U), 'delta': enc(delta), 'methods': methods,
                         'raw_base_or': any(m['accepted'] for m in methods if m['method'] != 'OCC')})
    operations = operational(raw_rows, A, v, nu, U, delta, e, case_sha, source_fingerprint)
    # Simulate information available on each standalone path. Factor-only future
    # candidates are never free; every transported witness retains its producer.
    paths = {}
    for label, use_occ in (('BASE_OR', False), ('OCC_OPERATIONAL', True)):
        event_ids = ['U']
        available = set()
        for row, op in zip(raw_rows, operations):
            accepted = False
            event_ids.extend(['C0'] + (['COCC'] if row['row'] != 'c0' else []) + (['CDH'] if row['row'] == 'c_star' else []))
            moved = list(op.get('baseline_certificates', [])) + (list(op.get('same_method_inherited', [])) if use_occ else [])
            reusable = next((w for w in moved if (w['origin']['row'], w['origin']['method'], w['origin']['attempt']) in available), None)
            if reusable:
                event_ids.append(reusable['event_id'])
                if use_occ and reusable['kind'] != 'occ': event_ids.append(reusable['event_id'].replace('TRANSPORT.', 'FALLBACK.', 1))
                break
            for method in row['methods']:
                if method['method'] == 'OCC' and not use_occ: continue
                event_ids.append(method['event_id'])
                available.update((row['row'], method['method'], j) for j, w in enumerate(method['attempts']) if w.get('PSD_valid'))
                if method['accepted']:
                    if use_occ and method['method'] != 'OCC':
                        match = next(f for f in op['fallback'] if f['origin']['row'] == row['row'] and f['origin']['method'] == method['method'])
                        event_ids.append(match['event_id'])
                    accepted = True; break
            if accepted: break
        paths[label] = reconstructed_cost(e, event_ids)
    return {'schema_version': 1, 'input_id': case['id'], 'base': case['base'], 'case_sha256': case_sha,
            'source_fingerprint': source_fingerprint, 'upper': upper, 'bounds': bounds, 'raw_rows': raw_rows,
            'operational_rows': operations, 'proof_pool': e.proofs, 'events': e.events, 'standalone_reconstructed_cost': paths,
            'resources': {'counts': e.counts, 'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
                          'max_monitored_rational_bits': a.max_bits, 'rational_operations': a.operations,
                          'bit_monitor_scope': 'explicit rational operands/results, not Fraction internal integer temporaries',
                          'worker_peak_rss_bytes_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          'matrix_dimension': len(A)},
            'truth_calls': 0, 'scientific_complete': True, 'next_scientific_stage_authorized': False}


def generate_case(case, source_fingerprint='synthetic', cached=True, bit_cap=RATIONAL_BITS):
    start, cpu = time.perf_counter(), time.process_time()
    engine = Engine(bit_cap, cached)
    try:
        return _generate_case(case, source_fingerprint, cached, bit_cap, engine)
    except Unresolved as ex:
        rows = [{'row': name, 'c': None, 'U': None, 'delta': None, 'raw_base_or': False,
                 'methods': [{'method': method, 'accepted': False, 'status': str(ex), 'attempts': [], 'proposal': None, 'witness': None} for method in METHODS]} for name in ROWS]
        return {'schema_version': 1, 'input_id': case['id'], 'base': case['base'], 'case_sha256': digest(case),
                'source_fingerprint': source_fingerprint, 'upper': None, 'bounds': None, 'raw_rows': rows,
                'operational_rows': [{'row': name, 'baseline_star': False, 'same_method_inherited': [], 'fallback': [], 'occ_operational': False,
                                      'raw_occ_generated': False, 'occ_specific_gain': False} for name in ROWS],
                'proof_pool': engine.proofs, 'events': engine.events, 'standalone_reconstructed_cost': {},
                'resources': {'counts': engine.counts, 'max_monitored_rational_bits': engine.a.max_bits,
                              'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu,
                              'worker_peak_rss_bytes_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                              'matrix_dimension': len(case['active'])},
                'truth_calls': 0, 'scientific_complete': False, 'case_unresolved_reason': str(ex),
                'next_scientific_stage_authorized': False}


def secondary_separator(case, record, primary_seal):
    e = Engine()
    a = e.a
    h, den, active, ext, nu, delta = decode_case(case, a)
    if digest(case) != record['case_sha256']: raise ValueError('SECONDARY_INPUT_BINDING')
    eligible = record['operational_rows'][-1]['occ_specific_gain']
    result = {'input_id': case['id'], 'primary_seal': primary_seal, 'eligible': eligible, 'separator': None,
              'case_sha256': record['case_sha256'], 'source_fingerprint': record['source_fingerprint'],
              'status': 'NOT_ELIGIBLE', 'proof_pool': {}, 'events': [], 'resources': {'counts': e.counts}}
    if not eligible: return result
    A, v = [[h[i][j] for j in active] for i in active], [h[i][ext] for i in active]
    U, c = dec(record['upper']['U']), dec(record['bounds']['c_star'])
    D = a.add(a.sub(c, U), delta)
    R = shifted(A, a.div(U, nu), a)
    M0 = plus(scaled(identity(len(A)), a.div(a.mul(delta, D), nu), a), scaled(outer(v, a), -1, a), a)
    proposal = record['raw_rows'][-1]['methods'][2].get('proposal')
    center = proposal['x'] if proposal else .5
    points = [0., 1., center] + [min(1., max(0., center + sign * 2 ** -bits)) for bits in (10, 20, 30, 40) for sign in (-1, 1)]
    vectors = []
    with e.stage('SECONDARY', 'SECONDARY'):
        try:
            for x in dict.fromkeys(points):
                mat = np.asarray(M0, float) + x * float(D) * np.asarray(R, float)
                _, numeric_vectors = e.numeric(mat, vectors=True)
                z = [a.div(round(float(t) * (1 << 40)), 1 << 40) for t in numeric_vectors[:, 0]]
                vectors.append((z, quadratic(R, z, a)))
            candidates = [(z, z, F(1)) for z, _ in vectors]
            for (z1, s1), (z2, s2) in combinations(vectors, 2):
                if s1 != s2:
                    t = a.div(a.neg(s2), a.sub(s1, s2))
                    if 0 <= t <= 1: candidates.append((z1, z2, t))
            valid_attempts = 0
            for z1, z2, t in candidates:
                W = plus(scaled(outer(z1, a), t, a), scaled(outer(z2, a), a.sub(1, t), a), a)
                tr = a.total(W[i][i] for i in range(len(A)))
                intercept = a.total(a.mul(W[i][j], M0[j][i]) for i in range(len(A)) for j in range(len(A)))
                slope = a.total(a.mul(W[i][j], R[j][i]) for i in range(len(A)) for j in range(len(A)))
                endpoint = a.add(intercept, a.mul(D, slope))
                if tr <= 0 or max(intercept, endpoint) >= 0: continue
                valid_attempts += 1
                if valid_attempts > 6: break
                proof = e.psd(W)
                if proof['accepted']:
                    result['separator'] = enc({'W': W, 'proof': proof['id'], 'endpoint_traces': [intercept, endpoint], 'trace': tr, 'U': U, 'c': c, 'delta': delta})
                    break
            result['status'] = 'EXACT_SEPARATOR' if result['separator'] else 'UNRESOLVED_FINITE_SEPARATOR_SEARCH'
        except Unresolved as ex: result['status'] = str(ex)
    result.update(proof_pool=e.proofs, events=e.events, resources={'counts': e.counts, 'max_monitored_rational_bits': a.max_bits})
    return result


def classify(records, secondary=()):
    gains = [r for r in records if r['operational_rows'][-1].get('occ_specific_gain')]
    bases = sorted({r['base'] for r in gains})
    by_id = {s['input_id']: s for s in secondary}
    separated = sorted({r['base'] for r in gains if by_id.get(r['input_id'], {}).get('status') == 'EXACT_SEPARATOR'
                        and dec(r['raw_rows'][-1]['methods'][0]['attempts'][0]['affine_margin']) < 0})
    status = 'NO_ADDITIONAL_ACCEPTANCE_OBSERVED' if not bases else 'SINGLE_BASE_GAIN_GATE_NOT_MET' if len(bases) == 1 else 'FINITE_PIPELINE_TRANSFER_GAIN'
    reasons = {}
    for r in records:
        for row in r['raw_rows']:
            for method in row['methods']:
                if not method['accepted']: reasons[method['status']] = reasons.get(method['status'], 0) + 1
    return {'status': status, 'family_separation': len(separated) >= 2,
            'family_separation_label': 'CERTIFIED_FAMILY_TRANSFER_SEPARATION' if len(separated) >= 2 else None,
            'gain_bases': bases, 'gain_cases': [r['input_id'] for r in gains], 'separator_bases_with_norm_false': separated,
            'gate_two_bases_met': len(bases) >= 2, 'unaccepted_reason_counts': reasons,
            'comparison_opportunities': {
                'valid_strong_bound_inputs': sum(bool(r['scientific_complete'] and dec(r['bounds']['c_star']) > dec(r['upper']['U'])) for r in records),
                'baseline_certified_inputs': sum(r['operational_rows'][-1]['baseline_star'] for r in records),
                'raw_occ_certified_inputs': sum(r['operational_rows'][-1]['raw_occ_generated'] for r in records),
                'whole_case_unresolved_inputs': sum(not r['scientific_complete'] for r in records)},
            'finite_failure_proves_family_infeasible': False,
            'denominator_inputs': len(records), 'primary_records': len(records) * 12,
            'mandatory_stop': True, 'next_scientific_stage_authorized': False, 'central_theme': None}
