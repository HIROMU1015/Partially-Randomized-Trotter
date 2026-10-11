"""Exact finite-witness audit for the occupation-aware N3 redesign.

No proposal solver, ground-state calculation, science runner or circuit library.
Review witnesses are data, never imported code. Not a transfer generator.
"""
from fractions import Fraction as F
from itertools import combinations
from functools import lru_cache
import hashlib
import json


CASE_SHA = 'b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16'
PACK_SHA = '9e99bda0d98ede7a5eea97002f36ab00378655884f35b0b0aec9af7692078d12'
LEDGER_SHA = '488c56f2cd8ffdf392dc6fc2912d65f54ed5eca00472f7b5067958229cad7fa7'


class Audit:
    def __init__(self, bit_cap=4096):
        self.bit_cap = bit_cap
        self.max_bits = 0
        self.checks = 0
        self.psd_checks = 0
        self.psd_cache_hits = 0
        self._psd_cache = {}

    def require(self, condition, reason):
        self.checks += 1
        if not condition:
            raise ValueError(reason)

    def q(self, value):
        value = F(value)
        self.max_bits = max(self.max_bits, abs(value.numerator).bit_length(), value.denominator.bit_length())
        if self.max_bits > self.bit_cap:
            raise ArithmeticError('UNRESOLVED_ARITHMETIC_CAP')
        return value

    def total(self, values):
        answer = F(0)
        for value in values:
            answer = self.q(answer + self.q(value))
        return answer

    def psd(self, matrix):
        """LDL from ORIGINAL entries, including exact reconstruction; zero tolerance."""
        n = len(matrix)
        self.require(n > 0 and all(len(row) == n for row in matrix), 'PSD shape')
        self.require(all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n)), 'PSD symmetry')
        key = tuple(tuple(self.q(x) for x in row) for row in matrix)
        if key in self._psd_cache:
            self.psd_cache_hits += 1
            return self._psd_cache[key]
        self.psd_checks += 1
        L = eye(n)
        pivots = []
        for k in range(n):
            pivot = self.q(matrix[k][k] - self.total(self.q(self.q(L[k][j] * L[k][j]) * pivots[j]) for j in range(k)))
            self.require(pivot >= 0, 'negative PSD pivot')
            pivots.append(pivot)
            for i in range(k + 1, n):
                residue = self.q(matrix[i][k] - self.total(self.q(self.q(L[i][j] * L[k][j]) * pivots[j]) for j in range(k)))
                if pivot == 0:
                    self.require(residue == 0, 'zero PSD pivot with nonzero row')
                    L[i][k] = F(0)
                else:
                    L[i][k] = self.q(residue / pivot)
        for i in range(n):
            for j in range(n):
                rebuilt = self.total(self.q(self.q(L[i][k] * L[j][k]) * pivots[k]) for k in range(n))
                self.require(rebuilt == matrix[i][j], 'LDL reconstruction mismatch')
        proof = {'pivots': [enc(p) for p in pivots], 'zero_pivots': pivots.count(F(0))}
        self._psd_cache[key] = proof
        return proof


def enc(value):
    if isinstance(value, F):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {k: enc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [enc(v) for v in value]
    return value


def rational(value, audit):
    audit.require(isinstance(value, list) and len(value) == 2 and all(type(x) is int for x in value) and value[1] > 0, 'rational schema')
    return audit.q(F(*value))


def matrix(value, audit):
    audit.require(isinstance(value, list) and len(value) > 0 and all(isinstance(row, list) and len(row) == len(value) for row in value), 'matrix shape')
    answer = [[rational(x, audit) for x in row] for row in value]
    audit.require(all(answer[i][j] == answer[j][i] for i in range(len(answer)) for j in range(len(answer))), 'matrix symmetry')
    return answer


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def affine(A, scale, B=None):
    return [[scale * x + (B[i][j] if B is not None else 0) for j, x in enumerate(row)] for i, row in enumerate(A)]


def shift(A, z):
    return [[x - (z if i == j else 0) for j, x in enumerate(row)] for i, row in enumerate(A)]


def outer(v):
    return [[x * y for y in v] for x in v]


def quadratic(A, v, audit):
    return audit.total(audit.q(audit.q(v[i] * A[i][j]) * v[j]) for i in range(len(v)) for j in range(len(v)))


def spectral_lower(G, k, witness, audit):
    audit.require(type(k) is int and 0 <= k <= len(G) and type(witness['k']) is int and witness['k'] == k, 'spectral particle count')
    if k == 0:
        audit.require(rational(witness['lower'], audit) == 0, 'empty spectral sum')
        return F(0)
    Z = matrix(witness['Q'], audit)
    audit.require(len(Z) == len(G), 'spectral matrix dimension')
    tau = rational(witness['tau'], audit)
    audit.psd(Z)
    audit.psd(affine(shift(G, tau), 1, Z))
    lower = audit.q(k * tau - audit.total(Z[i][i] for i in range(len(G))))
    audit.require(lower == rational(witness['lower'], audit), 'spectral trace bound')
    return lower


def decode_case(case, audit):
    allowed = {'id', 'base', 'modes', 'particles', 'active', 'external', 'scale', 'amplitude', 'direction', 'h', 'density'}
    audit.require(set(case) == allowed, 'unknown or oracle case field')
    h = matrix(case['h'], audit)
    active, d, nu = case['active'], case['external'], case['particles']
    n = len(h)
    audit.require(type(case['modes']) is int and case['modes'] == n, 'modes schema')
    audit.require(type(d) is int and 0 <= d < n, 'external index')
    audit.require(isinstance(active, list) and all(type(i) is int for i in active) and active == [i for i in range(n) if i != d], 'ordered active partition')
    audit.require(type(nu) is int and 1 <= nu <= len(active), 'particle count')
    density = {}
    for row in case['density']:
        audit.require(set(row) == {'i', 'j', 'coefficient'}, 'density schema')
        i, j = row['i'], row['j']
        audit.require(type(i) is int and type(j) is int and 0 <= i < n and 0 <= j < n and i != j, 'density index')
        key = tuple(sorted((i, j)))
        audit.require(key not in density, 'duplicate unordered density pair')
        x = rational(row['coefficient'], audit)
        audit.require(x >= 0, 'negative density outside proved class')
        density[key] = x
    return h, active, d, nu, density


def diagonal(h, density, occupied, audit):
    occupied = set(occupied)
    return audit.total([h[i][i] for i in occupied] + [x for (i, j), x in density.items() if i in occupied and j in occupied])


def check_saved(root):
    intake = root / 'artifacts/n3_occupation_redesign_preparation/2026-10-12/review_intake'
    audit = Audit()
    receipt = json.loads((intake.parent / 'intake_receipt.json').read_text())
    for item in receipt['appendix_files']:
        data = (intake / item['file']).read_bytes()
        audit.require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], 'intake file identity')
    review_path = intake / 'n13_transfer_recovery_scientific_review_2026-10-12.md'
    audit.require(hashlib.sha256(review_path.read_bytes()).hexdigest() == receipt['review_sha256'], 'review identity')
    pack_path = root / 'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json'
    ledger_path = root / 'artifacts/n3_direction_generator_v1/2026-10-11/run1/certificate_ledger.json'
    audit.require(hashlib.sha256(pack_path.read_bytes()).hexdigest() == PACK_SHA, 'frozen pack identity')
    audit.require(hashlib.sha256(ledger_path.read_bytes()).hexdigest() == LEDGER_SHA, 'frozen ledger identity')
    particle = json.loads((intake / 'posthoc_particle_certificate.json').read_text())
    stronger = json.loads((intake / 'density_bound_countercheck.json').read_text())
    case = particle['reconstructed_input']
    canonical = json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    audit.require(hashlib.sha256(canonical).hexdigest() == CASE_SHA == particle['input_case_sha256'] == stronger['input_case_sha256'], 'T3 case identity')
    pack_cases = json.loads(pack_path.read_text())['cases']
    audit.require(case in pack_cases, 'case differs from frozen input pack')
    ledger = json.loads(ledger_path.read_text())
    matching = [row for row in ledger if row['coefficient_input_sha256'] == CASE_SHA]
    audit.require(len(matching) == 1, 'T3 ledger binding')
    h, active, d, nu, density = decode_case(case, audit)
    m = len(active)
    A = [[h[i][j] for j in active] for i in active]
    v = [h[i][d] for i in active]
    U, delta = rational(particle['U'], audit), rational(particle['delta'], audit)
    audit.require(U == F(-4, 5) and delta == F(1, 250), 'fixed U/delta')
    audit.require(U == rational(matching[0]['bounds']['U'], audit), 'same frozen U')
    trial_values = [(occ, diagonal(h, density, occ, audit)) for occ in combinations(active, nu)]
    trial_occ, trial_energy = min(trial_values, key=lambda row: row[1])
    audit.require(trial_energy <= U, 'missing coefficient trial upper witness')
    Gext = [[A[i][j] + (density.get(tuple(sorted((active[i], d))), F(0)) if i == j else 0) for j in range(m)] for i in range(m)]
    c = audit.q(h[d][d] + spectral_lower(Gext, nu - 1, particle['external_bound_witness'], audit))
    audit.require(c == rational(particle['c_occupancy_certificate'], audit) and c > U, 'external occupation lower')
    D = audit.q(c - U + delta)
    kappa = rational(particle['kappa'], audit)
    audit.require(D == rational(particle['D'], audit) and 0 <= kappa <= D, 'Schur parameter domain')
    x = rational(particle['x'], audit)
    audit.require(kappa == audit.q(x * D), 'normalized parameter binding')
    K = affine(A, kappa, affine(outer(v), -1))
    low = spectral_lower(K, nu, particle['active_spectral_witness'], audit)
    margin = audit.q(low - audit.q(kappa * U) + audit.q(delta * D))
    audit.require(margin == rational(particle['affine_margin'], audit) and margin > 0, 'occupation condition')
    beta2 = audit.total(audit.q(t * t) for t in v)
    norm_margin = audit.q(delta * D - beta2)
    audit.require(norm_margin == rational(particle['norm_only_same_c_margin'], audit) and norm_margin < 0, 'same-c norm rejection')
    sep = particle['original_one_body_family_infeasibility_witness']
    zs = [[rational(x, audit) for x in sep[key]] for key in ('u', 'v')]
    t = rational(sep['weight'], audit)
    audit.require(all(len(z) == m for z in zs) and 0 <= t <= 1, 'PSD separating mixture')
    W = affine(outer(zs[0]), t, affine(outer(zs[1]), 1 - t))
    audit.psd(W)
    R = shift(A, U / nu)
    M0 = affine(eye(m), delta * D / nu, affine(outer(v), -1))
    slope = audit.q(t * quadratic(R, zs[0], audit) + (1 - t) * quadratic(R, zs[1], audit))
    intercept = audit.q(t * quadratic(M0, zs[0], audit) + (1 - t) * quadratic(M0, zs[1], audit))
    trace_W = audit.total(W[i][i] for i in range(m))
    audit.require(slope == rational(sep['slope_exact'], audit) == 0 and intercept == rational(sep['intercept'], audit) < 0 and trace_W == rational(sep['trace_W'], audit) > 0, 'old-family separating proof')
    audit.require(audit.q(intercept / trace_W) == rational(sep['normalized_intercept'], audit), 'normalized separator')
    audit.require(U == rational(stronger['U'], audit) and delta == rational(stronger['delta'], audit) and stronger['external_particle_number'] == nu - 1, 'stronger-control common task')
    rows = [(occ, diagonal(h, density, (*occ, d), audit)) for occ in combinations(active, nu - 1)]
    saved_rows = [(tuple(row['occupied_active']), rational(row['energy'], audit)) for row in stronger['diagonal_candidates']]
    audit.require(rows == saved_rows and len(rows) == 21, 'complete external diagonal enumeration')
    diagonal_min = min(energy for _, energy in rows)
    audit.require(diagonal_min == rational(stronger['diagonal_minimum'], audit) == F(-1, 3), 'external diagonal lower')
    O = [[A[i][j] if i != j else F(0) for j in range(m)] for i in range(m)]
    c2 = audit.q(diagonal_min + spectral_lower(O, nu - 1, stronger['offdiagonal_one_body_spectral_witness'], audit))
    margin2 = audit.q(delta * (c2 - U + delta) - beta2)
    audit.require(c2 == rational(stronger['c_diagonal_plus_hopping'], audit) and c2 > U and margin2 == rational(stronger['norm_only_affine_margin'], audit) > 0 and beta2 == rational(stronger['beta_squared'], audit), 'stronger norm-only certificate')
    return {'status': 'EXACT_REVIEW_WITNESSES_PASS', 'case_sha256': CASE_SHA,
            'classification': 'development_review_witness_audit_not_new_transfer',
            'checks': audit.checks, 'psd_matrices': audit.psd_checks, 'psd_cache_hits': audit.psd_cache_hits,
            'max_monitored_rational_bits': audit.max_bits, 'bit_cap': audit.bit_cap,
            'bit_monitor_scope': 'decoded rationals, LDL arithmetic/reconstruction and explicit q/total calls; helper construction temporaries not all instrumented',
            'trial_upper': {'occupation': trial_occ, 'energy': enc(trial_energy)},
            'same_information': enc({'U': U, 'delta': delta, 'c_occ': c, 'new_margin': margin, 'norm_margin': norm_margin, 'old_LMI_separator_normalized': intercept / trace_W}),
            'stronger_information': enc({'c_density_hopping': c2, 'norm_margin': margin2}),
            'numerical_eigensolver_calls': 0, 'many_body_diagonalizations': 0, 'optimizer_calls': 0,
            'old_scientific_records_changed': False, 'mandatory_stop': True,
            'next_scientific_batch_authorized': False, 'central_theme': None}


@lru_cache(maxsize=16)
def basis(modes, particles):
    return tuple(sum(1 << i for i in occupied) for occupied in combinations(range(modes), particles))


def hop(state, i, j):
    if not (state >> j & 1):
        return None
    sign = (-1) ** ((state & ((1 << j) - 1)).bit_count())
    state ^= 1 << j
    if state >> i & 1:
        return None
    sign *= (-1) ** ((state & ((1 << i) - 1)).bit_count())
    return state | (1 << i), sign


def sector_matrix(h, density, particles):
    states = basis(len(h), particles)
    lookup = {state: i for i, state in enumerate(states)}
    M = [[F(0) for _ in states] for _ in states]
    for j, state in enumerate(states):
        for a in range(len(h)):
            for b in range(len(h)):
                action = hop(state, a, b)
                if action is not None:
                    target, sign = action
                    M[lookup[target]][j] += sign * h[a][b]
        M[j][j] += sum((x for (a, b), x in density.items() if state >> a & 1 and state >> b & 1), F(0))
    return M


def exact_car_fixture(external, particles):
    """Tiny identity fixture only; no energy spectrum or scientific acceptance."""
    active = [i for i in range(4) if i != external]
    h = [[F(i + 1, 7) if i == j else F(i + j + 1, 19) for j in range(4)] for i in range(4)]
    density = {(0, 1): F(1, 3), (1, 3): F(2, 5)}
    full = sector_matrix(h, density, particles)
    states = basis(4, particles)
    p_indices = [i for i, s in enumerate(states) if not (s >> external & 1)]
    q_indices = [i for i, s in enumerate(states) if s >> external & 1]
    A = [[h[i][j] for j in active] for i in active]
    v = [h[i][external] for i in active]
    B = [[full[i][j] for j in q_indices] for i in p_indices]
    product = [[sum((B[i][k] * B[j][k] for k in range(len(q_indices))), F(0)) for j in range(len(p_indices))] for i in range(len(p_indices))]
    wanted = sector_matrix(outer(v), {}, particles)
    # Q basis is globally ordered; removal of d contributes this exact diagonal sign.
    q_signs = [(-1) ** sum(bool(states[j] >> i & 1) for i in active if i < external) for j in q_indices]
    external_h = [[A[i][j] + (density.get(tuple(sorted((active[i], external))), F(0)) if i == j else 0) for j in range(3)] for i in range(3)]
    active_density = {(active.index(i), active.index(j)): x for (i, j), x in density.items() if i in active and j in active}
    external_matrix = sector_matrix(external_h, active_density, particles - 1)
    wanted_C = affine(eye(len(q_indices)), h[external][external], external_matrix)
    actual_C = [[q_signs[i] * full[a][b] * q_signs[j] for j, b in enumerate(q_indices)] for i, a in enumerate(q_indices)]
    return product == wanted, actual_C == wanted_C
