"""Standard-library-only verification of saved post-hoc coefficient witnesses.

No numerical eigensolver, optimizer, many-body diagonalization, original science
module, or quantum circuit library is imported. The mathematical energy-error
implication is proved in the accompanying review; this checks its finite witnesses.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def scalar(x):
    if not isinstance(x, list) or len(x) != 2 or any(type(z) is not int for z in x):
        raise ValueError('Expected [integer numerator, integer denominator]')
    return F(x[0], x[1])

def matrix(x):
    a = [[scalar(z) for z in r] for r in x]
    if not a or any(len(r) != len(a) for r in a):
        raise ValueError('Expected nonempty square matrix')
    return a

def psd(a):
    n = len(a)
    assert all(len(r) == n for r in a)
    assert all(a[i][j] == a[j][i] for i in range(n) for j in range(n))
    b = [r[:] for r in a]
    for k in range(n):
        p = b[k][k]
        assert p >= 0
        if p == 0:
            assert all(b[k][j] == 0 for j in range(k + 1, n))
        else:
            for i in range(k + 1, n):
                for j in range(i, n):
                    b[i][j] -= b[i][k] * b[k][j] / p
                    b[j][i] = b[i][j]

def add(a, b): return [[x + y for x, y in zip(r, s)] for r, s in zip(a, b)]
def scale(a, x): return [[x * z for z in r] for r in a]
def eye(n): return [[F(i == j) for j in range(n)] for i in range(n)]
def outer(v): return [[x * y for y in v] for x in v]
def quad(a, v): return sum((v[i] * a[i][j] * v[j] for i in range(len(v)) for j in range(len(v))), F(0))

def check_spectral(g, cert, k):
    assert cert['k'] == k
    if k == 0:
        assert scalar(cert['lower']) == 0
        return F(0)
    q = matrix(cert['Q']); tau = scalar(cert['tau'])
    assert len(q) == len(g)
    psd(q); psd(add(add(g, scale(eye(len(g)), -tau)), q))
    lower = k * tau - sum((q[i][i] for i in range(len(q))), F(0))
    assert lower == scalar(cert['lower'])
    return lower

def main():
    p = json.loads((ROOT / 'posthoc_particle_certificate.json').read_text())
    b = json.loads((ROOT / 'density_bound_countercheck.json').read_text())
    case = p['reconstructed_input']
    digest = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    expected = 'b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16'
    assert digest == expected == p['input_case_sha256'] == b['input_case_sha256']
    h = matrix(case['h']); active = case['active']; d = case['external']; nu = case['particles']; m = len(active)
    assert (len(h), m, nu, d) == (8, 7, 3, 7)
    assert all(h[i][j] == h[j][i] for i in range(len(h)) for j in range(len(h)))
    density = [(r['i'], r['j'], scalar(r['coefficient'])) for r in case['density']]
    assert all(i != j and x >= 0 for i, j, x in density)
    a = [[h[i][j] for j in active] for i in active]; v = [h[i][d] for i in active]
    U = scalar(p['U']); delta = scalar(p['delta'])
    assert U == scalar(b['U']) and delta == scalar(b['delta']) and delta > 0
    def diagonal(occ):
        s = set(occ)
        return sum((h[i][i] for i in s), F(0)) + sum((x for i, j, x in density if i in s and j in s), F(0))
    assert min(diagonal(occ) for occ in combinations(active, nu)) <= U
    ext = [sum((x for i, j, x in density if {i, j} == {r, d}), F(0)) for r in active]
    gext = add(a, [[ext[i] if i == j else F(0) for j in range(m)] for i in range(m)])
    c = h[d][d] + check_spectral(gext, p['external_bound_witness'], nu - 1)
    assert c == scalar(p['c_occupancy_certificate']) and c > U
    D = c - U + delta; kappa = scalar(p['kappa'])
    assert D == scalar(p['D']) and 0 <= kappa <= D
    K = add(scale(a, kappa), scale(outer(v), -1))
    low = check_spectral(K, p['active_spectral_witness'], nu)
    margin = low - kappa * U + delta * D
    assert margin == scalar(p['affine_margin']) and margin > 0
    beta2 = sum((x*x for x in v), F(0))
    assert delta * D - beta2 == scalar(p['norm_only_same_c_margin']) < 0
    sep = p['original_one_body_family_infeasibility_witness']
    z1, z2 = [scalar(x) for x in sep['u']], [scalar(x) for x in sep['v']]
    t = scalar(sep['weight']); assert len(z1) == len(z2) == m and 0 <= t <= 1
    R = add(a, scale(eye(m), -U / nu))
    M0 = add(scale(eye(m), delta * D / nu), scale(outer(v), -1))
    slope = t * quad(R, z1) + (1 - t) * quad(R, z2)
    intercept = t * quad(M0, z1) + (1 - t) * quad(M0, z2)
    trW = t * sum((x*x for x in z1), F(0)) + (1-t) * sum((x*x for x in z2), F(0))
    assert slope == scalar(sep['slope_exact']) == 0
    assert intercept == scalar(sep['intercept']) < 0 and trW == scalar(sep['trace_W']) > 0
    assert intercept / trW == scalar(sep['normalized_intercept'])
    rows = list(combinations(active, nu - 1))
    expected_rows = {tuple(r['occupied_active']): scalar(r['energy']) for r in b['diagonal_candidates']}
    actual_rows = {occ: diagonal((*occ, d)) for occ in rows}
    assert expected_rows == actual_rows and len(rows) == 21
    diag_min = min(actual_rows.values()); assert diag_min == scalar(b['diagonal_minimum'])
    O = [[F(0) if i == j else a[i][j] for j in range(m)] for i in range(m)]
    c2 = diag_min + check_spectral(O, b['offdiagonal_one_body_spectral_witness'], nu - 1)
    margin2 = delta * (c2 - U + delta) - beta2
    assert c2 == scalar(b['c_diagonal_plus_hopping']) > U
    assert beta2 == scalar(b['beta_squared'])
    assert margin2 == scalar(b['norm_only_affine_margin']) > 0
    result = {'status': 'PASS', 'case_sha256': digest,
        'coefficient_trial_U_verified': True,
        'new_particle_spectral_certificate_verified': True,
        'old_LMI_infeasible_at_same_occupancy_bound_verified': True,
        'stronger_density_bound_norm_only_verified': True,
        'numerical_eigensolver_calls': 0, 'many_body_solves': 0,
        'registered_transfer_status_changed': False,
        'qualification': 'Exact finite-witness verification, not independent external scientific reproduction.'}
    (ROOT / 'saved_witness_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
