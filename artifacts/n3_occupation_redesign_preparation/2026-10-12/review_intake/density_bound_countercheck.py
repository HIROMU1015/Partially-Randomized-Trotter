"""Post-hoc stronger norm-only control for the SAME frozen T3 input.

No many-body eigensolve, ground state, or true gap is used. The external
sector has two particles, so its diagonal energy minimum is obtained by
21 coefficient-only pair evaluations. This is not asserted to scale
polynomially for arbitrary variable particle number.
"""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json
import review_checks as rc
from particle_number_certificate import spectral_lower, check_spectral

OUT = Path(__file__).resolve().parent

def main():
    case = rc.recipe(7, 3, F(2), 7, F(1, 4), F(1, 40), True)
    assert rc.sha(case) == 'b34f66cbeed196090dfd77ab999cb462db821ce12e9d21408020771f47a81f16'
    h, d, active = case['h'], case['external'], case['active']
    p = case['particles'] - 1
    diagonal_rows = []
    for occ in combinations(active, p):
        occupied = set(occ) | {d}
        energy = sum((h[i][i] for i in occupied), F(0))
        energy += sum((r['coefficient'] for r in case['density']
                       if r['i'] in occupied and r['j'] in occupied), F(0))
        diagonal_rows.append({'occupied_active': list(occ), 'energy': energy})
    dmin = min(r['energy'] for r in diagonal_rows)
    O = [[F(0) if i == j else h[i][j] for j in active] for i in active]
    spec = spectral_lower(O, p)
    check_spectral(O, spec)
    c = dmin + spec['lower']
    U = rc.trial_U(case)
    delta = rc.GOAL
    beta2 = sum((h[i][d] ** 2 for i in active), F(0))
    margin = delta * (c - U + delta) - beta2
    assert len(diagonal_rows) == 21
    assert dmin == F(-1, 3) and c > U and margin > 0
    record = {
        'classification': 'post_hoc_stronger_control_not_frozen_transfer_result',
        'case': case['id'], 'input_case_sha256': rc.sha(case),
        'U': U, 'delta': delta, 'external_particle_number': p,
        'diagonal_candidates': diagonal_rows, 'diagonal_minimum': dmin,
        'offdiagonal_one_body_spectral_witness': spec,
        'c_diagonal_plus_hopping': c, 'beta_squared': beta2,
        'norm_only_affine_margin': margin,
        'true_energy_or_gap_used': False,
        'conclusion': 'A stronger coefficient-only external bound allows norm-only to certify this same input. The new directional spectral family is therefore not indispensable for accepting this input.',
        'limitation': 'Exact diagonal minimization is polynomial here because external particle number equals two; no polynomial method for general particle number is claimed.'
    }
    (OUT / 'density_bound_countercheck.json').write_text(json.dumps(rc.enc(record), indent=2), encoding='utf-8')
    print(json.dumps({k: float(record[k]) for k in ['U', 'delta', 'diagonal_minimum', 'c_diagonal_plus_hopping', 'beta_squared', 'norm_only_affine_margin']}, indent=2))
    print('Coefficient-only stronger norm-control exact checks PASS; no many-body truth used.')

if __name__ == '__main__':
    main()
