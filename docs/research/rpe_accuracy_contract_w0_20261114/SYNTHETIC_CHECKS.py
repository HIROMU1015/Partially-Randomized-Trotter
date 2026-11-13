"""W0 artificial checks only: no saved raw/target/result, no MC, no H4 numbers.

Verifies statements used in REPORT.md against the UNCHANGED original decoder,
B's fixture (h=10, artificial loss [0,20]) and B's D.evaluate/_refine_states:

1. outside zero/tie histories, a total decoder (zero->hold, tie->scalar argmin)
   returns bit-identical energies to the original wrapper;
2. every decoder transition moves the estimate by <= pi/2^m (+ slack);
3. pointwise first-unsafe-round bound on the final phase error;
4. existing coarse/DP bounds also bound the total decoder's exact MSE, and the
   first-unsafe lemma bound (fallback to L_D near the principal edge) holds;
5. odd N on one axis makes X=Y=0 impossible.

total_decode() below is a verification oracle, NOT an adopted estimator and
not importable from src. Run: python -B <this file> [--out PATH].
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_rpe_deterministic_risk import case, D, S, a  # noqa: E402  (B's artificial fixture)

F = D.F
PI_LO = Fraction(math.pi)                          # binary64 pi < true pi
PI_UP = Fraction(math.nextafter(math.pi, math.inf))  # > true pi
TAU = Fraction(1, 10**9)                           # declared engineering slack (rad)
SOURCE = dict(scope='ARTIFICIAL_W0_CHECKS_ONLY',
              path='docs/research/rpe_accuracy_contract_w0_20261114/SYNTHETIC_CHECKS.py')


def circ_upper(x, y):
    """Upper bound of the real circular distance between two binary64 phases."""
    d = abs(Fraction(x) - Fraction(y))  # inputs lie in [-pi, pi], so d <= 2 pi
    return d if d <= PI_LO else min(d, 2*PI_UP - d)


def total_decode(coords, h):
    """Oracle: zero -> hold (m=0: +0.0), tolerance tie -> scalar lowest-ID argmin."""
    est = None; path = []; flags = []
    for m, (x, y) in enumerate(coords):
        zero = x == 0 and y == 0
        if m == 0:
            est = 0.0 if zero else float(F._wrap(math.atan2(y, x)))
            flags.append('ZERO_INIT' if zero else 'INIT'); path.append(est); continue
        if zero:
            flags.append('ZERO_HOLD'); path.append(est); continue
        principal = float(F._wrap(math.atan2(y, x)))
        value, _, best, runner, _ = F._nearest(np.array([principal]), np.array([est]), 1 << m)
        flags.append('TIE_ARGMIN' if abs(float(runner[0]-best[0])) <= D.TIE_TOLERANCE else 'DECODED')
        est = float(value[0]); path.append(est)
    return -est/float(h), path, flags


def lemma_radius(f, rho, M):
    """Real-arithmetic circular bound after first unsafe round f>=1 (upper pi)."""
    return Fraction(rho[f-1]) / 2**(f-1) + PI_UP*(Fraction(2, 2**f) - Fraction(1, 2**M))


def run_case(label, *, vector, raws, target=0.):
    M = len(vector)-1
    args = case(M=M, vectors=[tuple(vector)])
    for mean, raw in zip(args['signal_packet']['means'], raws):
        mean['raw'] = dict(real=raw[0], imag=raw[1])
    args['evaluation_target']['energy_ha'] = target
    args['options']['state_refinement'] = dict(source=SOURCE, maximum_frontier_states=10**6,
                                               maximum_transition_pairs=10**8)
    with ExitStack() as guards:
        for module, name in ((S, 'simulate'), (S, 'simulate_prepared'), (D.K, 'diagnose'),
                             (D.p, 'run_missing'), (D.p.SavedArtifacts, 'read')):
            guards.enter_context(patch.object(module, name,
                side_effect=AssertionError('MC/physical provider/saved data forbidden')))
        risk = D.evaluate(**args)[0]['diagnostic']['deterministic_risk']
    h = float(a.exact_scalar(args['config']['spec']['h'])); rho = args['options']['phase_radii']
    E = Fraction(target); theta = -Fraction(h)*E
    tables = [D.round_distribution(dict(m=m, raw=dict(real=r[0], imag=r[1]), N_X=n, N_Y=n),
              h=h, target_energy=target, radius=rho[m], probability_rule=D.EXACT_AFFINE,
              maximum_outcomes=10**6) for m, (n, r) in enumerate(zip(vector, raws))]
    support = [[(o, Fraction(o['weight'], t['denominator'])) for o in t['outcomes']] for t in tables]
    stats = dict(histories=0, decoded=0, zero=0, tie=0, zero_and_tie_path=0)
    mse_total = Fraction(0); max_excess = None; pointwise_ok = True; identical = True; mass = Fraction(0)
    edge_fallback_used = False
    LD = a.exact_scalar(risk['resolved_loss_upper']) if risk.get('resolved_loss_upper') else None
    # The good-prefix argument needs B's closure certificate (decoder == oracle on good rounds).
    cert_ok = (risk.get('decoder_certificate') or {}).get('status') == 'VERIFIED_FINITE_BINARY64_SUPPORT'
    for history in product(*support):
        p = math.prod(v[1] for v in history); mass += p; stats['histories'] += 1
        rows = [dict(m=m, X=dict(plus=o['X_plus'], shots=n), Y=dict(plus=o['Y_plus'], shots=n))
                for m, ((o, _), n) in enumerate(zip(history, vector))]
        original = S.decode_counts(rows, h=h, M=M, prior=[-1e3, 1e3])
        coords = [(S.count_mean(r['X']['plus'], r['X']['shots'])['mean'],
                   S.count_mean(r['Y']['plus'], r['Y']['shots'])['mean']) for r in rows]
        energy, path, flags = total_decode(coords, h)
        if original['status'] == S.DECODED:
            stats['decoded'] += 1
            identical &= original['energy_estimate_ha'] == energy
        elif original['status'] == 'UNRESOLVED_ZERO_COORDINATES':
            stats['zero'] += 1
            stats['zero_and_tie_path'] += 'TIE_ARGMIN' in flags
        else:
            stats['tie'] += 1
        for m in range(1, M+1):  # update-width lemma, every transition
            excess = circ_upper(path[m], path[m-1]) - PI_LO/2**m
            max_excess = excess if max_excess is None else max(max_excess, excess)
        good = [o['good'] for o, _ in history]
        f = next((m for m, g in enumerate(good) if not g), None)
        err = circ_upper(path[-1], theta)  # theta is exact rational; circ via same helper
        if cert_ok and f is None:
            pointwise_ok &= path[-1] == history[-1][0]['oracle_estimate']
        elif cert_ok and f >= 1:
            pointwise_ok &= err <= lemma_radius(f, rho, M) + TAU
        mse_total += p*(Fraction(energy) - E)**2
    assert mass == 1
    out = dict(label=label, vector=list(vector), raws=[list(r) for r in raws], target=target, h=h,
               certificate=(risk.get('decoder_certificate') or {}).get('status'), status=risk['status'],
               **stats, total_equals_original_on_decoded=bool(identical),
               max_transition_excess_over_pi_over_2m=float(max_excess) if max_excess is not None else None,
               transition_lemma_within_tau=max_excess is None or max_excess <= TAU,
               pointwise_first_unsafe_bound=bool(pointwise_ok), MSE_total_exact=float(mse_total))
    if risk['status'] != 'CONDITIONAL_BINARY64_DECODER_MODEL_BOUND':
        out['bound_checks'] = 'SKIPPED_' + str(risk.get('reason'))
        return out
    coarse = a.exact_scalar(risk['coarse_MSE_upper']); dp = risk['state_refinement']
    dp_lo, dp_hi = a.exact_scalar(dp['MSE_lower']), a.exact_scalar(dp['MSE_upper'])
    first = [a.exact_scalar(v['exact']) for v in risk['first_unsafe']]
    J = a.exact_scalar(risk['terminal_good_MSE_contribution'])
    lemma = J; naive = J  # naive = same caps WITHOUT the principal-edge guard (negative control)
    for f, pf in enumerate(first):
        if f == 0:
            lemma += pf*LD; naive += pf*LD; continue
        B = lemma_radius(f, rho, M) + TAU
        cap = min(((B/Fraction(h)) + Fraction(1, 10**12))**2, LD)
        naive += pf*cap
        if abs(theta) + B < PI_LO:   # no principal-edge crossing -> linear == circular
            lemma += pf*cap
        else:
            lemma += pf*LD; edge_fallback_used = edge_fallback_used or pf > 0
    complete = a.exact_scalar(dp['residual_probability']) == 0
    out.update(coarse_upper=float(coarse), dp_lower=float(dp_lo), dp_upper=float(dp_hi),
               lemma_total_decoder_upper=float(lemma), L_D=float(LD),
               L_D_fallback_used_for_f1_or_edge=edge_fallback_used,
               dp_lower_le_total=dp_lo <= mse_total, total_le_dp_upper=mse_total <= dp_hi,
               total_le_coarse=mse_total <= coarse, total_le_lemma=mse_total <= lemma,
               diagnostic_naive_edge_unguarded_bound_violated=mse_total > naive,
               diagnostic_lemma_below_partial_family_upper=bool(complete and lemma < dp_hi))
    return out


def large_q_stress(seed=20261114, n=200000):
    """Random and exact-midpoint transitions of the fast scalar-tie _nearest."""
    rng = np.random.default_rng(seed); worst = 0.
    for m in range(1, 13):
        q = 1 << m
        principal = F._wrap(rng.uniform(-math.pi, math.pi, n))
        estimate = F._wrap(rng.uniform(-math.pi, math.pi, n))
        # add exact candidate midpoints (computed tie geometry) to the batch
        k = rng.integers(0, q, n//10)
        mid = F._wrap((principal[:n//10] + 2*math.pi*k)/q + math.pi/q)
        estimate = np.concatenate([estimate, mid]); principal = np.concatenate([principal, principal[:n//10]])
        value, _, best, _, _ = F._nearest(principal, estimate, q)
        real = np.abs(np.angle(np.exp(1j*(value-estimate))))
        worst = max(worst, float(np.max(np.maximum(best, real) - math.pi/q)))
    return dict(transitions_per_round=n + n//10, rounds='m=1..12', max_excess_rad=worst,
                within_1e_12=worst <= 1e-12)


def zero_parity():
    ok = all(S.count_mean(k, n)['mean'] != 0 for n in range(1, 200, 2) for k in range(n+1))
    return dict(odd_N_never_zero_mean=ok, range='N=1..199 odd, all k')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out')
    out = parser.parse_args().out
    cases = [
        run_case('B_fixture_like', vector=(2, 2), raws=[(.25, -.5)]*2),
        run_case('zero_tie_rich', vector=(2, 2, 2), raws=[(0., 0.)]*3),
        run_case('mixed_parity', vector=(1, 3, 2), raws=[(.6, .1), (.3, -.2), (.5, .4)]),
        run_case('four_rounds', vector=(2, 2, 2, 2), raws=[(.5, .2)]*4),
        run_case('five_rounds_N1', vector=(1, 1, 1, 1, 1), raws=[(.4, .1)]*5),
        run_case('near_principal_edge', vector=(2, 2, 2),
                 raws=[(.6*math.cos(3*2**m), .6*math.sin(3*2**m)) for m in range(3)], target=-.3),
        run_case('odd_only', vector=(3, 1, 3), raws=[(0., 0.)]*3),
        # target phase pi-0.02: wrapped outputs make the UNGUARDED caps fail (negative control)
        run_case('principal_edge_wrap', vector=(1, 1, 1, 1, 1),
                 raws=[(.9*math.cos((math.pi-.02)*2**m), .9*math.sin((math.pi-.02)*2**m)) for m in range(5)],
                 target=-(math.pi-.02)/10),
    ]
    result = dict(source=SOURCE, real_or_saved_inputs=False, MC_calls=0, slack_rad=float(TAU),
                  cases=cases, large_q_stress=large_q_stress(), parity=zero_parity())
    checks = [c[k] for c in cases for k in ('total_equals_original_on_decoded',
              'transition_lemma_within_tau', 'pointwise_first_unsafe_bound')]
    checks += [c[k] for c in cases if 'coarse_upper' in c for k in
               ('dp_lower_le_total', 'total_le_dp_upper', 'total_le_coarse', 'total_le_lemma')]
    checks.append(cases[-1]['diagnostic_naive_edge_unguarded_bound_violated'])  # guard is necessary
    checks += [result['large_q_stress']['within_1e_12'], result['parity']['odd_N_never_zero_mean']]
    result['all_checks_pass'] = all(checks); result['check_count'] = len(checks)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if out:
        Path(out).write_text(text)
    print(text)
    if not result['all_checks_pass']:
        raise SystemExit('W0 synthetic check FAILED')


if __name__ == '__main__':
    main()
