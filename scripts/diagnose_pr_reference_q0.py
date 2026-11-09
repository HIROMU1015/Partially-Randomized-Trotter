"""Fixed Q0: saved m17 interval/count diagnosis; no circuit or sampling calls."""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import types
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
PLAN = '1b795c250c8c2fc0a463695b472aa6877951c620'
BASE = 'f2638a7845685a119619382b850441f5d8d33713'
SAVED = '5f672ce6884bd49231366875a9e24a539fcaec7a'
SELF = 'scripts/diagnose_pr_reference_q0.py'
FILES = [(SAVED, 'artifacts/pr_reference_v6p/2026-11-09/P_run01/'+p+'.json')
         for p in ('RESULTS', 'SOURCE', 'CHECKS')] + [
    (BASE, 'artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json'),
    (BASE, 'artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/GROUND_ALIAS.json')]
REUSE = ('src/trottertracks/track_c_compiler_a_18round.py',
         'src/trotterlib/rpe_resource_accounting.py',
         'src/trotterlib/rpe_hadamard_failure_validation.py',
         'src/trottertracks/pr_reference_bprime_accounting.py')
BITS = 192


def raw(ref, path):
    return subprocess.check_output(['git', 'show', ref+':'+path], cwd=ROOT)


def setup():
    mp.mp.prec = mp.iv.prec = BITS
    # Reuse the exact pinned interval constructor, without importing builders.
    tree = ast.parse(raw(BASE, REUSE[0]))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'I')
    ns = {'mp': mp}
    exec(compile(ast.Module(body=[node], type_ignores=[]), REUSE[0], 'exec'), ns)
    return ns['I']


I = None


def iv(x):
    if isinstance(x, F):
        return I(x.numerator)/x.denominator
    return I(x)


def endpoints(x):
    return tuple(mp.mpf(t) for t in x._mpi_)


def exact(t):
    sign, mantissa, exponent, bitcount = t
    assert bitcount >= 0
    return F(-mantissa if sign else mantissa)*F(2)**exponent


def pack(x):
    return {'exact_interval': [str(exact(t)) for t in x._mpi_],
            'display': [mp.nstr(p, 26) for p in endpoints(x)]}


def minimum(*xs):
    return I([min(endpoints(x)[0] for x in xs), min(endpoints(x)[1] for x in xs)])


def maximum(*xs):
    return I([max(endpoints(x)[0] for x in xs), max(endpoints(x)[1] for x in xs)])


def asin(x):
    """Monotone clipped inverse, using the old atan2/sqrt interval convention."""
    lo, hi = endpoints(x)
    lo, hi = max(mp.mpf(-1), min(mp.mpf(1), lo)), max(mp.mpf(-1), min(mp.mpf(1), hi))
    def at(v):
        if v == 1:
            return mp.iv.pi/2
        if v == -1:
            return -mp.iv.pi/2
        z = I(v)
        return mp.iv.atan2(z, mp.iv.sqrt(1-z*z))
    return I([endpoints(at(lo))[0], endpoints(at(hi))[1]])


def inverse_endpoint(x, rho, lower):
    # sin(v) belongs to [l/rho_U, h/rho_L] on the positive half;
    # the denominators interchange on the negative half. rho_U=1.
    def at(v):
        denominator = rho if (v < 0 if lower else v >= 0) else I(1)
        return asin(I(v)/denominator)
    a, b = endpoints(x)
    return I([endpoints(at(a))[0], endpoints(at(b))[1]])


def epsilon(n, alpha=F(1, 360)):
    # Inverse of the pinned _hoeffding_shots: 2 exp(-n eps^2/2).
    return mp.iv.sqrt(2*mp.iv.log(2/iv(alpha))/n)


def hull(n, k, rho, D, U, eps=None):
    """Full preimage hull on the monotone sine branch, in residual-phase units."""
    if eps is None:
        eps = epsilon(n)
    c = iv(F(2*k, n)-1)
    lower = maximum(-U, inverse_endpoint(c-eps, rho, True)-D)
    upper = minimum(U, inverse_endpoint(c+eps, rho, False)+D)
    if endpoints(upper)[1] < endpoints(lower)[0]:
        return None  # Count CI is incompatible with every model point.
    return lower, upper, maximum(I(0), upper-lower)


def candidate_counts(n, rho, D, U, eps):
    """Cover ALL counts: width is increasing then decreasing on c>=0.

    The only turning point is c*=rho_L sin(U-D)-eps, where the upper
    endpoint first reaches U. Reflection covers negative c. Neighboring
    lattice counts, the center and endpoints therefore contain every maximum.
    Integer rounding uses both directed endpoints; no floating grid is used.
    """
    star = maximum(I(0), rho*mp.iv.sin(U-D)-eps)
    index = n*(star+1)/2
    a, b = endpoints(index)
    keys = {0, n, n//2, (n+1)//2}
    keys.update(range(max(0, int(mp.floor(a))-1), min(n, int(mp.ceil(b))+1)+1))
    keys |= {n-k for k in tuple(keys)}
    return sorted(keys)


def worst(n, rho, D, U, full=False):
    eps = epsilon(n)
    counts = range(n+1) if full else candidate_counts(n, rho, D, U, eps)
    values = [(k, hull(n, k, rho, D, U, eps)) for k in counts]
    feasible = [(k, h) for k, h in values if h is not None]
    if not feasible:
        raise ValueError('No count is compatible with the model')
    bound = maximum(*(h[2] for _, h in feasible))
    witness = max(feasible, key=lambda v: endpoints(v[1][2])[0])
    return bound, witness, len(values)


def verdict(value, target):
    a, b = endpoints(value)
    c, d = endpoints(target)
    return 'PASS' if b <= c else ('FAIL' if a > d else 'INDETERMINATE')


def margin(n, rho, B, U, target):
    """A bracket on allowed feedback error for this n, not an adopted provider."""
    lo, hi = F(0), F(2, 5)
    assert verdict(worst(n, rho, B, U)[0], target) == 'PASS'
    if verdict(worst(n, rho, B+iv(hi), U)[0], target) == 'PASS':
        return {'at_least': str(hi), 'scope_upper_reached': True}
    for _ in range(80):
        mid = (lo+hi)/2
        status = verdict(worst(n, rho, B+iv(mid), U)[0], target)
        if status == 'INDETERMINATE':
            return {'status': status, 'bracket': [str(lo), str(hi)]}
        if status == 'PASS':
            lo = mid
        else:
            hi = mid
    return {'status': 'BRACKETED', 'bracket': [str(lo), str(hi)],
            'display': [mp.nstr(mp.mpf(v.numerator)/v.denominator, 24) for v in (lo, hi)],
            'lower_is_sufficient': True, 'upper_fails_this_certificate': True,
            'bisections': 80, 'not_a_provider_error_certificate': True}


def xy_certificate(rho, beta, max_total):
    """Optimize integer allocations within the SAME Hoeffding L2 certificate.

    alpha_X=A*nY/N and alpha_Y=A*nX/N minimize the squared coordinate
    radius for fixed integers. The continuous relaxation is a lower bound
    ONLY for this sufficient-certificate family, not any phase estimator.
    """
    A = F(1, 360)
    S = (rho*mp.iv.sin(beta))**2
    relaxation = 8*mp.iv.log(4/iv(A))/S
    start = max(2, int(mp.ceil(endpoints(relaxation)[0])))
    rows = []
    for total in range(start, max_total+1):
        best = None
        for nx in range(1, total//2+1):
            ny = total-nx
            ax, ay = A*F(ny, total), A*F(nx, total)
            lhs = epsilon(nx, ax)**2+epsilon(ny, ay)**2
            item = (nx, ny, ax, ay, lhs)
            if best is None or endpoints(lhs)[1] < endpoints(best[4])[1]:
                best = item
        nx, ny, ax, ay, lhs = best
        status = verdict(lhs, S)
        rows.append({'total': total, 'best_n': [nx, ny], 'alpha': [str(ax), str(ay)],
                     'squared_coordinate_radius': pack(lhs), 'threshold': pack(S), 'status': status,
                     'all_ordered_allocations_covered': total-1})
        if status != 'FAIL':
            break
    return {'continuous_relaxation': pack(relaxation), 'boundary_totals': rows,
            'minimum_certified_total': next((r['total'] for r in rows if r['status']=='PASS'), None),
            'scope': 'integer optimum of Hoeffding coordinate L2 sufficient certificate only; not a global shot optimum'}


def arm_run(task):
    global I
    I = setup()
    name, row = task
    rho = I(row['raw_radius_lower'])
    beta = I(row['statistical_phase_lower'])
    B = iv(F(2, 5))-beta
    U, target = iv(F(4, 5)), iv(F(4, 5))
    monotone = verdict(U+B+iv(F(2, 5)), mp.iv.pi/2)
    assert monotone == 'PASS'
    # Even infinite data do not identify a Pauli projection at center pi/4.
    witness_low = asin(rho)-mp.iv.pi/4-B
    witness_high = mp.iv.pi/4+B
    assert endpoints(witness_low)[0] > -endpoints(U)[0]
    assert endpoints(witness_high)[1] < endpoints(U)[0]
    pauli_width = witness_high-witness_low
    assert verdict(pauli_width, target) == 'FAIL'
    ideal = U-asin(rho*mp.iv.sin(U-B))+B
    records, passing, uncertain = [], [], []
    for n in range(1, 1253):
        width, (k, h), count = worst(n, rho, B, U)
        status = verdict(width, target)
        records.append({'n': n, 'worst_phase_diameter': pack(width), 'status': status,
                        'witness_k': k, 'witness_hull': [pack(h[0]), pack(h[1])],
                        'count_domain_size': n+1, 'analytic_envelope_counts_evaluated': count})
        if status == 'PASS':
            passing.append(n)
        elif status == 'INDETERMINATE':
            uncertain.append(n)
    if uncertain:
        return {'arm': name, 'status': 'INDETERMINATE', 'uncertain_n': uncertain, 'scan': records}
    first = passing[0] if passing else None
    # At the minimum and preceding count, independently enumerate ALL counts.
    full_checks = []
    if first is not None:
        for n in (first-1, first, 1252):
            fast = worst(n, rho, B, U)[0]
            slow = worst(n, rho, B, U, full=True)[0]
            assert verdict(fast, target) == verdict(slow, target)
            assert max(endpoints(fast)[0], endpoints(slow)[0]) <= min(endpoints(fast)[1], endpoints(slow)[1])
            full_checks.append({'n': n, 'all_count_scan': pack(slow), 'envelope_overlap': True,
                                'counts_enumerated': n+1})
    boundaries = {str(n): margin(n, rho, B, U, target) for n in sorted({first, 1251, 1252})
                  if n is not None and n in passing}
    return {'arm': name, 'status': 'CONDITIONAL_IDEAL_READOUT_COUNT_CERTIFICATE' if first else 'STOP_NO_PASSING_N',
            'rho_lower': pack(rho), 'rho_upper': 1, 'systematic_phase_upper': pack(B),
            'statistical_phase_lower': pack(beta), 'prior_residual_halfwidth': '4/5',
            'target_phase_diameter': '4/5', 'ideal_infinite_data_worst_diameter': pack(ideal),
            'monotone_even_with_scope_feedback_cap': monotone,
            'minimum_one_direction_n': first, 'passing_n': passing, 'scan': records,
            'feedback_margins': boundaries, 'full_count_crosschecks': full_checks,
            'single_pauli_counterexample': {'center_phase': 'pi/4', 'equal_projected_mean': pack(rho),
                'residuals': [pack(witness_low), pack(witness_high)], 'diameter': pack(pauli_width),
                'failure_even_with_infinite_samples': True,
                'both_Pauli_choices_fail_by_reflection': True},
            'xy_baseline': xy_certificate(rho, beta, 1252),
            'scope': 'unknown-radius worst-case signal set; not actual H4 performance, new RPE guarantee or circuit qualification'}


def inputs():
    docs = [json.loads(raw(ref, path)) for ref, path in FILES]
    result, provenance, checks, bp, alias = docs
    assert result['fixed_target'] == [17, 131072, 32, 88]
    assert checks['status'] == 'V6P_POSTRUN_CHECKS_PASS'
    assert provenance['inputs']['input_commit'] == BASE
    assert alias['all_spectrum_endpoint_margin_pass']
    assert alias['alias_proof'] == 'Gershgorin: delta0 ||H|| + .4 < pi; no E0 supplied to decoder'
    assert result['Bprime_saved'] == bp['rows'][17]
    assert [r['m'] for r in bp['rows']] == list(range(18))
    assert bp['alpha_per_axis'] == '1/720'
    return [('Bprime', result['Bprime_saved']), ('Reference_B1', result['conditional_reference'])], {
        'target': result['fixed_target'], 'prefix_success_assumption': 'saved m0..16 guarantee; no new prefix calculation',
        'm16_q': bp['rows'][16]['q'], 't17_over_t16': F(bp['rows'][17]['q'], bp['rows'][16]['q']).numerator,
        'prefix_failure': '34/720', 'm17_failure': '2/720', 'total_failure': '1/20',
        'unknown_state_energy_gap_and_true_means_used': False,
        'alias_fields_used': ['all_spectrum_endpoint_margin_pass', 'alias_proof', 'alias_window'],
        'raw_disks_kept_in_saved_beta_and_radius_no_readdition': True,
        'binomial_confidence': 'not adopted: old scipy point-tail/ppf routines lack uniform directed interval inversion; Hoeffding fallback on both arms',
        'matched_conditions': 'Linear H4 1A STO-3G DF12 LD3 Partial S2 h=.02 K2 compiler A canonical independent finite bits; exact original-H ground supplied, preparation excluded'}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def run(out):
    global I
    I = setup()
    out.mkdir(parents=True, exist_ok=False)
    source = {'plan_commit': PLAN, 'base_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'script_sha256': hashlib.sha256((ROOT/SELF).read_bytes()).hexdigest(),
              'python': sys.version, 'python_executable': sys.executable, 'mpmath': mp.__version__,
              'interval_bits': BITS, 'inputs': [], 'reused_source': []}
    for ref, path in FILES:
        data = raw(ref, path)
        source['inputs'].append({'commit': ref, 'path': path,
            'blob': subprocess.check_output(['git','rev-parse',ref+':'+path],cwd=ROOT,text=True).strip(),
            'sha256': hashlib.sha256(data).hexdigest()})
    for path in REUSE:
        source['reused_source'].append({'commit': BASE, 'path': path, 'sha256': hashlib.sha256(raw(BASE,path)).hexdigest()})
    write(out/'SOURCE.json', source)
    tasks, projected = inputs()
    write(out/'INPUTS.json', {**projected, 'arms': {name: {k: row[k] for k in
        ('raw_radius_lower','statistical_phase_lower','total_raw_disk_upper','shots_per_axis')} for name,row in tasks}})
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(arm_run, tasks))
    write(out/'RESULTS.json', {'status': 'Q0_FIXED_SCOPE_COMPLETE', 'arms': results,
        'elapsed_seconds': time.monotonic()-start, 'independent_arm_workers': len(tasks),
        'new_circuit_compile_sampling_RPE_total_price_synthesis_calls': 0})
    for item in source['inputs']:
        assert hashlib.sha256(raw(item['commit'],item['path'])).hexdigest() == item['sha256']
    write(out/'STOP.json', {'status': 'MANDATORY_STOP', 'scope': 'Q0 section 2 only',
        'reason': 'fixed two-arm finite diagnosis complete', 'next_owner': 'GPT/user',
        'Q_circuit_RPE_total_T_and_I_not_started': True, 'fixed_inputs_unchanged': True})
    print(json.dumps({r['arm']: {'status':r['status'], 'minimum_n':r.get('minimum_one_direction_n'),
        'xy_certificate_total':r.get('xy_baseline',{}).get('minimum_certified_total'),
        'feedback_margins':r.get('feedback_margins')} for r in results},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
