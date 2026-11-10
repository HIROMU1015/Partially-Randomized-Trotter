"""Evaluate new bound inputs, reusing run02 signals without retrying that run."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PRIOR = 'artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json'
PRIOR_SHA = '8ff3e10d6fb663714b68aa0e9b2affa0816bc04d92c509eede7dca8791fbfffb'
DELTA, LD, R = .02, 3, 4


def evaluate(private_dir, prior):
    import numpy as np
    from scipy.linalg import eigh, null_space, solve
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib.calpha_macrostep_bounds import (
        macrostep_diagnostics, certify_gap_from_brackets)

    data = p1.load_private_inputs(private_dir)
    if any(data['public'][key] != prior['input_identity'][key]
           for key in ('original_hamiltonian', 'original_state', 'df_hamiltonian', 'sector')):
        raise ValueError('SAME_ORIGINAL_INPUTS_REQUIRED')
    ix = np.ix_(data['indices'], data['indices'])
    hermitian = lambda x: (x + x.conj().T)/2
    old = prior['operator_quantities']
    h = hermitian(data['physical'])[ix]
    heads = [hermitian(x)[ix] for x in [data['one_body'], *data['fragments'][:LD]]]
    tail = hermitian(sum(data['fragments'][LD:], np.zeros((256,256), complex))
                     - old['identity']*np.eye(256))[ix]
    eigenvalues, vectors = eigh(tail)
    theta = DELTA/R
    rtheta = (vectors*(np.arctan(theta*eigenvalues)/theta))@vectors.conj().T
    rates = (math.log1p(theta**2*old['lambda_R']**2)
             - np.log1p(theta**2*eigenvalues**2))/(2*theta)
    gamma_op = (vectors*rates)@vectors.conj().T
    diagnostic, _, _ = macrostep_diagnostics(heads, rtheta, gamma_op, DELTA)
    diagnostic['head_append_order'] = ['one_body', 'fragment0', 'fragment1', 'fragment2']
    diagnostic['central_to_outer'] = ['fragment2', 'fragment1', 'fragment0', 'one_body']

    # Bracket seeds use the already saved energy/gap; exact inertia, rather
    # than their precision, establishes the eigenvalue counts. No state solve.
    e0 = Fraction.from_float(data['energy'])
    e1 = e0 + Fraction.from_float(old['saved_gap_diagnostic'])
    width = Fraction(1, 1 << 36)
    try:
        gap = certify_gap_from_brackets(h, [e0-width, e0+width], [e1-width, e1+width])
        gap_value = math.nextafter(float(Fraction(gap['gap_lower_exact'])), -math.inf)
        gap['gap_lower_float_downward'] = gap_value
    except ValueError as exc:
        gap = dict(certified_for_supplied_matrix=False, reason=str(exc))
        gap_value = None

    # State-dependent alternative: diagnostic only. Do not promote numerical
    # SVD, complement construction or solve into a resolvent certificate.
    psi = data['sector_state']
    complement = null_space(psi.conj()[None,:])
    g = complement.conj().T@gamma_op@psi
    bq = complement.conj().T@(h-data['energy']*np.eye(36))@complement
    bq = bq-1j*(complement.conj().T@gamma_op@complement-old['gamma']*np.eye(35))
    w = solve(bq, g)
    resolvent = dict(b=float(np.linalg.norm(g)), w_norm=float(np.linalg.norm(w)),
                     smallest_singular_value=float(np.linalg.svd(bq, compute_uv=False)[-1]),
                     solve_residual=float(np.linalg.norm(bq@w-g)),
                     complement_unitarity_defect=float(np.linalg.norm(complement.conj().T@complement-np.eye(35),2)),
                     invertibility_certified=False,
                     qualification='state_dependent_binary64_diagnostic_NOT_CERTIFIED')
    rows = []
    for previous in prior['rounds']:
        m, q, elapsed = previous['m'], previous['q'], previous['T_Ha_inverse']
        pf_comm = elapsed*DELTA**2*diagnostic['C_multi']
        pf_defect = q*diagnostic['eta_macro']
        other = {key: value for key, value in previous['epsilon_dyn_parts'].items()
                 if key not in ('head_internal', 'split')}
        epsilon = min(pf_comm, pf_defect)+math.fsum(other.values())
        variance_free = old['variance_gamma']*elapsed**2/2
        variance_gap = (min(variance_free, 2*old['variance_gamma']*elapsed/gap_value)
                        if gap_value is not None else None)
        variance_resolvent = min(variance_free, resolvent['b']*resolvent['w_norm']*(
            elapsed-math.expm1(-old['gamma']*elapsed)/old['gamma']))
        a = math.exp(-old['gamma']*elapsed)
        variants = {}
        for name, variance in [('gap_independent',variance_free),
                               ('certified_matrix_gap_conditional',variance_gap),
                               ('resolvent_diagnostic',variance_resolvent)]:
            if variance is None:
                variants[name] = dict(status='UNKNOWN')
                continue
            distance = variance+epsilon
            variants[name] = dict(variance_term=variance, d=distance,
                reference_radius=a, radius_lower_expression=a-distance,
                phase_bound_rad=math.asin(distance/a) if distance<a else None,
                satisfies_014=distance<=a*math.sin(.14),
                qualification='numerical_evaluation_of_conditional_formula_NOT_CERTIFIED')
        rows.append(dict(m=m, q=q, T_Ha_inverse=elapsed,
            saved_raw_signal=previous['raw_signal'],
            saved_observed_phase_error_rad=previous['observed_phase_error_rad'],
            PF_commutator=pf_comm, PF_macro_defect=pf_defect,
            PF_selected_diagnostic=min(pf_comm,pf_defect),
            unchanged_ledger_terms_from_run02=other,
            epsilon_dyn_excluding_unknown_implementation=epsilon,
            finite_bit_native_implementation_error=None, variants=variants))
    return dict(input_identity=data['public'], macrostep=diagnostic,
                gap_certificate=gap, resolvent=resolvent, rounds=rows,
                new_raw_signal_evaluations=0, new_damped_signal_evaluations=0,
                ground_state_solves=0, circuit_builds=0, shots_or_prices=0,
                implementation_inclusive_guarantee='UNKNOWN',
                qualification='diagnostics_and_exact_gap_for_explicit_matrix; NOT_comprehensive_guarantee')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    review = Path(args.review)
    if not review.is_file():
        raise ValueError('LATEST_SCIENTIFIC_REVIEW_ORIGINAL_REQUIRED')
    path = ROOT/PRIOR
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR_SHA:
        raise ValueError('PRESERVE_RUN02_BYTES')
    prior = json.loads(path.read_text())
    for name, expected in prior['SOURCE_SHA256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('PRESERVE_RUN02_SOURCE:'+name)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files = ['scripts/run_calpha_macrostep_gap.py','src/trotterlib/calpha_macrostep_bounds.py',
             'src/trotterlib/b2_p1_signal_validation.py']
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    result = dict(schema='calpha_macrostep_gap_diagnostic_v1',
        started_UTC=datetime.now(timezone.utc).isoformat(),
        base_SOURCE=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        SOURCE_SHA256=hashes, review_SHA256=hashlib.sha256(review.read_bytes()).hexdigest(),
        prior_result=dict(path=PRIOR,sha256=PRIOR_SHA), status='STARTED',
        fixed_conditions=prior['fixed_conditions'],
        environment=dict(affinity=len(os.sched_getaffinity(0)),
                         BLAS={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']}))
    started = time.monotonic()
    try:
        result.update(evaluate(args.private_dir,prior))
        result['status']='BOUND_DIAGNOSTICS_AND_GAP_CHECK_COMPLETE'
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc))
        raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h for name,h in hashes.items())
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main()
