"""C-alpha sector-corrected Step 1, dormant until separately permitted.

No scientific imports, input arrays, operators, or circuits on import/help.
Full dictionary means are constructed before any sector compression. The old
run01/runner are evidence, not modified or retried by this continuation.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import time

BASE = '0d5ca357ccfd8fcc850bef800c5bcadfde2a153e'
ROUNDS = (0, 17)
DELTA, R, K, LD = .02, 4, 0, 3
MATCH = 1e-9  # Existing P1 independent-match diagnostic, not a new guarantee.
OLD = 'artifacts/calpha_h4_step1/2026-11-10/run01/applicability.json'


def disk_diagnostic(gamma, variance, epsilon, elapsed, gap=None):
    """Evaluate the supplied conditional bound; never clamp arcsin or certify."""
    values=(gamma, variance, epsilon, elapsed)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('FINITE_NONNEGATIVE_BOUND_INPUTS_REQUIRED')
    if gap is not None and (not math.isfinite(gap) or gap <= 0):
        raise ValueError('POSITIVE_GAP_REQUIRED')
    scale=elapsed*elapsed/2
    if gap is not None:
        scale=min(scale,2*elapsed/gap)
    a=math.exp(-gamma*elapsed)
    d=variance*scale+epsilon
    valid=d<a
    return dict(reference_radius=a, variance_term=variance*scale, d=d,
        radius_lower_expression=a-d, positive_radius=valid,
        phase_bound_rad=math.asin(d/a) if valid else None,
        satisfies_014=valid and math.asin(d/a)<=.14,
        gap_used=gap, gap_kind='saved_numerical_diagnostic' if gap is not None else 'gap_independent',
        qualification='numerical_evaluation_of_conditional_bound; NOT_CERTIFIED',
        finite_bit_native_implementation_error=None)


def validate_rounds(rows):
    if tuple(row['m'] for row in rows)!=ROUNDS:
        raise ValueError('ONLY_M0_M17_EXACTLY_ONCE')


def verify_old_evidence(root):
    """Read preserved bytes; do not rerun the 46-atom diagnostic."""
    path=root/OLD
    old=json.loads(path.read_text())
    if old['status']!='TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED':
        raise ValueError('OLD_STOP_NOT_PRESERVED')
    for name,expected in old['SOURCE_SHA256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('OLD_EXECUTED_SOURCE_CHANGED:'+name)
    return dict(path=OLD,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        status=old['status'],source_hashes_unchanged=True,
        previous_primitive_violations=len(old['tail_dictionary_diagnostic']['single_Z_spin_violations']))


def sector_power_precision_diagnostic(step, state):
    """Existing 80/120-digit stored-matrix power diagnostic, narrowly ported.

    mp_matrix is from e217d145b's diagnose_b2_p1_strong_pf_numerics.py. Its old
    driver imports an unrelated strong-PF module absent from this base. This
    checks power arithmetic, not binary64 step construction or enclosure.
    """
    import mpmath
    rows={}
    for digits in (80,120):
        ctx=mpmath.mp.clone();ctx.dps=digits
        mp_matrix=lambda matrix:ctx.matrix([[ctx.mpc(float(complex(x).real),float(complex(x).imag))
                                            for x in row] for row in matrix])
        base=mp_matrix(step)
        psi=ctx.matrix([ctx.mpc(float(z.real),float(z.imag)) for z in state])
        power=base
        signals={0:(psi.H*power*psi)[0]}
        for index in range(1,18):
            power=power*power
            if index==17:
                signals[17]=(psi.H*power*psi)[0]
        rows[digits]={m:dict(real=ctx.nstr(z.real,ctx.dps),imag=ctx.nstr(z.imag,ctx.dps))
                      for m,z in signals.items()}
    ctx=mpmath.mp.clone();ctx.dps=120
    differences={}
    for m in ROUNDS:
        a=rows[80][m];b=rows[120][m]
        difference=abs(ctx.mpc(a['real'],a['imag'])-ctx.mpc(b['real'],b['imag']))
        differences[m]=ctx.nstr(difference,ctx.dps)
    return dict(precisions=[80,120],signals=rows,absolute_differences=differences,
        scope='preserving-reference stored-step power only; NOT full dictionary or step-construction certificate')


def numerical_evaluation(private_dir, root, prior_evidence, record=None):
    """Scientific entry. No caller may invoke this as a mock/syntax test."""
    import numpy as np
    from scipy.linalg import expm, eigh
    from trotterlib import b2_p1_signal_validation as p1
    from trotterlib import b2_p1_remaining_rounds as saved
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.rte import make_rte_config, finite_rte_operator_moments

    data=p1.load_private_inputs(private_dir)
    if record is not None:
        record['input_identity']=data['public']
    prior,screen=saved.checked_prior()
    preparation=saved.mean_preparation(data['ham'],screen,prior)
    config,distribution=make_rte_config(preparation.rte_preparation.symbolic_tail,
        evolution_time=DELTA,rte_steps=R,finite_taylor_order=K,
        truncation_tolerance=p1.legacy._explicit_cutoff_tolerance(
            preparation.exact_rte_lambda_r*DELTA/R,K),seed=0)
    lam=preparation.exact_rte_lambda_r
    identity=preparation.extracted_identity_coefficient
    phase=complex(np.exp(-1j*DELTA*(float(data['ham'].constant)+identity)))
    ix=data['indices']; outside=np.setdiff1d(np.arange(256),ix)
    sector=np.ix_(ix,ix)
    occupation=((np.arange(256)[:,None]>>np.arange(8))&1)
    spin_sector_description=data['spec']['sector']
    norm=lambda matrix:float(np.linalg.norm(matrix,2))
    hermitian=lambda matrix:(matrix+matrix.conj().T)/2
    cross=lambda matrix:dict(outward_column_norm=norm(matrix[np.ix_(outside,ix)]),
        inward_row_norm=norm(matrix[np.ix_(ix,outside)]))
    eye=np.eye(256,dtype=complex)
    coefficients=screen['coefficient_projection']['components']

    # Preserve exactly the saved signed coefficients, full mass and generic
    # eigensolver frame. Batched exterior minors reuse the P1 CAR helper.
    head_full=[]; lifted_tail=np.zeros((256,256),complex); frame_diagnostics=[]
    for k,matrix in enumerate((data['ham'].one_body,*data['ham'].g_matrices)):
        orbital,eigenvalues=diag_hermitian(matrix,sort='descending_abs',assume_hermitian=True)
        basis=data['ref'].fock_lift(orbital)
        if k==0 or k<=LD:
            energies=occupation@eigenvalues
            if k:
                energies=float(data['ham'].lambdas[k-1])*energies**2
            block=(basis*np.exp(-.5j*DELTA*energies))@basis.conj().T
            head_full.append(block)
            frame_diagnostics.append(dict(role='completed_head',fragment=k-1,
                basis_unitarity_defect=norm(basis.conj().T@basis-eye),sector=cross(block)))
        else:
            diagonal=np.zeros(256)
            for atom in coefficients:
                if atom['fragment']!=k-1:
                    continue
                signs=np.ones(256)
                for mode in atom['support']:
                    signs*=1-2*occupation[:,mode]
                diagonal+=atom['coefficient']*signs
            lifted_tail+=(basis*diagonal)@basis.conj().T
            frame_diagnostics.append(dict(role='tail_frame',fragment=k-1,
                basis_unitarity_defect=norm(basis.conj().T@basis-eye)))
    # CAR fragments define a preserving Hermitian analysis reference. Its
    # difference from the actual full signed dictionary is explicitly bridged.
    reference_tail=hermitian(sum(data['fragments'][LD:],np.zeros_like(eye))-identity*eye)
    completed_generators=[hermitian(data['one_body']),
                          *[hermitian(b) for b in data['fragments'][:LD]]]
    h_reference=hermitian(data['physical'])
    h_model=sum(completed_generators,np.zeros_like(eye))+reference_tail+(float(data['ham'].constant)+identity)*eye
    reference_errors=dict(tail_dictionary_minus_CAR_sector_columns=norm((lifted_tail-reference_tail)[:,ix]),
        tail_dictionary_minus_CAR_full=norm(lifted_tail-reference_tail),
        tail_dictionary_Hermiticity=norm(lifted_tail-lifted_tail.conj().T),
        tail_dictionary_sector=cross(lifted_tail),reference_tail_sector=cross(reference_tail),
        H_reference_sector=cross(h_reference),model_minus_H_reference=norm(h_model-h_reference),
        H_reference_minus_raw=norm(h_reference-data['physical']))
    if record is not None:
        record['sector_connection']=reference_errors
    if reference_errors['tail_dictionary_minus_CAR_full']>MATCH:
        raise ValueError('FIXED_DICTIONARY_REFERENCE_MISMATCH')
    if any(cross(b)['outward_column_norm']>MATCH for b in [reference_tail,h_reference,*completed_generators]):
        raise ValueError('REFERENCE_COMPLETED_OPERATORS_NOT_SECTOR_COMPATIBLE')

    rs=reference_tail[sector]; hs=h_reference[sector]
    head_car=[expm(-.5j*DELTA*b[sector]) for b in completed_generators]
    ref_moments=finite_rte_operator_moments(rs/lam,config)
    reference_step=p1.compose_step(head_car,ref_moments.attenuated_event_mean_operator,phase)
    dict_moments=finite_rte_operator_moments(lifted_tail/lam,config)
    actual_step=p1.compose_step(head_full,dict_moments.attenuated_event_mean_operator,phase)
    embedded=np.zeros((256,36),complex);embedded[ix,:]=reference_step
    macro_error=norm(actual_step[:,ix]-embedded)
    reference_errors.update(macrostep_column_connection_error=macro_error,
        macrostep_connection_definition='||(S_dictionary J - J S_CAR,S)||_2; includes outward sector components',
        macrostep_connection_assumptions='contraction and exact reference invariance; values below are float diagnostics',
        actual_step_norm=norm(actual_step),reference_step_norm=norm(reference_step),
        completed_head_diagnostics=frame_diagnostics,
        all_reconstruction_differences_counted_once='q*macro_error only; head eta compares CAR sweep to exact CAR head')

    theta=DELTA/R
    values,vectors=eigh(rs)
    r_values=np.arctan(theta*values)/theta
    gamma_values=(math.log1p(theta*theta*lam*lam)-np.log1p(theta*theta*values*values))/(2*theta)
    rt=(vectors*r_values)@vectors.conj().T
    gamma_op=(vectors*gamma_values)@vectors.conj().T
    psi=data['sector_state']
    gamma_mean=np.vdot(psi,gamma_op@psi)
    residual=gamma_op@psi-float(gamma_mean.real)*psi
    variance=float(np.vdot(residual,residual).real)
    if float(np.min(gamma_values))<0:
        raise ValueError('NEGATIVE_DAMPING_RATE_NO_CLAMP')
    d_operator=sum((b[sector] for b in completed_generators),np.zeros_like(rs))
    a=-1j*d_operator; b=-1j*rt-gamma_op
    comm=a@b-b@a
    c_split=norm(a@comm-comm@a)/24+norm(b@comm-comm@b)/12
    forward=np.eye(36,dtype=complex);reverse=forward.copy()
    for half in head_car:
        forward=half@forward
    for half in reversed(head_car):
        reverse=half@reverse
    exact_head=expm(-.5j*DELTA*d_operator)
    eta_f=norm(forward-exact_head);eta_r=norm(reverse-exact_head)
    arc=norm(rt-rs)
    state_result=p1.public_json(p1.STATE_RESULT,'artifacts/b2_hphys_ground_state/2026-11-06/run01/result.json')
    saved_gap=state_result['validation']['same_sector_gap_Ha']
    operator_quantities=dict(lambda_R=lam,identity=identity,theta=theta,
        probability_sum=math.fsum(c.probability for c in preparation.rte_preparation.symbolic_tail.components),
        B0=distribution.exact_finite_distribution,normalization_role='diagnostic only; physical mean already divided once',
        gamma=float(gamma_mean.real),gamma_imaginary_diagnostic=float(gamma_mean.imag),variance_gamma=variance,
        gamma_min_eigenvalue=float(np.min(gamma_values)),Rtheta_minus_R=arc,
        C_split=c_split,eta_forward=eta_f,eta_reverse=eta_r,
        model_error=reference_errors['model_minus_H_reference'],saved_gap_diagnostic=saved_gap,
        gap_certified_lower_bound=None,state_reference_residual=norm(hs@psi-data['energy']*psi),
        state_assumption='supplied exact-ground-state promise; float vector is not an interval certificate',
        finite_bit_native_implementation_error=None,
        saved_coefficient_projection_hash=screen['coefficient_projection']['projection_hash'],
        prior_executable_tail_reference=prior['normalization_law']['tail_hash'],
        coefficient_projection_is_executable_preparation=False)
    if record is not None:
        record['operator_quantities']=operator_quantities
    precision=sector_power_precision_diagnostic(reference_step,psi)
    by_m={row['m']:row for row in prior['rounds']}
    rows=[]
    if record is not None:
        record['rounds']=rows
        record['reference_power_precision_diagnostic']=precision
    pack=lambda z:dict(real=float(z.real),imag=float(z.imag),radius=float(abs(z)),phase_rad=float(np.angle(z)))
    # Full-space powers keep the tiny reconstructed off-sector blocks instead
    # of deleting them. No per-event compressed exponent or trajectory occurs.
    actual_powers={0:actual_step}
    reference_powers={0:reference_step}
    full=actual_step;small=reference_step
    for power_index in range(1,18):
        full=full@full;small=small@small
        if power_index==17:
            actual_powers[17]=full;reference_powers[17]=small
    for m in ROUNDS:
        q=1<<m;elapsed=q*DELTA
        z=complex(np.vdot(data['state'],actual_powers[m]@data['state']))
        z_sector=complex(np.vdot(psi,reference_powers[m]@psi))
        z_gamma=complex(np.vdot(psi,expm(elapsed*(-1j*hs-gamma_op))@psi))
        ground=complex(np.exp(-1j*data['energy']*elapsed))
        attenuation_reference=ground*math.exp(-float(gamma_mean.real)*elapsed)
        parts=dict(split=elapsed*DELTA**2*c_split,arctan=elapsed*arc,
            head_internal=q*(eta_f+eta_r),model=elapsed*operator_quantities['model_error'],
            full_dictionary_connection=q*macro_error,
            raw_reference_representation=math.expm1(elapsed*reference_errors['H_reference_minus_raw']))
        epsilon=math.fsum(parts.values())
        k2=by_m[m]['signals']['raw']
        old_z=complex(k2['real'],k2['imag'])
        hp=precision['signals'][120][m]
        hp_z=complex(float(hp['real']),float(hp['imag']))
        rows.append(dict(m=m,q=q,T_Ha_inverse=elapsed,raw_signal=pack(z),
            preserving_reference_signal=pack(z_sector),damped_reference_signal=pack(z_gamma),
            leading_damping_reference_signal=pack(attenuation_reference),
            ground_phase_reference_only=pack(ground),observed_phase_error_rad=p1.phase_distance(z,ground),
            dictionary_vs_sector_signal_difference=p1.signal_difference(z,z_sector),
            reference_power_float_vs_HP120=p1.signal_difference(z_sector,hp_z),
            epsilon_dyn_parts=parts,epsilon_dyn_excluding_unknown_implementation=epsilon,
            gap_independent=disk_diagnostic(float(gamma_mean.real),variance,epsilon,elapsed),
            numerical_gap_diagnostic=disk_diagnostic(float(gamma_mean.real),variance,epsilon,elapsed,saved_gap),
            K2_saved_raw_signal=k2,K0_K2_signal_difference=p1.signal_difference(z,old_z),
            K2_result_SOURCE=saved.PRIOR_COMMIT,qualification='ideal-law float diagnostic; implementation-inclusive guarantee UNKNOWN'))
    validate_rounds(rows)
    return dict(status='TWO_POINT_NUMERICAL_DIAGNOSTICS_COMPLETE',input_identity=data['public'],
        prior_evidence=prior_evidence,sector=spin_sector_description,operator_quantities=operator_quantities,
        sector_connection=reference_errors,rounds=rows,reference_power_precision_diagnostic=precision,
        scientific_calls=dict(new_raw_signal_points=2,new_damped_reference_points=2,
            ground_solve=0,circuit_build=0,compile=0,sampling=0,shot_accounting=0,other_round_signal_points=0),
        guarantee_status='not_certified',implementation_inclusive_guarantee='UNKNOWN')


def run(private_dir, output, sector_review, *, evaluator=None):
    root=Path(__file__).resolve().parents[1]
    review=Path(sector_review)
    if not review.is_file():
        raise ValueError('SECTOR_REVIEW_ORIGINAL_REQUIRED')
    output=Path(output).resolve()
    if output.exists() or output.is_relative_to(root/Path(OLD).parent):
        raise ValueError('NEW_NONEXISTENT_OUTPUT_REQUIRED_PRESERVE_RUN01')
    prior_evidence=verify_old_evidence(root)
    sources=[Path(__file__).relative_to(root).as_posix(),
        'src/trotterlib/b2_p1_signal_validation.py','src/trotterlib/b2_p1_remaining_rounds.py',
        'src/trotterlib/rte.py','src/trotterlib/df_trotter/decompose.py',
        'scripts/run_o1_m1_h4_operator_correctness.py']
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result=dict(schema='calpha_h4_step1_sector_corrected_v1',base_SOURCE=BASE,
        started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,
        sector_review_SHA256=hashlib.sha256(review.read_bytes()).hexdigest(),
        fixed_conditions=dict(K=K,r=R,L_D=LD,delta=DELTA,rounds=list(ROUNDS),
            systematic_budget=.14,sector='Nalpha=Nbeta=2;dimension36',frame='unchanged_generic',
            sampling='fresh_independent_full_space_trajectory; mean-only evaluation'),
        status='STARTED',mandatory_stop=True)
    try:
        evaluated=(numerical_evaluation(private_dir,root,prior_evidence,record=result)
                   if evaluator is None else evaluator(private_dir,root,prior_evidence))
        validate_rounds(evaluated['rounds'])
        result.update(evaluated)
    except Exception as exc:
        result.update(status='TECHNICAL_STOP_EXCEPTION',stop_reason=str(exc),exception_type=type(exc).__name__)
        raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        result['environment']=dict(python=platform.python_version(),affinity=len(os.sched_getaffinity(0)),
            BLAS={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')})
        (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sector-review',required=True)
    args=parser.parse_args()
    result=run(args.private_dir,args.output,args.sector_review)
    print(result['status'])
