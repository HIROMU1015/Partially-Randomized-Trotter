"""Enclose the contraction-weighted version of review (3.6).

This certifies the explicit dyadic CAR reference only. Preserved run02's
generic/full connection and native implementation are not interval-certified
by this program and cannot authorize the downstream RPE/cost gate.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def evaluate(operands):
    import numpy as np
    from trotterlib.calpha_interval_bounds import IntervalBounds, endpoint
    from trotterlib.calpha_macrostep_bounds import rational_inertia
    b=IntervalBounds(40); c=b.c
    old=json.loads((ROOT/'artifacts/calpha_h4_step1/2026-11-10/run02_sector_corrected/result.json').read_text())
    gap=json.loads((ROOT/'artifacts/calpha_h4_step1/2026-11-10/run03_macrostep_gap/result.json').read_text())['gap_certificate']
    state_run=json.loads((ROOT/'artifacts/calpha_h4_step1/2026-11-10/run04_state_defect/result.json').read_text())
    if hashlib.sha256(Path(operands).read_bytes()).hexdigest()!=state_run['operands_SHA256']:
        raise ValueError('FROZEN_PRIVATE_OPERANDS_REQUIRED')
    with np.load(operands,allow_pickle=False) as raw:
        a={key:raw[key].copy() for key in raw.files}
    n=len(a['tail']); ident=c.eye(n)
    for matrix in [a['tail'],a['h_reference'],*a['heads']]:
        if not np.array_equal(matrix,matrix.conj().T):
            raise ValueError('EXACT_DYADIC_HERMITICITY_REQUIRED')
    radius=Fraction.from_float(float(np.linalg.norm(a['tail'],2)))+Fraction(1,1<<36)
    counts=[rational_inertia(a['tail'],x) for x in [-radius,radius]]
    if [x['negative'] for x in counts]!=[0,n]:
        raise ValueError('TAIL_SPECTRAL_RADIUS_NOT_ENCLOSED')
    rnorm=b.real(radius); lam=b.real(float(a['lambda_R']))
    delta=b.real(.02); theta=delta/4
    rate=(c.ln(1+(theta*lam)**2)-c.ln(1+(theta*rnorm)**2))/(2*theta)
    if endpoint(rate)<=0:
        raise ValueError('STRICT_CONTRACTION_NOT_CERTIFIED')
    rate=b.real(endpoint(rate))
    print('tail log intervals',flush=True)
    rt,gamma=b.tail_log(b.matrix(a['tail']),lam,theta,rnorm)
    generator=-c.j*rt-gamma
    step=b.expm(delta*generator)
    for j in reversed(range(len(a['heads']))):
        print('head interval exponential '+str(j),flush=True)
        head=-c.j*b.matrix(a['heads'][j])
        half=b.expm(delta*head/2)
        step=half*step*half
        generator+=head
    print('reference interval exponential',flush=True)
    exact=b.expm(delta*generator)
    defect=step-exact
    psi=b.matrix([[x] for x in a['psi']])
    normpsi=c.sqrt((b.dagger(psi)*psi)[0,0].real)
    normalized=psi/normpsi
    h=b.matrix(a['h_reference'])
    energy=b.real(float(a['energy']))
    separation=b.real(Fraction(gap['first_excited_interval'][0]))-energy
    distance=(c.sqrt(2)*b.norm((h-energy*ident)*normalized)/separation).b
    gamma_max=(c.ln(1+(theta*lam)**2)/(2*theta)).b
    gamma_snapshot=(b.dagger(normalized)*gamma*normalized)[0,0].real
    gamma_upper=(gamma_snapshot+2*gamma_max*distance).b
    variance_upper=(b.norm((gamma-gamma_snapshot*ident)*normalized)+4*gamma_max*distance)**2
    variance_upper=variance_upper.b
    rows=[]
    for row in old['rounds']:
        q=row['q']; elapsed=q*delta
        print('state interval bound m'+str(row['m']),flush=True)
        pf=b.weighted_state_pf(defect,generator,a['eigenvalues'],a['eigenvectors'],
                              a['coefficients'],psi,delta,rate,q)
        # Normalization is an analysis reference; the saved state is unchanged.
        # Two distinct comparisons: PF vector on true ground vs snapshot, and
        # raw expectation on saved (slightly unnormalized) vector vs ground.
        pf_ground=(pf/normpsi.a+2*distance).b
        state_rep=((normpsi.b+1)*(distance+abs(normpsi-1).b)).b
        arctan=(elapsed*theta**2*rnorm**3/3).b
        variance=min(variance_upper*elapsed**2/2,
                     2*variance_upper*elapsed/b.real(Fraction(gap['gap_lower_exact'])),key=lambda x:endpoint(x,True)).b
        core=(pf_ground+state_rep+arctan+variance).b
        reference=c.exp(-gamma_upper*elapsed).a
        # Retain the old bridge/model/raw values only as explicitly unqualified
        # diagnostics, never as certified upper bounds or as zeros.
        unqualified={key:value for key,value in row['epsilon_dyn_parts'].items()
                     if key in ['model','full_dictionary_connection','raw_reference_representation']}
        total_diagnostic=core+sum((b.real(x) for x in unqualified.values()),c.mpf(0))
        values=dict(PF_weighted_state_upper=pf_ground, state_representation_upper=state_rep,
                    arctan_upper=arctan, variance_gap_upper=variance,
                    core_distance_upper=core,reference_radius_lower=reference,
                    allowable_distance=reference*c.sin(b.real('0.14')),
                    core_radius_lower=reference-core,
                    conditional_total_with_unqualified_values=total_diagnostic,
                    conditional_phase_with_unqualified_values=c.atan2(total_diagnostic,
                       c.sqrt(reference**2-total_diagnostic**2)) if endpoint(total_diagnostic,True)<endpoint(reference) else None)
        rows.append(dict(m=row['m'],q=q,values={key:None if value is None else
            dict(lower_exact=str(endpoint(value)),upper_exact=str(endpoint(value,True)),
                 lower_float=float(endpoint(value)),upper_float=float(endpoint(value,True)))
            for key,value in values.items()},
            core_satisfies_014=endpoint(core,True)<=endpoint(reference*c.sin(b.real('0.14'))),
            unqualified_ledger_values=unqualified,
            full_fixed_law_guarantee='UNKNOWN_GENERIC_FULL_CONNECTION_NOT_ENCLOSED',
            implementation_inclusive_guarantee='UNKNOWN'))
    return dict(interval_digits=40, exponential_degree=32, matrix_function_series_terms=7,
        reference_semantics='exact dyadic CAR head/tail/H_ref matrices; tail log functions enclosed by series',
        tail_norm_upper_exact=str(radius),tail_inertia=counts,
        rate_lower_exact=str(endpoint(rate)),rate_lower_float=float(endpoint(rate)),
        gamma_upper_exact=str(endpoint(gamma_upper,True)),
        variance_upper_exact=str(endpoint(variance_upper,True)),
        ground_ray_distance_upper=float(endpoint(distance,True)),
        rounds=rows, new_raw_signals=0,new_damped_signals=0,
        downstream_RPE_cost_gate=False,
        reason='Generic/full connection, model and raw terms retain run02 numerical-only qualification; finite-bit/native UNKNOWN.',
        mathematical_refinement='Exact telescoping identity retains ||F^(q-1-k)|| <= exp(-mu*delta*(q-1-k)); approximate eigenpairs enclosed by residual Duhamel bounds.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operands',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_calpha_interval_state_bound.py','src/trotterlib/calpha_interval_bounds.py',
           'src/trotterlib/calpha_macrostep_bounds.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,status='STARTED')
    start=time.monotonic()
    try:
        result.update(evaluate(args.operands));result['status']='REFERENCE_CORE_ENCLOSURE_COMPLETE'
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
