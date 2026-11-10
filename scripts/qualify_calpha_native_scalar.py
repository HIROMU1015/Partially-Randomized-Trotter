"""Exact phase-grouping bridge from qualified step builder to native emitter.

Only saved scalar input and completed interval records are read. No signal,
Hamiltonian construction, circuit build or compiler is invoked.
"""
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART='artifacts/calpha_h4_step1/2026-11-10/'


def evaluate(hdf_path):
    import h5py
    from trotterlib.calpha_interval_bounds import IntervalBounds,endpoint
    from trotterlib.calpha_rpe_accounting import account_raw_radius
    hdf_path=Path(hdf_path)
    if hashlib.sha256(hdf_path.read_bytes()).hexdigest()!='a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958':
        raise ValueError('FIXED_ORIGINAL_HDF5_REQUIRED')
    with h5py.File(hdf_path,'r') as f:nuclear=float(f['nuclear_repulsion'][()])
    profile=json.loads((ROOT/(ART+'run09_sampling_law/gate_profile.json')).read_text())
    identity=profile['native_identity']
    account=json.loads((ROOT/(ART+'run10_rpe_accounting/result.json')).read_text())
    core=json.loads((ROOT/(ART+'run06_reference_interval/result.json')).read_text())
    # Each ordinary native head delegates to exactly the qualified existing
    # block emitter. Only the top scalar is grouped differently: one phase at T.
    b=IntervalBounds(40);c=b.c;rows=[]
    phase_step=Fraction.from_float(float(-.02*nuclear))+Fraction.from_float(float(-.02*identity))
    gamma=b.real(Fraction(core['gamma_upper_exact']))
    for previous in account['rounds']:
        q=previous['q'];native_phase=float(-(nuclear+identity)*(.02*q))
        difference=abs(Fraction.from_float(native_phase)-q*phase_step)
        # Saved vector norm differs from one by <=1e-12 in the fixed loader;
        # 2 is a conservative simple upper bound, not a zero implementation term.
        error=2*b.real(difference)
        d=b.real(Fraction(previous['disk']['upper_exact']))+error
        a=c.exp(-gamma*b.real(.02)*q).a
        radius=(a-d).a
        good=endpoint(d,True)<=endpoint(a*c.sin(b.real('0.14')))
        if not good:raise ValueError('NATIVE_GROUPED_PHASE_BUDGET_FAILED:'+str(previous['m']))
        count=account_raw_radius(endpoint(radius),b)
        rows.append(dict(m=previous['m'],q=q,phase_grouping_error_exact=str(difference),
            native_disk_upper_exact=str(endpoint(d,True)),
            native_radius_lower_exact=str(endpoint(radius)),
            native_phase_upper_float=float(endpoint(c.atan2(d,c.sqrt(a*a-d*d)),True)),
            native_radius_lower_float=float(endpoint(radius)),satisfies_014=True,**count))
    return dict(status='ALL18_NATIVE_EMITTER_IDEAL_GATE_MODEL_QUALIFIED',
        nuclear_scalar=nuclear,tail_identity=identity,rows=rows,
        totals=dict(N_X=sum(x['N_X'] for x in rows),N_Y=sum(x['N_Y'] for x in rows),
                    N_all=sum(x['N_round'] for x in rows)),
        scope='Ordinary controlled native emitter, before transpilation; same exact gate meanings and finite-CDF/IID assumptions.',
        inherited_branch_conditions=account['branch_conditions'],
        native_compiler_and_device='UNKNOWN',new_scientific_operator_constructions=0,new_circuit_builds=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hdf5',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/qualify_calpha_native_scalar.py','src/trotterlib/calpha_rpe_accounting.py',
           'src/trottertracks/resource_applicability/ax2a_native_df.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,status='STARTED')
    try:result.update(evaluate(args.hdf5))
    except Exception as exc:result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:(out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
