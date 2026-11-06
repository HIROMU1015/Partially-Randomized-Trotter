"""One approved B operator-only invocation with physical model UNASSESSED.

Reuses fixed 7f9aea5 evaluate/criteria. No state, dynamics, fit, circuit or cost.
The user adopts B only for this numerical check, not as a physical redefinition.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import validate_o1_m2_hermitian_b as fixed
from trottertracks import o1_m2_residual_adapter as adapter

BASE='7f9aea5f22e0e19dc3be002f6f35d4b9648fd872'
INPUT_COMMIT='aced1647b81e8d28091058cdc7b6e3e1ee54896a'
FIXED_FILES=('scripts/validate_o1_m2_hermitian_b.py','scripts/validate_o1_m2_h4_operator.py',
    'src/trottertracks/o1_m2_residual_adapter.py','src/trottertracks/o1_m2_hermitian_b.py',
    'src/trottertracks/o1_m2_cutoff_free_pauli.py')


def supplemental_diagnostics(raw):
    """Requested full/sector closure, model changes and one-body bookkeeping.

    This is part of the same invocation, computed before the fixed final
    qualification. It does not introduce new tolerances or qualify model shifts.
    """
    import math
    import numpy as np
    from trottertracks.c1_stage1 import sector_indices
    candidate=fixed.b.derive_hermitian_b(raw)
    n=len(raw.physical.h)
    ref=fixed.previous.CARReference(n)
    names=('physical','core','residual')
    original={name:fixed.fsum_car(ref,getattr(raw,name)) for name in names}
    projected={name:fixed.fsum_car(ref,getattr(candidate,name)) for name in names}
    projected_raw=fixed.fsum_car(ref,candidate.projected_raw_residual)
    spaces={'full256':np.arange(1<<(2*n)), 'sector36':sector_indices(n,*raw.sector)}
    def maxima(matrix):
        return {name:float(np.max(abs(matrix[np.ix_(indices,indices)])))
                for name,indices in spaces.items()}
    kappa=np.array([[raw.physical.h[p,q]-.5*math.fsum(float(raw.physical.g[p,r,r,q]) for r in range(n))
                     for q in range(n)] for p in range(n)])
    residual_contraction=np.array([[.5*math.fsum(float(raw.residual.g[p,r,r,q]) for r in range(n))
                                   for q in range(n)] for p in range(n)])
    return dict(canonical_C_plus_R_minus_H=maxima(projected['core']+projected['residual']-projected['physical']),
        canonical_R_minus_P_raw_R=maxima(projected['residual']-projected_raw),
        model_change_max_entry={name:maxima(projected[name]-original[name]) for name in names},
        raw_antihermitian_half_max_entry={name:maxima((matrix-matrix.conj().T)/2)
                                        for name,matrix in original.items()},
        one_body=dict(original_g_kappa_max_error=float(np.max(abs(kappa-np.asarray(raw.general_core.spatial_one_body_tensor)))),
            raw_R_one_body_vs_half_R_contraction_max_error=float(np.max(abs(raw.residual.h-residual_contraction))),
            raw_R_h_max=float(np.max(abs(raw.residual.h))),raw_R_g_max=float(np.max(abs(raw.residual.g))),
            raw_R_g_sign_max_error=float(np.max(abs(raw.residual.g-(raw.physical.g-raw.core.g)))),
            projection_h_max_change={name:float(np.max(abs(getattr(candidate,name).h-getattr(raw,name).h))) for name in names}),
        all_saved_frames=len(raw.general_core.frames),sector=raw.sector,
        model_change_status='UNASSESSED',new_criterion_added=False)


def run(core_path,input_path):
    started=datetime.now(timezone.utc).isoformat();tick=time.perf_counter()
    hashes={}
    for name in FIXED_FILES:
        content=(ROOT/name).read_bytes()
        historical=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        if content!=historical: raise ValueError('Fixed SOURCE differs: '+name)
        hashes[name]=hashlib.sha256(content).hexdigest()
    hashes['scripts/run_o1_m2_b_limited_operator.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Plan is the unmodified SOURCE-fixed H4_PLAN, established before input decode.
    plan=fixed.H4_PLAN
    raw=adapter.load_saved_pair(core_path,input_path)
    supplemental=supplemental_diagnostics(raw)
    result=fixed.evaluate(raw,plan=plan,projection_limits=None)
    assert result['model_change_status']=='UNASSESSED'
    result.update(status='OPERATOR_CHECKS_PASS_MODEL_UNASSESSED' if result['operator_checks_passed'] else 'NOT_QUALIFIED',
        numeric_convention_B_adopted_for_this_validation=True,physical_hamiltonian_redefined=False,
        user_model_limits=None,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-tick,host=platform.node(),
        science_source=dict(fixed_commit=BASE,files_sha256=hashes),
        inputs=dict(commit=INPUT_COMMIT,binding=asdict(adapter.UPDATE005),
            core=str(core_path.relative_to(ROOT)),physical=str(input_path.relative_to(ROOT))),
        runtime=dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','openfermion','qiskit')}),
        supplemental=supplemental,
        mandatory_stop=True,scope_completed='single B operator-only invocation; no state/fit/dynamics/circuits/sampling/shots/cost')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Reserve a fresh result name before any decode; a failed invocation is not
    # silently repeated and cannot overwrite a previous raw result.
    with args.output.open('x',encoding='utf-8') as stream:
        try:
            result=run(args.core.resolve(),args.input.resolve())
        except Exception as exc:
            json.dump(dict(status='ERROR_STOP',error_type=type(exc).__name__,error=str(exc),mandatory_stop=True),stream,indent=2)
            stream.write('\n');raise
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return 0 if result['operator_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
