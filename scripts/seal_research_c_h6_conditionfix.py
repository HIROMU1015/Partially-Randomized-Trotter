"""Seal the explicitly requested conditioning repair; no numerical execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from seal_research_c_h6 import closure

ROOT=Path(__file__).resolve().parents[1]
BASE='30c6f47529d708a0dd1f67818fb455325b084fc8'
OLD_CONTRACT='docs/research/research_c_h6_contract_unapproved_20261106.json'
CONTRACT='docs/research/research_c_h6_conditionfix_contract_20261106.json'
SOURCE='artifacts/research_c_h6_conditionfix/2026-11-06/execution_preflight/SOURCE.json'


def main():
    c=json.loads(subprocess.check_output(['git','show',BASE+':'+OLD_CONTRACT],cwd=ROOT,text=True))
    c.update(status='FIXED_H6_CONDITIONING_REPAIR',execution_SOURCE=SOURCE,
        execution_SOURCE_sha256=None,
        fresh_output='artifacts/research_c_h6_independent_validation/2026-11-06-conditionfix/run01')
    c['numerical_repair']=dict(backend='H6_known_gauge_full_metric_centered_trust_exact_v1',
        parent_commit=BASE,parent_SOURCE_sha256='0be3f9e073172018fc51fe30433dea9280c6d415a1fcfdd1d94ccabd61ac46ef',
        source_changes=[
            'verify existing SVD reconstruction/orthogonality against longdouble physical head map; original safety factors256/1024 retained',
            'early structural rejection before optimizer; known LD-1 gauge representative fixed, no nongauge direction dropped',
            'full nonsingular metric on quotient, longdouble coordinate accumulation, centered objective differences at actual binary64 arrays',
            'SciPy trust-exact with existing analytic smooth-mass Hessian; j_maxiter1000; internal gtol1e-13; same both arms',
            'search-only analytic Hessian assembled with BLAS binary64; same-array LD primal/gradients and all independent acceptance checks retained; exact last-point primal cache',
            'unchanged independent same-array value/gradient, J threshold1e-5, U refinement-shift1e-6, physical DF, outer64/512 and all downstream conditions'],
        no_new_penalty=True,no_nongauge_truncation=True,scientific_comparison_conditions_unchanged=True,
        old_run_resume=0,old_run_retry=0)
    c['order'][0]='User requested numerical repair and fresh rerun; preserve parent SOURCE/raw/STOP; no approval artifact'
    c['decisions_before_execution']=[]
    c['optimization']['numerical_backend']=c['numerical_repair']['backend']
    c['optimization']['inner_optimizer']='SciPy1.14.1 trust-exact; quotient/full metric/centered cost; same scientific objective and gates'
    roots=['scripts/run_research_c_h6.py','scripts/seal_research_c_h6_conditionfix.py','scripts/seal_research_c_h6.py',
        'scripts/prepare_research_c_h6_validation.py','src/trotterlib/df_hamiltonian.py','src/trotterlib/rte.py',
        'src/trottertracks/c_h6_runtime.py','src/trottertracks/c_h6_fit_conditioning.py',
        'src/trottertracks/c_h6_downstream.py','src/trottertracks/c_h6_parallel.py',
        'src/trottertracks/c_h6_contract.py','tests/management/test_research_c_h6_conditioning.py',
        'tests/management/test_research_c_h6_runner.py']
    sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.c_h6_contract import configuration
    files=closure(ROOT,roots)
    manifest=dict(schema='H6_execution_SOURCE_v1',base_commit=BASE,
        files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        contract=configuration(c),dynamic_modules=['trotterlib.df_hamiltonian','trotterlib.rte','literal _legacy targets recursively included'],
        scope='explicitly requested H6 conditioning repair and same-condition fresh rerun',
        preserved_parent_SOURCE='artifacts/research_c_h6_runner/2026-11-06/execution_preflight/SOURCE.json')
    target=ROOT/SOURCE;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    c['execution_SOURCE_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT/CONTRACT).write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),SOURCE_sha256=c['execution_SOURCE_sha256'])))


if __name__=='__main__':main()
