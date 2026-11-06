"""Seal the bounded-metric arithmetic repair without scientific execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from seal_research_c_h6 import closure

ROOT=Path(__file__).resolve().parents[1]
BASE='d166c75c407f4b2f74f239018fe4bbc5cf9dd390'
PARENT_CONTRACT='docs/research/research_c_h6_conditionfix_contract_20261106.json'
CONTRACT='docs/research/research_c_h6_stablemetric_contract_20261106.json'
SOURCE='artifacts/research_c_h6_stablemetric/2026-11-06/execution_preflight/SOURCE.json'


def main():
    c=json.loads(subprocess.check_output(['git','show',BASE+':'+PARENT_CONTRACT],cwd=ROOT,text=True))
    c.update(status='FIXED_H6_STABLE_METRIC_REPAIR',execution_SOURCE=SOURCE,execution_SOURCE_sha256=None,
        fresh_output='artifacts/research_c_h6_independent_validation/2026-11-06-stablemetric/run01')
    version='H6_known_gauge_bounded_metric_centered_trust_exact_v2'
    c['numerical_repair'].update(backend=version,parent_commit=BASE,
        parent_SOURCE_sha256='d177976d3a8d644f19bec393c2231eb08928fe4e27f5385716769cd80eecdf0c')
    c['numerical_repair']['source_changes'] += [
        'inverse metric scales bounded with floor=sigma_max*sqrt(binary64 eps); all nongauge directions retain positive scaling, no truncated SVD or objective ridge',
        'centered cost anchored at actual decoded origin, including pack/unpack binary64 rounding',
        'saved failed-point regression binds exact stored x/J; original value and gradient checks remain, inconsistent points not relabeled PASS']
    c['optimization']['numerical_backend']=version
    roots=['scripts/run_research_c_h6.py','scripts/seal_research_c_h6_stablemetric.py','scripts/seal_research_c_h6.py',
        'scripts/regress_research_c_h6_stablemetric.py','scripts/prepare_research_c_h6_validation.py',
        'src/trotterlib/df_hamiltonian.py','src/trotterlib/rte.py','src/trottertracks/c_h6_runtime.py',
        'src/trottertracks/c_h6_fit_conditioning.py','src/trottertracks/c_h6_downstream.py',
        'src/trottertracks/c_h6_parallel.py','src/trottertracks/c_h6_contract.py',
        'tests/management/test_research_c_h6_conditioning.py','tests/management/test_research_c_h6_runner.py']
    sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.c_h6_contract import configuration
    files=closure(ROOT,roots)
    manifest=dict(schema='H6_execution_SOURCE_v1',base_commit=BASE,
        files={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        contract=configuration(c),dynamic_modules=['trotterlib.df_hamiltonian','trotterlib.rte','literal _legacy targets recursively included'],
        scope='bounded inverse-metric arithmetic repair; scientific scope and acceptance tolerances retained',
        preserved_parent_SOURCE='artifacts/research_c_h6_conditionfix/2026-11-06/execution_preflight/SOURCE.json')
    target=ROOT/SOURCE;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    c['execution_SOURCE_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT/CONTRACT).write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),SOURCE_sha256=c['execution_SOURCE_sha256'])))


if __name__=='__main__':main()
