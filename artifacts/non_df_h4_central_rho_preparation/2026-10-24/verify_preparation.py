"""Read-only metadata/SHA checks; no real-H4 numeric functions or QPY imports."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
pkg=types.ModuleType('rho_audit_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['rho_audit_private']=pkg
from rho_audit_private import h4_central_rho_plan as P
C=P.C


def main():
    checks={}
    def ok(name,value):
        C.require(value,name);checks[name]='PASS'
    inp=C.read(BASE/'inputs.json')['files'];old=C.read(BASE/'preservation_before.json')
    C.hashes({str(ROOT/k):v for k,v in inp.items()});ok('input_SHA',True)
    C.hashes({str(ROOT/k):v for k,v in old.items()});ok('old_preservation_SHA',True)
    targets=C.read(BASE/'targets.json')['targets']
    ok('six_unique_targets',len(targets)==6 and len({x['key'] for x in targets})==6)
    ok('three_ids_both_axes',all(len([x for x in targets if x['trajectory_id']==tid])==2 for _,_,tid in P.TARGETS))
    for t in targets:
        for name in ('source_body','controlled_body','input_qpy','compiled_qpy'):
            ok('saved_bytes:'+t['key']+':'+name,C.sha(ROOT/t[name])==inp[t[name]])
        ok('unknown_residual:'+t['key'],t['new_signal'] is None and t['real_rho_new'] is None)
    drafts=C.read(BASE/'contracts_DRAFT.json');P.require_draft(drafts)
    for d in drafts['candidates']:
        P.require_draft(d);ok('disabled_contract:'+d['id'],True)
        ok('no_forbidden_future_calls:'+d['id'],all(v==0 for v in d['forbidden_new_calls'].values()))
    ok('planned_2_plus_4_not_executed',[len(x['targets']) for x in drafts['candidates']]==[2,4,0])
    ok('no_new_grant_seal_ledger',not any(p.name in ('grant.json','seal.json','CONSUMED.json','call_ledger.json','active_contract.json') for p in BASE.rglob('*')))
    inspection=C.read(BASE/'inspection.json')
    ok('no_real_decode_numeric_backend',all(inspection[k]==0 for k in ('QPY_decodes','NPZ_decodes','quantum_or_numeric_backend_imports','real_state_actions','real_signal_evaluations','scientific_grants_issued')))
    ok('scientific_SOURCE_absent',inspection['new_scientific_SOURCE'] is None)
    try:P.execution_request(dict(execution_authorized=True,grant_valid=True))
    except PermissionError:ok('real_execution_unconditionally_refused',True)
    else:raise ValueError('EXECUTION_NOT_DISABLED')
    current=C.read(ROOT/'artifacts/validation_manifest.json');before=C.read(BASE/'manifest_before.json')
    ok('prior_manifest_rows_unchanged',all(current[k]==v for k,v in before.items()))
    changes=subprocess.check_output(['git','diff','--name-only','388b1e525d3cfbeba8f231acd42c813f67a7f531'],cwd=ROOT,text=True).splitlines()
    ok('no_previous_artifact_changes',not any(p.startswith('artifacts/') and not p.startswith(str(BASE.relative_to(ROOT))) and p!='artifacts/validation_manifest.json' for p in changes))
    for p in ('src/trotterlib/h4_central_rho_plan.py','scripts/prepare_h4_central_rho.py','tests/test_h4_central_rho_plan.py'):
        ast.parse((ROOT/p).read_text());ok('AST:'+p,True)
    report=ROOT/'docs/non_df_h4_central_rho_preparation_20261024.md'
    links=re.findall(r'\]\(([^)]+)\)',report.read_text())
    local_links=[(report.parent/x.split('#')[0]).resolve() for x in links if '://' not in x and not x.startswith('#')]
    # The report links to this audit's output, which is created at completion.
    ok('report_links',all(p.exists() or p==BASE/'verification.json' for p in local_links))
    for p in BASE.rglob('*.json'):C.read(p)
    ok('all_JSON_parse',True)
    for fn,number in [('artificial15_PASS.log',15),('reused_D2_12_PASS.log',12),('reused_semantics8_PASS.log',8)]:
        log=(BASE/fn).read_text();ok('artificial_log:'+fn,f'Ran {number} tests' in log and log.rstrip().endswith('OK'))
    ok('first_artificial_failure_preserved','FAILED (failures=1)' in (BASE/'artificial_first_failure.log').read_text())
    ok('audit_no_numeric_imports',not any(x.startswith(('qiskit','numpy','flint','pyscf','openfermion')) for x in sys.modules))
    result=dict(status='PASS_PREPARATION_ONLY',checks=checks,counts=dict(checks=len(checks),targets=6,inputs=len(inp),
        preserved_files=len(old),prior_manifest_rows=len(before),report_links=len(links),artificial_tests=35),
        real_H4_QPY_NPZ_decodes=0,real_H4_state_signal_calls=0,real_N_G=0,new_scientific_SOURCE=None,
        grants_issued=0,formal_certificate_accepted=False,scope='Primary-agent metadata/SHA checks and artificial tests; no independent scientific review',mandatory_STOP=True)
    C.save(BASE/'verification.json',result)
    C.require(all(p.exists() for p in local_links),'FINAL_REPORT_LINK_EXISTS')
    print(json.dumps(dict(status=result['status'],counts=result['counts'])))


if __name__=='__main__':main()
