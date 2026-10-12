"""Guarded ADDITIONAL off-domain/mock validation; no registered worker launch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/track_b_e_r3_identity_final_technical_audit/2026-10-12'
PREP=ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12'
sys.path.insert(0,str(ROOT/'src'))

def main():
    from trottertracks.algorithm_codesign.df_correction_comparison.worker import runtime_check
    from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits
    contract=json.loads((PREP/'contract_v1.json').read_bytes())
    runtime_check(contract,'identity')
    files=json.loads((PREP/'source_manifest_v1.json').read_bytes())['files']
    added=['tests/tracks/algorithm_codesign/test_e_r3_identity_final_technical.py',str(Path(__file__).relative_to(ROOT)),
           'scripts/tracks/algorithm_codesign/audit_e_r3_identity_final_technical.py']
    def identities():
        return [{'path':p,'bytes':len((ROOT/p).read_bytes()),'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
                for p in [r['path'] for r in files]+added]
    before=identities()
    run=ART/'additional_focused_run_01';run.mkdir(parents=True)
    xml=run/'tests.xml';done=run/'pytest_exit.json'
    os.environ['PYTHONPATH']=str(ROOT/'src');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['E_R3_FINAL_AUDIT_TEST_OUTPUT']=str(run)
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']:
        os.environ[k]='1'
    os.environ['QISKIT_PARALLEL']='FALSE'
    tests=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',added[0],'--junitxml='+str(xml)]
    wrapper='import json,subprocess,sys; r=subprocess.run(sys.argv[2:]); open(sys.argv[1],"x").write(json.dumps({"pytest_exit":r.returncode})); sys.exit(r.returncode)'
    supervisor=Supervisor(Limits(float('inf'),60,512,448),run,output_cap=8388608,log_cap=1048576)
    failure=None
    try:
        supervisor.phase([{'label':'additional-focused-tests','output':str(done),
                          'command':[sys.executable,'-B','-c',wrapper,str(done),*tests]}],Limits(float('inf'),60,512,448),1)
        supervisor.check_total()
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
    suites=[] if not xml.exists() else list(ET.parse(xml).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    result={'kind':'ADDITIONAL_FROZEN_S_MOCK_AND_OFF_DOMAIN_TESTS',
            'frozen_source':'8377c1d12439cabab3b97bcc1f57ac030a479785','command':tests,
            'source_files':before,'protected_source_unchanged':before==identities(),
            'test_counts':counts,'failure':failure,'resource_receipt':supervisor.receipt(),
            'runtime_identity_checked':contract['runtimes']['numeric'],
            'previous34_PASS': 'saved receipt verified; NOT rerun',
            'known_bug_reproductions_can_PASS_tests_while_blocking_readiness':True,
            'registered_diagnostic_calls':0,'native_transpile_body_calls':0,
            'additional_blocked_off_domain_native_entry_probe':1,
            'registration_authorization_commit_created':False,'science_calls':0,'mandatory_STOP':True}
    result['test_harness_passed']=failure is None and counts['tests']>0 and not any(counts[k] for k in ['failures','errors','skipped']) and result['protected_source_unchanged']
    with (ART/'additional_focused_tests_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['test_harness_passed','test_counts','failure','mandatory_STOP']}))
    if not result['test_harness_passed']:raise SystemExit(1)

if __name__=='__main__':main()
