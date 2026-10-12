"""One guarded regression batch for the repaired observer; registered calls zero."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12'
ART=BASE/'v2'
sys.path.insert(0,str(ROOT/'src'))

def main():
    from trottertracks.algorithm_codesign.df_correction_comparison.worker import runtime_check
    from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits
    contract=json.loads((ART/'contract_v1.json').read_bytes())
    runtime_check(contract,'identity')
    files={r['path'] for r in json.loads((BASE/'source_manifest_v1.json').read_bytes())['files']}
    tests=['tests/tracks/algorithm_codesign/test_e_r3_identity_diagnostic_preparation.py',
           'tests/tracks/algorithm_codesign/test_e_r3_identity_final_technical.py',
           'tests/tracks/algorithm_codesign/test_e_r3_identity_observer_repair.py']
    files.update(tests+[str(Path(__file__).relative_to(ROOT)),
                       'scripts/tracks/algorithm_codesign/prepare_e_r3_identity_observer_repair.py',
                       str((ART/'contract_v1.json').relative_to(ROOT))])
    def identities():
        return [{'path':p,'bytes':len((ROOT/p).read_bytes()),'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
                for p in sorted(files)]
    before=identities();run=ART/'focused_run_01';run.mkdir()
    xml=run/'tests.xml';done=run/'pytest_exit.json'
    env={'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1','QISKIT_PARALLEL':'FALSE',
         'E_R3_FINAL_AUDIT_TEST_OUTPUT':str(run),'E_R3_OFF_DOMAIN_TEST_OUTPUT':str(run),
         'E_R3_OBSERVER_REPAIR_OUTPUT':str(run)}
    env.update({k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']})
    os.environ.update(env)
    synthetic_tmp=Path(tempfile.mkdtemp(prefix='e-r3-observer-repair-tests-'))
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',*tests,
             '-k','not reproduce_profile_lifecycle_failure_misclassified_as_observation',
             '--basetemp='+str(synthetic_tmp),'--junitxml='+str(xml)]
    wrapper='import json,subprocess,sys; r=subprocess.run(sys.argv[2:]); open(sys.argv[1],"x").write(json.dumps({"pytest_exit":r.returncode})); sys.exit(r.returncode)'
    supervisor=Supervisor(Limits(float('inf'),60,512,448),run,output_cap=8388608,log_cap=1048576)
    failure=None
    try:
        supervisor.phase([{'label':'observer-repair-tests','output':str(done),
                          'command':[sys.executable,'-B','-c',wrapper,str(done),*command]}],Limits(float('inf'),60,512,448),1)
        supervisor.check_total()
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
    suites=[] if not xml.exists() else list(ET.parse(xml).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    entries=[t.get('name') for t in ET.parse(xml).getroot().iter('testcase')] if xml.exists() else []
    result={'kind':'OBSERVER_REPAIR_V2_SOURCE_BOUND_LOCAL_TESTS','command':command,
            'previous_science_source':'8377c1d12439cabab3b97bcc1f57ac030a479785',
            'test_counts':counts,'test_cases':entries,'failure':failure,'source_files':before,
            'source_files_unchanged_during_tests':before==identities(),
            'resource_receipt':supervisor.receipt(),'runtime_identity':contract['runtimes']['numeric'],
            'existing_preparation_regressions':34,'previous_final_audit_selected_regressions':26,
            'historical_bug_reproduction_tests_deselected':2,'new_repair_controls':17,
            'registration_array_matrix_payload_calls':0,'registered_runner_worker_launches':0,
            'native_transpile_body_calls':0,'Clifford_T_science_molecule_NPZ_GPU_LP_calls':0,
            'test_only_blocked_native_entry_probes':2,'synthetic_temporary_directory':str(synthetic_tmp),
            'runs':0,'retries':0,
            'diagnostic_execution_authorized':False,'science_execution_authorized':False,
            'full_test_suite':False,'mandatory_STOP':True}
    result['passed']=failure is None and counts['tests']==77 and not any(counts[k] for k in ['failures','errors','skipped']) and result['source_files_unchanged_during_tests']
    with (ART/'focused_tests_v2.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['passed','test_counts','failure','mandatory_STOP']}))
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
