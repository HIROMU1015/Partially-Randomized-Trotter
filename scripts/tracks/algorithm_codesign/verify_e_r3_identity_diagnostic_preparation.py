"""Source-bound NEW off-domain focused tests; no registered worker invocation."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12'
sys.path.insert(0,str(ROOT/'src'))


def main():
    from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits
    from trottertracks.algorithm_codesign.df_correction_comparison.worker import runtime_check
    contract=json.loads((ART/'contract_v1.json').read_bytes())
    runtime_check(contract,'identity')  # file metadata; no input reconstruction
    parser=argparse.ArgumentParser();parser.add_argument('--run-index',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    run=ART/('focused_tests_run_%02d'%args.run_index);run.mkdir()
    paths=['tests/tracks/algorithm_codesign/test_e_r3_identity_diagnostic_preparation.py',
        'scripts/tracks/algorithm_codesign/run_e_r3_identity_diagnostic.py',str(Path(__file__).relative_to(ROOT))]
    paths+=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic').glob('*.py'))
    def identities():
        return [{'path':p,'bytes':len((ROOT/p).read_bytes()),'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in paths]
    before=identities();xml=run/'tests.xml'
    command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',paths[0],'--junitxml='+str(xml)]
    os.environ['PYTHONPATH']=str(ROOT/'src');os.environ['E_R3_OFF_DOMAIN_TEST_OUTPUT']=str(run)
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['QISKIT_PARALLEL']='FALSE'
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS']:os.environ[k]='1'
    supervisor=Supervisor(Limits(float('inf'),60,512,448),run,output_cap=8*1024**2,log_cap=1024**2)
    done=run/'pytest_exit.json'
    wrapper='import json,subprocess,sys; r=subprocess.run(sys.argv[2:]); open(sys.argv[1],"x").write(json.dumps({"pytest_exit":r.returncode})); sys.exit(r.returncode)'
    failure=None
    try:
        supervisor.phase([{'label':'focused-tests','output':str(done),
            'command':[sys.executable,'-B','-c',wrapper,str(done),*command]}],Limits(float('inf'),60,512,448),1)
        supervisor.check_total()
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
    suites=[] if not xml.exists() else list(ET.parse(xml).getroot().iter('testsuite'))
    counts={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    result={'kind':'E_R3_DIAGNOSTIC_RUNNER_NEW_OFF_DOMAIN_FOCUSED_TESTS','command':command,
        'source_files':before,'source_files_unchanged_during_tests':before==identities(),
        'test_counts':counts,'failure':failure,'resource_receipt':supervisor.receipt(),
        'runtime_identity':'fixed numeric executable/packages/source-binary trees passed',
        'prior104_tests_repeated':False,'registered_input_array_or_matrix_or_payload_calls':0,
        'native_transpile_body_calls':0,'blocked_off_domain_transpile_entry_probe':1,
        'Clifford_T_CD_E_r6_G10_RA_D0_molecule_NPZ_GPU_LP_trajectory_shots_calls':0,
        'full_test_suite':False,'diagnostic_execution_authorized':False,'science_execution_authorized':False,
        'old_science_retry':0,'mandatory_STOP':True}
    result['passed']=failure is None and counts['tests']>0 and not any(counts[k] for k in ['failures','errors','skipped']) and result['source_files_unchanged_during_tests']
    result['preparation_test_invocation']=args.run_index
    result['repeat_reason']=None if args.run_index==1 else 'Add source precondition violation technical STOP and new dedicated test; original33 PASS receipt and draft source retained; not registered execution retry.'
    with (ART/('focused_tests_v%d.json'%args.run_index)).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'passed':result['passed'],'counts':counts,'failure':failure,'mandatory_STOP':True}))
    if not result['passed']:sys.exit(1)


if __name__=='__main__':main()
