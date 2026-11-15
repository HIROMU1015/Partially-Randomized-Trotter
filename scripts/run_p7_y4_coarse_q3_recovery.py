"""One approved coarse-Y4 recovery using unchanged saved inputs and AST guard."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_signal_q3 as O

CALLER='scripts/run_p7_y4_coarse_q3_recovery.py'
STUDY='docs/research/d_p7_y4_coarse_q3_recovery_20261116'
OUT=ROOT/STUDY
PYTHON='/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python'
FIXED=O.ref('278fb5214f1091efba2a05b305bc05190cd18ff6',O.STUDY+'/Q3_INPUTS.json')
PACKET=O.ref('f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53',O.STUDY+'/BOUND_SIGNAL_INPUT.json')
OLD='34783cd09098f86464ca4bbf2ede917994b9269e'


def runtime_check(F):
    O.require(Path(sys.executable).resolve()==Path(PYTHON).resolve(),'Use the fixed compatible dedicated Python')
    O.require((ROOT/O.CALLER).read_bytes()==O.blob(OLD,O.CALLER),'Original stopped caller changed')
    F.check_source()
    for path in ('src/trottertracks/rpe_total_risk.py','src/trottertracks/rpe_total_dp_parallel.py'):
        O.require((O.RUNTIME/path).read_bytes()==O.blob(O.B,path),'Pinned kernel/guard bytes differ')
    # Same constructor used by the PF worker; original hash check is executed.
    F.R.Observed(OUT,'preflight_constructor',workers=1,outcome_batch=8)
    node=ast.parse(textwrap.dedent(inspect.getsource(F.R.P.R.total_state_dp))).body[0]
    actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    O.require(actual==F.R.P.REFERENCE_DP_AST_SHA256,'Original AST guard mismatch')
    return dict(executable=sys.executable,resolved_executable=os.path.realpath(sys.executable),
        version=sys.version,platform=platform.platform(),expected_AST_hash=F.R.P.REFERENCE_DP_AST_SHA256,
        actual_AST_hash=actual,constructor='PASS_UNCHANGED_GUARD',guard_modified=False,
        kernel_source=F.R.P.REFERENCE_SOURCE,accelerator_source=F.R.ACCELERATOR,
        pinned_code_snapshot=O.B,D2_precedent='8ec7c8828')


def fixed_inputs(F):
    inputs=O.read(FIXED)
    O.require(len(inputs['cases'])==inputs['maximum_scientific_calls']==1,'Exactly one original case')
    case=inputs['cases'][0]
    O.require(case['configuration_id']==O.CID and case['signal_packet']==PACKET,'Wrong saved candidate/packet')
    O.require([r['N_X'] for r in case['allocation']['shots']]==O.VECTOR==
              [r['N_Y'] for r in case['allocation']['shots']],'Wrong fixed allocation')
    F.metadata(case)
    return inputs


def prepare():
    F=O.q3_runtime();env=runtime_check(F);inputs=fixed_inputs(F)
    OUT.mkdir(parents=True,exist_ok=True)
    O.write(OUT/'INPUTS.json',dict(original_fixed_inputs=FIXED,signal_packet=PACKET,inputs=inputs,
        runtime=env,previous_STOP=O.ref(OLD,O.STUDY+'/q3_run01/COMPLETION.json'),
        maximum_scientific_calls=1,new_signal=0,new_reference=0,build=0,compile=0,price=0,fit=0,
        additional_allocation=0,retries=0,output=STUDY+'/run01'))
    print('RECOVERY_PREPARED_FIXED_INPUTS',env,flush=True)


def execute():
    F=O.q3_runtime();src=O.source(STUDY+'/INPUTS.json')
    src['caller']=CALLER
    O.require((ROOT/CALLER).read_bytes()==O.blob(src['commit'],CALLER),'Unfixed recovery caller')
    package=json.loads((OUT/'INPUTS.json').read_text())
    O.require(package['inputs']==fixed_inputs(F) and package['runtime']==runtime_check(F),'Fixed runtime/input changed')
    delta=json.loads((OUT/'DELTA_CHECK.json').read_text())
    O.require(not delta['equivalent_results'] and not delta['matching_jobs'] and not delta['unresolved'],'Resolve duplicate work first')
    inputs=package['inputs'];case=inputs['cases'][0];out=OUT/'run01';out.mkdir();(out/case['label']).mkdir()
    src.update(runtime=package['runtime'],original_fixed_inputs=FIXED,signal_packet=PACKET)
    O.write(out/'EXECUTION.json',dict(source=src,started_UTC=O.utc(),maximum_scientific_calls=1,
        previous_STOP=package['previous_STOP'],price_calls=0,new_signal=0,retries=0))
    with F.installed_workers(out):result=F.M.supervise([case],inputs,src,out)
    exits=[c for p in out.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
           for c in json.loads(line)['children']]
    gate=dict(recorded_children=len(exits),all_exit_codes_zero=bool(exits) and all(c['exitcode']==0 for c in exits))
    ok=(not result['errors'] and not result['monitor_errors'] and gate['all_exit_codes_zero'] and
        len(result['receipts'])==1 and result['receipts'][0]['scientific_evaluation_calls']==1 and
        result['receipts'][0]['api_return_saved'] and result['receipts'][0]['status']!='UNKNOWN')
    O.write(out/'COMPLETION.json',dict(status='ONE_FIXED_Q3_SAVED' if ok else 'UNKNOWN_STOP_NO_RETRY',
        source=src,**result,process_exit_gate=gate,finished_UTC=O.utc(),coarse_cost=None,mandatory_stop=True))
    print('ONE_FIXED_Q3_SAVED' if ok else 'UNKNOWN_STOP_NO_RETRY',flush=True)
    return 0 if ok else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    if parser.parse_args().mode=='prepare':prepare()
    else:raise SystemExit(execute())
