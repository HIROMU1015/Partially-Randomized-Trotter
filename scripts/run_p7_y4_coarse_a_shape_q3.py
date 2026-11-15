"""One approved coarse-Y4 A_SHAPE Q3; saved signal and original PF worker."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_q3_recovery as R
O=R.O
CALLER='scripts/run_p7_y4_coarse_a_shape_q3.py'
STUDY='docs/research/d_p7_y4_coarse_a_shape_q3_20261116'
OUT=ROOT/STUDY
BASE=O.ref('278fb5214f1091efba2a05b305bc05190cd18ff6',O.STUDY+'/Q3_INPUTS.json')
INTEGER_SOURCE=O.ref(O.B,O.B_DIR+'INPUTS.json','cases',8,'allocation')
PREVIOUS=O.ref('e4972e62fbdfa75aadccb5b5f8181e40ba5d7cbc',R.STUDY+'/run01/COMPLETION.json')
LABEL='B0I_P7_Y4_32h_over_5__A_SHAPE'
ALLOCATION_ID='B0I_P7_Y4_32H_OVER5_A_SHAPE_SAVED_INTEGERS_01'
VECTOR=[37,34,32,29,27,24,22,19,17,14,11,9,6]


def validate_case(case):
    O.require(case['label']==LABEL and case['configuration_id']==O.CID,'Only coarse Y4 A_SHAPE is allowed')
    O.require(case['signal_packet']==R.PACKET,'Wrong saved coarse signal')
    allocation=case['allocation']
    O.require(allocation['allocation_id']==ALLOCATION_ID and allocation['shot_policy']=='USER_SUPPLIED','Wrong allocation identity')
    O.require(allocation['shots']==[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)],'Wrong fixed m/axis/vector')


def fixed_inputs(F):
    inputs=O.read(BASE);case=inputs['cases'][0]
    prior=O.read(O.ref(O.B,O.B_DIR+'INPUTS.json'))
    selected=[(i,c) for i,c in enumerate(prior['cases']) if c['label']=='B0I_P7_Y4_16h_over_3__A_SHAPE']
    O.require(len(selected)==1 and selected[0][0]==8,'Saved A_SHAPE source selection differs')
    old=selected[0][1]
    O.require(old['configuration_id']==O.CURRENT and old['allocation']==O.read(INTEGER_SOURCE),'Wrong integer source')
    O.require(old['evaluation_target']==case['evaluation_target'],'Original target differs')
    case.update(label=LABEL,allocation=deepcopy(old['allocation']))
    case['allocation']['allocation_id']=ALLOCATION_ID
    validate_case(case);F.metadata(case)
    for key in ('kernel_source','C_source','statistical_contract','options','adopted_evaluation','acceleration'):
        O.require(inputs[key]==prior[key],'Original evaluation contract differs: '+key)
    bounds=F.count_bounds(case['allocation']['shots'])
    O.require(bounds['maximum_round_outcomes']<=inputs['options']['maximum_round_outcomes'] and
        bounds['verification_pairs_upper']<=inputs['options']['maximum_verification_pairs'] and
        bounds['transition_pairs_upper']<=inputs['options']['total_state_refinement']['maximum_transition_pairs'],'Count-only finite scope exceeded')
    inputs.update(finite_scope_preflight=bounds,saved_current_comparison=O.ref(O.B,
        O.B_DIR+'run01/B0I_P7_Y4_16h_over_3__A_SHAPE/RESULT.json'))
    O.require(len(inputs['cases'])==inputs['maximum_scientific_calls']==inputs['actual_new_call_limit']==1,'Only one scientific call')
    return inputs


def prepare():
    F=O.q3_runtime();env=R.runtime_check(F,out=OUT);inputs=fixed_inputs(F)
    OUT.mkdir(parents=True,exist_ok=True)
    O.write(OUT/'INPUTS.json',dict(base_evaluation_inputs=BASE,integer_allocation_source=INTEGER_SOURCE,
        signal_packet=R.PACKET,inputs=inputs,runtime=env,previous_STOP=PREVIOUS,
        maximum_scientific_calls=1,new_signal=0,new_reference=0,build=0,compile=0,price=0,fit=0,
        additional_allocation=0,retries=0,output=STUDY+'/run01'))
    print('A_SHAPE_PREPARED',LABEL,inputs['finite_scope_preflight'],flush=True)


def execute():
    F=O.q3_runtime();src=O.source(STUDY+'/INPUTS.json');src['caller']=CALLER
    for path in (CALLER,R.CALLER):
        O.require((ROOT/path).read_bytes()==O.blob(src['commit'],path),'Unfixed caller/helper: '+path)
    package=json.loads((OUT/'INPUTS.json').read_text())
    O.require(package['inputs']==fixed_inputs(F) and package['runtime']==R.runtime_check(F,out=OUT),'Fixed runtime/input changed')
    delta=json.loads((OUT/'DELTA_CHECK.json').read_text())
    O.require(not delta['equivalent_results'] and not delta['matching_jobs'] and not delta['unresolved'],'Resolve duplicate work first')
    src.update(runtime=package['runtime'],base_evaluation_inputs=BASE,integer_allocation_source=INTEGER_SOURCE,signal_packet=R.PACKET)
    return R.run_case(F,package['inputs'],src,OUT/'run01',PREVIOUS)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    if parser.parse_args().mode=='prepare':prepare()
    else:raise SystemExit(execute())
