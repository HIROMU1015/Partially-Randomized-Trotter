"""One approved p7 S2 requested-width 0.1 Q3; same shots as saved coarse B."""
import argparse
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_p7_d01_reused_budget_caller',ROOT/'scripts/run_pf6_cost_budget_q3_fixed.py')
F=importlib.util.module_from_spec(spec);sys.modules[spec.name]=F;spec.loader.exec_module(F)
R,T,M=F.R,F.T,F.M
BASE='b4296b110b129d226dbef2743c14f725a27c3bf6'
READY='99930491819cabd94609c42005a8637ac06a787d'
DIRECTORY='docs/research/p7_s2_d01_same_shots_q3_results_20261115'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_p7_s2_d01_same_shots_q3_fixed.py'
LABEL='B0I_P7_S2_D01'
CID='e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc'
VECTOR=[59,55,51,47,43,39,35,31,27,23,19,13,4]
COMPARE_LABEL='B0I_P7_S2_16h_over_7__B_PRICE'
COMPARE_DIR='docs/research/pf6_cost_budget_q3_results_20261115/'


def check_source():
    F.check_source()
    for path in ('scripts/run_pf6_cost_budget_q3_fixed.py','scripts/summarize_pf_shortlist_total_q3_fixed.py'):
        if (ROOT/path).read_bytes()!=R.blob(BASE,path):
            raise ValueError('Reused caller SOURCE differs: '+path)


def build_inputs():
    check_source()
    prior=F.document(BASE,COMPARE_DIR+'INPUTS.json')
    coarse=next(c for c in prior['cases'] if c['label']==COMPARE_LABEL)
    if [r['N_X'] for r in coarse['allocation']['shots']]!=VECTOR or [r['N_Y'] for r in coarse['allocation']['shots']]!=VECTOR:
        raise ValueError('Comparison shot vector differs')
    if coarse['conditional_cost']!=435481090:
        raise ValueError('Saved coarse-only cost differs')
    ready=F.document(READY,'docs/research/rpe_required_n_cross_method_preparation_20261114/READINESS.json')
    entry=next(s for s in ready['series'] if s['series']=='TRUNCATED_PF_P7_B0I')
    if entry['original_configuration_id']!=CID:
        raise ValueError('Fine-width original configuration mismatch')
    case=dict(label=LABEL,configuration_id=CID,total_view_id=entry['total_configuration_id'],
        binding_id=entry['binding_id'],configuration=entry['configuration_source'],signal_packet=entry['packet_source'],
        allocation=dict(allocation_id='B0I_P7_S2_D01_SAME_N_AS_COARSE_BPRICE_01',shot_policy='USER_SUPPLIED',
            shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)]))
    binding,view=F.metadata(case)
    config=binding['original_configuration'];old=T._selected(ROOT,coarse['configuration'])
    if (config['spec']['formula']!='2nd' or config['spec']['L_D']!=7
            or config['spec']['delta_pf']!=config['spec']['h']
            or [r['q_PF'] for r in config['rounds']]!=[1<<m for m in range(13)]):
        raise ValueError('Original fine-width schedule differs')
    if config['spec']['binding']!=old['spec']['binding']:
        raise ValueError('Full physical H/state/snapshot/profile binding differs')
    if {k:v for k,v in config['spec'].items() if k!='delta_pf'}!={k:v for k,v in old['spec'].items() if k!='delta_pf'}:
        raise ValueError('Configuration fields other than requested width differ')
    target=view['signal_packet']['target']
    if target['binding']!=coarse['evaluation_target']['binding']:
        raise ValueError('Evaluation target physical binding differs')
    case['evaluation_target']=deepcopy(coarse['evaluation_target'])
    case['evaluation_target'].update(source=deepcopy(target['source']),
        original_target_source=deepcopy(target['source']),truth_proxy_source=deepcopy(target['truth_proxy_source']))
    case['physical_rows']=[dict(m=r['m'],q_PF=r['q_PF'],T=r['candidate']['T'],actual_delta_pf=r['candidate']['delta'],
        physical_candidate=view['signal_packet']['means'][r['m']]['provider_binding']['physical_context']['physical_candidate'])
        for r in config['rounds']]
    bounds=F.count_bounds(case['allocation']['shots'])
    options=deepcopy(prior['options'])
    if (bounds['maximum_round_outcomes']>options['maximum_round_outcomes']
            or bounds['verification_pairs_upper']>options['maximum_verification_pairs']
            or bounds['transition_pairs_upper']>options['total_state_refinement']['maximum_transition_pairs']):
        raise ValueError('Count-only scope exceeded')
    options['model_loss_contract']['scope']='ONE_P7_S2_D01_SAME_SHOTS_ACCURACY_COMPARISON'
    contract=deepcopy(prior['statistical_contract']);contract['id']='P7_S2_D01_SAME_VECTOR_AS_COARSE_TOTAL_Q3'
    return dict(cases=[case],maximum_scientific_calls=1,actual_new_call_limit=1,concurrent_configurations=1,
        comparison=dict(publication=BASE,label=COMPARE_LABEL,
            API_source=F.ref(BASE,COMPARE_DIR+'run01/'+COMPARE_LABEL+'/API_RETURN.json'),
            summary_source=F.ref(BASE,COMPARE_DIR+'SUMMARY.json'),
            coarse_conditional_cost_RZ=435481090,coarse_cost_applies_to='S2_16h_over_7_ONLY'),
        kernel_source=prior['kernel_source'],C_source=prior['C_source'],
        statistical_contract=contract,options=options,adopted_evaluation=deepcopy(prior['adopted_evaluation']),
        acceleration=deepcopy(prior['acceleration']),finite_scope_preflight=bounds,
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',signal_recomputed=False,conditional_cost=None,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def prepare():
    x=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,x)
    print('Fixed fine-width case:',x['cases'][0]['configuration_id'],x['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):
            raise ValueError('Caller/INPUTS not committed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs() or len(inputs['cases'])!=1 or inputs['maximum_scientific_calls']!=1:
        raise ValueError('Exactly the one fixed fine-width input required')
    output=ROOT/DIRECTORY/'run01';output.mkdir();(output/LABEL).mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1))
    with F.installed_workers(output):result=M.supervise(inputs['cases'],inputs,source,output)
    exits=[c for p in output.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
           for c in json.loads(line)['children']]
    exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=bool(exits) and all(c['exitcode']==0 for c in exits))
    status=('FIXED_SCIENTIFIC_WORKERS_SAVED' if not result['errors'] and not result['monitor_errors']
        and exit_gate['all_exit_codes_zero'] and len(result['receipts'])==1
        and result['receipts'][0]['status']!='UNKNOWN' else 'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,process_exit_gate=exit_gate,
        finished_UTC=R.utc(),mandatory_stop=True,sufficient_N=None,G=None,G_qualified=None))
    print('Finished:',status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
