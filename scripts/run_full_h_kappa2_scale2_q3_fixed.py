"""One fixed Full-H kappa2 scale2 Q3 call; reuse the saved worker/kernel."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_fixed_full_h_scale2_reuse', ROOT/'scripts/run_pr_df_fullh_total_q3_fixed.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
R = M.R
BASE = 'a82b0f0ec7f0f11d9981ddb2343fcd5378a87503'
PLAN = '4525e1b68883bb6a4960346f6406da4d01af2ccd'
PROPOSAL = 'docs/research/full_h_kappa2_next_shot_plan_20261115/PROPOSAL.json'
DIRECTORY = 'docs/research/full_h_kappa2_scale2_q3_results_20261115'
INPUT = DIRECTORY+'/INPUTS.json'
CALLER = 'scripts/run_full_h_kappa2_scale2_q3_fixed.py'
VECTOR = [128,120,112,104,94,86,76,68,60,50,42,34,24]
LABEL = 'FULL_H_KAPPA2'


def check_source():
    M.check_source()
    for path in ('scripts/run_pr_df_fullh_total_q3_fixed.py', 'scripts/run_pauli_q3_three_quarter.py'):
        if (ROOT/path).read_bytes() != R.blob(BASE,path):
            raise ValueError('Reused caller/count helper differs: '+path)
    if (ROOT/PROPOSAL).read_bytes() != R.blob(PLAN,PROPOSAL):
        raise ValueError('Fixed proposal differs')


def build_inputs():
    """Metadata binding and integer cardinalities only; no risk/probability call."""
    check_source()
    proposal = json.loads(R.blob(PLAN,PROPOSAL))
    previous = json.loads(R.blob(BASE,'docs/research/pr_df_fullh_q3_fixed_results_20261115/INPUTS.json'))
    case = deepcopy(next(c for c in previous['cases'] if c['label']==LABEL))
    if proposal['proposal']['N_X'] != VECTOR or proposal['proposal']['N_Y'] != VECTOR:
        raise ValueError('Fixed scale2 vector differs')
    if proposal['candidate'] != {k:case[k] for k in proposal['candidate']}:
        raise ValueError('Physical candidate/SOURCE differs')
    if proposal['physical_rows'] != [{k:row[k] for k in proposal['physical_rows'][m]}
                                     for m,row in enumerate(case['rows'])]:
        raise ValueError('Thirteen physical rows differ')
    if proposal['evaluation_target'] != case['evaluation_target']:
        raise ValueError('Evaluation target differs')
    for key in ('proposed_allocation','equivalent_Q3_result','reuse',
                'proposed_new_scientific_calls_if_missing','closure_and_total_scope'):
        case.pop(key,None)
    case['prior_control_allocation'] = deepcopy(case['allocation'])
    case['allocation'] = dict(allocation_id=proposal['proposal']['allocation_id'], shot_policy='USER_SUPPLIED',
                              shots=deepcopy(proposal['proposal']['shots']))
    binding,view = M.metadata(case)
    for m,mean in enumerate(view['signal_packet']['means']):
        physical = mean['provider_binding']['physical_context']['physical_candidate']
        if physical != case['rows'][m]['physical_candidate']:
            raise ValueError('Saved physical row differs')
    contract = deepcopy(previous['statistical_contract'])
    contract['id'] = 'FULL_H_KAPPA2_SINGLE_SCALE2_TOTAL_Q3'
    options = deepcopy(previous['options'])
    options['maximum_round_outcomes'] = 16641
    options['maximum_verification_pairs'] = 847519356
    options['total_state_refinement'] = dict(maximum_frontier_states=8192,maximum_transition_pairs=620478721)
    options['model_loss_contract']['scope'] = 'ONE_FULL_H_KAPPA2_SCALE2_ALL_EVENT_MODEL_DIAGNOSTIC'
    bounds = M.H.finite_bounds(VECTOR)
    if (bounds['maximum_round_outcomes']!=16641 or bounds['verification_pairs_upper']!=847519356
            or bounds['transition_pairs_upper']!=620478721):
        raise ValueError('Count-only scope differs')
    return dict(B_source=PLAN,previous_results=dict(commit=BASE,
        path='docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json',label=LABEL),
        fixed_plan=dict(commit=PLAN,path=PROPOSAL),cases=[case],held=[],maximum_scientific_calls=1,
        kernel_source=previous['kernel_source'],C_source=previous['C_source'],
        reused_worker=dict(commit=BASE,path='scripts/run_pr_df_fullh_total_q3_fixed.py'),
        statistical_contract=contract,options=options,adopted_evaluation=deepcopy(previous['adopted_evaluation']),
        acceleration=deepcopy(previous['acceleration']),concurrent_configurations=1,finite_scope_preflight=bounds,
        initial_state_support='NEW_N0_128_NO_REUSE_OF_OLD_2593_STATE_BOUND',
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',signal_recomputed=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def prepare():
    inputs=build_inputs(); (ROOT/DIRECTORY).mkdir(exist_ok=True)
    R.write_new(ROOT/INPUT,inputs)
    print('One Full-H scale2 input fixed; no science called:',inputs['finite_scope_preflight'],flush=True)


def validate_inputs(inputs):
    if inputs != build_inputs():
        raise ValueError('Caller/INPUTS differ from the one fixed proposal')


def execute():
    check_source(); sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not committed')
    inputs=json.loads((ROOT/INPUT).read_text()); validate_inputs(inputs)
    output=ROOT/DIRECTORY/'run01'; output.mkdir(); (output/LABEL).mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1,held=[]))
    result=M.supervise(inputs['cases'],inputs,source,output)
    status=('FIXED_SCIENTIFIC_WORKERS_SAVED' if not result['errors'] and not result['monitor_errors']
            and len(result['receipts'])==1 and result['receipts'][0]['status']!='UNKNOWN' else 'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,held=[],finished_UTC=R.utc(),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))
    print('Finished:',status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
