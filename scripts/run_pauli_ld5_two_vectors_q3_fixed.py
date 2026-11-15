"""LD5 total view, two fixed supplied allocations, sequential Q3, no search."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_ld5_saved_ld3_route',ROOT/'scripts/run_pauli_ld3_pf_budget_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
L,T,R,M=C.L,C.T,C.R,C.M
A=L.A
BASE='fb4d78781e2a4c63c9d334af246807785e669d6c'
PRICE=dict(commit='b8bcdf97b693fa7e99d3d35f44f5ce6d3a691201',
    path='docs/research/a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json')
CID='4d6e2c9c4511acb0e2d7f4d488e331009504de805ad7826fdeb703907fc919eb'
DIRECTORY='docs/research/pauli_ld5_two_vectors_q3_results_20261116'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_pauli_ld5_two_vectors_q3_fixed.py'
VECTORS=[('FULL_PF__B_PRICE',[48,45,42,39,36,33,29,26,23,19,16,8,2]),
         ('VP',[48,45,42,39,36,33,29,26,23,19,16,13,9])]


def check_source():
    L.check_source()
    for path in ('scripts/run_pauli_ld3_pf_budget_q3_fixed.py','scripts/run_pauli_ld3_vp_q3_fixed.py',
                 'src/trottertracks/rpe_total_binding.py'):
        if (ROOT/path).read_bytes()!=R.blob(BASE,path):raise ValueError('Reused LD3 SOURCE differs: '+path)


def metadata(case):
    accepted=A.accept_saved(ROOT,integration=T.C.interfaces(str(ROOT)).pipeline,handoff_source=A.LD5_HANDOFF)['results'][0]
    original=T._selected(ROOT,case['configuration'])
    if original!=accepted['configuration'] or original['id']!=CID:raise ValueError('LD5 original SOURCE/ID differs')
    binding=T.create(ROOT,original,original_source=case['configuration'])
    if case.get('binding_id') is not None and (binding['id']!=case['binding_id'] or
            binding['total_configuration']['id']!=case['total_configuration_id']):raise ValueError('LD5 total canonical binding differs')
    return binding,T.bind_signal(ROOT,binding,accepted['signal_packet'])


def joined_prices(table,binding,view):
    if (table['configuration_id']!=CID or table['configuration_source']!=binding['original_source']
            or table['signal_packet']!=A.LD5_HANDOFF or table['kind']!='MODEL_REFERENCE_PRICE_NOT_E_C'
            or any(table[k] is not None for k in ('E_C','CI','G','winner'))):
        raise ValueError('LD5 price SOURCE/kind/qualification differs')
    rows=sorted(enumerate(table['rows']),key=lambda v:(v[1]['m'],v[1]['axis']))
    if [(r['m'],r['axis']) for _,r in rows]!=[(m,a) for m in range(13) for a in ('X','Y')]:
        raise ValueError('26 distinct axes required')
    original=binding['original_configuration']
    for index,row in rows:
        m=row['m'];round=original['rounds'][m]
        physical=view['signal_packet']['means'][m]['canonical_origin']['provider_binding']['physical_context']
        if (row['physical_id']!=round['candidate']['id'] or physical['physical_candidate']['id']!=row['physical_id']
                or (row['q'],row['R'],row['events'])!=(round['q_PF'],round['R_outer'],round['structure']['event_count'])
                or physical['dictionary_identity']!=view['signal_packet']['dictionary_identity']
                or row['value_source']!='MODEL_APPLICATION_FROM_Q8R4_DECOMPOSITION'
                or row['direct_observation_at_this_q_R'] is not False or C.rational(row['C_ref_exact'])<=0
                or row['N']!=L.VECTOR[m]
                or C.rational(row['N_times_C_ref_exact'])!=row['N']*C.rational(row['C_ref_exact'])):
            raise ValueError('LD5 price/physical/law/VP row mismatch at '+str((m,row['axis'])))
    return rows


def build_inputs():
    check_source()
    old=C.document(dict(commit=BASE,path=L.INPUT));prior=C.document(dict(commit=BASE,path=C.INPUT))
    handoff=C.document(A.LD5_HANDOFF);table=C.document(PRICE)
    seed=dict(label='LD5_METADATA',configuration=handoff['configuration_source'],signal_packet=A.LD5_HANDOFF,
        original_configuration_id=CID,evaluation_target=deepcopy(old['cases'][0]['evaluation_target']))
    binding,view=metadata(seed);rows=joined_prices(table,binding,view);cases=[]
    for name,vector in VECTORS:
        case=deepcopy(seed);case.update(label='LD5__'+name,comparison_name=name,binding_id=binding['id'],
            total_configuration_id=binding['total_configuration']['id'],
            allocation=dict(allocation_id='PAULI_LD5_'+name+'_USER_FIXED_01',shot_policy='USER_SUPPLIED',
                shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(vector)]))
        saved=table['comparison'][name]
        if vector!=saved['N_per_axis']:raise ValueError('A exact comparison uses a different vector')
        case['price_terms']=[dict(m=r['m'],axis=r['axis'],N=vector[r['m']],C_ref=C.exact(C.rational(r['C_ref_exact'])),
            N_times_C_ref=C.exact(vector[r['m']]*C.rational(r['C_ref_exact'])),
            original_candidate_id=r['physical_id'],total_candidate_id=binding['total_configuration']['rounds'][r['m']]['candidate']['id'],
            price_source=dict(**PRICE,selector=['rows',i]),value_source=r['value_source'],price_qualified_for_candidate=False)
            for i,r in rows]
        cost=sum(C.rational(t['N_times_C_ref']) for t in case['price_terms'])
        if cost!=C.rational(saved['T_LD5']):raise ValueError('Allocation-specific exact sum differs from A comparison')
        case['conditional_cost']=C.exact(cost);case['A_comparison_source']=dict(**PRICE,selector=['comparison',name])
        case['finite_scope_preflight']=L.H.finite_bounds(vector);bounds=case['finite_scope_preflight'];options=old['options']
        if (bounds['maximum_round_outcomes']>options['maximum_round_outcomes'] or
                bounds['verification_pairs_upper']>options['maximum_verification_pairs'] or
                bounds['transition_pairs_upper']>options['total_state_refinement']['maximum_transition_pairs']):
            raise ValueError('Concrete count-only scope cannot contain this vector')
        if name=='FULL_PF__B_PRICE':
            ld3=next(v for v in prior['cases'] if v['label']=='FULL_PF__B_PRICE')
            ref=C.document(dict(commit=BASE,path=C.DIRECTORY+'/C_COMPARISON_JOIN.json'))['records'][3]['result_source']
        else:
            ld3=old['cases'][0];ref=C.document(dict(commit=BASE,path=L.DIRECTORY+'/RESULT_SOURCE.json'))
        if ld3['allocation']['shots']!=case['allocation']['shots']:raise ValueError('LD3 comparison vector differs')
        case['LD3_comparison']=dict(API_source=ref,conditional_cost=deepcopy(saved['T_LD3']),
            price_source=deepcopy(saved['T_LD3_source']),original_configuration_id=ld3['original_configuration_id'],
            total_configuration_id=ld3['total_configuration_id'])
        cases.append(case)
    return dict(B_source=BASE,cases=cases,evaluation_order=[c['label'] for c in cases],maximum_scientific_calls=2,
        concurrent_configurations=1,signal_source=A.LD5_HANDOFF,price_source=PRICE,configuration_source=handoff['configuration_source'],
        dictionary_source=handoff['dictionary_source'],dictionary_helper=handoff['B_intake_dictionary_identity'],
        statistical_contract=old['statistical_contract'],options=old['options'],adopted_evaluation=old['adopted_evaluation'],
        acceleration=old['acceleration'],kernel_source=old['kernel_source'],C_source=old['C_source'],
        duplicate_check='SCOPE_AND_PREFLIGHT.md',allocation_generation_calls=0,price_model_axes=26,
        qualified_for_candidate=False,E_C=None,CI=None,sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


def prepare():
    inputs=build_inputs();p=ROOT/DIRECTORY;p.mkdir(exist_ok=True);R.write_new(p/'INPUTS.json',inputs)
    binding,view=metadata(inputs['cases'][0]);R.write_new(p/'TOTAL_BINDING.json',binding)
    R.write_new(p/'TOTAL_SIGNAL_VIEW.json',view['signal_packet']);R.write_new(p/'C_SIGNAL_INPUT.json',view['signal_input'])
    for c in inputs['cases']:print(c['label'],C.rational(c['conditional_cost']),c['finite_scope_preflight'],flush=True)


def worker(case,inputs,source,output):
    with patch.object(M,'metadata',metadata),patch.object(M,'check_source',check_source):
        return L.SAVED_WORKER(case,inputs,source,output)


@contextmanager
def installed(output):
    with patch.object(L,'worker',worker),L.installed(output):yield


def run_cases(inputs,visit):
    history=[];calls=0;stop='TWO_FIXED_ALLOCATIONS_COMPLETED'
    for index,case in enumerate(inputs['cases'],1):
        result=visit(case,index);n=result['scientific_evaluation_calls']
        if n not in (0,1) or calls+n>2:raise ValueError('Fixed two-call scope exceeded')
        calls+=n;history.append(dict(label=case['label'],**result))
        if result['status']=='UNKNOWN':stop='MAJOR_GATE_OR_PERSISTENCE_FAILURE';break
    return dict(history=history,new_scientific_calls=calls,stop_reason=stop,
        unevaluated=[c['label'] for c in inputs['cases'] if c['label'] not in {r['label'] for r in history}],
        sufficient_N=None,G=None,G_qualified=None,mandatory_stop=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER,'src/trottertracks/rpe_pauli_signal_intake.py','src/trottertracks/rpe_c_interfaces.py'):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Unfixed input/caller/adapter')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Fixed two-allocation input differs')
    run=ROOT/DIRECTORY/'run01';run.mkdir();source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(run/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=2))
    def visit(case,index):
        step=run/f'{index:02}';step.mkdir();folder=step/case['label'];folder.mkdir()
        R.write_new(step/'SELECTION.json',dict(sequence=index,case=case,reason='USER_FIXED_ORDER_NO_FEASIBILITY_EARLY_STOP'))
        with installed(step):sup=M.supervise([case],inputs,source,step)
        exits=[c for p in step.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines() for c in json.loads(line)['children']]
        gate=bool(exits) and all(c['exitcode']==0 for c in exits)
        R.write_new(step/'SUPERVISION.json',dict(**sup,process_exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=gate)))
        try:
            result=C.validate_result(case,inputs,folder,source);result['execution_source']=result.pop('result_source')
            result['API_path']=str((folder/'API_RETURN.json').relative_to(ROOT)) if (folder/'API_RETURN.json').exists() else None
            if not gate or sup['errors'] or sup['monitor_errors']:result.update(status='UNKNOWN',reason='SUPERVISOR_OR_PROCESS_GATE')
        except Exception as exc:
            stages=folder/'STAGES.jsonl';count=sum(json.loads(v)['stage']=='SCIENTIFIC_CALL_START' for v in stages.read_text().splitlines()) if stages.exists() else 0
            result=dict(status='UNKNOWN',reason=type(exc).__name__+': '+str(exc),scientific_evaluation_calls=count,MSE_lower=None,MSE_upper=None)
        R.write_new(step/'OUTCOME.json',result);print(case['label'],result['status'],flush=True);return result
    result=run_cases(inputs,visit);R.write_new(run/'RESULT_INDEX.json',result)
    R.write_new(run/'STOP.json',dict(source=source,stop_reason=result['stop_reason'],new_scientific_calls=result['new_scientific_calls'],
        finished_UTC=R.utc(),mandatory_stop=True,automatic_retry=False))
    print('STOP:',result['stop_reason'],'calls:',result['new_scientific_calls'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
