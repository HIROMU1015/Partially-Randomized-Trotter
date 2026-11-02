#!/usr/bin/env python3
"""R1 source plan now; native resource evaluation only after separate approval."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_reallocation.model import events,description
from trottertracks.algorithm_codesign.rte_reallocation.native import lower_event,planned_angles
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,synthesize,synthesis_key
from trottertracks.algorithm_codesign.rte_reallocation.accounting import canonical_profile,confidence_budget,interval_ratio
from trottertracks.algorithm_codesign.rte_reallocation.launch import (
    verify_launch,verify_runtime,consume_marker,BudgetGuard,serialize_result,sha,
)

PREPARATION=ROOT/'artifacts/track_b_rte_reallocation_r1_source/2026-10-06'
CONTRACT=PREPARATION/'contract_v2.json'


def requests(contract):
    angles=planned_angles(contract['domain']['x'])
    req={synthesis_key(angle,eps):(angle,eps) for angle in angles.values() for eps in contract['native_operator_epsilons']}
    if len(req)!=contract['caps']['synthesis_keys']:raise ValueError('frozen key inventory mismatch')
    return req


def verify_plan(root,contract):
    p=json.loads((root/contract['key_inventory_path']).read_text())
    if p['keys']!=list(requests(contract)) or p['science_calls']!=0:
        raise PermissionError('registered result-prior keys changed')
    return p


def build_rows(contract,cache,guard,result):
    for context in contract['domain']['contexts']:
        arms=contract['arms_by_context'][context]
        for x in contract['domain']['x']:
            for sigma in contract['domain']['sigma']:
                for controlled in (False,True):
                    for eps in contract['native_operator_epsilons']:
                        group=[]
                        for arm in arms:
                            guard.check();start=time.monotonic()
                            es=events(context,F(x),sigma,arm)
                            acquisition=time.monotonic()-start
                            circuits=[lower_event(context,e,controlled) for e in es]
                            profile=canonical_profile(es,circuits,cache,eps,controlled)
                            classical={'coefficient_description_entries':len(description(F(x),arm)) if arm!='CTS_collected' else None,
                                'expected_I0_involution_index_draws':str(sum(F(record['canonical_probability_exact'])*(len(e.word)+int(isinstance(e.rotation,int))) for record,e in zip(profile['events'],es))) if arm!='CTS_collected' else None,
                                'enumerated_evaluator_events':len(es),
                                'CTS_collection_word_labels':14 if arm=='CTS_collected' else 0,
                                'CTS_collection_Pauli_multiplications':34 if arm=='CTS_collected' else 0,
                                'coefficient_and_event_acquisition_wall_seconds_diagnostic':acquisition,
                                'enumeration_is_pilot_evaluator_not_I0_sampler_preprocessing':True,
                                'cost_optimality_claim':False}
                            row={'context':context,'x':x,'sigma':sigma,'controlled':controlled,'epsilon':eps,'arm':arm,
                                 'evidence_role':'primary_native_controlled' if context=='distinct_basis' and controlled else 'control_or_diagnostic',
                                 'profile':profile,'C_classical':classical,
                                 'finite_confidence':confidence_budget(profile,contract['finite_confidence']) if controlled else {'status':'ORDINARY_DIAGNOSTIC_NO_COHERENT_TASK'},
                                 'science_GO':False,'mandatory_STOP':True}
                            result['resource_rows'].append(row);group.append(row)
                        a=next(r for r in group if r['arm']=='A')
                        # Descriptive ratios only. No post-result winner/materiality route.
                        comparisons=[]
                        for other in group:
                            if other['arm']=='A':continue
                            ap,bp=a['profile'],other['profile']
                            item={'baseline':other['arm'],'B_squared':interval_ratio(ap['ideal_B_squared'],bp['ideal_B_squared']),
                                  'native_expected_cost':{},'resource_vector_includes_classical_subvector':True}
                            for name in ('T','CX','1Q'):
                                av,bv=ap['E_native_cost'][name],bp['E_native_cost'][name]
                                item['native_expected_cost'][name]=interval_ratio({'lo':av,'hi':av},{'lo':bv,'hi':bv})
                            for name in ('G_T','G_CX'):
                                aa,bb=a['finite_confidence'],other['finite_confidence']
                                item[name]=interval_ratio({'lo':aa[name],'hi':aa[name]},{'lo':bb[name],'hi':bb[name]}) if name in aa and name in bb else {'status':'TASK_NOT_JOINTLY_ELIGIBLE','ratio':None}
                            comparisons.append(item)
                        result['comparison_rows'].append({'context':context,'x':x,'sigma':sigma,'controlled':controlled,
                                                          'epsilon':eps,'A_vs_registered_baselines':comparisons,'scientific_classification':None})


def run():
    # Pending source fails before tool import, keys, circuits, marker or resource evaluation.
    contract,auth,head=verify_launch(ROOT,CONTRACT)
    runtime=verify_runtime(ROOT,contract);plan=verify_plan(ROOT,contract)
    directory=ROOT/contract['result_directory']
    receipt={'source_commit':auth['source_commit'],'authorization_commit':head,'contract_sha256':sha(CONTRACT),
             'authorization_sha256':sha(ROOT/contract['authorization_path']),
             'tool_identity_sha256':contract['tool_identity']['sha256'],'key_inventory_sha256':sha(ROOT/contract['key_inventory_path']),
             'runs':1,'retries':0,'mandatory_STOP':True,'next_stage_authorized':False}
    marker=consume_marker(directory,receipt)
    result={**receipt,'one_shot_marker_sha256':sha(marker),'status':'INCONCLUSIVE','runtime_identity':runtime,
            'synthesis_rows':[],'resource_rows':[],'comparison_rows':[],
            'synthesis_attempts':0,'pygridsynth_invocations':0,'last_attempted_synthesis_key':None,
            'science_GO':False,'DF_wrapper_authorized':False,'molecule_GPU_trajectory_calls':0}
    guard=BudgetGuard(contract['caps']);cache={}
    try:
        with guard:
            configure(contract['interval_and_point_dps'])
            for key,(angle,epsilon) in requests(contract).items():
                guard.begin_key();result['synthesis_attempts']+=1;result['last_attempted_synthesis_key']=key
                def invoked():result['pygridsynth_invocations']+=1
                row=synthesize(angle,epsilon,contract['synthesizer_options'],on_synthesis=invoked,max_characters=contract['caps']['sequence_characters'])
                result['synthesis_rows'].append(row);row.update(guard.end_key())
                if len(row['sequence'])>contract['caps']['sequence_characters']:raise RuntimeError('sequence cap hit')
                if not row['error_pass']:raise ArithmeticError('strict operator guard failed; no retry')
                cache[key]=row
            if len(cache)!=len(plan['keys']):raise RuntimeError('incomplete synthesis inventory')
            build_rows(contract,cache,guard,result);guard.check()
            if len(result['resource_rows'])!=contract['planned_resource_rows']:raise RuntimeError('row inventory mismatch')
            result['status']='R1_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
    except BaseException as error:
        result['failure']=f'{type(error).__name__}: {error}'[:1500]
    result['resource_usage']=guard.usage();result['synthesis_calls']=result['pygridsynth_invocations']
    result['planned_synthesis_keys']=contract['caps']['synthesis_keys']
    try:payload=serialize_result(result,contract['caps']['output_bytes'])
    except BaseException as error:
        payload=serialize_result({**receipt,'status':'INCONCLUSIVE','failure':str(error)[:1000],
            'one_shot_marker_sha256':sha(marker),'partial_rows_not_saved':True,'mandatory_STOP':True,
            'synthesis_calls':result['pygridsynth_invocations'],'synthesis_attempts':result['synthesis_attempts'],
            'last_attempted_synthesis_key':result['last_attempted_synthesis_key'],'retries':0},contract['caps']['output_bytes'])
    with (directory/'result.json').open('x') as stream:stream.write(payload)
    print(json.dumps({'status':json.loads(payload)['status'],'mandatory_STOP':True,'result':str(directory/'result.json')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('plan','run'));args=parser.parse_args()
    if args.mode=='plan':
        c=json.loads(CONTRACT.read_text());p=verify_plan(ROOT,c)
        print(json.dumps({'keys':len(p['keys']),'resource_rows':c['planned_resource_rows'],
            'synthesis_calls':0,'science_resource_evaluations':0,'science_execution_authorized':False,'mandatory_STOP':True}))
    else:
        try:run()
        except Exception as error:parser.exit(2,f'launch rejected: {type(error).__name__}: {error}\n')


if __name__=='__main__':main()
