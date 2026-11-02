"""G8 delegated bounded on-demand native path; separate reference only afterward."""
import argparse
from dataclasses import asdict,is_dataclass
from fractions import Fraction as F
import hashlib,json,time,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.g7_generator import make_generator,DeterministicBits
from trottertracks.algorithm_codesign.g7_launch import verify_source,sha,protected_check,consume_marker
from trottertracks.algorithm_codesign.g8_bounds import budget
from trottertracks.algorithm_codesign.g8_pipeline import AcquisitionCache,NativePipeline
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,synthesize,validate_saved
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
from trottertracks.algorithm_codesign.rte_reallocation.launch import BudgetGuard
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime

PREP='artifacts/track_b_g8_on_demand_preparation/2026-10-10'


def plain(v):
    if isinstance(v,F):return str(v)
    if is_dataclass(v):return plain(asdict(v))
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v


def generator(spec,arm,c):
    return make_generator(tuple(map(F,spec['p'])),F(spec['x']),spec['m'],arm,
        root_bits=c['root_bits'],probability_bits=c['probability_bits'],eta=F(c['eta']),rho=F(c['rho']))


def trace(pipe,c,guard):
    bits=DeterministicBits(c['diagnostic_bitstream_seed']);digest=hashlib.sha256()
    first=None;start,time_cpu=time.monotonic(),time.process_time()
    for _ in range(c['diagnostic_trials_per_pass']):
        value=pipe.step(bits)
        if first is None and value is not None:first=value
        digest.update(json.dumps(plain(value),sort_keys=True).encode()+b'\n')
        guard.check()
    return {'digest':digest.hexdigest(),'wall_seconds':time.monotonic()-start,
            'CPU_seconds':time.process_time()-time_cpu,'bit_blocks':bits.counter,
            'requested_bits':bits.consumed_bits,'trials':c['diagnostic_trials_per_pass'],
            'first_live_description':first,'confidence_or_cost_frequency_estimation':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);a=p.parse_args()
    cp=ROOT/PREP/'contract_v1.json';c=json.loads(cp.read_text())
    before=verify_source(ROOT,a.source_commit,c);runtime=verify_runtime(ROOT,c)
    if sha(Path(sys.executable).resolve())!=c['runtime_executable_sha256']:raise PermissionError('fixed Python identity mismatch')
    out=ROOT/c['result_directory']
    marker=consume_marker(out,{'kind':'G8_DELEGATED_ON_DEMAND_TECHNICAL_BUNDLE','source_commit':a.source_commit,
        'contract_sha256':sha(cp),'authorization':c['authorization'],'runs':1,'retries':0,'mandatory_STOP':True})
    guard=BudgetGuard(c['caps']);rows=[];diagnostic={};reason=None
    status='G8_TECHNICAL_INCONCLUSIVE';context={};calls=0
    def backend(q):
        nonlocal calls
        guard.begin_key();calls+=1
        if calls>c['caps']['synthesis_keys']:raise RuntimeError('G8 native call cap; no retry')
        angle=Angle('atan',q)
        row=synthesize(angle,c['primitive_error'],c['synthesizer_options'],max_characters=c['caps']['sequence_characters'])
        row['acquisition_resource']=guard.end_key();row['first_live_context']=dict(context)
        validate_saved(row,angle,c['primitive_error'])
        print(json.dumps({'live_miss':str(q),'T_count':row['T_count'],'error_pass':True}),flush=True)
        return row
    cache=AcquisitionCache(backend,c['caps']['synthesis_keys'],c['cache']['acquisition_byte_cap'])
    gens=[]
    try:
        with guard:
            configure(c['interval_dps'])
            for spec in c['inputs']:
                for arm in c['arms']:
                    context={'input':spec['id'],'arm':arm,'pass':'cold'}
                    wall,cpu=time.monotonic(),time.process_time()
                    g=generator(spec,arm,c);plan=budget(g,F(c['provider_delta_parameter']),F(c['primitive_error']))
                    setup={'wall_seconds':time.monotonic()-wall,'CPU_seconds':time.process_time()-cpu}
                    if plan['N_per_axis']>c['caps']['shot_cap_per_axis']:raise RuntimeError('G8 shot cap; no rescue')
                    pipe=NativePipeline(g,cache,plan,F(c['provider_delta_parameter']),
                                        c['cache']['row_entries'],c['cache']['row_byte_cap'])
                    cold=trace(pipe,c,guard);cold_counts=pipe.counters();context['pass']='warm'
                    warm=trace(pipe,c,guard);total_counts=pipe.counters()
                    if cold['digest']!=warm['digest']:raise ArithmeticError('deterministic semantic replay mismatch')
                    keys=pipe.observed_keys
                    cold_charge={k:sum(cache.rows[q]['acquisition_resource'][k] for q in keys)
                                 for k in ('wall_seconds','cpu_seconds')}
                    rows.append({'input':spec['id'],'arm':arm,'p':g.p,'x':g.x,'m':g.m,'budget':plan,
                        'construction_and_budget_resource':setup,'cold_pass':cold,'warm_pass':warm,
                        'cold_counters':cold_counts,'total_counters':total_counts,
                        'isolated_cold_acquisition_charge_for_observed_keys':cold_charge,
                        'production_started_with_row_cache_empty':True,
                        'global_table_or_key_plan_inputs':0,'whole_physical_provider_instantiated':False})
                    gens.append((spec['id'],g,plan));guard.check()
            # All production is finished before parsing any saved native cost
            # table or importing the independent enumeration audit.
            production_finished=time.monotonic()
            from trottertracks.algorithm_codesign.g8_reference_audit import (saved_factorizations,
                independent_review_cap,oracle_IS_diagnostic)
            old=json.loads((ROOT/c['saved_G7_result']).read_text())
            start,cpu=time.monotonic(),time.process_time()
            oracle=[];rebudget=[];cap_check=None
            for name,g,plan in gens:
                old_row=next(r for r in old['rows'] if r['input']==name and r['arm']==g.arm)
                ref=old_row['small_support_reference'];N=plan['N_per_axis']
                rebudget.append({'input':name,'arm':g.arm,'N_per_axis':N,
                    'T_Rz_two_axes_saved_per_trial_cost':2*N*F(ref['per_trial_fixed_cost']['T_Rz']),
                    'provider_coefficients': [2*N*F(x) for x in ref['per_trial_provider_calls']],
                    'expected_quantum_calls':2*N*F(ref['digital_acceptance']),
                    'accepted_call_cap':plan['accepted_call_cap_two_axes'],
                    'hard_attempt_cap':plan['hard_attempt_cap_two_axes'],
                    'new_actual_provider_costs_acquired':False,'G7_reclassified':False})
                o=oracle_IS_diagnostic(g,old['synthesis_cache'],plan)
                oracle.append({'input':name,'arm':g.arm,'diagnostic':o})
                if name=='P5_general_order' and g.arm=='full_return':
                    cap_check=independent_review_cap(g,old_row)
                    if F(108,125)<plan['acceptance_upper_non_enumerative']:raise ArithmeticError('review coarse .864 bound fails')
            counts_match={q:row['sequence_sha256']==old['synthesis_cache'][q]['sequence_sha256']
                          for q,row in cache.rows.items() if q in old['synthesis_cache']}
            diagnostic={'saved_factorizations':saved_factorizations(old),'independent_G7_review_cap':cap_check,
                'finite_provider_rebudget_saved_G7_costs':rebudget,'oracle_IS_Rz_only':oracle,
                'observed_sequences_vs_G7_identity':counts_match,
                'reference_phase_started_after_all_production':True,
                'production_finished_monotonic':production_finished,
                'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
                'not_non_enumerative_runtime_or_new_baseline_optimality_evidence':True}
            guard.check();status='G8_ON_DEMAND_PATH_AND_CONDITIONAL_PROVIDER_BUDGET_COMPLETE'
    except Exception as error:reason=type(error).__name__+': '+str(error)
    after=protected_check(ROOT,c)
    if after['violations']:status,reason='G8_TECHNICAL_INCONCLUSIVE','protected history changed'
    result={'kind':'G8_KNOWN_DEVELOPMENT_CONTRACT_PATHWAY_NOT_NATIVE_PERFORMANCE_ADOPTION',
        'status':status,'technical_reason':reason,'source_commit':a.source_commit,'contract_sha256':sha(cp),
        'marker_sha256':sha(marker),'runtime':runtime,'rows':rows,'reference_diagnostic':diagnostic,
        'on_demand_acquired_sequences':cache.rows,'synthesis_calls':calls,
        'acquisition_cache':{'capacity':cache.capacity,'entries':len(cache.rows),'requests':cache.requests,
            'hits':cache.hits,'bytes':cache.bytes,'peak_bytes':cache.peak_bytes,'no_keys_injected':True},
        'resource':guard.usage(),'protected_before':before,'protected_after':after,
        'runs':1,'retries':0,'DF_molecule_NPZ_GPU_LP_quantum_measurement_trajectory_actual_provider_or_circuit_compile':0,
        'provider_delta_is_hypothetical':True,'uniform_Rz_synthesis_termination_for_all_events_proved':False,
        'fixed_trace_covers_all_events_or_all_native_angles':False,'source_correction_after_one_shot':False,
        'mandatory_STOP':True,'next_science_authorized':False,'novelty_or_main_method_adopted':False,
        'prefix_comparisons_usable':status!='G8_TECHNICAL_INCONCLUSIVE'}
    payload=(json.dumps(plain(result),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    if len(payload)>c['caps']['output_bytes']-16384:raise RuntimeError('G8 output cap; consumed marker remains STOP')
    with (out/'result_v1.json').open('xb') as f:f.write(payload)
    with (out/'STOP.json').open('x') as f:
        json.dump({'mandatory_STOP':True,'status':status,'next_science_authorized':False,'research_owner':'GPT / user'},f,indent=2);f.write('\n')
    print(json.dumps({'status':status,'synthesis_calls':calls,'rows':len(rows),'STOP':True}))


if __name__=='__main__':main()
