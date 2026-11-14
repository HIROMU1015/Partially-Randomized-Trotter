"""Exactly two fixed Pauli total/Q3 diagnostics; no retry, MC, N search or G.

Reuses B's immutable canonical/risk API and the previous fixed-run resource
monitor. Hooks only persist returned partial evidence and observe transitions;
they never change numerical inputs, operations, outputs or finite scopes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from trottertracks import rpe_pauli_signal_intake as A
from trottertracks import rpe_total_binding as T

BASE = 'eecbcdaed88b626b00858c7e94b15f598258d5ec'
ENGINE = '710aa69b663ef3313439762d25c80a652a03948f'
DIRECTORY = 'docs/research/pauli_total_q3_fixed_results_20261114'
INPUT = DIRECTORY+'/INPUTS.json'
MAPPING = 'docs/research/b_total_q3_canonical_integration_20261114/CANONICAL_BINDING_METADATA.json'
N = [64,60,56,52,47,43,38,34,30,25,21,17,12]


def blob(commit, path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def legacy():
    path=ROOT/'scripts/run_h4_dp_risk_fixed.py'
    spec=importlib.util.spec_from_file_location('_pauli_fixed_monitor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


L = legacy()
write_new, utc = L.write_new, L.utc


def check_source():
    paths=('src/trottertracks/rpe_pauli_signal_intake.py','src/trottertracks/rpe_total_binding.py',
        'src/trottertracks/rpe_total_risk.py','src/trottertracks/rpe_total_decoder.py',
        'src/trottertracks/rpe_c_interfaces.py','src/trottertracks/rpe_deterministic_risk.py',
        'src/trottertracks/rpe_decoder_fast.py','src/trottertracks/rpe_decoder_statistics.py',
        'src/trottertracks/rpe_signal_statistics_caller.py','src/trottertracks/pr_rpe_rmse_controller.py',
        'src/trottertracks/rpe_shot_allocation.py','src/trottertracks/rpe_price_implementation_guard.py',
        'src/trottertracks/pf_pr_rpe_pipeline.py','scripts/run_h4_dp_risk_fixed.py')
    for path in paths:
        if (ROOT/path).read_bytes()!=blob(BASE,path):
            raise ValueError('Fixed B SOURCE changed: '+path)
    for path in ('src/trottertracks/rpe_total_decoder.py','src/trottertracks/rpe_total_risk.py'):
        if blob(BASE,path)!=blob(ENGINE,path):raise ValueError('Kernel SOURCE mismatch')
    if T.C.SOURCE!='c628e9e8447c11c984064b922cbce6c52dc72d48':raise ValueError('C SOURCE differs')
    return list(paths)


def prepare():
    checked=check_source(); api=T.C.interfaces(str(ROOT)).pipeline
    accepted=A.accept_saved(ROOT,integration=api)
    mapping=json.loads(blob(BASE,MAPPING))['configurations']
    prior=json.loads(blob('e7ce23803b5644bd9df0e83c877b390aeb9405e0',
        'docs/research/h4_dp_risk_fixed_results_20261114/INPUTS.json'))
    cases=[]
    for index,item in enumerate(accepted['results']):
        source=dict(A.HANDOFF,selector=['packets',index,'configuration'])
        binding=T.create(ROOT,item['configuration'],original_source=source)
        view=T.bind_signal(ROOT,binding,item['signal_packet']); target=view['signal_packet']['target']
        expected=mapping[index]
        if (binding['original_configuration']['id']!=expected['original_configuration_id']
                or binding['total_configuration']['id']!=expected['total_configuration_id']
                or binding['id']!=expected['binding_id'] or len(view['signal_packet']['means'])!=13
                or target['reference_solver_energy_ha']!=-2.1663874486347634
                or target['ground_state_certified'] is not False
                or target['truth_proxy_adopted_for_new_simulation'] is not False):
            raise ValueError('Fixed canonical/target/qualification mismatch')
        cases.append(dict(label='LD'+str(index+1),original_source=source,
            original_configuration_id=expected['original_configuration_id'],
            total_configuration_id=expected['total_configuration_id'],binding_id=binding['id'],
            allocation=dict(allocation_id='PAULI_LD'+str(index+1)+'_USER_CONTROL_01',shot_policy='USER_SUPPLIED',
                shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(N)]),
            evaluation_target=dict(kind='SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY',
                energy_ha=target['reference_solver_energy_ha'],source=deepcopy(target['source']),
                truth_proxy_source=deepcopy(target['truth_proxy_source']),
                qualification_scope='RAW_MODEL_DIAGNOSTIC_ONLY_NOT_CERTIFIED_GROUND_ENERGY',
                original_target_source=deepcopy(target['source']),binding=deepcopy(target['binding']))))
    contract=deepcopy(prior['statistical_contract'])
    contract.update(id='PAULI_LD1_LD2_TOTAL_Q3_TWO_DIAGNOSTICS',claim_level='TOTAL_RAW_MODEL_DIAGNOSTIC_ONLY')
    options=dict(probability_rule=T.Q.D.EXACT_AFFINE,phase_radii=[1.]*13,
        maximum_round_outcomes=4225,maximum_verification_pairs=55043173,estimator=T.T.ESTIMATOR,
        model_loss_contract=dict(adopted=True,scope='TWO_PAULI_TOTAL_DECODER_ALL_EVENT_SQUARED_ERROR_DIAGNOSTICS_ONLY'),
        total_state_refinement=dict(maximum_frontier_states=8192,maximum_transition_pairs=141559036))
    inputs=dict(B_source=BASE,kernel_source=ENGINE,C_source=T.C.SOURCE,A_handoff=deepcopy(A.HANDOFF),
        scope='TWO_FIXED_PAULI_OPT_IN_TOTAL_MODEL_MSE_DIAGNOSTICS_NO_N_OR_G',cases=cases,
        statistical_contract=contract,options=options,
        reference_loss_family_source=dict(commit='e7ce23803b5644bd9df0e83c877b390aeb9405e0',
            path='docs/research/h4_dp_risk_fixed_results_20261114/INPUTS.json',
            selector=['statistical_contract','summary_options','unresolved_loss']),
        checked_B_files=checked,intake_source_blobs=accepted['source_blobs'],
        LD2_original_id_note='Chat ID contains duplicated text; pinned A packet and canonical metadata are authoritative.',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    out=ROOT/DIRECTORY;out.mkdir(exist_ok=True);write_new(ROOT/INPUT,inputs)
    print('Prepared exactly 2 canonical-bound cases / 26 saved round records; no risk called.',flush=True)


def worker(case, inputs, source, output):
    output=Path(output); label=case['label']; start=time.perf_counter();cpu=time.process_time()
    write_new(output/('START_'+label+'.json'),dict(label=label,pid=os.getpid(),parent_pid=os.getppid(),
        started_UTC=utc(),source=source))
    stage=output/('STAGES_'+label+'.jsonl')
    def event(name,**values):
        with stage.open('a') as stream:
            stream.write(json.dumps(dict(stage=name,UTC=utc(),wall_seconds=time.perf_counter()-start,**values))+'\n')
    partials={}; calls=0
    try:
        check_source(); api=T.C.interfaces(str(ROOT)).pipeline
        accepted=A.accept_saved(ROOT,integration=api)['results'][int(label[-1])-1]
        binding=T.create(ROOT,accepted['configuration'],original_source=case['original_source'])
        if binding['id']!=case['binding_id']:raise ValueError('Worker canonical binding mismatch')
        view=T.bind_signal(ROOT,binding,accepted['signal_packet'])
        allocation=deepcopy(case['allocation']);allocation['source']=deepcopy(source)
        contract=deepcopy(inputs['statistical_contract']);contract['source']=deepcopy(source)
        contract['simulation_policy']['source']=deepcopy(source)
        contract['summary_options']['unresolved_loss']['source']=deepcopy(source)
        options=deepcopy(inputs['options']);options['source']=deepcopy(source)
        options['model_loss_contract']['source']=deepcopy(source)
        options['total_state_refinement']['source']=deepcopy(source)
        # Explicit caller contracts are checked before the one scientific call.
        T.K.prepare(binding['original_configuration'],accepted['signal_packet'],[allocation],
            evaluation_target=case['evaluation_target'],statistical_contract=contract,integration=api)
        original_total=T.Q._total; original_lemma=T.Q.lemma_bound; original_dp=T.Q.total_state_dp
        original_step=T.T.step; progress=dict(transition_pairs_observed_after_m0=0,last_round=None)
        last_progress=[0.]
        def total_gate(entry,partial,opts):
            partials['partial_decoder_reference']=partial
            write_new(output/('PARTIAL_COARSE_'+label+'.json'),partial)
            certificate=partial.get('decoder_certificate') or {}
            if certificate.get('status')!=T.Q.VERIFIED:
                raise ValueError('RHO_CLOSURE_UNKNOWN:'+str(certificate.get('reason',partial.get('reason'))))
            event('GOOD_CLOSURE_VERIFIED_TOTAL_COUNT_TABLES_START')
            return original_total(entry,partial,opts)
        def lemma_gate(*args,**kwargs):
            value=original_lemma(*args,**kwargs);partials['lemma']=value
            write_new(output/('PARTIAL_LEMMA_'+label+'.json'),value)
            if value.get('MSE_upper') is None:raise ValueError('LEMMA_UNKNOWN:'+str(value.get('reason')))
            event('LEMMA_SAVED_TOTAL_DP_NEXT');return value
        def step_observer(principal,estimate,q,zero):
            result=original_step(principal,estimate,q,zero)
            progress['transition_pairs_observed_after_m0']+=len(principal);progress['last_round']=int(q).bit_length()-1
            now=time.perf_counter()
            if now-last_progress[0]>=30:
                event('TOTAL_DP_PROGRESS',**progress);last_progress[0]=now
            return result
        def dp_gate(*args,**kwargs):
            event('TOTAL_DP_START'); value=original_dp(*args,**kwargs);partials['total_state_dp']=value
            write_new(output/('PARTIAL_TOTAL_DP_'+label+'.json'),value)
            if value.get('MSE_upper') is None:raise ValueError('TOTAL_DP_UNKNOWN:'+str(value.get('reason')))
            if value['transition_pairs']>141559036 or any(r['retained_states']>8192 for r in value['rounds'][:-1]):
                raise ValueError('TOTAL_DP_FINITE_SCOPE_EXCEEDED')
            event('TOTAL_DP_RETURNED',status=value['status'],transition_pairs=value['transition_pairs'])
            return value
        event('CANONICAL_INPUT_VERIFIED_EVALUATION_START')
        with ExitStack() as stack:
            for module,name in ((T.Q.S,'simulate'),(T.Q.S,'simulate_prepared'),(T.K,'diagnose'),
                (T.Q.D.W,'diagnose'),(T.p,'run_missing'),(T.Q.D,'compare'),(T.Q.D.W,'compare'),
                (T.B,'cost'),(T.B,'select')):
                stack.enter_context(patch.object(module,name,side_effect=AssertionError('Outside approved scope: '+name)))
            for module,name,hook in ((T.Q,'_total',total_gate),(T.Q,'lemma_bound',lemma_gate),
                                    (T.Q,'total_state_dp',dp_gate),(T.T,'step',step_observer)):
                stack.enter_context(patch.object(module,name,hook))
            calls+=1
            records=T.evaluate(ROOT,binding,view['signal_packet'],[allocation],
                evaluation_target=case['evaluation_target'],statistical_contract=contract,source=source,options=options)
        if len(records)!=1:raise ValueError('One result required')
        record=records[0];risk=record['diagnostic']['deterministic_risk']
        if risk.get('MSE_upper') is None:raise ValueError('TOTAL_RISK_UNKNOWN:'+str(risk.get('reason')))
        lo=T.Q.a.exact_scalar(risk['MSE_lower']);hi=T.Q.a.exact_scalar(risk['MSE_upper'])
        if not 0<=lo<=hi:raise ValueError('MSE_BOUNDS_INCONSISTENT')
        if risk.get('q3',{}).get('limit') is not None:raise ValueError('Unapproved target filter')
        if any(record.get(k) is not None for k in ('n_packet','sufficient_N','global_RMSE_bound','G','G_qualified')):
            raise ValueError('Forbidden qualification/cost output')
        event('EVALUATION_COMPLETE')
        return dict(label=label,status='CONDITIONAL_OPT_IN_TOTAL_MODEL_BOUND_SAVED',source=source,record=record,
            performance=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid()),
            scientific_evaluation_calls=calls,MC_calls=0,old_partial_DP_calls=0,additional_allocations=False,
            target_precision_filter_performed=False,deltaE_added=False,ground_certified=False,
            sampler_law_transfer_certified=False,numerical_certificate=None,
            sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True)
    except Exception as exc:
        event('UNKNOWN_STOP',reason=str(exc))
        return dict(label=label,status='UNKNOWN_STOP_NO_RETRY',reason=str(exc),error_type=type(exc).__name__,
            source=source,partial_evidence_files=[p.name for p in output.glob('PARTIAL_*_'+label+'.json')],
            performance=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid()),
            scientific_evaluation_calls=calls,automatic_retry=False,sufficient_N=None,global_RMSE_bound=None,
            G=None,G_qualified=None,winner=None,mandatory_stop=True)


def execute():
    check_source();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,'scripts/'+Path(__file__).name):
        if (ROOT/path).read_bytes()!=blob(commit,path):raise ValueError('Runner/input not fixed at execution SOURCE')
    inputs=json.loads((ROOT/INPUT).read_text());output=ROOT/DIRECTORY/'run01';output.mkdir()
    source=dict(commit=commit,path=INPUT,caller='scripts/'+Path(__file__).name)
    write_new(output/'EXECUTION.json',dict(source=source,B_source=BASE,kernel_source=ENGINE,C_source=T.C.SOURCE,
        started_UTC=utc(),evaluations_requested=2,old_partial_DP_calls=0,MC_calls=0,prices_or_G=False))
    if [v['label'] for v in inputs['cases']]!=['LD1','LD2']:raise ValueError('Exactly LD1/LD2 required')
    start=time.perf_counter();completed=[];unknown=[];errors=[]
    with ProcessPoolExecutor(max_workers=len(inputs['cases'])) as pool:
        pending={pool.submit(worker,case,inputs,source,str(output)):case['label'] for case in inputs['cases']}
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                label=pending.pop(future)
                try:
                    result=future.result();write_new(output/('RESULT_'+label+'.json'),result)
                    completed.append(label)
                    if result['status'].startswith('UNKNOWN'):unknown.append(label)
                    print(label+': '+result['status'],flush=True)
                except Exception as exc:
                    errors.append(dict(label=label,type=type(exc).__name__,message=str(exc)))
            L.own_resources(output,list(pending.values()),start)
    write_new(output/'COMPLETION.json',dict(status='TWO_RESULTS_SAVED' if not errors and not unknown else 'UNKNOWN_STOP_NO_RETRY',
        completed=sorted(completed),unknown=unknown,errors=errors,wall_seconds=time.perf_counter()-start,
        finished_UTC=utc(),sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare' else execute)()
