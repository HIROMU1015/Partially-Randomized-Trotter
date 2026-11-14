"""At most three approved fixed Q3 calls; workers persist without the monitor."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import resource
import subprocess
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_saved_pf_worker',ROOT/'scripts/run_pf_shortlist_total_q3_fixed.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
T,R,H=P.T,P.R,P.H
REFERENCE_CHECK=P.check_source
BASE='0629a8cf359b3aa99400f79de8f61a5ef50d8162'
PLAN=dict(commit=BASE,path='docs/research/pr_df_fullh_q3_execution_plan_20261115/CANDIDATE_INPUT_INDEX.json')
DIRECTORY='docs/research/pr_df_fullh_q3_fixed_results_20261115'
INPUT=DIRECTORY+'/INPUTS.json'
LABELS=['PR_DF_LD1','PR_DF_LD2','FULL_H_KAPPA2']
FAILURE=None


def check_source():
    REFERENCE_CHECK()
    path='scripts/run_pf_shortlist_total_q3_fixed.py'
    if (ROOT/path).read_bytes()!=R.blob(BASE,path):raise ValueError('Reused worker/monitor SOURCE differs')


def metadata(case):
    original=T._selected(ROOT,case['configuration'])
    if case['signal_packet_kind']=='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED':
        api=T.C.interfaces(str(ROOT)).pipeline
        accepted=T.K.accept_saved(ROOT,bundle_source=case['signal_packet'],integration=api)
        if accepted['request']['configuration']!=original:raise ValueError('Fixed-grid configuration differs')
        packet=accepted['signal_packet'];packet_source=None
    elif case['signal_packet_kind']=='SAVED_B_SIGNAL_INPUT_DIRECT_BASELINE_BIND':
        packet=T._selected(ROOT,case['signal_packet']);packet_source=case['signal_packet']
    else:raise ValueError('Unknown saved intake route')
    if original['id']!=case['original_configuration_id'] or packet['configuration_id']!=original['id']:
        raise ValueError('Original ID/SOURCE mismatch')
    binding=T.create(ROOT,original,original_source=case['configuration'])
    if binding['id']!=case['binding_id'] or binding['total_configuration']['id']!=case['total_configuration_id']:
        raise ValueError('Canonical bridge mismatch')
    view=T.bind_signal(ROOT,binding,packet,original_packet_source=packet_source)
    if len(view['signal_packet']['means'])!=13 or original['spec']['law']['model']!='ideal_finite_RTE_law':
        raise ValueError('Complete ideal finite-K raw input required')
    target=packet['target']
    if target['reference_solver_energy_ha']!=-2.1663874486347634 or target['ground_state_certified'] is not False:
        raise ValueError('Fixed proxy/qualification differs')
    return binding,view


def finite_bounds_from_saved_support(vector):
    """Count-only bound using a saved COMPLETE first-round count support.

    No probabilities/phases are evaluated. New binomial supports are subsets
    of that full Cartesian count support; the fixed decoder maps them to a
    subset of the same first-round states, independently of their weights.
    """
    generic=H.finite_bounds(vector)
    previous='caf1417ef20d0a5ab2030a2d957ee4f09443be43'
    directory='docs/research/pauli_total_q3_accelerated_results_20261114/'
    prior=json.loads(R.blob(previous,directory+'INPUTS.json'))
    dp=json.loads(R.blob(previous,directory+'run01/PARTIAL_TOTAL_DP_LD1.json'))
    coarse=json.loads(R.blob(previous,directory+'run01/PARTIAL_COARSE_LD1.json'))
    if (vector!=[r['N_X']for r in prior['cases'][0]['allocation']['shots']]
            or vector!=[r['N_Y']for r in prior['cases'][0]['allocation']['shots']]
            or dp['estimator']!=T.T.ESTIMATOR):raise ValueError('Saved count support contract differs')
    first=dp['rounds'][0];cells=[(n+1)**2 for n in vector]
    if (coarse['rounds'][0]['cells']!=cells[0]
            or first['retained_states']!=first['grouped_count_outcomes']
            or T.Q.a.exact_scalar(first['residual_probability'])!=0):
        raise ValueError('Saved full first-round support was not retained')
    bound=cells[0]+first['retained_states']*cells[1]+8192*sum(cells[2:])
    return dict(generic,uncompressed_transition_pairs_upper=generic['transition_pairs_upper'],
        transition_pairs_upper=bound,first_round_full_support_states=first['retained_states'],
        support_source=dict(commit=previous,path=directory+'run01/PARTIAL_TOTAL_DP_LD1.json',selector=['rounds',0]),
        full_count_source=dict(commit=previous,path=directory+'run01/PARTIAL_COARSE_LD1.json',selector=['rounds',0,'cells']),
        rule='C0 + saved_complete_m0_states*C1 + frontier*sum(C2..C12); count/decoder geometry only')


def prepare():
    check_source();index=T._selected(ROOT,PLAN)
    old=json.loads(R.blob(BASE,'docs/research/pauli_q3_three_quarter_20261115/INPUTS.json'))
    cases=[];held=[]
    for value in index['cases']:
        case=deepcopy(value)
        case.update(configuration=deepcopy(value['configuration_source']),signal_packet=deepcopy(value['signal_packet_source']))
        allocation=value['proposed_allocation'];nx,ny=allocation['N_X'],allocation['N_Y']
        case['allocation']=dict(allocation_id=allocation['allocation_id'],shot_policy='USER_SUPPLIED',
            shots=[dict(m=m,N_X=x,N_Y=y)for m,(x,y)in enumerate(zip(nx,ny))])
        try:
            if nx!=[64,60,56,52,47,43,38,34,30,25,21,17,12] or nx!=ny:
                raise ValueError('Fixed vector mismatch')
            _,view=metadata(case);target=view['signal_packet']['target']
            case['evaluation_target']=dict(kind='SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY',
                energy_ha=-2.1663874486347634,source=deepcopy(target['source']),
                original_target_source=deepcopy(target['source']),truth_proxy_source=deepcopy(target['truth_proxy_source']),
                binding=deepcopy(target['binding']),qualification_scope='RAW_MODEL_DIAGNOSTIC_ONLY_NOT_CERTIFIED_GROUND_ENERGY')
            cases.append(case)
        except Exception as exc:held.append(dict(label=case['label'],reason=str(exc),scientific_calls=0))
    contract=deepcopy(old['statistical_contract']);contract['id']='PR_DF_FULL_H_THREE_FIXED_TOTAL_Q3'
    options=deepcopy(old['options']);options['model_loss_contract']['scope']='THREE_FIXED_PR_DF_FULL_H_TOTAL_ALL_EVENT_DIAGNOSTICS'
    bounds=finite_bounds_from_saved_support(index['cases'][0]['proposed_allocation']['N_X'])
    if (bounds['maximum_round_outcomes']>4225 or bounds['verification_pairs_upper']>55043173
            or bounds['transition_pairs_upper']>141559036):raise ValueError('Count-only scope cannot contain inputs')
    adopted=deepcopy(index['adopted_evaluation'])
    adopted['MSE_upper']=T.B.a.exact_value(H.Fraction('0.00159360010199040')**2)
    inputs=dict(B_source=BASE,fixed_plan=PLAN,cases=cases,held=held,maximum_scientific_calls=3,
        kernel_source=old['kernel_source'],C_source=T.C.SOURCE,
        reused_worker=dict(commit=BASE,path='scripts/run_pf_shortlist_total_q3_fixed.py'),
        statistical_contract=contract,options=options,adopted_evaluation=adopted,
        acceleration=deepcopy(old['acceleration']),concurrent_configurations=len(cases),
        finite_scope_preflight=bounds,signal_recomputed=False,duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    (ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,inputs)
    print('Metadata ready:',[c['label']for c in cases],'held:',held,'count bounds:',bounds,flush=True)


def init_worker(flag):
    global FAILURE
    FAILURE=flag;P.FAILURE=flag


def worker(case,inputs,source,output):
    """Reuse P.worker's unchanged numerics; persist BEFORE returning to parent."""
    output=Path(output);start=time.perf_counter();cpu=time.process_time()
    child_start=resource.getrusage(resource.RUSAGE_CHILDREN);calls=0;api_saved=False
    original=T.evaluate
    def capture(*args,**kwargs):
        nonlocal calls,api_saved
        if calls:raise ValueError('More than one scientific call prohibited')
        calls+=1
        records=original(*args,**kwargs)
        R.write_new(output/'API_RETURN.json',records);api_saved=True
        return records
    try:
        with patch.object(P,'metadata',metadata),patch.object(P,'check_source',check_source),patch.object(T,'evaluate',capture):
            result=P.worker(case,inputs,source,str(output))
        if calls!=result['scientific_evaluation_calls']:raise ValueError('Scientific call counter mismatch')
        result.update(input_disposition='NEW_EVALUATION',original_api_return_saved=api_saved,
            persistence_scope='WORKER_OWNED_BEFORE_SUPERVISOR_RETURN')
        R.write_new(output/'RESULT.json',result)
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu=child.ru_utime+child.ru_stime-child_start.ru_utime-child_start.ru_stime
        usage=dict(label=case['label'],source=source,wall_seconds=time.perf_counter()-start,
            worker_CPU_seconds=time.process_time()-cpu,child_CPU_seconds=child_cpu,
            process_tree_CPU_seconds=time.process_time()-cpu+child_cpu,
            worker_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            max_single_completed_child_peak_RSS_KiB=child.ru_maxrss,
            scope='THROUGH_API_AND_RESULT_PERSISTENCE_CHILD_PEAK_IS_NOT_PROCESS_TREE_SUM')
        R.write_new(output/'RESOURCE_FINAL.json',usage)
        receipt=dict(label=case['label'],status=result['status'],source=source,
            scientific_evaluation_calls=calls,api_return_saved=api_saved,result_saved=True,resource_saved=True,
            operational_status='WORKER_ARTIFACTS_SAVED',sufficient_N=None,G_qualified=None)
        R.write_new(output/'PERSISTED.json',receipt)
        return receipt
    except Exception as exc:
        if FAILURE is not None:FAILURE.set()
        R.write_new(output/'WORKER_PERSISTENCE_ERROR.json',dict(label=case['label'],type=type(exc).__name__,
            reason=str(exc),scientific_evaluation_calls=calls,api_return_saved=api_saved,mandatory_stop=True))
        raise


def supervise(cases,inputs,source,output,*,monitor=P.monitor_pending,poll_seconds=2):
    """Monitor errors stop unstarted calls, while started workers save locally."""
    output=Path(output);start=time.perf_counter();context=mp.get_context('fork');failure=context.Event()
    receipts=[];errors=[];monitor_errors=[];peaks={c['label']:{}for c in cases};monitor_enabled=True
    if cases:
        with ProcessPoolExecutor(max_workers=inputs['concurrent_configurations'],mp_context=context,
                                 initializer=init_worker,initargs=(failure,))as pool:
            pending={pool.submit(worker,c,inputs,source,str(output/c['label'])):c['label']for c in cases}
            while pending:
                done,_=wait(pending,timeout=poll_seconds,return_when=FIRST_COMPLETED)
                for future in done:
                    label=pending.pop(future)
                    try:
                        receipt=future.result();receipts.append(receipt)
                        R.write_new(output/label/'SUPERVISOR_RECEIPT.json',receipt)
                        if receipt['status']=='UNKNOWN':failure.set()
                        print(label+': '+receipt['status'],flush=True)
                    except Exception as exc:
                        failure.set();errors.append(dict(label=label,type=type(exc).__name__,reason=str(exc)))
                if pending and monitor_enabled:
                    try:monitor(output,pending,start,peaks)
                    except Exception as exc:
                        failure.set();monitor_enabled=False
                        monitor_errors.append(dict(type=type(exc).__name__,reason=str(exc)))
                        R.write_new(output/'MONITOR_ERROR.json',dict(errors=monitor_errors,scientific_retry=False))
    return dict(receipts=receipts,errors=errors,monitor_errors=monitor_errors,
        sampled_process_tree_peak_RSS_KiB={k:v.get('job')for k,v in peaks.items()},wall_seconds=time.perf_counter()-start)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,'scripts/'+Path(__file__).name):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('SOURCE not fixed')
    inputs=json.loads((ROOT/INPUT).read_text());labels=[c['label']for c in inputs['cases']]
    if len(labels)>3 or len(set(labels))!=len(labels)or not set(labels)<=set(LABELS):raise ValueError('Fixed cases required')
    output=ROOT/DIRECTORY/'run01';output.mkdir()
    for label in labels:(output/label).mkdir()
    source=dict(commit=sha,path=INPUT,caller='scripts/'+Path(__file__).name)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=3,held=inputs['held']))
    result=supervise(inputs['cases'],inputs,source,output)
    status=('FIXED_SCIENTIFIC_WORKERS_SAVED'if not result['errors']and not result['monitor_errors']
            and all(r['status']!='UNKNOWN'for r in result['receipts'])else'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,held=inputs['held'],finished_UTC=R.utc(),
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare'else execute)()
