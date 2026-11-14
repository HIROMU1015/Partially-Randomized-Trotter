"""Two permitted Q3 evaluations, scale 3/4 only; no price, N search or MC."""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('_previous_accelerated_q3',ROOT/'scripts/run_pauli_total_q3_accelerated.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
T=R.T
BASE='9c0e12d87bf83429e3411d295b16c47d913283d2'
PREVIOUS='caf1417ef20d0a5ab2030a2d957ee4f09443be43'
DIRECTORY='docs/research/pauli_q3_three_quarter_20261115'
INPUT=DIRECTORY+'/INPUTS.json'
EPSILON='0.00159360010199040'
R.L.BASE=BASE  # Existing checks now bind to the user's current B SOURCE.
FAILURE=None


def check_source():
    R.L.check_source()
    for sha,path in [(PREVIOUS,'scripts/run_pauli_total_q3_accelerated.py'),
        (R.ACCELERATOR,'src/trottertracks/rpe_total_dp_parallel.py')]:
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Reused SOURCE differs: '+path)


def finite_bounds(vector,frontier=8192):
    cells=[(n+1)**2 for n in vector]
    verification=sum(a*b for a,b in zip(cells,cells[1:]))
    states=min(cells[0],frontier);pairs=cells[0]
    for count in cells[1:]:
        pairs+=states*count;states=min(frontier,states*count)
    return dict(maximum_round_outcomes=max(cells),verification_pairs_upper=verification,
        transition_pairs_upper=pairs,rule='FULL_COUNT_SUPPORT_AND_FRONTIER_UPPER_BOUND_NO_PROBABILITY_EVALUATION')


def prepare():
    check_source()
    old=json.loads(R.blob(PREVIOUS,'docs/research/pauli_total_q3_accelerated_results_20261114/INPUTS.json'))
    cases=deepcopy(old['cases']);base=[r['N_X']for r in cases[0]['allocation']['shots']]
    vector=[max(1,(3*n+3)//4)for n in base];limits=finite_bounds(vector)
    if (limits['maximum_round_outcomes']>4225 or limits['verification_pairs_upper']>55043173
        or limits['transition_pairs_upper']>141559036):raise ValueError('Finite scope insufficient; do not execute')
    for case in cases:
        if [r['N_X']for r in case['allocation']['shots']]!=base or [r['N_Y']for r in case['allocation']['shots']]!=base:
            raise ValueError('Original control vectors differ')
        case['prior_control_allocation']=deepcopy(case['allocation'])
        case['allocation']=dict(allocation_id='PAULI_'+case['label']+'_USER_SCALE_3O4_01',shot_policy='USER_SUPPLIED',
            shots=[dict(m=m,N_X=n,N_Y=n)for m,n in enumerate(vector)])
    adopted=dict(estimator=T.T.ESTIMATOR,target='SAVED_DF10_SOLVER_ENERGY_CONDITIONAL_PROXY',
        loss='ALL_EVENT_SQUARED_ENERGY_ERROR',epsilon_E_decimal=EPSILON,
        MSE_upper=T.B.a.exact_value(Fraction(EPSILON)**2),scope='MODEL_MSE_FILTER_ONLY',
        measurement_law='SAVED_IDEAL_FINITE_K_RAW_IID_EXACT_AFFINE',
        minkowski='SEPARATE_REFERENCE_NOT_COMBINED',physical_total_error_guarantee=False)
    inputs=dict(B_source=BASE,kernel_source=old['kernel_source'],C_source=old['C_source'],A_handoff=old['A_handoff'],
        prior_control_inputs=dict(commit=PREVIOUS,path='docs/research/pauli_total_q3_accelerated_results_20261114/INPUTS.json'),
        scope='TWO_PAULI_Q3_SCALED_WHOLE_VECTORS_ONLY_NO_D_PRICE_B1',cases=cases,
        statistical_contract=deepcopy(old['statistical_contract']),options=deepcopy(old['options']),
        acceleration=deepcopy(old['acceleration']),adopted_evaluation=adopted,finite_scope_preflight=limits,
        maximum_scientific_calls=2,old_control_recomputed=False,PF_risk_calls=0,price_calls=0,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)
    (ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,inputs)
    print('Prepared',vector,limits,flush=True)


def assess(risk,contract,source):
    if risk.get('MSE_lower') is None or risk.get('MSE_upper') is None:
        return dict(status='UNKNOWN',reason='MISSING_MODEL_BOUND')
    lo=T.Q.a.exact_scalar(risk['MSE_lower']);hi=T.Q.a.exact_scalar(risk['MSE_upper'])
    tau=T.Q.a.exact_scalar(contract['MSE_upper'])
    if not 0<=lo<=hi or tau<=0:raise ValueError('Inconsistent bounds or threshold')
    limit=dict(scope='MODEL_MSE_FILTER_ONLY',source=deepcopy(source),MSE_upper=deepcopy(contract['MSE_upper']))
    summary=T.Q.q3_summary(risk,limit)
    status=('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'if summary['within_limit'] else
            'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'if lo>tau else 'INCONCLUSIVE')
    return dict(status=status,q3=summary,epsilon_E_decimal=contract['epsilon_E_decimal'],
        sufficient_N=None,global_RMSE_bound=None,G_qualified=None)


def init_worker(flag):
    global FAILURE
    FAILURE=flag


def worker(case,inputs,source,output):
    check_source();calls=[0];original=T.evaluate
    def single_call(*args,**kwargs):
        if FAILURE is not None and FAILURE.is_set():raise ValueError('Sibling failure before scientific call; not started')
        if calls[0]:raise ValueError('More than one scientific call prohibited')
        calls[0]+=1
        return original(*args,**kwargs)
    with patch.object(T,'evaluate',single_call):result=R.worker(case,inputs,source,output)
    result['scientific_evaluation_calls']=calls[0]
    result['reference_caller_status']=result['status']
    if result['status'].startswith('UNKNOWN'):
        if FAILURE is not None:FAILURE.set()
        return result
    try:
        assessment=assess(result['record']['diagnostic']['deterministic_risk'],inputs['adopted_evaluation'],source)
        result.update(status=assessment['status'],adopted_q3_assessment=assessment,
            adopted_evaluation=deepcopy(inputs['adopted_evaluation']),target_precision_filter_performed=True,
            prior_control_recomputed=False,PF_risk_calls=0,price_calls=0)
        if assessment['status']=='UNKNOWN'and FAILURE is not None:FAILURE.set()
    except Exception as exc:
        result.update(status='UNKNOWN',reason=str(exc))
        if FAILURE is not None:FAILURE.set()
    return result


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,'scripts/'+Path(__file__).name):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Unfixed runner/input SOURCE')
    inputs=json.loads((ROOT/INPUT).read_text())
    if [c['label']for c in inputs['cases']]!=['LD1','LD2']:raise ValueError('Exactly LD1/LD2 required')
    output=ROOT/DIRECTORY/'run01';output.mkdir()
    source=dict(commit=sha,path=INPUT,caller='scripts/'+Path(__file__).name)
    R.write_new(output/'EXECUTION.json',dict(source=source,B_source=BASE,kernel=inputs['kernel_source'],
        scope=inputs['scope'],started_UTC=R.utc(),maximum_scientific_calls=2,PF_risk_calls=0,price_calls=0))
    start=time.perf_counter();completed=[];errors=[];unknown=[];peaks={}
    context=mp.get_context('fork');failure=context.Event()
    with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=init_worker,initargs=(failure,))as pool:
        pending={pool.submit(worker,c,inputs,source,str(output)):c['label']for c in inputs['cases']}
        while pending:
            done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
            for future in done:
                label=pending.pop(future)
                try:
                    result=future.result();result['performance']['sampled_peak_process_tree_RSS_KiB']=peaks.get(label)
                    R.write_new(output/('RESULT_'+label+'.json'),result);completed.append(label)
                    if result['status'].startswith('UNKNOWN'):unknown.append(label);failure.set()
                    print(label+': '+result['status'],flush=True)
                except Exception as exc:
                    failure.set();errors.append(dict(label=label,type=type(exc).__name__,message=str(exc)))
            R.resources(output,list(pending.values()),start,peaks)
    R.write_new(output/'COMPLETION.json',dict(status='TWO_RESULTS_SAVED'if not unknown and not errors else'UNKNOWN_STOP_NO_RETRY',
        completed=sorted(completed),unknown=unknown,errors=errors,wall_seconds=time.perf_counter()-start,
        finished_UTC=R.utc(),sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare'else execute)()
