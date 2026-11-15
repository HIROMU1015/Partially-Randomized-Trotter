"""One permitted Full-H kappa2 ceil(3N/2) Q3 call; no retry or N search."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_reuse_full_h_scale2_caller',ROOT/'scripts/run_full_h_kappa2_scale2_q3_fixed.py')
S=importlib.util.module_from_spec(spec);sys.modules[spec.name]=S;spec.loader.exec_module(S)
M,R=S.M,S.R
BASE='422db93fc7bb599a7ea46c5b58e5dec0fda9ef08'
CALLER_SOURCE='8a8871021b83c139479d0f47d24d8304145e8cf4'
DIRECTORY='docs/research/full_h_kappa2_scale3o2_q3_results_20261115'
INPUT=DIRECTORY+'/INPUTS.json'
CALLER='scripts/run_full_h_kappa2_scale3o2_q3_fixed.py'
LABEL='FULL_H_KAPPA2'
VECTOR=[96,90,84,78,71,65,57,51,45,38,32,26,18]
ALLOCATION='FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01'
EXIT_DIRECTORY=None


def check_source():
    S.check_source()
    path='scripts/run_full_h_kappa2_scale2_q3_fixed.py'
    if (ROOT/path).read_bytes()!=R.blob(CALLER_SOURCE,path):raise ValueError('Reused caller SOURCE differs')


def build_inputs():
    check_source()
    previous=json.loads(R.blob(BASE,'docs/research/full_h_kappa2_scale2_q3_results_20261115/INPUTS.json'))
    case=deepcopy(previous['cases'][0]);control=case['prior_control_allocation']
    if VECTOR!=[(3*r['N_X']+1)//2 for r in control['shots']]:raise ValueError('Fixed ceil(3N/2) vector differs')
    prior=deepcopy(case['allocation'])
    case['allocation']=dict(allocation_id=ALLOCATION,shot_policy='USER_SUPPLIED',
        shots=[dict(m=m,N_X=v,N_Y=v)for m,v in enumerate(VECTOR)])
    binding,view=M.metadata(case)
    for m,mean in enumerate(view['signal_packet']['means']):
        if mean['provider_binding']['physical_context']['physical_candidate']!=case['rows'][m]['physical_candidate']:
            raise ValueError('Saved physical descriptor differs')
    options=deepcopy(previous['options']);bounds=M.H.finite_bounds(VECTOR)
    if (bounds['maximum_round_outcomes'],bounds['verification_pairs_upper'],bounds['transition_pairs_upper'])!=(9409,273175586,353666241):
        raise ValueError('Integer finite scope differs')
    options.update(maximum_round_outcomes=9409,maximum_verification_pairs=273175586,
        total_state_refinement=dict(maximum_frontier_states=8192,maximum_transition_pairs=353666241))
    options['model_loss_contract']['scope']='ONE_FULL_H_KAPPA2_USER_SCALE_3O2_ALL_EVENT_MODEL_DIAGNOSTIC'
    contract=deepcopy(previous['statistical_contract']);contract['id']='FULL_H_KAPPA2_SINGLE_SCALE_3O2_TOTAL_Q3'
    return dict(B_source=BASE,cases=[case],held=[],maximum_scientific_calls=1,
        kernel_source=previous['kernel_source'],C_source=previous['C_source'],
        reused_caller=dict(commit=CALLER_SOURCE,path='scripts/run_full_h_kappa2_scale2_q3_fixed.py'),
        reused_worker=previous['reused_worker'],statistical_contract=contract,options=options,
        adopted_evaluation=deepcopy(previous['adopted_evaluation']),acceleration=deepcopy(previous['acceleration']),
        concurrent_configurations=1,finite_scope_preflight=bounds,
        prior_allocations=dict(control=control,scale2=prior),
        previous_results=[dict(commit='a82b0f0ec7f0f11d9981ddb2343fcd5378a87503',
            path='docs/research/pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json'),
            dict(commit=BASE,path='docs/research/full_h_kappa2_scale2_q3_results_20261115/SUMMARY.json')],
        initial_state_support='NEW_N0_96_NO_REUSE_OF_PREVIOUS_SUPPORT_OR_CLOSURE',
        observation='COOPERATIVE_STAGE_AND_PROCESS_EXIT_LOGS_NO_THREADS_OR_ASYNCHRONOUS_FRAME_DUMPS',
        duplicate_check='SEE_SCOPE_AND_PREFLIGHT.md',signal_recomputed=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None)


class ObservedStages(R.Observed):
    """Log intervals outside propagation without touching the fixed DP body."""
    def segment(self,stage,**values):
        with(self.output/'SEGMENTS.jsonl').open('a')as f:
            f.write(json.dumps(dict(stage=stage,UTC=R.utc(),wall_seconds=time.perf_counter()-self.started,**values))+'\n')

    def propagate(self,*args,**kwargs):
        if getattr(self,'last_propagation',None)is not None:
            m,t=self.last_propagation
            self.segment('BETWEEN_PROPAGATIONS_COMPLETE',m=m,seconds=time.perf_counter()-t)
        value=super().propagate(*args,**kwargs)
        self.last_propagation=(args[3],time.perf_counter())
        self.segment('POST_PROPAGATION_START',m=args[3])
        return value

    def __call__(self,*args,**kwargs):
        self.last_propagation=None;self.segment('DP_CALL_START')
        value=super().__call__(*args,**kwargs)
        if self.last_propagation is not None:
            m,t=self.last_propagation
            self.segment('TERMINAL_POST_PROPAGATION_COMPLETE',m=m,seconds=time.perf_counter()-t)
        self.segment('DP_CALL_RETURN',status=value['status'])
        return value


class ObservedExecutor(ProcessPoolExecutor):
    """Keep OS exit codes of this run's scientific worker and DP children."""
    def shutdown(self,wait=True,*,cancel_futures=False):
        processes=list((self._processes or {}).values())
        super().shutdown(wait=wait,cancel_futures=cancel_futures)
        if EXIT_DIRECTORY is not None and processes:
            entries=[dict(pid=p.pid,exitcode=p.exitcode,signal=-p.exitcode if p.exitcode is not None and p.exitcode<0 else None)
                     for p in processes]
            with(Path(EXIT_DIRECTORY)/('PROCESS_EXITS_'+str(os.getpid())+'.jsonl')).open('a')as f:
                f.write(json.dumps(dict(parent_pid=os.getpid(),UTC=R.utc(),wait=wait,children=entries))+'\n')


@contextmanager
def observation(output):
    global EXIT_DIRECTORY
    previous=EXIT_DIRECTORY;EXIT_DIRECTORY=str(output)
    try:
        with patch.object(R,'Observed',ObservedStages),patch.object(M,'ProcessPoolExecutor',ObservedExecutor),\
                patch.object(R.P,'ProcessPoolExecutor',ObservedExecutor):yield
    finally:EXIT_DIRECTORY=previous


def prepare():
    inputs=build_inputs();(ROOT/DIRECTORY).mkdir(exist_ok=True);R.write_new(ROOT/INPUT,inputs)
    print('One scale3o2 metadata input fixed; no science called:',inputs['finite_scope_preflight'],flush=True)


def execute():
    check_source();sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (INPUT,CALLER):
        if (ROOT/path).read_bytes()!=R.blob(sha,path):raise ValueError('Caller/INPUTS not committed')
    inputs=json.loads((ROOT/INPUT).read_text())
    if inputs!=build_inputs():raise ValueError('Input differs from fixed one-case contract')
    output=ROOT/DIRECTORY/'run01';output.mkdir();(output/LABEL).mkdir()
    source=dict(commit=sha,path=INPUT,caller=CALLER)
    R.write_new(output/'EXECUTION.json',dict(source=source,started_UTC=R.utc(),maximum_scientific_calls=1,held=[]))
    with observation(output):result=M.supervise(inputs['cases'],inputs,source,output)
    exits=[c for path in output.glob('PROCESS_EXITS_*.jsonl')for line in path.read_text().splitlines()
           for c in json.loads(line)['children']]
    exit_gate=dict(recorded_children=len(exits),all_exit_codes_zero=bool(exits)and all(c['exitcode']==0 for c in exits))
    status=('FIXED_SCIENTIFIC_WORKERS_SAVED'if not result['errors']and not result['monitor_errors']
            and exit_gate['all_exit_codes_zero'] and len(result['receipts'])==1
            and result['receipts'][0]['status']!='UNKNOWN'else'UNKNOWN_STOP_NO_RETRY')
    R.write_new(output/'COMPLETION.json',dict(status=status,**result,held=[],finished_UTC=R.utc(),
        process_exit_gate=exit_gate,
        sufficient_N=None,global_RMSE_bound=None,G=None,G_qualified=None,winner=None,mandatory_stop=True))
    print('Finished:',status,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','execute'))
    (prepare if parser.parse_args().mode=='prepare'else execute)()
