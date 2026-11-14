"""Acquire only four PR-DF LD1 m3 normal paths with unchanged P1/P2 S/C backend.

m4 is metadata only. No replay, residual aggregation, fit, prediction or holdout.
"""
import argparse
from copy import deepcopy
import faulthandler
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import p1_efficient_zz as P, compiled_price_fast as F, efficient_controlled_zz as E
import run_p2_direct_pilot as R

BASE='39ddd59edf54a3a901e5302207b2765ddbd03fbb'
C_INDEX=('c99d511e2422b8f250490b11054f754a81e536c5','docs/research/pr_df_ld1_q3_price_admission_20261115/ADMISSION.json')
C_PLAN=('8eef0ad7a9b04e04d766f5e05b3b32eee50a794e','docs/research/pr_df_ld1_q3_price_admission_20261115/NEXT_VALIDATION_PLAN.md')
FACTS=(C_INDEX[0],'docs/research/pr_df_ld1_q3_price_admission_20261115/SOURCE_FACTS.json')
CAL_COHORT='D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1'
HOLD_COHORT='D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1'
PUBLIC='docs/research/d_pr_df_ld1_m3_results_20261115'
PRIVATE='/home/AbeHiromu/projects/d2-pr-df-ld1-m3-20261115-private/run01'
STATE={}

def blob(commit,path):return P._blob(ROOT,commit,path)
def read_ref(ref):return json.loads(blob(*ref))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def point(config,m,cohort,role):
    r=config['rounds'][m];c={k:r['physical_candidate'][k] for k in P.CANDIDATE_FIELDS}
    if P.p.candidate_id(c)!=c['id']:raise ValueError('Canonical physical candidate differs')
    seed=P.trial_seed_function(ROOT)
    row=dict(m=m,q_RPE=r['q_RPE'],q_PF=r['q_PF'],candidate=c,structure=deepcopy(r['structure']),
             V2_candidate_id=r['candidate']['id'],identity_configuration_id=config['id'],law_binding=deepcopy(r['law_binding']))
    cell=dict(id='H4_S0_m%d_q%d_R%d'%(m,c['q'],c['R_outer']),system='H4',m=m,candidate_id=c['id'],role=role,
              cohort=cohort,n=4,path_indices=[0,1,2,3],seeds=[[seed(c,cohort,i,j,0) for j in range(c['q'])] for i in range(4)])
    return row,cell

def make_inputs():
    plan=P.make_plan(ROOT,BASE)
    config=read_ref(P.B_CONFIG);admission=read_ref(C_INDEX);facts=read_ref(FACTS)
    r3,c3=point(config,3,CAL_COHORT,'CALIBRATION')
    r4,c4=point(config,4,HOLD_COHORT,'HOLDOUT_NOT_AUTHORIZED')
    if (c3['candidate_id'],r3['candidate']['q'],r3['candidate']['R_outer'],r3['candidate']['T'])!=('045843b2429bece6f29b',8,16,.8):
        raise ValueError('m3 scope differs')
    if (c4['candidate_id'],r4['candidate']['q'],r4['candidate']['R_outer'])!=('3df0cfd3356c358aeda8',16,32):
        raise ValueError('Future m4 metadata differs')
    for row,cell in ((r3,c3),(r4,c4)):
        for a in ('X','Y'):
            old=next(r for r in admission['rows'] if (r['m'],r['axis'])==(cell['m'],a))
            if old['physical_candidate']!=row['candidate'] or old['finite_paths']:
                raise ValueError('C fixed admission contains a different point or existing path')
    oldplan=read_ref((BASE,'docs/research/d_p2_direct_pilot_20261113/PLAN.json'))
    oldcells=oldplan['cells'];saved=[]
    for m in (0,6):
        for axis in ('X','Y'):
            i,r=next((i,r) for i,r in enumerate(admission['rows']) if (r['m'],r['axis'])==(m,axis))
            paths=[p for p in r['finite_paths'] if p['phase']=='P2' and p['acquisition_phase']=='fresh' and not p['replay']]
            if len(paths)!=4 or sorted(p['index'] for p in paths)!=[0,1,2,3]:raise ValueError('Saved P2 fresh metadata incomplete')
            saved.append(dict(m=m,axis=axis,admission=dict(commit=C_INDEX[0],path=C_INDEX[1],selector=['rows',i]),
                paths=[{k:p[k] for k in ('index','cohort','role','source','scientific_source','seeds','laws','original_receipt','axis_receipt')} for p in paths]))
    seed_sets=[{x for row in c['seeds'] for x in row} for c in [*oldcells,c3,c4]]
    if any(seed_sets[i]&seed_sets[j] for i in range(len(seed_sets)) for j in range(i)):raise ValueError('Calibration/holdout seed overlap')
    plan.update(schema='PR_DF_LD1_M3_INPUTS_V1',menus={'H4':dict(plan['menus']['H4'],candidates=[r3])},cells=[c3],
        source_label=BASE,compile_budget=8,trajectory_paths=4,calibration_saved_cells=oldcells,saved_P2_fresh_inputs=saved,
        future_holdout=dict(row=r4,cell=c4,execution_authorized=False,prices_opened=False),
        model_plan=dict(formula='saved_E_I[S] + theta0 + thetaq*q + thetaL*qR',calibration_rounds=[0,3,6],holdout_round=4,
          responses=['X','Y'],target='Q_SPECIFIED_FINITE_COHORT',residual_aggregation_authorized=False,fit_authorized=False,
          holdout_rule=dict(n_paths=4,axes=['X','Y'],O='same-axis sum of four compiled RZ / 4',signed_error='P-O',absolute_error='abs(P-O)',
            signed_relative='(P-O)/O',absolute_relative='abs(P-O)/O',zero_observed='relative null: ZERO_OBSERVED_PRICE',
            missing='O and errors null; never change n_paths',tolerance=None,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE')),
        C_plan=dict(commit=C_PLAN[0],path=C_PLAN[1]),C_admission=dict(commit=C_INDEX[0],path=C_INDEX[1]),
        source_facts=dict(commit=FACTS[0],path=FACTS[1]),scalar_saved=facts['scalar_saved'],expected_preparation_hash=facts['preparation_hash'],
        profile=facts['profile'],pipeline_source=BASE,public=PUBLIC,private=PRIVATE,
        not_in_scope=['m0/m6 recompile','P1/replay pool','m4 acquisition/opening','residual aggregation','fit','m4 prediction','law distance','CI','N/G','retry','additional paths'])
    cfg=json.loads(Path(plan['runtime']['input_config']).read_text())['systems']['H4']
    plan['saved_input_files']=cfg['inputs']
    return plan


def progress(path,stage,state,**fields):
    with Path(path).open('a') as out:
        out.write(json.dumps(dict(stage=stage,state=state,monotonic=time.perf_counter(),peak_rss_bytes=R.rss(),**fields))+'\n');out.flush()


def run_path(index, state=None):
    state=STATE if state is None else state
    plan,ctx,backend,science=(state[k] for k in ('plan','ctx','backend','science'))
    cell=plan['cells'][0];candidate=science.m.row_for(plan,cell)['candidate']
    private=Path(state['private']);log=private/f'PROGRESS_p{index}.jsonl';start=time.perf_counter()
    directory=private/'fresh'/cell['id']/str(index)
    compile_log=[];active_axis={'value':None};original_compile=ctx.compile;original_compose=ctx.h['compose'];original_struct=F.structural_pair
    original_write=science.write;stack=(private/f'STACK_p{index}.txt').open('a')
    faulthandler.dump_traceback_later(60,repeat=True,file=stack)
    def timed(name,fn):
        progress(log,name,'START')
        try:return fn()
        finally:progress(log,name,'END')
    def compile_one(wrapper,settings,**kw):
        if len(compile_log)>=2:raise ValueError('PER_PATH_TWO_COMPILE_LIMIT')
        axis=active_axis['value']
        if axis not in ('cosine','sine'):raise ValueError('AXIS_COMPILE_START_NOT_IDENTIFIED')
        if wrapper.num_qubits!=ctx.prep.num_system_qubits+1 or wrapper.num_clbits!=1 or dict(wrapper.count_ops()).get('measure')!=1:raise ValueError('FULL_MEASURED_WRAPPER_REQUIRED')
        entry=dict(axis=axis,status='STARTED');compile_log.append(entry);then=time.perf_counter()
        try:
            value=timed(axis+'_compile',lambda:original_compile(wrapper,settings,**kw));entry['status']='COMPLETE';return value
        finally:entry.update(seconds=time.perf_counter()-then,peak_rss_bytes=R.rss())
    def write_observed(path,value):
        name=Path(path).name
        if name in ('cosine.started.json','sine.started.json'):active_axis['value']=name.split('.')[0]
        return timed('save_'+name,lambda:original_write(path,value))
    ctx.compile=compile_one
    ctx.h['compose']=lambda *a,**k:timed('compose',lambda:original_compose(*a,**k))
    F.structural_pair=lambda *a,**k:timed('structural_pair',lambda:original_struct(*a,**k))
    science.write=write_observed
    try:
        requests,seeds,laws=timed('sample_requests',lambda:ctx.h['sample_requests'](candidate,ctx.prep,cell['cohort'],index))
        if seeds!=cell['seeds'][index]:raise ValueError('FROZEN_NORMAL_SEEDS_DIFFER')
        batches=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
        event_digest=science.m.digest(batches);event_metrics=P.event_metrics([e for row in batches for e in row]);del batches
        with E.efficient_controlled_zz(state['dfq']):
            result=timed('price_requests',lambda:backend.price_requests(cell,index,directory,requests,seeds,laws,conditional=False))
        if result['new_compiles']!=len(compile_log) or result['event_digest']!=event_digest:raise ValueError('COMPILE_COUNT_OR_SAME_EVENT_DIGEST_DIFFERS')
        axes={a:json.loads((directory/(a+'.json')).read_text()) for a in science.m.AXES}
        for ai,axis in enumerate(science.m.AXES):
            r=axes[axis]
            if r['compiler']!=plan['compiler']['settings_hash'] or r['fingerprint']!=result['fingerprints'][ai] or r['metrics']['rz_count']!=result['C'][ai]:
                raise ValueError('AXIS_RECEIPT_DIFFERS')
            if any(type(v) is not int or v<0 for v in (result['S'][ai],result['C'][ai])):raise ValueError('INVALID_INTEGER_SC_PRICE')
            if r['cache_hit']:raise ValueError('OLD_CACHE_ORIGINAL_EXECUTION_SOURCE_REQUIRES_ADMISSION_STOP')
        record=dict(result,m=cell['m'],phase='fresh',role=cell['role'],request=candidate,law_binding=science.m.row_for(plan,cell)['law_binding'],
            seeds=seeds,laws=laws,events=event_metrics,lowering=E.LOWERING_ID,scalar=ctx.scalar,scalar_hex=float(ctx.scalar).hex(),
            scalar_application='ONCE_AT_TOTAL_T',compiled={a:r['metrics'] for a,r in axes.items()},
            compile_log=compile_log,timing=json.loads((directory/'TECHNICAL_TIMING.json').read_text()),
            path_wall_seconds=time.perf_counter()-start,peak_rss_bytes=R.rss(),private_directory=str(directory),
            gate_status='INPUT_PREPARATION_REQUEST_LOWERING_COMPILER_FINGERPRINT_CHECKED',
            numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',full_wrapper_statevector_validation=False)
        R.write(directory/('M%d_PATH.json'%cell['m']),record);progress(log,'path','COMPLETE',index=index)
        return record
    except BaseException:
        R.write(private/f'STOP_p{index}.json',dict(index=index,compile_log=compile_log,error=traceback.format_exc(),partial_directory=str(directory)))
        progress(log,'path','STOP',index=index);raise
    finally:
        ctx.compile=original_compile;ctx.h['compose']=original_compose;F.structural_pair=original_struct;science.write=original_write
        faulthandler.cancel_dump_traceback_later();stack.close()


def acquire_four(out):
    context=mp.get_context('fork');pending={};rows=[]
    for index in range(4):
        recv,send=context.Pipe(duplex=False)
        def target(index=index,recv=recv,send=send):
            recv.close()
            try:send.send(dict(ok=True,value=run_path(index)))
            except BaseException:send.send(dict(ok=False,error=traceback.format_exc()))
            finally:send.close()
        process=context.Process(target=target);process.start();send.close();pending[recv]=(index,process)
        print('START_PATH',index,'pid',process.pid,flush=True)
    while pending:
        for recv in wait(pending,timeout=10):
            index,process=pending.pop(recv)
            try:message=recv.recv()
            except EOFError:message=dict(ok=False,error='Worker exited without a receipt')
            process.join();recv.close();record=dict(index=index,exitcode=process.exitcode,**message)
            R.write(out/f'p{index}.json',record);rows.append(record)
            print('END_PATH',index,'ok',record['ok'],'exit',process.exitcode,flush=True)
        if pending:print('INFLIGHT_PATHS',[v[0] for v in pending.values()],flush=True)
    return sorted(rows,key=lambda r:r['index'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--source');parser.add_argument('--inputs',type=Path)
    args=parser.parse_args()
    if args.prepare:R.write(args.prepare,make_inputs());return 0
    if not (args.execute and args.source and args.inputs):parser.error('Use --prepare or --execute --source --inputs')
    if git('rev-parse','HEAD')!=args.source or git('status','--porcelain','--untracked-files=no'):raise ValueError('PINNED_COMMITTED_EXECUTION_SOURCE_REQUIRED')
    plan=json.loads(args.inputs.read_text())
    if plan!=make_inputs():raise ValueError('FIXED_INPUTS_DIFFER')
    for path in ('src/trottertracks/p1_efficient_zz.py','src/trottertracks/p2_direct_pilot.py','src/trottertracks/compiled_price_fast.py','src/trottertracks/efficient_controlled_zz.py','scripts/h46_b3_fast_fingerprint.py'):
        if (ROOT/path).read_bytes()!=blob(BASE,path):raise ValueError('FIXED_P1_P2_SOURCE_DIFFERS:'+path)
    for item in plan['saved_input_files'].values():
        if R.file_sha(item['path'])!=item['sha256']:raise ValueError('SAVED_INPUT_FILE_DIFFERS')
    out=ROOT/PUBLIC/'run01';private=Path(PRIVATE)
    out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();records=[];parent_log=private/'PARENT_PROGRESS.jsonl'
    base=dict(execution_source=args.source,pipeline_source=BASE,inputs_sha256=R.file_sha(args.inputs),compile_budget=8,resources_before=R.resources(),mandatory_stop=True)
    stack=(private/'PARENT_STACK.txt').open('a');faulthandler.dump_traceback_later(60,repeat=True,file=stack)
    try:
        progress(parent_log,'context','START')
        science=P.load_c1_backend(ROOT)['r_price_phase1_science'];cls=F.make_backend_class(science.Backend,science)
        backend=cls(ROOT,P.execution_plan(plan,args.source,ROOT));ctx=backend.context('H4')
        if ctx.prep.preparation_hash!=plan['expected_preparation_hash'] or float(ctx.scalar).hex()!=plan['scalar_saved']['hex']:raise ValueError('PREPARATION_OR_SCALAR_DIFFERS')
        import trotterlib.df_rte_qiskit as dfq
        progress(parent_log,'context','END');faulthandler.cancel_dump_traceback_later();stack.close()
        STATE.update(plan=plan,ctx=ctx,backend=backend,science=science,dfq=dfq,private=str(private))
        R.write(out/'PREPARATION_CHECK.json',dict(source=args.source,preparation_hash=ctx.prep.preparation_hash,scalar=ctx.scalar,
            scalar_hex=float(ctx.scalar).hex(),fragment_order=list(ctx.prep.deterministic_fragment_indices),status='SAVED_CONTEXT_CHECKED',full_wrapper_statevector_validation=False))
        records=acquire_four(out)
        success=all(r['ok'] and r['exitcode']==0 for r in records) and len(records)==4
        logs=[]
        for record in records:
            log=(record.get('value') or {}).get('compile_log')
            if log is None:
                stop=private/f"STOP_p{record['index']}.json"
                log=json.loads(stop.read_text())['compile_log'] if stop.exists() else []
            logs.extend(log)
        compiles=sum(r['status']=='COMPLETE' for r in logs)
        R.write(out/'RESULT.json',dict(base,status='M3_SC_PAIRS_ACQUIRED' if success else 'STOP_PARTIAL',records=records,new_compiles=compiles,compile_attempts=len(logs),
            total_wall_seconds=time.perf_counter()-start,resources_after=R.resources(),residual_aggregation=False,fit=False,m4_acquisition=False,m4_prediction=False,
            all_law_E_C=None,law_distance=None,CI=None,N=None,G=None))
        return 0 if success else 2
    except BaseException:
        R.write(out/'STOP.json',dict(base,status='STOP',error=traceback.format_exc(),records=records,total_wall_seconds=time.perf_counter()-start))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if not stack.closed:stack.close()

if __name__=='__main__':raise SystemExit(main())
