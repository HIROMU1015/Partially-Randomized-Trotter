"""Fixed B0I p7/S2 width h: reuse six axes, acquire only m1/m2 X/Y.

Reuses the m0 caller's native child, fast workers, original seven gates and
T3's record builder/C guard. No predictor, fit, Q3, allocation or total cost.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_pf_m0_prices as M

MODULE = 'scripts/run_pf_p7_h_prices.py'
STUDY = 'docs/research/d_pf_p7_s2_h_direct_prices_20261115'
PRIVATE = M.T.PRIVATE_ROOT/'p7_s2_h_m1_m2/run01'
CONFIG = 'e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc'
COHORT = 'D_P7_S2_H_DIRECT_V1'  # Deterministic run label; no trajectory sampling.
PLAN_COMMIT = '486fa84ccf2b63c73f71d3adcedada72dc6ebe99'
PLAN_DIR = 'docs/research/d_pf_width_price_coverage_20261115/'
BASE = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
SOURCES = M.SOURCES+(MODULE,)
RAW = M.RAW+('ORIGINAL_TRIALS.json',)
TARGETS = {1:('7d1659503f9d99381006',.2,2),2:('7fb12f906ab2f8cd8083',.4,4)}
REUSE = {0:'d2da7de820833a4c6334',3:'398f55738920acb1d4a5',4:'b4156f8b791f26be5ae6'}
check,write = M.check,M.write


def read(ref):
    # Stdlib before provider install: never import the local scientific package.
    raw = subprocess.check_output(['git','show',ref['commit']+':'+ref['path']],cwd=ROOT)
    if ref['path'].endswith('.gz'):
        import gzip
        raw=gzip.decompress(raw)
    value=json.loads(raw)
    for key in ref.get('selector',[]): value=value[key]
    return value


def template():
    C,I=M.T._meta(); manifest=read(BASE); cfg=manifest['candidates'][1]['configuration']
    check(cfg['id']==CONFIG and cfg['spec']['L_D']==7 and cfg['spec']['formula']=='2nd','Wrong width-h configuration')
    plan=I.plan(ROOT,cfg,target=manifest['shared']['target'],signal_source=dict(price_only=MODULE),references=())
    fixed=read(dict(commit=PLAN_COMMIT,path=PLAN_DIR+'FIRST_FINITE_PLAN.json'))
    rows=[];cells=[]
    for m,(ident,T,q) in TARGETS.items():
        req=next(r for r in plan['pending']['requests'] if r['m']==m);c=req['physical_candidate']
        check(c==dict(system='H4',method='B0I',prefix=7,formula='2nd',T=T,q=q,delta=.1,R_outer=None,K=None,id=ident),'Target physical request differs')
        check(cfg['rounds'][m]['structure']['signed_weights']==[1.],'Signed S2 differs')
        for axis in ('X','Y'):
            row=dict(configuration_id=CONFIG,physical_id=ident,m=m,axis=axis,physical_candidate=c)
            check(row in fixed['targets'],'Not a fixed target axis'); rows.append(row)
        cells.append(dict(m=m,formula='2nd',row=M.T.native_row(c),role='P7_S2_H_DIRECT_ONLY'))
    check(len(rows)==len(fixed['targets'])==4,'Exactly four fixed target axes required')
    return plan,cells,rows


def reuse_trials():
    """Read preserved JSON only; preserve producer/cohort and never resimulate."""
    cover=read(dict(commit=PLAN_COMMIT,path=PLAN_DIR+'PRICE_COVERAGE.json'))
    result={}
    for m,ident in REUSE.items():
        rows=[r for r in cover['rows'] if r['configuration_id']==CONFIG and r['m']==m]
        check(len(rows)==2 and {r['axis'] for r in rows}=={'X','Y'} and all(r['physical_id']==ident for r in rows),'Reuse axis binding differs')
        if m==0:
            source=rows[0]['price_evidence'][0]['receipt'];old=read(source)
            ref=dict(commit='a7d066caed56a0c7d8e64c617606f99d0fe4e634',path=source['path'].replace('PRICES.json','RESULT.json'),selector=['results',0,'trial'])
            trial=read(ref);producer=old['source'];cohort=old['cohort']
        else:
            item=next(v for v in rows[0]['price_evidence'] if 'trial' in v);ref=item['trial']
            if m==3:
                original=read(ref)
                trial=dict(original,prices={a:original['prices'][a]['metrics'] for a in ('cosine','sine')})
                # Existing carrier-level label; native common trial has no
                # cohort field. Keep that absence in original_cohort below.
                cohort='SAVED_T0.8_COMMON'
            else:
                raw=Path(ref['path']).read_bytes();check(hashlib.sha256(raw).hexdigest()==ref['sha256'],'Original private trial changed')
                original=json.loads(raw);trial=original;cohort=original['cohort']
            producer=dict(commit=item['generation_SOURCE'],path='scripts/pf_common_run_native.py' if m==3 else 'scripts/h46_long_time_native.py')
            subprocess.check_output(['git','cat-file','-e',producer['commit']+':'+producer['path']],cwd=ROOT)
        check(trial['status']=='PASS' and trial['candidate']['q']==2**m,'Original trial status/q differs')
        context=cover['contexts']['7|2nd']
        check(trial['scalar'].hex()==context['scalar']['value'].hex(),'Original total scalar differs')
        check(all(g.get('PASS',True) and g['error']<g.get('tolerance',g.get('strictly_below',0)) for g in trial['checks']),'Original gate failed')
        for r in rows:
            axis='cosine' if r['axis']=='X' else 'sine'
            check(trial['prices'][axis]['rz_count']==r['saved_compiled_RZ'],'Original saved RZ differs')
        result[str(m)]=dict(trial=trial,original_locator=ref,original_producer=producer,cohort=cohort,
                            original_cohort=trial.get('cohort'),coverage_rows=rows,
                            trial_view='ORIGINAL_JSON_PRICES_PROJECTED_TO_METRICS' if m==3 else 'ORIGINAL_JSON')
    return result


def preflight():
    plan,cells,rows=template(); reused=reuse_trials()
    audit=M.duplicate_check((1,2),selected=rows)
    check(not (ROOT/STUDY/'run01/RESULT.json').exists() and not (PRIVATE/'child.json').exists(),'Namespace already acquired; no retry')
    # The old audit recognizes the shared caller; inspect this new thin caller.
    import os
    jobs=[]
    for pid in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid='],text=True).split():
        if int(pid)==os.getpid():continue
        try:a=Path('/proc',pid,'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError,PermissionError):continue
        if a and Path(a[0]).name.startswith('python') and any(Path(s).name==Path(MODULE).name for s in a[1:]) and any(s in a for s in ('--execute-approved','--child')):
            jobs.append(dict(pid=int(pid),caller=MODULE))
    if jobs:
        for r in audit['rows']:r.update(action='HELD',findings=r['findings']+[dict(running_jobs=jobs)])
    audit.update(plan_source=dict(commit=PLAN_COMMIT,path=PLAN_DIR+'FIRST_FINITE_PLAN.json'),configuration_source=dict(BASE,selector=['candidates',1,'configuration']),
                 B_Q3_result=dict(commit='bec032fc26093177f529754060df06d5375944b4',path='docs/research/p7_s2_d01_same_shots_q3_results_20261115/HANDOFF.md'),
                 reusable_axes=6,additional_caller_jobs=jobs,original_trials={m:{k:v for k,v in r.items() if k!='trial' and k!='coverage_rows'} for m,r in reused.items()})
    return audit


def records(plan,cell,trial,source,cohort,*,evidence_path=None):
    a,b,c=M.T.price_bindings(ROOT,plan,cell,trial,price_source=source,evidence_commit=M.S.PLACEHOLDER,evidence_path=evidence_path or STUDY+'/run01/IMPLEMENTATION_EVIDENCE.json')
    for price,side in zip(a,b,strict=True):
        price['cohort']=side['cohort']=cohort
        key=side['implementation_evidence']['selector'][0]
        c[key]={k:deepcopy(side[k]) for k in ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
    return a,b,c


def validate_job(job):
    check(job['route']=='TRUNCATED_PF' and job['configuration_id']==CONFIG,'Wrong child context')
    fixed=read(dict(commit=PLAN_COMMIT,path=PLAN_DIR+'FIRST_FINITE_PLAN.json'))
    allowed={r['physical_id']:r['physical_candidate'] for r in fixed['targets']}
    check(len(job['cells'])<=2 and len({c['row']['id'] for c in job['cells']})==len(job['cells']),'Duplicate/too many cells')
    check(all(c['m'] in TARGETS and c['row']['id']==TARGETS[c['m']][0] and c['row']==allowed[c['row']['id']] for c in job['cells']),'Foreign child cell')
    check(job['rounds']==[1,2],'Foreign child round scope')
    check(job['cohort']==COHORT,'Wrong deterministic run label')


def child(job_path):
    job=json.loads(Path(job_path).read_text());validate_job(job)
    for path in SOURCES:
        check((ROOT/path).read_bytes()==subprocess.check_output(['git','show',job['execution_SOURCE']+':'+path],cwd=ROOT),'Child SOURCE changed')
    return M.child(job_path)


def execute():
    source=M.S.source_gate(ROOT,SOURCES);audit=preflight();plan,cells,_=template();reused=reuse_trials()
    # All four axes must be resolved before a native pair is scheduled. A held
    # pair stays held; another resolved cell may proceed without duplication.
    eligible=[c for c in cells if all(r['action']=='NEW_COMPILE_MISSING_CONFIRMED' for r in audit['rows'] if r['physical_id']==c['row']['id'])]
    out=ROOT/STUDY/'run01';out.mkdir(parents=True,exist_ok=False);PRIVATE.mkdir(parents=True,exist_ok=False)
    write(out/'PREFLIGHT.json',audit);write(out/'ORIGINAL_TRIALS.json',reused)
    from pf_common_run_contract import resources,workers
    obs=resources();n=workers(obs,len(eligible),2*2**30,100*2**20)
    saved=read(dict(commit=M.BASELINE,path=M.T.OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    job=dict(route='TRUNCATED_PF',configuration_id=CONFIG,cohort=COHORT,rounds=[1,2],cells=eligible,public=str(out/'child_public'),private=str(PRIVATE),workers=max(1,n),
             expected_scalar=saved['facts']['scalar'],tail_scalar=saved['facts']['tail_identity'],preparation_hash=saved['trial']['preparation_hash'],execution_SOURCE=source)
    write(out/'JOB_METADATA.json',job);write(PRIVATE/'job.json',job)
    start=time.monotonic();code=None;data=dict(results=[],resources=None)
    if eligible and n:
        with (PRIVATE/'child.log').open('x') as log:
            code=subprocess.call([sys.executable,'-B',str(ROOT/MODULE),'--child',str(PRIVATE/'job.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        data=json.loads((PRIVATE/'child.json').read_text()) if (PRIVATE/'child.json').exists() else dict(results=[],error='Child output missing')
    elif eligible:data['error']='No available resource slot; no science entered'
    prices=[];sides=[];proofs={}
    for m,r in reused.items():
        origin=dict(commit=source,binding_caller=MODULE,kind='SAVED_DIRECT_RECEIPT_BINDING',original_producer=r['original_producer'],original_trial=r['original_locator'],original_cohort=r['original_cohort'],cohort_label_role='CARRIER_LABEL_NOT_A_NEW_SAMPLE' if r['original_cohort'] is None else 'ORIGINAL_COHORT')
        cell=dict(m=int(m));a,b,c=records(plan,cell,r['trial'],origin,r['cohort'])
        for p in a:p['diagnostic'].update(acquisition='REUSED',original_axis_price=next(v for v in r['coverage_rows'] if v['axis']==p['axis'])['price_evidence'])
        prices+=a;sides+=b;proofs.update(c)
    for r in data['results']:
        if not r['ok']:continue
        origin=dict(commit=source,caller=MODULE,underlying_caller=M.T.CALLER,native=M.T.NATIVE,wrapper=M.T.WRAPPER,stage='P7_S2_H_DIRECT_ONLY')
        a,b,c=records(plan,r['cell'],r['trial'],origin,COHORT)
        for p in a:p['diagnostic'].update(acquisition='NEW_COMPILE',compiled_wrapper=r['wrappers']['cosine' if p['axis']=='X' else 'sine'])
        prices+=a;sides+=b;proofs.update(c)
    for name,value in zip(M.RAW[1:],(prices,sides,proofs)):write(out/name,value)
    result=dict(execution_SOURCE=source,configuration_id=CONFIG,results=data['results'],context_gate=data.get('context_gate'),error=data.get('error'),
                child_exit_code=code,child_signal=-code if code is not None and code<0 else None,resources=obs,worker_resources=data.get('resources'),
                wall_seconds=time.monotonic()-start,child_wall_seconds=data.get('wall_seconds'),context_peak_RSS_bytes=data.get('context_peak_RSS_bytes'),
                reused_axis_records=6,new_axis_records=len(prices)-6,axis_records_before_C_guard=len(prices),scientific_retry=False,fit_calls=0,model_predictions=0,Q3_calls=0,N=None,G=None,mandatory_stop=True)
    write(out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('child_exit_code','reused_axis_records','new_axis_records','wall_seconds','error')}))


def bind(publication):
    plan,_,_=template()
    return bind_records(plan,STUDY+'/run01',publication)


def bind_records(plan,relative,publication,*,names=RAW,policy_id='P7_S2_H_DIRECT_IMPLEMENTATION_ONLY_V1',route='TRUNCATED_PF'):
    """Bind an already published native stage using the existing C guard."""
    out=ROOT/relative;fixed=M.S.committed(ROOT,relative,names,commit=publication)
    M.T._meta()
    from trottertracks import price_implementation_guard as G
    cfg=plan['configuration'];prices=json.loads((out/'PRICES.json').read_text());sides=json.loads((out/'SIDECARS_UNBOUND.json').read_text())
    corrections=[]
    for s in sides:
        check(s['implementation_evidence']['commit']==M.S.PLACEHOLDER,'Already bound')
        expected=relative+'/IMPLEMENTATION_EVIDENCE.json'
        if s['implementation_evidence']['path']!=expected:
            corrections.append(dict(axis=s['axis'],selector=s['implementation_evidence']['selector'],original=s['implementation_evidence']['path'],bound=expected))
        s['implementation_evidence']['commit']=publication
        s['implementation_evidence']['path']=expected
    cohorts={(cfg['rounds'][p['m']]['candidate']['id'],p['axis']):p['cohort'] for p in prices}
    scalar_quantity=M.F.SCALAR_QUANTITY if route=='FULL_PF' else M.T.SCALAR_QUANTITY
    policy=dict(id=policy_id,source=M.S.GUARD_SOURCE,profile=cfg['spec']['profile'],by_method={route:dict(native_policy=M.T.NATIVE_POLICY,rte_rotation_lowering=M.T.NO_RTE,scalar_quantity=scalar_quantity)})
    result=G.check(ROOT,cfg,prices,cohorts=cohorts,bindings=sides,policy=policy)
    check(len(result['checked'])==len(prices),'Unchecked axis receipt')
    write(out/'SIDECARS.json',sides);write(out/'GUARD_CHECK.json',dict(raw_evidence=fixed,guard=result,model_qualified=False,binding_path_corrections=corrections))
    print(json.dumps(dict(guarded_axes=len(prices))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata-output',type=Path);p.add_argument('--execute-approved',action='store_true');p.add_argument('--child',type=Path);p.add_argument('--bind-publication')
    a=p.parse_args()
    if a.metadata_output:write(a.metadata_output,preflight())
    elif a.execute_approved:execute()
    elif a.child:return child(a.child)
    elif a.bind_publication:bind(a.bind_publication)
    else:p.error('Select metadata, fixed approved execution, or receipt binding')
    return 0


if __name__=='__main__':sys.exit(main())
