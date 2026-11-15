"""Fixed coarse-Y4 calibration (m5/m6), then frozen m9 wrapper holdout."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_pf_p7_h_prices as P
M=P.M
check,write,read=P.check,P.write,P.read
MODULE='scripts/run_p7_y4_coarse_prices.py'
STUDY='docs/research/d_p7_y4_coarse_price_calibration_20261116'
OUT=ROOT/STUDY
PRIVATE=M.T.PRIVATE_ROOT/'coarse_y4_32o5'
CONFIG='dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0'
CONFIG_SOURCE=dict(commit='3179680ed9b4cff251b3401cabea2f6100cf47fb',path='docs/research/d_p7_y4_coarse_signal_q3_20261116/SIGNAL_INPUTS.json',selector=['configuration'])
PLAN_SOURCE=dict(commit='d01c780d035eb096186ef1b8aa6d5e580bca2a50',path='docs/research/d_pf_first_coarser_width_plan_20261116/FIRST_FINITE_PLAN.json')
Q3_SOURCE=dict(commit='8855f420a9fe2577efc77f099c5a39ab0cb9b7e6',path='docs/research/d_p7_y4_coarse_a_shape_q3_20261116/C_RESULT_JOIN.json')
STAGES=dict(CALIBRATION=(5,6),HOLDOUT=(9,))
TARGETS={5:('4eccd52bab8d4d790ab9',3.2,5),6:('77d24715893869133820',6.4,10),9:('f0a99e91997c44c59ad4',51.2,80)}
VECTOR=[37,34,32,29,27,24,22,19,17,14,11,9,6]
SOURCES=P.SOURCES+(MODULE,'references/h46_m5_tol_pilot_inputs_v1.json')
RAW=M.RAW
RUNS=dict(CALIBRATION='run02',HOLDOUT='run01')
NATIVE_METADATA='references/h46_m5_tol_pilot_inputs_v1.json'
METADATA_SOURCE='2a8f861ecbf89fec75f762a0cb27389c3317183d'
POLICY='P7_Y4_32O5_DIRECT_WRAPPER_IMPLEMENTATION_V1'


def template():
    C,I=M.T._meta();cfg=read(CONFIG_SOURCE);manifest=read(M.T.MANIFEST)
    check(cfg['id']==CONFIG and cfg['spec']['L_D']==7 and cfg['spec']['formula']=='4th','Wrong coarse candidate')
    plan=I.plan(ROOT,cfg,target=manifest['shared']['target'],signal_source=dict(price_only=MODULE),references=())
    fixed=read(PLAN_SOURCE);schedule=read(fixed['schedule_source'])
    cells={};rows=[]
    for req in plan['pending']['requests']:
        m=req['m'];c=req['physical_candidate'];s=schedule[m]
        check(c['id']==s['physical_candidate_id'] and c['q']==s['q_PF'],'Original physical schedule differs')
        check(req['structure']['signed_weights']==[1.3512071919596578,-1.7024143839193153,1.3512071919596578],'Signed Yoshida4 differs')
        for axis in ('X','Y'):rows.append(dict(configuration_id=CONFIG,m=m,axis=axis,physical_id=c['id'],physical_candidate=c))
        if m in TARGETS:
            ident,T,q=TARGETS[m]
            check(c==dict(system='H4',method='B0I',prefix=7,formula='4th',T=T,q=q,delta=.64,R_outer=None,K=None,id=ident),'Calibration/holdout physical condition differs')
            cells[m]=dict(m=m,formula='4th',row=M.T.native_row(c),role='CALIBRATION' if m in (5,6) else 'HOLDOUT')
    q3=read(Q3_SOURCE)
    check(q3['configuration_id']==CONFIG and q3['model_status']=='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD','No matching saved Q3 acceptance')
    check(q3['allocation']['shots']==[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(VECTOR)],'Wrong adopted A_SHAPE')
    return plan,cells,rows


def saved_early(plan):
    """Source-bound direct receipts; no circuit, recount or early-point fit."""
    _,I=M.T._meta();requests={(r['m'],r['axis']):r for r in I.price_requests(ROOT,plan)}
    facts=read(dict(commit=M.BASELINE,path=M.T.OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    output=[]
    for item in read(PLAN_SOURCE)['saved_direct_price_evidence']:
        m,axis=item['m'],item['axis'];entry=item['evidence'][0];ref=entry['receipt'];old=read(ref);req=requests[m,axis]
        check(old['physical_context']==req['physical_context'],'Early physical context differs')
        for key in ('axis','measurement_axis','observable','outcome_convention','units','quantity','P'):
            check(old[key]==req[key],'Early request differs: '+key)
        # m3 has an already C-admitted legacy native producer; preserve its
        # exact SOURCE and equivalence proof instead of relabelling it af284.
        native_source=(dict(commit='e2179f68f6979f3ad26392bff877fb422ddfb744',path='scripts/h46_m5_native.py') if m==3 else M.T.NATIVE)
        check(old['profile_extension']==dict(lowering_id=M.T.NATIVE_POLICY,source=native_source) and old['price_profile']==dict(plan['configuration']['spec']['profile'],lowering_id=M.T.NATIVE_POLICY),'Realized native/compiler profile differs')
        check(old['classification']=='DIRECT_COMPILED_DETERMINISTIC' and old['expectation']=='DETERMINISTIC','Not a direct deterministic price')
        admitted=read(entry['proof_of_current_admission'])
        check(admitted['physical_id']==item['physical_id']==old['physical_context']['physical_candidate']['id'] and admitted['axis']==axis and admitted['C']==old['mean']['rz_count'],'Original admission differs')
        if m==3:
            top=read(dict(commit=ref['commit'],path=ref['path']))
            origin=next(r['origin'] for r in top['rows'] if r['physical_id']==item['physical_id'] and r['axis']==axis)
            check(origin['generation_SOURCE']==native_source['commit'],'C-admitted legacy producer differs')
            trial_ref=origin['original_trial'];trial=read(trial_ref)
            side=next(s for s in top['sides'] if s['physical_context']['physical_candidate']['id']==item['physical_id'] and s['axis']==axis)
            metrics=trial['prices']['cosine' if axis=='X' else 'sine']['metrics']
        else:
            trial_path=ref['path'].replace('PRICES.json','RESULT.json')
            value=read(dict(commit=ref['commit'],path=trial_path))
            matches=[(i,r) for i,r in enumerate(value['results']) if r['cell']['row']['id']==item['physical_id']]
            check(len(matches)==1 and matches[0][1]['ok'],'Original trial missing or failed')
            index,result=matches[0];trial=result['trial'];metrics=trial['prices']['cosine' if axis=='X' else 'sine']
            trial_ref=dict(commit=ref['commit'],path=trial_path,selector=['results',index,'trial'])
            sides=read(dict(commit=ref['commit'],path=ref['path'].replace('PRICES.json','SIDECARS_UNBOUND.json')))
            side=next(s for s in sides if s['physical_context']['physical_candidate']['id']==item['physical_id'] and s['axis']==axis)
        check(trial['status']=='PASS' and metrics['rz_count']==old['mean']['rz_count'],'Original compiled trial differs')
        check(trial['scalar'].hex()==facts['facts']['scalar'].hex()==side['implementation']['resolved_scalar'].hex(),'Original scalar differs')
        check(side['implementation']['scalar_application']=='ONCE_AT_TOTAL_T' and side['implementation']['scalar_quantity']==M.T.SCALAR_QUANTITY,'Original scalar quantity differs')
        check(all(c.get('PASS',True) and c['error']<c.get('tolerance',c.get('strictly_below',0)) for c in trial['checks']),'Original action gate failed')
        output.append(dict(m=m,axis=axis,physical_id=item['physical_id'],physical_candidate=deepcopy(req['physical_context']['physical_candidate']),
            compiled_RZ=old['mean']['rz_count'],receipt=ref,original_trial=trial_ref,original_SOURCE=old['source'],original_cohort=old['cohort'],
            proof_of_current_admission=entry['proof_of_current_admission'],physical_context_equal=True,scalar_hex=trial['scalar'].hex(),
            profile=old['price_profile'],native_source=deepcopy(native_source),native_reuse_basis='EXISTING_C_ADMISSION_EQUIVALENCE' if m==3 else 'SAME_NATIVE_SOURCE',gate='ACCEPTED_ORIGINAL_DIRECT_PHYSICALLY_MATCHED',fit_role='EXCLUDED_EARLY_WIDTH_DIFFERS'))
    check(len(output)==10,'Exactly ten early direct axes required')
    return output


def live_jobs():
    jobs=[]
    for pid in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid='],text=True).split():
        if int(pid)==os.getpid():continue
        try:a=Path('/proc',pid,'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (FileNotFoundError,PermissionError):continue
        if a and Path(a[0]).name.startswith('python') and any(Path(x).name==Path(MODULE).name for x in a[1:]) and ('--execute' in a or '--child' in a):jobs.append(dict(pid=int(pid),caller=MODULE))
    return jobs


def prepare():
    plan,cells,rows=template();early=saved_early(plan)
    audit=M.duplicate_check((1,2),selected=[r for r in rows if r['m']>=5])
    jobs=live_jobs();check(not jobs,'Same candidate caller already running')
    check(all(r['action']=='NEW_COMPILE_MISSING_CONFIRMED' for r in audit['rows']),'Review matching later receipts/jobs before acquiring or modelling')
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'INPUTS.json',dict(configuration=CONFIG_SOURCE,configuration_id=CONFIG,cells={str(k):v for k,v in cells.items()},rows=rows,
        Q3_result=Q3_SOURCE,allocation=read(Q3_SOURCE)['allocation'],early_direct_axes=early,source_snapshot=dict(native=M.T.NATIVE,wrapper=M.T.WRAPPER),
        scalar_preparation_source=dict(commit=M.BASELINE,path=M.T.OUTPUT_ROOT+'/holdout/run01/RESULT.json',selector=['results',0]),
        stages={k:list(v) for k,v in STAGES.items()},maximum_new_compile=6,model_features=['one','q'],model_formula='C_axis(q)=alpha_axis+beta_axis*q',
        early_fit_excluded=True,holdout_refit=False,holdout_difference='OBSERVED_MINUS_FROZEN_PREDICTED',relative_difference_denominator='OBSERVED',
        holdout_tolerance=None,long_q640_error_bound=None,Q3_calls=0,new_signal=0,other_allocations=0))
    write(OUT/'PREFLIGHT.json',dict(duplicate_audit=audit,new_caller_jobs=jobs))
    print('PREPARED',len(early),'early axes; six new target axes; no science')


def validate_job(job):
    check(job['stage'] in STAGES and job['route']=='TRUNCATED_PF' and job['configuration_id']==CONFIG,'Wrong native stage/context')
    fixed=json.loads((OUT/'INPUTS.json').read_text());allowed=[fixed['cells'][str(m)] for m in STAGES[job['stage']]]
    check(job['rounds']==list(STAGES[job['stage']]) and job['cells']==allowed,'Wrong phase/cells/order')
    check(job['cohort']=='D_P7_Y4_32O5_'+job['stage']+'_V1','Wrong cohort/role')


def child(path):
    job=json.loads(Path(path).read_text());validate_job(job)
    for name in SOURCES+(STUDY+'/INPUTS.json',):
        check((ROOT/name).read_bytes()==subprocess.check_output(['git','show',job['execution_SOURCE']+':'+name],cwd=ROOT),'Unfixed child source/input')
    return M.child(path,validate_rounds=lambda r:check(tuple(r)==STAGES[job['stage']],'Foreign child round scope'))


def native_metadata_check():
    raw=(ROOT/NATIVE_METADATA).read_bytes()
    check(raw==subprocess.check_output(['git','show',METADATA_SOURCE+':'+NATIVE_METADATA],cwd=ROOT),'Fixed native metadata bytes differ')
    return dict(path=NATIVE_METADATA,commit=METADATA_SOURCE,git_blob=subprocess.check_output(['git','rev-parse',METADATA_SOURCE+':'+NATIVE_METADATA],cwd=ROOT,text=True).strip())


def execute(stage,frozen=None):
    source=M.S.source_gate(ROOT,SOURCES+(STUDY+'/INPUTS.json',STUDY+'/PLAN.md',STUDY+'/INITIALIZATION_RECOVERY.md'))
    native_metadata_check()
    if stage=='CALIBRATION':
        initial=json.loads((OUT/'calibration/run01/RESULT.json').read_text())
        check(not initial['results'] and initial['context_gate'] is None and initial['worker_resources'] is None and NATIVE_METADATA in initial['error'],'Only the preserved pre-import, zero-science failure may enter run02')
    check(not live_jobs(),'Equivalent price caller already running')
    plan,cells,_=template();fixed=json.loads((OUT/'INPUTS.json').read_text())
    check(fixed['cells']=={str(k):v for k,v in cells.items()},'Fixed physical inputs changed')
    if stage=='HOLDOUT':
        check(frozen is not None,'Freeze calibration and m9 predictions before holdout')
        frozen_value=read(dict(commit=frozen,path=STUDY+'/FROZEN_MODEL.json'))
        check((OUT/'FROZEN_MODEL.json').read_bytes()==subprocess.check_output(['git','show',frozen+':'+STUDY+'/FROZEN_MODEL.json'],cwd=ROOT),'Changed frozen model')
        check(frozen_value['configuration_id']==CONFIG and frozen_value['calibration_rounds']==[5,6] and frozen_value['holdout_round']==9 and set(frozen_value['models'])=={'X','Y'},'Wrong calibration/holdout design')
    out=OUT/stage.lower()/RUNS[stage];private=PRIVATE/stage.lower()/RUNS[stage]
    out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
    from pf_common_run_contract import resources,workers
    obs=resources();n=workers(obs,len(STAGES[stage]),2*2**30,100*2**20);check(n>0,'No available native resource slot')
    saved=read(fixed['scalar_preparation_source'])
    job=dict(stage=stage,route='TRUNCATED_PF',configuration_id=CONFIG,cohort='D_P7_Y4_32O5_'+stage+'_V1',rounds=list(STAGES[stage]),
        cells=[cells[m] for m in STAGES[stage]],public=str(out/'child_public'),private=str(private),workers=n,
        expected_scalar=saved['facts']['scalar'],tail_scalar=saved['facts']['tail_identity'],preparation_hash=saved['trial']['preparation_hash'],execution_SOURCE=source,frozen_model_source=frozen,runtime=dict(executable=sys.executable,version=sys.version))
    validate_job(job);write(out/'JOB_METADATA.json',job);write(private/'job.json',job)
    started=time.monotonic()
    with (private/'child.log').open('x') as log:code=subprocess.call([sys.executable,'-B',str(ROOT/MODULE),'--child',str(private/'job.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    data=json.loads((private/'child.json').read_text()) if (private/'child.json').exists() else dict(results=[],error='Missing child output')
    trace=out/'TRACE';trace.mkdir()
    for name in ['child.log','context_progress.jsonl']+[p.name for p in private.glob('progress_*.jsonl')]:
        if (private/name).exists():shutil.copyfile(private/name,trace/name)
    prices=[];sides=[];proofs={}
    for value in data['results']:
        if not value['ok']:continue
        origin=dict(commit=source,caller=MODULE,underlying_caller=M.T.CALLER,native=M.T.NATIVE,wrapper=M.T.WRAPPER,stage=stage)
        a,b,c=P.records(plan,value['cell'],value['trial'],origin,job['cohort'],evidence_path=STUDY+'/'+stage.lower()+'/'+RUNS[stage]+'/IMPLEMENTATION_EVIDENCE.json')
        for p in a:p['diagnostic'].update(acquisition='NEW_COMPILE',compiled_wrapper=value['wrappers']['cosine' if p['axis']=='X' else 'sine'])
        prices+=a;sides+=b;proofs.update(c)
    for name,value in zip(RAW[1:],(prices,sides,proofs)):write(out/name,value)
    ok=code==0 and len(prices)==2*len(STAGES[stage]) and not data.get('error')
    write(out/'RESULT.json',dict(execution_SOURCE=source,stage=stage,configuration_id=CONFIG,results=data['results'],context_gate=data.get('context_gate'),
        error=data.get('error'),child_exit_code=code,child_signal=-code if code<0 else None,resources=obs,worker_resources=data.get('resources'),
        wall_seconds=time.monotonic()-started,child_wall_seconds=data.get('wall_seconds'),context_peak_RSS_bytes=data.get('context_peak_RSS_bytes'),
        runtime=job['runtime'],valid_axes_before_C_guard=len(prices),status='ACQUIRED_PENDING_C_GUARD' if ok else 'UNKNOWN_STOP_NO_RETRY',scientific_retry=False,Q3_calls=0,new_signal=0,mandatory_stop=True))
    print('STAGE',stage,'AXES',len(prices),'EXIT',code,'WALL',time.monotonic()-started,flush=True)
    return 0 if ok else 2


def bind(stage,publication):
    plan,_,_=template()
    P.bind_records(plan,STUDY+'/'+stage.lower()+'/'+RUNS[stage],publication,names=RAW,policy_id=POLICY)


def calibrate(prices,design):
    check(len(prices)==4 and {(p['m'],p['axis']) for p in prices}=={(m,a) for m in (5,6) for a in ('X','Y')},'Only m5/m6 calibration axes')
    for p in prices:
        ident,T,q=TARGETS[p['m']];c=p['physical_context']['physical_candidate']
        check(p['configuration_id']==CONFIG and (c['id'],c['T'],c['q'],c['delta'].hex())==(ident,T,q,(.64).hex()),'Foreign calibration width/schedule')
    records=M.T.fit_inputs(prices,route='TRUNCATED_PF',formula='4th',width_over_h='32/5',roles={5:'CALIBRATION',6:'CALIBRATION'})
    V=M.T.c_model();models={a:V.fit([r for r in records if r['axis']==a],features=['one','q'],design_source=design) for a in ('X','Y')}
    predictions={a:M.T.enc(V.predict(model,dict(one=1,q=80))) for a,model in models.items()}
    return records,models,predictions


def freeze(calibration_commit):
    plan,_,_=template();rel=STUDY+'/calibration/'+RUNS['CALIBRATION']
    M.S.committed(ROOT,rel,RAW+('SIDECARS.json','GUARD_CHECK.json'),commit=calibration_commit)
    prices=read(dict(commit=calibration_commit,path=rel+'/PRICES.json'))
    design=dict(commit=prices[0]['source']['commit'],path=STUDY+'/PLAN.md')
    records,models,predictions=calibrate(prices,design)
    write(OUT/'FROZEN_MODEL.json',dict(configuration_id=CONFIG,calibration_rounds=[5,6],calibration_publication=calibration_commit,
        calibration_records=records,model_formula='C_axis(q)=alpha_axis+beta_axis*q',models=models,holdout_round=9,holdout_q=80,
        frozen_m9_predictions=predictions,holdout_refit=False,tolerance=None,m12_q=640,long_q_error_bound=None,frozen_before_m9=True,mandatory_stop_after_campaign=True))
    print('FROZEN',{a:models[a]['coefficients'] for a in models},'m9',predictions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',choices=STAGES)
    parser.add_argument('--frozen-commit');parser.add_argument('--child',type=Path)
    parser.add_argument('--bind',choices=STAGES);parser.add_argument('--publication');parser.add_argument('--freeze-calibration')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.child:return child(args.child)
    elif args.execute:return execute(args.execute,args.frozen_commit)
    elif args.bind:bind(args.bind,args.publication)
    elif args.freeze_calibration:freeze(args.freeze_calibration)
    else:parser.error('Select one fixed metadata/stage/binding/freeze mode')
    return 0


if __name__=='__main__':raise SystemExit(main())
