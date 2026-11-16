"""Selected Full-PF/B0I evidence completion; no shot search or Q3 implementation.

plan is metadata-only. complete needs the user's scoped science authorization.
Native circuit work runs in a fresh provider-first subprocess; publication keeps
all old sources and one-shot stops intact. Completed inputs return via registry.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
MODULE='scripts/prepare_pf_candidate.py'
STUDY='docs/research/d_pf_on_demand_evidence_20261116'
SOURCES=(MODULE,'src/trottertracks/pf_evidence_preparation.py','src/trottertracks/pf_candidate_space.py',
 'scripts/run_pf_m0_prices.py','scripts/run_pf_p7_h_prices.py','scripts/run_truncated_p7_prices.py')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load(path):return json.loads(Path(path).read_text())
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def gitref(path,*s):return dict(commit=head(),path=str(Path(path).relative_to(ROOT)),selector=list(s))
def publish(paths,message):
    names=[str(Path(p).relative_to(ROOT)) for p in paths]
    subprocess.run(['git','add','--sparse','--',*names],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT).strip():
        subprocess.run(['git','-c','user.name=HIROMU1015','-c','user.email=u411612g@ecs.osaka-u.ac.jp','commit','-m',message],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return head()
def source_gate():
    sha=head()
    for path in SOURCES:
        if (ROOT/path).read_bytes()!=subprocess.check_output(['git','show',sha+':'+path],cwd=ROOT):raise ValueError('Commit execution source first: '+path)
    return dict(commit=sha,caller=MODULE)
def event(out,stage,**kw):
    with (Path(out)/'STAGES.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,time=time.time(),**kw))+'\n')
    print(stage,kw,flush=True)


def comparison_method(cfg):
    s=cfg['spec']
    if s['method'] in ('FULL_PF','TRUNCATED_PF'):return s['method']
    if s['method']=='PF' and s.get('split',{}).get('original_PF_method') in ('FULL','B0I'):
        return 'TRUNCATED_PF' if s['split']['original_PF_method']=='B0I' else 'FULL_PF'
    raise ValueError('D handles canonical deterministic PF only')


def native_row(cfg,physical):
    import run_pf_m0_prices as M
    route=comparison_method(cfg)
    if physical['method']!=('FULL' if route=='FULL_PF' else 'B0I') or physical['prefix']!=cfg['spec']['L_D']:
        raise ValueError('Native candidate route/prefix mismatch')
    return M.ROUTES[route].native_row(physical)


def selected_configuration(item):
    from trottertracks import pf_candidate_space as K,pf_evidence_preparation as E
    if item.get('configuration_body') is not None:cfg=deepcopy(item['configuration_body'])
    elif item.get('configuration') and isinstance(item['configuration'],dict) and 'commit' in item['configuration']:cfg=E.read(ROOT,item['configuration'])
    elif item.get('configuration_id'):
        matches=[v for v in K.candidates(ROOT) if v['configuration_id']==item['configuration_id']]
        E.check(len(matches)==1,'Unknown/ambiguous selected PF configuration');cfg=matches[0]['configuration']
    else:cfg=K.configure(ROOT,**{k:item[k] for k in ('method','p','formula','delta_over_h')})
    E.check(comparison_method(cfg) in ('FULL_PF','TRUNCATED_PF'),'D handles deterministic PF only')
    return cfg


def plan_one(cfg,out):
    from trottertracks import pf_evidence_preparation as E
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source=source_gate();value=E.make_plan(ROOT,cfg,source)
    write(out/'CONFIGURATION.json',cfg);write(out/'PLAN.json',value)
    return value


def native_child(path):
    # Do not import C's package before the original install() resolves native.
    import run_pf_m0_prices as M
    job=load(path);allowed=load(job['fixed_plan'])
    cfg=allowed['configuration'];prefix=cfg['spec']['L_D'];route=comparison_method(cfg)
    if route!=job['route'] or prefix!=job['prefix']:raise ValueError('Native prefix/route binding differs')
    requests={r['m']:r['physical_candidate'] for r in allowed['prepared']['pending']['requests']}
    if any(c['row']!=native_row(cfg,requests[c['m']]) for c in job['cells']):raise ValueError('Native schedule differs')
    for file in SOURCES:
        if (ROOT/file).read_bytes()!=subprocess.check_output(['git','show',job['driver_source']['commit']+':'+file],cwd=ROOT):raise ValueError('Native source changed')
    return M.child(path,validate_rounds=lambda rounds: M.check(list(rounds)==[c['m'] for c in job['cells']],'Wrong selected native rounds'))


def preflight(value,out):
    import run_pf_m0_prices as M
    from trottertracks import pf_evidence_preparation as E
    existing={(r['m'],r['axis']) for r in value['direct_prices']}
    acquired={r['m'] for r in value['new_compile_cells']}
    targets=[dict(configuration_id=value['configuration']['id'],physical_id=r['physical_candidate']['id'],m=r['m'],axis=a,physical_candidate=r['physical_candidate'])
       for r in value['prepared']['pending']['requests'] for a in ('X','Y') if r['m'] in acquired and (r['m'],a) not in existing]
    audit=M.duplicate_check((1,2),selected=targets)
    E.check(all(r['action']=='NEW_COMPILE_MISSING_CONFIRMED' for r in audit['rows']),'Review existing receipt/job before new work')
    # Fixed import-time metadata dependency; no restore/prepare/compile here.
    dep='references/h46_m5_tol_pilot_inputs_v1.json'
    E.check((ROOT/dep).read_bytes()==subprocess.check_output(['git','show',head()+':'+dep],cwd=ROOT),'Native dependency not materialized')
    write(out/('PREFLIGHT_EXECUTION.json' if (out/'PREFLIGHT.json').exists() else 'PREFLIGHT.json'),audit)


def signals(value,out,source):
    from trottertracks import pf_evidence_preparation as E
    from trottertracks import saved_pf_mean_staging as S
    prepared=value['prepared'];folder=out/'signal_run01';folder.mkdir()
    means=[];ctx=None;new=0;started=time.perf_counter();status='STARTED';error=None
    try:
        for req,choice in zip(prepared['pending']['requests'],value['signal_rounds']):
            event(out,'SIGNAL_ROUND_START',m=req['m'],kind=choice['acquisition'])
            if choice['saved_source']:
                record=E.saved_record(ROOT,prepared,req,choice['saved_source'],source)
            else:
                if ctx is None:ctx=E.SavedOrNativeContext(ROOT,prepared)
                desc=req['domain']['reference'];reference=E.read(ROOT,desc['source']);reference=dict(reference,source=desc['source'],binding=desc['binding'],T=desc['T'])
                record=E.N.evaluate(ctx,req,reference,source=source);new+=1
            write(folder/('round_%d.json'%req['m']),record)
            if record['status']!='NUMERICAL_MEAN_AVAILABLE':raise ValueError('Numerical mean gate: '+str(record.get('reason')))
            means.append(E.bind_mean(prepared,req,record));event(out,'SIGNAL_ROUND_END',m=req['m'])
        packet=dict(configuration_id=value['configuration']['id'],source=source,target=prepared['target'],means=means)
        S.require_complete(packet);write(folder/'MEAN_HANDOFF.json',packet);status='COMPLETE_THIRTEEN_ROUNDS'
    except Exception as exc:error=str(exc);raise
    finally:write(folder/'STATUS.json',dict(status=status,error=error,new_means=new,reused_means=len(means)-new,new_step_count=0 if ctx is None else len(ctx.new_step_keys),wall_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,reference_calls=0,mandatory_stop=True))


def acquire(value,out,phase,ms,source):
    from trottertracks import pf_evidence_preparation as E
    import run_pf_m0_prices as M
    import run_pf_p7_h_prices as P
    folder=out/phase/'run01';private=ROOT/'.pf_evidence_private'/value['configuration']['id']/phase/'run01'
    folder.mkdir(parents=True);private.mkdir(parents=True,exist_ok=False)
    cfg=value['configuration'];p=cfg['spec']['L_D'];route=comparison_method(cfg)
    cells=[dict(m=r['m'],formula=r['physical_candidate']['formula'],row=native_row(cfg,r['physical_candidate']),role=phase.upper()) for r in value['prepared']['pending']['requests'] if r['m'] in ms]
    saved=M.read(dict(commit=M.BASELINE,path=(M.F if route=='FULL_PF' else M.T).OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    constant=saved['facts']['scalar'] if route=='FULL_PF' else saved['facts']['scalar']-saved['facts']['tail_identity']
    # Use the original saved H constant, not subtraction, for binary64 order.
    full=M.read(dict(commit=M.BASELINE,path=M.F.OUTPUT_ROOT+'/holdout/run01/RESULT.json'))['results'][0]
    constant=full['facts']['scalar'];tail=0. if p==10 else E.read(ROOT,E.scalar_source(p))
    from pf_common_run_contract import resources,workers
    obs=resources();n=workers(obs,len(cells),2*2**30,100*2**20);E.check(n>0,'No available worker slot')
    job=dict(route=route,prefix=p,rounds=ms,cells=cells,cohort='D_PF_SELECTED_'+cfg['id'][:12]+'_'+phase,public=str(folder/'child_public'),private=str(private),workers=n,expected_scalar=constant+tail,tail_scalar=tail,
      preparation_hash=saved['trial']['preparation_hash'] if p in (7,10) else None,execution_SOURCE=source['commit'],driver_source=source,fixed_plan=str(out/'PLAN.json'))
    # Non-p7 preparations get checked against exact prefix order/scalar; the
    # native preparation hash is recorded for their own request, never borrowed.
    if p not in (7,10):job['prefix_specific_preparation']=True
    write(folder/'JOB.json',job);started=time.perf_counter()
    with (private/'child.log').open('x') as log:r=subprocess.run([sys.executable,'-B',str(ROOT/MODULE),'--native-child',str(folder/'JOB.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result=load(private/'child.json') if (private/'child.json').exists() else dict(results=[],error='Native child result absent')
    trace=folder/'TRACE';trace.mkdir()
    for name in ['child.log','context_progress.jsonl']+[x.name for x in private.glob('progress_*.jsonl')]:
        if (private/name).exists():shutil.copyfile(private/name,trace/name)
    prices=[];sides=[];proofs={};module=M.F if route=='FULL_PF' else M.T
    for row in result['results']:
        if not row['ok']:continue
        kwargs={} if route=='FULL_PF' else dict(scalar_source=E.scalar_source(p))
        a,b,c=module.price_bindings(ROOT,value['prepared'],row['cell'],row['trial'],price_source=source,evidence_commit=M.S.PLACEHOLDER,evidence_path=str((folder/'IMPLEMENTATION_EVIDENCE.json').relative_to(ROOT)),**kwargs)
        for price,side in zip(a,b):
            price['cohort']=side['cohort']=job['cohort'];price['diagnostic']['compiled_wrapper']=row['wrappers']['cosine' if price['axis']=='X' else 'sine']
            key=side['implementation_evidence']['selector'][0];c[key]={k:deepcopy(side[k]) for k in ('price_source','identity','law_binding','axis','cohort','physical_context','profile_extension','implementation')}
        prices+=a;sides+=b;proofs.update(c)
    for name,data in [('PRICES.json',prices),('SIDECARS_UNBOUND.json',sides),('IMPLEMENTATION_EVIDENCE.json',proofs)]:write(folder/name,data)
    result.update(source=source,child_exit_code=r.returncode,child_signal=-r.returncode if r.returncode<0 else None,wall_seconds=time.perf_counter()-started)
    write(folder/'RESULT.json',result)
    E.check(r.returncode==0 and len(prices)==2*len(ms) and all(x['ok'] for x in result['results']),'Native STOP; preserve partial results, no retry')
    pub=publish([out],'Preserve selected PF '+phase+' measured wrapper receipts')
    P.bind_records(value['prepared'],str(folder.relative_to(ROOT)),pub,names=M.RAW,policy_id='PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1',route=route)
    bound=publish([out],'Admit selected PF '+phase+' receipts through C implementation guard')
    return [dict(m=x['m'],axis=x['axis'],physical_id=x['physical_context']['physical_candidate']['id'],unit_price_RZ=E.enc(x['mean']['rz_count']),kind='DIRECT_COMPILED_DETERMINISTIC',price_evidence=dict(source=E.ref(bound,str((folder/'PRICES.json').relative_to(ROOT)),i)),gate='C_IMPLEMENTATION_GUARD_PASS') for i,x in enumerate(prices)]


def freeze(value,prices,out,source):
    from trottertracks import pf_evidence_preparation as E
    from trottertracks import price_model_validation as V
    records=[];model={};prediction={};qs={r['m']:r['physical_candidate']['q'] for r in value['prepared']['pending']['requests']}
    for axis in ('X','Y'):
        records=[]
        for m in value['calibration_rounds']:
            row=next(x for x in prices if (x['m'],x['axis'])==(m,axis))
            records.append(V.record(cell_id=value['configuration']['id']+':m%d'%m,role='CALIBRATION',route=comparison_method(value['configuration']),axis=axis,features=dict(one=1,q=qs[m]),rz_count=row['unit_price_RZ'],kind='DIRECT_COMPILED_DETERMINISTIC',source=row['price_evidence'],implementation=dict(lowering_id='H46_LONG_TIME_NATIVE_COMPOSE',physical_recipe='C_ADMITTED_DETERMINISTIC_PREFIX_AND_SIGNED_PF'),profile=value['configuration']['spec']['profile']))
        model[axis]=V.fit(records,features=['one','q'],design_source=source)
        prediction[axis]=E.enc(V.predict(model[axis],dict(one=1,q=qs[value['holdout_round']])))
    path=out/'FROZEN_MODEL.json';write(path,dict(configuration_id=value['configuration']['id'],models=model,calibration_rounds=value['calibration_rounds'],holdout_round=value['holdout_round'],predictions=prediction,holdout_refit=False,long_q_error_bound=None))
    pub=publish([out],'Freeze selected PF width-specific model before independent holdout')
    return E.ref(pub,str(path.relative_to(ROOT)))


def finish(value,prices,out,frozen,source):
    from trottertracks import pf_evidence_preparation as E
    from trottertracks import price_model_validation as V
    model=E.read(ROOT,frozen);check=E.check;qs={r['m']:r['physical_candidate']['q'] for r in value['prepared']['pending']['requests']};m=value['holdout_round'];hold=[]
    for axis in ('X','Y'):
        row=next(x for x in prices if (x['m'],x['axis'])==(m,axis));obs=E.frac(row['unit_price_RZ']);pred=E.frac(model['predictions'][axis]);check(obs>0,'Undefined relative error')
        hold.append(dict(m=m,axis=axis,q=qs[m],predicted=E.enc(pred),observed=E.enc(obs),observed_minus_prediction=E.enc(obs-pred),signed_relative=E.enc((obs-pred)/obs),absolute_relative=E.enc(abs(obs-pred)/obs),status='EXACT_MATCH' if obs==pred else 'DEVIATION_NO_REFIT',receipt=row['price_evidence']))
    rows=[];known={(r['m'],r['axis']):r for r in prices}
    for req in value['prepared']['pending']['requests']:
        for axis in ('X','Y'):
            old=known.get((req['m'],axis))
            if old:r=deepcopy(old)
            else:
                check(E.frac(req['T'])/req['q_PF']==E.frac(value['actual_width']),'Conditional model used outside its actual width')
                r=dict(m=req['m'],axis=axis,physical_id=req['physical_candidate']['id'],unit_price_RZ=E.enc(V.predict(model['models'][axis],dict(one=1,q=req['q_PF']))),kind='CONDITIONAL_AFFINE_PRICE_REFERENCE',price_evidence=dict(source=frozen,axis=axis),gate='MODEL_REFERENCE_LONG_Q_UNCERTAINTY_UNKNOWN')
            r.update(q=req['q_PF'],T=deepcopy(req['T']),actual_delta_pf=value['configuration']['rounds'][req['m']]['actual_delta_pf'],unvalidated_extrapolation=req['q_PF']>qs[m] and not old,model_error_bound_RZ=None if old is None else 0)
            rows.append(r)
    write(out/'AXIS_PRICES.json',dict(configuration_id=value['configuration']['id'],axes=rows,holdout=hold,model=frozen,price_error_bound=None,CI=None,qualified_for_candidate=False,scope=value['price_scope']))
    pub=publish([out],'Publish selected PF 26-axis conditional prices and independent holdout comparison')
    return E.ref(pub,str((out/'AXIS_PRICES.json').relative_to(ROOT)))


def total_binding(cfg,cfgref):
    # Same B canonical binder in a fresh metadata-only process; never import
    # two generations of trottertracks into the native/scientific parent.
    problem=load(ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json')
    route=problem['execution_routes']['FULL_PF'][0];view=Path(route['view'])
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=view,text=True).strip()
    if actual!=route['view_commit']:raise ValueError('Pinned B binding view changed')
    code="import sys,json;sys.path.insert(0,'src');from trottertracks import rpe_total_binding as T;d=json.load(sys.stdin);b=T.create(d['root'],d['configuration'],original_source=d['source']);print(json.dumps({'id':b['id'],'total_configuration':{'id':b['total_configuration']['id']}}))"
    raw=subprocess.check_output([sys.executable,'-B','-c',code],cwd=view,input=json.dumps(dict(root=str(ROOT),configuration=cfg,source=cfgref)).encode())
    return json.loads(raw)


def register(value,out,prices,registry):
    from trottertracks import pf_evidence_preparation as E

    # Reuse the exact PF worker contract. The template allocation is unused:
    # B2 always replaces it with its own generated proposal before any Q3.
    template=E.read(ROOT,E.ref('b4296b110b129d226dbef2743c14f725a27c3bf6','docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json'))
    cfg=value['configuration'];cfgref=gitref(out/'CONFIGURATION.json');sigref=gitref(out/'signal_run01/MEAN_HANDOFF.json')
    binding=total_binding(cfg,cfgref);base=template['cases'][0]
    case=dict(label='PF_SEARCH_TEMPLATE_'+cfg['id'][:12],configuration_id=cfg['id'],configuration=cfgref,signal_packet=sigref,total_view_id=binding['total_configuration']['id'],binding_id=binding['id'],evaluation_target=base['evaluation_target'],allocation=dict(allocation_id='UNUSED_TEMPLATE_REPLACED_BY_OPTIMIZER',shot_policy='USER_SUPPLIED',shots=[dict(m=m,N_X=1,N_Y=1) for m in range(13)]))
    inputs={k:deepcopy(template[k]) for k in ('kernel_source','C_source','statistical_contract','options','adopted_evaluation','acceleration')}
    inputs.update(cases=[case],concurrent_configurations=1,maximum_scientific_calls=1)
    write(out/'Q3_INPUTS.json',inputs);publish([out],'Publish selected PF configuration signal and reusable Q3 route template')
    width=str(E.frac(cfg['spec']['delta_pf'])/SAVED_H())
    entry=dict(method=comparison_method(cfg),p=cfg['spec']['L_D'],formula=cfg['spec']['formula'],delta_over_h=width,configuration=cfgref,signal_packet=sigref,prices=prices,q3_inputs=gitref(out/'Q3_INPUTS.json'),total_configuration_id=binding['total_configuration']['id'],binding_id=binding['id'],price_scope=value['price_scope'])
    registry=Path(registry);data=load(registry) if registry.exists() else dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={})
    if cfg['id'] in data['configurations'] and data['configurations'][cfg['id']]!=entry:raise ValueError('Registry collision; preserve original entry')
    data['configurations'][cfg['id']]=entry
    registry.parent.mkdir(parents=True,exist_ok=True);registry.write_text(json.dumps(data,indent=2)+'\n')
    publish([registry],'Register completed selected PF inputs for normal optimizer reuse')
    return entry


def SAVED_H():
    from trottertracks.saved_pf_mean_staging import H
    return H


def complete(value,out,registry):
    from trottertracks import pf_evidence_preparation as E
    source=source_gate();preflight(value,out);pub=publish([out],'Fix selected PF input plan and duplicate preflight before science')
    source=dict(commit=pub,caller=MODULE,input_plan=E.ref(pub,str((out/'PLAN.json').relative_to(ROOT))))
    signals(value,out,source);publish([out],'Preserve selected PF saved and newly completed signal rounds')
    prices=deepcopy(value['direct_prices']);cal=[r['m'] for r in value['new_compile_cells'] if r['role']!='HOLDOUT']
    if cal:prices+=acquire(value,out,'calibration',cal,dict(commit=head(),caller=MODULE))
    frozen=freeze(value,prices,out,dict(commit=head(),path=str((out/'PLAN.json').relative_to(ROOT))))
    ms=[r['m'] for r in value['new_compile_cells'] if r['role']=='HOLDOUT']
    if ms:prices+=acquire(value,out,'holdout',ms,dict(commit=head(),caller=MODULE,frozen_model=frozen))
    price_source=finish(value,prices,out,frozen,source)
    entry=register(value,out,price_source,registry)
    write(out/'COMPLETION.json',dict(status='READY_FOR_SAME_OPTIMIZER',configuration_id=value['configuration']['id'],new_signal_means=value['new_signal_means'],new_compiles=value['maximum_new_compile'],new_Q3=0,registry_entry=entry,mandatory_stop=True))
    publish([out],'Publish selected PF evidence completion status')
    return dict(status='READY_FOR_SAME_OPTIMIZER',configuration_id=value['configuration']['id'],entry=entry,output=str(out))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path);p.add_argument('--output',type=Path);p.add_argument('--mode',choices=('plan','complete'));p.add_argument('--native-child',type=Path)
    args=p.parse_args()
    if args.native_child:return native_child(args.native_child)
    request=load(args.request);response=dict(schema='D_PF_EVIDENCE_COMPLETION_V1',items=[])
    output_root=Path(request['output_root']).resolve();registry=Path(request['registry_file']).resolve()
    if not output_root.is_relative_to(ROOT) or not registry.is_relative_to(ROOT):raise ValueError('Use this producer worktree for source-bound publication')
    for item in request['selected']:
        cfg=selected_configuration(item)
        saved=load(registry).get('configurations',{}).get(cfg['id']) if registry.exists() else None
        if saved and saved.get('signal_packet') and saved.get('prices') and saved.get('q3_inputs'):
            response['items'].append(dict(status='REUSED_COMPLETED_REGISTRY',configuration_id=cfg['id'],entry=saved));continue
        out=output_root/cfg['id']
        if args.mode=='complete' and (out/'PLAN.json').exists():
            if (out/'signal_run01').exists() or (out/'STOP.json').exists():raise ValueError('Preserve consumed scientific attempt; no automatic retry')
            value=load(out/'PLAN.json')
            if value['configuration']!=cfg:raise ValueError('Fixed plan belongs to another configuration')
            for name in ('CONFIGURATION.json','PLAN.json'):
                raw=subprocess.check_output(['git','show',head()+':'+str((out/name).relative_to(ROOT))],cwd=ROOT)
                if (out/name).read_bytes()!=raw:raise ValueError('Commit metadata plan before execution')
        else:value=plan_one(cfg,out)
        if args.mode=='complete':
            try:response['items'].append(complete(value,out,registry))
            except Exception as exc:
                failure=dict(status='UNKNOWN_PRESERVED_NO_AUTOMATIC_RETRY',configuration_id=cfg['id'],reason=type(exc).__name__+': '+str(exc),output=str(out))
                write(out/'STOP.json',failure);publish([out],'Preserve selected PF partial evidence and failure without retry')
                response['items'].append(failure);response['registry_file']=str(registry);write(args.output,response)
                raise
        else:response['items'].append(dict(status='PLAN_ONLY',configuration_id=cfg['id'],new_signal_means=value['new_signal_means'],maximum_new_compile=value['maximum_new_compile'],output=str(out)))
    response['registry_file']=str(registry);write(args.output,response);print(json.dumps({r['configuration_id']:r['status'] for r in response['items']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
