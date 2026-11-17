"""PF leaf phases using the existing D numerical compiler and C receipt guard.

The native compile body is reused from D d8780090f acquire(). Receipt admission
is a separate phase; neither phase chooses a candidate, shots or a new scope.
This adapter runs only in the task-owned provider workspace.
"""
import argparse
import importlib.util
import json
import resource
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


def load_provider(root):
    root=Path(root).resolve()
    sys.path[:0]=[str(root/'scripts'),str(root/'src')]
    spec=importlib.util.spec_from_file_location('fixed_pf_provider',root/'scripts/prepare_pf_candidate.py')
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    # Function globals resolve to the same fixed provider modules and helpers.
    globals().update(ROOT=root,native_row=provider.native_row,comparison_method=provider.comparison_method,
        write=provider.write,load=provider.load,publish=provider.publish,head=provider.head)
    return provider


def compile_receipts(value,out,phase,ms,source):
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
    pub=publish([out],'Preserve unadmitted PF compile receipts for separate audit')
    return dict(status='COMPILE_RECEIPTS_READY',phase=phase,rounds=ms,raw_commit=pub,relative=str(folder.relative_to(ROOT)),route=route)


def audit_receipts(provider,value,out,compiled):
    import run_pf_m0_prices as M
    import run_pf_p7_h_prices as P
    from trottertracks import pf_evidence_preparation as E
    folder=ROOT/compiled['relative']
    # The original C implementation guard reads committed raw receipts and
    # matching implementation evidence. It never recompiles a circuit.
    P.bind_records(value['prepared'],compiled['relative'],compiled['raw_commit'],names=M.RAW,
        policy_id='PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1',route=compiled['route'])
    bound=publish([out],'Admit transferred PF receipts through original C implementation guard')
    prices=load(folder/'PRICES.json')
    return [dict(m=x['m'],axis=x['axis'],physical_id=x['physical_context']['physical_candidate']['id'],
        unit_price_RZ=E.enc(x['mean']['rz_count']),kind='DIRECT_COMPILED_DETERMINISTIC',
        price_evidence=dict(source=E.ref(bound,str((folder/'PRICES.json').relative_to(ROOT)),i)),
        gate='C_IMPLEMENTATION_GUARD_PASS')for i,x in enumerate(prices)]


def run(provider,request):
    value=request['value'];out=ROOT/request['relative_output'];phase=request['phase']
    source=request['source']
    if phase=='preflight':
        provider.source_gate();provider.preflight(value,out)
        publication=publish([out],'Fix distributed PF input plan and duplicate preflight')
        return dict(status='PREFLIGHT_PASS',source=dict(commit=publication,caller='scripts/prepare_pf_candidate.py',
            input_plan=dict(commit=publication,path=str((out/'PLAN.json').relative_to(ROOT)))))
    if phase=='signal':
        provider.signals(value,out,source)
        publication=publish([out],'Preserve PF signal task for central adoption')
        return dict(status='SIGNALS_READY',publication=publication)
    if phase=='compile':return compile_receipts(value,out,request['cohort'],request['rounds'],source)
    if phase=='audit':return dict(status='AUDIT_PASS',prices=audit_receipts(provider,value,out,request['compiled']))
    if phase=='freeze':return dict(status='MODEL_FROZEN',model=provider.freeze(value,request['prices'],out,source))
    if phase=='finish':return dict(status='PRICES_READY',prices=provider.finish(value,request['prices'],out,request['model'],source))
    if phase=='register':
        # The central owned clone imports remote objects without switching its
        # HEAD. Bind its newly received signal files before gitref() is used.
        publish([out],'Bind received PF evidence before central registry publication')
        admitted=provider.register(value,out,request['prices'],ROOT/request['registry'])
        publish([out],'Publish distributed PF completed evidence')
        return dict(status='READY_FOR_SAME_OPTIMIZER',configuration_id=value['configuration']['id'],
            entry=admitted,output=str(out))
    raise ValueError('Unknown PF preparation phase')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider-root',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    provider=load_provider(a.provider_root);request=load(a.input)
    result=run(provider,request);write(a.output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
