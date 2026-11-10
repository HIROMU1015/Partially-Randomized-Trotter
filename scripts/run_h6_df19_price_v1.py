"""Approved H6 saved-data price V1; no circuits, fit, sampling, N or G.

The model is frozen before develop and remains unchanged for held-out cohorts.
Local 12x12 coefficient diagonalization extracts source-defined price descriptors;
no state/Hamiltonian is generated and no ground-state eigenproblem is solved.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.h6_df19_price_v1 import MODEL,predict
CARRIER='155529e20a8fac39e00786c40b57830b71203c4f'
PROVIDER='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
BASE='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
PUBLIC=ROOT/'docs/research/h6_df19_price_v1_20261110'
CONFIG=Path('/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/private/authorization_run01/H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json')
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
FILES=('src/trottertracks/h6_df19_price_v1.py','scripts/run_h6_df19_price_v1.py','src/trottertracks/track_d_native_resources.py')
WEIGHTS=(.12547397927357543,.13781643533753404,.1211920365713068,.14181961643121604,.11225428783206336,-.27711271089139133,.11225428783206336,.14181961643121604,.1211920365713068,.13781643533753404,.12547397927357543)
ALLOC={32:[3,3,3,3,2,5,2,3,2,3,3],16:[1,2,1,2,1,2,1,2,1,2,1]}
PRS=(('A','2606efdab20e3b8522e5',5,32,.8,'8f1a392f8ab2bcb6cf59cad513b96f15ff518cb6dc237233410b03bb14567938'),('B','2d2af6ca830dff14a695',7,16,1.6,'00df05ca58c97cc1be8ced7f3df2a6a103b9e44a47d2197f30b4e1119df3a28a'))


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def blob(ref,path):return git('show',ref+':'+path)
def digest(b):return hashlib.sha256(b).hexdigest()
def saved(name):
    b=blob(CARRIER,BASE+name)
    return json.loads(gzip.decompress(b) if name.endswith('.gz') else b)
def write(name,data):
    with (PUBLIC/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def reuse_functions(ref,path,names,namespace):
    nodes=[n for n in ast.parse(blob(ref,path)).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing pinned local function')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),ref+':'+path,'exec'),namespace)


def source_binding():
    ref=git('rev-parse','HEAD').decode().strip();hashes={}
    for name in FILES:
        b=(ROOT/name).read_bytes()
        if b!=blob(ref,name):raise ValueError('Commit scientific source before execution')
        hashes[name]=digest(b)
    return ref,hashes


def descriptors():
    import numpy as np
    import qiskit
    from qiskit.circuit import ParameterExpression
    if (np.__version__,qiskit.__version__)!=('1.26.4','1.3.0'):raise ValueError('Fixed numerical/gate-description runtime required')
    prov=saved('scientific_provenance.json');cfg=json.loads(CONFIG.read_bytes())['systems']['H6']
    if cfg['binding']!=prov['inputs']['H6']['binding'] or prov['provider_SOURCE']!=PROVIDER or prov['compiler']['hash']!=COMPILER:raise ValueError('H6/provider/compiler identity')
    hashes={}
    for name,meta in cfg['inputs'].items():
        b=Path(meta['path']).read_bytes()
        if digest(b)!=meta['sha256'] or len(b)!=meta['bytes']:raise ValueError('Saved H6 input changed')
        hashes[name]=digest(b)
    receipt=json.loads(Path(cfg['inputs']['H6_snapshot_receipt.json']['path']).read_bytes())
    if receipt['metadata']['hamiltonian_hash']!=cfg['binding']['hamiltonian']:raise ValueError('Completion H identity')
    with np.load(cfg['inputs']['H6_snapshot.npz']['path'],allow_pickle=False) as z:
        constant=float(z['constant']);matrices=[np.asarray(z['one_body'],complex),*np.asarray(z['g_matrices'],complex)];lambdas=np.asarray(z['lambdas'],float)
    if len(matrices)!=20 or any(m.shape!=(12,12) for m in matrices):raise ValueError('H6 DF19 coefficient scope')
    scope=dict(np=np,math=math,ParameterExpression=ParameterExpression,DiagonalEvolutionPrimitives=SimpleNamespace)
    reuse_functions(PROVIDER,'src/trotterlib/df_trotter/decompose.py',['diag_hermitian'],scope)
    reuse_functions(CARRIER,'scripts/h46_new2_native.py',['gaussian_operations'],scope)
    reuse_functions(PROVIDER,'src/trotterlib/df_trotter/ops.py',['_as_real','_is_symbolic','_is_effectively_zero','_real_or_symbolic','one_body_diagonal_primitives','df_squared_diagonal_primitives'],scope)
    specs=[]
    for i,matrix in enumerate(matrices):
        matrix=.5*(matrix+matrix.conj().T)
        U,values=scope['diag_hermitian'](matrix,sort='descending_abs',assume_hermitian=True)
        operations=scope['gaussian_operations'](U)
        basis=[dict(name=g.name,qubits=list(q),parameters=[str(t) for t in g.params],matrix_sha256=digest(np.ascontiguousarray(np.asarray(g.to_matrix(),dtype=np.complex128)).view(np.uint8).tobytes())) for g,q in operations]
        specs.append((basis,values,None if i==0 else lambdas[i-1]))
    pieces={}
    for T,q in ((.8,1),(1.6,1),(.8,2)):
        series=[]
        for w in WEIGHTS:
            half=(T*w/q)/2
            blocks=[dict(basis=b,diagonal=scope['one_body_diagonal_primitives'](v,half) if lam is None else scope['df_squared_diagonal_primitives'](v,lam,half)) for b,v,lam in specs]
            series.append(blocks)
        pieces[(T,q)]=series
    return pieces,constant,cfg['binding'],hashes


def validate_events(batches,allocation,known):
    if len(batches)!=11 or any(len(b)!=allocation[i] for i,b in enumerate(batches)):raise ValueError('Saved signed-piece allocation')
    for batch in batches:
        for event in batch:
            for i,app in enumerate(event['application_sequence']):
                if app['application_index']!=i:raise ValueError('Saved application order')
                if app['is_identity']:continue
                key=(app['basis_id'],app['basis_hash']);ops=app['basis_change_operations']
                if key in known:
                    if known[key]!=ops:raise ValueError('Changed ordered basis descriptor')
                else:
                    h=digest(json.dumps(dict(num_system_qubits=12,operations=ops,hash_policy='ordered_local_basis_operations_v1'),sort_keys=True,separators=(',',':')).encode())
                    if h!=app['basis_hash'] or app['basis_id']!='df-basis-'+h:raise ValueError('Saved H6 basis hash')
                    known[key]=ops


def _predict_job(job):
    blocks,q,scalar,T,batches,allocation=job
    return {axis:predict(blocks,q,scalar,T,axis,batches,allocation) for axis in ('cosine','sine')}


def require(c,method,prefix,T,q,R=None,K=None):
    expected=dict(system='H6',method=method,formula='4th(m5_best)',prefix=prefix,T=T,q=q,delta=T/q,R_outer=R,K=K)
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed H6 candidate mismatch')


def prepare_jobs(stage,pieces,constant):
    jobs=[];metadata=[];bindings=[];known={}
    pf_id,T,q=('76601611270ed9f0677d',.8,1) if stage=='develop' else ('a4f5ffc0f320644dadfa',1.6,1)
    pf=saved('reused_compiled_prices.json.gz' if stage=='develop' else 'exploration_costs.json.gz')[pf_id];require(pf['candidate'],'FULL',19,T,q)
    pf_actual={a:pf[a] if stage=='develop' else pf['mean_metrics'][a]['rz_count'] for a in ('cosine','sine')}
    jobs.append((pieces[(T,q)],q,constant,T,None,None));metadata.append(dict(family='PF_FULL19',candidate_id=pf_id,cohort='DETERMINISTIC',path_index=0,T=T,q_PF=q,actual=pf_actual))
    cohort,n=('EXPLORATION',9) if stage=='develop' else ('CONFIRMATION',33)
    table=saved('exploration_costs.json.gz' if stage=='develop' else 'confirmation_costs.json.gz')
    aggregate_checks=[]
    for family,cid,prefix,R,T,prephash in PRS:
        group=table[cid];require(group['candidate'],'PR',prefix,T,1,R,2)
        files=group['trial_files'];indices=[int(Path(f).parent.name) for f in files]
        if group['cohort']!=cohort or group['n']!=n or len(files)!=n or set(indices)!=set(range(n)):raise ValueError('Original whole-path cohort scope')
        aggregate_checks.append(dict(family='PR_'+family,expected=group['mean_metrics']))
        for f in sorted(files,key=lambda f:int(Path(f).parent.name)):
            path=Path(f);index=int(path.parent.name);raw=path.read_bytes();eraw=path.with_name('events_PRIVATE.json').read_bytes();result,event=json.loads(raw),json.loads(eraw)
            require(result['candidate'],'PR',prefix,T,1,R,2)
            if (result['status']!='PASS' or result['cohort']!=cohort or result['index']!=index or result['preparation_hash']!=prephash or event['cohort']!=cohort or result['conditional'] or event['conditional']):raise ValueError('Saved path identity')
            validate_events(event['events'],ALLOC[R],known)
            blocks=[series[:prefix+1] for series in pieces[(T,1)]]
            jobs.append((blocks,1,result['scalar'],T,event['events'],ALLOC[R]))
            metadata.append(dict(family='PR_'+family,candidate_id=cid,cohort=cohort,path_index=index,T=T,q_PF=1,actual={a:result['prices'][a]['rz_count'] for a in ('cosine','sine')}))
            bindings.append(dict(family='PR_'+family,cohort=cohort,index=index,result_sha256=digest(raw),events_sha256=digest(eraw),preparation_hash=prephash))
    return jobs,metadata,bindings,aggregate_checks


def summarize(rows):
    groups=[]
    for family in dict.fromkeys(r['family'] for r in rows):
        for axis in ('cosine','sine'):
            rs=[r for r in rows if r['family']==family and r['axis']==axis];n=len(rs)
            groups.append(dict(family=family,axis=axis,n_paths=n,mean_compiled=sum(r['compiled_rz'] for r in rs)/n,mean_predicted=sum(r['prediction']['structural_rz'] for r in rs)/n,mean_residual=sum(r['residual_rz'] for r in rs)/n,mean_absolute_error=sum(r['absolute_error'] for r in rs)/n,mean_absolute_relative_error=sum(r['absolute_relative_error'] for r in rs)/n,max_absolute_error=max(r['absolute_error'] for r in rs),max_absolute_relative_error=max(r['absolute_relative_error'] for r in rs),min_residual=min(r['residual_rz'] for r in rs),max_residual=max(r['residual_rz'] for r in rs),exact_count=sum(r['residual_rz']==0 for r in rs)))
    return groups


def run():
    started=time.monotonic();source,hashes=source_binding()
    if any((PUBLIC/name).exists() for name in ('DEVELOPMENT.json','HOLDOUT.json','AUXILIARY.json','PATH_RESIDUALS.csv')):raise ValueError('Preserve prior results; no rerun/overwrite')
    import qiskit
    calls=[]
    def forbidden(*a,**kw):calls.append(1);raise RuntimeError('Real circuit construction/transpile prohibited')
    all_rows=[]
    with patch.object(qiskit.QuantumCircuit,'__init__',forbidden),patch.object(qiskit,'transpile',forbidden):
        pieces,constant,binding,input_hashes=descriptors()
        workers=min(67,len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork')) as pool:
            for stage in ('develop','holdout'):
                if source_binding()!=(source,hashes):raise ValueError('Model changed before holdout')
                jobs,metadata,private,aggregate=prepare_jobs(stage,pieces,constant)
                predictions=list(pool.map(_predict_job,jobs,chunksize=1));rows=[]
                for meta,prediction in zip(metadata,predictions):
                    for axis in ('cosine','sine'):
                        observed=int(meta['actual'][axis]);pred=prediction[axis];delta=observed-pred['structural_rz']
                        rows.append(dict(**{k:v for k,v in meta.items() if k!='actual'},axis=axis,compiled_rz=observed,prediction=pred,residual_rz=delta,absolute_error=abs(delta),absolute_relative_error=abs(delta)/observed))
                summaries=summarize(rows)
                for check in aggregate:
                    for a in ('cosine','sine'):
                        summary=next(s for s in summaries if s['family']==check['family'] and s['axis']==a)
                        if summary['mean_compiled']!=check['expected'][a]['rz_count']:raise ValueError('Original aggregate/path consistency')
                result=dict(stage=stage,model=MODEL,model_source=source,model_sha256=hashes,provider_source=PROVIDER,price_carrier=CARRIER,compiler_hash=COMPILER,scientific_binding=binding,input_sha256=input_hashes,private_trial_bindings=private,rows=rows,summaries=summaries,exact_saved_reproduction=all(r['residual_rz']==0 for r in rows),approximate_accuracy_evaluated=True,formal_error_threshold=None,long_q_qualified=False,all_law_expectation=None,total_CI=None,mandatory_stop=True)
                write('DEVELOPMENT.json' if stage=='develop' else 'HOLDOUT.json',result);all_rows.extend(rows)
                print(json.dumps(dict(stage=stage,model_source=source,summaries=summaries),ensure_ascii=False),flush=True)
            # Optional saved PF q2 stress point: source receipt must match first.
            aux=saved('reused_compiled_prices.json.gz')['4ecf37965885c656e554'];require(aux['candidate'],'FULL',19,.8,2)
            qualified=len(aux['proof'])==2 and {p['axis'] for p in aux['proof']}=={'cosine','sine'} and all(p['receipt']['preparation_hash']=='1de5325d85ac3c95b7957bc6414097fb52fdac2a6d8294ff1623bcd105b73648' and p.get('common_native_identity') and p['receipt']['system']=='H6' and p['receipt']['prefix']==19 and p['receipt']['q']==2 and p['receipt']['formula']=='4th(m5_best)' and p['receipt']['compiler_settings_hash']==COMPILER for p in aux['proof'])
            if qualified:
                predictions=_predict_job((pieces[(.8,2)],2,constant,.8,None,None));rows=[]
                for a in ('cosine','sine'):
                    actual=int(aux[a]);pred=predictions[a];delta=actual-pred['structural_rz']
                    rows.append(dict(family='PF_AUX_Q2',candidate_id=aux['candidate']['id'],cohort='SAVED_DETERMINISTIC',path_index=0,T=.8,q_PF=2,axis=a,compiled_rz=actual,prediction=pred,residual_rz=delta,absolute_error=abs(delta),absolute_relative_error=abs(delta)/actual))
                write('AUXILIARY.json',dict(model_source=source,rows=rows,summaries=summarize(rows),fixed_delta_q_validation=False,mandatory_stop=True));all_rows.extend(rows)
            else:write('AUXILIARY.json',dict(status='UNQUALIFIED_SAVED_RECEIPT',rows=[],mandatory_stop=True))
    fields=('family','cohort','candidate_id','path_index','axis','T','q_PF','compiled_rz','predicted_rz','residual_rz','absolute_error','absolute_relative_error')
    with (PUBLIC/'PATH_RESIDUALS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in all_rows:writer.writerow({k:row['prediction']['structural_rz'] if k=='predicted_rz' else row[k] for k in fields})
    write('COMPLETION.json',dict(model_source=source,source_hashes=hashes,rows=len(all_rows),scope=dict(new_circuit_builds=len(calls),transpiles=0,sampling=0,fit_bootstrap=0,signal_N_G=0,new_q_prices=0),workers=workers,wall_seconds=time.monotonic()-started,all_law_expectation=None,long_q_qualified=False,full_RPE_provider_qualified=False,mandatory_stop=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();PUBLIC.mkdir(parents=True,exist_ok=True)
    try:run()
    except BaseException as exc:
        if not (PUBLIC/'FAILURE.json').exists():write('FAILURE.json',dict(reason=str(exc),source=git('rev-parse','HEAD').decode().strip(),completed_files=[p.name for p in PUBLIC.iterdir()],mandatory_stop=True))
        raise
