"""Fixed delta=1.6 H6 price-only pilot; isolated raw, read-only old cache.

Reuse pinned scientific functions, not historical launchers or contracts.
Phases: prepare -> q2 -> freeze -> q4 -> summarize (no implicit reruns).
"""
import ast, csv, gzip, hashlib, json, math, os, pickle, resource, subprocess, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
ROOT=Path(__file__).resolve().parents[1]
PROVIDER_ROOT=Path('/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/provider')
sys.path.insert(0,str(PROVIDER_ROOT/'src'))
import numpy as np
import qiskit
import trottertracks
trottertracks.__path__.append(str(ROOT/'src/trottertracks'))
from trottertracks.h6_df19_price_v1 import Accounting
from trottertracks.h6_df19_price_v2 import reuse
CARRIER='155529e20a8fac39e00786c40b57830b71203c4f'
PROVIDER='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
PUBLIC=ROOT/'docs/research/h6_df19_price_v2_20261110'
PRIVATE=Path('/home/AbeHiromu/projects/h6-df19-price-v2/20261110')
OLD_CACHE=Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/compiled_cache')
CONFIG=Path('/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/private/authorization_run01/H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json')
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
METRICS=('rz_count','rz_depth','cx_count','cx_depth','total_depth','circuit_size')
AXES=('cosine','sine')
TAIL=(.11225428783206336,.14181961643121604,.1211920365713068,.13781643533753404,.12547397927357543)
CTX=None


def blob(ref,path):return subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def require(v,message):
    if not v:raise ValueError(message)

def functions(ref,path,names,scope,strip_imports=()):
    nodes=[n for n in ast.parse(blob(ref,path)).body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==set(names),'Pinned functions missing')
    class Adapt(ast.NodeTransformer):
        def visit_ImportFrom(self,node):return None if node.module in strip_imports else node
    nodes=[Adapt().visit(n) for n in nodes]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),ref+':'+path,'exec'),scope)


def helpers():
    core=dict(math=math,TAIL=TAIL)
    functions(CARRIER,'scripts/h46_new2_core.py',['allocation'],core)
    functions(CARRIER,'src/trotterlib/h46_m5_plan.py',['weights','schedule'],core)
    native=dict(np=np,math=math,hashlib=hashlib,json=json,require=require,core=SimpleNamespace(**{k:core[k] for k in ('weights','allocation','schedule')}))
    functions(CARRIER,'scripts/h46_new2_native.py',['gaussian_operations','prepare'],native)
    functions(CARRIER,'scripts/h46_pr_pilot_repair.py',['canonical_tau'],native)
    functions(CARRIER,'scripts/h46_long_time_native.py',['trial_seed','sample_requests','compose'],native,('h46_m5_contract','h46_pr_pilot_repair'))
    return native


def source():
    ref=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    p='scripts/run_h6_df19_price_v2_pilot.py'
    require((ROOT/p).read_bytes()==blob(ref,p),'Commit pilot source before execution')
    return ref


def verify_provider():
    paths=set()
    for module in tuple(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if not p:continue
        try:relative=Path(p).relative_to(PROVIDER_ROOT)
        except ValueError:continue
        if str(relative).endswith('.py'):paths.add(str(relative))
    for path in sorted(paths):require((PROVIDER_ROOT/path).read_bytes()==blob(PROVIDER,path),'Provider SOURCE changed: '+path)
    return {path:sha((PROVIDER_ROOT/path).read_bytes()) for path in sorted(paths)}


def context():
    global CTX
    if CTX is not None:return CTX
    require((np.__version__,qiskit.__version__)==('1.26.4','1.3.0'),'Fixed runtime')
    cfg=read(CONFIG)['systems']['H6'];h=helpers()
    for meta in cfg['inputs'].values():
        b=Path(meta['path']).read_bytes();require(sha(b)==meta['sha256'] and len(b)==meta['bytes'],'Saved input bytes')
    from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
    ham,sector,full,state,md,*_=load_h6_snapshot(Path(cfg['inputs']['H6_snapshot.npz']['path']),read(cfg['inputs']['H6_snapshot_receipt.json']['path']),read(cfg['inputs']['H6_df_receipt.json']['path']))
    preps={p:h['prepare'](ham,p) for p in (7,19)}
    expected={7:'00df05ca58c97cc1be8ced7f3df2a6a103b9e44a47d2197f30b4e1119df3a28a',19:'1de5325d85ac3c95b7957bc6414097fb52fdac2a6d8294ff1623bcd105b73648'}
    for p,prep in preps.items():
        require(prep.preparation_hash==expected[p] and prep.hamiltonian_hash==cfg['binding']['hamiltonian'],'Preparation binding')
        require(prep.deterministic_fragment_indices==tuple(range(p)),'Saved fragment order')
    require(ham.constant+preps[7].extracted_identity_coefficient==.0194666612606853,'Saved scalar')
    require(preps[7].rte_preparation.symbolic_tail.lambda_r==.1649333839054104,'Saved lambda')
    import trotterlib.rte_compiled_cost as costs
    costs.transpile=partial(qiskit.transpile,num_processes=1)
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trotterlib.rte import CompilerSettings
    from trotterlib.rte_compiled_cost import transpile_and_measure_cost
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    CTX=SimpleNamespace(ham=ham,preps=preps,h=h,binding=cfg['binding'],inputs=cfg['inputs'],provider_sha256=verify_provider())
    return CTX


def candidate_plan():
    rows=[]
    for ref,path,q in [('95d01b6ab1d820921fcd2521324e41e82389713a','docs/research/h6_df19_rpe_d2_preparation_20261110/PLAN.json',2),('18fcd90f1614e3bf7b91bb02d8254f968e048a94','docs/research/h6_df19_rpe_m6_preparation_20261110/PLAN.json',4)]:
        d=json.loads(blob(ref,path))
        for r in d['rows']:
            c=r['candidate']
            if c['q']!=q or c['T']!=1.6*q:continue
            require((c['method'],c['prefix'],c['R_outer'],c['K']) in [('FULL',19,None,None),('PR',7,16,2)],'Pilot candidate')
            rows.append(dict(candidate=c,cohort='H6_PRICE_V2_20261110_Q'+str(q)+('_CALIBRATION' if q==2 else '_INDEPENDENT'),paths=8 if c['method']=='PR' else 1,carrier=ref,path=path))
    require(len(rows)==4,'Four fixed pilot groups')
    return rows


def prepare():
    ref=source();PRIVATE.mkdir(parents=True,exist_ok=False);ctx=context();groups=candidate_plan();h=ctx.h
    # Freeze exact scalar input/seed derivation/order before any new draw/build.
    seeds=[]
    for g in groups:
        g['seed_streams']=[]
        if g['candidate']['method']=='PR':
            for index in range(g['paths']):
                values=[h['trial_seed'](g['candidate'],g['cohort'],index,s['outer'],s['piece']) for s in h['core'].schedule('4th(m5_best)',g['candidate']['q'],'ABS_TIME_PER_OUTER',16,T=g['candidate']['T'])]
                g['seed_streams'].append(values);seeds.extend(values)
    require(len(seeds)==len(set(seeds))==528,'Distinct q2/q4 streams')
    previous=[]
    base='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
    coverage=[]
    for name in ('exploration_costs','confirmation_costs','reused_compiled_prices','reused_confirmation_costs','extension_parallel_costs','fixed_delta_exploration_costs'):
        data=json.loads(gzip.decompress(blob(CARRIER,base+name+'.json.gz')))
        found=[k for k,v in data.items() if v['candidate']['id'] in {g['candidate']['id'] for g in groups}]
        require(not found,'Existing matching direct price: reuse first')
        coverage.append(dict(table=name,rows=len(data),matching_candidates=found))
        for v in data.values():
            if v['candidate']['system']!='H6' or v['candidate']['method']!='PR':continue
            for f in v.get('trial_files',[]):
                ep=Path(f).with_name('events_PRIVATE.json')
                if ep.exists():previous.extend(read(ep).get('seeds',[]))
    require(not set(seeds).intersection(previous),'Seed collision in inspected H6 cohorts')
    write(PRIVATE/'INPUTS_PRIVATE.json',dict(source=ref,groups=groups,binding=ctx.binding,inputs=ctx.inputs,compiler=COMPILER,provider_source=PROVIDER,provider_sha256=ctx.provider_sha256,seed_rule='Exact pinned trial_seed; distinct cohort and candidate enter SHA256',application_order='pinned make_df_partial_s2_step_request and QiskitDFRTEEventCircuitBuilder, unsorted',allocation=list(h['core'].allocation(h['core'].weights('4th(m5_best)'),16)),old_cache_read_only=str(OLD_CACHE),maximum_new_full_wrapper_prices=36))
    public_groups=[{k:v for k,v in g.items() if k!='seed_streams'} for g in groups]
    write(PUBLIC/'PILOT_INPUTS.json',dict(source=ref,groups=public_groups,compiler_hash=COMPILER,scientific_binding=ctx.binding,input_sha256={k:v['sha256'] for k,v in ctx.inputs.items()},provider_source=PROVIDER,preparation_hash={str(p):prep.preparation_hash for p,prep in ctx.preps.items()},weights=list(h['core'].weights('4th(m5_best)')),R16_allocation=list(h['core'].allocation(h['core'].weights('4th(m5_best)'),16)),seed_values_private=True,distinct_new_streams=len(seeds),collisions_with_inspected_historical_streams=0,complete_historical_seed_coverage=False,prior_price_tables=coverage,old_cache_policy='Only exact wrapper fingerprint+compiler match; no writes',new_price_budget=36,scope=dict(signal_N_G=0,old_V1_recounts=0),metrics=['signed residual','MAE','RMSE','relative bias/RMSE/max','PR path residual SD','paired mean covariance'],selected_V2=read(PUBLIC/'SAVED_CORRECTIONS.json')['selected'],q4_refit=False,full_RPE_qualified=False))
    print('PREPARED',ref,flush=True)


def piece_blocks(prep,ctx):
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
    result=[]
    for w in ctx.h['core'].weights('4th(m5_best)'):
        bs=[]
        for b in prep.deterministic_blocks:
            diag=one_body_diagonal_primitives(np.array(b.diagonal_eigenvalues),1.6*w/2) if b.block_kind=='one_body' else df_squared_diagonal_primitives(np.array(b.diagonal_eta),b.lam,1.6*w/2)
            basis=[dict(name=o.name,qubits=list(q),parameters=[str(t) for t in o.params]) for o,q in b.runtime_basis_operations]
            bs.append(dict(basis=basis,diagonal=diag))
        result.append(bs)
    return result


def predict(blocks,q,scalar,T,axis,events):
    require(q in (2,4) and T==1.6*q,'New pilot scope')
    a=Accounting();a.gate('h',(12,),role='wrapper')
    for outer in range(q):
        for j,bs in enumerate(blocks):
            for b in bs:a.block(b,False)
            if events is not None:a.tail(events[11*outer+j])
            for b in reversed(bs):a.block(b,True)
    if scalar*T!=0:a.gate('p',(12,),(-scalar*T,),'scalar_once')
    if axis=='sine':a.gate('sdg',(12,),role='wrapper')
    a.gate('h',(12,),role='wrapper');a.gate('measure',(12,),role='measurement')
    a.features.update(signed_piece_occurrences=11*q,negative_piece_occurrences=q,piece_boundaries=11*q-1,outer_step_boundaries=q-1)
    if events is not None:a.features['head_tail_interfaces']=22*q
    return a.finish()


def trial(job):
    g,index=job;row=g['candidate'];start=time.monotonic();ctx=context();prep=ctx.preps[row['prefix']]
    path=PRIVATE/('q'+str(row['q']))/row['id']/str(index);path.mkdir(parents=True,exist_ok=False)
    try:
        pr=row['method']=='PR';requests,seeds,laws=ctx.h['sample_requests'](row,prep,g['cohort'],index) if pr else ([],[],[])
        if pr:require(seeds==g['seed_streams'][index],'Frozen seeds')
        events=[[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
        write(path/'events_PRIVATE.json',dict(events=events,seeds=seeds,laws=laws,cohort=g['cohort'],conditional=False))
        scalar=ctx.ham.constant+(prep.extracted_identity_coefficient if pr else 0.)
        blocks=piece_blocks(prep,ctx)
        predictions={axis:predict(blocks,row['q'],scalar,row['T'],axis,events if pr else None) for axis in AXES}
        # Predictions are checkpointed before observing this path's compiled price.
        write(path/'predictions.json',predictions)
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        from trottertracks.resource_applicability import ax2b_native_df_v5 as native
        from trotterlib.rte import CompilerSettings
        from trotterlib.rte_compiled_cost import transpile_and_measure_cost
        prices={};fingerprints={};hits=0
        with use_fingerprint_reuse() as fp:
            evolution=ctx.h['compose'](row,prep,requests,scalar,fp.fingerprint)
            write(path/'native.json',dict(instructions=len(evolution.circuit.data),instruction_bound=evolution.instruction_upper_bound,fingerprint=evolution.circuit_fingerprint))
            for axis in AXES:
                wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=len(evolution.circuit.data)+4)
                fingerprint=fp.fingerprint(wrapper);fingerprints[axis]=fingerprint
                old=OLD_CACHE/(fingerprint+'.json')
                if old.exists():
                    data=read(old);require(data['fingerprint']==fingerprint and data['compiler']==COMPILER,'Cache binding');metrics=data['metrics'];hits+=1
                    write(path/(axis+'_cache_hit.json'),dict(path=str(old),sha256=sha(old.read_bytes()),fingerprint=fingerprint,compiler=COMPILER))
                else:
                    write(path/(axis+'_compile_started.json'),dict(fingerprint=fingerprint,compiler=COMPILER))
                    compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0')
                    cost=transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
                    require(cost.compiler_settings_hash==COMPILER,'Compiler hash')
                    metrics={k:getattr(cost,k) for k in METRICS}
                    del cost
                require(fp.fingerprint(wrapper)==fingerprint,'Compiler mutated input')
                prices[axis]=metrics
                write(path/(axis+'_price.json'),dict(metrics=metrics,fingerprint=fingerprint,compiler=COMPILER))
                del wrapper
        result=dict(status='PASS_PRICE_ACQUIRED',candidate=row,cohort=g['cohort'],index=index,preparation_hash=prep.preparation_hash,scalar=scalar,prices=prices,fingerprints=fingerprints,predictions=predictions,cache_hits=hits,orders={str(k):sum(e['taylor_order']==k for batch in events for e in batch) for k in (0,2)},wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,conditional_sampling=False,signal_N_G_computed=False,full_wrapper_statevector_validation=False)
        write(path/'result.json',result);return result
    except BaseException as e:
        write(path/'STOP.json',dict(reason=str(e),traceback=traceback.format_exc()));raise


def run(q):
    ref=source();plan=read(PRIVATE/'INPUTS_PRIVATE.json');require(ref==plan['source'],'Pilot source changed')
    if q==4:require((PUBLIC/'Q4_FROZEN_MODELS.json').exists(),'Freeze coefficients before q4')
    ctx=context();groups=[g for g in plan['groups'] if g['candidate']['q']==q]
    jobs=[(g,i) for g in groups for i in range(g['paths'])]
    require(not (PRIVATE/('q'+str(q))).exists(),'No implicit rerun')
    # Nine independent whole-path jobs, each compiler uses one process.
    workers=min(len(jobs),len(os.sched_getaffinity(0)))
    print('START',q,'workers',workers,flush=True);results=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        futures={pool.submit(trial,j):j for j in jobs}
        for f in as_completed(futures):
            r=f.result();results.append(r);print('DONE',q,r['candidate']['method'],r['index'],r['prices']['cosine']['rz_count'],round(r['wall_seconds'],2),flush=True)
    results.sort(key=lambda r:(r['candidate']['method'],r['index']))
    write(PUBLIC/('Q'+str(q)+'_DIRECT.json'),dict(scientific_source=ref,compiler_hash=COMPILER,rows=results,workers=workers,nested_compiler_processes=1,new_full_wrapper_prices=sum(2-r['cache_hits'] for r in results),cache_hits=sum(r['cache_hits'] for r in results),full_RPE_qualified=False))


if __name__=='__main__':
    command=sys.argv[1]
    if command=='prepare':prepare()
    elif command in ('q2','q4'):run(int(command[1:]))
    else:raise ValueError('Use prepare/q2/q4')
