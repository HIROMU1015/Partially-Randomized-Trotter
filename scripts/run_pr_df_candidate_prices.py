"""Candidate-owned PR-DF price acquisition/model path, using fixed D2 workers.

Preparation preserves the supplied mapper's q/R/step. No prefix, kappa, step or
shot exploration is performed here. Root coordinator owns publication sequence.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from fractions import Fraction
import hashlib
import fcntl
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
CALLER='scripts/run_pr_df_candidate_prices.py'
FIXED_VIEW=ROOT.parent/'d2-pr-df-ld1-m3-acquisition-20261115'
VIEW_COMMIT='326feadd093920bbea29b498108aa4eae6f67a75'
REUSED='ea9e66092465d39dfcd4aeb6fea8e9f09808cabb'
ADAPTER_SOURCE='e827367629ca6c2920d9d908fc1631314399db86'
SOLVER=('d83737b6d5241962ce39c81d39c56af3506c0cdd','src/trottertracks/price_model_validation.py')
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
BASE_INPUTS=dict(commit='e0b3cad2a024070389f757160f3a20d7da02c5ff',path='docs/research/d_pr_df_ld_split_results_20261115/ACQUISITION_INPUTS.json')
SCALAR_CONSTANT=dict(commit='0defa286c53a1b9acf513be453907bcad83102b3',path='docs/research/d_pr_df_ld7_v0_results_20261116/generation/SCALAR.json',selector=['hamiltonian_constant'])
RUNTIME=None
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)


def blob(c,p):return subprocess.check_output(['git','show',c+':'+p],cwd=ROOT)
def read(loc):
    value=json.loads(blob(loc['commit'],loc['path']))
    for k in loc.get('selector',[]):value=value[k]
    return value

def enc(x):
    x=Fraction(x);return dict(numerator=x.numerator,denominator=x.denominator)
def dec(x):return Fraction(x['numerator'],x['denominator'])if isinstance(x,dict)else Fraction(x)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def fixed(source):
    if blob(source,CALLER)!=Path(__file__).read_bytes():raise ValueError('Unfixed DF price runner SOURCE')
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def runtime():
    global RUNTIME
    if RUNTIME is not None:return RUNTIME
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=FIXED_VIEW,text=True).strip()!=VIEW_COMMIT:raise ValueError('DF worker view differs')
    path='scripts/run_pr_df_ld_split.py'
    if (FIXED_VIEW/path).read_bytes()!=blob(REUSED,path):raise ValueError('Reused DF path worker bytes differ')
    sys.path[:0]=[str(FIXED_VIEW/'src'),str(FIXED_VIEW/'scripts')]
    spec=importlib.util.spec_from_file_location('_candidate_df_price_worker',FIXED_VIEW/path)
    R=importlib.util.module_from_spec(spec);sys.modules[spec.name]=R;spec.loader.exec_module(R)
    R.ROOT=ROOT;RUNTIME=R;return R


def solve(matrix,rhs):
    tree=ast.parse(blob(*SOLVER));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_solve')
    namespace=dict(Fraction=Fraction)
    exec(compile(ast.Module(body=[node],type_ignores=[]),SOLVER[0]+':_solve','exec'),namespace)
    return namespace['_solve'](matrix,rhs)


def design(config):
    rows=config['rounds'];s=config['spec'];bad=[]
    if s['method']!='PR'or s['split'].get('tail_dictionary'):bad.append('PR_DF_REQUIRED')
    if s['formula']!='2nd'or s['finite_K']!=2 or any(r['structure']['signed_weights']!=[1.]or len(r['structure']['allocation'])!=1 for r in rows):
        bad.append('PRICE_COEFFICIENTS_FOR_CHANGED_FORMULA_PIECES_OR_K_NOT_ADOPTED')
    if len({dec(r['actual_delta_pf'])for r in rows})!=1:bad.append('MULTIPLE_ACTUAL_STEPS_REQUIRE_SEPARATE_CALIBRATION_GROUPS')
    if len(rows)!=13 or [r['m']for r in rows]!=list(range(13)):bad.append('CURRENT_H4_13_ROUND_SCOPE_REQUIRED')
    mode='FULL_AFFINE_Q_R1'if all(r['R_outer']==1 for r in rows)else'FULL_AFFINE_QR'
    calibration=[0,6]if mode=='FULL_AFFINE_Q_R1'else[0,3,6]
    holdout=3 if mode=='FULL_AFFINE_Q_R1'else 4
    matrix=[[Fraction(1),Fraction(rows[m]['q_PF'])]+([]if mode=='FULL_AFFINE_Q_R1'else[Fraction(rows[m]['q_PF']*rows[m]['R_outer'])])for m in calibration]
    try:solve(matrix,[Fraction(0)]*len(matrix))
    except ValueError:bad.append('RANK_DEFICIENT_HEAD_Q_AND_TAIL_QR_COEFFICIENTS')
    return dict(status='CALIBRATION_DESIGN_READY'if not bad else'NEEDS_PRICE_RULE_OR_DESIGN_EXTENSION',reasons=bad,
                mode=mode,calibration_rounds=calibration,holdout_round=holdout,matrix=matrix)


def prepare(config_source,output,source,cohort,price_quote=None,tail_descriptor=None):
    fixed(source);config=read(config_source);output=ROOT/output;output.mkdir(parents=True,exist_ok=True)
    selected=design(config)
    if price_quote is not None:
        if price_quote.get('configuration_id')!=config['id']:raise ValueError('Price quote belongs to another configuration')
        write(output/'PRICE_QUOTE.json',price_quote)
        relevant=set(selected['calibration_rounds']+[selected['holdout_round']])
        existing=[dict(m=r['m'],axis=r['axis'],physical_id=r.get('physical_id'),direct_alternatives=r.get('direct_alternatives',[]))
                  for r in price_quote.get('rows',[])if r['m']in relevant and r.get('direct_alternatives')]
        if existing:
            selected['existing_receipt_cells']=existing
            selected['reasons'].append('RECONCILE_EXISTING_DIRECT_PATH_RECEIPTS_BEFORE_NEW_SAMPLING')
            selected['status']='NEEDS_SAVED_RECEIPT_REUSE_RESOLUTION'
    if selected['reasons']:
        write(output/'NEEDS.json',{k:v for k,v in selected.items()if k!='matrix'});return
    R=runtime();index=read(INDEX);descriptor=index['T20_reference_files']['H4'];data=Path(descriptor['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=descriptor['sha256']:raise ValueError('Saved tail descriptor bytes differ')
    context=json.loads(data);ld=config['spec']['L_D']
    if tail_descriptor is None:
        tail=context['prefixes'][str(ld)]
        tail_source=dict(index=INDEX,context_sha256=descriptor['sha256'],selector=['prefixes',str(ld)])
    else:
        extension_spec=importlib.util.spec_from_file_location('_df_price_descriptor_extension',ROOT/'src/trottertracks/pr_df_descriptor_extension.py')
        extension=importlib.util.module_from_spec(extension_spec);extension_spec.loader.exec_module(extension)
        tail=extension.validate(extension.read(ROOT,tail_descriptor),tail_descriptor,config);tail_source=tail_descriptor
    if config['spec']['split']['lambda_source']!=tail_source or float(dec(config['spec']['split']['lambda_R'])).hex()!=float(tail['lambda_R']).hex():raise ValueError('Candidate tail binding differs')
    constant=read(SCALAR_CONSTANT)
    if constant['source']['hamiltonian_hash']!=config['spec']['binding']['hamiltonian']:raise ValueError('Hamiltonian constant binding differs')
    scalar=constant['value']+tail['scalar'];scalar_record=dict(hamiltonian_constant=constant,tail_identity=dict(value=tail['scalar'],source=tail_source),total=scalar,total_hex=scalar.hex(),application='ONCE_AT_TOTAL_T')
    write(output/'SCALAR.json',scalar_record)
    seed=R.P.trial_seed_function(ROOT);cells=[]
    for m,n,role in [(m,4,'CALIBRATION')for m in selected['calibration_rounds']]+[(selected['holdout_round'],2,'HOLDOUT')]:
        request=R.D.from_b(config,m,config['rounds'][m]['physical_candidate'],
            input_source=dict(configuration=config_source,tail=tail_source,scalar=dict(hamiltonian_constant=SCALAR_CONSTANT,tail=tail_source)),
            execution_source=dict(commit=source,path=CALLER),expected_scalar=scalar)
        c=request['physical_context']['physical_candidate'];name=cohort+'_'+role
        seeds=[[seed(c,name,i,j,0)for j in range(c['q'])]for i in range(n)]
        cells.append(dict(m=m,role=role,request=request,cohort=name,indices=list(range(n)),seeds=seeds))
    all_seeds=[v for c in cells for row in c['seeds']for v in row]
    if len(set(all_seeds))!=len(all_seeds):raise ValueError('Calibration/holdout seed overlap')
    old=read(BASE_INPUTS)
    write(output/'INPUTS.json',dict(schema='PR_DF_CANDIDATE_PRICE_INPUTS_V1',source=source,configuration=config_source,configuration_id=config['id'],output_directory=rel(output),
        tail_descriptor=tail_descriptor,cells=cells,model_design={k:v for k,v in selected.items()if k!='matrix'},runtime=old['runtime'],saved_input_files=old['saved_input_files'],
        local_support=old['local_support'],reused_worker=REUSED,adapter_source=ADAPTER_SOURCE,scalar=scalar_record,
        max_new_paths=sum(len(c['indices'])for c in cells),max_wrapper_compiles=2*sum(len(c['indices'])for c in cells),
        full_wrapper_statevector_validation=False,numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',
        model_qualification='FINITE_COHORT_EXPLORATORY_REFERENCE_NOT_EXPECTATION'))


def load_inputs(path,commit,source):
    fixed(source);path=Path(path).resolve();raw=blob(commit,rel(path))
    if raw!=path.read_bytes():raise ValueError('Input bytes differ from fixed input commit')
    inputs=json.loads(raw)
    if inputs['source']!=source:raise ValueError('Execution SOURCE mismatch')
    return inputs


def _acquire(inputs,input_commit,stage,source,model_commit=None):
    R=runtime();base=inputs['output_directory'];R.PUBLIC=base+'/'+stage
    for path,want in inputs.get('local_support',{}).items():
        if hashlib.sha256((FIXED_VIEW/path).read_bytes()).hexdigest()!=want:raise ValueError('Reused support SOURCE differs: '+path)
    cells=[c for c in inputs['cells']if c['role']==('CALIBRATION'if stage=='calibration'else'HOLDOUT')]
    if stage=='holdout':
        if not model_commit:raise ValueError('Independent holdout requires published frozen model')
        model=read(dict(commit=model_commit,path=base+'/FROZEN_MODELS.json'));prices=read(dict(commit=model_commit,path=base+'/AXIS_PRICES.json'))
        if model['status']!='FROZEN_2_AXIS_MODELS'or len(prices['rows'])!=26 or any(dec(r['C_ref'])<=0 for r in prices['rows']):raise ValueError('Positive frozen model prerequisite')
    out=ROOT/R.PUBLIC/'acquisition';out.mkdir(parents=True,exist_ok=True);pending_jobs=[];rows=[];processes=[];start=time.perf_counter()
    for cell in cells:
        for index in cell['indices']:
            receipt=out/f'm{cell["m"]}'/f'p{index}'/'PATH_RECEIPT.json';marker=out/f'STARTED_m{cell["m"]}_p{index}.json';stop=out/f'STOP_m{cell["m"]}_p{index}.json'
            if receipt.exists():
                value=json.loads(receipt.read_text());record=value.get('record',{})
                if value.get('status')!='VALID'or record.get('request')!=cell['request']or record.get('seeds')!=cell['seeds'][index]or record.get('cohort')!=cell['cohort']or record.get('index')!=index:raise ValueError('Saved receipt identity/gate mismatch')
                rows.append(dict(value,disposition='REUSED_COMPLETED_PATH'));continue
            if marker.exists()or stop.exists():
                rows.append(dict(m=cell['m'],index=index,status='HELD_PRIOR_START_OR_FAILURE_NO_RETRY',compiles=[]));continue
            pending_jobs.append((cell,index))
    if pending_jobs:
        ham,metadata=R.restore(cells[0]['request'],inputs['saved_input_files'],inputs['runtime']);contexts={};runtimes={}
        for cell in cells:
            pc=cell['request']['physical_context'];r=R.D.NativeRuntime(ROOT,ham,binding=pc['binding'],input_source=pc['input_source'],execution_source=pc['execution_source'])
            runtimes[cell['m']]=r;contexts[cell['m']]=r.context(cell['request'])
        preparation=dict(metadata=metadata,contexts={str(m):dict(preparation_hash=c.prep.preparation_hash,scalar=c.scalar,head=list(c.prep.deterministic_fragment_indices),tail=list(c.prep.randomized_block_indices))for m,c in contexts.items()})
        if (out/'PREPARATION.json').exists():
            if json.loads((out/'PREPARATION.json').read_text())!=preparation:raise ValueError('Saved preparation identity differs on resume')
        else:write(out/'PREPARATION.json',preparation)
        R.STATE.update(inputs=dict(inputs,cache_root=str(out/'isolated_cache')),runtimes=runtimes,contexts=contexts)
        ctx=mp.get_context('fork');flag=ctx.Event()
        with ProcessPoolExecutor(max_workers=len(pending_jobs),mp_context=ctx,initializer=R.init_price,initargs=(flag,))as pool:
            futures={}
            for pair in pending_jobs:
                cell,index=pair;write(out/f'STARTED_m{cell["m"]}_p{index}.json',dict(source=source,inputs_commit=input_commit,epoch=time.time(),automatic_retry=False))
                futures[pool.submit(R.acquire_path,pair)]=(cell['m'],index)
            processes=list(pool._processes.values())
            while futures:
                done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    m,i=futures.pop(future)
                    try:value=future.result()
                    except BaseException:value=dict(m=m,index=i,status='WORKER_ERROR',error=traceback.format_exc(),compiles=[])
                    rows.append(dict(value,disposition='NEW_PATH'))
                print('PRICE_PROGRESS',stage,'finished',len(rows),'pending',list(futures.values()),flush=True)
    good=len(rows)==sum(len(c['indices'])for c in cells)and all(r['status']=='VALID'for r in rows)
    atomic(out/'RESULT.json',dict(status='ACQUIRED'if good else'STOP_PARTIAL',rows=sorted(rows,key=lambda r:(r['m'],r['index'])),
        new_paths=sum(r.get('disposition')=='NEW_PATH'for r in rows),reused_paths=sum(r.get('disposition')=='REUSED_COMPLETED_PATH'for r in rows),
        compile_attempts=sum(len(r['compiles'])for r in rows if r.get('disposition')=='NEW_PATH'),completed_compiles=sum(c['status']=='COMPLETE'for r in rows if r.get('disposition')=='NEW_PATH'for c in r['compiles']),
        source=source,inputs_commit=input_commit,model_commit=model_commit,elapsed_seconds=time.perf_counter()-start,
        process_exits=[dict(pid=p.pid,exitcode=p.exitcode)for p in processes],automatic_retry=False))
    return good


def acquire(inputs,input_commit,stage,source,model_commit=None):
    directory=ROOT/inputs['output_directory']/stage/'acquisition';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'DRIVER.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();raise ValueError('Another acquisition owns this candidate/stage')
    try:return _acquire(inputs,input_commit,stage,source,model_commit)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def means(inputs,raw_commit,stage):
    base=inputs['output_directory'];role='CALIBRATION'if stage=='calibration'else'HOLDOUT';out=[]
    for cell in [c for c in inputs['cells']if c['role']==role]:
        m=cell['m'];n=len(cell['indices'])
        for axis in ('X','Y'):
            paths=[]
            for index in cell['indices']:
                path=base+f'/{stage}/acquisition/m{m}/p{index}/PATH_RECEIPT.json';value=read(dict(commit=raw_commit,path=path));r=value['record'];a=next(v for v in r['records']if v['axis']==axis)
                receipt_path=base+f'/{stage}/acquisition/m{m}/p{index}/'+{'X':'cosine','Y':'sine'}[axis]+'.json';receipt=read(dict(commit=raw_commit,path=receipt_path))
                if value['status']!='VALID'or r['request']!=cell['request']or r['seeds']!=cell['seeds'][index]or r['cohort']!=cell['cohort']or r['index']!=index:raise ValueError('Price path identity mismatch')
                if a['C']!=a['metrics']['rz_count']or a['D']!=a['C']-a['S']or receipt['metrics']!=a['metrics']or receipt['fingerprint']!=a['fingerprint']or receipt['compiler']!=cell['request']['physical_context']['price_profile']['compiler']:raise ValueError('Axis receipt metric/fingerprint/compiler mismatch')
                paths.append(dict(index=index,axis=axis,cohort=cell['cohort'],seeds=r['seeds'],S=a['S'],C=a['C'],D=a['D'],fingerprint=a['fingerprint'],event_digest=r['event_digest'],receipt=dict(commit=raw_commit,path=receipt_path),path_receipt=dict(commit=raw_commit,path=path)))
            c=cell['request']['physical_context']['physical_candidate'];out.append(dict(m=m,axis=axis,q=c['q'],R=c['R_outer'],physical_id=c['id'],n_paths=n,cohort=cell['cohort'],mean_C=enc(sum(Fraction(p['C'])for p in paths)/n),paths=paths))
    return out


def freeze(inputs,raw_commit,source):
    cfg=read(inputs['configuration']);mode=inputs['model_design']['mode'];cells=means(inputs,raw_commit,'calibration');models={};rows=[]
    for axis in ('X','Y'):
        own=[c for c in cells if c['axis']==axis]
        matrix=[[Fraction(1),Fraction(c['q'])]+([]if mode=='FULL_AFFINE_Q_R1'else[Fraction(c['q']*c['R'])])for c in own]
        co=solve(matrix,[dec(c['mean_C'])for c in own]);model=dict(features=['one','q']if len(co)==2 else['one','q','qR'],offset=0,
            calibration=own,calibration_residuals=[enc(0)]*len(own),independent_validation='NOT_YET_OPENED',coefficients={k:enc(v)for k,v in zip(('one','q','qR'),co)})
        if len(co)==2:model.update(alpha=enc(co[0]),beta=enc(co[1]))
        models[axis]=model
        for r in cfg['rounds']:
            q,Rr=r['q_PF'],r['R_outer'];p=co[0]+co[1]*q+(co[2]*q*Rr if len(co)==3 else 0)
            if p<=0:raise ValueError('Nonpositive candidate model price; no clipping/refit')
            rows.append(dict(L_D=cfg['spec']['L_D'],configuration_id=cfg['id'],m=r['m'],axis=axis,physical_id=r['physical_candidate']['id'],q=q,R=Rr,C_ref=enc(p),
                application='CALIBRATION'if r['m']in inputs['model_design']['calibration_rounds']else'LONG_ROUND_EXTRAPOLATION'if r['m']>6 else'UNCHECKED_INTERPOLATION',law=r['law_binding']))
    output=ROOT/inputs['output_directory'];write(output/'FROZEN_MODELS.json',dict(status='FROZEN_2_AXIS_MODELS',source=source,raw_commit=raw_commit,mode=mode,models=models,
        model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',finite_cohort_is_law_expectation=False,holdout_not_opened=True,formula='FULL_WRAPPER_C_OFFSET_ZERO_NO_S_ADDITION'))
    write(output/'AXIS_PRICES.json',dict(source=source,rows=sorted(rows,key=lambda r:(r['m'],r['axis'])),positive_finite_axes=26))


def compare(inputs,raw_commit,model_commit,source):
    base=inputs['output_directory'];cells=means(inputs,raw_commit,'holdout');prices=read(dict(commit=model_commit,path=base+'/AXIS_PRICES.json'));rows=[]
    for c in cells:
        p=next(v for v in prices['rows']if(v['m'],v['axis'])==(c['m'],c['axis']));P=dec(p['C_ref']);O=dec(c['mean_C']);d=P-O
        rows.append(dict(**c,prediction=enc(P),observed=enc(O),signed_error=enc(d),absolute_error=enc(abs(d)),signed_relative_error=None if O==0 else enc(d/O),absolute_relative_error=None if O==0 else enc(abs(d)/O),relative_reason='ZERO_OBSERVED_PRICE'if O==0 else None))
    write(ROOT/base/'HOLDOUT_COMPARISON.json',dict(source=source,raw_commit=raw_commit,model_commit=model_commit,rows=rows,status='DESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED'if any(dec(r['observed'])==0 for r in rows)else'DESCRIPTIVE_COMPLETE',model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE',refit=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=('prepare','acquire','freeze','compare-holdout'));ap.add_argument('--source',required=True)
    ap.add_argument('--tail-descriptor',type=Path);ap.add_argument('--price-quote',type=Path);ap.add_argument('--configuration');ap.add_argument('--selector',default='[]');ap.add_argument('--output');ap.add_argument('--cohort');ap.add_argument('--input',type=Path);ap.add_argument('--inputs-commit');ap.add_argument('--stage',choices=('calibration','holdout'));ap.add_argument('--raw-commit');ap.add_argument('--model-commit');a=ap.parse_args()
    if a.mode=='prepare':
        commit,path=a.configuration.split(':',1);prepare(dict(commit=commit,path=path,selector=json.loads(a.selector)),a.output,a.source,a.cohort,None if a.price_quote is None else json.loads(a.price_quote.read_text()),None if a.tail_descriptor is None else json.loads(a.tail_descriptor.read_text()))
    else:
        inputs=load_inputs(a.input,a.inputs_commit,a.source)
        if a.mode=='acquire':return 0 if acquire(inputs,a.inputs_commit,a.stage,a.source,a.model_commit)else 2
        if a.mode=='freeze':freeze(inputs,a.raw_commit,a.source)
        else:compare(inputs,a.raw_commit,a.model_commit,a.source)
    return 0


if __name__=='__main__':raise SystemExit(main())
