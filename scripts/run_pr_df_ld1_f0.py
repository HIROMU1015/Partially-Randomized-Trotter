"""One-shot LD1 F0: saved probabilities only; no sampler, circuit, price or fit imports."""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
PLAN=dict(commit='950577f416c17a20d309a8053a4bb016027b724d',path='docs/research/d_pr_df_ld1_ec_plan_20261115/NEXT_F0_PLAN.json')
HELPER=dict(commit='d72dc4e6228bb380f4d1d000e5e030794fce7164',path='src/trottertracks/finite_k_sampler_law.py')
CDF=dict(commit='155529e20a8fac39e00786c40b57830b71203c4f',path='scripts/h46_long_time_native.py')
PROVIDER=dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe',path='src/trotterlib/rte.py')
PUBLIC='docs/research/d_pr_df_ld1_f0_law_diagnostic_20261115'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
@lru_cache(maxsize=None)
def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(ref):
    value=json.loads(blob(ref['commit'],ref['path']))
    for key in ref.get('selector',[]):value=value[key]
    return value

def safe(value):
    if isinstance(value,Fraction):return dict(numerator=value.numerator,denominator=value.denominator)
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value

def write(path,value):
    with Path(path).open('x') as f:json.dump(safe(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def progress(path,stage,state,**fields):
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(stage=stage,state=state,monotonic=time.perf_counter(),peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**fields))+'\n');f.flush()


def component_signature(prep):
    cs=prep['components']
    if len(cs)!=prep['component_count'] or len({r['id'] for r in cs})!=len(cs):raise ValueError('COMPONENT_COUNT_OR_IDS')
    for r in cs:
        p=r['probability']
        if type(p) not in (int,float) or not math.isfinite(p) or p<0 or float(p).hex()!=r['probability_hex']:raise ValueError('COMPONENT_PROBABILITY_HEX')
    return dict(preparation_hash=prep['hash'],components=deepcopy(cs))


def law_signature(law,row):
    for key,wanted in [('m',row['m']),('candidate_id',row['physical_id']),('q_PF',row['q']),('R_outer',row['R_outer']),('event_count',row['saved_event_positions'])]:
        if law[key]!=wanted:raise ValueError('LAW_IDENTITY:'+key)
    if law['orders']!=[0,2] or len(law['order_weights'])!=2 or len(law['order_probabilities'])!=2:raise ValueError('K2_ORDER_SUPPORT')
    if float(law['tau']).hex()!=law['tau_hex']:raise ValueError('LAW_TAU_HEX')
    fields=('m','candidate_id','q_PF','R_outer','event_count','tau','tau_hex','orders','order_weights','order_probabilities','event_normalization')
    for v in [law['tau'],law['event_normalization'],*law['order_weights'],*law['order_probabilities']]:
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('NONFINITE_LAW_METADATA')
    if law['event_normalization']<=0 or any(v<0 for v in law['order_weights']+law['order_probabilities']):raise ValueError('NEGATIVE_OR_ZERO_LAW_METADATA')
    return {k:deepcopy(law[k]) for k in fields}


def bind_inputs():
    plan=read(PLAN);prep=read(plan['saved_inputs']['identity_and_order'])
    csig=component_signature(prep)
    if csig['preparation_hash']!='b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7' or len(csig['components'])!=324:raise ValueError('FIXED_LD1_PREPARATION')
    tables=[dict(id='components',kind='component',source=plan['saved_inputs']['identity_and_order'],signature=csig)]
    for row in plan['saved_inputs']['law_rows']:
        tables.append(dict(id='order_m'+str(row['m']),kind='order',source=row['source'],row=deepcopy(row),signature=law_signature(read(row['source']),row)))
    sources=[PLAN,HELPER,CDF,PROVIDER,*[t['source'] for t in tables]]
    unique={(r['commit'],r['path']) for r in sources}
    return dict(schema='PR_DF_LD1_F0_FIXED_INPUTS_V1',plan=PLAN,target_cost_law='ideal_finite_RTE_law',helper=HELPER,cdf_helper=CDF,
        normalization_source=dict(PROVIDER,symbols=['_validate_components','iter_sample_rte_events']),
        source_files=[dict(commit=c,path=p,sha256=sha(blob(c,p))) for c,p in sorted(unique)],tables=tables,
        compile_count=0,path_count=0,assumptions=deepcopy(plan['assumptions']),output=plan['proposed_output'],
        mandatory_stop=True)


def load_fixed_functions():
    # Full pinned helper definitions; its optional saved_cdf_helper is not called.
    # Extract the same pinned pure cdf_law function without importing its launcher.
    scope=dict(__name__='_fixed_ld1_f0',__package__='trottertracks')
    exec(compile(blob(HELPER['commit'],HELPER['path']),HELPER['commit']+':'+HELPER['path'],'exec'),scope)
    tree=ast.parse(blob(CDF['commit'],CDF['path']))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cdf_law')
    cscope=dict(Fraction=Fraction)
    exec(compile(ast.Module(body=[node],type_ignores=[]),CDF['commit']+':cdf_law','exec'),cscope)
    return SimpleNamespace(**{k:scope[k] for k in ('categorical_audit','k2_event_bound','trajectory_bound','expectation_shift_bound')}),cscope['cdf_law']


def read_table(table,reader=read):
    data=reader(table['source'])
    actual=component_signature(data) if table['kind']=='component' else law_signature(data,table['row'])
    if actual!=table['signature']:raise ValueError('SAVED_TABLE_ORDER_OR_FIELDS_CHANGED:'+table['id'])
    if table['kind']=='component':
        raw=[r['probability'] for r in data['components']]
        # Literal provider operation, rte._validate_components / iter_sample_rte_events.
        total=math.fsum(raw);choice=[p/total for p in raw]
        labels=[dict(index=i,component_id=r['id'],basis_id=r['basis_id'],support=r['support']) for i,r in enumerate(data['components'])]
    else:
        raw=list(data['order_weights']);choice=list(data['order_probabilities']);labels=[dict(index=i,order=k) for i,k in enumerate(data['orders'])]
        # Literal finite_rte_distribution normalization, using only saved weights.
        if math.fsum(raw)!=data['event_normalization'] or [w/data['event_normalization'] for w in raw]!=choice:raise ValueError('SAVED_ORDER_NORMALIZATION_DIFFERS')
    return raw,choice,labels


def audit_one(table,helper,cdf,reader=read):
    raw,choice,labels=read_table(table,reader)
    audit=helper.categorical_audit(choice,cdf_law=cdf,ideal_weights=raw)
    if audit['cdf'][-1]!=1 or any(a>b for a,b in zip((Fraction(0),*audit['cdf']),audit['cdf'])):raise ValueError('CDF_NONMONOTONIC_OR_MASS')
    if sum(audit['grid_counts'])!=2**53:raise ValueError('GRID_MASS')
    changes=[]
    for before,after in (('P','Q'),('Q','F'),('F','G'),('P','G')):
        for i,(a,b) in enumerate(zip(audit[before],audit[after])):
            if (a>0 and b==0) or (a==0 and b>0):changes.append(dict(stage=before+'->'+after,label=labels[i],change='POSITIVE_TO_ZERO' if b==0 else 'ZERO_TO_POSITIVE',before=a,after=b))
    return dict(id=table['id'],kind=table['kind'],source=table['source'],status='DIAGNOSTIC_COMPLETE',
        input_reference_weight_hex=[float(v).hex() for v in raw],choice_input_hex=[float(v).hex() for v in choice],labels=labels,
        **audit,support_changes=changes,order2_mass={stage:audit[stage][1] for stage in ('P','Q','F','G')} if table['kind']=='order' else None)


def public_path(path):
    try:return str(Path(path).relative_to(ROOT))
    except ValueError:return str(path)


def run_tables(bundle,out,helper,cdf,reader=read):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);log=out/'PROGRESS.jsonl';start=time.perf_counter()
    results=[];bounds=[];errors=[];component=None;attempted=0
    for table in bundle['tables']:
        attempted+=1;began=time.perf_counter();progress(log,table['id'],'START')
        try:
            value=audit_one(table,helper,cdf,reader);write(out/(table['id']+'.json'),value)
            results.append(dict(id=table['id'],status=value['status'],path=public_path(out/(table['id']+'.json'))))
            if table['kind']=='component':component=value
            else:
                row=table['row'];ev=helper.k2_event_bound(value['P'],value['G'],component['TV']['total'],iid_components=True)
                path=helper.trajectory_bound(ev,row['saved_event_positions'],uniform_conditional_bound=True)
                bounds.append(dict(m=row['m'],physical_id=row['physical_id'],axes=['X','Y'],event_positions=row['saved_event_positions'],
                    source=table['source'],order_table=table['id']+'.json',component_table='components.json',event_TV_upper=ev,path_TV_upper=path,
                    informativeness='VACUOUS_EQUALS_ONE' if path==1 else 'NONVACUOUS_TV_BOUND_ONLY_NO_PRICE_THRESHOLD',
                    assumptions=['same event label denotes the same saved word/angle/phase','conditional order/component draws independent within each ideal/grid model',
                     'independent uniform 53-bit draws across event positions and restarted outer streams, or a justified bound uniform over matched histories'],
                    actual_PCG64_joint_distance=None,price_oscillation=None,price_shift=None,qualified_E_C=None))
            progress(log,table['id'],'END',seconds=time.perf_counter()-began)
        except BaseException:
            error=dict(id=table['id'],source=table['source'],reason=traceback.format_exc());errors.append(error);write(out/'STOP.json',error)
            progress(log,table['id'],'STOP',seconds=time.perf_counter()-began);break
    write(out/'ROUND_BOUNDS.json',dict(schema='LD1_F0_CONDITIONAL_BOUNDS_V1',rows=bounds,actual_PCG64_joint_distance=None,price_correction=None))
    axis=[dict(m=r['m'],axis=a,physical_id=r['physical_id'],round_selector=['rows',i],bounds_path=public_path(out/'ROUND_BOUNDS.json')) for i,r in enumerate(bounds) for a in ('X','Y')]
    write(out/'AXIS_MAP.json',dict(rows=axis,new_axis_specific_diagnostics=0))
    counts=dict(planned=len(bundle['tables']),reused=0,new_attempted=attempted,new_complete=len(results),anomalies=len(errors),withheld=len(bundle['tables'])-attempted,
                event_bounds=len(bounds),path_bounds=len(bounds),paths=0,builds=0,compiles=0)
    result=dict(schema='LD1_F0_RESULT_V1',status='F0_DIAGNOSTIC_COMPLETE' if not errors else 'STOP_PARTIAL',counts=counts,tables=results,errors=errors,
        elapsed_seconds=time.perf_counter()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        target_cost_law='ideal_finite_RTE_law',actual_PCG64_joint_distance=None,analytic_real_weight_error=None,price_oscillation=None,price_correction=None,
        E_S_recomputed=False,E_C=None,CI=None,price_prediction=None,fit=False,Q3=None,N=None,G=None,winner=None,mandatory_stop=True)
    write(out/'RESULT.json',result);return result


def execute(source,inputs):
    if git('rev-parse','HEAD')!=source or git('status','--porcelain','--untracked-files=no'):raise ValueError('COMMITTED_CLEAN_EXECUTION_SOURCE_REQUIRED')
    bundle=json.loads(Path(inputs).read_text())
    for ref in bundle['source_files']:
        if sha(blob(ref['commit'],ref['path']))!=ref['sha256']:raise ValueError('PINNED_SOURCE_BYTES_CHANGED')
    if bundle!=bind_inputs():raise ValueError('FIXED_INPUTS_CHANGED')
    helper,cdf=load_fixed_functions();out=ROOT/bundle['output']
    result=run_tables(bundle,out,helper,cdf)
    write(out/'EXECUTION.json',dict(execution_source=source,inputs=dict(commit=source,path=str(Path(inputs).relative_to(ROOT))),
        inputs_sha256=sha(Path(inputs).read_bytes()),helper=HELPER,cdf_helper=CDF,numpy_version=__import__('numpy').__version__,python_version=sys.version,
        outcome=result['status'],exit_code=0 if result['status']=='F0_DIAGNOSTIC_COMPLETE' else 2,
        parallelism='support-sized exact categorical operations; cached Git blobs; 14 small tables sequential, no process-pool overhead or event enumeration',mandatory_stop=True))
    print(json.dumps(dict(status=result['status'],counts=result['counts'],seconds=result['elapsed_seconds'])),flush=True)
    return 0 if result['status']=='F0_DIAGNOSTIC_COMPLETE' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',type=Path);parser.add_argument('--execute',action='store_true');parser.add_argument('--source');parser.add_argument('--inputs',type=Path)
    args=parser.parse_args()
    if args.prepare:write(args.prepare,bind_inputs());return 0
    if not(args.execute and args.source and args.inputs):parser.error('Use --prepare or --execute --source --inputs')
    return execute(args.source,args.inputs.resolve())

if __name__=='__main__':raise SystemExit(main())
