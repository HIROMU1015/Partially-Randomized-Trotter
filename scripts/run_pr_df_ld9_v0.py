"""LD9: one configuration, up to13 fixed-provider means, at most one V0 Q3.

Delegate the unchanged B mapper/provider/worker; stop on first numerical gate.
No price, sampler, compiler, fit or allocation search entry point.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import run_pr_df_ld7_v0 as S

ROOT=S.ROOT
CALLER='scripts/run_pr_df_ld9_v0.py'
PUBLIC='docs/research/d_pr_df_ld9_v0_results_20261116'
OUT=ROOT/PUBLIC
REUSED='1cbc24cfc0fb0972f75cdef81db6bf62c51141e4'
LD7='0defa286c53a1b9acf513be453907bcad83102b3'
LD7DIR='docs/research/d_pr_df_ld7_v0_results_20261116'
read,write,blob,exact,head=S.read,S.write,S.blob,S.exact,S.head
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)

def fixed(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Unfixed execution SOURCE')
    if blob(REUSED,'scripts/run_pr_df_ld7_v0.py')!=(ROOT/'scripts/run_pr_df_ld7_v0.py').read_bytes():raise ValueError('Reused helper SOURCE changed')
def local(name):return json.loads((OUT/name).read_text())
def delta(name='DELTA_CHECK.json'):
    d=local(name)
    if d['equivalent_results']or d['matching_jobs']or d['unresolved']:raise ValueError('Unreconciled duplicate/unknown work')

def inspect(config):
    rows=[]
    for r in config['rounds']:
        c=r['physical_candidate'];st=r['structure'];R=c['R_outer']
        ok=(c['prefix']==9 and c['method']=='PR'and c['formula']=='2nd'and c['K']==2 and c['delta']==.1 and c['q']==1<<r['m']
            and exact(r['T'])==Fraction(.1)*(1<<r['m'])and type(R)is int and R>=1 and R==r['R_outer']
            and st['formula_id']=='S2'and st['signed_weights']==[1.]and st['allocation']==[R]and st['event_count']==c['q']*R)
        rows.append(dict(m=r['m'],T=c['T'],T_exact=r['T'],q=c['q'],R=R,event_count=c['q']*R,physical_ID=c['id'],allocation=st['allocation'],gate_pass=ok))
    return dict(status='PASS'if len(rows)==13 and [r['m']for r in rows]==list(range(13))and all(r['gate_pass']for r in rows)else'STOP_IDENTITY_OR_MAPPER',
        rows=rows,R_forced_to_LD7=False,gate_before_signals=True)

def generate():
    start=S.stage_start('LD9 configuration');p=local('GENERATION_INPUTS.json');fixed(p['source']);delta()
    out=OUT/'generation';out.mkdir();write(out/'EXECUTION.json',dict(source=p['source'],inputs_commit=head(),new_configuration_max=1))
    _,metadata,provider,b=S.signal_modules();template=read(p['template']['commit'],p['template']['path']);index=read(S.INDEX['commit'],S.INDEX['path'])
    desc=index['T20_reference_files']['H4'];data=Path(desc['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=desc['sha256']:raise ValueError('Descriptor bytes differ')
    ctx=json.loads(data)
    if ctx['binding']!=template['inputs']['design']['binding']or ctx['prefixes']['9']!=p['tail_descriptor']:raise ValueError('Saved descriptor/binding differs')
    inputs=deepcopy(template['inputs']);inputs['design']['prefix']=9
    inputs['design']['source']=dict(input_source=deepcopy(inputs['design']['source']),composition_recipe=S.IMPLEMENTATION,status='PROPOSED_PREFIX_COMPOSITION_NOT_SCIENCE_RESULT')
    tail_source=dict(index=S.INDEX,context_sha256=desc['sha256'],selector=['prefixes','9']);tail=ctx['prefixes']['9']
    inputs['tail_norm'].update(prefix=9,lambda_R=tail['lambda_R'],source=tail_source)
    c=S.saved_constant(index['saved_inputs']['H4']['inputs'],ctx['binding']);total=c['value']+tail['scalar']
    write(out/'SCALAR.json',dict(hamiltonian_constant=c,tail_identity=dict(value=tail['scalar'],source=tail_source),total=total,total_hex=total.hex(),application='ONCE_AT_TOTAL_T',head=list(range(9)),tail=[9],one_body='SEPARATE_HEAD'))
    config=b.generate(b.a_spec(inputs,'PR'));b.validate(config)
    if str(exact(config['spec']['A_input']['kappa_paper']))!=p['kappa_exact']:raise ValueError('Common kappa changed')
    write(out/'CONFIGURATION.json',config);gate=inspect(config);write(out/'SCHEDULE_GATE.json',gate)
    if gate['status']!='PASS':return 2
    old=read(LD7,LD7DIR+'/generation/SCHEDULE_GATE.json')['rows']
    write(out/'LD7_LD9_SCHEDULE.json',dict(LD7_source=dict(commit=LD7,path=LD7DIR+'/generation/SCHEDULE_GATE.json'),rows=[dict(m=r['m'],T=r['T'],q=r['q'],LD7_R=old[i]['R'],LD7_event_count=old[i]['q']*old[i]['R'],LD7_physical_ID=old[i]['physical_ID'],LD9_R=r['R'],LD9_event_count=r['event_count'],LD9_physical_ID=r['physical_ID'])for i,r in enumerate(gate['rows'])]))
    plan=metadata.plan(config,{r['m']:r['physical_candidate']for r in config['rounds']},b_validate=b.validate,b_exact_scalar=exact,numerical_recipe=provider.numerical_recipe(S.IMPLEMENTATION))
    for job in plan['jobs']:provider.saved_h4_reference(ROOT,job)
    write(out/'SIGNAL_PLAN.json',plan);write(out/'COMPLETION.json',dict(status='CONFIGURATION_AND_SCHEDULE_FIXED',configuration_id=config['id'],new_configurations=1,signal_calls=0,resources=S.resources(start)))
    print(config['id'],[(r['m'],r['q'],r['R'])for r in gate['rows']],flush=True);return 0

def signals():
    start=S.stage_start('LD9 signals');p=local('SIGNAL_INPUTS.json');fixed(p['source']);delta('PHYSICAL_DELTA_CHECK.json')
    config=read(p['configuration']['commit'],p['configuration']['path']);gate=read(p['configuration']['commit'],PUBLIC+'/generation/SCHEDULE_GATE.json');scalar=read(p['configuration']['commit'],PUBLIC+'/generation/SCALAR.json')
    if gate!=inspect(config)or gate['status']!='PASS':raise ValueError('Mapper/identity gate differs')
    _,metadata,provider,b=S.signal_modules();b.validate(config);out=OUT/'signals';out.mkdir()
    write(out/'EXECUTION.json',dict(source=p['source'],inputs_commit=head(),numerical_source=S.IMPLEMENTATION,configuration=p['configuration']))
    plan=metadata.plan(config,{r['m']:r['physical_candidate']for r in config['rounds']},b_validate=b.validate,b_exact_scalar=exact,numerical_recipe=provider.numerical_recipe(S.IMPLEMENTATION));write(out/'PLAN.json',plan)
    records=[];errors=[];timings=[]
    try:
        print('START saved H/state restore',flush=True);ctx=provider.load_h4_context(ROOT,config)
        if ctx.constant!=scalar['hamiltonian_constant']['value']or ctx.tails[9]['scalar']!=scalar['tail_identity']['value']or(ctx.constant+ctx.tails[9]['scalar']).hex()!=scalar['total_hex']:raise ValueError('Loaded scalar components differ')
        print('END saved H/state restore',flush=True)
        for job in plan['jobs']:
            m=job['m'];t=time.perf_counter();print('START signal m'+str(m),flush=True)
            r=provider.evaluate(ctx,job,provider.saved_h4_reference(ROOT,job),root=ROOT);write(out/('m'+str(m)+'.json'),r);records.append(r)
            timings.append(dict(m=m,seconds=time.perf_counter()-t,status=r['status']));print('END signal m'+str(m)+' '+r['status'],flush=True)
            if r['status']=='NUMERICAL_FAILURE':break
    except Exception as exc:errors.append(dict(type=type(exc).__name__,reason=str(exc)))
    ok=len(records)==13 and not errors and all(r['status']=='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'for r in records)
    failure=next((r for r in records if r['status']=='NUMERICAL_FAILURE'),None)
    if failure:
        n=failure['numerical'];bv=n.get('normalizer')
        write(out/'FIRST_GATE_FAILURE.json',dict(m=failure['m'],physical_id=failure['candidate']['id'],normalizer=bv,normalizer_hex=None if bv is None else float(bv).hex(),numerical=n,error=failure.get('error'),
            numerical_failure_reasons=failure.get('numerical_failure_reasons'),Q3_status='NOT_STARTED_SIGNAL_GATE_FAILURE',retry=False))
    import numpy,scipy
    write(out/'RESULT.json',dict(status='13_MEANS_COMPLETE'if ok else'STOP_PARTIAL_NO_RETRY',records=records,errors=errors,timings=timings,attempted_signal_rounds=len(records),valid_signal_rounds=sum(r['status']=='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'for r in records),not_started_rounds=list(range(len(records),13)),
        source=p['source'],numerical_source=S.IMPLEMENTATION,runtime=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__),resources=S.resources(start),new_reference_actions=0,sampling=0,compile=0,automatic_retry=False))
    return 0 if ok else 2

def bundle(raw,configuration):
    # Reuse the original field-preserving bundle adapter in an owned output scope.
    S.PUBLIC=PUBLIC;S.OUT=OUT;S.bundle(raw,configuration)
    (OUT/'B_MEANS_LD7.json').rename(OUT/'B_MEANS_LD9.json')

def q3_prepare(source,packet_commit):
    import run_pr_df_ld_split_q3_resume as W
    fixed(source);env=W.runtime_check();packet=read(packet_commit,PUBLIC+'/B_MEANS_LD9.json');csrc=packet['configuration_source'];config=read(csrc['commit'],csrc['path'])
    binding=W.U.T.create(W.Q.VIEW,config,original_source=csrc);base=read(LD7,LD7DIR+'/Q3_INPUTS.json')['inputs']
    case=dict(label='LD9_V0',configuration=csrc,signal_packet=dict(commit=packet_commit,path=PUBLIC+'/B_MEANS_LD9.json'),signal_packet_kind='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED',original_configuration_id=config['id'],binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'],
        allocation=dict(allocation_id='PR_DF_LD9_V0',shot_policy='USER_SUPPLIED',shots=deepcopy(base['cases'][0]['allocation']['shots'])),evaluation_target=deepcopy(base['cases'][0]['evaluation_target']))
    case['evaluation_target']['binding']=deepcopy(packet['target']['binding']);W.U.metadata(case)
    inputs={k:deepcopy(base[k])for k in ('statistical_contract','options','adopted_evaluation','acceleration')};inputs.update(cases=[case],held=[],finite_scope_preflight=W.U.finite_bounds_from_saved_support(S.V0))
    write(OUT/'Q3_INPUTS.json',dict(source=source,inputs=inputs,runtime=env,maximum_scientific_calls=1,compile=0));print('INTAKE_CONSTRUCTOR_COUNT_PASS',flush=True)

def q3():
    import run_pr_df_ld_split_q3_resume as W
    p=local('Q3_INPUTS.json');fixed(p['source']);delta('Q3_DELTA_CHECK.json')
    if p['runtime']!=W.runtime_check():raise ValueError('Q3 runtime differs')
    inputs=p['inputs'];case=inputs['cases'][0];W.U.metadata(case);out=OUT/'q3';out.mkdir();(out/case['label']).mkdir()
    source=dict(commit=p['source'],path=CALLER,inputs=dict(commit=head(),path=PUBLIC+'/Q3_INPUTS.json'),runtime=p['runtime']);write(out/'EXECUTION.json',dict(source=source,maximum_scientific_calls=1))
    result=W.supervise(inputs['cases'],inputs,source,out)
    ok=len(result['receipts'])==1 and not result['errors']and not result['monitor_errors']and result['receipts'][0]['scientific_evaluation_calls']==1 and result['receipts'][0]['status']!='UNKNOWN'and all(p['exitcode']==0 for p in result['process_exitcodes'])
    write(out/'COMPLETION.json',dict(status='Q3_SAVED'if ok else'STOP_UNKNOWN_NO_RETRY',**result,mandatory_stop=True));return 0 if ok else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('generate','signals','bundle','q3_prepare','q3'));p.add_argument('--source');p.add_argument('--raw');p.add_argument('--configuration');p.add_argument('--packet');a=p.parse_args()
    if a.mode=='generate':raise SystemExit(generate())
    elif a.mode=='signals':raise SystemExit(signals())
    elif a.mode=='bundle':bundle(a.raw,a.configuration)
    elif a.mode=='q3_prepare':q3_prepare(a.source,a.packet)
    else:raise SystemExit(q3())
