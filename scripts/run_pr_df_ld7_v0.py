"""Adopted LD7 stages 0--2; one configuration, thirteen means, one V0 Q3.

Numerics/identities are delegated to their unchanged published implementations.
Each CLI stage runs in a fresh process; no molecular price/build/sampler entry.
"""
import argparse
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
PUBLIC='docs/research/d_pr_df_ld7_v0_results_20261116'
OUT=ROOT/PUBLIC
CALLER='scripts/run_pr_df_ld7_v0.py'
SIGNAL='fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
VIEW=ROOT.parent/'d2-pr-df-ld7-signal-source-20261116'
VIEW_COMMIT='714247619afb4714fd663eb15c0470ef32bdc7af'
B='437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
IMPLEMENTATION=dict(commit=SIGNAL,path='src/trottertracks/pr_rpe_signal_provider.py',recipe='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1')
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
PHYS=dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
V0=[64,60,56,52,47,43,38,34,30,25,21,17,12]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)

def blob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def read(commit,path):return json.loads(blob(commit,path))
def local(name):return json.loads((OUT/name).read_text())
def head():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def exact(v):return Fraction(v['numerator'],v['denominator'])if isinstance(v,dict)else Fraction(v)
def fixed_caller(source):
    if blob(source,CALLER)!=(ROOT/CALLER).read_bytes():raise ValueError('Caller differs from execution SOURCE')
def stage_start(mode):
    print('START '+mode,flush=True);return time.perf_counter()
def resources(start):
    r=resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_seconds=time.perf_counter()-start,user_seconds=r.ru_utime,system_seconds=r.ru_stime,peak_RSS_KiB=r.ru_maxrss)

def signal_modules():
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=VIEW,text=True).strip()!=VIEW_COMMIT:raise ValueError('Signal view SOURCE differs')
    for path in ('src/trottertracks/pr_rpe_signal_provider.py','src/trottertracks/pr_rpe_signal_plan.py','src/trottertracks/pr_rpe_rmse_s0_v2.py'):
        if (VIEW/path).read_bytes()!=blob(SIGNAL,path):raise ValueError('Pinned signal/metadata bytes differ: '+path)
    sys.path.insert(0,str(VIEW/'src'))
    from trottertracks import pr_rpe_rmse_s0_v2 as fixed,pr_rpe_signal_plan as metadata,pr_rpe_signal_provider as provider
    node=next(n for n in ast.parse(blob(SIGNAL,'src/trottertracks/pr_rpe_rmse_s0_v2.py')).body if isinstance(n,ast.FunctionDef)and n.name=='b_metadata')
    scope=dict(vars(fixed),B_RELEASE=B)
    exec(compile(ast.Module(body=[node],type_ignores=[]),CALLER+':B_metadata','exec'),scope)
    return fixed,metadata,provider,scope['b_metadata'](ROOT)

def saved_constant(inputs,binding):
    # Reuse only the published constant-member reader, not the price runtime.
    source='e827367629ca6c2920d9d908fc1631314399db86'
    path='src/trottertracks/ld12_price_requests.py'
    node=next(n for n in ast.parse(blob(source,path)).body if isinstance(n,ast.FunctionDef)and n.name=='_constant_from_inputs')
    def real(x):
        if not math.isfinite(x):raise ValueError('Nonfinite scalar')
        return x
    scope=dict(Path=Path,hashlib=hashlib,json=json,io=io,deepcopy=deepcopy,D=SimpleNamespace(_real=real))
    exec(compile(ast.Module(body=[node],type_ignores=[]),path+'@'+source,'exec'),scope)
    return scope['_constant_from_inputs'](inputs,binding,dict(**INDEX,selector=['saved_inputs','H4','inputs']))

def inspect_schedule(config):
    rows=[]
    for r in config['rounds']:
        c=r['physical_candidate'];s=r['structure']
        ok=(c['prefix']==7 and c['method']=='PR'and c['K']==2 and c['formula']=='2nd'
            and c['q']==1<<r['m'] and c['delta']==.1 and exact(r['T'])==Fraction(.1)*(1<<r['m'])
            and r['R_outer']==c['R_outer']==1 and s['formula_id']=='S2'
            and s['signed_weights']==[1.]and s['allocation']==[1]and s['event_count']==c['q'])
        rows.append(dict(m=r['m'],physical_ID=c['id'],T=c['T'],T_exact=r['T'],q=c['q'],R=c['R_outer'],
            signed_weights=s['signed_weights'],allocation=s['allocation'],single_piece_gate=ok))
    passed=len(rows)==13 and [r['m']for r in rows]==list(range(13))and all(r['single_piece_gate']for r in rows)
    return dict(status='PASS'if passed else'STOP_MAPPER_CONDITION',rows=rows,gate_before_signals=True)

def generate():
    start=stage_start('configuration');p=local('GENERATION_INPUTS.json');fixed_caller(p['source'])
    d=local('DELTA_CHECK.json')
    if d['equivalent_results']or d['matching_jobs']or d['unresolved']:raise ValueError('Unreconciled equivalent work')
    out=OUT/'generation';out.mkdir()
    write(out/'EXECUTION.json',dict(source=p['source'],inputs_commit=head(),maximum_new_configurations=1))
    fixed,metadata,provider,b=signal_modules()
    s0=read(p['template']['commit'],p['template']['path']);index=read(INDEX['commit'],INDEX['path'])
    desc=index['T20_reference_files']['H4'];data=Path(desc['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=desc['sha256']:raise ValueError('Tail descriptor hash mismatch')
    context=json.loads(data)
    if context['binding']!=s0['inputs']['design']['binding']:raise ValueError('Tail descriptor binding differs')
    inputs=deepcopy(s0['inputs']);inputs['design']['prefix']=7
    inputs['design']['source']=dict(input_source=deepcopy(inputs['design']['source']),composition_recipe=IMPLEMENTATION,status='PROPOSED_PREFIX_COMPOSITION_NOT_SCIENCE_RESULT')
    tail_source=dict(index=INDEX,context_sha256=desc['sha256'],selector=['prefixes','7'])
    tail=context['prefixes']['7']
    inputs['tail_norm'].update(prefix=7,lambda_R=tail['lambda_R'],source=tail_source)
    if tail!=p['tail_descriptor']:raise ValueError('Adopted descriptor differs')
    constant=saved_constant(index['saved_inputs']['H4']['inputs'],context['binding'])
    total=constant['value']+tail['scalar']
    write(out/'SCALAR.json',dict(hamiltonian_constant=constant,tail_identity=dict(value=tail['scalar'],source=tail_source),
        total=total,total_hex=total.hex(),application='ONCE_AT_TOTAL_T',head=[0,1,2,3,4,5,6],tail=[7,8,9],one_body='SEPARATE_HEAD'))
    config=b.generate(b.a_spec(inputs,'PR'));b.validate(config)
    if str(exact(config['spec']['A_input']['kappa_paper']))!=p['kappa_exact']:raise ValueError('Common kappa differs')
    write(out/'CONFIGURATION.json',config)
    gate=inspect_schedule(config);write(out/'SCHEDULE_GATE.json',gate)
    if gate['status']!='PASS':return 2
    pending=metadata.plan(config,{r['m']:r['physical_candidate']for r in config['rounds']},b_validate=b.validate,b_exact_scalar=exact,numerical_recipe=provider.numerical_recipe(IMPLEMENTATION))
    physical=read(PHYS['commit'],PHYS['path'])
    for job in pending['jobs']:provider.saved_h4_reference(ROOT,job)
    write(out/'SIGNAL_PLAN.json',pending)
    write(out/'COMPLETION.json',dict(status='CONFIGURATION_AND_13_ROW_GATE_PASS',configuration_id=config['id'],new_configurations=1,signal_calls=0,resources=resources(start)))
    print(config['id']+' 13/13 R=1 single-piece PASS',flush=True)
    return 0

def signals():
    start=stage_start('signals');p=local('SIGNAL_INPUTS.json');fixed_caller(p['source'])
    config=read(p['configuration']['commit'],p['configuration']['path'])
    gate=read(p['configuration']['commit'],PUBLIC+'/generation/SCHEDULE_GATE.json')
    if gate!=inspect_schedule(config)or gate['status']!='PASS':raise ValueError('Pre-signal mapper gate differs')
    scalar=read(p['configuration']['commit'],PUBLIC+'/generation/SCALAR.json')
    fixed,metadata,provider,b=signal_modules();b.validate(config)
    out=OUT/'signals';out.mkdir()
    write(out/'EXECUTION.json',dict(source=p['source'],inputs_commit=head(),numerical_source=IMPLEMENTATION,configuration=p['configuration']))
    pending=metadata.plan(config,{r['m']:r['physical_candidate']for r in config['rounds']},b_validate=b.validate,b_exact_scalar=exact,numerical_recipe=provider.numerical_recipe(IMPLEMENTATION))
    write(out/'PLAN.json',pending);records=[];errors=[];timings=[]
    try:
        print('START saved H/state restore and cached head/macro preparation',flush=True)
        ctx=provider.load_h4_context(ROOT,config)
        if (ctx.constant!=scalar['hamiltonian_constant']['value']or ctx.tails[7]['scalar']!=scalar['tail_identity']['value']
                or (ctx.constant+ctx.tails[7]['scalar']).hex()!=scalar['total_hex']):raise ValueError('Loaded scalar components differ')
        print('END saved H/state restore',flush=True)
        for job in pending['jobs']:
            m=job['m'];t=time.perf_counter();print('START signal m'+str(m),flush=True)
            record=provider.evaluate(ctx,job,provider.saved_h4_reference(ROOT,job),root=ROOT)
            write(out/('m'+str(m)+'.json'),record);records.append(record)
            timings.append(dict(m=m,seconds=time.perf_counter()-t,status=record['status']))
            print('END signal m'+str(m)+' '+record['status']+' seconds='+str(timings[-1]['seconds']),flush=True)
            if record['status']=='NUMERICAL_FAILURE':break
    except Exception as exc:errors.append(dict(type=type(exc).__name__,reason=str(exc)))
    ok=len(records)==13 and not errors and all(r['status']=='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'for r in records)
    import numpy,scipy
    write(out/'RESULT.json',dict(status='13_MEANS_COMPLETE'if ok else'STOP_PARTIAL_NO_RETRY',records=records,errors=errors,timings=timings,
        new_signal_rounds=len(records),saved_reference_pairs=len(records),new_reference_actions=0,sampling=0,compile=0,
        not_started_rounds=list(range(len(records),13)),source=p['source'],numerical_source=IMPLEMENTATION,
        runtime=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__),resources=resources(start),automatic_retry=False))
    return 0 if ok else 2

def bundle(raw_commit,config_commit):
    """Copy original raw fields; authoritative existing A intake validates all."""
    config=read(config_commit,PUBLIC+'/generation/CONFIGURATION.json')
    old=read(VIEW_COMMIT,'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD2.json')
    packet={k:deepcopy(v)for k,v in old.items()if k!='means'}
    packet.update(configuration_source=dict(commit=config_commit,path=PUBLIC+'/generation/CONFIGURATION.json'),configuration_id=config['id'],raw_result_carrier=raw_commit,means=[])
    packet['target']['binding']=deepcopy(config['spec']['binding'])
    for row in config['rounds']:
        m=row['m'];loc=dict(commit=raw_commit,path=PUBLIC+'/signals/m'+str(m)+'.json');rec=read(loc['commit'],loc['path'])
        if rec['status']!='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED':raise ValueError('Invalid raw mean')
        s=rec['signal'];mean=dict(m=m,identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),source=loc,implementation_source=IMPLEMENTATION,
            **{k:deepcopy(s[k])for k in ('raw','corrected','reference','exact_tail')},
            normalization=dict(kind='FINITE_LOG_B_SUPPLIED_BY_PRODUCER',**{k:s[k]for k in ('log_B','B','attenuation')},log_B_margin=rec['allowance']['log_B_margin'],source=loc,normalizer_one_ulp_log_sensitivity=s['allowance_evidence']['normalizer_one_ulp_log_sensitivity']),
            bias={k:deepcopy(s[k])for k in ('total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias')},
            uncertainty=dict(empirical_allowance=rec['allowance'],numerical_observations=rec['numerical'],maximum_norm_domain=s['allowance_evidence']['maximum_norm_domain'],raw_rigorous_error_bound=None,ground_identification_error=None,actual_sampler_law_error=None,finite_K_RMSE_transfer_bound=None),
            target=deepcopy(packet['target']),qualification='UNQUALIFIED',numerical_gate_pass=True,reference_source=rec['reference_source'],state_evidence=rec['state_evidence'])
        packet['means'].append(mean)
    write(OUT/'B_MEANS_LD7.json',packet)

def q3_prepare(packet_commit,source):
    # Fresh process: Q3 SOURCE modules must not mix with the signal view.
    import run_pr_df_ld_split_q3_resume as R
    fixed_caller(source);env=R.runtime_check()
    packet=read(packet_commit,PUBLIC+'/B_MEANS_LD7.json');csrc=packet['configuration_source'];config=read(csrc['commit'],csrc['path'])
    binding=R.U.T.create(R.Q.VIEW,config,original_source=csrc)
    base=read('a82b0f0ec7f0f11d9981ddb2343fcd5378a87503','docs/research/pr_df_fullh_q3_fixed_results_20261115/INPUTS.json')
    case=dict(label='LD7_V0',configuration=csrc,signal_packet=dict(commit=packet_commit,path=PUBLIC+'/B_MEANS_LD7.json'),
        signal_packet_kind='A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED',original_configuration_id=config['id'],
        binding_id=binding['id'],total_configuration_id=binding['total_configuration']['id'],
        allocation=dict(allocation_id='PR_DF_LD7_V0',shot_policy='USER_SUPPLIED',shots=[dict(m=m,N_X=n,N_Y=n)for m,n in enumerate(V0)]),
        evaluation_target=deepcopy(base['cases'][0]['evaluation_target']))
    case['evaluation_target']['binding']=deepcopy(packet['target']['binding'])
    R.U.metadata(case)
    inputs={k:deepcopy(base[k])for k in ('statistical_contract','options','adopted_evaluation','acceleration')}
    inputs.update(cases=[case],maximum_scientific_calls=1,held=[],finite_scope_preflight=R.U.finite_bounds_from_saved_support(V0))
    write(OUT/'Q3_INPUTS.json',dict(source=source,inputs=inputs,runtime=env,output=PUBLIC+'/q3',unchanged_AST_guard=True,compile=0))
    print('SOURCE_BOUND_A_INTAKE_CONSTRUCTOR_COUNT_PREFLIGHT_PASS',flush=True)

def q3():
    import run_pr_df_ld_split_q3_resume as R
    start=stage_start('LD7 V0 Q3');package=local('Q3_INPUTS.json');fixed_caller(package['source'])
    if package['runtime']!=R.runtime_check():raise ValueError('Fixed Q3 runtime differs')
    inputs=package['inputs'];case=inputs['cases'][0];R.U.metadata(case)
    out=OUT/'q3';out.mkdir();(out/case['label']).mkdir()
    source=dict(commit=package['source'],path=CALLER,inputs=dict(commit=head(),path=PUBLIC+'/Q3_INPUTS.json'),runtime=package['runtime'])
    write(out/'EXECUTION.json',dict(source=source,maximum_scientific_calls=1))
    result=R.supervise(inputs['cases'],inputs,source,out)
    ok=(len(result['receipts'])==1 and not result['errors']and not result['monitor_errors']and result['receipts'][0]['scientific_evaluation_calls']==1
        and result['receipts'][0]['status']!='UNKNOWN'and all(p['exitcode']==0 for p in result['process_exitcodes']))
    write(out/'COMPLETION.json',dict(status='Q3_SAVED'if ok else'STOP_UNKNOWN_NO_RETRY',**result,resources=resources(start),mandatory_stop=True))
    return 0 if ok else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('generate','signals','bundle','q3_prepare','q3'));p.add_argument('--raw');p.add_argument('--configuration');p.add_argument('--packet');p.add_argument('--source');a=p.parse_args()
    if a.mode=='generate':raise SystemExit(generate())
    elif a.mode=='signals':raise SystemExit(signals())
    elif a.mode=='bundle':bundle(a.raw,a.configuration)
    elif a.mode=='q3_prepare':q3_prepare(a.packet,a.source)
    else:raise SystemExit(q3())
