"""Approved CT-4B: two saved q128/r1 paths, strict e=-11, no scope expansion."""
from __future__ import annotations
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
from importlib.metadata import version
import hashlib,io,json,lzma,math,os,platform,subprocess,sys,tarfile,time

ROOT=Path(__file__).resolve().parents[2]
BASE='69e2c0ae4e79febe54df59b6ea7704238467a968'
OUT=ROOT/'artifacts/pr_reference_ct4/2026-11-07'
Q=128;R=1;EXPONENT=-11;EPSILON=F(10)**EXPONENT
TASKS=[(rare,arm,axis) for rare in (False,True) for arm in ('Bprime','reference') for axis in 'XY']
# Per-process BLAS work follows actual independent task count, not an OS cap.
stage=sys.argv[1] if len(sys.argv)>1 else ''
task_count=8
if stage=='validate' and (OUT/'LOCAL_TARGETS.json').exists():
    task_count=len(json.loads((OUT/'LOCAL_TARGETS.json').read_text())['records'])
elif stage=='synthesis':task_count=len(os.sched_getaffinity(0))
workers=min(max(1,task_count),len(os.sched_getaffinity(0)))
blas_threads=max(1,len(os.sched_getaffinity(0))//workers)
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]=str(blas_threads)
import numpy as np
import mpmath as mp
from qiskit import QuantumCircuit,qpy
from .track_c_h4 import common,save_json
from .track_c_compiler_a import compile_reference,PROFILE
from .track_c_compiler_a_18round import I,upper,lower
from . import pr_reference_ct1 as ct1
from . import pr_reference_bprime as bp
from . import pr_reference_transfer_qualification as transfer
from . import pr_reference_interference_pilot as pilot
from . import pr_reference_rotation_v3 as v3
from . import track_c_h4_synthesis as synth
from .pr_reference_ct0 import validate_record

TR='artifacts/pr_reference_transfer_qualification/2026-11-06/'
BP='artifacts/pr_reference_bprime/2026-11-07/'
PILOT='artifacts/pr_reference_interference_pilot/2026-11-06/'
INPUTS={}

@lru_cache(None)
def data(path):
    raw=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
    INPUTS[path]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return raw

def read(path):return json.loads(data(path))

def archive(path):
    with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
        return {m.name:t.extractfile(m).read() for m in t if m.isfile()}

def attach():
    if (OUT/'STOP.json').exists():raise RuntimeError('CT-4 mandatory STOP')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')
    ct1.OUT=OUT;ct1.BASE=BASE;synth.OUT=OUT

@lru_cache(None)
def samples():
    raw=archive(TR+'raw_prices.tar.xz');result=[]
    hashes=['6bd2c4a9f5e861db301fd2df1cdc86dd646dd06fbd883c097191835b14c0a10e',
            '688f3c52622170692c0f4846f1d07c3c54c8b83aae5e341e03057f5271abbcf8']
    seeds=[14787950215150025216,9772911353196937292]
    for rare in (False,True):
        name=f'raw/q128_r1_rare{int(rare)}_0.json';row=json.loads(raw[name])
        assert (row['q'],row['r'],row['j'],row['rare'],row['seed'])==(Q,R,0,rare,seeds[int(rare)])
        digest=hashlib.sha256(json.dumps(row['events'],sort_keys=True).encode()).hexdigest()
        assert digest==hashes[int(rare)] and len(row['events'])==Q
        row['event_sha256']=digest;row['archive_member']=name;result.append(row)
    return result

@lru_cache(None)
def baseline_rows():
    rows=[json.loads(b) for b in archive(BP+'paired_Bprime_prices.tar.xz').values()]
    result={rare:next(x for x in rows if (x['q'],x['r'],x['j'],x['rare'])==(Q,R,0,rare)) for rare in (False,True)}
    for rare,row in result.items():assert row['events_sha256']==samples()[int(rare)]['event_sha256']
    return result

def histogram(native):
    return Counter(float(i.operation.params[0]).hex() for i in native.data if i.operation.name=='rz')

def canon_hist(h):
    out=Counter()
    for key,n in h.items():out['0x0.0p+0' if float.fromhex(key)==0 else key]+=n
    return out

def prepare_one(task):
    attach();rare,arm,axis=task;row=samples()[int(rare)]
    name=f'rare{int(rare)}_{arm}_{axis}';path=OUT/'native'/f'{name}.qpy'
    meta=OUT/'native_meta'/f'{name}.json';start=time.monotonic()
    if meta.exists():return json.loads(meta.read_text())
    reused_technical_partial=path.exists()
    if reused_technical_partial:
        with path.open('rb') as f:native=qpy.load(f)[0]
    else:
        if arm=='Bprime':
            body,_,_=bp.build_saved(row)
        else:
            cfg,dist,events=bp.reconstructed(row);a=transfer.assemble(Q,R,cfg,dist,events)
            assert a['phase']==row['phase'],(name,a['phase'],row['phase'])
            body=pilot.reference_core(a['F'],a['G'],a['phase'])
        native=compile_reference(pilot.wrapper(body,axis))
        if arm=='reference':
            full=QuantumCircuit(17,1);full.reset(range(8,16));full.compose(native,inplace=True);native=full
        with path.open('wb') as f:qpy.dump(native,f)
    expected=(baseline_rows()[rare]['Bprime_values'] if arm=='Bprime' else row['values']['reference'])[axis=='Y']
    actual=pilot.metric(native);assert actual==expected,(name,actual,expected)
    ops=dict(native.count_ops());assert set(ops)<=set(('rz','sx','x','cx','reset','measure','barrier'))
    resets=[j for j,i in enumerate(native.data) if i.operation.name=='reset']
    assert resets==(list(range(8)) if arm=='reference' else [])
    assert ops.get('measure',0)==1 and native.num_qubits==(17 if arm=='reference' else 9)
    h=histogram(native);assert sum(h.values())==actual[0]
    result=dict(name=name,rare=rare,arm=arm,axis=axis,q=Q,r=R,j=0,seed=row['seed'],event_sha256=row['event_sha256'],
        qubits=native.num_qubits,metric=actual,saved_metric=expected,operations=ops,
        global_phase_hex=float(native.global_phase).hex(),signed_RZ_occurrences=dict(h),
        native_qpy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passed=True,
        compile_calls=0 if reused_technical_partial else 1,reused_completed_QPY_after_technical_error=reused_technical_partial,
        wall_seconds=time.monotonic()-start)
    save_json(meta,result);return result

def local_targets(keys):
    # Deterministic saved catalogue selection, only keys already in eight wholes.
    candidates=ct1.local_specs();records=v3.records()
    for arm in ('Bprime','reference'):
        for name in ('block:0:half','block:0:full'):
            rec=records[arm][name]
            candidates.append(dict(name=arm+'_'+name,role='D_edge_or_fused_boundary',arm=arm,n=rec['n'],
                lowering=rec['lowering'],global_phase=0.,origin='saved native D context'))
        for name in (('cz',) if arm=='Bprime' else ('z',)):
            rec=records[arm][name]
            candidates.append(dict(name=arm+'_'+name,role='RTE_product',arm=arm,n=rec['n'],lowering=rec['lowering'],global_phase=0.,origin='saved native product'))
    batches=archive('artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz')
    batch=next(json.loads(v) for v in batches.values() if json.loads(v)['context']==[True,True])
    for name in ('block:0:half','block:0:full'):
        rec=next(x for x in batch['records'] if x['name']==name)
        candidates.append(dict(name='actual_SOURCE_inverse_'+name,role='D_actual_SOURCE_inverse',arm='reference',
            n=rec['native']['n'],lowering=rec['native']['lowering'],global_phase=0.,origin='saved V5 actual inverse D lowering'))
    for row in samples():
        angle=row['phase'];candidates.append(dict(name=f'vacuum_P_rare{int(row["rare"])}',role='vacuum_P',arm='reference',n=1,
            lowering=[['rz',[0],[angle]]],global_phase=0.,origin='fixed trajectory literal vacuum P'))
    result=[];excluded=[]
    for spec in candidates:
        local={abs(float(p[0])).hex() for op,qs,p in spec['lowering'] if op=='rz'}
        if local<=keys:result.append(spec)
        else:excluded.append(dict(name=spec['name'],reason='not a subset of fixed whole dependency closure',missing_keys=sorted(local-keys)))
    assert any(x['role']=='D_actual_SOURCE_inverse' for x in result)
    return result,excluded

def prepare():
    if (OUT/'PREDECLARED.json').exists():raise RuntimeError('Already fixed; no repeated native preparation')
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('native','native_meta','reused','synthesis'):(OUT/folder).mkdir(exist_ok=True)
    (OUT/'.gitignore').write_text('common.pkl\nnative/\nnative_meta/\nreused/\nsynthesis/\n__pycache__/\n')
    raw=lzma.decompress(data(PILOT+'common.pkl.xz'));cache=OUT/'common.pkl'
    if cache.exists():assert cache.read_bytes()==raw
    else:cache.write_bytes(raw)
    attach()
    if not (OUT/'FIXED_INPUTS.json').exists():
        save_json(OUT/'FIXED_INPUTS.json',dict(base=BASE,q=Q,r=R,exponent=EXPONENT,epsilon=str(EPSILON),tasks=TASKS,
            trajectories=samples(),common_pickle_sha256=hashlib.sha256(raw).hexdigest(),
            local_selection='CT1 saved templates plus D0 half/full and actual inverse, CZ/Z, saved vacuum; filter to whole key closure before synthesis',
            new_sampling=False,new_SCF_DF_ground_solve=False,new_native_compile_scope=8))
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),python=sys.version,executable=sys.executable,
        packages={p:version(p) for p in ('qiskit','pygridsynth','numpy','scipy','mpmath','numba')},
        affinity=len(os.sched_getaffinity(0)),memory=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        disk_free_bytes=__import__('shutil').disk_usage(OUT).free,profile=PROFILE,workers=workers,
        BLAS_threads_per_process=blas_threads,shared_settings_changed=False))
    (OUT/'PREPARE_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(prepare_one,TASKS))
    save_json(OUT/'NATIVE_INPUT_CHECKS.json',dict(rows=rows,new_native_compiles=sum(x['compile_calls'] for x in rows),passed=True))
    roles=defaultdict(set)
    for row in rows:
        for signed in row['signed_RZ_occurrences']:roles[abs(float.fromhex(signed)).hex()].add('whole:'+row['name'])
    keys=set(roles);specs,excluded=local_targets(keys)
    save_json(OUT/'LOCAL_TARGETS.json',dict(records=specs,excluded=excluded,extra_synthesis_keys=0))
    existing,legacy=ct1.records_at(EXPONENT)
    reused={k:existing[k] for k in keys if k in existing};missing=sorted(keys-set(reused))
    for v in reused.values():save_json(OUT/'reused'/f'{v["key"]}.json',v)
    prior=read('artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json')['rows'][7]
    assert (prior['q'],prior['r'])==(Q,R)
    assert all(EPSILON<F(v['uniform_gate_error_strict_upper']) for a in prior['arms'].values() for v in a['axis'].values())
    pre=dict(base=BASE,q=Q,r=R,exponent=EXPONENT,epsilon=str(EPSILON),axes=list('XY'),
        trajectories=[dict(q=Q,r=R,j=0,rare=x['rare'],seed=x['seed'],event_sha256=x['event_sha256'],events=x['events'],vacuum_phase_hex=x['phase'].hex()) for x in samples()],
        all_positive_keys=sorted(keys),roles={k:sorted(v) for k,v in roles.items()},existing_qualified_keys=sorted(reused),new_synthesis_keys=missing,
        excluded_legacy_record_keys=legacy,CT0_round_endpoints=prior,local_targets=[x['name'] for x in specs],
        actual_signed_occurrences={x['name']:x['signed_RZ_occurrences'] for x in rows},
        numerical_check_tolerance=1e-10,phase_fit=False,precision_optimization=False,all_support_price=False,all18_T=False,
        whole_state_simulation=False,whole_dependency_closure='only eight fixed saved q128 whole native circuits; local tests add no key')
    save_json(OUT/'PREDECLARED.json',pre)
    save_json(OUT/'INPUTS.json',dict(base=BASE,inputs=INPUTS,common_pickle_sha256=hashlib.sha256(raw).hexdigest()))
    print(json.dumps(dict(stage='prepare',native=8,keys=len(keys),reused=len(reused),missing=len(missing),local=len(specs))),flush=True)

def synthesis():
    attach();(OUT/'SYNTHESIS_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    ct1.stage_b()

def count_hist(h,records):
    t=td=0;bound=F()
    for signed,n in h.items():
        if not n:continue
        assert F(n).denominator==1
        n=int(n)
        theta=float.fromhex(signed);v=records[abs(theta).hex()]
        t+=n*v['T'];td+=n*(v['Tdagger'] if theta>=0 else v['T']-v['Tdagger'])
        bound+=n*F(v['strict_error_upper'])
    return dict(T=t,Tdagger=td,strict_joint_operator_upper=str(bound))

def role_account(row,records):
    original=samples()[int(row['rare'])];parts=v3.saved_path_parts(original,row['arm'],row['axis'])
    actual=canon_hist(row['signed_RZ_occurrences']);aggregate=Counter();details=[]
    for role,part in parts.items():
        h=Counter()
        for token,n in part.items():
            key=v3.TOKENS[token]['angle_key'];assert not key.startswith('symbolic:')
            h['0x0.0p+0' if float.fromhex(key)==0 else key]+=n
        aggregate.update(h)
        available=all(abs(float.fromhex(k)).hex() in records for k,n in h.items() if n)
        value=count_hist(h,records) if available else None
        details.append(dict(role=role,signed_accounting_correction=role=='D_boundary_fusion_difference',
            signed_RZ_count=str(sum(h.values())),account=value,keys_available=available))
    matched=all(aggregate[k]==actual[k] for k in set(aggregate)|set(actual))
    return dict(name=row['name'],projection_matches_literal_histogram=matched,roles=details if matched else [],
        status='QUALIFIED_SIGNED_ROLE_ACCOUNTING' if matched else 'UNQUALIFIED_ATTRIBUTION',
        mismatched_keys=[] if matched else [k for k in set(aggregate)|set(actual) if aggregate[k]!=actual[k]])

def validate():
    attach();pre=ct1.predeclared()
    if (OUT/'PILOT_T_COUNTS.json').exists():raise RuntimeError('Already accounted')
    (OUT/'VALIDATE_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    ct1.word_matrices();ct1.simulator_check()
    specs=json.loads((OUT/'LOCAL_TARGETS.json').read_text())['records']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        local=list(pool.map(ct1.local_one,specs))
    save_json(OUT/'LOCAL_CORRECTNESS.json',dict(rows=local,passed=True,whole_state_simulation=False,
        BLAS_threads_per_process=blas_threads,workers=workers))
    records=ct1.selected_records();assert set(records)==set(pre['all_positive_keys'])
    rows=json.loads((OUT/'NATIVE_INPUT_CHECKS.json').read_text())['rows'];priced=[];roles=[];errors=[]
    mp.iv.prec=192
    for row in rows:
        account=count_hist(row['signed_RZ_occurrences'],records)
        old=pre['CT0_round_endpoints']['arms'][row['arm']];N=old['saved_shots_per_axis'];assert N==199
        for kind,e in [('finite_trajectory',F(account['strict_joint_operator_upper'])),
                       ('conditional_all_support_uniform_precision',old['axis'][row['axis']]['worst_case_RZ_upper']*EPSILON)]:
            ds=2*mp.iv.sqrt(I(2))*(I(e.numerator)/e.denominator)
            d=I(old['d_old'])+ds;rho=I(old['rho0'])-d;ratio=d/I(old['rho0'])
            beta=I(old['B_lower'])-mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))
            need=4*mp.iv.log(I(1440))/(rho*rho*mp.iv.sin(beta)**2)
            assert lower(rho)>0 and lower(beta)>0 and upper(need)<=N,(row['name'],kind)
            errors.append(dict(name=row['name'],kind=kind,strict_joint_operator_upper=str(e),
                synthesis_raw_disk_upper=upper(ds),d_total_upper=upper(d),rho_lower=lower(rho),beta_stat_lower=lower(beta),
                required_continuous_N_upper=upper(need),saved_N=N,saved_N_sufficient=True))
        priced.append(dict(name=row['name'],rare=row['rare'],arm=row['arm'],axis=row['axis'],q=Q,r=R,
            qubits=row['qubits'],metric=row['metric'],account=account,physical_RZ_occurrences=sum(row['signed_RZ_occurrences'].values())))
        roles.append(role_account(row,records))
    save_json(OUT/'ROLE_T_COUNTS.json',dict(rows=roles,new_fusion=False))
    save_json(OUT/'ERROR_AND_SHOTS.json',dict(rows=errors,passed=True,alpha='1/720',phase_cap='.4',total_failure='.05',new_N_schedule=False,
        energy_accuracy_Ha=1.5936e-4,compiler_and_phase_disks_added_once=True,
        full_law_CT_RPE_success='NOT_QUALIFIED: finite pilot words only',all18_T='NOT_ACQUIRED'))
    save_json(OUT/'PILOT_T_COUNTS.json',dict(rows=priced,scope='finite saved two trajectories, X/Y; not expectation',
        all_support_price=False,all18_T=False,T_depth=False))
    print(json.dumps(dict(stage='validate',local=len(local),rows=priced)),flush=True)

def finish():
    attach()
    for name in ('KEY_REUSE_AND_SYNTHESIS','WORD_CORRECTNESS','TECHNICAL_TESTS','LOCAL_CORRECTNESS','ERROR_AND_SHOTS','PILOT_T_COUNTS'):
        assert (OUT/(name+'.json')).exists(),name
    bundles=[]
    for filename,folders in [('native_qpy.tar.xz',['native','native_meta']),('strict_records.tar.xz',['reused','synthesis'])]:
        with tarfile.open(OUT/filename,'w:xz') as t:
            for folder in folders:
                for p in sorted((OUT/folder).glob('*')):
                    if p.suffix in ('.qpy','.json'):t.add(p,arcname=folder+'/'+p.name)
        b=(OUT/filename).read_bytes();bundles.append(dict(path=filename,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    save_json(OUT/'BUNDLES.json',bundles)
    paths=[Path(__file__),ROOT/'src/trottertracks/pr_reference_ct1.py',ROOT/'src/trottertracks/pr_reference_ct0.py',
        ROOT/'src/trottertracks/pr_reference_bprime.py',ROOT/'src/trottertracks/pr_reference_transfer_qualification.py',
        ROOT/'src/trottertracks/pr_reference_rotation_v3.py',ROOT/'src/trottertracks/track_c_h4_synthesis.py',
        ROOT/'src/trottertracks/track_c2_b_verify.py',ROOT/'src/trottertracks/track_c_compiler_a.py']
    save_json(OUT/'SOURCE.json',dict(base=BASE,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        old_SOURCE_results_STOPs_preserved=True,new_native_compiles=8,new_sampling=0,new_Hamiltonian_or_ground_solves=0,
        whole_state_simulation=0,whole_expanded_CT_build=0,all18_accounting=0))
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='approved CT-4B finite scope complete',
        next_owner='GPT/user',automatic_expansion=False,all_support_T='UNQUALIFIED',all18_T='NOT_ACQUIRED'))

def main():
    commands=dict(prepare=prepare,synthesis=synthesis,validate=validate,finish=finish)
    try:commands[stage]()
    except (AssertionError,ArithmeticError) as exc:
        save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',reason='scientific mismatch or strict acquisition failure',stage=stage,error=repr(exc),automatic_retry=False))
        raise

if __name__=='__main__':main()
