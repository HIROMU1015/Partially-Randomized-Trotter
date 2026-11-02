"""Dormant saved-cohort descriptor study. No fit/sampler/build/compiler path."""
from pathlib import Path
import importlib.metadata
import math
import subprocess,time
from . import c_df_basis_surrogate as features
from .c1_stage1_execution import read,sha,save
from .c1_stage3_integrated import Calls
from .c1_stage3_accuracy import METRICS

SCOPE='C_DF_BASIS_NATIVE_SAVED64_V1'
FRESH='artifacts/research_c_df_basis_surrogate/2026-10-29/run01'
LIMITS=dict(saved_bundle=2,identity_gate=2,static_features=2,trajectory_features=64,
    saved_raw_census=64,descriptive_models=12,surrogate_qualification=1,result_summary=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=read(path)
    for k,v in dict(scope=SCOPE,fresh_output=FRESH,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
            models={k:list(v) for k,v in features.MODELS.items()},folds='sample_index_mod4_within_each_cohort',
            DF_clusters=2,samples_per_cohort=32,event_count=128,retry=0,resume=0,new_DF_fit=0,
            sampling=0,new_build_compile=0).items():
        if c[k]!=v:raise ValueError('DESCRIPTOR_SCOPE:'+k)
    if set(c['cohorts'])!={'M4_Proposed','tail_P4'}:raise ValueError('TWO_FIXED_DF_COHORTS')
    for name,spec in c['cohorts'].items():
        if ([x['sample'] for x in spec['samples']]!=list(range(32))
                or spec['role']!=('proposed' if name=='M4_Proposed' else 'tail_p4')):raise ValueError('FIXED_COHORT_ROLE_AND32')
    if sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('DESCRIPTOR_SOURCE')
    for files in (read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in files.items():
            f=(root/p).resolve()
            if not f.is_relative_to(root) or sha(f)!=h:raise ValueError('DESCRIPTOR_INPUT:'+p)
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('DESCRIPTOR_ENV:'+p)
    if (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    return c


def load_bundle(root,spec):
    import numpy as np
    with np.load(root/spec['export'],allow_pickle=False) as x:U=x['U']
    return dict(U=U,tail=read(root/spec['tail']),SOURCE=read(root/spec['SOURCE']),STOP=read(root/spec['STOP']))


def identity(bundle,spec):
    if (bundle['SOURCE']['execution_commit']!=spec['execution_commit'] or bundle['STOP']['failed']
            or bundle['STOP']['scientific_units']!=spec['completed_calls']
            or bundle['STOP']['retry'] or bundle['STOP']['resume']):raise ValueError('SAVED_COHORT_IDENTITY')
    return dict(status='PASS',role=spec['role'],independent_DF_structures=1,paired=False)


def static(bundle):
    U=bundle['U'];atoms=features.atoms_from_records(bundle['tail']['atoms_catalog'])
    distances=features.geometry(atoms,U);law=features.law_features(atoms)
    if not math.isclose(law['lambda_R'],bundle['tail']['lambda_R'],rel_tol=1e-12,abs_tol=1e-14):raise ValueError('SAVED_LAW_MASS')
    q=law['q_frame'];T=127*math.fsum(q[l]*q[k]*v for (l,k),v in distances.items())
    record=dict(law=law,expected_overlap=T,head_F0_F1_overlap=features.overlap(U[0],U[1]),
        head_onebody_basis_descriptor=None,head_price=None,static_sample_count=1,
        caveat='two static DF vectors,never64 independent static observations')
    return atoms,distances,record


def trajectory(root,spec,row,atoms,distances):
    payload=read(root/row['trajectory']);metrics=read(root/row['metrics'])
    if (payload['sample']!=row['sample'] or payload['SOURCE_sha256']!=spec['sampling_SOURCE_sha256']
            or payload['final_JU_sha256']!=spec['export_sha256'] or payload['cohort']!=spec['cohort']
            or payload['microsteps']!=128 or payload['paired'] or payload['old_cohort_pooled']
            or payload['quantum_shots']):raise ValueError('SAMPLE_CORRESPONDENCE')
    if 'role' in payload and payload['role']!=spec['role']:raise ValueError('SAMPLE_ROLE')
    if any(type(metrics[k])is not int or metrics[k]<0 for k in METRICS):raise ValueError('SAVED_METRICS')
    return dict(sample=row['sample'],seed=payload['seed'],features=features.trajectory_features(payload,atoms,distances),
        metrics={k:metrics[k] for k in METRICS},paired=False)


def census(root,spec,row):
    from collections import Counter
    from qiskit import qpy
    with (root/row['raw_qpy']).open('rb') as f:circuits=qpy.load(f)
    if len(circuits)!=1 or circuits[0].num_qubits!=9:raise ValueError('RAW_QPY_SCHEMA')
    qc=circuits[0]
    if qc.metadata.get('arm')!=spec['role'] or qc.metadata.get('source')!=spec['export_sha256']:raise ValueError('RAW_QPY_PROVENANCE')
    return dict(opcode_counts=dict(Counter(x.operation.name for x in qc.data)),
        isolated_head_price=None,isolated_basis_prices=None,
        compiled_metrics_reused=True,new_circuit_build_compile=0)


def execute(root,c,out,call):
    cohorts={};static_records={};assessments={}
    for name,spec in c['cohorts'].items():
        d=out/name;d.mkdir()
        bundle=call('saved_bundle',name,lambda:load_bundle(root,spec))
        save(d/'identity.json',call('identity_gate',name,lambda:identity(bundle,spec)))
        atoms,distances,record=call('static_features',name,lambda:static(bundle));save(d/'static.json',record);static_records[name]=record
        rows=[]
        for row in spec['samples']:
            value=call('trajectory_features',name+str(row['sample']),lambda:trajectory(root,spec,row,atoms,distances))
            value['raw_census']=call('saved_raw_census',name+str(row['sample']),lambda:census(root,spec,row))
            save(d/f'sample{row["sample"]:02d}.json',value);rows.append(value)
        if len({x['seed'] for x in rows})!=32:raise ValueError('DUPLICATE_SEED')
        metrics={k:call('descriptive_models',name+':'+k,lambda k=k:features.describe_models(rows,k)) for k in METRICS}
        save(d/'models.json',metrics);cohorts[name]=rows;assessments[name]=metrics
    q=call('surrogate_qualification','primaryRZ withincohort only',lambda:features.qualification(assessments))
    value=call('result_summary','no new DF/price engine or scientific followup',lambda:dict(status='SAVED64_DESCRIPTOR_STUDY_COMPLETE',
        qualification=q,cohorts=assessments,static=static_records,DF_clusters=2,samples=64,
        paired=False,all_law_synthesis=None,cost_support=None,actual_CA_achieved=None,new_DF_fit=0,
        new_build_compile=0,retry=0,resume=0,mandatory_stop=True))
    save(out/'result.json',value);return value


def run(root,path):
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH;out.mkdir(parents=True)
    call=Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(out/'contract.json',c);save(out/'SOURCE.json',dict(read(root/c['source_manifest']),execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        return execute(root,c,out,call)
    except BaseException as error:
        call.stop.set();save(out/'failure.json',dict(error=str(error),first_failure=call.first_failure));raise
    finally:
        save(out/'STOP.json',dict(failed=call.stop.is_set(),scientific_units=sum(call.counts.values()),call_counts=dict(call.counts),retry=0,resume=0,mandatory_stop=True))
        save(out/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()))
