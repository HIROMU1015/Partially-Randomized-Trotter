"""Fixed H4/H6 theory execution, separate from historical grants and runners.

Only explicit approved manifests reach numerical imports. The original counter,
grouped expectation, Gaussian preparation and native provider are not modified.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
PANEL = {
    'H4': ('B1_p12_q1_2nd','B1_p12_q1_4th','B2_p3_q1_2nd_r4_K2','B2_p3_q8_2nd_r2_K4'),
    'H6': ('B1_p19_q8_4th','B2_p10_q8_2nd_r1_K2','B2_p15_q8_2nd_r2_K4'),
}


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m)
    return m


lib = load('_h46_lib', ROOT/'src/trotterlib/pr_theory_compiled.py')
m = lib.counter
grouped = load('_h46_grouped', ROOT/'src/trotterlib/native_df_grouped_expectation.py')
archive = load('_h46_archive', ROOT/'src/trotterlib/native_df_archive_metadata.py')


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def verify(plan, grant):
    source = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if not (plan['sealed'] is True and grant['approved'] is True and
            grant['runtime_authorization'] is True and grant['status']=='APPROVED' and
            grant['one_shot'] is True and grant['automatic_retry'] is False and
            plan['mandatory_stop'] is True):
        raise ValueError('EXPLICIT_ONE_SHOT_APPROVAL_REQUIRED')
    if source != plan['SOURCE'] or source != grant['SOURCE'] or grant['plan_digest'] != lib.digest(plan):
        raise ValueError('SOURCE_PLAN_GRANT_BINDING')
    if plan['stage'] not in (2,3,4) or grant['stage'] != plan['stage']:
        raise ValueError('FIXED_STAGE_REQUIRED')
    expected=PANEL['H4' if plan['stage']==2 else 'H6'] if plan['stage']!=4 else PANEL['H4']+PANEL['H6']
    if tuple(c['id'] for c in plan['candidates'])!=expected:
        raise ValueError('FIXED_SEVEN_PANEL_REQUIRED')
    for name, sha in plan['source_hashes'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or lib.file_hash(p)!=sha:
            raise ValueError('SOURCE_CLOSURE:'+name)
    provider=Path(plan['provider'])
    for name, sha in plan['provider_hashes'].items():
        if lib.file_hash(provider/name)!=sha:
            raise ValueError('PROVIDER_CLOSURE:'+name)
    for item in plan['inputs'].values():
        p=Path(item['path'])
        if p.stat().st_size!=item['bytes'] or lib.file_hash(p)!=item['sha256']:
            raise ValueError('INPUT_BYTE_BINDING:'+str(p))
    for item in plan.get('environment_files',[]):
        if lib.file_hash(item['path'])!=item['sha256']:
            raise ValueError('RUNTIME_FILE_BINDING')
    output=Path(plan['output']).resolve()
    if not output.is_relative_to('/home/AbeHiromu'):
        raise ValueError('HOME_OUTPUT_REQUIRED')
    return output


class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cupy','cupyx'):
            raise RuntimeError('GPU_IMPORT_FORBIDDEN')


def provider_modules(plan):
    sys.meta_path.insert(0,NoGPU())
    base=Path(plan['provider'])
    for name,relative in [('trotterlib','src/trotterlib'),('trotterlib.df_trotter','src/trotterlib/df_trotter'),
                          ('trottertracks','src/trottertracks'),('trottertracks.resource_applicability','src/trottertracks/resource_applicability')]:
        if name in sys.modules: raise ValueError('UNCLEAN_PROVIDER_NAMESPACE')
        mod=types.ModuleType(name);mod.__path__=[str(base/relative)];sys.modules[name]=mod
    actual={name:importlib.metadata.version(name) for name in plan['versions']}
    if actual!=plan['versions']:raise ValueError('VERSION_PROFILE')
    def forbidden(*a,**kw):raise RuntimeError('SAMPLING_OR_MOLECULAR_REGENERATION_FORBIDDEN')
    rte=importlib.import_module('trotterlib.rte')
    for name in ('iter_sample_rte_events','sample_rte_events'):
        setattr(rte,name,forbidden)
    return rte,importlib.import_module('trotterlib.df_trotter.ops')


def from_live(p):
    blocks=[];bases={}
    def basis(bid,bh,runtime):
        key=bid+':'+bh
        if key not in bases:
            value=m.basis_from_runtime(bid,bh,runtime)
            bases[key]=value
        return bases[key]
    for b in p.deterministic_blocks:
        one=b.original_fragment_index is None
        blocks.append(dict(one=one,basis=basis(b.basis_id,b.basis_hash,b.runtime_basis_operations),
                           diagonal=tuple(b.diagonal_eigenvalues if one else b.diagonal_eta),
                           lam=None if one else b.lam))
    components=[]
    for c in p.rte_preparation.symbolic_tail.components:
        components.append(dict(component_id=c.component_id,probability=c.probability,
            coefficient_sign=c.coefficient_sign,is_identity=c.is_identity,
            support=tuple(c.diagonal_pauli_support or ()),
            basis=None if c.is_identity else basis(c.basis_id,c.basis_hash,
                p.rte_preparation.basis_registry.operations(c.basis_id))))
    return dict(n=p.num_system_qubits,preparation_hash=p.preparation_hash,
        hamiltonian_hash=p.hamiltonian_hash,partition_hash=p.partition_hash,
        deterministic_indices=list(p.deterministic_fragment_indices),randomized_indices=list(p.randomized_block_indices),
        constant=p.constant_coefficient,extracted=p.extracted_identity_coefficient,
        lambda_r=p.exact_rte_lambda_r,blocks=blocks,components=components)


def from_graph(graph, ref):
    f=graph.fields(ref,'DFPartialS2Preparation');n=graph.scalar(f['num_system_qubits'])
    bases={};blocks=[]
    def basis(bid,bh,ops,runtime=None):
        key=(bid,bh)
        if key not in bases:
            desc=graph.basis_description(ops,n=n,basis_id=bid,basis_hash=bh)
            if runtime is not None:graph.compare_runtime_descriptions(runtime,desc)
            bases[key]=m.basis_from_runtime(bid,bh,[(types.SimpleNamespace(name=x['name'],params=x['parameters']),tuple(x['qubits'])) for x in desc])
        return bases[key]
    for br in graph.sequence(f['deterministic_blocks']):
        node=graph.node(br);b=node['fields'];one=node['type']=='DFDeterministicOneBodySpec'
        blocks.append(dict(one=one,basis=basis(b['basis_id'],b['basis_hash'],b['basis_change_operations'],b['runtime_basis_operations']),
                           diagonal=tuple(graph.scalar(b['diagonal_eigenvalues'] if one else b['diagonal_eta'])),
                           lam=None if one else graph.scalar(b['lam'])))
    tail=graph.fields(graph.fields(f['rte_preparation'],'DFRTEEventPreparation')['symbolic_tail'],'SymbolicRTETail')
    components=[]
    for cr in graph.sequence(tail['components']):
        c=graph.fields(cr,'RTEComponent')
        row={k:graph.scalar(c[k]) for k in ('component_id','probability','coefficient_sign','is_identity')}
        row['support']=tuple(graph.scalar(c['diagonal_pauli_support']) or ())
        row['basis']=None if row['is_identity'] else basis(c['basis_id'],c['basis_hash'],c['basis_change_operations'])
        components.append(row)
    return dict(n=n,preparation_hash=f['preparation_hash'],hamiltonian_hash=f['hamiltonian_hash'],partition_hash=f['partition_hash'],
        deterministic_indices=graph.scalar(f['deterministic_fragment_indices']),randomized_indices=graph.scalar(f['randomized_block_indices']),
        constant=graph.scalar(f['constant_coefficient']),extracted=graph.scalar(f['extracted_identity_coefficient']),
        lambda_r=graph.scalar(f['exact_rte_lambda_r']),blocks=blocks,components=components)


def describe(p):
    bases={}
    def ref(b):
        if b is None:return None
        key=b.basis_id+':'+b.source_hash
        bases[key]=dict(basis_id=b.basis_id,source_hash=b.source_hash,
                       operations=[dict(name=g.name,qubits=g.qubits,angle_hex=float(g.angle).hex(),beta_hex=float(g.beta).hex()) for g in b.operations])
        return key
    result={k:v for k,v in p.items() if k not in ('blocks','components')}
    result['blocks']=[dict(one=b['one'],basis=ref(b['basis']),diagonal_hex=[float(x).hex() for x in b['diagonal']],lambda_hex=None if b['lam'] is None else float(b['lam']).hex()) for b in p['blocks']]
    result['components']=[dict(component_id=c['component_id'],probability_hex=float(c['probability']).hex(),coefficient_sign=c['coefficient_sign'],is_identity=c['is_identity'],support=c['support'],basis=ref(c['basis'])) for c in p['components']]
    result['bases']=bases
    return result


def factory(p, ops):
    import numpy as np
    def primitive(j,t):
        b=p['blocks'][j]
        return (ops.one_body_diagonal_primitives(np.asarray(b['diagonal']),t) if b['one'] else
                ops.df_squared_diagonal_primitives(np.asarray(b['diagonal']),b['lam'],t))
    return primitive


def law(p,c,rte):
    tau=p['lambda_r']*(.8/c['q'])/c['r']
    dist=rte.finite_rte_distribution(tau,c['K'])
    z=sum(F(c['probability']) for c in p['components'])
    components=tuple(m.Component(x['component_id'],F(x['probability'])/z,x['basis'],x['support'],x['coefficient_sign']) for x in p['components'])
    weights={k:F(v) for k,v in zip(dist.orders,dist.order_probabilities)};w=sum(weights.values())
    orders={k:v/w for k,v in weights.items()}
    return components,orders,tau,dict(native_distribution=dist.to_dict(),component_sum=str(z),order_sum=str(w),
        tau_hex=tau.hex(),lambda_r_hex=float(p['lambda_r']).hex(),all_positive_orders_included=True,
        numerical_law_policy='EXACT_NORMALIZATION_OF_NATIVE_BINARY64_PROBABILITIES')


def packed_cost(cost):return {k:str(F(v)) for k,v in cost.as_dict().items()}


def check_saved_law(receipt, saved):
    if receipt['native_distribution']!=saved['native_distribution'] or receipt['tau_hex']!=saved['tau_hex'] or receipt['lambda_r_hex']!=saved['lambda_r_hex']:
        raise ValueError('PINNED_NATIVE_LAW_MISMATCH')


def evaluate(p,c,models,ops,rte):
    random=c['method']=='B2';rows=[]
    scalar=p['constant']+(p['extracted'] if random else 0.)
    primitive=factory(p,ops);bases=tuple(b['basis'] for b in p['blocks'])
    laws=None;tails={}
    if random:
        components,orders,tau,laws=law(p,c,rte)
    for model in models:
        pm='A' if model=='NATIVE_MIXED' else model
        tm='B' if model=='NATIVE_MIXED' else model
        if random:
            if tm not in tails:tails[tm]=grouped.expected_rte_grouped(m,components,orders,c['r'],tau,p['n'],tm)
            circuit=m.partial_prefix_circuit(bases,primitive,.8,c['q'],p['n'],'symmetric_directional',-scalar*.8,pm)
        else:
            circuit=m.deterministic_circuit(bases,primitive,.8,c['q'],c['formula'],'symmetric_directional',p['n'],-scalar*.8,pm)
        for axis,quadaxis in [('cosine','X'),('sine','Y')]:
            if random:
                cost=m.expected_partial_trajectory(tails[tm],c['q'],circuit,p['n'],quadaxis);lo,hi=cost.lower,cost.upper
            else:lo=hi=m.count(m.hadamard_wrapper(circuit,p['n'],quadaxis))
            rows.append(dict(candidate_id=c['id'],model=model,axis=axis,lower=packed_cost(lo),upper=packed_cost(hi),
                theory_kind='LAW_EXPECTATION' if random else 'DETERMINISTIC',theory_profile=m.PROFILE,
                preparation_hash=p['preparation_hash'],all_positive_orders_included=random,control='symmetric_directional'))
    return rows,laws


def journal(output, kind, **extra):
    with (output/'journal.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,**extra),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def theory_stage(plan, output):
    rte,ops=provider_modules(plan)
    preparations={};live={};input=plan['inputs']
    if plan['stage']==2:
        loader=importlib.import_module('trottertracks.resource_applicability.h4_common_panel_port_v1')
        ham,_,_,_,metadata,_=loader.h4_loader(Path(input['snapshot']['path']),read(input['snapshot_receipt']['path']),read(input['df_receipt']['path']))
        expected=read(input['prepared_summary']['path'])['preparations']
        prep_api=importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')
        for prefix in (3,12):
            journal(output,'prepare_saved_hamiltonian',prefix=prefix)
            p=prep_api._prepare(ham,prefix);record=from_live(p)
            old={x['preparation_hash'] for x in expected if x['cell_id'].split('_')[1]=='p'+str(prefix)}
            record['old_preparation_hashes']=sorted(old)
            record['old_preparation_hash_match']=old=={p.preparation_hash}
            # A different floating basis remains a new cohort; never relabel it.
            preparations[prefix]=record;live['faithful:p'+str(prefix)]=p
        codec=importlib.import_module('trottertracks.resource_applicability.h6_frozen_preparation_v1')
        write(output/'h4_preparations_codec.json',codec.encode_preparations(live))
        write(output/'snapshot_identity.json',metadata)
    else:
        graph=archive.FrozenMetadataGraph.from_file(input['archive']['path'],expected_sha256=input['archive']['sha256'])
        for prefix in (10,15,19):
            preparations[prefix]=from_graph(graph,graph.archive['codec']['roots']['faithful:p'+str(prefix)])
    write(output/'preparation_descriptions.json',{str(k):describe(p) for k,p in preparations.items()})
    results=[];laws=[]
    old_laws=read(input['old_laws']['path']) if plan['stage']==3 else []
    old_theory=read(input['old_theory']['path']) if plan['stage']==3 else []
    for c in plan['candidates']:
        journal(output,'theory_candidate_reserved',candidate_id=c['id'],models=plan['models'])
        print(json.dumps(dict(phase='THEORY',candidate=c['id'])),flush=True)
        rows,law_receipt=evaluate(preparations[c['prefix']],c,plan['models'],ops,rte)
        if plan['stage']==3:
            selected=[r for r in old_theory if r['system']=='H6' and r['candidate_id']==c['id']]
            if len(selected)!=4 or {r['preparation_hash'] for r in selected}!={preparations[c['prefix']]['preparation_hash']}:
                raise ValueError('PINNED_THEORY_PREPARATION_MISMATCH')
            if law_receipt is not None:
                previous=[r for r in old_laws if r['system']=='H6' and r['candidate_id']==c['id']]
                if len(previous)!=1:raise ValueError('UNIQUE_SAVED_LAW_REQUIRED')
                check_saved_law(law_receipt,previous[0])
        for row in rows:row['system']=plan['system']
        results.extend(rows)
        if law_receipt is not None:laws.append(dict(candidate_id=c['id'],**law_receipt))
    write(output/'theory.json',results);write(output/'laws.json',laws)
    if plan['stage']==3:
        write(output/'uniform_saved_reuse.json',[r for r in old_theory if r['system']=='H6' and r['candidate_id'] in PANEL['H6']])
    return dict(theory_wrapper_rows=len(results),candidates=len(plan['candidates']),
                new_compile=0,new_sampling=0,new_signal=0,new_SCF_DF_state=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--grant',required=True)
    parser.add_argument('--worker',type=int)
    a=parser.parse_args();plan=read(a.plan);grant=read(a.grant);output=verify(plan,grant)
    if a.worker is not None:
        if plan['stage']!=4 or not 0<=a.worker<7:raise ValueError('FIXED_SUBORDINATE_WORKER')
        parent=read(output/'parent_identity.json')
        if parent['pid']!=os.getppid() or parent['starttime']!=Path('/proc/'+str(os.getppid())+'/stat').read_text().rsplit(')',1)[1].split()[19]:
            raise ValueError('OWN_PARENT_IDENTITY')
        same=load('_h46_same',ROOT/'scripts/h46_same_event.py')
        same.worker(sys.modules[__name__],plan,a.worker,output)
        return
    output.mkdir(exist_ok=False)
    start=time.monotonic();terminal=dict(status='MANDATORY_STOP_FAILURE',first_stop=None,SOURCE=plan['SOURCE'],
        stage=plan['stage'],plan_digest=lib.digest(plan),grant_digest=lib.digest(grant),mandatory_stop=True,automatic_retry=False)
    try:
        if plan['stage']==4:
            same=load('_h46_same',ROOT/'scripts/h46_same_event.py')
            terminal.update(same.coordinator(sys.modules[__name__],plan,output,a.plan,a.grant))
        else:terminal.update(theory_stage(plan,output))
        for item in plan['inputs'].values():
            if lib.file_hash(item['path'])!=item['sha256']:raise ValueError('INPUT_CHANGED_DURING_RUN')
        # Certify every provider import, rather than trusting a checkout name.
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and name.startswith(('trotterlib.','trottertracks.')):
                relative=Path(path).resolve().relative_to(Path(plan['provider']).resolve()).as_posix()
                if plan['provider_hashes'].get(relative)!=lib.file_hash(path):raise ValueError('IMPORTED_PROVIDER_CLOSURE')
        terminal['status']='COMPLETE_MANDATORY_STOP'
    except BaseException as exc:
        terminal['first_stop']=repr(exc);raise
    finally:
        terminal.update(wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(output/'terminal.json',terminal)


if __name__=='__main__':main()
