"""Authorized metadata completion and theoretical counts; never sample/compile.

The private, hash-bound manifest supplies fixed provider/input/metadata paths.
The existing theory model is loaded by path independently of the science package.
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
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]


def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return sha(canonical(value))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def read(path):return json.loads(Path(path).read_text())
def fraction(value):
    f=F(value)
    return dict(numerator=str(f.numerator),denominator=str(f.denominator),decimal=float(f))
def costs(value):return {k:fraction(v) for k,v in value.as_dict().items()}
def forbid(*args,**kwargs):raise RuntimeError('SAMPLING_COMPILE_OR_FULL_NATIVE_BUILD_FORBIDDEN')


class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('cupy','cupyx'):
            raise RuntimeError('GPU_IMPORT_FORBIDDEN')


def namespace(name,path):
    if name in sys.modules:raise RuntimeError('UNCLEAN_SCIENCE_NAMESPACE')
    module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module


def execute(manifest,output):
    if (manifest.get('approved') is not True or manifest.get('runtime_authorization') is not True or
            manifest.get('scope')!='H6_H8_NATIVE_DF_THEORY_EVALUATION_V1'):
        raise RuntimeError('EXPLICIT_SCOPE_AUTHORIZATION_REQUIRED')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=manifest['SOURCE']:
        raise RuntimeError('SOURCE_HEAD_MISMATCH')
    for name,expected in manifest['source_hashes'].items():
        if sha((ROOT/name).read_bytes())!=expected:raise RuntimeError('SOURCE_BYTES_MISMATCH:'+name)
    for item in manifest['inputs'].values():
        h=hashlib.sha256()
        with Path(item['path']).open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        if h.hexdigest()!=item['sha256'] or Path(item['path']).stat().st_size!=item['bytes']:
            raise RuntimeError('INPUT_BYTES_MISMATCH')
    output=output.resolve()
    if not output.is_relative_to('/home/AbeHiromu'):raise RuntimeError('HOME_OUTPUT_REQUIRED')
    output.mkdir(exist_ok=False)
    emitted=0
    def write(name,value):
        nonlocal emitted
        data=canonical(value)+b'\n';emitted+=len(data)
        if emitted>manifest['limits']['output_bytes']:raise RuntimeError('OUTPUT_BUDGET')
        with (output/name).open('xb') as f:f.write(data)
    started=time.monotonic()
    def progress(phase,**extra):
        data=dict(phase=phase,elapsed_seconds=time.monotonic()-started,**extra)
        print(json.dumps(data),flush=True)
    m=load('_evaluation_counter',ROOT/'src/trotterlib/native_df_gate_model.py')
    grouped=load('_evaluation_grouped',ROOT/'src/trotterlib/native_df_grouped_expectation.py')
    graphlib=load('_evaluation_archive',ROOT/'src/trotterlib/native_df_archive_metadata.py')
    sys.meta_path.insert(0,NoGPU())
    provider=Path(manifest['provider_checkout'])
    for name,expected in manifest['provider_hashes'].items():
        if sha((provider/name).read_bytes())!=expected:raise RuntimeError('PROVIDER_SOURCE_MISMATCH:'+name)
    namespace('trotterlib',provider/'src/trotterlib')
    namespace('trotterlib.df_trotter',provider/'src/trotterlib/df_trotter')
    namespace('trottertracks',provider/'src/trottertracks')
    namespace('trottertracks.resource_applicability',provider/'src/trottertracks/resource_applicability')
    progress('IMPORT_FIXED_PROVIDER')
    prep_api=importlib.import_module('trotterlib.pr2_matched_accuracy_m1_execution')
    ops=importlib.import_module('trotterlib.df_trotter.ops')
    rte=importlib.import_module('trotterlib.rte')
    loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
    import numpy as np
    import qiskit
    qiskit.transpile=forbid
    for name in ('iter_sample_rte_events','sample_rte_events'):setattr(rte,name,forbid)
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    for name in ('build_deterministic_native','build_partial_native','partial_native_from_step_requests','build_native_hadamard_wrapper'):
        setattr(native,name,forbid)
    versions={name:importlib.metadata.version(name) for name in manifest['versions']}
    if versions!=manifest['versions']:raise RuntimeError('ENVIRONMENT_VERSION_MISMATCH')
    if len(list(Path('/proc/self/task').iterdir()))!=1:raise RuntimeError('INTERNAL_THREADS_NOT_ONE')
    basis_cache={}
    def basis(description,n,bid,bh):
        key=(n,bid,bh)
        if key in basis_cache:return basis_cache[key]
        if digest(dict(num_system_qubits=n,operations=description,hash_policy='ordered_local_basis_operations_v1'))!=bh:
            raise RuntimeError('BASIS_DESCRIPTION_HASH_MISMATCH')
        runtime=[(types.SimpleNamespace(name=x['name'],params=x['parameters']),tuple(x['qubits'])) for x in description]
        value=m.basis_from_runtime(bid,bh,runtime)
        basis_cache[key]=value
        return value
    preps={'H6':{},'H8':{}}
    archive_item=manifest['inputs']['h6_archive']
    graph=graphlib.FrozenMetadataGraph.from_file(archive_item['path'],expected_sha256=archive_item['sha256'])
    def from_graph(ref):
        f=graph.fields(ref,'DFPartialS2Preparation');n=graph.scalar(f['num_system_qubits']);blocks=[]
        for br in graph.sequence(f['deterministic_blocks']):
            b=graph.node(br);fields=b['fields'];description=graph.basis_description(fields['basis_change_operations'],n=n,basis_id=fields['basis_id'],basis_hash=fields['basis_hash'])
            graph.compare_runtime_descriptions(fields['runtime_basis_operations'],description)
            one=b['type']=='DFDeterministicOneBodySpec'
            blocks.append(dict(one=one,basis=basis(description,n,fields['basis_id'],fields['basis_hash']),
                diagonal=tuple(graph.scalar(fields['diagonal_eigenvalues'] if one else fields['diagonal_eta'])),
                lam=None if one else graph.scalar(fields['lam'])))
        tail=graph.fields(graph.fields(f['rte_preparation'],'DFRTEEventPreparation')['symbolic_tail'],'SymbolicRTETail')
        components=[]
        for cr in graph.sequence(tail['components']):
            c=graph.fields(cr,'RTEComponent');row={k:graph.scalar(c[k]) for k in ('component_id','probability','coefficient_sign','basis_id','basis_hash','diagonal_pauli_support','is_identity')}
            row['basis']=None if row['is_identity'] else basis(graph.basis_description(c['basis_change_operations'],n=n,basis_id=row['basis_id'],basis_hash=row['basis_hash']),n,row['basis_id'],row['basis_hash'])
            components.append(row)
        return dict(n=n,blocks=blocks,components=components,preparation_hash=f['preparation_hash'],
            constant=graph.scalar(f['constant_coefficient']),extracted=graph.scalar(f['extracted_identity_coefficient']),
            lambda_r=graph.scalar(f['exact_rte_lambda_r']))
    progress('H6_EXISTING_ARCHIVE')
    for key,ref in graph.archive['codec']['roots'].items():preps['H6'][key]=from_graph(ref)
    progress('H8_FIXED_SNAPSHOT_LOAD')
    inputs=manifest['inputs']
    ham,sector,full,state,metadata,_=loader.load_h8_snapshot(Path(inputs['h8_snapshot']['path']),read(inputs['h8_snapshot_receipt']['path']),read(inputs['h8_df_receipt']['path']))
    if metadata['hamiltonian_hash']!=manifest['H8_binding']['hamiltonian'] or metadata['state_hash']!=manifest['H8_binding']['state']:
        raise RuntimeError('H8_SCIENTIFIC_IDENTITY_MISMATCH')
    raw_h8={}
    for key,expected in sorted(manifest['H8_expected_preparation_hashes'].items()):
        kind,prefix=key.split(':p');prefix=int(prefix)
        progress('H8_PREPARATION',root=key)
        p=(prep_api._prepare_discard if kind=='discard' else prep_api._prepare)(ham,prefix)
        if p.preparation_hash!=expected:raise RuntimeError('H8_PREPARATION_HASH_MISMATCH:'+key)
        blocks=[];raw_blocks=[]
        for b in p.deterministic_blocks:
            one=b.original_fragment_index is None;description=[asdict(x) for x in b.basis_change_operations]
            blocks.append(dict(one=one,basis=basis(description,p.num_system_qubits,b.basis_id,b.basis_hash),
                diagonal=tuple(b.diagonal_eigenvalues if one else b.diagonal_eta),lam=None if one else b.lam))
            raw_blocks.append(dict(block_id=b.block_id,basis_id=b.basis_id,basis_hash=b.basis_hash,operations=description,
                diagonal_hex=[float(x).hex() for x in blocks[-1]['diagonal']],lambda_hex=None if one else float(b.lam).hex()))
        tail=p.rte_preparation.symbolic_tail;components=[];raw_components=[];registry={}
        for c in tail.components:
            row={k:getattr(c,k) for k in ('component_id','probability','coefficient_sign','basis_id','basis_hash','diagonal_pauli_support','is_identity')}
            if not c.is_identity and c.basis_hash not in registry:
                registry[c.basis_hash]=[asdict(x) for x in c.basis_change_operations]
            row['basis']=None if c.is_identity else basis(registry[c.basis_hash],p.num_system_qubits,c.basis_id,c.basis_hash)
            components.append(row);raw_components.append({k:v for k,v in row.items() if k!='basis'})
        preps['H8'][key]=dict(n=p.num_system_qubits,blocks=blocks,components=components,preparation_hash=p.preparation_hash,
            constant=p.constant_coefficient,extracted=p.extracted_identity_coefficient,lambda_r=p.exact_rte_lambda_r)
        raw_h8[key]=dict(preparation_hash=p.preparation_hash,blocks=raw_blocks,components=raw_components,bases=registry,
            constant_hex=float(p.constant_coefficient).hex(),extracted_hex=float(p.extracted_identity_coefficient).hex(),lambda_r_hex=float(p.exact_rte_lambda_r).hex())
    write('h8_preparation_export_private_v1.json',raw_h8)
    write('preparation_identity_receipt_v1.json',{
        system:{k:dict(preparation_hash=p['preparation_hash'],blocks=len(p['blocks']),
            components=len(p['components']),lambda_r_hex=float(p['lambda_r']).hex(),
            basis_ids=sorted({b['basis'].basis_id for b in p['blocks']}))
            for k,p in records.items()} for system,records in preps.items()})
    del ham,sector,full,state,graph,raw_h8
    candidate_index=read(inputs['candidate_index']['path']);b0i_scalars=manifest['B0I_tail_scalars']
    rows=[];primitive_receipts={};laws=[]
    for system,entries in candidate_index.items():
        unique={x['cell_id']:x for x in entries}
        for cid,entry in sorted(unique.items()):
            progress('THEORY_CANDIDATE',system=system,candidate=cid,completed=len({(x['system'],x['candidate_id']) for x in rows}))
            c=entry['cell'];method=c.get('method','B0I' if cid.startswith('B0I_') else None)
            key=('discard' if method in ('B0','B0I') else 'faithful')+':p'+str(c['prefix'])
            p=preps[system][key];n=p['n'];bases=tuple(b['basis'] for b in p['blocks'])
            def primitive(j,t):
                b=p['blocks'][j]
                obj=(ops.one_body_diagonal_primitives(np.asarray(b['diagonal']),t) if b['one'] else
                    ops.df_squared_diagonal_primitives(np.asarray(b['diagonal']),b['lam'],t))
                record=dict(phase=float(obj.global_phase).hex(),rz=[(q,float(a).hex()) for q,a in obj.rz],rzz=[(l,r,float(a).hex()) for l,r,a in obj.rzz])
                pk=system+'/'+key+'/'+str(j)+'/'+float(t).hex();current=digest(record)
                if pk in primitive_receipts and primitive_receipts[pk]!=current:raise RuntimeError('PRIMITIVE_DRIFT')
                primitive_receipts[pk]=current
                return obj
            scalar=p['constant']+(float(b0i_scalars[system][str(c['prefix'])+'/'+str(c['q'])]) if method=='B0I' else p['extracted'] if method=='B2' else 0.)
            random=method=='B2';occurrences={}
            if random:
                tau=p['lambda_r']*(.8/c['q'])/c['r'];dist=rte.finite_rte_distribution(tau,c['K'])
                total=sum(F(x['probability']) for x in p['components'])
                components=tuple(m.Component(x['component_id'],F(x['probability'])/total,x['basis'],tuple(x['diagonal_pauli_support']) if not x['is_identity'] else (),x['coefficient_sign']) for x in p['components'])
                weights={k:F(v) for k,v in zip(dist.orders,dist.order_probabilities)};z=sum(weights.values());orders={k:v/z for k,v in weights.items()}
                laws.append(dict(system=system,candidate_id=cid,tau_hex=tau.hex(),lambda_r_hex=float(p['lambda_r']).hex(),q=c['q'],r=c['r'],K=c['K'],R=c['R'],native_distribution=dist.to_dict(),component_probability_sum=fraction(total),order_probability_sum=fraction(z),Fraction_law_policy='EXACT_NORMALIZATION_OF_NATIVE_BINARY64_PROBABILITIES',new_sampling=0))
                for model in ('A','B'):occurrences[model]=grouped.expected_rte_grouped(m,components,orders,c['r'],tau,n,model,max_transitions=1_000_000)
            for model in ('A','B'):
                circuit=(m.partial_prefix_circuit(bases,primitive,.8,c['q'],n,policy='symmetric_directional',scalar_phase=-scalar*.8,model=model) if random else
                    m.deterministic_circuit(bases,primitive,.8,c['q'],c['formula'],policy='symmetric_directional',ancilla=n,scalar_phase=-scalar*.8,model=model))
                for axis in ('cosine','sine'):
                    if random:
                        result=m.expected_partial_trajectory(occurrences[model],c['q'],circuit,n,'X' if axis=='cosine' else 'Y');lo,hi=result.lower,result.upper;transitions=result.transitions
                    else:
                        lo=hi=m.count(m.hadamard_wrapper(circuit,n,'X' if axis=='cosine' else 'Y'));transitions=0
                    rows.append(dict(system=system,candidate_id=cid,cell=c,axis=axis,model=model,preparation_hash=p['preparation_hash'],lower=costs(lo),upper=costs(hi),exact_for_supplied_law=lo==hi,DP_grouped_transitions=transitions,theory_profile=m.PROFILE,control='symmetric_directional',state_preparation_excluded=True,measurement_gate_cost_excluded=True,full_native_numeric_fingerprint_rebuilt=False))
                del circuit
    write('theoretical_wrapper_costs_v1.json',rows);write('finite_law_receipts_v1.json',laws);write('primitive_receipts_v1.json',primitive_receipts)
    imported=[]
    for name,module in sorted(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith(('trotterlib.','trottertracks.')) and path:
            relative=Path(path).resolve().relative_to(provider).as_posix();data=Path(path).read_bytes()
            if manifest['provider_hashes'].get(relative)!=sha(data):raise RuntimeError('IMPORTED_PROVIDER_CLOSURE_MISMATCH:'+relative)
            imported.append(dict(path=relative,sha256=sha(data)))
    if any(x.startswith(('cupy','cupyx')) for x in sys.modules):raise RuntimeError('GPU_MODULE_PRESENT')
    if len(rows)!=832:raise RuntimeError('FIXED_CANDIDATE_COVERAGE_MISMATCH')
    write('execution_terminal_v1.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=manifest['SOURCE'],candidate_ids=208,theory_rows=len(rows),H8_preparations=7,H6_preparations_reused=7,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,versions=versions,imported_provider_closure=imported,new_sampling=0,new_compile=0,new_signal=0,new_molecular_generation=0,scientific_adoption=False,subagents=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=read(args.manifest)
    resource.setrlimit(resource.RLIMIT_AS,(manifest['limits']['AS_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_FSIZE,(manifest['limits']['output_bytes'],)*2)
    execute(manifest,args.output)


if __name__=='__main__':main()
