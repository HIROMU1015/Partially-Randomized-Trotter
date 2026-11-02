#!/usr/bin/env python3
"""One fixed Stage 3 native preflight and 36-target price pilot. No retry."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction as F
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from threading import Lock,Event
import traceback
from zoneinfo import ZoneInfo

import run_h4_p3_stage1 as previous

ROOT=Path(__file__).resolve().parents[1]
BASE='be3de04e6fb501fecc4a07f38ceea7c24ba10472'
S2='8bcf5851f99f980dc08c5885ace54874b554c815'
OLD='artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/'
OUT=ROOT/'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run01'
CONTRACTS=('docs/research/h4_p3_stage3_native_cost_design_20261102.md',
           'docs/research/h4_p3_stage3_preflight_readiness_20261102.md')
OWN=('scripts/run_h4_p3_stage3.py','src/trotterlib/matched_partition_stage3.py',
     'tests/test_matched_partition_stage3.py','docs/research/h4_p3_stage3_native_validation_20261102.md')


def modules():
    previous.modules()
    return importlib.import_module('stage1.matched_partition_stage3')


def sha(raw):return hashlib.sha256(raw).hexdigest()


def write(path,data):
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_qpy(path,circuit):
    from qiskit import qpy
    path=OUT/path;path.parent.mkdir(parents=True,exist_ok=True)
    stream=io.BytesIO();qpy.dump(circuit,stream);raw=stream.getvalue()
    with path.open('xb') as sink:
        with gzip.GzipFile(fileobj=sink,mode='wb',filename='',mtime=0) as compressed:compressed.write(raw)
    return dict(path=str(path.relative_to(OUT)),sha256=sha(path.read_bytes()),
        decoded_QPY_sha256=sha(raw),decoded_bytes=len(raw),stored_bytes=path.stat().st_size)


def code_binding():
    paths=set(OWN)|{'scripts/run_h4_p3_stage1.py'}
    for key,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if key.startswith('stage1.') and file:
            p=Path(file).resolve()
            if ROOT in p.parents:paths.add(str(p.relative_to(ROOT)))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes={p:sha((ROOT/p).read_bytes()) for p in sorted(paths)}
    for p,h in hashes.items():
        if sha(previous.gitbytes(commit,p))!=h:raise ValueError('UNFIXED_SOURCE:'+p)
    return commit,hashes


def inputs():
    import numpy as np
    ready=previous.gitbytes(BASE,CONTRACTS[1]).decode()
    rows=re.findall(r'\| `([^`]+\.(?:json|md|npz))` \| `([0-9a-f]{64})`',ready)
    if len(rows)!=7:raise ValueError('STAGE2_INPUT_TABLE')
    files={};saved={}
    for name,h in rows:
        path=OLD+name;raw=previous.gitbytes(S2,path)
        if sha(raw)!=h or (ROOT/path).read_bytes()!=raw:raise ValueError('STAGE2_INPUT_BYTES:'+path)
        files[path]=dict(ref=S2,sha256=h)
        if name.endswith('.json'):saved[name]=json.loads(raw)
    oldsource=saved['source_and_inputs.json']
    for path,row in oldsource['inputs']['files'].items():
        raw=previous.gitbytes(row['ref'],path)
        if sha(raw)!=row['sha256'] or (ROOT/path).read_bytes()!=raw:raise ValueError('ORIGINAL_INPUT_BYTES:'+path)
        files[path]=row
    if saved['STOP.json']['status']!='STOP_STAGE2_UNRESOLVED_COMPONENTS':raise ValueError('STAGE2_STOP')
    with np.load(io.BytesIO(previous.gitbytes(S2,OLD+'common_operators.npz')),allow_pickle=False) as a:
        t=a['T']
    path=previous.IBASE+'snapshot.npz'
    with np.load(io.BytesIO(previous.gitbytes(previous.INPUT_REF,path)),allow_pickle=False) as a:
        one=a['df_one_body'];nuclear=float(a['nuclear_repulsion'])
    dictionaries=saved['dictionaries.json'];norms=saved['results.json']['normalization']
    if one.shape!=(8,8) or t.shape!=(256,256) or nuclear!=2.29310124732:raise ValueError('FIXED_H4')
    if dictionaries['P']['component_count']!=184 or dictionaries['DF']['component_count']!=360:raise ValueError('DICTIONARY_CARDINALITY')
    if len(dictionaries['DF']['basis'])!=10:raise ValueError('DF_BASIS_CARDINALITY')
    if [x['fragment'] for x in dictionaries['DF']['basis']]!=list(range(10)):raise ValueError('DF_BASIS_ORDER')
    return one,t,nuclear,dictionaries,norms,files


def compact_event(event):
    row=event.to_dict()
    for app in row['application_sequence']:app.pop('basis_change_operations',None)
    row['basis_operation_lookup']='component ID -> law basis ID/hash -> registry metadata'
    return row


def execute():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    counts=dict(scientific_batches_started=0,T_eigendecompositions=0,DF_eigendecompositions=0,
        native_bases=0,reference_half_exponentials=0,sampling_calls=0,trajectories=0,event_positions=0,
        native_body_programs=0,wrapper_targets=0,compile_started=0,compile_completed=0,
        scientific_retries=0,new_signal_B_N_C_G=0,finite_bit_mean=0,SCF_FCI_DF=0,new_state_reference=0)
    checks={};prices={};input_qpy={};source={}
    price_lock=Lock();price_failed=Event()
    def sync_operations():
        current=sys.modules.get('stage1.matched_partition_stage3')
        if current:
            for name,key in [('T_eigendecompositions','T_eigendecompositions'),
                             ('native_bases','native_basis_lowerings'),
                             ('reference_half_exponentials','reference_half_exponentials')]:
                counts[name]=current.OPERATIONS.get(key,0)
    def status(state):
        sync_operations()
        write('execution_status.json',dict(state=state,counts=counts,elapsed_seconds=time.monotonic()-start))
    try:
        kernel=modules();kernel.OPERATIONS.clear();commit,code=code_binding()
        for p in CONTRACTS:
            if (ROOT/p).read_bytes()!=previous.gitbytes(BASE,p):raise ValueError('CONTRACT_CHANGED')
        runtime={x:importlib.metadata.version(x) for x in ('numpy','scipy','qiskit','qiskit-nature','openfermion')}
        if runtime['qiskit']!='1.3.0' or runtime['qiskit-nature']!='0.7.2':raise ValueError('NATIVE_COMPILER_FAMILY')
        available=max(1,len(os.sched_getaffinity(0))-math.ceil(os.getloadavg()[0]))
        workers=min(available,36)
        source=dict(commit=commit,contract_commit=BASE,code_sha256=code,runtime=runtime,
            python=sys.version,executable=sys.executable,execution_JST=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            conditions=dict(molecule='linear H4',distance_A=1.,basis='STO-3G',charge=0,multiplicity=1,
                full_dimension=256,system_qubits=8,ancilla=8,DF='saved tol-only1e-8 rank10',partition='A',
                arms=['M_B','M_C'],T_exact=str(kernel.TIME),q=8,r=4,K=4,events_per_trajectory=32,
                seeds={k:list(v) for k,v in kernel.SEEDS.items()},state_preparation='P=0; no state signal'),
            resources=dict(available_affinity_cpus=len(os.sched_getaffinity(0)),load=os.getloadavg(),
                workers=workers,BLAS_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                meminfo={x.split(':')[0]:x.split(':')[1].strip() for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith(('MemTotal:','MemAvailable:'))},
                reason='independent fixed native preflights and price targets; only own process settings'),
            policy='contract engineering gamma/scale policy; no phase alignment, no full-law guarantee',
            contracts={p:sha(previous.gitbytes(BASE,p)) for p in CONTRACTS})
        write('source_and_inputs.json',source)
        one,t,nuclear,dictionaries,norms,files=inputs();source['inputs']=files
        write('source_and_inputs.json',source);status('INPUTS_BOUND')
        counts['scientific_batches_started']=1
        retained=kernel.retained_preflight(one,t);counts.update(T_eigendecompositions=1,native_bases=1,reference_half_exponentials=1)
        checks['retained']=retained.checks;checks['retained_basis']=retained.basis.checks
        write('preflight.json',checks)
        common_qpy={}
        for name,c in [('basis_T',retained.basis.circuit),('half',retained.half),
                       ('half_actual_inverse',retained.inverse),('controlled_half',retained.controlled_half)]:
            common_qpy[name]=save_qpy('native/'+name+'.qpy.gz',c)
        import numpy as np
        def lower(b):
            u=np.empty((8,8),dtype=complex);u.real=b['U_real'];u.imag=b['U_imag']
            return b['fragment'],kernel.basis_from_saved(u)
        bases={}
        with ThreadPoolExecutor(max_workers=min(10,workers)) as pool:
            for index,basis in pool.map(lower,dictionaries['DF']['basis']):
                bases[index]=basis;counts['native_bases']+=1
                checks[f'DF_basis_{index}']=basis.checks
                common_qpy[f'basis_DF_{index}']=save_qpy(f'native/basis_DF_{index}.qpy.gz',basis.circuit)
                write('preflight.json',checks)
        write('native_bases.json',dict(T=retained.basis.metadata,DF={str(k):v.metadata for k,v in bases.items()},
            qpy=common_qpy,T_eigenvalues=retained.eps.tolist(),T_U_real=retained.basis.u.real.tolist(),
            T_U_imag=retained.basis.u.imag.tolist(),half_errors=retained.errors,
            common_half_fingerprint=kernel.canonical_qiskit_circuit_fingerprint(retained.controlled_half)))
        angle=kernel.scalar_angle(nuclear,4.679666433618258)
        scalar=kernel.QuantumCircuit(9);scalar.p(-float(angle),8)
        checks['scalar']=kernel.program_check(scalar,kernel.controlled(np.exp(-1j*float(angle))*np.eye(256)),
            float(abs(F(float(angle))-angle))+kernel.gamma(8)*(1+abs(float(angle))))
        kernel.require_pass({'scalar':checks['scalar']});write('preflight.json',checks)
        comps={name:kernel.component_inputs(name,dictionaries[name],norms[name],bases,8) for name in ('P','DF')}
        laws={name:dict(saved_Stage2_normalization=norms[name],
                components=[dict(row,probability=component.probability,
                    basis_id=component.basis_id,basis_hash=component.basis_hash)
                    for row,component in zip(dictionaries[name]['components'],value.components)],
                basis_metadata=[asdict(x) for x in value.registry.metadata()],
                sampling='single PCG64/np.choice stream per seed; 32 flat events, outer-major; no CDF mean',
                seeds=list(kernel.SEEDS[name])) for name,value in comps.items()}
        write('law_bindings.json',laws);status('COMMON_NATIVE_PREFLIGHT_PASS')
        # Exactly eight sampling calls, only after common basis/half/scalar PASS.
        trajectories={}
        for name,comp in comps.items():
            for index,seed in enumerate(kernel.SEEDS[name]):
                events=kernel.sample_rte_events(comp.components,comp.distribution,sample_count=32,seed=seed)
                key=f'{name}_{index}';trajectories[key]=(name,events)
                counts['sampling_calls']+=1;counts['trajectories']+=1;counts['event_positions']+=len(events)
                write(f'trajectories/{key}.json',dict(arm=name,index=index,seed=seed,event_count=len(events),
                    events=[dict(compact_event(e),outer_position=i//4,inner_position=i%4) for i,e in enumerate(events)]))
        status('TRAJECTORIES_FROZEN')
        shared=kernel.shared_parts(retained,angle);bodies=dict(shared)
        def native_program(item):
            key,(name,events)=item
            parts,sidecar=kernel.trajectory_parts(comps[name],events,retained,angle)
            oracle=kernel.DictionaryOracle(comps[name],dictionaries[name],8)
            result={}
            for part,c in parts.items():
                expected,error=oracle.trajectory(events,retained,angle,part=part)
                result[part]=kernel.program_check(c,expected,error)
                kernel.require_pass({key+'/'+part:result[part]})
            return key,parts,result,sidecar
        with ThreadPoolExecutor(max_workers=min(8,workers)) as pool:
            futures=[pool.submit(native_program,x) for x in trajectories.items()]
            for future in as_completed(futures):
                key,parts,result,sidecar=future.result()
                for part,c in parts.items():bodies[key+'/'+part]=c
                checks[key]=result;counts['native_body_programs']+=2
                write('preflight.json',checks);write(f'trajectories/{key}_native.json',dict(parts=result,outer_residual=sidecar))
        oracle=kernel.DictionaryOracle(comps['P'],dictionaries['P'],8)
        for part,c in shared.items():
            if part=='R':expected,error=oracle.trajectory((),retained,angle,part='R')
            else:expected,error=np.eye(512),0.
            checks[part]=kernel.program_check(c,expected,error);kernel.require_pass({part:checks[part]})
            counts['native_body_programs']+=1
        write('preflight.json',checks)
        targets={}
        for bodykey,c in sorted(bodies.items()):
            for axis,circuit in kernel.wrappers(c).items():
                key=bodykey+'/'+axis;targets[key]=circuit
                input_qpy[key]=save_qpy('prices/'+key+'/input.qpy.gz',circuit)
        counts['wrapper_targets']=len(targets)
        if len(targets)!=36 or counts['native_body_programs']!=18:raise ValueError('PRICE_SCOPE_CARDINALITY')
        write('price_inputs.json',input_qpy);status('ALL_NATIVE_PROGRAMS_PASS_PRICE_INPUTS_FROZEN')
        def compile_one(key,circuit):
            if price_failed.is_set():raise RuntimeError('COMPILE_NOT_STARTED_AFTER_OTHER_FAILURE')
            with price_lock:counts['compile_started']+=1
            try:
                compiled,metrics=kernel.price(circuit)
                output=save_qpy('prices/'+key+'/output.qpy.gz',compiled)
                metrics.update(input_QPY=input_qpy[key],output_QPY=output)
                write('prices/'+key+'/metrics.json',metrics)
                with price_lock:
                    prices[key]=metrics;counts['compile_completed']+=1
                return key,metrics
            except Exception:
                price_failed.set();raise
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futuremap={}
            for key,circuit in targets.items():
                futuremap[pool.submit(compile_one,key,circuit)]=key
            for future in as_completed(futuremap):
                try:key,metrics=future.result()
                except Exception:
                    price_failed.set()
                    for other in futuremap:other.cancel()
                    raise
                with price_lock:snapshot=dict(prices)
                write('prices.json',snapshot);status('COMPILING_FIXED_TARGETS')
        if counts['compile_completed']!=36:raise ValueError('INCOMPLETE_PRICES')
        interaction={}
        for key in trajectories:
            for axis in ('cosine','sine'):
                f,e,rr,empty=(prices[k+'/'+axis] for k in (key+'/F',key+'/E','R','Empty'))
                interaction[key+'/'+axis]={m:f[m]-rr[m]-e[m]+empty[m] for m in ('RZ','CX','size')}
        write('interaction.json',dict(formula='F - R - E + Empty',values=interaction,
            interpretation='fixed-program compiler nonadditivity, no unique causal allocation; depth not additive'))
        for p,h in code.items():
            if sha((ROOT/p).read_bytes())!=h:raise ValueError('SOURCE_CHANGED')
        sync_operations()
        finish=dict(state='COMPLETED',counts=counts,operations=dict(kernel.OPERATIONS),workers_joined=True,elapsed_seconds=time.monotonic()-start)
        write('execution_status.json',finish)
        write('STOP.json',dict(status='STOP_STAGE3_FIXED_COHORT_COST_COMPLETE_NO_AUTOMATIC_FOLLOWUP',
            **finish,Stage4_authorized=False,automatic_followup=False))
        print(json.dumps(dict(status='COMPLETED',source=commit,**finish)))
    except Exception as error:
        sync_operations()
        if prices:write('prices.json',prices)
        failure=dict(state='FAILED_NO_RETRY',counts=counts,source_commit=source.get('commit'),
            error=traceback.format_exc(),failed_checks=getattr(error,'checks',None),
            operations=dict(kernel.OPERATIONS) if 'kernel' in locals() else {},
            workers_joined=True,elapsed_seconds=time.monotonic()-start)
        write('preflight.json',checks);write('execution_status.json',failure)
        write('STOP.json',dict(status='STOP_STAGE3_PREFLIGHT_OR_EXECUTION_FAILURE_NO_RETRY',**failure,automatic_followup=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',required=True,action='store_true')
    parser.parse_args();execute()
