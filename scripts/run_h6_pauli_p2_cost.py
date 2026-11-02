#!/usr/bin/env python3
"""Fixed saved-trajectory P2 only; never sample, rebuild Full, or price N/G."""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
# Directory date is UTC, matching execution_UTC recorded below.
OUT = ROOT/'artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost'
HYBRID = '00958ae1945c780718b2deef07d5dab1096a5cd8'
RTE = '10cb3e5cca7834ebf75301bd91bd093ca51d4a6b'
P1 = 'bfb4ca75437a9b3605a8f57b5f82100615dd034f'
HBASE = 'artifacts/h6_hybrid_fairness_extension/2026-10-27/'
RBASE = 'artifacts/pauli_rte_event_extension/2026-10-26/'
MODEL = 'artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/model.json'
MODEL_HASH = '6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681'
HAM_HASH = '37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847'
CANDIDATES = dict(H32=('PR_p0078_q8_r32_K2',78,32),
                  H64=('PR_p0078_q8_r64_K2',78,64),
                  R64=('H6_p000_q8_r64_K2',0,64))
FILES = ('scripts/run_h6_pauli_p2_cost.py','scripts/restore_pauli_h2_h6_evidence.py',
         'src/trotterlib/pauli_mechanism_cost.py','src/trotterlib/h4_d1_adapter.py',
         'src/trotterlib/h4_d2_pauli_control.py','src/trotterlib/pauli_signal_adapter.py',
         'src/trotterlib/fixed_signal_evaluation.py','src/trotterlib/rte_compiled_cost.py',
         'src/trotterlib/rte.py','tests/test_pauli_mechanism_cost.py',
         'docs/research/h6_pauli_p2_cost_20261101.md')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gitbytes(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)


def modules():
    # Relevant pure modules need neither matplotlib nor OpenFermion __init__.
    for name,folder in [('p2_study','src/trotterlib'),
                        ('p2_study.df_trotter','src/trotterlib/df_trotter')]:
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(ROOT/folder)]
            sys.modules[name]=package
    from p2_study import pauli_mechanism_cost as kernel
    return kernel


def selected_members(commit,base,namespace,wanted):
    from restore_pauli_h2_h6_evidence import recover_member
    manifest=gitbytes(commit,base+'data_archives.json')
    record=next(x for x in json.loads(manifest) if x['namespace']==namespace)
    chunks=[]
    for part in record['parts']:
        raw=gitbytes(commit,base+part['path'])
        if sha(raw)!=part['sha256']:raise ValueError('ARCHIVE_PART_SHA')
        chunks.append(raw)
    raw=b''.join(chunks)
    if sha(raw)!=record['archive_sha256']:raise ValueError('ARCHIVE_SHA')
    mapping={record['members'][k]['archive_member']:k for k in wanted}
    found={};provenance={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r|xz') as archive:
        for member in archive:
            if member.name not in mapping:continue
            key=mapping[member.name];meta=record['members'][key]
            content=recover_member(archive.extractfile(member).read(),meta)
            found[key]=content
            provenance[key]=dict(commit=commit,manifest=base+'data_archives.json',
                manifest_sha256=sha(manifest),archive_sha256=record['archive_sha256'],
                parts=record['parts'],member=key,**meta)
            if len(found)==len(wanted):break
    if set(found)!=set(wanted):raise ValueError('MISSING_SAVED_MEMBER')
    return found,provenance


def inputs():
    modelraw=gitbytes(RTE,MODEL)
    if sha(modelraw)!=MODEL_HASH:raise ValueError('MODEL_SHA')
    model=json.loads(modelraw)
    if (model['Hamiltonian_hash']!=HAM_HASH or model['n_qubits']!=12
            or model['nonidentity_terms']!=918):raise ValueError('H6_MODEL')
    historical={}
    for commit,base,name in [(HYBRID,HBASE,'SOURCE.json'),
            (RTE,RBASE,'SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json')]:
        raw=gitbytes(commit,base+name);source=json.loads(raw)
        for path,digest in source['code_sha256'].items():
            if sha(gitbytes(source['commit'],path))!=digest:
                raise ValueError('HISTORICAL_SOURCE_SHA:'+path)
        historical[source['commit']]=dict(result_commit=commit,path=base+name,
            sha256=sha(raw),code_sha256=source['code_sha256'])
        # The saved price runners call these same wrapper/compiler functions.
        functions={}
        for path,names in [('src/trotterlib/fixed_signal_evaluation.py',
                           ('default_compiler','hadamard_wrapper')),
                          ('src/trotterlib/rte_compiled_cost.py',('transpile_and_measure_cost',))]:
            old=ast.parse(gitbytes(source['commit'],path));new=ast.parse((ROOT/path).read_text())
            for function in names:
                def descriptor(tree):
                    return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function))
                if descriptor(old)!=descriptor(new):raise ValueError('HISTORICAL_COMPILER_WRAPPER_PROFILE')
                functions[function]=sha(descriptor(old).encode())
        historical[source['commit']]['same_wrapper_compiler_functions']=functions
    members={};provenance={};prefixes={}
    for commit,base,namespace,tags in [(HYBRID,HBASE,'H6',('H32','H64')),
                                      (RTE,RBASE,'cases',('R64',))]:
        wanted=[]
        for tag in tags:
            cid=CANDIDATES[tag][0]
            prefix=('H6/batch01/cases/' if namespace=='H6' else 'cases/')+cid+'/'
            prefixes[tag]=prefix
            wanted.append(prefix+('spec.json' if namespace=='H6' else 'law.json'))
            wanted.append(prefix+'prices/confirmation/0000/result.json')
            wanted.extend(prefix+'prices/confirmation/0000/'+a+'_'+k+'.qpy.gz'
                          for a in ('cosine','sine') for k in ('input','compiled'))
        saved,prov=selected_members(commit,base,namespace,wanted)
        members.update(saved);provenance.update(prov)
    laws={};results={}
    for tag,(cid,ld,r) in CANDIDATES.items():
        p=prefixes[tag]
        law=json.loads(members[p+('law.json' if tag=='R64' else 'spec.json')])
        if tag!='R64':law=law['law']
        result=json.loads(members[p+'prices/confirmation/0000/result.json'])
        if any(law['config'][k]!=v for k,v in dict(L_D=ld,q=8,r=r,K=2,T=.8).items()):
            raise ValueError('LAW_CONFIG:'+tag)
        if (law['n_system']!=12 or law['Hamiltonian_hash']!=HAM_HASH
                or law['step_time']!=.1
                or law['scalar_step_binary64']!=-model['hamiltonian']['constant']*.1):
            raise ValueError('LAW_MODEL:'+tag)
        terms=model['hamiltonian']['terms']
        if [(x['label'],x['coefficient']) for x in law['retained']]!=[tuple(x) for x in terms[:ld]]:
            raise ValueError('RETAINED_ORDER:'+tag)
        expected_tail=[(label,1 if coefficient>0 else -1,abs(coefficient))
                       for label,coefficient in terms[ld:]]
        if [(x['label'],x['sign'],x['coefficient_abs']) for x in law['components']]!=expected_tail:
            raise ValueError('TAIL_COMPONENT_JOIN:'+tag)
        if (result['candidate_id']!=cid or result['cohort']!='confirmation' or result['index']!=0
                or result['qiskit_version']!='1.3.0' or not result['state_preparation_excluded']):
            raise ValueError('RESULT_IDENTITY:'+tag)
        laws[tag]=law;results[tag]=result
    return laws,results,members,prefixes,dict(model=dict(commit=RTE,path=MODEL,
        sha256=MODEL_HASH,Hamiltonian_hash=HAM_HASH),historical_SOURCE=historical,
        members=provenance)


def decode(raw):
    from qiskit import qpy
    circuits=qpy.load(io.BytesIO(gzip.decompress(raw)))
    if len(circuits)!=1:raise ValueError('ONE_CIRCUIT_QPY')
    return circuits[0]


def verify_full(laws,results,members,prefixes):
    kernel=modules();specs={};full={};checks={}
    for tag in CANDIDATES:
        spec=kernel.recipe(laws[tag],results[tag]);specs[tag]=spec
        _,phase=kernel.source_tokens(spec,'F')
        if abs(math.remainder(phase-spec['saved_body_phase'],2*math.pi))>1e-12:
            raise ValueError('SAVED_BODY_PHASE:'+tag)
        if results[tag]['orders']!=[x.taylor_order for x in spec['events']]:
            raise ValueError('EVENT_ORDERS:'+tag)
        rp=kernel.source_tokens(spec,'R')[1];ep=kernel.source_tokens(spec,'E')[1]
        checks[tag]=dict(axes={},event_schema='full' if tag=='R64' else 'compact',
            event_count=len(spec['events']),source_phase=dict(F=phase,R=rp,E=ep,Empty=0.),
            phase_modulo_error=abs(math.remainder(phase-rp-ep,2*math.pi)),
            event_records_sha256=sha(json.dumps(results[tag]['events'],sort_keys=True).encode()),
            restored_F_tokens_sha256=kernel.program_hash(dict(tokens=kernel.source_tokens(spec,'F')[0])),
            no_event_resampling=True,no_F_build_or_compile=True)
        if checks[tag]['phase_modulo_error']>1e-12:raise ValueError('PHASE_OWNERSHIP')
        full[tag]={}
        for axis in ('cosine','sine'):
            saved=results[tag]['axes'][axis];circuits={}
            for kind in ('input','compiled'):
                key=prefixes[tag]+'prices/confirmation/0000/'+axis+'_'+kind+'.qpy.gz'
                raw=members[key];uncompressed=gzip.decompress(raw);meta=saved[kind+'_qpy']
                if (sha(raw)!=meta['gzip_sha256'] or sha(uncompressed)!=meta['raw_sha256']
                        or len(raw)!=meta['gzip_bytes'] or len(uncompressed)!=meta['raw_bytes']):
                    raise ValueError('SAVED_QPY_RESULT_BINDING:'+key)
                circuits[kind]=decode(raw)
            sem=kernel.semantic_program(circuits['input'])
            check=kernel.compare_programs(sem,kernel.expected_full(spec,axis))
            actual=kernel.metrics(circuits['compiled'])
            for target,original in [('RZ','rz_count'),('CX','cx_count'),('total_depth','total_depth'),
                    ('rz_depth','rz_depth'),('cx_depth','cx_depth'),('circuit_size','circuit_size')]:
                if actual[target]!=saved[original]:raise ValueError('SAVED_METRIC:'+tag+axis+target)
            if abs(math.remainder(actual['global_phase']-saved['global_phase'],2*math.pi))>1e-12:
                raise ValueError('SAVED_COMPILED_PHASE')
            c=circuits['compiled']
            if (c.num_qubits!=13 or c.num_clbits!=1 or c.count_ops().get('measure')!=1
                    or c.find_bit(c.data[-1].qubits[0]).index!=12
                    or set(actual['gate_histogram'])-{'rz','sx','x','cx','measure'}):
                raise ValueError('COMPILED_BOUNDARY')
            checks[tag]['axes'][axis]=dict(**check,input_semantic_sha256=kernel.program_hash(sem),
                individual_saved_metrics_match=True,saved_qpy=saved)
            full[tag][axis]=dict(**actual,input_qpy=saved['input_qpy'],compiled_qpy=saved['compiled_qpy'],
                original_result=prefixes[tag]+'prices/confirmation/0000/result.json',reused=True)
    return specs,full,checks


def write_json(name,payload):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save_qpy(circuit,path):
    from qiskit import qpy
    stream=io.BytesIO();qpy.dump(circuit,stream);raw=stream.getvalue()
    zipped=bytearray(gzip.compress(raw,compresslevel=9,mtime=0));zipped[9]=3
    zipped=bytes(zipped);target=OUT/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(zipped)
    return dict(path=path,raw_sha256=sha(raw),gzip_sha256=sha(zipped),
                raw_bytes=len(raw),gzip_bytes=len(zipped))


def compile_one(name,input_record):
    start=time.monotonic();kernel=modules()
    from p2_study.rte_compiled_cost import transpile_and_measure_cost
    circuit=decode((OUT/input_record['path']).read_bytes())
    cost=transpile_and_measure_cost(circuit,kernel.default_compiler(),
                                    actual_circuit_fingerprint=input_record['raw_sha256'])
    compiled=cost.transpiled_circuit
    record=dict(**kernel.metrics(compiled),input_qpy=input_record,
                compiled_qpy=save_qpy(compiled,'programs/'+name+'_compiled.qpy.gz'),
                compiler=asdict(cost.compiler),compiler_settings_hash=cost.compiler_settings_hash,
                worker_pid=os.getpid(),elapsed_seconds=time.monotonic()-start)
    write_json('programs/'+name+'.json',record)
    return name,record


def execute():
    OUT.mkdir(parents=True,exist_ok=False)
    counts=dict(new_wrapper_programs=0,new_compile_submitted=0,new_compile_completed=0,
        failures=0,Full_rebuilt=0,Full_recompiled=0,Full_axes_reused=0,sampling=0,RNG_replay=0,
        state_reference_generation=0,P1_recalculation=0,N_G_SE_ranking=0)
    start=time.monotonic()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        codes={p:sha((ROOT/p).read_bytes()) for p in FILES}
        if any(sha(gitbytes(commit,p))!=digest for p,digest in codes.items()):
            raise ValueError('SOURCE_NOT_COMMITTED_OR_CHANGED')
        runtime={n:importlib.metadata.version(n) for n in ('qiskit','numpy','scipy')}
        if runtime['qiskit']!='1.3.0':raise ValueError('QISKIT_PROFILE')
        kernel=modules();compiler=asdict(kernel.default_compiler())
        laws,results,members,prefixes,provenance=inputs()
        source=dict(commit=commit,code_sha256=codes,inputs=provenance,runtime=runtime,
            python=sys.version,candidates=CANDIDATES,compiler=compiler,cohort='confirmation',index=0,
            directory_date_basis='UTC',execution_UTC=datetime.now(timezone.utc).isoformat(),
            affinity_CPUs=len(os.sched_getaffinity(0)),load_average=os.getloadavg(),
            meminfo={s.split(':')[0]:s.split(':')[1].strip() for s in Path('/proc/meminfo').read_text().splitlines()
                     if s.startswith(('MemTotal:','MemAvailable:'))},
            scientific_scope='P2_ONLY_SAVED_0000_MAX12_NEW_COMPILES',
            fixed_model='linear H6, 1 Angstrom, STO-3G, neutral singlet, JW, 918 nonidentity + constant',
            control='central rotation',wrapper='cosine / +Im sine full measured Hadamard; P=0')
        write_json('source_and_inputs.json',source)
        specs,full,checks=verify_full(laws,results,members,prefixes)
        counts['Full_axes_reused']=6
        write_json('semantic_checks.json',checks);write_json('saved_full.json',full)
        programs={};hash_to_name={};parts={};sidecars={}
        # Shared R requires exact token/phase identity before any R construction.
        r32=kernel.source_tokens(specs['H32'],'R');r64=kernel.source_tokens(specs['H64'],'R')
        if r32!=r64:raise ValueError('HYBRID_R_IDENTITY')
        for tag,part,owner in [('H32','R','Hybrid_R'),('R64','R','R64_R'),
                ('H32','E','H32_E'),('H64','E','H64_E'),('R64','E','R64_E'),('H32','0','Empty')]:
            wrappers,sidecar=kernel.part_wrappers(specs[tag],part);sidecars[owner]=sidecar
            for axis,circuit in wrappers.items():
                counts['new_wrapper_programs']+=1
                identity=kernel.program_hash(kernel.semantic_program(circuit));name=owner+'_'+axis
                if identity in hash_to_name:
                    parts[name]=hash_to_name[identity];continue
                hash_to_name[identity]=name;parts[name]=name
                programs[name]=dict(input_qpy=save_qpy(circuit,'programs/'+name+'_input.qpy.gz'),
                    input_semantic_sha256=identity,owner=owner,axis=axis)
        if len(programs)>12:raise ValueError('MAX12_SCOPE')
        write_json('parts.json',dict(programs=programs,aliases=parts,sidecars=sidecars,
            H32_H64_R_exact_tokens_and_phase_equal=True,Full_checked_before_parts=True))
        completed={};failures={}
        workers=min(len(programs),len(os.sched_getaffinity(0)))
        source['workers']=workers;write_json('source_and_inputs.json',source)
        # Each future is submitted once. Exiting the context reaps every child.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={}
            for name,record in programs.items():
                future=pool.submit(compile_one,name,record['input_qpy']);futures[future]=name
                counts['new_compile_submitted']+=1
            for future in as_completed(futures):
                name=futures[future]
                try:
                    key,value=future.result();completed[key]=value;counts['new_compile_completed']+=1
                except Exception:
                    failures[name]=traceback.format_exc();counts['failures']+=1
                write_json('execution_status.json',dict(state='RUNNING',counts=counts,
                    completed=list(completed),failures=failures))
        write_json('new_metrics.json',completed)
        if failures:raise RuntimeError('COMPILE_FAILED_NO_RETRY')
        tables={};interactions={}
        for tag in CANDIDATES:
            tables[tag]={};interactions[tag]={}
            for axis in ('cosine','sine'):
                names=dict(R=('R64_R' if tag=='R64' else 'Hybrid_R')+'_'+axis,
                           E=tag+'_E_'+axis,Empty='Empty_'+axis)
                row=dict(F=full[tag][axis],**{p:completed[parts[n]] for p,n in names.items()})
                tables[tag][axis]=row
                interactions[tag][axis]=kernel.interaction_metrics(row['F'],row['R'],row['E'],row['Empty'])
        write_json('metrics_and_interactions.json',dict(metrics=tables,interactions=interactions,
            interpretation='fixed-program deletion interaction, not independent causal contributions; depth descriptive only'))
        if any(sha((ROOT/p).read_bytes())!=digest for p,digest in codes.items()):
            raise ValueError('SOURCE_CHANGED_DURING_EXECUTION')
        write_json('execution_status.json',dict(state='COMPLETED',counts=counts,
            elapsed_seconds=time.monotonic()-start,workers_reaped=True,automatic_retries=0))
        write_json('STOP.json',dict(status='STOP_P2_COMPLETE_NO_AUTOMATIC_FOLLOWUP',
            P1_STOP='STOP_UNRESOLVED_COMPONENTS',P1_unchanged=True,
            pending='Publish and compare remote SHA/full text, then mandatory STOP; no P1N/P3 or additional science',
            counts=counts))
    except Exception:
        write_json('execution_status.json',dict(state='FAILED_NO_RETRY',counts=counts,
            elapsed_seconds=time.monotonic()-start,error=traceback.format_exc()))
        write_json('STOP.json',dict(status='STOP_P2_FAILURE_NO_RETRY',counts=counts,
            error=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true',required=True)
    parser.parse_args();execute()
