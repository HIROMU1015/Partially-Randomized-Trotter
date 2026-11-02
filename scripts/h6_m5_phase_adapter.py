"""Authorized saved-input adapter. Importing this module does no array I/O."""
import json
import os
from pathlib import Path
import time
import numpy as np
import h6_m5_phase_contract as c
import h6_m5_phase_engine as e


def load_vectors(cfg):
    paths=cfg['inputs'];n=c.scope()['nqubits'];length=2**n
    with np.load(paths['initial_npz']['path'],allow_pickle=False) as z:
        c.require(set(z.files)=={'low'},'INITIAL_ARRAY_KEYS');low=z['low']
    with np.load(paths['vectors_npz']['path'],allow_pickle=False) as z:
        c.require(set(z.files)=={'low','before','after','independent_expected_high'},'VECTOR_ARRAY_KEYS')
        vectors={name:z[name] for name in z.files}
    for name,v in dict(initial=low,**vectors).items():
        c.require(v.dtype==np.dtype('complex128'),'EXACT_SAVED_COMPLEX128:'+name)
        e.finite_vector(v,length//2 if name=='independent_expected_high' else length)
    c.require(np.array_equal(low,vectors['low']),'DUPLICATED_LOW_IDENTITY')
    c.require(not np.count_nonzero(low[length//2:]),'SAVED_ANCILLA_ZERO')
    return vectors


def saved_gate(cfg):
    vectors=load_vectors(cfg)
    saved=c.read(cfg['inputs']['saved_diagnostics']['path'])
    c.require(saved['status']=='DIAGNOSTICS_ONLY_NO_ACCEPTANCE_OVERRIDE' and
        not saved['raw_gate_PASS'] and not saved['acceptance_override'] and
        not saved['full_H_reference_phase_aligned'],'ORIGINAL_FAILED_DIAGNOSTIC')
    calculated=e.comparison(vectors['before'],vectors['after'],expected_hadamard=saved['independent_expected_hadamard'])
    numeric={k:v for k,v in saved.items() if type(v) is float}
    for k,v in numeric.items():
        c.require(calculated.get(k) is not None and abs(calculated[k]-v)<=c.scope()['saved_metric_and_ordinary_replay_atol'], 'SAVED_METRIC_REPRODUCTION:'+k)
    c.require(calculated['raw_gate_PASS']==saved['raw_gate_PASS'],'SAVED_RAW_GATE_REPRODUCTION')
    return dict(status='REPRODUCED',numeric_fields=len(numeric),atol=c.scope()['saved_metric_and_ordinary_replay_atol'],
        metrics=calculated,duplicate_low_exact=True,normalization_applied=False)


def load_circuit(cfg, kind):
    from qiskit import qpy
    role=kind.lower()+'_qpy'
    row=cfg['inputs'][role]
    c.require(c.sha(row['path'])==row['sha256'],'QPY_BYTES_BEFORE_LOAD')
    with Path(row['path']).open('rb') as f:
        circuits=qpy.load(f)
        c.require(f.read()==b'','NO_QPY_TRAILING_BYTES')
    c.require(len(circuits)==1,'EXACT_ONE_CIRCUIT_PER_ROLE')
    return circuits[0]


def circuit_gate(cfg, kind):
    qc=load_circuit(cfg,kind)
    result=e.validate_circuit(qc,kind,c.scope()['nqubits'])
    receipt=c.read(cfg['inputs'][kind.lower()+'_receipt']['path'])
    c.require(result['global_phase']==receipt['global_phase'],'SAVED_GLOBAL_PHASE_IDENTITY')
    return result


def save_vector(path, vector):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb') as f:
        np.savez(f,vector=vector);f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()
    return dict(file=path.name,bytes=path.stat().st_size,sha256=c.sha(path),dtype=str(vector.dtype),shape=list(vector.shape))


def worker(cfg, job, cancelled):
    """Called once per reserved job; ordinary failure stops future dispatch only."""
    c.require(job in c.JOBS,'REGISTERED_WORKER_JOB');c.consumed(cfg)
    c.require(c.read(Path(cfg['ledger'])/(job+'.json'))['job']==job,'RESERVED_REPLAY')
    start=time.monotonic();kind,mode=job.split('_');folder=Path(cfg['output'])/job
    folder.mkdir(exist_ok=False)
    vectors=load_vectors(cfg);qc=load_circuit(cfg,kind)
    schema=e.validate_circuit(qc,kind,c.scope()['nqubits'])
    c.require(not cancelled.is_set(),'OWN_SIGNAL_BEFORE_REPLAY')
    with (folder/'progress.jsonl').open('x') as journal:
        def progress(i,total):
            journal.write(json.dumps(dict(gates=i,total=total,elapsed=time.monotonic()-start))+'\n');journal.flush()
        progress(0,len(qc.data))
        v=(e.ordinary(qc,vectors['low']) if mode=='ORDINARY' else
            e.extended(qc,vectors['low'],progress=progress,cancelled=cancelled.is_set))
        progress(len(qc.data),len(qc.data))
    c.require(not cancelled.is_set(),'OWN_SIGNAL_AFTER_REPLAY')
    artifact=save_vector(folder/'vector.npz',v)
    error=float(np.linalg.norm(v-vectors['before' if kind=='WRAPPER' else 'after'])) if mode=='ORDINARY' else None
    result=dict(job=job,status='QUARANTINED_DIAGNOSTIC',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        vector=artifact,precision=e.precision(),schema=schema,ordinary_saved_vector_error=error,
        ordinary_reproduced=None if error is None else error<=c.scope()['saved_metric_and_ordinary_replay_atol'],
        wall_seconds=time.monotonic()-start,new_compile=0,new_sampling=0,acceptance_override=False)
    c.save(folder/'result.json',result)
    return result


def report(cfg, results):
    c.require(set(results)==set(c.JOBS),'EXACT_FOUR_REPLAY_RESULTS')
    vec={}
    for job,r in results.items():
        folder=Path(cfg['output'])/job
        c.require(c.read(folder/'result.json')==r and r['job']==job and r['SOURCE']==cfg['SOURCE'] and
            r['plan_sha256']==cfg['plan_sha256'],'RESULT_IDENTITY')
        c.require(c.sha(folder/'vector.npz')==r['vector']['sha256'],'RESULT_VECTOR_SHA')
        with np.load(folder/'vector.npz',allow_pickle=False) as z:
            c.require(z.files==['vector'],'RESULT_ARRAY_KEYS');v=z['vector']
        c.require(v.dtype==np.dtype(np.clongdouble if job.endswith('EXTENDED') else np.complex128),'RESULT_PRECISION')
        e.finite_vector(v,8192);vec[job]=v
        if job.endswith('ORDINARY'):
            c.require(r['ordinary_reproduced'] is True,'ORDINARY_REPLAY_DID_NOT_REPRODUCE')
    expected=c.read(cfg['inputs']['saved_diagnostics']['path'])['independent_expected_hadamard']
    return dict(status='DIAGNOSTIC_COMPLETE_NO_ACCEPTANCE_OVERRIDE',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        ordinary=e.comparison(vec['WRAPPER_ORDINARY'],vec['COMPILED_ORDINARY'],expected_hadamard=expected),
        extended=e.comparison(vec['WRAPPER_EXTENDED'],vec['COMPILED_EXTENDED'],expected_hadamard=expected,extended_metrics=True),
        precision_sensitivity={kind:e.comparison(vec[kind+'_ORDINARY'],vec[kind+'_EXTENDED'],extended_metrics=True) for kind in ('WRAPPER','COMPILED')},
        result_sha256={job:c.sha(Path(cfg['output'])/job/'result.json') for job in c.JOBS},
        interpretive_limit='SINGLE_SAVED_STATE_PRECISION_DIAGNOSTIC_NOT_CAUSAL_OR_UNITARY_CERTIFICATE',
        old_STOP_remains_STOP=True,accepted_new_cost_axes=0,sine_dispatch=0,accounting=False,
        phase_correction_applied=False,mandatory_STOP=True,next_phase_started=False)
