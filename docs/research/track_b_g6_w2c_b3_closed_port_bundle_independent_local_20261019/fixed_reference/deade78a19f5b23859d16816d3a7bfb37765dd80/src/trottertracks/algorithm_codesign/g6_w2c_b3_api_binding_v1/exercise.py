"""Fixed pair-engine port exerciser, NOT a second Global Bundle implementation.

No call/storage admission scheduler is implemented here. Global storage/STOP
acceptance is separately delegated to the unchanged fixed bundle's own harness.
"""
from contextlib import nullcontext
from pathlib import Path
import multiprocessing as mp
import os
import resource
import time
from .. import g6_w2c_b3_parallel_scheduler as P
from .. import g6_w2c_b3_bundle_contract_v1 as K
from . import static as S
from .ports import ArtificialEndpoint, ArtificialBits, install_guard, forbidden_imports
from .authority import ArtificialAuthority, approval
D=P.D

def write(path,value):
    with Path(path).open('xb',buffering=0) as f:
        f.write(P.safe_json(value)+b'\n')
        f.flush()

def pair(job):
    task,c,out,fault=job
    install_guard()
    S.preflight(c,c['runtime_root'])
    endpoint=ArtificialEndpoint(task,run=Path(out).parent.name,fault=fault)
    meter=D.Meter(D.clock)
    observer=P.CachedObserver([str(Path(c['runtime_root'])/p) for p in c['source_pins'] if p.endswith('.py')])
    D.SERIALIZATION_MODE='fused'
    before=D.clock()
    io=P.own_io()
    stream=None
    outcome=failure=None
    try:
        with observer if task['pass_name']=='instrumented' else nullcontext():
            stream=D.Stream(Path(out)/(task['key']+'.jsonl'),meter,byte_limit=1<<36,flush_rows=4096)
            outcome=D.run_pair(endpoint,lambda axis:ArtificialBits(task['arm'],axis),
                [dict(arm=task['arm'],layer=task['layer'],axis=axis,N=task['n']) for axis in ('X','Y')],
                stream,meter,phase=task['phase'],pass_name=task['pass_name'],correctness_trials=16)
            stream.close()
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),
                     trial_failure_provenance=getattr(exc,'b3_failure_provenance',None))
        if stream and not stream.closed:
            try:stream.close()
            except BaseException as close:failure['close_error']=str(close)
    total=D.delta(before,D.clock())
    row=dict(task=task,status='FAILED' if failure else 'OK',outcome=outcome,failure=failure,
             pid=os.getpid(),cpu_ns=total[0],wall_ns=total[1],coarse=meter.rows,
             reconciliation=meter.reconcile(total),fine=observer.meter.rows if task['pass_name']=='instrumented' else None,
             fine_reconciliation=observer.meter.reconcile(total) if task['pass_name']=='instrumented' else None,
             memory=D.memory_snapshot(),io={k:v-io.get(k,0) for k,v in P.own_io().items()},
             output_bytes=stream.bytes if stream else 0,output_rows=stream.rows if stream else 0,
             chain=stream.previous if stream else None,file_sha=stream.file_hash.hexdigest() if stream else None,
             forbidden_imports=forbidden_imports(),guard_installed=True)
    write(Path(out)/(task['key']+'.port.json'),dict(task=task,records=endpoint.records))
    write(Path(out)/(task['key']+'.worker.json'),row)
    return row

def run(c,root,*,n=256,fault_mode=None,fault_stage=''):
    install_guard()
    S.preflight(c,c['runtime_root'])
    S.need(type(n) is int and 16<=n<=256, 'ARTIFICIAL_PROFILE_RANGE')
    root=Path(root)
    out=root/'ports'
    out.mkdir()
    authority=ArtificialAuthority(c,root)
    b=authority.expected
    authority.claim(b,approval(b))
    write(root/'authority_binding.json',b)
    tasks,total=K.plan(n)
    start=D.clock()
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    rows=[]
    for phase in range(6):
        batch=[t for t in tasks if t['phase_index']==phase]
        jobs=[(t,c,str(out),fault_stage if t['condition']==fault_mode and t['pair_index']==1 else '') for t in batch]
        # All test workers install the import guard before constructing endpoints.
        with mp.get_context('spawn').Pool(1 if phase<3 else 8,maxtasksperchild=1) as pool:
            current=pool.map(pair,jobs,chunksize=1)
        rows.extend(current)
        if any(x['status']!='OK' for x in current):break
    authority.stop(b)
    write(root/'fake_snapshot.json',authority.snapshot())
    elapsed=D.delta(start,D.clock())
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(schema='b3_ARTIFICIAL_port_exercise_v1',candidate_digest=S.digest(c),
                run_root=str(root),
                measured_SOURCE=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=c['runtime_root'],text=True).strip(),
                n=n,planned=total,rows=rows,complete=len(rows)==48 and all(x['status']=='OK' for x in rows),
                global_bundle_equivalence=False,scientific_GO=False,mandatory_STOP=True,
                forbidden_imports=forbidden_imports(),outer=dict(parent_cpu_ns=elapsed[0],wall_ns=elapsed[1],
                    all_worker_cpu_ns=round((after.ru_utime+after.ru_stime-children.ru_utime-children.ru_stime)*1e9)),
                fixed_bundle_injection='UNIMPLEMENTED_IN_IMMUTABLE_BUNDLE',
                post_close='caller observes all pair streams and report; fixed bundle independently checks Storage V2')
    write(root/'port_report.json',report)
    write(root/'port_caller.json',dict(schema='b3_ARTIFICIAL_port_caller_v1',
        report_sha256=S.sha((root/'port_report.json').read_bytes()),report_write_flush_close_observed=True,
        forbidden_imports=forbidden_imports(),candidate_digest=S.digest(c)))
    return report
