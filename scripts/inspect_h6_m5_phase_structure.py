"""Read two saved QPY structures; never evaluate gates or sum real angles.

Scope is serialized metadata, exact binary64 representations, integer counts
and source lineage. No state file, simulator, transpiler or numerical ledger.
"""
from collections import Counter, deque
import concurrent.futures as futures
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import struct
import time

ALLOWED = {'WRAPPER': {'h','p','rz','rzz','cx','xx_plus_yy','measure'},
           'COMPILED': {'rz','sx','x','cx','measure'}}


def require(ok, reason):
    if not ok:raise ValueError(reason)


def numeric_bits(value):
    require(type(value) in (float,int) and math.isfinite(value),'FINITE_SERIALIZED_FLOAT_PARAMETER_ONLY')
    return struct.pack('>d',float(value)).hex()


def opposite_bits(a,b):
    return int(a,16) ^ int(b,16) == 1 << 63


def summarize(qc,kind,expected_qubits):
    require(kind in ALLOWED,'FIXED_CIRCUIT_KIND')
    require(qc.num_qubits==expected_qubits and qc.num_clbits==1 and qc.layout is None,'FIXED_QUBITS_LAYOUT')
    require(not qc.parameters,'NO_SYMBOLIC_PARAMETERS')
    counts=Counter();locations=Counter();pairs=Counter();head=[];tail=deque(maxlen=12)
    bit_types=Counter();adjacent_opposite_P=0;previous=None;measurement=False
    structural=hashlib.sha256();pstream=hashlib.sha256()
    for index,item in enumerate(qc.data):
        op=item.operation;name=op.name
        require(name in ALLOWED[kind],'UNKNOWN_SAVED_GATE:'+name)
        qs=tuple(qc.find_bit(q).index for q in item.qubits)
        cs=tuple(qc.find_bit(q).index for q in item.clbits)
        bits=[numeric_bits(p) for p in op.params]
        bit_types.update(type(p).__name__ for p in op.params)
        if name=='measure':
            require(not measurement and qs==(expected_qubits-1,) and cs==(0,),'TERMINAL_ANCILLA_MEASUREMENT')
            measurement=True
        else:require(not measurement,'NO_MID_MEASUREMENT')
        counts[name]+=1;locations[(name,qs)]+=1
        if len(qs)==2:pairs[qs]+=1
        row=dict(index=index,name=name,qubits=list(qs),clbits=list(cs),parameter_binary64_hex=bits)
        encoded=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode()
        structural.update(encoded)
        if name=='p':
            pstream.update(encoded)
            if previous and previous['name']=='p' and previous['qubits']!=row['qubits'] and opposite_bits(previous['parameter_binary64_hex'][0],bits[0]):
                adjacent_opposite_P+=1
        if len(head)<12:head.append(row)
        tail.append(row);previous=row
    require(measurement,'SAVED_MEASUREMENT_PRESENT')
    return dict(kind=kind,nqubits=qc.num_qubits,ancilla=expected_qubits-1,layout='None',
        global_phase_serialized=float(qc.global_phase),global_phase_binary64_hex=numeric_bits(qc.global_phase),
        operation_count=len(qc.data),gate_counts=dict(sorted(counts.items())),parameter_types=dict(bit_types),
        per_location_counts=[dict(gate=g,qubits=list(qs),count=n) for (g,qs),n in sorted(locations.items())],
        two_qubit_location_counts=[dict(qubits=list(qs),count=n) for qs,n in sorted(pairs.items())],
        adjacent_P_opposite_binary64_on_distinct_qubits=adjacent_opposite_P,
        ordered_metadata_sha256=structural.hexdigest(),ordered_P_metadata_sha256=pstream.hexdigest(),
        head=head,tail=list(tail),gate_matrices_evaluated=0,statevector_replays=0,
        phase_parameter_sums_computed=False,determinant_ledger_evaluated=False,
        gate_definitions_expanded=False,acceptance_override=False)


def inspect_job(job):
    """Deserialize only the fixed QPY role; byte checks bracket the metadata scan."""
    from qiskit import qpy
    import h6_m5_phase_contract as c
    start=time.monotonic();kind,row,qubits=job
    require(c.sha(row['path'])==row['sha256'],'QPY_INPUT_SHA')
    with Path(row['path']).open('rb') as f:
        circuits=qpy.load(f);require(f.read()==b'','QPY_TRAILING_BYTES')
    require(len(circuits)==1,'ONE_SAVED_CIRCUIT')
    result=summarize(circuits[0],kind,qubits)
    require(c.sha(row['path'])==row['sha256'],'QPY_UNCHANGED_AFTER_INSPECTION')
    result.update(input_sha256=row['sha256'],wall_seconds=time.monotonic()-start)
    return result


def inspect_pair(cfg):
    """Two independent metadata reads, not two new scientific experiments."""
    import h6_m5_phase_contract as c
    c.verify_seals(cfg)
    expected=c.read(Path(cfg['run_root'])/'QPY_schema_gate.json')
    jobs=[(kind,cfg['inputs'][kind.lower()+'_qpy'],13) for kind in ('WRAPPER','COMPILED')]
    pool=futures.ProcessPoolExecutor(max_workers=len(jobs),mp_context=mp.get_context('spawn'))
    processes=[]
    try:
        tasks={pool.submit(inspect_job,job):job[0] for job in jobs}
        processes=list(pool._processes.values())
        result={tasks[f]:f.result() for f in futures.as_completed(tasks)}
    finally:pool.shutdown(wait=True,cancel_futures=True)
    require(all(p.exitcode==0 and not p.is_alive() for p in processes),'STATIC_INSPECTION_EXIT_AND_REAP')
    for kind,row in result.items():
        require(row['gate_counts']==expected[kind]['gate_counts'] and row['global_phase_serialized']==expected[kind]['global_phase'],'PREVIOUS_STRUCTURE_IDENTITY')
    return dict(status='STATIC_METADATA_ONLY',circuits=result,workers=len(processes),all_reaped=True,
        scientific_grants_consumed=0,new_compile=0,new_statevector_replays=0,phase_parameter_sums_computed=False)
