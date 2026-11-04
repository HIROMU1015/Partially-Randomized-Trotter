"""Corrected one-shot A v2 runner. Default --describe reads metadata only.

--run selects scientific execution; it is not evidence of user permission.
This preparation task must not invoke --run or prepare_context on H4.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__=='__main__':
    for var in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[var]='1'  # This job only, before NumPy initializes.

import numpy as np
import fresh_iid_adapter as a

ROOT = a.ROOT
STATE_PATH = "artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz"
STATE_SHA = "793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4"
B_PATH = "artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/RESULT.json"
B_SHA = "b9b4615ed862a7a85dc6790692f8632d0997d53a1ad36ad59788572a64a51283"
H_PATH = "artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
H_FILE_SHA = "13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b"
H_SHA = "56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5"
STATE_FP = "a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90"
PREPARATION_SHA = "89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919"
LEGACY_PATHS = ("df_hamiltonian.py","df_partial_randomized_pf.py","df_partial_s2.py",
    "rte.py","rte_connected_cluster_cost_validation.py","finite_rte_signal_validation.py",
    "rpe_hadamard_failure_validation.py","research_direction_full_scope.py",
    "df_rte_tail.py","df_trotter/ops.py","df_trotter/decompose.py","df_trotter/model.py")


def describe():
    return {"status":"PREPARED_CORRECTED_NOT_SCIENTIFICALLY_EXECUTED", "coefficient_atol":1e-12, "inputs":[asdict(s) for s in a.fixed_axis_inputs()],
            "seed_namespace_public":a.SEED_NAMESPACE,"PRNG":"numpy.random.PCG64",
            "trajectory_calls":1928,"conditional_signal_calls":1928,"virtual_shots":1928,
            "event_count":8284,"ground_state_solves":0,"controlled_or_measurement_builds":0,
            "compiles":0,"diagnostic_comparisons":12,"diagnostic_family_alpha":.05,
            "equivalence_if_no_discrepancy":"UNDETERMINED","mandatory_STOP":True}


def source_and_input_identity():
    paths = ((a.B_RESULT,STATE_PATH,STATE_SHA),(a.B_RESULT,B_PATH,B_SHA),(a.CARRIER,H_PATH,H_FILE_SHA))
    records=[]
    for commit,path,expected in paths:
        raw=(ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected or raw!=a.fixed_bytes(commit,path):
            raise ValueError("Fixed input byte identity mismatch: "+path)
        records.append({"path":path,"commit":commit,"sha256":expected})
    for name in LEGACY_PATHS:
        path="src/trotterlib/"+name
        if (ROOT/path).read_bytes()!=a.fixed_bytes(a.CARRIER,path):
            raise ValueError("Legacy numerical SOURCE mismatch: "+path)
    return records


def restricted_basis(operations,n_qubits,indices,permutation):
    """Compose existing gate operators numerically; no new trajectory circuit."""
    from qiskit.quantum_info import Operator
    operator=Operator(np.eye(1<<n_qubits,dtype=np.complex128))
    for gate,qubits in operations:
        operator=operator.compose(Operator(gate),qargs=list(qubits))
    matrix=np.asarray(operator.data)
    inside=np.asarray(indices,dtype=np.int64)
    outside=np.ones(1<<n_qubits,dtype=bool);outside[inside]=False
    leakage=float(np.linalg.norm(matrix[np.ix_(np.flatnonzero(outside),inside)],ord="fro"))
    if leakage>a.MATRIX_ATOL:
        raise ValueError("Basis leaks from number sector")
    ordered=inside[permutation]
    return matrix[np.ix_(ordered,ordered)],leakage


def diagonal_from_primitives(primitives,ordered_indices):
    """Same global/RZ/RZZ phases as legacy append_diagonal_primitives."""
    phase=np.full(len(ordered_indices),float(primitives.global_phase))
    z=lambda q:1.-2.*((ordered_indices>>q)&1)
    for q,angle in primitives.rz:phase-=float(angle)*z(q)/2
    for p,q,angle in primitives.rzz:phase-=float(angle)*z(p)*z(q)/2
    return np.exp(1j*phase)


def numeric_half_operators(preparation,basis_fn,primitive_fn,ordered_indices):
    blocks=[]
    for block in preparation.deterministic_blocks:
        basis=basis_fn(block.basis_id,block.basis_hash,block.runtime_basis_operations)
        diagonal=diagonal_from_primitives(primitive_fn(block),ordered_indices)
        blocks.append((basis*diagonal[np.newaxis,:])@basis.conj().T)
    dimension=len(ordered_indices)
    scalar=np.exp(-.02j*(preparation.constant_coefficient+preparation.extracted_identity_coefficient))
    forward=np.eye(dimension,dtype=complex)*scalar
    reverse=np.eye(dimension,dtype=complex)
    for block in blocks:forward=block@forward
    for block in reversed(blocks):reverse=block@reverse
    return forward,reverse


def prepare_context():
    """New H4 numerical preparation: requires the separate scientific permission."""
    identity=source_and_input_identity()
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit')}
    if versions['qiskit']!='1.3.0':raise ValueError('Qiskit 1.3.0 required for the registered basis definitions')
    sys.path.insert(0,str(ROOT/"src"))
    from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot
    from trotterlib.df_hamiltonian import PhysicalSector
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash,split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    from trotterlib.research_direction_full_scope import _finite_inputs
    from trotterlib.finite_rte_signal_validation import _qiskit_to_openfermion_sector_permutation,_array_fingerprint
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives,df_squared_diagonal_primitives
    hamiltonian=load_connected_cluster_hamiltonian_snapshot(ROOT/H_PATH)
    if df_hamiltonian_hash(hamiltonian)!=H_SHA or hamiltonian.n_qubits!=8 or hamiltonian.n_blocks!=12:
        raise ValueError("Fixed H/model mismatch")
    with np.load(ROOT/STATE_PATH,allow_pickle=False) as saved:
        psi=np.asarray(saved["state"],dtype=np.complex128)
        sector=PhysicalSector.number_sector(n_qubits=8,n_electrons=4)
        if not np.array_equal(saved["basis_indices"],sector.basis_indices) or str(saved["hamiltonian_hash"])!=H_SHA:
            raise ValueError("State sector/H mismatch")
    if _array_fingerprint(psi)!=STATE_FP or abs(np.linalg.norm(psi)-1)>a.MATRIX_ATOL:
        raise ValueError("Saved B physical state mismatch")
    prep=prepare_df_partial_s2(hamiltonian,split_df_hamiltonian_by_ld(hamiltonian,3),
                               identity_policy="extract_identity_phase",coefficient_atol=1e-12)
    if (prep.preparation_hash!=PREPARATION_SHA or len(prep.rte_preparation.symbolic_tail.components)!=218
            or len(prep.deterministic_blocks)>4):
        raise ValueError("Saved preparation/component inventory mismatch: "
                         f"actual preparation_hash={prep.preparation_hash}; "
                         f"components={len(prep.rte_preparation.symbolic_tail.components)}; "
                         f"deterministic_blocks={len(prep.deterministic_blocks)}; coefficient_atol=1e-12")
    config,distribution=_finite_inputs(prep,delta_time=.02,rte_steps=1,finite_taylor_order=2)
    permutation=_qiskit_to_openfermion_sector_permutation(sector)
    ordered=sector.basis_indices[permutation]
    cache={};maximum_leakage=0.
    def basis_fn(basis_id,basis_hash,operations):
        nonlocal maximum_leakage
        key=(basis_id,basis_hash)
        if key not in cache:
            cache[key],leakage=restricted_basis(operations,8,sector.basis_indices,permutation)
            maximum_leakage=max(maximum_leakage,leakage)
        return cache[key]
    def primitive_fn(block):
        if block.block_kind=="one_body":
            return one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues),.02/2)
        return df_squared_diagonal_primitives(np.asarray(block.diagonal_eta),block.lam,.02/2)
    forward,reverse=numeric_half_operators(prep,basis_fn,primitive_fn,ordered)
    operators={}
    for component in prep.rte_preparation.symbolic_tail.components:
        definition=prep.tail_extraction.basis_definition(component.basis_id)
        basis=basis_fn(component.basis_id,component.basis_hash,definition.runtime_operations)
        parity=np.zeros(len(ordered),dtype=np.int64)
        for q in component.diagonal_pauli_support:parity^=(ordered>>q)&1
        signs=component.coefficient_sign*(1.-2.*parity)
        operators[component.component_id]=(basis*signs[np.newaxis,:])@basis.conj().T
    b=json.loads((ROOT/B_PATH).read_text())
    if b["status"]!="LOCAL_FIXED_REQUEST_MEANS_CONSISTENT_MANDATORY_STOP":
        raise ValueError("B evidence status changed")
    payload={"psi":psi,"forward":forward,"reverse":reverse,"operators":operators,
             "components":[asdict(c) for c in prep.rte_preparation.symbolic_tail.components],
             "distribution":distribution.to_dict()}
    provenance={"fixed_inputs":identity,"state_fingerprint":STATE_FP,"basis_cache_entries":len(cache),
        "package_versions":versions,
        "maximum_basis_sector_leakage":maximum_leakage,"preparation_hash":prep.preparation_hash,
        "tail_hash":prep.tail_extraction.tail_hash,"config":asdict(config),
        "component_count":len(operators),"half_construction":"legacy primitives and numeric composition of registered basis operations; no scalar double correction",
        "basis_definition_note":"Legacy DF preparation constructs Gaussian gate definitions internally; no controlled evolution or Hadamard circuits are built"}
    return payload,provenance


def restore_payload(payload):
    rte=a.legacy_rte()
    components=[]
    for raw in payload["components"]:
        raw=dict(raw)
        raw["basis_change_operations"]=tuple(rte.BasisChangeOperation(**op) for op in raw["basis_change_operations"])
        components.append(rte.RTEComponent(**raw))
    context={**payload,"components":tuple(components),
             "distribution":rte.RTEFiniteDistribution.from_dict(payload["distribution"])}
    for name in ("psi","forward","reverse"):context[name].setflags(write=False)
    for matrix in context["operators"].values():matrix.setflags(write=False)
    return context


_CONTEXT=None
def initialize_worker(payload):
    global _CONTEXT
    _CONTEXT=restore_payload(payload)


def shot_chunk(job,*,context=None,namespace=a.SEED_NAMESPACE):
    context=_CONTEXT if context is None else context
    spec,indices=job
    if len(set(indices))!=len(indices):raise ValueError("Duplicate shot request")
    scalar=a.legacy_functions()["_trajectory_signal"]
    def signal(events,q,r):
        return scalar(initial_state=context["psi"],forward=context["forward"],reverse=context["reverse"],
                      events=events,operators=context["operators"],q_m=q,rte_steps=r)
    return [a.one_shot(spec,i,components=context["components"],distribution=context["distribution"],
                       signal_fn=signal,namespace=namespace) for i in indices]


def execute(output):
    output.mkdir(parents=True,exist_ok=False)  # Fresh output, no resume/retry.
    started=time.monotonic();record={"execution_status":"STARTED","mandatory_STOP":True,
        "SOURCE_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"]).decode().strip(),
        "seed_namespace_public":a.SEED_NAMESPACE,"RNG":"PCG64; legacy _seed_for_shot with role-separated public master derivation",
        "ground_state_solves":0,"controlled_wrapper_builds":0,"compiles":0,"price_evaluations":0}
    try:
        inputs=a.fixed_axis_inputs();a.check_stream_keys(inputs)
        record['fixed_axis_inputs']=[asdict(s) for s in inputs]
        record['saved_mean_source']={'commit':a.V2,'path':'docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv',
                                    'selection':'ld3, m0/q1 and m3/q8, physical attenuated_event_mean_signal'}
        record['sampler_SOURCE_carrier']=a.CARRIER
        payload,provenance=prepare_context();record["context"]=provenance
        available=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
        arrays=[payload[n] for n in ("psi","forward","reverse")]+list(payload["operators"].values())
        array_bytes=sum(x.nbytes for x in arrays)
        meminfo=Path('/proc/meminfo').read_text() if Path('/proc/meminfo').exists() else ''
        memory=next((int(line.split()[1])*1024 for line in meminfo.splitlines() if line.startswith('MemAvailable:')),None)
        workers=min(available,sum(s.shots for s in inputs))
        if memory is not None and workers*array_bytes>memory:
            workers=max(1,int(memory//array_bytes))
        record["parallelism"]={"workers":workers,"available_CPUs":available,"context_array_bytes_per_worker":array_bytes,
                              "available_memory_bytes":memory,"BLAS_threads_per_worker":1,"start_method":"spawn"}
        size=max(1,math.ceil(max(s.shots for s in inputs)/workers))
        jobs=[(s,tuple(range(start,min(start+size,s.shots)))) for s in inputs for start in range(0,s.shots,size)]
        rows=[]
        with (output/"shots.jsonl").open('x') as raw:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_worker,initargs=(payload,)) as pool:
                futures=[pool.submit(shot_chunk,job) for job in jobs]
                try:
                    for future in as_completed(futures):
                        completed=future.result();rows.extend(completed)
                        for row in completed:raw.write(json.dumps(row,allow_nan=False)+'\n')
                        raw.flush()
                        record['harvested_shots']=len(rows)
                except BaseException:
                    for future in futures:future.cancel()
                    # Running bounded chunks can finish; cancelled queued chunks
                    # are not replaced. Partial data receive no scientific test.
                    raise
        record["axes"]=[a.summarize_axis(s,[r for r in rows if r['q']==s.q and r['axis']==s.axis]) for s in inputs]
        record["scientific_status"]="INCONSISTENT" if any(r["scientific_status"]=="INCONSISTENT" for r in record["axes"]) else "UNDETERMINED"
        record["execution_status"]="COMPLETED_MANDATORY_STOP"
        record["completed_trajectory_calls"]=len(rows);record["completed_virtual_shots"]=len(rows)
        record["completed_events"]=sum(len(r['events']) for r in rows)
    except BaseException as exc:
        record["execution_status"]="INTERRUPTED_OR_INVALID_MANDATORY_STOP"
        record["scientific_status"]="UNDETERMINED"
        record["error"]={"type":type(exc).__name__,"message":str(exc).replace(str(ROOT),'<repository>')}
        raise
    finally:
        record["wall_seconds"]=time.monotonic()-started
        (output/"RESULT.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true',help='Scientific action selection, only after separate permission')
    mode.add_argument('--describe',action='store_true',help='Metadata only (default)')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(),ensure_ascii=False,indent=2));return
    if args.output_dir is None:parser.error('--run requires --output-dir')
    execute(args.output_dir)


if __name__=='__main__':
    main()
