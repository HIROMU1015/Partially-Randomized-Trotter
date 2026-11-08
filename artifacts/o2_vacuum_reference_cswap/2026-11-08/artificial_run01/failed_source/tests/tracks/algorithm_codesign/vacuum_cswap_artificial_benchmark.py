"""Finite artificial full-wrapper comparison, no molecular inputs or sampling."""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import math
import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from trottertracks.algorithm_codesign import vacuum_reference_cswap as v
from trotterlib.research_direction_structure_pilot import _build_from_operations
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
from trottertracks.pr_dictionary_validation.gates import (
    compiled_wrapper, compile_cache_key, matrix_gate,
)
from test_vacuum_reference_cswap import initial, known_se, bit_probability

CASES=((1,2),(2,2),(2,12),(2,36),(3,2),(3,12),(3,36))
METHODS=('optimized_controlled','known_SE','PR_pathwise','known_cat_SE')
COMPILE_TOL=2e-10


def recorded(circuit):
    return dict(width=circuit.num_qubits,global_phase_hex=float(circuit.global_phase).hex(),
        operations=[dict(name=x.operation.name,qubits=[circuit.find_bit(q).index for q in x.qubits],
                         parameters=[float(p).hex() for p in x.operation.params])
                    for x in circuit.data])


def artificial_path(n,count):
    """Existing diagonal-only/source builder, with invented changing bases.

    Only n=1/count=2 is used for the trivial diagonal case. No long one-mode
    gate strings are used to manufacture a control-overhead advantage.
    """
    raw,controlled=[],[]
    for i in range(count):
        basis=QuantumCircuit(n)
        if n>1:
            basis.xx_plus_yy(.28+.03*(i%4),.11*(i%3),0,1)
        if n==3:
            basis.xx_plus_yy(-.19+.02*(i%5),.17,1,2)
        basis.rz(.12*(i%3-1),(i+1)%n)
        ops=[(x.operation,tuple(basis.find_bit(q).index for q in x.qubits)) for x in basis.data]
        support=tuple(sorted((i%n,(i+1)%n))) if n>1 and i%2 else (i%n,)
        angle=(-1)**i*(.2+.01*(i%7));scalar=.03*(i%3-1)
        for target,ctrl in ((raw,False),(controlled,True)):
            target.append(_build_from_operations(ops,num_system_qubits=n,
                support=support,angle=angle,controlled=ctrl,scalar_phase=scalar,
                control_policy='diagonal_only'))
    R,L=QuantumCircuit(n),QuantumCircuit(n)
    for c in raw[:count//2]:R.compose(c,inplace=True)
    for c in raw[count//2:]:L.compose(c,inplace=True)
    return L,R,controlled


def controlled_wrapper(blocks,n,axis):
    c=QuantumCircuit(n+1);c.h(n)
    for block in blocks:c.compose(block,inplace=True)
    if axis=='Y':c.sdg(n)
    c.h(n)
    bit=ClassicalRegister(1,'controlled_signal');c.add_register(bit);c.measure(n,bit[0])
    return c


def body(circuit):
    """Remove final signal readout and initial vacuum resets, WITHOUT RNG."""
    out=QuantumCircuit(circuit.num_qubits);out.global_phase=circuit.global_phase
    touched=set()
    for item in circuit.data:
        qs=[circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name=='reset':
            if set(qs)&touched:raise AssertionError('reset after nontrivial operation not audited')
            continue
        if item.operation.name=='measure':
            continue
        if item.clbits:raise AssertionError('unexpected dynamic instruction')
        touched.update(qs);out.append(item.operation,qs)
    return out


def compile_audit(raw,compiled,n):
    """All target basis columns, actual routing permutation, no phase fitting."""
    left,right=body(raw),body(compiled)
    perm=(compiled.layout.final_index_layout(filter_ancillas=True)
          if compiled.layout is not None else list(range(compiled.num_qubits)))
    if len(perm)!=raw.num_qubits or compiled.num_qubits!=raw.num_qubits:
        raise AssertionError('untracked routing ancillas')
    physical_indices=np.array([sum(((i>>q)&1)<<perm[q] for q in range(len(perm)))
                               for i in range(1<<len(perm))])
    maximum=0.
    for j in range(1<<n):
        psi=np.eye(1<<n,dtype=complex)[:,j]
        init=initial(psi,raw.num_qubits,range(n))
        ref=Statevector(init).evolve(left).data
        actual=Statevector(init).evolve(right).data[physical_indices]
        error=matrix_gate(actual,ref,COMPILE_TOL,label='COMPILED_ALLOWED_INPUT_ISOMETRY')
        maximum=max(maximum,error['max_entry'])
    return dict(max_entry_error=maximum,tolerance=COMPILE_TOL,phase_fitting=False,
                target_basis_columns=1<<n,final_logical_to_physical=perm,
                initial_reference_resets_removed_only_for_unitary_audit=True)


def metrics(c):
    counts=dict(c.count_ops())
    return dict(qubits=c.num_qubits,rz_count=counts.get('rz',0),cx_count=counts.get('cx',0),
        rz_depth=c.depth(lambda x:x.operation.name=='rz'),
        cx_depth=c.depth(lambda x:x.operation.name=='cx'),total_depth=c.depth(),
        reset_count=counts.get('reset',0),measure_count=counts.get('measure',0),
        circuit_size=c.size(),native_gate_counts=counts,
        T_count='UNKNOWN_NO_CLIFFORD_T_SYNTHESIS',
        physical_reset_readout_latency='UNKNOWN')


def run(output):
    import json
    profile=CompilerProfile();cache={};route_cache={};rows=[];inputs=[]
    calls=dict(native_compile=0,routing_compile=0,native_cache_hits=0,routing_cache_hits=0)
    start=time.perf_counter()
    for n,count in CASES:
        L,R,blocks=artificial_path(n,count)
        case=f'artificial_n{n}_blocks{count}'
        inputs.append(dict(case=case,L=recorded(L),R=recorded(R),
            split='existing block boundary at count/2; R first, L last',
            source_control_policy='diagonal_only',law='deterministic artificial path'))
        for axis in ('X','Y'):
            candidate=v.build_instrument(L,R,axis=axis,measure=True,initialize_reference=True)
            circuits={
                'optimized_controlled':controlled_wrapper(blocks,n,axis),
                'known_SE':known_se(L,R,axis,initialize_reference=True),
                'PR_pathwise':candidate.circuit,
                'known_cat_SE':v.build_instrument(L,R,axis=axis,measure=True,fanout=True,
                                                  initialize_reference=True).circuit}
            # Fair baseline: identical explicit gadget must have identical IR.
            context=dict(scope='invented_vacuum_benchmark',case=case,axis=axis)
            keys={m:compile_cache_key(c,profile,context) for m,c in circuits.items()}
            if keys['known_SE']!=keys['PR_pathwise']:
                raise AssertionError('known SE and candidate IR differ unexpectedly')
            # Independent check of each method's signal before compilation.
            psi=np.array([complex(j+1,(-1)**j) for j in range(1<<n)])
            psi/=np.linalg.norm(psi);p_ctrl=bit_probability(
                Statevector(initial(psi,n+1,range(n))).evolve(body(circuits['optimized_controlled'])).data,n)
            for method,raw in circuits.items():
                a=n if method=='optimized_controlled' else 2*n
                vec=Statevector(initial(psi,raw.num_qubits,range(n))).evolve(body(raw)).data
                if abs(bit_probability(vec,a)-p_ctrl)>COMPILE_TOL:
                    raise AssertionError('raw full-wrapper signal mismatch')
                key=keys[method];warm=key in cache;t=time.perf_counter()
                native=compiled_wrapper(raw,profile,cache=cache,identity_context=context)
                elapsed=time.perf_counter()-t
                calls['native_cache_hits' if warm else 'native_compile']+=1
                for topology in ('all_to_all','line_fixed_layout'):
                    routing_warm=False;route_seconds=0.
                    if topology=='all_to_all':
                        compiled=native
                    else:
                        routing_warm=key in route_cache;t=time.perf_counter()
                        if not routing_warm:
                            route_cache[key]=transpile(native,basis_gates=list(profile.basis_gates),
                                optimization_level=profile.optimization_level,
                                seed_transpiler=profile.seed_transpiler,
                                approximation_degree=profile.approximation_degree,
                                qubits_initially_zero=False,
                                coupling_map=CouplingMap.from_line(raw.num_qubits),
                                initial_layout=list(range(raw.num_qubits)),
                                layout_method='trivial',routing_method='sabre')
                        compiled=route_cache[key].copy();route_seconds=time.perf_counter()-t
                        calls['routing_cache_hits' if routing_warm else 'routing_compile']+=1
                    audit=compile_audit(raw,compiled,n)
                    row=dict(case=case,system_qubits=n,blocks=count,axis=axis,method=method,
                        topology=topology,metrics=metrics(compiled),audit=audit,
                        signal_probability_zero=bit_probability(vec,a),
                        control_policy='diagonal_only' if method=='optimized_controlled' else 'uncontrolled_lanes',
                        raw_cswap_count=raw.count_ops().get('cswap',0),
                        native_compile_seconds=elapsed,native_cache_hit=warm,
                        routing_compile_seconds=route_seconds,routing_cache_hit=routing_warm,
                        phase_ledger=None if method=='optimized_controlled' else candidate.ledger,
                        cache_key=key)
                    rows.append(row)
                    # Incremental evidence; a technical failure preserves completed rows.
                    (output/'BENCHMARK_RESULTS.json').write_text(json.dumps(dict(
                        status='IN_PROGRESS',profile=asdict(profile),calls=calls,rows=rows),indent=2)+'\n')
    # Standalone Fredkin decomposition under the same profile, not an FT T claim.
    fredkin=QuantumCircuit(3);fredkin.cswap(0,1,2)
    t=time.perf_counter();compiled=compiled_wrapper(fredkin,profile)
    calls['native_compile']+=1
    matrix_gate(__import__('qiskit.quantum_info',fromlist=['Operator']).Operator(compiled).data,
                __import__('qiskit.quantum_info',fromlist=['Operator']).Operator(fredkin).data,
                COMPILE_TOL,label='FREDKIN_COMPILE')
    primitive=dict(profile=asdict(profile),metrics=metrics(compiled),
                   compile_seconds=time.perf_counter()-t)
    (output/'CSWAP_DECOMPOSITION.json').write_text(json.dumps(primitive,indent=2)+'\n')
    (output/'ARTIFICIAL_INPUTS.json').write_text(json.dumps(dict(
        benchmark_paths=inputs,correctness_inputs='exact deterministic fixtures in hashed test source',
        two_path_QPE_counterexample='U and U^2, noncommuting split; see TEST_RESULTS'),indent=2)+'\n')
    report=dict(status='PASS',profile=asdict(profile),cells=len(rows),calls=calls,rows=rows,
        elapsed_seconds=time.perf_counter()-start,
        compile_pipeline='existing phase-preserving compiled_wrapper; optional second native routing pass for line',
        state_preparation_cost_excluded=True,reference_and_fanout_resets_included=True,
        classical_phase_path_cost='generation/compile seconds measured; online execution UNKNOWN',
        old_cache_reused=False,scientific_input_files=[])
    (output/'BENCHMARK_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
