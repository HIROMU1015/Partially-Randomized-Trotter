"""Bounded native DF circuit lowering; no molecular I/O, sampling or compile.

Qiskit system ordering, with the ancilla immediately after the system.
Prepared block specs and explicitly supplied ordinary-controlled RTE circuits
retain the legacy diagonal/global phase semantics. A finite mean is not a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate

from trotterlib.df_partial_s2 import (
    DFDeterministicOneBodySpec, DFDeterministicFragmentSpec,
    DFDeterministicBlockSpec, DFPartialS2StepRequest,
    QiskitDFPartialS2CircuitBuilder,
)
from trotterlib.df_trotter.ops import (
    DiagonalEvolutionPrimitives, append_diagonal_primitives,
    one_body_diagonal_primitives, df_squared_diagonal_primitives,
)
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.rpe_hadamard_interrogation import QiskitRPEHadamardInterrogationBuilder
from trotterlib.rte import require_integer_count
from .ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint

from .ax2a_control_plan import deterministic_control_plan


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _limit(value):
    return require_integer_count(value, name="max_instructions", minimum=1)


def _gate_only(circuit, n):
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits != n:
        raise ValueError("Circuit qubits do not match the declared register.")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("Evolution must be numeric with no classical bits.")
    _real(circuit.global_phase, "global_phase")
    if any(not isinstance(item.operation, Gate) for item in circuit.data):
        raise ValueError("Evolution must contain unitary Gates only.")


def validate_blocks(blocks, n):
    n = require_integer_count(n, name="num_system_qubits", minimum=1)
    for block in blocks:
        if not isinstance(block, (DFDeterministicOneBodySpec, DFDeterministicFragmentSpec)):
            raise TypeError("Expected an existing prepared DF deterministic block spec.")
        if block.num_system_qubits != n:
            raise ValueError("Block register size mismatch.")
        values = block.diagonal_eigenvalues if isinstance(block, DFDeterministicOneBodySpec) else block.diagonal_eta
        if len(values) != n:
            raise ValueError("Block diagonal dimension mismatch.")
        for value in values:
            _real(value, "diagonal coefficient")
        if isinstance(block, DFDeterministicFragmentSpec):
            _real(block.lam, "lambda")
        for gate, positions in block.runtime_basis_operations:
            if not isinstance(gate, Gate) or gate.is_parameterized():
                raise ValueError("Basis operations must be numeric unitary Gates.")
            indices = tuple(require_integer_count(i, name="basis index") for i in positions)
            if len(indices) != gate.num_qubits or len(set(indices)) != len(indices) or any(i >= n for i in indices):
                raise ValueError("Basis operation register mismatch.")
    return n


def block_instruction_bound(block, mode):
    """Conservative Qiskit instruction bound, not an expanded native-gate cost."""
    n = block.num_system_qubits
    pairs = 0 if isinstance(block, DFDeterministicOneBodySpec) else n*(n-1)//2
    basis = 2*len(block.runtime_basis_operations)
    if mode == "UNCONTROLLED":
        return basis + n + pairs
    if mode == "ORDINARY":
        return basis + n + pairs + 1
    if mode == "DIRECTIONAL":
        return basis + 3*n + 5*pairs + 1
    raise ValueError("Unknown control mode.")


def append_directional_diagonal(circuit, primitives: DiagonalEvolutionPrimitives, ancilla):
    """diag(D(-t),D(t)); parity rotations plus RZ(2*global_phase)."""
    for left, right, angle in primitives.rzz:
        if angle == 0:
            continue
        circuit.cx(left, right)
        circuit.cx(ancilla, right)
        circuit.rz(-angle, right)
        circuit.cx(ancilla, right)
        circuit.cx(left, right)
    for qubit, angle in primitives.rz:
        circuit.cx(ancilla, qubit)
        circuit.rz(-angle, qubit)
        circuit.cx(ancilla, qubit)
    if primitives.global_phase != 0:
        circuit.rz(2*primitives.global_phase, ancilla)


def _append_block(circuit, block, time, mode, ancilla):
    QiskitDFPartialS2CircuitBuilder._append_basis(circuit, block, inverse=True)
    if isinstance(block, DFDeterministicOneBodySpec):
        primitives = one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues), time)
    else:
        primitives = df_squared_diagonal_primitives(np.asarray(block.diagonal_eta), block.lam, time)
    if mode == "DIRECTIONAL":
        append_directional_diagonal(circuit, primitives, ancilla)
    else:
        append_diagonal_primitives(circuit, primitives, controlled=mode=="ORDINARY",
                                   ancilla_qubit=ancilla if mode=="ORDINARY" else None)
    QiskitDFPartialS2CircuitBuilder._append_basis(circuit, block, inverse=False)


@dataclass(frozen=True)
class NativeEvolution:
    circuit: QuantumCircuit
    num_system_qubits: int
    T: float
    q: int
    formula: str
    control_policy: str
    instruction_upper_bound: int
    circuit_fingerprint: str


def _finish(circuit, n, T, q, formula, policy, bound, cap):
    if len(circuit.data) > bound or len(circuit.data) > cap:
        raise RuntimeError("POST_BUILD_INSTRUCTION_BOUND")
    return NativeEvolution(circuit, n, T, q, formula, policy, bound,
                           canonical_qiskit_circuit_fingerprint(circuit))


def build_deterministic_native(blocks: Sequence[DFDeterministicBlockSpec], *,
                               num_system_qubits, T, q, formula, scalar,
                               control_policy, max_instructions):
    n = validate_blocks(blocks, num_system_qubits)
    T, scalar = _real(T,"T"), _real(scalar,"scalar")
    q = require_integer_count(q, name="q", minimum=1)
    cap = _limit(max_instructions)
    if formula not in ("2nd","4th") or control_policy not in ("ordinary","symmetric_directional"):
        raise ValueError("Unsupported formula or control policy.")
    # Bound before allocating the q-dependent schedule or a circuit.
    pieces = 1 if formula=="2nd" else 3
    modes = ("ORDINARY","ORDINARY") if control_policy=="ordinary" else ("UNCONTROLLED","DIRECTIONAL")
    bound = q*pieces*sum(block_instruction_bound(b, m) for b in blocks for m in modes)+1
    if bound > cap:
        raise RuntimeError("PRE_BUILD_INSTRUCTION_BUDGET")
    circuit = QuantumCircuit(n+1, name="ax2a_df_global_pf")
    for stage in deterministic_control_plan(len(blocks),T,q,formula) if blocks else ():
        mode = "ORDINARY" if control_policy=="ordinary" else stage.mode
        _append_block(circuit,blocks[stage.term],stage.time,mode,n)
    if scalar*T != 0:
        circuit.p(-scalar*T,n)
    return _finish(circuit,n,T,q,formula,control_policy,bound,cap)


def build_partial_native(blocks: Sequence[DFDeterministicBlockSpec], *,
                         num_system_qubits, T, q, scalar, controlled_tails,
                         control_policy, max_instructions):
    """B0/B2/B3 S2; supplied tails are already diag(I,U_event_sequence).

    Tail ordinary-control semantics must be certified by the caller/legacy
    RTE builder. No new control, normalization, event sampling or reordering.
    """
    n = validate_blocks(blocks,num_system_qubits)
    T, scalar = _real(T,"T"), _real(scalar,"scalar")
    q = require_integer_count(q,name="q",minimum=1)
    cap = _limit(max_instructions)
    if control_policy not in ("ordinary","symmetric_directional"):
        raise ValueError("Unsupported control policy.")
    if controlled_tails is not None and len(controlled_tails)!=q:
        raise ValueError("One explicit controlled tail per outer step is required.")
    tail_bound = 0
    if controlled_tails is not None:
        for tail in controlled_tails:
            _gate_only(tail,n+1)
            tail_bound += len(tail.data)
    modes = ("ORDINARY","ORDINARY") if control_policy=="ordinary" else ("UNCONTROLLED","DIRECTIONAL")
    bound = q*sum(block_instruction_bound(b,m) for b in blocks for m in modes)+tail_bound+1
    if bound > cap:
        raise RuntimeError("PRE_BUILD_INSTRUCTION_BUDGET")
    circuit = QuantumCircuit(n+1,name="ax2a_df_partial_s2")
    delta = T/q
    for index in range(q) if blocks or controlled_tails is not None else ():
        for block in blocks:
            _append_block(circuit,block,delta/2,modes[0],n)
        if controlled_tails is not None:
            circuit.compose(controlled_tails[index],inplace=True)
        for block in reversed(blocks):
            _append_block(circuit,block,delta/2,modes[1],n)
    if scalar*T != 0:
        circuit.p(-scalar*T,n)
    return _finish(circuit,n,T,q,"partial_2nd",control_policy,bound,cap)


def partial_native_from_step_requests(requests: Sequence[DFPartialS2StepRequest], *,
                                       control_policy, max_instructions):
    """Replay existing explicit RTE requests without calling any sampler."""
    from trotterlib.df_rte_qiskit import (
        QiskitDFRTEEventCircuitBuilder, estimate_df_rte_untranspiled_size_upper_bound,
    )
    if not requests or any(not isinstance(r,DFPartialS2StepRequest) for r in requests):
        raise ValueError("Explicit validated step requests are required.")
    first = requests[0]
    prep = first.preparation
    n = prep.num_system_qubits
    cap = _limit(max_instructions)
    if control_policy not in ("ordinary","symmetric_directional"):
        raise ValueError("Unsupported control policy.")
    if any(not r.controlled or r.ancilla_qubit!=n or
           r.preparation.preparation_hash!=prep.preparation_hash or
           r.step_time!=first.step_time for r in requests):
        raise ValueError("Step target/time/control identities do not match.")
    modes = ("ORDINARY","ORDINARY") if control_policy=="ordinary" else ("UNCONTROLLED","DIRECTIONAL")
    bound = len(requests)*sum(block_instruction_bound(b,m) for b in prep.deterministic_blocks for m in modes)+1
    bound += sum(estimate_df_rte_untranspiled_size_upper_bound(r.rte_occurrence)
                 for r in requests if r.rte_occurrence is not None)
    if bound>cap:
        raise RuntimeError("PRE_BUILD_INSTRUCTION_BUDGET")
    # Never use make_*_request or sample_occurrence_request here.
    builder = QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    tails = None if prep.is_deterministic_only else [
        builder.build_sequence(r.rte_occurrence,basis_plan=r.rte_basis_plan).circuit for r in requests]
    return build_partial_native(prep.deterministic_blocks,num_system_qubits=n,
        T=first.step_time*len(requests),q=len(requests),
        scalar=prep.constant_coefficient+prep.extracted_identity_coefficient,
        controlled_tails=tails,control_policy=control_policy,max_instructions=cap)


def make_native_block_action(block, *, max_instructions):
    """Full Qiskit-order vector callback; reuse legacy phase-aware evolution."""
    n = validate_blocks((block,),block.num_system_qubits)
    cap = _limit(max_instructions)
    if block_instruction_bound(block,"UNCONTROLLED")>cap:
        raise RuntimeError("PRE_BUILD_INSTRUCTION_BUDGET")
    def action(vector,time):
        value = np.asarray(vector,dtype=np.complex128)
        if value.shape!=(1<<n,) or not np.isfinite(value).all():
            raise ValueError("Expected finite full Qiskit-order vector.")
        circuit = QuantumCircuit(n)
        _append_block(circuit,block,_real(time,"time"),"UNCONTROLLED",None)
        # No normalization: finite corrected means can have non-unit norm.
        return simulate_statevector(circuit,value)
    return action


def build_native_hadamard_wrapper(evolution: NativeEvolution, *, axis,
                                  include_measurement=True, max_instructions):
    """Identical legacy H/U/(Sdg)/H/measure scope; no repeated-record spoofing."""
    if not isinstance(evolution,NativeEvolution) or axis not in ("cosine","sine"):
        raise ValueError("Native evolution and cosine/sine axis required.")
    if type(include_measurement) is not bool:
        raise ValueError("include_measurement must be boolean.")
    n = evolution.num_system_qubits
    _gate_only(evolution.circuit,n+1)
    if canonical_qiskit_circuit_fingerprint(evolution.circuit)!=evolution.circuit_fingerprint:
        raise ValueError("Evolution circuit mutated after fingerprinting.")
    cap = _limit(max_instructions)
    bound = len(evolution.circuit.data)+2+(axis=="sine")+include_measurement
    if bound>cap:
        raise RuntimeError("PRE_BUILD_INSTRUCTION_BUDGET")
    circuit = QiskitRPEHadamardInterrogationBuilder._new_wrapper_circuit(evolution.circuit)
    circuit.h(n)
    circuit.compose(evolution.circuit,inplace=True)
    if axis=="sine":
        circuit.sdg(n)
    circuit.h(n)
    if include_measurement:
        circuit.add_register(ClassicalRegister(1,"rpe_measure"))
        circuit.measure(n,0)
    return circuit
