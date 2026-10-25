"""Unchanged selected AX-2B Full PF definitions; science-only lazy import.

Source commit 31cdff47f3282d2898c153a1af22f90e500be4c6
src/trottertracks/resource_applicability/ax2b_native_df_v5.py
Function/class bodies are preserved. Fingerprinting uses the existing library
canonical implementation rather than importing the old diagnostics package.
No old wrapper, sampler, authorization or runtime environment is restored.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from trotterlib.df_partial_s2 import (DFDeterministicOneBodySpec,
    DFDeterministicFragmentSpec, DFDeterministicBlockSpec,
    QiskitDFPartialS2CircuitBuilder)
from trotterlib.df_trotter.ops import (DiagonalEvolutionPrimitives,
    append_diagonal_primitives, one_body_diagonal_primitives,
    df_squared_diagonal_primitives)
from trotterlib.rte import require_integer_count
from trotterlib.rte_compiled_cost import canonical_qiskit_circuit_fingerprint
from ._full_df_schedule import deterministic_control_plan


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _limit(value):
    return require_integer_count(value, name="max_instructions", minimum=1)


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
