from dataclasses import dataclass
import multiprocessing as mp
from typing import Sequence, Tuple

import numpy as np

from openfermion.ops import QubitOperator

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from .config import PFLabel
from .qiskit_time_evolution_utils import (
    apply_time_evolution,
    _get_w_list,
    term_to_sparse_pauli,
)
from .pf_decomposition import iter_pf_steps


@dataclass(frozen=True)
class CliqueHamiltonian:
    hamiltonian: SparsePauliOp | None
    exp_term_count: int


def build_clique_hamiltonian(
    commuting_clique: Sequence[QubitOperator],
    num_qubits: int,
) -> CliqueHamiltonian:
    """Convert one commuting clique to a reusable SparsePauliOp."""
    pauli_terms: list[tuple[str, float]] = []
    exp_term_count = 0
    for hamiltonian in commuting_clique:
        for term, coeff in hamiltonian.terms.items():
            if not term:
                continue
            label = ["I"] * int(num_qubits)
            for index, pauli_op_name in term:
                label[int(index)] = str(pauli_op_name)
            pauli_terms.append(("".join(label), float(np.real(coeff))))
            exp_term_count += 1
    if not pauli_terms:
        return CliqueHamiltonian(hamiltonian=None, exp_term_count=0)
    return CliqueHamiltonian(
        hamiltonian=SparsePauliOp.from_list(pauli_terms),
        exp_term_count=int(exp_term_count),
    )


def _build_clique_hamiltonian_worker(
    args: tuple[Sequence[QubitOperator], int],
) -> CliqueHamiltonian:
    commuting_clique, num_qubits = args
    return build_clique_hamiltonian(commuting_clique, int(num_qubits))


def build_clique_hamiltonians(
    commuting_cliques: Sequence[Sequence[QubitOperator]],
    num_qubits: int,
    *,
    processes: int = 1,
) -> tuple[CliqueHamiltonian, ...]:
    """Build reusable clique Hamiltonians, optionally in parallel."""
    task_args = [(tuple(clique), int(num_qubits)) for clique in commuting_cliques]
    if int(processes) <= 1 or len(task_args) <= 1:
        return tuple(_build_clique_hamiltonian_worker(args) for args in task_args)
    process_count = max(1, min(int(processes), len(task_args)))
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=process_count) as pool:
        return tuple(pool.map(_build_clique_hamiltonian_worker, task_args, chunksize=1))


def add_clique_to_circuit_grouper(
    commuting_clique: Sequence[QubitOperator],
    time: float | ParameterExpression,
    num_qubits: int,
    weight: float,
    circuit: QuantumCircuit,
) -> int:
    """
    可換クリーク中の項を和に束ねて一括で進化ゲートを追加。
    戻り値は、元の「指数項の数」（ゲート数ではなく、物理評価用のカウント）を返す。
    """
    # 係数付きの SparsePauliOp を加算して合成ハミルトニアン H_clique を構築
    clique_hamiltonian: SparsePauliOp | None = None
    exp_term_count = 0
    for hamiltonian in commuting_clique:
        for term, coeff in hamiltonian.terms.items():
            if not term:
                # 恒等項（定数項）は回路には入れない
                continue
            # PauliOp へ変換して加算
            pauli_op = term_to_sparse_pauli(tuple(term), num_qubits)
            pauli_op = coeff.real * pauli_op
            clique_hamiltonian = (
                pauli_op if clique_hamiltonian is None else (clique_hamiltonian + pauli_op)
            )
            exp_term_count += 1
    if clique_hamiltonian is None:
        return 0
    # 以前は各項 angle=coeff*w*t だったが、ここでは H=Σ coeff*P として time=w*t を与える
    evolution_gate = PauliEvolutionGate(
        clique_hamiltonian, time=(weight * time), synthesis=None
    )
    circuit.append(evolution_gate, range(num_qubits))
    return exp_term_count


def add_precomputed_clique_to_circuit_grouper(
    clique_hamiltonian: CliqueHamiltonian,
    time: float | ParameterExpression,
    num_qubits: int,
    weight: float,
    circuit: QuantumCircuit,
) -> int:
    """Append a precomputed clique Hamiltonian evolution gate."""
    if clique_hamiltonian.hamiltonian is None:
        return 0
    evolution_gate = PauliEvolutionGate(
        clique_hamiltonian.hamiltonian,
        time=(weight * time),
        synthesis=None,
    )
    circuit.append(evolution_gate, range(num_qubits))
    return int(clique_hamiltonian.exp_term_count)


def w_trotter_grouper(
    circuit: QuantumCircuit,
    commuting_cliques: Sequence[Sequence[QubitOperator]],
    time: float | ParameterExpression,
    num_qubits: int,
    pf_label: PFLabel,
) -> int:
    """与えられた w シリーズで PF 分解を回路に追加し、累計項数を返す。"""
    # PF 係数列に従ってクリークを順次追加
    weights = _get_w_list(pf_label)
    exp_term_count = 0
    for term_idx, weight in iter_pf_steps(len(commuting_cliques), weights):
        exp_term_count += add_clique_to_circuit_grouper(
            commuting_cliques[term_idx], time, num_qubits, weight, circuit
        )
    return exp_term_count


def w_trotter_grouper_precomputed(
    circuit: QuantumCircuit,
    clique_hamiltonians: Sequence[CliqueHamiltonian],
    time: float | ParameterExpression,
    num_qubits: int,
    pf_label: PFLabel,
) -> int:
    """Append PF steps using precomputed clique Hamiltonians."""
    weights = _get_w_list(pf_label)
    exp_term_count = 0
    for term_idx, weight in iter_pf_steps(len(clique_hamiltonians), weights):
        exp_term_count += add_precomputed_clique_to_circuit_grouper(
            clique_hamiltonians[term_idx], time, num_qubits, weight, circuit
        )
    return exp_term_count


def tEvolution_vector_grouper(
    commuting_cliques: Sequence[Sequence[QubitOperator]],
    time: float,
    num_qubits: int,
    state_vec: np.ndarray,
    pf_label: PFLabel,
) -> Tuple[float, Statevector, int]:
    """グルーピング済みハミルトニアンで時間発展回路を合成し、最終状態を返す。"""
    # 回路を構築して時間発展
    evolution_circuit = QuantumCircuit(num_qubits)
    exp_term_count = w_trotter_grouper(
        evolution_circuit, commuting_cliques, time, num_qubits, pf_label
    )
    final_statevector = apply_time_evolution(state_vec, evolution_circuit)
    return time, final_statevector, exp_term_count


def tEvolution_vector_grouper_precomputed(
    clique_hamiltonians: Sequence[CliqueHamiltonian],
    time: float,
    num_qubits: int,
    state_vec: np.ndarray,
    pf_label: PFLabel,
) -> Tuple[float, Statevector, int]:
    """Time evolution using reusable SparsePauliOp objects for each clique."""
    evolution_circuit = QuantumCircuit(num_qubits)
    exp_term_count = w_trotter_grouper_precomputed(
        evolution_circuit,
        clique_hamiltonians,
        time,
        num_qubits,
        pf_label,
    )
    final_statevector = apply_time_evolution(state_vec, evolution_circuit)
    return time, final_statevector, exp_term_count
