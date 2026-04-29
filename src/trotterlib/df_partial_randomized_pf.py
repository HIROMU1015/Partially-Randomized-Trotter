from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

from .analysis_utils import loglog_average_coeff, loglog_fit
from .config import (
    PARTIAL_RANDOMIZED_DF_CGS_CACHE_PATH,
    PARTIAL_RANDOMIZED_DF_GROUND_STATE_CACHE_DIR,
    PFLabel,
    pf_order,
)
from .df_gpu_statevector import (
    DFGPUParameterizedTemplate,
    build_parameterized_gpu_template,
    run_parameterized_gpu_template,
    simulate_statevector_gpu,
)
from .df_hamiltonian import (
    DFGroundStateResult,
    DFHamiltonian,
    PhysicalSector,
    solve_df_ground_state,
)
from .df_trotter.circuit import build_df_trotter_circuit, simulate_statevector
from .df_trotter.model import Block, DFModel
from .df_trotter.ops import (
    apply_D_one_body,
    apply_D_squared,
    build_df_blocks,
    build_df_blocks_givens,
    build_one_body_gaussian_block,
    build_one_body_gaussian_block_givens,
)
from .pf_decomposition import iter_pf_steps
from .product_formula import _get_w_list
from .partial_randomized_pf import (
    _PERTURBATION_NOISE_FLOOR,
    default_perturbation_t_values,
)


DFFragmentWeightRule: TypeAlias = Literal[
    "lambda_frobenius_squared",
    "abs_lambda",
]
DFEvolutionBackend: TypeAlias = Literal["gpu", "cpu", "auto"]
_DF_CGS_CACHE_SCHEMA_VERSION = 6
_DF_CGS_DEFINITION = "df_hd_deterministic_surrogate_v1"
_DF_GROUND_STATE_CACHE_SCHEMA_VERSION = 2
_DF_COST_BASIS_GATES = ("rz", "cx", "sx", "x")
_DF_TIME_WORKER_TEMPLATE: DFGPUParameterizedTemplate | None = None


def _get_pool_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


@dataclass(frozen=True)
class RankedDFFragment:
    """Single DF fragment sorted by a fixed representation-level weight rule."""

    rank: int
    original_index: int
    lam: float
    weight: float
    weight_rule: str


@dataclass(frozen=True)
class DFFragmentPartition:
    """DF-native H_D/H_R split for a single L_D."""

    ld: int
    deterministic_fragments: tuple[RankedDFFragment, ...]
    randomized_fragments: tuple[RankedDFFragment, ...]
    deterministic_block_indices: tuple[int, ...]
    randomized_block_indices: tuple[int, ...]
    lambda_r: float
    weight_rule: str


@dataclass(frozen=True)
class DFCgsFitResult:
    """C_gs,D fit for DF H_D. This is a deterministic surrogate, not a full bound."""

    representation_type: str
    cgs_definition: str
    pf_label: PFLabel
    order: int
    ld: int
    lambda_r: float
    t_values: tuple[float, ...]
    perturbation_errors: tuple[float, ...]
    coeff: float
    fit_coeff_fixed_order: float
    fit_slope: float | None
    fit_coeff: float | None
    evolution_backend: str
    gpu_ids: tuple[str, ...]
    chunk_splits: int
    optimization_level: int
    parallel_times: bool
    processes: int
    weight_rule: str
    df_rank_actual: int
    df_rank_requested: int | None
    df_tol_requested: float | None
    metadata: dict[str, Any]
    simulation_profiles: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def df_fragment_weight(
    hamiltonian: DFHamiltonian,
    block_index: int,
    *,
    weight_rule: DFFragmentWeightRule = "lambda_frobenius_squared",
) -> float:
    """Return the scalar weight used to sort and split DF fragments."""
    lam = float(hamiltonian.lambdas[int(block_index)])
    if weight_rule == "abs_lambda":
        return abs(lam)
    if weight_rule == "lambda_frobenius_squared":
        g_mat = np.asarray(hamiltonian.g_matrices[int(block_index)])
        return float(abs(lam) * (np.linalg.norm(g_mat, ord="fro") ** 2))
    raise ValueError(f"Unsupported DF fragment weight rule: {weight_rule}")


def rank_df_fragments(
    hamiltonian: DFHamiltonian,
    *,
    weight_rule: DFFragmentWeightRule = "lambda_frobenius_squared",
) -> tuple[RankedDFFragment, ...]:
    """Sort DF two-body fragments by descending weight."""
    fragments = [
        RankedDFFragment(
            rank=-1,
            original_index=idx,
            lam=float(hamiltonian.lambdas[idx]),
            weight=df_fragment_weight(
                hamiltonian,
                idx,
                weight_rule=weight_rule,
            ),
            weight_rule=weight_rule,
        )
        for idx in range(hamiltonian.n_blocks)
    ]
    fragments.sort(
        key=lambda fragment: (
            -round(fragment.weight, 12),
            fragment.original_index,
        )
    )
    return tuple(
        RankedDFFragment(
            rank=rank,
            original_index=fragment.original_index,
            lam=fragment.lam,
            weight=fragment.weight,
            weight_rule=fragment.weight_rule,
        )
        for rank, fragment in enumerate(fragments)
    )


def split_df_hamiltonian_by_ld(
    hamiltonian: DFHamiltonian,
    ld: int,
    *,
    ranked_fragments: Sequence[RankedDFFragment] | None = None,
    weight_rule: DFFragmentWeightRule = "lambda_frobenius_squared",
) -> DFFragmentPartition:
    """Split a fixed DF representation into H_D and H_R by fragment prefix length."""
    if ranked_fragments is None:
        ranked_fragments = rank_df_fragments(hamiltonian, weight_rule=weight_rule)
    ld = int(ld)
    if ld < 0 or ld > len(ranked_fragments):
        raise ValueError("ld must be between 0 and the number of DF fragments.")
    deterministic = tuple(ranked_fragments[:ld])
    randomized = tuple(ranked_fragments[ld:])
    return DFFragmentPartition(
        ld=ld,
        deterministic_fragments=deterministic,
        randomized_fragments=randomized,
        deterministic_block_indices=tuple(
            fragment.original_index for fragment in deterministic
        ),
        randomized_block_indices=tuple(fragment.original_index for fragment in randomized),
        lambda_r=float(sum(fragment.weight for fragment in randomized)),
        weight_rule=weight_rule,
    )


def select_df_h_d(
    hamiltonian: DFHamiltonian,
    partition: DFFragmentPartition,
) -> DFHamiltonian:
    """Build DF H_D from selected fragments while keeping one-body and constant terms."""
    return hamiltonian.select_blocks(partition.deterministic_block_indices)


def df_hamiltonian_to_model(hamiltonian: DFHamiltonian) -> DFModel:
    """Convert this project's DFHamiltonian to the circuit-builder DFModel."""
    return DFModel(
        lambdas=np.asarray(hamiltonian.lambdas, dtype=np.complex128),
        G_list=[np.asarray(g_mat, dtype=np.complex128) for g_mat in hamiltonian.g_matrices],
        one_body_correction=np.asarray(hamiltonian.one_body, dtype=np.complex128),
        constant_correction=float(hamiltonian.constant),
        N=int(hamiltonian.n_qubits),
    ).hermitize()


def build_df_hd_trotter_blocks(
    hamiltonian: DFHamiltonian,
    *,
    include_one_body: bool = True,
    diagonal_sort: str = "descending_abs",
) -> tuple[Block, ...]:
    """Build DF-native circuit blocks for H_D without converting fragments to Pauli terms."""
    model = df_hamiltonian_to_model(hamiltonian)
    blocks: list[Block] = []
    if include_one_body and np.linalg.norm(model.one_body_correction) > 1e-14:
        blocks.append(
            Block.from_one_body_gaussian(
                build_one_body_gaussian_block(
                    model.one_body_correction,
                    sort=diagonal_sort,
                )
            )
        )
    blocks.extend(
        Block.from_df(block)
        for block in build_df_blocks(model, sort=diagonal_sort)
    )
    return tuple(blocks)


def _nonbasis_ops(qc: QuantumCircuit, *, basis_gates: Sequence[str]) -> list[str]:
    basis_set = {name.lower() for name in basis_gates}
    ignore = {"barrier", "measure", "reset", "delay"}
    extras = {
        inst.operation.name
        for inst in qc.data
        if inst.operation.name.lower() not in basis_set
        and inst.operation.name.lower() not in ignore
    }
    return sorted(extras)


def _decompose_to_cost_basis(
    qc: QuantumCircuit,
    *,
    basis_gates: Sequence[str] = _DF_COST_BASIS_GATES,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> QuantumCircuit:
    qc_work = qc
    for _ in range(max(0, int(decompose_reps))):
        extras = _nonbasis_ops(qc_work, basis_gates=basis_gates)
        if not extras:
            break
        qc_work = qc_work.decompose(gates_to_decompose=extras, reps=1)
    return transpile(
        qc_work,
        basis_gates=list(basis_gates),
        optimization_level=int(optimization_level),
    )


def _rz_depth_from_circuit(
    qc: QuantumCircuit,
    *,
    basis_gates: Sequence[str] = _DF_COST_BASIS_GATES,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> dict[str, Any]:
    qc_cost = _decompose_to_cost_basis(
        qc,
        basis_gates=basis_gates,
        decompose_reps=decompose_reps,
        optimization_level=optimization_level,
    )
    dag = circuit_to_dag(qc_cost)
    dp: dict[DAGOpNode, int] = {}
    rz_depth = 0
    for node in dag.topological_op_nodes():
        max_pred = 0
        for pred in dag.predecessors(node):
            if isinstance(pred, DAGOpNode):
                max_pred = max(max_pred, dp.get(pred, 0))
        weight = 1 if node.op.name.lower() == "rz" else 0
        dp[node] = max_pred + weight
        rz_depth = max(rz_depth, dp[node])
    counts = qc_cost.count_ops()
    return {
        "rz_count": int(counts.get("rz", 0)) + int(counts.get("RZ", 0)),
        "rz_depth": int(rz_depth),
        "transpiled_size": int(qc_cost.size()),
        "transpiled_depth": int(qc_cost.depth()),
        "transpiled_count_ops": {str(key): int(value) for key, value in counts.items()},
    }


def _u_ops_rz_cost(
    u_ops: Sequence[tuple[Any, tuple[int, ...]]],
    num_qubits: int,
    *,
    basis_gates: Sequence[str],
    decompose_reps: int,
    optimization_level: int,
) -> dict[str, Any]:
    qc = QuantumCircuit(int(num_qubits))
    for gate, qubits in u_ops:
        qc.append(gate, list(qubits))
    return _rz_depth_from_circuit(
        qc,
        basis_gates=basis_gates,
        decompose_reps=decompose_reps,
        optimization_level=optimization_level,
    )


def _apply_d_block(qc: QuantumCircuit, block: Block, tau: float) -> None:
    if block.kind == "one_body_gaussian":
        apply_D_one_body(qc, block.payload.eps, tau)
        return
    if block.kind == "df":
        apply_D_squared(qc, block.payload.eta, block.payload.lam, tau)
        return
    raise ValueError(f"Unsupported DF cost block kind: {block.kind}")


def _is_nonzero_cost_angle(value: float, *, atol: float = 1e-12) -> bool:
    return abs(float(value)) > float(atol)


def _analytic_apply_rz(depths: list[int], qubit: int) -> None:
    depths[int(qubit)] += 1


def _analytic_apply_rzz(depths: list[int], q0: int, q1: int) -> None:
    next_depth = max(depths[int(q0)], depths[int(q1)]) + 1
    depths[int(q0)] = next_depth
    depths[int(q1)] = next_depth


def _real_float(value: Any) -> float:
    return float(np.real_if_close(value))


def _analytic_d_block_rz_cost(
    block: Block,
    tau: float,
    depths: list[int],
    *,
    atol: float = 1e-12,
) -> int:
    """Count D-only RZ gates and RZ-depth without constructing the circuit."""
    count = 0
    if block.kind == "one_body_gaussian":
        for k, eps_k in enumerate(np.asarray(block.payload.eps)):
            angle = -float(tau) * _real_float(eps_k)
            if _is_nonzero_cost_angle(angle, atol=atol):
                _analytic_apply_rz(depths, k)
                count += 1
        return count

    if block.kind == "df":
        eta = np.asarray(block.payload.eta)
        lam = _real_float(block.payload.lam)
        tau_internal = -float(tau)
        rz_angles = [0.0 for _ in range(len(eta))]

        for k in range(len(eta)):
            eta_k = _real_float(eta[k])
            rz_angles[k] += tau_internal * lam * eta_k * eta_k

        for k in range(len(eta)):
            eta_k = _real_float(eta[k])
            for j in range(k + 1, len(eta)):
                eta_j = _real_float(eta[j])
                beta = 2.0 * tau_internal * lam * eta_k * eta_j
                rz_angles[k] += beta / 2.0
                rz_angles[j] += beta / 2.0
                _analytic_apply_rzz(depths, k, j)
                count += 1

        for k, angle in enumerate(rz_angles):
            if _is_nonzero_cost_angle(angle, atol=atol):
                _analytic_apply_rz(depths, k)
                count += 1
        return count

    raise ValueError(f"Unsupported DF cost block kind: {block.kind}")


def _analytic_d_only_rz_cost(
    blocks: Sequence[Block],
    *,
    time: float,
    num_qubits: int,
    pf_label: PFLabel,
) -> dict[str, Any]:
    depths = [0 for _ in range(int(num_qubits))]
    rz_count = 0
    for term_idx, weight in iter_pf_steps(len(blocks), _get_w_list(pf_label)):
        rz_count += _analytic_d_block_rz_cost(
            blocks[int(term_idx)],
            float(weight) * float(time),
            depths,
        )
    return {
        "rz_count": int(rz_count),
        "rz_depth": int(max(depths, default=0)),
        "cost_method": "analytic_rz_rzz_dependency_v1",
    }


def _build_d_only_cost_circuit(
    blocks: Sequence[Block],
    *,
    time: float,
    num_qubits: int,
    pf_label: PFLabel,
) -> QuantumCircuit:
    qc = QuantumCircuit(int(num_qubits))
    for term_idx, weight in iter_pf_steps(len(blocks), _get_w_list(pf_label)):
        _apply_d_block(qc, blocks[term_idx], float(weight) * float(time))
    return qc


def df_deterministic_step_rz_cost(
    hamiltonian: DFHamiltonian,
    pf_label: PFLabel,
    *,
    time: float = 1.0,
    diagonal_sort: str = "descending_abs",
    basis_gates: Sequence[str] = _DF_COST_BASIS_GATES,
    decompose_reps: int = 8,
    optimization_level: int = 0,
) -> dict[str, Any]:
    """Count DF-project-style total_ref_rz_depth for one deterministic PF step."""
    model = df_hamiltonian_to_model(hamiltonian)
    blocks: list[Block] = []
    if np.linalg.norm(model.one_body_correction) > 1e-14:
        blocks.append(
            Block.from_one_body_gaussian(
                build_one_body_gaussian_block_givens(
                    model.one_body_correction,
                    sort=diagonal_sort,
                )
            )
        )
    blocks.extend(
        Block.from_df(block)
        for block in build_df_blocks_givens(model, sort=diagonal_sort)
    )

    u_costs: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        cost = _u_ops_rz_cost(
            block.payload.U_ops,
            model.N,
            basis_gates=basis_gates,
            decompose_reps=decompose_reps,
            optimization_level=optimization_level,
        )
        cost["block_index"] = int(idx)
        cost["block_kind"] = str(block.kind)
        u_costs.append(cost)

    u_total_count = 0
    u_total_depth = 0
    for term_idx, _weight in iter_pf_steps(len(blocks), _get_w_list(pf_label)):
        cost = u_costs[int(term_idx)]
        u_total_count += 2 * int(cost["rz_count"])
        u_total_depth += 2 * int(cost["rz_depth"])

    d_cost = _analytic_d_only_rz_cost(
        blocks,
        time=float(time),
        num_qubits=model.N,
        pf_label=pf_label,
    )
    d_total_count = int(d_cost["rz_count"])
    d_total_depth = int(d_cost["rz_depth"])
    return {
        "cost_definition": "df_project_total_ref_rz_depth_v1",
        "pf_label": str(pf_label),
        "time": float(time),
        "basis_gates": [str(gate) for gate in basis_gates],
        "decompose_reps": int(decompose_reps),
        "optimization_level": int(optimization_level),
        "num_qubits": int(model.N),
        "num_cost_blocks": int(len(blocks)),
        "u_ref_rz_count": int(u_total_count),
        "u_ref_rz_depth": int(u_total_depth),
        "d_ref_rz_count": int(d_total_count),
        "d_ref_rz_depth": int(d_total_depth),
        "total_ref_rz_count": int(u_total_count + d_total_count),
        "total_ref_rz_depth": int(u_total_depth + d_total_depth),
        "u_block_costs": u_costs,
        "d_only_cost": d_cost,
    }


def _collect_df_perturbation_errors(
    final_state_list: Sequence[tuple[float, np.ndarray]],
    energy: float,
    state_vec: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    times_out: list[float] = []
    error_list: list[float] = []
    for raw_time, evolved_raw in final_state_list:
        time = float(raw_time)
        psi0 = np.asarray(state_vec, dtype=np.complex128).reshape(-1)
        evolved = np.asarray(evolved_raw, dtype=np.complex128).reshape(-1)
        phase_factor = np.exp(-1j * energy * time)
        delta_state = evolved - phase_factor * psi0
        denom = time * np.sin(energy * time)
        if abs(denom) < 1e-12:
            denom = energy * (time**2)
        if denom == 0.0:
            error = 0.0
        else:
            error = float(abs(np.vdot(psi0, delta_state).real / denom))
        error_list.append(error)
        times_out.append(time)
    return tuple(times_out), tuple(error_list)


def _bit_reverse_permutation(num_qubits: int) -> np.ndarray:
    dim = 1 << int(num_qubits)
    perm = np.zeros(dim, dtype=np.int64)
    for idx in range(dim):
        value = idx
        reversed_value = 0
        for _ in range(int(num_qubits)):
            reversed_value = (reversed_value << 1) | (value & 1)
            value >>= 1
        perm[idx] = reversed_value
    return perm


def _to_qiskit_state_order(state: np.ndarray, num_qubits: int) -> np.ndarray:
    """Match the DF circuit builder's Qiskit statevector ordering."""
    vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    if vec.size != (1 << int(num_qubits)):
        raise ValueError("Statevector dimension does not match num_qubits.")
    return vec[_bit_reverse_permutation(int(num_qubits))]


def _fit_errors(
    *,
    pf_label: PFLabel,
    times_out: Sequence[float],
    perturbation_errors: Sequence[float],
) -> tuple[float, float, float | None, float | None]:
    order = pf_order(pf_label)
    positive_errors = [err for err in perturbation_errors if err > 0.0]
    if not positive_errors or max(positive_errors) < _PERTURBATION_NOISE_FLOOR:
        return 0.0, 0.0, None, None
    coeff = float(
        loglog_average_coeff(
            times_out,
            perturbation_errors,
            order,
            mask_nonpositive=True,
        )
    )
    fit_slope: float | None = None
    fit_coeff: float | None = None
    try:
        fit = loglog_fit(times_out, perturbation_errors, mask_nonpositive=True)
        fit_slope = fit.slope
        fit_coeff = fit.coeff
    except ValueError:
        pass
    return coeff, coeff, fit_slope, fit_coeff


def _assign_gpu_ids_to_times(
    t_values: Sequence[float],
    gpu_ids: Sequence[str],
) -> list[str]:
    visible_gpu_ids = [str(gpu_id) for gpu_id in gpu_ids if str(gpu_id) != ""]
    if not visible_gpu_ids:
        visible_gpu_ids = ["0"]
    return [
        visible_gpu_ids[idx % len(visible_gpu_ids)]
        for idx, _time_value in enumerate(t_values)
    ]


def _resolve_parallel_processes(
    *,
    num_times: int,
    num_gpus: int,
    processes: int | None,
) -> int:
    if num_times <= 0:
        return 0
    max_parallel = max(1, int(num_gpus))
    if processes is None:
        return min(int(num_times), max_parallel)
    return max(1, min(int(processes), int(num_times), max_parallel))


def _set_df_time_worker_template(
    template: DFGPUParameterizedTemplate | None,
) -> None:
    global _DF_TIME_WORKER_TEMPLATE
    _DF_TIME_WORKER_TEMPLATE = template


def _simulate_df_time_task(
    args: tuple[
        float,
        tuple[Block, ...],
        int,
        PFLabel,
        float,
        np.ndarray,
        str,
        int,
        int,
        bool,
    ],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    (
        time_value,
        blocks,
        num_qubits,
        pf_label,
        energy_shift,
        state_flat,
        gpu_id,
        chunk_splits,
        optimization_level,
        debug,
    ) = args
    if _DF_TIME_WORKER_TEMPLATE is not None:
        if int(chunk_splits) != 1:
            raise ValueError("Parameterized GPU template does not support chunk_splits > 1.")
        evolved, profile = run_parameterized_gpu_template(
            _DF_TIME_WORKER_TEMPLATE,
            time_value=float(time_value),
            gpu_ids=(str(gpu_id),),
            debug=bool(debug),
            debug_label=f"t={float(time_value)} gpu={gpu_id}",
        )
    else:
        qc = build_df_trotter_circuit(
            blocks,
            time=float(time_value),
            num_qubits=int(num_qubits),
            pf_label=pf_label,
            energy_shift=float(energy_shift),
        )
        evolved, profile = simulate_statevector_gpu(
            qc,
            state_flat,
            gpu_ids=(str(gpu_id),),
            chunk_splits=int(chunk_splits),
            optimization_level=int(optimization_level),
            debug=bool(debug),
            debug_label=f"t={float(time_value)} gpu={gpu_id}",
        )
    profile = dict(profile)
    profile["time"] = float(time_value)
    profile["assigned_gpu_id"] = str(gpu_id)
    return float(time_value), evolved, profile


def _df_hamiltonian_hash(
    hamiltonian: DFHamiltonian,
    *,
    weight_rule: str,
) -> str:
    payload = {
        "constant": round(float(hamiltonian.constant), 12),
        "one_body_shape": list(hamiltonian.one_body.shape),
        "one_body_norm": round(float(np.linalg.norm(hamiltonian.one_body)), 12),
        "one_body_hash": _array_hash_payload(hamiltonian.one_body),
        "lambdas": [round(float(value), 12) for value in hamiltonian.lambdas],
        "lambdas_hash": _array_hash_payload(hamiltonian.lambdas),
        "g_norms": [
            round(float(np.linalg.norm(g_mat, ord="fro")), 12)
            for g_mat in hamiltonian.g_matrices
        ],
        "g_matrix_hashes": [
            _array_hash_payload(g_mat) for g_mat in hamiltonian.g_matrices
        ],
        "metadata": _jsonable_metadata(hamiltonian.metadata),
        "weight_rule": weight_rule,
    }
    return _json_hash(payload)


def _sector_hash(sector: PhysicalSector) -> str:
    payload = {
        "n_qubits": int(sector.n_qubits),
        "dimension": int(sector.dimension),
        "basis_indices_sha256": hashlib.sha256(
            np.asarray(sector.basis_indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "n_electrons": sector.n_electrons,
        "nelec_alpha": sector.nelec_alpha,
        "nelec_beta": sector.nelec_beta,
        "sz_value": sector.sz_value,
    }
    return _json_hash(payload)


def _df_ground_state_cache_key_payload(
    *,
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    matrix_free_backend: str,
    matrix_free_threads: int | None,
    matrix_free_block_chunk_size: int | None,
    ground_state_ncv: int | None,
    ground_state_tol: float,
) -> dict[str, Any]:
    return {
        "schema_version": _DF_GROUND_STATE_CACHE_SCHEMA_VERSION,
        "hamiltonian_hash": _df_hamiltonian_hash(
            hamiltonian,
            weight_rule="ground_state",
        ),
        "sector_hash": _sector_hash(sector),
        "matrix_free_backend": matrix_free_backend,
        "matrix_free_threads": (
            None if matrix_free_threads is None else int(matrix_free_threads)
        ),
        "matrix_free_block_chunk_size": (
            None
            if matrix_free_block_chunk_size is None
            else int(matrix_free_block_chunk_size)
        ),
        "solver": "eigsh",
        "ground_state_ncv": None if ground_state_ncv is None else int(ground_state_ncv),
        "ground_state_tol": float(ground_state_tol),
        "expand_state": True,
    }


def _df_ground_state_result_from_npz(
    path: Path,
    sector: PhysicalSector,
    *,
    expected_cache_key: str | None = None,
    expected_cache_payload: dict[str, Any] | None = None,
) -> DFGroundStateResult | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["cache_schema_version"][()]) != _DF_GROUND_STATE_CACHE_SCHEMA_VERSION:
                return None
            if expected_cache_key is not None and str(data["cache_key"][()]) != str(
                expected_cache_key
            ):
                return None
            if expected_cache_payload is not None:
                expected_payload_hash = _json_hash(expected_cache_payload)
                if str(data["cache_payload_sha256"][()]) != expected_payload_hash:
                    return None
            state_vector = np.asarray(data["state_vector"], dtype=np.complex128)
            sector_state_vector = np.asarray(
                data["sector_state_vector"],
                dtype=np.complex128,
            )
            if state_vector.size != (1 << int(sector.n_qubits)):
                return None
            if sector_state_vector.size != sector.dimension:
                return None
            return DFGroundStateResult(
                energy=float(data["energy"][()]),
                state_vector=state_vector,
                sector_state_vector=sector_state_vector,
                sector=sector,
                converged=bool(data["converged"][()]),
                residual_norm=float(data["residual_norm"][()]),
                matvec_count=int(data["matvec_count"][()]),
                elapsed_s=float(data["elapsed_s"][()]),
                solver=str(data["solver"][()]),
                message=str(data["message"][()]),
            )
    except (OSError, KeyError, ValueError):
        return None


def _save_df_ground_state_npz(
    path: Path,
    ground_state: DFGroundStateResult,
    *,
    cache_key: str,
    cache_payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        cache_schema_version=np.asarray(_DF_GROUND_STATE_CACHE_SCHEMA_VERSION),
        cache_key=np.asarray(str(cache_key)),
        cache_payload_sha256=np.asarray(_json_hash(cache_payload)),
        hamiltonian_hash=np.asarray(str(cache_payload["hamiltonian_hash"])),
        sector_hash=np.asarray(str(cache_payload["sector_hash"])),
        energy=np.asarray(ground_state.energy),
        state_vector=np.asarray(ground_state.state_vector, dtype=np.complex128),
        sector_state_vector=np.asarray(
            ground_state.sector_state_vector,
            dtype=np.complex128,
        ),
        converged=np.asarray(bool(ground_state.converged)),
        residual_norm=np.asarray(float(ground_state.residual_norm)),
        matvec_count=np.asarray(int(ground_state.matvec_count)),
        elapsed_s=np.asarray(float(ground_state.elapsed_s)),
        solver=np.asarray(str(ground_state.solver)),
        message=np.asarray(str(ground_state.message)),
    )
    tmp_path.replace(path)


def get_or_compute_cached_df_ground_state(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    *,
    cache_dir: str | Path = PARTIAL_RANDOMIZED_DF_GROUND_STATE_CACHE_DIR,
    matrix_free_backend: str = "auto",
    matrix_free_threads: int | None = None,
    matrix_free_block_chunk_size: int | None = None,
    ground_state_ncv: int | None = None,
    ground_state_tol: float = 1e-10,
) -> tuple[DFGroundStateResult, dict[str, Any]]:
    """Return a cached DF ground state for a fixed H_D/sector/solver setting."""
    payload = _df_ground_state_cache_key_payload(
        hamiltonian=hamiltonian,
        sector=sector,
        matrix_free_backend=matrix_free_backend,
        matrix_free_threads=matrix_free_threads,
        matrix_free_block_chunk_size=matrix_free_block_chunk_size,
        ground_state_ncv=ground_state_ncv,
        ground_state_tol=ground_state_tol,
    )
    cache_key = _json_hash(payload)
    path = Path(cache_dir) / f"{cache_key}.npz"
    cached = _df_ground_state_result_from_npz(
        path,
        sector,
        expected_cache_key=cache_key,
        expected_cache_payload=payload,
    )
    if cached is not None:
        return cached, {
            "ground_state_cache_hit": True,
            "ground_state_cache_key": cache_key,
            "ground_state_cache_path": str(path),
        }

    ground_state = solve_df_ground_state(
        hamiltonian,
        sector,
        matrix_free_backend=matrix_free_backend,
        matrix_free_threads=matrix_free_threads,
        matrix_free_block_chunk_size=matrix_free_block_chunk_size,
        tol=ground_state_tol,
        ncv=ground_state_ncv,
        expand_state=True,
    )
    _save_df_ground_state_npz(
        path,
        ground_state,
        cache_key=cache_key,
        cache_payload=payload,
    )
    return ground_state, {
        "ground_state_cache_hit": False,
        "ground_state_cache_key": cache_key,
        "ground_state_cache_path": str(path),
    }


def _jsonable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.generic):
            clean[key] = value.item()
        elif isinstance(value, np.ndarray):
            clean[key] = value.tolist()
        elif isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = list(value)
        elif isinstance(value, dict):
            clean[key] = _jsonable_metadata(value)
        else:
            clean[key] = str(value)
    return clean


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash_payload(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.view(np.uint8).tobytes()).hexdigest(),
    }


def _default_cache_document() -> dict[str, Any]:
    return {
        "schema_version": _DF_CGS_CACHE_SCHEMA_VERSION,
        "cgs_definition": _DF_CGS_DEFINITION,
        "representation_type": "df",
        "entries": {},
    }


def load_df_cgs_json_cache(
    cache_path: str | Path = PARTIAL_RANDOMIZED_DF_CGS_CACHE_PATH,
) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.exists():
        return _default_cache_document()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_cache_document()
    if document.get("schema_version") != _DF_CGS_CACHE_SCHEMA_VERSION:
        return _default_cache_document()
    if not isinstance(document.get("entries"), dict):
        return _default_cache_document()
    return document


def save_df_cgs_json_cache(
    cache_document: dict[str, Any],
    cache_path: str | Path = PARTIAL_RANDOMIZED_DF_CGS_CACHE_PATH,
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(cache_document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _cache_key_payload(
    *,
    hamiltonian: DFHamiltonian,
    hamiltonian_hash: str,
    partition: DFFragmentPartition,
    pf_label: PFLabel,
    t_values: Sequence[float],
    evolution_backend: str,
    gpu_ids: Sequence[str],
    chunk_splits: int,
    optimization_level: int,
    parallel_times: bool,
    processes: int | None,
    use_parameterized_template: bool,
    diagonal_sort: str,
    ground_state_tol: float,
    ground_state_ncv: int | None,
) -> dict[str, Any]:
    return {
        "representation_type": "df",
        "cgs_definition": _DF_CGS_DEFINITION,
        "hamiltonian_hash": hamiltonian_hash,
        "molecule_type": hamiltonian.metadata.get("molecule_type"),
        "distance": hamiltonian.metadata.get("distance"),
        "basis": hamiltonian.metadata.get("basis"),
        "df_rank_actual": hamiltonian.n_blocks,
        "df_rank_requested": hamiltonian.metadata.get("df_rank_requested"),
        "df_tol_requested": hamiltonian.metadata.get("df_tol_requested"),
        "pf_label": pf_label,
        "order": pf_order(pf_label),
        "ld": int(partition.ld),
        "deterministic_block_indices": list(partition.deterministic_block_indices),
        "t_values": [float(value) for value in t_values],
        "evolution_backend": evolution_backend,
        "gpu_ids": [str(value) for value in gpu_ids],
        "chunk_splits": int(chunk_splits),
        "optimization_level": int(optimization_level),
        "parallel_times": bool(parallel_times),
        "processes": None if processes is None else int(processes),
        "use_parameterized_template": bool(use_parameterized_template),
        "circuit_builder": "df_trotter.build_df_trotter_circuit",
        "diagonal_sort": diagonal_sort,
        "weight_rule": partition.weight_rule,
        "ground_state_tol": float(ground_state_tol),
        "ground_state_ncv": None if ground_state_ncv is None else int(ground_state_ncv),
    }


def _fit_result_from_record(record: dict[str, Any]) -> DFCgsFitResult:
    return DFCgsFitResult(
        representation_type=str(record["representation_type"]),
        cgs_definition=str(record["cgs_definition"]),
        pf_label=str(record["pf_label"]),
        order=int(record["order"]),
        ld=int(record["ld"]),
        lambda_r=float(record["lambda_r"]),
        t_values=tuple(float(value) for value in record["t_values"]),
        perturbation_errors=tuple(
            float(value) for value in record["perturbation_errors"]
        ),
        coeff=float(record["coeff"]),
        fit_coeff_fixed_order=float(record["fit_coeff_fixed_order"]),
        fit_slope=None if record.get("fit_slope") is None else float(record["fit_slope"]),
        fit_coeff=None if record.get("fit_coeff") is None else float(record["fit_coeff"]),
        evolution_backend=str(record["evolution_backend"]),
        gpu_ids=tuple(str(value) for value in record.get("gpu_ids", ())),
        chunk_splits=int(record["chunk_splits"]),
        optimization_level=int(record["optimization_level"]),
        parallel_times=bool(record.get("parallel_times", False)),
        processes=int(record.get("processes", 1)),
        weight_rule=str(record["weight_rule"]),
        df_rank_actual=int(record["df_rank_actual"]),
        df_rank_requested=(
            None
            if record.get("df_rank_requested") is None
            else int(record["df_rank_requested"])
        ),
        df_tol_requested=(
            None
            if record.get("df_tol_requested") is None
            else float(record["df_tol_requested"])
        ),
        metadata=dict(record.get("metadata", {})),
        simulation_profiles=tuple(record.get("simulation_profiles", ())),
    )


def _record_from_fit_result(result: DFCgsFitResult) -> dict[str, Any]:
    return result.to_dict()


def fit_df_cgs_with_perturbation(
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    partition: DFFragmentPartition,
    pf_label: PFLabel,
    *,
    t_values: Sequence[float] | None = None,
    evolution_backend: DFEvolutionBackend = "gpu",
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    diagonal_sort: str = "descending_abs",
    matrix_free_backend: str = "auto",
    matrix_free_threads: int | None = None,
    matrix_free_block_chunk_size: int | None = None,
    ground_state_ncv: int | None = None,
    ground_state_tol: float = 1e-10,
    parallel_times: bool = True,
    processes: int | None = None,
    use_parameterized_template: bool = True,
    use_ground_state_cache: bool = True,
    ground_state_cache_dir: str | Path = PARTIAL_RANDOMIZED_DF_GROUND_STATE_CACHE_DIR,
    debug: bool = False,
) -> DFCgsFitResult:
    """Fit the DF H_D deterministic surrogate C_gs,D with a DF-native circuit."""
    if t_values is None:
        molecule_type = int(hamiltonian.metadata.get("molecule_type", 2))
        t_values = default_perturbation_t_values(molecule_type, pf_label)
    t_values = tuple(float(value) for value in t_values)
    order = pf_order(pf_label)
    h_d = select_df_h_d(hamiltonian, partition)
    if partition.ld == 0 and h_d.n_blocks == 0 and np.linalg.norm(h_d.one_body) <= 1e-14:
        df_step_cost = {
            "cost_definition": "df_project_total_ref_rz_depth_v1",
            "pf_label": str(pf_label),
            "total_ref_rz_count": 0,
            "total_ref_rz_depth": 0,
            "num_cost_blocks": 0,
        }
        return DFCgsFitResult(
            representation_type="df",
            cgs_definition=_DF_CGS_DEFINITION,
            pf_label=pf_label,
            order=order,
            ld=partition.ld,
            lambda_r=partition.lambda_r,
            t_values=tuple(),
            perturbation_errors=tuple(),
            coeff=0.0,
            fit_coeff_fixed_order=0.0,
            fit_slope=None,
            fit_coeff=None,
            evolution_backend=evolution_backend,
            gpu_ids=tuple(str(value) for value in gpu_ids),
            chunk_splits=int(chunk_splits),
            optimization_level=int(optimization_level),
            parallel_times=bool(parallel_times),
            processes=0,
            weight_rule=partition.weight_rule,
            df_rank_actual=hamiltonian.n_blocks,
            df_rank_requested=hamiltonian.metadata.get("df_rank_requested"),
            df_tol_requested=hamiltonian.metadata.get("df_tol_requested"),
            metadata={
                "surrogate_note": "DF H_D deterministic surrogate",
                "df_step_cost": df_step_cost,
            },
        )

    df_step_cost = df_deterministic_step_rz_cost(
        h_d,
        pf_label,
        time=1.0,
        diagonal_sort=diagonal_sort,
    )

    ground_state_cache_metadata: dict[str, Any]
    if use_ground_state_cache:
        ground_state, ground_state_cache_metadata = get_or_compute_cached_df_ground_state(
            h_d,
            sector,
            cache_dir=ground_state_cache_dir,
            matrix_free_backend=matrix_free_backend,
            matrix_free_threads=matrix_free_threads,
            matrix_free_block_chunk_size=matrix_free_block_chunk_size,
            ground_state_ncv=ground_state_ncv,
            ground_state_tol=ground_state_tol,
        )
    else:
        ground_state = solve_df_ground_state(
            h_d,
            sector,
            matrix_free_backend=matrix_free_backend,
            matrix_free_threads=matrix_free_threads,
            matrix_free_block_chunk_size=matrix_free_block_chunk_size,
            tol=ground_state_tol,
            ncv=ground_state_ncv,
            expand_state=True,
        )
        ground_state_cache_metadata = {
            "ground_state_cache_hit": False,
            "ground_state_cache_disabled": True,
        }
    state_flat = _to_qiskit_state_order(
        ground_state.state_vector,
        h_d.n_qubits,
    )
    blocks = build_df_hd_trotter_blocks(h_d, diagonal_sort=diagonal_sort)
    template: DFGPUParameterizedTemplate | None = None
    template_profile: dict[str, Any] | None = None
    if (
        evolution_backend != "cpu"
        and use_parameterized_template
        and int(chunk_splits) == 1
    ):
        time_parameter = Parameter("t")
        template_qc = build_df_trotter_circuit(
            blocks,
            time=time_parameter,
            num_qubits=h_d.n_qubits,
            pf_label=pf_label,
            energy_shift=h_d.constant,
        )
        template = build_parameterized_gpu_template(
            template_qc,
            state_flat,
            time_parameter_name=time_parameter.name,
            gpu_ids=gpu_ids,
            optimization_level=int(optimization_level),
            debug=debug,
            debug_label="df_cgs_template",
        )
        template_profile = dict(template.prepare_profile)

    final_state_list: list[tuple[float, np.ndarray]] = []
    profiles: list[dict[str, Any]] = []
    resolved_processes = 1
    if evolution_backend == "cpu":
        for time_value in t_values:
            raw_time = float(time_value)
            qc = build_df_trotter_circuit(
                blocks,
                time=raw_time,
                num_qubits=h_d.n_qubits,
                pf_label=pf_label,
                energy_shift=h_d.constant,
            )
            evolved = simulate_statevector(qc, state_flat)
            final_state_list.append((raw_time, evolved))
            profiles.append({"backend": "qiskit_statevector_cpu", "time": raw_time})
    else:
        assigned_gpu_ids = _assign_gpu_ids_to_times(t_values, gpu_ids)
        resolved_processes = _resolve_parallel_processes(
            num_times=len(t_values),
            num_gpus=len(set(assigned_gpu_ids)),
            processes=processes,
        )
        task_args = [
            (
                float(time_value),
                tuple(blocks),
                int(h_d.n_qubits),
                pf_label,
                float(h_d.constant),
                state_flat,
                str(assigned_gpu_ids[idx]),
                int(chunk_splits),
                int(optimization_level),
                bool(debug),
            )
            for idx, time_value in enumerate(t_values)
        ]
        _set_df_time_worker_template(template)
        try:
            if parallel_times and resolved_processes > 1:
                ctx = _get_pool_context()
                with ctx.Pool(
                    processes=resolved_processes,
                    initializer=_set_df_time_worker_template,
                    initargs=(template,),
                ) as pool:
                    raw_results = list(
                        pool.map(_simulate_df_time_task, task_args, chunksize=1)
                    )
            else:
                resolved_processes = 1
                raw_results = [_simulate_df_time_task(args) for args in task_args]
        finally:
            _set_df_time_worker_template(None)
        raw_results.sort(key=lambda item: item[0])
        final_state_list = [(time_value, evolved) for time_value, evolved, _ in raw_results]
        profiles = [dict(profile) for _, _, profile in raw_results]

    times_out, perturbation_errors = _collect_df_perturbation_errors(
        final_state_list,
        float(np.real(ground_state.energy)),
        state_flat,
    )
    coeff, fixed_coeff, fit_slope, fit_coeff = _fit_errors(
        pf_label=pf_label,
        times_out=times_out,
        perturbation_errors=perturbation_errors,
    )
    return DFCgsFitResult(
        representation_type="df",
        cgs_definition=_DF_CGS_DEFINITION,
        pf_label=pf_label,
        order=order,
        ld=partition.ld,
        lambda_r=partition.lambda_r,
        t_values=times_out,
        perturbation_errors=perturbation_errors,
        coeff=coeff,
        fit_coeff_fixed_order=fixed_coeff,
        fit_slope=fit_slope,
        fit_coeff=fit_coeff,
        evolution_backend=evolution_backend,
        gpu_ids=tuple(str(value) for value in gpu_ids),
        chunk_splits=int(chunk_splits),
        optimization_level=int(optimization_level),
        parallel_times=bool(parallel_times and evolution_backend != "cpu"),
        processes=int(resolved_processes),
        weight_rule=partition.weight_rule,
        df_rank_actual=hamiltonian.n_blocks,
        df_rank_requested=hamiltonian.metadata.get("df_rank_requested"),
        df_tol_requested=hamiltonian.metadata.get("df_tol_requested"),
        metadata={
            "surrogate_note": "DF H_D deterministic surrogate; not a full partial-randomized error bound",
            "ground_state_converged": ground_state.converged,
            "ground_state_residual_norm": ground_state.residual_norm,
            "ground_state_cache": ground_state_cache_metadata,
            "df_step_cost": df_step_cost,
            "deterministic_block_indices": list(partition.deterministic_block_indices),
            "randomized_block_indices": list(partition.randomized_block_indices),
            "parallel_times": bool(parallel_times and evolution_backend != "cpu"),
            "processes": int(resolved_processes),
            "use_parameterized_template": bool(template is not None),
            "parameterized_template_profile": template_profile,
        },
        simulation_profiles=tuple(profiles),
    )


def get_or_compute_cached_df_cgs_fit(
    *,
    hamiltonian: DFHamiltonian,
    sector: PhysicalSector,
    partition: DFFragmentPartition,
    pf_label: PFLabel,
    cache_document: dict[str, Any] | None = None,
    cache_path: str | Path = PARTIAL_RANDOMIZED_DF_CGS_CACHE_PATH,
    t_values: Sequence[float] | None = None,
    evolution_backend: DFEvolutionBackend = "gpu",
    gpu_ids: Sequence[str] = ("0",),
    chunk_splits: int = 1,
    optimization_level: int = 0,
    diagonal_sort: str = "descending_abs",
    matrix_free_backend: str = "auto",
    matrix_free_threads: int | None = None,
    matrix_free_block_chunk_size: int | None = None,
    ground_state_ncv: int | None = None,
    ground_state_tol: float = 1e-10,
    parallel_times: bool = True,
    processes: int | None = None,
    use_parameterized_template: bool = True,
    use_ground_state_cache: bool = True,
    ground_state_cache_dir: str | Path = PARTIAL_RANDOMIZED_DF_GROUND_STATE_CACHE_DIR,
    debug: bool = False,
) -> DFCgsFitResult:
    if t_values is None:
        molecule_type = int(hamiltonian.metadata.get("molecule_type", 2))
        t_values = default_perturbation_t_values(molecule_type, pf_label)
    t_values = tuple(float(value) for value in t_values)
    if cache_document is None:
        cache_document = load_df_cgs_json_cache(cache_path)

    hamiltonian_hash = _df_hamiltonian_hash(
        hamiltonian,
        weight_rule=partition.weight_rule,
    )
    key_payload = _cache_key_payload(
        hamiltonian=hamiltonian,
        hamiltonian_hash=hamiltonian_hash,
        partition=partition,
        pf_label=pf_label,
        t_values=t_values,
        evolution_backend=evolution_backend,
        gpu_ids=gpu_ids,
        chunk_splits=chunk_splits,
        optimization_level=optimization_level,
        parallel_times=parallel_times,
        processes=processes,
        use_parameterized_template=use_parameterized_template,
        diagonal_sort=diagonal_sort,
        ground_state_tol=ground_state_tol,
        ground_state_ncv=ground_state_ncv,
    )
    cache_key = _json_hash(key_payload)
    entries = cache_document["entries"]
    record = entries.get(cache_key)
    if isinstance(record, dict):
        try:
            return _fit_result_from_record(record)
        except (KeyError, TypeError, ValueError):
            entries.pop(cache_key, None)

    fit_result = fit_df_cgs_with_perturbation(
        hamiltonian,
        sector,
        partition,
        pf_label,
        t_values=t_values,
        evolution_backend=evolution_backend,
        gpu_ids=gpu_ids,
        chunk_splits=chunk_splits,
        optimization_level=optimization_level,
        diagonal_sort=diagonal_sort,
        matrix_free_backend=matrix_free_backend,
        matrix_free_threads=matrix_free_threads,
        matrix_free_block_chunk_size=matrix_free_block_chunk_size,
        ground_state_ncv=ground_state_ncv,
        ground_state_tol=ground_state_tol,
        parallel_times=parallel_times,
        processes=processes,
        use_parameterized_template=use_parameterized_template,
        use_ground_state_cache=use_ground_state_cache,
        ground_state_cache_dir=ground_state_cache_dir,
        debug=debug,
    )
    entries[cache_key] = _record_from_fit_result(fit_result)
    save_df_cgs_json_cache(cache_document, cache_path)
    return fit_result
