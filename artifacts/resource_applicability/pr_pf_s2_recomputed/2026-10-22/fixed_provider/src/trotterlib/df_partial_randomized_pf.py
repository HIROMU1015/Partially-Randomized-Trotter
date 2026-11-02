from __future__ import annotations

import hashlib
import json
import math
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
    AerSimulator,
    DFGPUParameterizedTemplate,
    build_parameterized_gpu_template,
    run_parameterized_gpu_template,
    simulate_statevector_gpu,
)
from .df_hamiltonian import (
    DFGroundStateResult,
    DFHamiltonian,
    PhysicalSector,
    _NUMBA_AVAILABLE,
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
from .rte import require_integer_count
from .partial_randomized_pf import (
    _PERTURBATION_NOISE_FLOOR,
)


DFFragmentWeightRule: TypeAlias = Literal[
    "lambda_frobenius_squared",
    "abs_lambda",
]
DFEvolutionBackend: TypeAlias = Literal["gpu", "cpu", "auto"]
DFPhaseBiasStatus: TypeAlias = Literal[
    "ok",
    "true_zero",
    "low_overlap",
    "branch_ambiguous",
    "ground_state_unconverged",
    "residual_dominated",
    "fit_unstable",
    "below_noise_floor",
]
_DF_CGS_CACHE_SCHEMA_VERSION = 8
_DF_CGS_DEFINITION = "df_phase_bias_surrogate_v3"
_DF_GROUND_STATE_CACHE_SCHEMA_VERSION = 4
_DF_PHASE_BIAS_ESTIMATE_KIND = "state_specific_phase_bias_surrogate"
_DF_PHASE_BIAS_ESTIMATOR_VERSION = "df_phase_bias_v1"
_DF_PHASE_BIAS_STATUSES = {
    "ok",
    "true_zero",
    "low_overlap",
    "branch_ambiguous",
    "ground_state_unconverged",
    "residual_dominated",
    "fit_unstable",
    "below_noise_floor",
}
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
    """DF-native H_D/H_R split for one L_D.

    ``lambda_r`` is retained for compatibility and is a fragment-ranking
    proxy, not the exact involutory RTE one-norm.
    """

    ld: int
    deterministic_fragments: tuple[RankedDFFragment, ...]
    randomized_fragments: tuple[RankedDFFragment, ...]
    deterministic_block_indices: tuple[int, ...]
    randomized_block_indices: tuple[int, ...]
    lambda_r: float
    weight_rule: str

    @property
    def ranking_proxy_lambda_r(self) -> float:
        """Explicit name for the legacy fragment-weight tail sum."""
        return self.lambda_r


@dataclass(frozen=True)
class DFCgsFitResult:
    """State-specific DF phase-bias fit; never a rigorous PR error bound."""

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
    estimate_kind: str = _DF_PHASE_BIAS_ESTIMATE_KIND
    is_rigorous_bound: bool = False
    estimator_status: DFPhaseBiasStatus = "ok"
    signed_phase_biases: tuple[float, ...] = ()
    relative_overlap_magnitudes: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _DFPhaseBiasSeries:
    times: tuple[float, ...]
    absolute_biases: tuple[float, ...]
    signed_biases: tuple[float, ...]
    overlap_magnitudes: tuple[float, ...]
    unwrapped_phases: tuple[float, ...]
    status: DFPhaseBiasStatus
    minimum_branch_cut_clearance: float
    maximum_adjacent_phase_increment: float


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
            -fragment.weight,
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
    ld = require_integer_count(ld, name="ld")
    ranked_fragments = tuple(ranked_fragments)
    if len(ranked_fragments) != hamiltonian.n_blocks:
        raise ValueError("ranked_fragments must cover every DF fragment exactly once.")
    ranked_indices = tuple(fragment.original_index for fragment in ranked_fragments)
    if len(set(ranked_indices)) != len(ranked_indices) or set(ranked_indices) != set(
        range(hamiltonian.n_blocks)
    ):
        raise ValueError("ranked_fragments must be a permutation of all DF fragments.")
    if tuple(fragment.rank for fragment in ranked_fragments) != tuple(
        range(hamiltonian.n_blocks)
    ):
        raise ValueError("ranked_fragments ranks must be contiguous in tuple order.")
    if any(fragment.weight_rule != weight_rule for fragment in ranked_fragments):
        raise ValueError("ranked_fragments weight rule does not match weight_rule.")
    if ld > len(ranked_fragments):
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
    if include_one_body and np.any(np.asarray(model.one_body_correction) != 0.0):
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
    if np.any(np.asarray(model.one_body_correction) != 0.0):
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


def _validate_phase_times(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    if not times:
        raise ValueError(f"{name} must not be empty.")
    if any(not np.isfinite(value) or value <= 0.0 for value in times):
        raise ValueError(f"{name} must contain finite positive times.")
    if len(set(times)) != len(times):
        raise ValueError(f"{name} must not contain duplicate times.")
    return tuple(sorted(times))


def default_df_phase_bias_t_values(
    molecule_type: int,
    pf_label: PFLabel,
) -> tuple[float, ...]:
    """Return the versioned, small-time log grid used by the DF estimator."""
    if int(molecule_type) < 1:
        raise ValueError("molecule_type must be positive.")
    # Higher-order formulas need a slightly later window to remain above the
    # statevector noise floor.  Both grids remain geometric so fixed-order
    # coefficients can be compared across nested fit windows.
    start = 0.01 if pf_order(pf_label) <= 4 else 0.025
    return tuple(float(start * (2**index)) for index in range(4))


def _block_operator_norm_upper_bound(block: Block) -> float:
    if block.kind == "one_body_gaussian":
        return float(np.sum(np.abs(np.asarray(block.payload.eps, dtype=np.complex128))))
    if block.kind == "df":
        eta_l1 = float(
            np.sum(np.abs(np.asarray(block.payload.eta, dtype=np.complex128)))
        )
        return float(abs(float(block.payload.lam)) * eta_l1 * eta_l1)
    raise ValueError(f"No DF phase-rate bound is defined for block kind {block.kind!r}.")


def _pf_phase_rate_upper_bound(
    blocks: Sequence[Block],
    pf_label: PFLabel,
    energy: float,
    *,
    energy_shift: float = 0.0,
) -> float:
    block_bounds = tuple(_block_operator_norm_upper_bound(block) for block in blocks)
    product_variation = math.fsum(
        abs(float(weight)) * block_bounds[int(term_index)]
        for term_index, weight in iter_pf_steps(len(blocks), _get_w_list(pf_label))
    )
    # The scalar circuit shift cancels from the relative survival phase.  Use
    # E-shift here so the certificate is both shift-invariant and conservative.
    return float(abs(float(energy) - float(energy_shift)) + product_variation)


def _phase_tracking_times(
    fit_times: Sequence[float],
    *,
    phase_rate_upper_bound: float,
    minimum_overlap: float,
    branch_clearance: float,
    maximum_tracking_points: int,
) -> tuple[tuple[float, ...], bool, float | None]:
    """Return a 0-anchored, a-priori branch-safe simulation grid when feasible."""
    fit = _validate_phase_times(fit_times, name="fit times")
    maximum_tracking_points = require_integer_count(
        maximum_tracking_points,
        name="maximum_tracking_points",
        minimum=1,
    )
    if len(fit) > maximum_tracking_points:
        raise ValueError(
            "fit grid exceeds maximum_tracking_points before branch refinement."
        )
    if not math.isfinite(minimum_overlap) or not 0.0 < minimum_overlap <= 1.0:
        raise ValueError("minimum_overlap must be finite and in (0, 1].")
    if not math.isfinite(branch_clearance) or not 0.0 < branch_clearance < math.pi:
        raise ValueError("branch_clearance must be finite and in (0, pi).")
    if not math.isfinite(phase_rate_upper_bound) or phase_rate_upper_bound < 0.0:
        raise ValueError("phase_rate_upper_bound must be finite and non-negative.")
    if phase_rate_upper_bound == 0.0:
        return fit, True, None

    maximum_step = (
        (math.pi - branch_clearance) * minimum_overlap / phase_rate_upper_bound
    )
    required_intervals = int(math.ceil(max(fit) / maximum_step))
    tracking = {
        float(index * max(fit) / required_intervals)
        for index in range(1, required_intervals + 1)
    }
    tracking.update(fit)
    if len(tracking) > maximum_tracking_points:
        return fit, False, maximum_step
    return tuple(sorted(tracking)), True, maximum_step


def _collect_df_phase_bias_series(
    final_state_list: Sequence[tuple[float, np.ndarray]],
    energy: float,
    state_vec: np.ndarray,
    *,
    fit_times: Sequence[float] | None = None,
    minimum_overlap: float = 0.25,
    branch_clearance: float = 0.1,
    branch_certified: bool = False,
) -> _DFPhaseBiasSeries:
    """Estimate the shift-invariant survival-phase bias on selected fit times."""
    if not math.isfinite(float(energy)):
        raise ValueError("energy must be finite.")
    ordered_states = sorted(
        ((float(time), np.asarray(state, dtype=np.complex128).reshape(-1))
         for time, state in final_state_list),
        key=lambda item: item[0],
    )
    tracking_times = _validate_phase_times(
        tuple(time for time, _state in ordered_states),
        name="tracking times",
    )
    if tuple(time for time, _state in ordered_states) != tracking_times:
        raise ValueError("tracking times must be unique after normalization.")
    selected_times = (
        tracking_times
        if fit_times is None
        else _validate_phase_times(fit_times, name="fit times")
    )
    tracking_set = set(tracking_times)
    if any(time not in tracking_set for time in selected_times):
        raise ValueError("Every fit time must be present in the tracking grid.")

    psi0 = np.asarray(state_vec, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(psi0))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("state_vec must have finite nonzero norm.")
    psi0 = psi0 / norm
    relative_overlaps: list[complex] = []
    for time, evolved in ordered_states:
        if evolved.shape != psi0.shape:
            raise ValueError("Evolved state dimension differs from state_vec.")
        evolved_norm = float(np.linalg.norm(evolved))
        if not math.isfinite(evolved_norm) or evolved_norm == 0.0:
            raise ValueError("Evolved states must have finite nonzero norm.")
        normalized_evolved = evolved / evolved_norm
        overlap = np.vdot(psi0, normalized_evolved)
        relative_overlaps.append(complex(np.exp(1j * energy * time) * overlap))

    magnitudes = np.asarray([abs(value) for value in relative_overlaps], dtype=float)
    principal_phases = np.asarray([np.angle(value) for value in relative_overlaps])
    anchored_phases = np.unwrap(np.concatenate(([0.0], principal_phases)))
    phases = anchored_phases[1:]
    increments = np.diff(anchored_phases)
    maximum_increment = float(np.max(np.abs(increments))) if increments.size else 0.0
    minimum_clearance = float(math.pi - maximum_increment)
    status: DFPhaseBiasStatus = "ok"
    if np.any(magnitudes < minimum_overlap):
        status = "low_overlap"
    elif (
        not branch_certified
        or maximum_increment >= math.pi - branch_clearance
        or minimum_clearance < branch_clearance
    ):
        status = "branch_ambiguous"

    index_by_time = {time: index for index, time in enumerate(tracking_times)}
    selected_indices = [index_by_time[time] for time in selected_times]
    signed = tuple(
        float(-phases[index] / selected_times[position])
        for position, index in enumerate(selected_indices)
    )
    selected_magnitudes = tuple(float(magnitudes[index]) for index in selected_indices)
    selected_phases = tuple(float(phases[index]) for index in selected_indices)
    return _DFPhaseBiasSeries(
        times=selected_times,
        absolute_biases=tuple(abs(value) for value in signed),
        signed_biases=signed,
        overlap_magnitudes=selected_magnitudes,
        unwrapped_phases=selected_phases,
        status=status,
        minimum_branch_cut_clearance=minimum_clearance,
        maximum_adjacent_phase_increment=maximum_increment,
    )


def _classify_df_phase_bias_status(
    phase_series: _DFPhaseBiasSeries,
    *,
    ground_state_converged: bool,
    ground_state_residual_norm: float,
    order: int,
    fit_slope: float | None,
    window_relative_spread: float,
    residual_dominance_factor: float,
    fit_slope_tolerance: float,
    fit_window_relative_tolerance: float,
) -> DFPhaseBiasStatus:
    """Apply the versioned estimator-status precedence independently of I/O."""
    if not ground_state_converged:
        return "ground_state_unconverged"
    if phase_series.status != "ok":
        return phase_series.status
    maximum_bias = max(phase_series.absolute_biases, default=0.0)
    if maximum_bias < _PERTURBATION_NOISE_FLOOR:
        return "below_noise_floor"
    residual_floor = residual_dominance_factor * float(ground_state_residual_norm)
    if maximum_bias <= residual_floor:
        return "residual_dominated"
    if (
        fit_slope is not None
        and len(phase_series.times) >= 3
        and abs(fit_slope - order) > fit_slope_tolerance
    ):
        return "fit_unstable"
    if window_relative_spread > fit_window_relative_tolerance:
        return "fit_unstable"
    return "ok"


def _collect_df_perturbation_errors(
    final_state_list: Sequence[tuple[float, np.ndarray]],
    energy: float,
    state_vec: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compatibility wrapper for the shift-invariant phase-bias series."""
    series = _collect_df_phase_bias_series(
        final_state_list,
        energy,
        state_vec,
        branch_certified=False,
    )
    return series.times, series.absolute_biases


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


def _fit_window_coefficients(
    *,
    pf_label: PFLabel,
    times: Sequence[float],
    errors: Sequence[float],
) -> tuple[float, ...]:
    """Fit nested windows used only as a screening-stability diagnostic."""
    if len(times) < 3:
        return ()
    windows = ((0, len(times)), (0, len(times) - 1), (1, len(times)))
    coefficients: list[float] = []
    for start, stop in windows:
        window_errors = tuple(float(value) for value in errors[start:stop])
        if not window_errors or max(window_errors) < _PERTURBATION_NOISE_FLOOR:
            continue
        coefficients.append(
            float(
                loglog_average_coeff(
                    times[start:stop],
                    window_errors,
                    pf_order(pf_label),
                    mask_nonpositive=True,
                )
            )
        )
    return tuple(coefficients)


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
            psi0=state_flat,
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
        "constant_float_hex": float(hamiltonian.constant).hex(),
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


def df_hamiltonian_hash(
    hamiltonian: DFHamiltonian,
    *,
    weight_rule: str = "lambda_frobenius_squared",
) -> str:
    """Public canonical hash for DF circuit/preparation provenance."""
    return _df_hamiltonian_hash(hamiltonian, weight_rule=weight_rule)


def _sector_hash(sector: PhysicalSector) -> str:
    payload = {
        "n_qubits": int(sector.n_qubits),
        "dimension": int(sector.dimension),
        "basis_indices_sha256": hashlib.sha256(
            np.asarray(sector.basis_indices, dtype="<i8").tobytes()
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
    resolved_backend = _resolve_matrix_free_backend(matrix_free_backend)
    return {
        "schema_version": _DF_GROUND_STATE_CACHE_SCHEMA_VERSION,
        "hamiltonian_hash": _df_hamiltonian_hash(
            hamiltonian,
            weight_rule="ground_state",
        ),
        "sector_hash": _sector_hash(sector),
        "matrix_free_backend_requested": matrix_free_backend,
        "matrix_free_backend_resolved": resolved_backend,
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
    resolved_backend = _resolve_matrix_free_backend(matrix_free_backend)
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
            "matrix_free_backend_requested": matrix_free_backend,
            "matrix_free_backend_resolved": resolved_backend,
        }

    ground_state = solve_df_ground_state(
        hamiltonian,
        sector,
        matrix_free_backend=resolved_backend,
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
        "matrix_free_backend_requested": matrix_free_backend,
        "matrix_free_backend_resolved": resolved_backend,
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


def _float_cache_value(value: float) -> str:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("Cache-key floating values must be finite.")
    return normalized.hex()


def _resolve_evolution_backend(backend: DFEvolutionBackend) -> Literal["cpu", "gpu"]:
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        return "gpu"
    if backend != "auto":
        raise ValueError("evolution_backend must be 'cpu', 'gpu', or 'auto'.")
    if AerSimulator is None:
        return "cpu"
    try:
        devices = tuple(AerSimulator(method="statevector").available_devices())
    except Exception:
        return "cpu"
    return "gpu" if "GPU" in devices else "cpu"


def _resolve_matrix_free_backend(backend: str) -> str:
    if backend not in ("auto", "numba", "python"):
        raise ValueError("matrix_free_backend must be 'auto', 'numba', or 'python'.")
    if backend == "auto":
        return "numba" if _NUMBA_AVAILABLE else "python"
    return backend


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
    if document.get("cgs_definition") != _DF_CGS_DEFINITION:
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
    sector: PhysicalSector,
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
    matrix_free_backend: str,
    matrix_free_threads: int | None,
    matrix_free_block_chunk_size: int | None,
    minimum_overlap: float,
    branch_clearance: float,
    maximum_tracking_points: int,
    residual_dominance_factor: float,
    fit_slope_tolerance: float,
    fit_window_relative_tolerance: float,
) -> dict[str, Any]:
    return {
        "representation_type": "df",
        "cgs_definition": _DF_CGS_DEFINITION,
        "hamiltonian_hash": hamiltonian_hash,
        "sector_hash": _sector_hash(sector),
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
        "fit_grid_float_hex": [_float_cache_value(value) for value in t_values],
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
        "matrix_free_backend": matrix_free_backend,
        "matrix_free_threads": (
            None if matrix_free_threads is None else int(matrix_free_threads)
        ),
        "matrix_free_block_chunk_size": (
            None
            if matrix_free_block_chunk_size is None
            else int(matrix_free_block_chunk_size)
        ),
        "ground_state_tol_float_hex": _float_cache_value(ground_state_tol),
        "ground_state_ncv": None if ground_state_ncv is None else int(ground_state_ncv),
        "estimator_definition": _DF_PHASE_BIAS_ESTIMATOR_VERSION,
        "phase_tracking_grid_policy": "zero_anchored_norm_certified_v1",
        "minimum_overlap_float_hex": _float_cache_value(minimum_overlap),
        "branch_clearance_float_hex": _float_cache_value(branch_clearance),
        "maximum_tracking_points": int(maximum_tracking_points),
        "residual_dominance_factor_float_hex": _float_cache_value(
            residual_dominance_factor
        ),
        "fit_slope_tolerance_float_hex": _float_cache_value(fit_slope_tolerance),
        "fit_window_policy": "full_drop_first_drop_last_v1",
        "fit_window_relative_tolerance_float_hex": _float_cache_value(
            fit_window_relative_tolerance
        ),
    }


def _fit_result_from_record(record: dict[str, Any]) -> DFCgsFitResult:
    status = str(record.get("estimator_status", "ok"))
    if status not in _DF_PHASE_BIAS_STATUSES:
        raise ValueError("Cached DF phase-bias estimator status is unsupported.")
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
        estimate_kind=str(
            record.get("estimate_kind", _DF_PHASE_BIAS_ESTIMATE_KIND)
        ),
        is_rigorous_bound=bool(record.get("is_rigorous_bound", False)),
        estimator_status=status,
        signed_phase_biases=tuple(
            float(value) for value in record.get("signed_phase_biases", ())
        ),
        relative_overlap_magnitudes=tuple(
            float(value)
            for value in record.get("relative_overlap_magnitudes", ())
        ),
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
    minimum_overlap: float = 0.25,
    branch_clearance: float = 0.1,
    maximum_tracking_points: int = 4096,
    residual_dominance_factor: float = 10.0,
    fit_slope_tolerance: float | None = None,
    fit_window_relative_tolerance: float = 0.5,
    require_usable_estimate: bool = True,
) -> DFCgsFitResult:
    """Fit a shift-invariant DF survival-phase-bias surrogate."""
    if t_values is None:
        molecule_type = int(hamiltonian.metadata.get("molecule_type", 2))
        t_values = default_df_phase_bias_t_values(molecule_type, pf_label)
    fit_times = _validate_phase_times(t_values, name="fit times")
    order = pf_order(pf_label)
    resolved_evolution_backend = _resolve_evolution_backend(evolution_backend)
    resolved_matrix_free_backend = _resolve_matrix_free_backend(matrix_free_backend)
    maximum_tracking_points = require_integer_count(
        maximum_tracking_points,
        name="maximum_tracking_points",
        minimum=1,
    )
    if not math.isfinite(residual_dominance_factor) or residual_dominance_factor <= 0.0:
        raise ValueError("residual_dominance_factor must be finite and positive.")
    if fit_slope_tolerance is None:
        fit_slope_tolerance = max(1.0, 0.5 * order)
    if not math.isfinite(fit_slope_tolerance) or fit_slope_tolerance <= 0.0:
        raise ValueError("fit_slope_tolerance must be finite and positive.")
    if (
        not math.isfinite(fit_window_relative_tolerance)
        or fit_window_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "fit_window_relative_tolerance must be finite and positive."
        )
    h_d = select_df_h_d(hamiltonian, partition)
    if (
        partition.ld == 0
        and h_d.n_blocks == 0
        and not np.any(np.asarray(h_d.one_body) != 0.0)
    ):
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
            t_values=fit_times,
            perturbation_errors=(0.0,) * len(fit_times),
            coeff=0.0,
            fit_coeff_fixed_order=0.0,
            fit_slope=None,
            fit_coeff=None,
            evolution_backend=resolved_evolution_backend,
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
                "surrogate_note": (
                    "State-specific DF survival-phase-bias surrogate; "
                    "not a rigorous partial-randomized error bound"
                ),
                "screening_usable": True,
                "estimator_definition": _DF_PHASE_BIAS_ESTIMATOR_VERSION,
                "estimator_status": "true_zero",
                "is_rigorous_bound": False,
                "sector_hash": _sector_hash(sector),
                "fit_grid_float_hex": [value.hex() for value in fit_times],
                "tracking_grid_float_hex": [value.hex() for value in fit_times],
                "tracking_grid_with_zero_anchor_float_hex": [
                    float(0.0).hex(),
                    *(value.hex() for value in fit_times),
                ],
                "df_truncation_value": hamiltonian.metadata.get(
                    "df_truncation_value"
                ),
                "df_step_cost": df_step_cost,
            },
            estimator_status="true_zero",
            signed_phase_biases=(0.0,) * len(fit_times),
            relative_overlap_magnitudes=(1.0,) * len(fit_times),
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
            matrix_free_backend=resolved_matrix_free_backend,
            matrix_free_threads=matrix_free_threads,
            matrix_free_block_chunk_size=matrix_free_block_chunk_size,
            ground_state_ncv=ground_state_ncv,
            ground_state_tol=ground_state_tol,
        )
    else:
        ground_state = solve_df_ground_state(
            h_d,
            sector,
            matrix_free_backend=resolved_matrix_free_backend,
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
    phase_rate_upper_bound = _pf_phase_rate_upper_bound(
        blocks,
        pf_label,
        float(np.real(ground_state.energy)),
        energy_shift=float(h_d.constant),
    )
    simulation_times, branch_certified, maximum_tracking_step = _phase_tracking_times(
        fit_times,
        phase_rate_upper_bound=phase_rate_upper_bound,
        minimum_overlap=minimum_overlap,
        branch_clearance=branch_clearance,
        maximum_tracking_points=maximum_tracking_points,
    )
    template: DFGPUParameterizedTemplate | None = None
    template_profile: dict[str, Any] | None = None
    if (
        resolved_evolution_backend != "cpu"
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
    if resolved_evolution_backend == "cpu":
        for time_value in simulation_times:
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
        assigned_gpu_ids = _assign_gpu_ids_to_times(simulation_times, gpu_ids)
        resolved_processes = _resolve_parallel_processes(
            num_times=len(simulation_times),
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
            for idx, time_value in enumerate(simulation_times)
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

    phase_series = _collect_df_phase_bias_series(
        final_state_list,
        float(np.real(ground_state.energy)),
        state_flat,
        fit_times=fit_times,
        minimum_overlap=minimum_overlap,
        branch_clearance=branch_clearance,
        branch_certified=branch_certified,
    )
    coeff, fixed_coeff, fit_slope, fit_coeff = _fit_errors(
        pf_label=pf_label,
        times_out=phase_series.times,
        perturbation_errors=phase_series.absolute_biases,
    )
    window_coefficients = _fit_window_coefficients(
        pf_label=pf_label,
        times=phase_series.times,
        errors=phase_series.absolute_biases,
    )
    window_relative_spread = (
        0.0
        if len(window_coefficients) < 2
        else (
            max(window_coefficients) - min(window_coefficients)
        )
        / max(max(window_coefficients), _PERTURBATION_NOISE_FLOOR)
    )
    estimator_status = _classify_df_phase_bias_status(
        phase_series,
        ground_state_converged=ground_state.converged,
        ground_state_residual_norm=ground_state.residual_norm,
        order=order,
        fit_slope=fit_slope,
        window_relative_spread=window_relative_spread,
        residual_dominance_factor=residual_dominance_factor,
        fit_slope_tolerance=fit_slope_tolerance,
        fit_window_relative_tolerance=fit_window_relative_tolerance,
    )
    screening_usable = estimator_status in ("ok", "true_zero", "below_noise_floor")
    result = DFCgsFitResult(
        representation_type="df",
        cgs_definition=_DF_CGS_DEFINITION,
        pf_label=pf_label,
        order=order,
        ld=partition.ld,
        lambda_r=partition.lambda_r,
        t_values=phase_series.times,
        perturbation_errors=phase_series.absolute_biases,
        coeff=coeff,
        fit_coeff_fixed_order=fixed_coeff,
        fit_slope=fit_slope,
        fit_coeff=fit_coeff,
        evolution_backend=resolved_evolution_backend,
        gpu_ids=tuple(str(value) for value in gpu_ids),
        chunk_splits=int(chunk_splits),
        optimization_level=int(optimization_level),
        parallel_times=bool(parallel_times and resolved_evolution_backend != "cpu"),
        processes=int(resolved_processes),
        weight_rule=partition.weight_rule,
        df_rank_actual=hamiltonian.n_blocks,
        df_rank_requested=hamiltonian.metadata.get("df_rank_requested"),
        df_tol_requested=hamiltonian.metadata.get("df_tol_requested"),
        metadata={
            "surrogate_note": (
                "State-specific DF survival-phase-bias surrogate; not a rigorous "
                "partial-randomized error bound"
            ),
            "screening_usable": screening_usable,
            "estimator_definition": _DF_PHASE_BIAS_ESTIMATOR_VERSION,
            "estimator_status": estimator_status,
            "is_rigorous_bound": False,
            "sector_hash": _sector_hash(sector),
            "fit_grid_float_hex": [value.hex() for value in phase_series.times],
            "tracking_grid_float_hex": [value.hex() for value in simulation_times],
            "minimum_overlap": float(minimum_overlap),
            "branch_clearance": float(branch_clearance),
            "branch_certified": bool(branch_certified),
            "phase_rate_upper_bound": float(phase_rate_upper_bound),
            "maximum_tracking_step": (
                None
                if maximum_tracking_step is None
                else float(maximum_tracking_step)
            ),
            "tracking_grid_with_zero_anchor_float_hex": [
                float(0.0).hex(),
                *(value.hex() for value in simulation_times),
            ],
            "minimum_branch_cut_clearance": (
                phase_series.minimum_branch_cut_clearance
            ),
            "maximum_adjacent_phase_increment": (
                phase_series.maximum_adjacent_phase_increment
            ),
            "residual_dominance_factor": float(residual_dominance_factor),
            "fit_slope_tolerance": float(fit_slope_tolerance),
            "fit_window_policy": "full_drop_first_drop_last_v1",
            "fit_window_coefficients": list(window_coefficients),
            "fit_window_relative_spread": float(window_relative_spread),
            "fit_window_relative_tolerance": float(
                fit_window_relative_tolerance
            ),
            "df_truncation_value": hamiltonian.metadata.get("df_truncation_value"),
            "ground_state_energy": float(np.real(ground_state.energy)),
            "ground_state_converged": ground_state.converged,
            "ground_state_residual_norm": ground_state.residual_norm,
            "ground_state_cache": ground_state_cache_metadata,
            "df_step_cost": df_step_cost,
            "deterministic_block_indices": list(partition.deterministic_block_indices),
            "randomized_block_indices": list(partition.randomized_block_indices),
            "parallel_times": bool(
                parallel_times and resolved_evolution_backend != "cpu"
            ),
            "processes": int(resolved_processes),
            "use_parameterized_template": bool(template is not None),
            "parameterized_template_profile": template_profile,
        },
        simulation_profiles=tuple(profiles),
        estimator_status=estimator_status,
        signed_phase_biases=phase_series.signed_biases,
        relative_overlap_magnitudes=phase_series.overlap_magnitudes,
    )
    if require_usable_estimate and not screening_usable:
        raise RuntimeError(
            "DF phase-bias estimate is not usable for screening: "
            f"status={estimator_status}."
        )
    return result


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
    minimum_overlap: float = 0.25,
    branch_clearance: float = 0.1,
    maximum_tracking_points: int = 4096,
    residual_dominance_factor: float = 10.0,
    fit_slope_tolerance: float | None = None,
    fit_window_relative_tolerance: float = 0.5,
    require_usable_estimate: bool = True,
) -> DFCgsFitResult:
    if t_values is None:
        molecule_type = int(hamiltonian.metadata.get("molecule_type", 2))
        t_values = default_df_phase_bias_t_values(molecule_type, pf_label)
    t_values = _validate_phase_times(t_values, name="fit times")
    resolved_evolution_backend = _resolve_evolution_backend(evolution_backend)
    resolved_matrix_free_backend = _resolve_matrix_free_backend(matrix_free_backend)
    resolved_fit_slope_tolerance = (
        max(1.0, 0.5 * pf_order(pf_label))
        if fit_slope_tolerance is None
        else float(fit_slope_tolerance)
    )
    maximum_tracking_points = require_integer_count(
        maximum_tracking_points,
        name="maximum_tracking_points",
        minimum=1,
    )
    if not math.isfinite(minimum_overlap) or not 0.0 < minimum_overlap <= 1.0:
        raise ValueError("minimum_overlap must be finite and in (0, 1].")
    if not math.isfinite(branch_clearance) or not 0.0 < branch_clearance < math.pi:
        raise ValueError("branch_clearance must be finite and in (0, pi).")
    if not math.isfinite(residual_dominance_factor) or residual_dominance_factor <= 0.0:
        raise ValueError("residual_dominance_factor must be finite and positive.")
    if (
        not math.isfinite(resolved_fit_slope_tolerance)
        or resolved_fit_slope_tolerance <= 0.0
    ):
        raise ValueError("fit_slope_tolerance must be finite and positive.")
    if (
        not math.isfinite(fit_window_relative_tolerance)
        or fit_window_relative_tolerance <= 0.0
    ):
        raise ValueError(
            "fit_window_relative_tolerance must be finite and positive."
        )
    if cache_document is None:
        cache_document = load_df_cgs_json_cache(cache_path)
    elif (
        cache_document.get("schema_version") != _DF_CGS_CACHE_SCHEMA_VERSION
        or cache_document.get("cgs_definition") != _DF_CGS_DEFINITION
        or not isinstance(cache_document.get("entries"), dict)
    ):
        cache_document = _default_cache_document()

    hamiltonian_hash = _df_hamiltonian_hash(
        hamiltonian,
        weight_rule=partition.weight_rule,
    )
    key_payload = _cache_key_payload(
        hamiltonian=hamiltonian,
        hamiltonian_hash=hamiltonian_hash,
        sector=sector,
        partition=partition,
        pf_label=pf_label,
        t_values=t_values,
        evolution_backend=resolved_evolution_backend,
        gpu_ids=gpu_ids,
        chunk_splits=chunk_splits,
        optimization_level=optimization_level,
        parallel_times=parallel_times,
        processes=processes,
        use_parameterized_template=use_parameterized_template,
        diagonal_sort=diagonal_sort,
        ground_state_tol=ground_state_tol,
        ground_state_ncv=ground_state_ncv,
        matrix_free_backend=resolved_matrix_free_backend,
        matrix_free_threads=matrix_free_threads,
        matrix_free_block_chunk_size=matrix_free_block_chunk_size,
        minimum_overlap=minimum_overlap,
        branch_clearance=branch_clearance,
        maximum_tracking_points=maximum_tracking_points,
        residual_dominance_factor=residual_dominance_factor,
        fit_slope_tolerance=resolved_fit_slope_tolerance,
        fit_window_relative_tolerance=fit_window_relative_tolerance,
    )
    cache_key = _json_hash(key_payload)
    entries = cache_document["entries"]
    record = entries.get(cache_key)
    if isinstance(record, dict):
        try:
            if record.get("cache_key") != cache_key:
                raise ValueError("Cached Cgs key mismatch.")
            if record.get("cache_key_payload_sha256") != _json_hash(key_payload):
                raise ValueError("Cached Cgs payload mismatch.")
            fit_result = _fit_result_from_record(record["result"])
            if (
                fit_result.representation_type != "df"
                or fit_result.cgs_definition != _DF_CGS_DEFINITION
                or fit_result.estimate_kind != _DF_PHASE_BIAS_ESTIMATE_KIND
                or fit_result.is_rigorous_bound
                or fit_result.pf_label != pf_label
                or fit_result.order != pf_order(pf_label)
                or fit_result.ld != partition.ld
                or fit_result.weight_rule != partition.weight_rule
                or fit_result.t_values != tuple(t_values)
                or fit_result.evolution_backend != resolved_evolution_backend
                or fit_result.metadata.get("sector_hash") != _sector_hash(sector)
                or fit_result.metadata.get("estimator_definition")
                != _DF_PHASE_BIAS_ESTIMATOR_VERSION
            ):
                raise ValueError("Cached DF phase-bias result metadata mismatch.")
            if require_usable_estimate and not bool(
                fit_result.metadata.get("screening_usable", False)
            ):
                raise RuntimeError(
                    "Cached DF phase-bias estimate is not usable for screening: "
                    f"status={fit_result.estimator_status}."
                )
            return fit_result
        except RuntimeError:
            raise
        except (KeyError, TypeError, ValueError):
            entries.pop(cache_key, None)

    fit_result = fit_df_cgs_with_perturbation(
        hamiltonian,
        sector,
        partition,
        pf_label,
        t_values=t_values,
        evolution_backend=resolved_evolution_backend,
        gpu_ids=gpu_ids,
        chunk_splits=chunk_splits,
        optimization_level=optimization_level,
        diagonal_sort=diagonal_sort,
        matrix_free_backend=resolved_matrix_free_backend,
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
        minimum_overlap=minimum_overlap,
        branch_clearance=branch_clearance,
        maximum_tracking_points=maximum_tracking_points,
        residual_dominance_factor=residual_dominance_factor,
        fit_slope_tolerance=resolved_fit_slope_tolerance,
        fit_window_relative_tolerance=fit_window_relative_tolerance,
        require_usable_estimate=require_usable_estimate,
    )
    entries[cache_key] = {
        "cache_key": cache_key,
        "cache_key_payload_sha256": _json_hash(key_payload),
        "result": _record_from_fit_result(fit_result),
    }
    save_df_cgs_json_cache(cache_document, cache_path)
    return fit_result
