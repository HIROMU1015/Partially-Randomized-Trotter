from __future__ import annotations

import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from openfermion.ops import QubitOperator
from qiskit import QuantumCircuit

from .analysis_utils import loglog_average_coeff
from .chemistry_hamiltonian import (
    ham_ground_energy,
    jw_hamiltonian_maker,
    min_hamiltonian_grouper,
)
from .config import (
    BETA,
    DECOMPO_NUM,
    DEFAULT_BASIS,
    DEFAULT_DISTANCE,
    PFLabel,
    PF_RZ_LAYER,
    P_DIR,
    pf_order,
)
from .df_gpu_statevector import simulate_statevector_gpu
from .io_cache import load_data
from .partial_randomized_pf import default_perturbation_t_values
from .pf_decomposition import iter_pf_steps
from .product_formula import _get_w_list
from .qiskit_time_evolution_grouping import (
    tEvolution_vector_grouper,
    w_trotter_grouper,
)
from .qiskit_time_evolution_pyscf import make_fci_vector_from_pyscf_solver_grouper
from .rz_layers import (
    _layers_from_z_terms,
    estimate_rz_layers_from_grouping,
    extract_z_like_terms_from_qubit_group,
)
from .uwc import UWCConfig, preprocess_qubit_hamiltonian, qubit_num_terms


_ZERO_TOL = 1e-12
_GROUPING_ORIGINAL_PF_LABEL_MAP: dict[str, str] = {
    "2nd": "w2",
    "4th": "w3",
    "4th(new_2)": "wmy4",
    "8th(Morales)": "w8",
    "10th(Morales)": "w1016",
    "8th(Yoshida)": "wyoshida",
}


@dataclass(frozen=True)
class GroupedHamiltonianData:
    molecule_type: int
    molecule: str
    basis: str
    distance: float
    ham_name: str
    num_qubits: int
    jw_hamiltonian: QubitOperator
    cliques: tuple[tuple[QubitOperator, ...], ...]
    ground_energy: float
    state_vector: np.ndarray
    grouping_rule: str
    identity_coeff: float


@dataclass(frozen=True)
class GroupedPFQPERow:
    molecule: str
    molecule_type: int
    method: str
    uwc_method: str
    uwc_objective: str
    pf_label: str
    order: int
    grouping_rule: str
    num_groups: int
    num_pauli_terms: int
    alpha: float
    qpe_iteration_factor: float
    step_pauli_rotations: int
    total_pauli_rotations: float
    step_rz_layers: int
    total_rz_layers: float
    step_t_depth: float
    total_t_depth: float
    cost_metric: str
    g_total: float
    cost_ratio_vs_grouped_baseline: float
    alpha_ratio_vs_grouped_baseline: float
    step_cost_ratio_vs_grouped_baseline: float
    qpe_beta: float
    target_error: float
    fitting_rule: str
    time_grid: tuple[float, ...]
    alpha_source: str
    uwc_hamiltonian_hash: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GroupedUWCComparisonResult:
    molecule: str
    molecule_type: int
    basis: str
    distance: float
    spin_charge_sector: dict[str, Any]
    pf_label: str
    order: int
    grouping_rule: str
    cost_metric: str
    rows: tuple[GroupedPFQPERow, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroupedAlphaFitResult:
    alpha: float
    times: tuple[float, ...]
    errors: tuple[float, ...]
    backend: str
    requested_backend: str
    gpu_ids: tuple[str, ...]
    parallel_processes: int
    chunk_splits: int
    optimization_level: int
    profiles: tuple[dict[str, Any], ...]

    def __iter__(self):
        yield self.alpha
        yield self.times
        yield self.errors


def _real_coeff(value: complex, *, atol: float = 1e-10) -> float:
    coeff = complex(value)
    if abs(coeff.imag) > atol:
        raise ValueError(f"Expected a real coefficient, got {value}.")
    return float(coeff.real)


def _identity_coeff(operator: QubitOperator) -> float:
    return float(sum(_real_coeff(coeff) for term, coeff in operator.terms.items() if term == ()))


def _particle_number_expectation_from_state(
    state_vector: np.ndarray,
    *,
    num_qubits: int,
) -> tuple[float, float]:
    state = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    if state.size != (1 << int(num_qubits)):
        raise ValueError("Statevector dimension does not match num_qubits.")
    probabilities = np.abs(state) ** 2
    norm = float(np.sum(probabilities))
    if norm <= 0.0:
        raise ValueError("Cannot infer particle number from a zero-norm state.")
    probabilities = probabilities / norm
    occupations = np.fromiter(
        (int(index).bit_count() for index in range(state.size)),
        dtype=float,
        count=state.size,
    )
    expectation = float(np.dot(probabilities, occupations))
    variance = float(np.dot(probabilities, (occupations - expectation) ** 2))
    return expectation, variance


def _infer_particle_number_from_state(
    state_vector: np.ndarray,
    *,
    num_qubits: int,
    atol: float = 1e-6,
) -> int:
    expectation, variance = _particle_number_expectation_from_state(
        state_vector,
        num_qubits=int(num_qubits),
    )
    rounded = int(round(expectation))
    if abs(expectation - rounded) > atol or variance > atol:
        raise ValueError(
            "Cannot infer an integer particle-number sector from the grouped "
            f"reference state: expectation={expectation:.12g}, variance={variance:.3e}."
        )
    return rounded


def _with_grouped_reference_particle_number(
    uwc_config: UWCConfig | Mapping[str, Any] | None,
    data: GroupedHamiltonianData,
) -> UWCConfig | Mapping[str, Any] | None:
    if uwc_config is None:
        return None
    if isinstance(uwc_config, UWCConfig):
        method = str(uwc_config.method)
        enabled = bool(uwc_config.enabled)
        parameters = dict(uwc_config.parameters)
        if (
            enabled
            and method in {"bliss", "orbital_optimization_bliss"}
            and "target_particle_number" not in parameters
        ):
            parameters["target_particle_number"] = _infer_particle_number_from_state(
                data.state_vector,
                num_qubits=data.num_qubits,
            )
            parameters["target_particle_number_source"] = "grouped_reference_state"
            return replace(uwc_config, parameters=parameters)
        return uwc_config

    config_dict = dict(uwc_config)
    method = str(config_dict.get("method", "none"))
    enabled = bool(config_dict.get("enabled", method != "none"))
    parameters = dict(config_dict.get("parameters", {}))
    if (
        enabled
        and method in {"bliss", "orbital_optimization_bliss"}
        and "target_particle_number" not in parameters
    ):
        parameters["target_particle_number"] = _infer_particle_number_from_state(
            data.state_vector,
            num_qubits=data.num_qubits,
        )
        parameters["target_particle_number_source"] = "grouped_reference_state"
        config_dict["parameters"] = parameters
    return config_dict


def _sum_qubit_operators(operators: Sequence[QubitOperator]) -> QubitOperator:
    total = QubitOperator()
    for operator in operators:
        total += operator
    total.compress(abs_tol=_ZERO_TOL)
    return total


def _split_into_single_term_ops(operator: QubitOperator) -> tuple[QubitOperator, ...]:
    return tuple(QubitOperator(term, coeff) for term, coeff in operator.terms.items())


def _normalize_cliques(raw_cliques: Sequence[Any]) -> tuple[tuple[QubitOperator, ...], ...]:
    cliques: list[tuple[QubitOperator, ...]] = []
    for clique in raw_cliques:
        if isinstance(clique, QubitOperator):
            cliques.append(_split_into_single_term_ops(clique))
        else:
            terms: list[QubitOperator] = []
            for operator in clique:
                if isinstance(operator, QubitOperator):
                    terms.extend(_split_into_single_term_ops(operator))
                else:
                    raise TypeError(f"Unsupported clique item type: {type(operator)!r}")
            cliques.append(tuple(terms))
    return tuple(cliques)


def _nonidentity_term_count(operator: QubitOperator) -> int:
    return sum(
        1
        for term, coeff in operator.terms.items()
        if term != () and abs(_real_coeff(coeff)) > _ZERO_TOL
    )


def grouped_nonidentity_term_counts(
    cliques: Sequence[Sequence[QubitOperator]],
) -> tuple[int, ...]:
    """Return non-identity Pauli-term counts for each grouped clique."""

    return tuple(
        _nonidentity_term_count(_sum_qubit_operators(clique))
        for clique in cliques
    )


def _pauli_term_commutes(
    left: tuple[tuple[int, str], ...],
    right: tuple[tuple[int, str], ...],
) -> bool:
    n_anticommute = 0
    left_dict = dict(left)
    right_dict = dict(right)
    for qubit in set(left_dict) | set(right_dict):
        left_axis = left_dict.get(qubit, "I")
        right_axis = right_dict.get(qubit, "I")
        if left_axis == "I" or right_axis == "I":
            continue
        if left_axis != right_axis:
            n_anticommute += 1
    return n_anticommute % 2 == 0


def _single_term_commutes_with_group(
    term_operator: QubitOperator,
    group_operator: QubitOperator,
) -> bool:
    if len(term_operator.terms) != 1:
        raise ValueError("Expected one Pauli term.")
    ((candidate_term, _candidate_coeff),) = term_operator.terms.items()
    if candidate_term == ():
        return True
    for group_term, coeff in group_operator.terms.items():
        if group_term == () or abs(_real_coeff(coeff)) <= _ZERO_TOL:
            continue
        if not _pauli_term_commutes(candidate_term, group_term):
            return False
    return True


def _add_delta_terms_to_cliques(
    base_cliques: Sequence[Sequence[QubitOperator]],
    delta: QubitOperator,
) -> tuple[tuple[QubitOperator, ...], ...]:
    groups = [_sum_qubit_operators(clique) for clique in base_cliques]
    if not groups:
        groups.append(QubitOperator())

    for term_operator in _split_into_single_term_ops(delta):
        if not term_operator.terms:
            continue
        ((pauli_term, coeff),) = term_operator.terms.items()
        if abs(_real_coeff(coeff)) <= _ZERO_TOL:
            continue
        if pauli_term == ():
            groups[0] += term_operator
            groups[0].compress(abs_tol=_ZERO_TOL)
            continue
        for index, group in enumerate(groups):
            if _single_term_commutes_with_group(term_operator, group):
                groups[index] += term_operator
                groups[index].compress(abs_tol=_ZERO_TOL)
                break
        else:
            groups.append(term_operator)

    return tuple(_split_into_single_term_ops(group) for group in groups)


def _grouping_hamiltonian_name(molecule_type: int) -> str:
    molecule = f"H{int(molecule_type)}"
    if int(molecule_type) % 2 == 0:
        return f"{molecule}_sto-3g_singlet_distance_100_charge_0_grouping"
    return f"{molecule}_sto-3g_triplet_1+_distance_100_charge_1_grouping"


def _grouping_coeff_target_names(
    ham_name: str,
    pf_label: str,
    *,
    use_original: bool,
) -> list[str]:
    labels: list[str] = []
    if use_original:
        legacy = _GROUPING_ORIGINAL_PF_LABEL_MAP.get(pf_label)
        if legacy is not None:
            labels.append(legacy)
    labels.append(pf_label)
    labels = list(dict.fromkeys(labels))
    targets: list[str] = []
    for label in labels:
        targets.append(f"{ham_name}_Operator_{label}_ave")
        targets.append(f"{ham_name}_Operator_{label}")
    return targets


def load_grouped_alpha_artifact(
    molecule_type: int,
    pf_label: str,
    *,
    use_original: bool = False,
) -> tuple[float, str]:
    """Load the existing grouped-Hamiltonian Trotter coefficient artifact."""

    ham_name = _grouping_hamiltonian_name(molecule_type)
    last_error: Exception | None = None
    for target in _grouping_coeff_target_names(
        ham_name,
        pf_label,
        use_original=use_original,
    ):
        try:
            data = load_data(target, gr=True, use_original=use_original)
        except Exception as exc:
            last_error = exc
            continue
        coeff = float(data.get("coeff") if isinstance(data, Mapping) else data)
        if not math.isfinite(coeff) or coeff <= 0.0:
            raise ValueError(f"Invalid grouped coefficient artifact: {target}")
        return coeff, target
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"Grouping coefficient not found: {ham_name} / {pf_label}")


def qpe_iteration_factor(
    alpha: float,
    order: int,
    epsilon: float,
    *,
    qpe_beta: float = BETA,
) -> float:
    """Return F(alpha, p, epsilon) for deterministic grouped PF+QPE."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if order <= 0:
        raise ValueError("order must be positive.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if qpe_beta <= 0.0:
        raise ValueError("qpe_beta must be positive.")
    return float(
        qpe_beta
        * ((1.0 + order) / (order * epsilon))
        * ((alpha * (1.0 + order) / epsilon) ** (1.0 / order))
    )


def rotation_synthesis_t_depth(
    alpha: float,
    order: int,
    epsilon: float,
    step_pauli_rotations: int,
    qpe_factor: float,
) -> float:
    """Existing grouping T-depth proxy from cost_extrapolation.py."""

    if step_pauli_rotations <= 0:
        return 0.0
    t_step = (epsilon / alpha * (order + 1.0)) ** (1.0 / order)
    eps_rot = (t_step * 0.01 * epsilon) / (step_pauli_rotations * qpe_factor)
    if eps_rot <= 0.0:
        return math.inf
    return float(3.0 * np.log2(1.0 / eps_rot))


def grouped_step_pauli_rotations(
    cliques: Sequence[Sequence[QubitOperator]],
    pf_label: PFLabel,
) -> int:
    """Count grouped one-step Pauli rotations after applying the PF sequence."""

    label = str(pf_label)
    per_group = [
        _nonidentity_term_count(_sum_qubit_operators(clique))
        for clique in cliques
    ]
    weights = _get_w_list(str(pf_label))
    return int(
        sum(
            per_group[group_idx]
            for group_idx, _weight in iter_pf_steps(len(per_group), weights)
        )
    )


def _qubit_group_z_terms(
    cliques: Sequence[Sequence[QubitOperator]],
) -> list[dict[frozenset[int], complex]]:
    return [
        extract_z_like_terms_from_qubit_group(_sum_qubit_operators(clique))
        for clique in cliques
    ]


def _assign_delta_z_terms(
    z_terms_by_group: list[dict[frozenset[int], complex]],
    base_cliques: Sequence[Sequence[QubitOperator]],
    delta: QubitOperator,
) -> list[dict[frozenset[int], complex]]:
    groups = [_sum_qubit_operators(clique) for clique in base_cliques]
    if not z_terms_by_group:
        z_terms_by_group.append({})
        groups.append(QubitOperator())

    out = [dict(item) for item in z_terms_by_group]
    for term_operator in _split_into_single_term_ops(delta):
        ((pauli_term, coeff),) = term_operator.terms.items()
        if pauli_term == () or abs(_real_coeff(coeff)) <= _ZERO_TOL:
            continue
        support = frozenset(qubit for qubit, _axis in pauli_term)
        target_index: int | None = None
        for index, group in enumerate(groups):
            if _single_term_commutes_with_group(term_operator, group):
                target_index = index
                break
        if target_index is None:
            target_index = len(out)
            out.append({})
            groups.append(term_operator)
        out[target_index][support] = out[target_index].get(support, 0.0) + coeff
    return out


def grouped_step_rz_layers(
    cliques: Sequence[Sequence[QubitOperator]],
    pf_label: PFLabel,
    *,
    molecule_type: int | None = None,
    delta: QubitOperator | None = None,
    use_reference_group_layers: bool = False,
) -> int:
    """
    Count grouped one-step RZ layers for the current grouped Hamiltonian.

    For the reference H-chain grouping, ``use_reference_group_layers=True``
    regenerates the same per-group Z-support rule used to make RZ_LAYER_DIR, then
    injects UWC delta supports before summing over the PF sequence.
    """

    if use_reference_group_layers and molecule_type is not None:
        _n_layers, _layers, z_terms_by_group = estimate_rz_layers_from_grouping(
            int(molecule_type)
        )
        if delta is not None:
            z_terms_by_group = _assign_delta_z_terms(z_terms_by_group, cliques, delta)
    else:
        z_terms_by_group = _qubit_group_z_terms(cliques)

    group_layers = tuple(
        len(_layers_from_z_terms(z_terms))
        for z_terms in z_terms_by_group
    )
    weights = _get_w_list(str(pf_label))
    return int(
        sum(
            group_layers[group_idx]
            for group_idx, _weight in iter_pf_steps(len(group_layers), weights)
        )
    )


@lru_cache(maxsize=None)
def build_grouped_hamiltonian_data(
    molecule_type: int,
    *,
    distance: float = DEFAULT_DISTANCE,
    basis: str = DEFAULT_BASIS,
) -> GroupedHamiltonianData:
    """Build the existing grouped-Hamiltonian baseline data."""

    if basis != DEFAULT_BASIS:
        raise ValueError(
            "The current grouped baseline artifacts use DEFAULT_BASIS only. "
            f"Got basis={basis!r}."
        )

    jw_hamiltonian, _hf_energy, ham_name, num_qubits = jw_hamiltonian_maker(
        int(molecule_type),
        distance,
    )
    identity = _identity_coeff(jw_hamiltonian)
    molecule = f"H{int(molecule_type)}"

    if int(molecule_type) in (2, 3):
        ground_energy, state_vec, _ = ham_ground_energy(
            jw_hamiltonian,
            n_qubits=num_qubits,
            return_max_eig=False,
        )
        grouped_ops, _grouped_name = min_hamiltonian_grouper(jw_hamiltonian, ham_name)
        cliques = tuple((op,) for op in grouped_ops)
        grouping_rule = "min_hamiltonian_grouper"
    else:
        if float(distance) != float(DEFAULT_DISTANCE):
            raise ValueError(
                "make_fci_vector_from_pyscf_solver_grouper currently follows the "
                f"default distance={DEFAULT_DISTANCE}; got distance={distance}."
            )
        raw_cliques, num_qubits, ground_energy, state_vec = (
            make_fci_vector_from_pyscf_solver_grouper(int(molecule_type))
        )
        cliques = _normalize_cliques(raw_cliques)
        grouping_rule = "Almost_optimal_grouper"

    return GroupedHamiltonianData(
        molecule_type=int(molecule_type),
        molecule=molecule,
        basis=basis,
        distance=float(distance),
        ham_name=ham_name,
        num_qubits=int(num_qubits),
        jw_hamiltonian=jw_hamiltonian,
        cliques=cliques,
        ground_energy=float(ground_energy),
        state_vector=np.asarray(state_vec, dtype=np.complex128).reshape(-1),
        grouping_rule=grouping_rule,
        identity_coeff=identity,
    )


def _build_grouped_trotter_circuit(
    cliques: Sequence[Sequence[QubitOperator]],
    *,
    time: float,
    num_qubits: int,
    pf_label: PFLabel,
) -> tuple[QuantumCircuit, int]:
    circuit = QuantumCircuit(int(num_qubits))
    exp_term_count = w_trotter_grouper(
        circuit,
        cliques,
        float(time),
        int(num_qubits),
        str(pf_label),
    )
    return circuit, int(exp_term_count)


def _has_instruction_name(circuit: QuantumCircuit, names: set[str]) -> bool:
    lowered = {name.lower() for name in names}
    return any(
        getattr(instruction.operation, "name", "").lower() in lowered
        for instruction in circuit.data
    )


def _decompose_grouped_circuit_for_gpu(circuit: QuantumCircuit) -> QuantumCircuit:
    prepared = circuit
    for _ in range(8):
        if not _has_instruction_name(
            prepared,
            {"PauliEvolution", "rzz", "xx_plus_yy"},
        ):
            break
        prepared = prepared.decompose(reps=1)
    return prepared


def _assign_gpu_ids_to_times(
    t_values: Sequence[float],
    gpu_ids: Sequence[str],
) -> list[str]:
    visible_gpu_ids = [str(gpu_id) for gpu_id in gpu_ids if str(gpu_id) != ""]
    if not visible_gpu_ids:
        visible_gpu_ids = ["0"]
    return [
        visible_gpu_ids[index % len(visible_gpu_ids)]
        for index, _time_value in enumerate(t_values)
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


def _simulate_grouped_time_task(
    args: tuple[
        float,
        tuple[tuple[QubitOperator, ...], ...],
        int,
        str,
        np.ndarray,
        str,
        int,
        int,
        bool,
    ],
) -> tuple[float, np.ndarray, int, dict[str, Any]]:
    (
        time_value,
        cliques,
        num_qubits,
        pf_label,
        state_flat,
        gpu_id,
        chunk_splits,
        optimization_level,
        debug,
    ) = args
    raw_time = -float(time_value)
    circuit, exp_term_count = _build_grouped_trotter_circuit(
        cliques,
        time=raw_time,
        num_qubits=int(num_qubits),
        pf_label=pf_label,
    )
    input_num_instructions = len(circuit.data)
    circuit = _decompose_grouped_circuit_for_gpu(circuit)
    evolved, profile = simulate_statevector_gpu(
        circuit,
        state_flat,
        gpu_ids=(str(gpu_id),),
        chunk_splits=int(chunk_splits),
        optimization_level=int(optimization_level),
        debug=bool(debug),
        debug_label=f"grouped t={float(time_value)} gpu={gpu_id}",
    )
    profile = dict(profile)
    profile["time"] = float(time_value)
    profile["raw_time"] = raw_time
    profile["assigned_gpu_id"] = str(gpu_id)
    profile["exp_term_count"] = int(exp_term_count)
    profile["grouped_gpu_input_num_instructions"] = int(input_num_instructions)
    profile["grouped_gpu_decomposed_num_instructions"] = int(len(circuit.data))
    return raw_time, np.asarray(evolved, dtype=np.complex128).reshape(-1), exp_term_count, profile


def _run_grouped_trotter_gpu(
    cliques: Sequence[Sequence[QubitOperator]],
    *,
    num_qubits: int,
    state_vector: np.ndarray,
    pf_label: PFLabel,
    t_values: Sequence[float],
    gpu_ids: Sequence[str],
    processes: int | None,
    chunk_splits: int,
    optimization_level: int,
    debug: bool,
) -> tuple[
    tuple[tuple[float, np.ndarray, int], ...],
    tuple[dict[str, Any], ...],
    int,
]:
    normalized_cliques = tuple(tuple(clique) for clique in cliques)
    state_flat = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    assigned_gpu_ids = _assign_gpu_ids_to_times(t_values, gpu_ids)
    num_processes = _resolve_parallel_processes(
        num_times=len(t_values),
        num_gpus=len(set(assigned_gpu_ids)),
        processes=processes,
    )
    tasks = [
        (
            float(time_value),
            normalized_cliques,
            int(num_qubits),
            str(pf_label),
            state_flat,
            str(gpu_id),
            int(chunk_splits),
            int(optimization_level),
            bool(debug),
        )
        for time_value, gpu_id in zip(t_values, assigned_gpu_ids, strict=True)
    ]

    if num_processes <= 1:
        task_results = [_simulate_grouped_time_task(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(num_processes),
            mp_context=context,
        ) as executor:
            task_results = list(executor.map(_simulate_grouped_time_task, tasks))

    final_states: list[tuple[float, np.ndarray, int]] = []
    profiles: list[dict[str, Any]] = []
    for raw_time, evolved, exp_term_count, profile in task_results:
        final_states.append((float(raw_time), evolved, int(exp_term_count)))
        profiles.append(dict(profile))
    return tuple(final_states), tuple(profiles), int(num_processes)


def _run_grouped_trotter_cpu(
    cliques: Sequence[Sequence[QubitOperator]],
    *,
    num_qubits: int,
    state_vector: np.ndarray,
    pf_label: PFLabel,
    t_values: Sequence[float],
) -> tuple[tuple[tuple[float, np.ndarray, int], ...], tuple[dict[str, Any], ...]]:
    state_flat = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    final_states: list[tuple[float, np.ndarray, int]] = []
    profiles: list[dict[str, Any]] = []
    for time_value in t_values:
        raw_time, statevector, exp_term_count = tEvolution_vector_grouper(
            cliques,
            -float(time_value),
            int(num_qubits),
            state_flat,
            str(pf_label),
        )
        final_states.append(
            (
                float(raw_time),
                np.asarray(statevector.data, dtype=np.complex128).reshape(-1),
                int(exp_term_count),
            )
        )
        profiles.append(
            {
                "backend": "qiskit_statevector_cpu",
                "time": float(time_value),
                "raw_time": float(raw_time),
                "exp_term_count": int(exp_term_count),
            }
        )
    return tuple(final_states), tuple(profiles)


def _fit_grouped_alpha_from_states(
    final_states: Sequence[tuple[float, np.ndarray, int]],
    *,
    state_vector: np.ndarray,
    energy_without_identity: float,
    order: int,
) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    state_flat = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    state_col = state_flat.reshape(-1, 1)
    times_out: list[float] = []
    errors: list[float] = []
    for raw_time, evolved_state, _exp_term_count in final_states:
        time = -float(raw_time)
        evolved = np.asarray(evolved_state, dtype=np.complex128).reshape(-1, 1)
        phase_factor = np.exp(1j * float(energy_without_identity) * time)
        delta_state = (evolved - (phase_factor * state_col)) / (1j * time)
        overlap = state_col.conj().T @ delta_state
        overlap = overlap.real / np.cos(float(energy_without_identity) * time)
        errors.append(float(np.abs(overlap.real).item()))
        times_out.append(time)

    alpha = float(
        loglog_average_coeff(
            times_out,
            errors,
            order,
            mask_nonpositive=True,
        )
    )
    return max(0.0, alpha), tuple(times_out), tuple(errors)


def fit_grouped_trotter_alpha(
    cliques: Sequence[Sequence[QubitOperator]],
    *,
    num_qubits: int,
    state_vector: np.ndarray,
    energy_without_identity: float,
    pf_label: PFLabel,
    t_values: Sequence[float],
    evolution_backend: str = "cpu",
    gpu_ids: Sequence[str] = ("0",),
    parallel_processes: int | None = None,
    chunk_splits: int = 1,
    gpu_optimization_level: int = 0,
    gpu_debug: bool = False,
) -> GroupedAlphaFitResult:
    """Fit the fixed-order grouped PF perturbative coefficient."""

    if not t_values:
        raise ValueError("t_values must contain at least one time point.")
    backend = str(evolution_backend).lower()
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("evolution_backend must be 'cpu', 'gpu', or 'auto'.")
    if int(chunk_splits) <= 0:
        raise ValueError("chunk_splits must be positive.")
    gpu_id_tuple = tuple(str(gpu_id) for gpu_id in gpu_ids if str(gpu_id) != "")
    if not gpu_id_tuple:
        gpu_id_tuple = ("0",)
    order = pf_order(str(pf_label))
    state_flat = np.asarray(state_vector, dtype=np.complex128).reshape(-1)
    time_grid = tuple(float(value) for value in t_values)
    profiles: tuple[dict[str, Any], ...]
    num_processes = 1

    if backend in {"gpu", "auto"}:
        try:
            final_states, profiles, num_processes = _run_grouped_trotter_gpu(
                cliques,
                num_qubits=int(num_qubits),
                state_vector=state_flat,
                pf_label=str(pf_label),
                t_values=time_grid,
                gpu_ids=gpu_id_tuple,
                processes=parallel_processes,
                chunk_splits=int(chunk_splits),
                optimization_level=int(gpu_optimization_level),
                debug=bool(gpu_debug),
            )
            used_backend = "gpu"
        except (ImportError, RuntimeError) as exc:
            if backend == "gpu":
                raise
            final_states, cpu_profiles = _run_grouped_trotter_cpu(
                cliques,
                num_qubits=int(num_qubits),
                state_vector=state_flat,
                pf_label=str(pf_label),
                t_values=time_grid,
            )
            profiles = (
                {
                    "backend": "auto_fallback",
                    "requested_backend": backend,
                    "fallback_backend": "cpu",
                    "fallback_reason": str(exc),
                },
                *cpu_profiles,
            )
            used_backend = "cpu"
            num_processes = 1
    else:
        final_states, profiles = _run_grouped_trotter_cpu(
            cliques,
            num_qubits=int(num_qubits),
            state_vector=state_flat,
            pf_label=str(pf_label),
            t_values=time_grid,
        )
        used_backend = "cpu"

    alpha, times_out, errors = _fit_grouped_alpha_from_states(
        final_states,
        state_vector=state_flat,
        energy_without_identity=float(energy_without_identity),
        order=int(order),
    )
    return GroupedAlphaFitResult(
        alpha=float(alpha),
        times=times_out,
        errors=errors,
        backend=used_backend,
        requested_backend=backend,
        gpu_ids=gpu_id_tuple,
        parallel_processes=int(num_processes),
        chunk_splits=int(chunk_splits),
        optimization_level=int(gpu_optimization_level),
        profiles=tuple(dict(profile) for profile in profiles),
    )


def _spin_charge_sector(molecule_type: int) -> dict[str, Any]:
    from .chemistry_hamiltonian import geo

    _geometry, multiplicity, charge = geo(int(molecule_type))
    return {
        "multiplicity": int(multiplicity),
        "charge": int(charge),
    }


def _target_hamiltonian_state(
    hamiltonian: QubitOperator,
    *,
    num_qubits: int,
    fallback_state: np.ndarray,
    fallback_energy: float,
    use_fallback: bool,
) -> tuple[np.ndarray, float]:
    if use_fallback:
        return np.asarray(fallback_state, dtype=np.complex128).reshape(-1), float(fallback_energy)
    energy, state_vec, _ = ham_ground_energy(
        hamiltonian,
        n_qubits=int(num_qubits),
        return_max_eig=False,
    )
    return np.asarray(state_vec, dtype=np.complex128).reshape(-1), float(energy)


def _row(
    *,
    data: GroupedHamiltonianData,
    method: str,
    uwc_method: str,
    uwc_objective: str,
    pf_label: str,
    grouping_rule: str,
    alpha: float,
    step_pauli_rotations: int,
    step_rz_layers: int,
    target_error: float,
    qpe_beta: float,
    cost_metric: str,
    baseline: GroupedPFQPERow | None,
    time_grid: Sequence[float],
    alpha_source: str,
    uwc_hamiltonian_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GroupedPFQPERow:
    order = pf_order(pf_label)
    qpe_factor = qpe_iteration_factor(
        alpha,
        order,
        target_error,
        qpe_beta=qpe_beta,
    )
    total_pauli_rotations = float(step_pauli_rotations) * qpe_factor
    total_rz_layers = float(step_rz_layers) * qpe_factor
    step_t_depth = rotation_synthesis_t_depth(
        alpha,
        order,
        target_error,
        int(step_pauli_rotations),
        qpe_factor,
    )
    total_t_depth = float(step_rz_layers) * step_t_depth * qpe_factor
    step_cost = step_rz_layers if cost_metric == "rz_layers" else step_pauli_rotations
    g_total = float(step_cost) * qpe_factor

    if baseline is None:
        cost_ratio = 1.0
        alpha_ratio = 1.0
        step_ratio = 1.0
    else:
        baseline_step = (
            baseline.step_rz_layers
            if cost_metric == "rz_layers"
            else baseline.step_pauli_rotations
        )
        cost_ratio = g_total / baseline.g_total
        alpha_ratio = alpha / baseline.alpha
        step_ratio = float(step_cost) / float(baseline_step)

    return GroupedPFQPERow(
        molecule=data.molecule,
        molecule_type=data.molecule_type,
        method=method,
        uwc_method=uwc_method,
        uwc_objective=uwc_objective,
        pf_label=pf_label,
        order=order,
        grouping_rule=grouping_rule,
        num_groups=len(data.cliques),
        num_pauli_terms=qubit_num_terms(data.jw_hamiltonian),
        alpha=float(alpha),
        qpe_iteration_factor=float(qpe_factor),
        step_pauli_rotations=int(step_pauli_rotations),
        total_pauli_rotations=float(total_pauli_rotations),
        step_rz_layers=int(step_rz_layers),
        total_rz_layers=float(total_rz_layers),
        step_t_depth=float(step_t_depth),
        total_t_depth=float(total_t_depth),
        cost_metric=cost_metric,
        g_total=float(g_total),
        cost_ratio_vs_grouped_baseline=float(cost_ratio),
        alpha_ratio_vs_grouped_baseline=float(alpha_ratio),
        step_cost_ratio_vs_grouped_baseline=float(step_ratio),
        qpe_beta=float(qpe_beta),
        target_error=float(target_error),
        fitting_rule=f"fixed_order_average_p{order}",
        time_grid=tuple(float(value) for value in time_grid),
        alpha_source=alpha_source,
        uwc_hamiltonian_hash=uwc_hamiltonian_hash,
        metadata=metadata,
    )


def compare_grouped_uwc_pf_qpe(
    molecule_type: int,
    *,
    pf_label: PFLabel,
    uwc_config: UWCConfig | Mapping[str, Any] | None = None,
    target_error: float,
    distance: float = DEFAULT_DISTANCE,
    basis: str = DEFAULT_BASIS,
    qpe_beta: float = BETA,
    cost_metric: str = "rz_layers",
    t_values: Sequence[float] | None = None,
    baseline_alpha_source: str = "artifact",
    use_original_grouped_artifact: bool = False,
    reuse_original_ground_state_for_uwc: bool | None = None,
    use_reference_rz_layers: bool = True,
    alpha_backend: str = "cpu",
    alpha_gpu_ids: Sequence[str] = ("0",),
    alpha_parallel_processes: int | None = None,
    alpha_chunk_splits: int = 1,
    alpha_gpu_optimization_level: int = 0,
    alpha_gpu_debug: bool = False,
) -> GroupedUWCComparisonResult:
    """
    Compare existing grouped PF+QPE against UWC-applied grouped PF+QPE.

    The grouped baseline uses the existing grouped coefficient artifact and the
    DECOMPO_NUM/PF_RZ_LAYER tables by default. UWC never reuses those step costs:
    the UWC Hamiltonian is regrouped by adding its Pauli-term delta to compatible
    grouped cliques, then Pauli rotations, RZ layers, and alpha are recomputed.
    """

    if cost_metric not in {"rz_layers", "pauli_rotations"}:
        raise ValueError("cost_metric must be 'rz_layers' or 'pauli_rotations'.")
    if target_error <= 0.0:
        raise ValueError("target_error must be positive.")
    label = str(pf_label)
    if label not in P_DIR:
        raise KeyError(f"Unsupported PF label: {pf_label}")

    data = build_grouped_hamiltonian_data(
        int(molecule_type),
        distance=distance,
        basis=basis,
    )
    if t_values is None:
        t_values = default_perturbation_t_values(int(molecule_type), label)
    time_grid = tuple(float(value) for value in t_values)

    molecule_key = data.molecule
    baseline_fit: GroupedAlphaFitResult | None = None
    if baseline_alpha_source == "artifact":
        baseline_alpha, alpha_artifact = load_grouped_alpha_artifact(
            int(molecule_type),
            label,
            use_original=use_original_grouped_artifact,
        )
        baseline_alpha_source_label = f"grouped_artifact:{alpha_artifact}"
    elif baseline_alpha_source == "fit":
        baseline_fit = fit_grouped_trotter_alpha(
            data.cliques,
            num_qubits=data.num_qubits,
            state_vector=data.state_vector,
            energy_without_identity=data.ground_energy - data.identity_coeff,
            pf_label=label,
            t_values=time_grid,
            evolution_backend=alpha_backend,
            gpu_ids=alpha_gpu_ids,
            parallel_processes=alpha_parallel_processes,
            chunk_splits=alpha_chunk_splits,
            gpu_optimization_level=alpha_gpu_optimization_level,
            gpu_debug=alpha_gpu_debug,
        )
        baseline_alpha = baseline_fit.alpha
        baseline_alpha_source_label = f"computed_grouped_fit:{baseline_fit.backend}"
    else:
        raise ValueError("baseline_alpha_source must be 'artifact' or 'fit'.")

    baseline_step_pauli = int(DECOMPO_NUM[molecule_key][label])
    baseline_step_rz = int(PF_RZ_LAYER[molecule_key][label])
    baseline_group_sizes = grouped_nonidentity_term_counts(data.cliques)
    baseline_row = _row(
        data=data,
        method="grouped_baseline",
        uwc_method="none",
        uwc_objective="none",
        pf_label=label,
        grouping_rule=data.grouping_rule,
        alpha=baseline_alpha,
        step_pauli_rotations=baseline_step_pauli,
        step_rz_layers=baseline_step_rz,
        target_error=target_error,
        qpe_beta=qpe_beta,
        cost_metric=cost_metric,
        baseline=None,
        time_grid=time_grid,
        alpha_source=baseline_alpha_source_label,
        metadata={
            "baseline_step_pauli_source": "DECOMPO_NUM",
            "baseline_step_rz_source": "PF_RZ_LAYER",
            "group_sizes": baseline_group_sizes,
            "alpha_fit_backend": None if baseline_fit is None else baseline_fit.backend,
            "alpha_requested_backend": (
                None if baseline_fit is None else baseline_fit.requested_backend
            ),
            "alpha_gpu_ids": None if baseline_fit is None else baseline_fit.gpu_ids,
            "alpha_parallel_processes": (
                None if baseline_fit is None else baseline_fit.parallel_processes
            ),
            "fit_times": None if baseline_fit is None else baseline_fit.times,
            "fit_errors": None if baseline_fit is None else baseline_fit.errors,
            "gpu_profiles": None if baseline_fit is None else baseline_fit.profiles,
        },
    )

    uwc_config_for_grouped_reference = _with_grouped_reference_particle_number(
        uwc_config,
        data,
    )
    preprocessing = preprocess_qubit_hamiltonian(
        data.jw_hamiltonian,
        uwc_config_for_grouped_reference,
        n_qubits=data.num_qubits,
        target_ld=None,
    )
    target_hamiltonian = preprocessing.hamiltonian
    delta = target_hamiltonian + (-1.0 * data.jw_hamiltonian)
    delta.compress(abs_tol=_ZERO_TOL)
    uwc_cliques = _add_delta_terms_to_cliques(data.cliques, delta)
    uwc_hamiltonian_from_groups = _sum_qubit_operators(
        [_sum_qubit_operators(clique) for clique in uwc_cliques]
    )
    uwc_identity = _identity_coeff(target_hamiltonian)

    if reuse_original_ground_state_for_uwc is None:
        reuse_original_ground_state_for_uwc = preprocessing.config.method == "bliss"
    uwc_state, uwc_energy = _target_hamiltonian_state(
        target_hamiltonian,
        num_qubits=data.num_qubits,
        fallback_state=data.state_vector,
        fallback_energy=data.ground_energy,
        use_fallback=bool(reuse_original_ground_state_for_uwc),
    )
    uwc_fit = fit_grouped_trotter_alpha(
        uwc_cliques,
        num_qubits=data.num_qubits,
        state_vector=uwc_state,
        energy_without_identity=uwc_energy - uwc_identity,
        pf_label=label,
        t_values=time_grid,
        evolution_backend=alpha_backend,
        gpu_ids=alpha_gpu_ids,
        parallel_processes=alpha_parallel_processes,
        chunk_splits=alpha_chunk_splits,
        gpu_optimization_level=alpha_gpu_optimization_level,
        gpu_debug=alpha_gpu_debug,
    )
    uwc_alpha = uwc_fit.alpha
    uwc_step_pauli = grouped_step_pauli_rotations(uwc_cliques, label)
    uwc_group_sizes = grouped_nonidentity_term_counts(uwc_cliques)
    use_reference_group_layers = bool(use_reference_rz_layers)
    uwc_step_rz = grouped_step_rz_layers(
        uwc_cliques,
        label,
        molecule_type=data.molecule_type,
        delta=delta,
        use_reference_group_layers=use_reference_group_layers,
    )

    uwc_data = GroupedHamiltonianData(
        molecule_type=data.molecule_type,
        molecule=data.molecule,
        basis=data.basis,
        distance=data.distance,
        ham_name=data.ham_name,
        num_qubits=data.num_qubits,
        jw_hamiltonian=uwc_hamiltonian_from_groups,
        cliques=uwc_cliques,
        ground_energy=uwc_energy,
        state_vector=uwc_state,
        grouping_rule=f"{data.grouping_rule}+uwc_delta_regroup",
        identity_coeff=uwc_identity,
    )
    uwc_row = _row(
        data=uwc_data,
        method="uwc_grouped",
        uwc_method=preprocessing.metadata["uwc_method"],
        uwc_objective=preprocessing.metadata["uwc_objective"],
        pf_label=label,
        grouping_rule=uwc_data.grouping_rule,
        alpha=uwc_alpha,
        step_pauli_rotations=uwc_step_pauli,
        step_rz_layers=uwc_step_rz,
        target_error=target_error,
        qpe_beta=qpe_beta,
        cost_metric=cost_metric,
        baseline=baseline_row,
        time_grid=time_grid,
        alpha_source=f"computed_grouped_fit:{uwc_fit.backend}",
        uwc_hamiltonian_hash=preprocessing.metadata["uwc_hamiltonian_hash"],
        metadata={
            "uwc_metadata": preprocessing.metadata,
            "fit_times": uwc_fit.times,
            "fit_errors": uwc_fit.errors,
            "baseline_num_groups": len(data.cliques),
            "uwc_num_groups": len(uwc_cliques),
            "baseline_num_pauli_terms": qubit_num_terms(data.jw_hamiltonian),
            "uwc_num_pauli_terms": qubit_num_terms(uwc_hamiltonian_from_groups),
            "baseline_group_sizes": baseline_group_sizes,
            "uwc_group_sizes": uwc_group_sizes,
            "baseline_step_pauli_rotations": baseline_row.step_pauli_rotations,
            "uwc_step_pauli_rotations": uwc_step_pauli,
            "baseline_step_rz_layers": baseline_row.step_rz_layers,
            "uwc_step_rz_layers": uwc_step_rz,
            "alpha_fit_backend": uwc_fit.backend,
            "alpha_requested_backend": uwc_fit.requested_backend,
            "alpha_gpu_ids": uwc_fit.gpu_ids,
            "alpha_parallel_processes": uwc_fit.parallel_processes,
            "alpha_chunk_splits": uwc_fit.chunk_splits,
            "alpha_gpu_optimization_level": uwc_fit.optimization_level,
            "gpu_profiles": uwc_fit.profiles,
            "uwc_step_pauli_source": "regrouped_uwc_hamiltonian",
            "uwc_step_rz_source": (
                "reference_group_z_terms_plus_uwc_delta"
                if use_reference_group_layers
                else "qubit_support_greedy_layers"
            ),
        },
    )

    return GroupedUWCComparisonResult(
        molecule=data.molecule,
        molecule_type=data.molecule_type,
        basis=data.basis,
        distance=data.distance,
        spin_charge_sector=_spin_charge_sector(int(molecule_type)),
        pf_label=label,
        order=pf_order(label),
        grouping_rule=data.grouping_rule,
        cost_metric=cost_metric,
        rows=(baseline_row, uwc_row),
    )


def save_grouped_uwc_comparison(
    result: GroupedUWCComparisonResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path
