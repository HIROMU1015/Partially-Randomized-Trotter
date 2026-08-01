"""One explicit partial-S2 step combining deterministic DF blocks and RTE."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate

from .df_hamiltonian import DFHamiltonian
from .df_partial_randomized_pf import (
    DFFragmentPartition,
    df_hamiltonian_hash,
    df_hamiltonian_to_model,
    rank_df_fragments,
)
from .df_rte_circuit import (
    DFRTEEventPreparation,
    DFRTEEventSequenceCircuitRequest,
)
from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from .df_rte_tail import (
    DFBasisRegistry,
    DFTailExtraction,
    IdentityPolicy,
    extract_df_tail_from_hamiltonian,
    prepare_df_rte_event_inputs,
)
from .df_trotter.ops import (
    append_diagonal_primitives,
    build_df_blocks,
    build_one_body_gaussian_block,
    df_squared_diagonal_primitives,
    one_body_diagonal_primitives,
)
from .rte import (
    BasisChangeOperation,
    RTEConfig,
    RTEFiniteDistribution,
    finite_rte_attenuation,
    make_rte_config,
    require_integer_count,
)


BasisConstructionPolicy: TypeAlias = Literal["fermionic_gaussian_jw"]
DeterministicBlockKind: TypeAlias = Literal["one_body", "df_fragment"]


def _real_tuple(values: np.ndarray, *, name: str) -> tuple[float, ...]:
    array = np.asarray(values)
    if np.iscomplexobj(array) and np.max(np.abs(array.imag), initial=0.0) > 1e-12:
        raise ValueError(f"{name} has a non-negligible imaginary part.")
    return tuple(float(value) for value in np.real(array))


@dataclass(frozen=True)
class DFDeterministicOneBodySpec:
    """Executable one-body Gaussian block in deterministic application order."""

    block_id: str
    block_kind: Literal["one_body"]
    original_fragment_index: None
    rank: None
    basis_id: str
    basis_hash: str
    basis_change_operations: tuple[BasisChangeOperation, ...]
    diagonal_eigenvalues: tuple[float, ...]
    num_system_qubits: int
    order_index: int
    runtime_basis_operations: tuple[tuple[object, tuple[int, ...]], ...] = field(
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class DFDeterministicFragmentSpec:
    """Executable squared DF fragment retaining original index and rank."""

    block_id: str
    block_kind: Literal["df_fragment"]
    original_fragment_index: int
    rank: int
    basis_id: str
    basis_hash: str
    basis_change_operations: tuple[BasisChangeOperation, ...]
    diagonal_eta: tuple[float, ...]
    lam: float
    num_system_qubits: int
    order_index: int
    runtime_basis_operations: tuple[tuple[object, tuple[int, ...]], ...] = field(
        repr=False,
        compare=False,
    )


DFDeterministicBlockSpec: TypeAlias = (
    DFDeterministicOneBodySpec | DFDeterministicFragmentSpec
)


@dataclass(frozen=True)
class DFPartialS2Preparation:
    """Dense-free, hash-bound inputs for one partial-S2 circuit step."""

    hamiltonian_hash: str
    partition_hash: str
    preparation_hash: str
    ld: int
    num_system_qubits: int
    deterministic_blocks: tuple[DFDeterministicBlockSpec, ...]
    deterministic_block_order: tuple[str, ...]
    deterministic_fragment_indices: tuple[int, ...]
    randomized_block_indices: tuple[int, ...]
    constant_coefficient: float
    tail_extraction: DFTailExtraction
    rte_preparation: DFRTEEventPreparation
    ranking_proxy_lambda_r: float
    exact_rte_lambda_r: float
    identity_policy: IdentityPolicy
    coefficient_atol: float
    threshold_dropped_component_count: int
    threshold_dropped_coefficient_l1: float
    threshold_operator_error_bound: float
    extracted_identity_coefficient: float
    basis_construction_policy: BasisConstructionPolicy
    diagonal_sort: str
    product_formula: Literal["2nd"] = "2nd"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ld", require_integer_count(self.ld, name="ld"))
        object.__setattr__(
            self,
            "num_system_qubits",
            require_integer_count(
                self.num_system_qubits,
                name="num_system_qubits",
                minimum=1,
            ),
        )
        if not self.hamiltonian_hash or not self.partition_hash or not self.preparation_hash:
            raise ValueError("Partial-S2 preparation hashes must not be empty.")
        if self.product_formula != "2nd":
            raise ValueError("DF partial-S2 preparation supports only second order.")
        if self.deterministic_block_order != tuple(
            block.block_id for block in self.deterministic_blocks
        ):
            raise ValueError("Deterministic block order does not match block specs.")
        if tuple(block.order_index for block in self.deterministic_blocks) != tuple(
            range(len(self.deterministic_blocks))
        ):
            raise ValueError("Deterministic block order indices must be contiguous.")
        fragment_indices = tuple(
            block.original_fragment_index
            for block in self.deterministic_blocks
            if isinstance(block, DFDeterministicFragmentSpec)
        )
        if fragment_indices != self.deterministic_fragment_indices:
            raise ValueError("Deterministic fragment indices do not match block specs.")
        if set(fragment_indices).intersection(self.randomized_block_indices):
            raise ValueError("Deterministic and randomized fragments overlap.")
        if not math.isclose(
            self.exact_rte_lambda_r,
            self.tail_extraction.rte_lambda_r,
            abs_tol=1e-14,
        ):
            raise ValueError("exact_rte_lambda_r does not match tail extraction.")
        if self.rte_preparation.symbolic_tail.tail_hash != self.tail_extraction.tail_hash:
            raise ValueError("RTE preparation does not match tail extraction.")
        if self.extracted_identity_coefficient != (
            self.tail_extraction.deterministic_identity_coefficient
        ):
            raise ValueError("Extracted identity coefficient is inconsistent.")
        if not math.isfinite(self.constant_coefficient):
            raise ValueError("constant_coefficient must be finite.")
        if not math.isfinite(self.coefficient_atol) or self.coefficient_atol < 0.0:
            raise ValueError("coefficient_atol must be finite and non-negative.")
        for name in (
            "threshold_dropped_component_count",
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name),
            )

    @property
    def is_deterministic_only(self) -> bool:
        return self.exact_rte_lambda_r == 0.0


def _partition_hash(
    hamiltonian_hash: str,
    partition: DFFragmentPartition,
) -> str:
    payload = {
        "hamiltonian_hash": hamiltonian_hash,
        "ld": partition.ld,
        "deterministic_fragments": [
            asdict(fragment) for fragment in partition.deterministic_fragments
        ],
        "randomized_fragments": [
            asdict(fragment) for fragment in partition.randomized_fragments
        ],
        "ranking_proxy_lambda_r": partition.ranking_proxy_lambda_r,
        "weight_rule": partition.weight_rule,
        "partition_hash_policy": "ranked_df_prefix_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_partition(
    hamiltonian: DFHamiltonian,
    partition: DFFragmentPartition,
) -> None:
    ld = require_integer_count(partition.ld, name="partition.ld")
    if ld > hamiltonian.n_blocks:
        raise ValueError("partition.ld exceeds the number of DF fragments.")
    expected = rank_df_fragments(
        hamiltonian,
        weight_rule=partition.weight_rule,
    )
    if partition.deterministic_fragments != expected[:ld]:
        raise ValueError("Deterministic fragments are not the ranked L_D prefix.")
    if partition.randomized_fragments != expected[ld:]:
        raise ValueError("Randomized fragments are not the ranked suffix.")
    deterministic = partition.deterministic_block_indices
    randomized = partition.randomized_block_indices
    if deterministic != tuple(item.original_index for item in expected[:ld]):
        raise ValueError("Deterministic block order differs from ranked prefix order.")
    if randomized != tuple(item.original_index for item in expected[ld:]):
        raise ValueError("Randomized block order differs from ranked suffix order.")
    if set(deterministic).intersection(randomized):
        raise ValueError("Deterministic and randomized fragments overlap.")
    if set((*deterministic, *randomized)) != set(range(hamiltonian.n_blocks)):
        raise ValueError("DF partition does not cover every two-body fragment.")


def _block_hash_payload(block: DFDeterministicBlockSpec) -> dict[str, object]:
    common: dict[str, object] = {
        "block_id": block.block_id,
        "block_kind": block.block_kind,
        "original_fragment_index": block.original_fragment_index,
        "rank": block.rank,
        "basis_id": block.basis_id,
        "basis_hash": block.basis_hash,
        "basis_change_operations": [
            asdict(operation) for operation in block.basis_change_operations
        ],
        "num_system_qubits": block.num_system_qubits,
        "order_index": block.order_index,
    }
    if isinstance(block, DFDeterministicOneBodySpec):
        common["diagonal_eigenvalues"] = block.diagonal_eigenvalues
    else:
        common["diagonal_eta"] = block.diagonal_eta
        common["lam"] = block.lam
    return common


def prepare_df_partial_s2(
    hamiltonian: DFHamiltonian,
    partition: DFFragmentPartition,
    *,
    identity_policy: IdentityPolicy = "extract_identity_phase",
    coefficient_atol: float = 0.0,
    diagonal_sort: str = "descending_abs",
    basis_construction_policy: BasisConstructionPolicy = "fermionic_gaussian_jw",
) -> DFPartialS2Preparation:
    """Prepare ranked deterministic blocks and the exact symbolic RTE tail."""
    if basis_construction_policy != "fermionic_gaussian_jw":
        raise ValueError("Unsupported basis construction policy.")
    if not diagonal_sort:
        raise ValueError("diagonal_sort must not be empty.")
    if not math.isfinite(coefficient_atol) or coefficient_atol < 0.0:
        raise ValueError("coefficient_atol must be finite and non-negative.")
    _validate_partition(hamiltonian, partition)
    hamiltonian_fingerprint = df_hamiltonian_hash(
        hamiltonian,
        weight_rule=partition.weight_rule,
    )
    partition_fingerprint = _partition_hash(hamiltonian_fingerprint, partition)
    model = df_hamiltonian_to_model(hamiltonian)
    registry = DFBasisRegistry()
    deterministic_blocks: list[DFDeterministicBlockSpec] = []

    if np.linalg.norm(model.one_body_correction) > 1e-14:
        one_body = build_one_body_gaussian_block(
            model.one_body_correction,
            sort=diagonal_sort,
        )
        definition = registry.register(
            one_body.U_ops,
            num_system_qubits=model.N,
        )
        deterministic_blocks.append(
            DFDeterministicOneBodySpec(
                block_id="df-one-body-correction",
                block_kind="one_body",
                original_fragment_index=None,
                rank=None,
                basis_id=definition.basis_id,
                basis_hash=definition.basis_hash,
                basis_change_operations=definition.metadata.operations,
                diagonal_eigenvalues=_real_tuple(one_body.eps, name="one_body eps"),
                num_system_qubits=model.N,
                order_index=0,
                runtime_basis_operations=tuple(one_body.U_ops),
            )
        )

    all_fragment_blocks = build_df_blocks(model, sort=diagonal_sort)
    rank_by_index = {
        fragment.original_index: fragment.rank
        for fragment in partition.deterministic_fragments
    }
    for original_index in partition.deterministic_block_indices:
        block = all_fragment_blocks[original_index]
        definition = registry.register(
            block.U_ops,
            num_system_qubits=model.N,
        )
        deterministic_blocks.append(
            DFDeterministicFragmentSpec(
                block_id=f"df-fragment-{original_index}",
                block_kind="df_fragment",
                original_fragment_index=original_index,
                rank=rank_by_index[original_index],
                basis_id=definition.basis_id,
                basis_hash=definition.basis_hash,
                basis_change_operations=definition.metadata.operations,
                diagonal_eta=_real_tuple(block.eta, name="fragment eta"),
                lam=float(np.real_if_close(block.lam)),
                num_system_qubits=model.N,
                order_index=len(deterministic_blocks),
                runtime_basis_operations=tuple(block.U_ops),
            )
        )

    tail_extraction = extract_df_tail_from_hamiltonian(
        f"partial-s2-tail-{hamiltonian_fingerprint[:16]}-ld-{partition.ld}",
        hamiltonian,
        partition.randomized_block_indices,
        identity_policy=identity_policy,
        coefficient_atol=coefficient_atol,
        diagonal_sort=diagonal_sort,
        ranking_weight_rule=partition.weight_rule,
    )
    if not math.isclose(
        tail_extraction.ranking_proxy_lambda_r or 0.0,
        partition.ranking_proxy_lambda_r,
        rel_tol=1e-13,
        abs_tol=1e-14,
    ):
        raise ValueError("Tail ranking proxy does not match the DF partition.")
    rte_preparation = prepare_df_rte_event_inputs(tail_extraction)
    metadata = tail_extraction.extraction_metadata
    deterministic_tuple = tuple(deterministic_blocks)
    preparation_payload = {
        "hamiltonian_hash": hamiltonian_fingerprint,
        "partition_hash": partition_fingerprint,
        "deterministic_blocks": [
            _block_hash_payload(block) for block in deterministic_tuple
        ],
        "tail_hash": tail_extraction.tail_hash,
        "identity_policy": identity_policy,
        "coefficient_atol": float(coefficient_atol),
        "basis_construction_policy": basis_construction_policy,
        "diagonal_sort": diagonal_sort,
        "product_formula": "2nd",
        "preparation_hash_policy": "df_partial_s2_preparation_v1",
    }
    encoded = json.dumps(
        preparation_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DFPartialS2Preparation(
        hamiltonian_hash=hamiltonian_fingerprint,
        partition_hash=partition_fingerprint,
        preparation_hash=hashlib.sha256(encoded).hexdigest(),
        ld=partition.ld,
        num_system_qubits=model.N,
        deterministic_blocks=deterministic_tuple,
        deterministic_block_order=tuple(
            block.block_id for block in deterministic_tuple
        ),
        deterministic_fragment_indices=partition.deterministic_block_indices,
        randomized_block_indices=partition.randomized_block_indices,
        constant_coefficient=float(hamiltonian.constant),
        tail_extraction=tail_extraction,
        rte_preparation=rte_preparation,
        ranking_proxy_lambda_r=partition.ranking_proxy_lambda_r,
        exact_rte_lambda_r=tail_extraction.rte_lambda_r,
        identity_policy=identity_policy,
        coefficient_atol=float(coefficient_atol),
        threshold_dropped_component_count=(
            metadata.threshold_dropped_component_count
        ),
        threshold_dropped_coefficient_l1=(
            metadata.threshold_dropped_coefficient_l1
        ),
        threshold_operator_error_bound=metadata.threshold_operator_error_bound,
        extracted_identity_coefficient=(
            tail_extraction.deterministic_identity_coefficient
        ),
        basis_construction_policy=basis_construction_policy,
        diagonal_sort=diagonal_sort,
    )


@dataclass(frozen=True)
class DFPartialS2StepRequest:
    preparation: DFPartialS2Preparation
    step_time: float
    rte_config: RTEConfig | None
    rte_distribution: RTEFiniteDistribution | None
    rte_occurrence: DFRTEEventSequenceCircuitRequest | None
    controlled: bool = False
    ancilla_qubit: int | None = None
    seed: int | None = None
    pf_label: Literal["2nd"] = "2nd"

    def __post_init__(self) -> None:
        if self.pf_label != "2nd":
            raise ValueError("DF partial-S2 supports only pf_label='2nd'.")
        if not math.isfinite(self.step_time):
            raise ValueError("step_time must be finite.")
        if self.controlled and self.ancilla_qubit is None:
            raise ValueError("A controlled partial-S2 step requires ancilla_qubit.")
        if self.ancilla_qubit is not None:
            ancilla = require_integer_count(self.ancilla_qubit, name="ancilla_qubit")
            object.__setattr__(self, "ancilla_qubit", ancilla)
            if self.controlled and ancilla < self.preparation.num_system_qubits:
                raise ValueError("Ancilla qubit must not overlap the system register.")
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                require_integer_count(self.seed, name="seed"),
            )
        if self.preparation.is_deterministic_only:
            if any(
                item is not None
                for item in (
                    self.rte_config,
                    self.rte_distribution,
                    self.rte_occurrence,
                )
            ):
                raise ValueError("A deterministic-only step must not contain RTE data.")
            return
        if (
            self.rte_config is None
            or self.rte_distribution is None
            or self.rte_occurrence is None
            or self.seed is None
        ):
            raise ValueError("A randomized partial-S2 step requires complete RTE data.")
        config = self.rte_config
        distribution = self.rte_distribution
        occurrence = self.rte_occurrence
        tail = self.preparation.rte_preparation.symbolic_tail
        if not math.isclose(config.evolution_time, self.step_time, abs_tol=1e-14):
            raise ValueError("RTE evolution_time must equal partial-S2 step_time.")
        if config.tail_id != tail.tail_id or config.tail_hash != tail.tail_hash:
            raise ValueError("RTE config tail identity does not match preparation.")
        if not math.isclose(config.lambda_r, self.preparation.exact_rte_lambda_r):
            raise ValueError("RTE config must use exact_rte_lambda_r.")
        if config.finite_taylor_order != distribution.finite_taylor_order:
            raise ValueError("RTE config and distribution Taylor cutoffs differ.")
        if not math.isclose(
            config.dimensionless_step_time,
            distribution.dimensionless_step_time,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("RTE config and distribution step times differ.")
        if not math.isclose(
            config.distribution_normalization,
            distribution.exact_finite_distribution,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise ValueError("RTE distribution normalization mismatch.")
        if len(occurrence.events) != config.rte_steps:
            raise ValueError("RTE occurrence event count must equal config.rte_steps.")
        if occurrence.occurrence_rte_steps != config.rte_steps:
            raise ValueError("RTE occurrence metadata must equal config.rte_steps.")
        if occurrence.tail_id != tail.tail_id or occurrence.tail_hash != tail.tail_hash:
            raise ValueError("RTE occurrence tail identity does not match preparation.")
        if occurrence.component_specs != self.preparation.rte_preparation.component_specs:
            raise ValueError("RTE occurrence component specs do not match preparation.")
        components = {
            component.component_id: component for component in tail.components
        }
        for event in occurrence.events:
            if event.taylor_order not in distribution.orders:
                raise ValueError("RTE occurrence contains an unsupported Taylor order.")
            if not math.isclose(
                event.event_normalization,
                distribution.exact_finite_distribution,
                rel_tol=1e-14,
                abs_tol=1e-15,
            ):
                raise ValueError("RTE event normalization does not match distribution.")
            order_index = distribution.orders.index(event.taylor_order)
            expected_angle = math.atan(
                config.dimensionless_step_time / (event.taylor_order + 1)
            )
            if not math.isclose(event.rotation_angle, expected_angle, abs_tol=1e-14):
                raise ValueError("RTE event rotation angle does not match distribution.")
            expected_phase = complex((-1) ** (event.taylor_order // 2))
            if event.phase != expected_phase:
                raise ValueError("RTE event Taylor phase does not match its order.")
            component_probability = math.prod(
                components[component_id].probability
                for component_id in event.selected_component_ids
            )
            expected_probability = (
                distribution.unnormalized_order_weights[order_index]
                * component_probability
                / distribution.exact_finite_distribution
            )
            if not math.isclose(
                event.event_probability,
                expected_probability,
                rel_tol=1e-13,
                abs_tol=1e-15,
            ):
                raise ValueError("RTE event probability does not match distribution.")
        if occurrence.controlled != self.controlled:
            raise ValueError("Partial-S2 and RTE controlled conditions must match.")
        if occurrence.ancilla_qubit != self.ancilla_qubit:
            raise ValueError("Partial-S2 and RTE ancilla positions must match.")


def make_df_partial_s2_step_request(
    preparation: DFPartialS2Preparation,
    *,
    step_time: float,
    rte_steps: int = 1,
    truncation_tolerance: float = 1e-10,
    finite_taylor_order: int | None = None,
    seed: int = 0,
    controlled: bool = False,
    ancilla_qubit: int | None = None,
    cancel_adjacent_equal_bases: bool = True,
    pf_label: Literal["2nd"] = "2nd",
) -> DFPartialS2StepRequest:
    """Create one valid sampled request or a deterministic-only request."""
    if preparation.is_deterministic_only:
        return DFPartialS2StepRequest(
            preparation=preparation,
            step_time=float(step_time),
            rte_config=None,
            rte_distribution=None,
            rte_occurrence=None,
            controlled=controlled,
            ancilla_qubit=ancilla_qubit,
            seed=None,
            pf_label=pf_label,
        )
    config, distribution = make_rte_config(
        preparation.rte_preparation.symbolic_tail,
        evolution_time=float(step_time),
        rte_steps=rte_steps,
        truncation_tolerance=truncation_tolerance,
        finite_taylor_order=finite_taylor_order,
        seed=seed,
    )
    occurrence = preparation.rte_preparation.sample_occurrence_request(
        config,
        distribution,
        seed=seed,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        cancel_adjacent_equal_bases=cancel_adjacent_equal_bases,
    )
    return DFPartialS2StepRequest(
        preparation=preparation,
        step_time=float(step_time),
        rte_config=config,
        rte_distribution=distribution,
        rte_occurrence=occurrence,
        controlled=controlled,
        ancilla_qubit=ancilla_qubit,
        seed=seed,
        pf_label=pf_label,
    )


@dataclass(frozen=True)
class DFPartialS2AdditiveCircuits:
    forward_deterministic_half: QuantumCircuit
    rte_occurrence: QuantumCircuit
    reverse_deterministic_half: QuantumCircuit
    forward_fingerprint: str
    rte_fingerprint: str
    reverse_fingerprint: str


@dataclass(frozen=True)
class DFPartialS2CircuitResult:
    circuit: QuantumCircuit
    hamiltonian_hash: str
    partition_hash: str
    preparation_hash: str
    step_time: float
    ld: int
    controlled: bool
    ancilla_qubit: int | None
    deterministic_block_order: tuple[str, ...]
    randomized_block_indices: tuple[int, ...]
    randomized_event_count: int
    rte_steps: int
    finite_taylor_order: int | None
    exact_distribution_normalization: float
    attenuation_factor: float
    exact_rte_lambda_r: float
    ranking_proxy_lambda_r: float
    identity_policy: IdentityPolicy
    coefficient_atol: float
    threshold_dropped_component_count: int
    threshold_dropped_coefficient_l1: float
    threshold_error_bound: float
    constant_phase: float
    extracted_identity_phase: float
    rte_relative_phase: float
    rte_sequence_fingerprint: str | None
    basis_reuse_policy: Literal["disabled", "raw_adjacent_equal_basis", "none"]
    compiler_independent_fingerprint: str
    untranspiled_circuit_size: int
    untranspiled_circuit_depth: int
    circuit_qubit_count: int


class QiskitDFPartialS2CircuitBuilder:
    """Build one explicit deterministic-half/RTE/reverse-half S2 step."""

    @staticmethod
    def _new_circuit(request: DFPartialS2StepRequest) -> QuantumCircuit:
        preparation = request.preparation
        if request.controlled:
            return QuantumCircuit(request.ancilla_qubit + 1)
        return QuantumCircuit(preparation.num_system_qubits)

    @staticmethod
    def _append_basis(
        circuit: QuantumCircuit,
        block: DFDeterministicBlockSpec,
        *,
        inverse: bool,
    ) -> None:
        operations = list(block.runtime_basis_operations)
        if inverse:
            operations.reverse()
        for gate, qubits in operations:
            circuit.append(gate.inverse() if inverse else gate, list(qubits))

    def _append_block(
        self,
        circuit: QuantumCircuit,
        block: DFDeterministicBlockSpec,
        request: DFPartialS2StepRequest,
    ) -> None:
        self._append_basis(circuit, block, inverse=True)
        half_time = request.step_time / 2.0
        if isinstance(block, DFDeterministicOneBodySpec):
            primitives = one_body_diagonal_primitives(
                np.asarray(block.diagonal_eigenvalues),
                half_time,
            )
        else:
            primitives = df_squared_diagonal_primitives(
                np.asarray(block.diagonal_eta),
                block.lam,
                half_time,
            )
        append_diagonal_primitives(
            circuit,
            primitives,
            controlled=request.controlled,
            ancilla_qubit=request.ancilla_qubit,
        )
        self._append_basis(circuit, block, inverse=False)

    @staticmethod
    def _append_step_phases(
        circuit: QuantumCircuit,
        request: DFPartialS2StepRequest,
    ) -> tuple[float, float]:
        constant_phase = -request.step_time * request.preparation.constant_coefficient
        identity_phase = (
            -request.step_time
            * request.preparation.extracted_identity_coefficient
        )
        for phase in (constant_phase, identity_phase):
            if abs(phase) <= 1e-15:
                continue
            if request.controlled:
                circuit.append(PhaseGate(phase), [request.ancilla_qubit])
            else:
                circuit.global_phase += phase
        return float(constant_phase), float(identity_phase)

    @staticmethod
    def _part_fingerprint(
        request: DFPartialS2StepRequest,
        role: str,
        rte_fingerprint: str | None = None,
    ) -> str:
        payload = {
            "preparation_hash": request.preparation.preparation_hash,
            "hamiltonian_hash": request.preparation.hamiltonian_hash,
            "partition_hash": request.preparation.partition_hash,
            "ld": request.preparation.ld,
            "deterministic_block_order": request.preparation.deterministic_block_order,
            "step_time": request.step_time,
            "tail_hash": request.preparation.tail_extraction.tail_hash,
            "exact_rte_lambda_r": request.preparation.exact_rte_lambda_r,
            "finite_taylor_order": (
                None
                if request.rte_config is None
                else request.rte_config.finite_taylor_order
            ),
            "rte_fingerprint": rte_fingerprint,
            "identity_policy": request.preparation.identity_policy,
            "coefficient_atol": request.preparation.coefficient_atol,
            "controlled": request.controlled,
            "ancilla_qubit": request.ancilla_qubit,
            "basis_reuse_policy": (
                "none"
                if request.rte_occurrence is None
                else "raw_adjacent_equal_basis"
                if request.rte_occurrence.cancel_adjacent_equal_bases
                else "disabled"
            ),
            "role": role,
            "fingerprint_policy": "df_partial_s2_circuit_v1",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _build_parts(
        self,
        request: DFPartialS2StepRequest,
    ) -> tuple[DFPartialS2AdditiveCircuits, object | None]:
        forward = self._new_circuit(request)
        self._append_step_phases(forward, request)
        for block in request.preparation.deterministic_blocks:
            self._append_block(forward, block, request)

        rte_circuit = self._new_circuit(request)
        rte_result = None
        rte_fingerprint = self._part_fingerprint(request, "empty_rte_occurrence")
        if request.rte_occurrence is not None:
            rte_result = QiskitDFRTEEventCircuitBuilder(
                basis_registry=request.preparation.rte_preparation.basis_registry
            ).build_sequence(request.rte_occurrence)
            rte_circuit = rte_result.circuit
            rte_fingerprint = rte_result.circuit_fingerprint

        reverse = self._new_circuit(request)
        for block in reversed(request.preparation.deterministic_blocks):
            self._append_block(reverse, block, request)
        parts = DFPartialS2AdditiveCircuits(
            forward_deterministic_half=forward,
            rte_occurrence=rte_circuit,
            reverse_deterministic_half=reverse,
            forward_fingerprint=self._part_fingerprint(request, "forward_half"),
            rte_fingerprint=rte_fingerprint,
            reverse_fingerprint=self._part_fingerprint(request, "reverse_half"),
        )
        return parts, rte_result

    def build_additive_circuits(
        self,
        request: DFPartialS2StepRequest,
    ) -> DFPartialS2AdditiveCircuits:
        """Return matching forward/RTE/reverse circuits for additive costing."""
        parts, _rte_result = self._build_parts(request)
        return parts

    def build_step(
        self,
        request: DFPartialS2StepRequest,
    ) -> DFPartialS2CircuitResult:
        parts, rte_result = self._build_parts(request)
        circuit = self._new_circuit(request)
        qubits = tuple(range(circuit.num_qubits))
        for part in (
            parts.forward_deterministic_half,
            parts.rte_occurrence,
            parts.reverse_deterministic_half,
        ):
            circuit.compose(part, qubits=qubits, inplace=True)
        constant_phase = -request.step_time * request.preparation.constant_coefficient
        extracted_phase = (
            -request.step_time
            * request.preparation.extracted_identity_coefficient
        )
        config = request.rte_config
        rte_fingerprint = (
            None if rte_result is None else rte_result.circuit_fingerprint
        )
        full_fingerprint = self._part_fingerprint(
            request,
            "complete_partial_s2_step",
            rte_fingerprint,
        )
        if request.rte_occurrence is None:
            reuse_policy: Literal[
                "disabled", "raw_adjacent_equal_basis", "none"
            ] = "none"
        elif request.rte_occurrence.cancel_adjacent_equal_bases:
            reuse_policy = "raw_adjacent_equal_basis"
        else:
            reuse_policy = "disabled"
        return DFPartialS2CircuitResult(
            circuit=circuit,
            hamiltonian_hash=request.preparation.hamiltonian_hash,
            partition_hash=request.preparation.partition_hash,
            preparation_hash=request.preparation.preparation_hash,
            step_time=request.step_time,
            ld=request.preparation.ld,
            controlled=request.controlled,
            ancilla_qubit=request.ancilla_qubit,
            deterministic_block_order=(
                request.preparation.deterministic_block_order
            ),
            randomized_block_indices=request.preparation.randomized_block_indices,
            randomized_event_count=(
                0
                if request.rte_occurrence is None
                else len(request.rte_occurrence.events)
            ),
            rte_steps=0 if config is None else config.rte_steps,
            finite_taylor_order=(
                None if config is None else config.finite_taylor_order
            ),
            exact_distribution_normalization=(
                1.0 if config is None else config.distribution_normalization
            ),
            attenuation_factor=(
                1.0 if config is None else finite_rte_attenuation(config)
            ),
            exact_rte_lambda_r=request.preparation.exact_rte_lambda_r,
            ranking_proxy_lambda_r=request.preparation.ranking_proxy_lambda_r,
            identity_policy=request.preparation.identity_policy,
            coefficient_atol=request.preparation.coefficient_atol,
            threshold_dropped_component_count=(
                request.preparation.threshold_dropped_component_count
            ),
            threshold_dropped_coefficient_l1=(
                request.preparation.threshold_dropped_coefficient_l1
            ),
            threshold_error_bound=(
                request.preparation.threshold_operator_error_bound
            ),
            constant_phase=float(constant_phase),
            extracted_identity_phase=float(extracted_phase),
            rte_relative_phase=(
                0.0 if rte_result is None else rte_result.relative_ancilla_phase
            ),
            rte_sequence_fingerprint=rte_fingerprint,
            basis_reuse_policy=reuse_policy,
            compiler_independent_fingerprint=full_fingerprint,
            untranspiled_circuit_size=int(circuit.size()),
            untranspiled_circuit_depth=int(circuit.depth() or 0),
            circuit_qubit_count=circuit.num_qubits,
        )
