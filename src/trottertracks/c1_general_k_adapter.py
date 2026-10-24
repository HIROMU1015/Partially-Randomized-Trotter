"""Opt-in number-density data adapter; no fit, circuit construction or sampling.

The import path uses only the standard library. Existing trotterlib types and
the diagonal emitter are imported lazily at explicit bridge calls. Gaussian
operations must be supplied by a separately authorized caller; U conventions
are recorded here, not certified by this adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Complex, Real
from typing import Any

SCHEMA = "C1_general_K_number_v1"
FFSIM_SOURCE = "558d3517385778c733f1aa226bb61b8f30f4b0f3"
COEFFICIENT_CONVENTION = "D=sum_ab_K_ab_n_a_n_b_including_diagonal_and_both_orders"
ROTATION_CONVENTION = "columns_define_rotated_creation_operators"
SPIN_ORDER = "interleaved_2i_plus_sigma"


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _complex(value: Any, name: str) -> complex:
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise ValueError(f"{name} must be a real or complex number")
    try:
        result = complex(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(value: Any, name: str, *, real: bool = False) -> tuple:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty square matrix") from exc
    n = len(rows)
    if not n or any(len(row) != n for row in rows):
        raise ValueError(f"{name} must be a nonempty square matrix")
    convert = _real if real else _complex
    return tuple(tuple(convert(x, name) for x in row) for row in rows)


def _symmetric(value: Any, name: str) -> tuple:
    result = _matrix(value, name, real=True)
    if any(result[i][j] != result[j][i]
           for i in range(len(result)) for j in range(i)):
        raise ValueError(f"{name} must be real symmetric; no implicit symmetrization")
    return result


def _index(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative Python integer")
    return value


def _enabled(value: bool) -> None:
    if value is not True:
        raise ValueError("The general-K route requires enable_general_k=True")


def _digest(value: Any) -> str:
    def encode(item):
        if isinstance(item, complex):
            return {"real": item.real, "imag": item.imag}
        raise TypeError(type(item).__name__)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=encode).encode()
    return hashlib.sha256(data).hexdigest()


def _spin_lift(matrix: tuple) -> tuple:
    n = len(matrix)
    return tuple(tuple(matrix[a // 2][b // 2] if a % 2 == b % 2 else 0j
                       for b in range(2 * n)) for a in range(2 * n))


def _spin_density(J: tuple) -> tuple:
    n = len(J)
    return tuple(tuple(J[a // 2][b // 2] / 2.0 for b in range(2 * n))
                 for a in range(2 * n))


@dataclass(frozen=True)
class GeneralKFrame:
    """One original frame, without auxiliary square splitting or phase removal.

    K is real symmetric. orbital_rotation is the spin-orbital V in the common
    interleaved basis. Unitarity and its Gaussian implementation remain separate
    scientific checks; only shape, entries and declared conventions are checked.
    """

    frame_id: str
    group_id: str
    original_index: int
    K: tuple
    orbital_rotation: tuple
    source_identity: str
    input_identity: str
    spatial_J: tuple | None = None
    spatial_orbital_rotation: tuple | None = None
    representation: str = "number"
    spin_order: str = SPIN_ORDER
    rotation_convention: str = ROTATION_CONVENTION
    coefficient_convention: str = COEFFICIENT_CONVENTION

    def __post_init__(self):
        for name in ("frame_id", "group_id", "source_identity", "input_identity"):
            _identity(getattr(self, name), name)
        _index(self.original_index, "original_index")
        if (self.representation != "number" or self.spin_order != SPIN_ORDER
                or self.rotation_convention != ROTATION_CONVENTION
                or self.coefficient_convention != COEFFICIENT_CONVENTION):
            raise ValueError("Unsupported representation/index/coefficient convention")
        K = _symmetric(self.K, "K")
        V = _matrix(self.orbital_rotation, "orbital_rotation")
        if len(K) != len(V):
            raise ValueError("K and orbital_rotation dimensions differ")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "orbital_rotation", V)
        if (self.spatial_J is None) != (self.spatial_orbital_rotation is None):
            raise ValueError("Spatial J and U provenance must be supplied together")
        if self.spatial_J is not None:
            J = _symmetric(self.spatial_J, "spatial_J")
            U = _matrix(self.spatial_orbital_rotation, "spatial_orbital_rotation")
            if len(J) != len(U) or K != _spin_density(J) or V != _spin_lift(U):
                raise ValueError("Spatial J/U and spin K/V provenance disagree")
            object.__setattr__(self, "spatial_J", J)
            object.__setattr__(self, "spatial_orbital_rotation", U)

    @property
    def num_system_qubits(self) -> int:
        return len(self.K)

    @property
    def frame_hash(self) -> str:
        return _digest({"namespace": SCHEMA, **asdict(self)})

    @property
    def logical_basis_id(self) -> str:
        # A namespaced request ID, never a claim of validated Gaussian action.
        return "c1-general-k-basis:" + _digest(dict(
            namespace=SCHEMA, V=self.orbital_rotation,
            source=self.source_identity, input=self.input_identity,
            convention=self.rotation_convention))

    @property
    def interleaved_to_block_permutation(self) -> tuple[int, ...]:
        if self.spatial_J is None:
            raise ValueError("A general spin K need not have a spatial spin lift")
        n = len(self.spatial_J)
        return tuple(a // 2 + (a % 2) * n for a in range(2 * n))

    def gaussian_request(self) -> dict:
        """Data only: no synthesis, matrix reference or circuit is invoked."""
        return dict(basis_id=self.logical_basis_id, V=self.orbital_rotation,
                    helper="trotterlib.df_trotter.ops.U_to_qiskit_ops_jw_givens",
                    helper_argument="V; helper internally uses BogoliubovTransform(V.T)",
                    rotation_convention=self.rotation_convention,
                    application_order=("basis_inverse", "central", "basis_forward"),
                    independent_CAR_validation="pending")


@dataclass(frozen=True)
class GeneralKModel:
    """Corrected spin-orbital kappa and original constant are owned only here.

    Kappa is checked for shape/finite entries, without a projection or an
    unapproved numerical Hermiticity tolerance. That scientific gate is pending.
    """

    one_body_tensor: tuple
    constant: float
    frames: tuple[GeneralKFrame, ...]
    source_identity: str
    input_identity: str
    spatial_one_body_tensor: tuple | None = None
    representation: str = "number"
    one_body_role: str = "ffsim_corrected_kappa"

    def __post_init__(self):
        _identity(self.source_identity, "source_identity")
        _identity(self.input_identity, "input_identity")
        if self.representation != "number" or self.one_body_role != "ffsim_corrected_kappa":
            raise ValueError("Expected corrected kappa in number representation")
        kappa = _matrix(self.one_body_tensor, "one_body_tensor")
        frames = tuple(self.frames)
        if any(not isinstance(f, GeneralKFrame) for f in frames):
            raise TypeError("GeneralKModel accepts only explicit GeneralKFrame records")
        if len({f.frame_id for f in frames}) != len(frames):
            raise ValueError("frame_id must be unique; frames are never merged")
        if tuple(f.original_index for f in frames) != tuple(range(len(frames))):
            raise ValueError("Frames must retain their original outer order")
        if any(len(f.K) != len(kappa) or f.source_identity != self.source_identity
               or f.input_identity != self.input_identity for f in frames):
            raise ValueError("Frame dimension/provenance differs from model")
        object.__setattr__(self, "one_body_tensor", kappa)
        object.__setattr__(self, "constant", _real(self.constant, "constant"))
        object.__setattr__(self, "frames", frames)
        if self.spatial_one_body_tensor is not None:
            spatial = _matrix(self.spatial_one_body_tensor, "spatial_one_body_tensor")
            if kappa != _spin_lift(spatial):
                raise ValueError("Spatial and spin corrected kappa disagree")
            object.__setattr__(self, "spatial_one_body_tensor", spatial)

    @property
    def original_outer_rank(self) -> int:
        return len(self.frames)

    @property
    def cache_identity(self) -> str:
        return SCHEMA + ":" + _digest(asdict(self))

    def reference_input(self) -> dict:
        """Original number-density data, not the candidate Z expansion.

        A future CAR reference must independently contract J/U or K/n, and must
        obtain physical h/g from input_identity separately. No operator is made.
        """
        return dict(namespace=SCHEMA, source_identity=self.source_identity,
                    input_identity=self.input_identity, constant=self.constant,
                    corrected_kappa=self.one_body_tensor,
                    frames=tuple(dict(frame_id=f.frame_id, group_id=f.group_id,
                                      U=f.spatial_orbital_rotation, J=f.spatial_J,
                                      V=f.orbital_rotation,
                                      raw_K=f.K if f.spatial_J is None else None)
                                 for f in self.frames),
                    one_body_hermiticity="pending_separate_scientific_gate",
                    independent_reference_status="not_evaluated")


def from_ffsim_number_arrays(*, one_body_tensor, diag_coulomb_mats,
                             orbital_rotations, constant, z_representation,
                             frame_ids, group_ids, source_identity, input_identity,
                             one_body_role, enable_general_k=False) -> GeneralKModel:
    """Import arrays/mocks without importing ffsim, fitting or re-correcting h.

    Input one_body_tensor is spatial corrected kappa, not the original h.
    z_representation must explicitly be False; Z inputs are never auto-converted.
    """
    _enabled(enable_general_k)
    if z_representation is not False or one_body_role != "ffsim_corrected_kappa":
        raise ValueError("Only explicit number representation / corrected kappa is supported")
    if isinstance(frame_ids, (str, bytes)) or isinstance(group_ids, (str, bytes)):
        raise ValueError("frame_ids and group_ids must be sequences of identifiers")
    kappa = _matrix(one_body_tensor, "one_body_tensor")
    Js, Us, ids, groups = map(tuple, (diag_coulomb_mats, orbital_rotations,
                                    frame_ids, group_ids))
    if len({len(Js), len(Us), len(ids), len(groups)}) != 1:
        raise ValueError("J/U/frame/group counts differ")
    frames = []
    for index, (J, U, frame_id, group_id) in enumerate(zip(Js, Us, ids, groups)):
        J = _symmetric(J, "J")
        U = _matrix(U, "U")
        if len(J) != len(kappa) or len(U) != len(kappa):
            raise ValueError("Spatial J/U/kappa shapes differ")
        frames.append(GeneralKFrame(frame_id, group_id, index, _spin_density(J),
                                   _spin_lift(U), source_identity, input_identity,
                                   spatial_J=J, spatial_orbital_rotation=U))
    return GeneralKModel(_spin_lift(kappa), constant, tuple(frames),
                         source_identity, input_identity,
                         spatial_one_body_tensor=kappa)


def general_k_coefficients(K) -> tuple[tuple[tuple[int, ...], float], ...]:
    """Exact-support I/Z/ZZ expansion; no threshold, tail norm or sampling law."""
    K = _symmetric(K, "K")
    n = len(K)
    try:
        terms = [((), math.fsum((*[x for row in K for x in row],
                                *[K[a][a] for a in range(n)])) / 4.0)]
        terms.extend(((a,), -0.5 * math.fsum(K[a])) for a in range(n))
        terms.extend(((a, b), 0.5 * K[a][b])
                     for a in range(n) for b in range(a + 1, n))
        return tuple((support, _real(value, "derived coefficient"))
                     for support, value in terms)
    except OverflowError as exc:
        raise ValueError("Coefficient arithmetic overflowed") from exc


def diagonal_primitives(frame: GeneralKFrame, tau):
    """Bridge to the existing primitive type. Owns only this frame's phase."""
    if not isinstance(frame, GeneralKFrame):
        raise TypeError("An explicit GeneralKFrame is required")
    tau = _real(tau, "tau")
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives
    coefficients = general_k_coefficients(frame.K)
    phase = _real(-tau * coefficients[0][1], "phase")
    rz, rzz = [], []
    for support, coefficient in coefficients[1:]:
        angle = _real(2.0 * tau * coefficient, "rotation angle")
        if angle != 0.0:  # exact-zero pruning only, no scientific cutoff
            (rz if len(support) == 1 else rzz).append((*support, angle))
    return DiagonalEvolutionPrimitives(global_phase=phase, rz=tuple(rz), rzz=tuple(rzz))


def append_general_k_diagonal(circuit, frame, tau, *, controlled,
                             ancilla_qubit=None, enable_general_k=False):
    """Delegate to the established emitter; never construct a circuit here."""
    _enabled(enable_general_k)
    if not isinstance(frame, GeneralKFrame):
        raise TypeError("An explicit GeneralKFrame is required")
    if type(controlled) is not bool:
        raise ValueError("controlled must be bool")
    if controlled:
        _index(ancilla_qubit, "ancilla_qubit")
        if ancilla_qubit < frame.num_system_qubits:
            raise ValueError("Ancilla must be outside the system register")
    elif ancilla_qubit is not None:
        raise ValueError("An uncontrolled call must not supply an ancilla")
    from trotterlib.df_trotter.ops import append_diagonal_primitives
    primitives = diagonal_primitives(frame, tau)
    append_diagonal_primitives(circuit, primitives, controlled=controlled,
                               ancilla_qubit=ancilla_qubit)
    return primitives


@dataclass(frozen=True)
class BoundGeneralKFrame:
    """Deterministic-spec preparation and unnormalized RTE component sidecar.

    This is deliberately NOT DFDeterministicFragmentSpec / DFPartialS2Preparation:
    partition weights, identity policy and finite target have not been selected.
    """

    frame: GeneralKFrame
    order_index: int
    basis_definition: Any = field(repr=False, compare=False)
    block_kind: str = field(default="general_k_fragment", init=False)
    validation_status: str = field(default="NOT_NUMERICALLY_VALIDATED", init=False)

    def __post_init__(self):
        _index(self.order_index, "order_index")
        if (self.basis_definition.num_system_qubits != self.frame.num_system_qubits
                or self.basis_definition.basis_id != self.frame.logical_basis_id):
            raise ValueError("Bound basis does not match the general frame")

    def diagonal_components(self):
        """Candidate components only: no tail extraction/normalization/sampling."""
        from trotterlib.df_rte_tail import DFDiagonalPauliComponent
        basis = self.basis_definition
        return tuple(DFDiagonalPauliComponent(
            component_id=f"c1-general-k:{self.frame.frame_hash}:{support}",
            coefficient=value, coefficient_abs=abs(value),
            coefficient_sign=1 if value > 0 else -1,
            df_fragment_id=f"c1-general-k:{self.frame.frame_hash}",
            basis_id=basis.basis_id, basis_hash=basis.basis_hash,
            diagonal_pauli_support=support,
            basis_change_operations=basis.metadata.operations)
            for support, value in general_k_coefficients(self.frame.K) if value != 0.0)

    def component_specs(self):
        from trotterlib.df_rte_circuit import (
            DFRTEComponentCircuitSpec, DFRTEIdentityCircuitSpec)
        specs = []
        for component in self.diagonal_components():
            args = dict(component_id=component.component_id,
                        coefficient_abs=component.coefficient_abs,
                        coefficient_sign=component.coefficient_sign,
                        df_fragment_id=component.df_fragment_id,
                        basis_id=component.basis_id, basis_hash=component.basis_hash,
                        diagonal_pauli_support=component.diagonal_pauli_support,
                        num_system_qubits=self.frame.num_system_qubits)
            if component.is_identity:
                specs.append(DFRTEIdentityCircuitSpec(**args))
            else:
                specs.append(DFRTEComponentCircuitSpec(
                    **args, basis_change_operations=component.basis_change_operations))
        return tuple(specs)


def bind_frame(frame: GeneralKFrame, *, order_index, basis_registry,
               runtime_operations, enable_general_k=False) -> BoundGeneralKFrame:
    """Bind caller-supplied Gaussian ops; no synthesis or D/R choice is made.

    The established registry may evaluate local Operator fingerprints. Real
    operation binding therefore belongs to future authorized science; technical
    tests mock the operation-description boundary without evaluating matrices.
    """
    _enabled(enable_general_k)
    _index(order_index, "order_index")
    if not isinstance(frame, GeneralKFrame):
        raise TypeError("An explicit GeneralKFrame is required")
    basis = basis_registry.register(tuple(runtime_operations),
                                    num_system_qubits=frame.num_system_qubits,
                                    basis_id=frame.logical_basis_id)
    return BoundGeneralKFrame(frame, order_index, basis)
