"""Explicit H4-A (T_eff / V_square) dictionary adapter.

Importing this module performs no I/O or scientific evaluation. Loading a saved
snapshot, building operators, and extracting a rotated DF dictionary are separate
explicit calls. The latter uses the existing Gaussian/DF machinery and may build
basis circuits; it is not needed to construct the Pauli/factor representations.
No time evolution, state preparation, selector, sampling or cost model is added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

IdentityPolicy = Literal["faithful_identity_in_tail", "extract_identity_phase"]


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.complex128, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite shape {shape}")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class MatchedAInputs:
    """Supplied spin tensors/factors and original-H diagnostic state, unscaled.

    ``spin_two`` already includes the original source's 1/2 prefactor. Spin
    orbitals are interleaved; state indices are Qiskit little-endian. Physical
    indices are metadata, never an instruction to project individual events.
    Numerical comparisons/tolerances and ground-state certification are external.
    """

    nuclear_repulsion: float
    spin_one: np.ndarray
    spin_two: np.ndarray
    df_one_body: np.ndarray
    df_lambdas: np.ndarray
    df_g_matrices: np.ndarray
    state: np.ndarray
    physical_qiskit_indices: np.ndarray

    def __post_init__(self) -> None:
        one = np.asarray(self.spin_one)
        if one.ndim != 2 or one.shape[0] != one.shape[1] or one.shape[0] < 2 or one.shape[0] % 2:
            raise ValueError("Expected an even number of interleaved spin orbitals")
        n = one.shape[0]
        lambdas = np.asarray(self.df_lambdas)
        if lambdas.ndim != 1 or len(lambdas) < 1:
            raise ValueError("Expected a nonempty saved DF factor list")
        for name, shape in (
            ("spin_one", (n, n)), ("spin_two", (n,) * 4),
            ("df_one_body", (n, n)), ("df_lambdas", lambdas.shape),
            ("df_g_matrices", (len(lambdas), n, n)), ("state", (1 << n,)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if np.any(self.df_lambdas.imag != 0):
            raise ValueError("DF weights must be real; do not silently discard imaginary parts")
        weights = self.df_lambdas.real.copy()
        weights.flags.writeable = False
        object.__setattr__(self, "df_lambdas", weights)
        scalar = complex(self.nuclear_repulsion)
        if not np.isfinite(scalar) or scalar.imag != 0:
            raise ValueError("nuclear_repulsion must be finite and real")
        object.__setattr__(self, "nuclear_repulsion", float(scalar.real))
        indices = np.asarray(self.physical_qiskit_indices)
        if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= 1 << n)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Expected distinct integer physical Qiskit indices")
        indices = indices.astype(np.int64, copy=True)
        indices.flags.writeable = False
        object.__setattr__(self, "physical_qiskit_indices", indices)

    @property
    def num_qubits(self) -> int:
        return len(self.spin_one)


def load_h4_a_snapshot(path: Any) -> MatchedAInputs:
    """Load only the H4-A fields from an explicitly supplied saved NPZ.

    This is a scientific-input load, not a preflight. Call only within an
    authorized scope. No default path, search, fallback to DF_state, re-DF or
    ground-state solve exists. Fixed commit/hash provenance is checked by the
    caller; matching dimensions alone do not identify the physical input.
    """
    names = tuple(MatchedAInputs.__dataclass_fields__)
    with np.load(path, allow_pickle=False) as saved:
        inputs = MatchedAInputs(**{name: saved[name] for name in names})
    if (inputs.num_qubits != 8 or len(inputs.df_lambdas) != 10
            or len(inputs.physical_qiskit_indices) != 36):
        raise ValueError("H4-A requires saved spin8 / rank10 / sector36 metadata")
    return inputs


@dataclass(frozen=True)
class MatchedASplit:
    inputs: MatchedAInputs
    df_hamiltonian: Any
    pauli_retained: Any
    pauli_residual: Any
    original_pauli: Any
    one_body_correction: np.ndarray
    source_one_body_correction: np.ndarray

    def pauli_full(self) -> Any:
        """c_N I + the SAME T_eff + V_square; original-target error is separate."""
        from .pauli_signal_adapter import PauliHamiltonian

        n = self.inputs.num_qubits
        return PauliHamiltonian.from_terms(
            [("I" * n, self.inputs.nuclear_repulsion),
             ("I" * n, self.pauli_retained.constant),
             ("I" * n, self.pauli_residual.constant),
             *self.pauli_retained.terms, *self.pauli_residual.terms],
            num_qubits=n, model_id="M1_A_matched_full",
        )


def prepare_matched_a(inputs: MatchedAInputs) -> MatchedASplit:
    """Construct explicit Pauli operators and preserve the saved native DF form.

    P and S use the same saved square factors. P is NOT original_H - T_eff:
    that would transfer DF approximation error into only one dictionary.
    This call performs operator algebra, not a circuit build or simulation.
    OpenFermion's internal coefficient tolerances still apply: the absence of
    an additional adapter cutoff is not a guarantee of exact tiny-term retention.
    """
    from openfermion import FermionOperator, InteractionOperator, get_fermion_operator, jordan_wigner
    from openfermion.circuits.low_rank import get_chemist_two_body_coefficients
    from .df_hamiltonian import DFHamiltonian
    from .pauli_signal_adapter import PauliHamiltonian

    n = inputs.num_qubits
    for name, matrices in (("spin_one", (inputs.spin_one,)),
                           ("df_one_body", (inputs.df_one_body,)),
                           ("DF G", inputs.df_g_matrices)):
        if any(not np.allclose(g, g.conj().T, rtol=0, atol=1e-12) for g in matrices):
            raise ValueError(f"{name} must be Hermitian; inputs are never symmetrized here")

    def one_body(matrix: np.ndarray) -> Any:
        op = FermionOperator()
        for p, q in np.argwhere(matrix != 0):
            op += FermionOperator(((int(p), 1), (int(q), 0)), complex(matrix[p, q]))
        return op

    residual = FermionOperator()
    for weight, g in zip(inputs.df_lambdas, inputs.df_g_matrices, strict=True):
        a = one_body(g)
        residual += float(weight) * (a * a)

    def pauli(op: Any, model_id: str) -> Any:
        return PauliHamiltonian.from_openfermion(
            jordan_wigner(op), num_qubits=n, model_id=model_id,
            metadata={"physical_split": "T_eff/V_square", "explicit_coefficient_cutoff": 0.0},
        )

    source_correction, _ = get_chemist_two_body_coefficients(inputs.spin_two, spin_basis=True)
    correction = _array(inputs.df_one_body - inputs.spin_one, (n, n), "one_body_correction")
    source_correction = _array(source_correction, (n, n), "source_one_body_correction")
    df = DFHamiltonian(
        constant=inputs.nuclear_repulsion, one_body=inputs.df_one_body,
        lambdas=inputs.df_lambdas, g_matrices=tuple(inputs.df_g_matrices),
        metadata={"physical_split": "T_eff/V_square", "saved_factor_order_preserved": True,
                  "state_field": "state", "state_role": "original_H_diagnostic_only"},
    )
    original = InteractionOperator(inputs.nuclear_repulsion, inputs.spin_one, inputs.spin_two)
    return MatchedASplit(
        inputs, df, pauli(one_body(inputs.df_one_body), "M1_A_T_eff"),
        pauli(residual, "M1_A_V_square"), pauli(get_fermion_operator(original), "M1_original_H"),
        correction, source_correction,
    )


def pauli_tail_coefficients(
    split: MatchedASplit, *, identity_policy: IdentityPolicy,
) -> tuple[tuple[tuple[str, float], ...], float]:
    """Return the explicit P dictionary plus the residual-only scalar transfer.

    T_eff's electronic identity stays in the common retained block, and c_N
    stays outside it. No scalar transfer here is the total full-H scalar.
    """
    if identity_policy not in ("faithful_identity_in_tail", "extract_identity_phase"):
        raise ValueError("Unsupported identity policy")
    tail = split.pauli_residual
    if identity_policy == "extract_identity_phase":
        return tail.terms, tail.constant
    identity = (("I" * split.inputs.num_qubits, tail.constant),) if tail.constant != 0 else ()
    return (*identity, *tail.terms), 0.0


def extract_matched_df_tail(split: MatchedASplit, *, identity_policy: IdentityPolicy) -> Any:
    """Explicit existing S dictionary entry; may generate Gaussian basis circuits.

    This is deliberately NOT called by loading, preparation, import or tests of
    numerical Pauli/factor algebra. Requires the separately permitted scope.
    """
    from .df_rte_tail import extract_df_tail_from_hamiltonian

    return extract_df_tail_from_hamiltonian(
        "M1_A_V_square", split.df_hamiltonian,
        tuple(range(split.df_hamiltonian.n_blocks)),
        identity_policy=identity_policy, coefficient_atol=0.0,
    )
