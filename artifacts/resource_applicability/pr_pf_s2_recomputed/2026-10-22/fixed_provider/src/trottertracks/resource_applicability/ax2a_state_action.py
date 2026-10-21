"""AX-2A implementation-only state actions; no molecular I/O or sampling.

Callbacks must use one common basis and carry their own numerical accuracy.
Synthetic checks do not certify a molecular sector or an H6/H8 reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from trotterlib.pf_decomposition import iter_pf_steps
from trotterlib.product_formula import _get_w_list
from trotterlib.rte import finite_rte_distribution, require_integer_count

Matvec = Callable[[np.ndarray], np.ndarray]
Evolution = Callable[[np.ndarray, float], np.ndarray]


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _vector(value: np.ndarray, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError("Expected a nonempty finite one-dimensional vector.")
    if dimension is not None and result.size != dimension:
        raise ValueError("State action changed the vector dimension.")
    return result


def _initial(value: np.ndarray) -> np.ndarray:
    result = _vector(value).copy()
    if abs(np.linalg.norm(result) - 1.0) > 1e-12:
        raise ValueError("Initial state must be normalized; no implicit renormalization.")
    return result


@dataclass
class ActionBudget:
    """Operation counters, independent of CPU/RAM allocation or wall limits."""

    max_tail_matvecs: int
    max_deterministic_actions: int
    tail_matvecs: int = 0
    deterministic_actions: int = 0

    def __post_init__(self) -> None:
        for name in ("max_tail_matvecs", "max_deterministic_actions"):
            require_integer_count(getattr(self, name), name=name)

    def tail(self) -> None:
        if self.tail_matvecs >= self.max_tail_matvecs:
            raise RuntimeError("TAIL_MATVEC_BUDGET")
        self.tail_matvecs += 1

    def deterministic(self) -> None:
        if self.deterministic_actions >= self.max_deterministic_actions:
            raise RuntimeError("DETERMINISTIC_ACTION_BUDGET")
        self.deterministic_actions += 1


def finite_taylor_action(
    normalized_matvec: Matvec, vector: np.ndarray, tau: float, K: int,
    *, budget: ActionBudget,
) -> np.ndarray:
    """Apply source-defined paired Taylor numerator through degree K+1.

    Horner uses K+1 matvecs and O(dimension) vector storage. The mean is
    nonunitary: intermediate and final norms are intentionally not rescaled.
    """
    tau = _real(tau, "tau")
    K = require_integer_count(K, name="K")
    if K % 2:
        raise ValueError("K must be even.")
    state = _vector(vector)
    if tau == 0:
        return state.copy()
    current = state.copy()
    for degree in range(K + 1, 0, -1):
        budget.tail()
        product = _vector(normalized_matvec(current), state.size)
        current = _vector(state + (-1j * tau / degree) * product, state.size)
    return current


def _outer(
    state: np.ndarray, deterministic: Sequence[Evolution], delta: float,
    phase_energy: float, tail: Matvec | None, budget: ActionBudget,
) -> np.ndarray:
    current = np.exp(-1j * phase_energy * delta) * state
    for action in deterministic:
        budget.deterministic()
        current = _vector(action(current, delta / 2), state.size)
    if tail is not None:
        current = _vector(tail(current), state.size)
    for action in reversed(deterministic):
        budget.deterministic()
        current = _vector(action(current, delta / 2), state.size)
    return current


@dataclass(frozen=True)
class PartialSignal:
    corrected: complex
    raw: complex | None
    log_normalization: float
    normalization: float | None
    raw_status: str


def partial_s2_signal(
    state: np.ndarray, deterministic: Sequence[Evolution],
    normalized_tail: Matvec | None, *, lambda_r: float, T: float,
    q: int, r: int, K: int, phase_energy: float, budget: ActionBudget,
) -> PartialSignal:
    """Connect M1 forward/tail/reverse semantics without many-body matrices.

    phase_energy is constant + extracted tail identity. normalized_tail is
    (sum of randomized DF fragments - extracted identity)/lambda_r;
    it must exclude one-body and the molecular constant.
    """
    initial = _initial(state)
    q = require_integer_count(q, name="q", minimum=1)
    r = require_integer_count(r, name="r", minimum=1)
    K = require_integer_count(K, name="K")
    if K % 2:
        raise ValueError("K must be even.")
    lambda_r = _real(lambda_r, "lambda_r")
    T, phase_energy = _real(T, "T"), _real(phase_energy, "phase_energy")
    if lambda_r < 0 or (lambda_r == 0) != (normalized_tail is None):
        raise ValueError("Empty tail requires lambda_r=0 and normalized_tail=None.")
    delta = T / q
    tau = lambda_r * delta / r
    b = finite_rte_distribution(tau, K).exact_finite_distribution
    log_B = q * r * math.log(b)
    if not math.isfinite(log_B):
        raise ValueError("NORMALIZATION_NONFINITE")
    B = math.exp(log_B) if log_B <= math.log(np.finfo(float).max) else None
    attenuation = math.exp(-log_B)
    raw_available = attenuation > 0

    def occurrence(vector: np.ndarray, raw: bool) -> np.ndarray:
        current = vector
        for _ in range(r):
            current = finite_taylor_action(normalized_tail, current, tau, K, budget=budget)
            if raw:
                current = current / b
        return current

    corrected, raw = initial.copy(), initial.copy()
    for _ in range(q):
        corrected = _outer(
            corrected, deterministic, delta, phase_energy,
            (lambda v: occurrence(v, False)) if normalized_tail else None, budget,
        )
        if raw_available:
            raw = _outer(
                raw, deterministic, delta, phase_energy,
                (lambda v: occurrence(v, True)) if normalized_tail else None, budget,
            )
    return PartialSignal(
        complex(np.vdot(initial, corrected)),
        complex(np.vdot(initial, raw)) if raw_available else None,
        log_B, B, "AVAILABLE" if raw_available else "RAW_NORMALIZATION_UNDERFLOW",
    )


def deterministic_pf_state(
    state: np.ndarray, actions: Sequence[Evolution], *, T: float, q: int,
    formula: str, scalar: float, budget: ActionBudget,
) -> np.ndarray:
    """Global second/fourth-order DF-term PF; not just inner-H_D upgrade."""
    current = _initial(state)
    q = require_integer_count(q, name="q", minimum=1)
    T, scalar = _real(T, "T"), _real(scalar, "scalar")
    if formula not in ("2nd", "4th"):
        raise ValueError("Only legacy second-order and standard Yoshida fourth supported.")
    # One term is exactly solvable; the legacy merged multi-term iterator
    # is not used for this degenerate case.
    stages = ((0, 1.0),) if len(actions) == 1 else tuple(
        iter_pf_steps(len(actions), _get_w_list(formula))
    )
    for _ in range(q):
        for index, fraction in stages:
            budget.deterministic()
            current = _vector(actions[index](current, T * fraction / q), current.size)
    return _vector(np.exp(-1j * scalar * T) * current)


def project_primitive_checked(
    full_action: Matvec, state: np.ndarray, indices: Sequence[int],
    full_dimension: int, *, leakage_tolerance: float = 1e-12,
) -> np.ndarray:
    """Check every primitive before projection; no net-sector inference.

    Failure requires a number-sector/full-vector implementation, not silent
    truncation. Passing this check for one state is not a general certificate.
    """
    full_dimension = require_integer_count(full_dimension, name="full_dimension", minimum=1)
    selected = tuple(require_integer_count(i, name="index") for i in indices)
    tolerance = _real(leakage_tolerance, "leakage_tolerance")
    if tolerance < 0 or len(set(selected)) != len(selected):
        raise ValueError("Invalid leakage tolerance or duplicate sector indices.")
    if not selected or any(i >= full_dimension for i in selected):
        raise ValueError("Sector indices out of range.")
    value = _vector(state, len(selected))
    lifted = np.zeros(full_dimension, dtype=np.complex128)
    lifted[list(selected)] = value
    acted = _vector(full_action(lifted), full_dimension)
    outside = acted.copy()
    outside[list(selected)] = 0
    if np.linalg.norm(outside) > tolerance * max(1.0, np.linalg.norm(value)):
        raise ValueError("PRIMITIVE_LEAVES_SECTOR")
    return acted[list(selected)]


def df_tail_operator(
    hamiltonian, sector, randomized_indices, *, extracted_identity: float,
    primitive_sector_certified: bool = False,
):
    """Reuse existing sector matvec, removing one-body/constant exactly once.

    Input basis is OpenFermion sector order, not Qiskit order. Preparation
    must certify this sector for each squared one-body primitive before use.
    No integral/DF/state generation, eigensolve or molecular I/O occurs here.
    """
    from trotterlib.df_hamiltonian import DFHamiltonian, df_linear_operator

    if primitive_sector_certified is not True:
        raise ValueError("Primitive-sector preservation must be certified before DF projection.")
    if sector.n_qubits != hamiltonian.n_qubits:
        raise ValueError("Hamiltonian and sector dimensions differ.")
    indices = tuple(require_integer_count(i, name="fragment index") for i in randomized_indices)
    if len(set(indices)) != len(indices) or any(i >= hamiltonian.n_blocks for i in indices):
        raise ValueError("Invalid randomized fragment indices.")
    constant = -_real(extracted_identity, "extracted_identity")
    tail = DFHamiltonian(
        constant=constant, one_body=np.zeros_like(hamiltonian.one_body),
        lambdas=hamiltonian.lambdas[list(indices)].copy(),
        g_matrices=tuple(hamiltonian.g_matrices[i] for i in indices),
        metadata={"operator_role": "AX2A_tail_without_one_body_or_constant"},
    )
    return df_linear_operator(tail, sector, backend="python")


def eigenphase_reference(state: np.ndarray, hermitian_matvec: Matvec, T: float):
    """Phase surrogate and T*residual allowance for a caller-certified Hermitian H.

    Does not assert eigenstate or ground-state certification. The allowance
    excludes solver/roundoff error and must be supplemented in AX-2B.
    """
    initial = _initial(state)
    T = _real(T, "T")
    image = _vector(hermitian_matvec(initial), initial.size)
    energy = complex(np.vdot(initial, image))
    if abs(energy.imag) > 1e-12:
        raise ValueError("Non-real Rayleigh energy for declared Hermitian operator.")
    residual = float(np.linalg.norm(image - energy.real * initial))
    return {
        "signal": complex(np.exp(-1j * energy.real * T)),
        "rayleigh_energy": energy.real,
        "residual": residual,
        "signal_allowance": abs(T) * residual,
        "ground_state_certified": False,
    }
