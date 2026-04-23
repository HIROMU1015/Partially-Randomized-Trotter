from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from openfermion.ops import QubitOperator
from scipy.optimize import minimize_scalar

from .analysis_utils import loglog_average_coeff, loglog_fit
from .chemistry_hamiltonian import ham_ground_energy, ham_list_maker, jw_hamiltonian_maker
from .config import (
    PARTIAL_RANDOMIZED_CGS_CACHE_PATH,
    PARTIAL_RANDOMIZED_BOUNDARY_REL_TOL,
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_GRID,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
    PARTIAL_RANDOMIZED_Q_MAX,
    PARTIAL_RANDOMIZED_Q_MIN,
    PERTURBATION_FIT_SPAN,
    PERTURBATION_FIT_STARTS,
    PERTURBATION_FIT_STEP,
    P_DIR,
    PFLabel,
    pf_order,
)
from .pf_decomposition import iter_pf_steps
from .qiskit_time_evolution_ungrouped import tEvolution_vector


DEFAULT_PF_LABELS: tuple[PFLabel, ...] = (
    "2nd",
    "4th",
    "4th(new_2)",
    "4th(new_3)",
    "8th(Yoshida)",
    "8th(Morales)",
    "10th(Morales)",
)
_HIGH_ORDER_TIME_MODE_LABELS = frozenset({"8th(Morales)", "10th(Morales)"})
_PERTURBATION_NOISE_FLOOR = 1e-12
_MINIMIZE_Q_ATOL = 1e-8
_MINIMIZE_KAPPA_ATOL = 1e-6
_STABLE_COEFF_DECIMALS = 12
_CGS_CACHE_SCHEMA_VERSION = 3
_CGS_CACHE_DEFINITION = "hd_surrogate_v1"
_CGS_CACHE_SORT_RULE = (
    "abs_coeff_desc_rounded12_then_original_index_then_pauli_term"
)

KappaMode: TypeAlias = Literal["fixed", "optimize", "sweep"]
RandomizedMethod: TypeAlias = Literal["qdrift", "rte"]


@dataclass(frozen=True)
class RankedPauliTerm:
    """Single non-identity Pauli term sorted by descending absolute coefficient."""

    rank: int
    original_index: int
    pauli_term: tuple[tuple[int, str], ...]
    coeff: float
    abs_coeff: float
    operator: QubitOperator


@dataclass(frozen=True)
class SortedPauliHamiltonian:
    """Hamiltonian metadata plus non-identity Pauli terms in sorted order."""

    molecule_type: int
    distance: float
    ham_name: str
    num_qubits: int
    identity_coeff: float
    sorted_terms: tuple[RankedPauliTerm, ...]

    @property
    def num_terms(self) -> int:
        return len(self.sorted_terms)


@dataclass(frozen=True)
class HamiltonianPartition:
    """Deterministic/randomized split for a single L_D."""

    ld: int
    deterministic_terms: tuple[RankedPauliTerm, ...]
    randomized_terms: tuple[RankedPauliTerm, ...]
    lambda_r: float


@dataclass(frozen=True)
class PerturbationFitResult:
    """Fixed-order perturbative fit used as C_gs^(p)(L_D)."""

    pf_label: PFLabel
    order: int
    ld: int
    t_values: tuple[float, ...]
    perturbation_errors: tuple[float, ...]
    coeff: float
    fit_coeff_fixed_order: float
    fit_slope: float | None
    fit_coeff: float | None


@dataclass(frozen=True)
class ErrorBudgetResult:
    """Optimal error split for one fixed candidate or one fixed (candidate, kappa)."""

    q_ratio: float
    eps_qpe: float
    eps_trot: float
    kappa: float | None
    b_value: float
    boundary_hit_q: bool
    boundary_hit_kappa: bool
    g_det: float
    g_rand: float
    g_total: float


@dataclass(frozen=True)
class KappaSweepPoint:
    """Optimal q result for one fixed kappa in sensitivity mode."""

    kappa: float
    q_opt: float
    eps_qpe_opt: float
    eps_trot_opt: float
    b_value: float
    boundary_hit_q: bool
    g_det: float
    g_rand: float
    g_total: float


@dataclass(frozen=True)
class CandidateResult:
    """Single candidate point in the (p, L_D) scan."""

    pf_label: PFLabel
    order: int
    ld: int
    deterministic_step_cost: int
    lambda_r: float
    c_gs: float
    fit_coeff_fixed_order: float
    fit_slope: float | None
    fit_coeff: float | None
    q_opt: float
    eps_qpe_opt: float
    eps_trot_opt: float
    eps_qpe: float
    eps_trot: float
    kappa_opt: float | None
    b_opt: float
    boundary_hit_kappa: bool
    boundary_hit_q: bool
    randomized_method: str
    g_rand_input: float
    b0: float
    g_det: float
    g_rand: float
    g_total: float
    t_values: tuple[float, ...]
    perturbation_errors: tuple[float, ...]
    kappa_sweep: tuple[KappaSweepPoint, ...] = ()
    legacy_random_prefactor: float | None = None


@dataclass(frozen=True)
class PartialRandomizedStudyResult:
    """Full scan result plus the best candidate."""

    molecule_type: int
    distance: float
    ham_name: str
    num_qubits: int
    epsilon_total: float
    random_prefactor: float | None
    randomized_method: str
    g_rand_input: float
    b0: float
    kappa_mode: str
    kappa_value: float | None
    kappa_min: float
    kappa_max: float
    kappa_grid: tuple[float, ...]
    ld_values: tuple[int, ...]
    pf_labels: tuple[PFLabel, ...]
    total_terms: int
    best: CandidateResult
    candidates: tuple[CandidateResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _real_coeff(coeff: complex, *, atol: float = 1e-10) -> float:
    coeff_complex = complex(coeff)
    if abs(coeff_complex.imag) > atol:
        raise ValueError(f"Pauli coefficient has non-negligible imaginary part: {coeff}")
    return float(coeff_complex.real)


def _sum_operators(terms: Sequence[RankedPauliTerm]) -> QubitOperator:
    op = QubitOperator()
    for term in terms:
        op += term.operator
    return op


def _normalize_randomized_method(method: str) -> RandomizedMethod:
    normalized = method.strip().lower()
    if normalized not in {"qdrift", "rte"}:
        raise ValueError(f"Unsupported randomized method: {method}")
    return normalized  # type: ignore[return-value]


def _normalize_kappa_mode(mode: str) -> KappaMode:
    normalized = mode.strip().lower()
    if normalized not in {"fixed", "optimize", "sweep"}:
        raise ValueError(f"Unsupported kappa mode: {mode}")
    return normalized  # type: ignore[return-value]


def _normalize_q_bounds(q_bounds: tuple[float, float] | None = None) -> tuple[float, float]:
    bounds = (PARTIAL_RANDOMIZED_Q_MIN, PARTIAL_RANDOMIZED_Q_MAX)
    if q_bounds is None:
        return bounds
    q_min, q_max = float(q_bounds[0]), float(q_bounds[1])
    if not (0.0 < q_min < q_max < 1.0):
        raise ValueError(f"q bounds must satisfy 0 < q_min < q_max < 1, got {q_bounds}")
    return q_min, q_max


def _normalize_kappa_bounds(
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
) -> tuple[float, float]:
    kappa_min = float(kappa_min)
    kappa_max = float(kappa_max)
    if kappa_min <= 0.0:
        raise ValueError("kappa_min must be positive.")
    if kappa_max <= kappa_min:
        raise ValueError("kappa_max must be larger than kappa_min.")
    return kappa_min, kappa_max


def _normalize_kappa_grid(kappa_grid: Sequence[float] | None) -> tuple[float, ...]:
    if kappa_grid is None:
        values = PARTIAL_RANDOMIZED_KAPPA_GRID
    else:
        values = tuple(float(value) for value in kappa_grid)
    unique_values = tuple(sorted(set(values)))
    if not unique_values:
        raise ValueError("kappa_grid must not be empty.")
    if unique_values[0] <= 0.0:
        raise ValueError("All kappa grid values must be positive.")
    return unique_values


def _boundary_hit(value: float, lower: float, upper: float) -> bool:
    margin = max(1e-12, PARTIAL_RANDOMIZED_BOUNDARY_REL_TOL * (upper - lower))
    return (value <= lower + margin) or (value >= upper - margin)


def _deterministic_scale(
    order: int,
    deterministic_step_cost_value: int,
    c_gs: float,
) -> float:
    if order <= 0:
        raise ValueError("order must be positive.")
    if deterministic_step_cost_value <= 0 or c_gs <= 0.0:
        return 0.0
    return float(deterministic_step_cost_value * (c_gs ** (1.0 / order)))


def _default_cgs_cache_document() -> dict[str, Any]:
    return {
        "schema_version": _CGS_CACHE_SCHEMA_VERSION,
        "cgs_definition": _CGS_CACHE_DEFINITION,
        "sort_rule": _CGS_CACHE_SORT_RULE,
        "entries": {},
    }


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_coeff(value: float) -> float:
    return float(round(value, _STABLE_COEFF_DECIMALS))


def _sorted_hamiltonian_hash(sorted_hamiltonian: SortedPauliHamiltonian) -> str:
    payload = {
        "molecule_type": sorted_hamiltonian.molecule_type,
        "distance": sorted_hamiltonian.distance,
        "ham_name": sorted_hamiltonian.ham_name,
        "num_qubits": sorted_hamiltonian.num_qubits,
        "sorted_terms": [
            {
                "rank": term.rank,
                "original_index": term.original_index,
                "pauli_term": [[qubit, axis] for qubit, axis in term.pauli_term],
                "coeff": _stable_coeff(term.coeff),
            }
            for term in sorted_hamiltonian.sorted_terms
        ],
    }
    return _json_hash(payload)


def load_cgs_json_cache(
    cache_path: str | Path = PARTIAL_RANDOMIZED_CGS_CACHE_PATH,
) -> dict[str, Any]:
    """Load the JSON cache for perturbative C_gs fits."""
    path = Path(cache_path)
    if not path.exists():
        return _default_cgs_cache_document()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_cgs_cache_document()
    if not isinstance(document, dict):
        return _default_cgs_cache_document()
    if document.get("schema_version") != _CGS_CACHE_SCHEMA_VERSION:
        return _default_cgs_cache_document()
    if not isinstance(document.get("entries"), dict):
        return _default_cgs_cache_document()
    return document


def save_cgs_json_cache(
    cache_document: dict[str, Any],
    cache_path: str | Path = PARTIAL_RANDOMIZED_CGS_CACHE_PATH,
) -> Path:
    """Persist the JSON cache atomically."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(cache_document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _cgs_cache_key_payload(
    *,
    sorted_hamiltonian: SortedPauliHamiltonian,
    sorted_hamiltonian_hash: str,
    pf_label: PFLabel,
    ld: int,
    t_values: Sequence[float],
) -> dict[str, Any]:
    return {
        "cgs_definition": _CGS_CACHE_DEFINITION,
        "sort_rule": _CGS_CACHE_SORT_RULE,
        "sorted_hamiltonian_hash": sorted_hamiltonian_hash,
        "molecule_type": sorted_hamiltonian.molecule_type,
        "distance": sorted_hamiltonian.distance,
        "ham_name": sorted_hamiltonian.ham_name,
        "num_qubits": sorted_hamiltonian.num_qubits,
        "pf_label": pf_label,
        "order": pf_order(pf_label),
        "ld": int(ld),
        "t_values": [float(value) for value in t_values],
        "noise_floor": _PERTURBATION_NOISE_FLOOR,
    }


def _cgs_cache_record(
    *,
    sorted_hamiltonian: SortedPauliHamiltonian,
    sorted_hamiltonian_hash: str,
    fit_result: PerturbationFitResult,
) -> dict[str, Any]:
    return {
        "cgs_definition": _CGS_CACHE_DEFINITION,
        "sort_rule": _CGS_CACHE_SORT_RULE,
        "sorted_hamiltonian_hash": sorted_hamiltonian_hash,
        "molecule_type": sorted_hamiltonian.molecule_type,
        "distance": sorted_hamiltonian.distance,
        "ham_name": sorted_hamiltonian.ham_name,
        "num_qubits": sorted_hamiltonian.num_qubits,
        "pf_label": fit_result.pf_label,
        "order": fit_result.order,
        "ld": fit_result.ld,
        "t_values": [float(value) for value in fit_result.t_values],
        "perturbation_errors": [float(value) for value in fit_result.perturbation_errors],
        "coeff": float(fit_result.coeff),
        "fit_coeff_fixed_order": float(fit_result.fit_coeff_fixed_order),
        "fit_slope": None if fit_result.fit_slope is None else float(fit_result.fit_slope),
        "fit_coeff": None if fit_result.fit_coeff is None else float(fit_result.fit_coeff),
        "noise_floor": _PERTURBATION_NOISE_FLOOR,
    }


def _perturbation_fit_result_from_cache_record(
    record: dict[str, Any],
) -> PerturbationFitResult:
    return PerturbationFitResult(
        pf_label=str(record["pf_label"]),
        order=int(record["order"]),
        ld=int(record["ld"]),
        t_values=tuple(float(value) for value in record.get("t_values", [])),
        perturbation_errors=tuple(
            float(value) for value in record.get("perturbation_errors", [])
        ),
        coeff=float(record["coeff"]),
        fit_coeff_fixed_order=float(record.get("fit_coeff_fixed_order", record["coeff"])),
        fit_slope=(
            None if record.get("fit_slope") is None else float(record["fit_slope"])
        ),
        fit_coeff=(
            None if record.get("fit_coeff") is None else float(record["fit_coeff"])
        ),
    )


def get_or_compute_cached_cgs_fit(
    *,
    sorted_hamiltonian: SortedPauliHamiltonian,
    sorted_hamiltonian_hash: str,
    partition: HamiltonianPartition,
    pf_label: PFLabel,
    cache_document: dict[str, Any],
    cache_path: str | Path = PARTIAL_RANDOMIZED_CGS_CACHE_PATH,
) -> PerturbationFitResult:
    """
    Return a perturbative C_gs fit from the JSON cache or compute and persist it.

    The cache key does not store H_D explicitly. It uses the full sorted Hamiltonian
    hash together with the deterministic prefix length L_D and the fit settings.
    """
    t_values = default_perturbation_t_values(sorted_hamiltonian.molecule_type, pf_label)
    key_payload = _cgs_cache_key_payload(
        sorted_hamiltonian=sorted_hamiltonian,
        sorted_hamiltonian_hash=sorted_hamiltonian_hash,
        pf_label=pf_label,
        ld=partition.ld,
        t_values=t_values,
    )
    cache_key = _json_hash(key_payload)
    entries = cache_document["entries"]
    record = entries.get(cache_key)
    if isinstance(record, dict):
        try:
            return _perturbation_fit_result_from_cache_record(record)
        except (KeyError, TypeError, ValueError):
            entries.pop(cache_key, None)

    fit_result = fit_cgs_with_perturbation(
        sorted_hamiltonian,
        partition,
        pf_label,
        t_values=t_values,
    )
    entries[cache_key] = _cgs_cache_record(
        sorted_hamiltonian=sorted_hamiltonian,
        sorted_hamiltonian_hash=sorted_hamiltonian_hash,
        fit_result=fit_result,
    )
    save_cgs_json_cache(cache_document, cache_path)
    return fit_result


def build_sorted_pauli_hamiltonian(
    molecule_type: int,
    distance: float = 1.0,
) -> SortedPauliHamiltonian:
    """
    Build the JW Hamiltonian and sort non-identity Pauli terms by descending |coeff|.

    The identity term is excluded from ranking/splitting because it only contributes
    a global phase and does not affect L_D or lambda_R.
    """
    jw_hamiltonian, _hf_energy, ham_name, num_qubits = jw_hamiltonian_maker(
        molecule_type, distance
    )
    identity_coeff = 0.0
    ranked_terms: list[RankedPauliTerm] = []

    for original_index, term_operator in enumerate(ham_list_maker(jw_hamiltonian)):
        if len(term_operator.terms) != 1:
            raise ValueError("Expected a single Pauli term per QubitOperator.")
        (pauli_term, coeff_raw), = term_operator.terms.items()
        coeff = _real_coeff(coeff_raw)
        if pauli_term == ():
            identity_coeff += coeff
            continue
        ranked_terms.append(
            RankedPauliTerm(
                rank=-1,
                original_index=original_index,
                pauli_term=tuple(pauli_term),
                coeff=coeff,
                abs_coeff=abs(coeff),
                operator=QubitOperator(pauli_term, coeff),
            )
        )

    ranked_terms.sort(
        key=lambda term: (
            -_stable_coeff(term.abs_coeff),
            term.original_index,
            str(term.pauli_term),
        )
    )
    sorted_terms = tuple(
        RankedPauliTerm(
            rank=rank,
            original_index=term.original_index,
            pauli_term=term.pauli_term,
            coeff=term.coeff,
            abs_coeff=term.abs_coeff,
            operator=term.operator,
        )
        for rank, term in enumerate(ranked_terms, start=1)
    )
    return SortedPauliHamiltonian(
        molecule_type=molecule_type,
        distance=distance,
        ham_name=ham_name,
        num_qubits=num_qubits,
        identity_coeff=identity_coeff,
        sorted_terms=sorted_terms,
    )


def split_hamiltonian_by_ld(
    sorted_hamiltonian: SortedPauliHamiltonian,
    ld: int,
) -> HamiltonianPartition:
    """Split the sorted Pauli list into deterministic H_D and randomized H_R."""
    if ld < 0 or ld > sorted_hamiltonian.num_terms:
        raise ValueError(f"L_D must be in [0, {sorted_hamiltonian.num_terms}], got {ld}")
    deterministic_terms = sorted_hamiltonian.sorted_terms[:ld]
    randomized_terms = sorted_hamiltonian.sorted_terms[ld:]
    lambda_r = float(sum(term.abs_coeff for term in randomized_terms))
    return HamiltonianPartition(
        ld=ld,
        deterministic_terms=deterministic_terms,
        randomized_terms=randomized_terms,
        lambda_r=lambda_r,
    )


def default_perturbation_t_values(
    molecule_type: int,
    pf_label: PFLabel,
    *,
    step: float = PERTURBATION_FIT_STEP,
    span: float = PERTURBATION_FIT_SPAN,
) -> tuple[float, ...]:
    """
    Return the notebook-style time grid used for perturbative C_gs fitting.

    Existing work uses a shorter-time window for lower-order PFs and a later
    window for the Morales higher-order formulas; this function preserves that
    convention.
    """
    if molecule_type not in PERTURBATION_FIT_STARTS:
        raise KeyError(f"No perturbation-fit time window configured for H{molecule_type}")
    mode = 1 if pf_label in _HIGH_ORDER_TIME_MODE_LABELS else 0
    t_start = PERTURBATION_FIT_STARTS[molecule_type][mode]
    num_points = max(2, int(round(span / step)))
    return tuple(round(t_start + step * idx, 10) for idx in range(num_points))


def _collect_perturbation_errors(
    final_state_list: Sequence[tuple[float, Any]],
    energy: float,
    state_vec: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Existing perturbative error proxy, copied verbatim in logic for H_D-only use."""
    state_col = np.asarray(state_vec, dtype=np.complex128).reshape(-1, 1)
    times_out: list[float] = []
    error_list: list[float] = []
    for raw_time, statevector in final_state_list:
        time = -float(raw_time)
        evolved = np.asarray(statevector.data, dtype=np.complex128).reshape(-1, 1)
        phase_factor = np.exp(1j * energy * time)
        delta_state = (evolved - (phase_factor * state_col)) / (1j * time)
        overlap = state_col.conj().T @ delta_state
        overlap = overlap.real / np.cos(energy * time)
        error_list.append(float(np.abs(overlap.real).item()))
        times_out.append(time)
    return tuple(times_out), tuple(error_list)


def fit_cgs_with_perturbation(
    sorted_hamiltonian: SortedPauliHamiltonian,
    partition: HamiltonianPartition,
    pf_label: PFLabel,
    *,
    t_values: Sequence[float] | None = None,
) -> PerturbationFitResult:
    """
    Estimate C_gs^(p)(L_D) from perturbative eigenvalue-error scaling on H_D.

    This intentionally uses the exact eigenstate assumption and fixed-order fit
    from the existing research workflow rather than a rigorous worst-case bound.
    """
    order = pf_order(pf_label)
    if partition.ld == 0:
        return PerturbationFitResult(
            pf_label=pf_label,
            order=order,
            ld=partition.ld,
            t_values=tuple(),
            perturbation_errors=tuple(),
            coeff=0.0,
            fit_coeff_fixed_order=0.0,
            fit_slope=None,
            fit_coeff=None,
        )

    if t_values is None:
        t_values = default_perturbation_t_values(sorted_hamiltonian.molecule_type, pf_label)
    t_values = tuple(float(t) for t in t_values)

    deterministic_operator = _sum_operators(partition.deterministic_terms)
    deterministic_terms = [term.operator for term in partition.deterministic_terms]
    energy, state_vec, _max_eig = ham_ground_energy(
        deterministic_operator,
        n_qubits=sorted_hamiltonian.num_qubits,
        return_max_eig=False,
    )
    state_flat = np.asarray(state_vec, dtype=np.complex128).reshape(-1)

    final_state_list = [
        tEvolution_vector(
            deterministic_terms,
            -time_value,
            sorted_hamiltonian.num_qubits,
            state_flat,
            pf_label,
        )
        for time_value in t_values
    ]
    times_out, perturbation_errors = _collect_perturbation_errors(
        final_state_list,
        float(np.real(energy)),
        np.asarray(state_vec, dtype=np.complex128),
    )

    positive_errors = [err for err in perturbation_errors if err > 0.0]
    if not positive_errors or max(positive_errors) < _PERTURBATION_NOISE_FLOOR:
        return PerturbationFitResult(
            pf_label=pf_label,
            order=order,
            ld=partition.ld,
            t_values=times_out,
            perturbation_errors=perturbation_errors,
            coeff=0.0,
            fit_coeff_fixed_order=0.0,
            fit_slope=None,
            fit_coeff=None,
        )

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

    return PerturbationFitResult(
        pf_label=pf_label,
        order=order,
        ld=partition.ld,
        t_values=times_out,
        perturbation_errors=perturbation_errors,
        coeff=max(0.0, coeff),
        fit_coeff_fixed_order=max(0.0, coeff),
        fit_slope=fit_slope,
        fit_coeff=fit_coeff,
    )


def deterministic_step_cost(num_terms: int, pf_label: PFLabel) -> int:
    """Approximate A_p(L_D) as the number of Pauli evolutions in one PF step on H_D."""
    if num_terms <= 0:
        return 0
    from .product_formula import _get_w_list

    return sum(1 for _ in iter_pf_steps(num_terms, _get_w_list(pf_label)))


def randomized_gamma(method: str) -> float:
    """Return the gamma factor used in the simplified randomized prefactor."""
    normalized = _normalize_randomized_method(method)
    return 1.0 if normalized == "qdrift" else 2.0


def randomized_prefactor_b0(
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
) -> float:
    """Return B0 = (280/9) * G_rand * gamma * (0.1*pi)^2."""
    if g_rand < 0.0:
        raise ValueError("g_rand must be non-negative.")
    gamma = randomized_gamma(randomized_method)
    return float((280.0 / 9.0) * g_rand * gamma * ((0.1 * math.pi) ** 2))


def randomized_prefactor_B(
    kappa: float,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
) -> float:
    """Return B(kappa) = B0 * kappa * exp(2 / kappa)."""
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    exponent = 2.0 / kappa
    if exponent >= 700.0:
        return math.inf
    return float(randomized_prefactor_b0(randomized_method, g_rand) * kappa * math.exp(exponent))


def _total_cost_given_prefactor(
    *,
    epsilon_total: float,
    q_ratio: float,
    order: int,
    deterministic_scale: float,
    lambda_r: float,
    b_value: float,
    kappa: float | None,
    q_bounds: tuple[float, float],
    kappa_bounds: tuple[float, float] | None = None,
) -> ErrorBudgetResult:
    eps_qpe = epsilon_total * q_ratio
    eps_trot = epsilon_total * math.sqrt(max(0.0, 1.0 - (q_ratio * q_ratio)))

    g_det = 0.0
    if deterministic_scale > 0.0:
        if eps_qpe <= 0.0 or eps_trot <= 0.0:
            g_det = math.inf
        else:
            g_det = deterministic_scale / (eps_qpe * (eps_trot ** (1.0 / order)))

    g_rand = 0.0
    if lambda_r > 0.0:
        if eps_qpe <= 0.0:
            g_rand = math.inf
        else:
            g_rand = b_value * (lambda_r**2) / (eps_qpe**2)

    boundary_hit_q = _boundary_hit(q_ratio, q_bounds[0], q_bounds[1])
    boundary_hit_kappa = False
    if kappa is not None and kappa_bounds is not None:
        boundary_hit_kappa = _boundary_hit(kappa, kappa_bounds[0], kappa_bounds[1])

    return ErrorBudgetResult(
        q_ratio=float(q_ratio),
        eps_qpe=float(eps_qpe),
        eps_trot=float(eps_trot),
        kappa=None if kappa is None else float(kappa),
        b_value=float(b_value),
        boundary_hit_q=boundary_hit_q,
        boundary_hit_kappa=boundary_hit_kappa,
        g_det=float(g_det),
        g_rand=float(g_rand),
        g_total=float(g_det + g_rand),
    )


def total_cost_given_q_kappa(
    *,
    epsilon_total: float,
    q_ratio: float,
    order: int,
    deterministic_step_cost_value: int,
    c_gs: float,
    lambda_r: float,
    kappa: float | None,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    q_bounds: tuple[float, float] | None = None,
    kappa_bounds: tuple[float, float] | None = None,
) -> ErrorBudgetResult:
    """
    Evaluate G_total(q, kappa) with B(kappa) for one candidate.

    If lambda_r == 0, the randomized side is turned off explicitly and kappa is
    treated as irrelevant.
    """
    q_bounds_norm = _normalize_q_bounds(q_bounds)
    deterministic_scale = _deterministic_scale(order, deterministic_step_cost_value, c_gs)
    if lambda_r == 0.0:
        return _total_cost_given_prefactor(
            epsilon_total=epsilon_total,
            q_ratio=q_ratio,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=0.0,
            b_value=0.0,
            kappa=None,
            q_bounds=q_bounds_norm,
            kappa_bounds=None,
        )
    if kappa is None:
        raise ValueError("kappa must be provided when lambda_r > 0.")
    b_value = randomized_prefactor_B(
        kappa,
        randomized_method=randomized_method,
        g_rand=g_rand,
    )
    return _total_cost_given_prefactor(
        epsilon_total=epsilon_total,
        q_ratio=q_ratio,
        order=order,
        deterministic_scale=deterministic_scale,
        lambda_r=lambda_r,
        b_value=b_value,
        kappa=kappa,
        q_bounds=q_bounds_norm,
        kappa_bounds=kappa_bounds,
    )


def _optimize_q_for_prefactor(
    *,
    epsilon_total: float,
    order: int,
    deterministic_scale: float,
    lambda_r: float,
    b_value: float,
    q_bounds: tuple[float, float],
    kappa: float | None,
    kappa_bounds: tuple[float, float] | None = None,
) -> ErrorBudgetResult:
    if epsilon_total <= 0.0:
        raise ValueError("epsilon_total must be positive.")

    q_min, q_max = q_bounds
    if deterministic_scale == 0.0 and lambda_r == 0.0:
        q_ratio = float((q_min + q_max) / 2.0)
        return _total_cost_given_prefactor(
            epsilon_total=epsilon_total,
            q_ratio=q_ratio,
            order=order,
            deterministic_scale=0.0,
            lambda_r=0.0,
            b_value=0.0,
            kappa=None,
            q_bounds=q_bounds,
            kappa_bounds=None,
        )

    if deterministic_scale == 0.0:
        return _total_cost_given_prefactor(
            epsilon_total=epsilon_total,
            q_ratio=q_max,
            order=order,
            deterministic_scale=0.0,
            lambda_r=lambda_r,
            b_value=b_value,
            kappa=kappa,
            q_bounds=q_bounds,
            kappa_bounds=kappa_bounds,
        )

    def objective(q_ratio: float) -> float:
        return _total_cost_given_prefactor(
            epsilon_total=epsilon_total,
            q_ratio=q_ratio,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=lambda_r,
            b_value=b_value,
            kappa=kappa,
            q_bounds=q_bounds,
            kappa_bounds=kappa_bounds,
        ).g_total

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=q_bounds,
        options={"xatol": _MINIMIZE_Q_ATOL},
    )
    q_ratio = float(result.x)
    if not result.success:
        grid = np.linspace(q_min, q_max, 4096)
        q_ratio = float(grid[int(np.argmin([objective(float(q)) for q in grid]))])

    return _total_cost_given_prefactor(
        epsilon_total=epsilon_total,
        q_ratio=q_ratio,
        order=order,
        deterministic_scale=deterministic_scale,
        lambda_r=lambda_r,
        b_value=b_value,
        kappa=kappa,
        q_bounds=q_bounds,
        kappa_bounds=kappa_bounds,
    )


def optimize_error_budget(
    *,
    epsilon_total: float,
    order: int,
    deterministic_step_cost_value: int,
    c_gs: float,
    lambda_r: float,
    random_prefactor: float,
) -> ErrorBudgetResult:
    """
    Backward-compatible fixed-B optimizer.

    This preserves the previous interface where the randomized-side prefactor
    was passed directly as a constant B.
    """
    if random_prefactor < 0.0:
        raise ValueError("random_prefactor must be non-negative.")
    q_bounds = _normalize_q_bounds()
    deterministic_scale = _deterministic_scale(order, deterministic_step_cost_value, c_gs)
    if lambda_r == 0.0:
        return _optimize_q_for_prefactor(
            epsilon_total=epsilon_total,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=0.0,
            b_value=0.0,
            q_bounds=q_bounds,
            kappa=None,
            kappa_bounds=None,
        )
    return _optimize_q_for_prefactor(
        epsilon_total=epsilon_total,
        order=order,
        deterministic_scale=deterministic_scale,
        lambda_r=lambda_r,
        b_value=random_prefactor,
        q_bounds=q_bounds,
        kappa=None,
        kappa_bounds=None,
    )


def optimize_error_budget_and_kappa(
    *,
    epsilon_total: float,
    order: int,
    deterministic_step_cost_value: int,
    c_gs: float,
    lambda_r: float,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    q_bounds: tuple[float, float] | None = None,
) -> ErrorBudgetResult:
    """Optimize q and, when requested, kappa for one fixed (p, L_D) candidate."""
    q_bounds_norm = _normalize_q_bounds(q_bounds)
    kappa_mode_norm = _normalize_kappa_mode(kappa_mode)
    randomized_method_norm = _normalize_randomized_method(randomized_method)
    kappa_bounds = _normalize_kappa_bounds(kappa_min, kappa_max)
    deterministic_scale = _deterministic_scale(order, deterministic_step_cost_value, c_gs)

    if lambda_r == 0.0:
        return _optimize_q_for_prefactor(
            epsilon_total=epsilon_total,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=0.0,
            b_value=0.0,
            q_bounds=q_bounds_norm,
            kappa=None,
            kappa_bounds=None,
        )

    if kappa_mode_norm == "fixed":
        if not (kappa_bounds[0] <= kappa_value <= kappa_bounds[1]):
            raise ValueError(
                f"kappa_value must lie in [{kappa_bounds[0]}, {kappa_bounds[1]}], "
                f"got {kappa_value}"
            )
        b_value = randomized_prefactor_B(
            float(kappa_value),
            randomized_method=randomized_method_norm,
            g_rand=g_rand,
        )
        return _optimize_q_for_prefactor(
            epsilon_total=epsilon_total,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=lambda_r,
            b_value=b_value,
            q_bounds=q_bounds_norm,
            kappa=float(kappa_value),
            kappa_bounds=kappa_bounds,
        )

    def objective(kappa: float) -> float:
        b_value = randomized_prefactor_B(
            float(kappa),
            randomized_method=randomized_method_norm,
            g_rand=g_rand,
        )
        return _optimize_q_for_prefactor(
            epsilon_total=epsilon_total,
            order=order,
            deterministic_scale=deterministic_scale,
            lambda_r=lambda_r,
            b_value=b_value,
            q_bounds=q_bounds_norm,
            kappa=float(kappa),
            kappa_bounds=kappa_bounds,
        ).g_total

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=kappa_bounds,
        options={"xatol": _MINIMIZE_KAPPA_ATOL},
    )
    kappa_opt = float(result.x)
    if not result.success:
        grid = np.geomspace(kappa_bounds[0], kappa_bounds[1], 512)
        kappa_opt = float(grid[int(np.argmin([objective(float(k)) for k in grid]))])
    b_value = randomized_prefactor_B(
        kappa_opt,
        randomized_method=randomized_method_norm,
        g_rand=g_rand,
    )
    return _optimize_q_for_prefactor(
        epsilon_total=epsilon_total,
        order=order,
        deterministic_scale=deterministic_scale,
        lambda_r=lambda_r,
        b_value=b_value,
        q_bounds=q_bounds_norm,
        kappa=kappa_opt,
        kappa_bounds=kappa_bounds,
    )


def sweep_kappa_for_candidate(
    *,
    epsilon_total: float,
    order: int,
    deterministic_step_cost_value: int,
    c_gs: float,
    lambda_r: float,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    kappa_grid: Sequence[float] | None = None,
    q_bounds: tuple[float, float] | None = None,
) -> tuple[KappaSweepPoint, ...]:
    """Run sensitivity analysis over a user-specified kappa grid for one candidate."""
    q_bounds_norm = _normalize_q_bounds(q_bounds)
    kappa_grid_norm = _normalize_kappa_grid(kappa_grid)
    randomized_method_norm = _normalize_randomized_method(randomized_method)
    deterministic_scale = _deterministic_scale(order, deterministic_step_cost_value, c_gs)

    points: list[KappaSweepPoint] = []
    for kappa in kappa_grid_norm:
        if lambda_r == 0.0:
            budget = _optimize_q_for_prefactor(
                epsilon_total=epsilon_total,
                order=order,
                deterministic_scale=deterministic_scale,
                lambda_r=0.0,
                b_value=0.0,
                q_bounds=q_bounds_norm,
                kappa=None,
                kappa_bounds=None,
            )
            b_value = 0.0
        else:
            b_value = randomized_prefactor_B(
                kappa,
                randomized_method=randomized_method_norm,
                g_rand=g_rand,
            )
            budget = _optimize_q_for_prefactor(
                epsilon_total=epsilon_total,
                order=order,
                deterministic_scale=deterministic_scale,
                lambda_r=lambda_r,
                b_value=b_value,
                q_bounds=q_bounds_norm,
                kappa=kappa,
                kappa_bounds=(kappa_grid_norm[0], kappa_grid_norm[-1]),
            )
        points.append(
            KappaSweepPoint(
                kappa=float(kappa),
                q_opt=budget.q_ratio,
                eps_qpe_opt=budget.eps_qpe,
                eps_trot_opt=budget.eps_trot,
                b_value=float(b_value),
                boundary_hit_q=budget.boundary_hit_q,
                g_det=budget.g_det,
                g_rand=budget.g_rand,
                g_total=budget.g_total,
            )
        )
    return tuple(points)


def _normalize_pf_labels(pf_labels: Sequence[PFLabel] | None) -> tuple[PFLabel, ...]:
    labels = DEFAULT_PF_LABELS if pf_labels is None else tuple(pf_labels)
    unknown = [label for label in labels if label not in P_DIR]
    if unknown:
        raise KeyError(f"Unsupported PF labels: {unknown}")
    return tuple(labels)


def _normalize_ld_values(
    total_terms: int,
    ld_values: Sequence[int] | None,
    *,
    ld_step: int = 1,
) -> tuple[int, ...]:
    if ld_step <= 0:
        raise ValueError("ld_step must be positive.")
    if ld_values is None:
        values = list(range(0, total_terms + 1, ld_step))
        if values[-1] != total_terms:
            values.append(total_terms)
    else:
        values = sorted(set(int(ld) for ld in ld_values))
    if not values:
        raise ValueError("No L_D values were provided.")
    if values[0] < 0 or values[-1] > total_terms:
        raise ValueError(f"L_D values must lie in [0, {total_terms}].")
    return tuple(values)


def analyze_partial_randomized_pf(
    molecule_type: int,
    *,
    epsilon_total: float,
    distance: float = 1.0,
    pf_labels: Sequence[PFLabel] | None = None,
    ld_values: Sequence[int] | None = None,
    ld_step: int = 1,
    random_prefactor: float | None = None,
    kappa_mode: str = "optimize",
    kappa_value: float = PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    kappa_min: float = PARTIAL_RANDOMIZED_KAPPA_MIN,
    kappa_max: float = PARTIAL_RANDOMIZED_KAPPA_MAX,
    kappa_grid: Sequence[float] | None = None,
    randomized_method: str = PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    g_rand: float = PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    q_bounds: tuple[float, float] | None = None,
) -> PartialRandomizedStudyResult:
    """
    Scan PF label and L_D for the simplified partially randomized PF cost model.

    Main approximation choices in this first-stage implementation:
    - H is handled as a Pauli LCU and split by descending |coeff|.
    - A_p(L_D) is approximated by the number of Pauli evolutions in one PF step.
    - C_gs^(p)(L_D) is a surrogate obtained from perturbative fits on H_D only.
    - In the main path, the randomized prefactor is B(kappa) = B0*kappa*exp(2/kappa).

    Backward compatibility:
    - If random_prefactor is given, the legacy fixed-B path is used instead of
      the kappa-derived prefactor.
    """
    sorted_hamiltonian = build_sorted_pauli_hamiltonian(molecule_type, distance)
    pf_labels_norm = _normalize_pf_labels(pf_labels)
    ld_values_norm = _normalize_ld_values(
        sorted_hamiltonian.num_terms,
        ld_values,
        ld_step=ld_step,
    )
    q_bounds_norm = _normalize_q_bounds(q_bounds)
    kappa_mode_norm = _normalize_kappa_mode(kappa_mode)
    kappa_bounds = _normalize_kappa_bounds(kappa_min, kappa_max)
    kappa_grid_norm = _normalize_kappa_grid(kappa_grid)
    randomized_method_norm = _normalize_randomized_method(randomized_method)
    b0 = randomized_prefactor_b0(randomized_method_norm, g_rand)
    sorted_hamiltonian_hash = _sorted_hamiltonian_hash(sorted_hamiltonian)
    cgs_cache_document = load_cgs_json_cache()

    fit_cache: dict[tuple[PFLabel, int], PerturbationFitResult] = {}
    candidate_results: list[CandidateResult] = []

    for ld in ld_values_norm:
        partition = split_hamiltonian_by_ld(sorted_hamiltonian, ld)
        for pf_label in pf_labels_norm:
            fit_key = (pf_label, ld)
            if fit_key not in fit_cache:
                fit_cache[fit_key] = get_or_compute_cached_cgs_fit(
                    sorted_hamiltonian=sorted_hamiltonian,
                    sorted_hamiltonian_hash=sorted_hamiltonian_hash,
                    partition=partition,
                    pf_label=pf_label,
                    cache_document=cgs_cache_document,
                )
            fit_result = fit_cache[fit_key]
            order = pf_order(pf_label)
            step_cost = deterministic_step_cost(ld, pf_label)
            kappa_sweep_points: tuple[KappaSweepPoint, ...] = tuple()

            if random_prefactor is not None:
                budget = optimize_error_budget(
                    epsilon_total=epsilon_total,
                    order=order,
                    deterministic_step_cost_value=step_cost,
                    c_gs=fit_result.coeff,
                    lambda_r=partition.lambda_r,
                    random_prefactor=random_prefactor,
                )
            elif kappa_mode_norm == "sweep":
                kappa_sweep_points = sweep_kappa_for_candidate(
                    epsilon_total=epsilon_total,
                    order=order,
                    deterministic_step_cost_value=step_cost,
                    c_gs=fit_result.coeff,
                    lambda_r=partition.lambda_r,
                    randomized_method=randomized_method_norm,
                    g_rand=g_rand,
                    kappa_grid=kappa_grid_norm,
                    q_bounds=q_bounds_norm,
                )
                best_sweep_point = min(kappa_sweep_points, key=lambda point: point.g_total)
                if partition.lambda_r == 0.0:
                    budget = ErrorBudgetResult(
                        q_ratio=best_sweep_point.q_opt,
                        eps_qpe=best_sweep_point.eps_qpe_opt,
                        eps_trot=best_sweep_point.eps_trot_opt,
                        kappa=None,
                        b_value=0.0,
                        boundary_hit_q=best_sweep_point.boundary_hit_q,
                        boundary_hit_kappa=False,
                        g_det=best_sweep_point.g_det,
                        g_rand=best_sweep_point.g_rand,
                        g_total=best_sweep_point.g_total,
                    )
                else:
                    budget = ErrorBudgetResult(
                        q_ratio=best_sweep_point.q_opt,
                        eps_qpe=best_sweep_point.eps_qpe_opt,
                        eps_trot=best_sweep_point.eps_trot_opt,
                        kappa=best_sweep_point.kappa,
                        b_value=best_sweep_point.b_value,
                        boundary_hit_q=best_sweep_point.boundary_hit_q,
                        boundary_hit_kappa=_boundary_hit(
                            best_sweep_point.kappa,
                            kappa_grid_norm[0],
                            kappa_grid_norm[-1],
                        ),
                        g_det=best_sweep_point.g_det,
                        g_rand=best_sweep_point.g_rand,
                        g_total=best_sweep_point.g_total,
                    )
            else:
                budget = optimize_error_budget_and_kappa(
                    epsilon_total=epsilon_total,
                    order=order,
                    deterministic_step_cost_value=step_cost,
                    c_gs=fit_result.coeff,
                    lambda_r=partition.lambda_r,
                    randomized_method=randomized_method_norm,
                    g_rand=g_rand,
                    kappa_mode=kappa_mode_norm,
                    kappa_value=kappa_value,
                    kappa_min=kappa_bounds[0],
                    kappa_max=kappa_bounds[1],
                    q_bounds=q_bounds_norm,
                )

            candidate_results.append(
                CandidateResult(
                    pf_label=pf_label,
                    order=order,
                    ld=ld,
                    deterministic_step_cost=step_cost,
                    lambda_r=partition.lambda_r,
                    c_gs=fit_result.coeff,
                    fit_coeff_fixed_order=fit_result.fit_coeff_fixed_order,
                    fit_slope=fit_result.fit_slope,
                    fit_coeff=fit_result.fit_coeff,
                    q_opt=budget.q_ratio,
                    eps_qpe_opt=budget.eps_qpe,
                    eps_trot_opt=budget.eps_trot,
                    eps_qpe=budget.eps_qpe,
                    eps_trot=budget.eps_trot,
                    kappa_opt=budget.kappa,
                    b_opt=budget.b_value,
                    boundary_hit_kappa=budget.boundary_hit_kappa,
                    boundary_hit_q=budget.boundary_hit_q,
                    randomized_method=randomized_method_norm,
                    g_rand_input=float(g_rand),
                    b0=float(b0),
                    g_det=budget.g_det,
                    g_rand=budget.g_rand,
                    g_total=budget.g_total,
                    t_values=fit_result.t_values,
                    perturbation_errors=fit_result.perturbation_errors,
                    kappa_sweep=kappa_sweep_points,
                    legacy_random_prefactor=random_prefactor,
                )
            )

    best_result = min(candidate_results, key=lambda result: result.g_total)
    return PartialRandomizedStudyResult(
        molecule_type=molecule_type,
        distance=distance,
        ham_name=sorted_hamiltonian.ham_name,
        num_qubits=sorted_hamiltonian.num_qubits,
        epsilon_total=epsilon_total,
        random_prefactor=random_prefactor,
        randomized_method=randomized_method_norm,
        g_rand_input=float(g_rand),
        b0=float(b0),
        kappa_mode=kappa_mode_norm,
        kappa_value=None if random_prefactor is not None else float(kappa_value),
        kappa_min=kappa_bounds[0],
        kappa_max=kappa_bounds[1],
        kappa_grid=kappa_grid_norm,
        ld_values=ld_values_norm,
        pf_labels=pf_labels_norm,
        total_terms=sorted_hamiltonian.num_terms,
        best=best_result,
        candidates=tuple(candidate_results),
    )


def kappa_sweep_rows(result: PartialRandomizedStudyResult) -> list[dict[str, Any]]:
    """Flatten sweep-mode results into CSV-friendly rows."""
    rows: list[dict[str, Any]] = []
    for candidate in result.candidates:
        if not candidate.kappa_sweep:
            continue
        for point in candidate.kappa_sweep:
            rows.append(
                {
                    "molecule_type": result.molecule_type,
                    "ham_name": result.ham_name,
                    "epsilon_total": result.epsilon_total,
                    "pf_label": candidate.pf_label,
                    "order": candidate.order,
                    "ld": candidate.ld,
                    "lambda_r": candidate.lambda_r,
                    "c_gs": candidate.c_gs,
                    "randomized_method": candidate.randomized_method,
                    "g_rand_input": candidate.g_rand_input,
                    "b0": candidate.b0,
                    "kappa": point.kappa,
                    "b_value": point.b_value,
                    "q_opt": point.q_opt,
                    "eps_qpe_opt": point.eps_qpe_opt,
                    "eps_trot_opt": point.eps_trot_opt,
                    "boundary_hit_q": point.boundary_hit_q,
                    "g_det": point.g_det,
                    "g_rand": point.g_rand,
                    "g_total": point.g_total,
                }
            )
    return rows


def save_partial_randomized_result(
    result: PartialRandomizedStudyResult,
    output_path: str | Path,
) -> Path:
    """Save the scan result as JSON for later analysis."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def save_kappa_sweep_csv(
    result: PartialRandomizedStudyResult,
    output_path: str | Path,
) -> Path:
    """Save sweep-mode rows as CSV."""
    rows = kappa_sweep_rows(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "molecule_type",
        "ham_name",
        "epsilon_total",
        "pf_label",
        "order",
        "ld",
        "lambda_r",
        "c_gs",
        "randomized_method",
        "g_rand_input",
        "b0",
        "kappa",
        "b_value",
        "q_opt",
        "eps_qpe_opt",
        "eps_trot_opt",
        "boundary_hit_q",
        "g_det",
        "g_rand",
        "g_total",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
