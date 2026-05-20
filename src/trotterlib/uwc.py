from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
from openfermion import count_qubits
from openfermion.linalg import get_sparse_operator, jw_number_indices
from openfermion.ops import QubitOperator
from scipy.optimize import minimize_scalar


_STABLE_COEFF_DECIMALS = 12
_ZERO_TOL = 1e-12
_UWC_METHODS = frozenset(
    {
        "none",
        "simple_shift",
        "test_shift",
        "bliss",
        "orbital_optimization",
        "orbital_optimization_bliss",
    }
)
_UWC_OBJECTIVE_ALIASES = {
    "l1": "l1_norm",
    "l1_norm": "l1_norm",
    "hamiltonian_l1_norm": "l1_norm",
    "lambda_r": "lambda_r",
    "randomized_lambda": "lambda_r",
    "estimated_total_cost": "estimated_total_cost",
    "estimated_g_total": "estimated_total_cost",
    "total_pr_pf_cost": "estimated_total_cost",
}


@dataclass(frozen=True)
class UWCConfig:
    """Configuration for Hamiltonian preprocessing by UWC-style transforms."""

    enabled: bool = False
    method: str = "none"
    objective: str = "l1_norm"
    target_ld: int | None = None
    optimizer_settings: Mapping[str, Any] = field(default_factory=dict)
    max_iterations: int = 0
    seed: int | None = None
    use_cache: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)
    sector_energy_tolerance: float = 1e-8
    sector_energy_check: str = "warn"
    max_sector_dimension_for_check: int = 256


@dataclass(frozen=True)
class UWCPreprocessingResult:
    """Raw and processed Hamiltonians plus comparison metadata."""

    original_hamiltonian: QubitOperator
    hamiltonian: QubitOperator
    config: UWCConfig
    metadata: dict[str, Any]
    warnings: tuple[str, ...] = ()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_coeff(value: float) -> float:
    return float(round(float(value), _STABLE_COEFF_DECIMALS))


def _real_coeff(coeff: complex, *, atol: float = 1e-10) -> float:
    coeff_complex = complex(coeff)
    if abs(coeff_complex.imag) > atol:
        raise ValueError(f"Pauli coefficient has non-negligible imaginary part: {coeff}")
    return float(coeff_complex.real)


def _copy_qubit_operator(operator: QubitOperator) -> QubitOperator:
    copied = QubitOperator()
    for pauli_term, coeff in operator.terms.items():
        copied += QubitOperator(pauli_term, coeff)
    copied.compress(abs_tol=_ZERO_TOL)
    return copied


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _UWC_METHODS:
        raise ValueError(f"Unsupported UWC method: {method}")
    return normalized


def _normalize_objective(objective: str) -> str:
    normalized = str(objective).strip().lower()
    try:
        return _UWC_OBJECTIVE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported UWC objective: {objective}") from exc


def _normalize_sector_energy_check(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"warn", "error", "off"}:
        raise ValueError("sector_energy_check must be one of 'warn', 'error', or 'off'.")
    return normalized


def normalize_uwc_config(config: UWCConfig | Mapping[str, Any] | None) -> UWCConfig:
    """Return a validated UWCConfig with normalized method/objective names."""

    if config is None:
        config = UWCConfig()
    elif isinstance(config, Mapping):
        config = UWCConfig(**dict(config))

    if not isinstance(config, UWCConfig):
        raise TypeError("uwc_config must be a UWCConfig, mapping, or None.")

    method = _normalize_method(config.method)
    enabled = bool(config.enabled) and method != "none"
    if not enabled:
        method = "none"
    objective = _normalize_objective(config.objective)

    target_ld = None if config.target_ld is None else int(config.target_ld)
    if target_ld is not None and target_ld < 0:
        raise ValueError("target_ld must be non-negative when provided.")
    max_iterations = int(config.max_iterations)
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative.")
    sector_energy_tolerance = float(config.sector_energy_tolerance)
    if sector_energy_tolerance <= 0.0:
        raise ValueError("sector_energy_tolerance must be positive.")
    max_sector_dimension_for_check = int(config.max_sector_dimension_for_check)
    if max_sector_dimension_for_check < 0:
        raise ValueError("max_sector_dimension_for_check must be non-negative.")

    return replace(
        config,
        enabled=enabled,
        method=method,
        objective=objective,
        target_ld=target_ld,
        optimizer_settings=dict(config.optimizer_settings),
        max_iterations=max_iterations,
        parameters=dict(config.parameters),
        sector_energy_tolerance=sector_energy_tolerance,
        sector_energy_check=_normalize_sector_energy_check(config.sector_energy_check),
        max_sector_dimension_for_check=max_sector_dimension_for_check,
    )


def canonical_qubit_operator_terms(
    hamiltonian: QubitOperator,
    *,
    include_identity: bool = True,
) -> list[dict[str, Any]]:
    """Return a deterministic JSON-friendly representation of Pauli terms."""

    terms: list[dict[str, Any]] = []
    for pauli_term, coeff_raw in hamiltonian.terms.items():
        if pauli_term == () and not include_identity:
            continue
        coeff = _real_coeff(coeff_raw)
        if abs(coeff) <= _ZERO_TOL:
            continue
        normalized_term = tuple((int(qubit), str(axis)) for qubit, axis in pauli_term)
        terms.append(
            {
                "pauli_term": [[qubit, axis] for qubit, axis in normalized_term],
                "coeff": _stable_coeff(coeff),
            }
        )
    terms.sort(key=lambda item: (item["pauli_term"], item["coeff"]))
    return terms


def qubit_hamiltonian_hash(
    hamiltonian: QubitOperator,
    *,
    n_qubits: int | None = None,
) -> str:
    """
    Hash the full QubitOperator term list, including identity coefficients.

    This metadata hash is intentionally independent from the C_gs sorted-prefix
    cache hash used by the PR-PF model.
    """

    if n_qubits is None:
        n_qubits = count_qubits(hamiltonian)
    payload = {
        "hash_type": "full_qubit_operator_v1",
        "num_qubits": int(n_qubits),
        "terms": canonical_qubit_operator_terms(hamiltonian, include_identity=True),
    }
    return _json_hash(payload)


def qubit_l1_norm(hamiltonian: QubitOperator) -> float:
    """Return the non-identity Pauli coefficient L1 norm used by this pipeline."""

    return float(
        sum(
            abs(_real_coeff(coeff))
            for pauli_term, coeff in hamiltonian.terms.items()
            if pauli_term != () and abs(_real_coeff(coeff)) > _ZERO_TOL
        )
    )


def qubit_num_terms(hamiltonian: QubitOperator) -> int:
    """Count non-identity Pauli terms with nonzero coefficients."""

    return sum(
        1
        for pauli_term, coeff in hamiltonian.terms.items()
        if pauli_term != () and abs(_real_coeff(coeff)) > _ZERO_TOL
    )


def lambda_r_from_qubit_operator(hamiltonian: QubitOperator, target_ld: int) -> float:
    """Compute lambda_R after selecting the target_ld largest |coeff| terms."""

    target_ld = int(target_ld)
    if target_ld < 0:
        raise ValueError("target_ld must be non-negative.")
    weights = sorted(
        (
            abs(_real_coeff(coeff))
            for pauli_term, coeff in hamiltonian.terms.items()
            if pauli_term != () and abs(_real_coeff(coeff)) > _ZERO_TOL
        ),
        reverse=True,
    )
    if target_ld > len(weights):
        raise ValueError(f"target_ld must be in [0, {len(weights)}], got {target_ld}")
    return float(sum(weights[target_ld:]))


def number_operator(n_qubits: int) -> QubitOperator:
    """Return the Jordan-Wigner particle-number operator N."""

    n_qubits = int(n_qubits)
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative.")
    operator = QubitOperator((), 0.5 * n_qubits)
    for qubit in range(n_qubits):
        operator += QubitOperator(((qubit, "Z"),), -0.5)
    operator.compress(abs_tol=_ZERO_TOL)
    return operator


def number_sector_shift_operator(
    *,
    n_qubits: int,
    target_particle_number: int,
    theta: float,
    power: int = 1,
) -> QubitOperator:
    """Return theta * (N - N_target)^power for a number-sector BLISS shift."""

    if power not in {1, 2}:
        raise ValueError("number-sector BLISS shift power must be 1 or 2.")
    base = number_operator(n_qubits) + QubitOperator((), -float(target_particle_number))
    shift = base if power == 1 else base * base
    shift = float(theta) * shift
    shift.compress(abs_tol=_ZERO_TOL)
    return shift


def _scale_nonidentity_terms(hamiltonian: QubitOperator, scale: float) -> QubitOperator:
    transformed = QubitOperator()
    for pauli_term, coeff in hamiltonian.terms.items():
        factor = 1.0 if pauli_term == () else float(scale)
        transformed += QubitOperator(pauli_term, coeff * factor)
    transformed.compress(abs_tol=_ZERO_TOL)
    return transformed


def _apply_simple_shift(
    hamiltonian: QubitOperator,
    *,
    n_qubits: int,
    parameters: Mapping[str, Any],
) -> tuple[QubitOperator, dict[str, Any]]:
    transformed = _copy_qubit_operator(hamiltonian)
    applied: dict[str, Any] = {}

    if "coefficient_scale" in parameters:
        scale = float(parameters["coefficient_scale"])
        transformed = _scale_nonidentity_terms(transformed, scale)
        applied["coefficient_scale"] = scale

    default_shift = 0.0 if "coefficient_scale" in parameters else 1e-3
    shift = float(parameters.get("shift", parameters.get("z_shift", default_shift)))
    if shift != 0.0:
        qubit = int(parameters.get("qubit", 0))
        if not (0 <= qubit < int(n_qubits)):
            raise ValueError(f"simple_shift qubit must lie in [0, {n_qubits}).")
        transformed += QubitOperator(((qubit, "Z"),), shift)
        applied["shift"] = shift
        applied["qubit"] = qubit

    transformed.compress(abs_tol=_ZERO_TOL)
    return transformed, applied


def _target_particle_number(parameters: Mapping[str, Any]) -> int:
    if "target_particle_number" in parameters:
        return int(parameters["target_particle_number"])
    if "n_electrons" in parameters:
        return int(parameters["n_electrons"])
    raise ValueError(
        "BLISS UWC requires parameters['target_particle_number'] "
        "or parameters['n_electrons']."
    )


def _bliss_power(parameters: Mapping[str, Any]) -> int:
    raw = parameters.get("power", parameters.get("bliss_power", 1))
    if isinstance(raw, str):
        raw = raw.strip().lower()
        if raw == "linear":
            return 1
        if raw == "quadratic":
            return 2
    return int(raw)


def _theta_bounds(settings: Mapping[str, Any]) -> tuple[float, float]:
    raw = settings.get("theta_bounds", settings.get("bounds", (-1.0, 1.0)))
    if not isinstance(raw, SequenceABC) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise ValueError("optimizer_settings['theta_bounds'] must contain two values.")
    lower = float(raw[0])
    upper = float(raw[1])
    if not lower < upper:
        raise ValueError("theta_bounds must satisfy lower < upper.")
    return lower, upper


def _optimize_bliss_theta(
    hamiltonian: QubitOperator,
    *,
    n_qubits: int,
    target_particle_number: int,
    power: int,
    objective: str,
    target_ld: int,
    optimizer_settings: Mapping[str, Any],
    max_iterations: int,
) -> float:
    if objective == "estimated_total_cost":
        raise NotImplementedError(
            "UWC objective='estimated_total_cost' is reserved for a later PR-PF "
            "cost-model integration."
        )
    bounds = _theta_bounds(optimizer_settings)

    def value(theta: float) -> float:
        shifted = hamiltonian + number_sector_shift_operator(
            n_qubits=n_qubits,
            target_particle_number=target_particle_number,
            theta=float(theta),
            power=power,
        )
        shifted.compress(abs_tol=_ZERO_TOL)
        if objective == "l1_norm":
            return qubit_l1_norm(shifted)
        if objective == "lambda_r":
            return lambda_r_from_qubit_operator(shifted, target_ld)
        raise ValueError(f"Unsupported UWC objective: {objective}")

    options: dict[str, Any] = {"xatol": float(optimizer_settings.get("xatol", 1e-8))}
    if max_iterations > 0:
        options["maxiter"] = max_iterations
    result = minimize_scalar(value, method="bounded", bounds=bounds, options=options)
    if result.success:
        return float(result.x)

    grid_size = int(optimizer_settings.get("fallback_grid_size", 129))
    grid_size = max(3, grid_size)
    grid = np.linspace(bounds[0], bounds[1], grid_size)
    values = [value(float(theta)) for theta in grid]
    return float(grid[int(np.argmin(values))])


def _basis_indices_for_number_sector(n_qubits: int, target_particle_number: int) -> np.ndarray:
    indices = np.asarray(
        jw_number_indices(int(target_particle_number), int(n_qubits)),
        dtype=np.int64,
    )
    indices.sort()
    return indices


def _diagonal_pauli_value(
    pauli_term: tuple[tuple[int, str], ...],
    basis_index: int,
    *,
    n_qubits: int,
) -> float | None:
    value = 1.0
    for qubit, axis in pauli_term:
        if axis != "Z":
            return None
        bit = (int(basis_index) >> (int(n_qubits) - 1 - int(qubit))) & 1
        value *= 1.0 if bit == 0 else -1.0
    return value


def _max_abs_diagonal_on_basis(
    operator: QubitOperator,
    *,
    n_qubits: int,
    basis_indices: np.ndarray,
) -> float:
    max_abs = 0.0
    for basis_index in np.asarray(basis_indices, dtype=np.int64):
        diagonal = 0.0
        for pauli_term, coeff in operator.terms.items():
            diagonal_value = _diagonal_pauli_value(pauli_term, int(basis_index), n_qubits=n_qubits)
            if diagonal_value is None:
                return float("inf")
            diagonal += _real_coeff(coeff) * diagonal_value
        max_abs = max(max_abs, abs(diagonal))
    return float(max_abs)


def _sector_ground_energy(
    hamiltonian: QubitOperator,
    *,
    n_qubits: int,
    basis_indices: np.ndarray,
) -> float:
    basis_indices = np.asarray(basis_indices, dtype=np.int64)
    if basis_indices.size == 0:
        raise ValueError("Cannot check ground energy in an empty physical sector.")
    sparse = get_sparse_operator(hamiltonian, n_qubits=int(n_qubits)).tocsc()
    sector_matrix = sparse[basis_indices, :][:, basis_indices].toarray()
    evals = np.linalg.eigvalsh(np.asarray(sector_matrix, dtype=np.complex128))
    return float(np.min(np.real_if_close(evals)))


def _handle_sector_check_failure(config: UWCConfig, message: str) -> None:
    if config.sector_energy_check == "error":
        raise ValueError(message)
    if config.sector_energy_check == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _check_number_sector_preservation(
    original: QubitOperator,
    processed: QubitOperator,
    shift: QubitOperator,
    *,
    n_qubits: int,
    target_particle_number: int,
    config: UWCConfig,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if config.sector_energy_check == "off":
        return {"checked": False, "reason": "disabled"}, tuple()

    if target_particle_number < 0 or target_particle_number > n_qubits:
        raise ValueError(
            "target_particle_number must lie in "
            f"[0, {n_qubits}], got {target_particle_number}."
        )
    sector_dimension = math.comb(int(n_qubits), int(target_particle_number))
    check: dict[str, Any] = {
        "checked": True,
        "sector_type": "number",
        "target_particle_number": int(target_particle_number),
        "sector_dimension": int(sector_dimension),
        "tolerance": float(config.sector_energy_tolerance),
    }
    emitted_warnings: list[str] = []

    if sector_dimension > config.max_sector_dimension_for_check:
        check.update(
            {
                "max_abs_shift_on_sector": 0.0,
                "ground_energy_checked": True,
                "ground_energy_check_method": "algebraic_number_sector_zero_shift",
                "ground_energy_difference_bound": 0.0,
            }
        )
        return check, tuple()

    basis_indices = _basis_indices_for_number_sector(n_qubits, target_particle_number)
    max_abs_shift = _max_abs_diagonal_on_basis(
        shift,
        n_qubits=n_qubits,
        basis_indices=basis_indices,
    )
    check["max_abs_shift_on_sector"] = max_abs_shift
    if max_abs_shift > config.sector_energy_tolerance:
        message = (
            "BLISS shift is not zero on the target number sector: "
            f"max_abs_shift={max_abs_shift:.3e}, "
            f"tol={config.sector_energy_tolerance:.3e}."
        )
        emitted_warnings.append(message)
        _handle_sector_check_failure(config, message)

    if basis_indices.size <= config.max_sector_dimension_for_check:
        original_energy = _sector_ground_energy(
            original,
            n_qubits=n_qubits,
            basis_indices=basis_indices,
        )
        processed_energy = _sector_ground_energy(
            processed,
            n_qubits=n_qubits,
            basis_indices=basis_indices,
        )
        difference = abs(processed_energy - original_energy)
        check.update(
            {
                "ground_energy_checked": True,
                "ground_energy_check_method": "explicit_diagonalization",
                "original_ground_energy": original_energy,
                "uwc_ground_energy": processed_energy,
                "ground_energy_difference": difference,
            }
        )
        if difference > config.sector_energy_tolerance:
            message = (
                "BLISS shift changed the target-sector ground energy: "
                f"|delta|={difference:.3e}, tol={config.sector_energy_tolerance:.3e}."
            )
            emitted_warnings.append(message)
            _handle_sector_check_failure(config, message)
    return check, tuple(emitted_warnings)


def _apply_bliss_shift(
    hamiltonian: QubitOperator,
    *,
    n_qubits: int,
    config: UWCConfig,
    target_ld: int,
) -> tuple[QubitOperator, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    parameters = dict(config.parameters)
    target_particle_number = _target_particle_number(parameters)
    power = _bliss_power(parameters)
    theta_raw = parameters.get("theta", None)
    if theta_raw is None and config.max_iterations > 0:
        theta = _optimize_bliss_theta(
            hamiltonian,
            n_qubits=n_qubits,
            target_particle_number=target_particle_number,
            power=power,
            objective=config.objective,
            target_ld=target_ld,
            optimizer_settings=config.optimizer_settings,
            max_iterations=config.max_iterations,
        )
        optimized = True
    else:
        theta = 0.0 if theta_raw is None else float(theta_raw)
        optimized = False

    shift = number_sector_shift_operator(
        n_qubits=n_qubits,
        target_particle_number=target_particle_number,
        theta=theta,
        power=power,
    )
    processed = hamiltonian + shift
    processed.compress(abs_tol=_ZERO_TOL)

    check, emitted_warnings = _check_number_sector_preservation(
        hamiltonian,
        processed,
        shift,
        n_qubits=n_qubits,
        target_particle_number=target_particle_number,
        config=config,
    )
    applied = {
        "shift_type": "number_sector",
        "target_particle_number": int(target_particle_number),
        "theta": float(theta),
        "power": int(power),
        "optimized": optimized,
    }
    if "target_particle_number_source" in parameters:
        applied["target_particle_number_source"] = str(
            parameters["target_particle_number_source"]
        )
    return processed, applied, check, emitted_warnings


def _metadata(
    *,
    original: QubitOperator,
    processed: QubitOperator,
    config: UWCConfig,
    n_qubits: int,
    target_ld: int,
    applied_parameters: Mapping[str, Any],
    sector_check: Mapping[str, Any] | None,
    emitted_warnings: Sequence[str],
) -> dict[str, Any]:
    original_lambda_r = lambda_r_from_qubit_operator(original, target_ld)
    uwc_lambda_r = lambda_r_from_qubit_operator(processed, target_ld)
    preprocessor = "uwc" if config.enabled else "none"
    metadata = {
        "preprocessor": preprocessor,
        "uwc_method": config.method,
        "uwc_objective": config.objective,
        "uwc_target_ld": int(target_ld),
        "target_ld": int(target_ld),
        "uwc_parameters": dict(applied_parameters),
        "uwc_config": asdict(config),
        "uwc_optimizer_settings": dict(config.optimizer_settings),
        "uwc_max_iterations": int(config.max_iterations),
        "uwc_seed": config.seed,
        "uwc_use_cache": bool(config.use_cache),
        "original_hamiltonian_hash": qubit_hamiltonian_hash(
            original,
            n_qubits=n_qubits,
        ),
        "uwc_hamiltonian_hash": qubit_hamiltonian_hash(
            processed,
            n_qubits=n_qubits,
        ),
        "original_l1_norm": qubit_l1_norm(original),
        "uwc_l1_norm": qubit_l1_norm(processed),
        "original_num_terms": qubit_num_terms(original),
        "uwc_num_terms": qubit_num_terms(processed),
        "original_lambda_r": original_lambda_r,
        "uwc_lambda_r": uwc_lambda_r,
        "original_lambda_r_at_target_ld": original_lambda_r,
        "uwc_lambda_r_at_target_ld": uwc_lambda_r,
        "warnings": list(emitted_warnings),
    }
    if sector_check is not None:
        metadata["sector_preservation_check"] = dict(sector_check)
    return metadata


def preprocess_qubit_hamiltonian(
    hamiltonian: QubitOperator,
    config: UWCConfig | Mapping[str, Any] | None = None,
    *,
    n_qubits: int | None = None,
    target_ld: int | None = None,
) -> UWCPreprocessingResult:
    """
    Apply UWC-style Hamiltonian preprocessing before PR-PF sorting/splitting.

    The returned Hamiltonian is still a plain QubitOperator. Downstream sorting,
    L_D splitting, C_gs fitting, and cost optimization can therefore reuse the
    existing Pauli pipeline without UWC-specific product-formula branches.
    """

    config_norm = normalize_uwc_config(config)
    if n_qubits is None:
        n_qubits = count_qubits(hamiltonian)
    n_qubits = int(n_qubits)
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative.")

    effective_target_ld = (
        int(config_norm.target_ld)
        if config_norm.target_ld is not None
        else (0 if target_ld is None else int(target_ld))
    )
    if effective_target_ld < 0:
        raise ValueError("target_ld must be non-negative.")

    original = _copy_qubit_operator(hamiltonian)
    processed = _copy_qubit_operator(hamiltonian)
    applied_parameters: dict[str, Any] = {}
    sector_check: dict[str, Any] | None = None
    emitted_warnings: tuple[str, ...] = tuple()

    if config_norm.enabled:
        if config_norm.method in {"simple_shift", "test_shift"}:
            processed, applied_parameters = _apply_simple_shift(
                original,
                n_qubits=n_qubits,
                parameters=config_norm.parameters,
            )
        elif config_norm.method == "bliss":
            processed, applied_parameters, sector_check, emitted_warnings = _apply_bliss_shift(
                original,
                n_qubits=n_qubits,
                config=config_norm,
                target_ld=effective_target_ld,
            )
        elif config_norm.method in {"orbital_optimization", "orbital_optimization_bliss"}:
            raise NotImplementedError(
                f"UWC method '{config_norm.method}' is reserved for the future "
                "orbital-optimization implementation."
            )

    metadata = _metadata(
        original=original,
        processed=processed,
        config=config_norm,
        n_qubits=n_qubits,
        target_ld=effective_target_ld,
        applied_parameters=applied_parameters,
        sector_check=sector_check,
        emitted_warnings=emitted_warnings,
    )
    return UWCPreprocessingResult(
        original_hamiltonian=original,
        hamiltonian=processed,
        config=config_norm,
        metadata=metadata,
        warnings=emitted_warnings,
    )
