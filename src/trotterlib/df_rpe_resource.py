"""Direct Level-5-R compiled-cost provider for RPE resource accounting.

The provider in this module deliberately returns the cost of the controlled
time-evolution subcircuit only.  It does not construct a Hadamard test, state
preparation, measurement, or a complete RPE circuit.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated import RepeatedCircuitConstructionPolicy
from .df_partial_s2_repeated_cost import (
    CompiledRepeatedPartialS2CostEstimate,
    estimate_exact_compiled_repeated_partial_s2_cost,
    estimate_monte_carlo_compiled_repeated_partial_s2_cost,
)
from .rpe_resource_accounting import RPERoundCompiledCost, RPERoundCostRequest
from .rte import (
    CompilerSettings,
    RTEConfig,
    RTEFiniteDistribution,
    require_integer_count,
)
from .rte_compiled_cost import (
    TranspiledCircuitCostCache,
    canonical_backend_fingerprint_or_none,
    compiler_settings_hash,
)


DFRPERoundCostEvaluationMethod: TypeAlias = Literal["exact", "monte_carlo"]
DF_RPE_COMPILED_COST_PROVIDER_VERSION = "df_level5r_direct_v1"


@dataclass(frozen=True)
class DFLevel5RCompiledCostProvider:
    """Adapt the existing short repeated partial-S2 compiler to one RPE round.

    ``sample_count`` is the classical Monte Carlo size used to estimate an
    expected compiled cost.  It is metadata, not a quantum shot multiplier.
    The controlled evolution has ordinary ``diag(I, U)`` semantics, with the
    ancilla immediately after the system register.
    """

    compiler: CompilerSettings
    evaluation_method: DFRPERoundCostEvaluationMethod = "exact"
    sample_count: int | None = None
    seed: int | None = None
    construction_policy: RepeatedCircuitConstructionPolicy = (
        "boundary_optimized"
    )
    maximum_repetition_count: int = 4
    maximum_trajectories: int = 10_000
    maximum_samples: int = 10_000
    maximum_untranspiled_circuit_size: int = 100_000
    maximum_retained_provenance_records: int = 1_024
    maximum_build_requests: int = 1_000_000
    maximum_transpile_requests: int = 1_000_000
    maximum_planned_instruction_applications: int = 100_000_000
    cache: TranspiledCircuitCostCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    backend: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiler, CompilerSettings):
            raise TypeError("compiler must be a CompilerSettings instance.")
        if self.evaluation_method not in ("exact", "monte_carlo"):
            raise ValueError("evaluation_method must be 'exact' or 'monte_carlo'.")
        if self.construction_policy not in (
            "raw_concatenation",
            "boundary_optimized",
        ):
            raise ValueError("Unsupported repeated circuit construction policy.")

        for name, minimum in (
            ("maximum_repetition_count", 1),
            ("maximum_trajectories", 1),
            ("maximum_samples", 1),
            ("maximum_untranspiled_circuit_size", 1),
            ("maximum_retained_provenance_records", 0),
            ("maximum_build_requests", 1),
            ("maximum_transpile_requests", 1),
            ("maximum_planned_instruction_applications", 1),
        ):
            object.__setattr__(
                self,
                name,
                require_integer_count(getattr(self, name), name=name, minimum=minimum),
            )

        if self.evaluation_method == "exact":
            if self.sample_count is not None or self.seed is not None:
                raise ValueError(
                    "Exact compiled-cost evaluation does not accept sample_count "
                    "or seed."
                )
            return
        if self.sample_count is None or self.seed is None:
            raise ValueError(
                "Monte Carlo compiled-cost evaluation requires sample_count and seed."
            )
        object.__setattr__(
            self,
            "sample_count",
            require_integer_count(
                self.sample_count,
                name="sample_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "seed",
            require_integer_count(self.seed, name="seed"),
        )
        if self.sample_count > self.maximum_samples:
            raise ValueError(
                f"sample_count={self.sample_count} exceeds "
                f"maximum_samples={self.maximum_samples}."
            )

    def __call__(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        """Return one round's cosine/sine evolution-subcircuit expectations."""
        return self.evaluate(request)

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        """Build and transpile the requested directly constructible round."""
        if not isinstance(request, RPERoundCostRequest):
            raise TypeError("request must be an RPERoundCostRequest instance.")
        preparation = request.preparation
        if not isinstance(preparation, DFPartialS2Preparation):
            raise TypeError("request.preparation must be a DFPartialS2Preparation.")
        deterministic_only = preparation.is_deterministic_only

        specification = request.specification
        repetition_count = require_integer_count(
            specification.q_m,
            name="specification.q_m",
            minimum=1,
        )
        if repetition_count > self.maximum_repetition_count:
            raise ValueError(
                f"q_m={repetition_count} exceeds the directly constructed "
                "short-round guard "
                f"maximum_repetition_count={self.maximum_repetition_count}."
            )
        round_time = float(specification.t_m)
        if not math.isfinite(round_time):
            raise ValueError("specification.t_m must be finite.")
        step_time = round_time / repetition_count
        if not math.isfinite(step_time):
            raise ValueError("Derived partial-S2 step_time must be finite.")

        rte_steps = require_integer_count(
            request.rte_steps_per_occurrence,
            name="rte_steps_per_occurrence",
            minimum=0 if deterministic_only else 1,
        )
        finite_order = require_integer_count(
            request.finite_taylor_order,
            name="finite_taylor_order",
        )
        if finite_order % 2:
            raise ValueError(
                "finite_taylor_order must be a non-negative even integer."
            )

        config = request.rte_config
        distribution = request.rte_distribution
        if deterministic_only:
            if rte_steps != 0 or finite_order != 0:
                raise ValueError(
                    "A deterministic-only DF tail requires the canonical "
                    "rte_steps_per_occurrence=0 and finite_taylor_order=0."
                )
            if config is not None or distribution is not None:
                raise ValueError(
                    "A deterministic-only DF tail requires no RTE config or "
                    "distribution."
                )
            # There is one deterministic trajectory.  Repeating it S_MC times
            # would only waste classical compilation work and would not change
            # its expected cost.
            estimate = self._estimate_exact(
                preparation,
                step_time=step_time,
                repetition_count=repetition_count,
                rte_config=None,
                rte_distribution=None,
            )
            actual_method: DFRPERoundCostEvaluationMethod = "exact"
            actual_sample_count = None
        else:
            if config is None or distribution is None:
                raise ValueError(
                    "A randomized DF tail requires an RTE config and distribution."
                )
            if config.rte_steps != rte_steps:
                raise ValueError(
                    "RTE config rte_steps does not match "
                    "rte_steps_per_occurrence."
                )
            if config.finite_taylor_order != finite_order:
                raise ValueError(
                    "RTE config cutoff does not match finite_taylor_order."
                )
            if not math.isclose(
                config.evolution_time,
                step_time,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError(
                    "RTE config evolution_time must equal the partial-S2 "
                    "delta_time."
                )
            if self.evaluation_method == "exact":
                estimate = self._estimate_exact(
                    preparation,
                    step_time=step_time,
                    repetition_count=repetition_count,
                    rte_config=config,
                    rte_distribution=distribution,
                )
                actual_method = "exact"
                actual_sample_count = None
            else:
                # __post_init__ establishes these two values for MC mode.
                if self.sample_count is None or self.seed is None:  # pragma: no cover
                    raise RuntimeError("Monte Carlo provider is not configured.")
                estimate = self._estimate_monte_carlo(
                    preparation,
                    step_time=step_time,
                    repetition_count=repetition_count,
                    rte_config=config,
                    rte_distribution=distribution,
                )
                actual_method = "monte_carlo"
                actual_sample_count = self.sample_count

        backend_fingerprint = canonical_backend_fingerprint_or_none(self.backend)
        backend_context_canonical = backend_fingerprint is not None
        cost_model_payload = {
            "provider_version": DF_RPE_COMPILED_COST_PROVIDER_VERSION,
            "compiler_settings_hash": compiler_settings_hash(self.compiler),
            "backend_fingerprint": backend_fingerprint,
            "backend_type": (
                None
                if self.backend is None
                else (
                    f"{type(self.backend).__module__}."
                    f"{type(self.backend).__qualname__}"
                )
            ),
            "evaluation_method": self.evaluation_method,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "construction_policy": self.construction_policy,
            "controlled": True,
            "ancilla_policy": "immediately_after_system_register",
            "circuit_cost_scope": "compiled_time_evolution_subcircuit",
            "fidelity_level": 5,
            "fingerprint_policy": "df_level5r_rpe_cost_model_v1",
        }
        cost_model_fingerprint = (
            hashlib.sha256(
                json.dumps(
                    cost_model_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if backend_context_canonical
            else None
        )
        return RPERoundCompiledCost(
            cosine_expected_cost=estimate.expected_cost,
            sine_expected_cost=estimate.expected_cost,
            cosine_standard_error=estimate.standard_error,
            sine_standard_error=estimate.standard_error,
            evaluation_method=actual_method,
            classical_sample_count=actual_sample_count,
            circuit_cost_scope="compiled_time_evolution_subcircuit",
            cost_model_fingerprint=cost_model_fingerprint,
            metadata=(
                ("provider_version", DF_RPE_COMPILED_COST_PROVIDER_VERSION),
                ("cost_model_fingerprint_policy", "df_level5r_rpe_cost_model_v1"),
                (
                    "compiler_settings_hash",
                    cost_model_payload["compiler_settings_hash"],
                ),
                ("backend_fingerprint", backend_fingerprint),
                ("backend_context_canonical", backend_context_canonical),
                ("requested_evaluation_method", self.evaluation_method),
                ("level5r_estimate_kind", estimate.estimate_kind),
                ("level5r_evaluation_mode", estimate.evaluation_mode),
                ("construction_policy", estimate.boundary_optimization_policy),
                ("fidelity_level", estimate.fidelity_level),
                ("repetition_count", estimate.repetition_count),
                ("maximum_repetition_count", self.maximum_repetition_count),
                ("rte_steps_per_repetition", estimate.rte_steps_per_repetition),
                ("controlled", estimate.controlled),
                ("ancilla_qubit", estimate.ancilla_qubit),
                ("trajectory_space_size", estimate.trajectory_space_size),
                ("processed_trajectory_count", estimate.processed_trajectory_count),
                ("configured_classical_sample_count", self.sample_count),
                ("deterministic_only", deterministic_only),
                ("deterministic_exact_short_circuit", deterministic_only),
                ("ordinary_control_semantics", "diag(I,U)"),
                ("hadamard_test_included", False),
                ("state_preparation_included", False),
                ("measurements_included", False),
            ),
        )

    def _estimate_exact(
        self,
        preparation: DFPartialS2Preparation,
        *,
        step_time: float,
        repetition_count: int,
        rte_config: RTEConfig | None,
        rte_distribution: RTEFiniteDistribution | None,
    ) -> CompiledRepeatedPartialS2CostEstimate:
        return estimate_exact_compiled_repeated_partial_s2_cost(
            preparation,
            step_time,
            repetition_count,
            rte_config,
            rte_distribution,
            self.compiler,
            controlled=True,
            ancilla_qubit=preparation.num_system_qubits,
            construction_policy=self.construction_policy,
            evaluation_mode="selected_only",
            maximum_trajectories=self.maximum_trajectories,
            maximum_untranspiled_circuit_size=(
                self.maximum_untranspiled_circuit_size
            ),
            maximum_retained_provenance_records=(
                self.maximum_retained_provenance_records
            ),
            maximum_build_requests=self.maximum_build_requests,
            maximum_transpile_requests=self.maximum_transpile_requests,
            maximum_planned_instruction_applications=(
                self.maximum_planned_instruction_applications
            ),
            cache=self.cache,
            backend=self.backend,
        )

    def _estimate_monte_carlo(
        self,
        preparation: DFPartialS2Preparation,
        *,
        step_time: float,
        repetition_count: int,
        rte_config: RTEConfig,
        rte_distribution: RTEFiniteDistribution,
    ) -> CompiledRepeatedPartialS2CostEstimate:
        if self.sample_count is None or self.seed is None:  # pragma: no cover
            raise RuntimeError("Monte Carlo provider is not configured.")
        return estimate_monte_carlo_compiled_repeated_partial_s2_cost(
            preparation,
            step_time,
            repetition_count,
            rte_config,
            rte_distribution,
            self.compiler,
            sample_count=self.sample_count,
            seed=self.seed,
            maximum_samples=self.maximum_samples,
            controlled=True,
            ancilla_qubit=preparation.num_system_qubits,
            construction_policy=self.construction_policy,
            evaluation_mode="selected_only",
            maximum_untranspiled_circuit_size=(
                self.maximum_untranspiled_circuit_size
            ),
            maximum_retained_provenance_records=(
                self.maximum_retained_provenance_records
            ),
            maximum_build_requests=self.maximum_build_requests,
            maximum_transpile_requests=self.maximum_transpile_requests,
            maximum_planned_instruction_applications=(
                self.maximum_planned_instruction_applications
            ),
            cache=self.cache,
            backend=self.backend,
        )
