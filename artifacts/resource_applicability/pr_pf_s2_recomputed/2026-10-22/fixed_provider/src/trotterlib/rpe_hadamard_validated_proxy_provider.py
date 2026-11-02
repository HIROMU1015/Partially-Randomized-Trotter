"""Resource-accounting provider for a holdout-validated Hadamard cost proxy.

The provider is deliberately narrow.  It accepts only proxy predictions at
repetition counts that were directly tested as unused holdout points, and it
returns the cost of one measured Hadamard interrogation without state
preparation.  It does not extrapolate beyond the validated holdout set.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxyValidationResult,
)
from .rpe_hadamard_interrogation import RPE_HADAMARD_INTERROGATION_SCOPE
from .rpe_resource_accounting import (
    RPE_COST_METRICS,
    RPERoundCompiledCost,
    RPERoundCostRequest,
)
from .rte import CircuitCost, CompilerSettings
from .rte_compiled_cost import compiler_settings_hash


RPE_HADAMARD_VALIDATED_PROXY_PROVIDER_VERSION = (
    "rpe_hadamard_validated_proxy_provider_v1"
)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ValidatedRPEHadamardCompiledCostProxyProvider:
    """Expose a passed holdout proxy at explicitly validated ``q_m`` only."""

    validation: RPEHadamardCompiledCostProxyValidationResult
    compiler: CompilerSettings

    def __post_init__(self) -> None:
        if not isinstance(
            self.validation,
            RPEHadamardCompiledCostProxyValidationResult,
        ):
            raise TypeError(
                "validation must be an "
                "RPEHadamardCompiledCostProxyValidationResult."
            )
        if not isinstance(self.compiler, CompilerSettings):
            raise TypeError("compiler must be a CompilerSettings instance.")
        if not self.validation.overall_pass:
            raise ValueError("A failed holdout validation cannot feed accounting.")
        if (
            compiler_settings_hash(self.compiler)
            != self.validation.proxy.compiler_settings_fingerprint
        ):
            raise ValueError(
                "Compiler settings do not match the validated proxy context."
            )

    @property
    def validated_q_m_values(self) -> tuple[int, ...]:
        return self.validation.validated_q_m_values

    @property
    def cost_model_fingerprint(self) -> str:
        proxy = self.validation.proxy
        return _fingerprint(
            {
                "provider_version": RPE_HADAMARD_VALIDATED_PROXY_PROVIDER_VERSION,
                "validation_fingerprint": self.validation.validation_fingerprint,
                "fit_fingerprint": proxy.fit_fingerprint,
                "compiler_settings_fingerprint": (
                    proxy.compiler_settings_fingerprint
                ),
                "validated_q_m_values": list(self.validated_q_m_values),
                "circuit_scope": RPE_HADAMARD_INTERROGATION_SCOPE,
                "prediction_policy": "validated_holdout_q_m_only",
            }
        )

    def _cost(self, q_m: int, *, axis: str) -> CircuitCost:
        proxy = self.validation.proxy
        values = {
            metric: float(proxy.predict(q_m, axis=axis, metric=metric))
            for metric in RPE_COST_METRICS
        }
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError("Validated proxy produced an invalid circuit cost.")
        return CircuitCost(
            **values,
            compiler=self.compiler,
            fidelity_level=5,
            estimate_kind="validated_hadamard_affine_proxy",
        )

    def evaluate(self, request: RPERoundCostRequest) -> RPERoundCompiledCost:
        if not isinstance(request, RPERoundCostRequest):
            raise TypeError("request must be an RPERoundCostRequest.")
        proxy = self.validation.proxy
        q_m = request.specification.q_m
        if q_m not in self.validated_q_m_values:
            raise ValueError(
                f"q_m={q_m} was not directly validated as a holdout point; "
                f"validated values are {self.validated_q_m_values}."
            )
        if request.preparation.preparation_hash != proxy.preparation_fingerprint:
            raise ValueError("Preparation does not match the validated proxy.")
        if request.preparation.hamiltonian_hash != proxy.hamiltonian_fingerprint:
            raise ValueError("Hamiltonian does not match the validated proxy.")
        if request.preparation.partition_hash != proxy.partition_fingerprint:
            raise ValueError("DF partition does not match the validated proxy.")
        if request.preparation.ld != proxy.ld:
            raise ValueError("L_D does not match the validated proxy.")
        if request.preparation.num_system_qubits != proxy.num_system_qubits:
            raise ValueError("System size does not match the validated proxy.")
        if request.specification.delta_time != proxy.delta_time:
            raise ValueError("delta_time does not match the validated proxy.")
        if request.rte_steps_per_occurrence != proxy.rte_steps_per_occurrence:
            raise ValueError("r_m does not match the validated proxy.")
        if request.finite_taylor_order != proxy.finite_taylor_order:
            raise ValueError("K_m does not match the validated proxy.")

        metadata = (
            ("provider_version", RPE_HADAMARD_VALIDATED_PROXY_PROVIDER_VERSION),
            ("validation_fingerprint", self.validation.validation_fingerprint),
            ("fit_fingerprint", proxy.fit_fingerprint),
            ("proxy_fingerprint", proxy.proxy_fingerprint),
            ("validated_q_m_values", self.validated_q_m_values),
            ("prediction_policy", "validated_holdout_q_m_only"),
            ("backend_context_canonical", proxy.backend_fingerprint is not None),
        )
        return RPERoundCompiledCost(
            cosine_expected_cost=self._cost(q_m, axis="cosine"),
            sine_expected_cost=self._cost(q_m, axis="sine"),
            cosine_standard_error=None,
            sine_standard_error=None,
            evaluation_method="holdout_validated_affine_proxy",
            classical_sample_count=None,
            circuit_cost_scope=RPE_HADAMARD_INTERROGATION_SCOPE,
            cost_model_fingerprint=self.cost_model_fingerprint,
            metadata=metadata,
        )
