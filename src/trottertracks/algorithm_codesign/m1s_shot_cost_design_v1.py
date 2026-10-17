"""Non-numerical shot/cost design metadata. There is no scientific runner.

This module does not read results, compute shots, compile circuits, or issue
authorization. Even an adopted-looking dictionary cannot enable execution.
"""

BASE_COMMIT = "81d6103ec89d485cb7336c3a90bd95dad66992aa"
ARTIFACT = "artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1"
CONTRACT = ARTIFACT + "/verification_contract_candidate.json"
SCHEMA = ARTIFACT + "/verification_contract_candidate.schema.json"
FORMULA_IDS = ("P1", "P2", "P3", "P4", "S1", "S2", "S3", "C1", "C2", "C3")
PENDING_CHOICES = (
    "D1_phase_requirement_policy",
    "D2_confidence_policy",
    "D3_phase_and_state_budget",
    "D4_coverage_acceptance_policy",
    "D5_cost_metric_and_scope",
    "D6_native_implementation_policy",
    "D7_decision_and_stage_policy",
)
CLOSED_FLAGS = (
    "adopted", "coverage_model_adopted", "contract_grants_execution",
    "scientific_execution_authorized", "run_ready", "platform_certified",
    "numerical_engine_implemented",
)
ZERO_FIELDS = (
    "new_scientific_calculations", "saved_numerical_reanalysis", "shot_forecasts",
    "native_compiles", "RPE_simulations", "authorizations_issued",
    "grants_issued", "markers_issued",
)


class DesignOnlyError(RuntimeError):
    """A decision packet cannot authorize scientific execution."""


def validate_candidate(candidate):
    """Validate non-numerical invariants; never inspect a scientific input."""
    if candidate.get("base_commit") != BASE_COMMIT:
        raise DesignOnlyError("Base identity differs from this design version")
    if any(candidate.get(key) is not False for key in CLOSED_FLAGS):
        raise DesignOnlyError("All design-only flags must remain closed")
    if candidate.get("future_source_identity") is not None:
        raise DesignOnlyError("No future scientific SOURCE has been fixed")
    if candidate.get("execution_authorization") is not None:
        raise DesignOnlyError("No execution authorization belongs to this packet")
    choices = candidate.get("scientific_choices", {})
    if set(choices) != set(PENDING_CHOICES):
        raise DesignOnlyError("Scientific choice inventory differs")
    if any(value is not None for value in choices.values()):
        raise DesignOnlyError("Unadopted scientific choices must remain null")
    if candidate.get("formula_ids") != list(FORMULA_IDS):
        raise DesignOnlyError("Formula inventory differs")
    if any(type(candidate.get(key)) is not int or candidate[key] != 0
           for key in ZERO_FIELDS):
        raise DesignOnlyError("This packet contains no scientific execution")
    if candidate.get("mandatory_STOP") is not True:
        raise DesignOnlyError("Mandatory STOP must remain in effect")


def request_execution(*args, **kwargs):
    """Unconditional guard, including when passed altered/adopted flags."""
    raise DesignOnlyError(
        "DESIGN_ONLY_NOT_EXECUTABLE: prepare a new fixed SOURCE and contract, "
        "obtain formal adoption and separate explicit scientific approval"
    )
