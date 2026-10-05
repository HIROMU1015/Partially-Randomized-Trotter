"""Review-only, pure in-memory JSON validation, with no launch or authorization IO.

Fictional stage examples are explicitly marked; acceptance never authorizes work.
The inherited v1 module is loaded by the synthetic test runner from its pinned path.
"""
import hashlib
import json
import math
import re
import contract_validator_v1 as inherited

GIB = 2**30
DISTANCES = ["0.70", "0.80", "0.90", "1.10", "1.40", "1.60"]
HASH_FIELDS = ("hamiltonian_sha256", "df_sha256", "state_sha256",
               "sector_sha256", "order_sha256", "coordinate_sha256")
STAGES = (
    "CONTRACT_V2_AWAITING_REVIEW", "CONTRACT_CONDITIONS_APPROVED",
    "SCIENCE_SOURCE_IMPLEMENTED", "SCIENCE_SOURCE_FROZEN",
    "INPUT_GENERATION_PLAN_READY", "INPUT_GENERATION_AUTHORIZATION_REVIEWED",
    "INPUT_GENERATION_LAUNCH_APPROVED", "INPUTS_FROZEN_STOP",
    "INPUT_BOUND_SIGNAL_PLAN_SEALED", "SIGNAL_AUTHORIZATION_REVIEWED",
    "SIGNAL_LAUNCH_APPROVED", "MAP_COMPLETE_STOP",
)
ACTIONS = ("approve_contract", "implement_source", "freeze_source",
           "prepare_generation_plan", "review_generation_authorization",
           "explicit_generation_launch", "generate_freeze_inputs_then_stop",
           "seal_input_bound_signal_plan", "review_signal_authorization",
           "explicit_signal_launch", "signal_compile_map_then_stop")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan_fingerprint(plan):
    body = {k: v for k, v in plan.items() if k != "plan_fingerprint"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_zero_plan(plan, schema_validator):
    schema_validator.validate(plan)
    require(plan_fingerprint(plan) == plan["plan_fingerprint"], "plan digest mismatch")
    require(plan["distances_angstrom"] == DISTANCES, "distance scope")
    require(len(plan["templates"]) == 218, "template coverage")
    require(len({t["template_id"] for t in plan["templates"]}) == 218, "duplicate template")
    counts = {m: sum(t["method"] == m for t in plan["templates"]) for m in ("B0", "B1", "B2", "B3")}
    require(counts == {"B0": 20, "B1": 4, "B2": 145, "B3": 49}, "method coverage")
    require(sum(s["logical_wrapper_slots"] for s in plan["symbolic_slots"]) == 74784,
            "wrapper coverage")
    require(plan["model"]["system_qubits"] == 8 and plan["model"]["ancilla_count"] == 1
            and plan["model"]["ancilla_index"] == 8 and plan["model"]["total_qubits"] == 9,
            "qubit scope")


def select_workers(observation, schema_validator):
    """Evaluate fictional observations only; do not inspect or reserve host memory."""
    schema_validator.validate(observation)
    for field in ("available_bytes", "allowed_cpu_count", "requested_workers", "observation_age_seconds"):
        require(type(observation[field]) is int, "integer observation required")
    require(not observation["memory_pressure"], "memory pressure: STOP")
    available = observation["available_bytes"]
    ceiling = min(12, observation["requested_workers"], observation["allowed_cpu_count"])
    for workers in range(ceiling, 0, -1):
        if available >= required_available(workers):
            return workers
    raise ValueError("minimum worker cannot be admitted: STOP")


def required_available(workers):
    require(type(workers) is int and 1 <= workers <= 12, "worker range")
    return 8*GIB + workers*8*GIB + 16*GIB


def validate_frozen_inputs(inputs):
    require(type(inputs) is list and len(inputs) == 6, "six frozen inputs required")
    require([v["distance_angstrom"] for v in inputs] == DISTANCES, "input geometry coverage/order")
    for value in inputs:
        require(set(value) == {"distance_angstrom", *HASH_FIELDS}, "frozen input hash fields")
        for field in HASH_FIELDS:
            require(type(value[field]) is str and re.fullmatch(r"[0-9a-f]{64}", value[field]) is not None,
                    "null or invalid frozen input: " + field)


def validate_transition(current, action, context):
    """Check a hypothetical transition. Never execute, sign, or issue authorization."""
    require(context.get("artifact_scope") == "SYNTHETIC_CONTRACT_TEST_ONLY", "fictional scope required")
    require(context.get("actual_authorizations") == [], "actual authorization forbidden")
    require(current in STAGES[:-1], "unknown or terminal stage")
    index = STAGES.index(current)
    require(action == ACTIONS[index], "stage skip or operation before authorization")
    require(context["actual_signal_or_cost_before_authorization"] is False,
            "actual signal/cost cannot be used to pass preauthorization gates")
    flags = context["flags"]
    def flag(key):
        require(flags.get(key) is True, "missing prerequisite: " + key)
    flag("contract_conditions_approved")
    if index >= 1:
        flag("separate_implementation_instruction")
        flag("synthetic_source_gates_passed")
    if index >= 2:
        source = context["source_commit"]
        require(type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None, "actual source commit required")
        require(source == context["expected_source_commit"], "independent source binding")
        hashes = context["source_hashes"]
        require(type(hashes) is dict and len(hashes) > 0 and
                all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
                "source hashes required")
    if index >= 3:
        flag("generation_plan_source_bound")
        require(context["generation_plan_fingerprint"] == context["expected_generation_plan_fingerprint"],
                "independent generation plan binding")
    if index >= 4:
        flag("generation_authorization_reviewed")
        validate_authorization_example(context, "generation")
    if index >= 5:
        flag("generation_explicit_launch")
        flag("resources_admitted")
    if index >= 6:
        flag("generation_numerical_gates_passed")
        flag("generation_complete_stop")
        validate_frozen_inputs(context["frozen_inputs"])
    if index >= 7:
        flag("input_bound_signal_plan_sealed")
        require(context["signal_plan_fingerprint"] == context["expected_signal_plan_fingerprint"],
                "independent input-bound plan binding")
    if index >= 8:
        flag("signal_authorization_reviewed")
        validate_authorization_example(context, "signal")
        require(context["signal_authorization_example"] != context["generation_authorization_example"],
                "generation authorization cannot authorize signal/compile")
    if index >= 9:
        flag("signal_explicit_launch")
    return STAGES[index+1]


def validate_authorization_example(context, kind):
    example = context[kind + "_authorization_example"]
    require(example.get("marker") == "ARTIFICIAL_NOT_AUTHORIZATION", "fictional authorization required")
    require(example.get("scope") == ("INPUT_GENERATION_ONLY" if kind == "generation" else "SIGNAL_COMPILE_ONLY"),
            "separate authorization scope")
    require(example.get("result_prior") is True, "result-prior authorization example required")
    require(example.get("source_commit") == context["expected_source_commit"], "authorization source mismatch")
    require(example.get("plan_fingerprint") == context["expected_" + ("generation" if kind == "generation" else "signal") + "_plan_fingerprint"],
            "authorization plan mismatch")


def validate_numeric_gate_observations(measurements):
    """Compare fictional scalar observations only; no states, matrices or solver."""
    limits = {"state_norm_absolute": 1e-12, "reference_residual_L2_Ha": 1e-9,
              "relative_Hermiticity": 1e-12, "imaginary_energy_Ha": 1e-11,
              "outside_sector_max": 1e-12, "full_sector_consistency_max": 1e-12}
    require(set(measurements) == {*limits, "sector_gap_Ha"}, "measurement fields")
    for key, value in measurements.items():
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative gate observation")
    for key, limit in limits.items():
        require(measurements[key] <= limit, "numerical gate STOP: " + key)
    require(measurements["sector_gap_Ha"] > 1e-10, "degenerate sector: STOP")


def pressure_requires_stop(*, available_bytes, worker_rss_bytes, driver_rss_bytes,
                           admitted_workers, psi_full_avg10, oom_event_delta):
    require(type(available_bytes) is int and available_bytes >= 0, "memory observation")
    required_available(admitted_workers)
    require(type(worker_rss_bytes) is list and len(worker_rss_bytes) == admitted_workers,
            "owned worker observations")
    require(all(type(v) is int and v >= 0 for v in [driver_rss_bytes, *worker_rss_bytes]), "RSS observations")
    require(type(psi_full_avg10) in (int, float) and math.isfinite(psi_full_avg10) and psi_full_avg10 >= 0,
            "PSI observation")
    require(type(oom_event_delta) is int and oom_event_delta >= 0, "OOM delta")
    return (available_bytes < 16*GIB or driver_rss_bytes > 8*GIB or
            any(v > 8*GIB for v in worker_rss_bytes) or psi_full_avg10 > 0 or oom_event_delta > 0)


def validate_record(*args, **kwargs):
    # The unchanged v1 verifier remains the owner/cache/digest source of truth.
    return inherited.validate_record(*args, **kwargs)
