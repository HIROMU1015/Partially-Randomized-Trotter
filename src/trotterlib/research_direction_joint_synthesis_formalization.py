"""Formalize P-A v1 and audit which mechanism the completed holdouts exercise."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .parallel_validation_executor import atomic_write_json
from .research_direction_full_scope import fingerprint
from .research_direction_joint_synthesis_blind_validation import (
    validate_blind_validation_artifact,
)
from .research_direction_joint_synthesis_pilot import (
    validate_joint_synthesis_pilot_artifact,
)


SCHEMA_VERSION = "research_direction_joint_synthesis_formalization_v1"
METHOD = "pa_v1_dp_formalization_and_empirical_mechanism_audit_v1"
STAGE = "P-A-v1-formalization"

EXPECTED_PILOT_FINGERPRINT = (
    "1a9840a4ee46daa6e3749272593acf3e8ae29be9fcecc1a05b0bd9ea817fbc37"
)
EXPECTED_BLIND_FINGERPRINT = (
    "78af3474898dbf989780ea5f2881cb5b61595609dd2698164b9846c1ce1c5919"
)


@dataclass(frozen=True, order=True)
class IntervalObjective:
    """The exact lexicographic objective used by frozen P-A v1."""

    twice_basis_operation_count: int
    segment_count: int
    support_union_size_sum: int
    full_mode_penalty: int

    def plus(
        self,
        *,
        basis_operation_count: int,
        support_union_size: int,
        mode: str,
    ) -> "IntervalObjective":
        if mode not in {"full", "support_union"}:
            raise ValueError(f"Unsupported interval mode: {mode!r}.")
        return IntervalObjective(
            self.twice_basis_operation_count + 2 * int(basis_operation_count),
            self.segment_count + 1,
            self.support_union_size_sum + int(support_union_size),
            self.full_mode_penalty + int(mode == "full"),
        )


def solve_lexicographic_interval_dp(
    length: int,
    options: Mapping[tuple[int, int, str], tuple[int, int]],
) -> tuple[IntervalObjective, tuple[tuple[int, int, str], ...]]:
    """Solve the formal v1 recurrence for a precomputed interval-option table."""
    if length <= 0:
        raise ValueError("length must be positive.")
    zero = IntervalObjective(0, 0, 0, 0)
    best: list[
        tuple[IntervalObjective, tuple[tuple[int, int, str], ...]] | None
    ] = [(zero, ())] + [None] * length
    for stop in range(1, length + 1):
        candidates = []
        for start in range(stop):
            prefix = best[start]
            if prefix is None:
                continue
            for mode in ("full", "support_union"):
                key = (start, stop, mode)
                if key not in options:
                    raise ValueError(f"Missing interval option {key!r}.")
                operation_count, support_size = options[key]
                objective = prefix[0].plus(
                    basis_operation_count=operation_count,
                    support_union_size=support_size,
                    mode=mode,
                )
                candidates.append((objective, (*prefix[1], key)))
        best[stop] = min(candidates, key=lambda item: (item[0], item[1]))
    selected = best[length]
    if selected is None:
        raise RuntimeError("Interval recurrence did not cover the run.")
    return selected


def _selected_objective(run: Mapping[str, Any]) -> IntervalObjective:
    objective = IntervalObjective(0, 0, 0, 0)
    for segment in run["segments"]:
        objective = objective.plus(
            basis_operation_count=int(segment["basis_operation_count"]),
            support_union_size=int(segment["support_union_size"]),
            mode=str(segment["mode"]),
        )
    return objective


def _transition_count(metadata: Mapping[str, Any]) -> int:
    # Two modes are considered for every one of n(n+1)/2 contiguous intervals.
    return sum(
        int(run["run_length"]) * (int(run["run_length"]) + 1)
        for run in metadata["runs"]
    )


def _audit_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    run_count = 0
    segment_count = 0
    full_segments = 0
    union_segments = 0
    split_records = 0
    multi_application_records = 0
    transition_mismatches = 0
    objective_mismatches = 0
    implied_extra_applications = 0

    for record in records:
        metadata = record["interval_metadata"]
        run_count += int(metadata["run_count"])
        segment_count += int(metadata["selected_segment_count"])
        split_records += int(
            int(metadata["selected_segment_count"]) > int(metadata["run_count"])
        )
        multi_application_records += int(
            int(metadata["selected_multi_application_interval_count"]) > 0
        )
        transition_mismatches += int(
            int(metadata["dp_transition_count"]) != _transition_count(metadata)
        )
        applications = sum(int(run["run_length"]) for run in metadata["runs"])
        implied_extra_applications += applications - int(record["sequence_length"])
        for run in metadata["runs"]:
            objective = _selected_objective(run)
            objective_mismatches += int(
                objective.twice_basis_operation_count
                != int(run["selected_proxy_basis_operation_count"])
                or objective.segment_count != int(run["segment_count"])
            )
            for segment in run["segments"]:
                full_segments += int(segment["mode"] == "full")
                union_segments += int(segment["mode"] == "support_union")

    return {
        "record_count": len(records),
        "run_count": run_count,
        "selected_segment_count": segment_count,
        "records_with_within_run_split": split_records,
        "records_with_multi_application_interval": multi_application_records,
        "selected_full_segments": full_segments,
        "selected_support_union_segments": union_segments,
        "dp_transition_formula_mismatches": transition_mismatches,
        "selected_objective_record_mismatches": objective_mismatches,
        "implied_extra_applications_beyond_one_rotation_per_event": (
            implied_extra_applications
        ),
    }


def _stratum_audit(stratum: Mapping[str, Any]) -> dict[str, Any]:
    holdout_rows = list(stratum["holdout_rows"])
    probes = list(stratum["operator_equivalence_probes"])
    event_orders = [
        int(order) for row in holdout_rows for order in row["event_orders"]
    ]
    holdout = _audit_records(holdout_rows)
    operator_probes = _audit_records(probes)
    return {
        "holdout": holdout,
        "operator_probes": operator_probes,
        "holdout_event_count": len(event_orders),
        "holdout_nonzero_taylor_order_event_count": sum(
            int(order != 0) for order in event_orders
        ),
        "holdout_maximum_taylor_order_observed": max(event_orders, default=0),
        "within_run_split_observed": (
            holdout["records_with_within_run_split"] > 0
            or operator_probes["records_with_within_run_split"] > 0
        ),
        "nonzero_taylor_structure_observed": (
            any(order != 0 for order in event_orders)
            or operator_probes[
                "implied_extra_applications_beyond_one_rotation_per_event"
            ]
            > 0
        ),
    }


def evaluate_joint_synthesis_formalization(
    pilot: Mapping[str, Any],
    blind: Mapping[str, Any],
) -> dict[str, Any]:
    """Formalize frozen v1 and audit the completed empirical mechanism coverage."""
    validate_joint_synthesis_pilot_artifact(pilot)
    validate_blind_validation_artifact(blind)

    strata = {
        stratum_id: _stratum_audit(stratum)
        for stratum_id, stratum in blind["strata"].items()
    }
    holdout_records = sum(row["holdout"]["record_count"] for row in strata.values())
    probe_records = sum(
        row["operator_probes"]["record_count"] for row in strata.values()
    )
    split_records = sum(
        row["holdout"]["records_with_within_run_split"]
        + row["operator_probes"]["records_with_within_run_split"]
        for row in strata.values()
    )
    nonzero_events = sum(
        row["holdout_nonzero_taylor_order_event_count"] for row in strata.values()
    )
    inferred_probe_extra_applications = sum(
        row["operator_probes"][
            "implied_extra_applications_beyond_one_rotation_per_event"
        ]
        for row in strata.values()
    )
    transition_mismatches = sum(
        row[part]["dp_transition_formula_mismatches"]
        for row in strata.values()
        for part in ("holdout", "operator_probes")
    )
    objective_mismatches = sum(
        row[part]["selected_objective_record_mismatches"]
        for row in strata.values()
        for part in ("holdout", "operator_probes")
    )
    h4_digest_matches = (
        pilot["holdout"]["event_stream_digest"]
        == blind["strata"]["h4_compiler_transfer_opt2"]["holdout"][
            "event_stream_digest"
        ]
    )

    checks = {
        "pilot_fingerprint_matches": pilot.get("content_fingerprint")
        == EXPECTED_PILOT_FINGERPRINT,
        "blind_fingerprint_matches": blind.get("content_fingerprint")
        == EXPECTED_BLIND_FINGERPRINT,
        "blind_validation_passed": bool(blind.get("overall_pass"))
        and bool(blind["decision"]["blind_validation_passed"]),
        "h4_pilot_and_opt2_event_stream_match": h4_digest_matches,
        "all_48_holdout_rows_audited": holdout_records == 48,
        "all_6_operator_probes_audited": probe_records == 6,
        "dp_transition_formula_matches_all_records": transition_mismatches == 0,
        "selected_objective_metadata_matches_all_runs": objective_mismatches == 0,
    }

    return {
        "input_fingerprints": {
            "pilot": pilot["content_fingerprint"],
            "blind": blind["content_fingerprint"],
        },
        "formal_problem": {
            "domain": (
                "Each maximal consecutive run with one source basis is partitioned "
                "into contiguous nonempty intervals. Every interval selects either "
                "the registered full basis or the deterministic completion preserving "
                "the union of all diagonal supports in that interval."
            ),
            "objective_tuple": [
                "twice_basis_operation_count",
                "segment_count",
                "support_union_size_sum",
                "full_mode_penalty",
            ],
            "objective_order": "lexicographic_minimum",
            "recurrence": (
                "DP[t] = min over 0<=s<t and mode in {full,support_union} "
                "of DP[s] plus interval_objective(s,t,mode)"
            ),
            "optimality_scope": (
                "Global optimum only within the finite contiguous-interval, two-mode "
                "candidate family and the frozen basis-operation-count proxy."
            ),
            "transition_count_per_run_of_length_n": "n*(n+1)",
            "dp_time_with_precomputed_interval_options": "O(n^2) per run",
            "dp_predecessor_space": "O(n) per run",
            "candidate_table_space_if_materialized": "O(n^2) per run",
            "excluded_from_optimality_claim": [
                "compiled RZ global optimality",
                "arbitrary Gaussian completions",
                "direct pairwise relative-basis transition synthesis",
                "noncontiguous grouping",
                "coupling-aware or noise-aware routing",
            ],
        },
        "operator_equivalence_contract": {
            "statement": (
                "For every Z/ZZ application with support S, the selected one-particle "
                "basis preserves the source-unitary columns indexed by S. Therefore "
                "the conjugated diagonal operator is unchanged. Products remain "
                "unchanged in event order; cancelling adjacent inverse/equal basis "
                "changes is exact; basis-independent scalar phases are accumulated "
                "with the same global or controlled relative-phase convention."
            ),
            "assumptions": [
                "source and selected basis definitions are unitary",
                "all columns in each application support are preserved",
                "basis-plan source hashes and support labels match the application",
                "basis changes are uncontrolled and only the diagonal action is controlled",
                "identity and event scalar phases use the existing accumulation rule",
            ],
            "implementation_certificate": (
                "preserved_columns_max_abs_residual <= 1e-12 at builder acceptance; "
                "six completed dense operator probes additionally passed 1e-10"
            ),
        },
        "empirical_mechanism_audit": {
            "strata": strata,
            "holdout_record_count": holdout_records,
            "operator_probe_count": probe_records,
            "records_with_within_run_split": split_records,
            "holdout_nonzero_taylor_order_event_count": nonzero_events,
            "operator_probe_implied_extra_applications": (
                inferred_probe_extra_applications
            ),
            "one_segment_per_source_run_matches_selected_plan_on_all_records": (
                split_records == 0
            ),
            "incremental_interval_partition_plan_changes_observed": split_records,
            "incremental_interval_partition_compiled_benefit_identified": False,
            "interpretation": (
                "The completed transfer evidence supports run-local full/support-union "
                "basis selection. It does not empirically distinguish the interval "
                "partitioning layer from a one-segment-per-source-run baseline, and it "
                "does not exercise nonzero Taylor-order event structure."
            ),
        },
        "decision": {
            "status": (
                "pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished"
            ),
            "primary_theme_status": (
                "conditional_candidate_pending_nondegenerate_mechanism_validation"
            ),
            "secondary_theme": "P-C_geometry_energy_difference",
            "next_action": (
                "preregister_a_small_forced_structure_comparison_against_an_explicit_"
                "one_segment_per_source_run_baseline"
            ),
            "h12_or_long_rpe_required_next": False,
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_circuit_compilation_performed": False,
            "new_physical_simulation_performed": False,
            "dp_global_compiled_cost_optimality_claimed": False,
            "interval_partition_advantage_claimed": False,
            "nonzero_taylor_order_transfer_validated": False,
            "literature_novelty_established": False,
            "h12_evaluated": False,
            "rpe_or_final_total_cost_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_joint_synthesis_formalization_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_joint_synthesis_formalization_artifact(payload)
    return payload


def validate_joint_synthesis_formalization_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported P-A formalization schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported P-A formalization method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("P-A formalization artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("P-A formalization status does not match checks.")
    if payload.get("decision", {}).get("status") != (
        "pa_v1_formalized_but_interval_mechanism_not_empirically_distinguished"
    ):
        raise ValueError("P-A formalization decision was overstated or changed.")
    audit = payload.get("empirical_mechanism_audit", {})
    if audit.get("records_with_within_run_split") != 0:
        raise ValueError("Frozen formalization artifact must preserve the zero-split result.")
    if audit.get("holdout_nonzero_taylor_order_event_count") != 0:
        raise ValueError("Frozen formalization artifact must preserve Taylor coverage.")
    scope = payload.get("scope", {})
    for key in (
        "new_circuit_compilation_performed",
        "new_physical_simulation_performed",
        "dp_global_compiled_cost_optimality_claimed",
        "interval_partition_advantage_claimed",
        "nonzero_taylor_order_transfer_validated",
        "literature_novelty_established",
        "h12_evaluated",
        "rpe_or_final_total_cost_evaluated",
        "scientific_superiority_claimed",
    ):
        if scope.get(key) is not False:
            raise ValueError(f"P-A formalization artifact overstates scope: {key}.")


def write_joint_synthesis_formalization_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_joint_synthesis_formalization_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)
