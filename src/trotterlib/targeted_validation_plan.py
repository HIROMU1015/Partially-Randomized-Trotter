"""Metadata-only targeted Track A campaign. No scientific dispatch or imports.

Counts are proposed workload ceilings, never observations or precision claims.
The frontier is a dependency simulation, not a launcher or an authorization gate.
"""
from copy import deepcopy
from graphlib import TopologicalSorter, CycleError
import hashlib
import json

BASE = "cf8e93eef50333bef41f6d75803f6b819d679343"
NAMESPACE = "TRACK_A_TARGETED_PARALLEL_20261018_DRAFT_V1"
AXES = ["cosine", "sine"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def pr_cell(p, q, r=4, K=4):
    return dict(id=f"B2_p{p}_q{q}_2nd_r{r}_K{K}", method="B2", prefix=p,
                q=q, r=r, K=K, R=q*r, formula="2nd")


def node(name, family, dependencies=(), **scope):
    return dict(id=name, family=family, dependencies=list(dependencies),
                approval_unit=name, mandatory_stop=True, approval_required=True,
                production_adapter_ready=False, scope=scope)


def build_plan():
    boundary = [pr_cell(p, q, K=K) for p in (6, 8)
                for q in (2, 4, 8) for K in (2, 4)]
    boundary += [pr_cell(7, q, r=8) for q in (4, 8)]
    boundary += [pr_cell(7, 16), pr_cell(6, 8, r=8)]
    rare = [dict(candidate=pr_cell(7, q), stratum=s, n=8)
            for q in (4, 8) for s in ("s2_multi", "s4")]
    times = [dict(T=T, q=q, PR=pr_cell(7, q), n_PR=8,
                  deterministic=[dict(method=m, prefix=p, formula=f, q=q)
                    for m, p, f in (("B0", 15, "2nd"), ("B0I", 15, "4th"),
                                    ("B1", 28, "2nd"), ("B1", 28, "4th"))])
             for T, q in (("0.4", 2), ("1.6", 8))]
    nodes = [
        node("A_SIGNAL_H8", "PR_BOUNDARY", candidates=boundary, T="0.8",
             endpoints="existing_five_eps_both_bias_factor1", new_state=False),
        node("A_COST_H8", "PR_BOUNDARY", ["A_SIGNAL_H8"], candidates=deepcopy(boundary),
             n_per_candidate=8, eligibility_gate="at_least_one_registered_endpoint",
             first_trajectory_control="each_candidate_before_remaining_seven"),
        node("B_RARE_H8", "RARE_NATIVE", groups=rare, T="0.8",
             first_trajectory_control="each_q_stratum_before_remaining_seven",
             law="original_conditional_full_count_support_not_fixed_event_count"),
        node("C_EPS_H8", "SAVED_EPSILON", epsilon=["0.04", "0.025", "0.015",
             "0.000875", "0.00075", "0.000625"], allowance_factor="1",
             bias=["USUAL_SIGNED_SAVED", "EMPIRICAL_NONCANCELLATION"],
             deterministic_candidates=184, PR_candidates=6, n_PR=32,
             comparison="frozen_six_axis_reflected_candidate_set",
             bootstrap=0, new_sampling=0, new_compile=0),
    ]
    for name, system, candidate, endpoint in (
            ("D_H8_S2", "H8", "B0_p15_q8_2nd", "usual_epsilon_0.001"),
            ("D_H8_S4", "H8", "B0I_p16_q4_4th", "noncancellation_epsilon_0.0005"),
            ("D_H6_S4", "H6", "B0I_p11_q4_4th", "usual_epsilon_0.0005")):
        nodes.append(node(name, "DIRECT_CONTROL", system=system, candidate=candidate,
            endpoint=endpoint, axes=AXES.copy(), within_pair="serial_cosine_then_sine",
            model_threshold="relative_error_ge_0.02_STOP_each_axis",
            prediction="must_bind_existing_saved_prediction_before_seal_no_new_fit"))
    nodes.extend([
        node("E_TIME_SIGNAL_H8", "TIME", panels=times, delta="0.2",
             new_state=False, new_reference=True, reoptimization=False),
        node("E_TIME_COST_H8", "TIME", ["E_TIME_SIGNAL_H8"], panels=times,
             first_trajectory_control="each_panel_PR_before_remaining_seven"),
    ])
    for system in ("H6", "H8"):
        nodes.append(node(f"F_INPUTS_{system}", "GEOMETRY_INPUT", system=system,
            distance_angstrom="1.40", basis="STO-3G", DF="tol_only_1e-8_cutoff0",
            rank="determined_by_new_input_not_fixed_to_1A_rank",
            generation="signed_order_weighted_Hermitian_projection",
            release="freeze_input_hash_rank_state_reference_then_mandatory_STOP"))
        nodes.append(node(f"F_SIGNAL_{system}", "GEOMETRY_SIGNAL", [f"F_INPUTS_{system}"],
            system=system, T="0.8", q=4, PR_prefix="ceil(rank/4)",
            deterministic_prefix="ceil(3*rank/4)", r=4, K=4,
            deterministic=["B0I_S2", "B0I_S4", "FULL_S2", "FULL_S4"],
            new_approval_after_input_freeze=True, reoptimization=False))
        nodes.append(node(f"F_COST_{system}", "GEOMETRY_COST", [f"F_SIGNAL_{system}"],
            system=system, PR_n=8, deterministic_pairs=4,
            first_trajectory_control="each_instance_PR_before_remaining_seven"))
    plan = dict(schema_version=1, kind="METADATA_ONLY_UNAPPROVED_CAMPAIGN",
        base_commit=BASE, seed_namespace=NAMESPACE, approved=False,
        scientific_dispatch_available=False, axes=AXES.copy(),
        new_scientific_executions=0, nodes=nodes,
        reuse_only=["RARE_SENSITIVITY_run01", "SELECTION_SAVED_run01",
                   "H6_fixed_delta_T_panel", "H4_geometry_results",
                   "H8_original_72_and_confirmation_union6_n32"],
        excluded=["V2_NO_GO_fit", "H5_H7_changes", "new_independent_holdout",
                  "automatic_expansion_retry_refit", "automatic_final_reaccount"],
        execution_prerequisites=["sealed_SOURCE_and_input_hashes", "fresh_runtime_compiler",
            "package_specific_explicit_approval_and_new_grant", "current_resource_probe",
            "native_control_budget_and_seed_collision_audit", "owned_process_exit_receipts"],
        resource_policy=dict(permanent_worker_cap=None, permanent_RAM_cap=None,
            selection="fresh_affinity_and_memory_probe_plus_existing_measured_task_sizes",
            sharing="cache_readonly_primitives_preparation_and_reference_by_input_T_hash",
            nested_parallelism="account_for_workers_times_library_threads",
            transient_PSI="record_and_reobserve_without_forced_scientific_restart",
            actual_OOM_input_corruption_or_scientific_mismatch="STOP_affected_scope"),
        inference_limits=["targeted_checks_not_exhaustive_optimality",
            "n8_pilot_not_power_or_population_mean_guarantee",
            "empirical_allowance_not_rigorous_bias_bound", "finite_signal_P0_not_energy_RPE",
            "model_and_measured_prices_remain_labeled"])
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (plan.get("approved") is not False or plan.get("scientific_dispatch_available") is not False
            or plan.get("new_scientific_executions") != 0):
        raise ValueError("PLANNING_ONLY_NO_SCIENTIFIC_AUTHORIZATION")
    if plan["axes"] != AXES:
        raise ValueError("PAIRED_AXES_REQUIRED")
    nodes = plan["nodes"]
    ids = [n["id"] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("DUPLICATE_NODE")
    graph = {n["id"]: set(n["dependencies"]) for n in nodes}
    for n in nodes:
        if not n["mandatory_stop"] or not n["approval_required"] or n["production_adapter_ready"]:
            raise ValueError("UNSEALED_SCOPE_REQUIRES_PREPARATION_APPROVAL_AND_STOP")
        if n["approval_unit"] != n["id"] or not graph[n["id"]] <= set(ids):
            raise ValueError("SEPARATE_APPROVAL_OR_DEPENDENCY_INVALID")
        scope = n["scope"]
        cells = scope.get("candidates", [])
        if len({c["id"] for c in cells}) != len(cells):
            raise ValueError("DUPLICATE_CANDIDATE")
        for c in cells:
            if c != pr_cell(c["prefix"], c["q"], c["r"], c["K"]):
                raise ValueError("PR_OCCURRENCE_OR_ID_MISMATCH")
        if n["family"] == "GEOMETRY_SIGNAL" and not scope["new_approval_after_input_freeze"]:
            raise ValueError("GEOMETRY_FREEZE_IS_NOT_EXECUTION_APPROVAL")
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as e:
        raise ValueError("DEPENDENCY_CYCLE") from e
    a = next(n for n in nodes if n["id"] == "A_SIGNAL_H8")["scope"]["candidates"]
    b = next(n for n in nodes if n["id"] == "A_COST_H8")["scope"]["candidates"]
    if a != b:
        raise ValueError("SIGNAL_COST_SCOPE_MISMATCH")


def planned_workload(plan):
    """Ceilings for proposed scopes; includes both axes, excludes control probes.

    Signal algebra, reference checks, and native control probes must be budgeted
    separately at sealing. No estimated scientific costs or probabilities here.
    """
    validate_plan(plan)
    scopes = {n["id"]: n["scope"] for n in plan["nodes"]}
    a = len(scopes["A_COST_H8"]["candidates"])*scopes["A_COST_H8"]["n_per_candidate"]
    b = sum(g["n"] for g in scopes["B_RARE_H8"]["groups"])
    c = scopes["C_EPS_H8"]
    panels = scopes["E_TIME_COST_H8"]["panels"]
    e = sum(p["n_PR"] + len(p["deterministic"]) for p in panels)
    f = sum(scopes[f"F_COST_{s}"]["PR_n"] + scopes[f"F_COST_{s}"]["deterministic_pairs"]
            for s in ("H6", "H8"))
    return dict(PR_boundary=dict(new_cells=len(scopes["A_SIGNAL_H8"]["candidates"]),
            trajectory_ceiling=a, native_axis_compile_ceiling=2*a),
        rare_native=dict(groups=len(scopes["B_RARE_H8"]["groups"]),
            trajectories=b, native_axis_compiles=2*b),
        epsilon_saved=dict(new_epsilons=len(c["epsilon"]),
            candidate_endpoint_rows=(c["deterministic_candidates"]+c["PR_candidates"])
                *len(c["epsilon"])*len(c["bias"]), native_compiles=0),
        direct_control=dict(configurations=3, native_axis_compile_ceiling=6),
        time=dict(new_T=2, native_axis_compile_ceiling=2*e),
        geometry=dict(new_instances=2, native_axis_compile_ceiling=2*f,
            input_freeze_and_separate_execution_approval_required=True),
        native_axis_compile_ceiling_total=2*(a+b+e+f)+6,
        budget_complete=False, control_probe_budget="pending_adapter_seal_not_zero")


def simulate_frontier(plan, *, states=None, prepared=(), approved=()):
    """Return eligible *simulated* nodes. Never launch, write grants, or poll jobs."""
    validate_plan(plan)
    states = deepcopy(states or {})
    ids = {n["id"] for n in plan["nodes"]}
    if (set(states) | set(prepared) | set(approved)) - ids:
        raise ValueError("UNKNOWN_NODE")
    if any(s not in ("PENDING", "RUNNING", "PASS", "STOP") for s in states.values()):
        raise ValueError("UNKNOWN_STATE")
    failed = {i for i, s in states.items() if s == "STOP"}
    blocked = set(failed)
    while True:
        following = blocked | {n["id"] for n in plan["nodes"]
                               if set(n["dependencies"]) & blocked}
        if following == blocked:
            break
        blocked = following
    frontier = [n["id"] for n in plan["nodes"] if n["id"] not in blocked
        and states.get(n["id"], "PENDING") == "PENDING"
        and n["id"] in prepared and n["id"] in approved
        and all(states.get(d) == "PASS" for d in n["dependencies"])]
    return dict(kind="SIMULATION_ONLY_NO_DISPATCH", ready=frontier,
                blocked=sorted(blocked), scientific_executions=0)
