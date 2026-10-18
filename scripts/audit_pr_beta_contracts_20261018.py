"""Fixed Git blobs, saved metadata, and artificial/mock checks only.

Does not import trotterlib/numpy/scipy/Qiskit, read NPZ/private runtime,
evaluate saved shot/cost formulas, optimize real candidates, or sample.
Executing extracted functions is restricted to explicitly synthetic fixtures.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIXED = "f29eafbf4412193e8e8660d9f051889c2cba684b"
BASE = "e62ddb59173d100c113be7a33581515d78f152ee"
OUT = ROOT / "artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18"
LIB = "src/trotterlib/"
WP11 = "artifacts/research_direction_full_opt2/"
RESULTS = {
    "allocation": "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_beta_alpha_sensitivity_v1.json",
    "wp04": "artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json",
    "wp01_v1_history": "artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v1.json",
    "wp01_v2": "artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json",
    "wp11": WP11 + "2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json",
    "wp11_audit": WP11 + "2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json",
    "phase4": "artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json",
    "long_schedule": "artifacts/rpe_delta_round_schedule_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json",
    "wp01_screening": "artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json",
    "wp05a": "artifacts/research_direction_full_scope/2026-09-22/wp05a_full_controlled_interrogation_connection_v1.json",
    "wp05b": "artifacts/research_direction_full_scope_extension/2026-09-22/wp05b_q8_delta_0p01_full_scope_extension_v1.json",
    "wp05br": "artifacts/research_direction_full_scope_replication/2026-09-22/wp05br_r32_32trajectory_replication_v1.json",
    "a0": WP11 + "2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json",
}
SPECS = {k: (FIXED, v) for k, v in RESULTS.items()}
SPECS["signal_grid"] = (FIXED, "artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json")
for name in [
    "rpe_resource_accounting", "research_direction_ablation",
    "research_direction_decision_cost", "research_direction_full_opt2",
    "research_direction_full_opt2_extension_analysis", "finite_rte_signal_validation",
    "rpe_four_round_phase_validation", "rpe_delta_round_schedule_validation",
    "rpe_allocation_sensitivity_validation", "research_direction_prevalidation",
    "rpe_hadamard_compiled_cost_proxy", "rpe_hadamard_compiled_cost_benchmark", "df_partial_s2", "rte",
]:
    SPECS[name] = (FIXED, LIB + name + ".py")
for name in [
    "research_direction_ablation", "research_direction_decision_cost",
    "research_direction_full_opt2_extension_analysis", "rpe_four_round_phase_validation",
    "rpe_allocation_sensitivity_validation",
]:
    SPECS["runner_" + name] = (FIXED, "scripts/run_" + name + ".py")
for name in [
    "rpe_certification_boundary", "rpe_resource_accounting_guards",
    "research_direction_decision_cost", "research_direction_ablation",
    "research_direction_full_opt2_extension_analysis",
]:
    SPECS["tests_" + name] = (FIXED, "tests/test_" + name + ".py")
for name in [
    "research_direction_ablation", "research_direction_decision_cost",
    "research_direction_full_opt2", "rpe_four_round_phase_validation",
    "rpe_delta_round_schedule_validation", "rpe_allocation_sensitivity_validation",
]:
    SPECS["doc_" + name] = (FIXED, "docs/" + name + ".md")
for label, path in {
    "initial_manifest": "2026-09-24/wp11_all_r_opt2_initial_20260924_023012.manifest.json",
    "extension_manifest": "2026-09-24/wp11_all_r_opt2_extension_20260924_222811.manifest.json",
    "initial_aggregate": "2026-09-24/wp11_all_r_opt2_initial_20260924_023012/aggregate.json",
    "extension_aggregate": "2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/aggregate.json",
    "initial_provenance": "2026-09-24/wp11_all_r_opt2_initial_20260924_023012/run_provenance.json",
    "extension_provenance": "2026-09-24/wp11_all_r_opt2_fresh32_20260924_232418/run_provenance.json",
}.items():
    SPECS[label] = (FIXED, WP11 + path)
SPECS["alpha_audit"] = (BASE, "docs/research/pr_alpha_contract_audit_20261018.md")
SPECS["track_a_shot"] = ("fa75c7997e135da7b7a0c4e23949e47252bd7a2e", "scripts/d0_frozen_shot_rule_v1.py")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def extract(source, names, ns):
    """Compile named, inspected definitions; execute no source module imports."""
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if set(names) != {n.name for n in nodes}:
        raise ValueError("AST_DEFINITION_IDENTITY")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<fixed-beta-definitions-synthetic-only>", "exec"), ns)
    return ns


def synthetic_checks(blobs):
    checks = []
    def check(name, predicate, classification="PASS"):
        if not predicate:
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True, classification=classification))
    def raises(name, exc, call, classification="PASS"):
        try:
            call()
        except exc:
            check(name, True, classification)
        else:
            raise AssertionError(name)

    module = ModuleType("_beta_artificial_fixture")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(math=math, Fraction=Fraction, dataclass=dataclass)
    class Preparation:
        product_formula = "2nd"
        is_deterministic_only = True
        exact_rte_lambda_r = 0.0
        threshold_operator_error_bound = 0.0
        hamiltonian_hash = partition_hash = preparation_hash = "ARTIFICIAL_ONLY"
        ld = 12
        coefficient_atol = 0.0
    class Candidate(SimpleNamespace):
        @property
        def q_m(self):
            return self.specification.q_m
    def integer(v, *, name, minimum=0):
        if type(v) is not int or v < minimum:
            raise ValueError(name)
        return v
    ns.update(
        DFPartialS2Preparation=Preparation, RPERoundCandidate=Candidate,
        require_integer_count=integer, _normalize_cost_metric=lambda x: x,
        MAXIMUM_RPE_ROUND_INDEX=62, RPE_NUMERICAL_PHASE_MODEL_LIMIT=math.pi/2,
        RPE_STRICT_BRANCH_CERTIFICATION_LIMIT=math.pi/3,
        RPE_UNIT_UNATTENUATED_SIGNAL_RADIUS=1.0,
        DEFAULT_RPE_CIRCUIT_COST_SCOPE="compiled_time_evolution_subcircuit",
        _KNOWN_EMPIRICAL_PF_SOURCES={"paper_d6_empirical_surrogate", "df_phase_bias_surrogate_v3"},
    )
    extract(blobs["rpe_resource_accounting"], [
        "_finite_nonnegative", "_finite_positive", "_conservative_nonnegative_sum",
        "RPERoundSpecification", "RPEErrorAllocation", "RPEHadamardSamplingPolicy",
        "RPEPFErrorModel", "_hoeffding_shots", "evaluate_rpe_round_candidate",
    ], ns)
    allocation, model = ns["RPEErrorAllocation"], ns["RPEPFErrorModel"]
    def candidate(*, beta=.2, stat=.2, rigorous=True, policy=None, preparation=None, pf=0.0):
        return ns["evaluate_rpe_round_candidate"](
            preparation or Preparation(), ns["RPERoundSpecification"](0, 1.0),
            allocation(0.0, 0.0, stat, .025, .025),
            model(pf, "ARTIFICIAL_ONLY", rigorous), beta_rpe=beta,
            rte_steps_per_occurrence=0, finite_taylor_order=0, cost_metric="rz_count",
            hadamard_sampling_policy=policy or ns["RPEHadamardSamplingPolicy"]("fresh_iid_per_hadamard_shot", True),
        )
    valid = candidate()
    check("artificial_rigorous_candidate_is_conditionally_certified", valid.feasible and valid.guarantee_status == "certified")
    check("empirical_is_not_certified", candidate(rigorous=False).guarantee_status == "empirical_screening")
    check("branch_margin_overrides_empirical", candidate(beta=math.pi/3, rigorous=False).guarantee_status == "not_certified")
    check("branch_lower_ULP_is_accepted", candidate(beta=math.nextafter(math.pi/3, 0)).guarantee_status == "certified")
    check("budget_ULP_feasible_is_not_certificate", candidate(stat=math.nextafter(.2, math.inf)).feasible and candidate(stat=math.nextafter(.2, math.inf)).guarantee_status == "not_certified")
    check("actual_pf_ULP_is_rechecked", candidate(pf=math.ulp(0.0)).feasible and candidate(pf=math.ulp(0.0)).guarantee_status == "not_certified")
    threshold = Preparation(); threshold.threshold_operator_error_bound = .01
    check("threshold_error_is_not_certified", candidate(preparation=threshold).guarantee_status == "not_certified")
    unspecified = ns["RPEHadamardSamplingPolicy"]()
    check("missing_IID_is_not_certified", candidate(policy=unspecified).guarantee_status == "not_certified")
    for value in [math.nan, math.inf, -.1]:
        raises("reject_nonfinite_or_negative_allocation_" + str(value), ValueError, lambda v=value: allocation(v, 0, .2, .025, .025))
    for alpha in [0, 1, math.nan]:
        raises("reject_invalid_alpha_" + str(alpha), ValueError, lambda a=alpha: allocation(0, 0, .2, a, .025))
    raises("empirical_source_cannot_be_marked_rigorous", ValueError, lambda: model(1, "paper_d6_empirical_surrogate", True))
    exact = sum(map(Fraction.from_float, [.01, .02, .3]), Fraction())
    check("budget_sum_rounds_outward", Fraction.from_float(ns["_conservative_nonnegative_sum"]((.01,.02,.3))) >= exact)
    shots = ns["_hoeffding_shots"]
    check("artificial_shots_obey_ceiling", shots(.5, .025) == math.ceil(8 * math.log(80)))
    raises("zero_coordinate_is_rejected", ValueError, lambda: shots(0, .025))
    raises("nonfinite_coordinate_is_rejected", ValueError, lambda: shots(math.nan, .025))
    raises("overflow_is_explicit", OverflowError, lambda: shots(1e-160, .025))
    raises("coordinate_square_underflow_escapes_as_ZeroDivisionError", ZeroDivisionError,
           lambda: shots(1e-200, .025), "REPRODUCED_IMPLEMENTATION_BUG")
    raises("candidate_tiny_positive_stat_budget_escapes_as_ZeroDivisionError", ZeroDivisionError,
           lambda: candidate(stat=1e-200), "REPRODUCED_IMPLEMENTATION_BUG")
    check("zero_stat_budget_returns_infeasible", not candidate(stat=0).feasible)

    # Real ablation projection plus real candidate evaluator, artificial deterministic provider.
    extract(blobs["research_direction_ablation"], ["_continuous_hoeffding_shots", "_evaluate_pair_round", "_select_schedule", "_weighted_alpha", "evaluate_ablation_scenario"], ns)
    artificial_provider = SimpleNamespace(rte_steps=0, finite_taylor_order=0, predict=lambda axis, q: (7.0, 0.0))
    def projected(stat=.2):
        return ns["_evaluate_pair_round"](
            Preparation(), model(0, "ARTIFICIAL_ONLY", False), artificial_provider,
            round_index=0, delta_time=1, beta_rpe=.2, beta_profile=(0,0,stat),
            alpha_cosine=.025, alpha_sine=.025, rte_seed=0,
        )
    row = projected()
    check("projection_preserves_axis_shots_and_cost_scope_arithmetic", row["compiled_rz_point_estimate"] == 14 * row["axes"]["cosine"]["shots"])
    check("projection_drops_guarantee_and_assumptions", not {"guarantee_status", "certification_reasons", "assumptions"}.intersection(row))
    check("projection_excludes_infeasible_candidate", projected(0) is None)
    # Exercise actual selector; scientific round evaluator replaced by a small lookup table.
    selector = ns["_select_schedule"]
    ns["SCHEDULE_POLICIES"] = ("fixed_compiled_rz", "round_component_applications", "round_compiled_rz")
    providers = [SimpleNamespace(pair=(2,2)), SimpleNamespace(pair=(1,2))]
    def lookup(prep, pf, provider, *, round_index, **kwargs):
        pair = provider.pair
        if pair == (1,2) and round_index == 1:
            return None
        return dict(r_m=pair[0], compiled_rz_point_estimate=1.0, shot_weighted_component_application_proxy=1.0)
    ns["_evaluate_pair_round"] = lookup
    selector_kw = dict(maximum_round_index=1, delta_time=1, beta_rpe=.2, beta_profile=(0,0,.2), alpha_by_axis={(i,b): .01 for i in (0,1) for b in ("cosine","sine")}, rte_seed=0)
    selected = selector(None, None, providers, schedule_policy="round_compiled_rz", **selector_kw)
    check("selector_tie_break_and_updated_infeasibility", [r["r_m"] for r in selected] == [1,2])
    selected = selector(None, None, providers, schedule_policy="fixed_compiled_rz", **selector_kw)
    check("fixed_selector_requires_all_rounds", [r["r_m"] for r in selected] == [2,2])

    ns["ALPHA_POLICIES"] = ("uniform", "cost_sensitivity_weighted")
    alpha_calls = []
    def alpha_fixture(*args, maximum_round_index, alpha_by_axis, **kwargs):
        alpha_calls.append(dict(alpha_by_axis))
        output = []
        for i in range(maximum_round_index+1):
            r = dict(row)
            r.update(round_index=i, q_m=1 << i)
            r["axes"] = {b:dict(a, alpha=alpha_by_axis[(i,b)], predicted_rz_count_per_interrogation=(1.0 if i==0 else 3.0)) for b,a in row["axes"].items()}
            output.append(r)
        return output
    ns["_select_schedule"] = alpha_fixture
    ablation_kw = dict(ld=12, pf_coefficient=0, cost_models=[artificial_provider], beta_profile_label="ARTIFICIAL_ONLY",
                       beta_profile=(0,0,.2), schedule_policy="round_compiled_rz", alpha_policy="cost_sensitivity_weighted",
                       maximum_round_index=1, delta_time=1, beta_rpe=.2, alpha_total=.05, rte_seed=0)
    converged = ns["evaluate_ablation_scenario"](Preparation(), maximum_alpha_iterations=3, **ablation_kw)
    check("alpha_iteration_reselects_and_checks_final_schedule", converged["alpha_converged"] and len(alpha_calls)>=3)
    check("alpha_allocation_uses_cost_and_coordinate_weights", math.isclose(converged["rounds"][0]["axes"]["cosine"]["alpha"], .00625))
    incomplete = ns["evaluate_ablation_scenario"](Preparation(), maximum_alpha_iterations=1, **ablation_kw)
    check("alpha_iteration_limit_does_not_claim_convergence", not incomplete["alpha_converged"])
    final_calls = []
    def changed_final_schedule(*args, **kwargs):
        final_calls.append(True)
        rows = alpha_fixture(*args, **kwargs)
        if len(final_calls)==1:
            for r in rows:
                for a in r["axes"].values(): a["predicted_rz_count_per_interrogation"] = 1.0
        return rows
    ns["_select_schedule"] = changed_final_schedule
    inconsistent = ns["evaluate_ablation_scenario"](Preparation(), maximum_alpha_iterations=3, **ablation_kw)
    check("final_schedule_change_revokes_apparent_alpha_convergence", not inconsistent["alpha_converged"] and len(final_calls)==2)

    # Execute exact grid/refinement controller against synthetic changing schedules.
    class ArtificialGrid:
        @staticmethod
        def linspace(a,b,n):
            return [a + (b-a)*i/(n-1) for i in range(n)]
        @staticmethod
        def asarray(values):
            return list(values)
    ns["np"] = ArtificialGrid
    ns["MAXIMUM_ALPHA_ITERATIONS"] = 20
    extract(blobs["research_direction_decision_cost"], ["_scenario_summary", "_grid_search", "_evaluate_profile"], ns)
    profile = ns["_evaluate_profile"]
    seen = []
    ns["evaluate_ablation_scenario"] = lambda *a, **k: seen.append(k) or k
    profile(None, [], ld=3, delta_time=1, maximum_round_index=1, pf_coefficient=1,
            beta_pf=.01, beta_rte=.02, beta_rpe=.4, alpha_total=.05, rte_seed=0)
    check("profile_recomputes_stat_budget_for_updated_beta", math.isclose(seen[-1]["beta_profile"][2], .37) and seen[-1]["schedule_policy"] == "round_compiled_rz")
    raises("profile_rejects_nonpositive_stat_budget", RuntimeError, lambda: profile(None, [], ld=3, delta_time=1, maximum_round_index=1, pf_coefficient=1, beta_pf=.4, beta_rte=0, beta_rpe=.4, alpha_total=.05, rte_seed=0))
    ns["MAXIMUM_ALPHA_ITERATIONS"] = 20
    schedules_seen = set()
    def fake_profile(*args, beta_pf, beta_rte, beta_rpe, ld, **kwargs):
        used_pf = .001 if beta_pf >= .002 else .0005
        used_rte = 0 if ld == 12 else (.0002 if beta_rte >= .0005 else .00002)
        if used_pf > beta_pf or used_rte > beta_rte:
            raise RuntimeError("ARTIFICIAL_INFEASIBLE_UPDATED_SCHEDULE")
        schedule = "artificial_r1" if beta_pf >= .002 else "artificial_r2"
        schedules_seen.add(schedule)
        return dict(beta_pf_budget=beta_pf, beta_rte_budget=beta_rte, beta_stat_budget=beta_rpe-beta_pf-beta_rte,
                    maximum_empirical_pf_phase_proxy=used_pf, maximum_finite_rte_phase_bound=used_rte,
                    total_compiled_rz_point_estimate=beta_pf+beta_rte,
                    alpha_converged=True, conservative_calibration_95_half_width=0,
                    scenario_intervals={"local_5_percent_plus_calibration": [0,1]}, total_shots=2,
                    selected_r_k_pairs=[schedule], final_round_cost_fraction=1, last_three_round_cost_fraction=1)
    ns["_evaluate_profile"] = fake_profile
    grid_kw = dict(delta_time=1, maximum_round_index=1, pf_coefficient=1, beta_rpe=.4, alpha_total=.05, rte_seed=0, progress=None)
    grid = ns["_grid_search"](None, [], ld=3, **grid_kw)
    best = grid["best"]
    check("refinement_rechecks_updated_schedule_constraints", best["beta_pf_budget"] >= best["maximum_empirical_pf_phase_proxy"] and best["beta_rte_budget"] >= best["maximum_finite_rte_phase_bound"] and len(schedules_seen)==2)
    check("coarse_fine_and_boundary_executed", grid["coarse_grid_shape"] == [32,32] and grid["fine_grid_shape"] == [25,25] and grid["constraint_boundary_refined_count"] > 0)
    deterministic = ns["_grid_search"](None, [], ld=12, **grid_kw)
    check("tail_free_boundary_keeps_zero_RTE", deterministic["best"]["beta_rte_budget"] == 0)
    ns["_evaluate_profile"] = lambda *a, **k: dict(alpha_converged=False)
    raises("nonconverged_candidates_are_excluded", RuntimeError, lambda: ns["_grid_search"](None, [], ld=3, **grid_kw))
    del sys.modules[module.__name__]
    return checks


def contracts(objects):
    """Copy saved values; inspect sums/constraints only, never recompute shots/cost."""
    rows = []
    groups = [("wp04", str(i), s, objects["wp04"]["configuration"]["delta_time"]) for i,s in enumerate(objects["wp04"]["scenarios"])]
    for family in ("wp01_v2", "wp11"):
        candidates = objects[family]["candidates"] if family == "wp01_v2" else objects[family]["coherent_reoptimization"]["candidates"]
        groups += [(family,k,v["best"],v["delta_time"]) for k,v in candidates.items()]
    counters = Counter()
    for family,key,s,delta in groups:
        budgets = {k:s[k] for k in ("beta_pf_budget", "beta_rte_budget", "beta_stat_budget")}
        beta_exact = sum(map(Fraction.from_float, budgets.values()), Fraction())
        alpha_values = [a["alpha"] for r in s["rounds"] for a in r["axes"].values()]
        checks = dict(
            beta_sum_matches_with_feasibility_tolerance=abs(float(beta_exact)-.4) <= 1e-15,
            beta_sum_within_represented_budget_without_tolerance=beta_exact <= Fraction.from_float(.4),
            alpha_sum_matches_saved_total=math.isclose(math.fsum(alpha_values), s["alpha_total_allocated"], rel_tol=0, abs_tol=1e-15),
            alpha_exact_sum_within_represented_total=sum(map(Fraction.from_float, alpha_values),Fraction()) <= Fraction.from_float(.05),
            saved_round_phase_usages_within_allocations=all(r["empirical_pf_phase_proxy"] <= s["beta_pf_budget"] and (r["finite_rte_phase_bound"] or 0) <= s["beta_rte_budget"] for r in s["rounds"]),
            saved_coordinate_widths_positive=all(math.isfinite(r["epsilon_coordinate"]) and r["epsilon_coordinate"]>0 for r in s["rounds"]),
            saved_shot_fields_positive_integers=all(type(a["shots"]) is int and a["shots"]>0 for r in s["rounds"] for a in r["axes"].values()),
        )
        for name,passed in checks.items(): counters[name+":"+str(passed)] += 1
        counters[family+"_scenarios"] += 1
        counters[family+"_rounds"] += len(s["rounds"])
        if family == "wp11":
            compiler=objects["wp11"]["configuration"]["compiler"]
        elif family == "wp01_v2":
            compiler=objects["wp05a"]["configuration"]["compiler"]
        else:
            compiler=objects["wp04_provider_metadata"][0]["compiler_settings"]
        rows.append(dict(family=family, candidate=key, ld=s["ld"], delta_time=delta,
                         system=dict(molecule="linear_H4",geometry_angstrom=1.0,basis="STO-3G",df_rank=12),
                         compiler=compiler, state_preparation_included=False,
                         circuit_cost_scope="single_hadamard_interrogation_without_state_preparation",
                         saved_result=RESULTS[family], beta=budgets, alpha_total_allocated=s["alpha_total_allocated"],
                         schedule_policy=s["schedule_policy"], alpha_policy=s["alpha_policy"], alpha_converged=s["alpha_converged"],
                         rounds=s["rounds"], saved_field_checks=checks,
                         guarantee_status_in_saved_rounds=any("guarantee_status" in r for r in s["rounds"])))
    return rows, dict(counters)


def saved_physical_join(objects):
    """Join existing rows and compare saved bounds; no signal/shot evaluation."""
    grid=objects["signal_grid"]
    selected=objects["wp11"]["coherent_reoptimization"]["best_by_ld"]["3"]
    by_key={(p["q_m"],p["r_m"],p["K_m"]):p for p in grid["points"]}
    joined=[]
    for row in selected["best"]["rounds"]:
        key=(row["q_m"],row["r_m"],row["K_m"])
        point=by_key.get(key)
        if point is None:
            joined.append(dict(key=key,status="MISSING")); continue
        state=next(s for s in point["state_results"] if s["state_label"]=="physical_df_ground_state")
        joined.append(dict(key=key,status="PARAMETER_MATCH_PREPARATION_NOT_IDENTICAL",
                           physical_saved_state=state,
                           wp11_saved_model_radius_lower=row["conservative_radius_lower_bound"],
                           physical_pf_error_within_wp11_budget=state["pf_vs_exact_phase_error"] <= selected["best"]["beta_pf_budget"],
                           physical_rte_bound_within_wp11_budget=state["finite_rte_phase_error_bound"] <= selected["best"]["beta_rte_budget"],
                           observed_physical_radius_above_wp11_model_lower=state["observed_attenuated_radius"] >= row["conservative_radius_lower_bound"]))
    return dict(operation="SAVED_VALUE_ID_JOIN_AND_INEQUALITY_ONLY", rows=joined,
                delta_matches=grid["request"]["delta_time"]==selected["delta_time"],
                grid_request=grid["request"], grid_states=grid["states"], grid_summary=grid["summary"],
                grid_hamiltonian=grid["hamiltonian"],
                grid_threshold_metadata={k:v for k,v in grid["partial_s2"].items() if "threshold" in k},
                same_preparation_verified=False,
                grid_preparation_hash=grid["hamiltonian"]["preparation_hash"],
                wp04_ld3_cost_preparation_hashes=sorted({p["preparation_fingerprint"] for p in objects["wp04_provider_metadata"] if p["ld"]=="3"}),
                preparation_identity_comparisons=[dict(pair=p["pair"],hamiltonian_matches=p["hamiltonian_fingerprint"]==grid["hamiltonian"]["hamiltonian_hash"],partition_matches=p["partition_fingerprint"]==grid["hamiltonian"]["partition_hash"],preparation_matches=p["preparation_fingerprint"]==grid["hamiltonian"]["preparation_hash"]) for p in objects["wp04_provider_metadata"] if p["ld"]=="3"],
                limitation="Parameter-ID match only: grid coefficient_atol=1e-12 drops 108 terms; WP11 accounting SOURCE uses default coefficient_atol=0. Hashes differ. Numeric observations support only the saved grid, not identity with WP11, exact eigenstate/radius certification or WP11 statistical success. No transfer-error recalculation performed.")


def main():
    blobs, inventory, objects = {}, [], {}
    cache = {}
    def read_blob(commit, path, label):
        if path.endswith(".npz") or not path.startswith(("src/", "scripts/", "tests/", "docs/", "artifacts/")):
            raise ValueError("NON_METADATA_INPUT_FORBIDDEN")
        key = (commit,path)
        if key not in cache:
            raw = git("show",commit+":"+path)
            cache[key] = raw
            inventory.append(dict(label=label, commit=commit, path=path, git_blob=git("rev-parse",commit+":"+path).decode().strip(), bytes=len(raw), sha256=digest(raw)))
        return cache[key]
    for label,(commit,path) in SPECS.items():
        raw = read_blob(commit,path,label)
        blobs[label] = raw.decode()
        if path.endswith(".json"):
            objects[label] = json.loads(raw)
    provider_metadata=[]
    for ld,pairs in objects["wp04"]["source_evidence"]["compiled_cost_calibrations"].items():
        for pair,refs in pairs.items():
            entry=dict(ld=ld,pair=pair,files={})
            for kind,ref in refs.items():
                raw=read_blob(FIXED,ref["path"],"wp04_provider:"+ld+":"+pair+":"+kind)
                if digest(raw)!=ref["sha256"]:
                    raise ValueError("WP04_PROVIDER_HASH_MISMATCH")
                entry["files"][kind]=ref
                if kind=="dataset":
                    dataset=json.loads(raw)
                    entry.update(compiler_settings=dataset["compiler_settings"],circuit_scope=dataset["circuit_scope"],
                                 hamiltonian_fingerprint=dataset["hamiltonian_fingerprint"],
                                 partition_fingerprint=dataset["partition_fingerprint"],
                                 preparation_fingerprint=dataset["preparation_fingerprint"])
            provider_metadata.append(entry)
    objects["wp04_provider_metadata"]=provider_metadata
    if not all(p["compiler_settings"]==provider_metadata[0]["compiler_settings"] for p in provider_metadata):
        raise ValueError("WP04_COMPILER_CONTEXT_MISMATCH")
    if not all(objects[k]["configuration"]["compiler"]==objects["wp05a"]["configuration"]["compiler"] for k in ("wp05a","wp05b","wp05br")):
        raise ValueError("WP01_PROVIDER_COMPILER_CONTEXT_MISMATCH")
    fingerprints = {}
    for label in RESULTS:
        o = objects[label]
        unsigned = {k:v for k,v in o.items() if k != "content_fingerprint"}
        observed = digest(canonical(unsigned))
        fingerprints[label] = dict(saved=o.get("content_fingerprint"), canonical_sha256=observed, matches=observed==o.get("content_fingerprint"))
    grid_unsigned={k:v for k,v in objects["signal_grid"].items() if k!="validation_fingerprint"}
    grid_hash=digest(canonical(grid_unsigned))
    fingerprints["signal_grid"]=dict(saved=objects["signal_grid"]["validation_fingerprint"],canonical_sha256=grid_hash,matches=grid_hash==objects["signal_grid"]["validation_fingerprint"])
    lineage = []
    for label in RESULTS:
        o=objects[label]; p=o.get("provenance",{})
        for path,wanted in p.get("source_sha256",{}).items():
            # Declared paths only; read at fixed commit, never from private run directories.
            raw=read_blob(FIXED,path,"declared_source:"+label)
            lineage.append(dict(result=label, path=path, saved_sha256=wanted, fixed_sha256=digest(raw), matches=digest(raw)==wanted,
                                recorded_git_commit=p.get("git_commit"), evidence_status=p.get("evidence_status")))
    rows,counts=contracts(objects)
    synthetic=synthetic_checks(blobs)
    physical_join=saved_physical_join(objects)
    preparation_def=next(n for n in ast.parse(blobs["df_partial_s2"]).body if isinstance(n,ast.FunctionDef) and n.name=="prepare_df_partial_s2")
    defaults={a.arg:d for a,d in zip(preparation_def.args.kwonlyargs,preparation_def.args.kw_defaults)}
    physical_join["wp11_accounting_coefficient_atol_source_default"]=ast.literal_eval(defaults["coefficient_atol"])
    expected_snapshot=next(ast.literal_eval(n.value) for n in ast.parse(blobs["research_direction_full_opt2"]).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="EXPECTED_SNAPSHOT_SHA256" for t in n.targets))
    joins={
        "wp04_signal_grid_sha256": objects["wp04"]["source_evidence"]["ld3_selected_schedule_signal_grid"]["sha256"] == digest(cache[SPECS["signal_grid"]]),
        "wp04_signal_grid_fingerprint": objects["wp04"]["source_evidence"]["ld3_selected_schedule_signal_grid"]["validation_fingerprint"] == objects["signal_grid"]["validation_fingerprint"],
        "wp11_initial_aggregate_fingerprint": objects["wp11_audit"]["source_artifacts"]["initial_aggregate_fingerprint"] == objects["initial_aggregate"]["aggregate_fingerprint"],
        "wp11_extension_aggregate_fingerprint": objects["wp11_audit"]["source_artifacts"]["extension_aggregate_fingerprint"] == objects["extension_aggregate"]["aggregate_fingerprint"],
        "wp04_wp01_declared_snapshot_sha256": objects["wp04"]["source_evidence"]["snapshot"]["sha256"] == objects["wp01_v2"]["source_evidence"]["snapshot"]["sha256"],
        "wp11_source_expected_snapshot_sha256": expected_snapshot==objects["wp04"]["source_evidence"]["snapshot"]["sha256"],
    }
    for source in ("wp05a","wp05b","wp05br"):
        joins["wp01_"+source+"_sha256"]=objects["wp01_v2"]["source_evidence"][source]["sha256"] == digest(cache[SPECS[source]])
    if not all(joins.values()):
        raise ValueError("METADATA_JOIN_MISMATCH")
    fingerprint_mismatches=[k for k,v in fingerprints.items() if not v["matches"]]
    if fingerprint_mismatches:
        raise ValueError("ARTIFACT_FINGERPRINT_MISMATCH:"+str(fingerprint_mismatches))
    wp11=objects["wp11"]
    best=wp11["coherent_reoptimization"]["best_by_ld"]
    snapshot={ld:dict(delta_time=v["delta_time"], maximum_round_index_M=v["maximum_round_index_M"],q_max=v["q_max"],
                      **{k:v["best"][k] for k in ["beta_pf_budget","beta_rte_budget","beta_stat_budget","alpha_total_allocated","maximum_empirical_pf_phase_proxy","maximum_finite_rte_phase_bound","minimum_conservative_radius_lower_bound","total_shots","total_compiled_rz_point_estimate","last_three_round_cost_fraction"]}) for ld,v in best.items()}
    findings=[
        dict(id="B01", status="CONDITIONAL", topic="Physical state applicability", evidence="Exact effective partial-S2 eigenstate, unit unattenuated radius, alias-free branch remain assumptions. WP04 grid has parameter-ID matches for all 18 selected WP11 PR rows, but different preparation metadata (B08); only supplemental saved-grid observations, not an exact WP11 validation."),
        dict(id="B02", status="INSUFFICIENT_EVIDENCE", topic="Guarantee classification projection", evidence="_evaluate_pair_round gates on feasible and drops guarantee_status/certification_reasons/assumptions; all_rounds_feasible is numerical, not a certificate. WP11 scope explicitly disclaims superiority/final-total-cost."),
        dict(id="B03", status="IMPLEMENTATION_BUG", topic="Tiny positive coordinate exception", evidence="Artificial coordinate 1e-200 and candidate beta_stat=1e-200 yield ZeroDivisionError when epsilon**2 underflows. Candidate only catches OverflowError. No evidence that saved selected rows enter this path; no production source/results changed."),
        dict(id="B04", status="CONDITIONAL", topic="Boundary refinement", evidence="Actual grid/refinement controller recomputes profiles and reselects schedules; synthetic schedule-change and nonconvergence checks pass. max(1e-12,abs(used)*1e-10) is numeric slack, not model headroom or global-optimum proof."),
        dict(id="B05", status="CONSISTENT", topic="Comparison fairness", evidence="WP11 uses total beta .4, alpha .05, same opt2 compiler and state-preparation-free full-Hadamard scope. Both candidates reoptimized; deterministic RTE budget zero is admissible. Separate opt1/WP04 series retained."),
        dict(id="B06", status="CONDITIONAL", topic="Represented-value budget sums", evidence="Saved beta/alpha sums are checked with feasibility tolerance and separately with exact rational sums of represented binary64 inputs. Strict overshoots, if present, do not invalidate the declared empirical screening or constitute rigorous coverage; no totals/shots/costs are repaired."),
        dict(id="B07", status="INSUFFICIENT_EVIDENCE", topic="Historical closure", evidence="Saved source hashes are compared individually. Local dirty generation metadata, mutable/missing historical dependencies, absent private checkpoints/runtime/NPZ remain separate from fixed committed snapshots; no clean scientific regeneration claimed."),
        dict(id="B08", status="MODEL_MISMATCH", topic="Physical-grid preparation mismatch", evidence="WP04 grid coefficient_atol=1e-12 drops 108 terms (saved operator error bound 4.8294649659693557e-14); WP11 accounting prepare_df_partial_s2 call uses default coefficient_atol=0. WP04 cost dataset and grid Hamiltonian/partition hashes agree for L_D3 but preparation hashes differ. Parameter matches do not establish identical operators or transfer of strict radius bounds. No scientific impact magnitude or shot/cost correction computed."),
    ]
    report=dict(schema_version="pr_beta_contract_audit_v1", audit_date_jst="2026-10-18", fixed_science_snapshot=FIXED, audit_base=BASE,
                status="TECHNICAL_AUDIT_COMPLETED_WITH_FINDINGS", scientific_certification=False,
                allowed_operations=["fixed_blob_read","hash_metadata","saved_budget_sum_and_constraint_checks","synthetic_AST_functions_with_mock_dependencies"],
                prohibited_operations_executed=dict(real_sampling=0,physical_signal_generation=0,native_build=0,compile=0,saved_shot_recalculation=0,cost_reaccounting=0,real_reoptimization=0,monte_carlo=0),
                inventory=inventory, fingerprints=fingerprints, historical_source_hash_comparison=lineage,
                auditor_script_sha256=digest(Path(__file__).read_bytes()),
                metadata_joins=joins, wp11_pr_existing_physical_grid_join=physical_join,
                wp04_provider_metadata=provider_metadata,
                saved_contract_summary=counts, selected_wp11_saved_fields=snapshot,
                saved_wp11_comparison=wp11["coherent_reoptimization"]["comparison"],
                saved_wp11_scope=wp11["scope"], saved_wp11_compiler=wp11["configuration"]["compiler"],
                saved_wp11_task_counts=objects["wp11_audit"]["combined_counts"],
                synthetic_checks=synthetic, synthetic_checks_count=len(synthetic),
                findings=findings, mandatory_stop=True,
                interpretation="Author's technical checks only; no independent scientific review; reproduced defects are findings, not scientific PASS.")
    OUT.mkdir(parents=True,exist_ok=True)
    for name,obj in [("technical_validation.json",report),("beta_contracts.json",dict(schema_version="saved_beta_contract_projection_v1",operation="COPY_SAVED_FIELDS_AND_TECHNICAL_BUDGET_CHECKS_NO_SHOT_OR_COST_REDERIVATION",contracts=rows,historical_allocation_contracts=objects["allocation"]["scenarios"],physical_four_round_contract=objects["phase4"]["configuration"]))]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=report["status"], fixed_files=len(inventory), synthetic_checks=len(synthetic), contracts=counts,
                          fingerprint_mismatches=[k for k,v in fingerprints.items() if not v["matches"]],
                          historical_hash_mismatches=[dict(result=v["result"],path=v["path"]) for v in lineage if not v["matches"]],
                          findings=[dict(id=f["id"],status=f["status"]) for f in findings]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
