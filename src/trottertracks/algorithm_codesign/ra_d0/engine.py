"""Future saved-table one-shot, fixed anchor-first and per-stage A/B workflow.

Review source has no authorization. Synthetic fixtures use the same kernels.
"""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from .backend import evaluate
from .exact import E, kappa_upper, log_interval
from .freeze import identity, write_freeze, load_freeze, query_recipe, coverage_contexts, classification
from .guard import BudgetGuard, TechnicalFailure, ResourceCap
from .numerical import RESOURCES, build_numerical_lp, certify_nominal
from .lp import check_farkas_output


def saved_B0_vectors(data, n, ell):
    """Saved law only; retain its original midpoint rounding bias accounting."""
    output = []
    for profile in data["B0_saved_profiles"]:
        saved = profile["original_profile"]
        h = (E-F(saved["coefficient_and_strict_synthesis_bias_upper"]))/F(saved["implemented_B"])
        if h < kappa_upper(n, ell) or saved["workspace_qubits_beyond_2_system"] > 1:
            continue
        resources = {r: str(2*n*(F(saved["E_native_cost"][r])+(F(5, 2) if r == "1Q" else 0)))
                     for r in RESOURCES}
        output.append({"source": "B0_saved:"+profile["arm"]+":"+profile["epsilon"], "resources": resources})
    return output


class OneShotEngine:
    def __init__(self, table, grid, output, input_identities, permit=None, synthetic=False,
                 backend=evaluate, synthetic_caps=None):
        if synthetic:
            if table.get("domain") != "SYNTHETIC":
                raise PermissionError("real table may not enter the synthetic engine")
        elif permit is None:
            raise PermissionError("registered one-shot requires separate authorization")
        else:
            permit.assert_active()
        self.table, self.grid = table, grid
        self.output, self.inputs, self.permit = Path(output), input_identities, permit
        self.synthetic, self.backend = synthetic, backend
        self.guard = BudgetGuard("SYNTHETIC" if synthetic else "REGISTERED_SAVED_TABLE", synthetic_caps)
        self.ell = log_interval(10560)[1]
        self.rows = []
        self.point_outcomes = []
        self.stage_receipts = []
        self.last_task = None

    def solve(self, lp, call_id, baseline):
        if self.synthetic:
            lp = replace(lp, domain="SYNTHETIC")
        self.last_task = {"call_id": call_id, "baseline": baseline}
        return self.backend(lp, self.guard, call_id, baseline, self.permit)

    def save_full(self, value):
        key = identity(value)
        self.guard.write(self.output/"certificates.jsonl.gz", {"certificate_sha256": key, "value": value}, True)
        return key

    def budget_stage(self, stage, points):
        self.guard.current_phase = "BUDGET_FREEZE"
        frozen = []
        for x, n, tag in points:
            self.guard.check()
            data = self.table["tables"][x]
            saved_vectors = saved_B0_vectors(data, n, self.ell)
            minima = []
            infeasible_objectives = []
            for resource in RESOURCES:
                lp = build_numerical_lp(data, n, resource, self.ell, "B2", robust=True)
                result = self.solve(lp, f"{stage}:{x}:{n}:minimum:{resource}", "B2")
                cert = None
                minimum_status = "TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION"
                if result["status"] == 0:
                    cert = certify_nominal(data, lp, result["nominal_primal"], n,
                                           self.ell, "B2", resource)
                    if cert["certified"]:
                        minimum_status = "B2_MINIMUM_CERTIFIED_FEASIBLE"
                elif result["status"] == "CERTIFIED_INFEASIBLE":
                    # Reverify the returned exact ray against this very LP.
                    # A status string or Boolean certificate flag is insufficient.
                    try:
                        verified = check_farkas_output(lp, result["full_auxiliary_primal"])
                        if verified["certified_infeasible"]:
                            minimum_status = "B2_CERTIFIED_INFEASIBLE_AT_N"
                            infeasible_objectives.append(resource)
                    except (KeyError, ValueError, ArithmeticError):
                        pass
                full = {"kind": "B2_SINGLE_RESOURCE_MINIMUM", "stage": stage,
                        "x": x, "n": n, "tag": tag, "objective": resource,
                        "minimum_status": minimum_status,
                        "solver": result, "implementation_certificate": cert}
                key = self.save_full(full)
                if minimum_status == "TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION":
                    raise TechnicalFailure("TECHNICAL_INCONCLUSIVE_BUDGET_GENERATION")
                minima.append({"objective": resource, "status": minimum_status,
                               "certificate_sha256": key,
                               "resources": cert["resources"] if cert else None})
            if infeasible_objectives:
                point_status = "B2_POINT_CERTIFIED_INFEASIBLE"
                budget_vectors, queries = [], []
            else:
                point_status = "BUDGET_READY"
                vectors = saved_vectors + [{"source": "B2_minimum:"+m["objective"],
                                            "resources": m["resources"]} for m in minima]
                budget_vectors, queries = query_recipe(x, n, tag, vectors)
            frozen.append({"x": x, "n": n, "tag": tag, "minima": minima,
                           "point_status": point_status,
                           "certified_infeasible_objectives": infeasible_objectives,
                           "same_n_feasible_B0_profile_IDs": [v["source"] for v in saved_vectors],
                           "descriptive_B3_feasibility_query_scheduled": False,
                           "budget_vectors": budget_vectors, "queries": queries})
        path = self.output/stage/"budget_freeze.json"
        checksum = write_freeze(path, stage, frozen, self.inputs, self.guard)
        self.stage_receipts.append({"stage": stage, "path": str(path.relative_to(self.output)), "SHA256": checksum})
        return path, checksum

    def compare_stage(self, path, checksum):
        body = load_freeze(path, checksum, self.inputs)
        self.guard.current_phase = "PAIRED_COMPARISON"
        for point in body["points"]:
            if point["point_status"] == "B2_POINT_CERTIFIED_INFEASIBLE":
                outcome = {k: point[k] for k in ("x", "n", "tag", "point_status", "minima",
                           "certified_infeasible_objectives", "same_n_feasible_B0_profile_IDs")}
                outcome.update({"B3_feasibility_executed": False, "paired_queries": 0,
                                "primary_comparison_eligible": False, "strict_witness": False})
                self.guard.write(self.output/"point_outcomes.jsonl.gz", outcome, True)
                self.point_outcomes.append(outcome)
                continue
            data = self.table["tables"][point["x"]]
            for query in point["queries"]:
                self.guard.check()
                qid, n, resource, caps = query["query_id"], query["n"], query["objective"], query["caps"]
                lp2 = build_numerical_lp(data, n, resource, self.ell, "B2", caps, robust=False)
                result2 = self.solve(lp2, qid+":B2", "B2")
                lp3 = build_numerical_lp(data, n, resource, self.ell, "B3", caps, robust=True)
                result3 = self.solve(lp3, qid+":B3", "B3")
                cert3 = None
                if result3["status"] == 0:
                    cert3 = certify_nominal(data, lp3, result3["nominal_primal"], n, self.ell, "B3", resource, caps)
                    if not cert3["certified"]:
                        self.save_full({"kind": "TECHNICAL_FAILURE", "query": query, "solver_B2": result2,
                                        "solver_B3": result3, "certificate_B3": cert3})
                        raise TechnicalFailure("UNCERTIFIED_NUMERICAL_POINT")
                lower = result2["dual_certificate"]["lower"] if result2["status"] == 0 else None
                upper = F(cert3["objective_upper"]) if cert3 else None
                witness = lower is not None and upper is not None and upper < lower
                descriptive = lower is None and upper is not None
                full = {"query": query, "solver_B2": result2, "solver_B3": result3,
                        "certificate_B3": cert3}
                full_hash = identity(full)
                # Full vectors for all minima/witnesses/infeasibility/failures.
                # Other completed queries retain only the mandated compact
                # bounds/status/hash record, not full solver vectors.
                if witness or result2["status"] == "CERTIFIED_INFEASIBLE" or result3["status"] == "CERTIFIED_INFEASIBLE":
                    self.save_full(full)
                cert2_hash = identity(result2)
                cert3_hash = identity({"solver": result3, "implementation": cert3})
                row = {**query, "B2_status": "CERTIFIED_LOWER_BOUND" if lower is not None else "CERTIFIED_INFEASIBLE",
                       "B2_lower": str(lower) if lower is not None else None,
                       "B3_status": "CERTIFIED_IMPLEMENTATION" if upper is not None else "CERTIFIED_INFEASIBLE",
                       "B3_upper": str(upper) if upper is not None else None,
                       "certified_comparison": lower is not None and upper is not None,
                       "strict_witness": witness, "classification": "STRICT_DEGREE_LOCAL_WITNESS" if witness else
                       "B3_ONLY_FEASIBLE_DESCRIPTIVE" if descriptive else "NO_CERTIFIED_STRICT_WITNESS",
                       "certificate_hashes": {"B2": cert2_hash, "B3": cert3_hash, "full": full_hash}}
                self.guard.write(self.output/"paired_queries.jsonl.gz", row, True)
                self.rows.append({k: row[k] for k in ("x", "n", "tag", "strict_witness", "certified_comparison")})
            outcome = {k: point[k] for k in ("x", "n", "tag", "point_status")}
            outcome.update({"paired_queries": len(point["queries"]), "primary_comparison_eligible": True})
            self.guard.write(self.output/"point_outcomes.jsonl.gz", outcome, True)
            self.point_outcomes.append(outcome)
        # Immutable budget bytes are checked again after every complete batch.
        load_freeze(path, checksum, self.inputs)

    def stages(self):
        anchors = [(x, n, "PRIMARY_ANCHOR") for x in ("1/8", "1/4")
                   for n in self.grid["grids"][x]["anchor_shots"]]
        self.compare_stage(*self.budget_stage("P1_ANCHORS", anchors))
        successes = {x: any(r["x"] == x and r["strict_witness"] for r in self.rows)
                     for x in ("1/8", "1/4")}
        contexts = coverage_contexts(successes)
        if contexts:
            coverage = [(x, p["n"], "COVERAGE_GRID") for x in contexts
                        for p in self.grid["grids"][x]["points"] if p["tag"] == "COVERAGE_GRID"]
            if coverage:
                self.compare_stage(*self.budget_stage("P2_CONDITIONAL_COVERAGE", coverage))
        return classification(self.rows, complete=True)

    def run(self):
        reason = None
        try:
            with self.guard.enforce_OS_limits():
                status = self.stages()
        except (TechnicalFailure, MemoryError) as error:
            reason = str(error) or "MEMORY_FAILURE"
            status = "D0_TECHNICAL_INCONCLUSIVE"
            details = getattr(error, "details", None)
            if details:
                self.guard.write(self.output/"certificates.jsonl.gz", {
                    "certificate_sha256": identity(details), "value": details,
                    "kind": "TECHNICAL_FAILURE", "last_task": self.last_task},
                    compressed_append=True, terminal=True)
            self.guard.write(self.output/"technical_failure.json", {"reason": reason, "last_task": self.last_task,
                              "available_solver_vectors": "see certificates where returned; none for interrupted call"}, terminal=True)
        except Exception as error:
            reason = type(error).__name__+": "+str(error)
            status = "D0_TECHNICAL_INCONCLUSIVE"
            self.guard.write(self.output/"technical_failure.json", {"reason": reason, "last_task": self.last_task}, terminal=True)
        result = {"schema": "ra_d0_development_result_v3", "classification": status,
                  "technical_reason": reason, "stage_freezes": self.stage_receipts,
                  "completed_queries": len(self.rows), "resource_usage": self.guard.usage(),
                  "certified_comparable_queries": sum(r.get("certified_comparison", False) for r in self.rows),
                  "B2_certified_infeasible_points": sum(p["point_status"] == "B2_POINT_CERTIFIED_INFEASIBLE"
                                                       for p in self.point_outcomes),
                  "completed_budget_ready_points": sum(p["point_status"] == "BUDGET_READY"
                                                       for p in self.point_outcomes),
                  "NO_REGISTERED_WITNESS_scope": "completed comparable queries only; skipped infeasible points are not negative evidence",
                  "input_identities": self.inputs, "runs": 1, "retries": 0,
                  "mandatory_STOP": True, "next_stage_authorized": False,
                  "registered_optimization": not self.synthetic,
                  "synthesis_science_circuit_matrix_trajectory_GPU_calls": 0}
        self.guard.write(self.output/"result.json", result, terminal=True)
        return result
