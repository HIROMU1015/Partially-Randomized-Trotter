"""Future one-shot PM-1 science path. Importing this module is science-free.

No authorization is supplied by the preparation step. All scientific imports
and NPZ access are behind validate_execution_gate; no retry/resume exists.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import time

from . import pm1_discard_contract as c


def environment_identity():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in c.ENVIRONMENT if name != "python"}}


def signal_record(candidate, mean, target):
    c.require(all(math.isfinite(x) for x in (mean.real, mean.imag, target.real, target.imag)), "nonfinite signal")
    c.require(abs(mean) <= 1.0 + 1e-10, "deterministic signal radius gate failed")
    bias = {"real": abs(mean.real - target.real), "imag": abs(mean.imag - target.imag)}
    allowance = {k: 0.05 / math.sqrt(2.0) - v for k, v in bias.items()}
    shots = {k: None if a <= 0.0 else math.ceil(2.0 / a**2 * math.log(2.0 / 0.025)) for k, a in allowance.items()}
    eligible = all(v is not None for v in shots.values())
    record = {"candidate_id": candidate["candidate_id"], "candidate_fingerprint": candidate["candidate_fingerprint"],
              "corrected_mean": {"real": mean.real, "imag": mean.imag}, "exact_full_H_target": {"real": target.real, "imag": target.imag},
              "normalization_multiplier": 1.0, "axis_bias": bias, "axis_allowance": allowance,
              "axis_shots": shots, "accuracy_eligible": eligible,
              "total_shots": sum(shots.values()) if eligible else None,
              "total_discard_plus_pf_bias_abs": abs(mean - target), "pure_discard_bias_abs": None,
              "pure_pf_bias_abs": None, "random_actions": 0, "finite_truncation_bias_abs": 0.0}
    record["signal_record_fingerprint"] = c.fingerprint(record)
    return record


def validate_execution_gate(root, plan, authorization, plan_bytes, authorization_relative):
    """Fail before source imports, NPZ metadata access, registry or output creation."""
    c.validate_plan(plan)
    c.require(plan["source_commit"] is not None and plan["source_binding"] == "COMMIT_BLOB_BOUND", "source commit not frozen")
    c.require(authorization.get("schema_version") == c.AUTH_VERSION and
              authorization.get("status") == "PM1_EXECUTION_AUTHORIZED_ONCE", "separate PM1 authorization required")
    c.require(authorization.get("source_commit") == plan["source_commit"], "authorization source mismatch")
    c.require(authorization.get("plan_fingerprint") == plan["plan_fingerprint"] and
              authorization.get("plan_sha256") == c.hashlib.sha256(plan_bytes).hexdigest(), "authorization plan mismatch")
    c.require(c.canonical(authorization.get("resource_caps")) == c.canonical(c.CAPS) and
              c.canonical(authorization.get("permissions")) == c.canonical(c.PERMISSIONS),
              "authorization permissions or resource cap mismatch")
    c.require(authorization.get("final_review_approved") is True, "final review barrier not passed")
    c.require(authorization.get("fixed_project_root") == str(root.resolve()), "fixed root mismatch")
    out = Path(authorization.get("output_relative", ""))
    c.require(not out.is_absolute() and ".." not in out.parts and len(out.parts) >= 4 and
              out.parts[:3] == ("artifacts", "resource_applicability", "pr2_pm1_discard_execution"), "invalid fixed output")
    head = c.git(root, "rev-parse", "HEAD").decode().strip()
    c.require(head != plan["source_commit"], "authorization must follow source commit")
    c.git(root, "merge-base", "--is-ancestor", c.EVIDENCE_COMMIT, plan["source_commit"])
    c.git(root, "merge-base", "--is-ancestor", plan["source_commit"], head)
    c.require(c.read_text_bytes(root, authorization_relative) == c.git(root, "show", f"{head}:{authorization_relative}"),
              "authorization not committed or modified")
    c.require(c.source_inventory(root, plan["source_commit"]) == plan["source_sha256"], "source identity mismatch")
    values, audit = c.load_saved_inputs(root)
    refs, target = c.frozen_references(values)
    c.require(c.make_plan(refs, target, plan["source_sha256"], audit, source_commit=plan["source_commit"]) == plan,
              "saved comparison identity mismatch")
    c.require(environment_identity() == c.ENVIRONMENT, "environment identity mismatch")
    c.require(all(os.environ.get(k) == v for k, v in c.THREAD_ENVIRONMENT.items()), "thread/environment gate failed")
    c.require(os.environ.get("PYTHONPATH") == "src", "PYTHONPATH must be replaced with src")
    return out


class _DevelopmentBackend:
    """Private production adapter; constructed only AFTER all authorization gates."""
    def __init__(self, root, audit):
        # These existing helper functions are reused, never edited or run as runners.
        from trotterlib import pr2_matched_accuracy_m1_execution as m1
        self.m1 = m1
        counters = {"development_raw_hash_checks": 0, "development_npz_loads": 0}
        audit["development_load_attempts"] += 1
        try:
            self.hamiltonian, self.state, self.metadata = m1._load_development_only(root, counters)
        finally:
            audit["development_hash_attempts"] = counters["development_raw_hash_checks"]
            audit["development_load_calls"] = counters["development_npz_loads"]
        audit["development_loads_completed"] = 1
        c.require(self.hamiltonian.n_qubits == 8 and self.hamiltonian.n_blocks == 12, "development model gate failed")
        # Only prefix 0..4 and the same one-body/scalar terms are needed.
        self.one_body, self.fragments, self.reconstruction = m1._dense_block_operators(
            self.hamiltonian.select_blocks(tuple(range(5))))
        self.block_cache, self.preparations = {}, {}

    def evaluate(self, candidate, target):
        import numpy as np
        rank = candidate["rank"]
        if rank not in self.preparations:
            p = self.m1._prepare_discard(self.hamiltonian, rank)
            d = self.m1._preparation_eigensystems(p, self.one_body, self.fragments, self.block_cache)
            self.preparations[rank] = p, d
        p, d = self.preparations[rank]
        evolved = self.state.copy()
        phase = np.exp(-1j * candidate["delta"] * (p.constant_coefficient + p.extracted_identity_coefficient))
        for _ in range(candidate["q"]):
            evolved = self.m1._apply_outer_step(evolved, d, delta=candidate["delta"], phase=phase, tail_action=None)
        c.require(abs(float(np.linalg.norm(evolved)) - 1.0) <= 1e-10, "state-action norm gate failed")
        return complex(np.vdot(self.state, evolved))

    def compile(self, candidate):
        from trotterlib.rte import CompilerSettings
        from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
            RPEHadamardCompiledCostBenchmarkRequest, generate_rpe_hadamard_compiled_cost_benchmark_dataset)
        request = RPEHadamardCompiledCostBenchmarkRequest(
            preparation=self.preparations[candidate["rank"]][0], delta_time=candidate["delta"],
            calibration_repetition_counts=(candidate["q"],), holdout_repetition_counts=(),
            rte_steps_per_occurrence=0, finite_taylor_order=0, rte_config=None, rte_distribution=None,
            compiler=CompilerSettings(**{**c.COMPILER, "basis_gates": tuple(c.COMPILER["basis_gates"])}),
            evaluation_method="exact", sample_count=None, seed=None,
            generation_id=f"pm1-{candidate['candidate_fingerprint']}", maximum_repetition_count=8,
            maximum_trajectories=1, maximum_samples=1, maximum_retained_trajectory_records=1,
            maximum_build_requests=2, maximum_transpile_requests=2,
            maximum_untranspiled_circuit_size=1_000_000, maximum_planned_instruction_applications=2_000_000,
            construction_policy="boundary_optimized", cache=None)
        result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
        c.require(result.dataset.complete and len(result.dataset.records) == 2, "incomplete wrapper compile; no retry")
        c.require(len(result.estimates) == 1, "unexpected compile estimate count")
        estimate = result.estimates[0]
        c.require(estimate.processed_trajectory_count == 1 and estimate.shared_axis_trajectory_set and
                  estimate.actual_build_requests == 2 and estimate.actual_cache_requests == 2 and
                  estimate.transpile_cache_hit_count == 0, "compile workload/cache gate failed")
        axes = {point.axis: point.to_dict() for point in result.dataset.records}
        return axes


def _check_axes(axes, candidate):
    c.require(set(axes) == set(c.AXES), "missing or extra compile axis")
    for axis, record in axes.items():
        c.require(record["axis"] == axis and record["status"] == "complete", "axis identity/status mismatch")
        c.require(record["evaluation_method"] == "exact" and record["state_preparation_included"] is False and
                  record["measurement_included"] is True and record["additional_control_applied"] is False,
                  "wrapper semantics mismatch")
        c.require(record["quantum_shots_executed"] == 0 and record["backend_execution_included"] is False,
                  "quantum/backend execution is forbidden")
        c.require(c.canonical(record["transpile_configuration"]) == c.canonical(c.COMPILER), "compiler mismatch")
        c.require(record["q_m"] == candidate["q"] and record["repetition_count"] == candidate["q"] and
                  record["delta_time"] == candidate["delta"] and record["t_m"] == candidate["T"] and
                  record["r_m"] == 0 and record["K_m"] == 0, "signal/cost configuration mismatch")
        c.require(all(math.isfinite(record[m]) and record[m] >= 0 for m in c.METRICS), "invalid compiled metric")


def evaluate_eight(plan, backend, audit, records=None):
    """Bounded orchestration; injected backend enables molecular-data-free tests."""
    c.validate_plan(plan)
    target = complex(plan["saved_full_H_target"]["real"], plan["saved_full_H_target"]["imag"])
    records = [] if records is None else records
    c.require(not records, "no partial ledger reuse")
    # Complete the fixed signal ledger before any candidate compilation.
    for candidate in plan["candidates"]:
        c.require(audit["signal_attempts"] < c.CAPS["candidate_signals"], "signal budget exceeded")
        audit["signal_attempts"] += 1
        s = signal_record(candidate, backend.evaluate(candidate, target), target)
        audit["signal_completed"] += 1
        records.append({"candidate": candidate, "signal": s, "axes": None, "work": None})
    for row in records:
        c.require(audit["wrapper_reservations"] + 2 <= c.CAPS["full_wrappers"], "wrapper budget exceeded")
        audit["wrapper_reservations"] += 2
        audit["unresolved_wrapper_reservations"] += 2
        axes = backend.compile(row["candidate"])
        _check_axes(axes, row["candidate"])
        row["axes"] = axes
        row["wrapper_keys"] = {axis: c.wrapper_key(plan, row["candidate"], axis) for axis in c.AXES}
        row["signal_cost_candidate_fingerprint"] = row["candidate"]["candidate_fingerprint"]
        row["signal_record_fingerprint"] = row["signal"]["signal_record_fingerprint"]
        if row["signal"]["accuracy_eligible"]:
            shots = row["signal"]["axis_shots"]
            row["work"] = {m: shots["real"] * axes["cosine"][m] + shots["imag"] * axes["sine"][m] for m in c.METRICS}
        audit["wrappers_completed"] += 2
        audit["unresolved_wrapper_reservations"] -= 2
    return records


def new_audit():
    return {"development_load_attempts": 0, "development_hash_attempts": 0, "development_load_calls": 0,
            "development_loads_completed": 0, "signal_attempts": 0, "signal_completed": 0,
            "wrapper_reservations": 0, "wrappers_completed": 0, "unresolved_wrapper_reservations": 0,
            "random_sampling": 0, "held_out_access": 0, "gpu_operations": 0, "quantum_shots": 0,
            "cache_reuse": 0, "cpu_processes": 1, "automatic_research_decision": False}


def result_payload(plan, records, audit, error=None):
    complete = error is None
    if complete:
        c.require(len(records) == 8 and audit["signal_completed"] == 8 and audit["wrappers_completed"] == 16,
                  "incomplete successful result")
    eligible = [r for r in records if r["work"] is not None]
    comparisons = [{"candidate_id": r["candidate"]["candidate_id"], "reference_candidate_id": ref["candidate_id"],
                    "primary_RZ_ratio_point_only": r["work"]["rz_count"] / ref["work"]["rz_count"]}
                   for r in eligible for ref in plan["saved_comparators"]] if complete else []
    result = {"schema_version": c.RESULT_VERSION, "status": c.COMPLETE_STATUS if complete else c.FAILURE_STATUS,
              "plan_fingerprint": plan["plan_fingerprint"], "source_commit": plan["source_commit"],
              "candidate_records": records, "saved_comparators": plan["saved_comparators"],
              "comparison_scope": plan["comparison_scope"], "point_comparisons": comparisons,
              "execution_audit": dict(audit), "failure_reason": error,
              "actual_wrapper_count_if_failure": None if audit["unresolved_wrapper_reservations"] else audit["wrappers_completed"],
              "research_decision": None, "mandatory_stop_reached": True, "next_stage_authorized": False,
              "scientific_ineligibility_is_not_implementation_failure": True,
              "additional_actions_authorized": False}
    result["result_fingerprint"] = c.fingerprint(result)
    return result


def run_once(root, plan_path, authorization_path):
    """Future production entry point. Preparation never calls it."""
    root = Path(root)
    plan_relative, auth_relative = str(Path(plan_path).relative_to(root)), str(Path(authorization_path).relative_to(root))
    plan_bytes = c.read_text_bytes(root, plan_relative)
    plan, auth = json.loads(plan_bytes), json.loads(c.read_text_bytes(root, auth_relative))
    output_relative = validate_execution_gate(root, plan, auth, plan_bytes, auth_relative)
    output = root / output_relative
    c.require(not output.exists(), "output already exists; no resume/retry")
    registry = root / "artifacts/resource_applicability/pr2_pm1_discard_execution_registry"
    registry.mkdir(parents=True, exist_ok=True)
    # Exclusive, root-local consumed authorization. Failure does not release it.
    with (registry / (c.fingerprint(auth) + ".json")).open("x", encoding="utf-8") as handle:
        json.dump({"authorization_fingerprint": c.fingerprint(auth), "output": str(output_relative), "consumed": True}, handle)
    output.mkdir(parents=True, exist_ok=False)
    audit, records, start, error = new_audit(), [], time.monotonic(), None
    try:
        backend = _DevelopmentBackend(root, audit)
        # Keep partial ledger even when a later compile fails.
        evaluate_eight(plan, backend, audit, records)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    audit["wall_seconds"] = time.monotonic() - start
    audit["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = result_payload(plan, records, audit, error)
    payloads = {"result.json": result}
    if error is None:
        payloads["PM1_COMPLETE.json"] = {"status": c.COMPLETE_STATUS, "result_fingerprint": result["result_fingerprint"], "next_stage_authorized": False}
    entries = []
    for name, payload in payloads.items():
        data = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        (output / name).write_bytes(data)
        entries.append({"path": name, "bytes": len(data), "sha256": c.hashlib.sha256(data).hexdigest()})
    (output / "manifest.json").write_text(json.dumps({"files": entries, "next_stage_authorized": False}, indent=2) + "\n")
    return result
