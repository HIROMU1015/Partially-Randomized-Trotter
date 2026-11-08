#!/usr/bin/env python3
"""Finish saved PASS summaries only: no build, simulator, compile or postcheck."""
import hashlib
import json
from pathlib import Path
import subprocess

import run_track_d_head_tail_h4 as prior

ROOT = prior.ROOT
OLD = ROOT/prior.OUT
OUT = ROOT/"artifacts/track_d_head_tail_h4/2026-11-08/run02_summary_completion"
METRICS = ("rz_count", "cx_count", "rz_depth", "cx_depth", "total_depth", "circuit_size", "qubit_count")


def snapshot():
    return {str(p.relative_to(OLD)): prior.sha(p.read_bytes()) for p in sorted(OLD.rglob("*"))
            if p.is_file() and "matplotlib-cache" not in p.parts}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    before = snapshot()
    success = False
    try:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
        paths = ("scripts/complete_track_d_head_tail_h4_summary.py",
                 "docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/SUMMARY_COMPLETION_PLAN.md")
        for p in paths:
            if (ROOT/p).read_bytes() != subprocess.check_output(["git", "-C", str(ROOT), "show", head+":"+p]):
                raise ValueError("UNFIXED_COMPLETION_SOURCE")
        prior.write(OUT/"SOURCE.json", dict(execution_source=head,
            source_files={p: prior.sha((ROOT/p).read_bytes()) for p in paths},
            original_execution_source=prior.read(OLD/"SOURCE.json")["execution_source"],
            original_run_sha256=before, scope="uncompleted saved summary only; no new scientific circuit calls"))
        failure = prior.read(OLD/"FAILURE.json")
        calls = prior.read(OLD/"CALL_COUNTS.json")
        if failure != dict(type="ValueError", error="Invalid norm order 'fro' for vectors"):
            raise ValueError("ONLY_SAVED_SCALAR_SHAPE_ERROR")
        if calls["new_P0_compile"] != 4 or calls["new_compiled_postcheck"] != 4 or calls["science_retry"]:
            raise ValueError("EXACTLY_FOUR_COMPLETED_WRAPPERS")
        if prior.read(OLD/"RAW_CHECKS.json")["status"] != "PASS" or prior.read(OLD/"RAW_EQUALITY.json")["status"] != "PASS":
            raise ValueError("SAVED_RAW_PASS")
        prior.bootstrap()
        import numpy as np
        from threadpoolctl import threadpool_limits
        from trottertracks.c1_stage3_molecular import component_check
        from trottertracks.track_d_cut_h4_reuse import checked_errors
        observations, rows, checks = {}, [], []
        for old in prior.read(OLD/"BASELINE_REUSE.json")["wrappers"]:
            rows.append(dict(configuration=old["configuration"], axis=old["axis"],
                metrics={k: old["metrics"][k] for k in METRICS}, origin="qualified saved C6a", status="PASS"))
        for method, configuration in (("symmetric_directional", "S1"), ("reference", "R1")):
            for axis in ("X", "Y"):
                name = method+"_"+axis
                post = prior.read(OLD/(name+"_post.json"))
                if post["status"] != "PASS": raise ValueError("SAVED_POSTCHECK_PASS")
                checked_errors(post["checks"])
                rows.append(dict(configuration=configuration, axis=axis, metrics={k: post["metrics"][k] for k in METRICS},
                                 origin="new original-run P0/postcheck PASS", status="PASS"))
                with np.load(OLD/"local"/(name+"_observable.npz"), allow_pickle=False) as z:
                    observations[method, axis] = z["observable"].copy()
        with np.load(OLD/"local/references.npz", allow_pickle=False) as z:
            target = z["V"].copy()
        phase = prior.read(OLD/"local/raw/conditions.json")["phase_chi"]
        def same(a, b, label, *, signal=False):
            value = component_check(a, b, 1e-10, label=label)
            if signal and value["frobenius"] > 2e-10: raise ValueError("SIGNAL_ALLOWANCE")
            checks.append(value)
        with threadpool_limits(limits=1):
            for axis in ("X", "Y"):
                same(observations["reference", axis], observations["symmetric_directional", axis], "compiled_paired_"+axis, signal=True)
            for method in ("symmetric_directional", "reference"):
                coherence = observations[method, "X"]+1j*observations[method, "Y"]
                same(coherence, target, method+":compiled_full_complex_coherence", signal=True)
                # A 1x1 MATRIX, as required by the unchanged Frobenius checker.
                # Values, tolerances and analytic chi are exactly the old call.
                same(np.array([[coherence[0, 0]]]), np.array([[np.exp(1j*phase)]]), method+":analytic_chi")
        prior.write(OUT/"PAIRED_AND_CHI.json", dict(status="PASS", checks=checks, input="saved full256 observable matrices",
            component_atol=1e-10, signal_frobenius_atol=2e-10, rtol=0, phase_fit=False, no_new_state_action=True))
        prior.write(OUT/"RESOURCES.json", rows)
        comparisons = []
        for axis in ("X", "Y"):
            t = {r["configuration"]: r["metrics"] for r in rows if r["axis"] == axis}
            comparisons.append(dict(axis=axis, **t,
                delta_S={k: t["S1"][k]-t["S0"][k] for k in METRICS},
                delta_R={k: t["R1"][k]-t["R0"][k] for k in METRICS},
                relative_before={k: t["R0"][k]-t["S0"][k] for k in METRICS},
                relative_after={k: t["R1"][k]-t["S1"][k] for k in METRICS},
                difference_in_differences={k: t["R1"][k]-t["S1"][k]-t["R0"][k]+t["S0"][k] for k in METRICS}))
        prior.write(OUT/"COMPARISONS.json", comparisons)
        if snapshot() != before: raise ValueError("ORIGINAL_RUN_CHANGED")
        prior.write(OUT/"PRESERVATION.json", dict(status="PASS", original_run_files=len(before), old_STOP_preserved=True,
            old_SOURCE_results_private_payload_preserved=True, wrapper_postcheck_reexecuted=False))
        success = True
    except BaseException as e:
        prior.write(OUT/"FAILURE.json", dict(type=type(e).__name__, error=str(e)))
        raise
    finally:
        prior.write(OUT/"CALL_COUNTS.json", dict(new_sampling=0, new_core=0, new_wrapper=0, new_raw=0,
            new_P0_compile=0, new_compiled_postcheck=0, saved_postcheck_PASS_reused=4, saved_observable_loads=4,
            completed_original_scope_aggregate_checks=6 if success else None, science_retry=0, fit_N_13round=0))
        prior.write(OUT/"STOP.json", dict(status="PASS_MANDATORY_STOP" if success else "FAIL_MANDATORY_STOP",
            mandatory_stop=True, scope="saved summary completion only", original_compile_total=4,
            original_POST4_PASS_reused=True, next_stage_authorized=False, retry=0))
    print("SAVED_SUMMARY_PASS_MANDATORY_STOP")


if __name__ == "__main__": main()
