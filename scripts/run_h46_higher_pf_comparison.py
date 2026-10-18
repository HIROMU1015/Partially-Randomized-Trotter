"""Saved-only q comparison. Defaults to scope; no numerical execution authorized.

This readiness release deliberately cannot launch native acquisition/sampling.
A future execution requires a sealed plan, exact SOURCE/input closure, explicit
approval, independent one-shot grant, and a new output directory. No retry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import signal

from h46_higher_pf_adapter import ROOT, kernel


def authorize(plan, grant):
    k = kernel()
    if plan.get("status") != "SEALED_APPROVED" or grant.get("valid") is not True:
        raise ValueError("EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED")
    if plan.get("phase") != "B_SAVED_ONLY" or grant.get("phase") != plan["phase"]:
        raise ValueError("SCOPE_REQUIRES_SEPARATE_APPROVAL")
    if not grant.get("approval_reference") or grant.get("plan_sha256") != k.digest(plan):
        raise ValueError("GRANT_PLAN_BINDING")
    if grant.get("SOURCE") != plan.get("SOURCE") or not plan.get("SOURCE"):
        raise ValueError("SOURCE_BINDING")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != plan["SOURCE"]:
        raise ValueError("SEALED_SOURCE_CHECKOUT_REQUIRED")
    if not plan.get("closure") or not plan.get("input_bindings") or not grant.get("grant_id"):
        raise ValueError("SEALED_CLOSURE_INPUT_GRANT_REQUIRED")
    for item in plan["closure"] + plan["input_bindings"]:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("CLOSURE_PATH")
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("SOURCE_OR_INPUT_CLOSURE_MISMATCH")


def compare_rows(rows, *, independent_candidates=False):
    from d0_frozen_shot_rule_v1 import u_aware_shots
    k = kernel()
    evaluated = [k.evaluate_row(row, u_aware_shots) for row in rows]
    strata = sorted({(r["system"], r["formula_id"]) for r in evaluated})
    winners = [{"system": system, "formula_id": formula,
                **k.select_q([r for r in evaluated if (r["system"], r["formula_id"]) == (system, formula)])}
               for system, formula in strata]
    differences = []
    for i, left in enumerate(evaluated):
        for right in evaluated[i+1:]:
            if left["system"] != right["system"] or left["mean_exact"] is None or right["mean_exact"] is None:
                continue
            from fractions import Fraction
            diagnostic = (k.comparison(left["cost"]["summaries"]["RZ"], right["cost"]["summaries"]["RZ"],
                                       independent_candidates=True) if independent_candidates else
                          {"status": "CANDIDATE_INDEPENDENCE_NOT_ASSERTED", "rigorous_interval": None})
            differences.append({"left": left["candidate_id"], "right": right["candidate_id"],
                                "point_difference_exact": str(Fraction(left["mean_exact"])-Fraction(right["mean_exact"])),
                                "uncertainty": diagnostic})
    return {"rows": evaluated, "selections": winners, "pairwise_diagnostics": differences, "mandatory_STOP": True,
            "native_compile_calls": 0, "sampling_calls": 0, "next_phase_started": False}


def claim_grant(ledger, grant):
    ledger.mkdir(exist_ok=True)
    marker = ledger/(hashlib.sha256(grant["grant_id"].encode()).hexdigest()+".json")
    with marker.open("x") as file:
        json.dump({"grant": grant, "state": "CONSUMED_NO_RETRY"}, file)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "READINESS_ONLY", "phase": "B_SAVED_ONLY",
                          "q_grid": [1, 2, 4, 8], "new_science_authorized": False}))
        return
    try:
        if not args.plan or not args.grant:
            raise ValueError("EXPLICIT_NUMERICAL_EXECUTION_APPROVAL_REQUIRED")
        plan, grant = json.loads(args.plan.read_text()), json.loads(args.grant.read_text())
        authorize(plan, grant)  # before loading real signal/cost inputs
        if args.output is None or args.output.exists():
            raise ValueError("UNUSED_OUTPUT_REQUIRED")
        # Persistent exclusive grant marker: STOP/partial failure never frees it.
        ledger = ROOT/".higher_pf_execution_ledger"
        claim_grant(ledger, grant)
        args.output.mkdir()
        try:
            def interrupted(signum, frame):
                raise InterruptedError(f"OWN_RUN_SIGNAL_{signum}")
            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            from h46_higher_pf_adapter import load_rows
            rows, metadata = load_rows()
            result = compare_rows(rows)
            authorize(plan, grant)  # source/input remain identical at completion
            (args.output/"result.json").write_text(json.dumps({**result, "input_inventory": metadata}, indent=2, allow_nan=False)+"\n")
        except BaseException as exc:
            (args.output/"STOP.json").write_text(json.dumps({"reason": type(exc).__name__+":"+str(exc),
                                                          "retry": False, "mandatory_STOP": True})+"\n")
            raise
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, str(exc)+"\n")


if __name__ == "__main__":
    main()
