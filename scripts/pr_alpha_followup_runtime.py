"""Independent B1/B2 one-shot runners. No real input is read before admission.

Plans generated in this preparation are UNAPPROVED and inputs are not bound.
There is intentionally no approval/grant issuance CLI and no automatic retry.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import base64
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/trotterlib/pr_alpha_followup.py"
_spec = importlib.util.spec_from_file_location("pr_alpha_followup_core", CORE)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)  # stdlib-only; no inputs or numeric backend loaded


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def read_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            core.require(key not in out, "DUPLICATE_JSON_KEY")
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))


def validate_grant(plan, grant, stage):
    core.require(stage in ("B1", "B2") and plan.get("stage") == grant.get("stage") == stage, "STAGE_SCOPE")
    core.require(grant.get("approved_by_user") is True and grant.get("runtime_authorization") is True,
                 "EXPLICIT_SEPARATE_APPROVAL_REQUIRED")
    core.require(plan.get("schema") == "PR_ALPHA_FOLLOWUP_PLAN_V1" and plan.get("sealed") is True, "UNSEALED_PLAN")
    core.require(grant.get("plan_sha256") == digest_bytes(canonical(plan)), "PLAN_BINDING")
    core.require(grant.get("source_commit") == plan.get("source_commit") and plan.get("source_commit"), "SOURCE_BINDING")
    core.require(grant.get("one_shot") is True and grant.get("retry") is False and grant.get("auto_next_stage") is False,
                 "ONE_SHOT_NO_RETRY_NO_SCOPE_EXPANSION")
    core.require(plan.get("mandatory_stop") is True and plan.get("cross_stage_dispatch") is False, "MANDATORY_STOP")
    core.require(plan.get("scope", {}).get("contract") == core.CONTRACT, "WRONG_CONTRACT_NO_RPE_OR_H4_TRANSFER")
    core.require(plan["scope"].get("allocation") == "EQUAL_TWO_AXES", "NONUNIFORM_ALLOCATION_NOT_APPROVED")
    core.require(plan.get("inputs_bound") is True, "INPUTS_NOT_BOUND")
    core.require(plan.get("input", {}).get("sha256") and plan["input"].get("path"), "MISSING_INPUT_BINDING")
    core.require(grant.get("input_sha256") == plan["input"]["sha256"], "INPUT_GRANT_BINDING")
    if stage == "B2":
        grid = plan["scope"]["alpha_total_grid"]
        core.require(len(grid) == len(set(grid)) and "0.05" in grid and all(0 < core.rational(x) < 1 for x in grid), "ALPHA_GRID")
    core.require(grant.get("output") and grant.get("marker"), "OUTPUT_AND_MARKER_REQUIRED")
    output, marker = Path(grant["output"]), Path(grant["marker"])
    core.require(output.is_absolute() and marker.is_absolute(), "ABSOLUTE_RUN_PATHS")
    core.require(marker.parent == output.parent and marker.name == output.name + ".oneshot", "STAGE_LOCAL_MARKER")
    return output, marker


def source_gate(plan):
    for row in plan["source_files"]:
        path = Path(row["path"])
        core.require(not path.is_absolute() and ".." not in path.parts, "SOURCE_PATH")
        current = ROOT / path
        core.require(not current.is_symlink(), "SOURCE_SYMLINK")
        raw = current.read_bytes()
        original = subprocess.check_output(["git", "-C", str(ROOT), "show", plan["source_commit"] + ":" + str(path)])
        core.require(raw == original and digest_bytes(raw) == row["sha256"], "USED_SOURCE_CHANGED:" + str(path))
    runtime_gate(plan["runtime"])


def runtime_gate(profile):
    core.require(sys.version.split()[0] == profile["python_version"], "PYTHON_RUNTIME_CHANGED")
    core.require(digest_bytes(Path(sys.executable).read_bytes()) == profile["python_executable_sha256"], "PYTHON_EXECUTABLE_CHANGED")
    for saved in profile["packages"]:
        dist = importlib.metadata.distribution(saved["name"])
        core.require(dist.version == saved["version"], "NUMERIC_RUNTIME_VERSION_CHANGED")
        record = dist.read_text("RECORD")
        core.require(record is not None and digest_bytes(record.encode()) == saved["record_sha256"], "NUMERIC_RUNTIME_RECORD_CHANGED")
        for entry in dist.files:
            if entry.hash is None:
                continue
            raw = dist.locate_file(entry).read_bytes()
            expected = entry.hash
            observed = base64.urlsafe_b64encode(hashlib.new(expected.mode, raw).digest()).rstrip(b"=").decode()
            core.require(observed == expected.value, "NUMERIC_RUNTIME_FILE_CHANGED")


def load_bound_input(plan):
    spec = plan["input"]
    path = Path(spec["path"])
    core.require(path.is_absolute() and not path.is_symlink(), "INPUT_PATH")
    raw = path.read_bytes()
    core.require(len(raw) == spec["bytes"] and digest_bytes(raw) == spec["sha256"], "USED_INPUT_CHANGED")
    data = read_json(path)
    core.require(data.get("schema") in ("PR_ALPHA_BOUND_NATIVE_SCALARS_V1", "PR_ALPHA_BOUND_NATIVE_SCALARS_V2"), "CANONICAL_INPUT_SCHEMA")
    core.require(data.get("payload_kind") == plan["payload_kind"], "PAYLOAD_KIND")
    core.require(data.get("original_source_receipts"), "ORIGINAL_INPUT_PROVENANCE_REQUIRED")
    for receipt in data["original_source_receipts"]:
        core.require(receipt.get("sha256") and receipt.get("origin") and receipt.get("projection_fields"), "SCALAR_PROVENANCE")
    return data


def decode_bound_records(plan, data):
    """Called by compute after the one-shot marker is consumed, never by binders."""
    if data.get("schema") == "PR_ALPHA_BOUND_NATIVE_SCALARS_V2":
        spec = importlib.util.spec_from_file_location("pr_alpha_scalar_codec", ROOT / "src/trotterlib/pr_alpha_scalar_input.py")
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        records = [codec.decode_record(r) for r in data["records"]]
        if plan["stage"] == "B1":
            records = [codec.decode_binomial_normalization(r) for r in records]
        return records
    return data["records"]


def case_key(row):
    return row["system"], row["cell_id"], row["epsilon"], row["bias_kind"], row["allowance_factor"]


def scope_gate(plan, records):
    expected = {tuple(r) for r in plan["scope"]["case_keys"]}
    observed = {case_key(r) for r in records}
    core.require(len(observed) == len(records) and observed == expected, "EXACT_CANDIDATE_EPSILON_BIAS_SCOPE")
    core.require(len(records) == plan["budget"]["input_rows"], "INPUT_ROW_BUDGET")
    for r in records:
        if not r.get("missing_fields") and r.get("scalar_encoding") != "SAVED_RECIPES_V1":
            core.native_record(r)


class ScientificStop(ValueError):
    def __init__(self, reason, partial):
        super().__init__(reason)
        self.partial = partial


def compute(plan, data):
    records = data["records"]
    scope_gate(plan, records)
    records = decode_bound_records(plan, data)
    for r in records:
        if not r.get("missing_fields"):
            core.native_record(r)
    if plan["stage"] == "B1":
        core.require(2 * sum(not r.get("missing_fields") for r in records) <= plan["budget"]["binomial_axis_evaluations"], "B1_CALL_BUDGET")
        rows = []
        for r in records:
            if r.get("missing_fields"):
                rows.append(dict(cell_id=r["cell_id"], system=r["system"], epsilon=r["epsilon"],
                                 status="INCONCLUSIVE_SOURCE_MISSING", missing_fields=r["missing_fields"]))
            else:
                rows.append(core.binomial_case(r))
                if any(a["alpha_compliance"] == "EXCEEDS_BUDGET" for a in rows[-1]["axes"].values()):
                    raise ScientificStop("B1_DECLARED_ALPHA_EXCEEDED:" + r["cell_id"], dict(rows=rows))
        calls = sum(2 for r in rows if "axes" in r)
        core.require(calls <= plan["budget"]["binomial_axis_evaluations"], "B1_CALL_BUDGET")
        return dict(rows=rows, binomial_axis_evaluations=calls, shot_recalculations=0, cost_reaccounting=0)
    rows, groups = [], defaultdict(list)
    grid = plan["scope"]["alpha_total_grid"]
    core.require(len(records) * len(grid) == plan["budget"]["accounting_rows"], "B2_ROW_BUDGET")
    # Baseline reproduction is required before any new-alpha calculation.
    grid = sorted(grid, key=lambda value: value != "0.05")
    # One parsed snapshot and original paired costs are reused for the full grid.
    for r in records:
        for alpha in grid:
            if r.get("missing_fields"):
                row = {k:r[k] for k in ("cell_id", "system", "method", "epsilon", "bias_kind", "allowance_factor")}
                row.update(alpha_total=float(alpha), alpha_real=float(alpha)/2, alpha_imag=float(alpha)/2,
                           status="MISSING", accounting=None, G_exact=None,
                           missing_fields=r["missing_fields"], population_winner_certified=False)
            else:
                row = core.reaccount(r, alpha)
            rows.append(row)
            key = tuple(row[k] for k in ("system", "epsilon", "bias_kind", "allowance_factor", "alpha_total"))
            groups[key].append(row)
    core.require(len(rows) == plan["budget"]["accounting_rows"], "B2_ROW_BUDGET")
    comparisons = [core.compare(group) for group in groups.values()]
    switches = []
    for group_key in sorted({k[:4] for k in groups}):
        relevant = [c for c in comparisons if tuple(c[k] for k in ("system", "epsilon", "bias_kind", "allowance_factor")) == group_key]
        baseline = next(c for c in relevant if c["alpha_total"] == .05)
        for c in relevant:
            switches.append(dict(system=c["system"], epsilon=c["epsilon"], bias_kind=c["bias_kind"],
                alpha_total=c["alpha_total"], rank=c["rank"], baseline_rank=baseline["rank"],
                rank_changed=c["rank"] != baseline["rank"],
                winner_changed={f: (c["families"][f]["best"] or {}).get("cell_id") != (baseline["families"][f]["best"] or {}).get("cell_id") for f in ("PR", "PF")}))
    return dict(rows=rows, comparisons=comparisons, switches=switches,
                accounting_rows=len(rows), accounting_evaluations=sum(r["accounting"] is not None for r in rows),
                binomial_axis_evaluations=0)


def execute(plan, grant, stage, *, loader=load_bound_input, source_check=source_gate, calculator=compute):
    output, marker = validate_grant(plan, grant, stage)  # before real inputs/deps
    source_check(plan)
    core.require(not output.exists() and not marker.exists(), "ONE_SHOT_ALREADY_USED")
    data = loader(plan)
    source_hash = digest_bytes(canonical(plan["source_files"]))
    fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(dict(stage=stage, plan_sha256=grant["plan_sha256"], consumed=True), f)
    terminal = dict(stage=stage, source_commit=plan["source_commit"], source_closure_sha256=source_hash,
                    mandatory_stop=True, retry=False, auto_next_stage=False, payload_kind=plan["payload_kind"])
    try:
        output.mkdir(mode=0o700)
        started = time.perf_counter()
        result = calculator(plan, data)
        source_check(plan)
        spec = plan["input"]
        core.require(digest_bytes(Path(spec["path"]).read_bytes()) == spec["sha256"], "POST_USED_INPUT_CHANGED")
        (output / "result.json").write_bytes(canonical(result) + b"\n")
        inconclusive = any(r.get("status") in ("INCONCLUSIVE_SOURCE_MISSING", "MISSING", "UNDETERMINED", "LOG_DOMAIN_ONLY")
                           or any(a.get("alpha_compliance", "").startswith("UNDETERMINED") for a in r.get("axes", {}).values())
                           for r in result.get("rows", []))
        terminal.update(status="COMPLETE_WITH_INCONCLUSIVE_MANDATORY_STOP" if inconclusive else "COMPLETE_MANDATORY_STOP",
                        wall_seconds=time.perf_counter()-started, input_sha256=spec["sha256"],
                        real_scientific_stage_executed=plan["payload_kind"] == "REAL")
    except BaseException as exc:
        terminal.update(status="STOP", reason=type(exc).__name__ + ":" + str(exc), retry=False)
        if output.is_dir():
            if hasattr(exc, "partial"):
                (output / "partial_result.json").write_bytes(canonical(exc.partial) + b"\n")
            (output / "terminal.json").write_bytes(canonical(terminal) + b"\n")
        raise
    (output / "terminal.json").write_bytes(canonical(terminal) + b"\n")
    return terminal


def main(stage):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = read_json(args.plan)
    if not args.execute:
        print(json.dumps(dict(stage=stage, status="PREPARATION_ONLY", approved=False,
                              inputs_bound=plan["inputs_bound"], budget=plan["budget"],
                              real_scientific_calls=0)))
        return 0
    try:
        core.require(args.grant is not None, "EXPLICIT_SEPARATE_APPROVAL_REQUIRED")
        grant = read_json(args.grant)
        output, _ = validate_grant(plan, grant, stage)
        if not args.driver:
            # Parent is a native-exit witness. No shell, retry or next-stage dispatch.
            command = [sys.executable, str(Path(sys.argv[0]).resolve()), "--plan", str(args.plan.resolve()),
                       "--grant", str(args.grant.resolve()), "--execute", "--driver"]
            child = subprocess.Popen(command)
            try:
                code = child.wait()
            except BaseException:
                child.terminate(); child.wait(); raise
            if output.is_dir():
                (output / "native_wait.json").write_bytes(canonical(dict(native_exit_code=code,
                    wait_completed=True, stage=stage, next_stage_authorized=False)) + b"\n")
            return 0 if code == 0 else 1
        print(json.dumps(execute(plan, grant, stage)))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="REFUSED_OR_STOP", reason=str(exc), stage=stage,
                              retry=False, next_stage_authorized=False)), file=sys.stderr)
        return 2
