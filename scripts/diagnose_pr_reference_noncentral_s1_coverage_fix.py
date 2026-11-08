"""Independent one-shot S1 scope: change coverage only, reuse old arithmetic.

No old main, pareto_ids, offer construction, guard, quantum or RPE code is
loaded. Fixed legacy bodies are extracted from Git; arm expressions are exact
AST copies. Each arm checks only templates that its expression references.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
import copy
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c96e0a3a31fb0db5f1ae9a8903567012e21b78f9"
INPUT_BASE = "800554fbb64c21e584bb29a1a8f52490ab4163e0"
OLD_RUN = "5fbdf0b9d7046f8fce04405757c1ee075e42fb4b"
OLD_SOURCE = "scripts/diagnose_pr_reference_noncentral_saved.py"
OLD_HELPER = "scripts/audit_pr_reference_noncentral_reuse.py"
META = "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
META_SHA = "396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027"
AREA = ROOT / "artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08"
PREFLIGHT = AREA / "PREFLIGHT.json"
OUT = AREA / "science/run01"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixed_blob(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT)


@lru_cache(None)
def load_legacy():
    old = fixed_blob(SOURCE_COMMIT, OLD_SOURCE)
    helper = fixed_blob(SOURCE_COMMIT, OLD_HELPER)
    assert sha(old) == "6346c2591d8e22d1d15571493c8796a2567f4fdefdfcc586be22e72eb068598c"
    assert sha(helper) == "b4fdd5cac3989b8084d6846ffc6b55914c4f00cf54d4479aa526b549547685fc"
    ns = dict(Counter=Counter, defaultdict=defaultdict, F=F, lru_cache=lru_cache,
        hashlib=hashlib, io=io, json=json, lzma=lzma, Path=Path,
        subprocess=subprocess, tarfile=tarfile, ROOT=ROOT, BASE=INPUT_BASE, INPUTS={})
    parsed = ast.parse(old)
    wanted = {"blob", "read", "archive", "template_weights", "rational_json"}
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in functions} == wanted
    weighted = next(n for n in ast.parse(helper).body if isinstance(n, ast.FunctionDef)
                    and n.name == "weighted_template_max")
    exec(compile(ast.Module(body=[weighted] + functions, type_ignores=[]),
                 SOURCE_COMMIT + ":S1_pure_functions_only", "exec"), ns)
    parent = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "weighted_parts")
    original_return = parent.body[-1]
    assert isinstance(original_return, ast.Return)
    arm_expressions = {k.arg: k.value for k in original_return.value.keywords}
    assert set(arm_expressions) == {"Bprime", "reference"}
    # Keep pair_max, lane and the q/r assertion unchanged. Replace only the
    # global None gate and the eager two-arm return with a scoped arm return.
    prefix = parent.body[:-2]
    assert [type(n).__name__ for n in prefix] == ["Expr", "FunctionDef", "FunctionDef", "Assert"]
    assert isinstance(parent.body[-2], ast.If)
    arm_functions, expression_hashes = {}, {}
    for arm, expr in arm_expressions.items():
        node = copy.deepcopy(parent)
        node.name = "weighted_" + arm
        node.body = copy.deepcopy(prefix) + [ast.Return(value=copy.deepcopy(expr))]
        assert ast.dump(node.body[-1].value) == ast.dump(expr)
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(module, SOURCE_COMMIT + ":weighted_parts:" + arm, "exec"), ns)
        arm_functions[arm] = ns[node.name]
        expression_hashes[arm] = sha(ast.dump(expr).encode())
    assert not {"main", "pareto_ids", "index_matches", "archive_metadata"} & ns.keys()
    evidence = dict(source_commit=SOURCE_COMMIT, old_source_sha256=sha(old),
        helper_sha256=sha(helper), unchanged_functions=sorted(wanted | {"weighted_template_max"}),
        unchanged_arm_expression_AST_sha256=expression_hashes,
        only_change="scoped coverage and per-arm evaluation; arithmetic ASTs unchanged",
        S2_functions_loaded=False)
    return ns, arm_functions, evidence


def required_templates(arm, ids):
    pair = [name + bid for bid in ids for name in ("basis:", "basis_inverse:")]
    if arm == "Bprime":
        return dict(b=["left_all", "right_all", "left_no0", "right_no0", "boundary_control", "cz"] + pair)
    if arm == "reference":
        d = ["block:0:full"] + [f"block:{j}:half" for j in range(4)]
        return dict(ref=d + ["z", "cswap"] + pair, inv=d)
    raise ValueError(arm)


def scoped_parts(q, r, b, ref, inv, alphabet, arm, arm_functions):
    maps = dict(b=b, ref=ref, inv=inv)
    required = required_templates(arm, alphabet[arm])
    missing = [dict(mapping=key, template=name,
                    reason="template_absent" if name not in maps[key] else "guard_coverage_missing")
               for key, names in required.items() for name in names
               if name not in maps[key] or maps[key][name] is None]
    # No dummy values, no fallback record, no arithmetic from the other arm.
    parts = None if missing else arm_functions[arm](q, r, b, ref, inv, alphabet)
    return dict(parts=parts, required_templates=required, missing_templates=missing)


def missing_details(missing, batches, guards):
    output = []
    for m in missing:
        contexts = []
        for batch in batches[m["mapping"]]:
            for rec in batch["records"]:
                if rec["name"] != m["template"]:
                    continue
                native = rec["native"] if m["mapping"] == "inv" else rec
                keys = {abs(float(p[0])).hex() for op, qs, p in native["lowering"] if op == "rz"}
                contexts.append(dict(context=batch.get("key", str(batch.get("context", batch.get("request")))),
                                     missing_literals=sorted(keys - guards.keys())))
        output.append(dict(**m, contexts=contexts))
    return output


def save(name, value, serializer):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     default=serializer, allow_nan=False) + "\n")


def evaluate(pref, ns, arm_functions, evidence):
    raw = (ROOT / META).read_bytes()
    assert sha(raw) == META_SHA
    meta = json.loads(lzma.decompress(raw))
    assert meta["base_commit"] == INPUT_BASE and not meta["current_policy_changed"]
    c0path = "artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"
    c0 = ns["archive"](c0path)
    assert sha(ns["blob"](c0path)) == meta["source_inputs"][c0path]["sha256"]
    guards = {r["theta_hex"]: F(r["strict_error_upper"]) for r in c0}
    # Read only CURRENT certificates. No offers, record ranking or T fields.
    for entry in meta["entries"]:
        current = entry["current_word"]
        if current["kind"] == "N4_85_composite":
            guards[entry["target_literal"]] = F(current["Q_saved"]) + F(current["G_saved"])
    batches = dict(b=ns["archive"]("artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz"),
        ref=[x for x in ns["archive"]("artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz")
             if x["request"][0] == "reference"],
        inv=ns["archive"]("artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz"))
    weights = {key: ns["template_weights"](value, guards, inverse=key == "inv") for key, value in batches.items()}
    condition = ns["read"]("artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json")["condition"]
    assert (condition["m"], condition["q"], condition["r"], condition["bits"]) == (17, 131072, 32, 88)
    assert {a: v["saved_shots_per_axis"] for a, v in condition["arms"].items()} == {"Bprime": 626, "reference": 630}
    tail = tuple(ns["read"]("artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE1_STRUCTURE.json")["source_basis_probabilities"])
    full = {arm: tuple(k[6:] for k in weights[key] if k.startswith("basis:"))
            for arm, key in (("Bprime", "b"), ("reference", "ref"))}
    assert tail and all(set(tail) <= set(ids) for ids in full.values())
    old = ns["read"]("artifacts/pr_reference_post_v0_v1_budget/2026-11-08/RESULTS.json")["fixed_noncentral_loads"]
    rows, coverage = [], []
    for variant, alphabet in (("weighted_all_catalogue_basis", full),
                              ("weighted_SOURCE_RTE_basis", {arm: tail for arm in full})):
        for arm in ("Bprime", "reference"):
            case = scoped_parts(131072, 32, weights["b"], weights["ref"], weights["inv"], alphabet, arm, arm_functions)
            coverage.append(dict(variant=variant, arm=arm, alphabet=list(alphabet[arm]),
                required_templates=case["required_templates"],
                missing=missing_details(case["missing_templates"], batches, guards)))
            parts = case["parts"]
            if parts is not None:
                assert all(isinstance(v, F) and v >= 0 for v in parts.values())
            for axis in "XY":
                total = None if parts is None else sum(parts.values(), F(0))
                budget = F.from_float(condition["arms"][arm]["joint_synthesis_error_retaining_saved_N_inclusive_upper"])
                previous = F(next(v["known_noncentral_certificate_load"] for v in old if (v["arm"], v["axis"]) == (arm, axis)))
                rows.append(dict(variant=variant, arm=arm, axis=axis, parts=parts, noncentral_upper=total,
                    saved_V1_noncentral_upper=previous, no_worse_than_saved_V1=None if total is None else total <= previous,
                    saved_joint_budget=budget, residual=None if total is None else budget-total,
                    central_scalar_event_emitter_shell_domain_terminal="UNQUALIFIED_NOT_ZERO", m17_RPE="UNQUALIFIED"))
    assert len(rows) == 8
    monotonic = []
    for arm in ("Bprime", "reference"):
        for axis in "XY":
            a, b = [v for v in rows if (v["arm"], v["axis"]) == (arm, axis)]
            relation = None if a["noncentral_upper"] is None or b["noncentral_upper"] is None else b["noncentral_upper"] <= a["noncentral_upper"]
            if relation is not None:
                assert relation
            monotonic.append(dict(arm=arm, axis=axis, SOURCE_le_all_catalogue=relation))
    for path, info in pref["fixed_inputs"].items():
        assert sha(ns["blob"](path)) == info["sha256"]
    serializer = ns["rational_json"]
    save("RESULTS.json", dict(status="S1_SCOPED_COVERAGE_ONE_SHOT_COMPLETE", rows=rows,
        template_weights=weights, full_joint_RPE="UNQUALIFIED", S2_reexecuted=False,
        global_T="NOT_EVALUATED", new_policy_adopted=False), serializer)
    save("COVERAGE.json", coverage, serializer)
    save("CHECKS.json", dict(status="FIXED_S1_SCOPE_COMPLETE", aggregate_rows=8,
        evaluated_rows=sum(v["noncentral_upper"] is not None for v in rows),
        comparison_only_when_both_defined=monotonic, nonnegative_finite_Fraction_bounds=True,
        S2_functions_loaded=False, S2_offer_construction_or_frontier_calls=0,
        new_guard_synthesis_compile_RPE_T_calls=0, old_outputs_changed=False), serializer)
    return rows


def main():
    pref = json.loads(PREFLIGHT.read_text())
    assert not OUT.exists(), "Independent one-shot output already exists; no retry/resume"
    assert sha(Path(__file__).read_bytes()) == pref["SOURCE_sha256"]
    for path, info in pref["protected_SOURCE"].items():
        assert sha((ROOT / path).read_bytes()) == info["sha256"]
    assert sha((ROOT / META).read_bytes()) == META_SHA
    ns, arm_functions, evidence = load_legacy()
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    serializer = ns["rational_json"]
    save("SOURCE_INPUTS.json", dict(frozen_SOURCE_sha256=pref["SOURCE_sha256"], reuse=evidence,
        input_base=INPUT_BASE, metadata_sha256=META_SHA, fixed_inputs=pref["fixed_inputs"],
        old_run=OLD_RUN, preflight_sha256=sha(PREFLIGHT.read_bytes())), serializer)
    save("ENVIRONMENT.json", pref["environment"], serializer)
    try:
        rows = evaluate(pref, ns, arm_functions, evidence)
        assert sha(Path(__file__).read_bytes()) == pref["SOURCE_sha256"]
        save("STOP.json", dict(status="MANDATORY_STOP", independent_one_shot=True,
            reason="Fixed corrected S1 scope complete", S2_reexecuted=False,
            next_owner="GPT/user", elapsed_seconds=time.monotonic()-started), serializer)
        print(json.dumps(dict(status="S1_SCOPED_COVERAGE_ONE_SHOT_COMPLETE", aggregate_rows=len(rows),
            evaluated_rows=sum(v["noncentral_upper"] is not None for v in rows), S2_reexecuted=False)))
    except Exception:
        save("FAILURE.json", dict(status="FAILED_ONE_SHOT_NO_RETRY", traceback=traceback.format_exc()), serializer)
        save("STOP.json", dict(status="MANDATORY_STOP", reason="One-shot failed; preserve SOURCE and results",
                               next_owner="GPT/user", S2_reexecuted=False), serializer)
        raise


if __name__ == "__main__":
    main()
