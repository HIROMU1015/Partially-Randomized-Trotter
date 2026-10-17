"""Stage S AST/schema/closed-guard/identity/preservation audit; no saved science decode."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.algorithm_codesign.m1s_stage_s_v1 import spec, control
ART = ROOT / spec.ART
INDEXES = ["PROJECT_MAP.md", "VALIDATION_STATUS.md", "docs/README.md", "scripts/README.md",
           "src/trotterlib/README.md", "tests/README.md", "docs/research/研究概要・現状.md"]
MANIFEST = "artifacts/validation_manifest.json"
MANIFEST_KEY = "track_b_m1s_stage_s_preparation_20261017"
DOCS = ["docs/tracks/algorithm_codesign/m1s_stage_s_cap_edge_contract_candidate_20261017.md",
        "docs/tracks/algorithm_codesign/m1s_stage_s_open_cap_semantics_and_technical_handoff_20261017.md"]
TEST = "tests/tracks/algorithm_codesign/test_m1s_stage_s_preparation.py"
CHECKER = "scripts/tracks/algorithm_codesign/check_m1s_stage_s_preparation_static.py"
NOTE = "docs/research/研究ノート/2026-10-17_m1s_stage_s_cap_edge_preparation.md"
CHECKS = []


def check(name, ok):
    CHECKS.append({"name": name, "status": "PASS" if ok else "FAIL"})


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args],
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))


def digest(b):
    return hashlib.sha256(b).hexdigest()


def blob(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()


def load(path):
    # Only candidate/schema/new metadata: never a saved scientific result JSON.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--allow-unfrozen-source", action="store_true")
    args = parser.parse_args()
    c = load(ART / "execution_contract_candidate.json")
    cs = load(ART / "execution_contract_candidate.schema.json")
    rs = load(ART / "result_schema.json")
    for name, schema in [("candidate", cs), ("result", rs)]:
        Draft202012Validator.check_schema(schema)
        check("schema_well_formed:" + name, True)
    check("candidate_valid", Draft202012Validator(cs).is_valid(c))
    control.validate_contract(c)
    for k in ["adopted", "coverage_model_adopted", "contract_grants_execution",
              "scientific_execution_authorized", "run_ready"]:
        check("closed:" + k, c[k] is False)
    check("no_science_selection", c["selected_science"] is None)
    for k in ["cpu_time_cap_seconds", "wall_time_cap_seconds", "cpu_core_cap", "worker_cap",
              "gpu_cap", "memory_cap_bytes"]:
        check("no_artificial_cap:" + k, c["resources"][k] is None)
    check("proposal_not_adoption", c["proposed_science"] == spec.PROPOSED_SCIENCE)
    for path in [p for p in spec.SOURCE_FILES if p.endswith(".py")] + [TEST, CHECKER]:
        tree = ast.parse((ROOT / path).read_text())
        check("AST:" + path, isinstance(tree, ast.Module))
        imports = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imports.extend(i.name for i in n.names)
            elif isinstance(n, ast.ImportFrom):
                imports.append(n.module or "")
        check("no_old_engine_or_scientific_backend:" + path,
              not any(any(t in m for t in ["numpy", "scipy", "mpmath", "qiskit", "trotterlib",
                                           "m1s_saved_signal", "m1s_finite_signal", "m1s_gamma_pi3"])
                      for m in imports))
        check("no_eval_exec_compile_or_shared_process_signal:" + path,
              not any(isinstance(n, ast.Call) and ((isinstance(n.func, ast.Name) and
                      n.func.id in ["eval", "exec", "compile"]) or
                      (isinstance(n.func, ast.Attribute) and n.func.attr in ["kill", "killpg", "setrlimit"]))
                      for n in ast.walk(tree)))
    for path, name in [(spec.PACKAGE+"/adapter.py", "prepare_saved"),
                       (spec.PACKAGE+"/analysis.py", "evaluate"),
                       (spec.PACKAGE+"/control.py", "run")]:
        tree = ast.parse((ROOT/path).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        check("permit_first:" + name, ast.unparse(fn.body[0]) == "require_permit(permit)")
    ctrl = (ROOT/spec.PACKAGE/"control.py").read_text()
    authorization = ctrl.split("def authorize(", 1)[1].split("def seed_result", 1)[0]
    check("adoption_before_git_authority_and_input", authorization.index("validate_contract(c, adopted=True)") <
          authorization.index("a_path =") < authorization.index("for expected in INPUTS.values()"))
    runbody = ctrl.split("def run(", 1)[1]
    check("marker_before_science_decode_and_engine_import",
          runbody.index('open("x"') < runbody.index("from .analysis import evaluate") < runbody.index("data ="))
    check("output_failure_keeps_last_prefix", 'output / "failure.json"' in runbody and '"last_valid_prefix": "result.json"' in runbody)
    try:
        control.authorize(ROOT, spec.ART+"/execution_contract_candidate.json", None, None, None)
    except control.ExecutionClosed:
        check("candidate_guard_rejects", True)
    else:
        check("candidate_guard_rejects", False)
    check("no_analysis_imported_by_guard", "trottertracks.algorithm_codesign.m1s_stage_s_v1.analysis" not in sys.modules)
    proc = subprocess.run([sys.executable, "-B", str(ROOT/"scripts/tracks/algorithm_codesign/run_m1s_stage_s_v1.py")],
                          cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    denial = json.loads(proc.stdout) if proc.stdout else {}
    check("actual_default_CLI_closed_without_science", proc.returncode == 2
          and denial.get("status") == "EXECUTION_CLOSED" and denial.get("science_inputs_decoded") is False)
    kernels = (ROOT/spec.PACKAGE/"kernels.py").read_text()
    ktree = ast.parse(kernels)
    kfn = next(n for n in ktree.body if isinstance(n, ast.FunctionDef) and n.name == "open_cap_integer")
    check("floor_plus_one_not_endpoint_ceil", "lo.numerator // lo.denominator + 1" in ast.unparse(kfn)
          and "hi.numerator // hi.denominator + 1" in ast.unparse(kfn) and "ceil" not in ast.unparse(kfn).split('"""')[-1])
    for token in ["CAP_EDGE_STRICT_INTEGER_WITNESS", "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED",
                  "ARITHMETIC_UNRESOLVED", "UNATTAINED_SUPREMUM_NOT_NATIVE_COST_OR_RPE_SUCCESS"]:
        check("kernel_interpretation:"+token, token in kernels)
    analysis = (ROOT/spec.PACKAGE/"analysis.py").read_text()
    check("affinity_job_based_workers", "min(job_count, available)" in analysis)
    check("only_owned_Pool_termination", "pool.terminate()" in analysis and "pool.join()" in analysis)
    check("no_real_cap_grid_or_refinement", c["resources"]["max_refinements"] == 0
          and spec.PROPOSED_SCIENCE["gamma_policy"] == "OPEN_PI_OVER_THREE_CAP_LEFT_LIMIT_NO_GRID")
    # Copied generic arithmetic methods must be AST-identical to the frozen origin.
    old_tree = ast.parse(git("show", "677dd6957e0e1913b70ca5f999458835ae8a2e23:src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v2/bounds.py"))
    new_tree = ast.parse((ROOT/spec.PACKAGE/"arithmetic.py").read_text())
    def methods(tree):
        cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Bounds")
        return {n.name: ast.dump(n, include_attributes=False) for n in cl.body if isinstance(n, ast.FunctionDef)}
    om, nm = methods(old_tree), methods(new_tree)
    for name, value in nm.items():
        check("explicit_generic_reuse_AST:"+name, om.get(name) == value)
    registry = load(ART / "preserved_registry_metadata.json")["rows"]
    keys = [tuple(r[k] for k in ["split", "R", "delta", "T", "comparator"]) for r in registry]
    check("all48_unique_registry", len(keys) == len(set(keys)) == 48 and set(keys) == set(spec.contrast_keys()))
    check("preserved_old_status_and_unevaluated", all(r["old_pi3_gate_status"] == spec.old_status(k)
          and r["new_stage_s_status"] == "NOT_EVALUATED" for r, k in zip(registry, keys)))
    for g,t,n,state in [("G3",None,16,"SATISFIED"),("G6","4",8,"SATISFIED"),
                         ("G6","1",8,"NOT_SATISFIED"),("G9",None,16,"G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED")]:
        group = [r for r in registry if r["split"] == g and (t is None or r["T"] == t)]
        check("old_classification:"+g+":"+str(t), len(group) == n and all(r["old_pi3_gate_status"] == state for r in group))
    identities = load(ART / "input_identity.json")
    for e in identities["entries"]:
        raw = (ROOT/e["path"]).read_bytes()
        check("input_identity:"+e["path"], digest(raw) == e["sha256"] and len(raw) == e["bytes"])
        if "commit" in e:
            check("fixed_input_blob:"+e["role"], raw == git("show", e["commit"]+":"+e["path"]))
    check("review_exact_bytes", digest((ROOT/spec.REVIEW).read_bytes()) == spec.REVIEW_SHA256)
    identity = c["source_identity"]
    if identity is None:
        check("source_freeze_pending_allowed", args.allow_unfrozen_source)
    else:
        check("source_inventory_exact", set(identity["files"]) == set(spec.SOURCE_FILES))
        git("merge-base", "--is-ancestor", spec.BASE, identity["commit"])
        for p, e in identity["files"].items():
            raw = (ROOT/p).read_bytes()
            check("fixed_source:"+p, raw == git("show", identity["commit"]+":"+p)
                  and digest(raw) == e["sha256"] and len(raw) == e["bytes"])
    artificial = load(ART / "artificial_test_results.json")
    check("artificial_tests_PASS", artificial["status"] == "PASS" and artificial["failures"] == artificial["errors"] == 0)
    check("artificial_tests_not_saved_science", artificial["saved_science_decoded"] is False)
    for p in DOCS:
        body = (ROOT/p).read_text()
        for token in ["未採択", "mandatory STOP"]:
            check("document_scope:"+p+":"+token, token in body)
        destinations = re.findall(r"\]\(([^)]+)\)", body) + re.findall(r"^\[[^\]]+\]:\s+(\S+)", body, re.MULTILINE)
        for dest in destinations:
            if not dest.startswith(("https:", "http:", "#")):
                check("document_link:"+dest, ((ROOT/p).parent / dest.split("#")[0]).exists())
    for p in INDEXES:
        old = git("show", spec.BASE+":"+p)
        check("prepend_only:"+p, (ROOT/p).read_bytes().endswith(old))
    oldm = json.loads(git("show", spec.BASE+":"+MANIFEST))
    newm = load(ROOT/MANIFEST)
    check("old_manifest_entries_preserved", all(newm.get(k) == v for k,v in oldm.items())
          and set(newm)-set(oldm) == {MANIFEST_KEY})
    mismatches = []
    inspected = 0
    for record in git("ls-tree", "-rz", spec.BASE).split(b"\0"):
        if not record:
            continue
        header, rawp = record.split(b"\t",1)
        mode, kind, oid = header.decode().split()
        p = rawp.decode()
        if kind != "blob" or p in INDEXES+[MANIFEST]:
            continue
        q = ROOT/p
        raw = os.readlink(q).encode() if mode == "120000" and q.is_symlink() else q.read_bytes() if q.is_file() else None
        inspected += 1
        if raw is None or blob(raw) != oid:
            mismatches.append(p)
    check("all_old_nonindex_blobs_preserved", not mismatches)
    changed = set(filter(None,git("diff","--name-only","-z",spec.BASE).decode().split("\0")))
    changed.update(filter(None,git("ls-files","--others","--exclude-standard","-z").decode().split("\0")))
    allowed = set(INDEXES+[MANIFEST,TEST,CHECKER,NOTE,spec.REVIEW]+DOCS+spec.SOURCE_FILES)
    unexpected = [p for p in changed if p not in allowed and not p.startswith(spec.ART+"/")]
    check("only_authorized_paths_changed", not unexpected)
    new = [p for p in changed if not git("ls-tree",spec.BASE,"--",p)]
    check("no_authority_grant_marker_or_result_files_issued", not any(
        any(t in Path(p).name.lower() for t in ["authorization", "grant", "marker", "one_shot_consumed"])
        for p in new) and not any(p.startswith(spec.OUTPUT_PREFIX) or p.startswith(spec.AUTH_PREFIX) for p in new))
    snap = load(args.snapshot)
    parent = Path(snap["root"])
    parent_changed = []
    for p, expected in snap["tracked"].items():
        q = parent/p
        actual = digest(q.read_bytes()) if q.is_file() else None
        if actual != expected:
            parent_changed.append(p)
    check("parent_tracked_bytes_preserved", not parent_changed)
    check("parent_HEAD_preserved", git("rev-parse","HEAD",root=parent).decode().strip() == snap["root_HEAD"])
    ix = Path(git("rev-parse","--git-path","index",root=parent).decode().strip())
    if not ix.is_absolute():
        ix = parent/ix
    check("parent_index_preserved", digest(ix.read_bytes()) == snap["root_index_sha256"])
    check("parent_status_preserved", git("status","--short","--untracked-files=no",root=parent).decode() == snap["root_status"])
    check("original_stage_s_review_preserved", digest((parent/"m1s_stage_s_scientific_conditions_review_20261017.md").read_bytes()) == snap["stage_s_review_sha256"])
    changed_heads = [p for p, head in snap["worktree_HEADs"].items()
                     if git("rev-parse","HEAD",root=Path(p)).decode().strip() != head]
    check("prior_worktree_HEADs_preserved", not changed_heads)
    preservation = [c for c in CHECKS if any(t in c["name"] for t in
                    ["preserved","prepend_only","authorized_paths","no_authority_grant_marker"]) ]
    dump(ART/"preservation_audit.json", {"status":"PASS" if all(c["status"] == "PASS" for c in preservation) else "FAIL",
        "old_nonindex_blobs_inspected":inspected,"old_blob_mismatches":mismatches,"unexpected_paths":unexpected,
        "parent_tracked_files":len(snap["tracked"]),"parent_changed":parent_changed,
        "prior_worktree_HEADs":len(snap["worktree_HEADs"]),"changed_prior_HEADs":changed_heads,
        "other_worktree_untracked_inventory_claimed":False,"checks":preservation})
    report={"status":"PASS" if all(c["status"] == "PASS" for c in CHECKS) else "FAIL",
        "scope":"LOCAL_AST_METADATA_SCHEMA_GUARD_IDENTITY_PRESERVATION_ONLY",
        "checks_PASS":sum(c["status"] == "PASS" for c in CHECKS),
        "checks_FAIL":sum(c["status"] == "FAIL" for c in CHECKS),"checks":CHECKS,
        "source_frozen":identity is not None,"saved_scientific_JSON_decoded":False,
        "saved_integer_shot_calculations":0,"new_scientific_calculations":0,
        "new_authorizations_grants_markers":0,"mandatory_STOP":True}
    dump(ART/"static_validation.json",report)
    print(json.dumps({k:report[k] for k in ["status","checks_PASS","checks_FAIL","source_frozen"]}))
    for c in CHECKS:
        if c["status"] != "PASS":
            print(c["name"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
