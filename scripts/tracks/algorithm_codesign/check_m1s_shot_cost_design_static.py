"""Metadata/AST/schema/guard/identity audit only; no scientific JSON decoding.

Use --preservation-snapshot for the separately saved parent/worktree snapshot.
This is a local technical audit, not a numerical correctness test or CI result.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/track_b_m1s_shot_cost_preparation/2026-10-17/v1"
SOURCE = "src/trottertracks/algorithm_codesign/m1s_shot_cost_design_v1.py"
CHECKER = "scripts/tracks/algorithm_codesign/check_m1s_shot_cost_design_static.py"
DOCROOT = "docs/tracks/algorithm_codesign/"
PACKET = DOCROOT + "m1s_shot_cost_feasibility_decision_packet_20261017.md"
DOCS = [PACKET, DOCROOT + "m1s_shot_cost_verification_contract_candidate_20261017.md",
        DOCROOT + "m1s_shot_cost_source_resource_and_handoff_20261017.md"]
NOTE = "docs/research/研究ノート/2026-10-17_m1s_shot_cost_feasibility_preparation.md"
INDEXES = ["PROJECT_MAP.md", "VALIDATION_STATUS.md", "docs/README.md",
           "scripts/README.md", "src/trotterlib/README.md", "docs/research/研究概要・現状.md"]
MANIFEST = "artifacts/validation_manifest.json"
MANIFEST_KEY = "track_b_m1s_shot_cost_preparation_20261017"
CHECKS = []


def check(name, condition):
    CHECKS.append({"name": name, "status": "PASS" if condition else "FAIL"})


def git(*args, root=ROOT):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))


def read_metadata(path):
    """Only new design metadata and the old inventory manifest use this call."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def blob_digest(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preservation-snapshot", type=Path)
    args = parser.parse_args()
    spec_tree = ast.parse((ROOT / SOURCE).read_text())
    check("spec_AST_valid", isinstance(spec_tree, ast.Module))
    check("spec_no_imports", not any(isinstance(n, (ast.Import, ast.ImportFrom))
                                     for n in ast.walk(spec_tree)))
    functions = {n.name: n for n in spec_tree.body if isinstance(n, ast.FunctionDef)}
    guard_body = functions["request_execution"].body
    check("guard_unconditional_raise_after_docstring",
          len(guard_body) == 2 and isinstance(guard_body[0], ast.Expr)
          and isinstance(guard_body[1], ast.Raise))
    check("spec_no_numeric_runner", set(functions) == {"validate_candidate", "request_execution"})
    calls = [n for n in ast.walk(spec_tree) if isinstance(n, ast.Call)]
    check("spec_no_file_or_evaluation_calls", not any(
        isinstance(n.func, ast.Name) and n.func.id in
        {"open", "eval", "exec", "compile", "float", "ceil", "sin", "log", "pow"}
        for n in calls))
    check("checker_AST_valid", isinstance(ast.parse((ROOT / CHECKER).read_text()), ast.Module))
    spec = importlib.util.spec_from_file_location("shot_cost_design_only", ROOT / SOURCE)
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    c = read_metadata(ROOT / design.CONTRACT)
    schema = read_metadata(ROOT / design.SCHEMA)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    check("candidate_schema_valid", validator.is_valid(c))
    design.validate_candidate(c)
    check("candidate_invariants_valid", True)
    for key in design.CLOSED_FLAGS:
        check("closed:" + key, c[key] is False)
        altered = copy.deepcopy(c)
        altered[key] = True
        check("schema_rejects_open:" + key, not validator.is_valid(altered))
        try:
            design.validate_candidate(altered)
        except design.DesignOnlyError:
            rejected = True
        else:
            rejected = False
        check("spec_rejects_open:" + key, rejected)
    for key in design.PENDING_CHOICES:
        altered = copy.deepcopy(c)
        altered["scientific_choices"][key] = {"unapproved_choice": "not_adopted"}
        check("schema_rejects_choice:" + key, not validator.is_valid(altered))
    for payload in [None, c, {key: True for key in design.CLOSED_FLAGS}]:
        try:
            design.request_execution(payload)
        except design.DesignOnlyError:
            rejected = True
        else:
            rejected = False
        check("guard_rejects_payload:" + str(len(CHECKS)), rejected)
    altered = copy.deepcopy(c)
    altered["unregistered_field"] = None
    check("schema_rejects_unknown_fields", not validator.is_valid(altered))
    for key in design.ZERO_FIELDS:
        check("no_execution:" + key, type(c[key]) is int and c[key] == 0)
    check("new_time_caps_unset", all(
        r["CPU_time_cap_seconds"] is None and r["wall_time_cap_seconds"] is None
        for key, r in c["proposed_resources"].items() if key != "status"))
    check("no_future_runner_or_authority", c["future_source_identity"] is None
          and c["execution_authorization"] is None)

    # Only the newly transcribed category registry is decoded, never saved science.
    registry = read_metadata(ART / "preserved_gate_registry.json")
    rows = registry["rows"]
    ids = [(r["split"], r["delta_Ha_inverse"], r["R"], r["T_Ha_inverse"], r["comparator"])
           for r in rows]
    check("48_unique_metadata_contrasts", len(ids) == len(set(ids)) == 48)
    check("complete_cartesian_inventory", set(ids) == {
        (g, d, r, t, m) for g in ["G3", "G6", "G9"] for d in ["1/8", "1/16"]
        for r in [2, 4] for t in ["1", "4"] for m in ["ordinary", "independent_pm"]})
    for g, t, count, status in [
        ("G3", None, 16, "SATISFIED"), ("G6", "4", 8, "SATISFIED"),
        ("G6", "1", 8, "NOT_SATISFIED"),
        ("G9", None, 16, "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED")]:
        group = [r for r in rows if r["split"] == g and (t is None or r["T_Ha_inverse"] == t)]
        check("preserved_group:" + g + ":" + str(t),
              len(group) == count and all(r["old_gate_classification"] == status for r in group))
    check("metadata_not_new_scientific_results", all(
        r["new_integer_classification"] == r["new_cost_classification"] == "NOT_EVALUATED"
        for r in rows) and registry["raw_result_decoded"] is False)
    check("posthoc_not_held_out", registry["not_held_out_independent_validation"] is True)

    identities = read_metadata(ART / "input_identity.json")
    for entry in identities["entries"]:
        raw = (ROOT / entry["path"]).read_bytes()  # bytes only, no science JSON decoder
        check("identity:" + entry["path"], len(raw) == entry["bytes"]
              and digest(raw) == entry["sha256"] and blob_digest(raw) == entry["git_blob_sha1"]
              and raw == git("show", entry["commit"] + ":" + entry["path"]))
    check("identity_no_numeric_decoding", identities["scientific_raw_json_or_csv_decoded"] is False)
    pending = read_metadata(ART / "pending_scientific_decisions.json")
    check("scientific_fields_null", pending["fields"] == c["scientific_choices"]
          and all(v is None for v in pending["fields"].values()))
    packet = (ROOT / PACKET).read_text()
    for formula in design.FORMULA_IDS:
        check("formula_documented:" + formula, "\\tag{" + formula + "}" in packet)
    for p in DOCS:
        body = (ROOT / p).read_text()
        for token in ["未採択", "mandatory STOP"]:
            check("scope:" + p + ":" + token, token in body)
        # Resolve local Markdown links, without fetching or interpreting their contents.
        for dest in re.findall(r"\]\(([^)]+)\)", body):
            if dest.startswith(("https:", "http:", "#")):
                continue
            check("link:" + p + ":" + dest, ((ROOT / p).parent / dest.split("#")[0]).exists())
        for dest in re.findall(r"^\[[^\]]+\]:\s+(\S+)", body, re.MULTILINE):
            if not dest.startswith(("https:", "http:", "#")):
                check("reference_link:" + p + ":" + dest,
                      ((ROOT / p).parent / dest.split("#")[0]).exists())
    check("raw_total_no_double_count", "再加算しない" in packet)
    check("visibility_not_total_cost", "整数化を捨てた" in packet and "C2" in packet)
    check("target_bridge_and_missing_round", "中間T=2" in packet and "raw→目的" in packet)

    # Index additions preserve the exact prior text, and the manifest keeps old values.
    for p in INDEXES:
        old = git("show", design.BASE_COMMIT + ":" + p)
        new = (ROOT / p).read_bytes()
        check("index_prepend_only:" + p, new.endswith(old) and PACKET.split("/")[-1].encode() in new)
    old_manifest = json.loads(git("show", design.BASE_COMMIT + ":" + MANIFEST))
    new_manifest = read_metadata(ROOT / MANIFEST)
    check("old_manifest_entries_preserved", all(new_manifest.get(k) == v for k, v in old_manifest.items())
          and set(new_manifest) - set(old_manifest) == {MANIFEST_KEY})

    # Inspect every old non-index Git blob in this isolated worktree.
    records = git("ls-tree", "-rz", design.BASE_COMMIT).split(b"\0")
    mismatches = []
    inspected = 0
    for record in records:
        if not record:
            continue
        header, rawpath = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        p = rawpath.decode()
        if p in INDEXES or p == MANIFEST or kind != "blob":
            continue
        q = ROOT / p
        raw = os.readlink(q).encode() if mode == "120000" and q.is_symlink() else q.read_bytes() if q.is_file() else None
        inspected += 1
        if raw is None or blob_digest(raw) != oid:
            mismatches.append(p)
    check("all_old_nonindex_blobs_preserved", not mismatches)
    changed = set(filter(None, git("diff", "--name-only", "-z", design.BASE_COMMIT).decode().split("\0")))
    changed.update(filter(None, git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")))
    artprefix = str(ART.relative_to(ROOT)) + "/"
    allowed_exact = set(INDEXES + [MANIFEST, SOURCE, CHECKER, NOTE] + DOCS)
    unexpected = sorted(p for p in changed if p not in allowed_exact and not p.startswith(artprefix))
    check("only_authorized_new_paths_and_indexes", not unexpected)
    newpaths = [p for p in changed if not git("ls-tree", design.BASE_COMMIT, "--", p)]
    check("no_new_execution_authority_or_marker_files",
          not any(any(token in Path(p).name.lower() for token in ["authorization", "grant", "marker"])
                  for p in newpaths))
    parent_audit = {"snapshot_supplied": False, "status": "NOT_CHECKED"}
    if args.preservation_snapshot:
        snapshot = read_metadata(args.preservation_snapshot)
        parent = Path(snapshot["root"])
        root_changed = []
        for p, expected in snapshot["tracked"].items():
            q = parent / p
            actual = digest(q.read_bytes()) if q.is_file() else None
            if actual != expected:
                root_changed.append(p)
        parent_start = len(CHECKS)
        check("parent_tracked_bytes_preserved", not root_changed)
        check("parent_HEAD_preserved", git("rev-parse", "HEAD", root=parent).decode().strip() == snapshot["root_HEAD"])
        index = Path(git("rev-parse", "--git-path", "index", root=parent).decode().strip())
        if not index.is_absolute():
            index = parent / index
        check("parent_index_preserved", digest(index.read_bytes()) == snapshot["root_index_sha256"])
        check("parent_status_preserved", git("status", "--short", "--untracked-files=no", root=parent).decode() == snapshot["root_status"])
        review = parent / "m1s_gamma_v1_completed_results_scientific_review_20261017.md"
        check("original_review_preserved", digest(review.read_bytes()) == snapshot["review_original_sha256"])
        heads_changed = [p for p, head in snapshot["worktree_HEADs"].items()
                         if git("rev-parse", "HEAD", root=Path(p)).decode().strip() != head]
        check("prior_worktree_HEADs_preserved", not heads_changed)
        parent_audit = {"snapshot_supplied": True,
                        "status": "PASS" if all(c["status"] == "PASS" for c in CHECKS[parent_start:]) else "FAIL",
                        "tracked_files": len(snapshot["tracked"]), "prior_worktree_HEADs": len(snapshot["worktree_HEADs"]),
                        "changed_tracked": root_changed, "changed_HEADs": heads_changed,
                        "scope_limit": "Other worktree HEAD identities checked; their untracked contents are not inventoried"}
    else:
        check("parent_snapshot_required_for_completion", False)
    preservation_checks = [c for c in CHECKS if any(t in c["name"] for t in
        ["preserved", "index_prepend", "authorized_new_paths", "no_new_execution", "snapshot_required"])]
    dump(ART / "preservation_audit.json", {
        "status": "PASS" if all(c["status"] == "PASS" for c in preservation_checks) else "FAIL",
        "base_commit": design.BASE_COMMIT, "old_nonindex_blobs_inspected": inspected,
        "old_blob_mismatches": mismatches, "unexpected_changed_paths": unexpected,
        "parent": parent_audit, "checks": preservation_checks,
        "scientific_results_reclassified": False})
    report = {"status": "PASS" if all(c["status"] == "PASS" for c in CHECKS) else "FAIL",
              "scope": "LOCAL_STATIC_METADATA_SCHEMA_GUARD_IDENTITY_AND_PRESERVATION_ONLY",
              "checks_PASS": sum(c["status"] == "PASS" for c in CHECKS),
              "checks_FAIL": sum(c["status"] == "FAIL" for c in CHECKS), "checks": CHECKS,
              "numeric_correctness_tested": False, "runtime_tested": False,
              "saved_scientific_JSON_decoded": False, "new_scientific_calculations": 0,
              "shot_forecasts": 0, "native_compiles": 0, "new_authorizations_grants_markers": 0,
              "mandatory_STOP": True}
    dump(ART / "static_validation.json", report)
    print(json.dumps({k: report[k] for k in ["status", "checks_PASS", "checks_FAIL", "scope"]}))
    for c in CHECKS:
        if c["status"] != "PASS":
            print(c["name"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
