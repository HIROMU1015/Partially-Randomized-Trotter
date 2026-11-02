"""Read-only final verification plus new local audit/manifest output."""
import ast
import hashlib
import json
import pathlib
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from jsonschema import Draft202012Validator

HERE = pathlib.Path(__file__).parent
REPO = pathlib.Path.cwd()
PREP = REPO / "artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05"
BASE = "c2ab34fed49bb1fb104d39fe83b36858a2c92c2a"
DOCS = [
    "PROJECT_MAP.md", "docs/README.md", "docs/research/README.md",
    "scripts/README.md", "src/trotterlib/README.md",
    "docs/research/研究概要・現状.md", "docs/research/研究ノート/2026-10-06.md",
    "docs/research/研究ノート/README.md",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    git = ["git", "-c", "core.sparseCheckout=true", "-c", "core.sparseCheckoutCone=false"]
    assert subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip() == BASE
    assert subprocess.check_output(git + ["branch", "--show-current"], text=True).strip() == "track-a-h4-geometry-contract-20261006"
    assert subprocess.check_output(git + ["remote", "get-url", "origin"], text=True).strip() == "https://github.com/HIROMU1015/Partially-Randomized-Trotter"
    assert not subprocess.check_output(git + ["diff", "--cached", "--name-only"], text=True).strip()
    subprocess.run(git + ["diff", "--check"], check=True)
    changed = subprocess.check_output(git + ["diff", "--name-only", "-z"], text=True).split("\0")
    changed = sorted(p for p in changed if p)
    assert set(changed) <= set(DOCS), changed
    prep_manifest = read(PREP / "preparation_artifact_manifest_v0.json")
    assert sha(PREP / "preparation_artifact_manifest_v0.json") == "33d8a7a5986bd4a29cbe7d678874d54d859614bfab8643b88de2e9f613651741"
    for item in prep_manifest["files"]:
        assert sha(PREP / item["path"]) == item["sha256"]
    static = read(PREP / "static_audit_v0.json")
    for item in static["source_hashes"] + static["allowed_json_identity"]:
        assert sha(REPO / item["path"]) == item["sha256"]
    for name in ["VALIDATION_STATUS.md", "artifacts/validation_manifest.json"]:
        assert (REPO / name).read_bytes() == subprocess.check_output(git + ["show", BASE + ":" + name])
    for directory in ["src/trotterlib", "src/trottertracks", "scripts", "tests"]:
        # Git diff uses existing blobs; do not visit excluded scientific input paths.
        altered = subprocess.check_output(git + ["diff", "--name-only", "--", directory], text=True).splitlines()
        assert all(p in {"scripts/README.md", "src/trotterlib/README.md"} for p in altered)
    schemas = list(HERE.glob("*schema_v1.json"))
    assert len(schemas) == 6
    for schema in schemas:
        Draft202012Validator.check_schema(read(schema))
    plan = read(HERE / "zero_compute_plan_v1.json")
    Draft202012Validator(read(HERE / "plan_schema_v1.json")).validate(plan)
    Draft202012Validator(read(HERE / "scope_schema_v1.json")).validate(read(HERE / "scope_v1.json"))
    assert all(v == 0 for v in plan["current_execution_counts"].values())
    assert plan["execution_plan_sealed"] is False and plan["science_execution_authorized"] is False
    assert len(plan["templates"]) == 218
    assert plan["future_resource_caps"]["total_logical_wrapper_records"] == 74784
    assert plan["future_resource_caps"]["cpu_worker_max"] == 12
    tests = read(HERE / "contract_tests_result_v1.json")
    additional = read(HERE / "additional_contract_checks_v1.json")
    assert tests["passed"] == 111 and additional["passed"] == 18
    assert tests["failed"] == additional["failed"] == tests["skipped"] == additional["skipped"] == 0
    assert tests["science_build_compile_transpile_calls"] == additional["science_build_compile_transpile"] == 0
    assert not tests["protected_access_attempts"] and not tests["science_import_attempts"]
    assert read(HERE / "environment_binding_audit_v1.json")["metadata_differences"] == []
    for path in HERE.iterdir():
        assert path.is_file() and not path.is_symlink()
        assert path.suffix in {".py", ".md", ".json"}
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == ".json":
            read(path)
    audit = {
        "status": plan["status"], "final_local_audit": "PASS",
        "completed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "handoff_commit": plan["handoff_commit"], "base_head_commit": BASE,
        "branch": "track-a-h4-geometry-contract-20261006", "worktree": str(REPO),
        "new_contract_bundle": str(HERE), "contract_local_uncommitted": True,
        "published_preparation_files_unchanged": 25, "old_source_hashes_unchanged": 247,
        "saved_evidence_json_hashes_unchanged": 6, "old_validation_manifest_status_unchanged": True,
        "metadata_dependencies_match": 45, "schemas_checked": 6,
        "synthetic_JSON_tests_passed": 129, "failed": 0, "skipped": 0,
        "historical_synthetic_transpiles": {"comparison": 120, "axis_phase": 4, "full_operator": 4, "total": 128},
        "new_synthetic_transpiles": 0, "current_execution_counts": plan["current_execution_counts"],
        "allowed_tracked_doc_index_changes": changed,
        "unresolved_decisions": plan["unresolved_decisions"],
        "science_execution_authorized": False, "execution_plan_sealed": False,
        "next_stage_authorized": False, "research_decision": None, "mandatory_stop": True,
        "guard_and_validation_limit": "local contract JSON checks; no science serializer/atomic runtime/physical operator implementation or validation; not immutable CI"}
    with (HERE / "final_audit_v1.json").open("x") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2); handle.write("\n")
    files = sorted(list(HERE.iterdir()) + [REPO / name for name in DOCS])
    manifest = {
        "schema_version": "h4-contract-preparation-artifact-manifest-v1",
        "status": plan["status"], "base_commit": BASE, "handoff_commit": plan["handoff_commit"],
        "evidence_scope": "new local contract-only artifacts and documentary discovery updates; not scientific/CI/external evidence",
        "files": [{"path": str(p.relative_to(REPO)), "sha256": sha(p), "bytes": p.stat().st_size} for p in files],
        "manifest_self_excluded": True, "mandatory_stop": True,
        "scientific_execution_authorized": False, "commit_push": 0}
    with (HERE / "artifact_manifest_v1.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps({
        "status": audit["status"], "final_audit": "PASS",
        "bundle_files": len(list(HERE.iterdir())), "document_index_files": len(DOCS),
        "tests_passed": 129, "science_counts": plan["current_execution_counts"],
        "zero_compute_plan_sha256": sha(HERE / "zero_compute_plan_v1.json"),
        "contract_draft_sha256": sha(HERE / "CONTRACT_DRAFT_v1.md"),
        "manifest_sha256": sha(HERE / "artifact_manifest_v1.json")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
