"""Hash only explicit source, metadata, lightweight JSON and review documents.

Never enumerate or access molecular snapshots/runtime/cache. The old manifest is
checked against pinned commit blobs, separately from the updated documentary HEAD.
"""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).parent
REPO=HERE.parents[3]
OLD=HERE.parent/"2026-10-06"
PREP=REPO/"artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05"
BASE="7c1a3d43f61c5501a9e79206b7c60933f94b1077"
DOCS=["PROJECT_MAP.md","docs/README.md","docs/research/README.md",
      "scripts/README.md","src/trotterlib/README.md","docs/research/研究概要・現状.md",
      "docs/research/研究ノート/README.md","docs/research/研究ノート/2026-10-06.md"]


def sha(content):return hashlib.sha256(content).hexdigest()
def load(path):return json.loads(path.read_text())
def write(name,value):
    with (HERE/name).open("x") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
def blob(path):
    return subprocess.run(["git","show",BASE+":"+path],cwd=REPO,capture_output=True,check=True).stdout


def main():
    plan=load(HERE/"zero_compute_plan_v2.json")
    result=load(HERE/"contract_tests_result_v2.json")
    assert result["passed"]==320 and result["failed"]==result["skipped"]==0
    assert result["new_test_source_sha256"]==sha((HERE/"run_contract_tests_v2.py").read_bytes())
    prefix=OLD.relative_to(REPO).as_posix()+"/"
    manifest_bytes=blob(prefix+"artifact_manifest_v1.json")
    assert sha(manifest_bytes)=="68cc9c28faaf6ab1b43799d5ffc18ece633d244634f5146239dbc93770c530f7"
    old_manifest=json.loads(manifest_bytes)
    for entry in old_manifest["files"]:
        content=blob(entry["path"])
        assert sha(content)==entry["sha256"] and len(content)==entry["bytes"],entry["path"]
    old_paths=[f["path"] for f in old_manifest["files"] if f["path"].startswith(prefix)]+[prefix+"artifact_manifest_v1.json"]
    assert len(old_paths)==26
    for path in old_paths:assert (REPO/path).read_bytes()==blob(path),path
    prep_manifest=PREP/"preparation_artifact_manifest_v0.json"
    assert sha(prep_manifest.read_bytes())=="33d8a7a5986bd4a29cbe7d678874d54d859614bfab8643b88de2e9f613651741"
    pm=load(prep_manifest)
    for entry in pm["files"]:
        data=(PREP/entry["path"]).read_bytes()
        assert sha(data)==entry["sha256"] and len(data)==entry["bytes"],entry["path"]
    assert len(pm["files"])+1==25
    static=load(PREP/"static_audit_v0.json")
    for entry in static["source_hashes"]:
        assert sha((REPO/entry["path"]).read_bytes())==entry["sha256"],entry["path"]
    assert len(static["source_hashes"])==247
    for entry in static["allowed_json_identity"]:
        assert sha((REPO/entry["path"]).read_bytes())==entry["sha256"],entry["path"]
    assert len(static["allowed_json_identity"])==6
    for path in ["VALIDATION_STATUS.md","artifacts/validation_manifest.json"]:
        assert (REPO/path).read_bytes()==blob(path),path
    identity={"schema_version":"h4-contract-identity-preservation-v2","base_commit":BASE,
              "v1_manifest_sha256":sha(manifest_bytes),"v1_manifest_33_entries_verified_against":"base commit blobs; not mutated current documentary files",
              "v1_bundle_files_byte_identical":26,"published_preparation_files_byte_identical":25,
              "old_source_hashes_unchanged":247,"saved_evidence_JSON_hashes_unchanged":6,
              "documentary_paths_updated_only_in_v2_manifest":DOCS,
              "validation_status_manifest_unchanged":True,"excluded_inputs_accessed":0,
              "molecular_snapshot_materialization":False,"scope":"source text and permitted lightweight records only"}
    write("identity_preservation_audit_v2.json",identity)
    installed=Path("/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages")
    sources={
        "pyscf/scf/hf.py":[(116,117,"sqrt(conv_tol) gradient default; proposed explicit raw gradient"),
                              (194,196,"main convergence uses energy AND gradient"),
                              (211,230,"normal extra check has relaxed thresholds and OR; independent proposal gate required"),
                              (1324,1332,"canonical MO eigensolver and first largest real component sign"),
                              (1664,1687,"config-sensitive defaults, proposed explicit SCF/CDIIS/direct settings"),
                              (436,440,"minao atom fallback for charge>96; H4 proposal rejects fallback, no molecule tested")],
        "pyscf/scf/diis.py":[(40,66,"normal CDIIS, space8/damp0/rollback0; different from post-failure rescue")],
        "pyscf/gto/mole.py":[(2300,2318,"unit config-sensitive; explicit Angstrom/symmetry/cart/incore proposals")],
        "openfermionpyscf/_run_pyscf.py":[(36,60,"geometry and RHF path"),(77,94,"integral transform/order"),(130,132,"legacy controls implicit"),(202,206,"legacy run_pyscf unconditionally saves; not future safe controlled source")],
        "openfermion/circuits/low_rank.py":[(112,126,"spin transform and real/symmetric/eigh"),(133,159,"l1-derived weights, reversed argsort returned order, explicit rank12 overrides tolerance")],
        "openfermion/chem/molecular_data.py":[(367,402,"spin-orbital interleaving and EQ_TOLERANCE zeroing"),(1031,1037,"InteractionOperator factor1/2")],
        "openfermion/config.py":[(16,16,"EQ_TOLERANCE=1e-8")],
        "openfermion/linalg/sparse_tools.py":[(292,346,"fixed alpha/beta JW occupation bit(7-p)")],
    }
    findings=[]
    for relative,ranges in sources.items():
        content=(installed/relative).read_bytes()
        ast.parse(content.decode())
        findings.append({"installed_source":relative,"absolute_path":str(installed/relative),
                         "sha256":sha(content),"findings":[{"line_start":a,"line_end":b,"finding":f} for a,b,f in ranges]})
    repo_source_ranges={"src/trotterlib/df_hamiltonian.py":(405,428,"spin_sector sorted index order"),
                       "src/trotterlib/pr2_s0_s1_validation.py":(159,174,"sector state phase pivot"),
                       "src/trotterlib/df_partial_randomized_pf.py":(216,221,"generic Frobenius re-sort is separate policy"),
                       "src/trotterlib/pr2_matched_accuracy_m1_execution.py":(208,249,"dense eigh and residual/imaginary energy gates")}
    for relative,(a,b,f) in repo_source_ranges.items():
        content=(REPO/relative).read_bytes();ast.parse(content.decode())
        findings.append({"repository_source":relative,"sha256":sha(content),"line_start":a,"line_end":b,"finding":f})
    observations=load(OLD/"environment_binding_audit_v1.json")["dependency_observations"]
    checked=[]
    for name,previous in observations.items():
        distribution=importlib.metadata.distribution(name)
        assert distribution.version==previous["version"],name
        record=distribution.read_text("RECORD")
        assert record is not None and sha(record.encode())==previous["installed_record_sha256"],name
        checked.append(name)
    assert len(checked)==45
    write("static_source_audit_v2.json",{
        "schema_version":"h4-contract-static-source-v2","status":"READ_ONLY_SOURCE_AND_METADATA_MATCH",
        "observed_jst":datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),"source_findings":findings,
        "installed_metadata_versions_RECORDs_match_v1":checked,"dependency_count":45,
        "science_imports":0,"molecular_calculations":0,"global_config_changed":0,
        "actual_effective_SCF_defaults_imported":False,"explicit_review_controls_proposed":True,
        "scientific_choices_pending_review":plan["unresolved_decisions"],
        "full_wheel_binary_independent_equivalence_established":False})
    # AST-only check that new preparation source contains no forbidden imports.
    blocked={"numpy","scipy","qiskit","openfermion","openfermionpyscf","pyscf","trotterlib","trottertracks","torch","cupy"}
    for path in HERE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):assert not any(a.name.split(".")[0] in blocked for a in node.names),path
            if isinstance(node,ast.ImportFrom):assert node.module.split(".")[0] not in blocked,path
    write("final_audit_v2.json",{
        "schema_version":"h4-contract-final-audit-v2","status":plan["status"],
        "audit_stage":"before explicitly authorized publication; not scientific execution",
        "base_commit":BASE,"branch":"track-a-h4-geometry-contract-v2-20261006",
        "plan_sha256":sha((HERE/"zero_compute_plan_v2.json").read_bytes()),"plan_fingerprint":plan["plan_fingerprint"],
        "new_synthetic_cases_passed":320,"failed":0,"skipped":0,"schemas_checked":8,
        "historical_v1_cases_saved_not_rerun":129,"historical_v1_runners_executed":0,
        "development_failed_attempts_saved_separately":[
            {"log":"contract_tests_v2_attempt01_startup.log","failure":"guard rejected Python .pyc lookup before read; no scientific input or cache read","completed_suite":False},
            {"log":"contract_tests_v2_attempt02_fixture.log","failure":"unsupported wrapper semantics should be rejected by frozen schema, not accepted as a foreign owner; corrected test expectation","completed_suite":False}],
        "final_suite_scientific_import_attempts":result["scientific_import_attempts"],
        "final_suite_protected_access_attempts":result["protected_access_attempts"],
        "historical_synthetic_transpiles":{"comparison":120,"axis_phase":4,"full_operator":4,"total":128},
        "current_execution_counts":plan["current_execution_counts"],
        "old_evidence_preservation":identity,
        "science_execution_authorized":False,"source_port_authorized":False,
        "input_generation_authorized":False,"execution_plan_sealed":False,
        "next_stage_authorized":False,"automatic_research_decision_authorized":False,
        "research_decision":None,"mandatory_stop":True,
        "pending_reviews":plan["unresolved_decisions"],"publication_authorized":True,
        "science_output_or_registry_created":False,
        "limitations":"JSON review contract, not molecular/numerical/physical/atomic production implementation or CI/external evidence"})
    files=sorted([str(p.relative_to(REPO)) for p in HERE.iterdir() if p.is_file() and p.name!="artifact_manifest_v2.json"]+DOCS)
    assert all(Path(p).suffix in {".py",".md",".json",".log"} for p in files)
    entries=[]
    for name in files:
        path=REPO/name;assert not path.is_symlink()
        data=path.read_bytes();entries.append({"path":name,"sha256":sha(data),"bytes":len(data)})
    write("artifact_manifest_v2.json",{
        "schema_version":"h4-contract-preparation-artifact-manifest-v2","status":plan["status"],
        "base_commit":BASE,"files":entries,"manifest_self_excluded":True,
        "v1_manifest_verified_against_base_blobs":True,"v1_manifest_rewritten":False,
        "evidence_scope":"review-only preparation, source audits and synthetic JSON; no scientific evidence",
        "scientific_execution_authorized":False,"mandatory_stop":True})
    print(json.dumps({"manifest_entries":len(entries),"bundle_files":len(entries)-8+1,
                      "old_v1":26,"old_source":247,"published_preparation":25,"saved_JSON":6,
                      "dependencies":45,"tests":320,"science_execution":0}))


if __name__=="__main__":main()
