"""Verify saved evidence and preservation only; never evaluates a science case."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from trotterlib import h2_non_df_process_recovery as r
from trotterlib.h2_non_df_process_integration import validate_saved_case
from trotterlib.isolated_qiskit_execution import read, save, sha


def main():
    base, root = r.BASE, r.ROOT
    plan, source, grant = r.checked_binding()
    run = root/plan["execution"]["run_namespace"]
    stop=read(run/"STOP.json")
    assert stop["status"]=="COMPLETE_MANDATORY_STOP"
    for path, expected in stop["result_hashes"].items():
        assert sha(run/path)==expected,path
    processes=read(run/"processes.json")
    assert len(processes)==2 and all(p["exit_code"]==0 for p in processes)
    assert all(not Path(f"/proc/{p['pid']}").exists() for p in processes)
    statuses=[read(run/name/"status.json") for name in r.CASES]
    overlap=min(s["ended"] for s in statuses)-max(s["started"] for s in statuses)
    assert overlap>0
    comparison=read(base/"comparison.json")
    assert len(comparison["cases"])==4 and len(comparison["table"])==20
    max_probability_error=0.
    for case in comparison["cases"]:
        folder=root/case["run"]
        assert case["result_sha256"]==sha(folder/"result.json")
        assert case["precision_rows_sha256"]==sha(folder/"precision_rows.json")
        result, rows, validation=[read(folder/name) for name in ["result.json","precision_rows.json","validation.json"]]
        validate_saved_case(case["case"],result,rows,validation,plan["science"])
        max_probability_error=max(max_probability_error,result["circuit_cost"]["compiled_signal_max_error"])
    assert all(row["eligible"] for row in comparison["table"])
    plot=read(base/"figure/provenance.json")
    assert plot["comparison_sha256"]==sha(base/"comparison.json")
    assert len(plot["points"])==20
    for path,expected in plot["images"].items():
        assert sha(base/"figure"/path)==expected
    original_manifest=json.loads(subprocess.check_output(["git","show",r.BASE_COMMIT+":artifacts/validation_manifest.json"],cwd=root))
    current_manifest=read(root/"artifacts/validation_manifest.json")
    assert all(current_manifest[k]==v for k,v in original_manifest.items())
    original_worktree=root.parent/"non-df-h2-pilot-20261022"
    assert subprocess.check_output(["git","status","--porcelain"],cwd=original_worktree)==b""
    for name,expected in read(r.OLD/"result_inventory.json")["raw_result_hashes"].items():
        assert sha(original_worktree/"artifacts/non_df_h2_pilot/2026-10-22"/name)==expected
    allowlist={"PROJECT_MAP.md","VALIDATION_STATUS.md","artifacts/validation_manifest.json","docs/README.md",
               "docs/research/研究概要・現状.md","scripts/README.md","src/trotterlib/README.md"}
    unchanged, artifact_count, changed = 0,0,[]
    entries=subprocess.check_output(["git","ls-tree","-rz",r.BASE_COMMIT],cwd=root).split(b"\0")
    for entry in entries:
        if not entry: continue
        metadata,path=entry.split(b"\t",1)
        mode,kind,expected=metadata.split()
        name=path.decode()
        if name in allowlist: continue
        p=root/name
        content=os.readlink(p).encode() if mode==b"120000" else p.read_bytes()
        actual=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        if actual!=expected.decode(): changed.append(name)
        unchanged+=1
        artifact_count+=int(name.startswith("artifacts/"))
    assert not changed,changed
    # Derive test counts from JUnit, keeping the initial overlapping run separate.
    first=ET.parse(base/"technical/regression.xml").getroot().find("testsuite")
    final=ET.parse(base/"technical/recovery-final.xml").getroot().find("testsuite")
    assert int(first.attrib["failures"])==0 and int(first.attrib["errors"])==0
    assert int(first.attrib["tests"])-int(first.attrib["skipped"])==325
    assert int(final.attrib["tests"])==18 and int(final.attrib["failures"])==0
    assert read(base/"technical/contention/diagnostic.json")["borrow_exception_reproduced"]
    assert read(base/"technical/contention_processes/diagnostic.json")["status"]=="PASS"
    preservation=dict(status="PASS",base_commit=r.BASE_COMMIT,old_manifest_entries_identical=len(original_manifest),
        unchanged_original_tracked_files=unchanged,unchanged_original_artifact_files=artifact_count,
        changed_protected_files=changed,documentation_update_allowlist=sorted(allowlist),
        original_worktree_clean=True,old_raw_records_original_and_new_worktree_identical=True,
        source_manifest_bytes_unchanged=True,old_grant_marker_ledger_STOP_figures_preserved=True)
    save(base/"preservation.json",preservation)
    save(base/"verification.json",dict(status="PASS",SOURCE=source["commit"],
        bound_source_files_verified=len(source["files"]),old_verification=r.verify_old(),
        scientific_child_exit_codes=[p["exit_code"] for p in processes],all_scientific_children_reaped=True,
        scientific_case_overlap_seconds=overlap,accepted_cases=4,precision_rows=20,eligible_rows=20,
        exact_CNG_identities=True,paired_covariance_SE_verified=True,
        max_sampled_compiled_probability_error=max_probability_error,unique_technical_tests_passed=328,
        preserved_old_manifest_entries=len(original_manifest),new_SCF=0,new_FCI=0,new_DF_decompositions=0,
        new_state_solves=0,old_successes_recomputed=False,push=False,review_started=False,
        verifier_sha256=sha(__file__)))
    print(json.dumps(dict(status="PASS",old_files=unchanged,old_artifacts=artifact_count,
        SOURCE_files=len(source["files"]),preserved_manifest_entries=len(original_manifest)),sort_keys=True))


if __name__=="__main__":
    main()
