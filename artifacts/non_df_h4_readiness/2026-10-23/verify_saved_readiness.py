"""Read-only post-commit verification; no scientific or artificial execution."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from trotterlib.isolated_qiskit_execution import read,save,sha
from trotterlib.non_df_h4_readiness import h4_contract_draft,rare_event_acquisition_draft
from trotterlib.saved_rte_cost_diagnostics import diagnose

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
BASE_COMMIT="c8990de7721545faff88b636108034dedb80a463"


def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args])


def main():
    source=read(BASE/"SOURCE.json")
    assert not source["scientific_authorization"] and not source["H4_scientific_dispatch_enabled"]
    assert subprocess.run(["git","merge-base","--is-ancestor",source["commit"],"HEAD"],cwd=ROOT).returncode==0
    for path,expected in source["files"].items(): assert sha(ROOT/path)==expected,path
    assert read(BASE/"H4_execution_contract_DRAFT.json")==h4_contract_draft()
    assert read(BASE/"rare_event_acquisition_DRAFT.json")==rare_event_acquisition_draft()
    assert read(BASE/"rare_event_diagnosis.json")==diagnose(ROOT)
    # Frozen H2 source and raw evidence must remain byte-identical; no execution.
    old=ROOT/"artifacts/non_df_h2_process_recovery/2026-10-23"
    for path,expected in read(old/"SOURCE.json")["files"].items(): assert sha(ROOT/path)==expected,path
    oldstop=read(old/"STOP.json")
    assert oldstop["result_inventory_sha256"]==sha(old/"result_inventory.json")
    for path,expected in read(old/"result_inventory.json")["files"].items(): assert sha(old/path)==expected,path
    doc_updates={"PROJECT_MAP.md","VALIDATION_STATUS.md","artifacts/validation_manifest.json","docs/README.md",
        "docs/research/研究概要・現状.md","scripts/README.md","src/trotterlib/README.md"}
    unchanged,artifacts=0,0
    for record in git("ls-tree","-rz",BASE_COMMIT).split(b"\0"):
        if not record:continue
        metadata,name=record.split(b"\t",1)
        mode,kind,expected=metadata.split()
        name=name.decode()
        if name in doc_updates:continue
        p=ROOT/name
        content=os.readlink(p).encode() if mode==b"120000" else p.read_bytes()
        actual=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
        assert actual==expected.decode(),name
        unchanged+=1;artifacts+=int(name.startswith("artifacts/"))
    previous=json.loads(git("show",BASE_COMMIT+":artifacts/validation_manifest.json"))
    current=read(ROOT/"artifacts/validation_manifest.json")
    assert all(current[k]==v for k,v in previous.items())
    oldwd=ROOT.parent/"non-df-h2-process-recovery-20261023"
    assert subprocess.check_output(["git","status","--porcelain"],cwd=oldwd)==b""
    tests=ET.parse(BASE/"technical/final_tests.xml").getroot().find("testsuite")
    counts={k:int(tests.attrib[k]) for k in ["tests","failures","errors","skipped"]}
    assert counts==dict(tests=371,failures=0,errors=0,skipped=1),counts
    per_module=Counter(t.attrib["classname"] for t in tests.findall("testcase") if t.find("skipped") is None)
    assert sum(per_module.values())==370
    assert per_module["tests.test_isolated_qiskit_supervisor"]==31
    assert per_module["tests.test_non_df_h4_readiness"]==11
    q=BASE/"technical/qiskit_preflight"
    assert read(q/"summary.json")["status"]=="PASS"
    assert read(q/"dimension/summary.json")["status"]=="PASS"
    assert read(q/"dimension/summary.json")["molecular"] is False
    final_stops=list((BASE/"technical/final_workers").rglob("STOP.json"))+list(q.rglob("STOP.json"))
    pids=set();pending=[];stop_statuses=Counter();unreaped=[]
    for path in final_stops:
        report=read(path)
        stop_statuses[report["status"]]+=1
        assert report["all_children_reaped"] and not report["pending_owned_processes"],str(path)
        for ident in report["identities"]:
            pids.add(ident["pid"])
            p=Path(f"/proc/{ident['pid']}/stat")
            if p.exists():
                text=p.read_text();fields=text[text.rindex(")")+2:].split()
                assert int(fields[19])!=ident["start_ticks"],"PROCESS_STILL_OWNED:"+str(ident)
    # The deliberately failed STOP write is evidenced on the separately saved
    # supervisor stderr and direct process report, never represented as PASS.
    storage=BASE/"technical/final_workers/test_STOP_storage_failure_is_reported_without_claiming_success"
    assert "STOP_PERSISTENCE_FAILED" in (storage/"run.stderr.log").read_text()
    assert not (storage/"run/STOP.json").exists()
    row=read(storage/"run/case0/process.json")
    assert row["exit_code"]==0 and not Path(f"/proc/{row['pid']}").exists()
    pids.add(row["pid"])
    verification=dict(status="PASS",technical_SOURCE=source["commit"],source_files_verified=len(source["files"]),
        original_tracked_files_unchanged=unchanged,original_artifacts_unchanged=artifacts,
        original_manifest_entries_identical=len(previous),original_H2_worktree_clean=True,
        original_H2_SOURCE_and_result_inventory_verified=True,old_CNG_figures_grants_markers_ledgers_STOP_unchanged=True,
        technical_tests_passed=370,new_supervisor_tests=31,new_readiness_tests=11,
        previous_safe_regression_tests_passed=328,skipped=1,molecular_tests_deselected=5,
        passed_by_module=dict(sorted(per_module.items())),
        final_supervisor_STOP_count=len(final_stops),final_supervisor_STOP_statuses=dict(stop_statuses),
        owned_process_identities_verified_gone=len(pids),all_recorded_final_children_reaped=True,
        deliberate_STOP_write_failure_verified=True,
        small_four_case_overlap_seconds=read(q/"summary.json")["overlap_seconds"],
        synthetic_eight_qubit_overlap_seconds=read(q/"dimension/summary.json")["overlap_seconds"],
        eight_qubit_errors=read(q/"dimension/summary.json")["errors"],
        H4_scientific_calls=0,H2_new_sampling=0,H2_RNG_replay=0,H2_new_compile=0,
        new_population_cost_evaluation=0,new_scientific_ranking=0,scientific_review_started=False,
        H4_grant_valid=False,push=False,verifier_sha256=sha(__file__))
    if (BASE/"verification.json").exists(): assert read(BASE/"verification.json")==verification
    else:save(BASE/"verification.json",verification)
    print(json.dumps(dict(status="PASS",passed=370,unchanged_files=unchanged,old_manifest_entries=len(previous),
        final_STOP_count=len(final_stops),recorded_owned_processes_gone=len(pids)),sort_keys=True))


if __name__=="__main__":
    main()
