#!/usr/bin/env python3
"""Stdlib-only saved-source audit; no science imports/tests/execution."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OLD_FREEZE = "artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json"
ARTIFACT = "artifacts/resource_applicability/track_a_ax2b_post_review_preparation/2026-10-10"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root):
    root = Path(root)
    freeze = json.loads((root/OLD_FREEZE).read_text())
    frozen = {**freeze["science_source_hashes"], **freeze["preparation_validation_file_hashes"]}
    mismatches = [p for p,expected in frozen.items() if sha(root/p) != expected]
    if mismatches:
        raise ValueError("FROZEN_SOURCE_CHANGED:" + repr(mismatches))
    plan = json.loads((root/ARTIFACT/"h6_preparation_proposal_v1.json").read_text())
    if (plan["status"] != "H6_NOT_AUTHORIZED" or plan["contract_status"] != "DRAFT_NOT_AUTHORIZATION"
            or plan["actual_rank"] is not None or plan["science_authorized"] is not False
            or plan["input_generation_authorized"] is not False or plan["launch_allowed"] is not False
            or plan["input_binding"] is not None or plan["assigned_resources"] is not None
            or plan["quantum_shots"] is not None or plan["G"] is not None
            or plan["mandatory_stop"] is not True):
        raise ValueError("DRAFT_OR_AUTHORIZATION_CHANGED")
    suites = ET.parse(root/ARTIFACT/"synthetic_tests_final.junit.xml").getroot()
    counts = {key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
    if counts != {"tests":49,"failures":0,"errors":0,"skipped":0}:
        raise ValueError("SAVED_SYNTHETIC_TEST_RECORD")
    return {"schema":"track_a_ax2b_post_review_saved_source_audit_v1", "status":"PREPARATION_STATIC_AUDIT_PASS",
            "old_frozen_sources_checked":len(frozen), "source_mismatches":[], "saved_synthetic_test_counts":counts,
            "tests_rerun_by_audit":False,"science_executed_by_audit":False,
            "H6_status":"H6_NOT_AUTHORIZED","contract_status":"DRAFT_NOT_AUTHORIZATION","mandatory_stop":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(ROOT)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False); stream.write("\n")
    print(report["status"])


if __name__ == "__main__":
    main()
