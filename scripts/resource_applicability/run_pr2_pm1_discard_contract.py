#!/usr/bin/env python3
"""Print a zero-science PM-1 preparation bundle. No project output writes."""
import argparse
import json
from pathlib import Path

from trottertracks.resource_applicability import pm1_discard_contract as c


def build_bundle(root, source_commit=None):
    values, audit = c.load_saved_inputs(root)
    refs, target = c.frozen_references(values)
    sources = c.source_inventory(root, source_commit)
    plan = c.make_plan(refs, target, sources, audit, source_commit=source_commit)
    c.validate_plan(plan)
    keys = [c.wrapper_key(plan, candidate, axis) for candidate in plan["candidates"] for axis in c.AXES]
    c.require(len(keys) == len(set(keys)) == 16, "wrapper identity collision")
    return {"zero_science_plan_v1.json": plan, "plan_schema_v1.json": c.plan_schema(),
            "reserved_result_schema_v1.json": c.result_schema(),
            "source_inventory_v1.json": {"source_commit": source_commit, "source_binding": plan["source_binding"], "sha256": sources},
            "wrapper_identity_plan_v1.json": {"plan_fingerprint": plan["plan_fingerprint"], "wrapper_keys": keys, "wrapper_count": 16},
            "preparation_access_audit_v1.json": {
                "phase": "resumed_text_only_preparation", "saved_JSON_inputs": audit,
                "molecular_path_policy": "literal_only_not_resolved_statted_hashed_loaded",
                "current_preparation_npz_stat_hash_load": 0, "current_science_evaluations": 0,
                "current_H4_circuit_build_compile": 0, "current_random_sampling": 0,
                "current_held_out_access": 0, "current_GPU_operations": 0,
                "prior_attempt_incident": {"npz_stat_hash_accesses": 4, "npz_loads": 0,
                    "science_evaluations": 0, "disclosed_to_user": True,
                    "user_directed_resumption_under_document_only_scope": True,
                    "used_M2_held_out_is_not_fresh_blind_data": True},
                "execution_authorized": False, "next_stage_authorized": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-commit", help="Optional existing science-source commit for blob binding; NOT authorization")
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.project_root, args.source_commit), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
