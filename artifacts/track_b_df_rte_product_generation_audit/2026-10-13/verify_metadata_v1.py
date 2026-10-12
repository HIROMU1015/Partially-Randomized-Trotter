"""Offline documentation/source/saved-metadata verifier; no science imports.

Uses Git objects, JSON, CSV and Python AST only. This is not an algorithm runner,
scientific test, numerical certificate, or authorization gate for a new run.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
from urllib.parse import unquote


BASE = "4192077723770f4846cc391ea0da96c1e115f2fa"
SOURCE = "321cd8c3066194fe98741cbebbcdba1515d1deaa"
FOLDER = "artifacts/track_b_df_rte_product_generation_audit/2026-10-13"
ORIGINAL = "docs/research/track_b_post_closure_research_direction_review_20261013.md"
SHA = "caeef1dc9d4f74b7e6b7e7ec3671c3039f42ddbb2232d95dc10f8ea019e21289"
INDEXES = (
    "PROJECT_MAP.md",
    "docs/README.md",
    "docs/research/README.md",
    "docs/tracks/algorithm_codesign/README.md",
    "docs/research/研究概要・現状.md",
)
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / FOLDER


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checks = []

    def check(name, condition):
        checks.append({"name": name, "pass": bool(condition)})

    original = (ROOT / ORIGINAL).read_bytes()
    check("original_review_bytes", len(original) == 41995)
    check("original_review_sha256", hashlib.sha256(original).hexdigest() == SHA)
    provenance = read("input_provenance_v1.json")
    receipt = read("input_publication_receipt_v1.json")
    check("input_provenance_identity", provenance["original_review"]["sha256"] == SHA)
    check("input_remote_retrieval_record", receipt["original_bytes_equal"] and receipt["sha256"] == SHA)
    check("input_commit_original_bytes", git("show", receipt["input_commit"] + ":" + ORIGINAL) == original)

    inventory = read("source_identity_inventory_v1.json")
    check("inventory_source_and_saved_file_count", len(inventory["entries"]) == 14)
    for entry in inventory["entries"]:
        obj = entry["commit"] + ":" + entry["path"]
        b = git("show", obj)
        check("input_identity:" + entry["path"],
              len(b) == entry["bytes"] and hashlib.sha256(b).hexdigest() == entry["sha256"]
              and git("rev-parse", obj).decode().strip() == entry["git_blob"])
        if entry["path"].endswith(".py"):
            nodes = ast.walk(ast.parse(b))
            symbols = [{"name": n.name, "line": n.lineno, "end_line": n.end_lineno}
                       for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
            check("static_AST_positions:" + entry["path"], symbols == entry["symbols"])

    saved = read("saved_cost_scope_v1.json")
    rare = json.loads(git("show", SOURCE + ":" + saved["input_path"]))
    check("eight_saved_only_rows", len(saved["rows"]) == len(rare["rows"]) == 8)
    for actual, old in zip(saved["rows"], rare["rows"]):
        check("saved_law_fields:" + actual["cell_id"],
              actual["order_probabilities_saved"] == old["orders"]
              and actual["observed_order_slot_counts_saved"] == old["observed_counts"]
              and actual["R"] == old["R"] and actual["trajectories"] == old["trajectories"])
        check("saved_slot_counts:" + actual["cell_id"],
              sum(old["observed_counts"].values()) == old["R"] * old["trajectories"])
        check("descriptive_arithmetic:" + actual["cell_id"],
              actual["nonzero_order_trajectory_probability_descriptive_complement"]
              == 1.0 - old["all_zero_order_trajectory_probability"]
              and actual["expected_product_slots_per_trajectory_descriptive"]
              == old["R"] * sum(int(k) * p for k, p in old["orders"].items()))
        check("unknown_not_zero:" + actual["cell_id"],
              all(actual[k] is None for k in ("basis_span_dimension", "conditional_rare_wrapper_cost",
                                             "population_improvable_cost_upper")))

    prefix = "artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/"
    rows = list(csv.DictReader(io.StringIO(git("show", SOURCE + ":" + prefix + "wrapper_metrics_v1.csv").decode())))
    confirmation = [r for r in rows if r["phase"] == "confirmation"]
    check("512_saved_confirmation_wrapper_rows", len(confirmation) == 512)
    check("public_csv_has_no_order_or_span_columns", not {"event_orders", "observed_orders", "span_rank"}.intersection(rows[0]))
    check("saved_cost_scope_no_science", saved["new_science"] == 0 and not saved["confidence_or_winner_reclassification"])

    boundary = read("boundary_record_v1.json")
    check("scientific_operations_all_zero", all(v == 0 for v in boundary["scientific_operations"].values()))
    check("research_and_execution_boundary", boundary["status"] == "STOP_FOR_GPT_SCOPE_DECISION"
          and not boundary["registered_execution_authorized"] and not boundary["gate_3_prepared"]
          and not boundary["new_algorithm_adopted"] and not boundary["research_closure_decided_by_codex"]
          and boundary["mandatory_STOP"])

    for path in INDEXES:
        check("append_only_index:" + path, (ROOT / path).read_bytes().startswith(git("show", BASE + ":" + path)))
    manifest = read("evidence_manifest_v1.json")
    for entry in manifest["files"]:
        b = (ROOT / entry["path"]).read_bytes()
        check("output_identity:" + entry["path"], len(b) == entry["bytes"] and hashlib.sha256(b).hexdigest() == entry["sha256"])
    for path in manifest["markdown_local_link_scope"]:
        text = (ROOT / path).read_text(encoding="utf-8")
        if path in INDEXES:
            old = git("show", BASE + ":" + path).decode()
            text = text[len(old):]
        for dest in re.findall(r"\]\(([^)]+)\)", text):
            if dest.startswith(("http:", "https:", "#")):
                continue
            target = ((ROOT / path).parent / unquote(dest.split("#", 1)[0])).resolve()
            check("local_link:" + path + ":" + dest, target.exists() and target.is_relative_to(ROOT))

    result = {
        "schema_version": 1,
        "status": "METADATA_PASS" if all(c["pass"] for c in checks) else "METADATA_FAIL",
        "checks_count": len(checks), "passed": sum(c["pass"] for c in checks),
        "failed": [c for c in checks if not c["pass"]], "checks": checks,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "scientific_tests_executed": 0, "registered_modules_imported": 0,
        "scope": "Static Git/AST and saved JSON/CSV metadata bookkeeping only; no mathematical proof or performance certification.",
        "mandatory_STOP": True,
    }
    if args.write:
        (OUT / "metadata_validation_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "METADATA_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
