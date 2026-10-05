"""Verify the display/document bundle; never run science or molecular tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
BUILDER = ROOT / "scripts/resource_applicability/build_track_a_manuscript_figures.py"
SPEC = importlib.util.spec_from_file_location("track_a_display", BUILDER)
display = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(display)
DOCS = tuple("docs/manuscripts/" + name for name in (
    "track_a_resource_study_v0_1.md", "track_a_resource_study_supplement_v0_1.md",
    "track_a_resource_study_claim_audit_v0_1.md", "track_a_publication_readiness_review_request_v0_1.md"))
TESTS = ("tests/tracks/resource_applicability/test_manuscript_figures.py",
         "tests/tracks/resource_applicability/test_manuscript_bundle.py")
SAFE_SUFFIXES = {".md", ".py", ".json", ".csv", ".png", ".svg", ".pdf"}
STEMS = ("figure_1_development_cost_components", "figure_2_development_precision",
         "figure_3_frozen_transfer_precision", "figure_4_same_R_competition",
         "figure_S1_preparation_sensitivity")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(root, relative):
    raw = (root / relative).read_bytes()
    return {"path": relative, "bytes": len(raw), "sha256": digest(raw)}


def safe_link(root, document, target, pending_outputs=()):
    if target.startswith(("https://", "http://", "#", "mailto:")):
        return None
    target = target.split("#", 1)[0]
    path = Path(os.path.normpath(document.parent / target))
    if (not path.is_relative_to(root) or path.suffix not in SAFE_SUFFIXES
            or any(part in {".runtime", ".registry"} for part in path.parts)):
        raise ValueError("Non-allowlisted document target: " + target)
    if path in pending_outputs:
        return str(path.relative_to(root))  # Checked again after exporting this audit.
    # Only stat an explicitly permitted documentation/display target.
    if not path.is_file():
        raise ValueError("Broken document link: " + str(path))
    return str(path.relative_to(root))


def compare_rows(actual, expected):
    if len(actual) != len(expected):
        raise ValueError("Display row count mismatch")
    for observed, saved in zip(actual, expected):
        if any(observed.get(key) != str(value) for key, value in saved.items()):
            raise ValueError("Display value differs from saved source")


def verify(root, pending_outputs=()):
    data, inputs = display.verify_inputs(root)
    reference_sources = []
    for relative in ("src/trotterlib/rte.py", "src/trotterlib/pr2_matched_accuracy_m1_execution.py"):
        raw = (root / relative).read_bytes()  # Source text only; never import a science module.
        blob = subprocess.check_output(["git", "show", display.EVIDENCE_COMMIT + ":" + relative], cwd=root)
        if raw != blob:
            raise ValueError("Methods reference source differs from evidence commit")
        reference_sources.append(dict(file_identity(root, relative), commit_blob_identical=True))
    domains, fig1, minima, global_min, fig4 = display.display_selection(data)
    out = root / display.DEFAULT_OUTPUT
    csv_expected = {
        "figure_1_values.csv": fig1,
        "figure_2_method_minima.csv": minima,
        "figure_2_point_minimum_settings.csv": global_min,
        "figure_3_fixed_five.csv": [r for e in sorted(domains["transfer_fixed_five"]) for r in domains["transfer_fixed_five"][e]],
        "figure_4_values.csv": fig4,
        "original_precision_223_candidates.csv": domains["development"][.05] + domains["transfer_fixed_five"][.05],
    }
    expected_names = {s + "." + ext for s in STEMS for ext in ("png", "svg", "pdf")} | set(csv_expected) | {"display_audit.json"}
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["kind"] != "MANUSCRIPT_ASSET_MANIFEST_NOT_SCIENCE_MANIFEST":
        raise ValueError("Wrong manifest kind")
    if (len(manifest["files"]) != 22 or {r["path"] for r in manifest["files"]} != expected_names
            or {p.name for p in out.iterdir()} != expected_names | {"manifest.json"}):
        raise ValueError("Unexpected asset file set")
    for row in manifest["files"]:
        path = out / row["path"]
        if path.is_symlink():
            raise ValueError("Asset symlink not allowed")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError("Asset hash mismatch: " + row["path"])
    for name, expected in csv_expected.items():
        compare_rows(display.csv_rows((out / name).read_bytes()), expected)
    audit = json.loads((out / "display_audit.json").read_text())
    if audit["builder_sha256"] != digest(BUILDER.read_bytes()) or audit["input_identity"] != inputs:
        raise ValueError("Builder/input audit mismatch")
    if any(audit[k] != 0 for k in ("new_signal", "new_trajectory", "new_compile", "molecular_data_access", "GPU_operations")):
        raise ValueError("Display is not science-free")
    links = []
    for relative in DOCS:
        document = root / relative
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            checked = safe_link(root, document, target, pending_outputs)
            if checked:
                links.append({"document": relative, "target": checked})
    return {
        "status": "AUTHOR_BUNDLE_VERIFICATION_PASS_NOT_INDEPENDENT_REVIEW",
        "evidence_commit": display.EVIDENCE_COMMIT, "design_commit": display.DESIGN_COMMIT,
        "input_identity": inputs, "input_commit_blob_matches": 9, "asset_files_verified": 22,
        "display_rows": {name: len(rows) for name, rows in csv_expected.items()},
        "local_document_links": links,
        "document_identity": [file_identity(root, p) for p in DOCS],
        "source_identity": [file_identity(root, p) for p in (
            "scripts/resource_applicability/build_track_a_manuscript_figures.py",
            "scripts/resource_applicability/verify_track_a_manuscript_bundle.py", *TESTS)],
        "methods_reference_source_identity": reference_sources,
        "asset_manifest_identity": file_identity(root, display.DEFAULT_OUTPUT + "/manifest.json"),
        "claim_audit_note": "Semantic claim checks are the author's manual audit, not proved by keyword matching.",
        "new_signal": 0, "new_trajectory": 0, "new_compile": 0, "molecular_data_access": 0,
        "GPU_operations": 0, "scientific_next_stage_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/resource_applicability/track_a_manuscript_audit/2026-10-05")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Do not overwrite an existing verification bundle")
    pending = (args.output_dir / "verification.json",)
    before = verify(ROOT, pending)
    command = [str(Path(os.sys.executable).absolute()), "-m", "pytest", "-q", *TESTS, "-p", "no:cacheprovider"]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    ended = datetime.now(timezone.utc).isoformat()
    if result.returncode or re.search(r"\b\d+ (failed|skipped|errors?)\b", result.stdout):
        raise ValueError("Manuscript synthetic test gate failed: " + result.stdout + result.stderr)
    matched = re.search(r"(\d+) passed", result.stdout)
    if not matched:
        raise ValueError("Test pass count missing")
    after = verify(ROOT, pending)
    if before != after:
        raise ValueError("Bundle changed during tests")
    after["local_tests"] = {
        "command_argv": command, "cwd": str(ROOT), "python": platform.python_version(),
        "environment_overrides": {k: env[k] for k in (
            "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "started_utc": started, "ended_utc": ended, "passed": int(matched[1]),
        "failed": 0, "skipped": 0, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr, "immutable_CI": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "verification.json").write_text(json.dumps(after, indent=2, ensure_ascii=False) + "\n")
    if verify(ROOT) != before:
        raise ValueError("Exported audit link/bundle verification failed")
    raw = (args.output_dir / "verification.json").read_bytes()
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "kind": "MANUSCRIPT_VERIFICATION_MANIFEST_NOT_SCIENCE_MANIFEST",
        "files": [{"path": "verification.json", "bytes": len(raw), "sha256": digest(raw)}]}, indent=2) + "\n")
    print(json.dumps({"status": after["status"], "asset_files": 22,
                      "tests_passed": int(matched[1]), "new_science": 0}))


if __name__ == "__main__":
    main()
