from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from trotterlib.research_direction_full_opt2_extension_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    AUDIT_SCHEMA_VERSION,
    evaluate_coherent_extension_analysis,
    evaluate_extension_audit,
    validate_extension_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/research_direction_full_opt2/2026-09-24"
FINAL = ROOT / "artifacts/research_direction_full_opt2/2026-09-25"
AUDIT_PATH = FINAL / "wp11_all_r_opt2_fresh32_audit_20260925_065827.json"
ANALYSIS_PATH = FINAL / "wp11_all_r_opt2_coherent_analysis_20260925_065827.json"
RAW_OUTPUTS = (
    BASE / "wp11_all_r_opt2_initial_20260924_023012",
    BASE / "wp11_all_r_opt2_fresh32_20260924_232418",
)
RAW_EVIDENCE_AVAILABLE = all(
    all(
        (output / directory).is_dir()
        for directory in ("checkpoints", "tasks", "worker_results")
    )
    for output in RAW_OUTPUTS
)


def _artifact(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _audit() -> dict[str, object]:
    return evaluate_extension_audit(
        project_root=ROOT,
        initial_manifest_path=BASE
        / "wp11_all_r_opt2_initial_20260924_023012.manifest.json",
        initial_output_dir=BASE / "wp11_all_r_opt2_initial_20260924_023012",
        extension_manifest_path=BASE
        / "wp11_all_r_opt2_extension_20260924_222811.manifest.json",
        extension_output_dir=BASE / "wp11_all_r_opt2_fresh32_20260924_232418",
        initial_analysis_path=BASE
        / "wp11_all_r_opt2_analysis_20260924_222811.json",
    )


def test_fresh32_audit_confirms_51_complete_and_both_gates() -> None:
    body = _artifact(AUDIT_PATH)
    validate_extension_artifact(body, schema_version=AUDIT_SCHEMA_VERSION)
    assert body["overall_pass"] is True
    assert body["combined_counts"] == {
        "expected": 51,
        "completed": 51,
        "failed": 0,
    }
    assert body["gate_summary"]["all_proxy_cells_pass"] is True
    assert body["gate_summary"]["maximum_direct_rz_relative_standard_error"] <= 0.02
    assert body["gate_summary"]["maximum_selected_all_metrics_holdout_error"] <= 0.05
    assert body["seed_audit"]["cross_batch_task_seed_overlap"] == []
    assert body["seed_audit"]["cross_batch_trajectory_seed_overlap"] == []
    direct = body["direct_compiled_rz_measurements"]
    assert len(direct) == 14
    assert direct["ld3:delta0.02:r32"]["source_partition"] == "fresh32_extension"
    assert set(direct["ld3:delta0.02:r32"]["q"]) == {"1", "2", "8"}
    assert direct["ld3:delta0.02:r32"]["q"]["8"]["role"] == "fixed_holdout"
    assert direct["ld3:delta0.02:r32"]["q"]["8"]["compiled_rz"][
        "support_run_le_1"
    ]["cosine"]["standard_error"] > 0.0


def test_fresh32_audit_fingerprint_is_tamper_evident() -> None:
    artifact = _artifact(AUDIT_PATH)
    validate_extension_artifact(artifact, schema_version=AUDIT_SCHEMA_VERSION)
    tampered = deepcopy(artifact)
    tampered["combined_counts"]["completed"] = 50
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_extension_artifact(tampered, schema_version=AUDIT_SCHEMA_VERSION)


def test_fresh32_evidence_runs_coherent_reoptimization() -> None:
    body = _artifact(ANALYSIS_PATH)
    validate_extension_artifact(body, schema_version=ANALYSIS_SCHEMA_VERSION)
    assert body["overall_pass"] is True
    assert body["status"] == "coherent_opt2_reoptimization_complete"
    assert set(body["coherent_reoptimization"]["candidates"]) == {
        "ld3:delta0.01",
        "ld3:delta0.02",
        "ld12:delta0.01",
        "ld12:delta0.02",
    }
    difference = body["mixed_to_coherent_difference"]
    assert difference["ld3_compiled_rz_relative_change"] == pytest.approx(
        0.038917956428395684
    )
    assert difference["advantage_fraction_change"] < 0.0


@pytest.mark.skipif(
    not RAW_EVIDENCE_AVAILABLE,
    reason="raw M06-F task/checkpoint/worker evidence is not tracked by Git",
)
def test_raw_fresh32_evidence_rebuilds_when_available() -> None:
    audit = _audit()
    assert audit["overall_pass"] is True
    body = evaluate_coherent_extension_analysis(
        project_root=ROOT,
        initial_manifest_path=BASE
        / "wp11_all_r_opt2_initial_20260924_023012.manifest.json",
        initial_output_dir=BASE / "wp11_all_r_opt2_initial_20260924_023012",
        extension_manifest_path=BASE
        / "wp11_all_r_opt2_extension_20260924_222811.manifest.json",
        extension_output_dir=BASE / "wp11_all_r_opt2_fresh32_20260924_232418",
        initial_analysis_path=BASE
        / "wp11_all_r_opt2_analysis_20260924_222811.json",
    )
    committed = _artifact(ANALYSIS_PATH)
    assert body["mixed_to_coherent_difference"][
        "ld3_compiled_rz_relative_change"
    ] == pytest.approx(
        committed["mixed_to_coherent_difference"][
            "ld3_compiled_rz_relative_change"
        ],
        rel=1e-6,
    )
