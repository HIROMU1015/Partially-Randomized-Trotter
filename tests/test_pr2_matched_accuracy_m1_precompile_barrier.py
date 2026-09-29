from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_contract.py"
BARRIER_SOURCE = (
    ROOT / "src/trotterlib/pr2_matched_accuracy_m1_precompile_barrier.py"
)
RUNNER = ROOT / "scripts/run_pr2_matched_accuracy_m1_precompile_barrier.py"
ARTIFACT_ROOT = ROOT / "artifacts/pr2_matched_accuracy_m1_contract/2026-09-29"
AUTHORIZATION = (
    ARTIFACT_ROOT
    / "pr2_matched_accuracy_m1_preexecution_amendment_authorization_v2.json"
)
DRY_RUN = (
    ARTIFACT_ROOT
    / "pr2_matched_accuracy_m1_precompile_barrier_dry_run_v2.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract():
    return _load_module("pr2_m1_contract_v1_barrier_test", V1_SOURCE)


@pytest.fixture(scope="module")
def barrier():
    return _load_module("pr2_m1_precompile_barrier_test", BARRIER_SOURCE)


@pytest.fixture(scope="module")
def limited_selection(contract):
    candidates = contract.enumerate_base_candidates()
    return contract.selector_fixture(candidates)["selection"]


def _clear_selection(selection):
    clear = copy.deepcopy(selection)
    clear["unselected_proxy_frontier_count"] = 0
    clear["unselected_proxy_frontier_fingerprints"] = []
    clear["unselected_boundary_fingerprints"] = []
    clear["unselected_tail_challenger_fingerprints"] = []
    clear["selection_limited"] = False
    clear["selection_limited_reasons"] = []
    return clear


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_limited_selection_is_a_terminal_precompile_stop(
    barrier, limited_selection
):
    result = barrier.evaluate_precompile_barrier(limited_selection)
    assert result["status"] == "SELECTION_LIMITED"
    assert result["selection_limited"] is True
    assert result["m1_b_direct_compile_eligible"] is False
    assert result["compile_jobs_materialized_at_barrier"] == 0
    assert result["mandatory_stop_reached"] is True
    assert result["winner_claim_permitted"] is False
    with pytest.raises(barrier.SelectionLimitedStop):
        barrier.build_m1_b_compile_plan(
            limited_selection,
            deterministic_or_discard_fingerprints=[],
        )


def test_clear_selection_can_only_create_a_zero_compute_cell_plan(
    contract, barrier, limited_selection
):
    clear = _clear_selection(limited_selection)
    result = barrier.evaluate_precompile_barrier(clear)
    assert result["status"] == "M1_A_COMPLETE_M1_B_ELIGIBLE"
    assert result["m1_b_direct_compile_eligible"] is True
    assert result["m1_b_direct_compile_authorized_by_current_amendment"] is False
    deterministic = [
        item["candidate_fingerprint"]
        for item in contract.enumerate_base_candidates()
        if item["method"] in {"B0", "B1"}
    ]
    plan = barrier.build_m1_b_compile_plan(
        clear,
        deterministic_or_discard_fingerprints=deterministic,
    )
    assert plan["deterministic_or_discard_compile_cell_count"] == 16
    assert plan["random_compile_cell_count"] == 16
    assert plan["compile_cell_count"] == 32
    assert plan["plan_only"] is True
    assert plan["circuits_built"] == 0
    assert plan["circuit_compilations"] == 0
    assert plan["full_wrappers_compiled"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compile_cap", 17),
        ("selected_count", 15),
        ("selection_limited", False),
        ("selection_limited_reasons", []),
    ],
)
def test_selector_tampering_is_rejected(
    barrier, limited_selection, field, value
):
    tampered = copy.deepcopy(limited_selection)
    tampered[field] = value
    with pytest.raises(ValueError):
        barrier.validate_compile_selection(tampered)


def test_authorization_hashes_and_permissions_are_frozen():
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["status"] == (
        "M1_PREEXECUTION_AMENDMENT_V2_FROZEN_SCIENCE_NOT_AUTHORIZED"
    )
    assert authorization["prior_art_gate"]["decision"] == (
        "PROCEED_RESOURCE_STUDY"
    )
    assert authorization["prior_art_gate"]["papers_added"] == [
        "arXiv:2603.13495",
        "arXiv:2603.22778",
    ]
    for relative, expected in authorization["required_source_hashes"].items():
        assert _sha256(ROOT / relative) == expected
    permissions = authorization["permissions"]
    assert permissions["amendment_dry_run_authorized"] is True
    assert permissions["m1_scientific_execution_authorized"] is False
    assert permissions["development_npz_load_authorized"] is False
    assert permissions["held_out_access_authorized"] is False
    assert permissions["signal_evaluation_authorized"] is False
    assert permissions["trajectory_sampling_authorized"] is False
    assert permissions["circuit_build_authorized"] is False
    assert permissions["direct_compile_authorized"] is False
    assert permissions["s3_authorized"] is False
    assert permissions["automatic_next_stage"] is None


def test_committed_dry_run_exercises_both_branches_without_science():
    payload = json.loads(DRY_RUN.read_text(encoding="utf-8"))
    unsigned = dict(payload)
    observed = unsigned.pop("artifact_fingerprint")
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == observed
    assert payload["limited_fixture"]["barrier"]["status"] == (
        "SELECTION_LIMITED"
    )
    assert payload["limited_fixture"]["compile_plan_created"] is False
    assert payload["limited_fixture"]["compile_jobs_created"] == 0
    assert payload["clear_control"]["barrier"]["status"] == (
        "M1_A_COMPLETE_M1_B_ELIGIBLE"
    )
    assert all(value == 0 for value in payload["zero_compute_counters"].values())
    assert payload["permissions"]["m1_scientific_execution_authorized"] is False
    assert payload["mandatory_stop_reached"] is True


def test_runner_is_non_overwriting_and_reproduces_dry_run(tmp_path):
    output = tmp_path / "dry_run.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--authorization",
            str(AUTHORIZATION),
            "--artifact",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        DRY_RUN.read_text(encoding="utf-8")
    )
    assert json.loads(completed.stdout)["limited_compile_jobs_created"] == 0
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--authorization",
                str(AUTHORIZATION),
                "--artifact",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_new_source_is_standard_library_only_and_has_no_science_access():
    allowed = {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "importlib",
        "json",
        "pathlib",
        "sys",
        "types",
        "typing",
    }
    for path in (BARRIER_SOURCE, RUNNER):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= allowed
        text = path.read_text(encoding="utf-8")
        assert "numpy" not in text
        assert "qiskit" not in text
        assert "np.load" not in text
        assert "transpile(" not in text
