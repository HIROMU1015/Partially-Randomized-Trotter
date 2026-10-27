"""Artificial descriptor routing only; no scientific planner/backend is run."""
from dataclasses import replace
from fractions import Fraction
import importlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from trottertracks.b2d_adapter.adapter import BlockExponential
from trottertracks.b2d_evaluator import research_entry as entry


@pytest.mark.parametrize("stage", entry.STAGES)
def test_stage_plan_and_mock(stage):
    request = entry.artificial_request(stage)
    result = entry.run_mock(request)
    assert result["historical_state"] == "FINAL_PAUSE_PRESERVED"
    assert result["scientific_stage_evaluated"] is False
    assert all(value is None for value in result["qualifications"].values())
    evaluated = result["evaluation"]
    assert evaluated["mode"] == "MOCK_ONLY"
    assert sum(evaluated["call_counts"].values()) == request.call_budget
    row = evaluated["rows"][0]["outputs"]["descriptor_record"]
    assert row["source"] == row["cohort"] == "ARTIFICIAL_ONLY"
    assert row["signed_order"][1]["evolution_time"] == "-1/3"
    assert row["actual_adjoint"][0]["evolution_time"] == "1/3"
    assert row["actual_adjoint"][1]["evolution_time"] == "-1/2"
    assert row["processor"] == row["signed_order"][:1]
    assert result["locator"]["source_commit"] != entry.FROZEN_BASE


@pytest.mark.parametrize("mode", ["scientific", "real", "resume", "execute"])
def test_modes_closed_before_backend(mode):
    backend = entry.MockTransport()
    with pytest.raises(entry.EntryError):
        entry.run_mock(entry.artificial_request("B0"), mode=mode, backend=backend)
    assert backend.calls == []


@pytest.mark.parametrize("patch", [
    {"source": "physical"}, {"cohort": "saved"}, {"law": "shared"},
    {"method": "unknown"}, {"call_budget": 2}, {"call_budget": True},
    {"status": "STOP"}, {"status": "FAILED"}, {"status": "MISSING"},
    {"status": "UNQUALIFIED"}, {"signed_order": ()}, {"processor": ()},
    {"signed_order": (BlockExponential("real-D", Fraction(1), False, None),)},
    {"signed_order": (BlockExponential("mock-D", 1.0, False, None),)},
    {"signed_order": (BlockExponential("mock-D", Fraction(1), True, 2),)},
])
def test_invalid_binding_never_dispatches(patch):
    backend = entry.MockTransport()
    with pytest.raises(entry.EntryError):
        entry.run_mock(replace(entry.artificial_request("B0"), **patch), backend=backend)
    assert backend.calls == []


def test_q4_requires_own_method():
    with pytest.raises(entry.EntryError):
        entry.run_mock(replace(entry.artificial_request("Full-DF-q4"), method="S2"))


def test_no_native_backend_or_reuse():
    with pytest.raises(entry.EntryError):
        entry.run_mock(entry.artificial_request("A0"), backend=object())
    backend = entry.MockTransport()
    entry.run_mock(entry.artificial_request("A0"), backend=backend)
    with pytest.raises(entry.EntryError):
        entry.run_mock(entry.artificial_request("A0"), backend=backend)


def test_existing_evaluator_is_called(monkeypatch):
    calls = []
    original = entry.evaluate
    def traced(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(entry, "evaluate", traced)
    entry.run_mock(entry.artificial_request("Outer"))
    assert calls == [{"mock_only": True}]


def test_record_collision(tmp_path):
    record = entry.run_mock(entry.artificial_request("B1"))
    path = entry.write_record(tmp_path / "new", record)
    assert json.loads(path.read_text()) == record
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        entry.write_record(path.parent, record)
    assert path.read_bytes() == before


def test_cli_plan_and_mock(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/run_b2d_research_entry.py"
    for mode in ("plan", "mock"):
        run = subprocess.run([sys.executable, str(script), "--stage", "B1", "--mode", mode],
                             capture_output=True, text=True, check=True)
        assert json.loads(run.stdout)["mode"] == mode
    bad = subprocess.run([sys.executable, str(script), "--stage", "B1", "--mode", "resume"],
                         capture_output=True, text=True)
    assert bad.returncode != 0


def test_no_numerical_backend_import_in_clean_process():
    command = "from trottertracks.b2d_evaluator.research_entry import *; run_mock(artificial_request('B0')); import sys; assert not any(x in sys.modules for x in ['numpy','qiskit','mpmath','scipy','trottertracks.b2d_evaluator.dense_backend','trottertracks.b2d_evaluator.b0_native'])"
    subprocess.run([sys.executable, "-c", command], check=True)
