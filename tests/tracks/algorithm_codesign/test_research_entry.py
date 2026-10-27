"""Pure schema, fixed references and artificial descriptor/storage tests only."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import pytest
from trottertracks.algorithm_codesign import research_entry as entry

STUDIES = ("G6", "M1-S-gamma", "M1-S-pi3")


@pytest.mark.parametrize("study", STUDIES)
def test_separate_descriptor_and_mock(study, tmp_path):
    descriptor = entry.describe(study)
    result = entry.run_mock(entry.artificial_context(study))
    assert result["scope"] == descriptor["scope"]
    assert result["scientific_values"] is None
    assert result["scientific_GO"] is None
    assert result["physical_success_guaranteed"] is False
    assert result["artificial_record"]["study"] == study
    assert result["artificial_record"]["cohort"] == "ARTIFICIAL_ONLY"
    path = entry.write_record(tmp_path / study, result)
    assert entry.storage.load_json(path) == result
    with pytest.raises(FileExistsError):
        entry.write_record(path.parent, result)


@pytest.mark.parametrize("study", STUDIES)
@pytest.mark.parametrize("bad", ["source", "scope", "cohort", "STOP", "FAILED", "UNQUALIFIED"])
def test_mismatches_cannot_dispatch(study, bad, monkeypatch):
    context = entry.artificial_context(study)
    field = {"source": "scientific_source", "scope": "scope", "cohort": "cohort"}.get(bad, "status")
    context = replace(context, **{field: bad})
    def forbidden(*args, **kwargs):
        raise AssertionError("serializer must not be reached")
    monkeypatch.setattr(entry.Evidence, "create", forbidden)
    with pytest.raises(entry.EntryError):
        entry.run_mock(context)


@pytest.mark.parametrize("bad", ["scientific", "real", "resume", "native", "generator"])
def test_closed_stages(bad):
    with pytest.raises(entry.EntryError):
        entry.describe("G6", stage=bad)
    with pytest.raises(entry.EntryError):
        entry.run_mock(entry.artificial_context("G6"), stage=bad)


def test_no_combined_study():
    with pytest.raises(entry.EntryError):
        entry.describe(["G6", "M1-S-gamma"])
    with pytest.raises(entry.EntryError):
        entry.describe("unknown")


def test_math_and_historical_states_stay_distinct():
    gamma, pi3 = entry.describe("M1-S-gamma"), entry.describe("M1-S-pi3")
    assert gamma["scope"] != pi3["scope"]
    assert gamma["scientific_source"] != pi3["scientific_source"]
    assert gamma["forecast_kind"] != pi3["forecast_kind"]
    assert gamma["input_locator"]["commit"] != pi3["input_locator"]["commit"]
    assert gamma["historical_state"] == pi3["historical_state"] == "ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL"
    assert gamma["retry"] is pi3["retry"] is False
    # Original candidate flags are not promoted by descriptor integration.
    assert entry.gamma_spec.template()["adopted"] is False
    assert entry.pi3_spec.template()["decision_gate_adopted"] is False


def test_spec_return_values_are_not_mutated():
    d = entry.describe("M1-S-gamma")
    d["input_locator"]["commit"] = "changed"
    d["input"]["commit"] = "changed"
    assert entry.describe("M1-S-gamma")["input"]["commit"] != "changed"
    assert entry.gamma_spec.INPUT["commit"] != "changed"


def test_cli_and_import_without_scientific_modules():
    script = Path(__file__).resolve().parents[3] / "scripts/tracks/algorithm_codesign/research_entry.py"
    for study in STUDIES:
        run = subprocess.run([sys.executable, str(script), "--study", study, "--stage", "mock"],
                             check=True, text=True, capture_output=True)
        assert json.loads(run.stdout)["study"] == study
    code = "from trottertracks.algorithm_codesign.research_entry import *; run_mock(artificial_context('G6')); run_mock(artificial_context('M1-S-pi3')); import sys; assert not any(x in sys.modules for x in ['numpy','scipy','mpmath','qiskit','trottertracks.algorithm_codesign.g6_f3_generator','trottertracks.algorithm_codesign.m1s_gamma_pi3_v1.control'])"
    subprocess.run([sys.executable, "-c", code], check=True)
