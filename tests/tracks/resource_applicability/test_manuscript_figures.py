"""Synthetic/static tests only: no evidence or molecular input is opened."""
import ast
import importlib.util
import math
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "scripts/resource_applicability/build_track_a_manuscript_figures.py"
spec = importlib.util.spec_from_file_location("manuscript_figures", SOURCE)
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def test_missing_is_nan_not_zero():
    assert math.isnan(figures.value({"accuracy_eligible": "False", "primary_RZ_P0": "MISSING"}, "primary_RZ_P0"))


def test_eligible_missing_rejected():
    with pytest.raises(ValueError):
        figures.value({"accuracy_eligible": "True", "primary_RZ_P0": "MISSING"}, "primary_RZ_P0")


@pytest.mark.parametrize("flag", ["true", "", "0"])
def test_invalid_flags_rejected(flag):
    with pytest.raises(ValueError):
        figures.eligible({"accuracy_eligible": flag})


def test_additional_discard_identity_retained():
    assert figures.FIG1_IDS[0] == "PM1-B0-rank5-q1-r0-K0"
    assert figures.method(figures.FIG1_IDS[0]) == "B0"


def test_transfer_not_expanded():
    assert len(set(figures.TRANSFER_IDS)) == 5
    assert not any("PM1" in c or "-q2-" in c or "-q4-" in c for c in figures.TRANSFER_IDS)


def test_inputs_are_explicit_saved_csv_json_only():
    assert len(figures.INPUTS) == 9
    assert all(Path(p).suffix in (".csv", ".json") for p in figures.INPUTS)
    assert all("runtime" not in p.lower() and "registry" not in p.lower() for p in figures.INPUTS)


def test_no_scientific_imports_or_shot_recomputation():
    tree = ast.parse(SOURCE.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    assert not set(imports) & {"trotterlib", "trottertracks", "qiskit", "pyscf", "cupy", "scipy"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in {"ceil", "load", "save"} for n in ast.walk(tree))


def test_corrupt_identity_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "INPUTS", {"saved.csv": "0" * 64})
    (tmp_path / "saved.csv").write_text("x\n1\n")
    monkeypatch.setattr(figures.subprocess, "check_output", lambda *a, **kw: b"x\n1\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="identity mismatch"):
        figures.build(tmp_path, output)
    assert not output.exists()


def test_csv_preserves_candidate_and_missing(tmp_path):
    output = tmp_path / "display.csv"
    figures.export_csv(output, [{"candidate_id": "PM1-B0-example", "work": "MISSING"}])
    rows = figures.csv_rows(output.read_bytes())
    assert rows == [{"candidate_id": "PM1-B0-example", "work": "MISSING"}]


def test_preparation_display_is_affine_not_log_log_endpoint_interpolation():
    xs, ys = figures.affine_segment_display({"P_min": "1", "P_max": "100", "offset": "100", "slope": "2"}, "offset", "slope")
    assert len(xs) == len(ys) == 101
    assert xs[50] == pytest.approx(10)
    assert ys[50] == pytest.approx(120)
    assert ys[0] == pytest.approx(102)
    assert ys[-1] == pytest.approx(300)
