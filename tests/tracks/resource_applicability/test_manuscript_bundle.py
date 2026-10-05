"""Verifier helper tests use synthetic rows and temporary Markdown only."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "scripts/resource_applicability/verify_track_a_manuscript_bundle.py"
spec = importlib.util.spec_from_file_location("manuscript_bundle", SOURCE)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_exact_display_rows():
    bundle.compare_rows([{"candidate": "PM1-B0", "missing": "MISSING"}],
                        [{"candidate": "PM1-B0", "missing": "MISSING"}])


def test_row_count_mismatch():
    with pytest.raises(ValueError, match="count"):
        bundle.compare_rows([], [{"x": "1"}])


def test_altered_value_rejected():
    with pytest.raises(ValueError, match="differs"):
        bundle.compare_rows([{"x": "0"}], [{"x": "MISSING"}])


@pytest.mark.parametrize("target", ["held_out.npz", "state.npy", ".runtime/x.json", "../outside.md"])
def test_unsafe_link_rejected_before_file_probe(tmp_path, monkeypatch, target):
    def forbidden(*args, **kwargs):
        raise AssertionError("Protected path must not be probed")
    monkeypatch.setattr(Path, "is_file", forbidden)
    with pytest.raises(ValueError, match="Non-allowlisted"):
        bundle.safe_link(tmp_path, tmp_path / "doc.md", target)


def test_local_markdown_link(tmp_path):
    (tmp_path / "target.md").write_text("synthetic")
    assert bundle.safe_link(tmp_path, tmp_path / "doc.md", "target.md#part") == "target.md"


def test_external_link_not_opened(tmp_path):
    assert bundle.safe_link(tmp_path, tmp_path / "doc.md", "https://example.com/") is None
