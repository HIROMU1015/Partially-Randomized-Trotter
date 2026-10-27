"""Real writer/import regression on artificial records in fresh pytest paths.

Run with PRT_VALIDATION_SCOPE=df or pauli and the corresponding pinned src path.
No scientific API is called; normal package imports retain the eager initializer.
"""
import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import trotterlib
from trotterlib import parallel_validation_executor as executor


def pipeline_record():
    scope = os.environ.get("PRT_VALIDATION_SCOPE", "df")
    if scope == "df":
        from trottertracks.resource_applicability.pipeline import compose, write_new_result_directory
        from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter
        from trottertracks.resource_applicability.pipeline_fixture import synthetic_context
        return compose(synthetic_context(), H4GeometryAdapter()), write_new_result_directory
    if scope == "pauli":
        from trottertracks.pauli_jw.pipeline import compose, write_new_result
        from trottertracks.pauli_jw.adapters import PauliStudyAdapter
        from trottertracks.pauli_jw.pipeline_fixture import synthetic_context
        return compose(synthetic_context(), PauliStudyAdapter("h4-hybrid")), write_new_result
    raise ValueError("Select one pinned DF or Pauli checkout")


def test_normal_eager_exports_and_real_writer_identity():
    import trotterlib.rte as rte
    assert trotterlib.make_rte_config is rte.make_rte_config
    assert trotterlib.CompilerSettings is rte.CompilerSettings
    assert executor.atomic_write_json.__module__ == "trotterlib.parallel_validation_executor"
    assert Path(trotterlib.__file__).name == "__init__.py"


def test_real_pipeline_writer_new_json_and_collision(tmp_path):
    record, writer = pipeline_record()
    target = writer(record, tmp_path / "new")
    assert json.loads(target.read_text()) == json.loads(json.dumps(record.to_dict()))
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        writer(record, target.parent)
    assert target.read_bytes() == before
    assert list(target.parent.iterdir()) == [target]


def test_atomic_replace_observes_complete_temp_and_old_destination(tmp_path, monkeypatch):
    path = tmp_path / "artificial.json"
    executor.atomic_write_json(path, {"old": True})
    before = path.read_bytes()
    original = executor.os.replace
    observations = []
    def inspect_replace(source, destination):
        assert Path(destination) == path
        assert path.read_bytes() == before
        assert json.loads(Path(source).read_text()) == {"new": [1, 2, None]}
        assert Path(source).parent == path.parent
        observations.append(True)
        return original(source, destination)
    monkeypatch.setattr(executor.os, "replace", inspect_replace)
    executor.atomic_write_json(path, {"new": [1, 2, None]})
    assert observations == [True]
    assert json.loads(path.read_text()) == {"new": [1, 2, None]}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("stage,error_type", [("fsync", OSError), ("replace", OSError),
                                             ("replace", KeyboardInterrupt)])
def test_write_failure_or_interruption_preserves_old_file(tmp_path, monkeypatch, stage, error_type):
    path = tmp_path / "old.json"
    executor.atomic_write_json(path, {"old": "unchanged"})
    before = path.read_bytes()
    error = error_type("ARTIFICIAL_INJECTED_WRITE_FAILURE")
    def failed(*args, **kwargs):
        raise error
    monkeypatch.setattr(executor.os, stage, failed)
    with pytest.raises(error_type) as raised:
        executor.atomic_write_json(path, {"new": "must not publish"})
    assert raised.value is error
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_preserves_old_file(tmp_path):
    path = tmp_path / "old.json"
    executor.atomic_write_json(path, {"old": True})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        executor.atomic_write_json(path, {"bad": float("nan")})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_stream_write_failure_removes_temporary_and_propagates(tmp_path, monkeypatch):
    path = tmp_path / "old.json"
    executor.atomic_write_json(path, {"old": True})
    before = path.read_bytes()
    original = Path.open
    error = OSError("ARTIFICIAL_STREAM_WRITE_FAILURE")
    class FailedStream:
        def write(self, value):
            raise error
    @contextmanager
    def open_with_failure(self, *args, **kwargs):
        with original(self, *args, **kwargs) as handle:
            yield FailedStream() if self.name.startswith(".old.json.tmp.") else handle
    monkeypatch.setattr(Path, "open", open_with_failure)
    with pytest.raises(OSError) as raised:
        executor.atomic_write_json(path, {"new": True})
    assert raised.value is error
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_failed_new_publication_has_no_partial_or_implicit_retry(tmp_path, monkeypatch):
    record, writer = pipeline_record()
    directory = tmp_path / "new"
    error = OSError("ARTIFICIAL_REPLACE_FAILURE")
    def failed(*args, **kwargs):
        raise error
    with monkeypatch.context() as patch:
        patch.setattr(executor.os, "replace", failed)
        with pytest.raises(OSError) as raised:
            writer(record, directory)
        assert raised.value is error
    assert directory.is_dir() and list(directory.iterdir()) == []
    # File publication is atomic. A failed wrapper leaves its fresh directory;
    # the collision guard refuses reuse, rather than silently retrying it.
    with pytest.raises(FileExistsError):
        writer(record, directory)


def test_existing_output_file_is_not_changed(tmp_path):
    record, writer = pipeline_record()
    existing = tmp_path / "existing"
    existing.write_bytes(b"ARTIFICIAL_OLD_FILE")
    with pytest.raises(FileExistsError):
        writer(record, existing)
    assert existing.read_bytes() == b"ARTIFICIAL_OLD_FILE"


def test_real_core_binding_is_import_only():
    if os.environ.get("PRT_VALIDATION_SCOPE", "df") == "df":
        from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter
        from trottertracks.resource_applicability.pipeline_fixture import synthetic_context
        bound = H4GeometryAdapter().bind_core(synthetic_context())
        assert bound.hamiltonian is trotterlib.DFHamiltonian
        assert bound.prepare_s2 is trotterlib.prepare_df_partial_s2
        assert bound.configure_rte is trotterlib.make_rte_config
    else:
        from trottertracks.pauli_jw import core_stages
        from trottertracks.pauli_jw.adapters import PauliStudyAdapter
        from trottertracks.pauli_jw.pipeline_fixture import synthetic_context
        for scope in ("h4-hybrid", "h6-hybrid", "h6-rte"):
            adapter = PauliStudyAdapter(scope)
            bound = core_stages.bind_existing_core(adapter, synthetic_context(scope))
            assert bound.kind == "real_unexecuted"
            for name, locator in adapter.api_paths(synthetic_context(scope)).items():
                callback = getattr(bound, name)
                assert callback is (getattr(importlib.import_module(locator[0]), locator[1])
                                    if locator is not None else None)
