"""No real inputs or compile: prevent recurrence of missing native lowering."""
import importlib.metadata as md
import json
import sys
import pytest
from trottertracks.resource_applicability import h6_server_native_dependencies_v2 as d


@pytest.mark.parametrize('name',list(d.REQUIRED))
def test_missing_dependency_fails_closed(monkeypatch,name):
    def version(package):
        if package==name:raise md.PackageNotFoundError(package)
        return d.REQUIRED[package]
    monkeypatch.setattr(d.md,'version',version)
    with pytest.raises(ValueError,match='DEPENDENCY_MISSING:'+name):d.versions()


@pytest.mark.parametrize('name',list(d.REQUIRED))
def test_changed_dependency_version_fails_closed(monkeypatch,name):
    monkeypatch.setattr(d.md,'version',lambda package:'99.0' if package==name else d.REQUIRED[package])
    with pytest.raises(ValueError,match='DEPENDENCY_VERSION:'+name):d.versions()


def test_installed_frozen_native_import_only():
    result=d.native_dependencies()
    assert result['versions']==d.REQUIRED
    assert result['native_class'].endswith('.BogoliubovTransform')
    assert len(result['source_sha256'])==64
    assert not result['scientific_circuit_constructed'] and not result['transpile_performed']


def test_transitive_import_failure_is_not_compatible(monkeypatch):
    from trotterlib.df_trotter import ops
    def broken():raise ImportError('fixture transitive dependency')
    monkeypatch.setattr(ops,'_import_bogoliubov_transform',broken)
    with pytest.raises(ValueError,match='LOWERING_IMPORT_FAILED'):d.native_dependencies()


@pytest.mark.parametrize('tamper',['none','missing','changed'])
def test_native_profile_binding_before_launch(tmp_path,monkeypatch,tamper):
    from trottertracks.resource_applicability import h6_server_contract_v1 as c
    native={'fixture':'native lowering'}
    profile=dict(environment={'fixture':True},python=sys.executable,venv=sys.prefix,
                 critical_file_hashes=[],native_lowering_dependencies=native)
    if tamper=='missing':del profile['native_lowering_dependencies']
    if tamper=='changed':profile['native_lowering_dependencies']={'fixture':'other lowering'}
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    monkeypatch.setattr(d,'native_dependencies',lambda:native)
    manifest={'environment':profile['environment'],'profile':{'repository_path':'profile.json','sha256':c.file_hash(path)}}
    if tamper=='none':assert c.verify_profile(tmp_path,manifest)==profile
    else:
        with pytest.raises(ValueError,match='SERVER_NATIVE_LOWERING_PROFILE'):c.verify_profile(tmp_path,manifest)
