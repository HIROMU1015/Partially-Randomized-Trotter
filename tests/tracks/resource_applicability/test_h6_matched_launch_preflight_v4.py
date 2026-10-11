"""Filesystem and synthetic child-process tests; no molecular/science runner."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/resource_applicability/h6_matched_launch_preflight_v4.py'
spec = importlib.util.spec_from_file_location('launch_preflight_v4_test', SCRIPT)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


@pytest.mark.parametrize('exists', [False, True])
def test_missing_or_existing_parent_ready_leaf_remains_unused(tmp_path, exists):
    parent = tmp_path / 'synthetic' / 'namespace'
    if exists:
        parent.mkdir(parents=True)
    receipt = launch.output_parent(tmp_path, 'synthetic/namespace/leaf',
                                   namespace='synthetic/', prepare_parent=True)
    assert parent.is_dir() and not (parent / 'leaf').exists()
    assert receipt['created_directories'] == ([] if exists else ['synthetic', 'synthetic/namespace'])
    assert receipt['actual_write_probe_passed'] and receipt['probe_removed']
    assert not list(tmp_path.rglob('.h6-launch-parent-check-*'))


def test_check_only_missing_parent_rejects_without_creating_it(tmp_path):
    with pytest.raises(ValueError, match='OUTPUT_PARENT_MISSING'):
        launch.output_parent(tmp_path, 'synthetic/n/leaf', namespace='synthetic/')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('kind', ['directory', 'file', 'dangling_symlink'])
def test_used_leaf_rejected_and_existing_bytes_preserved(tmp_path, kind):
    leaf = tmp_path / 'synthetic' / 'leaf'
    leaf.parent.mkdir()
    if kind == 'directory':
        leaf.mkdir()
        (leaf / 'saved').write_bytes(b'IMMUTABLE')
    elif kind == 'file':
        leaf.write_bytes(b'IMMUTABLE')
    else:
        leaf.symlink_to(tmp_path / 'absent')
    with pytest.raises(ValueError, match='OUTPUT_LEAF_ALREADY_USED'):
        launch.output_parent(tmp_path, 'synthetic/leaf', namespace='synthetic/', prepare_parent=True)
    if kind == 'directory':
        assert (leaf / 'saved').read_bytes() == b'IMMUTABLE'
    elif kind == 'file':
        assert leaf.read_bytes() == b'IMMUTABLE'
    else:
        assert leaf.is_symlink()


@pytest.mark.parametrize('path', ['/outside/leaf', 'synthetic/../leaf', 'synthetic/./leaf',
                                  'synthetic//leaf', 'wrong/leaf', 'synthetic'])
def test_invalid_repository_paths_have_no_side_effects(tmp_path, path):
    with pytest.raises(ValueError, match='OUTPUT_REPOSITORY_PATH'):
        launch.output_parent(tmp_path, path, namespace='synthetic/', prepare_parent=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('kind', ['symlink', 'file'])
def test_non_directory_or_symlink_ancestor_rejected(tmp_path, kind):
    node = tmp_path / 'synthetic'
    target = tmp_path / 'protected'
    target.mkdir()
    if kind == 'symlink':
        node.symlink_to(target, target_is_directory=True)
    else:
        node.write_bytes(b'KEEP')
    with pytest.raises(ValueError, match='OUTPUT_ANCESTOR_NOT_REAL_DIRECTORY'):
        launch.output_parent(tmp_path, 'synthetic/new/leaf', namespace='synthetic/', prepare_parent=True)
    assert not list(target.iterdir())


def test_actual_write_permission_checked(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('Permission-denial case requires an unprivileged test process.')
    parent = tmp_path / 'synthetic'
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with pytest.raises(PermissionError):
            launch.output_parent(tmp_path, 'synthetic/leaf', namespace='synthetic/')
        assert not (parent / 'leaf').exists()
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize('state', ['missing_parent', 'existing_parent', 'used_leaf'])
def test_real_child_matches_frozen_leaf_mkdir_contract(tmp_path, state):
    parent = tmp_path / 'synthetic' / 'namespace'
    leaf = parent / 'leaf'
    if state != 'missing_parent':
        parent.mkdir(parents=True)
    if state == 'used_leaf':
        leaf.mkdir()
        (leaf / 'saved').write_bytes(b'KEEP')
    # Child imports ONLY the new metadata helper, then a synthetic stand-in
    # performs the exact pathlib mkdir call. The fixed H6 runner is not invoked.
    code = '''import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from h6_matched_launch_preflight_v4 import output_parent
root = Path(sys.argv[2]); leaf = root / 'synthetic/namespace/leaf'
try:
    output_parent(root, 'synthetic/namespace/leaf', namespace='synthetic/', prepare_parent=True)
except ValueError as error:
    assert sys.argv[3] == 'used_leaf' and 'OUTPUT_LEAF_ALREADY_USED' in str(error)
    assert (leaf / 'saved').read_bytes() == b'KEEP'
    print('REJECTED_BEFORE_SYNTHETIC_CHILD'); sys.exit(0)
assert not leaf.exists()
leaf.mkdir(exist_ok=False)
(leaf / 'synthetic_only').write_text('no science')
assert not any(p in sys.modules for p in ('numpy','scipy','qiskit','numba','openfermion'))
print('SYNTHETIC_CHILD_MKDIR_PASS')
'''
    result = subprocess.run([sys.executable, '-c', code, str(SCRIPT.parent), str(tmp_path), state],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert ('REJECTED_BEFORE_SYNTHETIC_CHILD' if state == 'used_leaf'
            else 'SYNTHETIC_CHILD_MKDIR_PASS') in result.stdout


def test_manifest_identity_only_changes_output_and_launch_metadata(monkeypatch, tmp_path):
    old = dict(source_commit=launch.SCIENCE_COMMIT, plan={'science': 'unchanged'},
               reuse_registry={'bytes': 'unchanged'}, exclusive_output={'old': 'unused'},
               source_hashes={'science.py': 'same'}, assigned_resources={'workers': 2})
    monkeypatch.setattr(launch, 'original_manifest', lambda root: old)
    new = launch.new_manifest(tmp_path, 'a' * 40, {'helper': 'b'}, {'test': 'c'})
    restored = dict(new)
    restored.pop('launch_preparation_identity')
    restored['exclusive_output'] = old['exclusive_output']
    assert restored == old
    assert old['exclusive_output'] == {'old': 'unused'}
    assert new['exclusive_output']['repository_path'] == launch.OUTPUT_PATH


def test_old_grant_not_valid_for_new_manifest(tmp_path):
    from trottertracks.resource_applicability.h6_matched_contract_v3 import validate_grant
    from trottertracks.resource_applicability.ax2a_preparation import digest
    manifest = dict(source_commit=launch.SCIENCE_COMMIT, reuse_registry={'same': 'bytes'},
                    assigned_resources={'assigned_cpus': [0, 2, 5, 6]}, output='new')
    grant = dict(schema='h6_matched_authorization_v3', kind='H6_MATCHED_RESOURCE_SUPPLEMENT_V3',
                 approved_by_user=True, authorization_only=True, manifest_digest=digest({**manifest, 'output': 'old'}),
                 source_commit=launch.SCIENCE_COMMIT, reuse_registry_digest=digest(manifest['reuse_registry']),
                 assigned_cpus=[0, 2, 5, 6], one_shot=True, retry=False, resume=False,
                 exclusive_output=str(tmp_path / 'leaf'), H8_authorized=False)
    with pytest.raises(ValueError, match='FRESH_SUPPLEMENT'):
        validate_grant(manifest, grant, tmp_path / 'leaf')
