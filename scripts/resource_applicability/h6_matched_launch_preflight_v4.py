#!/usr/bin/env python3
"""Metadata and output-parent checks only. This module never launches science."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
SCIENCE_COMMIT = '5a903aea2492c2d211c1e15e7737df0d50694c12'
PREDECESSOR_COMMIT = '6a3ce771ba343bd5f53b6991e4be1aef506351c7'
OLD_MANIFEST = 'artifacts/resource_applicability/track_a_h6_matched_preparation_v3/2026-10-11/sealed_manifest_v3.json'
OLD_MANIFEST_SHA = '7796f92d442207a985fe2594b2d88ce74694d69a4e041e4188a05a16e5200f4c'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_matched_supplement_v3/2026-10-11/'
OUTPUT_PATH = NAMESPACE + 'launch_v2'
PREPARATION_PATH = 'artifacts/resource_applicability/track_a_h6_matched_launch_preparation_v4/2026-10-11'
INVOCATION_ID = 'H6_MATCHED_SUPPLEMENT_V3_LAUNCH_V2_20261011'
METADATA_SOURCES = (
    'scripts/resource_applicability/h6_matched_launch_preflight_v4.py',
    'scripts/resource_applicability/prepare_track_a_h6_matched_launch_v4.py',
)
VALIDATION_SOURCE = 'tests/tracks/resource_applicability/test_h6_matched_launch_preflight_v4.py'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    if Path(path).stat().st_size > 16 * 2**20:
        raise ValueError('LAUNCH_METADATA_SIZE_CAP')
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def output_parent(root, repository_path, *, prepare_parent=False, namespace=NAMESPACE):
    """Prepare only missing ancestors; reject any existing leaf and symlink.

    A short exclusive file, created relative to an O_NOFOLLOW directory fd,
    checks actual write permission and is removed before returning. The science
    output leaf is never created here. This is a point-in-time check, not a lock.
    """
    root = Path(root).resolve(strict=True)
    if (not isinstance(repository_path, str) or Path(repository_path).is_absolute()
            or any(p in ('', '.', '..') for p in repository_path.split('/'))
            or not repository_path.startswith(namespace)
            or len(Path(repository_path).parts) <= len(Path(namespace).parts)):
        raise ValueError('OUTPUT_REPOSITORY_PATH')
    output = root / repository_path
    if os.path.lexists(output):
        raise ValueError('OUTPUT_LEAF_ALREADY_USED')
    current = root
    for part in Path(repository_path).parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError('OUTPUT_ANCESTOR_NOT_REAL_DIRECTORY:' + str(current))
    created = []
    current = root
    for part in Path(repository_path).parts[:-1]:
        current = current / part
        if not current.exists():
            if not prepare_parent:
                raise ValueError('OUTPUT_PARENT_MISSING:' + str(current))
            current.mkdir(exist_ok=False)
            created.append(str(current.relative_to(root)))
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ValueError('OUTPUT_ANCESTOR_NOT_REAL_DIRECTORY:' + str(current))
    if not os.access(output.parent, os.W_OK | os.X_OK, effective_ids=True):
        raise PermissionError('OUTPUT_PARENT_NOT_WRITABLE')
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    probe = '.h6-launch-parent-check-' + secrets.token_hex(16)
    probe_created = False
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                     0o600, dir_fd=directory_fd)
        probe_created = True
        try:
            os.write(fd, b'prelaunch-permission-check\n')
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if probe_created:
            os.unlink(probe, dir_fd=directory_fd)
        os.close(directory_fd)
    if os.path.lexists(output):
        raise ValueError('OUTPUT_LEAF_ALREADY_USED')
    st = output.parent.stat()
    return dict(status='OUTPUT_PARENT_READY_LEAF_UNUSED', repository_path=repository_path,
                absolute_path=str(output), parent_path=str(output.parent),
                parent_device=st.st_dev, parent_inode=st.st_ino, parent_mode=oct(stat.S_IMODE(st.st_mode)),
                created_directories=created, actual_write_probe_passed=True, probe_removed=True,
                final_leaf_exists=False, scientific_runner_invocations=0)


def frozen_contract(root):
    sys.path.insert(0, str(Path(root) / 'src'))
    from trottertracks.resource_applicability import h6_matched_contract_v3
    return h6_matched_contract_v3


def original_manifest(root):
    path = Path(root) / OLD_MANIFEST
    if file_hash(path) != OLD_MANIFEST_SHA:
        raise ValueError('PREDECESSOR_MANIFEST_BYTES')
    return read(path)


def new_manifest(root, metadata_commit, metadata_hashes, validation_hashes):
    old = original_manifest(root)
    result = copy.deepcopy(old)
    result['exclusive_output'] = dict(repository_path=OUTPUT_PATH, absolute_path=str(Path(root) / OUTPUT_PATH))
    result['launch_preparation_identity'] = dict(
        schema='h6_matched_launch_preparation_v4', invocation_id=INVOCATION_ID,
        predecessor_publication_commit=PREDECESSOR_COMMIT,
        predecessor_manifest_sha256=OLD_MANIFEST_SHA,
        metadata_source_commit=metadata_commit, metadata_source_hashes=metadata_hashes,
        validation_source_hashes=validation_hashes,
        old_authorization_retired=True, fresh_authorization_required=True,
        output_parent_policy='prepare missing ancestors, probe writes, keep leaf absent, recheck before runner',
        launch_transport='foreground maintained execution session; no detached sandbox owner',
        scientific_runner_unchanged=True, scientific_runner_invocations_in_preparation=0)
    return result


def verify_sealed(root, manifest):
    """Static saved bytes, Git commits, environment and unchanged science only."""
    root = Path(root)
    contract = frozen_contract(root)
    from trottertracks.resource_applicability.h6_matched_reuse_v3 import committed_hashes, verify_evidence
    original = original_manifest(root)
    identity = manifest.get('launch_preparation_identity', {})
    hashes = identity.get('metadata_source_hashes', {})
    validation = identity.get('validation_source_hashes', {})
    if set(hashes) != set(METADATA_SOURCES) or set(validation) != {VALIDATION_SOURCE}:
        raise ValueError('LAUNCH_METADATA_SOURCE_CLOSURE')
    commit = identity.get('metadata_source_commit')
    if not isinstance(commit, str) or len(commit) != 40 or any(x not in '0123456789abcdef' for x in commit):
        raise ValueError('LAUNCH_METADATA_SOURCE_COMMIT')
    expected = new_manifest(root, commit, hashes, validation)
    if manifest != expected or manifest['source_commit'] != SCIENCE_COMMIT:
        raise ValueError('LAUNCH_SCIENCE_OR_IDENTITY_CHANGED')
    contract.validate_manifest(manifest)
    contract.check_inherited_limits()
    if manifest['assigned_resources'] != contract.resources([0, 2, 5, 6]):
        raise ValueError('LAUNCH_RESOURCES')
    contract.verify_sources(root, SCIENCE_COMMIT, manifest['source_hashes'])
    committed_hashes(root, SCIENCE_COMMIT, manifest['validation_source_hashes'])
    committed_hashes(root, commit, {**hashes, **validation})
    verify_evidence(root, manifest['reuse_registry'])
    if (contract.verify_parent(root) != manifest['input_identity']
            or contract.environment() != manifest['environment']
            or manifest['input_identity'] != original['input_identity']):
        raise ValueError('LAUNCH_INPUT_ENVIRONMENT')
    return dict(status='STATIC_IDENTITY_PASS', science_source_commit=SCIENCE_COMMIT,
                metadata_source_commit=commit, science_plan_digest=contract.digest(manifest['plan']),
                reuse_registry_digest=contract.digest(manifest['reuse_registry']),
                science_and_reuse_unchanged=True, scientific_runner_invocations=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--prepare-parent', action='store_true')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--authorization-sha256')
    args = parser.parse_args()
    if file_hash(args.manifest) != args.manifest_sha256:
        raise ValueError('LAUNCH_MANIFEST_SHA')
    if bool(args.authorization) != bool(args.authorization_sha256):
        parser.error('Authorization and its SHA256 must be supplied together.')
    if args.receipt.exists():
        raise ValueError('PREFLIGHT_RECEIPT_ALREADY_EXISTS')
    manifest = read(args.manifest)
    result = verify_sealed(ROOT, manifest)
    if args.authorization:
        if file_hash(args.authorization) != args.authorization_sha256:
            raise ValueError('LAUNCH_AUTHORIZATION_SHA')
        grant = read(args.authorization)
        frozen_contract(ROOT).validate_grant(manifest, grant, ROOT / OUTPUT_PATH)
        result['fresh_grant_validated'] = True
    else:
        result['fresh_grant_validated'] = False
    result['output_parent'] = output_parent(ROOT, OUTPUT_PATH, prepare_parent=args.prepare_parent)
    result.update(H6_status='H6_NOT_AUTHORIZED', contract_status='DRAFT_NOT_AUTHORIZATION',
                  mandatory_stop=True, next_stage_authorized=False, science_launched=False)
    write_exclusive(args.receipt, result)
    print('METADATA_PREFLIGHT_PASS SCIENCE_NOT_LAUNCHED H6_NOT_AUTHORIZED')


if __name__ == '__main__':
    main()
