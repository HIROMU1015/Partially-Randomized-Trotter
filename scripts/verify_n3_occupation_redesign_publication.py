#!/usr/bin/env python3
"""Read-only remote-object verification of the frozen N3 preparation packet."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

COMMIT = 'c2955ba0ef391eaf9102c711c378ab73b766fc9f'
BASE = '64cf5ed227b7e4a74b37453cbe733db5f5d51f56'


def verify(bare, export):
    def git(*args):
        return subprocess.check_output(['git', '--git-dir=' + str(bare), *args])

    if (bare / 'objects/info/alternates').exists():
        raise ValueError('independent remote audit cannot use alternates')
    inventory_path = 'artifacts/n3_occupation_redesign_preparation/2026-10-12/file_inventory.json'
    inventory = json.loads(git('show', COMMIT + ':' + inventory_path))
    expected = {r['path']: r for r in inventory['files']}
    paths = list(expected) + [inventory_path,
        'artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json',
        'artifacts/n3_direction_generator_v1/2026-10-11/run1/certificate_ledger.json']
    rows = []
    for path in paths:
        data = git('show', COMMIT + ':' + path)
        row = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        if path in expected and row != expected[path]:
            raise ValueError(('fixed inventory mismatch', path))
        destination = export / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append(row)

    def tree(revision):
        return {row.split(b'\t', 1)[1].decode(): row.split(b'\t', 1)[0].decode()
                for row in git('ls-tree', '-r', '-z', revision).split(b'\0') if row}
    base, current = tree(BASE), tree(COMMIT)
    changed = set(expected) | {inventory_path}
    for path in base:
        if path not in changed and base[path] != current.get(path):
            raise ValueError(('protected tree entry changed', path))
    module = export / 'src/trottertracks/representation_exploration/n3_occupation_certificate_audit.py'
    spec = importlib.util.spec_from_file_location('occupation_saved_remote_audit', module)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    audit = json.loads(json.dumps(checker.check_saved(export)))
    stored = json.loads((export / 'artifacts/n3_occupation_redesign_preparation/2026-10-12/independent_witness_audit.json').read_text())
    if any(stored[k] != v for k, v in audit.items()):
        raise ValueError('remote exact witnesses differ from saved audit')
    return {'status': 'REMOTE_PREPARATION_PACKET_PASS', 'preparation_commit': COMMIT,
            'retrieved_blobs': len(rows), 'retrieved_bytes': sum(r['bytes'] for r in rows),
            'checks': audit['checks'], 'PSD_matrices': audit['psd_matrices'],
            'protected_base_tree_entries': len(base) - len(set(base) & changed),
            'numerical_eigensolver_calls': 0, 'many_body_solves': 0,
            'new_scientific_batches': 0, 'next_scientific_batch_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bare-repository', type=Path, required=True)
    parser.add_argument('--export-directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bare_repository, args.export_directory), indent=2))
