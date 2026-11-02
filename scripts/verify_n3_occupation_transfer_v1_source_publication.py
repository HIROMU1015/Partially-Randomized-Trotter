#!/usr/bin/env python3
"""Verify GitHub-retrieved source/plan and synthetic proofs only; no generator."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

PAYLOAD = '6d877855656ee93c896228c6842c9928089c5b67'
SOURCE = '83f51cd8a4848572c66a5535c67d0d0a5f3868cd'
BASE = '505576ba95ede12d38f55deea32f7ccd7c6d9255'


def verify(bare, export):
    def git(*args): return subprocess.check_output(['git', '--git-dir=' + str(bare), *args])
    if (bare / 'objects/info/alternates').exists(): raise ValueError('alternates prohibited')
    inventory_path = 'artifacts/n3_occupation_transfer_v1/2026-10-12/preparation/file_inventory.json'
    plan_path = 'docs/research/n3_occupation_transfer_v1_execution_plan_20261012.json'
    inventory = json.loads(git('show', PAYLOAD + ':' + inventory_path))
    plan = json.loads(git('show', PAYLOAD + ':' + plan_path))
    paths = sorted(set([r['path'] for r in inventory['files']] + [inventory_path, plan['input_path']] + list(plan['source_sha256'])))
    lookup = {}
    for path in paths:
        if Path(path).is_absolute() or '..' in Path(path).parts: raise ValueError('unsafe path')
        data = git('show', PAYLOAD + ':' + path)
        lookup[path] = {'path': path, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        target = export / path; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
    for r in inventory['files']:
        if lookup[r['path']] != r: raise ValueError('inventory mismatch')
    for path, sha in plan['source_sha256'].items():
        if lookup[path]['sha256'] != sha or git('show', SOURCE + ':' + path) != (export / path).read_bytes(): raise ValueError('source byte identity')
    if lookup[plan['input_path']]['sha256'] != plan['input_sha256'] or lookup[plan['comparison_contract_path']]['sha256'] != plan['comparison_contract_sha256']: raise ValueError('input/contract identity')
    def tree(rev):
        return {r.split(b'\t', 1)[1].decode(): r.split(b'\t', 1)[0].decode() for r in git('ls-tree', '-r', '-z', rev).split(b'\0') if r}
    old, new = tree(BASE), tree(PAYLOAD); changed = {r['path'] for r in inventory['files']} | {inventory_path}
    for path in old:
        if path not in changed and old[path] != new.get(path): raise ValueError('protected tree entry changed')
    checker_path = export / 'src/trottertracks/representation_exploration/n3_occupation_transfer_check_v1.py'
    spec = importlib.util.spec_from_file_location('remote_stdlib_occ_check', checker_path)
    checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
    data = export / 'artifacts/n3_occupation_transfer_v1/2026-10-12/preparation'
    primary = (data / 'synthetic_primary_ledger.json').read_bytes()
    result = checker.check_saved(json.loads((data / 'synthetic_pack.json').read_bytes()), json.loads(primary),
        json.loads((data / 'synthetic_secondary_ledger.json').read_bytes()), plan['source_fingerprint'], hashlib.sha256(primary).hexdigest())
    if result != json.loads((data / 'synthetic_saved_audit.json').read_bytes()): raise ValueError('synthetic exact audit mismatch')
    schema = checker.Checker()
    pack = json.loads((export / plan['input_path']).read_bytes())
    for case in pack['cases']: schema.decode(case)
    if len(pack['cases']) != 8: raise ValueError('fixed case count')
    return {'status': 'REMOTE_SOURCE_PLAN_AND_SYNTHETIC_AUDIT_PASS', 'source_commit': SOURCE,
            'payload_commit': PAYLOAD, 'blobs': len(paths), 'bytes': sum(r['bytes'] for r in lookup.values()),
            'synthetic_saved_checks': result['checks'], 'new8_schema_checks': schema.checks,
            'new8_predicate_evaluations': 0, 'numerical_eigensolver_calls': 0, 'ground_solves': 0,
            'new_scientific_batches': 0, 'scientific_batch_authorized': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bare-repository', type=Path, required=True)
    p.add_argument('--export-directory', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(verify(args.bare_repository, args.export_directory), indent=2))
