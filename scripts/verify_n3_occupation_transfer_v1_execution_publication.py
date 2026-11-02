#!/usr/bin/env python3
"""Read-only GitHub-object/export and saved-witness audit; never runs producers."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = 'artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/publication_receipt.json'
PLAN = 'docs/research/n3_occupation_transfer_v1_execution_plan_20261012.json'
RUN = 'artifacts/n3_occupation_transfer_v1/2026-10-12/run1'


def verify(bare, export, receipt):
    def git(*args):
        return subprocess.check_output(['git', '--git-dir=' + str(bare), *args])

    def blob(path, revision=None):
        return git('show', (revision or receipt['payload_commit']) + ':' + path)

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def require(condition, label):
        if not condition:
            raise ValueError(label)

    require(not (bare / 'objects/info/alternates').exists(), 'local alternates')
    lookup = {}
    for row in receipt['files']:
        data = blob(row['path'])
        require(len(data) == row['bytes'] and digest(data) == row['sha256'], 'remote blob ' + row['path'])
        require((export / row['path']).read_bytes() == data, 'export bytes ' + row['path'])
        lookup[row['path']] = row
    require(len(lookup) == receipt['remote_blobs'], 'duplicate/count mismatch')
    require(sum(r['bytes'] for r in lookup.values()) == receipt['remote_bytes'], 'size sum')
    plan = json.loads(blob(PLAN))
    require(digest(blob(PLAN)) == receipt['plan_sha256'], 'plan binding')
    require(digest(json.dumps(plan['source_sha256'], sort_keys=True, separators=(',', ':')).encode()) == plan['source_fingerprint'], 'source fingerprint')
    for path, sha in plan['source_sha256'].items():
        require(digest(blob(path)) == sha and blob(path) == blob(path, plan['source_commit']), 'frozen source ' + path)
    require(digest(blob(plan['input_path'])) == plan['input_sha256'], 'input binding')
    require(digest(blob(plan['comparison_contract_path'])) == plan['comparison_contract_sha256'], 'contract binding')
    inventory = json.loads(blob('artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/file_inventory.json'))
    for row in inventory['files']:
        require(lookup.get(row['path']) == row, 'payload inventory ' + row['path'])
    preserve = json.loads(blob('artifacts/n3_occupation_transfer_v1/2026-10-12/execution_audit/preservation_and_execution_audit.json'))

    def tree(revision):
        return {row.split(b'\t', 1)[1].decode(): row.split(b'\t', 1)[0]
                for row in git('ls-tree', '-r', '-z', revision).split(b'\0') if row}

    old, new = tree(receipt['base_commit']), tree(receipt['payload_commit'])
    allowed = set(preserve['allowed_existing_modified_paths'])
    protected = 0
    for path in old:
        if path not in allowed:
            require(old[path] == new[path], 'old protected tree ' + path)
            protected += 1
    require(protected == preserve['protected_base_tree_entries'], 'protected tree count')
    old_manifest = json.loads(blob('artifacts/validation_manifest.json', receipt['base_commit']))
    new_manifest = json.loads(blob('artifacts/validation_manifest.json'))
    require(new_manifest['result_sets'][:-1] == old_manifest['result_sets'], 'prior manifest entries')
    require(all(new_manifest[k] == old_manifest[k] for k in old_manifest if k != 'result_sets'), 'prior manifest fields')
    auth_data = blob('docs/research/n3_occupation_transfer_v1_execution_authorization_20261012.json')
    auth = json.loads(auth_data)
    launch = json.loads(blob(RUN + '/launch_receipt.json'))
    require(digest(auth_data) == launch['authorization_sha256'], 'authorization binding')
    require(auth['scientific_batch_authorized'] is True and auth['scientific_attempts'] == 1 and auth['scientific_retries'] == 0, 'one-shot authorization')
    require(auth['source_commit'] == plan['source_commit'] and auth['source_fingerprint'] == plan['source_fingerprint']
            and auth['input_sha256'] == plan['input_sha256'] and auth['plan_sha256'] == receipt['plan_sha256'], 'authorization hashes')
    primary_data = blob(RUN + '/primary_ledger.json')
    seal = digest(primary_data)
    require(json.loads(blob(RUN + '/primary_seal.json'))['sha256'] == seal, 'primary seal')
    checker_path = export / 'src/trottertracks/representation_exploration/n3_occupation_transfer_check_v1.py'
    spec = importlib.util.spec_from_file_location('n3_occ_v1_published_execution_checker', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    audit = checker.check_saved(json.loads(blob(plan['input_path'])), json.loads(primary_data),
                                json.loads(blob(RUN + '/secondary_ledger.json')), plan['source_fingerprint'], seal)
    require(audit == json.loads(blob(RUN + '/independent_saved_audit.json')), 'saved audit')
    require(audit == receipt['remote_saved_witness_audit'], 'receipt arithmetic')
    result = json.loads(blob(RUN + '/result.json'))
    require(result['gain_bases'] == audit['gain_bases'] and result['primary_seal'] == seal, 'result binding')
    require(result['status'] == 'NO_ADDITIONAL_ACCEPTANCE_OBSERVED', 'fixed result')
    stop = json.loads(blob(RUN + '/MANDATORY_STOP.json'))
    require(stop['mandatory_stop'] is True and stop['next_scientific_stage_authorized'] is False and stop['central_theme'] is None, 'mandatory STOP')
    return {'status': 'N3_OCC_V1_EXECUTION_PUBLICATION_SAVED_AUDIT_PASS', 'payload_commit': receipt['payload_commit'],
            'remote_blobs': len(lookup), 'remote_bytes': sum(r['bytes'] for r in lookup.values()),
            'frozen_source_files': len(plan['source_sha256']), 'protected_base_tree_entries': protected,
            'old_manifest_entries_preserved': len(old_manifest['result_sets']), 'saved_witness_audit': audit,
            'new_scientific_batches': 0, 'mandatory_stop': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bare-repository', type=Path, required=True)
    parser.add_argument('--export-directory', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((ROOT / RECEIPT).read_bytes())
    print(json.dumps(verify(args.bare_repository, args.export_directory, receipt), indent=2))


if __name__ == '__main__':
    main()
