#!/usr/bin/env python3
"""Check selected archived publication bytes at fixed commits, without execution.

The original publication receipts define the file hashes. Historical verifiers,
branch HEAD observations, execution contracts and scientific results stay intact.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen


RAW = 'https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'
MAX_BYTES = 5 * 1024 * 1024
TEXT_SUFFIXES = {'.md', '.py', '.json', '.txt', '.csv', '.tsv', '.yaml', '.yml', '.toml'}
CASES = {
    'e-r3-preparation': {
        'publication_commit': '0cf991d4371250dc6a2d8b346ed4ed7414f4a198',
        'content_commit': '8377c1d12439cabab3b97bcc1f57ac030a479785',
        'receipt_path': 'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/publication_receipt_v1.json',
        'receipt_blob': 'c88ad32571179ec88ab262d0197734abe9ce5eea',
        'content_field': 'fixed_source_commit',
        'count_field': 'retrieved_file_count',
    },
    'b3-api-preflight': {
        'publication_commit': '82896d53fa97235fa848d6c0833eb193af8b99d6',
        'content_commit': 'abfeab376010a8790bb0498e80fe1eade4a49051',
        'receipt_path': 'docs/research/track_b_g6_w2c_b3_real_api_binding_preflight_candidate_20261019/publication_receipt.json',
        'receipt_blob': '5b6975d1bd1ea84963a552f73b92193c693c1871',
        'content_field': 'content_commit',
        'count_field': 'file_count',
    },
}


def fixed_url(commit, path):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('A full fixed commit OID is required')
    parts = PurePosixPath(path)
    if not path or parts.is_absolute() or '..' in parts.parts or str(parts) != path:
        raise ValueError('Invalid repository path: ' + path)
    return RAW + commit + '/' + quote(path, safe='/')


def blob_oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def read_file(commit, path, local_git=None):
    url = fixed_url(commit, path)
    if local_git is None:
        with urlopen(url, timeout=30) as response:
            data = response.read(MAX_BYTES + 1)
    else:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
        cmd = ['git', '--no-optional-locks', '-C', str(local_git)]
        obj = commit + ':' + path
        size = int(subprocess.check_output(cmd + ['cat-file', '-s', obj], env=env))
        if size > MAX_BYTES:
            raise ValueError('File exceeds metadata/source scope: ' + path)
        data = subprocess.check_output(cmd + ['show', obj], env=env)
    if len(data) > MAX_BYTES:
        raise ValueError('File exceeds metadata/source scope: ' + path)
    return data


def select_rows(receipt, case):
    if receipt.get(case['content_field']) != case['content_commit']:
        raise ValueError('Receipt content commit differs from the fixed snapshot')
    rows = receipt['files']
    if len(rows) != receipt[case['count_field']]:
        raise ValueError('Receipt file count mismatch')
    selected, skipped, seen = [], [], set()
    for row in rows:
        path = row['path']
        url = fixed_url(case['content_commit'], path)
        if row['url'] != url:
            raise ValueError('Receipt URL is not the fixed content snapshot: ' + path)
        if path in seen:
            raise ValueError('Duplicate receipt path: ' + path)
        seen.add(path)
        if type(row['bytes']) is not int or row['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('Invalid byte identity: ' + path)
        parts = PurePosixPath(path)
        if row['bytes'] > MAX_BYTES or parts.suffix not in TEXT_SUFFIXES or {'raw', 'checkpoint', 'checkpoints', 'cache'} & set(parts.parts):
            skipped.append(path)
        else:
            selected.append(row)
    if not selected:
        raise ValueError('No report, SOURCE or metadata files selected')
    return selected, skipped


def verify(case_name, local_git=None):
    case = CASES[case_name]
    receipt_bytes = read_file(case['publication_commit'], case['receipt_path'], local_git)
    if blob_oid(receipt_bytes) != case['receipt_blob']:
        raise ValueError('Original publication receipt blob mismatch')
    selected, skipped = select_rows(json.loads(receipt_bytes), case)

    def check(row):
        data = read_file(case['content_commit'], row['path'], local_git)
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Fixed publication bytes differ: ' + row['path'])
        return len(data)

    with ThreadPoolExecutor() as pool:
        checked_bytes = sum(pool.map(check, selected))
    return {
        'status': 'PASS_SELECTED_FIXED_PUBLICATION_BYTES',
        'case': case_name,
        'read_method': 'local_fixed_git_objects' if local_git is not None else 'fixed_public_raw_urls',
        'publication_commit': case['publication_commit'],
        'content_commit': case['content_commit'],
        'receipt_blob': case['receipt_blob'],
        'original_receipt_files': len(selected) + len(skipped),
        'checked_listed_files': len(selected),
        'checked_listed_bytes': checked_bytes,
        'receipt_itself_checked_separately': True,
        'skipped_files': skipped,
        'historical_branch_HEAD_recheck': 'NOT_REQUIRED_FOR_FIXED_COMMIT_BYTES',
        'scientific_validation': 'NOT_PERFORMED',
        'scientific_execution_calls': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--local-git', type=Path, help='Read existing fixed Git objects instead of the network; no fetch')
    args = parser.parse_args()
    try:
        result = verify(args.case, args.local_git)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, 'Fixed publication byte check failed: ' + str(exc) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
