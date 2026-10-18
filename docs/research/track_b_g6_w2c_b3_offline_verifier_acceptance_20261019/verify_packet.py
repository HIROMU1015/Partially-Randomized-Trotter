"""Read-only local packet/source preservation check; no scientific data parsing.

Hashes pre-existing blobs as bytes only. Does not inspect another worktree.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent
BASE = '564409487a35e15d73f1d0f0c2611824b400ab02'
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], env=ENV)


def main():
    inventory = json.loads((PACKET/'source_inventory.json').read_text())
    checked = []
    for row in inventory['new_or_modified_sources_and_tests']:
        assert sha256((ROOT/row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    for row in inventory['references']:
        b = git('show', row['commit']+':'+row['path'])
        assert len(b) == row['bytes'] and sha256(b).hexdigest() == row['sha256'], row['path']
    # ls-tree -z avoids Git's quoted Unicode filenames; no other worktree access.
    tree = git('ls-tree', '-rz', BASE).split(b'\0')
    for raw in tree:
        if not raw:
            continue
        meta, name = raw.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        assert kind == 'blob', 'Unexpected submodule'
        p = ROOT/os.fsdecode(name)
        actual = os.readlink(p).encode() if mode == '120000' else p.read_bytes()
        # Git hash-object --stdin is read-only (no -w).
        actual_oid = subprocess.check_output(['git', 'hash-object', '--stdin'], input=actual, env=ENV).decode().strip()
        assert actual_oid == oid, str(p)
        checked.append(name.decode())
    results = json.loads((PACKET/'artificial_test_results.json').read_text())
    assert sum(r['pass_count'] for r in results) == 360
    assert all(r['successful'] and not r['fail'] and not r['error'] and not r['skip'] and
               not r['forbidden_science_modules_imported'] for r in results)
    import gzip
    for row in results:
        data = gzip.decompress((PACKET/'logs'/(row['log']+'.gz')).read_bytes())
        assert sha256(data).hexdigest() == row['log_sha256']
        assert sha256((ROOT/row['source_path']).read_bytes()).hexdigest() == row['source_sha256']
    from trottertracks.algorithm_codesign.g6_w2c_b3_offline_result import verify_bundle
    from trottertracks.algorithm_codesign.g6_w2c_b3_offline_cost import aggregate
    v = verify_bundle(PACKET/'artificial_inputs/manifest.json')
    assert v['status'] == 'VERIFIED_COMPLETE'
    assert aggregate(v)['status'] == 'COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS'
    assert v == json.loads((PACKET/'artificial_example/verification_report.json').read_text())
    assert aggregate(v) == json.loads((PACKET/'artificial_example/cost_summary.json').read_text())
    print(json.dumps(dict(existing_base_blobs_preserved=len(checked), references=len(inventory['references']),
                          source_test_hashes=len(inventory['new_or_modified_sources_and_tests']),
                          tests_pass=360, report_recomputed=True, scientific_calls=0, mandatory_STOP=True)))


if __name__ == '__main__':
    main()
