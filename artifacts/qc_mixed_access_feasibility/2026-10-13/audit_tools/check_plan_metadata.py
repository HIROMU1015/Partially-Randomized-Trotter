"""Check plan schema, provenance and STOP;no molecular/scientific imports."""
from pathlib import Path
from fractions import Fraction
import difflib
import hashlib
import json
import math
import re
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
base = 'b575ff1a4e659857367a266cb59acb958ba446fd'
prefix = Path('artifacts/qc_mixed_access_feasibility/2026-10-13')
out = root / prefix

def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)

contract = json.loads((out / 'feasibility_contract.json').read_text())
limits = contract['limits']
assert contract['methods'] == ['P2', 'E2', 'E4', 'D2', 'D4', 'R2', 'R4']
times = contract['grid']['times_atomic']
steps = contract['grid']['steps']
assert len(contract['methods']) * len(times) * len(steps) == limits['PF_error_queries'] == 154
assert len({Fraction(t, r) for t in times for r in steps}) * 7 == limits['distinct_ideal_seed_matrices'] == 91
assert 154 * len(contract['grid']['PF_budget_fractions']) * len(contract['resource_model']['rotation_sensitivity_kappa']) == limits['resource_model_rows'] == 1386
assert math.comb(8, 4) == limits['nu_input_columns'] == 70
assert 2**8 == limits['full_Fock_dimension'] == 256
assert limits['compile_or_T_synthesis'] == limits['FCI_ground_gap_eigh'] == 0
assert limits['attempts'] == 1 and limits['retries'] == 0
assert limits['wall_seconds'] is None and limits['CPU_seconds'] is None
assert contract['science_runner'] is None and contract['source_execution_SHA'] is None
assert contract['target']['pack_SHA256'] is None
assert contract['next_execution_authorization'] is None and contract['mandatory_stop']
status = json.loads((out / 'preparation_status.json').read_text())
assert all(v == 0 for v in status['counts'].values()) and not status['execution_ready']
ready = json.loads((out / 'launch_readiness.json').read_text())
assert ready['scope_ready_for_user_approval'] and not ready['launch_allowed']

old = json.loads(git('show', base + ':artifacts/validation_manifest.json'))
new = json.loads((root / 'artifacts/validation_manifest.json').read_text())
assert old['result_sets'] == new['result_sets'][:len(old['result_sets'])]
assert {k:v for k,v in old.items() if k != 'result_sets'} == {k:v for k,v in new.items() if k != 'result_sets'}
assert 'description' not in new['relevant_commits'][7]
for entry in new['result_sets'][len(old['result_sets']):]:
    assert set(entry) == {'id', 'status', 'summary', 'counts', 'completion_criteria'}
    assert entry['status'] == 'source_present_no_current_ci'
    assert all(type(v) is int and v >= 0 for v in entry['counts'].values())

bindings = json.loads((out / 'base_source_and_evidence_index.json').read_text())['files']
for item in bindings:
    data = (root / item['path']).read_bytes()
    assert data == git('show', base + ':' + item['path'])
    assert hashlib.sha256(data).hexdigest() == item['sha256']

allowed = {
    'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'artifacts/validation_manifest.json',
    'docs/README.md', 'docs/research/README.md', 'scripts/README.md', 'src/trotterlib/README.md',
    'docs/research/研究概要・現状.md', 'docs/research/研究目的・研究課題.md',
    'docs/research/研究方法・解析手順.md', 'docs/research/数値実験・評価計画.md',
    'docs/research/研究ノート/README.md', 'docs/research/研究ノート/2026-10-13_qc_relative_block_prevalidation.md',
    'docs/research/研究ノート/2026-10-13_qc_mixed_access_feasibility.md',
    'docs/research/qc_mixed_access_feasibility_plan_v1_20261013.md',
    'docs/research/qc_mixed_access_feasibility_approval_scope_v1_20261013.md',
}
paths = set(p.decode() for p in git('diff', '--name-only', '-z', base).split(b'\0') if p)
paths.update(p.decode() for p in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if p)
links = []
for rel in sorted(paths):
    assert rel in allowed or rel.startswith(str(prefix) + '/'), rel
    path = root / rel
    if path.suffix == '.json':
        json.loads(path.read_text())
    if path.suffix != '.md':
        continue
    current = path.read_text().splitlines()
    try:
        previous = git('show', base + ':' + rel).decode().splitlines()
    except subprocess.CalledProcessError:
        previous = []
    additions = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=previous, b=current).get_opcodes():
        if tag in ('insert', 'replace'):
            additions.extend(current[j1:j2])
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', '\n'.join(additions)):
        if target.startswith(('https:', 'http:', '#', 'mailto:')):
            continue
        q = (path.parent / target.split('#')[0]).resolve()
        assert q.exists(), (rel, target)
        links.append({'source': rel, 'target': q.relative_to(root).as_posix()})

run = subprocess.run([sys.executable, 'scripts/check_validation_manifest.py'], cwd=root,
                     capture_output=True, text=True)
assert run.returncode == 1 and "manifest.relevant_commits[7] has invalid keys: missing ['description']" in run.stderr
(out / 'global_manifest_failure.json').write_text(json.dumps({
    'exit_code': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr,
    'existing_failure_preserved': True}, indent=2) + '\n')
record = {
    'status': 'PLAN_METADATA_AND_PROTECTED_IDENTITY_PASS', 'base_commit': base,
    'old_manifest_entries_preserved': len(old['result_sets']),
    'all_non_result_set_fields_preserved': True, 'priority_blobs_unchanged': len(bindings),
    'allowed_documentation_metadata_changes_only': True, 'registered_slots_accounting_checked': True,
    'added_relative_links_checked': len(links), 'links': links,
    'source_or_input_execution_readiness': False, 'launch_authorization_present': False,
    'scientific_programs_or_tests_executed': 0, 'global_manifest_check': 'KNOWN_DESCRIPTION_FAILURE',
    'mandatory_stop': True, 'qualification': 'metadata/bytes/design arithmetic only;not molecular results or CI',
}
(out / 'static_preparation_check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k != 'links'}, ensure_ascii=False))
