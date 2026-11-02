"""Identity and documentation checks only;never import or execute science sources."""
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import difflib

root = pathlib.Path(sys.argv[1]).resolve()
base = '25cbc6e79480230afb841201b6917b987cc2a3d7'
prefix = pathlib.Path('artifacts/qc_relative_block_method_cost_audit/2026-10-13')
out = root / prefix

def git(*args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)

old = json.loads(git('show', base + ':artifacts/validation_manifest.json'))
new = json.loads((root / 'artifacts/validation_manifest.json').read_bytes())
assert new['result_sets'][:len(old['result_sets'])] == old['result_sets']
assert {k: v for k, v in old.items() if k != 'result_sets'} == {k: v for k, v in new.items() if k != 'result_sets'}
assert 'description' not in new['relevant_commits'][7]
old_ids = {entry['id'] for entry in old['result_sets']}
for entry in new['result_sets'][len(old['result_sets']):]:
    assert set(entry) == {'id', 'status', 'summary', 'counts', 'completion_criteria'}
    assert entry['id'] not in old_ids and entry['status'] == 'source_present_no_current_ci'
    assert entry['summary'] and entry['completion_criteria']
    assert all(type(v) is int and v >= 0 for v in entry['counts'].values())
bound = json.loads((out / 'base_evidence_identity.json').read_text())
for entry in bound['files']:
    raw = (root / entry['path']).read_bytes()
    assert raw == git('show', base + ':' + entry['path'])
    assert hashlib.sha256(raw).hexdigest() == entry['sha256']

changed = [p.decode() for p in git('diff', '--name-only', '-z', base).split(b'\0') if p]
untracked = [p.decode() for p in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if p]
allowed_modified = {
    'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'artifacts/validation_manifest.json',
    'docs/README.md', 'docs/research/README.md', 'scripts/README.md', 'src/trotterlib/README.md',
    'docs/research/研究概要・現状.md', 'docs/research/研究目的・研究課題.md',
    'docs/research/先行研究と未解決点.md', 'docs/research/研究ノート/README.md',
    'docs/research/研究ノート/2026-10-13_qc_relative_block_prevalidation.md',
}
new_docs = {
    'docs/research/qc_relative_block_method_cost_audit_v1_20261013.md',
    'docs/research/qc_relative_block_method_cost_handoff_v1_20261013.md',
}
for rel in set(changed + untracked):
    assert rel in allowed_modified | new_docs or rel.startswith(str(prefix) + '/'), rel

links = []
for rel in sorted(set(changed + untracked)):
    p = root / rel
    if p.suffix == '.json':
        json.loads(p.read_bytes())
    if p.suffix != '.md':
        continue
    current = p.read_text().splitlines()
    try:
        previous = git('show', base + ':' + rel).decode().splitlines()
    except subprocess.CalledProcessError:
        previous = []
    additions = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=previous, b=current).get_opcodes():
        if tag in ('insert', 'replace'):
            additions.extend(current[j1:j2])
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', '\n'.join(additions)):
        if target.startswith(('http://', 'https://', '#', 'mailto:', 'app:')):
            continue
        q = (p.parent / target.split('#')[0]).resolve()
        assert q.exists(), (rel, target)
        links.append({'source': rel, 'target': q.relative_to(root).as_posix()})

status = json.loads((out / 'audit_status.json').read_text())
assert status['mandatory_stop'] and not status['next_stage_authorized']
assert not status['phase2_detailed_backend_activated'] and not status['new_GPT_review_started']
assert all(v == 0 for v in status['counts'].values())
run = subprocess.run([sys.executable, 'scripts/check_validation_manifest.py'], cwd=root,
                     capture_output=True, text=True)
expected = "Validation manifest FAILED: manifest.relevant_commits[7] has invalid keys: missing ['description']"
assert run.returncode == 1 and expected in run.stderr, (run.returncode, run.stdout, run.stderr)
(out / 'global_manifest_existing_failure.log').write_text(run.stdout + run.stderr)
record = {
    'status': 'STATIC_METADATA_AND_PRESERVATION_PASS', 'base_commit': base,
    'old_manifest_entries_preserved': len(old['result_sets']),
    'non_result_set_manifest_fields_preserved': True,
    'new_manifest_entry_schema_checked': True,
    'priority_blobs_unchanged': len(bound['files']),
    'all_other_tracked_source_artifacts_unchanged': True,
    'allowed_changes_only': True, 'added_relative_links_checked': len(links), 'links': links,
    'scientific_programs_or_tests_executed': 0,
    'global_manifest_check': {'exit_code': run.returncode, 'status': 'KNOWN_DESCRIPTION_FAILURE_PRESERVED'},
    'qualification': 'file hashes/JSON/links/static scope only;not synthetic/scientific/CI verification',
}
(out / 'static_integrity_check.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in record.items() if k != 'links'}, ensure_ascii=False))
