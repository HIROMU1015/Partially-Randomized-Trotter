"""Read-only provenance and document audit; no scientific calculations."""
from pathlib import Path
import hashlib
import json
import os
import re
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = '4cf2827c14fec96613a60bb1427287b30766d5da'
OUT = ROOT / 'artifacts/track_b_d3_preparation/2026-10-13/v1'


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    start, cpu = time.monotonic(), time.process_time()
    expected = {}
    for p in ('artifacts/track_b_rte_development_source/2026-10-13/v1/source_manifest_v1.json',
              'artifacts/track_b_d2_boundary_source/2026-10-13/v1/source_manifest_v1.json'):
        expected.update(json.loads((ROOT/p).read_text())['critical_sha256'])
    expected.update(json.loads((ROOT/'artifacts/track_b_d2_boundary_source/2026-10-13/v1/protected_hashes_v1.json').read_text())['sha256'])
    for p, h in expected.items():
        assert not p.lower().endswith('.npz')
        assert digest(ROOT/p) == h, p
    changed = [p.decode() for p in git('diff', '--name-only', '-z', BASE).split(b'\0') if p]
    allowed = {
        'PROJECT_MAP.md', 'docs/README.md', 'docs/research/研究概要・現状.md',
        'docs/tracks/algorithm_codesign/README.md', 'scripts/README.md',
        'tests/README.md', 'src/trotterlib/README.md', 'artifacts/validation_manifest.json',
        'docs/research/track_b_d1_d2_integrated_scientific_review_20261013.md',
        'artifacts/track_b_d3_preparation/2026-10-13/v1/review_intake_v1.json'}
    assert set(changed) <= allowed
    old_manifest = json.loads(git('show', BASE + ':artifacts/validation_manifest.json'))
    assert json.loads((ROOT/'artifacts/validation_manifest.json').read_text())['result_sets'] == old_manifest['result_sets']
    results = git('ls-tree', '-r', '--name-only', BASE,
                  'artifacts/track_b_rte_development_d1_result/2026-10-13/v1',
                  'artifacts/track_b_d2_saved_boundary_result/2026-10-13/v1').decode().splitlines()
    for p in results:
        assert hashlib.sha256(git('show', BASE + ':' + p)).hexdigest() == digest(ROOT/p)
    parent = Path('/home/abe/Project/Partially Randomized Trotter')
    before = json.loads(Path('/tmp/track-b-d3-parent-before.json').read_text())
    def parent_git(*args):
        return subprocess.check_output(['git', '-C', str(parent), *args])
    head = parent_git('rev-parse', 'HEAD').decode().strip()
    assert head == before['HEAD']
    index = parent/parent_git('rev-parse', '--git-path', 'index').decode().strip()
    assert digest(index) == before['index_sha256']
    raw = parent_git('status', '--porcelain=v1', '-z')
    assert hashlib.sha256(raw).hexdigest() == before['status_bytes_sha256']
    matches = 0
    for row in before['entries']:
        if 'sha256' in row:
            assert digest(parent/row['path']) == row['sha256']
            matches += 1
    review = Path('docs/research/track_b_d1_d2_integrated_scientific_review_20261013.md')
    assert (ROOT/review).read_bytes() == (parent/review.name).read_bytes()
    links = 0
    for p in [ROOT/'docs/tracks/algorithm_codesign/d3_mathematical_confirmation_20261013.md',
              ROOT/'docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md',
              ROOT/'docs/research/研究ノート/2026-10-13_track_b_d3_preparation.md']:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' not in target:
                assert (p.parent/target).exists(), (str(p), target)
                links += 1
    for p in OUT.glob('*.json'):
        json.loads(p.read_text())
    c = json.loads((OUT/'comparison_design_contract_v1.json').read_text())
    assert c['science_execution_authorized'] is False and c['source_commit'] is None
    assert all(not (ROOT/c['planned_launch'][key]).exists() for key in ('authorization_path', 'result_namespace'))
    assert digest(c['acquisition']['runtime_executable']) == c['acquisition']['runtime_executable_sha256']
    return dict(
        status='D3_PREPARATION_SCOPE_AND_PROTECTION_PASS', base_commit=BASE,
        old_D1_D2_source_and_protected_hashes_checked=len(expected),
        old_result_files_byte_identity_checked=len(results),
        old_scientific_semantic_tests_repeated=0, old_scientific_classification_changed=False,
        parent_HEAD=head, parent_branch=before['branch'], parent_index_status_unchanged=True,
        parent_dirty_file_hashes_unchanged=matches, root_review_byte_identical=True,
        new_document_relative_links_checked=links, new_contract_JSON_valid=True,
        old_manifest_result_sets_unchanged=True, changed_tracked_paths=changed,
        registered_resource_revaluation=0, registered_new_budget_or_rank=0,
        registered_pairs=0, new_synthesis=0, new_science_input=0,
        new_circuit_or_IR_generation=0, matrix_evaluation=0, sampling=0,
        trajectory=0, LP=0, DF_or_molecule_NPZ_access=0, GPU_query_or_use=0,
        science_runner=0, authorization_created=False, marker_created=False,
        future_science_authorized=False, mandatory_STOP=True,
        runtime_executable_hash_matches=True, host_logical_CPUs=os.cpu_count(),
        protection_audit_wall_seconds=time.monotonic()-start,
        protection_audit_CPU_seconds=time.process_time()-cpu,
        protection_audit_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        earlier_audit_attempts=[
            'quoted Unicode paths/intake paths not included in first path-set assertion',
            'second link check preceded creation of referenced manifest checkpoint'],
        earlier_attempts_old_source_or_result_identity_failure=False)


if __name__ == '__main__':
    value = main()
    path = OUT/'protection_and_scope_receipt_v1.json'
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: value[k] for k in (
        'status', 'old_D1_D2_source_and_protected_hashes_checked',
        'old_result_files_byte_identity_checked', 'parent_dirty_file_hashes_unchanged')}))
