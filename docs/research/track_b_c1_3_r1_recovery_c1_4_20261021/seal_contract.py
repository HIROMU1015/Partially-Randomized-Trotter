"""Metadata-only pre-execution seal. Does not open scientific NPZ arrays."""
import ast
import difflib
import json
from pathlib import Path
import shutil
from support import HERE, ROOT, PRIOR, R1, BASE, MODEL, digest, environment, git, save, verify_rows

def row(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': digest(path)}

def main():
    assert not (HERE / 'result').exists()
    assert json.loads((HERE / 'phase_a_checks.json').read_text())['status'] == 'PASS'
    assert json.loads((HERE / 'artificial_checks.json').read_text())['status'] == 'PASS'
    for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
    old = json.loads((PRIOR / 'execution_contract.json').read_text())
    contract = dict(old)
    contract.update(schema='C1_3_R1_NEW_FIXED_ONE_SHOT_V1', base_commit=BASE,
        run_id='C1-3-R1-H4-selected-20261021',
        previous_contract_sha256=digest(PRIOR / 'execution_contract.json'),
        previous_failed_run_preserved=True, previous_authorization_reused=False,
        state_hash_policy='dtype.str + canonical JSON shape + C-order bytes; fixed pr2_s0_s1_validation::_array_hash semantics',
        state_hash_policy_source='243eea8e840f3c472753f7935e510391626fd325:src/trotterlib/pr2_s0_s1_validation.py:145',
        shared_connection={'adapter': row(PRIOR / 'adapter.py'), 'reference': row(PRIOR / 'reference.py'),
                           'modification': 'none; import original immutable connection modules'},
        failure_schema='C1_3_R1_FAILURE_V1; first failure immutable; logger errors kept separately',
        terminal_schema='C1_3_R1_TERMINAL_V1',
        classifications=['C1_3_CORRECTNESS_PASS', 'C1_3_PRECISION_INSUFFICIENT_STOP',
            'C1_3_OPERATOR_MISMATCH_STOP', 'C1_3_INPUT_IDENTITY_STOP',
            'C1_3_TECHNICAL_INCONCLUSIVE_STOP', 'C1_3_GUARANTEE_UNRESOLVED_STOP'],
        classification_policy='journal.classify_failure: precision category; explicit operator/adjoint gates; explicit guarantee gates; restoration/schedule identity; remaining technical failure',
        output='result/ original outputs plus state_identity.json; raw stdout/stderr and resource observations retained',
        runs=1, retry=0)
    added = set(contract) - set(old)
    allowed_metadata = {'schema', 'base_commit', 'output'}
    assert all(contract[k] == old[k] for k in old if k not in allowed_metadata)
    assert contract['events'] == [{'word': u, 'child': c} for u, c in
         (([], 0), ([], 9), ([0, 2], 36), ([36, 0], 9), ([0, 2, 36, 9], 2), ([9, 36, 2, 0], 36))]
    assert contract['model_sha256'] == MODEL and contract['runs'] == 1 and contract['retry'] == 0
    save(HERE / 'execution_contract.json', contract)
    save(HERE / 'contract_comparison.json', {'scientific_fields_equal_to_old_contract': True,
         'all_old_fields_checked_except_metadata': sorted(set(old) - allowed_metadata),
         'changed_existing_metadata_fields': sorted(allowed_metadata), 'added_technical_metadata': sorted(added),
         'event_precision_state_hash_tolerances_and_runtime_policy_changed': False,
         'old_contract_sha256': digest(PRIOR / 'execution_contract.json'),
         'new_contract_sha256': digest(HERE / 'execution_contract.json')})
    old_manifest = json.loads((PRIOR / 'source_manifest.json').read_text())
    verify_rows(old_manifest['files'])
    for name in ('input_manifest.json', 'runtime_manifest.json'):
        data = json.loads((PRIOR / name).read_text()); verify_rows(data['files'])
        shutil.copyfile(PRIOR / name, HERE / name)
    save(HERE / 'environment_identity.json', environment())
    assert environment() == json.loads((PRIOR / 'environment_identity.json').read_text())
    policy_path = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-b-method-note-v1-1-20261020/docs/research/track_b_g6_w2c_b3_limited_pilot_independent_acceptance_20261019/fixed_reference/tree/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/research_project_instructions_unified_2026-10-17.md')
    save(HERE / 'policy_receipt.json', {'unified_instructions_filename': policy_path.name,
         'SHA256': digest(policy_path), 'actually_read': True, 'sections': 'entire 0-12, especially 8/8.4',
         'projects_AGENTS_SHA256': digest(ROOT.parent.parent / 'AGENTS.md'),
         'primary_agent_only': True, 'private_review_originals_not_copied': True,
         'authorization': 'current explicit bundled C1-3-R1 to C1-4 request; no C1-5'})
    prior_baseline = json.loads((PRIOR / 'preservation_baseline.json').read_text())
    for item in prior_baseline['private_originals']:
        assert digest(item['path']) == item['sha256']
    save(HERE / 'preservation_baseline.json', {'base_commit': BASE,
         'rule': 'only additions under this new namespace; all prior tracked blobs and old STOP/grants retained',
         'private_originals': prior_baseline['private_originals'],
         'unrelated_worktrees': 'untouched; unrelated HEAD/status are not run guards'})
    differences = []
    for oldname, newname in (('run_c13.py', 'run_c13_r1.py'), ('support.py', 'support.py')):
        differences.extend(difflib.unified_diff((PRIOR / oldname).read_text().splitlines(True),
             (HERE / newname).read_text().splitlines(True), fromfile='sealed_prior/' + oldname, tofile='recovery/' + newname, n=0))
    (HERE / 'technical_changes.patch').write_text(''.join(differences))
    science = [r for r in old_manifest['files'] if r['path'].startswith('src/')]
    prior = [row(path) for path in sorted(PRIOR.rglob('*')) if path.is_file()]
    current = [row(path) for path in sorted(HERE.rglob('*')) if path.is_file() and path.name != 'source_manifest.json']
    save(HERE / 'source_manifest.json', {'base_commit': BASE,
         'fixed_science_commit': old_manifest['fixed_science_commit'], 'files': science + prior + current,
         'new_adapter_or_sampler_algorithm': False, 'all_old_source_and_failure_preserved': True,
         'self_hash_bound_by_new_authorization': True})
    print(json.dumps({'sealed': True, 'science_sources': len(science), 'prior_files': len(prior),
                      'new_files': len(current), 'science_condition_changes': 0, 'H4_scientific_attempts': 0}))

if __name__ == '__main__': main()
