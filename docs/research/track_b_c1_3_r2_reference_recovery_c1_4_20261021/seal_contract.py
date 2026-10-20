"""Phase C metadata seal, after artificial and local-matrix gates; no event execution."""
import ast
import difflib
import json
from pathlib import Path
import shutil
from support import HERE, ROOT, PRIOR, RECOVERY, R1, BASE, MODEL, digest, environment, save, verify_rows

def row(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': digest(path)}

def main():
    assert not (HERE / 'result').exists()
    for name in ('phase_a_checks.json', 'artificial_checks.json', 'phase_b/summary.json'):
        assert json.loads((HERE / name).read_text())['status'] == 'PASS'
    for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
    phase_b = json.loads((HERE / 'phase_b/execution_identity.json').read_text())
    assert phase_b['helper_sha256'] == digest(HERE / 'bitwise.py')
    assert phase_b['reference_sha256'] == digest(HERE / 'reference.py')
    old = json.loads((RECOVERY / 'execution_contract.json').read_text()); contract = dict(old)
    contract.update(schema='C1_3_R2_NEW_FIXED_ONE_SHOT_V1', base_commit=BASE,
        run_id='C1-3-R2-H4-selected-20261021', previous_contract_sha256=digest(RECOVERY / 'execution_contract.json'),
        shared_connection={'adapter': row(PRIOR / 'adapter.py'), 'reference': row(HERE / 'reference.py'),
            'bitwise_helper': row(HERE / 'bitwise.py'), 'state_identity_helper': row(RECOVERY / 'state_identity.py'),
            'journal': row(RECOVERY / 'journal.py'),
            'modification': 'saved local hex reconstruction only; all operator/norm/phase/XY definitions unchanged'},
        phase_A_PASS=row(HERE / 'phase_a_checks.json'), phase_B_PASS=row(HERE / 'phase_b/summary.json'))
    allowed = {'schema', 'base_commit', 'run_id', 'previous_contract_sha256', 'shared_connection'}
    assert all(contract[k] == old[k] for k in old if k not in allowed)
    original = ast.parse((PRIOR / 'reference.py').read_text()); current = ast.parse((HERE / 'reference.py').read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    left, right = functions(original), functions(current)
    assert set(left) == set(right) and all(left[n] == right[n] for n in left if n != 'saved_basis')
    save(HERE / 'execution_contract.json', contract)
    save(HERE / 'contract_comparison.json', {'all_scientific_fields_equal_to_R1': True,
         'checked_fields': sorted(set(old) - allowed), 'technical_metadata_changes_only': sorted(allowed),
         'unchanged_reference_function_ASTs': sorted(set(left) - {'saved_basis'}),
         'saved_basis_change': 'only raw local complex reconstruction, not composition/order/target/norm',
         'old_contract_sha256': digest(RECOVERY / 'execution_contract.json'),
         'new_contract_sha256': digest(HERE / 'execution_contract.json')})
    for filename in ('input_manifest.json', 'runtime_manifest.json'):
        verify_rows(json.loads((RECOVERY / filename).read_text())['files']); shutil.copyfile(RECOVERY / filename, HERE / filename)
    save(HERE / 'environment_identity.json', environment())
    assert environment() == json.loads((RECOVERY / 'environment_identity.json').read_text())
    baseline = json.loads((RECOVERY / 'preservation_baseline.json').read_text())
    for item in baseline['private_originals']: assert digest(item['path']) == item['sha256']
    save(HERE / 'preservation_baseline.json', {'base_commit': BASE,
         'rule': 'additions only under new namespace; prior SOURCE/results/authorizations/STOP unchanged',
         'private_originals': baseline['private_originals'], 'other_worktrees': 'untouched, not globally frozen'})
    shutil.copyfile(RECOVERY / 'policy_receipt.json', HERE / 'policy_receipt.json')
    changes = []
    for folder, oldname, newname in ((PRIOR, 'reference.py', 'reference.py'), (RECOVERY, 'run_c13_r1.py', 'run_c13_r2.py'),
                                    (RECOVERY, 'support.py', 'support.py')):
        changes.extend(difflib.unified_diff((folder / oldname).read_text().splitlines(True),
             (HERE / newname).read_text().splitlines(True), fromfile='prior/' + oldname, tofile='R2/' + newname, n=0))
    (HERE / 'technical_changes.patch').write_text(''.join(changes))
    previous_manifest = json.loads((RECOVERY / 'source_manifest.json').read_text())
    science = [r for r in previous_manifest['files'] if r['path'].startswith('src/')]; verify_rows(science)
    frozen = science + [row(p) for directory in (PRIOR, RECOVERY) for p in sorted(directory.rglob('*')) if p.is_file()]
    current = [row(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'source_manifest.json']
    save(HERE / 'source_manifest.json', {'base_commit': BASE, 'fixed_science_commit': previous_manifest['fixed_science_commit'],
         'files': frozen + current, 'self_hash_bound_by_new_authorization': True,
         'previous_failed_runs_unchanged': True, 'new_sampler_or_operator_algorithm': False})
    print(json.dumps({'phase_C_sealed': True, 'protected_files': len(frozen), 'new_files': len(current),
                      'scientific_condition_changes': 0, 'phase_C_scientific_attempts': 0}))

if __name__ == '__main__': main()
