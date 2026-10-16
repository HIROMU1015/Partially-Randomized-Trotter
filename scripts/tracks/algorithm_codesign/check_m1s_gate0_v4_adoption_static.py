#!/usr/bin/env python3
"""Read-only contract adoption audit. No scientific imports or formula evaluation."""
from pathlib import Path
import ast
import hashlib
import importlib.abc
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BASE = "44277e03448ee51dbe9afdd2dc6560337757cc6d"
ART = "artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1"


class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit'} or fullname.endswith('.science'):
            raise RuntimeError('Numerical import forbidden: ' + fullname)


def main():
    sys.meta_path.insert(0, BlockNumerics())
    formula_calls = []
    def block_budget(frame, event, arg):
        if (event == 'call' and frame.f_code.co_filename.endswith('/m1s_gate0_v4/budget.py')
                and frame.f_code.co_name in {'__init__', 'q', 'l1', 'gamma', 'up', 'global_limits', 'law_limits'}):
            formula_calls.append(frame.f_code.co_name)
            raise RuntimeError('Formula evaluation forbidden')
    sys.setprofile(block_budget)
    sys.path.insert(0, str(ROOT / 'src'))
    from trottertracks.algorithm_codesign.m1s_gate0_v4 import control as c
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'PASS': bool(ok)})
        if not ok:
            raise AssertionError(name)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args])
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    def get(name):
        return c.read_json_bytes((ROOT / ART / name).read_bytes())
    formal = get('execution_contract_v4_adopted.json')
    record = get('user_final_contract_adoption_record.json')
    candidate_path = record['candidate_contract_path']
    candidate_raw = git('show', BASE + ':' + candidate_path)
    candidate = c.read_json_bytes(candidate_raw)
    check('reviewed commit fixed', record['reviewed_candidate_commit'] == BASE)
    check('candidate hash bound', sha(candidate_raw) == record['candidate_contract_sha256'])
    check('formal contract hash bound', sha((ROOT / record['formal_contract_path']).read_bytes()) == record['formal_contract_sha256'])
    check('user formal adoption status', formal['adopted'] is True and formal['status'] == 'ADOPTED')
    check('conditional model formally accepted', formal['formula_policy']['coverage_model_adopted'] is True)
    check('conditional evidence retained', formal['evidence_classification'] == c.EVIDENCE_CLASSIFICATION == 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    check('record explicitly is adoption only', record['record_kind'] == 'FINAL_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION')
    check('science explicitly withheld', record['scientific_execution_authorized'] is False and record['user_explicitly_withheld_scientific_execution'] is True)
    check('user quotation retained', record['user_message_quote'] in formal['adoption_record'])
    check('record source bound', record['source_commit'] == formal['source_commit'] and record['source_sha256'] == formal['source_sha256'])
    changed = get('candidate_to_adopted_semantic_diff.json')['changed_fields']
    check('only four adoption metadata fields changed', {r['path'] for r in changed} == {
        '.status', '.adopted', '.adoption_record', '.formula_policy.coverage_model_adopted'})
    normalized = json.loads(json.dumps(formal))
    for key in ('status', 'adopted', 'adoption_record'):
        normalized[key] = candidate[key]
    normalized['formula_policy']['coverage_model_adopted'] = candidate['formula_policy']['coverage_model_adopted']
    check('all contract substance identical to adopted candidate', normalized == candidate)
    check('all 17 precision items', set(formal['precision']) == set(c.EPS_KEYS) | {'dtype'})
    check('precision policy unchanged', formal['precision'] == candidate['precision'])
    check('runtime and limits unchanged', formal['runtime'] == candidate['runtime'] and formal['limits'] == candidate['limits'])
    check('conditional assumptions unchanged', formal['formula_policy']['assumptions'] == candidate['formula_policy']['assumptions'])
    result = c.validate_config(formal, require_adopted=True)
    check('adopted contract JSON complete', result['execution_ready'] and not result['errors'] and not result['pending'])
    check('schema check performs no science', result['scientific_processing_performed'] is False)
    check('no numerical budget helper called', not formula_calls)
    sys.setprofile(None)
    for path, expected in formal['source_sha256'].items():
        raw = (ROOT / path).read_bytes()
        check('source bytes/hash fixed: ' + path, raw == git('show', formal['source_commit'] + ':' + path) and sha(raw) == expected)
    check('source is earlier ancestor', subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', formal['source_commit'], BASE], check=False).returncode == 0)
    input_raw = (ROOT / formal['input']['path']).read_bytes()
    check('input raw bytes/hash fixed; no array decoding', len(input_raw) == 8652 and sha(input_raw) == formal['input']['sha256'])
    scope = formal['scientific_scope_record']
    scope_raw = (ROOT / scope['path']).read_bytes()
    check('scope adoption record hash fixed', sha(scope_raw) == scope['sha256'])
    scope_data = c.read_json_bytes(scope_raw)
    for key in ('input', 'split_ids', 'basis', 'law', 'state', 'decisions'):
        check('adopted scientific scope unchanged: ' + key, formal[key] == scope_data[key])
    table = get('precision_specification_17_adopted.json')
    check('17 specification rows', table['precision_adopted'] is True and len(table['items']) == 17)
    for row in table['items']:
        check('precision item adopted and unchanged: ' + row['key'], row['formally_adopted'] is True and row['adopted_value_or_rule'] == formal['precision'][row['key']])
    status = get('adoption_status.json')
    check('contract adopted; actual execution not permitted', status['formal_contract_adopted'] is True and status['actual_execution_permitted'] is False and status['scientific_execution_authorized'] is False)
    check('no execution authorization/grant/marker', status['new_scientific_execution_authorizations'] == status['new_grants'] == status['new_markers'] == 0)
    check('no runtime or numerical certification', record['coverage_model_conditions_certified'] is False and status['runtime_or_numerical_correctness_verified'] is False)
    for row in get('reviewed_candidate_public_identity.json')['files']:
        check('reviewed candidate remote bytes verified: ' + row['path'], row['GitHub_git_local_bytes_equal'] and sha((ROOT / row['path']).read_bytes()) == row['sha256'])
    tree = ast.parse((ROOT / 'src/trottertracks/algorithm_codesign/m1s_gate0_v4/control.py').read_text())
    auth = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'authorize')
    auth_text = ast.unparse(auth)
    check('future distinct execution authorization schema retained', 'm1s_gate0_separate_execution_authorization_v4' in auth_text)
    check('source-contract-authorization commit ordering retained', 'sc in (head, cc)' in auth_text and 'cc == head' in auth_text)
    check('separate user execution approval retained', 'explicit_user_scientific_execution_approval' in auth_text)
    check('exclusive output retry guard retained', 'Output already exists; no implicit retry/overwrite' in auth_text)
    indexes = ['PROJECT_MAP.md', 'scripts/README.md', 'src/trotterlib/README.md', 'docs/README.md', 'docs/research/研究概要・現状.md', 'VALIDATION_STATUS.md']
    for path in indexes:
        check('earlier index body byte-preserved: ' + path, (ROOT / path).read_bytes().endswith(git('show', BASE + ':' + path)))
    before_manifest = c.read_json_bytes(git('show', BASE + ':artifacts/validation_manifest.json'))
    now_manifest = c.read_json_bytes((ROOT / 'artifacts/validation_manifest.json').read_bytes())
    check('earlier manifest entries unchanged', all(now_manifest[k] == v for k, v in before_manifest.items()))
    check('one adoption manifest entry added', set(now_manifest) - set(before_manifest) == {'track_b_m1s_gate0_v4_formal_adoption_20261016'})
    check('earlier manifest body byte-preserved', (ROOT / 'artifacts/validation_manifest.json').read_bytes().endswith(git('show', BASE + ':artifacts/validation_manifest.json')[2:]))
    docs = ['docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v4_formally_adopted_20261016.md',
            'docs/research/研究ノート/2026-10-16_track_b_M1S_Gate0_v4_formal_adoption.md']
    for path in docs + indexes:
        text = (ROOT / path).read_text()
        if path in indexes:
            text = text.split('\n\n', 1)[0]
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                continue
            check('local link: ' + path + ' -> ' + target, (ROOT / path).parent.joinpath(target.split('#')[0]).exists())
    preserved = []
    allowed = set(indexes + ['artifacts/validation_manifest.json'])
    for path in filter(None, git('ls-tree', '-r', '--name-only', '-z', BASE).decode().split('\0')):
        if path in allowed:
            continue
        raw = (ROOT / path).read_bytes()
        if raw != git('show', BASE + ':' + path):
            raise AssertionError('Frozen base file changed: ' + path)
        preserved.append({'path': path, 'bytes': len(raw), 'sha256': sha(raw)})
    check('all earlier SOURCE/candidates/results/adoption/auth/marker bytes preserved', True)
    check('numerical libraries and science never imported', not any(n in sys.modules for n in ('numpy', 'scipy', 'openfermion', 'qiskit')) and not any(n.endswith('.m1s_gate0_v4.science') for n in sys.modules))
    return {'schema': 'm1s_gate0_v4_adoption_static_validation_v1', 'PASS': len(checks), 'checks': checks,
        'schema_validation': result,
        'contract_configuration_complete': True, 'actual_execution_permitted': False,
        'scientific_execution_authorized': False,
        'earlier_base_files_preserved': len(preserved),
        'base_inventory_order': 'git ls-tree -r --name-only -z BASE excluding index append paths',
        'base_inventory_sha256': sha(json.dumps(preserved, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()),
        'evidence_classification': c.EVIDENCE_CLASSIFICATION,
        'old_static_99_and_document_214_checks': 'Prior candidate evidence retained; not rerun as a complete suite',
        'runtime_or_numerical_correctness_verified': False,
        'new_scientific_calculations': 0, 'new_Pauli_tables': 0, 'NPZ_arrays_decoded': 0,
        'numerical_formula_evaluations': 0, 'new_numerical_correctness_tests': 0,
        'new_scientific_execution_authorizations': 0, 'new_grants': 0, 'new_markers': 0,
        'mandatory_STOP': True}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
