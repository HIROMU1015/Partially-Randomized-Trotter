#!/usr/bin/env python3
"""Adoption metadata/AST/byte-identity checks. Never execute reference kernels.

Raw NPZ bytes may be hashed only inside raw_identity(); no array decoding,
numerical correctness testing, allowance evaluation or execution authority.
"""
import argparse
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
BASE = '2ba0858c3c92b61581cfbf6cdfb5039d25a174c9'
ART = 'artifacts/track_b_m1s_independent_contract_adoption/2026-10-16/v1'
PREP = 'artifacts/track_b_m1s_gate0_independent_preparation/2026-10-16/v1'


class NoNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit', 'zipfile', 'fractions', 'decimal'} or fullname.endswith(('.reference', '.execution')) or '.m1s_gate0_v5' in fullname or '.m1s_gate0_v4' in fullname:
            raise RuntimeError('Numerical import forbidden in adoption checks: '+fullname)


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.PIPE)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report')
    args = parser.parse_args()
    sys.meta_path.insert(0, NoNumericalImports())
    counts = {'NPZ_raw_byte_hash_reads': 0, 'NPZ_array_decodes': 0, 'reference_kernel_calls': 0}
    raw_hash_active = False

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)) and str(arguments[0]).lower().endswith(('.npz', ".npz'")):
            if not raw_hash_active:
                raise RuntimeError('Only raw NPZ byte identity allowed')
            counts['NPZ_raw_byte_hash_reads'] += 1

    def profile(frame, event, argument):
        if event == 'call' and frame.f_code.co_filename.endswith(('/m1s_gate0_independent_v1/reference.py', '/m1s_gate0_independent_v1/execution.py')):
            counts['reference_kernel_calls'] += 1
            raise RuntimeError('Reference calls forbidden')

    sys.addaudithook(audit)
    sys.setprofile(profile)
    sys.path.insert(0, str(ROOT/'src'))
    from trottertracks.algorithm_codesign.m1s_gate0_independent_v1 import control as c

    def load(p):
        return c.read_json((ROOT/p).read_bytes())

    def raw_identity(p):
        nonlocal raw_hash_active
        raw_hash_active = True
        try:
            return (ROOT/p).read_bytes()
        finally:
            raw_hash_active = False

    checks = []

    def check(name, ok):
        checks.append({'name': name, 'status': 'PASS' if ok else 'FAIL'})
        if not ok:
            raise AssertionError(name)

    candidate_path = PREP+'/independent_validation_contract_candidate.json'
    formal_path = ART+'/independent_validation_contract_adopted.json'
    candidate_raw = (ROOT/candidate_path).read_bytes()
    candidate, formal = load(candidate_path), load(formal_path)
    check('fixed_candidate_SHA256', sha(candidate_raw) == '0635dc992eeda64d4d19b2c61569391c6179ad7d74547262da50875731c1896f')
    check('candidate_bytes_unchanged', candidate_raw == git('show', BASE+':'+candidate_path))
    check('candidate_still_not_adopted', candidate['adopted'] is False and candidate['status'] == 'CANDIDATE_NOT_ADOPTED')
    changed = {k for k in candidate if not c._same(candidate[k], formal[k])}
    check('only_three_metadata_fields_changed', set(candidate) == set(formal) and changed == {'status', 'adopted', 'adoption_record'})
    check('all_scientific_conditions_typed_identical', all(c._same(formal[k], candidate[k]) for k in set(candidate)-changed))
    status = c.validate_contract(formal, require_adopted=True)
    check('adopted_contract_complete', status['valid'] and status['contract_complete'] and not status['pending'] and not status['errors'])
    check('adoption_does_not_permit_execution', status['actual_execution_permitted'] is False and formal['contract_grants_execution'] is False)
    record_raw = (ROOT/formal['adoption_record']['path']).read_bytes()
    record = c.read_json(record_raw)
    context = load(ART+'/adoption_context.json')
    quote = (ROOT/ART/'user_adoption_instruction.md').read_bytes()
    check('record_hash_bound', sha(record_raw) == formal['adoption_record']['sha256'])
    check('record_kind_matches_future_guard', record['record_kind'] == 'INDEPENDENT_VALIDATION_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION')
    check('record_explicitly_not_execution_authority', record['scientific_execution_authorized'] is False and record['contract_grants_execution'] is False)
    check('current_user_quote_bound', sha(quote) == record['user_instruction_sha256'] == context['current_user_instruction']['sha256'] and quote.decode() == context['current_user_instruction']['text'])
    check('quote_explicit_adoption_and_no_science', '正式採択します' in quote.decode() and '科学計算は承認しません。' in quote.decode() and BASE in quote.decode())
    check('adoption_context_hash', sha((ROOT/record['context_path']).read_bytes()) == record['context_sha256'])
    check('candidate_context_binding', record['candidate_commit'] == context['candidate_commit'] == BASE and record['candidate_sha256'] == context['candidate_sha256'] == sha(candidate_raw))
    check('no_circular_contract_record_hash', record['formal_contract_path'] == formal_path and 'formal_contract_sha256' not in record)
    for field, key in [('accepted_model', 'model'), ('accepted_arithmetic', 'arithmetic'), ('accepted_comparison', 'comparison'), ('accepted_resources', 'resources'), ('accepted_runtime', 'runtime')]:
        check('record_adopts_unchanged:'+key, c._same(record[field], formal[key]))
    check('full_HD_limit_explicitly_accepted', record['accepted_scope_limit']['full_scope_PASS_possible'] is False and record['accepted_scope_limit']['v5_rerun_to_recover_missing_matrices_authorized'] is False)
    check('FAIL_priority_and_missing_scope_UNDETERMINED', formal['comparison']['overall'] == candidate['comparison']['overall'] and formal['comparison']['full_HD_saved_path_match'] == 'UNDETERMINED_NOT_SAVED_BY_V5')
    check('separate_execution_approval_remains_required', formal['requires_separate_explicit_user_execution_approval'] is True and record['requires_separate_explicit_user_execution_approval'] is True)
    check('one_shot_no_retry_failure_STOP', formal['one_shot_consumed_on_output_reservation'] is True and formal['automatic_retry'] is False and formal['mandatory_STOP'] is True and record['mandatory_STOP'] is True)
    identity = load(ART+'/identity_ledger.json')
    check('formal_and_record_identity_ledger', identity['adopted_contract']['sha256'] == sha((ROOT/formal_path).read_bytes()) and identity['adoption_record'] == formal['adoption_record'])
    check('identity_source_and_v5_bindings', identity['independent_source'] == {'commit': formal['source_commit'], 'files_sha256': formal['source_sha256']} and identity['v5_source'] == formal['v5_source'] and identity['v5_proof'] == formal['v5_proof'])

    for p, digest in formal['source_sha256'].items():
        raw = (ROOT/p).read_bytes()
        ast.parse(raw, filename=p)
        check('SOURCE_hash_and_frozen_bytes:'+p, raw == git('show', formal['source_commit']+':'+p) and sha(raw) == digest)
    for binding, key in [(formal['v5_source'], 'sha256'), (formal['v5_proof'], 'files_sha256')]:
        commit = binding.get('commit', binding.get('proof_commit'))
        for p, digest in binding[key].items():
            raw = (ROOT/p).read_bytes()
            check('v5_SOURCE_proof_hash:'+p, raw == git('show', commit+':'+p) and sha(raw) == digest)
    for binding in (formal['comparator'], formal['v5_contract']):
        raw = (ROOT/binding['path']).read_bytes()
        check('comparator_contract_raw_bytes_only:'+binding['path'], raw == git('show', binding['commit']+':'+binding['path']) and sha(raw) == binding['sha256'])
    inp = formal['input']
    raw = raw_identity(inp['path'])
    check('input_raw_hash_only_no_decode', len(raw) == inp['bytes'] and sha(raw) == inp['sha256'] and raw == git('show', BASE+':'+inp['path']))
    common = identity['common_instructions']
    check('latest_common_instructions_preserved', sha((ROOT/common['path']).read_bytes()) == common['sha256'] and (ROOT/common['path']).read_bytes() == git('show', BASE+':'+common['path']))
    chain = [formal['v5_proof']['proof_commit'], formal['v5_source']['commit'], formal['v5_contract']['commit'], formal['base_result_commit'], formal['source_commit'], BASE]
    for earlier, later in zip(chain, chain[1:]):
        check('commit_order:'+earlier[:8]+'<'+later[:8], earlier != later and subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', earlier, later], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)

    package = 'src/trottertracks/algorithm_codesign/m1s_gate0_independent_v1/'
    trees = {name: ast.parse((ROOT/package/(name+'.py')).read_bytes()) for name in ('control', 'reference', 'execution')}
    functions = {name: {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} for name, tree in trees.items()}
    auth = ast.unparse(functions['control']['authorize'])
    main_text = ast.unparse(functions['control']['main'])
    compare = ast.unparse(functions['reference']['compare_saved'])
    law = ast.unparse(functions['reference']['compare_law'])
    run = ast.unparse(functions['execution']['run_authorized'])
    check('guard_checks_record_hash_and_no_execution_kind', 'record_kind' in auth and 'scientific_execution_authorized' in auth and "record.get('scientific_execution_authorized') is not False" in auth and 'adoption[\'sha256\']' in auth)
    check('guard_requires_later_separate_authority_commit', 'cc == head' in auth and 'sc in (cc, head)' in auth and 'explicit_user_scientific_execution_approval' in auth)
    check('guard_rejects_old_authority_namespace', 'New independent-verification authority namespace required' in auth)
    check('scientific_dispatch_after_full_guard', main_text.index('permit = authorize(') < main_text.index('from .execution import run_authorized'))
    check('full_scope_UNDETERMINED_unless_FAIL_in_SOURCE', "'status': 'FAIL' if primary == 'FAIL' else 'UNDETERMINED'" in compare and 'UNDETERMINED_NOT_SAVED_BY_V5' in law)
    check('available_scope_separately_reported', 'available_comparator_status' in compare and 'available_comparator_status' in run)
    check('references_frozen_before_comparator_read', run.index('independent = reference.build(input_raw)') < run.index('comparator_raw =') < run.index('reference.compare_saved('))
    for group in ('comparison', 'resources', 'arithmetic', 'runtime'):
        for key in formal[group]:
            mutated = copy.deepcopy(formal)
            mutated[group][key] = 'UNAUTHORIZED_CHANGE'
            check('reject_condition_change:'+group+'.'+key, not c.validate_contract(mutated, require_adopted=True)['valid'])
    for key, value in [('adopted', 1), ('contract_grants_execution', True), ('automatic_retry', True)]:
        mutated = copy.deepcopy(formal)
        mutated[key] = value
        check('reject_metadata_permission_mutation:'+key, not c.validate_contract(mutated, require_adopted=True)['valid'])
    for argv in (['validate-config', '--contract', str(ROOT/formal_path)], ['run'], ['run', '--contract', 'not-read.json', '--authorization-commit', 'not-a-sha', '--authorization-path', 'old-v5-authority']):
        with redirect_stdout(io.StringIO()) as output:
            code = c.main(argv)
        response = json.loads(output.getvalue())
        expected = 0 if argv[0] == 'validate-config' else 2
        check('metadata_validation_or_execution_refusal:'+repr(argv[:1]), code == expected and response.get('actual_execution_permitted') is False)
    for path, commit in [('artifacts/track_b_m1s_gate0_v5_execution_authorization/2026-10-16/v1/execution_authorization.json', '8da63d672f779c947dc295eb2c5d524d927ac66b'), (c.AUTH_PREFIX+'not_created.json', 'not-a-sha')]:
        try:
            c.authorize(ROOT, formal_path, commit, path)
        except c.ExecutionClosed:
            refused = True
        else:
            refused = False
        check('old_or_absent_authority_refused:'+path, refused)
    check('no_new_scientific_authority_or_result', not (ROOT/c.AUTH_PREFIX).exists() and not (ROOT/c.OUTPUT_PREFIX).exists())
    inherited = load(ART+'/inherited_source_static_validation.json')
    check('inherited_SOURCE_suite_143_PASS', inherited['PASS'] == 143 and inherited['FAIL'] == 0 and inherited['reference_kernel_calls'] == 0 and inherited['NPZ_reads'] == 0)
    readiness = load(ART+'/execution_readiness.json')
    check('readiness_adopted_but_not_authorized', readiness['validation_contract_adopted'] is True and readiness['scientific_execution_authorized'] is False and readiness['actual_execution_permitted'] is False and readiness['full_scope_PASS_possible_with_current_data'] is False)
    check('no_array_decode_or_reference_calls', counts['NPZ_array_decodes'] == counts['reference_kernel_calls'] == 0 and counts['NPZ_raw_byte_hash_reads'] == 1)
    report = {'schema': 'm1s_independent_contract_adoption_static_v1', 'PASS': len(checks), 'FAIL': 0, 'checks': checks,
              'contract_status': status, **counts, 'inherited_source_static_PASS': 143,
              'synthetic_numerical_tests': 0, 'Pauli_generation': 0, 'new_W_v_rho_evaluations': 0,
              'runtime_or_numerical_correctness_verified': False, 'new_scientific_calculations': 0,
              'new_authorizations_grants_markers': 0, 'mandatory_STOP': True}
    if args.write_report:
        (ROOT/args.write_report).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'contract_status')}))


if __name__ == '__main__':
    main()
