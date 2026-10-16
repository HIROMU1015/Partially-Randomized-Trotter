#!/usr/bin/env python3
"""AST, JSON and authority-refusal checks only; numerical evaluation forbidden."""
import argparse
import ast
import copy
import hashlib
import importlib.abc
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))


class BlockNumerics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'openfermion', 'qiskit'} or fullname.endswith('.science'):
            raise RuntimeError('Forbidden scientific import: ' + fullname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract')
    parser.add_argument('--write-report')
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockNumerics())
    accesses = {'NPZ': 0, 'numeric_budget_calls': 0}
    def block_npz(event, argv):
        if event == 'open' and isinstance(argv[0], (str, bytes)) and str(argv[0]).lower().endswith(('.npz', ".npz'")):
            accesses['NPZ'] += 1
            raise RuntimeError('NPZ access is forbidden in static checks')
    sys.addaudithook(block_npz)
    def deny_budget(frame, event, arg):
        if event == 'call' and frame.f_code.co_filename.endswith('/m1s_gate0_v5/budget.py') and frame.f_code.co_name not in {'<module>', '_Budget', 'formula_policy', 'is_formula'}:
            accesses['numeric_budget_calls'] += 1
            raise RuntimeError('Numerical budget invocation forbidden: ' + frame.f_code.co_name)
    sys.setprofile(deny_budget)
    from trottertracks.algorithm_codesign.m1s_gate0_v5 import control as c
    from trottertracks.algorithm_codesign.m1s_gate0_v5.budget import FORMULA_IDS
    from trottertracks.algorithm_codesign.m1s_gate0_tail_bound_spec import COUNTS
    checks = []
    def check(name, value):
        checks.append({'name': name, 'status': 'PASS' if value else 'FAIL'})
        if not value:
            raise AssertionError(name)
    def parse(path):
        return ast.parse((ROOT / path).read_bytes(), filename=path)
    def functions(tree):
        return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    def dump(node):
        return ast.dump(node, include_attributes=False)
    trees = {path: parse(path) for path in c.SOURCE_PATHS}
    for path in trees:
        check('syntax:' + path, True)
    bp = 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/budget.py'
    sp = 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/science.py'
    cp = 'src/trottertracks/algorithm_codesign/m1s_gate0_v5/control.py'
    bt, st, ct = trees[bp], trees[sp], trees[cp]
    bf, sf, cf = functions(bt), functions(st), functions(ct)
    oldb = functions(parse('src/trottertracks/algorithm_codesign/m1s_gate0_v4/budget.py'))
    olds = functions(parse('src/trottertracks/algorithm_codesign/m1s_gate0_v4/science.py'))
    check('no_budget_top_level_calculation', all(isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) or isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict) for n in bt.body))
    check('budget_permit_first', ast.unparse(bf['__init__'].body[0]) == 'require_permit(permit)')
    for name in ('local_coefficient_ledger', 'local_coefficient_ledgers'):
        check(name + ':permit_first', ast.unparse(bf[name].body[0]) == 'require_permit(self.permit)')
    check('law_permit_first', ast.unparse(bf['law_limits'].body[1]) == 'require_permit(self.permit)')
    for name in ('q', 'l1', 'gamma', 'up', 'global_limits'):
        check('global_arithmetic_unchanged:' + name, dump(bf[name]) == dump(oldb[name]))
    original_init = oldb['__init__']
    check('global_constructor_prefix_unchanged', [dump(n) for n in bf['__init__'].body[:len(original_init.body)]] == [dump(n) for n in original_init.body])
    for name in ('_sum', '_combine', '_multiply', '_jw_ladder', '_one_body_pauli', '_occupation_ladder', '_reference_one_body', '_apply_pauli', '_pauli_matrix', '_real_map', '_norm', 'load', 'gate', 'finite'):
        check('unchanged_pipeline_AST:' + name, dump(sf[name]) == dump(olds[name]))
    oldprep = copy.deepcopy(olds['prepare'])
    newprep = copy.deepcopy(sf['prepare'])
    for node in ast.walk(newprep):
        if isinstance(node, ast.If):
            node.body[:] = [n for n in node.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'tail_coefficient_ledgers' for t in n.targets))]
    check('prepare_only_adds_local_ledger', dump(newprep) == dump(oldprep))
    oldsplit, newsplit = olds['split'], sf['split']
    boundary = next(i for i,n in enumerate(oldsplit.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'lam == 0')
    check('split_coefficient_identity_reconstruction_AST_unchanged', [dump(n) for n in oldsplit.body[:boundary]] == [dump(n) for n in newsplit.body[:boundary]])
    init_text = ast.unparse(bf['__init__'])
    local_text = ast.unparse(bf['local_coefficient_ledger'])
    law_text = ast.unparse(bf['law_limits'])
    prep_text = ast.unparse(sf['prepare'])
    split_text = ast.unparse(sf['split'])
    tail_cache = next(n for n in bf['__init__'].body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'tail_block_scales' for t in n.targets))
    check('tail_scale_is_raw_input_mass', dump(tail_cache.value) == dump(ast.parse('tuple(l*g*g for l,g in zip(ls,gs))', mode='eval').body) and 'sum(self.tail_block_scales[count:])' in init_text)
    check('tail_index_713_plus_block_count', COUNTS['tail_gamma_base'] == 713 and "COUNTS['tail_gamma_base'] + r" in init_text and 'self.gamma(index) * scale' in init_text)
    check('tail_models_are_fixed_G3_G6_G9', 'for count in (3, 6, 9)' in init_text and 'r = 12 - count' in init_text)
    check('local_alpha_before_coefficients', prep_text.index('self.budget =') < prep_text.index('local_coefficient_ledgers()') < prep_text.index('self.bilinears ='))
    check('no_observed_count_or_metric_in_tail_alpha', all(x not in init_text[init_text.index('self.tail_block_scales ='):] for x in ('contributions', 'residual', 'lambda_R', 'rho', 'reconstruction')))
    check('local_ledger_explicitly_conditional', 'AUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWED' in local_text and 'resource_cap_or_observed_counter_used' in local_text)
    check('split_is_explicit_law_argument', ast.unparse(bf['law_limits'].args) == 'self, coefficients, lam, precision, *, split_count')
    check('law_uses_local_alpha', 'alpha_R = self.tail_models[split_count][3]' in law_text and 'self.alpha' not in law_text)
    check('no_scalar_fallback', 'values[key] = precision[key]' not in law_text and "values['W_floor'] = precision['W_floor']" not in law_text)
    check('no_ratio_gate_from_feasibility_lemma', 'S_R_over_lambda_hat_max' not in init_text + law_text + split_text and 'L_over_lambda_hat_max' not in init_text + law_text + split_text)
    check('law_before_metrics', split_text.index('law_limits(') < split_text.index('hpsi =') < split_text.index('W ='))
    check('precision_failure_before_metrics', split_text.index('PRECISION_UNRESOLVED_BEFORE_METRICS') < split_text.index('hpsi ='))
    check('per_split_precision_no_global_mutation', 'self.eps.update(resolved)' not in split_text)
    # The propagation assignment sequence is identical to v4, substituting alpha_R only.
    start = 'L'
    def assignments(fn):
        out = []
        active = False
        for n in fn.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == start for t in n.targets):
                active = True
            if active and isinstance(n, (ast.Assign, ast.AugAssign)):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ledger' for t in n.targets):
                    break
                out.append(ast.unparse(n))
        return out
    check('W_v_propagation_assignments_unchanged', [s.replace('self.alpha', 'alpha_R') for s in assignments(oldb['law_limits'])] == assignments(bf['law_limits']))
    floor_old = next(n for n in oldb['law_limits'].body if isinstance(n, ast.If) and "is_formula(precision['W_floor']" in ast.unparse(n.test))
    floor_new = next(n for n in bf['law_limits'].body if isinstance(n, ast.If) and "is_formula(precision['W_floor']" in ast.unparse(n.test))
    check('rho_floor_expression_unchanged', [dump(n) for n in floor_old.body] == [dump(n) for n in floor_new.body])
    check('rho_interval_and_metric_tail_unchanged', [dump(n) for n in oldsplit.body[oldsplit.body.index(next(n for n in oldsplit.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'probabilities')):]] == [dump(n) for n in newsplit.body[newsplit.body.index(next(n for n in newsplit.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'probabilities')):]])
    run_text = ast.unparse(sf['run_authorized'])
    check('separate_top_level_global_local_ledgers', 'global_formula_resolution_before_metrics' in run_text and 'tail_coefficient_resolution_before_metrics' in run_text)
    check('failed_run_keeps_both_ledgers', run_text.count('engine.tail_coefficient_ledgers') == 2 and run_text.count('engine.budget_ledger') == 2)
    check('all_three_rows_preserved', "for s in ('G3', 'G6', 'G9')" in run_text)
    for category in ('INPUT_MODEL_NOT_APPLICABLE_TECHNICAL_STOP','RECONSTRUCTION_UNRESOLVED','ORACLE_QUALITY_UNRESOLVED','PRECISION_UNRESOLVED','MEMORY_LIMIT_OR_ALLOCATION_FAILURE'):
        check('failure_category_retained:' + category, category in ast.unparse(st))
    check('exclusive_result_container', 'output.mkdir(parents=True, exist_ok=False)' in run_text)
    check('process_and_file_limits_retained', 'RLIMIT_AS' in run_text and 'RLIMIT_FSIZE' in run_text)
    check('strict_JSON_and_output_cap', 'allow_nan=False' in run_text and "limits['output_bytes']" in run_text)
    auth_text = ast.unparse(cf['authorize'])
    main_text = ast.unparse(cf['main'])
    check('guard_before_science_import', main_text.index('authorize(') < main_text.index('from .science import run_authorized'))
    check('new_authority_schema', 'm1s_gate0_separate_execution_authorization_v5' in auth_text and 'm1s_gate0_separate_execution_authorization_v4' not in auth_text)
    check('new_authority_namespace', c.AUTHORIZATION_PREFIX == 'artifacts/track_b_m1s_gate0_v5_execution_authorization/' and 'auth_path.startswith(AUTHORIZATION_PREFIX)' in auth_text)
    check('new_result_namespace', c.OUTPUT_PREFIX == 'artifacts/track_b_m1s_gate0_v5_result/' and 'output.startswith(OUTPUT_PREFIX)' in auth_text)
    check('proof_source_contract_authority_chain', "proof['proof_commit'], sc" in auth_text and 'sc in (head, cc)' in auth_text and 'cc == head' in auth_text)
    check('proof_model_identity_bound', 'proof/model identity mismatch' in auth_text)
    check('frozen_npz_checked_before_permit', auth_text.index('Frozen input byte identity mismatch') < auth_text.index('return _Permit'))
    check('scope_record_not_execution_permission', 'SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION' in auth_text)
    check('clean_HEAD_and_no_overwrite_guards', 'Dedicated execution checkout must be clean' in auth_text and 'Output already exists' in auth_text)
    template = c.candidate()
    check('template_has_17_items', len(c.EPS_KEYS) == len(template['precision']) - 1 == 17)
    check('template_valid_but_not_ready', c.validate_config(template)['valid_static_shape'] and not c.validate_config(template)['execution_ready'])
    check('candidate_adoption_and_coverage_false', template['adopted'] is False and template['formula_policy']['coverage_model_adopted'] is False)
    check('all_9_formula_selectors_allowlisted', all(template['precision'][k] == {'formula_id': fid} for k,fid in FORMULA_IDS.items()))
    original = json.loads((ROOT / 'artifacts/track_b_m1s_gate0_v4_formal_adoption/2026-10-16/v1/execution_contract_v4_adopted.json').read_bytes())
    changed = {k for k in template['precision'] if template['precision'][k] != original['precision'][k]}
    check('only_3_precision_formula_IDs_changed', changed == {'W_error_budget','v_error_budget','W_floor'})
    check('runtime_caps_unchanged', template['runtime'] == original['runtime'] and template['limits'] == original['limits'])
    for key in c.EPS_KEYS:
        bad = c.candidate()
        bad['precision'][key] = {'formula_id': 'UNAPPROVED'} if key in FORMULA_IDS else (1 if bad['precision'][key] == 0 else 0)
        check('precision_change_rejected:' + key, not c.validate_config(bad)['valid_static_shape'])
    for key in FORMULA_IDS:
        bad = c.candidate();bad['precision'][key]['expression'] = 'fit_to_observed_result'
        check('arbitrary_selector_expression_rejected:' + key, not c.validate_config(bad)['valid_static_shape'])
    for key,value in [('tail_gamma_base',27781),('tail_scale','full_H_scale'),('oracle_absolute_residual_quality_hartree',1e-7)]:
        bad = c.candidate();bad['formula_policy'][key] = value
        check('policy_mutation_rejected:' + key, not c.validate_config(bad)['valid_static_shape'])
    for area,key,value in [('limits','wall_time_seconds',1),('limits','memory_bytes',8589934592),('runtime','workers',2),('runtime','backend','gpu')]:
        bad = c.candidate();bad[area][key] = value
        check('runtime_mutation_rejected:' + key, not c.validate_config(bad)['valid_static_shape'])
    check('old_v4_contract_rejected', not c.validate_config(original)['valid_static_shape'])
    try:
        c.require_permit(None)
    except ValueError:
        check('missing_permit_refused', True)
    else:
        check('missing_permit_refused', False)
    for path,commit,label in [('artifacts/track_b_m1s_gate0_execution_authorization/2026-10-16/v4/execution_authorization.json',None,'legacy_authority_path'),(c.AUTHORIZATION_PREFIX+'not_created.json',None,'absent_authority_commit'),(c.AUTHORIZATION_PREFIX+'not_created.json','0'*40,'wrong_HEAD')]:
        try:
            c.authorize(ROOT, 'unused_contract.json', commit, path)
        except ValueError:
            check('authority_refusal_before_NPZ:' + label, True)
        else:
            check('authority_refusal_before_NPZ:' + label, False)
    for command in (['status'], ['schema']):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = c.main(command)
        result = json.loads(buffer.getvalue())
        check('safe_CLI:' + command[0], code == 0 and isinstance(result, dict))
    if args.contract:
        fixed = c.read_json_bytes((ROOT / args.contract).read_bytes())
        result = c.validate_config(fixed)
        check('final_candidate_static_complete', result['valid_static_shape'] and not result['execution_ready'])
        check('candidate_pending_only_adoption', set(result['pending']) == {'formal_execution_contract_adoption','conditional_roundoff_model_adoption','adoption_record'})
    check('no_NPZ_access', accesses['NPZ'] == 0)
    check('no_numeric_budget_invocation', accesses['numeric_budget_calls'] == 0)
    check('no_science_or_numeric_import', not any(n.split('.')[0] in {'numpy','scipy','openfermion','qiskit'} or n.endswith('.m1s_gate0_v5.science') for n in sys.modules))
    sys.setprofile(None)
    report = {'schema':'m1s_gate0_v5_static_source_validation_v1','PASS':len(checks),'FAIL':0,'checks':checks,'NPZ_reads':accesses['NPZ'],'numerical_formula_evaluations':accesses['numeric_budget_calls'],'Pauli_generation':0,'W_v_rho_evaluations':0,'scientific_runner_invocations':0,'runtime_or_numerical_correctness_verified':False,'independent_mathematical_acceptance_performed':False,'evidence_classification':c.EVIDENCE_CLASSIFICATION,'mandatory_STOP':True}
    if args.write_report:
        output = ROOT / args.write_report
        if not output.resolve().is_relative_to(ROOT / 'artifacts/track_b_m1s_gate0_v5_source_preparation'):
            raise ValueError('New v5 preparation report namespace required')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, indent=2))


if __name__ == '__main__':
    main()
