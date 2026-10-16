#!/usr/bin/env python3
"""Static design checks only: AST/JSON, integer counts and dyadic inequalities.

Never decode NPZ, generate Pauli coefficients, evaluate an input-dependent error
budget, import a scientific module, or execute Gate0. This is not a proof assistant
or a numerical/platform correctness test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ART = 'artifacts/track_b_m1s_gate0_tail_bound/2026-10-16/v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    checks = []
    def check(name, value):
        checks.append({'name': name, 'status': 'PASS' if value else 'FAIL'})
    def obj(path):
        return json.loads((ROOT / path).read_bytes())
    def dyadic_less(terms, target_exponent):
        # sum(n/2**p) < 1/2**target_exponent, using integers only.
        shift = max([target_exponent] + [p for n, p in terms])
        return sum(n << (shift - p) for n, p in terms) < (1 << (shift - target_exponent))
    spec_path = 'src/trottertracks/algorithm_codesign/m1s_gate0_tail_bound_spec.py'
    spec_raw = (ROOT / spec_path).read_bytes()
    tree = ast.parse(spec_raw)
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)}
    check('declarative_spec_only', all(isinstance(n, ast.Assign) or isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for n in tree.body))
    check('no_execution_or_adoption', values['INTEGRATION_IMPLEMENTED'] is False and values['FORMAL_ADOPTION'] is False)
    model = obj(ART + '/tail_local_model_candidate.json')
    candidate = obj(ART + '/precision_contract_v5_design_candidate.json')
    old = obj(candidate['base_formal_contract']['path'])
    check('frozen_base_contract_hash', hashlib.sha256((ROOT / candidate['base_formal_contract']['path']).read_bytes()).hexdigest() == candidate['base_formal_contract']['sha256'])
    check('frozen_coefficient_pipeline_hash', hashlib.sha256((ROOT / model['source_target_path']).read_bytes()).hexdigest() == model['source_target_sha256'])
    check('spec_hash', hashlib.sha256(spec_raw).hexdigest() == model['spec_source_sha256'])
    c = values['COUNTS']
    check('count_ledger_matches_spec', c == model['counts'])
    check('one_body_structural_support', c['one_body_support_max'] == 1 + 8 + 4 * (8 * 7 // 2) == 121)
    check('one_body_per_word_group', c['one_body_group_max'] == c['n_modes'] ** 2 == 64)
    check('square_per_word_group_uses_bijection', c['square_group_max'] == c['one_body_support_max'] == 121)
    check('leaf_phase_and_coefficient_products', c['multiply_leaf_gamma_index'] == (c['phase_products_per_word'] + c['coefficient_products_per_word']) * c['complex_product_real_budget'] == 80)
    check('bilinear_index', c['bilinear_gamma_index'] == c['multiply_leaf_gamma_index'] + c['bilinear_group_max'] + c['fsum_padding'] == 116)
    check('one_body_index', c['one_body_gamma_index'] == c['bilinear_gamma_index'] + 8 + c['one_body_group_max'] + c['fsum_padding'] == 220)
    check('square_index_has_both_inputs', c['square_gamma_index'] == 2 * c['one_body_gamma_index'] + c['multiply_leaf_gamma_index'] + c['square_group_max'] + c['fsum_padding'] == 673)
    check('block_scale_index', c['scaled_block_gamma_index'] == c['square_gamma_index'] + 8 == 681)
    check('tail_index', c['tail_gamma_base'] == c['scaled_block_gamma_index'] + c['fsum_padding'] == 713)
    check('max_rank12_index', c['tail_gamma_max_rank12'] == c['tail_gamma_base'] + c['df_rank'] == 725)
    for split, count in [('G3', 3), ('G6', 6), ('G9', 9)]:
        check(split + '_fixed_tail_index', model['tail_gamma_indices'][split] == 713 + 12 - count)
    check('no_observed_counter_or_resource_count', model['observed_counter_used'] is False and model['resource_caps_are_not_gamma_path_counts'] is True)
    # gamma_k <= 2 k u <= 2^-p, with ku <= 1/2. No floating arithmetic.
    for name, count, exponent in [('tail', 725, 42), ('fsum', 65536 + 32, 35), ('assembly', 8 * 65536 + 32, 32), ('dot', 8 * 256 + 32, 40), ('division', 8, 49), ('mean', 4 * 65536 + 32, 33), ('variance', 16 * 65536 + 32, 31)]:
        check('dyadic_gamma_' + name, 2 * count <= 2**53 and 2 * count * 2**exponent <= 2**53)
    inequalities = [
        ('lambda_error_ratio', [(1, 39), (1, 34)], 33),
        ('ec', [(1, 38), (1, 31)], 30),
        ('matvec', [(1, 35), (1, 67)], 34),
        ('division', [(1, 43), (1, 75)], 42),
        ('y', [(1, 30), (1, 27), (1, 34), (1, 42)], 26),
        ('quotient_numerator', [(1, 39), (1, 48)], 38),
        ('subtraction', [(17, 53)], 48),
        ('W_precision_ceiling', [(1, 25), (1, 52), (1, 36), (1, 48)], 24),
        ('ep', [(1, 30), (1, 48)], 29),
        ('mass_defect', [(1, 34), (1, 48)], 33),
        ('delta', [(1, 29), (3, 37), (4, 33)], 28),
        ('v_precision_ceiling', [(1, 27), (1, 25), (1, 55), (1, 26)], 24),
    ]
    for name, terms, target in inequalities:
        check('dyadic_' + name, dyadic_less(terms, target))
    check('A_upper_two', 9**2 < 2 * 8**2)
    check('am_upper_two', 9**3 < 2 * 8**3)
    check('variance_center_magnitude_upper_four', 25 < 4 * 8)
    check('sufficient_domain_not_an_execution_gate', candidate['feasibility_lemma_is_not_new_execution_gate'] is True and values['FEASIBILITY_DOMAIN']['is_new_execution_gate'] is False)
    check('17_precision_items', len(candidate['precision']) - 1 == candidate['precision_item_count'] == 17)
    changed = {key for key in old['precision'] if candidate['precision'][key] != old['precision'][key]}
    check('only_three_law_selectors_changed', changed == {'W_error_budget', 'v_error_budget', 'W_floor'})
    check('all_precision_ceilings_unchanged', all(candidate['precision'][key] == old['precision'][key] for key in old['precision'] if key not in changed))
    check('runtime_and_caps_unchanged', candidate['runtime'] == old['runtime'] and candidate['limits'] == old['limits'])
    check('scope_unchanged', all(candidate['scientific_scope'][key] == old[key] for key in candidate['scientific_scope']))
    check('same_conditional_assumptions', candidate['conditional_assumptions'] == model['conditional_assumptions'] == old['formula_policy']['assumptions'])
    check('residual_quality_unchanged', candidate['unchanged_global_formula_policy']['oracle_absolute_residual_quality_hartree'] == old['formula_policy']['oracle_absolute_residual_quality_hartree'] == 1e-8)
    check('candidate_not_execution_contract', candidate['schema'] != old['schema'] and candidate['execution_ready'] is False and candidate['scientific_execution_authorized'] is False)
    check('candidate_not_adopted', candidate['adopted'] is False and candidate['coverage_model_adopted'] is False)
    check('runtime_integration_explicitly_pending', candidate['implementation']['required_new_scientific_source_commit'] is None and model['integration_implemented'] is False)
    report = {'schema': 'm1s_gate0_tail_bound_static_validation_v1', 'PASS': sum(x['status'] == 'PASS' for x in checks), 'FAIL': sum(x['status'] == 'FAIL' for x in checks), 'checks': checks, 'scope': 'AST/JSON, structural integer counts and symbolic dyadic inequalities only', 'NPZ_decodes': 0, 'Pauli_generation': 0, 'real_or_synthetic_input_budget_evaluations': 0, 'W_v_rho_evaluations': 0, 'Gate0_runner_invocations': 0, 'is_formal_proof_assistant': False, 'is_numerical_or_platform_certificate': False, 'mandatory_STOP': True}
    if args.write_report:
        (ROOT / ART / 'static_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, indent=2))
    return 1 if report['FAIL'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
