"""Pinned SOURCE/selection checks only. No scientific modules or result arithmetic.

Run from the repository root with Python. Reads Git blobs, not private inputs.
The emitted JSON is a plan/technical observation, never a compiled price record.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

D = '01e068af7e06d9f1f5621a5b3800c30eb32ecb06'
C = 'd7c09ef2e2521bb1815d1d11970529fc7030c2e3'
A = '3ae055fa7a696b13320e00ec79a5ba15c4ed182e'
A_FOLLOWUP = 'd48d8732ee7ef521aab5d67a0945439364f5a2f8'
B = 'ce0517de6e7b58cf4bf4c69f99f998d86099c406'
LIGHT = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'
T20 = '7a8bc7f132e166390c943e6b32c92cc6abe1baf4'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
FULLH_SIGNAL = '367dc98d832ecb4d769f2a3da3d174bc697331a5'
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
OUT = Path(__file__).resolve().parent


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path]).decode()


def node(text, name):
    return next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def check_source():
    facts = {}
    old = blob(PROVIDER, 'src/trotterlib/df_rte_qiskit.py')
    rotation = ast.unparse(node(old, '_append_central_application'))
    facts['legacy_controlled_ZZ_uses_RZZGate_then_control1'] = (
        'RZZGate(2.0 * angle)' in rotation and 'gate.control(1)' in rotation)
    new = blob(D, 'src/trottertracks/efficient_controlled_zz.py')
    method = node(new, '_append_central_application')
    branch = next(n for n in method.body if isinstance(n, ast.If))
    facts['new_branch_is_controlled_rotation_support2_only'] = all(
        s in ast.unparse(branch.test) for s in
        ('controlled', "application.role == 'rotation'", 'len(support) == 2'))
    gates = [ast.unparse(n.value) for n in branch.body if isinstance(n, ast.Expr)]
    facts['new_ZZ_order_wires_angle'] = gates == [
        'circuit.cx(s0, s1)',
        'circuit.append(CRZGate(2.0 * event.unsigned_rotation_angle), [ancilla_qubit, s1])',
        'circuit.cx(s0, s1)']
    facts['other_cases_delegate_and_swap_is_restored'] = (
        'return legacy(' in ast.unparse(method)
        and 'finally:' in ast.unparse(node(new, 'efficient_controlled_zz'))
        and 'module.QiskitDFRTEEventCircuitBuilder = original' in new)
    full = blob(LIGHT, 'scripts/h46_full_h_pure_rte.py')
    t20 = blob(T20, 'scripts/h46_full_h_t20_opt.py')
    facts['old_fullH_calls_native_cost_trial_without_new_lowering_activation'] = (
        'native.cost_trial(job)' in ast.unparse(node(full, 'cost_job'))
        and 'native.cost_trial(job)' in ast.unparse(node(t20, 'worker'))
        and all(s not in full + t20 for s in
                ('efficient_controlled_zz', 'CONTROLLED_ZZ_CX_CRZ_CX_V1')))
    prep = ast.unparse(node(full, 'prepare_full_h'))
    facts['fullH_one_body_randomized_head_empty_own_scalar_rule'] = all(
        s in prep for s in ('tail.components + extra', 'math.fsum(one.diagonal_eigenvalues) / 2',
                            'math.fsum((tail.deterministic_identity_coefficient, one_scalar))',
                            'deterministic_blocks=()'))
    facts['light_prepare_SOURCE_unchanged_since_execution'] = (
        ast.dump(node(full, 'prepare_full_h')) == ast.dump(node(blob(
            'ac33da5f0cb038a780a447746a38e478bb0cec61',
            'scripts/h46_full_h_pure_rte.py'), 'prepare_full_h')))
    native = blob(CARRIER, 'scripts/h46_long_time_native.py')
    facts['PF_branch_has_no_RTE_builder_and_scalar_once'] = (
        'if requests else None' in ast.unparse(node(native, 'compose'))
        and 'build_deterministic_native' in ast.unparse(node(native, 'compose'))
        and "qc.p(-scalar * row['T'], n)" in ast.unparse(node(native, 'compose')))
    cost = ast.unparse(node(native, 'cost_trial'))
    facts['full_measured_XY_and_original_representative_checks'] = all(
        s in cost for s in ("('cosine', 'sine')", 'include_measurement=True',
                           "'low_control'", "'independent_occupation_vector'", "'compiled_vector_'"))
    facts['catalog_compose_and_cost_functions_same_as_carrier'] = all(
        ast.dump(node(native, name)) == ast.dump(node(blob(
            'af284d853890155bacd19e61850c196bd1d5c660',
            'scripts/h46_long_time_native.py'), name)) for name in ('compose', 'cost_trial'))
    assert all(facts.values()), facts
    return facts, full


def main():
    facts, full = check_source()
    packet = json.loads(blob(A, 'docs/research/h4_full_pf_heff_scan_results_20261114/C_HANDOFF.json'))
    cells = []
    # Select A's existing physical rows; do not regenerate candidates or coverage.
    for code, i, m, role in [('P-S2', 0, 5, 'HOLDOUT_INPUT_AFTER_C_MODEL_FREEZE'),
                             ('P-M5-CAL', 2, 4, 'NEW_CALIBRATION_INPUT'),
                             ('P-Y4', 1, 5, 'HOLDOUT_INPUT_AFTER_C_MODEL_FREEZE'),
                             ('P-M5', 2, 5, 'HOLDOUT_INPUT_AFTER_C_MODEL_FREEZE')]:
        item = packet['shortlist'][i]; row = item['rounds'][m]
        cells.append(dict(cell=code, configuration_id=item['configuration_id'], m=m,
            physical=row['physical_candidate'], canonical_candidate_id=row['canonical_candidate_id'],
            source=dict(commit=A, path='docs/research/h4_full_pf_heff_scan_results_20261114/C_HANDOFF.json',
                        shortlist_index=i, round=m),
            role=role, axes=['X', 'Y'], deterministic_replicas_per_axis=1,
            normal_paths=0, cohort='D_S1V1_' + code.replace('-', '_') + '_V1', index=0,
            RNG_seeds=[], seed_transpiler=17, new_compile_maximum=2,
            lowering=dict(rte_rotation_applicability='NOT_APPLICABLE_NO_RTE',
                deterministic_native_policy='M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1',
                source=dict(commit='e7dca5f7301c8633b17a59b6a7161331303aef2d',
                    path='docs/research/d_full_pf_full_h_v2_final_receipts_20261113/HANDOFF_C.md')),
            source_for_scientific_execution=None, approved=False))
    plan_path = 'docs/research/truncated_fullh_signal_results_20261114/run01/FULL_H_K2_QDYADIC_R448/PLAN.json'
    plan = json.loads(blob(FULLH_SIGNAL, plan_path))
    row = plan['requests'][0]['physical_candidate']
    assert (row['T'], row['q'], row['R_outer'], row['K']) == (.1, 1, 448, 2)
    # Original Full-H seed function only: no setup/context/sample/cost_job calls.
    funcs = ast.Module(body=[node(full, 'digest'), node(full, 'trial_seed')], type_ignores=[])
    env = dict(hashlib=hashlib, json=json)
    exec(compile(funcs, LIGHT + ':seed_functions_only', 'exec'), env)
    cohort = 'D_S1_FULL_H_NEW_ZZ_M0_V1'
    seeds = [env['trial_seed'](row, cohort, 0, 0, 0)]
    profile = plan['configuration']['spec']['profile']
    assert 'lowering_id' not in profile
    cells.append(dict(cell='F-H', m=0, configuration_id=plan['configuration']['id'], physical=row,
        source=dict(commit=FULLH_SIGNAL, path=plan_path, request_index=0),
        role='ONE_FRESH_NORMAL_PATH_TECHNICAL_EVIDENCE_NOT_LAW_MEAN_OR_MODEL_HOLDOUT',
        axes=['X', 'Y'], normal_paths=1, cohort=cohort, index=0, RNG_seeds=seeds,
        seed_source=dict(commit=LIGHT, path='scripts/h46_full_h_pure_rte.py', function='trial_seed'),
        seed_transpiler=17, new_compile_maximum=2,
        lowering=dict(rte_rotation_applicability='REQUIRED', lowering_id='CONTROLLED_ZZ_CX_CRZ_CX_V1',
            source=dict(commit=D, path='src/trottertracks/efficient_controlled_zz.py')),
        source_for_scientific_execution=None, approved=False))
    for c in cells:
        c.update(proposed_public='docs/research/d_s1_v1_price_execution_20261114/' + c['cell'] + '/run01',
                 proposed_private='/home/AbeHiromu/projects/d-s1-v1-price-20261114-private/' + c['cell'] + '/run01')
        assert not Path(c['proposed_public']).exists() and not Path(c['proposed_private']).exists()
    result = dict(status='STATIC_SOURCE_AND_SELECTION_CHECKS_ONLY_NOT_SCIENTIFIC_EXECUTION',
        sources=dict(D=D, C=C, A=A, A_latest_unexecuted_followup=A_FOLLOWUP, B=B,
                     audit='51d324ea9c83bc12f1e866da0fa409cef926f74a'),
        source_checks=facts, current_C_fullH_profile_omits_lowering_id=True,
        duplicate_receipt_audit=False, price_values_read_or_recalculated=False,
        new_preparation_sampling_build_compile_fit_G=0, mandatory_stop=True)
    (OUT/'SOURCE_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT/'REPRESENTATIVE_CELLS.json').write_text(json.dumps(dict(
        status='UNAPPROVED_ALTERNATIVES_NOT_A_COMBINED_EXECUTION_MENU', cells=cells,
        C_all_PF_option_max_compile=8, P_S2_is_subset_of_C_all_PF_option=True,
        default_m12_or_PR_extension=False, mandatory_stop=True), indent=2) + '\n')
    print('PASS: 10 pinned SOURCE assertions; A/C physical selectors copied; Full-H seed=' + str(seeds[0]))
    print('No scientific modules imported; no price values read; science calls=0. No comparison guard implemented here.')


if __name__ == '__main__':
    main()
