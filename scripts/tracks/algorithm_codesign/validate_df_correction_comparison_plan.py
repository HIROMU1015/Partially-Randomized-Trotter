"""Validate planning metadata only; no matrix, compiler, synthesis or runner calls."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'artifacts/track_b_df_correction_comparison_preparation/2026-10-11'
CHECKS = []


def check(name, condition):
    CHECKS.append({'name': name, 'pass': bool(condition)})
    if not condition:
        raise AssertionError(name)


def main():
    c = json.loads((ART/'contract_v1.json').read_text())
    a = json.loads((ART/'authorization.json').read_text())
    stop = json.loads((ART/'STOP.json').read_text())
    check('preparation_not_execution', c['science_execution_authorized'] is False and c['runs_authorized_now'] == 0 and c['runner_implemented'] is False)
    check('authorization_pending', a['science_execution_authorized'] is False and a['runs'] == 0 and a['source_commit'] is None)
    check('STOP', stop['mandatory_STOP'] is True and stop['new_execution_marker_created'] is False)
    d = c['CD']
    check('CD_recipe', (d['seed'], d['modes'], d['block_count'], d['L_D']) == (6121107, 3, 3, 2))
    check('CD_coefficients', list(map(F, d['coefficients'])) == [F(9,20), F(7,20), F(1,4)])
    check('CD_ranks_signs', d['primitive_ranks'] == [1,2,1] and d['primitive_signs'] == [1,-1,-1])
    check('time_grid', F(d['t']) == 1 and list(map(F,d['deltas'])) == [F(1,8),F(1,16),F(1,32)])
    check('steps', [F(d['t'])/F(x) for x in d['deltas']] == d['steps'])
    check('gamma_kappa', F(d['gamma']) == F(5,4) and F(d['kappa']) == 4)
    check('finite_RTE', (d['RTE_microsteps'],d['paired_cutoff'],d['mean_degree']) == (1,4,5))
    check('state_explicit', d['initial_state']['indices'] == [1,2,4,3,5,6] and d['initial_state']['amplitudes_before_division_sqrt6'] == ['1','i','1','1','-i','1'])
    check('state_is_proposal', d['initial_state']['proposal_requires_contract_confirmation'] is True)
    check('costs_symbolic', d['cost_policy'] == 'UNKNOWN_SYMBOLIC_UNTIL_SAME_CD_INPUT_COST_AVAILABLE')
    conf = c['confidence']
    check('epsilon', F(conf['complex_epsilon']) == F(1,10000) and F(conf['axis_epsilon']) == F(1,20000))
    check('complex_axis_sufficiency', 2*F(conf['axis_epsilon'])**2 < F(conf['complex_epsilon'])**2)
    check('alpha_families', F(conf['X_family_alpha']) == F(1,40) and F(conf['Y_family_alpha']) == F(1,40) and F(conf['total_alpha']) == F(1,20))
    check('union_bound', conf['tasks'] == 6 and conf['axes'] == 2 and 12*F(conf['axis_task_alpha']) == F(conf['total_alpha']))
    check('shot_execution_zero', conf['actual_quantum_shots_planned'] == 0)
    budget = c['error']['axis_maxima']
    check('equal_common_budget', list(map(F,budget.values())) == [F(1,40000),F(1,200000),F(1,200000),F(1,1000000)])
    check('budget_leaves_statistics', sum(map(F,budget.values())) < F(conf['axis_epsilon']))
    check('new_bounds_not_run', c['error']['new_bounds_computed_now'] is False and c['error']['new_operational_certificates_verified_now'] is False)
    check('digital_fixed', c['digital']['denominator_bits'] == 48 and c['digital']['near_zero_drop'] is False)
    e = c['E']
    check('E_recipe', e['seed'] == 7101107 and e['active_modes'] == [3,6] and e['primitive_ranks'] == [2,2,2] and e['primitive_signs'] == [1,-1,1])
    check('E_angle', F(e['theta']) == F(1,32) and F(e['kappa']) == 4 and F(e['u']) == F(1,128))
    check('domains_separate', e['multiply_with_CD_frequency'] is False and c['T']['reuse_T_angles_across_CD_and_E'] is False)
    t = c['T']
    check('synthesizer_frozen', t['synthesizer'] == 'pygridsynth' and t['version'] == '2.0.0' and t['mpmath'] == '1.3.0' and t['dps'] == 100)
    check('strict_phase', t['options']['up_to_phase'] is False)
    check('gate_precision', F(t['gate_precision']) == F(1,40000000000))
    check('key_cap', t['selection']['maximum_logical_keys'] == 64 and t['synthesis_calls_cap'] == 64 and 2*16+2*8+2*4+8 == 64)
    check('no_adaptive_fill', t['selection']['fill_empty_slots'] is False and t['actual_synthesis_now'] == 0)
    check('coverage_rule', 'T_INCONCLUSIVE_INCOMPLETE_COVERAGE' in t['full_T_policy'] and 'EXCLUDED' in t['RZ_times_average_T'])
    check('occurrence_error_allocation', t['RZ_occurrence_cap_per_circuit']*F(t['gate_precision']) <= F(e['local_synthesis_allowance']))
    check('local_error_allocation', F(e['local_synthesis_allowance'])+F(e['local_numeric_allowance']) == F(e['local_unitary_error_cap']))
    check('resource_limits', c['caps']['native']['compile_calls'] == 22 and c['caps']['synthesis']['keys'] == 64 and c['caps']['retry'] == 0)
    check('parallelism_memory', c['parallelism']['native_workers'] == 2 and 2*c['caps']['native']['worker_RSS_MiB']+256 <= c['caps']['native']['aggregate_RSS_MiB'])
    check('parallelism_synthesis_memory', c['parallelism']['synthesis_workers'] == 2 and 2*c['caps']['synthesis']['worker_RSS_MiB']+256 <= c['caps']['synthesis']['aggregate_RSS_MiB'])
    check('hard_guard_pending', c['caps']['guard_implemented_now'] is False)
    check('no_future_source_auth', c['future_source_commit'] is None and c['future_gate']['marker_created_now'] is False and c['retries'] == 0)
    inp = json.loads((ART/'input_intake.json').read_text())
    for r in inp['inputs']:
        b = (ROOT/r['repository_path']).read_bytes()
        check('input.'+Path(r['repository_path']).name, len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256'])
    with zipfile.ZipFile(ROOT/c['inputs']['zip']) as z:
        check('zip_CRC_and_members', z.testzip() is None and len(z.infolist()) == 9)
        for r in json.loads(z.read('manifest.json'))['files']:
            b = z.read(r['path'])
            check('ZIP_manifest.'+r['path'], len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256'])
    for r in json.loads((ART/'reference_source_manifest.json').read_text())['files']:
        b = (ROOT/r['path']).read_bytes()
        check('unchanged_reference.'+r['path'], hashlib.sha256(b).hexdigest() == r['sha256'] and b == subprocess.check_output(['git','show',r['fixed_commit']+':'+r['path']],cwd=ROOT))
    rt = json.loads((ART/'runtime_identity_preparation.json').read_text())
    check('runtime_inspection_only', rt['synthesis_calls'] == 0 and rt['science_calls'] == 0 and rt['status'] == 'METADATA_SOURCE_IDENTITY_PASS')
    scope = json.loads((ART/'preparation_scope.json').read_text())
    check('all_science_counters_zero', all(v == 0 for v in scope['executed'].values()))
    result = {'kind':'PREREGISTRATION_METADATA_VALIDATION_NOT_SCIENCE_TESTS', 'status':'PASS', 'checks':CHECKS, 'checks_passed':len(CHECKS),
              'contract_sha256':hashlib.sha256((ART/'contract_v1.json').read_bytes()).hexdigest(),
              'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'comparison_matrix_compile_synthesis_calls':0, 'production_tests':0, 'full_suite':False, 'mandatory_STOP':True}
    with (ART/'preparation_validation.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':'PASS','metadata_checks':len(CHECKS),'science_compile_synthesis':0,'mandatory_STOP':True}))


if __name__ == '__main__':
    main()
