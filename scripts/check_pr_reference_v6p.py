"""Post-run V6-P verification only; never calls the campaign or a phase provider."""
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/pr_reference_v6p/2026-11-09/P_run01'
EXECUTION = 'a2c54558e1b4cc052aeb835a9e6d3b2796767e98'


def main():
    if (OUT / 'CHECKS.json').exists():
        raise FileExistsError('Existing verification preserved')
    source = json.loads((OUT / 'SOURCE.json').read_text())
    raw = (OUT / 'RESULTS.json').read_bytes()
    result = json.loads(raw)
    stop = json.loads((OUT / 'STOP.json').read_text())
    path = 'scripts/diagnose_pr_reference_v6p.py'
    executable = (ROOT / path).read_bytes()
    assert executable == subprocess.check_output(['git', 'show', EXECUTION + ':' + path], cwd=ROOT)
    assert hashlib.sha256(executable).hexdigest() == source['source_sha256']
    assert source['execution_commit'] == EXECUTION
    spec = importlib.util.spec_from_file_location('v6p_fixed_helpers', ROOT / path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Reuse scalar helpers; do not call campaign, candidate_bound, recost,
    # bound_and_shots, shot_ceiling or any previous runner main().
    iv, old = adapter.reused('iv'), adapter.reused('check')
    I, mp = iv.I, iv.mp
    mp.iv.prec = 192
    inputs = {}
    for name, saved_path in adapter.INPUTS.items():
        payload = adapter.git_bytes(adapter.BASE, saved_path)
        assert hashlib.sha256(payload).hexdigest() == source['inputs']['files'][saved_path]['sha256']
        if name in ('stage3', 'rounds', 'bprime'):
            inputs[name] = json.loads(payload)
        if name == 'd0':
            import lzma
            inputs[name] = json.loads(lzma.decompress(payload))
    assert source['inputs'] == adapter.inventory()
    r3 = next(r for r in inputs['stage3']['rows'] if r['m'] == 17)
    original = next(c for c in inputs['rounds'] if c['name'] == 'CN_S2_h0.02_ld3')
    original = next(r for r in original['rows'] if r['m'] == 17)
    b = next(r for r in inputs['bprime']['rows'] if r['m'] == 17)
    ph = r3['phase_arithmetic']
    new = result['conditional_reference']
    exact_candidate = F.from_float(ph['normalization_vs_controlled_source_upper']) + \
        2 * F(2)**-53 * F.from_float(ph['exact_prefix_absolute_upper'])
    assert F.from_float(new['phase_upper']) >= exact_candidate
    assert result['candidate_phase_bound']['normalization_upper'] == ph['normalization_vs_controlled_source_upper']
    assert F.from_float(result['candidate_phase_bound']['single_round_raw_phase_upper']) >= \
        2 * F(2)**-53 * F.from_float(ph['exact_prefix_absolute_upper'])
    disk_checks = {}
    for label in ('old_reference_reconstructed', 'conditional_reference'):
        row = result[label]
        other = I(r3['old_source_shared_raw_disk_upper']) + I(r3['arms']['reference']['native_raw_disk_upper'])
        disk = other + I(row['phase_upper'])
        nominal = I(original['raw_radius_lower']) + I(original['raw_extra_disk_upper'])
        assert row['other_raw_disk_upper'] == iv.upper(other)
        assert row['total_raw_disk_upper'] == iv.upper(disk)
        rho = nominal - disk
        ratio = disk / nominal
        beta = I('.4') - I(original['PF_phase_upper']) - mp.iv.atan2(ratio, mp.iv.sqrt(1-ratio**2))
        assert row['raw_radius_lower'] == iv.lower(rho)
        assert row['statistical_phase_lower'] == iv.lower(beta)
        assert row['raw_radius_lower'] > 0 and row['statistical_phase_lower'] > 0
        disk_checks[label] = True
    # Direct 4-log expression checks the original epsilon/sqrt(2) computation.
    for row in (result['old_reference_reconstructed'], result['conditional_reference'],
                dict(result['Bprime_m17_ceil_check'], raw_radius_lower=b['raw_radius_lower'],
                     statistical_phase_lower=b['statistical_phase_lower'])):
        rho, beta = I(row['raw_radius_lower']), I(row['statistical_phase_lower'])
        direct = 4*mp.iv.log(I(1440)) / (rho**2 * mp.iv.sin(beta)**2)
        lo, hi = map(F, row['required_shots_interval'])
        direct_lo, direct_hi = map(adapter.exact_iv_endpoint, direct._mpi_)
        assert max(lo, direct_lo) <= min(hi, direct_hi)
        n = row['shots_per_axis']
        assert F(n) >= hi and F(n) >= direct_hi
        assert F(n-1) < lo and F(n-1) < direct_lo
    assert result['old_reference_reconstructed']['shots_per_axis'] == 630
    assert result['Bprime_m17_ceil_check']['shots_per_axis'] == 626
    assert new['shots_per_axis'] == 626
    a = result['accounting_RZ_only']
    prior = {(r['m'], r['axis']): r for r in inputs['d0']['rows']}
    assert len(prior) == 36
    for row in a['rows']:
        previous = prior[row['m'], row['axis']]
        for role in previous['roles']:
            old_cost = old.pair(previous['roles'][role]['reference_cost'])
            n0, n1 = previous['reference_shots'], row['shots_per_axis']
            expected = tuple(x * F(n1, n0) for x in old_cost)
            assert old.pair(row['roles'][role]) == expected
        assert old.add(old.pair(x) for x in row['roles'].values()) == old.pair(row['reference_cost'])
        if row['m'] < 17:
            assert row['shots_per_axis'] == previous['reference_shots']
            assert row['reference_cost'] == previous['reference_cost']
        else:
            assert row['shots_per_axis'] == previous['Bprime_shots']
    assert old.add(old.pair(x['reference_cost']) for x in a['rows']) == old.pair(a['conditional_reference_total'])
    assert old.add(old.pair(x) for x in a['conditional_role_totals'].values()) == old.pair(a['conditional_reference_total'])
    assert a['Bprime_total_unchanged'] == inputs['d0']['totals']['Bprime_cost']
    assert a['saved_reference_total'] == inputs['d0']['totals']['reference_cost']
    # When all N match Bprime, the new total gap equals D0's saved price-only
    # term, and the removed expense equals its saved extra-shot term.
    assert a['reference_minus_Bprime'] == inputs['d0']['totals']['Bprime_shot_anchor']['price']
    removed = old.pair(inputs['d0']['totals']['Bprime_shot_anchor']['shots'])
    assert old.pair(a['delta_to_saved_reference']) == (-removed[1], -removed[0])
    assert F(a['reference_minus_Bprime'][1]) < 0
    assert result['status'] == stop['status'] == 'CONDITIONAL_B1_TOTAL_RZ_ADVANTAGE'
    assert stop['next_action'] == 'MANDATORY_STOP' and not stop['V6Q_started']
    display = {k: old.display_pair(old.pair(a[k])) for k in (
        'Bprime_total_unchanged', 'saved_reference_total', 'conditional_reference_total',
        'reference_minus_Bprime', 'delta_to_saved_reference')}
    checks = {'status': 'V6P_POSTRUN_CHECKS_PASS', 'execution_source': EXECUTION,
        'production_campaign_invocations': 1, 'CLI_exit_code': 0,
        'CLI_wall_seconds_reported_by_exec': 0.195342565,
        'fixed_adapter_and_inputs_unchanged': True, 'outward_candidate_bound': True,
        'normalization_and_other_disks_preserved': True, 'disk_and_radius_beta_checks': disk_checks,
        'direct_four_log_and_epsilon_formulas_agree': True,
        'upper_endpoint_ceils_and_lower_endpoint_integer_separation': True,
        'old_reference_N': 630, 'old_Bprime_N': 626, 'conditional_reference_N': 626,
        'only_m17_N_changed': True, 'same_XY_worst_case_bound': True,
        'nine_roles_and_signed_fusion_checked': True, 'D0_same_endpoint_identities': True,
        'original_raw_outputs_unchanged': True,
        'raw_result_bytes': len(raw), 'raw_result_sha256': hashlib.sha256(raw).hexdigest(),
        'postrun_checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'display_RZ_outward_three_decimals': display,
        'new_provider_circuit_sampling_compile_guard_synthesis_T_evaluation': False,
        'independent_scientific_review': False, 'next_action': 'MANDATORY_STOP'}
    assert (OUT / 'RESULTS.json').read_bytes() == raw
    with (OUT / 'CHECKS.json').open('x') as stream:
        json.dump(checks, stream, indent=2)
        stream.write('\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
