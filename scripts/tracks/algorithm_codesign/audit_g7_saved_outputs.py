"""Metadata/rational-arithmetic audit ONLY; no generator, matrix or synthesis."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT / 'artifacts/track_b_g7_budget_control_preparation/2026-10-10'
OUT = ROOT / 'artifacts/track_b_g7_budget_control_result/2026-10-10/v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    started, cpu = time.monotonic(), time.process_time()
    result_path = OUT / 'result_v1.json'
    result_digest = sha(result_path)
    result = json.loads(result_path.read_text())
    contract = json.loads((PREP / 'contract_v1.json').read_text())
    manifest = json.loads((PREP / 'source_manifest_v1.json').read_text())
    inventory = json.loads((PREP / 'synthesis_key_inventory_v1.json').read_text())
    checks = {}
    checks['source_hashes_unchanged'] = all(sha(ROOT / p) == v for p, v in manifest['sha256'].items())
    ledger = json.loads((PREP / 'prior_protected_hashes.json').read_text())
    violations = []
    for path, row in ledger.items():
        if path.lower().endswith('.npz'): raise PermissionError('NPZ access forbidden')
        data = (ROOT / path).read_bytes()
        if path in contract['append_only_paths']: data = data[:row['bytes']]
        if hashlib.sha256(data).hexdigest() != row['sha256']: violations.append(path)
    checks['protected_hashes_unchanged'] = not violations
    checks['marker_and_contract_identity'] = (sha(OUT / 'one_shot_consumed.json') == result['marker_sha256']
        and sha(PREP / 'contract_v1.json') == result['contract_sha256'])
    checks['one_shot_complete'] = (result['runs'] == 1 and result['retries'] == 0
        and result['synthesis_calls'] == 24 and result['completed_keys'] == 24 and len(result['rows']) == 8
        and result['mandatory_STOP'] is True and result['next_science_authorized'] is False)
    cache = result['synthesis_cache']
    checks['only_registered_keys'] = set(cache) == set(inventory['positive_tangent_keys'])
    checks['sequence_count_identity_and_strict_error'] = all(
        row['sequence_sha256'] == hashlib.sha256(row['sequence'].encode()).hexdigest()
        and row['T_count'] == row['sequence'].count('T') + row['sequence'].count('t')
        and row['Tdagger_count'] == row['sequence'].count('t')
        and row['one_qubit_count'] == len(row['sequence']) - row['sequence'].count('W')
        and row['global_W_count'] == row['sequence'].count('W')
        and row['angle_key'] == 'atan:' + key + ':scale:1'
        and row['epsilon'] == contract['primitive_error'] and row['error_pass'] is True
        and 0 <= F(row['strict_operator_error_upper']) <= F(contract['primitive_error'])
        for key, row in cache.items())
    checks['common_margin_and_familywise'] = len({r['budget']['remaining'] for r in result['rows']}) == 1
    checks['common_margin_and_familywise'] &= all(16 * F(r['budget']['alpha_axis']) == F(1, 20)
        and not r['budget']['uses_exact_signal'] and not r['budget']['uses_reference_B_new_or_m2'] for r in result['rows'])
    checks['confidence_N_and_total_resources'] = True
    # Independently recalculate totals from SAVED sufficient N/per-trial costs.
    for row in result['rows']:
        b, ref, total = row['budget'], row['small_support_reference'], row['total_conditional_resource']
        N, z = b['N_per_axis'], F(ref['digital_acceptance'])
        checks['confidence_N_and_total_resources'] &= (
            F(ref['digital_weight_second_moment']) <= F(b['m2_upper'])
            and F(total['T_Rz']) == 2 * N * F(ref['per_trial_fixed_cost']['T_Rz'])
            and list(map(F, total['T_provider_coefficients'])) == [2 * N * F(v) for v in ref['per_trial_provider_calls']]
            and F(total['expected_quantum_calls_two_axes']) == 2 * N * z
            and total['quantum_call_hard_cap_two_axes'] == 2 * N
            and F(total['1Q_fixed_including_Re_Im_readout']) == 2 * N * F(ref['per_trial_fixed_cost']['1Q_fixed_no_readout']) + 5 * N * z)
    checks['caps_and_no_retry'] = (result['resource']['wall_seconds'] < contract['caps']['wall_seconds']
        and result['resource']['cpu_seconds'] < contract['caps']['cpu_seconds']
        and result['resource']['peak_RSS_KiB'] <= contract['caps']['RSS_MiB'] * 1024
        and all(v['acquisition_resource']['wall_seconds'] < contract['caps']['per_key_wall_seconds']
                and v['acquisition_resource']['cpu_seconds'] < contract['caps']['per_key_cpu_seconds'] for v in cache.values()))
    checks['science_boundary'] = (result['new_inputs_LP_DF_molecule_NPZ_GPU_quantum_measurements_trajectories'] == 0
        and result['native_whole_circuit_build_or_compile'] == 0 and result['exact_signal_evaluations'] == 0)
    checks['raw_result_and_marker_unchanged'] = result_digest == sha(result_path) and sha(OUT / 'one_shot_consumed.json') == result['marker_sha256']
    report = {'kind': 'G7_SAVED_METADATA_AND_RATIONAL_REAGGREGATION_ONLY', 'passed': all(checks.values()),
        'checks': checks, 'protected_paths': len(ledger), 'protected_violations': violations,
        'source_commit': result['source_commit'], 'raw_result_sha256': result_digest,
        'marker_sha256': result['marker_sha256'], 'contract_sha256': result['contract_sha256'],
        'additional_synthesis_matrix_generator_quantum_calls': 0,
        'wall_seconds': time.monotonic() - started, 'CPU_seconds': time.process_time() - cpu,
        'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'mandatory_STOP': True}
    with (OUT / 'saved_output_audit.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks), 'protected_paths': len(ledger)}))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
