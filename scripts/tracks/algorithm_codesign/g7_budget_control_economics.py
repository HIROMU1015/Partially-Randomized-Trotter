"""G7 fixed delegated implementation-economics acquisition; one shot then STOP."""
import argparse
from dataclasses import asdict, is_dataclass
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.g7_generator import make_generator, budget, DeterministicBits
from trottertracks.algorithm_codesign.g7_reference import reference_events, resource_reference
from trottertracks.algorithm_codesign.g7_provider import provider_contract, bind_synthesized_provider_ir
from trottertracks.algorithm_codesign.g7_launch import sha, git, verify_source, protected_check, consume_marker
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure, synthesize, validate_saved
from trottertracks.algorithm_codesign.rte_reallocation.launch import BudgetGuard
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime

PREP = 'artifacts/track_b_g7_budget_control_preparation/2026-10-10'


def plain(value):
    if isinstance(value, F): return str(value)
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def generators(contract):
    def build(spec, arm):
        wall, cpu = time.monotonic(), time.process_time()
        gen = make_generator(tuple(map(F, spec['p'])), F(spec['x']), spec['m'], arm,
                root_bits=contract['root_bits'], probability_bits=contract['probability_bits'],
                eta=F(contract['eta']), rho=F(contract['rho']))
        gen.construction_resource = {'wall_seconds': time.monotonic() - wall,
                                     'CPU_seconds': time.process_time() - cpu}
        return spec['id'], gen
    return [build(spec, arm) for spec in contract['inputs'] for arm in contract['arms']]


def inventory(gens):
    keys, rows = set(), []
    for name, gen in gens:
        events = list(reference_events(gen)); angles = {e['ratio'] for e in events}; keys |= angles
        rows.append({'input': name, 'arm': gen.arm, 'reference_events': len(events),
                     'positive_tangent_keys': list(map(str, sorted(angles)))})
    return {'positive_tangent_keys': list(map(str, sorted(keys))), 'key_count': len(keys), 'rows': rows,
            'negative_primitives': 'actual adjoint of the one acquired positive sequence; no resynthesis',
            'enumeration_role': 'small independent accounting/key reference, never production input'}


def totals(plan, ref):
    # N attempts on each axis; zero trials make no quantum calls/readout.
    N, Z = plan['N_per_axis'], ref['digital_acceptance']
    return {'T_Rz': 2 * N * ref['per_trial_fixed_cost']['T_Rz'],
            'T_provider_coefficients': [2 * N * q for q in ref['per_trial_provider_calls']],
            'CX_fixed': 2 * N * ref['per_trial_fixed_cost']['CX_fixed'],
            'CX_provider_coefficients': [2 * N * q for q in ref['per_trial_provider_calls']],
            '1Q_fixed_including_Re_Im_readout': 2 * N * ref['per_trial_fixed_cost']['1Q_fixed_no_readout'] + 5 * N * Z,
            '1Q_provider_coefficients': [2 * N * q for q in ref['per_trial_provider_calls']],
            'expected_quantum_calls_two_axes': 2 * N * Z,
            'quantum_call_hard_cap_two_axes': 2 * N,
            'workspace_beyond_system': 2,
            'state_preparation_cost_per_accepted_trial': 'common unspecified conditional provider/preparation cost; add 2*N*Z times it'}


def contrasts(rows):
    result = []
    for f in (r for r in rows if r['arm'] == 'full_return'):
        for b in (r for r in rows if r['input'] == f['input'] and r['arm'] != 'full_return'):
            ft, bt = f['total_conditional_resource'], b['total_conditional_resource']
            delta = ft['T_Rz'] - bt['T_Rz']
            slopes = [x - y for x, y in zip(ft['T_provider_coefficients'], bt['T_provider_coefficients'])]
            nz = ft['expected_quantum_calls_two_axes'] - bt['expected_quantum_calls_two_axes']
            result.append({'input': f['input'], 'baseline': b['arm'], 'full_minus_baseline_T_Rz': delta,
                           'full_minus_baseline_T_provider_coefficients': slopes,
                           'full_minus_baseline_expected_preparation_calls': nz,
                           'sign_rule': 'full improves conditional T iff intercept + dot(slopes,T_Q) + call_delta*T_prep < 0',
                           'provider_independent_sufficient_improvement': delta < 0 and all(x <= 0 for x in slopes) and nz <= 0,
                           'provider_independent_no_improvement': delta >= 0 and all(x >= 0 for x in slopes) and nz >= 0,
                           'full_minus_baseline_fixed_CX': ft['CX_fixed'] - bt['CX_fixed'],
                           'full_minus_baseline_fixed_1Q': ft['1Q_fixed_including_Re_Im_readout'] - bt['1Q_fixed_including_Re_Im_readout']})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    contract_path = ROOT / PREP / 'contract_v1.json'
    contract = json.loads(contract_path.read_text())
    before = verify_source(ROOT, args.source_commit, contract)
    runtime = verify_runtime(ROOT, contract)
    if sha(Path(sys.executable).resolve()) != contract['runtime_executable_sha256']:
        raise PermissionError('fixed interpreter changed')
    out = ROOT / contract['result_directory']
    marker = consume_marker(out, {'kind': 'G7_DELEGATED_LIMITED_IMPLEMENTATION_ECONOMICS',
        'source_commit': args.source_commit, 'contract_sha256': sha(contract_path),
        'authorization': contract['authorization'], 'runs': 1, 'retries': 0, 'mandatory_STOP': True})
    cache, rows, reason, calls = {}, [], None, 0
    status = 'G7_TECHNICAL_INCONCLUSIVE'
    guard = BudgetGuard(contract['caps'])
    try:
        with guard:
            gens = generators(contract)
            if inventory(gens) != json.loads((ROOT / contract['inventory_path']).read_text()):
                raise PermissionError('fixed symbolic key inventory changed')
            plans = [budget(g, F(contract['epsilon_axis']), F(contract['primitive_error'])) for _, g in gens]
            if any(p['N_per_axis'] > contract['caps']['shot_cap_per_axis'] for p in plans):
                raise RuntimeError('registered shot budget cap exceeded')
            configure(contract['interval_dps'])
            def on_call():
                nonlocal calls
                calls += 1
                if calls > contract['caps']['synthesis_keys']:
                    raise RuntimeError('synthesis key cap exceeded; no retry')
            for tangent in json.loads((ROOT / contract['inventory_path']).read_text())['positive_tangent_keys']:
                guard.begin_key()
                angle = Angle('atan', F(tangent))
                row = synthesize(angle, contract['primitive_error'], contract['synthesizer_options'],
                                  on_synthesis=on_call, max_characters=contract['caps']['sequence_characters'])
                row['acquisition_resource'] = guard.end_key()
                cache[tangent] = row
                validate_saved(row, angle, contract['primitive_error'])
                print(json.dumps({'acquired_key': tangent, 'T_count': row['T_count'], 'strict_error_pass': True}), flush=True)
            for (name, gen), plan in zip(gens, plans):
                guard.check()
                bits = DeterministicBits(contract['diagnostic_bitstream_seed'])
                w0, c0 = time.monotonic(), time.process_time()
                trace = [gen.sample(bits) for _ in range(contract['diagnostic_trials_per_row'])]
                digest = hashlib.sha256(json.dumps(plain(trace), sort_keys=True).encode()).hexdigest()
                diagnostic = {'trials': len(trace), 'zero_trials': sum(e is None for e in trace),
                    'bit_requests': bits.counter, 'consumed_bits': bits.consumed_bits, 'trace_sha256': digest,
                    'wall_seconds': time.monotonic() - w0, 'CPU_seconds': time.process_time() - c0,
                    'used_for_confidence_or_cost_estimation': False, 'quantum_measurements': 0}
                wr, cr = time.monotonic(), time.process_time()
                events = list(reference_events(gen))
                ref = resource_reference(gen, events, cache)
                if ref['digital_weight_second_moment'] > plan['m2_upper'] or max(e['weight'] for e in events) > plan['range_upper']:
                    raise ArithmeticError('budget bound violated by separate exact reference')
                reference_resource = {'wall_seconds': time.monotonic() - wr, 'CPU_seconds': time.process_time() - cr}
                native_description = bind_synthesized_provider_ir(events[0], cache)
                description_digest = hashlib.sha256(json.dumps(native_description, sort_keys=True).encode()).hexdigest()
                row_keys = {str(e['ratio']) for e in events}
                cold = {k: sum(cache[q]['acquisition_resource'][k] for q in row_keys)
                        for k in ('wall_seconds', 'cpu_seconds')}
                rows.append({'input': name, 'arm': gen.arm, 'p': gen.p, 'x': gen.x, 'm': gen.m,
                    'budget': plan, 'production_interface_diagnostic': diagnostic,
                    'production_construction_resource': gen.construction_resource,
                    'cold_unique_angle_acquisition_resource_if_this_arm_alone': cold,
                    'conditional_native_description_example': native_description,
                    'conditional_native_description_sha256': description_digest,
                    'small_support_reference': ref, 'reference_enumeration_resource': reference_resource,
                    'total_conditional_resource': totals(plan, ref)})
                guard.check()
            if calls != contract['caps']['synthesis_keys'] or len(rows) != 8:
                raise RuntimeError('incomplete registered acquisition')
            status = 'G7_LIMITED_IMPLEMENTATION_ECONOMICS_COMPLETE_AWAITING_GPT_REVIEW'
    except Exception as error:
        reason = type(error).__name__ + ': ' + str(error)
    after = protected_check(ROOT, contract)
    if after['violations']:
        status, reason = 'G7_TECHNICAL_INCONCLUSIVE', 'protected history changed'
    result = {'kind': 'G7_KNOWN_DEVELOPMENT_CONDITIONAL_QUERY_NATIVE_RZ_CPU_VECTOR',
        'status': status, 'technical_reason': reason, 'source_commit': args.source_commit,
        'contract_sha256': sha(contract_path), 'marker_sha256': sha(marker), 'runtime': runtime,
        'provider_contract': provider_contract(), 'synthesis_calls': calls, 'completed_keys': len(cache),
        'synthesis_cache': cache, 'rows': rows,
        'conditional_T_contrasts': contrasts(rows) if status != 'G7_TECHNICAL_INCONCLUSIVE' else [],
        'primary_cost': 'synthesized Rz T + per-label controlled-Q T + common preparation T; no molecular total',
        'resource': guard.usage(), 'runs': 1, 'retries': 0, 'protected_before': before, 'protected_after': after,
        'new_inputs_LP_DF_molecule_NPZ_GPU_quantum_measurements_trajectories': 0,
        'native_whole_circuit_build_or_compile': 0, 'exact_signal_evaluations': 0,
        'synthetic_matrix_checks': 'focused pre-freeze semantic tests only',
        'all_methods_same_fixed_error_and_shot_rule': True, 'mandatory_STOP': True,
        'novelty_established': False, 'next_science_authorized': False,
        'partial_rows_valid_for_final_decision': status != 'G7_TECHNICAL_INCONCLUSIVE'}
    payload = (json.dumps(plain(result), indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if len(payload) > contract['caps']['output_bytes'] - 16384:
        raise RuntimeError('output cap; marker remains consumed; no retry')
    with (out / 'result_v1.json').open('xb') as stream: stream.write(payload)
    with (out / 'STOP.json').open('x') as stream:
        json.dump({'mandatory_STOP': True, 'status': status, 'next_science_authorized': False,
                   'research_owner': 'GPT / user', 'retries': 0}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': status, 'keys': len(cache), 'rows': len(rows), 'STOP': True}))


if __name__ == '__main__': main()
