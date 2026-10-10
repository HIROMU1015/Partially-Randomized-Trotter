"""Future G10 v3. Preparation only; no execution without separate authority."""
import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path
from trottertracks.algorithm_codesign.g10_launch import verify_launch
from trottertracks.algorithm_codesign.g7_launch import consume_marker, sha
from trottertracks.algorithm_codesign.g10_io import (
    IOBudgetGuard, OutputSession, protected_check_streaming as protected_check,
)

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT/'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3'


def collect(c, result, guard, io):
    from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure, synthesize, validate_saved
    from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
    from trottertracks.algorithm_codesign.g7_generator import DeterministicBits
    from trottertracks.algorithm_codesign.g10_generator import generators, arm_names
    from trottertracks.algorithm_codesign.g10_reference import reference_events, cts_events
    from trottertracks.algorithm_codesign.g10_comparison import plan, row, rebudget_anchor
    from trottertracks.algorithm_codesign.g10_saved import affine_policy_lower
    p, x = tuple(map(F, c['input']['p'])), F(c['input']['x'])
    def start_clock():
        return time.monotonic(), time.process_time()

    def record_stage(kind, start, **context):
        result['classical_accounting'].append({
            'stage': kind, 'wall_seconds': time.monotonic()-start[0],
            'CPU_seconds': time.process_time()-start[1], **context})
        io.snapshot(kind)

    configure(c['interval_dps'])
    old_raw = (ROOT/c['reuse_G9_result']).read_bytes()
    if hashlib.sha256(old_raw).hexdigest() != c['reuse_G9_sha256']:
        raise PermissionError('saved G9 anchor identity changed')
    old = json.loads(old_raw)
    del old_raw
    io.snapshot('saved_G9_decoded_raw_released')
    if old['status'] != 'G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE':
        raise PermissionError('G9 anchor incomplete')
    pending = []
    for m in c['new_degrees']:
        start = start_clock()
        gs = generators(p, x, m)
        record_stage('production_precompute_and_group_laws', start, m=m,
                     arms=[g.arm for g in gs], reference_table_inputs=False)
        # Plans precede event traversal and do not receive any event table.
        bs = [plan(c, m, g=g) for g in gs]
        for g in gs:
            bits = DeterministicBits(c['interface_bitstream_seed']+':'+str(m)+':'+g.arm)
            start = start_clock()
            accepted, zeros = 0, 0
            max_coefficient_bits = max_weight_bits = 0
            keys = set()
            for _ in range(c['interface_trials_per_arm']):
                guard.check()
                event = g.sample(bits)
                if event is None:
                    zeros += 1
                else:
                    accepted += 1
                    keys.add(str(event['ratio']))
                    a, w = event['coefficient'], event['weight']
                    max_coefficient_bits = max(max_coefficient_bits,
                        a.numerator.bit_length(), a.denominator.bit_length())
                    max_weight_bits = max(max_weight_bits,
                        w.numerator.bit_length(), w.denominator.bit_length())
            result['production_interface_traces'].append({
                'm': m, 'arm': g.arm, 'fixed_trials': accepted+zeros,
                'accepted': accepted, 'pre_quantum_zero': zeros,
                'queried_angle_tangents': sorted(keys), 'consumed_bits': bits.consumed_bits,
                'wall_seconds': time.monotonic()-start[0],
                'CPU_seconds': time.process_time()-start[1],
                'root_bits': c['root_bits'], 'probability_bits_per_draw': c['probability_bits'],
                'largest_coefficient_integer_bits_diagnostic': max_coefficient_bits,
                'largest_weight_integer_bits_diagnostic': max_weight_bits,
                'table_normalizer_native_cost_signal_inputs': False,
                'statistical_or_scaling_inference': False})
        start = start_clock()
        ce, cert = cts_events(p, x, m)
        record_stage('reference_Pauli_acquisition_and_CTS_construction', start, m=m,
                     events=len(ce), explicit_cheap_I1=True)
        result['CTS_certificates'][str(m)] = cert
        for g, b in zip(gs, bs):
            start = start_clock()
            es = list(reference_events(g))
            record_stage('small_reference_event_traversal', start, m=m, arm=g.arm,
                         events=len(es), production_reads_reference=False)
            pending.append((m, g.arm, es, b))
        pending.append((m, 'matched_CTS', ce, plan(c, m, cts=ce)))
        del gs, bs, ce, cert, g, b, es, bits, event, keys
    needed = {str(e['ratio']) for _, _, es, _ in pending for e in es if e['ratio']}
    # Shared saved cache includes every frozen P5 anchor sequence, even if
    # one is unused in a new degree. No old wrapper cost is transferred.
    needed |= set(old['synthesis_cache'])
    if len(needed) > c['cache_entries']:
        raise RuntimeError('static key/cache cap')
    for key in sorted(needed, key=F):
        guard.check()
        start = start_clock()
        angle = Angle('atan', F(key))
        if key in old['synthesis_cache']:
            saved = old['synthesis_cache'][key]
            validate_saved(saved, angle, F(c['primitive_error']))
            result['synthesis_cache'][key] = saved
            result['reused_keys'] += 1
            acquisition = 'reuse_G9_fixed_identity'
        else:
            if result['new_synthesis_calls'] >= c['caps']['synthesis_keys']:
                raise RuntimeError('registered new synthesis cap')
            guard.begin_key()
            result['new_synthesis_calls'] += 1
            saved = synthesize(angle, c['primitive_error'], c['synthesizer_options'],
                               max_characters=c['caps']['sequence_characters'])
            saved['resource'] = guard.end_key()
            if not saved['error_pass']:
                raise ArithmeticError('strict new Rz guard failed')
            result['synthesis_cache'][key] = saved
            acquisition = 'new_G10_once'
        result['inventory'].append({'ratio': key, 'acquisition': acquisition,
                                    'sequence_sha256': saved['sequence_sha256'],
                                    'acquisition_wall_seconds': time.monotonic()-start[0],
                                    'acquisition_CPU_seconds': time.process_time()-start[1],
                                    'strict_error_pass': saved['error_pass']})
        cache_bytes = len(json.dumps(result['synthesis_cache']).encode())
        if cache_bytes > c['cache_bytes']:
            raise RuntimeError('registered cache byte cap')
    result['synthesis_cache_bytes'] = cache_bytes
    start = start_clock()
    result['rows'] = [rebudget_anchor(r, c) for r in old['rows'] if r['primary']]
    del old
    record_stage('saved_m5_policy_rebudget_only', start, rows=len(result['rows']))
    for m, arm, events, b in pending:
        guard.check()
        io.snapshot('before_reference_row:'+str(m)+':'+arm)
        start = start_clock()
        result['rows'].append(row(m, arm, events, b, result['synthesis_cache'], guard))
        record_stage('small_reference_matrix_and_native_accounting', start, m=m, arm=arm,
                     events=len(events), production_scaling_evidence=False)
    pending.clear()
    del pending, m, arm, events, b, saved, angle, needed
    io.snapshot('all_rows_retained_intermediates_released')
    expected = {(m, a) for m in c['degrees'] for a in arm_names(m)}
    got = {(r['degree'], r['arm']) for r in result['rows']}
    if got != expected or len(result['rows']) != c['rows']:
        raise ArithmeticError('complete registered row set mismatch')
    if sum(len(r['events']) for r in result['rows']) > c['caps']['event_bindings']:
        raise RuntimeError('reference event cap')
    for r in result['rows']:
        r['fixed_dictionary_policy_lower'] = affine_policy_lower(
            r['events'], r['budget']['remaining'], r['budget']['alpha_axis'])
    del r
    io.snapshot('before_final_protected_check')
    result['protected_after'] = protected_check(ROOT, c)
    if result['protected_after']['violations']:
        raise PermissionError('protected history changed')


def execute(source):
    c, auth, head, before = verify_launch(ROOT, PREP/'contract_v3.json', source)
    # Importing this runner or rejecting pending authority never opens science data.
    from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime
    runtime = verify_runtime(ROOT, c)
    if sha(Path(sys.executable).resolve()) != c['runtime_executable_sha256']:
        raise PermissionError('fixed executable identity mismatch')
    marker = consume_marker(ROOT/c['result_directory'], {
        'kind': 'G10_DEGREE_MATCHED_NATIVE_V3_ONE_SHOT', 'source_commit': source,
        'execution_HEAD': head, 'contract_sha256': sha(PREP/'contract_v3.json'),
        'authorization_sha256': sha(ROOT/c['authorization_path']), 'authorization': auth,
        'runs': 1, 'retries': 0, 'mandatory_STOP': True})
    result = {'status': 'G10_TECHNICAL_INCONCLUSIVE', 'technical_reason': None,
              'source_commit': source, 'execution_HEAD': head, 'authorization_commit': head,
              'contract_sha256': sha(PREP/'contract_v3.json'),
              'authorization_sha256': sha(ROOT/c['authorization_path']), 'marker_sha256': sha(marker),
              'runtime': runtime, 'rows': [], 'synthesis_cache': {}, 'inventory': [],
              'runs': 1, 'retries': 0, 'new_synthesis_calls': 0, 'reused_keys': 0,
              'production_interface_traces': [], 'CTS_certificates': {},
              'classical_accounting': [],
              'protected_before': before, 'mandatory_STOP': True, 'next_science_authorized': False,
              'method_or_novelty_adopted': False, 'prefix_rows_usable_for_final_research_decision': False,
              'actual_quantum_shots_trajectory_GPU_DF_molecule_NPZ_LP': 0,
              'across_degrees_equal_exponential_accuracy_claim': False}
    guard = IOBudgetGuard(c['caps'])

    provenance = {k: result[k] for k in ('source_commit', 'execution_HEAD',
                  'contract_sha256', 'authorization_sha256', 'marker_sha256', 'runtime')}
    io = OutputSession(ROOT/c['result_directory'], c['caps']['output_bytes'], guard, provenance)
    status = 'G10_TECHNICAL_INCONCLUSIVE'
    with guard:
        try:
            collect(c, result, guard, io)
            guard.check()
            io.snapshot('scientific_body_returned_before_output')
            result['resource'] = guard.usage()
            result['status'] = 'G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE'
            result['prefix_rows_usable_for_final_research_decision'] = True
            identity = io.write_result(result)
            io.success(identity)
            status = result['status']
        except Exception as exc:
            counts = (len(result['rows']), result['new_synthesis_calls'], result['reused_keys'])
            result.clear()
            try:
                io.failure(exc, *counts)
            except Exception as receipt_exc:
                print(json.dumps({'status': status, 'failure_receipt_failed':
                    type(receipt_exc).__name__, 'mandatory_STOP': True, 'retries': 0}))
    print(json.dumps({'status': status, 'retries': 0, 'mandatory_STOP': True,
                      'next_science_authorized': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    execute(parser.parse_args().source_commit)
