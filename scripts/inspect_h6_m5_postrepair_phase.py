"""Saved callback metadata and artificial scalar checks; no molecular execution.

This module never loads QPY/NPZ, imports the native adapter, or invokes a pass
manager, transpiler, circuit action, phase correction, or acceptance override.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def phase_hex(value):
    require(isinstance(value, str), 'PHASE_HEX_STRING')
    result = float.fromhex(value)
    require(math.isfinite(result) and result.hex() == value, 'FINITE_CANONICAL_PHASE_HEX')
    return value


def summarize_callbacks(rows, initial_hex):
    """Classify stored bit changes, without attributing numerical error."""
    previous = phase_hex(initial_hex)
    require(isinstance(rows, list) and bool(rows), 'NONEMPTY_CALLBACK_LOG')
    frequencies, transitions = Counter(), []
    for index, row in enumerate(rows):
        require(type(row['count']) is int and row['count'] == index, 'CONTIGUOUS_CALLBACK_ORDER')
        require(isinstance(row['pass_name'], str) and bool(row['pass_name']), 'PASS_NAME')
        require(type(row['pass_seconds']) in (int, float) and math.isfinite(row['pass_seconds']) and row['pass_seconds'] >= 0, 'FINITE_PASS_SECONDS')
        counts = row['gate_counts']
        require(isinstance(counts, dict) and counts and all(type(n) is int and n >= 0 for n in counts.values()), 'NONNEGATIVE_INTEGER_GATE_COUNTS')
        require(set(counts) <= {'rz', 'sx', 'x', 'cx', 'id', 'barrier', 'measure'} and row['unexpected_names'] == [], 'REGISTERED_BASIS_NAMES_ONLY')
        require(row['scalar_observation'] == 'BASIS_NAMES_ONLY_NOT_GATE_IDENTITY_OR_CAUSAL_PROOF', 'OBSERVATION_LIMIT_LABEL')
        current = phase_hex(row['global_phase_binary64_hex'])
        frequencies[row['pass_name']] += 1
        if current != previous:
            transitions.append(dict(callback_count=index, pass_name=row['pass_name'],
                before_hex=previous, after_hex=current, gate_counts=dict(counts)))
        previous = current
    return dict(status='SAVED_CALLBACK_METADATA_ONLY', callback_records=len(rows),
        pass_counts=dict(sorted(frequencies.items())), phase_bit_change_records=len(transitions),
        phase_bit_change_pass_counts=dict(sorted(Counter(r['pass_name'] for r in transitions).items())),
        transitions=transitions, first_phase_bit_change=None if not transitions else transitions[0]['callback_count'],
        initial_phase_binary64_hex=initial_hex, final_phase_binary64_hex=previous,
        first_numerical_error_pass=None, causal_attribution_established=False,
        phase_change_is_not_an_error_test=True, unchanged_counts_are_not_gate_identity=True,
        intermediate_statevectors_available=False, per_replacement_phase_increments_available=False,
        real_phase_differences_or_determinants_computed=0, QPY_NPZ_loads=0,
        molecular_compile=0, molecular_replay=0, accepted_cost_rows=0)


def artificial_scalar_probe(*, artificial=False):
    """Expose sequential-addition loss using fixed synthetic scalar inputs.

    The exact rational sum is the original binary64 base. No molecular phase
    or observed error is used to choose the increments or alter a circuit.
    """
    require(artificial is True, 'ARTIFICIAL_SCALAR_ONLY')
    from qiskit.dagcircuit import DAGCircuit
    base, increment, count = 1.7, 2.0**-54, 65536
    exact_sum = Fraction.from_float(base) + count*Fraction.from_float(increment)
    exact_sum -= Fraction.from_float(count*increment)
    require(exact_sum == Fraction.from_float(base), 'ARTIFICIAL_EXACT_ZERO_INCREMENT')
    model = base
    dag = DAGCircuit()  # Empty DAG: no qubits, gates, state or circuit execution.
    dag.global_phase = base
    for _ in range(count):
        model = (model + increment) % (2*math.pi)
        dag.global_phase += increment
    model = (model - count*increment) % (2*math.pi)
    dag.global_phase -= count*increment
    actual = float(dag.global_phase)
    require(model.hex() == actual.hex(), 'INSTALLED_SETTER_MATCHES_FLOAT_ADD_WRAP_MODEL')
    require(actual != float(exact_sum), 'ARTIFICIAL_ADDITION_LOSS_OBSERVED')
    batched = DAGCircuit()
    batched.global_phase = float(exact_sum)
    require(float(batched.global_phase).hex() == base.hex(), 'ARTIFICIAL_SINGLE_ASSIGNMENT_CONTROL')
    return dict(status='ARTIFICIAL_SCALAR_COUNTEREXAMPLE_NOT_H6_CAUSAL_PROOF',
        initial_phase=base, small_increment=increment, small_increment_count=count,
        final_increment=-count*increment, exact_rational_sum_float=float(exact_sum),
        sequential_model_phase=model, installed_empty_DAG_phase=actual,
        signed_scalar_error=actual-float(exact_sum), single_assignment_phase=float(batched.global_phase),
        exact_sum_matches_original_base=True, artificial_inputs_only=True,
        no_modulo_boundary_crossed_in_this_example=True,
        H6_error_magnitude_explained=False, old_error_used_to_tune_inputs=False,
        molecular_inputs=0, molecular_compile=0, molecular_replay=0,
        production_installed=False, postcompile_correction=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def inspect_manifest(manifest):
    """Read precisely five SHA-bound JSON/JSONL receipts, never binary arrays."""
    require(set(manifest) == {'origin_SOURCE', 'origin_plan_sha256', 'inputs'}, 'STATIC_MANIFEST_KEYS')
    inputs = manifest['inputs']
    require(set(inputs) == {'callback_log', 'lowering_receipt', 'compiled_receipt', 'completion_witness', 'worker_STOP'}, 'EXACT_STATIC_METADATA_ROLES')
    loaded = {}
    for role, item in inputs.items():
        path = Path(item['path'])
        require(path.suffix == ('.jsonl' if role == 'callback_log' else '.json'), 'METADATA_FILES_ONLY')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'INPUT_SHA:'+role)
        text = path.read_text()
        loaded[role] = [json.loads(s) for s in text.splitlines()] if role == 'callback_log' else json.loads(text)
    for role in ('lowering_receipt', 'compiled_receipt', 'completion_witness', 'worker_STOP'):
        require(loaded[role]['SOURCE'] == manifest['origin_SOURCE'] and loaded[role]['plan_sha256'] == manifest['origin_plan_sha256'], 'ORIGIN_IDENTITY:'+role)
    require(loaded['worker_STOP']['reason'] == 'NUMERIC_DISAGREEMENT:compiled_vs_lowered' and loaded['completion_witness']['status'] == 'STOP', 'EXISTING_STOP_REQUIRED')
    result = summarize_callbacks(loaded['callback_log'], loaded['lowering_receipt']['stored_global_phase_binary64_hex'])
    require(result['final_phase_binary64_hex'] == float(loaded['compiled_receipt']['schema']['global_phase']).hex(), 'FINAL_PHASE_RECEIPT_JOIN')
    require(loaded['callback_log'][-1]['gate_counts'] == loaded['compiled_receipt']['schema']['gate_counts'], 'FINAL_COUNTS_RECEIPT_JOIN')
    for role, item in inputs.items():
        require(sha(item['path']) == item['sha256'], 'INPUT_UNCHANGED:'+role)
    result.update(origin_SOURCE=manifest['origin_SOURCE'], origin_plan_sha256=manifest['origin_plan_sha256'],
        input_binding=[dict(role=role, bytes=row['bytes'], sha256=row['sha256']) for role,row in inputs.items()],
        old_STOP_changed=False, all_three_raw_gates_PASS=False, mandatory_STOP=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect-saved-log', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--output')
    args = parser.parse_args()
    if not args.inspect_saved_log:
        print(json.dumps(dict(status='STATIC_SCOPE_ONLY', molecular_compile=0, molecular_replay=0, active_grants_issued=0)))
        return
    require(args.manifest and args.output, 'EXPLICIT_METADATA_MANIFEST_AND_NEW_OUTPUT_REQUIRED')
    result = inspect_manifest(json.loads(Path(args.manifest).read_text()))
    with Path(args.output).open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],callback_records=result['callback_records'],phase_bit_change_records=result['phase_bit_change_records'],first_numerical_error_pass=None,molecular_compile=0,molecular_replay=0)))


if __name__ == '__main__':
    main()
