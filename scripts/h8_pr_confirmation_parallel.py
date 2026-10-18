"""Owned parallel probes and paired axis compiles; no sampling at import.

Each trajectory builds its evolution once, then forks read-only copy-on-write
views. Four independent native probes and one event oracle share a control map;
cosine/sine compile share a second map. Every descendant has a native wait4
receipt. CPU groups are assigned by the outer authorized driver, not oversubscribed.
"""
import math
from pathlib import Path
import time

import pr_cost_contract as c

POLICY = 'H8_CONFIRMATION_GROUPED_PROBES_AXES_V1'


def results(receipt):
    c.require(receipt['status'] == 'PASS' and receipt['all_reaped'] is True
              and not receipt['cleanup_errors'], 'PARALLEL_NATIVE_COMPLETION')
    values = {}
    for handle in receipt['workers']:
        path = Path(handle['directory']) / 'task_result.json'
        c.require(handle['reaped'] and handle['exit_code'] == 0
                  and handle['result_sha256'] == c.sha(path), 'PARALLEL_CHILD_RESULT_SHA')
        c.require(handle['task_id'] not in values, 'UNIQUE_PARALLEL_CHILD')
        values[handle['task_id']] = c.read(path)
    return values


def control(evolution, requests, context, writer, tolerance, cpus, pool, font_seed,
            *, oracle=None):
    m = context.m
    state = context.qstate
    np = m.np
    n = state.size.bit_length() - 1
    if oracle is None:
        from h6_dba_science import event_oracle
        oracle = event_oracle
    # The calls match the old four probes + one algebraic event oracle exactly.
    def probe(task, directory):
        kind = task['id']
        low = np.concatenate((state, np.zeros_like(state)))
        if kind == 'low':
            value = m.simulate_statevector(evolution.circuit, low)
            return dict(errors=[float(np.linalg.norm(value - low))])
        if kind == 'high':
            high = np.concatenate((np.zeros_like(state), state))
            value = m.simulate_statevector(evolution.circuit, high)
            z = complex(np.vdot(state, value[len(state):]))
            return dict(z=[z.real, z.imag], errors=[float(np.linalg.norm(value[:len(state)])),
                        abs(float(np.linalg.norm(value)) - 1)])
        if kind == 'oracle':
            z = complex(oracle(requests, m, context.native, state))
            return dict(z=[z.real, z.imag])
        wrapper = m.build_native_hadamard_wrapper(evolution, axis=kind,
            include_measurement=False, max_instructions=1000000)
        return dict(value=float(m.hadamard_expectation(m.simulate_statevector(wrapper, low), n)))
    receipt = pool.map_owned([dict(id=k) for k in ('low', 'high', 'cosine', 'sine', 'oracle')],
        probe, cpus=cpus, limits=dict(max_workers=len(cpus)),
        directory=writer.directory / 'parallel_control', deadline=math.inf, font_seed=font_seed)
    values = results(receipt)
    z = complex(*values['high']['z'])
    errors = values['low']['errors'] + values['high']['errors'] + [
        abs(values['cosine']['value'] - z.real), abs(values['sine']['value'] - z.imag),
        abs(z - complex(*values['oracle']['z']))]
    maximum = max(errors)
    c.require(all(math.isfinite(x) for x in errors) and maximum < tolerance,
              'NATIVE_SCALAR_CONTROL_EVENT_ORACLE')
    writer.write('control.json', dict(status='PASS', max_error=maximum, tolerance=tolerance,
        control_probes=4, independent_event_oracles=1, full_Fock_scalar_once=True,
        reference_rotated=False, ordinary_ancilla_branch=True, parallel_native=True,
        native_receipt_sha256=c.sha(writer.directory / 'parallel_control/parallel_receipt.json')))


def compile_axes(evolution, requests, context, task, event_digest, directory, cpus,
                 pool, font_seed, native):
    m = context.m
    def compile_axis(axis_task, work):
        axis = axis_task['id']
        c.require(c.digest(native.event_values(requests)) == event_digest, 'PAIRED_EVENT_MUTATION')
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        with use_fingerprint_reuse():
            wrapper = m.build_native_hadamard_wrapper(evolution, axis=axis,
                include_measurement=True, max_instructions=1000000)
            fingerprint = m.canonical_qiskit_circuit_fingerprint(wrapper)
            start = time.monotonic()
            value = m.transpile_and_measure_cost(wrapper, context.compiler,
                actual_circuit_fingerprint=fingerprint)
            c.require(value.compiler_settings_hash == context.binding['compiler']
                      and value.actual_circuit_fingerprint == fingerprint
                      and m.canonical_qiskit_circuit_fingerprint(wrapper) == fingerprint, 'COMPILE_IDENTITY')
            metrics = {key: getattr(value, attr) for key, attr in dict(RZ='rz_count', CX='cx_count',
                depth='total_depth', RZ_depth='rz_depth', CX_depth='cx_depth', size='circuit_size').items()}
            c.require(all(type(x) is int and x >= 0 for x in metrics.values())
                      and metrics['RZ'] > 0, 'INTEGER_COMPILED_METRICS')
            return dict(metrics, seed=task['seed'], event_digest=event_digest,
                compiler_hash=value.compiler_settings_hash, fingerprint=fingerprint,
                wall_seconds=time.monotonic() - start)
    receipt = pool.map_owned([dict(id=axis) for axis in c.core.AXES], compile_axis,
        cpus=cpus[:2], limits=dict(max_workers=min(2, len(cpus))),
        directory=Path(directory) / 'parallel_axes', deadline=math.inf, font_seed=font_seed)
    return results(receipt)


def compile_one(context, task, directory, source, target, law, log_B, margin,
                native, pool, cpus, font_seed, cohort):
    cell = task['cell']
    prep = context.preparations[cell['prefix']]
    writer = native.Writer(directory)
    budget = c.Budget(directory, dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R'],
        native_evolution=1, wrapper_compile=2, control_probe=4, independent_event_oracle=1))
    for kind, count in dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R']).items():
        budget.take(kind, count)
    requests, distribution = native.steps(prep, cell, task['seed'])
    c.require([r.seed for r in requests] == native.seed_hierarchy(task['seed'], cell['q'])[1:],
              'ORIGINAL_STEP_SEED_API')
    original_law = distribution.to_dict()
    c.require(original_law == law and c.digest(original_law) == target['original_law_sha256'],
              'ORIGINAL_UNCONDITIONAL_LAW')
    events = native.event_values(requests)
    event_digest = c.digest(events)
    c.require(task.get('expected_prior_event_digest', event_digest) == event_digest,
              'ORIGINAL_EVENT_STREAM_NO_REDRAW')
    c.require(len(events) == cell['q'] and all(len(step) == cell['r'] for step in events),
              'EXACT_OCCURRENCE_EVENTS')
    c.require(prep.preparation_hash == target['preparation_sha256'], 'EXACT_PREPARATION_IDENTITY')
    observed = cell['R'] * math.log(distribution.exact_finite_distribution)
    c.require(math.isfinite(observed) and abs(observed - log_B) <= margin, 'ORIGINAL_NORMALIZATION')
    writer.write('trajectory.json', dict(cell=cell, seed=task['seed'], SOURCE=source,
        events=events, event_digest=event_digest, original_RTE_distribution=original_law,
        cohort=cohort, conditional=False, quantum_weights_changed=False))
    budget.take('native_evolution')
    evolution = context.native.partial_native_from_step_requests(requests,
        control_policy='symmetric_directional', max_instructions=1000000)
    if task['replica'] == 0:
        budget.take('control_probe', 4)
        budget.take('independent_event_oracle')
        control(evolution, requests, context, writer, 1e-8, cpus[:5], pool, font_seed)
    budget.take('wrapper_compile', 2)
    axes = compile_axes(evolution, requests, context, task, event_digest, directory,
                        cpus, pool, font_seed, native)
    for axis, value in axes.items():
        writer.write('measured_' + axis + '.json', value)
    return dict(cell_id=cell['id'], replica=task['replica'], seed=task['seed'], SOURCE=source,
        event_digest=event_digest, binding=context.binding, axes=axes, cohort=cohort,
        calls=budget.used, normalization_unchanged=True, synthetic=False,
        parallel_native=True, axes_native_receipt_sha256=c.sha(Path(directory) / 'parallel_axes/parallel_receipt.json'))


def verify_descendants(directory, result):
    directory = Path(directory)
    path = directory / 'parallel_axes/parallel_receipt.json'
    c.require(c.sha(path) == result['axes_native_receipt_sha256'], 'AXIS_NATIVE_RECEIPT_SHA')
    axes = results(c.read(path))
    c.require(axes == result['axes'] and set(axes) == set(c.core.AXES), 'BOTH_AXES_NATIVE_EXIT0')
    if result['replica'] == 0:
        control_receipt = c.read(directory / 'control.json')
        path = directory / 'parallel_control/parallel_receipt.json'
        c.require(c.sha(path) == control_receipt['native_receipt_sha256'], 'CONTROL_NATIVE_RECEIPT_SHA')
        c.require(set(results(c.read(path))) == {'low', 'high', 'cosine', 'sine', 'oracle'},
                  'FOUR_PROBES_ONE_ORACLE_NATIVE_EXIT0')
