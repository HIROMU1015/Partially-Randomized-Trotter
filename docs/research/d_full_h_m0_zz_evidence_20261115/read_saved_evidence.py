"""Read pinned m0 evidence only; no builder, sampler, compiler or simulator calls.

QPY load deserializes saved wrappers. Existing pure phase/XY readers are selected
from pinned ASTs, without importing their scientific runners. RZ metrics are
quoted from original JSON; circuit prices are never counted here.
"""
import ast
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import types

ROOT = Path(__file__).resolve().parents[3]
SOURCE = '7f85f20e38af4df1a649ac239e57bfe76d0899ee'
RESULT = 'b25d7da6dedebc4cd22e8d6f00e6de75d7072292'
HANDOFF = '74cd9c2625a6338793f1beef7b434c765a3a7e67'
C_START = '426b8c4f14b4a00869c73f7b2ae4ecdc638a0107'
BASE = 'docs/research/d_full_h_m0_reference_20261115/'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
NATIVE = 'af284d853890155bacd19e61850c196bd1d5c660'
FULL = '9b83b96b83f62142c9d87858dc24a94db7ac5d61'
ZZ = '01e068af7e06d9f1f5621a5b3800c30eb32ecb06'
INPUTS = {}


def raw(commit, path):
    value = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)
    INPUTS[commit + ':' + path] = hashlib.sha256(value).hexdigest()
    return value


def read(commit, path):
    return json.loads(raw(commit, path))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def readers(commit, path, names, env):
    tree = ast.parse(raw(commit, path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PHASE_VALIDATION_ATOL'
                                                for t in node.targets):
            env['_PHASE_VALIDATION_ATOL'] = ast.literal_eval(node.value)
    selected = []
    for name in names:
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(nodes) == 1, 'Ambiguous pure reader: ' + name)
        node = nodes[0]
        node.decorator_list = []
        selected.append(node)
    # Python AST execution only; this never invokes circuit compilation/building.
    exec('from __future__ import annotations\n' + ast.unparse(ast.Module(body=selected, type_ignores=[])), env)


def inspect():
    from qiskit import qpy
    job = read(SOURCE, BASE + 'FROZEN_JOB.json')
    handoff = read(HANDOFF, BASE + 'HANDOFF_C.json')
    trial = read(RESULT, BASE + 'run01/NATIVE_TRIAL/result.json')
    events = read(RESULT, BASE + 'run01/NATIVE_TRIAL/events_PRIVATE.json')
    run = read(RESULT, BASE + 'run01/RESULT.json')
    context = read(RESULT, BASE + 'run01/H4_context.json')
    proposal = read(job['rule_source']['commit'], job['rule_source']['path'])
    require(trial['candidate'] == job['row'] == handoff['physical'], 'Physical binding')
    require(events['cohort'] == trial['cohort'] == job['cohort'], 'Cohort binding')
    require(trial['index'] == job['index'] == 0 and events['seeds'] == job['seeds'], 'Path/seed binding')
    require(events['conditional'] is False and trial['conditional'] is False, 'Conditional draw')
    require(run['child']['normal_sampler_calls'] == 1, 'One original draw')
    require(context['deterministic_blocks'] == 0, 'No deterministic head evidence')
    require(run['child']['context_check']['status'] == 'PASS', 'Original context check')
    require(trial['preparation_hash'] == context['preparation_hash'] == proposal['target_preparation_hash'], 'Preparation')
    require(trial['scalar'] == context['scalar'] == proposal['scalar']['total'], 'Scalar binding')
    require(len(events['events']) == 1 and len(events['events'][0]) == 9, 'Event allocation')
    require(handoff['C_0X'] is None and handoff['C_0Y'] is None, 'Original prices remain null')

    applications = []
    for r, batch in enumerate(events['events']):
        for k, event in enumerate(batch):
            for j, app in enumerate(event['application_sequence']):
                support = app['diagonal_pauli_support']
                # The native normal sampler fixes controlled=True, ancilla=8.
                eligible = (not app['is_identity'] and app['role'] == 'rotation'
                            and support is not None and len(support) == 2)
                basis_system_only = all(all(0 <= q < 8 for q in op['qubits']) for op in app['basis_change_operations'])
                require(basis_system_only, 'Basis metadata touches ancilla')
                applications.append(dict(selector=['events', r, k, 'application_sequence', j],
                    component_id=app['component_id'], role=app['role'], is_identity=app['is_identity'],
                    central_support=support, taylor_order=event['taylor_order'], phase=event['phase'],
                    basis_system_only=basis_system_only, replacement_target=eligible))
    targets = sum(a['replacement_target'] for a in applications)
    require(len(applications) == 9 and targets == 0, 'Expected independently checked zero-target path')
    require(all(a['role'] == 'rotation' and not a['is_identity'] and len(a['central_support']) == 1 for a in applications),
            'Unexpected central application')

    env = dict(math=math, json=json, types=types,
               PHASE_GATE_ID='SCALAR_EVENT_PHASE_ANCILLA_SIGNATURE_V1',
               LOWERING_ID='CONTROLLED_ZZ_CX_CRZ_CX_V1')
    readers(PROVIDER, 'src/trotterlib/df_rte_qiskit.py',
            ['_event_phase_angle', '_identity_application_phase'], env)
    provider = types.SimpleNamespace(_event_phase_angle=env['_event_phase_angle'],
        QiskitDFRTEEventCircuitBuilder=types.SimpleNamespace(_identity_application_phase=env['_identity_application_phase']))
    readers(SOURCE, 'scripts/run_full_h_kappa2_p1.py',
            ['_event_view', 'central_phase', 'request_phase', 'expected_ancilla_ops', 'ancilla_ops',
             'scalar_event_phase_gate', 'wrapper_evolution_bounds', 'xy_phase_gate'], env)
    wrappers, axes = {}, []
    for row in handoff['axis_rows']:
        axis = row['axis']
        for key in ('compile_record', 'wrapper', 'compiled_circuit'):
            loc = row[key]
            require(hashlib.sha256(raw(loc['commit'], loc['path'])).hexdigest() == loc['sha256'], 'Original bytes: ' + key)
        record = read(row['compile_record']['commit'], row['compile_record']['path'])
        require(record['fingerprint'] == row['fingerprint'], 'Own-axis fingerprint')
        require(record['compiler'] == job['profile']['compiler'], 'Compiler profile')
        native_axis = 'cosine' if axis == 'X' else 'sine'
        require(record['metrics'] == trial['prices'][native_axis], 'Saved metrics only; no price recount')
        circuits = qpy.load(io.BytesIO(gzip.decompress(raw(row['wrapper']['commit'], row['wrapper']['path']))))
        require(len(circuits) == 1, 'One saved wrapper')
        qc = circuits[0]
        require(qc.num_qubits == 9 and qc.num_clbits == 1 and float(qc.global_phase) == 0.0, 'Wrapper registers/phase')
        start, stop = env['wrapper_evolution_bounds'](qc, 8, native_axis)
        names = sorted({i.operation.name for i in qc.data})
        require(set(names) <= {'crz', 'p', 'xx_plus_yy', 'h', 'sdg', 'measure'}, 'Unknown/opaque wrapper instruction')
        expected_classes = dict(crz='CRZGate', p='PhaseGate', xx_plus_yy='XXPlusYYGate',
                                h='HGate', sdg='SdgGate', measure='Measure')
        classes = {i.operation.name: i.operation.base_class.__name__ for i in qc.data}
        require(all(i.operation.base_class.__name__ == expected_classes[i.operation.name] for i in qc.data),
                'Unexpected custom instruction class')
        require(all(len(i.qubits) <= 2 for i in qc.data), 'Unexpected three-wire gate')
        require(all(i.operation.name == 'crz' or len(i.qubits) == 1
                    for i in qc.data if any(qc.find_bit(q).index == 8 for q in i.qubits)), 'Unknown ancilla gate')
        measurements = [(k, [qc.find_bit(q).index for q in i.qubits], [qc.find_bit(c).index for c in i.clbits])
                        for k, i in enumerate(qc.data) if i.operation.name == 'measure']
        require(measurements == [(len(qc.data)-1, [8], [0])], 'Measured full wrapper')
        phase = env['scalar_event_phase_gate'](qc, 8, events['events'], trial['scalar'], proposal['scalar']['total'],
                                              job['row']['T'], provider=provider, start=start, stop=stop)
        require(phase['status'] == 'PASS', 'Saved-wrapper phase signature')
        wrappers[native_axis] = qc
        axes.append(dict(axis=axis, receipt=row['compile_record'], wrapper=row['wrapper'], compiled_circuit=row['compiled_circuit'],
            fingerprint=row['fingerprint'], compiler_profile=job['profile'],
            observed_RZ_quoted_from_original_record=record['metrics']['rz_count'], per_shot_RZ=None,
            original_implementation_gate_return='NOT_PERSISTED',
            current_read_only_phase_check=phase, evolution_bounds=[start, stop],
            unique_top_level_operation_names=names, standard_operation_classes=classes,
            all_operations_at_most_two_wires=True,
            measurement=measurements, global_phase=0.0))
    xy = env['xy_phase_gate'](wrappers, 8, events['events'], trial['scalar'], proposal['scalar']['total'],
                              job['row']['T'], provider=provider)
    require(xy['status'] == 'PASS', 'Current read-only X/Y check')
    required = {'low_control', 'high_control_leakage', 'independent_occupation_vector', 'Hadamard_cosine',
                'Hadamard_sine', 'compiled_vector_cosine', 'compiled_vector_sine'}
    require(trial['status'] == 'PASS' and {g['check'] for g in trial['checks']} == required, 'Original seven gates')
    require(all(math.isfinite(g['error']) and g['error'] < g['tolerance'] and g['phase_alignment'] is False
                for g in trial['checks']), 'Original numerical evidence')
    progress = raw(RESULT, BASE + 'run01/MONITORING/PROGRESS.jsonl').decode()
    require('xy_phase_gate' in progress and 'path_checks_complete' not in progress, 'Original STOP stage')
    # Read only the source definitions supporting the whole-wrapper scope proof.
    for commit, path in [(FULL, 'scripts/h46_full_h_pure_rte.py'),
                         (NATIVE, 'scripts/h46_long_time_native.py'),
                         (NATIVE, 'src/trottertracks/resource_applicability/ax2b_native_df_v5.py'),
                         (ZZ, 'src/trottertracks/efficient_controlled_zz.py'),
                         (SOURCE, 'scripts/full_h_p2_acceleration.py'),
                         (SOURCE, 'scripts/run_full_h_kappa2_p2.py'),
                         (SOURCE, 'scripts/run_full_h_m0_reference.py')]:
        raw(commit, path)
    raw(C_START, 'docs/research/full_h_m0_reference_total_20261115/REPORT.md')
    return dict(status='READ_ONLY_TECHNICAL_EVIDENCE_COMPLETE_NOT_PRICE_ADMISSION',
        C_start=C_START, execution_source=SOURCE, result_commit=RESULT, D1_handoff=HANDOFF,
        physical=job['row'], cohort=job['cohort'], path_index=0, seeds=job['seeds'], independent_paths=1,
        zero_target_evidence=dict(target_count_before_substitution=targets, target_counter_used_as_basis=False,
            applications=applications, no_deterministic_blocks=context['deterministic_blocks'],
            full_wrapper_scope='SOURCE call graph + saved context + all event applications + saved wrapper instruction/readout signatures',
            counter_original_return=None, counter_original_return_status='NOT_PERSISTED'),
        original_saved=dict(context_check='PASS', native_status=trial['status'], numeric_checks=trial['checks'],
            sampler_calls=run['child']['normal_sampler_calls'], composite_gate='STOP',
            parity_return='NOT_PERSISTED', scalar_event_return='NOT_PERSISTED', XY_return='NOT_PERSISTED',
            runtime_lowering_marker='NOT_PERSISTED', legacy_evolution_QPY='NOT_SAVED'),
        current_read_only_checks=dict(phase_and_XY=xy, axes=axes,
            legacy_new_equivalence='FIXED_SOURCE_BRANCH_ARGUMENT_ONLY_NO_LEGACY_CIRCUIT_COMPARISON',
            phase_XY_provenance='NEW_READ_ONLY_CHECK_OF_ORIGINAL_SAVED_QPY_NOT_ORIGINAL_RUN_GATE_RETURN'),
        unchanged_admission=dict(C_0X=None, C_0Y=None, total=None, accepted_axes=0, held_axes=2),
        new_work_counts=dict(sampling=0, redraw=0, circuit_build=0, circuit_compile=0, simulation=0, RZ_recount=0,
                             accounting_runs=0, repeated_18_tests=0),
        inputs_sha256=INPUTS, mandatory_stop=True)


if __name__ == '__main__':
    value = inspect()
    destination = Path(__file__).with_name('EVIDENCE.json')
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=value['status'], target_count=0, saved_numeric_gates=7,
                         current_read_only_phase_XY=value['current_read_only_checks']['phase_and_XY']['status'],
                         accepted_axes=0, C_0X=None, C_0Y=None, total=None)))
