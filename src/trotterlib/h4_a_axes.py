"""A_AXES: reuse controlled QPY, measure 255 PR + one PF sine, then STOP.

No quantum import before authority. Scientific builders and compiler are reused.
Only namespace/status labels and the finite task count bind the existing
cohort authority/native launcher to this separately authorized execution.
"""
import ast
from collections import Counter
from fractions import Fraction
import inspect
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import FunctionType, SimpleNamespace
from . import h4_a_cohort as H

C, G, D2 = H.C, H.G, H.D2
ROOT = H.ROOT
BASE = ROOT / 'artifacts/non_df_h4_a_axes/2026-10-24'
SCRIPT = ROOT / 'scripts/run_h4_a_axes.py'
PYTHON = H.PYTHON
CASES = ('PF',) + tuple(f'PR{i:04d}' for i in range(256) if i != 1)
LIMITS = dict(saved_controlled_body_load=256, sine_wrapper_build=256, cost_transpile=256,
              paired_statistics_batch=1, body_replay=0, central_control_build=0,
              cosine_compile=0, sampling=0, state_signal=0, N_G=0)
KIND = 'H4_A_AXES_ACTIVE_V1'
JOB_ENV = H.JOB_ENV


def rebound(function, replacements=None, **bindings):
    """Reuse tested control flow; only explicitly listed literals may differ."""
    tree = ast.parse(inspect.getsource(function))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            key = (type(node.value), node.value)
            if key in (replacements or {}):
                node.value = replacements[key]
    env = dict(function.__globals__, **bindings)
    exec(compile(tree, function.__code__.co_filename + ':A_AXES_binding', 'exec'), env)
    return env[function.__name__]


def fixed():
    result = dict(H.fixed())
    result.pop('axis')
    result['axes'] = ['cosine', 'sine']
    return result


def scope(c):
    C.require(c['kind'] == KIND and all(c[k] is True for k in ('execution_authorized', 'grant_valid', 'dispatch_enabled')), 'AXES_AUTHORITY')
    C.require(c['fixed'] == fixed() and c['new_targets'] == list(CASES) and c['axis'] == 'sine', 'AXES_FIXED_TARGETS')
    C.require(c['call_limits'] == LIMITS and c['per_target_compile_max'] == 1, 'AXES_CALL_SCOPE')
    C.require(c['mandatory_STOP'] and c['automatic_retry'] is False and c['automatic_resume'] is False
              and c['automatic_next_stage'] is False, 'AXES_STOP_SCOPE')
    C.require(c['technical_recovery_authorized'] is True and c['recovery_requires_new_SOURCE_seal_grant'] is True, 'AXES_RECOVERY_BOUNDARY')
    C.require(c['job_environment'] == JOB_ENV and 1 <= c['worker_count'] <= 256, 'AXES_RESOURCES')


def verify_seal(s, *, runtime=True, committed=True):
    C.require(s['kind'] == 'H4_A_AXES_SEAL_V1', 'AXES_SEAL_KIND')
    function = FunctionType(D2.verify_seal.__code__, dict(D2.__dict__, BASE=BASE))
    function(dict(s, kind='H4_D2_SOURCE_INPUT_RUNTIME_SEAL_V1'), runtime=runtime, committed=committed)


def authority(*, consume=False, task=False):
    function = rebound(H.authority, {(str, 'H4_A_COHORT_COS_ONE_SHOT'): 'H4_A_AXES_ONE_SHOT'},
                       BASE=BASE, LIMITS=LIMITS, scope=scope, verify_seal=verify_seal)
    return function(consume=consume, task=task)


def input_manifest():
    """SHA/metadata only; no QPY loads or new circuit/statistics evaluation."""
    from .h4_conditional_cng import extract
    verify_join = extract('src/trotterlib/rte_event_audit.py', 'verify_join', dict(digest=C.digest))
    old_rows = C.read(H.BASE / 'run01/cohort_costs.json')['rows']
    pr_records = H.records()
    compiler = H.fixed()['compiler']
    source_cache = {}
    C.require(len(old_rows) == 256 and [r['index'] for r in old_rows] == list(range(256)), 'AXES_OLD_COHORT')
    target_pf = next(t for t in C.read(D2.BASE / 'active_contract.json')['targets'] if t['case'] == 'pauli_jw_PF')
    rows = []
    for method, index in [('PF', 0)] + [('PR', i) for i in range(256)]:
        key = 'PF' if method == 'PF' else f'PR{index:04d}'
        if method == 'PF' or index == 0:
            folder = D2.BASE / 'run01/variants' / ('pauli_jw_' + method)
            origin = D2.BASE
            body = folder / 'controlled_body.qpy'
            native = origin / 'run01/native_tasks' / ('compile_pauli_jw_' + method) / 'native_exit.json'
            source = '74515d8eb4f073c40fc783681a3775dbca89e9b1'
        elif index == 1:
            folder = G.BASE / 'run01/variants/cosine'
            origin = G.BASE
            body = origin / 'run01/controlled_body.qpy'
            native = origin / 'run01/native_tasks/compile_cosine/native_exit.json'
            source = 'ee7d281b5daead1d0d5e81e00c9a0ed4727792b0'
        else:
            folder = H.BASE / 'run01/trajectories' / f'{index:04d}'
            origin = H.BASE
            body = None  # Extract the unchanged body from the saved cosine QPY.
            native = origin / 'run01/native_tasks' / ('compile_' + f'{index:04d}') / 'native_exit.json'
            source = '34ef0a7ca2412a61f5215a4ce1192e99f3226b2a'
        if origin not in source_cache:
            source_cache[origin] = (C.read(origin / 'run01/STOP.json'), C.read(origin / 'run01/output_inventory.json')['files'])
        stop, inventory = source_cache[origin]
        receipt = C.read(native)
        C.require(stop['status'].endswith('COMPLETE_MANDATORY_STOP') and stop['all_owned_reaped']
                  and receipt['exit_code'] == 0 and receipt['reaped'] and receipt['native_wait'], 'AXES_OLD_NATIVE')
        m, inp = C.read(folder / 'metrics.json'), C.read(folder / 'input.json')
        C.require(m['SOURCE'] == source and m['compiler'] == compiler
                  and m['input_fingerprint'] == inp['input_fingerprint']
                  and C.sha(folder / 'input.qpy') == inp['input_qpy_sha256'], 'AXES_OLD_PROFILE_INPUT')
        join_path = target_pf['price_join'] if method == 'PF' else pr_records[index]['price_join']
        join = C.read(ROOT / join_path)
        verify_join(join, expected_case='pauli_jw_' + method)
        C.require(join['binding']['SOURCE'] == '0c329b30a81accb35c212be8defa66fee0dd0f12', 'AXES_ORIGINAL_SOURCE')
        if method == 'PF':
            C.require(m['RZ'] == 15028 and m['case'] == 'pauli_jw_PF', 'AXES_PF_REUSE')
        else:
            saved = old_rows[index]
            C.require(saved['trajectory_id'] == join['trajectory_id'] and all(saved[k] == m[k] for k in ('RZ', 'CX', 'depth', 'rz_depth', 'cx_depth')), 'AXES_COSINE_PRICE_ID')
            C.require(m.get('axis', 'cosine') == 'cosine', 'AXES_COSINE_AXIS')
        paths = [folder / name for name in ('input.json', 'input.qpy', 'metrics.json', 'output.qpy')] + [native]
        if body is not None:
            paths.append(body)
        for path in paths:
            C.require(C.sha(path) == inventory[str(path.relative_to(origin / 'run01'))], 'AXES_OLD_INVENTORY')
        row = dict(key=key, method=method, trajectory_index=index, trajectory_id=join['trajectory_id'],
                   event_record_hash=join['event_record_hash'], trajectory_record_hash=join['trajectory_record_hash'],
                   price_join=join_path, price_join_sha256=C.sha(ROOT / join_path),
                   cosine_folder=str(folder.relative_to(ROOT)), cosine_SOURCE=source,
                   cosine_input_fingerprint=m['input_fingerprint'],
                   body_source='saved_controlled_body' if body else 'extract_saved_cosine_input',
                   body_qpy=str(body.relative_to(ROOT)) if body else None,
                   input_files={str(path.relative_to(ROOT)): C.sha(path) for path in paths})
        if key == 'PR0001':
            sine = G.BASE / 'run01/variants/sine'
            sine_m, sine_i = C.read(sine / 'metrics.json'), C.read(sine / 'input.json')
            C.require(sine_m['RZ'] == 1129 and sine_m['axis'] == 'sine' and sine_m['trajectory_id'] == row['trajectory_id']
                      and sine_m['compiler'] == m['compiler'] and sine_m['SOURCE'] == source
                      and sine_m['input_fingerprint'] == sine_i['input_fingerprint'], 'AXES_REUSED_SINE')
            extra = [sine / name for name in ('input.json', 'input.qpy', 'metrics.json', 'output.qpy')]
            extra.append(G.BASE / 'run01/native_tasks/compile_sine/native_exit.json')
            for path in extra:
                C.require(C.sha(path) == inventory[str(path.relative_to(origin / 'run01'))], 'AXES_SINE_INVENTORY')
            row['reused_sine_folder'] = str(sine.relative_to(ROOT))
            row['input_files'].update({str(path.relative_to(ROOT)): C.sha(path) for path in extra})
        rows.append(row)
    C.require(len(rows) == 257 and len({(r['method'], r['trajectory_id']) for r in rows}) == 257, 'AXES_UNIQUE_INPUTS')
    C.require([r['key'] for r in rows if r['key'] != 'PR0001'] == list(CASES), 'AXES_NEW_SCOPE')
    return dict(status='PASS_SAVED_METADATA_257_COSINE_AND_1_SINE', rows=rows, new_scientific_calls=0)


def prepare():
    C.require(not (BASE / 'seal.json').exists() and not (BASE / 'run01').exists(), 'AXES_NAMESPACE_USED')
    approval = C.read(BASE / 'approval.json')
    text = (BASE / 'user_instruction.txt').read_text()
    C.require('A_AXES_FOR_COMPLEX_Gの限定科学実行を承認' in text
              and approval['instruction_sha256'] == C.sha(BASE / 'user_instruction.txt'), 'AXES_EXPLICIT_APPROVAL')
    tests = C.read(BASE / 'artificial/new_tests.json')
    C.require(tests['tests'] == 14 and tests['errors'] == tests['failures'] == 0, 'AXES_ARTIFICIAL_TESTS')
    old = C.read(H.BASE / 'seal.json')
    H.verify_seal(old)
    manifest = input_manifest()
    C.save(BASE / 'case_inputs.json', manifest)
    code, inputs = dict(old['code_files']), dict(old['input_files'])
    for path in ('src/trotterlib/h4_a_axes.py', 'scripts/run_h4_a_axes.py', 'tests/test_h4_a_axes.py'):
        code[path] = C.sha(ROOT / path)
    for row in manifest['rows']:
        inputs.update(row['input_files'])
        inputs[row['price_join']] = row['price_join_sha256']
    for path in (BASE / 'approval.json', BASE / 'user_instruction.txt', BASE / 'output_schema.json',
                 BASE / 'resource_plan.json', BASE / 'case_inputs.json', BASE / 'artificial/new_tests.json',
                 H.BASE / 'run01/cohort_costs.json', H.BASE / 'run01/statistics.json',
                 H.OLD_RUN / 'pauli_jw_PF/result.json'):
        inputs[str(path.relative_to(ROOT))] = C.sha(path)
    protected = dict(C.read(H.BASE / 'preservation_before.json'))
    for path in H.BASE.rglob('*'):
        if path.is_file() and 'mplconfig' not in path.parts and '__pycache__' not in path.parts:
            protected[str(path.relative_to(ROOT))] = C.sha(path)
    C.save(BASE / 'preservation_before.json', protected)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    seal = dict(kind='H4_A_AXES_SEAL_V1', SOURCE=source, code_files=code, input_files=inputs,
                python_sha256=C.sha(sys.executable), runtime_manifest=old['runtime_manifest'],
                runtime_manifest_sha256=old['runtime_manifest_sha256'], approval_sha256=C.sha(BASE / 'approval.json'),
                preservation_sha256=C.sha(BASE / 'preservation_before.json'), real_H4_calls_before_seal=0, UTC=C.utc())
    verify_seal(seal)
    C.save(BASE / 'seal.json', seal)
    contract = dict(kind=KIND, science_SOURCE=source, fixed=fixed(), new_targets=list(CASES), axis='sine',
        execution_authorized=True, grant_valid=True, dispatch_enabled=True, call_limits=LIMITS,
        per_target_compile_max=1, mandatory_STOP=True, automatic_retry=False, automatic_resume=False, automatic_next_stage=False,
        technical_recovery_authorized=True, recovery_requires_new_SOURCE_seal_grant=True,
        recovery_policy='Only diagnosed technical failure/incomplete original targets; preserve consumed calls and prior runs; carry verified successes; no blind retry or scientific changes',
        job_environment=JOB_ENV, worker_count=C.read(BASE / 'resource_plan.json')['worker_count'],
        seal_sha256=C.sha(BASE / 'seal.json'), approval_sha256=seal['approval_sha256'],
        output_schema_sha256=C.sha(BASE / 'output_schema.json'), output_namespace=str(BASE / 'run01'))
    scope(contract)
    C.save(BASE / 'active_contract.json', contract)
    ns = BASE / 'run01'
    ns.mkdir()
    grant = dict(kind='H4_A_AXES_ONE_SHOT', SOURCE=source, namespace=str(ns),
        contract_sha256=C.sha(BASE / 'active_contract.json'), seal_sha256=C.sha(BASE / 'seal.json'),
        approval_sha256=seal['approval_sha256'], one_shot=True, automatic_retry=False, call_limits=LIMITS)
    grant['grant_id'] = C.digest(grant)
    C.save(ns / 'grant.json', grant)
    C.save(ns / 'call_ledger.json', dict(limits=LIMITS, used=dict.fromkeys(LIMITS, 0), events=[]))
    return grant


def circuit_signature(circuit):
    """Explicit operands only, no matrix, lazy definition or synthesis."""
    return dict(n_qubits=circuit.num_qubits, n_clbits=circuit.num_clbits, global_phase=float(circuit.global_phase),
        instructions=[dict(name=item.operation.name, params=[float(p) for p in item.operation.params],
            qubits=[circuit.find_bit(q).index for q in item.qubits], clbits=[circuit.find_bit(c).index for c in item.clbits]) for item in circuit.data])


def controlled_from_cosine(cosine, direct=None):
    from qiskit import QuantumCircuit
    s = circuit_signature(cosine)
    n = cosine.num_qubits - 1
    head = dict(name='h', params=[], qubits=[n], clbits=[])
    measure = dict(name='measure', params=[], qubits=[n], clbits=[0])
    C.require(n >= 1 and cosine.num_clbits == 1 and not cosine.parameters and len(s['instructions']) >= 3
              and s['instructions'][0] == head and s['instructions'][-2] == head
              and s['instructions'][-1] == measure, 'AXES_COSINE_WRAPPER')
    C.require(all(not item.clbits and getattr(item.operation, 'condition', None) is None
                  and item.operation.name in ('h', 's', 'sdg', 'cx', 'cy', 'cz', 'crz', 'p') for item in cosine.data[1:-2]), 'AXES_CONTROLLED_BODY_DOMAIN')
    if direct is None:
        body = QuantumCircuit(cosine.num_qubits)
        body.global_phase = cosine.global_phase
        for item in cosine.data[1:-2]:
            body.append(item.operation, [body.qubits[cosine.find_bit(q).index] for q in item.qubits])
    else:
        body = direct
    b = circuit_signature(body)
    C.require(body.num_qubits == cosine.num_qubits and body.num_clbits == 0 and not body.parameters
              and b['global_phase'] == s['global_phase'] and b['instructions'] == s['instructions'][1:-2], 'AXES_SAVED_BODY_COSINE_CORRESPONDENCE')
    return body, dict(status='PASS_UNCHANGED_CONTROLLED_BODY', direct_saved_body=direct is not None,
                      global_phase=b['global_phase'], ancilla=n, body_instruction_count=len(body.data),
                      body_signature_sha256=C.digest(b), cosine_signature_sha256=C.digest(s),
                      phase_policy='Preserve entire saved body and global phase; no per-event phase readdition', new_certificate=False)


def task_run(case):
    c, s, g = authority(task=True)
    C.require(case in CASES, 'AXES_TASK_SCOPE')
    ns = BASE / 'run01'
    reservation = C.read(ns / 'native_tasks' / ('compile_' + case) / 'reservation.json')
    C.require(reservation['kind'] == 'compile' and reservation['case'] == case and reservation['grant_id'] == g['grant_id'], 'AXES_RESERVED_CALL')
    folder = ns / 'trajectories' / case
    folder.mkdir()
    C.save(folder / 'task_claim.json', dict(pid=os.getpid(), UTC=C.utc(), case=case))
    try:
        row = next(r for r in C.read(BASE / 'case_inputs.json')['rows'] if r['key'] == case)
        C.require(C.sha(BASE / 'case_inputs.json') == s['input_files'][str((BASE / 'case_inputs.json').relative_to(ROOT))], 'AXES_MANIFEST_BYTES')
        for path, digest in row['input_files'].items():
            C.require(C.sha(ROOT / path) == digest, 'AXES_TASK_INPUT_SHA')
        from . import h4_d1_adapter as A
        a = A.api()
        D2.S.require_compiler(a, H.fixed()['compiler'])
        C.require(a.qiskit.__version__ == '1.3.0', 'AXES_QISKIT')
        C.CallBudget(ns, LIMITS).take('saved_controlled_body_load')
        with (ROOT / row['cosine_folder'] / 'input.qpy').open('rb') as f:
            saved = a.qpy.load(f)
        C.require(len(saved) == 1 and saved[0].num_qubits == 9
                  and a.canonical_qiskit_circuit_fingerprint(saved[0]) == row['cosine_input_fingerprint'], 'AXES_SAVED_COSINE_INPUT')
        direct = None
        if row['body_qpy']:
            with (ROOT / row['body_qpy']).open('rb') as f:
                bodies = a.qpy.load(f)
            C.require(len(bodies) == 1, 'AXES_SINGLE_CONTROLLED_BODY')
            direct = bodies[0]
        body, binding = controlled_from_cosine(saved[0], direct)
        binding.update(target=row, body_replay=0, central_control_build=0)
        C.save(folder / 'source_binding.json', binding)
        C.CallBudget(ns, LIMITS).take('sine_wrapper_build')
        sine = a.hadamard_wrapper(body, 'sine', already_controlled=True)
        sig = circuit_signature(sine)
        C.require(sig['instructions'][-3] == dict(name='sdg', params=[], qubits=[8], clbits=[])
                  and sig['instructions'][1:-3] == circuit_signature(body)['instructions']
                  and sig['global_phase'] == float(body.global_phase), 'AXES_SINE_WRAPPER_POSITION_PHASE')
        inp = A.save_input(SimpleNamespace(circuit=sine, body=body, sidecar=dict(control_policy='REUSED_D2_CONTROLLED_BODY',
                           source_binding='source_binding.json', ancilla=8, phase_policy=binding['phase_policy'])), folder)
        inp.update(SOURCE=s['SOURCE'], case=case, axis='sine', trajectory_id=row['trajectory_id'], input_qpy_sha256=C.sha(folder / 'input.qpy'))
        C.save(folder / 'input.json', inp)
        C.save(folder / 'cost_transpile_started.json', dict(UTC=C.utc(), nonrefundable=True, reservation=reservation))
        m = A.compile_saved(folder)
        C.require(m['input_fingerprint'] == inp['input_fingerprint'], 'AXES_COMPILED_INPUT_FINGERPRINT')
        m.update(SOURCE=s['SOURCE'], case=case, method=row['method'], axis='sine', target=row,
                 input_qpy_sha256=inp['input_qpy_sha256'], output_qpy_sha256=C.sha(folder / 'output.qpy'),
                 runtime_manifest_sha256=s['runtime_manifest_sha256'])
        C.save(folder / 'metrics.json', m)
        C.save(folder / 'task_finished.json', dict(status='COMPLETE', pid=os.getpid(), UTC=C.utc()))
        return 0
    except BaseException:
        C.save(folder / 'task_error.json', dict(traceback=traceback.format_exc(), no_blind_retry=True, UTC=C.utc()))
        raise


def dispatch(ns, cases):
    return FunctionType(H.dispatch.__code__, dict(H.__dict__, CASES=CASES, LIMITS=LIMITS, SCRIPT=SCRIPT))(ns, cases)


def recovery_plan(original_targets, verified_successes, diagnosis):
    """Read-only recovery boundary, never retries or issues a grant itself."""
    C.require(diagnosis.get('class') == 'TECHNICAL' and bool(diagnosis.get('cause'))
              and diagnosis.get('scientific_conditions_changed') is False
              and diagnosis.get('unresolved_semantic_mismatch') is False, 'AXES_RECOVERY_DIAGNOSIS_REQUIRED')
    C.require(set(original_targets) <= set(CASES) and set(verified_successes) <= set(original_targets), 'AXES_RECOVERY_SCOPE')
    return dict(targets=[t for t in original_targets if t not in verified_successes],
                reused_verified_successes=list(verified_successes), previous_consumed_calls_preserved=True,
                new_SOURCE_seal_contract_namespace_grant_required=True, execution_enabled=False,
                scope='No new target, scientific condition, compiler or metric; no retry of validated success')


def paired_statistics(ns):
    import csv
    import numpy as np
    from .h4_conditional_cng import extract
    stats = extract('src/trotterlib/fixed_signal_evaluation.py', '_statistics', dict(np=np, math=math, Fraction=Fraction))
    paired = []
    for row in C.read(BASE / 'case_inputs.json')['rows']:
        cos = C.read(ROOT / row['cosine_folder'] / 'metrics.json')
        folder = ROOT / row['reused_sine_folder'] if row['key'] == 'PR0001' else ns / 'trajectories' / row['key']
        sine = C.read(folder / 'metrics.json')
        C.require(sine['axis'] == 'sine' and sine['compiler'] == cos['compiler'] == H.fixed()['compiler'], 'AXES_STATS_PROFILE')
        if row['key'] != 'PR0001':
            receipt = C.read(ns / 'native_tasks' / ('compile_' + row['key']) / 'native_exit.json')
            C.require(receipt['exit_code'] == 0 and receipt['native_wait'] and receipt['reaped'], 'AXES_STATS_NATIVE')
            C.require(sine['target'] == row and sine['input_qpy_sha256'] == C.sha(folder / 'input.qpy')
                      and sine['output_qpy_sha256'] == C.sha(folder / 'output.qpy'), 'AXES_STATS_BINDING')
        else:
            C.require(sine['trajectory_id'] == row['trajectory_id'], 'AXES_STATS_REUSED_ID')
        join = C.read(ROOT / row['price_join'])
        orders = Counter(e['taylor_order'] for e in join['events'])
        metrics = ('RZ', 'CX', 'depth', 'rz_depth', 'cx_depth')
        paired.append(dict(case=row['method'], index=row['trajectory_index'], trajectory_id=row['trajectory_id'],
            event_record_hash=row['event_record_hash'], trajectory_record_hash=row['trajectory_record_hash'],
            cosine={k: cos[k] for k in metrics}, sine={k: sine[k] for k in metrics},
            signed_sine_minus_cosine={k: sine[k]-cos[k] for k in metrics},
            generic_RZ=join['compiled_RZ'], order_counts={str(k): orders[k] for k in (0, 2, 4)},
            cosine_origin=row['cosine_folder'], sine_origin=str(folder.relative_to(ROOT)),
            cosine_SOURCE=cos['SOURCE'], sine_SOURCE=sine['SOURCE'], sine_reused=row['key'] == 'PR0001'))
    pr = [r for r in paired if r['case'] == 'PR']
    pf = next(r for r in paired if r['case'] == 'PF')
    C.require(len(pr) == 256 and len({r['trajectory_id'] for r in pr}) == 256
              and [r['index'] for r in pr] == list(range(256)), 'AXES_STATS_COMPLETENESS')
    values = {a: [r[a]['RZ'] for r in pr] for a in ('cosine', 'sine')}
    means = {a: Fraction(sum(v), 256) for a, v in values.items()}
    C.require(means['cosine'] == Fraction(C.read(H.BASE / 'run01/statistics.json')['central_cosine_RZ']['mean_exact']), 'AXES_OLD_COSINE_MEAN')
    covariance = sum((Fraction(x)-means['cosine'])*(Fraction(y)-means['sine']) for x,y in zip(values['cosine'], values['sine']))/255
    summary = dict(status='COMPLETE_257_PAIRED_PR256_PF1', new_unique_targets=256, initial_compile_attempts=256,
        PR=dict(n=256, axes={a: H.summarize(v, stats, np) for a,v in values.items()},
            paired_RZ_difference=H.summarize([r['signed_sine_minus_cosine']['RZ'] for r in pr], stats, np),
            sample_covariance_RZ=float(covariance), sample_covariance_RZ_exact=str(covariance),
            RZ_equal_count=sum(r['cosine']['RZ'] == r['sine']['RZ'] for r in pr),
            RZ_unequal_count=sum(r['cosine']['RZ'] != r['sine']['RZ'] for r in pr),
            equal_count_by_metric={k:sum(r['cosine'][k] == r['sine'][k] for r in pr) for k in ('RZ','CX','depth','rz_depth','cx_depth')},
            generic_comparison={}, order_groups={}), PF=pf,
        PF_over_PR_mean={a:float(Fraction(pf[a]['RZ'])/means[a]) for a in values},
        DF_saved_reference={case:C.read(H.OLD_RUN / case / 'result.json')['circuit_cost']['axes'] for case in ('df_native_PF','df_native_PR')},
        cross_representation_pairing=False, population_mean_upper=None, new_N_G=False,
        compiled_error_certificate=False, price_inputs_for_conditional_CNG_complete=True,
        precision_or_G_guarantee=False, nominal_CI='Student_t95; exact finite cohort mean; no rare-event or population upper guarantee')
    for axis in values:
        generic = [r['generic_RZ'][axis] for r in pr]
        summary['PR']['generic_comparison'][axis] = dict(saved_generic_RZ=H.summarize(generic, stats, np),
            paired_central_minus_generic_RZ=H.summarize([r[axis]['RZ']-r['generic_RZ'][axis] for r in pr], stats, np),
            aggregate_reduction_percent=100*(1-float(means[axis]/Fraction(sum(generic),256))))
    for label, group in [('only_order0',[r for r in pr if r['order_counts']['2']==r['order_counts']['4']==0]),
                         ('contains_order2',[r for r in pr if r['order_counts']['2']>0]),
                         ('contains_order4',[r for r in pr if r['order_counts']['4']>0])]:
        summary['PR']['order_groups'][label] = dict(n=len(group), axes={a:H.summarize([r[a]['RZ'] for r in group],stats,np) if group else None for a in values}, interpretation='DESCRIPTIVE_NOT_CAUSAL; absent cost unknown, not zero')
    C.save(ns / 'paired_costs.json', dict(rows=paired))
    with (ns / 'paired_costs.csv').open('x', newline='') as f:
        names = ['case','index','trajectory_id','cosine_RZ','sine_RZ','signed_sine_minus_cosine_RZ','cosine_CX','sine_CX','cosine_depth','sine_depth','generic_cosine_RZ','generic_sine_RZ','sine_reused']
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for r in paired:
            writer.writerow(dict(case=r['case'], index=r['index'], trajectory_id=r['trajectory_id'],
                cosine_RZ=r['cosine']['RZ'], sine_RZ=r['sine']['RZ'], signed_sine_minus_cosine_RZ=r['signed_sine_minus_cosine']['RZ'],
                cosine_CX=r['cosine']['CX'], sine_CX=r['sine']['CX'], cosine_depth=r['cosine']['depth'], sine_depth=r['sine']['depth'],
                generic_cosine_RZ=r['generic_RZ']['cosine'], generic_sine_RZ=r['generic_RZ']['sine'], sine_reused=r['sine_reused']))
    C.save(ns / 'statistics.json', summary)
    plot_paired(ns, pr, summary)


def plot_paired(ns, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12.5,3.7))
    for axis, color in [('cosine','#287d8e'),('sine','#b36a2a')]:
        axes[0].hist([r[axis]['RZ'] for r in rows],bins='auto',histtype='step',label=axis,color=color,linewidth=1.6)
    axes[0].set(xlabel='Compiled RZ',ylabel='Saved PR trajectories',title='Fixed cohort n=256')
    axes[0].legend()
    axes[1].hist([r['signed_sine_minus_cosine']['RZ'] for r in rows],bins='auto',color='#287d8e')
    axes[1].set(xlabel='Sine minus cosine RZ (same ID)',ylabel='Trajectories',title='Paired difference')
    axes[2].scatter([r['cosine']['RZ'] for r in rows],[r['sine']['RZ'] for r in rows],s=13,alpha=.45)
    axes[2].set(xlabel='Cosine RZ',ylabel='Sine RZ',title='Observed covariance; no tail bound')
    fig.suptitle('H4 / 1.00 A / STO-3G / Pauli central control / S2 T=.8 q8 r4 K4')
    fig.tight_layout()
    for extension in ('png','svg','pdf'):
        fig.savefig(ns/('paired_costs.'+extension),dpi=170)
    plt.close(fig)


def worker():
    ns = BASE / 'run01'
    c, s, g = authority(task=True)
    C.save(ns / 'ENGINE_CLAIM.json', dict(pid=os.getpid(), UTC=C.utc()))
    (ns / 'trajectories').mkdir()
    try:
        for start in range(0,len(CASES),c['worker_count']):
            dispatch(ns,CASES[start:start+c['worker_count']])
            C.save(ns/('wave_'+str(start)+'.json'),dict(completed_targets=list(CASES[start:start+c['worker_count']]),UTC=C.utc()))
        C.CallBudget(ns,LIMITS).take('paired_statistics_batch')
        paired_statistics(ns)
        C.save(ns/'ENGINE_STOP.json',dict(status='A_AXES_COMPLETE_MANDATORY_STOP',new_unique_successes=256,total_compile_attempts=256,technical_interruption=False,recovery_runs=0,mandatory_STOP=True))
        return 0
    except BaseException:
        C.save(ns/'ENGINE_STOP.json',dict(status='A_AXES_FAILED_DIAGNOSE_BEFORE_RECOVERY',traceback=traceback.format_exc(),mandatory_STOP=True,technical_recovery_requires_diagnosis=True))
        raise


def launch():
    labels = {(str,'NEW_A_COHORT_COS_ENGINE_ONLY'):'NEW_A_AXES_ENGINE_ONLY',
              (str,'A_COHORT_COS_COMPLETE_MANDATORY_STOP'):'A_AXES_COMPLETE_MANDATORY_STOP',
              (str,'A_COHORT_COS_FAILED_NO_RETRY'):'A_AXES_FAILED_DIAGNOSE_BEFORE_RECOVERY',
              (int,254):256}
    return rebound(H.launch,labels,BASE=BASE,SCRIPT=SCRIPT,authority=authority,verify_seal=verify_seal)()
