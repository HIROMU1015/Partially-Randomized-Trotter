"""Finite D2 connector. Authority precedes every quantum import and task.

Reuse D1's compiler, durable files, locked nonrefundable call budget, resource
observations and native witness. All old namespaces remain read-only.
"""
import ast
from fractions import Fraction
from types import SimpleNamespace
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from . import h4_d1_source_control as S

C = S.C
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/non_df_h4_d2_execution/2026-10-24'
PREP = ROOT/'artifacts/non_df_h4_d2_preparation/2026-10-24'
SCRIPT = ROOT/'scripts/run_h4_d2.py'
PYTHON = S.PYTHON
CASES = ('pauli_jw_PF', 'pauli_jw_PR')
LIMITS = dict(saved_body_validation=2, central_control_build=2, cost_transpile=2,
              generic_full_build=0, generic_baseline_compile=0, sampling=0, state_signal=0, N_G=0)
CALLS = dict(validate='saved_body_validation', build='central_control_build', compile='cost_transpile')
KIND = 'H4_D2_ACTIVE_ONE_SHOT_V1'


def scope(c):
    plan = C.read(PREP/'contract_DRAFT.json')
    C.require(c.get('kind') == KIND, 'D2_CONTRACT_KIND')
    C.require(all(c.get(k) is True for k in ('execution_authorized', 'grant_valid', 'dispatch_enabled')), 'D2_NOT_AUTHORIZED')
    for key in ('fixed', 'targets', 'future_scientific_call_limits', 'intervention', 'meaningful_difference'):
        C.require(c[key] == plan[key], 'D2_FIXED:'+key)
    C.require(c['call_limits'] == LIMITS and c['future_scientific_call_limits'] == LIMITS, 'D2_CALL_SCOPE')
    C.require(c['mandatory_STOP'] is True and c['automatic_retry'] is False
              and c['automatic_resume'] is False and c['automatic_D3'] is False, 'D2_STOP_SCOPE')


def verify_seal(s, *, runtime=True, committed=True):
    C.require(s['kind'] == 'H4_D2_SOURCE_INPUT_RUNTIME_SEAL_V1', 'D2_SEAL_KIND')
    C.require(C.sha(sys.executable) == s['python_sha256'], 'D2_PYTHON')
    for group in ('code_files', 'input_files'):
        S.D.relative_hashes(s[group])
    if committed:
        for p, h in s['code_files'].items():
            C.require(hashlib.sha256(subprocess.check_output(['git', 'show', s['SOURCE']+':'+p], cwd=ROOT)).hexdigest() == h,
                      'D2_UNCOMMITTED_CODE:'+p)
    C.require(C.sha(BASE/'approval.json') == s['approval_sha256'], 'D2_APPROVAL')
    C.require(C.sha(BASE/'preservation_before.json') == s['preservation_sha256'], 'D2_PRESERVATION_MAP')
    S.D.relative_hashes(C.read(BASE/'preservation_before.json'))
    manifest = ROOT/s['runtime_manifest']
    C.require(C.sha(manifest) == s['runtime_manifest_sha256'], 'D2_RUNTIME_MANIFEST')
    if runtime:
        C.hashes(C.read(manifest)['files'])


def prepare():
    C.require(not (BASE/'seal.json').exists() and not (BASE/'run01').exists(), 'D2_NAMESPACE_USED_NO_RETRY')
    approval = C.read(BASE/'approval.json')
    C.require(approval['verbatim_user'] == '次の作業に進んで大丈夫'
              and approval['D2_execution_authorized'] is True and approval['maximum_compile_calls'] == 2
              and approval['cases'] == list(CASES), 'D2_APPROVAL_SCOPE')
    for name, count in (('artificial/adapter.json', 12), ('artificial/control.json', 10)):
        t = C.read(BASE/name)
        C.require(t['tests'] == count and t['failures'] == 0 and t['errors'] == 0, 'D2_ARTIFICIAL_TESTS')
    prep = C.read(PREP/'preparation.json')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code = dict(prep['code_files'])
    for p in ('src/trotterlib/h4_d2_control.py', 'scripts/run_h4_d2.py', 'tests/test_h4_d2_control.py'):
        code[p] = C.sha(ROOT/p)
    inputs = dict(prep['input_files'])
    for p in (PREP/'contract_DRAFT.json', PREP/'preparation.json', BASE/'approval.json',
              BASE/'artificial/adapter.json', BASE/'artificial/control.json'):
        inputs[str(p.relative_to(ROOT))] = C.sha(p)
    protected = dict(C.read(PREP/'preservation_before.json'))
    for p in PREP.iterdir():
        if p.is_file():
            protected[str(p.relative_to(ROOT))] = C.sha(p)
    C.save(BASE/'preservation_before.json', protected)
    manifest = ROOT/'artifacts/non_df_h4_d1_execution/2026-10-24/runtime.json'
    s = dict(kind='H4_D2_SOURCE_INPUT_RUNTIME_SEAL_V1', SOURCE=source,
             prior_D1_SOURCE='9843b803878db6c3aee7ab5e7c6bb4818572c944',
             code_files=code, input_files=inputs, python_sha256=C.sha(sys.executable),
             runtime_manifest=str(manifest.relative_to(ROOT)), runtime_manifest_sha256=C.sha(manifest),
             approval_sha256=C.sha(BASE/'approval.json'), preservation_sha256=C.sha(BASE/'preservation_before.json'),
             UTC=C.utc(), real_H4_calls_before_seal=0)
    verify_seal(s)
    C.save(BASE/'seal.json', s)
    c = dict(C.read(PREP/'contract_DRAFT.json'), kind=KIND, status='APPROVED_SEALED_D2_ONE_SHOT',
             preparation_commit=approval['base_commit'],
             execution_authorized=True, grant_valid=True, dispatch_enabled=True, call_limits=LIMITS,
             scientific_SOURCE=source, science_SOURCE=source, runtime_seal=str(BASE/'seal.json'),
             approval_sha256=s['approval_sha256'], seal_sha256=C.sha(BASE/'seal.json'),
             production_dispatcher_status='IMPLEMENTED_SEALED_FOR_THIS_D2_ONLY',
             grant=str(BASE/'run01/grant.json'),
             output_namespace=str(BASE/'run01'), launch_command=[PYTHON, '-B', str(SCRIPT), '--launch'])
    # Retain the disabled proposal as an immutable reference; authority is new.
    scope(c)
    C.save(BASE/'active_contract.json', c)
    ns = BASE/'run01'
    ns.mkdir()
    g = dict(kind='H4_D2_SCIENCE_ONE_SHOT', SOURCE=source, namespace=str(ns),
             contract_sha256=C.sha(BASE/'active_contract.json'), seal_sha256=C.sha(BASE/'seal.json'),
             approval_sha256=s['approval_sha256'], one_shot=True, automatic_retry=False, call_limits=LIMITS)
    g['grant_id'] = C.digest(g)
    C.save(ns/'grant.json', g)
    C.save(ns/'call_ledger.json', dict(limits=LIMITS, used=dict.fromkeys(LIMITS, 0), events=[]))
    return g


def authority(*, consume=False, task=False):
    ns = BASE/'run01'
    c, s, g = C.read(BASE/'active_contract.json'), C.read(BASE/'seal.json'), C.read(ns/'grant.json')
    scope(c)
    C.require(g['kind'] == 'H4_D2_SCIENCE_ONE_SHOT' and g['namespace'] == str(ns), 'D2_GRANT_NAMESPACE')
    C.require(g['grant_id'] == C.digest({k:v for k,v in g.items() if k != 'grant_id'}), 'D2_GRANT_DIGEST')
    C.require(g['SOURCE'] == c['science_SOURCE'] == s['SOURCE'], 'D2_GRANT_SOURCE')
    C.require(g['contract_sha256'] == C.sha(BASE/'active_contract.json')
              and g['seal_sha256'] == C.sha(BASE/'seal.json') == c['seal_sha256'], 'D2_GRANT_BINDING')
    C.require(g['approval_sha256'] == c['approval_sha256'] == s['approval_sha256'], 'D2_APPROVAL_BINDING')
    C.require(g['one_shot'] is True and g['automatic_retry'] is False and g['call_limits'] == LIMITS, 'D2_GRANT_SCOPE')
    C.require(not (ns/'STOP.json').exists() and not (ns/'ENGINE_STOP.json').exists(), 'D2_ALREADY_STOPPED')
    verify_seal(s, runtime=not task, committed=not task)
    if task:
        C.require(C.read(ns/'CONSUMED.json')['grant_id'] == g['grant_id'] and (ns/'LAUNCH_CLAIM.json').is_file(), 'D2_TASK_NOT_LAUNCHED')
    if consume:
        C.require(not (ns/'CONSUMED.json').exists(), 'D2_CONSUMED_NO_RETRY')
        C.save(ns/'CONSUMED.json', dict(grant_id=g['grant_id'], pid=os.getpid(), UTC=C.utc(), nonrefundable=True))
    return c, s, g


def validate_body(target, folder, a):
    from .h4_d2_pauli_control import _tokens
    source = ROOT/target['body_qpy']
    C.require(C.sha(source) == target['body_qpy_sha256'], 'D2_SAVED_BODY_SHA')
    with source.open('rb') as f:
        saved = a.qpy.load(f)
    C.require(len(saved) == 1 and saved[0].num_qubits == 8 and saved[0].num_clbits == 0, 'D2_BODY_SHAPE')
    body = saved[0]
    tokens, phase = _tokens(body)  # Domain guard; no circuit is emitted.
    inspector = S.inspector()
    actual = [inspector.primitive_token(item, body) for item in body.data]
    model = C.read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json')
    join = C.read(ROOT/target['price_join'])
    a.verify_join(join, expected_case=target['case'])
    expected, phase_ledger = inspector.pauli_recipe(model, join['method'], join)
    C.require(actual == expected, 'D2_SOURCE_INSTRUCTION_TOKEN_MISMATCH')
    prior = C.read(ROOT/target['D1_input_metadata'])
    C.require(phase == prior['sidecar']['naming']['global_phase'], 'D2_SAVED_AGGREGATE_PHASE_MISMATCH')
    evidence = C.read(ROOT/'artifacts/non_df_h4_d1_reproducibility_design/2026-10-24/saved_QPY_structure.json')[target['case']]
    C.require(evidence['current_body_phase_match'] and evidence['current_base_body_token_match']
              and evidence['body_global_phase'] == phase and evidence['body_qpy_sha256'] == C.sha(source), 'D2_PRIOR_SOURCE_B_EDGE')
    shutil.copyfile(source, folder/'source_body.qpy')
    data = dict(status='PASS_SAVED_BODY_SOURCE_B', body_qpy_sha256=C.sha(source),
                source_instruction_count=len(tokens), actual_tokens_sha256=C.digest(actual),
                expected_tokens_sha256=C.digest(expected), phase_update_SOURCE_ledger=phase_ledger,
                aggregate_phase=phase, historical_hash_match=target['historical_hash_match'],
                historical_root_cause='UNKNOWN_UNCHANGED', historical_program_identity='NOT_ESTABLISHED', new_certificate=False)
    C.save(folder/'validation.json', data)
    return data


def task_run(kind, case):
    c, s, g = authority(task=True)
    C.require(kind in CALLS and case in CASES, 'D2_TASK_SCOPE')
    ns = BASE/'run01'
    folder = ns/'variants'/case
    reservation = C.read(ns/'native_tasks'/(kind+'_'+case)/'reservation.json')
    C.require(reservation['kind'] == kind and reservation['case'] == case
              and reservation['grant_id'] == g['grant_id'], 'D2_TASK_NOT_RESERVED')
    if kind == 'validate':
        folder.mkdir()
    elif kind == 'build':
        C.require(C.read(ns/'stage_A.json')['status'] == 'PASS_BOTH_SAVED_BODY_SOURCE_B', 'D2_BUILD_BEFORE_BOTH_VALIDATED')
    else:
        C.require(C.read(ns/'stage_B.json')['status'] == 'PASS_BOTH_CONTROLLED_INPUTS_SAVED', 'D2_COMPILE_BEFORE_BOTH_BUILT')
    C.save(folder/(kind+'_claim.json'), dict(pid=os.getpid(), UTC=C.utc(), case=case))
    try:
        from . import h4_d1_adapter as A
        a = A.api()
        C.require(a.qiskit.__version__ == '1.3.0', 'D2_QISKIT_VERSION')
        S.require_compiler(a, c['fixed']['compiler'])
        target = next(t for t in c['targets'] if t['case'] == case)
        if kind == 'validate':
            validate_body(target, folder, a)
        elif kind == 'build':
            from .h4_d2_pauli_control import full_wrapper
            valid = C.read(folder/'validation.json')
            C.require(C.sha(folder/'source_body.qpy') == valid['body_qpy_sha256'], 'D2_LOCAL_BODY_CHANGED')
            with (folder/'source_body.qpy').open('rb') as f:
                body = a.qpy.load(f)[0]
            circuit, result = full_wrapper(body)
            data = A.save_input(SimpleNamespace(circuit=circuit, body=result.body, sidecar=result.sidecar), folder)
            with (folder/'controlled_body.qpy').open('xb') as f:
                a.qpy.dump(result.body, f)
            data.update(SOURCE=s['SOURCE'], case=case, input_qpy_sha256=C.sha(folder/'input.qpy'),
                        source_body_qpy_sha256=valid['body_qpy_sha256'],
                        historical_input_fingerprint=target['historical_input_fingerprint'],
                        historical_hash_match=target['historical_hash_match'],
                        generic_current_input_fingerprint=target['baseline_current_input_fingerprint'],
                        new_input_need_not_equal_generic=True)
            C.save(folder/'input.json', data)
        else:
            inp = C.read(folder/'input.json')
            C.require(C.sha(folder/'input.qpy') == inp['input_qpy_sha256'], 'D2_LOCAL_INPUT_CHANGED')
            C.save(folder/'cost_transpile_started.json', dict(UTC=C.utc(), nonrefundable=True))
            metrics = A.compile_saved(folder)
            C.require(metrics['input_fingerprint'] == inp['input_fingerprint'], 'D2_COMPILED_INPUT_FINGERPRINT')
            metrics.update(SOURCE=s['SOURCE'], case=case, generic_baseline_RZ=target['baseline_RZ_quote'])
            C.save(folder/'metrics.json', metrics)
        C.save(folder/(kind+'_finished.json'), dict(pid=os.getpid(), UTC=C.utc(), scope='NEW_D2_TASK_ONLY'))
        return 0
    except BaseException:
        C.save(folder/(kind+'_error.json'), dict(traceback=traceback.format_exc(), UTC=C.utc(), no_retry=True))
        raise


def batch(ns, kind, cases):
    C.require(kind in CALLS and len(set(cases)) == len(cases) and set(cases) <= set(CASES), 'D2_BATCH_SCOPE')
    children = []
    try:
        for case in cases:
            C.CallBudget(ns, LIMITS).take(CALLS[kind])
            d = ns/'native_tasks'/(kind+'_'+case)
            d.mkdir(parents=True)
            C.save(d/'reservation.json', dict(kind=kind, case=case, grant_id=C.read(ns/'grant.json')['grant_id'], UTC=C.utc()))
            command = [PYTHON, '-B', str(SCRIPT), '--task', '--kind', kind, '--case', case]
            with (d/'stdout.log').open('xb') as out, (d/'stderr.log').open('xb') as err:
                child = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            children.append((case, child, d))
            C.save(d/'launch.json', dict(pid=child.pid, identity=C._proc_identity(child.pid), argv=command, UTC=C.utc()))
        last = 0
        while any(p.poll() is None for _, p, _ in children):
            if time.monotonic()-last > 30:
                S.D.record_resources(ns)
                last = time.monotonic()
            time.sleep(.2)
        codes = {}
        for case, child, d in children:
            codes[case] = child.wait()
            C.save(d/'native_exit.json', dict(pid=child.pid, exit_code=codes[case], native_wait=True, reaped=True, UTC=C.utc()))
        C.require(all(v == 0 for v in codes.values()), 'D2_TASK_FAILURE:'+str(codes))
        return codes
    finally:
        for _, child, _ in children:
            if child.poll() is None:
                child.terminate()
        for _, child, d in children:
            if not (d/'native_exit.json').exists():
                code = child.wait()
                C.save(d/'native_exit.json', dict(pid=child.pid, exit_code=code, native_wait=True, reaped=True, UTC=C.utc()))


def comparisons(metrics, targets):
    rows = {}
    for t in targets:
        before = C.read(ROOT/t['D1_metrics'])
        after = metrics[t['case']]
        rows[t['case']] = {}
        for key in ('RZ', 'CX', 'depth', 'rz_depth', 'cx_depth'):
            b, a = before[key], after[key]
            rows[t['case']][key] = dict(generic=b, central=a, signed_central_minus_generic=a-b,
                central_over_generic_exact=str(Fraction(a, b)) if b else None,
                ratio_denominator='D1 generic '+key)
    return rows


def worker():
    ns = BASE/'run01'
    c, s, g = authority(task=True)
    C.save(ns/'ENGINE_CLAIM.json', dict(pid=os.getpid(), UTC=C.utc()))
    (ns/'variants').mkdir()
    stage = 'A'
    try:
        batch(ns, 'validate', CASES)
        C.save(ns/'stage_A.json', dict(status='PASS_BOTH_SAVED_BODY_SOURCE_B', cases=list(CASES), new_certificate=False))
        stage = 'B'
        batch(ns, 'build', CASES)
        C.save(ns/'stage_B.json', dict(status='PASS_BOTH_CONTROLLED_INPUTS_SAVED', cases=list(CASES)))
        stage = 'C'
        batch(ns, 'compile', CASES)
        metrics = {case:C.read(ns/'variants'/case/'metrics.json') for case in CASES}
        C.save(ns/'comparisons.json', dict(metrics=metrics, comparisons=comparisons(metrics, c['targets']),
            scope='FIXED_SAME_BODY_IMPLEMENTATION_PLUS_COMPILER_NOT_REPRESENTATION_ONLY_OR_POPULATION'))
        C.save(ns/'ENGINE_STOP.json', dict(status='D2_COMPLETE_MANDATORY_STOP', stage=stage, cost_compiles=2, mandatory_STOP=True))
        return 0
    except BaseException:
        C.save(ns/'ENGINE_STOP.json', dict(status='D2_FAILED_NO_RETRY', stage=stage,
            traceback=traceback.format_exc(), mandatory_STOP=True))
        raise


def launch():
    ns = BASE/'run01'
    c, s, g = authority(consume=True)
    C.save(ns/'LAUNCH_CLAIM.json', dict(pid=os.getpid(), UTC=C.utc(), grant_id=g['grant_id']))
    tree = ast.parse((ROOT/'src/trotterlib/h4_certificate_control.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'witness')
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and n.value == 'NEW_C_WORKER_ONLY':
            n.value = 'NEW_D2_ENGINE_ONLY'
    env = dict(vars(C))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reused D2 native witness>', 'exec'), env)
    status = env['witness'](ns/'native', [PYTHON, '-B', str(SCRIPT), '--worker'])
    problem = None
    try:
        verify_seal(s)
    except BaseException as exc:
        problem = repr(exc)
    engine = C.read(ns/'ENGINE_STOP.json') if (ns/'ENGINE_STOP.json').exists() else dict(status='MISSING_ENGINE_STOP')
    native = C.read(ns/'native/worker_native_exit.json')
    stop = C.read(ns/'native/STOP.json')
    tasks = [C.read(p) for p in (ns/'native_tasks').glob('*/native_exit.json')]
    success = (status == 0 and problem is None and stop['all_owned_reaped'] and len(tasks) == 6
               and all(t['exit_code'] == 0 and t['reaped'] for t in tasks) and engine['status'] == 'D2_COMPLETE_MANDATORY_STOP')
    C.save(ns/'STOP.json', dict(status='D2_COMPLETE_MANDATORY_STOP' if success else 'D2_FAILED_NO_RETRY',
        native_engine_exit=native, all_owned_reaped=stop['all_owned_reaped'], direct_task_native_receipts=tasks,
        preservation_error=problem, old_preservation_PASS=problem is None, ledger=C.read(ns/'call_ledger.json'),
        mandatory_STOP=True, automatic_retry=False, original_H4_native_verified=False, formal_certificate_accepted=False, UTC=C.utc()))
    C.save(ns/'output_inventory.json', dict(files={str(p.relative_to(ns)):C.sha(p) for p in ns.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name != 'output_inventory.json'}))
    return dict(status=C.read(ns/'STOP.json')['status'], native_engine_exit=native['exit_code'],
                ledger=C.read(ns/'call_ledger.json')['used'])
