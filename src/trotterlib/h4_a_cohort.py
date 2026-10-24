"""Saved Pauli PR cosine cohort: 254 new tasks, two reused outputs, one batch.

Authority imports only stdlib. Reuse A_GATE body-only replay, D2 lift/batch,
D1 QPY compiler, locked CallBudget, native witness and existing t statistics.
"""
import ast
from collections import Counter
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import FunctionType, SimpleNamespace
from . import h4_a_gate as G

C = G.C
D2 = G.D2
ROOT = G.ROOT
BASE = ROOT / 'artifacts/non_df_h4_a_cohort_cos/2026-10-24'
SCRIPT = ROOT / 'scripts/run_h4_a_cohort_cos.py'
PYTHON = G.PYTHON
CASES = tuple(f'{i:04d}' for i in range(2, 256))
COHORT = G.DESIGN / 'cohort_manifest.json'
KIND = 'H4_A_COHORT_COS_ACTIVE_V1'
LIMITS = dict(body_replay=254, central_control_build=254, cosine_wrapper_build=254,
              cost_transpile=254, cost_statistics_batch=1, sampling=0,
              sine_compile=0, generic_control_build=0, state_signal=0, N_G=0)
JOB_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               QISKIT_PARALLEL='FALSE')
OLD_RUN = ROOT / 'artifacts/non_df_h4_pilot/2026-10-23/run01/run'


def records():
    rows = C.read(COHORT)['records']
    C.require(len(rows) == 256 and [r['trajectory_index'] for r in rows] == list(range(256)), 'COHORT_INDICES')
    C.require(len({r['trajectory_id'] for r in rows}) == 256, 'COHORT_UNIQUE_IDS')
    return rows


def fixed():
    value = dict(C.read(D2.BASE / 'active_contract.json')['fixed'])
    C.require(value['axis'] == 'cosine', 'COSINE_ONLY')
    return value


def scope(c):
    C.require(c['kind'] == KIND and all(c[k] is True for k in ('execution_authorized', 'grant_valid', 'dispatch_enabled')), 'COHORT_AUTHORITY')
    C.require(c['fixed'] == fixed() and c['new_indices'] == list(range(2, 256)) and c['reuse_indices'] == [0, 1], 'COHORT_FIXED_SCOPE')
    C.require(c['call_limits'] == LIMITS and c['axis'] == 'cosine' and c['per_trajectory_compile_max'] == 1, 'COHORT_CALL_SCOPE')
    C.require(c['mandatory_STOP'] is True and c['automatic_retry'] is False and c['automatic_resume'] is False
              and c['automatic_next_stage'] is False, 'COHORT_STOP_SCOPE')
    C.require(c['job_environment'] == JOB_ENV and isinstance(c['worker_count'], int) and 1 <= c['worker_count'] <= len(CASES), 'COHORT_RESOURCES')


def verify_seal(s, *, runtime=True, committed=True):
    C.require(s['kind'] == 'H4_A_COHORT_COS_SEAL_V1', 'COHORT_SEAL_KIND')
    fun = FunctionType(D2.verify_seal.__code__, dict(D2.__dict__, BASE=BASE))
    fun(dict(s, kind='H4_D2_SOURCE_INPUT_RUNTIME_SEAL_V1'), runtime=runtime, committed=committed)


def reuse_results():
    """Metadata/bytes only: no QPY load, new circuit, compile or statistics."""
    rows = records()
    result = []
    for index, base, folder, expected_source, expected_RZ in (
        (0, D2.BASE, D2.BASE / 'run01/variants/pauli_jw_PR', '74515d8eb4f073c40fc783681a3775dbca89e9b1', 1145),
        (1, G.BASE, G.BASE / 'run01/variants/cosine', 'ee7d281b5daead1d0d5e81e00c9a0ed4727792b0', 1129),
    ):
        contract, seal = C.read(base / 'active_contract.json'), C.read(base / 'seal.json')
        C.require(contract['science_SOURCE'] == seal['SOURCE'] == expected_source, 'REUSE_SOURCE')
        old_fixed = dict(contract['fixed'])
        old_fixed.pop('axes', None)
        old_fixed['axis'] = 'cosine'
        C.require(old_fixed == fixed(), 'REUSE_FIXED_PROFILE')
        target = next(t for t in contract['targets'] if t['case'] == 'pauli_jw_PR') if index == 0 else contract['target']
        C.require(target['price_join'] == rows[index]['price_join'], 'REUSE_TRAJECTORY')
        stop = C.read(base / 'run01/STOP.json')
        expected_stop = 'D2_COMPLETE_MANDATORY_STOP' if index == 0 else 'A_GATE_COMPLETE_MANDATORY_STOP'
        C.require(stop['status'] == expected_stop and stop['native_engine_exit']['exit_code'] == 0 and stop['all_owned_reaped'], 'REUSE_NATIVE')
        inventory = C.read(base / 'run01/output_inventory.json')['files']
        for name in ('input.qpy', 'output.qpy', 'input.json', 'metrics.json'):
            p = folder / name
            C.require(C.sha(p) == inventory[str(p.relative_to(base / 'run01'))], 'REUSE_BYTES:' + name)
        inp, metrics = C.read(folder / 'input.json'), C.read(folder / 'metrics.json')
        C.require(metrics['RZ'] == expected_RZ and metrics['compiler'] == fixed()['compiler'], 'REUSE_METRICS')
        C.require(metrics['input_fingerprint'] == inp['input_fingerprint'] and C.sha(folder / 'input.qpy') == inp['input_qpy_sha256'], 'REUSE_INPUT')
        if index == 1:
            C.require(metrics['trajectory_id'] == rows[index]['trajectory_id'] and metrics['axis'] == 'cosine', 'REUSE_ID_AXIS')
        else:
            C.require(metrics['case'] == 'pauli_jw_PR', 'REUSE_D2_CASE')
        result.append(dict(index=index, target=rows[index], SOURCE=expected_source,
                           folder=str(folder.relative_to(ROOT)), metrics=metrics,
                           input_qpy_sha256=C.sha(folder / 'input.qpy'), output_qpy_sha256=C.sha(folder / 'output.qpy'),
                           new_scientific_calls=0, status='PASS_REUSE_SOURCE_INPUT_PROFILE_NATIVE'))
    return result


def prepare():
    C.require(not (BASE / 'seal.json').exists() and not (BASE / 'run01').exists(), 'COHORT_NAMESPACE_USED')
    approval = C.read(BASE / 'approval.json')
    text = (BASE / 'user_instruction.txt').read_text()
    C.require('A_COHORT_COSの限定科学実行を承認' in text and '254本のみ' in text
              and approval['instruction_sha256'] == C.sha(BASE / 'user_instruction.txt'), 'COHORT_EXPLICIT_APPROVAL')
    tests = C.read(BASE / 'artificial/new_tests.json')
    C.require(tests['tests'] == 12 and tests['errors'] == tests['failures'] == 0, 'COHORT_ARTIFICIAL_TESTS')
    old = C.read(G.BASE / 'seal.json')
    G.verify_seal(old)
    D2.verify_seal(C.read(D2.BASE / 'seal.json'), runtime=False)
    reuse = reuse_results()
    C.save(BASE / 'reuse_compatibility.json', dict(status='PASS', records=reuse, new_scientific_calls=0))
    code, inputs = dict(old['code_files']), dict(old['input_files'])
    for p in ('src/trotterlib/h4_a_cohort.py', 'scripts/run_h4_a_cohort_cos.py', 'tests/test_h4_a_cohort.py', 'src/trotterlib/h4_conditional_cng.py'):
        code[p] = C.sha(ROOT / p)
    inputs[str(COHORT.relative_to(ROOT))] = C.sha(COHORT)
    for row in records():
        C.require(C.sha(ROOT / row['price_join']) == row['price_join_sha256'], 'COHORT_JOIN_SHA')
        inputs[row['price_join']] = row['price_join_sha256']
    for p in (BASE / 'approval.json', BASE / 'user_instruction.txt', BASE / 'output_schema.json', BASE / 'artificial/new_tests.json',
              BASE / 'resource_plan.json', BASE / 'reuse_compatibility.json', D2.BASE / 'active_contract.json', G.BASE / 'active_contract.json',
              D2.BASE / 'run01/variants/pauli_jw_PF/metrics.json',
              *(OLD_RUN / case / 'result.json' for case in ('pauli_jw_PR', 'df_native_PF', 'df_native_PR'))):
        inputs[str(p.relative_to(ROOT))] = C.sha(p)
    protected = dict(C.read(G.BASE / 'preservation_before.json'))
    for p in G.BASE.rglob('*'):
        if p.is_file() and 'mplconfig' not in p.parts and '__pycache__' not in p.parts:
            protected[str(p.relative_to(ROOT))] = C.sha(p)
    C.save(BASE / 'preservation_before.json', protected)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    seal = dict(kind='H4_A_COHORT_COS_SEAL_V1', SOURCE=source, code_files=code, input_files=inputs,
                python_sha256=C.sha(sys.executable), runtime_manifest=old['runtime_manifest'],
                runtime_manifest_sha256=old['runtime_manifest_sha256'], approval_sha256=C.sha(BASE / 'approval.json'),
                preservation_sha256=C.sha(BASE / 'preservation_before.json'), real_H4_calls_before_seal=0, UTC=C.utc())
    verify_seal(seal)
    C.save(BASE / 'seal.json', seal)
    plan = C.read(BASE / 'resource_plan.json')
    contract = dict(kind=KIND, science_SOURCE=source, execution_authorized=True, grant_valid=True, dispatch_enabled=True,
                    fixed=fixed(), axis='cosine', new_indices=list(range(2, 256)), reuse_indices=[0, 1],
                    call_limits=LIMITS, per_trajectory_compile_max=1, worker_count=plan['worker_count'], job_environment=JOB_ENV,
                    automatic_retry=False, automatic_resume=False, automatic_next_stage=False, mandatory_STOP=True,
                    seal_sha256=C.sha(BASE / 'seal.json'), approval_sha256=seal['approval_sha256'],
                    cohort_manifest_sha256=C.sha(COHORT), output_schema_sha256=C.sha(BASE / 'output_schema.json'),
                    output_namespace=str(BASE / 'run01'))
    scope(contract)
    C.save(BASE / 'active_contract.json', contract)
    ns = BASE / 'run01'
    ns.mkdir()
    grant = dict(kind='H4_A_COHORT_COS_ONE_SHOT', SOURCE=source, namespace=str(ns),
                 contract_sha256=C.sha(BASE / 'active_contract.json'), seal_sha256=C.sha(BASE / 'seal.json'),
                 approval_sha256=seal['approval_sha256'], one_shot=True, automatic_retry=False, call_limits=LIMITS)
    grant['grant_id'] = C.digest(grant)
    C.save(ns / 'grant.json', grant)
    C.save(ns / 'call_ledger.json', dict(limits=LIMITS, used=dict.fromkeys(LIMITS, 0), events=[]))
    return grant


def authority(*, consume=False, task=False):
    ns = BASE / 'run01'
    c, s, g = C.read(BASE / 'active_contract.json'), C.read(BASE / 'seal.json'), C.read(ns / 'grant.json')
    scope(c)
    C.require(g['kind'] == 'H4_A_COHORT_COS_ONE_SHOT' and g['namespace'] == str(ns), 'COHORT_GRANT_NAMESPACE')
    C.require(g['grant_id'] == C.digest({k: v for k, v in g.items() if k != 'grant_id'}), 'COHORT_GRANT_DIGEST')
    C.require(g['SOURCE'] == c['science_SOURCE'] == s['SOURCE'] and g['call_limits'] == LIMITS, 'COHORT_GRANT_SOURCE')
    C.require(g['contract_sha256'] == C.sha(BASE / 'active_contract.json')
              and g['seal_sha256'] == c['seal_sha256'] == C.sha(BASE / 'seal.json')
              and g['approval_sha256'] == c['approval_sha256'] == s['approval_sha256'], 'COHORT_GRANT_BINDING')
    C.require(g['one_shot'] is True and g['automatic_retry'] is False, 'COHORT_ONE_SHOT')
    C.require(not (ns / 'STOP.json').exists() and not (ns / 'ENGINE_STOP.json').exists(), 'COHORT_ALREADY_STOPPED')
    if task:
        # Full runtime/all-input/preservation checks occur once before and after
        # execution. Each child checks the used code, model and its own join.
        C.require(C.sha(sys.executable) == s['python_sha256'], 'COHORT_PYTHON')
        D2.S.D.relative_hashes(s['code_files'])
        for p in (G.MODEL, str(COHORT.relative_to(ROOT))):
            C.require(C.sha(ROOT / p) == s['input_files'][p], 'COHORT_SHARED_INPUT')
        C.require(C.read(ns / 'CONSUMED.json')['grant_id'] == g['grant_id'] and (ns / 'LAUNCH_CLAIM.json').exists(), 'COHORT_NOT_LAUNCHED')
    else:
        verify_seal(s)
    if consume:
        C.require(not (ns / 'CONSUMED.json').exists(), 'COHORT_CONSUMED_NO_RETRY')
        C.save(ns / 'CONSUMED.json', dict(grant_id=g['grant_id'], pid=os.getpid(), nonrefundable=True, UTC=C.utc()))
    return c, s, g


def validate_join(row, model, a):
    C.require(C.sha(ROOT / row['price_join']) == row['price_join_sha256'], 'COHORT_JOIN_BYTES')
    join = C.read(ROOT / row['price_join'])
    a.verify_join(join, expected_case='pauli_jw_PR')
    for key in ('trajectory_index', 'trajectory_id', 'event_record_hash', 'trajectory_record_hash'):
        C.require(join[key] == row[key], 'COHORT_JOIN_BINDING:' + key)
    C.require(join['binding']['SOURCE'] == '0c329b30a81accb35c212be8defa66fee0dd0f12' and join['q'] == 8 and join['r'] == 4, 'COHORT_SAVED_SOURCE')
    case = model['cases']['PR']
    C.require(model['n_qubits'] == 8 and model['step_time'] == .1 and len(model['hamiltonian']['terms']) == 184
              and len(case['blocks']) == 27 and all(case['config'][k] == v for k, v in [('T', .8), ('q', 8), ('r', 4), ('K', 4)]), 'COHORT_MODEL')
    for event in join['events']:
        C.require(event['taylor_order'] in (0, 2, 4), 'COHORT_SAVED_ORDER')
        for app in event['application_sequence']:
            comp = case['components'][int(app['component_id'])]
            C.require(comp['sign'] == app['coefficient_sign'] and comp['coefficient_abs'] == app['coefficient_abs']
                      and set(comp['label']) != {'I'}, 'COHORT_COMPONENT')
    return join


def task_run(case):
    c, s, g = authority(task=True)
    C.require(case in CASES, 'COHORT_TASK_INDEX')
    ns = BASE / 'run01'
    row = records()[int(case)]
    reservation = C.read(ns / 'native_tasks' / ('compile_' + case) / 'reservation.json')
    C.require(reservation['kind'] == 'compile' and reservation['case'] == case and reservation['grant_id'] == g['grant_id'], 'COHORT_TASK_RESERVATION')
    folder = ns / 'trajectories' / case
    folder.mkdir()  # Exclusive trajectory claim, before any quantum import.
    C.save(folder / 'task_claim.json', dict(pid=os.getpid(), trajectory_id=row['trajectory_id'], UTC=C.utc()))
    try:
        from . import h4_d1_adapter as A
        from .h4_d2_pauli_control import central_control_body
        a = A.api()
        D2.S.require_compiler(a, c['fixed']['compiler'])
        C.require(a.qiskit.__version__ == '1.3.0', 'COHORT_QISKIT')
        raw = C.read(ROOT / G.MODEL)
        join = validate_join(row, raw, a)
        model = dict(representation='pauli_jw', n_qubits=8, step_time=raw['step_time'],
                     hamiltonian=dict(constant=raw['hamiltonian']['constant']),
                     cases={'PR': {k: raw['cases']['PR'][k] for k in ('config', 'blocks', 'components')}})
        inspector = D2.S.inspector()
        expected, rules = inspector.pauli_recipe(model, 'PR', join)
        budget = C.CallBudget(ns, LIMITS)
        budget.take('body_replay')
        replay = G.body_only(model, join)
        actual = [inspector.primitive_token(item, replay.body) for item in replay.body.data]
        binding = dict(status='PASS_SOURCE_TOKENS' if actual == expected else 'FAIL_SOURCE_TOKENS',
                       target=row, actual_tokens_sha256=C.digest(actual), expected_tokens_sha256=C.digest(expected),
                       numeric_tokens_match=actual == expected, source_instruction_count=len(actual),
                       replay_sidecar=replay.sidecar, phase_SOURCE_rules=rules,
                       historical_hash='FAIL_UNCHANGED', historical_cause='UNKNOWN', new_certificate=False)
        C.save(folder / 'source_binding.json', binding)
        C.require(actual == expected, 'COHORT_BODY_SOURCE_TOKENS')
        with (folder / 'source_body.qpy').open('xb') as f:
            a.qpy.dump(replay.body, f)
        budget.take('central_control_build')
        result = central_control_body(replay.body)
        C.require(result.sidecar['source_instruction_count'] == len(actual)
                  and result.sidecar['aggregate_source_global_phase'] == replay.sidecar['aggregate_phase'], 'COHORT_PHASE_MAPPING')
        C.save(folder / 'controlled_mapping.json', result.sidecar)
        budget.take('cosine_wrapper_build')
        circuit = a.hadamard_wrapper(result.body, 'cosine', already_controlled=True)
        sidecar = {k: v for k, v in result.sidecar.items() if k != 'instruction_mapping'}
        sidecar['instruction_mapping_file'] = 'controlled_mapping.json'
        inp = A.save_input(SimpleNamespace(circuit=circuit, body=result.body, sidecar=sidecar), folder)
        inp.update(SOURCE=s['SOURCE'], axis='cosine', trajectory_id=row['trajectory_id'],
                   input_qpy_sha256=C.sha(folder / 'input.qpy'), source_body_sha256=C.sha(folder / 'source_body.qpy'))
        C.save(folder / 'input.json', inp)
        C.save(folder / 'cost_transpile_started.json', dict(UTC=C.utc(), nonrefundable=True, reservation=reservation))
        metrics = A.compile_saved(folder)
        C.require(metrics['input_fingerprint'] == inp['input_fingerprint'], 'COHORT_COMPILED_INPUT')
        metrics.update(SOURCE=s['SOURCE'], axis='cosine', target=row, generic_cosine_RZ=join['compiled_RZ']['cosine'],
                       event_order_counts=dict(Counter(str(e['taylor_order']) for e in join['events'])),
                       input_qpy_sha256=inp['input_qpy_sha256'], output_qpy_sha256=C.sha(folder / 'output.qpy'),
                       runtime_manifest_sha256=s['runtime_manifest_sha256'])
        C.save(folder / 'metrics.json', metrics)
        C.save(folder / 'task_finished.json', dict(status='COMPLETE', pid=os.getpid(), UTC=C.utc()))
        return 0
    except BaseException:
        C.save(folder / 'task_error.json', dict(traceback=traceback.format_exc(), no_retry=True, UTC=C.utc()))
        raise


def dispatch(ns, cases):
    fun = FunctionType(D2.batch.__code__, dict(D2.__dict__, CASES=CASES, LIMITS=LIMITS,
                                              CALLS=dict(compile='cost_transpile'), SCRIPT=SCRIPT))
    return fun(ns, 'compile', cases)


def summarize(values, stats, np):
    output = stats(values)
    output.update(n=len(values), sample_standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                  minimum=min(values), maximum=max(values),
                  quantile_definition='numpy.linear', quantiles={str(q): float(np.quantile(values, q, method='linear'))
                      for q in (.05, .25, .5, .75, .95, .99)},
                  nominal95_interval=[output['mean'] - output['nominal95_halfwidth'], output['mean'] + output['nominal95_halfwidth']]
                      if output['nominal95_halfwidth'] is not None else None)
    return output


def statistics_batch(ns, reuse):
    """Called once after all 254 native exits. No new quantum operation."""
    import csv
    import numpy as np
    from .h4_conditional_cng import extract
    stats = extract('src/trotterlib/fixed_signal_evaluation.py', '_statistics', dict(np=np, math=math, Fraction=Fraction))
    output = []
    for record in records():
        i = record['trajectory_index']
        if i < 2:
            saved = reuse[i]
            m = saved['metrics']
            origin = saved['folder']
        else:
            folder = ns / 'trajectories' / f'{i:04d}'
            m = C.read(folder / 'metrics.json')
            C.require(m['target'] == record and C.sha(folder / 'input.qpy') == m['input_qpy_sha256']
                      and C.sha(folder / 'output.qpy') == m['output_qpy_sha256'], 'COHORT_STATS_SAVED_BINDING')
            receipt = C.read(ns / 'native_tasks' / ('compile_' + f'{i:04d}') / 'native_exit.json')
            C.require(receipt['exit_code'] == 0 and receipt['reaped'] and receipt['native_wait'], 'COHORT_STATS_NATIVE')
            origin = str(folder.relative_to(ROOT))
        C.require(m['compiler'] == fixed()['compiler'], 'COHORT_STATS_COMPILER')
        join = C.read(ROOT / record['price_join'])
        orders = Counter(e['taylor_order'] for e in join['events'])
        old, new = join['compiled_RZ']['cosine'], m['RZ']
        output.append(dict(index=i, trajectory_id=record['trajectory_id'], event_record_hash=record['event_record_hash'],
                           trajectory_record_hash=record['trajectory_record_hash'], axis='cosine', reused=i < 2,
                           SOURCE=m['SOURCE'], origin=origin, RZ=new, CX=m['CX'], depth=m['depth'],
                           rz_depth=m['rz_depth'], cx_depth=m['cx_depth'], generic_RZ=old,
                           signed_central_minus_generic_RZ=new-old,
                           reduction_percent=100 * (1 - new / old), reduction_fraction_exact=str(1 - Fraction(new, old)),
                           order0_events=orders[0], order2_events=orders[2], order4_events=orders[4]))
    C.require(len(output) == 256 and len({r['trajectory_id'] for r in output}) == 256, 'COHORT_STATS_COMPLETENESS')
    x, old = [r['RZ'] for r in output], [r['generic_RZ'] for r in output]
    baseline = C.read(OLD_RUN / 'pauli_jw_PR/result.json')['circuit_cost']['axes']['cosine']
    C.require(Fraction(sum(old), 256) == Fraction(baseline['mean_exact']) == Fraction('9321.1015625'), 'COHORT_PAIRED_GENERIC_MEAN')
    pf = C.read(D2.BASE / 'run01/variants/pauli_jw_PF/metrics.json')['RZ']
    df_pf = C.read(OLD_RUN / 'df_native_PF/result.json')['circuit_cost']['axes']['cosine']['mean']
    df_pr = C.read(OLD_RUN / 'df_native_PR/result.json')['circuit_cost']['axes']['cosine']['mean']
    C.require(pf == 15028 and df_pf == 135355 and df_pr == 23678.44140625, 'COHORT_BASELINE_QUOTES')
    mean = Fraction(sum(x), 256)
    summary = dict(status='COMPLETE_EXACT_SAVED_COHORT', n=256, new_successes=254, reused=2,
                   central_cosine_RZ=summarize(x, stats, np), generic_cosine_RZ=summarize(old, stats, np),
                   paired_signed_difference=summarize([r['signed_central_minus_generic_RZ'] for r in output], stats, np),
                   individual_reduction_percent=summarize([r['reduction_percent'] for r in output], stats, np),
                   aggregate_mean_reduction_percent=100 * (1 - float(mean / Fraction(sum(old), 256))),
                   aggregate_reduction_fraction_exact=str(1 - mean / Fraction(sum(old), 256)),
                   central_PR_over_PF_exact=str(mean / pf), central_PF_over_PR=float(pf / mean),
                   baseline_RZ=dict(pauli_central_PF=pf, df_native_PF=df_pf, df_native_PR_saved_mean=df_pr),
                   cross_representation_pairing=False, order_groups={}, new_N_G=False,
                   nominal_interval='Student_t_95_df255_NOT_POPULATION_UPPER_OR_RARE_EVENT_GUARANTEE',
                   population_mean_upper=None, compiled_error_certificate=False,
                   uncertainty='Exact finite cohort mean; nominal MC inference only; different Pauli/DF laws and implementation/compiler dependence retained')
    for label, selected in [('only_order0', [r for r in output if r['order2_events'] == r['order4_events'] == 0]),
                            ('contains_order2', [r for r in output if r['order2_events'] > 0]),
                            ('contains_order4', [r for r in output if r['order4_events'] > 0])]:
        summary['order_groups'][label] = dict(n=len(selected), central_RZ=summarize([r['RZ'] for r in selected], stats, np) if selected else None,
            mean_reduction_percent=float(np.mean([r['reduction_percent'] for r in selected])) if selected else None,
            interpretation='DESCRIPTIVE_NOT_CAUSAL; absent stratum is unknown, never zero cost')
    C.save(ns / 'cohort_costs.json', dict(rows=output))
    with (ns / 'cohort_costs.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    C.save(ns / 'statistics.json', summary)
    plot_costs(ns, output, summary)
    return summary


def plot_costs(ns, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist([r['RZ'] for r in rows], bins='auto', color='#287d8e')
    axes[0].set(xlabel='Central-control cosine RZ', ylabel='Saved trajectories', title='Fixed cohort (n=256)')
    axes[1].hist([r['reduction_percent'] for r in rows], bins='auto', color='#a66826')
    axes[1].set(xlabel='Reduction vs same-ID generic (%)', ylabel='Saved trajectories', title='Paired comparison')
    labels = ['Pauli PR\ncentral', 'Pauli PR\ngeneric', 'Pauli PF\ncentral', 'DF PR\nnative', 'DF PF\nnative']
    values = [summary['central_cosine_RZ']['mean'], summary['generic_cosine_RZ']['mean'],
              summary['baseline_RZ']['pauli_central_PF'], summary['baseline_RZ']['df_native_PR_saved_mean'], summary['baseline_RZ']['df_native_PF']]
    axes[2].bar(range(5), values, color=['#287d8e', '#b8b8b8', '#287d8e', '#9c6a9b', '#9c6a9b'])
    axes[2].set_xticks(range(5), labels, fontsize=8)
    axes[2].set(yscale='log', ylabel='Cosine RZ / wrapper (log scale)', title='Saved costs; Pauli/DF unpaired')
    fig.suptitle('H4 / 1.00 A / STO-3G / S2 T=0.8 q=8 r=4 K=4; no state preparation')
    fig.tight_layout()
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(ns / ('cost_comparison.' + extension), dpi=170)
    plt.close(fig)


def worker():
    ns = BASE / 'run01'
    c, s, g = authority(task=True)
    C.save(ns / 'ENGINE_CLAIM.json', dict(pid=os.getpid(), UTC=C.utc()))
    (ns / 'trajectories').mkdir()
    try:
        reuse = reuse_results()
        size = c['worker_count']
        for start in range(0, len(CASES), size):
            dispatch(ns, CASES[start:start+size])
            C.save(ns / ('wave_' + str(start) + '.json'), dict(completed_indices=[int(x) for x in CASES[start:start+size]], UTC=C.utc()))
        C.CallBudget(ns, LIMITS).take('cost_statistics_batch')
        statistics_batch(ns, reuse)
        C.save(ns / 'ENGINE_STOP.json', dict(status='A_COHORT_COS_COMPLETE_MANDATORY_STOP', mandatory_STOP=True, new_successes=254))
        return 0
    except BaseException:
        C.save(ns / 'ENGINE_STOP.json', dict(status='A_COHORT_COS_FAILED_NO_RETRY', traceback=traceback.format_exc(), mandatory_STOP=True,
                                           exact_256_mean_complete=False, missing_results_are_not_zero=True))
        raise


def launch():
    c, s, g = authority(consume=True)
    ns = BASE / 'run01'
    C.save(ns / 'LAUNCH_CLAIM.json', dict(pid=os.getpid(), grant_id=g['grant_id'], UTC=C.utc()))
    tree = ast.parse((ROOT / 'src/trotterlib/h4_certificate_control.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'witness')
    for item in ast.walk(node):
        if isinstance(item, ast.Constant) and item.value == 'NEW_C_WORKER_ONLY':
            item.value = 'NEW_A_COHORT_COS_ENGINE_ONLY'
    env = dict(vars(C))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reused cohort native witness>', 'exec'), env)
    code = env['witness'](ns / 'native', [PYTHON, '-B', str(SCRIPT), '--worker'])
    error = None
    try:
        verify_seal(s)
    except BaseException as exc:
        error = repr(exc)
    engine = C.read(ns / 'ENGINE_STOP.json') if (ns / 'ENGINE_STOP.json').exists() else dict(status='MISSING_ENGINE_STOP')
    native = C.read(ns / 'native/worker_native_exit.json')
    tasks = [C.read(p) for p in (ns / 'native_tasks').glob('*/native_exit.json')]
    finished = [p for p in (ns / 'trajectories').glob('*/task_finished.json')]
    success = (code == 0 and error is None and C.read(ns / 'native/STOP.json')['all_owned_reaped'] and len(tasks) == len(finished) == 254
               and all(t['exit_code'] == 0 and t['reaped'] for t in tasks) and engine['status'] == 'A_COHORT_COS_COMPLETE_MANDATORY_STOP')
    C.save(ns / 'STOP.json', dict(status='A_COHORT_COS_COMPLETE_MANDATORY_STOP' if success else 'A_COHORT_COS_FAILED_NO_RETRY',
        native_engine_exit=native, native_task_count=len(tasks), all_tasks_exit0_reaped=all(t['exit_code'] == 0 and t['reaped'] for t in tasks),
        all_owned_reaped=C.read(ns / 'native/STOP.json')['all_owned_reaped'], preservation_error=error,
        ledger=C.read(ns / 'call_ledger.json'), mandatory_STOP=True, automatic_next_stage=False,
        original_H4_native_verified=False, formal_certificate_accepted=False, UTC=C.utc()))
    C.save(ns / 'output_inventory.json', dict(files={str(p.relative_to(ns)): C.sha(p) for p in ns.rglob('*') if p.is_file() and p.name != 'output_inventory.json'}))
    return dict(status=C.read(ns / 'STOP.json')['status'], ledger=C.read(ns / 'call_ledger.json')['used'])
