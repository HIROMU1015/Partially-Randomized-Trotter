"""H6 fixed-candidate confirmation adapter; no science at import or preparation.

H8's tested paired accounting and native acceptance are specialized through
private function globals, never by editing its module or its sealed SOURCE.
The parallel kernel and H6 A_COST context are called unchanged after approval.
Phase 0 proposals have no n/seed registry and cannot pass the execution gate.
"""
import argparse
from fractions import Fraction
import math
import os
from pathlib import Path
import signal
import sys
from types import FunctionType, SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_pr_confirmation as shared
import pr_cost_batch as batch

c = shared.c
ROOT = shared.ROOT
COHORT = 'H6_INDEPENDENT_UNCONDITIONAL_CONFIRMATION_V1'
NAMESPACE = 'H6_NEW_PR_HOLDOUT_V1'
PLAN = 'artifacts/resource_applicability/track_a_h6_pr_confirmation_phase0/2026-10-19/plan_PROPOSED.json'
TARGETS = (
    dict(scope='H6_CONFIRM_P4_Q2', cell=dict(id='B2_p4_q2_2nd_r4_K2', method='B2',
         prefix=4, q=2, r=4, K=2, R=8, formula='2nd'), epsilon=.05,
         fixed_PF_id='B0I_p7_q1_2nd', expected_direction=-1),
    dict(scope='H6_CONFIRM_P4_Q8', cell=dict(id='B2_p4_q8_2nd_r4_K2', method='B2',
         prefix=4, q=8, r=4, K=2, R=32, formula='2nd'), epsilon=.0005,
         fixed_PF_id='B0I_p11_q4_4th', expected_direction=1),
)
SCOPES = {t['scope']: t['cell']['id'] for t in TARGETS}
FILES = tuple(sorted(set((*batch.FILES, *shared.FILES, PLAN,
    'scripts/h6_pr_confirmation.py', 'scripts/prepare_h6_pr_confirmation.py',
    'scripts/launch_h6_pr_confirmation.py', 'tests/test_h6_pr_confirmation.py',
    'scripts/h6_pr_confirmation_seal.py', 'scripts/prepare_h6_pr_confirmation_seal.py',
    'tests/test_h6_pr_confirmation_seal.py'))))


def specialize(function, **constants):
    """Reuse an unmodified H8 body with isolated H6 constants/dependencies."""
    result = FunctionType(function.__code__, dict(function.__globals__, **constants),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def check_target(target):
    c.core.cell_key(target['cell'])
    expected = next((t for t in TARGETS if t['scope'] == target['scope']), None)
    c.require(expected is not None and type(target.get('expected_direction')) is int
              and all(target.get(k) == v for k, v in expected.items()),
              'FIXED_H6_CANDIDATE_Q_R_K_PF_DIRECTION')


def check_law(law):
    c.require(law.get('finite_taylor_order') == 2 and law.get('orders') == [0, 2],
              'H6_K2_SUPPORT_ONLY')


def check_events(events, cell):
    c.require(len(events) == cell['q'] and all(len(step) == cell['r'] for step in events)
              and all(type(e['taylor_order']) is int and e['taylor_order'] in (0, 2)
                      for step in events for e in step), 'H6_K2_EVENT_SUPPORT')


def checked_native(native):
    """Keep the sampler untouched, reject unexpected K4 events before native build."""
    def steps(prep, cell, master):
        requests, law = native.steps(prep, cell, master)
        check_law(law.to_dict())
        check_events(native.event_values(requests), cell)
        return requests, law
    return SimpleNamespace(Writer=native.Writer, steps=steps,
        event_values=native.event_values, seed_hierarchy=native.seed_hierarchy)


def derive_registry(targets, source, plan_digest, excluded, n_by_scope):
    """Metadata only. In Phase 0 this is exercised with artificial identities only."""
    c.require(len(source) == 40 and len(plan_digest) == 64
              and all(x in '0123456789abcdef' for x in source + plan_digest), 'SEED_SOURCE_PLAN')
    c.require({t['scope'] for t in targets} == set(SCOPES)
              and len(targets) == len(SCOPES) and set(n_by_scope) == set(SCOPES), 'TWO_H6_SCOPES')
    taken, tasks = set(excluded), []
    for target in targets:
        check_target(target)
        n = n_by_scope[target['scope']]
        c.require(type(n) is int and n >= 2, 'FINAL_FIXED_N_REQUIRED')
        for replica in range(n):
            key = [NAMESPACE, source, plan_digest, target['cell'], replica]
            master = int(c.digest(key)[:16], 16)
            values = shared.hierarchy(master, target['cell']['q'])
            c.require(len(values) == len(set(values)) and taken.isdisjoint(values),
                      'SEED_COLLISION_NO_REDRAW')
            taken.update(values)
            tasks.append(dict(id=target['cell']['id'] + '_' + str(replica),
                cell=target['cell'], scope=target['scope'], replica=replica, seed=master))
    return tasks


def calls_for(target, n):
    check_target(target)
    c.require(type(n) is int and n >= 2, 'FINAL_FIXED_N_REQUIRED')
    return dict(trajectory=n, occurrence=n * target['cell']['q'], RTE_event=n * target['cell']['R'],
                native_evolution=n, wrapper_compile=2*n, control_probe=4, independent_event_oracle=1)


def validate_prior_seeds(registry, payload):
    c.require(registry['excluded_inventory_complete'] is True, 'COMPLETE_SEED_INVENTORY_REQUIRED')
    old = set(c.read(payload['new_seed_registry']['old_seed_exclusions'])['values'])
    tasks = payload['new_seed_registry']['tasks']
    c.require(len(tasks) == 128, 'EXCLUDE_ALL_ORIGINAL_H6_A128')
    for task in tasks:
        old.update(shared.hierarchy(task['seed'], task['cell']['q']))
    c.require(old <= set(registry['excluded']), 'OLD_H6_SEEDS_MISSING_FROM_FINAL_EXCLUSIONS')


def point_account(pairs, shots, target, n, binding):
    check_target(target)
    c.require(type(n) is int and n >= 2, 'FINAL_FIXED_N_REQUIRED')
    c.require(all(p['cell_id'] == target['cell']['id'] and p['binding'] == binding for p in pairs),
              'H6_CANDIDATE_INPUT_MIXING')
    result = specialize(shared.point_account, N=n, COHORT=COHORT)(pairs, shots, target['fixed_PF_G_exact'])
    delta = Fraction(result['fixed_PF_G_exact']) - Fraction(result['mean_exact'])
    direction = 1 if delta > 0 else -1 if delta < 0 else 0
    result.pop('point_decision')
    result.update(point_direction=direction, expected_direction=target['expected_direction'],
        reproduction_status='INCONCLUSIVE' if direction == 0 else
            'REPRODUCED' if direction == target['expected_direction'] else 'NOT_REPRODUCED',
        PR_CHEAPER=None if direction == 0 else direction > 0)
    return result


def accept(task, handle, spec):
    check_target(spec['target'])
    c.require(task['cell'] == spec['target']['cell'], 'FIXED_H6_TASK')
    check_law(spec['law'])
    result, trace = specialize(shared.accept, COHORT=COHORT)(task, handle, spec)
    check_events(trace['events'], task['cell'])
    return result, trace


def verify_proposal(plan):
    c.require(plan['schema'] == 'H6_CONFIRMATION_PHASE0_V1' and plan['cohort'] == COHORT,
              'H6_PHASE0_SCHEMA_COHORT')
    c.require(len(plan['targets']) == 2 and {t['scope'] for t in plan['targets']} == set(SCOPES),
              'TWO_H6_SCOPES')
    for target in plan['targets']:
        check_target(target)
        check_law(target['law'])
        c.require(target['n'] is None, 'PHASE0_NO_NUMERIC_N')
    c.require(all(plan[k] is False for k in ('approved', 'runtime_authorization', 'technical_sealed',
              'retry', 'next_stage_authorized')), 'PHASE0_NOT_EXECUTION_APPROVAL')
    c.require(plan['seed_registry'] is None and plan['calls'] is None and plan['SOURCE'] is None,
              'PHASE0_NO_FINAL_SEEDS_CALLS_SOURCE')
    c.require(plan['binding']['system'] == 'H6' and plan['binding']['rank'] == 19
              and plan['binding']['rank_policy'] == 'tol_only_1e-8', 'H6_INPUT_BINDING')
    c.core.same_binding(plan['binding'], c.read(ROOT / c.REGISTRY)['systems']['H6']['binding'])
    c.require(plan['metric'] == 'saved_shots_times_compiled_RZ_cosine_plus_sine'
              and plan['bias_mode'] == 'USUAL_SIGNED' and plan['allowance_factor'] == 1
              and plan['P'] == 0, 'FIXED_H6_ACCOUNTING')
    design = plan['numerical_design']
    c.require(design['approved'] is False and all(design[k] is None
              for k in ('n_results', 'power_results', 'visibility_results')), 'PHASE0_NO_NUMERIC_DESIGN')
    return dict(status='PASS_PHASE0_PROPOSAL_ONLY', scientific_calls=0, launch_ready=False)


def check_payload(spec, payload, record):
    c.require(payload['native_adapter']['binding'] == spec['binding'], 'NATIVE_INPUT_BINDING')
    target = spec['target']
    saved = record['precision_rows'][str(target['epsilon'])]['1.0']
    c.require(record['cell'] == target['cell'] and c.digest(record) == target['A_signal_record_sha256']
              and c.digest(saved) == target['accounting_sha256'] and saved['cost_acquisition_eligible'] is True
              and spec['shots'] == {a: saved['axes'][ch]['shots'] for a, ch in c.core.CHANNEL.items()},
              'FROZEN_H6_SIGNAL_SAVED_SHOTS')


def sealed(spec):
    c.require(spec.get('schema') == 'H6_CONFIRMATION_EXECUTION_V1'
              and spec.get('technical_sealed') is True, 'FINAL_H6_EXECUTION_SEAL_REQUIRED')
    check_target(spec['target'])
    c.require(spec['scope'] == spec['target']['scope'] and spec['cohort'] == COHORT, 'H6_SCOPE_COHORT')
    c.require(spec['calls'] == calls_for(spec['target'], spec['n']), 'EXACT_H6_CALL_BUDGET')
    from h6_pr_confirmation_seal import FINAL_N
    c.require(spec['n_by_scope'] == FINAL_N and spec['n'] == FINAL_N[spec['scope']],
              'FINAL_ADOPTED_H6_N')
    c.require(spec['technical_digest'] == shared.technical_digest(spec), 'TECHNICAL_DIGEST')
    c.require(all(spec[k] is False for k in ('approved', 'runtime_authorization', 'retry',
                                          'next_stage_authorized')), 'SEALED_NOT_AUTHORIZED')
    output, lock = Path(spec['output']).resolve(), Path(spec['lock']).resolve()
    c.require(not output.is_relative_to(lock) and not lock.is_relative_to(output), 'SEPARATE_OUTPUT_LOCK')


def authorize(spec, grant):
    # Before reading input/runtime/provider bytes or loading a science module.
    c.require(all(grant.get(k) is True for k in ('approved', 'approved_by_user',
              'runtime_authorization', 'one_shot')), 'EXPLICIT_H6_SCIENCE_APPROVAL_REQUIRED')
    sealed(spec)
    c.require(all(grant.get(k) == spec[k] for k in ('SOURCE', 'scope', 'technical_digest'))
              and grant.get('retry') is False and grant.get('next_stage_authorized') is False, 'H6_GRANT_BINDING')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'USER_APPROVAL_RECEIPT')
    approval = c.read(grant['approval_receipt'])
    c.require(approval.get('approved_by_user') is True and approval.get('stage') == 'H6_CONFIRMATION_SCIENCE'
              and approval['SOURCE'] == spec['SOURCE'] and approval['n_by_scope'] == spec['n_by_scope']
              and approval['seed_registry_sha256'] == spec['registry_sha256']
              and approval['calls_by_scope'][spec['scope']] == spec['calls'], 'EXACT_FINAL_SCIENCE_APPROVAL')
    cpus = grant['allowed_cpus']
    c.require(cpus and all(type(x) is int for x in cpus) and len(cpus) == len(set(cpus))
              and set(cpus) <= os.sched_getaffinity(0)
              and set(cpus).isdisjoint(grant['concurrent_job_cpus']), 'FRESH_NONINTERFERING_CPUS')
    c.require(type(grant['driver_cpu']) is int and grant['driver_cpu'] in os.sched_getaffinity(0)
              and grant['driver_cpu'] not in cpus
              and grant['driver_cpu'] not in grant['concurrent_job_cpus'], 'SEPARATE_OWN_DRIVER_CPU')
    from h6_pr_confirmation_seal import process_slots
    c.require(type(grant.get('memory_admission_estimate_bytes')) is int
              and grant['memory_admission_estimate_bytes'] >=
              spec['memory_estimate_per_process_bytes'] * process_slots(len(cpus)),
              'MEMORY_ADMISSION_ESTIMATE_NOT_A_CAP')


def verify(spec):
    sealed(spec)
    c.require(set(spec['source_hashes']) == set(FILES), 'COMPLETE_H6_SOURCE_CLOSURE')
    specialize(batch.verify, sealed=sealed)(spec)
    plan = c.read(ROOT / PLAN)
    verify_proposal(plan)
    c.require(spec['design_digest'] == c.digest(plan) and spec['target'] in plan['targets']
              and spec['binding'] == plan['binding'], 'FIXED_PHASE0_SELECTION_BINDING')
    payload = c.read(spec['payload'])
    covered = {(str(Path(r['path']).resolve()), r['sha256']) for r in spec['inputs']}
    for name in (spec['payload'], payload['A_signal'], payload['native_adapter']['native_config'],
                 spec['registry'], spec['runtime_receipt'], payload['new_seed_registry']['old_seed_exclusions']):
        c.require((str(Path(name).resolve()), c.sha(name)) in covered, 'UNCOVERED_H6_INPUT')
    old = c.read(payload['native_adapter']['native_config'])
    c.require(all((str(Path(r['path']).resolve()), r['sha256']) in covered for r in old['inputs']),
              'UNCOVERED_NATIVE_INPUT')
    c.require(all(any(Path(p).resolve().is_relative_to(Path(root).resolve())
                  for root in spec['provider_closure']) for p in payload['native_adapter']['import_paths']),
              'UNCOVERED_H6_PROVIDER_IMPORT')
    record = next(r for r in c.read(payload['A_signal'])['signals']
                  if r['cell']['id'] == spec['target']['cell']['id'])
    target = spec['target']
    check_payload(spec, payload, record)
    check_law(spec['law'])
    c.require(spec['law'] == target['law'], 'FROZEN_K2_LAW')
    c.require(c.sha(spec['registry']) == spec['registry_sha256'], 'SEED_REGISTRY_SHA')
    registry = c.read(spec['registry'])
    validate_prior_seeds(registry, payload)
    c.require(spec['n'] == spec['n_by_scope'][spec['scope']] and registry['n_by_scope'] == spec['n_by_scope']
              and registry['SOURCE'] == spec['SOURCE'] and registry['design_digest'] == spec['design_digest']
              and registry['seed_plan_digest'] == spec['seed_plan_digest']
              and registry['excluded_inventory_complete'] is True, 'FINAL_H6_SEED_BINDING')
    c.require(registry['tasks'] == derive_registry(plan['targets'], spec['SOURCE'], registry['seed_plan_digest'],
              registry['excluded'], spec['n_by_scope']), 'EXACT_H6_SEED_REGISTRY')
    from h6_pr_confirmation_seal import verify_final_binding
    verify_final_binding(spec, registry)
    return dict(status='PASS_BYTES_ONLY', scientific_calls=0)


def execute(spec, grant, output):
    # Only run() dispatches here after the SOURCE/input/runtime/approval gate.
    import pr_cost_native as native
    import pr_cost_batch_science as science
    import h8_key_compile_parallel as pool
    import h8_pr_confirmation_parallel as parallel
    payload = c.read(spec['payload'])
    target = spec['target']
    context = science.native_context(dict(system='H6', binding=spec['binding']), payload, [target['cell']], output)
    c.require(context.preparations[4].preparation_hash == target['preparation_sha256'], 'H6_SAVED_P4_IDENTITY')
    adapter = checked_native(native)
    tasks = [t for t in c.read(spec['registry'])['tasks'] if t['scope'] == spec['scope']]
    c.require(len(tasks) == spec['n'], 'EXACT_FIXED_H6_N')
    budget = c.Budget(output, spec['calls'])
    completed, traces, receipts = {}, {}, []
    cpus = grant['allowed_cpus']
    groups = {cpus[i]: cpus[i:i+2] for i in range(0, len(cpus), 2)}
    def callback(task, directory):
        leader = next(iter(os.sched_getaffinity(0)))
        group = cpus[:5] if task['replica'] == 0 else groups[leader]
        return parallel.compile_one(context, task, directory, spec['SOURCE'], target, spec['law'],
            target['saved_log_B'], target['log_B_margin'], adapter, pool, group,
            payload['native_adapter']['font_seed'], COHORT)
    def reserve(task):
        counts = dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
                      native_evolution=1, wrapper_compile=2)
        if task['replica'] == 0:
            counts.update(control_probe=4, independent_event_oracle=1)
        for kind, count in counts.items():
            budget.take(kind, count)
    def finish(task, handle):
        result, trace = accept(task, handle, spec)
        c.require(task['id'] not in completed, 'DUPLICATE_H6_COMPLETION')
        completed[task['id']], traces[task['id']] = result, trace
    last_observation = [0.]
    def observe():
        import time
        if time.monotonic() - last_observation[0] >= 5:
            c.save(output / ('resources_' + str(time.monotonic_ns()) + '.json'), shared.effective_fresh(output))
            last_observation[0] = time.monotonic()
    for name, items, leaders in (
        ('representative', [t for t in tasks if t['replica'] == 0], [cpus[0]]),
        ('workers', [t for t in tasks if t['replica'] != 0], list(groups))):
        if name == 'workers':
            c.require(any(t['replica'] == 0 and t['id'] in completed for t in tasks), 'FIRST_CONTROL_REQUIRED')
        receipts.append(pool.map_owned(items, callback, cpus=leaders,
            limits=dict(max_workers=len(leaders), cleanup_TERM_seconds=10.), directory=output / name,
            deadline=math.inf, font_seed=payload['native_adapter']['font_seed'],
            reserve=reserve, on_complete=finish, guard=observe))
    c.require(len(completed) == spec['n'] and budget.used == spec['calls']
              and all(r['all_reaped'] and not r['cleanup_errors'] for r in receipts), 'COMPLETE_H6_SCOPE_AND_REAP')
    c.save(output / 'native_receipt.json', dict(maps=receipts, all_reaped=True))
    pairs = [completed[t['id']] for t in tasks]
    # Accounting only the new cohort; no secondary endpoints or reselection.
    orders = [e['taylor_order'] for trace in traces.values() for step in trace['events'] for e in step]
    return dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'], scope=spec['scope'],
        primary=point_account(pairs, spec['shots'], target, spec['n'], spec['binding']), paired_costs=pairs,
        event_order_counts={str(k): orders.count(k) for k in (0, 2)}, calls=budget.used,
        old_samples_pooled=False, unobserved_rare_costs_not_zero_filled=True,
        automatic_retry=False, next_stage_authorized=False)


def run(spec, grant):
    return specialize(shared.run, authorize=authorize, verify=verify, execute=execute)(spec, grant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('OWN_H6_SIGNAL_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.proposal:
            c.require(not args.spec and not args.grant, 'PROPOSAL_VERIFY_ONLY')
            print(verify_proposal(c.read(args.proposal)))
            return 0
        c.require(args.spec is not None and args.grant is not None, 'EXPLICIT_H6_SCIENCE_APPROVAL_REQUIRED')
        return run(c.read(args.spec), c.read(args.grant))
    except (Exception, KeyboardInterrupt) as error:
        print('STOP: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
