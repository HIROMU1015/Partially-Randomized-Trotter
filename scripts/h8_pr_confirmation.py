"""Fixed H8 independent n48 confirmation. No scientific work at import.

The metadata preparation command cannot issue an execution grant. Scientific
dispatch requires a SOURCE/input/seed-bound user grant before native imports.
The original S2/RTE kernel functions are reused; exploration outputs are never
relabeled as confirmation samples.
"""
from fractions import Fraction as F
import argparse
import importlib.util
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pr_cost_contract as c

COHORT = 'INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48'
N = 48
SCOPES = {'H8_CONFIRM_P8_Q2': 'B2_p8_q2_2nd_r4_K4',
          'H8_CONFIRM_P7_Q16': 'B2_p7_q16_2nd_r4_K4'}
DESIGN = 'artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/plan_PROPOSED.json'
FILES = ('scripts/h8_pr_confirmation.py', 'scripts/prepare_h8_pr_confirmation.py',
         'scripts/launch_h8_pr_confirmation.py',
         'scripts/h8_pr_confirmation_parallel.py', 'scripts/prepare_h8_pr_confirmation_resume.py',
         'tests/test_h8_pr_confirmation.py', 'scripts/pr_cost_native.py',
         'scripts/pr_cost_contract.py', 'scripts/h8_key_compile_parallel.py',
         'src/trotterlib/pr_cost_integration.py', DESIGN)


def load_own(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hierarchy(master, q):
    # Original direct_master_to_step_seeds_v1, including the master exclusion.
    if q == 1:
        return [master]
    rng = random.Random(master)
    return [master] + [rng.randrange(0, 2**63) for _ in range(q)]


def derive_registry(targets, source, seed_plan_digest, excluded):
    c.require(len(source) == 40 and len(seed_plan_digest) == 64, 'SEED_SOURCE_PLAN')
    taken = set(excluded)
    tasks = []
    for target in targets:
        c.require(SCOPES.get(target['scope']) == target['cell']['id'], 'FIXED_CONFIRMATION_TARGET')
        for replica in range(N):
            key = ['H8_NEW_PR_HOLDOUT_V1', source, seed_plan_digest, target['cell'], replica]
            master = int(c.digest(key)[:16], 16)
            values = hierarchy(master, target['cell']['q'])
            c.require(len(values) == len(set(values)) and taken.isdisjoint(values), 'SEED_COLLISION_NO_REDRAW')
            taken.update(values)
            tasks.append(dict(id=target['cell']['id'] + '_' + str(replica),
                cell=target['cell'], scope=target['scope'], replica=replica, seed=master))
    return tasks


def validate_registry(tasks, targets, source, seed_plan_digest, excluded):
    c.require(tasks == derive_registry(targets, source, seed_plan_digest, excluded),
              'EXACT_FROZEN_INDEPENDENT_SEED_REGISTRY')


def technical_digest(spec):
    return c.digest({k: v for k, v in spec.items() if k != 'technical_digest'})


def verify(spec, *, source=True):
    c.require(spec['scope'] in SCOPES and spec['technical_sealed'] is True,
              'TECHNICAL_SEAL_REQUIRED')
    c.require(spec['technical_digest'] == technical_digest(spec), 'TECHNICAL_DIGEST')
    c.require(spec['approved'] is False and spec['runtime_authorization'] is False
              and spec['retry'] is False and spec['next_stage_authorized'] is False,
              'TECHNICAL_SEAL_NOT_EXECUTION_APPROVAL')
    c.require(spec['cohort'] == COHORT and spec['n'] == N, 'FIXED_CONFIRMATION_N48')
    c.require(spec['target']['cell']['id'] == SCOPES[spec['scope']], 'FIXED_TARGET')
    c.require(set(spec['source_hashes']) == set(FILES), 'OWN_SOURCE_CLOSURE')
    if source:
        c.verify_source(ROOT, spec['SOURCE'], spec['source_hashes'])
    c.verify_byte_rows(spec['inputs'])
    c.require(c.sha(spec['python']) == spec['python_sha256'], 'PYTHON_BINARY')
    receipt = c.read(spec['runtime_receipt'])
    c.require(c.sha(spec['runtime_receipt']) == spec['runtime_receipt_sha256'], 'RUNTIME_RECEIPT')
    for path, expected in receipt['expected'].items():
        st = Path(path).stat()
        if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != receipt['verified_stats'][path]:
            c.require(c.sha(path) == expected, 'RUNTIME_BYTES_CHANGED')
    for base, files in spec['provider_closure'].items():
        for relative, expected in files.items():
            path = Path(relative)
            c.require(not path.is_absolute() and '..' not in path.parts, 'PROVIDER_RELATIVE_PATH')
            c.require(c.sha(Path(base) / path) == expected, 'PROVIDER_BYTES_CHANGED')
    payload = c.read(spec['payload'])
    c.require(payload['native_adapter']['binding'] == spec['binding'], 'NATIVE_INPUT_BINDING')
    records = {r['cell']['id']: r for r in c.read(payload['A_signal'])['signals']}
    target = spec['target']
    record = records[target['cell']['id']]
    c.require(c.digest(record) == target['A_signal_record_sha256'], 'FROZEN_SIGNAL_RECORD')
    for mode, field, hash_key in [('primary', 'precision_rows', 'primary_accounting_sha256'),
            ('secondary', 'empirical_noncancellation_precision_rows', 'secondary_accounting_sha256')]:
        saved = record[field][str(target['primary_epsilon'])]['1.0']
        c.require(c.digest(saved) == target[hash_key] and saved['cost_acquisition_eligible'] is True,
                  'FROZEN_SAVED_SHOTS_' + mode)
        c.require(spec['shots'][mode] == {a: saved['axes'][ch]['shots']
                  for a, ch in c.core.CHANNEL.items()}, 'AXIS_SHOT_MAPPING')
    registry = c.read(spec['registry'])
    design = c.read(ROOT / DESIGN)
    dataset_source = spec.get('seed_derivation_SOURCE', spec['SOURCE'])
    c.require(registry['targets'] == design['targets'] and registry['SOURCE'] == dataset_source
              and spec['target'] in design['targets'] and spec['design_digest'] == design['design_digest'],
              'FIXED_PUBLISHED_SELECTION')
    validate_registry(registry['tasks'], registry['targets'], dataset_source,
                      registry['seed_plan_digest'], registry['excluded'])
    if spec.get('resume_manifest'):
        c.require(spec.get('parallel_scheduling') == 'H8_CONFIRMATION_GROUPED_PROBES_AXES_V1',
                  'FIXED_PARALLEL_RESUME_POLICY')
        resume = load_own('h8_confirmation_resume_gate', 'scripts/prepare_h8_pr_confirmation_resume.py')
        resume.verify_manifest(spec)
    return dict(status='PASS_BYTES_ONLY', scientific_calls=0)


def authorize(spec, grant):
    c.require(all(grant.get(k) is True for k in ('approved', 'approved_by_user',
              'runtime_authorization', 'one_shot')), 'EXPLICIT_EXECUTION_GRANT_REQUIRED')
    c.require(grant.get('SOURCE') == spec['SOURCE'] and grant.get('scope') == spec['scope']
              and grant.get('technical_digest') == spec['technical_digest'], 'GRANT_BINDING')
    c.require(grant.get('retry') is False and grant.get('next_stage_authorized') is False,
              'NO_RETRY_EXTENSION_OR_NEXT_STAGE')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'USER_APPROVAL_RECEIPT')
    approval = c.read(grant['approval_receipt'])
    c.require(approval['approved_by_user'] is True and approval['n_per_candidate'] == N
              and approval['wrapper_compiles'] == 192 and set(approval['scopes']) == set(SCOPES)
              and approval['design_digest'] == spec['design_digest'], 'EXACT_APPROVED_SCOPE')
    if spec.get('resume_manifest'):
        c.require(grant.get('explicit_resume_authorized') is True
                  and approval.get('repair_parallel_resume_authorized') is True
                  and approval.get('seed_registry_sha256') == c.sha(spec['registry']),
                  'EXPLICIT_REPAIR_PARALLEL_HANDOFF_APPROVAL')
    cpus = grant['allowed_cpus']
    c.require(cpus and all(type(x) is int for x in cpus) and len(cpus) == len(set(cpus))
              and set(cpus) <= os.sched_getaffinity(0), 'FRESH_OWN_CPU_AFFINITY')
    c.require(set(cpus).isdisjoint(grant['concurrent_job_cpus']), 'NONINTERFERING_CPU_ASSIGNMENT')
    if spec.get('parallel_scheduling'):
        c.require(len(cpus) >= 2, 'PARALLEL_CPU_GROUP_REQUIRED')
    c.require(type(grant['driver_cpu']) is int and grant['driver_cpu'] in os.sched_getaffinity(0)
              and grant['driver_cpu'] not in cpus
              and grant['driver_cpu'] not in grant['concurrent_job_cpus'], 'SEPARATE_OWN_DRIVER_CPU')


def effective_fresh(directory):
    value = c.fresh(directory)
    relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    observations = []
    while group.is_relative_to(Path('/sys/fs/cgroup')):
        row = dict(path=str(group))
        for key in ('memory.max', 'memory.current', 'memory.events', 'memory.pressure', 'cpu.max'):
            path = group / key
            if path.exists():
                row[key] = path.read_text().strip()
        if row.get('memory.max', 'max') != 'max':
            value['memory_available_bytes'] = min(value['memory_available_bytes'],
                max(0, int(row['memory.max']) - int(row['memory.current'])))
        observations.append(row)
        if group == Path('/sys/fs/cgroup'):
            break
        group = group.parent
    value['cgroups'] = observations
    return value


def fresh_gate(spec, grant):
    observation = effective_fresh(Path(spec['output']).parent)
    c.require(observation['memory_available_bytes'] >= grant['memory_admission_estimate_bytes']
              and observation['disk_free_bytes'] >= spec['output_estimate_bytes']
              and observation['free_inodes'] >= spec['inode_estimate'], 'FRESH_RESOURCE_ADMISSION')
    c.require(grant['quota_policy'] in ('AVAILABLE_VERIFIED', 'NO_QUOTA_CONFIRMED',
              'USER_ACCEPTS_UNEXPOSED_QUOTA_ENOSPC_FAIL_CLOSED'), 'EXPLICIT_QUOTA_POLICY')
    if grant['quota_policy'] == 'AVAILABLE_VERIFIED':
        c.require(grant['quota_remaining_bytes'] >= spec['output_estimate_bytes']
                  and grant['quota_remaining_inodes'] >= spec['inode_estimate'], 'QUOTA_ADMISSION')
    for name in ('output', 'lock'):
        path = c.home_path(spec[name])
        c.require(not path.exists(), 'UNUSED_OUTPUT_LOCK')
    return observation


def compile_one(context, task, directory, source, target, law, log_B, margin, native):
    """Same S2 requests/control/compiler as A; fresh cohort and event stream."""
    cell = task['cell']
    m = context.m
    prep = context.preparations[cell['prefix']]
    writer = native.Writer(directory)
    budget = c.Budget(directory, dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R'],
        native_evolution=1, wrapper_compile=2, control_probe=4, independent_event_oracle=1))
    for kind, count in dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R']).items():
        budget.take(kind, count)
    requests, distribution = native.steps(prep, cell, task['seed'])
    c.require([s.seed for s in requests] == hierarchy(task['seed'], cell['q'])[1:], 'ORIGINAL_STEP_SEED_API')
    original_law = distribution.to_dict()
    c.require(original_law == law and c.digest(original_law) == target['original_law_sha256'], 'ORIGINAL_UNCONDITIONAL_LAW')
    events = native.event_values(requests)
    event_digest = c.digest(events)
    c.require(len(events) == cell['q'] and all(len(step) == cell['r'] for step in events), 'EXACT_OCCURRENCE_EVENTS')
    c.require(prep.preparation_hash == target['preparation_sha256'], 'EXACT_PREPARATION_IDENTITY')
    observed = cell['R'] * math.log(distribution.exact_finite_distribution)
    c.require(math.isfinite(observed) and abs(observed - log_B) <= margin, 'ORIGINAL_NORMALIZATION')
    writer.write('trajectory.json', dict(cell=cell, seed=task['seed'], SOURCE=source,
        events=events, event_digest=event_digest, original_RTE_distribution=original_law,
        cohort=COHORT, conditional=False, quantum_weights_changed=False))
    budget.take('native_evolution')
    evolution = context.native.partial_native_from_step_requests(requests,
        control_policy='symmetric_directional', max_instructions=1000000)
    if task['replica'] == 0:
        budget.take('control_probe', 4)
        budget.take('independent_event_oracle')
        native.control(evolution, requests, context, writer, 1e-8)
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    axes = {}
    with use_fingerprint_reuse():
        for axis in c.core.AXES:
            c.require(c.digest(native.event_values(requests)) == event_digest, 'PAIRED_EVENT_MUTATION')
            wrapper = m.build_native_hadamard_wrapper(evolution, axis=axis,
                include_measurement=True, max_instructions=1000000)
            fingerprint = m.canonical_qiskit_circuit_fingerprint(wrapper)
            budget.take('wrapper_compile')
            start = time.monotonic()
            result = m.transpile_and_measure_cost(wrapper, context.compiler,
                actual_circuit_fingerprint=fingerprint)
            c.require(result.compiler_settings_hash == context.binding['compiler']
                      and result.actual_circuit_fingerprint == fingerprint
                      and m.canonical_qiskit_circuit_fingerprint(wrapper) == fingerprint, 'COMPILE_IDENTITY')
            metrics = {key: getattr(result, attr) for key, attr in dict(RZ='rz_count', CX='cx_count',
                depth='total_depth', RZ_depth='rz_depth', CX_depth='cx_depth', size='circuit_size').items()}
            c.require(all(type(x) is int and x >= 0 for x in metrics.values())
                      and metrics['RZ'] > 0, 'INTEGER_COMPILED_METRICS')
            axes[axis] = dict(metrics, seed=task['seed'], event_digest=event_digest,
                compiler_hash=result.compiler_settings_hash, fingerprint=fingerprint,
                wall_seconds=time.monotonic() - start)
            writer.write('measured_' + axis + '.json', axes[axis])
    return dict(cell_id=cell['id'], replica=task['replica'], seed=task['seed'], SOURCE=source,
        event_digest=event_digest, binding=context.binding, axes=axes, cohort=COHORT,
        calls=budget.used, normalization_unchanged=True, synthetic=False)


def native_identity(value):
    c.require(isinstance(value, (list, tuple)) and len(value) == 4
              and all(type(x) is int for x in value), 'TYPED_NATIVE_IDENTITY')
    return tuple(value)


def accept(task, handle, spec):
    directory = Path(handle['directory'])
    c.require(handle['reaped'] and handle['exit_code'] == 0
              and handle['result_sha256'] == c.sha(directory / 'task_result.json'), 'NATIVE_EXIT_RESULT_SHA')
    started = c.read(directory / 'started.json')
    c.require(native_identity(started['identity']) == native_identity(handle['identity'])
              and started['task'] == task['id']
              and handle['identity'][0] == handle['pid'] and handle['identity'][3] == os.getuid(),
              'NATIVE_START_IDENTITY')
    result = c.read(directory / 'task_result.json')
    trace = c.read(directory / 'trajectory.json')
    c.require(result['SOURCE'] == spec['SOURCE'] and trace['SOURCE'] == spec['SOURCE']
              and result['cohort'] == trace['cohort'] == COHORT
              and result['synthetic'] is False and trace['conditional'] is False
              and trace['quantum_weights_changed'] is False and result['normalization_unchanged'] is True,
              'INDEPENDENT_COHORT_ONLY')
    c.require(result['cell_id'] == task['cell']['id'] and trace['cell'] == task['cell']
              and result['replica'] == task['replica'] and result['seed'] == trace['seed'] == task['seed']
              and result['binding'] == spec['binding'], 'FROZEN_TASK_BINDING')
    c.require(result['event_digest'] == trace['event_digest'] == c.digest(trace['events'])
              and trace['original_RTE_distribution'] == spec['law'], 'EVENT_LAW_BINDING')
    c.require(len(trace['events']) == task['cell']['q']
              and all(len(step) == task['cell']['r'] for step in trace['events'])
              and all(type(e['taylor_order']) is int and e['taylor_order'] in (0, 2, 4)
                      for step in trace['events'] for e in step), 'COMPLETE_FINITE_TAYLOR_EVENTS')
    c.require(set(result['axes']) == set(c.core.AXES), 'EXACT_PAIR')
    for axis, item in result['axes'].items():
        c.require(item == c.read(directory / ('measured_' + axis + '.json'))
                  and item['seed'] == task['seed'] and item['event_digest'] == result['event_digest']
                  and item['compiler_hash'] == spec['binding']['compiler'], 'AXIS_BINDING')
        c.require(all(type(item[k]) is int and item[k] >= 0 for k in
                  ('RZ', 'CX', 'depth', 'RZ_depth', 'CX_depth', 'size')) and item['RZ'] > 0
                  and len(item['fingerprint']) == 64 and math.isfinite(item['wall_seconds'])
                  and item['wall_seconds'] >= 0, 'ACCEPTED_INTEGER_NATIVE_METRICS')
    expected = dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
        native_evolution=1, wrapper_compile=2)
    if task['replica'] == 0:
        expected.update(control_probe=4, independent_event_oracle=1)
        control = c.read(directory / 'control.json')
        c.require(control['status'] == 'PASS' and control['tolerance'] == 1e-8
                  and 0 <= control['max_error'] < 1e-8 and control['control_probes'] == 4
                  and control['independent_event_oracles'] == 1 and control['full_Fock_scalar_once'] is True
                  and control['reference_rotated'] is False and control['ordinary_ancilla_branch'] is True,
                  'FIRST_FRESH_CONTROL')
    c.require(result['calls'] == expected, 'EXACT_TASK_CALLS')
    for kind, total in expected.items():
        reservations = [c.read(p) for p in directory.glob(kind + '_*_reserved.json')]
        c.require(sum(x['count'] for x in reservations) == total
                  and max(x['total'] for x in reservations) == total
                  and all(x['nonrefundable'] is True for x in reservations), 'BEFORE_CALL_RESERVATIONS')
    if result.get('parallel_native'):
        parallel = load_own('h8_confirmation_parallel_accept', 'scripts/h8_pr_confirmation_parallel.py')
        parallel.verify_descendants(directory, result)
    return result, trace


def point_account(pairs, shots, fixed_pf):
    c.require(len(pairs) == N and {p['replica'] for p in pairs} == set(range(N))
              and len({p['seed'] for p in pairs}) == N
              and all(p['cohort'] == COHORT for p in pairs), 'COMPLETE_FRESH_48_ONLY')
    c.require(set(shots) == set(c.core.AXES) and all(type(x) is int and x > 0 for x in shots.values()), 'SAVED_SHOTS')
    values = []
    for pair in pairs:
        c.require(set(pair['axes']) == set(c.core.AXES), 'BOTH_AXES')
        for axis, value in pair['axes'].items():
            c.require(type(value['RZ']) is int and value['RZ'] > 0
                      and value['seed'] == pair['seed'] and value['event_digest'] == pair['event_digest'], 'PAIRED_INTEGER_COST')
        values.append(sum(shots[a] * pair['axes'][a]['RZ'] for a in c.core.AXES))
    mean = F(sum(values), N)
    variance = sum((F(v) - mean)**2 for v in values) / (N - 1)
    pf = None if fixed_pf is None else F(fixed_pf)
    c.require(pf is None or pf > 0, 'POSITIVE_FIXED_PF')
    return dict(n=N, mean_exact=str(mean), sample_SD=math.sqrt(variance),
        paired_SE=math.sqrt(variance / N), fixed_PF_G_exact=None if pf is None else str(pf),
        PR_reduction_percent_exact=None if pf is None else str(100 * (1 - mean / pf)),
        point_decision='FIXED_PF_NOT_ELIGIBLE_NO_COMPARISON' if pf is None else
            'POINT_REPRODUCED' if mean < pf else 'POINT_NOT_REPRODUCED' if mean > pf else 'EXACT_POINT_TIE',
        cohort=COHORT, population_conclusion='UNRESOLVED_WITHOUT_VALID_ALL_SUPPORT_INFERENCE',
        confidence_interval=None, p_value=None, old_samples_pooled=False)


def execute(spec, grant, output):
    # Load own modules before provider paths are added, preventing shadow imports.
    native = load_own('pr_cost_native', 'scripts/pr_cost_native.py')
    pool = load_own('h8_confirmation_owned_pool', 'scripts/h8_key_compile_parallel.py')
    payload = c.read(spec['payload'])
    target = spec['target']
    registry = c.read(spec['registry'])
    tasks = [t for t in registry['tasks'] if t['scope'] == spec['scope']]
    c.require(len(tasks) == N, 'EXACT_SCOPE_48')
    context = native.load_context('H8', payload['native_adapter'], [target['cell']], output)
    signal_record = next(r for r in c.read(payload['A_signal'])['signals']
                         if r['cell']['id'] == target['cell']['id'])
    completed, traces, provenance = {}, {}, {}
    if spec.get('resume_manifest'):
        resume = c.read(spec['resume_manifest'])
        origin = c.read(resume['origin_spec'])
        by_id = {t['id']: t for t in tasks}
        for key, handle in resume['accepted'].items():
            value, trace = accept(by_id[key], handle, origin)
            completed[key], traces[key] = value, trace
            provenance[key] = dict(SOURCE=origin['SOURCE'], directory=handle['directory'], reused=True)
        for task in tasks:
            if task['id'] in resume['previous_event_digests']:
                task['expected_prior_event_digest'] = resume['previous_event_digests'][task['id']]
    budget = c.Budget(output, spec['calls'])
    def callback(task, directory):
        if spec.get('parallel_scheduling'):
            parallel = load_own('h8_confirmation_parallel_worker', 'scripts/h8_pr_confirmation_parallel.py')
            leader = next(iter(os.sched_getaffinity(0)))
            cpus = grant['allowed_cpus'][:5] if task['replica'] == 0 else groups[leader]
            return parallel.compile_one(context, task, directory, spec['SOURCE'], target, spec['law'],
                signal_record['log_B'], signal_record['allowance']['log_B_margin'], native, pool,
                cpus, payload['native_adapter']['font_seed'], COHORT)
        return compile_one(context, task, directory, spec['SOURCE'], target, spec['law'],
            signal_record['log_B'], signal_record['allowance']['log_B_margin'], native)
    def finish(task, handle):
        result, trace = accept(task, handle, spec)
        c.require(task['id'] not in completed, 'DUPLICATE_COMPLETION')
        completed[task['id']] = result
        traces[task['id']] = trace
        provenance[task['id']] = dict(SOURCE=spec['SOURCE'], directory=handle['directory'], reused=False)
    def reserve(task):
        for kind, count in dict(trajectory=1, occurrence=task['cell']['q'], RTE_event=task['cell']['R'],
                native_evolution=1, wrapper_compile=2, control_probe=4 if task['replica'] == 0 else 0,
                independent_event_oracle=1 if task['replica'] == 0 else 0).items():
            if count:
                budget.take(kind, count)
    observed_at = [0.]
    def observe():
        if time.monotonic() - observed_at[0] >= 5:
            fresh = effective_fresh(output)
            # Transient pressure/CPU contention is diagnostic, not a kill rule.
            # Actual allocation, I/O, native or identity failures remain STOPs.
            c.save(output / ('resources_' + str(time.monotonic_ns()) + '.json'), fresh)
            observed_at[0] = time.monotonic()
    cpus = grant['allowed_cpus']
    width = 2 if spec.get('parallel_scheduling') else 1
    groups = {cpus[i]: cpus[i:i+width] for i in range(0, len(cpus), width)}
    pending = [t for t in tasks if t['id'] not in completed]
    receipts = []
    def batch(items, name, leaders):
        if not items:return
        receipt = pool.map_owned(items, callback, cpus=leaders,
            limits=dict(max_workers=len(leaders), cleanup_TERM_seconds=10.), directory=output / name,
            deadline=math.inf, font_seed=payload['native_adapter']['font_seed'], guard=observe,
            reserve=reserve, on_complete=finish)
        receipts.append(receipt)
    batch([t for t in pending if t['replica'] == 0], 'representative', [cpus[0]])
    c.require(tasks[0]['id'] in completed, 'FIRST_CONTROL_BEFORE_DEPENDENTS')
    batch([t for t in pending if t['replica'] != 0], 'workers', list(groups))
    c.save(output / 'native_receipt.json', dict(maps=receipts,
        all_reaped=all(r['all_reaped'] for r in receipts), reused_task_count=len(tasks)-len(pending)))
    c.require(len(completed) == N and budget.used == spec['calls'], 'EXACT_COMPLETE_SCOPE')
    pairs = [completed[t['id']] for t in tasks]
    orders = [e['taylor_order'] for trace in traces.values() for step in trace['events'] for e in step]
    primary = point_account(pairs, spec['shots']['primary'], target['fixed_PF_G_exact'])
    secondary = point_account(pairs, spec['shots']['secondary'], spec['secondary_PF_G_exact'])
    secondary['interpretation'] = 'EMPIRICAL_NONCANCELLATION_DESCRIPTIVE_ONLY'
    return dict(status='COMPLETE_MANDATORY_STOP', scope=spec['scope'], SOURCE=spec['SOURCE'],
        primary=primary, secondary=secondary, paired_costs=pairs,
        event_order_counts={str(order): orders.count(order) for order in (0, 2, 4)},
        task_provenance=provenance, seed_derivation_SOURCE=spec.get('seed_derivation_SOURCE', spec['SOURCE']),
        carry_reservations=c.read(spec['resume_manifest'])['old_reservations'] if spec.get('resume_manifest') else [],
        unobserved_rare_costs_not_zero_filled=True, calls=budget.used,
        new_signal=0, new_PF_compile=0, new_fit=0, new_quantum_shots=0,
        automatic_retry=False, next_stage_authorized=False)


def run(spec, grant):
    authorize(spec, grant)
    verify(spec)
    fresh = fresh_gate(spec, grant)
    output = c.home_path(spec['output'])
    c.claim_attempt(spec['lock'], spec['SOURCE'], spec['scope'])
    output.mkdir(exist_ok=False)
    try:
        for name in ('TMPDIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'XDG_CACHE_HOME'):
            path = output / 'scratch' / name
            path.mkdir(parents=True, exist_ok=False)
            os.environ[name] = str(path)
        import tempfile
        tempfile.tempdir = os.environ['TMPDIR']
        font = c.read(spec['payload'])['native_adapter']['font_seed']
        (Path(os.environ['MPLCONFIGDIR']) / font['name']).write_bytes(Path(font['path']).read_bytes())
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        c.save(output / 'startup_gate.json', dict(status='PASS', SOURCE=spec['SOURCE'],
            technical_digest=spec['technical_digest'], fresh=fresh, pid=os.getpid()))
        os.sched_setaffinity(0, {grant['driver_cpu']})
        result = execute(spec, grant, output)
        c.save(output / 'result.json', result)
        verify(spec)
        c.save(output / 'terminal.json', dict(status='COMPLETE_MANDATORY_STOP',
            SOURCE=spec['SOURCE'], result_sha256=c.sha(output / 'result.json'),
            next_stage_authorized=False, retry=False))
        return 0
    except BaseException as error:
        c.first_stop(output, type(error).__name__ + ':' + str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('OWN_RUN_SIGNAL_' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.verify_only:
            c.require(not args.grant, 'VERIFY_ONLY_NO_GRANT')
            print(verify(c.read(args.spec)))
            return 0
        c.require(args.grant is not None, 'EXPLICIT_EXECUTION_GRANT_REQUIRED')
        return run(c.read(args.spec), c.read(args.grant))
    except (Exception, KeyboardInterrupt) as error:
        print('STOP: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
