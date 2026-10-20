# 固定公開SOURCEの検査条件抜粋

コードは実行していない。SOURCE中の条件と実行時receiptは別の証拠である。全文の固定URLとSHAを併記する。

## scripts/h8_pr_confirmation.py

SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8`; file SHA256 `de45b22f85f8d5bd22208220ea19f75699ce89ea5011c64ec352ccb0af36bde5`. [全文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation.py)

### hierarchy (lines 44–49)

```python
def hierarchy(master, q):
    # Original direct_master_to_step_seeds_v1, including the master exclusion.
    if q == 1:
        return [master]
    rng = random.Random(master)
    return [master] + [rng.randrange(0, 2**63) for _ in range(q)]
```

### derive_registry (lines 52–66)

```python
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
```

### validate_registry (lines 69–71)

```python
def validate_registry(tasks, targets, source, seed_plan_digest, excluded):
    c.require(tasks == derive_registry(targets, source, seed_plan_digest, excluded),
              'EXACT_FROZEN_INDEPENDENT_SEED_REGISTRY')
```

### technical_digest (lines 74–75)

```python
def technical_digest(spec):
    return c.digest({k: v for k, v in spec.items() if k != 'technical_digest'})
```

### verify (lines 78–129)

```python
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
```

### authorize (lines 132–157)

```python
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
```

### native_identity (lines 257–260)

```python
def native_identity(value):
    c.require(isinstance(value, (list, tuple)) and len(value) == 4
              and all(type(x) is int for x in value), 'TYPED_NATIVE_IDENTITY')
    return tuple(value)
```

### accept (lines 263–316)

```python
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
```

## scripts/h8_pr_confirmation_parallel.py

SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8`; file SHA256 `46eb3489155479f71d616105b0fe24eb129ad8d351edfadbd180d46f93efff34`. [全文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/h8_pr_confirmation_parallel.py)

### results (lines 17–27)

```python
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
```

### verify_descendants (lines 150–161)

```python
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
```

## scripts/prepare_h8_pr_confirmation_resume.py

SOURCE `745a674546f2cb6258526fbfde1d9a00601689f8`; file SHA256 `fd1d019b0a058bc7c288f5e88ee3e54940dd3556fa353db88e3d7dfef1179ebe`. [全文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/745a674546f2cb6258526fbfde1d9a00601689f8/scripts/prepare_h8_pr_confirmation_resume.py)

### verify_manifest (lines 105–117)

```python
def verify_manifest(spec):
    manifest = c.read(spec['resume_manifest'])
    contents = dict(manifest)
    digest = contents.pop('manifest_digest')
    c.require(c.digest(contents) == digest and manifest['quiescence_observations'] >= 2
              and manifest['scope'] == spec['scope']
              and manifest['registry_sha256'] == c.sha(spec['registry'])
              and manifest['seed_derivation_SOURCE'] == spec['seed_derivation_SOURCE'], 'RESUME_MANIFEST_BINDING')
    c.verify_byte_rows(manifest['files'])
    observed = collect(manifest['origin_spec'], manifest['origin_root'],
                       manifest['parent_wait'], manifest['parent_identity'])
    c.require(all(observed[k] == manifest[k] for k in observed), 'RECHECK_ORIGIN_ACCEPTANCE')
    return manifest
```

`authorize`のCPU配置部分は実行時の条件である。本packetの過去grant照合はSOURCE/technical digest/承認receipt/scopeの結合を対象とし、現在のCPU配置で過去実行を再認可しない。registry導出の再検証は前回の原本照合に記録されている。ここではseed値を提供せず、新たなseed生成・event samplingをしない。
