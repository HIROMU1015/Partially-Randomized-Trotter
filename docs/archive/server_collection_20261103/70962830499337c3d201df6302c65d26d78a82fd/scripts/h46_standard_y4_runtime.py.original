"""Thin binding of the unchanged c028 acquisition, storage and end audit."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import h46_standard_y4_contract as c

RT = None
WARM = False


def fresh_observation(run_root, disk_proxy=0):
    """Reuse the old observer and honor visible ancestor CPU quotas."""
    from run_h46_m5_signal import fresh_resources
    obs = fresh_resources(dict(run_root=str(run_root), systems={'probe': {}},
        resources=dict(system_RSS_proxy_bytes={'probe': 1}, disk_proxy_bytes=disk_proxy)))
    controls = []
    row = next((s for s in Path('/proc/self/cgroup').read_text().splitlines()
                if s.startswith('0::')), None)
    if row:
        p = Path('/sys/fs/cgroup') / row[3:].lstrip('/')
        while p.is_relative_to('/sys/fs/cgroup'):
            if (p / 'cpu.max').is_file():
                quota, period = (p / 'cpu.max').read_text().split()
                controls.append(dict(cpu_max=[quota, period]))
                if quota != 'max':
                    obs['idle_CPU_equivalents'] = min(obs['idle_CPU_equivalents'],
                        int(quota) / int(period))
            if p == Path('/sys/fs/cgroup'):
                break
            p = p.parent
    obs['ancestor_cpu_controls'] = controls
    obs['affinity_count'] = len(os.sched_getaffinity(0))
    return obs


def fresh(cfg):
    import h46_pr_pilot_repair as repair
    lease = c.read(os.environ['H246_CONFIRMATION_LEASE'])
    s = cfg['stage'][:2]
    c.require(lease['configs'].get(cfg['stage']) == dict(SOURCE=cfg['SOURCE'],
        plan_sha256=cfg['plan_sha256']) and lease['runtime_caps'] is False,
        'FRESH_OWN_LEASE_BINDING')
    obs = fresh_observation(cfg['run_root'], lease['peak_growth_proxy_bytes'])
    workers = repair.worker_count(len(cfg['tasks']), obs['idle_CPU_equivalents'],
        obs['effective_available_RAM_bytes'], cfg['resources']['worker_RSS_proxy_bytes'],
        historical_allocation=lease['workers'][s])
    fs = os.statvfs(Path(cfg['run_root']).parent)
    c.require(fs.f_favail >= cfg['resources']['inode_proxy'], 'FRESH_INODES')
    return dict(workers=workers, observation=obs, joint_workers=lease['workers'],
        worker_RSS_proxy_bytes=cfg['resources']['worker_RSS_proxy_bytes'],
        inode_free=fs.f_favail, runtime_caps=False, proxy_is_upper_bound=False,
        storage_peak_proxy_bytes=lease['peak_growth_proxy_bytes'],
        IO_PSI_observation_only=Path('/proc/pressure/io').read_text())


def prewarm(cfg, unused):
    """Count actual setup entry points; no setup is allowed in a worker."""
    global WARM
    c.authorize(cfg)
    c.consumed(cfg)
    c.require(not WARM, 'NO_SECOND_PREPARATION')
    import h46_m5_signal_adapter as adapter
    import h46_m5_tol_native as tol
    import h46_new2_native as native
    import h46_new2_core as core
    ledger = Path(cfg['run_root']) / 'context_ledger'
    ledger.mkdir(exist_ok=False)
    counts = {k: 0 for k in c.context_scope()}

    def reserve(kind):
        i = counts[kind]
        c.require(i < cfg['context_scope'][kind], 'SETUP_CALL_BUDGET:' + kind)
        c.save(ledger / (kind + '__' + str(i) + '.json'),
            dict(kind=kind, index=i, SOURCE=cfg['SOURCE'],
                 plan_sha256=cfg['plan_sha256'], state='RESERVED_NONREFUNDABLE'))
        counts[kind] += 1

    def counted(fn, kind):
        def wrapper(*args, **kwargs):
            reserve(kind)
            return fn(*args, **kwargs)
        return wrapper

    targets = [(adapter, 'base_context', 'saved_reference_read'),
        (tol, 'restore', 'saved_snapshot_restore'),
        (native, 'OccupationOracle', 'occupation_oracle_initialization'),
        (native, 'prepare', 'gaussian_prepare'),
        (core, 'matrix_only_tail_coefficients', 'matrix_only_tail_coefficients'),
        (core, 'spectrum', 'tail_spectrum')]
    originals = [(m, name, getattr(m, name)) for m, name, _ in targets]
    try:
        reserve('bootstrap')
        reserve('setup')
        for m, name, kind in targets:
            setattr(m, name, counted(getattr(m, name), kind))
        ctx = RT.context(cfg, cfg['stage'][:2], c.GROUPS[cfg['stage'][:2]]['prefix'])
        c.require(ctx.cut.preparation_hash == cfg['preparation_hash'], 'FIXED_CUT_PREPARATION')
        c.require(counts == cfg['context_scope'], 'EXACT_COMPLETED_CONTEXT_CALLS')
        WARM = True
        return dict(contexts=1, counted_entry_points=counts, RNG_calls=0,
            signal_evaluations=0, compile_calls=0,
            context_readonly_between_forked_workers=True,
            internal_diagonal_matrix_work='UNCHANGED_C028_FUNCTION_BODIES')
    finally:
        for m, name, fn in originals:
            setattr(m, name, fn)


def guarded_context(cfg, system, prefix):
    # The native worker must inherit the parent-prepared context; a cache miss
    # would otherwise perform extra unbudgeted preparation inside a worker.
    c.require(WARM, 'PARENT_PREWARM_REQUIRED_NO_WORKER_SETUP')
    ctx = RT.RUNTIME.h46_context
    cache = RT.RUNTIME.native_namespace['CACHE']
    key = (cfg['SOURCE'], system, c.digest(cfg['systems'][system]['inputs']))
    c.require(key in cache and prefix in cache[key]['prefixes'], 'READONLY_CONTEXT_CACHE_HIT')
    return ctx(cfg, system, prefix)


def worker(task):
    from trottertracks.resource_applicability import ax2b_native_df_v5 as api
    cfg = RT.RUNTIME.runner_namespace['WORKER_CFG']
    counts = dict(partial_native_build=0, hadamard_wrapper_build=0)
    originals = {name: getattr(api, name) for name in
        ('build_partial_native', 'build_native_hadamard_wrapper')}

    def partial(*args, **kwargs):
        counts['partial_native_build'] += 1
        c.require(counts['partial_native_build'] <= 12, 'EXACT_PARTIAL_NATIVE_BUILDS')
        return originals['build_partial_native'](*args, **kwargs)

    def wrapper(*args, **kwargs):
        i = counts['hadamard_wrapper_build']
        c.reserve(cfg, task, 'wrapper_build', i)
        counts['hadamard_wrapper_build'] += 1
        return originals['build_native_hadamard_wrapper'](*args, **kwargs)

    completed = False
    try:
        api.build_partial_native, api.build_native_hadamard_wrapper = partial, wrapper
        result = RT.worker(task)
        c.require(counts == dict(partial_native_build=12, hadamard_wrapper_build=2),
            'EXACT_COMPLETED_BUILD_DISPATCH')
        completed = True
        return result
    finally:
        for name, fn in originals.items():
            setattr(api, name, fn)
        out = Path(cfg['output']) / task['id']
        if out.is_dir():
            c.save(out / 'build_dispatch.json', dict(SOURCE=cfg['SOURCE'],
                plan_sha256=cfg['plan_sha256'], task_id=task['id'],
                counts=counts, completed=completed,
                partial_native_build_is_nested_in_native_piece_compositions=True))


def runtime(cfg):
    global RT
    c.authorize(cfg)
    import h246_confirmation_runtime as old
    if RT is None:
        RT, _ = old.rebind(old, dict(c=c, rebind=old.rebind,
            cohort_code=old.cohort_code))
    r = RT.get_runtime(cfg)
    cli = str(c.ROOT / 'scripts/run_h46_standard_y4.py')
    r.runner_namespace['__file__'] = cli
    r.runner.supervise.__globals__['__file__'] = cli
    r.runner_namespace['worker'] = worker
    r.runner_namespace['fresh'] = fresh
    r.runner.execute.__kwdefaults__ = {'target': worker}
    r.runner_namespace['repair'] = SimpleNamespace(enabled=RT.enabled, prewarm=prewarm)
    r.native_namespace['context'] = guarded_context
    return r


def audit(cfg):
    rt = runtime(cfg)
    result = rt.auditor.audit(cfg)
    root = Path(cfg['run_root'])
    expected = {kind + '__' + str(i) + '.json' for kind, n in cfg['context_scope'].items()
                for i in range(n)}
    actual = {p.name for p in (root / 'context_ledger').glob('*.json')}
    errors = dict(result['errors'])
    if actual != expected:
        errors['context_ledger'] = 'INCOMPLETE_OR_EXTRA_CONTEXT_CALLS'
    for p in (root / 'context_ledger').glob('*.json'):
        row = c.read(p)
        if row['SOURCE'] != cfg['SOURCE'] or row['plan_sha256'] != cfg['plan_sha256']:
            errors['context_binding'] = 'SOURCE_OR_PLAN_MISMATCH'
    build_joins = []
    for task in cfg['tasks']:
        path = Path(cfg['output']) / task['id'] / 'build_dispatch.json'
        if not path.is_file():
            errors[task['id']] = 'MISSING_BUILD_DISPATCH'
            continue
        row = c.read(path)
        if not (row['SOURCE'] == cfg['SOURCE'] and row['plan_sha256'] == cfg['plan_sha256']
                and row['task_id'] == task['id'] and row['completed'] and
                row['counts'] == dict(partial_native_build=12, hadamard_wrapper_build=2)):
            errors[task['id']] = 'BUILD_DISPATCH_MISMATCH'
        build_joins.append(dict(task_id=task['id'], sha256=c.sha(path)))
    report = dict(status='STOP' if errors else 'CONTEXT_AND_ACQUISITION_AUDIT_PASS',
        errors=errors, build_joins=build_joins, end_audit_sha256=c.sha(root / 'end_audit.json'),
        SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], retry=False,
        mandatory_STOP=True)
    c.save(root / 'context_end_audit.json', report)
    return report


def launch(package_path, configs):
    """Fresh joint admission; unrelated worktrees and host PSI cannot kill jobs."""
    from launch_h246_confirmation import allocate
    p = c.verify_package(package_path)
    cfgs = [c.read(path) for path in configs]
    c.require({v['stage'] for v in cfgs} == set(c.STAGES) and len(cfgs) == 2,
        'TWO_STAGES_ONCE')
    for cfg in cfgs:
        c.authorize(cfg)
        c.require(not Path(cfg['run_root']).exists(), 'NO_RETRY')
    state = Path(p['assets']) / 'controller_run01'
    state.mkdir(exist_ok=False)
    pending = list(zip(configs, cfgs))
    reports = []
    while pending:
        obs = fresh_observation(state / 'unused')
        lease = allocate(obs, [cfg for _, cfg in pending])
        chosen = [(path, cfg) for path, cfg in pending if cfg['stage'][:2] in lease['workers']]
        lease['configs'] = {cfg['stage']: dict(SOURCE=cfg['SOURCE'],
            plan_sha256=cfg['plan_sha256']) for _, cfg in chosen}
        lp = state / ('lease_' + str(len(reports)) + '.json')
        c.save(lp, lease)
        active = []
        for path, cfg in chosen:
            log = (state / (cfg['stage'] + '_controller.log')).open('xb')
            env = dict(os.environ, H246_CONFIRMATION_LEASE=str(lp))
            proc = __import__('subprocess').Popen([cfg['runtime']['python'], '-B',
                str(c.ROOT / 'scripts/run_h46_standard_y4.py'), '--supervise',
                '--config', path], env=env, stdout=log, stderr=__import__('subprocess').STDOUT)
            active.append((proc, path, cfg, log))
        for proc, path, cfg, log in active:
            code = proc.wait()
            log.close()
            au = __import__('subprocess').run([cfg['runtime']['python'], '-B',
                str(c.ROOT / 'scripts/run_h46_standard_y4.py'), '--audit', '--config', path],
                capture_output=True, text=True)
            report = dict(stage=cfg['stage'], supervisor_exit=code, audit_exit=au.returncode,
                audit_stdout=au.stdout, audit_stderr=au.stderr, retry=False)
            c.save(state / (cfg['stage'] + '_finish.json'), report)
            reports.append(report)
        pending = [x for x in pending if x not in chosen]
        if any(r['supervisor_exit'] or r['audit_exit'] for r in reports):
            break
    success = len(reports) == 2 and not any(r['supervisor_exit'] or r['audit_exit'] for r in reports)
    c.save(state / 'CONTROLLER_FINISHED.json', dict(reports=reports,
        unstarted=[cfg['stage'] for _, cfg in pending], success=success,
        retry=False, mandatory_STOP=True))
    return success
